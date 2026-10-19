"""Independent 2D normalization, conditional density, correlation and tails."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm

from research_experiments.mobile_ev_mle import continuous_eig_sampling as s
from research_experiments.mobile_ev_mle.continuous_eig_proposals import (SequentialGaussianBoxProposal,
    log_standard_normal_interval,sample_standard_normal_interval,clip_last_bit_roundoff)
from research_experiments.mobile_ev_mle.test_continuous_eig_sampling import toy_component


def proposal(uniform=.2,correlation=.8):
    c = toy_component()
    sd = np.array([.12,.09])
    covariance = np.diag(sd**2)
    covariance[0,1] = covariance[1,0] = correlation*np.prod(sd)
    return SequentialGaussianBoxProposal(c,[[1.,1.]],covariance[None],uniform)


def test_deep_tail_normalizers_and_inverse_do_not_underflow_or_escape():
    a = np.array([-101.,100.,-.001,-3.,35.])
    b = np.array([-100.,101.,.001,2.,36.])
    logz = log_standard_normal_interval(a,b)
    assert np.isfinite(logz).all()
    assert logz[0] < -5000 and logz[1] < -5000
    np.testing.assert_allclose(logz[0],logz[1],atol=1e-12)
    np.testing.assert_allclose(np.exp(logz[2:4]),ndtr(b[2:4])-ndtr(a[2:4]),rtol=1e-11)
    rows = np.broadcast_to(a,(3000,len(a)))
    upper = np.broadcast_to(b,rows.shape)
    draws = sample_standard_normal_interval(rows,upper,np.random.default_rng(26613))
    assert np.isfinite(draws).all() and np.all(draws >= rows) and np.all(draws <= upper)
    assert draws[:,1].std() > 0


def test_interval_rejects_degenerate_normalization_instead_of_dropping_points():
    with pytest.raises(ValueError):
        log_standard_normal_interval([1.],[1.])
    with pytest.raises(ValueError):
        log_standard_normal_interval([np.nan],[2.])


def test_two_dimensional_density_matches_independent_conditional_formula():
    p = proposal(0.2)
    beta = np.array([[1.,2.],[.9,1.75],[1.15,2.3]])
    phi = beta/p.component.nominal_beta
    C = p.covariances_phi[0]
    L = np.linalg.cholesky(C)
    mu = np.ones(2)
    eta0 = (phi[:,0]-mu[0])/L[0,0]
    firstZ = norm.cdf((1.2-mu[0])/L[0,0])-norm.cdf((.8-mu[0])/L[0,0])
    first = norm.pdf(eta0)/L[0,0]/firstZ
    mean1 = mu[1]+L[1,0]*eta0
    secondZ = norm.cdf((1.2-mean1)/L[1,1])-norm.cdf((.8-mean1)/L[1,1])
    second = norm.pdf((phi[:,1]-mean1)/L[1,1])/L[1,1]/secondZ
    expected = first*second/np.prod(p.component.nominal_beta)
    np.testing.assert_allclose(np.exp(p.sequential_logpdf(beta,0)),expected,rtol=2e-13)
    np.testing.assert_allclose(np.exp(p.logpdf(beta)),.2*np.exp(p.component.prior_logconstant)+.8*expected,rtol=2e-13)


def test_conditional_integral_and_full_two_dimensional_quadrature_equal_one():
    p = proposal(.2,.93)
    c = p.component
    def density(x,y):
        return float(np.exp(p.sequential_logpdf(np.array([x,y]),0)))
    # Independent integration over y leaves the exact first marginal.
    L = np.linalg.cholesky(p.covariances_phi[0])
    Z0 = norm.cdf(.2/L[0,0])-norm.cdf(-.2/L[0,0])
    for x in (.805,.97,1.195):
        integrated = quad(lambda y:density(x,y),c.lower[1],c.upper[1],epsabs=1e-8)[0]
        expected = norm.pdf((x-1.)/L[0,0])/L[0,0]/Z0
        assert integrated == pytest.approx(expected,rel=1e-8)
    joint = quad(lambda x:quad(lambda y:density(x,y),c.lower[1],c.upper[1],epsabs=1e-8)[0],
                 c.lower[0],c.upper[0],epsabs=1e-7)[0]
    assert joint == pytest.approx(1.,abs=2e-7)


@pytest.mark.parametrize("sign",[1.,-1.])
def test_box_supported_samples_keep_correlation_sign_and_importance_target_one(sign):
    p = proposal(.2,sign*.93)
    beta,logq = p.sample(40000,np.random.default_rng(27613 if sign > 0 else 28613))
    assert np.isfinite(p.component.prior_logpdf(beta)).all()
    corr = np.corrcoef(beta.T)[0,1]
    assert sign*corr > .4
    # Full exact q is evaluated independently again, no component-density shortcut.
    np.testing.assert_allclose(logq,p.logpdf(beta),rtol=0,atol=0)
    w = np.exp(p.component.prior_logpdf(beta)-logq)
    assert abs(w.mean()-1.) < 5*w.std(ddof=1)/np.sqrt(len(w))
    assert w.max() <= 5.+1e-12


def test_permuted_order_beta_scale_and_multicomponent_mixture_density():
    base = proposal(.2)
    # Swap diagonal ordering to force the second beta coordinate first.
    C = np.array([[.003,.001],[.001,.02]])
    p = SequentialGaussianBoxProposal(base.component,[[1.,1.],[.98,1.02]],np.array([C,C*1.4]),.2)
    assert p.payload()["coordinate_orders"][0] == [1,0]
    beta = np.array([[1.,2.],[.88,2.3]])
    expected = .2*np.exp(base.component.prior_logpdf(beta))+.4*np.exp(p.sequential_logpdf(beta,0))+.4*np.exp(p.sequential_logpdf(beta,1))
    np.testing.assert_allclose(np.exp(p.logpdf(beta)),expected,rtol=2e-13)
    assert p.logpdf([.7,2.]) == -np.inf
    assert p.payload()["standard_box_truncated_MVN"] is False
    draws,ld = p.sample(2000,np.random.default_rng(29613))
    assert np.isfinite(ld).all() and np.isfinite(base.component.prior_logpdf(draws)).all()


def test_near_corner_strong_correlations_never_require_unknown_MVN_normalizer():
    c = toy_component()
    C = np.array([[.013,-.0129],[-.0129,.013]])
    p = SequentialGaussianBoxProposal(c,[[.8003,.8003]],C[None],.2)
    beta,lq = p.sample(4000,np.random.default_rng(30613))
    assert np.isfinite(c.prior_logpdf(beta)).all() and np.isfinite(lq).all()
    # Original MVN box probability is nowhere required or estimated.
    assert p.payload()["conditional_normalizers"] == "past-dependent Zj included exactly"


def test_nonfinite_means_rejected_and_large_clips_raise_instead_of_hidden_projection():
    c = toy_component()
    with pytest.raises(ValueError,match="finite"):
        SequentialGaussianBoxProposal(c,[[np.nan,1.]],np.eye(2)[None],.2)
    with pytest.raises(FloatingPointError,match="roundoff tolerance"):
        clip_last_bit_roundoff([1.20001],[.8],[1.2],name="test")
    ledger = {}
    actual = clip_last_bit_roundoff([np.nextafter(1.2,np.inf)],[.8],[1.2],name="test",diagnostics=ledger)
    assert actual[0] == 1.2
    assert 0 < ledger["test"]["maximum_absolute_correction"] < 1e-14


def test_cross_zero_inverse_handles_last_float_below_one_without_infinite_quantile():
    class LastUniform:
        def random(self,shape):
            return np.full(shape,np.nextafter(1.,0.))
    value = sample_standard_normal_interval([-1.],[100.],LastUniform())
    assert np.isfinite(value).all() and value[0] > 7 and value[0] < 100
