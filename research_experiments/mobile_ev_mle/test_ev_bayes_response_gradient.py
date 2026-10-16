"""Independent Gaussian derivative checks, including a sign reversal."""
import numpy as np
import pytest
from scipy.stats import norm
from . import ev_bayes_small_probability as probability
from . import ev_bayes_response_gradient as gradient


def bank(mu,cov):
    return dict(means=np.asarray(mu,float)[None],covariances=np.asarray(cov,float)[None],
        prior=np.array([.5,.5]),log_prior=np.log([.5,.5]),weights=np.ones(1))


def shifted(b,dm,dc,h):
    return dict(b,means=b['means']+h*dm,covariances=b['covariances']+h*dc)


def test_full_bc_direction_matches_fd_with_unequal_correlated_covariances():
    b=bank([[.1,-.3],[.8,.4]],[[[.8,.2],[.2,.6]],[[1.2,-.1],[-.1,.5]]])
    dm=np.array([[[.2,.3],[-.1,.5]]]);dc=np.array([[[[.1,.02],[.02,-.08]],[[.2,-.04],[-.04,.15]]]])
    result=gradient.bhattacharyya_directional(b,dm,dc);h=1e-5
    plus=probability.bhattacharyya_bank(shifted(b,dm,dc,h));minus=probability.bhattacharyya_bank(shifted(b,dm,dc,-h))
    assert result['dC_full']==pytest.approx((plus['bound']-minus['bound'])/(2*h),abs=2e-10)
    assert result['pairs'][0]['dB_full']==pytest.approx((plus['pairs'][0]['B']-minus['pairs'][0]['B'])/(2*h),abs=2e-10)
    assert abs(result['dC_covariance'])>1e-4


def test_mean_only_bc_gradient_can_have_opposite_sign_to_full_gradient():
    # delta(u)=u, sigma(u)=exp(2u), at u=1. Separation grows in volts,
    # but grows more slowly than its uncertainty. B=u² exp(-4u)/8 declines.
    u=1.;variance=np.exp(4*u)
    b=bank([[0.],[u]],[[[variance]],[[variance]]])
    result=gradient.bhattacharyya_directional(b,np.array([[[0.],[1.]]]),np.full((1,2,1,1),4*variance))
    pair=result['pairs'][0]
    assert pair['dB_mean']>0 and pair['dB_full']<0
    assert pair['dB_full']==pytest.approx(np.exp(-4)/8*(2-4))
    assert result['dC_mean']<0 and result['dC_full']>0


def test_cholesky_derivative_matches_finite_difference():
    S=np.array([[1.2,.3],[.3,.7]]);dS=np.array([[.1,.05],[.05,-.02]]);h=1e-5
    dL=gradient.cholesky_directional(np.linalg.cholesky(S),dS)
    finite=(np.linalg.cholesky(S+h*dS)-np.linalg.cholesky(S-h*dS))/(2*h)
    np.testing.assert_allclose(dL,finite,atol=2e-11,rtol=1e-9)


def test_equal_gaussian_risk_gradient_matches_analytic_envelope_and_crn():
    # Equal priors, means +/-a*u, shared sigma=exp(b*u): R=Phi(-a*u/sigma).
    u=.8;a=.7;c=.4;sigma=np.exp(c*u);variance=sigma**2
    b=bank([[-a*u],[a*u]],[[[variance]],[[variance]]])
    dm=np.array([[[-a],[a]]]);dc=np.full((1,2,1,1),2*c*variance)
    noise=np.random.default_rng(144).standard_normal((1,2,65536,1))
    result=gradient.risk_directional(b,noise,dm,dc)
    exact=-norm.pdf(a*u/sigma)*(a/sigma)*(1-c*u)
    for name in ('score_full','path_full'):
        assert abs(result[name]['value']-exact)<5*result[name]['standard_error']+2e-4
    # A Gaussian bank locally varies with these derivatives; shared draws
    # differentiate the finite-sample risk itself, independently of the score.
    h=1e-5
    finite=gradient.risk_crn_finite_difference(shifted(b,dm,dc,h),shifted(b,dm,dc,-h),noise,h)
    assert finite['value']==pytest.approx(result['path_full']['value'],abs=2e-6)
    assert result['path_mean']['value']+result['path_covariance']['value']==pytest.approx(result['path_full']['value'],abs=2e-15)
    assert result['score_mean']['value']+result['score_covariance']['value']==pytest.approx(result['score_full']['value'],abs=2e-15)


def test_common_translation_has_zero_path_and_bc_derivative():
    b=bank([[-.3],[.5]],[[[.7]],[[1.2]]]);dm=np.ones((1,2,1));noise=np.random.default_rng(11).standard_normal((1,2,4096,1))
    bc=gradient.bhattacharyya_directional(b,dm)
    risk=gradient.risk_directional(b,noise,dm)
    assert bc['dC_full']==0
    assert risk['path_full']['value']==0
    assert abs(risk['score_full']['value'])<5*risk['score_full']['standard_error']


def test_unequal_covariance_posterior_path_matches_common_noise_fd():
    b=bank([[.1,-.3],[.8,.4]],[[[.8,.2],[.2,.6]],[[1.2,-.1],[-.1,.5]]])
    dm=np.array([[[.2,.3],[-.1,.5]]]);dc=np.array([[[[.1,.02],[.02,-.08]],[[.2,-.04],[-.04,.15]]]])
    noise=np.random.default_rng(45).standard_normal((1,2,4096,2));h=1e-5
    result=gradient.risk_directional(b,noise,dm,dc)
    finite=gradient.risk_crn_finite_difference(shifted(b,dm,dc,h),shifted(b,dm,dc,-h),noise,h)
    assert finite['value']==pytest.approx(result['path_full']['value'],abs=2e-6)
    assert abs(result['score_full']['value']-result['path_full']['value'])<5*(
        result['score_full']['standard_error']+result['path_full']['standard_error'])


def test_nonfinite_or_nonsymmetric_directions_are_rejected():
    b=bank([[0.],[1.]],[[[1.]],[[1.]]])
    with pytest.raises(ValueError):gradient.bhattacharyya_directional(b,np.full((1,2,1),np.nan))
    with pytest.raises(ValueError):gradient.risk_directional(b,np.zeros((1,2,1,1)),np.zeros((1,2,1)))
