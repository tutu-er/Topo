"""Independent fixed-F identities and probability checks, no grid data."""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import block_diag
from scipy.special import expit, logit

try:
    from .fixed_f import (build_fixed_f, build_fixed_f_full, DenseQuadraticEvidence,
                          conditional_mean_posterior, full_covariance,
                          fixed_covariance, fixed_covariance_sha256,
                          guarantees, thresholds)
    from .core import event_probability
except ImportError:
    from fixed_f import (build_fixed_f, build_fixed_f_full, DenseQuadraticEvidence,
                         conditional_mean_posterior, full_covariance,
                         fixed_covariance, fixed_covariance_sha256,
                         guarantees, thresholds)
    from core import event_probability


def fixture():
    base = np.array([[1.2, -.4], [.3, .8], [-.2, 1.1]])
    effect = np.array([[[.4, -.2], [.1, .6]], [[.3, .1], [-.4, .5]],
                       [[-.2, .8], [.6, -.1]]])
    cov = np.array([[[.4, .1], [.1, .2]], [[.3, -.15], [-.15, .075]],
                    [[0., 0.], [0., .7]]])
    u = np.array([[.3, -.2, .5], [-.4, .1, -.3]])
    return base, effect, cov, u


def test_score_matches_independent_dense_time_block_calculation():
    base, effect, cov, u = fixture()
    model = build_fixed_f(base, effect, cov)
    mean = (base+np.einsum("tnr,rt->tn",effect,u)).ravel()
    dense = np.eye(mean.size)+block_diag(*cov)/4.
    direct = mean@np.linalg.solve(dense,mean)/8.+np.linalg.slogdet(dense)[1]/2.
    assert model.kind == "fixed_f_negative_log_laplace_transform"
    assert model.value(u) == pytest.approx(direct,abs=2e-14)
    assert np.linalg.eigvalsh(model.Q).min() >= -1e-14


def test_fixed_f_score_gradient_matches_independent_difference():
    base, effect, cov, u = fixture()
    model = build_fixed_f(base,effect,cov)
    numerical = np.empty_like(u)
    for index in np.ndindex(u.shape):
        up, down = u.copy(),u.copy()
        up[index] += 1e-6;down[index] -= 1e-6
        numerical[index] = (model.value(up)-model.value(down))/2e-6
    np.testing.assert_allclose(model.gradient(u),numerical,atol=3e-10,rtol=3e-10)


def test_scalar_score_matches_direct_gaussian_exponential_integral():
    base = np.array([[1.7]])
    effect = np.array([[[.8]]])
    cov = np.array([[[.6]]])
    u = np.array([[-.3]])
    model = build_fixed_f(base,effect,cov)
    mean = float(base[0,0]+effect[0,0,0]*u[0,0])
    integral = quad(lambda z: math.exp(-(mean+math.sqrt(.6)*z)**2/8.)
                    *math.exp(-z*z/2.)/math.sqrt(2.*math.pi),
                    -np.inf,np.inf,epsabs=1e-12)[0]
    assert math.exp(-model.value(u)) == pytest.approx(integral,abs=2e-13)


def test_bounds_hold_for_nonzero_gaussian_power_covariance():
    mean, variance, initial = 4., .5, .2
    model = build_fixed_f([[mean]],[[[0.]]],[[[variance]]])
    bound = guarantees(model.value([[0.]]),initial,.9)
    density = lambda z: math.exp(-z*z/2.)/math.sqrt(2.*math.pi)
    true_mean = quad(lambda z: conditional_mean_posterior(
        (mean+math.sqrt(variance)*z)**2,initial)*density(z),
        -9.,9.,epsabs=2e-10)[0]
    true_event = quad(lambda z: event_probability(
        (mean+math.sqrt(variance)*z)**2,.9,initial,"F")*density(z),
        -9.,9.,epsabs=2e-10)[0]
    assert true_mean >= bound["mean_posterior_lower_bound"]
    assert true_event >= bound["posterior_event_lower_bound"]
    assert bound["mean_posterior_lower_bound"] > .9
    assert bound["posterior_event_lower_bound"] > .5


def test_two_thresholds_certify_their_own_requirements_not_each_other():
    cut = thresholds(q_star=.9,eta=.9,beta=.9,log_odds0=0.)
    assert cut["mean"] == pytest.approx(math.log(5.))
    assert cut["probability"] == pytest.approx(logit(.9)/2.+math.log(10.))
    for mode in ("mean","probability"):
        S = 8.*cut[mode]
        model = build_fixed_f([[math.sqrt(S)]],[[[0.]]],[[[0.]]])
        score = model.value([[0.]])
        lower = guarantees(score,0.,.9)
        if mode == "mean":
            assert lower["mean_posterior_lower_bound"] == pytest.approx(.9)
            assert conditional_mean_posterior(S) >= .9
        else:
            assert lower["posterior_event_lower_bound"] == pytest.approx(.9)
            assert event_probability(S,.9,state="F") >= .9
    assert cut["probability"] > cut["mean"]


def test_fixed_covariance_fingerprint_is_price_independent_and_inputs_unchanged():
    base,effect,cov,u=fixture()
    original=cov.copy();fingerprint=fixed_covariance_sha256(cov)
    model=build_fixed_f(base,effect,cov)
    first=model.value(u);second=model.value(2*u)
    assert first != second
    assert fixed_covariance_sha256(cov)==fingerprint
    np.testing.assert_array_equal(cov,original)
    changed=cov.copy();changed[0,0,0]+=.01
    assert fixed_covariance_sha256(changed)!=fingerprint


def test_zero_singular_covariance_and_zero_separation_are_valid():
    C=np.array([[[1.,1.],[1.,1.]]])
    np.testing.assert_allclose(fixed_covariance(C),C,atol=1e-14)
    model=build_fixed_f(np.zeros((1,2)),np.zeros((1,2,1)),np.zeros((1,2,2)))
    assert model.value([[3.]])==0.
    assert conditional_mean_posterior(0.,.7)==pytest.approx(expit(.7))
    assert conditional_mean_posterior(0.,.7,"C")==pytest.approx(expit(-.7))
    bound=guarantees(0.,0.,.9)
    assert bound["mean_posterior_lower_bound"]==.5
    assert bound["posterior_event_lower_bound"]==0.


def test_exact_conditional_mean_is_not_posterior_at_expected_log_odds():
    S=2.*logit(.9)
    assert conditional_mean_posterior(S)==pytest.approx(.790176328914923,abs=2e-12)
    assert expit(S/2.)==pytest.approx(.9)
    values=conditional_mean_posterior(np.array([0.,S]),.3,"F")
    assert values.shape==(2,)
    assert conditional_mean_posterior(S,.3,"C")==pytest.approx(
        conditional_mean_posterior(S,-.3,"F"),abs=2e-13)


def test_probability_bounds_are_stable_for_extreme_history():
    for initial in (-1000.,1000.):
        result=guarantees(np.array([0.,2.,1e4]),initial)
        for name in ("mean_posterior_lower_bound","posterior_event_lower_bound"):
            assert np.isfinite(result[name]).all()
            assert np.all((result[name]>=0.)&(result[name]<=1.))
    assert guarantees(0.,1000.)["mean_posterior_lower_bound"]==1.
    assert guarantees(0.,-1000.)["mean_posterior_lower_bound"]==0.


def test_invalid_covariance_shapes_and_probability_inputs_are_rejected():
    base,effect,cov,_=fixture()
    bad=cov.copy();bad[0,0,1]+=.1
    with pytest.raises(ValueError):build_fixed_f(base,effect,bad)
    bad=cov.copy();bad[0,0,0]=-1.
    with pytest.raises(ValueError):build_fixed_f(base,effect,bad)
    with pytest.raises(ValueError):build_fixed_f(base,effect,cov[:1])
    with pytest.raises(ValueError):build_fixed_f(base,effect[0],cov)
    with pytest.raises(ValueError):build_fixed_f(base,np.full_like(effect,np.nan),cov)
    with pytest.raises(ValueError):thresholds(q_star=1.)
    with pytest.raises(ValueError):thresholds(eta=.5)
    with pytest.raises(ValueError):thresholds(beta=0.)
    with pytest.raises(ValueError):guarantees(-.1)
    with pytest.raises(ValueError):conditional_mean_posterior(-1.)
    with pytest.raises(ValueError):conditional_mean_posterior(1.,state="bad")


def test_dense_builder_matches_block_builder_and_resource_major_gradient():
    base,effect,cov,u=fixture()
    block=build_fixed_f(base,effect,cov)
    dense=build_fixed_f_full(base,effect,block_diag(*cov))
    assert dense.shape==block.shape==u.shape
    assert dense.Q.shape==(u.size,u.size)
    assert dense.value(u)==pytest.approx(block.value(u),abs=2e-14)
    np.testing.assert_allclose(dense.gradient(u),block.gradient(u),atol=2e-14)
    np.testing.assert_allclose(dense.scaled(-1).gradient(u),-dense.gradient(u))
    assert dense.scaled(-1).value(u)==pytest.approx(-dense.value(u))


def test_full_builder_preserves_cross_slot_covariance_not_just_diagonal_blocks():
    base,effect,_,u=fixture()
    factors=np.array([[.5,.1],[-.2,.4],[.3,.2],[.1,-.4],[.6,.3],[-.3,.5]])
    cov=factors@factors.T
    dense=build_fixed_f_full(base,effect,cov)
    mean=(base+np.einsum("tnr,rt->tn",effect,u)).ravel()
    H=np.eye(mean.size)+cov/4.
    direct=mean@np.linalg.solve(H,mean)/8.+np.linalg.slogdet(H)[1]/2.
    assert dense.value(u)==pytest.approx(direct,abs=3e-14)
    diagonal_blocks=np.array([cov[2*t:2*t+2,2*t:2*t+2] for t in range(3)])
    assert abs(dense.value(u)-build_fixed_f(base,effect,diagonal_blocks).value(u))>1e-5
    numerical=np.empty_like(u)
    for index in np.ndindex(u.shape):
        up,down=u.copy(),u.copy();up[index]+=1e-6;down[index]-=1e-6
        numerical[index]=(dense.value(up)-dense.value(down))/2e-6
    np.testing.assert_allclose(dense.gradient(u),numerical,atol=3e-10,rtol=3e-10)


def test_full_covariance_validation_and_dense_shape_rejections():
    np.testing.assert_array_equal(full_covariance(np.zeros((2,2))),np.zeros((2,2)))
    np.testing.assert_allclose(full_covariance([[1.,1.],[1.,1.]]),np.ones((2,2)),atol=1e-14)
    with pytest.raises(ValueError):full_covariance([[1.,.3],[.2,1.]])
    with pytest.raises(ValueError):full_covariance([[1.,2.],[2.,1.]])
    with pytest.raises(ValueError):DenseQuadraticEvidence(np.eye(2),np.zeros(2),0.,1,3)
    with pytest.raises(ValueError):DenseQuadraticEvidence(np.eye(2),np.zeros(2),0.,0,2)
    with pytest.raises(ValueError):build_fixed_f_full([[0.]],[[[0.]]],np.eye(2))
