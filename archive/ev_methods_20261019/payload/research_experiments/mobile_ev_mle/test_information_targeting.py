"""Independent task-weight algebra, floor and failure audits."""
from types import SimpleNamespace
from inspect import signature
import numpy as np
import pytest
from .likelihood import RadialCandidate
from .information_targeting import history_target_weights


def inputs(losses=(0.,2.,4.)):
    pairs=((0,1),(0,2),(1,2))
    cs=[RadialCandidate.from_supports("clade_"+"".join(map(str,p)),3,(p,)) for p in pairs]
    fits={c.name:SimpleNamespace(success=True,loss=l) for c,l in zip(cs,losses)}
    return cs,fits


def test_exact_half_sse_softmax_and_floor_formula():
    cs,fs=inputs()
    out=history_target_weights(cs,fs)
    expected=.05+.85*np.exp(-np.arange(3))/(1+np.exp(-1)+np.exp(-2))
    assert out["success"]
    np.testing.assert_allclose(list(out["weights"].values()),expected,rtol=1e-14)
    assert sum(out["weights"].values())==pytest.approx(1.)
    assert not out["weights_are_topology_probabilities"]
    assert out["minimum_weight_floor"]==pytest.approx(.05)


def test_shifted_large_losses_do_not_overflow_and_keep_all_positive():
    cs,fs=inputs((1e12,1e12+2000.,1e12+4000.))
    out=history_target_weights(cs,fs)
    assert out["success"]
    np.testing.assert_allclose(list(out["weights"].values()),[.9,.05,.05],atol=1e-14)
    assert len(out["necessary_targets_retained"])==3


def test_equal_loss_and_complete_floor_produce_uniform_tasks():
    cs,fs=inputs((7.,7.,7.))
    np.testing.assert_allclose(list(history_target_weights(cs,fs)["weights"].values()),np.full(3,1/3))
    cs,fs=inputs((7.,17.,27.))
    np.testing.assert_allclose(list(history_target_weights(cs,fs,epsilon=1.)["weights"].values()),np.full(3,1/3))


def test_half_sse_cost_translation_invariance_and_temperature():
    cs,fs=inputs((3.,5.,9.));out=history_target_weights(cs,fs,tau=2.)
    cs,fs=inputs((203.,205.,209.));shifted=history_target_weights(cs,fs,tau=2.)
    np.testing.assert_allclose(list(out["weights"].values()),list(shifted["weights"].values()))
    cs,fs=inputs((3.,5.,9.));hot=history_target_weights(cs,fs,tau=20.)
    assert hot["weights"][(0,1)]<out["weights"][(0,1)]


@pytest.mark.parametrize("bad",[np.inf,np.nan,-1.])
def test_required_nonfinite_or_negative_fit_fails_entire_weight_rule(bad):
    cs,fs=inputs((1.,bad,3.));out=history_target_weights(cs,fs)
    assert not out["success"] and out["weights"] is None
    assert len(out["target_records"])==3
    assert out["target_records"][1]["failure"]=="invalid_historical_loss"
    assert out["necessary_targets_retained"]==[[0,1],[0,2],[1,2]]


def test_missing_candidate_failed_fit_and_duplicates_remain_failures():
    cs,fs=inputs()
    out=history_target_weights(cs[:2],fs)
    assert not out["success"] and len(out["target_records"])==3
    fs[cs[0].name].success=False
    out=history_target_weights(cs,fs)
    assert not out["success"] and out["weights"] is None
    fs[cs[0].name].success=True
    out=history_target_weights(cs+[cs[0]],fs)
    assert not out["success"] and any(f["kind"]=="duplicate_target_support" for f in out["failures"])


def test_star_is_not_given_zero_information_or_treated_as_deleted():
    cs,fs=inputs();cs.append(RadialCandidate.from_supports("star",3))
    out=history_target_weights(cs,fs)
    assert out["success"] and len(out["weights"])==3
    assert len(cs)==4


@pytest.mark.parametrize("kw",[{"epsilon":0.},{"epsilon":1.01},{"tau":0.},{"tau":np.inf}])
def test_invalid_policy_parameters_are_rejected(kw):
    cs,fs=inputs()
    with pytest.raises(ValueError):history_target_weights(cs,fs,**kw)


def test_entropy_first_order_condition_and_minimum_are_independently_checked():
    cs,fs=inputs((0.,2.,4.));out=history_target_weights(cs,fs)
    v=np.array([r["entropy_regularized_component"] for r in out["target_records"]])
    costs=np.array([r["cost_half_SSE"] for r in out["target_records"]])
    stationarity=costs+np.log(3*v)+1
    np.testing.assert_allclose(stationarity,np.full(3,stationarity[0]),atol=1e-14)
    objective=lambda w:np.dot(w,costs)+np.dot(w,np.log(3*w))
    for other in (np.full(3,1/3),np.array([.8,.1,.1]),np.array([.2,.7,.1])):
        assert objective(v)<objective(other)
    assert not {"truth","future_noise","future_observations"}&set(signature(history_target_weights).parameters)