"""Cross-component invariants; simulator truth is isolated from deployment."""
import inspect
import numpy as np
from .candidates import propose, normalize_family
from .design import EVConstraints, enumerate_actions, select_action
from .simulation import generate, observe, model
from .run_study import threshold


def test_ev_is_in_meter_total_exactly_once_and_noise_is_paired():
    path=generate(9)
    cons=EVConstraints()
    actions,_=enumerate_actions(8,3,cons,max_actions=9)
    obs0,_=observe(path,slice(12,None))
    obs1,pred=observe(path,slice(12,None),actions[0])
    assert np.allclose(obs1.p-obs0.p,actions[0].p,atol=1e-16)
    assert np.allclose(obs1.q-obs0.q,actions[0].q,atol=1e-16)
    assert np.allclose(pred["p"],path.p[12:]+actions[0].p)
    assert np.ptp(path.source_q/path.source_p)>.01


def test_n3_repairs_cover_declared_class_without_truth():
    path=generate(10)
    obs,_=observe(path,slice(0,12))
    candidates,diag=propose(obs,run_wzzt=False)
    families={normalize_family(c.supports,3) for c in candidates}
    assert families=={(),((0,1),),((0,2),),((1,2),)}
    assert diag["uses_truth"] is False
    assert "truth" not in inspect.signature(propose).parameters
    assert "truth" not in inspect.signature(select_action).parameters
    assert "future_noise" not in inspect.signature(select_action).parameters


def test_calibration_order_statistic_and_small_sample_limit():
    assert threshold(range(20),.1)==18.
    assert threshold(range(8),.1) is None
    assert threshold([],.1) is None


def test_replay_seed_and_calibrated_bias_sharing():
    a,b=generate(100),generate(100)
    ao,_=observe(a,slice(0,12));bo,_=observe(b,slice(0,12))
    assert np.array_equal(ao.p,bo.p)
    assert np.array_equal(ao.v,bo.v)
    meter=model(3)
    assert np.allclose(meter.bias_p,a.bias["p"])
    assert np.allclose(meter.bias_v,a.bias["v"])
