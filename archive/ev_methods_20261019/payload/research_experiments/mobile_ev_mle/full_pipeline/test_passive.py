"""Independent structure, observation boundary and real-component checks."""
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from research_experiments.mobile_ev_mle.likelihood import Observations, MeasurementModel
from . import passive


def fixture(n=6, t=48, seed=12031):
    rng=np.random.default_rng(seed)
    p=rng.uniform(.015,.06,(t,n));q=p*rng.uniform(.2,.45,(t,n))
    family=tuple(s for s in ((0,1),(2,3),(0,1,2,3)) if len(s)<n)
    y=.008*(p+.5*q).sum(axis=1)[:,None]*np.ones((1,n))
    for s in family:
        y[:,s] += .025*(p[:,s]+.5*q[:,s]).sum(axis=1)[:,None]
    y += .035*(p+.5*q)
    bp=np.linspace(-.0001,.0001,n);bq=-bp/2;bv=bp/4
    meter=MeasurementModel(.0006,.0004,.00012,bias_p=bp,bias_q=bq,bias_v=bv)
    obs=Observations(p+bp,q+bq,np.sqrt(1-y)+bv,root_v=1.)
    return obs,meter,(p,q,y)


def signature(candidates):
    return [c.supports for c in candidates]


def assert_laminar(candidates):
    for c in candidates:
        supports=list(map(set,c.supports))
        assert all(not a&b or a<=b or b<=a for a in supports for b in supports)
        assert len(set(c.supports)) == len(c.supports)


def test_calibration_before_squared_voltage_target():
    obs,meter,expected=fixture()
    calibrated=passive._calibrated(obs,meter)
    for actual,want in zip(calibrated,expected):
        np.testing.assert_allclose(actual,want,atol=4e-16,rtol=1e-13)


@pytest.mark.parametrize("n",[6,10])
def test_real_rnj_and_core_bootstrap_on_supported_sizes(n):
    obs,meter,_=fixture(n=n)
    old=tuple(getattr(obs,k).copy() for k in ("p","q","v"))
    bank,record=passive.propose(obs,meter,
        dict(run_wzzt=False,bootstrap_replicates=1,max_candidates=6),seed=101)
    assert 1<=len(bank)<=6
    assert record["native"]["rnj"]
    assert record["bootstrap"][0]["families"]
    assert record["budget"]["bootstrap_replicates_with_proposals"]==1
    assert all(np.array_equal(a,getattr(obs,k)) for a,k in zip(old,("p","q","v")))
    assert_laminar(bank)
    assert not record["extra_likelihood"] and not record["ranking_is_AC_evidence"]
    assert record["source_inputs_empty"] and not record["source_estimated_by_wzzt"]
    assert record["generated_count"]==record["retained_count"]+record["truncated_count"]
    json.dumps(record,allow_nan=False)


def test_true_labels_and_future_fields_do_not_change_proposals():
    obs,meter,_=fixture(n=4,t=24)
    fields={k:getattr(obs,k) for k in ("p","q","v","root_v")}
    a=SimpleNamespace(**fields,truth="star",future_p=np.full((8,4),100.))
    b=SimpleNamespace(**fields,truth="different_hidden_tree",future_p=np.full((8,4),-100.))
    config=dict(run_wzzt=False,bootstrap_replicates=1,max_candidates=5)
    first,record1=passive.propose(a,meter,config,seed=22)
    second,record2=passive.propose(b,meter,config,seed=22)
    assert signature(first)==signature(second)
    assert record1["bank"]==record2["bank"]
    assert not record1["truth_seen"] and not record1["future_seen"]
    params=inspect.signature(passive.propose).parameters
    assert list(params)==["observations","meter","config","seed"]
    with pytest.raises(ValueError,match="unknown passive"):
        passive.propose(obs,meter,dict(truth="forbidden"),seed=22)


def test_wzzt_receives_empty_source_and_no_certified_clades(monkeypatch):
    obs,meter,_=fixture(n=4,t=18)
    captured={}
    def failed(*args,**kwargs):
        captured.update(kwargs)
        raise RuntimeError("injected WZZT failure")
    monkeypatch.setattr(passive.bridge,"fit_source_aware_wzzt",failed)
    bank,record=passive.propose(obs,meter,dict(bootstrap_replicates=0,max_candidates=3),seed=0)
    assert bank and record["native"]["diagnostics"]["wzzt"]["called"]
    assert captured["source_inputs"] is None
    assert captured["certified_supports"]==()
    assert any(f["stage"]=="native_wzzt" for f in record["failures"])
    assert all(r["replaceable"] and not r["certified_supports"] for r in record["bank"])


def test_real_wzzt_is_called_and_keeps_native_diagnostics():
    obs,meter,_=fixture(n=4,t=12)
    bank,record=passive.propose(obs,meter,dict(bootstrap_replicates=0,
        wzzt_time_limit=.4,wzzt_max_atoms=5,max_candidates=5),seed=10)
    native=record["native"]["diagnostics"]["wzzt"]
    assert native["called"] and "error" not in native
    assert native["solver_attempts"]
    assert native["mandatory_supports"]==[[i] for i in range(4)]
    assert not native["global_topology_optimality"]
    assert record["native"]["wzzt"]
    assert_laminar(bank)


def test_repairs_are_limited_and_all_clades_replaceable():
    seeds=[((0,1,2),(0,1),(3,4))]
    observed=[*seeds,((0,1,2),(1,2)),((3,4,5),(3,4))]
    repairs=passive._repair_families(seeds,observed,6,4)
    assert repairs and len(repairs)<=4
    assert any(op=="delete_clade" for _,op in repairs)
    assert any(op in ("replace_disputed_clade","merge_observed_clade") for _,op in repairs)
    for family,_ in repairs:
        assert passive.bridge.normalize_family(family,6)==family
        assert family not in observed
    # Even a unanimously proposed internal edge can be deleted.
    stable=[((0,1,2),(0,1))]
    assert any(len(f)<len(stable[0]) for f,_ in passive._repair_families(stable,stable,6,2))


@pytest.mark.parametrize("config",[dict(max_candidates=0),dict(bootstrap_replicates=-1),
    dict(block_length=0),dict(kappa=-.2),dict(wzzt_time_limit=0),dict(tolerance_factors=())])
def test_invalid_public_options_fail_before_solver(config):
    obs,meter,_=fixture(n=4,t=12)
    with pytest.raises(ValueError):
        passive.propose(obs,meter,config)


def test_invalid_crossing_or_uncalibrated_input_rejected():
    with pytest.raises(ValueError,match="laminar"):
        passive._family(((0,1),(1,2)),4)
    obs,meter,_=fixture(n=4,t=12)
    meter=MeasurementModel(meter.p_sigma,meter.q_sigma,meter.v_sigma,fit_bias=True)
    with pytest.raises(ValueError,match="fixed independently calibrated"):
        passive.propose(obs,meter)
