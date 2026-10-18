"""Reference-network statistics and zero-edge diagnostics; no scientific fits."""
from copy import deepcopy
from hashlib import sha256
import json
import math

import numpy as np
import pytest

from .. import likelihood as lk
from . import analysis as a


def entry(candidate,loss,success=True):
    return dict(loss=float(loss),success=success,usable=True,
        state=dict(r=[.01]*candidate.m,x=[.005]*candidate.m,
                   source_p=[0.]*8,source_q=[0.]*8))


def record(index=0,phase="calibration",network="fixture_a",source="none",delta=0.):
    star=lk.RadialCandidate.from_supports("star",4,[])
    cherry=lk.RadialCandidate.from_supports("cherry",4,[[0,1]])
    structures=[dict(name=c.name,n=4,supports=c.supports) for c in (star,cherry)]
    comparisons={"star":dict(fixed=entry(star,8.),joint=entry(star,8.)),
        "cherry":dict(fixed=entry(cherry,8.+2*delta),joint=entry(cherry,6.))}
    action=dict(methods=list(a.METHODS),pre_refit=dict(comparisons=comparisons),
        post_refit=None,candidate_union=structures,failures={})
    return dict(spec=dict(phase=phase,cluster_index=index,n=4,network=network,source_mode=source),
        config=dict(zero_equivalence_tolerance_pu=1e-8),
        truth_evaluation_only=dict(supports=cherry.supports,r=[.01]*cherry.m,x=[.005]*cherry.m),
        candidate_structures=structures,
        planning=dict(history=dict(candidates={"star":entry(star,8.),"cherry":entry(cherry,8.+2*delta)})),
        evaluation=dict(actions={"same":action}),status="complete")


def config(count=1):
    return dict(network_names=["fixture_a","fixture_b"],
        network_observation_counts=dict(fixture_a=4,fixture_b=4),
        phase_counts=dict(development=count,calibration=count,evaluation=count),
        source_modes=["none","PV"],methods=["P-SQ","G-SQ"],alpha=.1,
        statistical_bootstrap=3,statistical_seed=123)


def test_new_condition_manifest_not_legacy_counts():
    cfg=config(2)
    rows=a.condition_manifest(cfg,"evaluation")
    assert len(rows)==8 and {r["network"] for r in rows}=={"fixture_a","fixture_b"}
    assert {r["n"] for r in rows}=={4}
    assert "terminal_counts" not in cfg and "evaluation_clusters" not in cfg


def test_missing_record_preserves_declared_network_and_denominator(tmp_path):
    cfg=config();a.io.write(tmp_path/"evaluation/one.json.gz",record(phase="evaluation"))
    audit=a.old.Audit();rows=a._load_phase(tmp_path,"evaluation",cfg,audit)
    assert len(rows)==4
    assert sum(r["status"]=="missing" for r in rows)==3
    assert {r["spec"]["network"] for r in rows}=={"fixture_a","fixture_b"}
    assert all(r["spec"]["n"]==4 for r in rows)
    assert not audit.record()["passed"]


def test_network_separate_thresholds_same_measurement_count_and_source_blind():
    records=[record(i,network=network,source=source,delta=i*scale)
        for i in range(9) for network,scale in (("fixture_a",1),("fixture_b",4))
        for source in ("none","PV")]
    thresholds=a.calibration_thresholds(records,methods=("G-SQ",),source_modes=("none","PV"))
    ka=a.source_blind_key("G-SQ",4,"preFIX","fixture_a")
    kb=a.source_blind_key("G-SQ",4,"preFIX","fixture_b")
    assert thresholds["source_blind_applied"][ka]["value"]==8
    assert thresholds["source_blind_applied"][kb]["value"]==32
    first=record(20,phase="evaluation",source="none");second=deepcopy(first);second["spec"]["source_mode"]="PV"
    x=a.decision_rows([first],thresholds,methods=("G-SQ",))
    y=a.decision_rows([second],thresholds,methods=("G-SQ",))
    assert [r["threshold_value"] for r in x]==[r["threshold_value"] for r in y]
    with pytest.raises(ValueError,match="calibration records only"):
        a.calibration_thresholds([first])


def test_pairing_does_not_merge_equal_n_different_networks():
    rows=a.decision_rows([record(network="fixture_a"),record(network="fixture_b")],{},methods=("P-SQ","G-SQ"))
    pairs=a.paired_rows(rows)
    assert len(pairs)==2 and {p["network"] for p in pairs}=={"fixture_a","fixture_b"}


def test_zero_edges_have_absolute_and_positive_only_relative_errors():
    raw=record();truth=raw["truth_evaluation_only"]
    truth["r"][0]=0.;truth["x"][0]=0.
    view=a.route_view(raw,"G-SQ","preJNT")
    state=view["entries"]["cherry"]["state"]
    state["r"]=[.011]*len(state["r"]);state["x"]=[.0055]*len(state["x"])
    state["r"][0]=.003;state["x"][0]=.004
    out=a.rx_error(raw,view)
    assert out["available"] and out["virtual_zero_edge_count"]==1
    assert out["r"]["n_zero_edges"]==1 and out["r"]["n_positive_edges"]==len(state["r"])-1
    assert out["r"]["mean_absolute_relative"]==pytest.approx(.1)
    assert out["r"]["zero_truth_edges_absolute_pu"]["mean"]==.003
    assert out["x"]["zero_truth_edges_absolute_pu"]["mean"]==.004
    assert out["r"]["rmse_pu"]>0


def test_closed_cone_accepts_virtual_zero_without_division():
    assert a.cone_contains([0.,.01],[0.,.005],(.1,1.))
    assert not a.cone_contains([0.,.01],[.01,.005],(.1,1.))
    assert not a.cone_contains([.01],[.02],(.1,1.))
    assert not a.cone_contains([np.nan],[0.],(.1,1.))


def test_final_clade_f1_and_failure_denominator():
    raw=record(delta=2.)
    fixed=a.final_topology_metrics(raw,"G-SQ","preFIX")
    joint=a.final_topology_metrics(raw,"G-SQ","preJNT")
    assert fixed["available"] and fixed["f1"]==0 and not fixed["exact"]
    assert joint["available"] and joint["f1"]==1 and joint["exact"]
    failed=deepcopy(raw)
    failed["evaluation"]["actions"]["same"]["pre_refit"]["comparisons"]["star"]["joint"]["success"]=False
    other=a.final_topology_metrics(failed,"G-SQ","preJNT")
    summary=a.summarize_clades([joint,other],bootstrap=3)
    assert summary["n_total_conditions"]==2 and summary["scored"]==1 and summary["unscored"]==1
    assert summary["f1"]["mean"]==1. and summary["exact_rate_all"]==.5


def test_service_voltage_failure_not_ac_optimizer_failure():
    raw=record();act=raw["evaluation"]["actions"]["same"]
    act["physical"]={"voltage_domain_satisfied":True,"service_voltage_09_11_satisfied":False}
    act["failures"]["actual_service_voltage"]="executed service bound failed"
    hist=a.decide(a.route_view(raw,"G-SQ","history"),0.)
    out=a.decide(a.route_view(raw,"G-SQ","postJNT"),0.)
    assert hist["available"] and not hist["physical_failure"]
    assert out["service_voltage_failure"] and out["physical_failure"] and out["unresolved"]
    assert not out["calculation_domain_failure"] and not out["optimization_failure"]
    assert act["pre_refit"]["comparisons"]["cherry"]["joint"]["success"]


def test_structural_refinement_and_contraction_do_not_replace_strict_label():
    raw=record();action=raw["evaluation"]["actions"]["same"]
    shape=next(s for s in raw["candidate_structures"] if s["name"]=="cherry")
    c=lk.RadialCandidate.from_supports("cherry",4,shape["supports"])
    k=c.supports.index((0,1))
    action["pre_refit"]["comparisons"]["cherry"]["joint"]["state"]["r"][k]=0.
    action["pre_refit"]["comparisons"]["cherry"]["joint"]["state"]["x"][k]=0.
    truth=lk.RadialCandidate.from_supports("true_star",4,[])
    raw["truth_evaluation_only"]=dict(supports=truth.supports,r=[.01]*truth.m,x=[.005]*truth.m)
    # Strict star candidate is absent; the refined cherry can represent it.
    raw["candidate_structures"]=[shape];action["candidate_union"]=[shape]
    view=a.route_view(raw,"G-SQ","postJNT");decision=a.decide(view,0.)
    diag=a.zero_equivalence_diagnostic(raw,view)
    final=a.final_topology_metrics(raw,"G-SQ","postJNT")
    assert not decision["truth_present"] and decision["unique_wrong"]
    assert diag["structural_refinement_coverage"] and diag["fitted_contraction_truth_coverage"]
    assert final["predicted"]=="cherry" and not final["exact"]
    assert final["contracted"]["exact"] and final["contracted"]["removed_count"]==1
    assert a.decide(view,0.)==decision
    assert diag["tolerance_impedance_magnitude_pu"]==1e-8


def test_phase_tables_are_by_network_and_have_final_f1():
    raw=[record(network="fixture_a"),record(network="fixture_b")]
    summary=a.phase_summary(raw,{},("P-SQ","G-SQ"),bootstrap=3,seed=1)
    assert summary["n_condition_records"]==2
    assert set(summary["by_network"])=={"fixture_a","fixture_b"}
    for row in summary["by_network"].values():
        final=row["tables"]["G-SQ"]["postJNT"]
        assert final["n_total_conditions"]==1 and final["clades"]["f1"]["mean"]==1.


def test_four_proxy_domains_do_not_invent_unexecuted_ac_actions():
    raws=[record(index=i,phase="evaluation",network="fixture_b") for i in range(2)]
    for raw in raws:
        raw["planning"]["families"]={"P-SQ":{
            d:dict(score=s) for d,s in zip(("fixed_charge","fixed_bidir","mobile_charge","mobile_bidir"),(1.,2.,3.,4.))}}
    result=a.small_spatial_ablation(raws,{},bootstrap=3)
    assert result["n_condition_records"]==2 and result["n_date_clusters"]==2
    assert len(result["proxy_parent_child_comparisons"]["P-SQ"])==4
    assert all(x["n_paired_scores"]==2 and x["decrease"]==0 for x in result["proxy_parent_child_comparisons"]["P-SQ"].values())
    assert result["actual_fixed_bidir_to_optional_mobile"]["P-SQ"]["n_all_pairs"]==2
    assert not result["charge_only_actual_comparison_available"]
    assert set(result["charge_only_actual_record_counts"].values())=={0}


def _frozen(tmp_path,cfg):
    payload=dict(config=cfg,source_sha256={},selector_reads_truth=False)
    identity=sha256(json.dumps(a.io.serial(payload),sort_keys=True).encode()).hexdigest()
    a.io.write(tmp_path/"freeze.json",dict(payload,identity=identity))


def test_finish_new_config_freezes_before_opening_evaluation(tmp_path,monkeypatch):
    cfg=config();_frozen(tmp_path,cfg)
    for phase in cfg["phase_counts"]:
        for name in cfg["network_names"]:
            for source in cfg["source_modes"]:
                a.io.write(tmp_path/phase/(name+"_"+source+".json.gz"),record(phase=phase,network=name,source=source))
    original=a.io.read;opened=[]
    def guarded(path):
        if "evaluation" in str(path):
            assert (tmp_path/"thresholds.json").exists()
        opened.append(str(path));return original(path)
    monkeypatch.setattr(a.io,"read",guarded)
    result,audit=a.finish(tmp_path,bootstrap=2)
    assert audit["passed"] and audit["forward_AC_replay_calls"]==0
    assert result["expected_conditions_by_phase"]=={p:4 for p in cfg["phase_counts"]}
    assert result["phases"]["evaluation"]["n_condition_records"]==4
    assert result["declared_network_sizes"]=={"fixture_a":4,"fixture_b":4}
    text=(tmp_path/"results.md").read_text(encoding="utf-8")
    assert "fixture_a" in text and "6/8" not in text and "64-condition" not in text


def test_freeze_rejects_wrong_manifest_even_equal_file_count(tmp_path):
    cfg=config();_frozen(tmp_path,cfg)
    for i in range(4):
        raw=record(index=i,network="fixture_a")
        a.io.write(tmp_path/"calibration"/(str(i)+".json.gz"),raw)
    with pytest.raises(RuntimeError,match="condition manifest"):
        a.freeze_thresholds(tmp_path,expected_conditions=4)
