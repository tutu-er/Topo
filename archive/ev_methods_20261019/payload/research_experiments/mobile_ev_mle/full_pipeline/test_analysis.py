"""Pure statistical and evidence-boundary tests; no fitting or planning."""
from copy import deepcopy
import math

import numpy as np
import pytest

from . import analysis as a


def entry(loss,success=True):
    return dict(loss=loss,success=success,usable=True,
        state=dict(r=[.01]*5,x=[.005]*5,source_p=[0.]*8,source_q=[0.]*8))


def record(index=0,phase="calibration",truth=((0,1),),true_pre_loss=12.):
    structures=[dict(name="star",n=4,supports=[]),dict(name="cherry",n=4,supports=[[0,1]])]
    comparisons={"star":dict(fixed=entry(8.),joint=entry(8.)),
        "cherry":dict(fixed=entry(true_pre_loss),joint=entry(6.))}
    action=dict(methods=list(a.METHODS),pre_refit=dict(comparisons=comparisons),
        post_refit=None,candidate_union=structures)
    return dict(spec=dict(phase=phase,cluster_index=index,n=4,source_mode="DER",seed=index),
        truth_evaluation_only=dict(supports=truth,r=[],x=[]),candidate_structures=structures,
        planning=dict(history=dict(candidates={"star":entry(8.),"cherry":entry(true_pre_loss)})),
        evaluation=dict(actions={"same":action}))


def test_clade_metrics_count_entire_nontrivial_tree():
    truth=[list(range(6)),[0,1,2],[0,1],[3,4],*[ [j] for j in range(6) ]]
    predicted=[[0,1,2],[0,1],[3,5]]
    result=a.clade_metrics(predicted,truth,6)
    assert not result["exact"] and result["matched_clades"]==2
    assert result["f1"]==pytest.approx(2/3)
    assert a.clade_metrics([],[],6)["f1"]==1
    assert a.signature([[1,0],[0,1],list(range(6)),[0]],6)==((0,1),)
    with pytest.raises(ValueError,match="crossing"):
        a.signature([[0,1],[1,2]],6)


def test_half_sse_calibration_order_and_shortage():
    threshold=a.threshold_quantile(range(12),.1)
    assert threshold["finite"] and threshold["value"]==11
    assert threshold["order_statistic_1_based"]==12
    short=a.threshold_quantile(range(8),.1)
    assert not short["finite"] and a.threshold_value(short)==math.inf
    assert a.threshold_quantile(range(9),.1)["value"]==8
    with pytest.raises(ValueError):a.threshold_quantile([-1.])


def test_fixed_route_survives_joint_failure():
    raw=record()
    raw["evaluation"]["actions"]["same"]["pre_refit"]["comparisons"]["cherry"]["joint"]["success"]=False
    assert a.route_view(raw,"G-SQ","preFIX")["fit_available"]
    joint=a.decide(a.route_view(raw,"G-SQ","preJNT"),0.)
    assert joint["fit_failure"] and joint["unresolved"]
    assert joint["retained"]==["cherry","star"] and not joint["false_reject"]


def test_actual_voltage_failure_blocks_only_future_routes():
    raw=record()
    raw["evaluation"]["actions"]["same"]["physical"]=dict(voltage_domain_satisfied=False)
    assert a.route_view(raw,"G-SQ","history")["fit_available"]
    for route in ("preFIX","preJNT","postJNT"):
        row=a.decide(a.route_view(raw,"G-SQ",route),0.)
        assert row["physical_failure"] and row["unresolved"] and row["fit_failure"]
        assert not row["optimization_failure"]
        assert row["retained"]==["cherry","star"]


def test_true_delta_and_margin_have_opposite_reference():
    view=a.route_view(record(),"G-SQ","preFIX")
    decision=a.decide(view,1.)
    assert decision["Delta"]==2
    assert decision["true_margin_half_SSE"]==-2
    assert decision["point_error"] and decision["false_reject"]
    assert decision["unique_wrong"]
    broad=a.decide(view,math.inf)
    assert broad["point_error"] and broad["unresolved"] and not broad["false_reject"]
    correct=a.decide(a.route_view(record(),"G-SQ","preJNT"),0.)
    assert correct["unique_correct"] and correct["true_margin_half_SSE"]==1


def test_post_union_merges_old_and_only_new_refits():
    raw=record(truth=((2,3),))
    action=raw["evaluation"]["actions"]["same"]
    action["candidate_union"]=deepcopy(action["candidate_union"])+[dict(name="new",n=4,supports=[[2,3]])]
    action["post_refit"]=dict(comparisons={"new":dict(joint=entry(4.))})
    assert not a.route_view(raw,"P-SQ","preJNT")["truth_present"]
    post=a.route_view(raw,"P-SQ","postJNT")
    assert post["truth_present"] and post["true_name"]=="new"
    assert set(post["losses"])=={"star","cherry","new"}
    assert a.decide(post,0.)["unique_correct"]
    action["post_refit"]["comparisons"]["new"]["joint"]["success"]=False
    assert a.decide(a.route_view(raw,"P-SQ","postJNT"),0.)["retained_count"]==3


def test_duplicate_post_comparison_rejected():
    raw=record();raw["evaluation"]["actions"]["same"]["post_refit"]=dict(comparisons={"star":dict(joint=entry(1))})
    with pytest.raises(ValueError,match="only newly"):
        a.route_view(raw,"P-SQ","postJNT")


def test_calibration_reads_only_calibration_and_preserves_failed_denominator():
    records=[record(j,true_pre_loss=8+2*j) for j in range(12)]
    frozen=a.calibration_thresholds(records)
    cell=frozen["cells"]["G-SQ|4|DER|preFIX"]
    assert cell["n_all"]==12 and cell["n"]==12 and cell["value"]==11
    assert frozen["n_independent_date_clusters"]==12
    with pytest.raises(ValueError,match="calibration records only"):
        a.calibration_thresholds([record(phase="evaluation")])
    with pytest.raises(ValueError,match="duplicate"):
        a.calibration_thresholds([records[0],records[0]])
    for raw in records[:4]:
        raw["evaluation"]["actions"]["same"]["pre_refit"]["comparisons"]["star"]["fixed"]["success"]=False
    short=a.calibration_thresholds(records)["cells"]["G-SQ|4|DER|preFIX"]
    assert short["n_all"]==12 and short["fit_failure"]==4 and short["n"]==8
    assert not short["finite"]


def test_applied_threshold_is_source_blind_max_not_oracle_routing():
    records=[]
    for j in range(12):
        records.append(record(j,true_pre_loss=8+2*j))
        none=record(j,true_pre_loss=8+4*j);none["spec"]["source_mode"]="none";records.append(none)
    frozen=a.calibration_thresholds(records,source_modes=("none","DER"))
    applied=frozen["source_blind_applied"]["G-SQ|4|preFIX"]
    assert applied["value"]==22 and applied["n_independent_date_clusters"]==12
    assert frozen["cells"]["G-SQ|4|DER|preFIX"]["value"]==11
    assert frozen["cells"]["G-SQ|4|none|preFIX"]["value"]==22
    left=record(21,phase="evaluation");right=deepcopy(left);right["spec"]["source_mode"]="none"
    rows1=a.decision_rows([left],frozen,methods=("G-SQ",))
    rows2=a.decision_rows([right],frozen,methods=("G-SQ",))
    assert [r["threshold_value"] for r in rows1]==[r["threshold_value"] for r in rows2]
    assert [r["retained"] for r in rows1]==[r["retained"] for r in rows2]
    assert all(not r["source_label_used_for_threshold"] for r in rows1+rows2)
    # A missing preregistered source stratum must not silently disappear.
    missing=a.calibration_thresholds(records[::2],source_modes=("none","DER"))
    assert not missing["source_blind_applied"]["G-SQ|4|preFIX"]["finite"]


def test_missing_truth_separate_from_optimization_failure_and_point_error():
    raw=record(truth=((2,3),))
    row=a.decide(a.route_view(raw,"P-SQ","preJNT"),0.)
    assert not row["truth_present"] and row["point_error"]
    assert row["conditional_point_error"] is None and row["false_reject"] is None
    row.update(cluster_index=0)
    summary=a.summarize_decisions([row])
    assert summary["candidate_missing"]==1 and summary["fit_failure"]==0
    assert summary["top1_wrong"]==1 and summary["conditional_top1_evaluable"]==0


def test_numeric_ties_remain_unresolved_at_zero_threshold():
    raw=record();comparison=raw["evaluation"]["actions"]["same"]["pre_refit"]["comparisons"]
    comparison["cherry"]["fixed"]["loss"]=8.
    decision=a.decide(a.route_view(raw,"P-SQ","preFIX"),0.)
    assert decision["numerical_tie"] and decision["predicted"] is None
    assert decision["unresolved"] and not decision["unique_wrong"]


def test_date_cluster_bootstrap_keeps_four_conditions_together():
    rows=[dict(cluster_index=cluster,n=n,source_mode=s,value=float(cluster))
        for cluster in range(2) for n in (6,8) for s in ("none","DER")]
    def statistic(sample):
        counts={cluster:sum(r["cluster_index"]==cluster for r in sample) for cluster in range(2)}
        assert all(v%4==0 for v in counts.values())
        return np.mean([r["value"] for r in sample])
    result=a.cluster_bootstrap(rows,statistic,replicates=100)
    assert result["n_clusters"]==2 and result["estimate"]==.5
    assert result["valid_replicates"]==100
    assert result["lower"]==0. and result["upper"]==1.


def test_pairing_uses_cluster_and_condition_and_keeps_failures():
    raw=[record(0,phase="evaluation"),record(1,phase="evaluation")]
    rows=a.decision_rows(raw,{"cells":{}})
    # Same actual action has the same inference, but counts as its own
    # hypothetical deployment for each method; not extra independent data.
    summary=a.paired_summary(rows,replicates=20)
    assert summary["n_all_pairs"]==2 and summary["n_conditional_point_pairs"]==2
    assert summary["conditional_point_error_delta"]["estimate"]==0.
    assert summary["conditional_point_error_delta"]["n_clusters"]==2
    table=a.summarize_decisions([r for r in rows if r["method"]=="P-SQ" and r["route"]=="postJNT"])
    assert table["n_total_conditions"]==2 and table["unresolved"]==2


def test_freeze_calibration_never_reads_evaluation(tmp_path,monkeypatch):
    from . import io
    output=tmp_path/"run"
    io.write(output/"freeze.json",dict(identity="fixed-policy"))
    for j in range(12):io.write(output/"calibration"/("history_%03d.json.gz"%j),record(j))
    io.write(output/"evaluation"/"oracle.json.gz",dict(forbidden_future="must not be opened"))
    read=io.read;opened=[]
    def checked(path):
        opened.append(str(path))
        assert "evaluation" not in str(path)
        return read(path)
    monkeypatch.setattr(io,"read",checked)
    first=a.freeze_thresholds(output,expected_conditions=12)
    second=a.freeze_thresholds(output,expected_conditions=12)
    assert first==second and (output/"thresholds.json").exists()
    assert first["n_independent_date_clusters"]==12
    with pytest.raises(RuntimeError,match="calibration incomplete"):
        a.freeze_thresholds(output,expected_conditions=13)


def test_phase_summary_keeps_new_candidate_and_computation_separate():
    records=[record(j,phase="evaluation") for j in range(2)]
    thresholds={"cells":{}}
    result=a.phase_summary(records,thresholds,("P-SQ","G-SQ"),bootstrap=3,seed=17)
    assert result["n_condition_records"]==2 and result["n_independent_date_clusters"]==2
    assert result["tables"]["G-SQ"]["preFIX"]["conditional_top1_wrong"]==2
    assert result["tables"]["G-SQ"]["preJNT"]["conditional_top1_wrong"]==0
    assert result["paired_joint_vs_fixed"]["G-SQ"]["right_fixes_left"]==2
    assert result["tables"]["G-SQ"]["preJNT"]["unresolved"]==2
    assert result["tables"]["G-SQ"]["preJNT"]["intervals"]["unique_correct_all"]["seed"]==17


def test_fixed_alias_cost_is_explicit_upper_bound():
    raw=record()
    raw["planning"]["cost"]=dict(passive_pre_seconds=1,history_seconds=2,
        baseline_completion_seconds=.1,random_seconds=.2,P_SQ_seconds=999,
        **{"P-SQ_seconds":3,"response_setup_seconds":.5,"G-SQ_seconds":4,"G-AQ_seconds":5,
           "selection_total_seconds":16.})
    action=raw["evaluation"]["actions"]["same"]
    action["pre_refit"]["seconds"]=6;action["post_passive"]=dict(seconds=2.)
    costs=a.deployment_costs(raw,["G-SQ","G-SQ/F±"])
    mobile=costs["policies"]["G-SQ"];fixed=costs["policies"]["G-SQ/F±"]
    assert mobile["deployment_lower_seconds"] is not None
    assert fixed["fixed_alias_full_family_upper_bound"]
    assert fixed["deployment_lower_seconds"] is None
    assert fixed["selection_component_lower_seconds"] is None
    assert fixed["deployment_upper_seconds"]==mobile["deployment_upper_seconds"]
