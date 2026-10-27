"""Observed-data and diagnostics contracts, plus one bounded real-core check."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_experiments.wzzt_dr_selection.end_to_end import wzzt_adapter as adapter
core = adapter.core


def observations(name="train", seed=123, n=2, rows=12, r=None, x=None):
    rng = np.random.default_rng(seed)
    labels = pd.Index([f"meter_{i+4}" for i in range(n)])
    index = pd.date_range("2020-01-01", periods=rows, freq="h")
    p = pd.DataFrame(rng.uniform(.005, .03, size=(rows,n)), index=index, columns=labels)
    q = pd.DataFrame(rng.uniform(.001, .02, size=(rows,n)), index=index, columns=labels)
    r = np.eye(n)*.2 if r is None else r
    x = np.eye(n)*.05 if x is None else x
    root = pd.Series(1.02 + .0001*np.arange(rows), index=index)
    y = p.to_numpy() @ r.T + q.to_numpy() @ x.T
    v = pd.DataFrame(np.sqrt(root.to_numpy()[:,None]**2-y), index=index, columns=labels)
    return adapter.make_observed_scenario(p,q,v,root,name),r,x


def fake_result(train, valid, *, stop="no_significant_one_atom_gain", timeout=False):
    _,r,x = observations()
    labels=tuple(train[0]["P_terminal"].columns)
    supports=tuple((i,) for i in range(len(labels)))
    train_mae,_,_=core.evaluate_l1_matrices(train,r,x)
    valid_mae,valid_se,_=core.evaluate_l1_matrices(valid,r,x)
    diag=core.SolverDiagnostics(status=0,success=True,message="contract stub LP",objective=train_mae,
        dual_bound=None,mip_gap=None,node_count=None,runtime_seconds=0.,variable_count=10,
        binary_variable_count=0,constraint_count=5)
    point=core.LaminarL1PathPoint(iteration=0,support_indices=supports,
        support_labels=tuple((label,) for label in labels),r_values=np.diag(r),x_values=np.diag(x),
        r_matrix=r,x_matrix=x,train_mae=train_mae,validation_mae=valid_mae,validation_se=valid_se,
        accepted_gain=None,solver=diag)
    attempts=()
    if timeout:
        attempted=core.SolverDiagnostics(status=1,success=False,message="contract stub time limit",
            objective=.01,dual_bound=.001,mip_gap=.9,node_count=5,runtime_seconds=.001,
            variable_count=30,binary_variable_count=2,constraint_count=40)
        attempts=(core.ExtensionSolution(support=(0,1),r_values=np.array([.2,.2,0.]),
            x_values=np.array([.05,.05,0.]),objective=.01,diagnostics=attempted),)
    return core.LaminarL1Result(terminal_labels=labels,support_indices=supports,
        support_labels=point.support_labels,r_values=point.r_values,x_values=point.x_values,
        r_matrix=r,x_matrix=x,train_mae=train_mae,validation_mae=valid_mae,selected_path_index=0,
        path=(point,),attempted_extensions=attempts,stop_reason=stop,r_upper_bound=2.,
        x_upper_bound=2.,bound_expansions=0)


def test_raw_observed_root_target_and_copy():
    scenario,_,_=observations()
    expected=scenario["V_root_observed"].to_numpy()[:,None]**2-scenario["V_terminal"].to_numpy()**2
    np.testing.assert_array_equal(scenario["drop_target"],expected)
    assert abs(scenario["drop_target"].to_numpy().mean()) > 1e-4
    assert scenario["units"]["drop_target"] == "pu2"


@pytest.mark.parametrize("defect",["time","columns","root_time","nonfinite","negative_voltage"])
def test_meter_alignment_and_measurement_defects_rejected(defect):
    s,_,_=observations()
    p,q,v,root=(s[k].copy() for k in ("P_terminal","Q_terminal","V_terminal","V_root_observed"))
    if defect=="time": q=q.iloc[::-1]
    if defect=="columns": v=v.iloc[:,::-1]
    if defect=="root_time": root=root.iloc[:-1]
    if defect=="nonfinite": p.iloc[0,0]=np.nan
    if defect=="negative_voltage": v.iloc[0,0]=-1.
    with pytest.raises(ValueError): adapter.make_observed_scenario(p,q,v,root,"invalid")


@pytest.mark.parametrize("defect",["target","unit","validation_reuse"])
def test_fit_contract_rejects_invalid_input_before_core(monkeypatch,defect):
    train,_,_=observations()
    valid,_,_=observations("validation",456)
    def forbidden(*args,**kwargs): raise AssertionError("invalid input reached core")
    monkeypatch.setattr(core,"fit_laminar_l1_sensitivity",forbidden)
    if defect=="target": train["drop_target"].iloc[0,0]+=1.
    if defect=="unit": train["units"]["P_terminal"]="MW"
    if defect=="validation_reuse": valid["name"]="train"
    with pytest.raises(ValueError): adapter.fit_observed_wzzt([train],[valid])


def test_actual_entry_called_once_without_truth_or_fixed_extra_blocks(monkeypatch,tmp_path):
    train,_,_=observations()
    valid,_,_=observations("validation",456)
    train["truth_supports"]=[(0,1)]
    called=[]
    def spy(scenarios,**kw):
        called.append((scenarios,kw))
        return fake_result(scenarios,kw["validation_scenarios"])
    monkeypatch.setattr(core,"fit_laminar_l1_sensitivity",spy)
    path=tmp_path/"result.json"
    result=adapter.fit_observed_wzzt([train],[valid],checkpoint_path=path)
    assert len(called)==1
    forwarded,kwargs=called[0]
    assert "truth_supports" not in forwarded[0]
    assert kwargs["initial_supports"]==[(0,),(1,)]
    assert kwargs["max_atoms"]==3 and kwargs["mip_rel_gap"]==0.
    assert kwargs["r_upper_bound"]==kwargs["x_upper_bound"]==2.
    assert result["status"]=="complete" and result["has_model"]
    assert not result["rnj_stage_called"] and not result["global_tree_optimality_claim"]
    assert result["selected"]["solver"]["numerical_optimum_certified"]
    assert json.loads(path.read_text())["summary"]["selected_path_index"]==0


def test_timeout_keeps_certified_fit_and_unproved_attempt(monkeypatch):
    train,_,_=observations()
    valid,_,_=observations("validation",456)
    monkeypatch.setattr(core,"fit_laminar_l1_sensitivity",
        lambda s,**kw: fake_result(s,kw["validation_scenarios"],stop="extension_not_proven_optimal",timeout=True))
    result=adapter.fit_observed_wzzt([train],[valid])
    summary=adapter.summarize_fit(result)
    assert result["status"]=="incomplete" and result["has_model"]
    assert summary["unproved_attempt_count"]==1 and summary["last_attempt_status"]==1
    assert summary["last_attempt_mip_gap"]==.9
    assert result["selected"]["solver"]["numerical_optimum_certified"]


def test_failure_preserves_preceding_progress_checkpoint(monkeypatch,tmp_path):
    train,_,_=observations()
    valid,_,_=observations("validation",456)
    def fail(s,**kw):
        fit=fake_result(s,kw["validation_scenarios"])
        kw["progress_callback"]({"event":"initialized","path":fit.path,"attempted_extensions":(),
            "bound_expansions":0,"stop_reason":"in_progress"})
        raise RuntimeError("simulated fixed-support LP failure")
    monkeypatch.setattr(core,"fit_laminar_l1_sensitivity",fail)
    path=tmp_path/"failure.json"
    result=adapter.fit_observed_wzzt([train],[valid],checkpoint_path=path)
    assert result["status"]=="failed" and not result["has_model"]
    assert len(result["last_certified_path"])==1
    assert json.loads(path.read_text())["exception_type"]=="RuntimeError"


def test_truth_evaluation_uses_labels_and_ignores_zero_support(monkeypatch):
    train,_,_=observations()
    valid,_,_=observations("validation",456)
    monkeypatch.setattr(core,"fit_laminar_l1_sensitivity",lambda s,**kw:fake_result(s,kw["validation_scenarios"]))
    result=adapter.fit_observed_wzzt([train],[valid])
    result["selected"]["support_labels"].append(["meter_4","meter_5"])
    result["selected"]["r_values"].append(0.)
    result["selected"]["x_values"].append(0.)
    evaluated=adapter.evaluate_effective_supports(result,[("meter_4",),("meter_5",)])
    assert evaluated["full_effective_family_recovered"]
    assert evaluated["all_effective_supports"]["predicted_count"]==2
    assert not evaluated["probability_claim"]
    with pytest.raises(ValueError): adapter.evaluate_effective_supports(result,[(0,)])


def test_actual_core_four_terminal_free_support_run(tmp_path):
    n=4
    z=np.array([1.,1.,0.,0.])
    r=np.eye(n)*.2+.1*np.outer(z,z)
    x=np.eye(n)*.05+.02*np.outer(z,z)
    train,_,_=observations("train",321,n=n,rows=16,r=r,x=x)
    valid,_,_=observations("validation",654,n=n,rows=16,r=r,x=x)
    result=adapter.fit_observed_wzzt([train],[valid],{"time_limit_seconds":3.},tmp_path/"actual_core.json")
    assert result["has_model"] and result["status"]=="complete",adapter.summarize_fit(result)
    assert result["summary"]["path_length"]>=2
    assert result["summary"]["train_mae"] < 1e-9
    evaluation=adapter.evaluate_effective_supports(result,[("meter_4",),("meter_5",),("meter_6",),("meter_7",),("meter_4","meter_5")])
    assert evaluation["full_effective_family_recovered"],evaluation
