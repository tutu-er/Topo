"""Independent fixtures for failure denominators, binding, and objective replay."""
import json

import numpy as np
import pytest

try:
    from . import adaptive_result_audit as audit
except ImportError:
    import adaptive_result_audit as audit


METHODS = ["none", "constant_fixed", "refit_milp"]


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def record(method, losses, *, success=None, selected_failure=False, physical=True):
    return dict(method=method, losses=losses, success=success or [True]*len(losses),
        status="complete" if all(success or [True]*len(losses)) else "joint_fit_failure",
        selection_status="selection_failure_fallback" if selected_failure else "complete",
        operational_voltage_ok=physical, fits=[], action={}, seconds=1., ac_calls=5,
        true_loss=losses[0], true_gap=None if losses[0] is None else losses[0]-min(x for x in losses if x is not None),
        impedance_relative_rmse=.2)


def trial(seed, *, losses=None, true_index=0):
    losses = losses or {method:[30.,32.,31.,31.] for method in METHODS}
    return dict(history_id=str(seed), seed=seed, candidate_names=["star","c01","c02","c12"],
        true_index=true_index, independent_history_unit=True, selection_uses_truth=False,
        selection_uses_future_noise=False, records=[record(method, losses[method]) for method in METHODS],
        selection={}, truth_topology=0, case="strong")


def batch(tmp_path, name, rows, *, policy="policy", source="a"*64, stress=None):
    folder=tmp_path/name; folder.mkdir()
    write_json(folder/"config.json", dict(policy_id=policy, population_id="population",
        source_sha256={"likelihood.py":source}, methods=METHODS, count=len(rows),
        policy={"gates_influence_shadow_policy":False}, stress=stress))
    for row in rows:
        write_json(folder/f"trial_{row['seed']}.json", row)
    return folder


def calibrated_batch(tmp_path):
    return batch(tmp_path,"calibration",[trial(i) for i in range(1,21)])


def test_calibrated_denominators_and_paired_improvement(tmp_path):
    cal=calibrated_batch(tmp_path)
    first=trial(100,losses={"none":[30.,30.,30.,30.],"constant_fixed":[30.,30.,30.,30.],
                          "refit_milp":[25.,60.,60.,60.]})
    failed=trial(101)
    failed["records"][2]=record("refit_milp",[None,30.,40.,40.],success=[False,True,True,True],selected_failure=True)
    omitted=trial(102,true_index=None)
    ev=batch(tmp_path,"evaluation",[first,failed,omitted])
    result=audit.run_audit(cal,ev,tmp_path/"report",replay=False)
    assert result["calibration"]["relative"]["threshold"]==0.
    assert result["calibration"]["absolute"]["threshold"]==30.
    method=result["methods"]["refit_milp"]
    assert method["events"]["unique_correct_retention"]["count"]==1
    assert method["events"]["unique_correct_retention"]["history_count"]==3
    assert method["events"]["fit_failure"]["count"]==1
    assert method["events"]["truth_omission"]["count"]==1
    assert method["events"]["selection_failure"]["count"]==1
    assert method["events"]["false_rejection"]["count"]==0
    assert method["measurements"]["true_loss"]["missing_or_failed_count"]==1
    gates=json.loads((tmp_path/"report"/"history_gates.json").read_text())
    assert gates[1]["methods"]["refit_milp"]["retained"]==[True]*4
    assert result["paired_against"]["constant_fixed"]["refit_milp"]["events"]["unique_correct_retention"]["improved"]==1
    assert (tmp_path/"report"/"numerical_bounds.csv").exists()


def test_failure_in_calibration_is_not_deleted(tmp_path):
    rows=[trial(i) for i in range(1,21)]
    rows[-1]["records"][0]=record("none",[None,30.,40.,40.],success=[False,True,True,True])
    cal=batch(tmp_path,"calibration",rows)
    ev=batch(tmp_path,"evaluation",[trial(100)])
    result=audit.run_audit(cal,ev,tmp_path/"report",replay=False)
    assert result["calibration"]["relative"]["vacuous"]
    assert result["calibration"]["absolute"]["vacuous"]
    assert result["calibration"]["compute_failure_count"]==1
    assert result["calibration"]["relative"]["sample_count"]==20
    assert result["methods"]["none"]["events"]["unresolved"]["count"]==1


@pytest.mark.parametrize("change",["policy","source","overlap"])
def test_calibration_context_cannot_be_reused_or_changed(tmp_path,change):
    cal=calibrated_batch(tmp_path)
    ev=batch(tmp_path,"evaluation",[trial(1 if change=="overlap" else 100)],
        policy="other" if change=="policy" else "policy", source="b"*64 if change=="source" else "a"*64)
    with pytest.raises(ValueError,match="differs|overlap"):
        audit.run_audit(cal,ev,tmp_path/"report",replay=False)


def test_stress_requires_explicit_descriptive_scope(tmp_path):
    cal=calibrated_batch(tmp_path)
    ev=batch(tmp_path,"stress",[trial(100)],stress="source_drift")
    with pytest.raises(ValueError,match="out of population"):
        audit.run_audit(cal,ev,tmp_path/"report",replay=False)
    result=audit.run_audit(cal,ev,tmp_path/"report",allow_stress=True,replay=False)
    assert result["out_of_population_stress"]
    assert not result["marginal_rank_gate_claim"]


def test_invalid_bound_cannot_hide_inside_iteration(tmp_path):
    row=trial(100)
    row["selection"]={"methods":{"refit_milp":{"records":[{"pair_values":[{"q_lower":9.,"q_upper":2.}]}]}}}
    folder=batch(tmp_path,"pilot",[row])
    with pytest.raises(ValueError,match="lower > upper"):
        audit.pilot_summary(folder,tmp_path/"report",replay=False)


def test_direction_lower_solver_bound_not_claimed_full_proxy_bound():
    value={"solver_records":[{"kind":"direction_lower","incumbent":1.,"bound":2.}],
           "proxy_lower":10.,"proxy_upper":30.}
    assert audit.validate_numeric_bounds(value)==2
    with pytest.raises(ValueError,match="incumbent > best-bound"):
        audit.validate_numeric_bounds({"kind":"secant_upper","incumbent":3.,"bound":2.})


def test_missing_saved_latents_are_unavailable_not_verified(tmp_path):
    np.savez(tmp_path/"trial_100_observations.npz",history_p=np.zeros((1,3)))
    result=audit.replay_saved_fit(tmp_path,trial(100),record("none",[1.,2.,3.,4.]),{"name":"star"})
    assert result["status"]=="unavailable"


def test_saved_state_gaussian_objective_replay(tmp_path):
    # The exact physical generator is independent from the optimizer and the
    # collector's residual arithmetic. Offset two meters by known sigmas so
    # the saved Gaussian objective must be 1^2+2^2=5.
    from research_experiments.mobile_ev_mle.likelihood import ac_predict
    from research_experiments.mobile_ev_mle.candidates import RadialCandidate
    from research_experiments.mobile_ev_mle.simulation import model
    c=RadialCandidate.from_supports("star",3,())
    p=np.full((2,3),.12);q=np.full((2,3),.04);sp=np.full(2,.1);sq=np.full(2,.035)
    r=np.array([.005,.03,.03,.03]);x=np.array([.003,.01,.01,.01])
    physical=ac_predict(c,r,x,p,q,sp,sq)
    meter=model(3)
    observed={k:physical[k].copy() for k in ("p","q","v","master_p","master_q","root_v")}
    for k in ("p","q","v"):observed[k]+=getattr(meter,"bias_"+k)
    observed["p"][0,0]+=.0006;observed["v"][1,2]+=2*.00012
    normalizer=sum(float(np.broadcast_to(np.log(2*np.pi*getattr(meter,k+"_sigma")**2),observed[k].shape).sum())
                   for k in ("p","q","v","master_p","master_q"))
    arrays={f"history_{k}":value[:1] for k,value in observed.items()}
    arrays.update({f"none_{k}":value[1:] for k,value in observed.items()})
    arrays.update({"none__star__fit_"+k:v for k,v in dict(p=p,q=q,source_p=sp,source_q=sq).items()})
    np.savez(tmp_path/"trial_100_observations.npz",**arrays)
    fit=dict(name="star",supports=[list(s) for s in c.supports],r_pu=r.tolist(),x_pu=x.tolist(),
             loss=5.,loglik=-.5*(5.+normalizer))
    result=audit.replay_saved_fit(tmp_path,trial(100),record("none",[5.,6.,7.,8.]),fit)
    assert result["status"]=="verified"
    assert result["replay_loss"]==pytest.approx(5.,abs=1e-10)
    fit["loss"]=4.
    assert audit.replay_saved_fit(tmp_path,trial(100),record("none",[5.,6.,7.,8.]),fit)["status"]=="mismatch"


def test_duplicate_history_ids_and_partial_methods_rejected(tmp_path):
    row=trial(100);row["records"].pop()
    folder=batch(tmp_path,"pilot",[row])
    with pytest.raises(ValueError,match="all frozen methods"):
        audit.load_batch(folder)


def test_rolling_false_rejection_is_any_round_not_final_only(tmp_path):
    cal_rows=[trial(i) for i in range(1,21)]
    for row in cal_rows:
        original=row["records"]
        row["records"]=[dict(record,round=round) for round in (0,1) for record in original]
    cal=batch(tmp_path,"calibration",cal_rows)
    row=trial(100)
    first=[]
    for method in METHODS:
        rec=record(method,[40.,10.,20.,20.]);rec["round"]=0;first.append(rec)
    last=[]
    for method in METHODS:
        rec=record(method,[25.,60.,60.,60.]);rec["round"]=1;last.append(rec)
    row["records"]=first+last
    ev=batch(tmp_path,"evaluation",[row])
    result=audit.run_audit(cal,ev,tmp_path/"report",replay=False)
    assert result["methods"]["refit_milp"]["events"]["false_rejection"]["count"]==1
    assert result["methods"]["refit_milp"]["events"]["unique_correct_retention"]["count"]==1


def test_incomplete_predeclared_batches_cannot_give_inferential_report(tmp_path):
    cal=calibrated_batch(tmp_path)
    ev=batch(tmp_path,"evaluation",[trial(100)])
    config=json.loads((ev/"config.json").read_text());config["count"]=2
    write_json(ev/"config.json",config)
    with pytest.raises(ValueError,match="incomplete predeclared"):
        audit.run_audit(cal,ev,tmp_path/"report",replay=False)


def test_one_round_calibration_cannot_be_reused_for_two_round_trace(tmp_path):
    cal=calibrated_batch(tmp_path)
    row=trial(100);row["records"]=[dict(record,round=round) for round in (0,1) for record in row["records"]]
    ev=batch(tmp_path,"evaluation",[row])
    with pytest.raises(ValueError,match="method-round trace differs"):
        audit.run_audit(cal,ev,tmp_path/"report",replay=False)


def test_rolling_proxy_selection_merges_exact_original_round_and_counters():
    row=dict(selection=dict(methods=dict(rolling=dict(rounds={
        "1":dict(original_selection_index=0,total_selection_seconds=3.),
        "2":dict(original_selection_index=1,selection_ac_predict_calls=17)}))),
        selections=[dict(proxy_search=dict(proxy_lower=2.,records=[dict(pair_values=[])])),
                    dict(proxy_search=dict(proxy_lower=5.,actual_rollback=[dict(name="pair2")]))])
    first=audit._selection_for_record(row,dict(method="rolling",round=1))
    last=audit._selection_for_record(row,dict(method="rolling",round=2))
    assert first["proxy_lower"]==2. and first["total_selection_seconds"]==3.
    assert last["proxy_lower"]==5. and last["selection_ac_predict_calls"]==17
    assert last["actual_rollback"][0]["name"]=="pair2"
    missing=audit._selection_for_record(row,dict(method="rolling",round=3))
    assert missing=={}


def test_rolling_joint_archive_replays_each_round_without_truth_or_future(tmp_path):
    from research_experiments.mobile_ev_mle.likelihood import ac_predict
    from research_experiments.mobile_ev_mle.candidates import RadialCandidate
    from research_experiments.mobile_ev_mle.simulation import model
    c=RadialCandidate.from_supports("star",3,())
    meter=model(3);r=np.array([.005,.03,.03,.03]);x=np.array([.003,.01,.01,.01])
    arrays={};fits={}
    for round_id,count,offset in ((1,2,1.),(2,4,3.)):
        p=np.full((count,3),.12);q=np.full((count,3),.04)
        sp=np.full(count,.1);sq=np.full(count,.035)
        physical=ac_predict(c,r,x,p,q,sp,sq)
        observed={k:physical[k].copy() for k in ("p","q","v","master_p","master_q","root_v")}
        for key in ("p","q","v"):observed[key]+=getattr(meter,"bias_"+key)
        observed["v"][0,1]+=offset*.00012
        norm=sum(float(np.broadcast_to(np.log(2*np.pi*getattr(meter,key+"_sigma")**2),observed[key].shape).sum())
                 for key in ("p","q","v","master_p","master_q"))
        arrays.update({f"round_{round_id}_joint_{k}":v for k,v in observed.items()})
        arrays.update({f"round_{round_id}__star__fit_{k}":v for k,v in
                       dict(p=p,q=q,source_p=sp,source_q=sq).items()})
        fits[round_id]=dict(name="star",supports=[list(s) for s in c.supports],r_pu=r.tolist(),
            x_pu=x.tolist(),loss=offset**2,loglik=-.5*(offset**2+norm))
    np.savez(tmp_path/"trial_300_observations.npz",**arrays)
    # No truth, action or future-only arrays exist in this archive.
    for round_id in (1,2):
        result=audit.replay_saved_fit(tmp_path,dict(seed=300),dict(method="rolling",round=round_id),fits[round_id])
        assert result["status"]=="verified"
        assert result["replay_loss"]==pytest.approx(float((1 if round_id==1 else 3)**2),abs=1e-9)
    assert audit.replay_saved_fit(tmp_path,dict(seed=300),dict(method="rolling",round=3),fits[2])["status"]=="unavailable"


def test_frozen_proxy_tolerance_preserves_raw_crossing_instead_of_zero_gap():
    value=dict(proxy_lower=26.825963154828287,proxy_upper=26.825959746504417)
    with pytest.raises(ValueError,match="invalid lower > upper"):
        audit.validate_numeric_bounds(value)
    assert audit.validate_numeric_bounds(value,proxy_tolerance=1e-6)==1
    ledger=audit.raw_bound_crossings(value)
    assert len(ledger)==1 and ledger[0]["raw_gap"]<0
    assert ledger[0]["within_declared_tolerance"] and not ledger[0]["strictly_ordered"]
    assert not ledger[0]["strict_optimality_certificate"]
    with pytest.raises(ValueError,match="invalid lower > upper"):
        audit.validate_numeric_bounds(dict(proxy_lower=1.0001,proxy_upper=1.),proxy_tolerance=1e-6)
    with pytest.raises(ValueError,match="invalid lower > upper"):
        audit.validate_numeric_bounds(dict(q_lower=1.0000002,q_upper=1.),proxy_tolerance=1e-6)
