"""Frozen headers, complete-trajectory schema, exact counters and checkpoints."""
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

try:
    from . import adaptive_rolling_batch as batch
except ImportError:
    import adaptive_rolling_batch as batch


def _fake_raw(seed, controller_id):
    rounds=[]; selections=[]
    for r,t in ((1,14),(2,20)):
        fits=[dict(name=n,loss=float(t+i),loglik=-1.,success=True,diagnostics={"ac_evaluations":1})
              for i,n in enumerate(("star","clade_01","clade_02","clade_12"))]
        rounds.append(dict(round=r, complete=True, losses=[t+i for i in range(4)], success=[True]*4,
            fits=fits, fit_failures={}, joint_observation_hash="fixture_hash", true_index=0,
            executed=dict(u_kw=np.tile([3.,0.,0.],(2 if r==1 else 6,1)),
                grid_energy_kwh=1.5 if r==1 else 4.5, sites=[0]*(2 if r==1 else 6), driving_energy_kwh=0.),
            operational_voltage_ok=True))
        selections.append(dict(status="complete", total_selection_seconds=.2,
            proxy_search=dict(proxy_lower=1.,proxy_upper=2.,absolute_gap=1.,relative_gap=.5,status="budget_or_iteration_limit")))
    score=batch.rolling.policy_trial_scores([[row["losses"] for row in rounds]],0,
        success=[[row["success"] for row in rounds]])
    from dataclasses import asdict
    return dict(seed=seed, history_id=str(seed), policy_id=controller_id, population_id="raw_pop",
        policy_seed=seed+1, case="weak", truth_topology=0, true_index=0,
        candidate_names=["star","clade_01","clade_02","clade_12"], rounds=rounds, selections=selections,
        score_record=asdict(score), independent_history_unit=True, seconds=.9)


def _metrics():
    return dict(selection_calls=[dict(round=r,seconds=.2,ac_calls=3,returned=True) for r in (1,2)],
        joint_fit_calls=[dict(round=r,seconds=.3,ac_calls=5,returned=True) for r in (1,2)],
        whole_trial_ac_calls=19, whole_trial_seconds=.9, unclassified_ac_calls=3,
        call_definition="fixture counts, not a timing experiment")


def _frozen(monkeypatch):
    monkeypatch.setattr(batch,"source_hashes",lambda:{"fixture.py":"a"*64})


def _fake_worker(monkeypatch, calls):
    def counted(seed, *, policy, stress, output):
        calls.append(seed)
        config=batch.batch_config(phase="pilot",count=1,start_seed=seed,policy=policy,stress=stress)
        raw=_fake_raw(seed,config["controller_policy_id"])
        batch.dump(output/f"rolling_{seed}.json",raw)
        np.savez_compressed(output/f"rolling_{seed}_observations.npz",
            round_1_joint_p=np.full((14,3),.12), round_2_joint_p=np.full((20,3),.12),
            round_1__star__fit_source_p=np.full(14,.1), round_2__star__fit_source_p=np.full(20,.1))
        return raw,_metrics()
    monkeypatch.setattr(batch,"_counted_trial",counted)


def test_phase_header_freezes_devices_two_rounds_sources_and_new_policy(monkeypatch):
    _frozen(monkeypatch)
    config=batch.batch_config(phase="calibration",count=64,start_seed=920001)
    assert config["phase"]=="calibration" and config["mandatory_rounds"]==2
    assert config["methods"]==[batch.METHOD] and config["round_ids"]==[1,2]
    assert config["initial_ev_config"]["net_grid_energy_kwh"]==6.
    assert config["operational_requirements"]["executed_slots"]==[2,6]
    assert config["source_sha256"]=={"fixture.py":"a"*64}
    assert config["policy"]["fit_max_nfev"]==300
    assert not config["may_reuse_one_round_gates"]
    assert config["policy_id"]!=batch._hash(dict(policy=config["policy"],source_hashes=config["source_sha256"]))
    assert batch.batch_config(phase="eval",count=1,start_seed=1)["phase"]=="evaluation"
    with pytest.raises(ValueError,match="calibration"):
        batch.batch_config(phase="calibration",count=1,start_seed=1,stress="source_drift")


def test_normalization_keeps_both_original_rounds_and_maps_collector_contract(monkeypatch):
    _frozen(monkeypatch)
    config=batch.batch_config(phase="pilot",count=2,start_seed=1)
    raw=_fake_raw(1,config["controller_policy_id"])
    raw["selections"][0]["status"]="selection_failure_operational_fallback"
    normalized=batch.normalize_trial(raw,_metrics(),config)
    assert normalized["rounds"] is raw["rounds"] and normalized["selections"] is raw["selections"]
    assert [r["round"] for r in normalized["records"]]==[1,2]
    assert normalized["records"][0]["selection_status"]=="selection_failure_fallback"
    assert normalized["selection"]["methods"][batch.METHOD]["status"]=="selection_failure_fallback"
    assert [r["ac_calls"] for r in normalized["records"]]==[5,5]
    assert all(r["ac_calls_exact_including_failed_fits"] for r in normalized["records"])
    assert normalized["records"][0]["actual_action"]["energy_kwh"]==1.5
    assert len(normalized["records"][1]["action"]["p"])==6
    assert normalized["controller_policy_id"]==raw["policy_id"]
    raw["policy_id"]="one_round_incompatible"
    with pytest.raises(ValueError,match="controller law"):
        batch.normalize_trial(raw,_metrics(),config)


def test_batch_publishes_exact_round_archive_and_strictly_resumes(monkeypatch,tmp_path):
    _frozen(monkeypatch); calls=[];_fake_worker(monkeypatch,calls)
    config,rows=batch.run_batch(tmp_path,phase="pilot",count=2,start_seed=910001)
    assert calls==[910001,910002] and len(rows)==2
    assert (tmp_path/"trial_910001.json").exists() and not (tmp_path/"rolling_910001.json").exists()
    archive=np.load(tmp_path/"trial_910001_observations.npz")
    assert archive["round_1_joint_p"].shape==(14,3)
    assert archive["round_2__star__fit_source_p"].shape==(20,)
    batch.run_batch(tmp_path,phase="pilot",count=2,start_seed=910001)
    assert calls==[910001,910002]
    summary=json.loads((tmp_path/"summary.json").read_text())
    assert summary["all_requested_histories_attempted"] and summary["history_count"]==2
    with pytest.raises(ValueError,match="changed policy"):
        batch.run_batch(tmp_path,phase="pilot",count=2,start_seed=910001,policy={"fit_max_nfev":301})


def test_budget_checkpoint_never_claims_requested_count_complete(monkeypatch,tmp_path):
    _frozen(monkeypatch); calls=[];_fake_worker(monkeypatch,calls)
    start=datetime(2026,10,12,18,tzinfo=timezone.utc);cutoff=start+timedelta(seconds=1)
    values=iter([start,start,cutoff+timedelta(seconds=1)])
    _,rows=batch.run_batch(tmp_path,phase="evaluation",count=3,start_seed=930001,
        deadline_utc=cutoff.isoformat(),_now=lambda:next(values))
    assert len(rows)==1 and calls==[930001]
    summary=json.loads((tmp_path/"summary.json").read_text())
    assert summary["stop_reason"]=="wall_clock_budget_checkpoint"
    assert summary["requested_history_count"]==3 and summary["history_count"]==1
    assert not summary["all_requested_histories_attempted"]
    assert json.loads((tmp_path/"progress.json").read_text())["state"]=="checkpoint"


def test_expired_budget_starts_no_history_and_naive_deadline_rejected(monkeypatch,tmp_path):
    _frozen(monkeypatch);calls=[];_fake_worker(monkeypatch,calls)
    now=datetime(2026,10,12,18,tzinfo=timezone.utc)
    _,rows=batch.run_batch(tmp_path,phase="pilot",count=2,start_seed=910001,
        deadline_utc=(now-timedelta(seconds=1)).isoformat(),_now=lambda:now)
    assert not rows and not calls
    assert not json.loads((tmp_path/"summary.json").read_text())["all_requested_histories_attempted"]
    with pytest.raises(ValueError,match="timezone"):
        batch._deadline("2026-10-12T18:00:00")


def test_source_change_during_trial_is_saved_as_pending_not_completed(monkeypatch,tmp_path):
    hashes=iter([{"fixture":"a"*64},{"fixture":"a"*64},{"fixture":"b"*64}])
    monkeypatch.setattr(batch,"source_hashes",lambda:next(hashes))
    # Worker avoids constructing a second source manifest inside this test.
    def worker(seed,*,policy,stress,output):
        ev=batch._public_config(); ids=batch.rolling.policy_identifiers(policy,ev)
        raw=_fake_raw(seed,ids["policy_id"])
        np.savez_compressed(output/f"rolling_{seed}_observations.npz",round_1_joint_p=np.zeros((14,3)))
        return raw,_metrics()
    monkeypatch.setattr(batch,"_counted_trial",worker)
    _,rows=batch.run_batch(tmp_path,phase="pilot",count=2,start_seed=910001)
    assert not rows and (tmp_path/"pending_source_drift_910001.json").exists()
    assert not (tmp_path/"trial_910001.json").exists()
    assert json.loads((tmp_path/"summary.json").read_text())["stop_reason"]=="source_changed_during_history_checkpoint"


def test_ac_counters_include_failed_entries_and_restore_all_functions(monkeypatch,tmp_path):
    from research_experiments.mobile_ev_mle import likelihood,symmetric_ac_local,simulation
    def physical(kind="ok"):
        if kind=="fail":raise ValueError("deliberate AC exception")
        return {}
    for module in (likelihood,symmetric_ac_local,simulation):monkeypatch.setattr(module,"ac_predict",physical)
    def selection(*args,**kwargs):
        likelihood.ac_predict();symmetric_ac_local.ac_predict()
    def fitted(*args,**kwargs):
        likelihood.ac_predict()
        try:likelihood.ac_predict("fail")
        except ValueError:pass
    def trial(*args,**kwargs):
        simulation.ac_predict()
        for _ in (1,2):
            batch.rolling.plan_round();batch.rolling.fit_bank();simulation.ac_predict()
        return {"fixture":True}
    monkeypatch.setattr(batch.rolling,"plan_round",selection)
    monkeypatch.setattr(batch.rolling,"fit_bank",fitted)
    monkeypatch.setattr(batch.rolling,"rolling_trial",trial)
    _,metrics=batch._counted_trial(1,policy={},stress=None,output=tmp_path)
    assert metrics["whole_trial_ac_calls"]==11 and metrics["unclassified_ac_calls"]==3
    assert [r["ac_calls"] for r in metrics["selection_calls"]]==[2,2]
    assert [r["ac_calls"] for r in metrics["joint_fit_calls"]]==[2,2]
    assert batch.rolling.plan_round is selection and batch.rolling.fit_bank is fitted
    assert all(module.ac_predict is physical for module in (likelihood,symmetric_ac_local,simulation))


def test_trace_exception_retains_counter_evidence_and_never_fake_completes(monkeypatch,tmp_path):
    from research_experiments.mobile_ev_mle import likelihood
    def physical(*args,**kwargs):return {}
    monkeypatch.setattr(likelihood,"ac_predict",physical)
    def selection(*args,**kwargs):likelihood.ac_predict()
    def fitting(*args,**kwargs):likelihood.ac_predict();raise ValueError("failed complete trace")
    def trial(*args,**kwargs):batch.rolling.plan_round();batch.rolling.fit_bank()
    monkeypatch.setattr(batch.rolling,"plan_round",selection)
    monkeypatch.setattr(batch.rolling,"fit_bank",fitting)
    monkeypatch.setattr(batch.rolling,"rolling_trial",trial)
    with pytest.raises(batch.TraceExecutionFailure) as caught:
        batch._counted_trial(1,policy={},stress=None,output=tmp_path)
    assert caught.value.metrics["whole_trial_ac_calls"]==2
    assert caught.value.metrics["joint_fit_calls"][0]["returned"] is False
    assert batch.rolling.fit_bank is fitting and likelihood.ac_predict is physical
    _frozen(monkeypatch)
    config=batch.batch_config(phase="pilot",count=1,start_seed=1)
    failed=batch._failure_trial(1,config,caught.value)
    assert not failed["score_record"]["complete"] and failed["score_record"]["compute_failure"]
    assert np.isinf(failed["score_record"]["relative_gap"])
    assert failed["records"][0]["ac_calls"]==1 and failed["records"][1]["ac_calls"] is None
    assert all(r["full_round_evidence_unavailable_due_to_trace_failure"] for r in failed["records"])


def test_one_round_calibration_identifier_cannot_be_reused_for_rolling(monkeypatch):
    from dataclasses import asdict
    from research_experiments.mobile_ev_mle.adaptive_statistics import calibrate_policy_gates,apply_candidate_gates
    _frozen(monkeypatch);config=batch.batch_config(phase="calibration",count=20,start_seed=940001)
    score=batch.rolling.policy_trial_scores([[[1.,2.,3.,4.],[2.,3.,4.,5.]]],0)
    calibration=calibrate_policy_gates([asdict(score)]*20,[str(i) for i in range(20)],
        policy_id=config["policy_id"],population_id=config["population_id"])
    with pytest.raises(ValueError,match="policy"):
        apply_candidate_gates([1.,2.,3.,4.],calibration,history_id="evaluation_history",
            policy_id="one_round_policy",population_id=config["population_id"],candidate_names=["star","01","02","12"])
