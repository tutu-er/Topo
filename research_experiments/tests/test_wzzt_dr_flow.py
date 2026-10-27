"""Orchestration boundaries and resume tests; estimator is an explicit stub."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research_experiments.wzzt_dr_selection.end_to_end import runner


@pytest.fixture
def fixture_run(tmp_path, monkeypatch):
    config = {"case_key": "paper15", "action_date": "2021-02-09", "validation_date": "2021-02-10",
        "source": {"forecast_days": 10, "history_fit_days": 3},
        "planning": {"objective": "d_opt", "budget_nok": 10000., "ridge": 1e-6,
            "power_limits_kw": 1000., "max_daily_energy_change_fraction": .5},
        "physical": {"voltage_min_pu": .9, "voltage_max_pu": 1.1},
        "seeds": {"history": 100, "action": 200, "validation": 300, "response_action": 400, "response_validation": 500},
        "wzzt": {}}
    data = tmp_path / "data"; calibration = tmp_path / "calibration"; output = tmp_path / "run"
    (data / "raw").mkdir(parents=True); (calibration / "outputs").mkdir(parents=True); output.mkdir()
    mean_rows = []; price_rows = []
    for i, menu in enumerate(runner.price_response.MENUS):
        for band in runner.price_response.BANDS: mean_rows.append({"menu": menu, "band": band, "factor": .9-i*.1})
        for hour in range(1,25): price_rows.append({"Price_signal":menu,"Hour":hour,"Experiment_price_NOK_kWh":1.})
    pd.DataFrame(mean_rows).to_csv(calibration / "outputs" / "control_training_menu_factors.csv", index=False)
    pd.DataFrame(price_rows).to_csv(data / "raw" / "price_signals.csv", index=False)
    terminals = [101,102]
    def frame(day, value): return pd.DataFrame(value, index=pd.date_range(day,periods=24,freq="h"),columns=terminals)
    history = {str(day.date()):frame(str(day.date()),float(i+1)) for i,day in enumerate(pd.bdate_range(end="2021-02-05",periods=10))}
    bundle = {"history_days": history,
        "actual_days": {"action":frame("2021-02-09",100.),"validation":frame("2021-02-10",200.)},
        "response_banks": {phase:{m:np.full((1,24,1),factor) for m in runner.price_response.MENUS}
            for phase,factor in (("action",.95),("validation",1.05))},
        "bank_metadata": {"fixture":True}, "data_manifest": {"fixture":True}}
    runner._pickle(output / runner.BUNDLE_FILE,bundle)
    runner._json(output / "prepared_receipt.json",{"config_hash":runner._fingerprint(config),
        "bundle_sha256":runner._sha(output / runner.BUNDLE_FILE)})
    net=SimpleNamespace(base_mva=1.,terminal_to_original={101:1,102:2},metadata={"impedance_scale":1.})
    monkeypatch.setattr(runner.physical_case,"build_reference_case",lambda key:net)
    monkeypatch.setattr(runner.physical_case,"synthetic_q_from_reference_pf",lambda p,**kw:.2*p)
    physical_calls=[]; fit_calls=[]
    def simulate(net,p,q,*,seed,name,**kw):
        physical_calls.append({"seed":seed,"name":name,"p":p.copy()})
        observed={"name":name,"P_terminal":(p+1)/1000.,"Q_terminal":(q+2)/1000.,
            "V_terminal":p*0+1.,"root_voltage":pd.Series(1.01,index=p.index)}
        return {"observed":observed,"truth":{"truth_supports":[[101],[102]]},"checks":{
            "usable_ac_solution":True,"voltage_limits_pass":True,"nonconverged_samples":0}}
    monkeypatch.setattr(runner.physical_case,"simulate_observations",simulate)
    def fit(training,validation,config,checkpoint_path):
        fit_calls.append({"training":training,"validation":validation})
        result={"status":"incomplete","has_model":True,"summary":{"stop_reason":"extension_not_proven_optimal",
            "selected_path_index":0,"path_length":1,"support_labels":[[101],[102]],"train_mae":.01,
            "validation_mae":.02,"bound_expansions":0},"attempted_extensions":[],"elapsed_seconds":0.}
        runner._json(checkpoint_path,result)
        return result
    monkeypatch.setattr(runner.wzzt_adapter,"fit_observed_wzzt",fit)
    monkeypatch.setattr(runner.wzzt_adapter,"evaluate_effective_supports",lambda fit,truth:{"full_effective_family_recovered":False})
    return config,data,calibration,output,physical_calls,fit_calls


def test_forecast_only_observed_history_and_independent_actual_batches(fixture_run):
    config,data,calibration,output,physical_calls,fit_calls=fixture_run
    summary=runner.run(config,data,calibration,output)
    plan=json.loads((output/"plan.json").read_text())
    np.testing.assert_allclose(plan["forecast_p_kw"],6.5)
    np.testing.assert_allclose(plan["forecast_q_kvar"],3.1)
    assert len(plan["candidates"])==13
    assert len(physical_calls)==16 and len(fit_calls)==3
    assert summary["technical_flow_success"] and summary["fit_incomplete_count"]==3
    assert summary["fit_complete_count"]==0 and summary["strategy_denominator"]==3
    for call in fit_calls:
        assert len(call["training"])==4 and len(call["validation"])==1
        assert sum(len(s["P_terminal"]) for s in call["training"])==96
        assert call["training"][-1]["P_terminal"].index[0].date().isoformat()=="2021-02-09"
        assert call["validation"][0]["P_terminal"].index[0].date().isoformat()=="2021-02-10"
    assert {c["seed"] for c in physical_calls if c["name"].endswith("_action")}=={200}
    assert {c["seed"] for c in physical_calls if c["name"].endswith("_validation")}=={300}
    fixed=runner._unpickle(output/"strategies"/"fixed_P2_first_node"/"action.pkl")
    valid=runner._unpickle(output/"strategies"/"fixed_P2_first_node"/"validation.pkl")
    assert fixed["response"]["p_kw"][0,0]==95. and valid["response"]["p_kw"][0,0]==210.


def test_completed_resume_does_not_repeat_ac_or_core_and_rejects_changed_config(fixture_run):
    config,data,calibration,output,physical_calls,fit_calls=fixture_run
    first=runner.run(config,data,calibration,output)
    second=runner.run(config,data,calibration,output,resume=True)
    assert first==second and len(physical_calls)==16 and len(fit_calls)==3
    config["planning"]["budget_nok"]+=1
    with pytest.raises(ValueError,match="fingerprint"):runner.run(config,data,calibration,output,resume=True)


def test_lock_prevents_duplicate_and_only_owner_removes(tmp_path):
    with runner._exclusive_lock(tmp_path,"hash"):
        original=(tmp_path/"runner.lock").read_text()
        with pytest.raises(RuntimeError,match="lock already exists"):
            with runner._exclusive_lock(tmp_path,"hash"):pass
        assert (tmp_path/"runner.lock").read_text()==original
    assert not (tmp_path/"runner.lock").exists()


def test_invalid_ac_is_retained_and_all_three_denominators_recorded(fixture_run,monkeypatch):
    config,data,calibration,output,physical_calls,fit_calls=fixture_run
    original=runner.physical_case.simulate_observations
    def fail_action(*args,**kw):
        result=original(*args,**kw)
        if kw["name"].endswith("_action"):
            result["observed"]=None
            result["checks"].update({"usable_ac_solution":False,"nonconverged_samples":1})
        return result
    monkeypatch.setattr(runner.physical_case,"simulate_observations",fail_action)
    result=runner.run(config,data,calibration,output)
    assert result["recorded_strategies"]==result["failed_or_skipped_count"]==3
    assert not result["technical_flow_success"] and not fit_calls
    assert all(s["stage_status"]=="skipped_invalid_ac" for s in result["strategies"])


def test_changed_future_load_response_bank_and_truth_leave_entire_plan_unchanged(fixture_run,monkeypatch):
    config,data,calibration,output,physical_calls,fit_calls=fixture_run
    runner.run(config,data,calibration,output)
    original_plan=json.loads((output/"plan.json").read_text())
    changed_output=output.parent/"changed_future_run"
    changed_output.mkdir()
    changed_bundle=runner._unpickle(output/runner.BUNDLE_FILE)
    changed_bundle["actual_days"]["action"]*=1.3
    changed_bundle["actual_days"]["validation"]*=.7
    for menu in runner.price_response.MENUS:
        changed_bundle["response_banks"]["action"][menu][:]=.8
        changed_bundle["response_banks"]["validation"][menu][:]=1.2
    runner._pickle(changed_output/runner.BUNDLE_FILE,changed_bundle)
    runner._json(changed_output/"prepared_receipt.json",{
        "config_hash":runner._fingerprint(config),
        "bundle_sha256":runner._sha(changed_output/runner.BUNDLE_FILE)})
    simulate=runner.physical_case.simulate_observations
    def changed_truth(*args,**kwargs):
        result=simulate(*args,**kwargs)
        if not kwargs["name"].startswith("history_"):
            result["truth"]["truth_supports"]=[[101],[102],[101,102]]
        return result
    evaluated_truths=[]
    def evaluate(fit,truth):
        evaluated_truths.append(truth)
        return {"full_effective_family_recovered":False}
    monkeypatch.setattr(runner.physical_case,"simulate_observations",changed_truth)
    monkeypatch.setattr(runner.wzzt_adapter,"evaluate_effective_supports",evaluate)
    runner.run(config,data,calibration,changed_output)
    changed_plan=json.loads((changed_output/"plan.json").read_text())
    assert original_plan==changed_plan
    assert evaluated_truths==[[[101],[102],[101,102]]]*3
    original_actual=runner._unpickle(output/"strategies"/"fixed_P2_first_node"/"action.pkl")
    changed_actual=runner._unpickle(changed_output/"strategies"/"fixed_P2_first_node"/"action.pkl")
    assert original_actual["response"]["p_kw"][0,0]==95.
    assert changed_actual["response"]["p_kw"][0,0]==104.
    assert len(fit_calls)==6  # All six are the explicit fixture stub; no MILP.


def test_prepared_mean_csv_change_is_rejected_before_any_execution(fixture_run):
    config,data,calibration,output,physical_calls,fit_calls=fixture_run
    mean_csv=calibration/"outputs"/"control_training_menu_factors.csv"
    price_csv=data/"raw"/"price_signals.csv"
    receipt_path=output/"prepared_receipt.json"
    receipt=json.loads(receipt_path.read_text())
    receipt["data_manifest"]={"source_hashes":{
        str(mean_csv):runner._sha(mean_csv),str(price_csv):runner._sha(price_csv)}}
    runner._json(receipt_path,receipt)
    modified=pd.read_csv(mean_csv)
    modified.loc[0,"factor"]+=.01
    modified.to_csv(mean_csv,index=False)
    with pytest.raises(ValueError,match="prepared planning-input hash mismatch") as error:
        runner.run(config,data,calibration,output)
    assert str(mean_csv) in str(error.value)
    assert physical_calls==[] and fit_calls==[]
    assert not (output/"run_state.json").exists()
