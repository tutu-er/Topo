"""Replay frozen, paired DR observations through the finite-pool identifier.

This replaces only the identification layer. Forecasts, prices, whole-day
response draws, AC outputs and physical/payment audits are reused unchanged.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys
import traceback

import numpy as np
import pandas as pd

from ..end_to_end import wzzt_adapter as observed
from ..end_to_end.runner import _exclusive_lock

STRATEGIES = ("no_reward", "fixed_P2_first_node", "planned_d_opt")
EXTRA_STRATEGY = "fixed_same_menu_first_node"
MODEL_NAMES = ("rnj_bank", "baseline25", "multistart_original_rule")
SAMPLE_COUNTS = (24, 48, 96)
DEFAULT_SAMPLE_COUNTS = (24, 96)
SPARSE_HOURS = (0, 4, 8, 12, 16, 20)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    observed._write_json(Path(path), value)


def load_pickle(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def observed_only(simulation):
    """The estimator receives measurements, never simulation truth/checks."""
    item = simulation["observed"]
    if item is None:
        raise ValueError("source AC simulation has no usable observations")
    return observed.make_observed_scenario(item["P_terminal"], item["Q_terminal"],
        item["V_terminal"], item["root_voltage"], item["name"])


def subsample_training(scenarios, count):
    if count not in SAMPLE_COUNTS or len(scenarios) != 4:
        raise ValueError("locked design requires four physical days and N24, N48 or N96")
    result = []
    seen_dates = set()
    for scenario in scenarios:
        frame = scenario["P_terminal"]
        if not isinstance(frame.index, pd.DatetimeIndex) or len(frame) != 24:
            raise ValueError("each source day must contain 24 indexed hourly observations")
        date = frame.index[0].normalize()
        if not frame.index.equals(pd.date_range(date, periods=24, freq="h")) or date in seen_dates:
            raise ValueError("source days must be distinct complete chronological physical days")
        seen_dates.add(date)
        rows = np.arange(0, 24, 96 // count)
        point = {}
        for key, value in scenario.items():
            if isinstance(value, (pd.DataFrame, pd.Series)):
                if not value.index.equals(frame.index):
                    raise ValueError("source channel indices differ")
                point[key] = value.iloc[rows].copy()
            else:
                point[key] = value
        result.append(point)
    if sum(len(s["P_terminal"]) for s in result) != count:
        raise ValueError("unexpected sample count")
    return result


def source_files(source, additional=None):
    summary = read(source / "summary.json")
    if summary.get("status") != "complete":
        raise ValueError("frozen source run is not complete")
    dates = summary["wzzt_training_dates"]
    if len(dates) != 3:
        raise ValueError("locked source requires three history-fit days")
    paths = [source / name for name in ("summary.json", "plan.json", "response_draws.json", "run_state.json")]
    paths += [source / "history" / (day + ".pkl") for day in dates]
    for strategy in STRATEGIES:
        paths += [source / "strategies" / strategy / name for name in ("result.json", "action.pkl", "validation.pkl")]
    if additional is not None:
        paths += [additional / name for name in ("result.json", "action.pkl", "validation.pkl")]
    required = paths if additional is None else paths[:-3]
    missing_required = [str(path) for path in required if not path.is_file()]
    if missing_required:
        raise FileNotFoundError("missing frozen source files: " + repr(missing_required))
    return summary, paths


def support_metrics(point, truth, labels):
    """Truth enters only post-fit scoring. Internal and effective-family scores differ."""
    labels = list(labels)
    family = point.get("support_labels")
    if family is None:
        family = [[labels[i] for i in support] for support in point.get("support_indices", point.get("supports", []))]
    r = point.get("r_values", point.get("r", [])); x = point.get("x_values", point.get("x", []))
    if len(family) != len(r) or len(family) != len(x):
        raise ValueError("model family and weight lengths differ")
    actual = {frozenset(s) for s in truth}
    active = {frozenset(s) for s, a, b in zip(family, r, x) if max(a, b) > 1e-9}
    stored = {frozenset(s) for s in family}
    internal = lambda f: {s for s in f if 1 < len(s) < len(labels)}
    wanted = internal(actual); predicted = internal(active)
    roots = {s for s in wanted if not any(s < t for t in wanted)}
    tp = len(wanted & predicted); fp = len(predicted - wanted); fn = len(wanted - predicted)
    denom = 2 * tp + fp + fn
    return {"internal_true_count": len(wanted), "internal_matched": tp,
        "internal_false": fp, "internal_missing": fn, "internal_f1": 1.0 if denom == 0 else 2*tp/denom,
        "internal_exact": predicted == wanted, "root_true_count": len(roots),
        "root_matched": len(roots & predicted), "complete_effective_family": active == actual,
        "stored_support_count": len(stored), "effective_support_count": len(active),
        "zero_weight_retained_supports": [sorted(s) for s in sorted(stored-active, key=lambda s:(len(s),tuple(sorted(s))))],
        "extra_effective_supports": [sorted(s) for s in sorted(active-actual,key=lambda s:(len(s),tuple(sorted(s))))],
        "missing_effective_supports": [sorted(s) for s in sorted(actual-active,key=lambda s:(len(s),tuple(sorted(s))))],
        "truth_role": "post-fit evaluation only", "zero_tolerance": 1e-9}


def evaluate_fit(fit, truth, labels):
    if not all(name in fit.get("models", {}) for name in MODEL_NAMES):
        raise ValueError("identifier did not return all three declared models")
    models = {}
    for name in MODEL_NAMES:
        point = fit["models"][name]
        item = {"final": support_metrics(point, truth, labels),
            "train_mae": point.get("train_mae", point.get("objective")),
            "validation_mae": point.get("validation_mae"),
            "selection_validation_mae": point.get("selection_validation_mae"),
            "selected_origin": point.get("origin"), "selection_index": point.get("selection_index")}
        before = point.get("selected_uncontracted", point.get("pre_contraction_model"))
        if before is not None:
            item["before_final_contraction"] = support_metrics(before, truth, labels)
        else:
            item["before_final_contraction"] = None
        models[name] = item
    pool_labels = fit.get("pool_support_labels")
    if pool_labels is None and fit.get("pool_support_indices") is not None:
        pool_labels = [[labels[i] for i in s] for s in fit["pool_support_indices"]]
    pool = None
    if pool_labels is not None:
        true_internal = {frozenset(s) for s in truth if 1 < len(s) < len(labels)}
        available = {frozenset(s) for s in pool_labels}
        pool = {"internal_true_count":len(true_internal), "internal_true_available":len(true_internal & available),
            "internal_true_recall":1.0 if not true_internal else len(true_internal & available)/len(true_internal),
            "missing_true_internal":[sorted(s) for s in sorted(true_internal-available,key=lambda s:(len(s),tuple(sorted(s))))],
            "role":"post-fit pool diagnostic; no truth-based candidate filtering"}
    searches = fit.get("searches", {})
    reasons = {name:run.get("reason",run.get("stop_reason")) for name,run in searches.items()}
    return {"models":models, "pool_coverage":pool, "search_stop_reasons":reasons,
        "search_wall_limited":any(reason == "pool_wall_limit" for reason in reasons.values()),
        "validation_role":"model selection, not untouched predictive test"}


def _periods(action, validation):
    return {name:{"payment_nok":p["response"]["total_payment_nok"],
            "response_audit":p["response"]["audit"], "physical_checks":p["simulation"]["checks"]}
        for name,p in (("action",action),("validation",validation))}


def summarize(records, protocol):
    result = {"status":"complete" if len(records)==protocol["task_denominator"] else "running",
        "task_denominator":protocol["task_denominator"], "recorded_tasks":len(records),
        "evaluated_tasks":sum(r["status"]=="evaluated" for r in records),
        "failed_or_unavailable_tasks":sum(r["status"]!="evaluated" for r in records),
        "search_wall_limited_tasks":sum(r.get("evaluation",{}).get("search_wall_limited",False) for r in records),
        "records":records, "new_price_optimizations":0, "new_AC_simulations":0,
        "independent_realizations":1, "sample_counts_share_trajectories":True,
        "new_generalization_test":False, "scope":protocol["scope"]}
    comparisons=[]
    by_key={(r["strategy"],r["N"]):r for r in records}
    for r in records:
        base=by_key.get(("no_reward",r["N"]))
        if r["strategy"]=="no_reward" or not base or r["status"]!="evaluated" or base["status"]!="evaluated":continue
        for model in MODEL_NAMES:
            current=r["evaluation"]["models"][model]["final"]
            previous=base["evaluation"]["models"][model]["final"]
            comparisons.append({"strategy":r["strategy"],"N":r["N"],"model":model,
                "internal_f1_delta":current["internal_f1"]-previous["internal_f1"],
                "root_matched_delta":current["root_matched"]-previous["root_matched"],
                "complete_effective_family":current["complete_effective_family"],
                "no_reward_complete_effective_family":previous["complete_effective_family"]})
    result["paired_vs_no_reward"]=comparisons
    matched=[]
    for count in protocol["sample_counts"]:
        planned=by_key.get(("planned_d_opt",count));fixed=by_key.get((EXTRA_STRATEGY,count))
        if not planned or not fixed or planned["status"]!="evaluated" or fixed["status"]!="evaluated":continue
        for model in MODEL_NAMES:
            a=planned["evaluation"]["models"][model]["final"]
            b=fixed["evaluation"]["models"][model]["final"]
            matched.append({"N":count,"model":model,"comparison":"planned_d_opt minus fixed_same_menu_first_node",
                "internal_f1_delta":a["internal_f1"]-b["internal_f1"],"root_matched_delta":a["root_matched"]-b["root_matched"],
                "planned_complete_effective_family":a["complete_effective_family"],
                "fixed_complete_effective_family":b["complete_effective_family"],
                "planned_action_payment_nok":planned["periods"]["action"]["payment_nok"],
                "fixed_action_payment_nok":fixed["periods"]["action"]["payment_nok"]})
    result["paired_vs_fixed_same_menu"]=matched
    return result


def run(source_run, output_dir, *, additional_strategy=None, resume=False, search_seconds=25.0, sample_sizes=DEFAULT_SAMPLE_COUNTS, fitter=None):
    sample_sizes=tuple(sample_sizes)
    if not sample_sizes or len(set(sample_sizes))!=len(sample_sizes) or any(n not in SAMPLE_COUNTS for n in sample_sizes):
        raise ValueError("sample sizes must be a nonempty unique subset of24,48,96")
    source=Path(source_run).resolve(); out=Path(output_dir).resolve()
    additional=None if additional_strategy is None else Path(additional_strategy).resolve()
    source_summary, paths=source_files(source,additional)
    source_hashes={str(p):sha(p) for p in paths if p.is_file()}
    source_missing=[str(p) for p in paths if not p.is_file()]
    strategies=list(STRATEGIES)+([] if additional is None else [EXTRA_STRATEGY])
    implementation_files=sorted(Path(__file__).parent.glob("*.py"))
    core_files=sorted(Path(observed.core.__file__).parents[1].rglob("*.py"))
    sources={str(p.resolve()):sha(p) for p in [*implementation_files,*core_files,Path(observed.__file__)]}
    state_source=read(source/'run_state.json')
    planning=read(source/'plan.json')
    contract={"version":"dr_pool_v2_frozen_pilot","source_run":str(source),
        "additional_strategy":None if additional is None else str(additional),
        "strategies":strategies,"sample_counts":list(sample_sizes),"task_denominator":len(sample_sizes)*len(strategies),
        "training_physical_days":4,"observation_hours_by_N":{str(n):list(range(0,24,96//n)) for n in sample_sizes},
        "validation_rows":24,"planning_grid_rows_per_day":24,"planning_history_rows":240,
        "sparse_plan_reoptimized":False,"payments_and_operations_grid":"full original24h, not sparse identification rows",
        "source_input_sha256":source_hashes,"source_missing_files":source_missing,
        "source_sha256":sources,"source_seeds":state_source['provenance']['config']['seeds'],
        "search_seconds_per_start":float(search_seconds),"source_plan_objective":planning['objective'],
        "source_plan_assignment":planning['selected_assignment'],"model_names":list(MODEL_NAMES),
        "scope":"One previously generated paired semi-synthetic DR realization, reused at two nested identification sample sizes; replacement identifier pilot, not independent price superiority or response-calibration validation",
        "truth_to_estimator":False,"mean_response_status":"provisional; menu-specific mean did not pass prior calibration",
        "validation_role":"used for model and initial-source selection; no untouched predictive test",
        "response_model":"held-out whole-day aggregate treatment/control factors, not individual causal responses or a fitted participation-probability model",
        "source_simulation_reuse":True,"source_plan_reuse":True,
        "environment":{"python":sys.version,"executable":sys.executable,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__}}
    identity=fingerprint({k:v for k,v in contract.items() if k!='environment'})
    out.mkdir(parents=True,exist_ok=True)
    with _exclusive_lock(out,identity):
        protocol_path=out/'protocol.json'
        if protocol_path.exists():
            prior=read(protocol_path)
            if not resume:raise FileExistsError("existing run; use --resume without changing inputs/configuration")
            if prior['fingerprint']!=identity:raise ValueError("resume fingerprint changed")
            protocol=prior
        else:
            protocol={**contract,"created_at":now(),"fingerprint":identity}
            write(protocol_path,protocol)
        records=[]
        history=[observed_only(load_pickle(source/'history'/(date+'.pkl'))) for date in source_summary['wzzt_training_dates']]
        for count in sample_sizes:
            for strategy in strategies:
                task=out/'tasks'/f'{strategy}__N{count}'; task.mkdir(parents=True,exist_ok=True)
                record_path=task/'task_record.json'
                if record_path.exists():
                    records.append(read(record_path));continue
                record={"strategy":strategy,"N":count,"status":"running","created_at":now(),"new_identifier_calls":0}
                try:
                    folder=additional if strategy==EXTRA_STRATEGY else source/'strategies'/strategy
                    required=[folder/name for name in ('result.json','action.pkl','validation.pkl')]
                    missing=[str(p) for p in required if not p.is_file()]
                    if missing:
                        record.update(status='source_unavailable',missing_files=missing,
                            source_failure_record=read(required[0]) if required[0].exists() else None)
                    else:
                        prior=read(required[0]); action=load_pickle(required[1]); valid=load_pickle(required[2])
                        train=subsample_training(history+[observed_only(action['simulation'])],count)
                        validation=[observed_only(valid['simulation'])]
                        record.update(assignment=prior['assignment'],planning_audit=prior.get('planning_audit'),
                            periods=_periods(action,valid),training_dates=[str(s['P_terminal'].index[0].date()) for s in train],
                            validation_dates=[str(validation[0]['P_terminal'].index[0].date())],
                            training_rows=count,validation_rows=len(validation[0]['P_terminal']),
                            source_period_sha256={str(p):source_hashes[str(p)] for p in required})
                        result_path=task/'identifier'/'result.json'
                        if result_path.exists():
                            fit=read(result_path)
                            if fit.get('status')!='complete':
                                record.update(status='interrupted_or_failed_identifier',identifier_status=fit.get('status'),
                                    note='Existing identifier checkpoint retained; no automatic re-solve')
                            else:record['reused_completed_identifier']=True
                        else:
                            if fitter is None:
                                from .adapter import fit_observed
                                current_fitter=fit_observed
                            else:current_fitter=fitter
                            record['new_identifier_calls']=1
                            fit=current_fitter(train,validation,task/'identifier',root_label=1,search_seconds=search_seconds)
                        if record['status']=='running':
                            if fit.get('status')!='complete':
                                record.update(status='identifier_failed',identifier_status=fit.get('status'))
                            else:
                                labels=list(train[0]['P_terminal'].columns)
                                # Truth is accessed only after the estimator returns.
                                truth=action['simulation']['truth']['truth_supports']
                                if truth!=valid['simulation']['truth']['truth_supports']:
                                    raise ValueError('source action/validation truth differs')
                                record.update(status='evaluated',evaluation=evaluate_fit(fit,truth,labels),
                                    identifier_seconds=fit.get('seconds',fit.get('elapsed_seconds')),
                                    identifier_result=str(result_path),identifier_source_sha256=fit.get('source_sha256'))
                except Exception as exc:
                    record.update(status='failed',exception={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
                record['finished_at']=now();write(record_path,record);records.append(record)
                write(out/'summary.json',summarize(records,protocol))
        summary=summarize(records,protocol)
        summary['source_inputs_unchanged']=all(Path(p).is_file() and sha(p)==h for p,h in source_hashes.items())
        summary['source_missing_unchanged']=all(not Path(p).exists() for p in source_missing)
        summary['finished_at']=now();summary['protocol_sha256']=sha(protocol_path)
        if not summary['source_inputs_unchanged'] or not summary['source_missing_unchanged']:
            summary['status']='source_changed'
        write(out/'summary.json',summary)
        return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--additional-strategy',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--sample-sizes',type=int,nargs='+',default=list(DEFAULT_SAMPLE_COUNTS))
    args=parser.parse_args(argv)
    result=run(args.source_run,args.output_dir,additional_strategy=args.additional_strategy,resume=args.resume,sample_sizes=args.sample_sizes)
    print(json.dumps({k:result[k] for k in ('status','task_denominator','recorded_tasks','evaluated_tasks','failed_or_unavailable_tasks','search_wall_limited_tasks')},indent=2))
    return 0 if result['status']=='complete' and result['failed_or_unavailable_tasks']==0 else 2
