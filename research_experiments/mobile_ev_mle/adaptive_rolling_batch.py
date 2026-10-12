"""Frozen two-round batch runner and collector-compatible trace normalization.

The controller itself remains unchanged. Process-local wrappers count every
entry to the research AC solver, including exceptions, and delimit history-
only selection and full joint-fit calls. They are always restored. Workers
must be separate processes, not concurrent threads in this process.

The deadline is checked before each independent history. An in-flight history
finishes and is recorded; partial requested batches remain explicit budget
checkpoints. Resume requires exact policy, device, source and runtime headers.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy

try:
    from . import adaptive_rolling_study as rolling
    from .adaptive_milp_study import source_hashes as primary_source_hashes, POPULATION
    from .run_study import dump
except ImportError:
    import adaptive_rolling_study as rolling
    from research_experiments.mobile_ev_mle.adaptive_milp_study import source_hashes as primary_source_hashes, POPULATION
    from research_experiments.mobile_ev_mle.run_study import dump

HERE = Path(__file__).resolve().parent
METHOD = "rolling_refit_milp"
METHODS = (METHOD,)
ROUNDS = (1, 2)
ARCHIVE_CONTRACT = dict(version="rolling_joint_states_v1",
    latent="round_{round}__{candidate}__fit_{p|q|source_p|source_q}",
    observations="round_{round}_joint_{p|q|v|master_p|master_q|root_v}",
    executed="round_{round}_observed_{channel}", truth_needed_for_replay=False)


class TraceExecutionFailure(RuntimeError):
    """Retain exact call/time accounting even when no full trace was returned."""
    def __init__(self, error, metrics):
        super().__init__(str(error))
        self.metrics = metrics


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        default=lambda a:np.asarray(a).tolist()).encode()).hexdigest()


def source_hashes():
    hashes = dict(primary_source_hashes())
    hashes["adaptive_rolling_study.py"] = hashlib.sha256(Path(rolling.__file__).read_bytes()).hexdigest()
    hashes["adaptive_rolling_batch.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # Staging runs can import helpers from a different path. Record actual
    # loaded helper code too; installed package runs produce matching hashes.
    for function in (rolling.build_history_context, rolling.adaptive_search, rolling.execute_prefix):
        module = sys.modules[function.__module__]
        hashes["loaded::"+Path(module.__file__).name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    return hashes


def _public_config():
    access = np.ones((8, 3), bool); access[:2, 2] = False
    return rolling.EVConfig(access=access)


def batch_config(*, phase, count, start_seed, policy=None, stress=None):
    if phase == "eval": phase = "evaluation"
    if phase not in ("pilot", "calibration", "evaluation", "stress"):
        raise ValueError("unknown batch phase")
    if (isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0
            or isinstance(start_seed, bool) or not isinstance(start_seed, (int, np.integer)) or start_seed < 0):
        raise ValueError("count and start_seed must be nonnegative integers")
    if stress not in (None, "bias", "source_drift", "load_trend"):
        raise ValueError("unsupported rolling stress")
    if phase == "calibration" and stress is not None:
        raise ValueError("reference calibration cannot silently use a stress population")
    p = rolling.rolling_policy_spec(policy)
    ev = _public_config(); ids = rolling.policy_identifiers(p, ev, stress=stress)
    hashes = source_hashes()
    requirements = dict(planned_horizons=[8, 6], executed_slots=[2, 6], original_grid_task_kwh=6.,
        source_node=1, tracking_gain=1., audit_bank="star,clade_01,clade_02,clade_12 every round",
        carry=["SOC", "site", "grid_energy", "throughput", "access", "terminal_constraints"],
        allowed_travel_slots=1, statistical_pruning=False)
    result = dict(phase=phase, count=int(count), start_seed=int(start_seed), policy=p,
        population={**POPULATION, "observation_policy": rolling.ROLLING_VERSION,
            "history_points":p["history_points"], "future_points":8, "stress":stress},
        policy_id=_hash(dict(controller_policy_id=ids["policy_id"], policy=p, initial_ev=asdict(ev),
            requirements=requirements, source_sha256=hashes)), population_id=ids["population_id"],
        controller_policy_id=ids["policy_id"], source_sha256=hashes, stress=stress, methods=list(METHODS),
        mandatory_rounds=2, round_ids=list(ROUNDS), initial_ev_config=asdict(ev),
        operational_requirements=requirements, latent_archive_contract=ARCHIVE_CONTRACT,
        history_unit="one independent generator path and its entire two-round shadow trajectory",
        requires_independent_rolling_calibration=True, may_reuse_one_round_gates=False,
        predeclared_schedule=dict(fast_pilot_seconds_threshold=45., alpha_total=.1,
            if_pilot_mean_at_most_threshold=dict(calibration=64, evaluation=96),
            otherwise=dict(calibration=32, evaluation=64), choose_before_calibration=True,
            limited_power=True),
        runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    # Normalize array fields exactly as they will be serialized, for resume.
    return json.loads(json.dumps(result, default=lambda a:np.asarray(a).tolist()))


@contextmanager
def _count_ac_calls():
    from research_experiments.mobile_ev_mle import likelihood, symmetric_ac_local, simulation
    count = {"count":0}
    originals = []
    for module in (likelihood, symmetric_ac_local, simulation):
        original = module.ac_predict
        def counted(*args, _original=original, **kwargs):
            count["count"] += 1
            return _original(*args, **kwargs)
        originals.append((module, original)); module.ac_predict = counted
    try:
        yield count
    finally:
        for module, original in originals:
            module.ac_predict = original


def _counted_trial(seed, *, policy, stress, output):
    """Wrap whole trial and delimit calls without adding selector arguments."""
    metrics = dict(selection_calls=[], joint_fit_calls=[])
    original_selection, original_fit = rolling.plan_round, rolling.fit_bank
    with _count_ac_calls() as counter:
        def counted_selection(*args, **kwargs):
            before = counter["count"]; began = perf_counter(); succeeded = False
            try:
                result = original_selection(*args, **kwargs); succeeded = True
                return result
            finally:
                metrics["selection_calls"].append(dict(round=len(metrics["selection_calls"])+1,
                    ac_calls=counter["count"]-before, seconds=perf_counter()-began, returned=succeeded))
        def counted_fit(*args, **kwargs):
            before = counter["count"]; began = perf_counter(); succeeded = False
            try:
                result = original_fit(*args, **kwargs); succeeded = True
                return result
            finally:
                metrics["joint_fit_calls"].append(dict(round=len(metrics["joint_fit_calls"])+1,
                    ac_calls=counter["count"]-before, seconds=perf_counter()-began, returned=succeeded))
        rolling.plan_round, rolling.fit_bank = counted_selection, counted_fit
        began = perf_counter()
        error = None
        try:
            result = rolling.rolling_trial(seed, policy=policy, stress=stress, output=output)
        except Exception as caught:
            error = caught
        finally:
            rolling.plan_round, rolling.fit_bank = original_selection, original_fit
            metrics["whole_trial_ac_calls"] = counter["count"]
            metrics["whole_trial_seconds"] = perf_counter()-began
            metrics["unclassified_ac_calls"] = counter["count"]-sum(
                x["ac_calls"] for k in ("selection_calls", "joint_fit_calls") for x in metrics[k])
            metrics["call_definition"] = "entries to research likelihood/symmetric/simulation ac_predict; exceptions included; not flow iterations"
    if error is not None:
        raise TraceExecutionFailure(error, metrics) from error
    return result, metrics


def normalize_trial(raw, metrics, config):
    """Keep original rounds/selections and add standard method/round records."""
    if raw.get("policy_id") != config["controller_policy_id"]:
        raise ValueError("returned controller law differs from the frozen rolling policy")
    if [r.get("round") for r in raw["rounds"]] != list(ROUNDS) or len(raw["selections"]) != 2:
        raise ValueError("full rolling trace must preserve both mandatory rounds")
    if raw["candidate_names"] != ["star", "clade_01", "clade_02", "clade_12"]:
        raise ValueError("all four audit candidates must be retained in fixed order")
    selector = {row["round"]:row for row in metrics["selection_calls"]}
    fitting = {row["round"]:row for row in metrics["joint_fit_calls"]}
    records, selection_rounds = [], {}
    for index, row in enumerate(raw["rounds"]):
        r = row["round"]; selected = raw["selections"][index]
        selection_status = ("complete" if selected.get("status") == "complete" else "selection_failure_fallback")
        fit_count = fitting.get(r)
        executed = row["executed"]; u = np.asarray(executed["u_kw"], float)
        action = dict(name=f"rolling_round_{r}", family=METHOD, actual_executed_prefix=True,
            p=(u/rolling.BASE_KW).tolist(), q=np.zeros_like(u).tolist(), u_kw=u.tolist(),
            energy_kwh=executed["grid_energy_kwh"], max_kw=float(np.max(abs(u), initial=0.)),
            sites=executed["sites"], driving_energy_kwh=executed["driving_energy_kwh"])
        returned_calls = sum(f.get("diagnostics", {}).get("ac_evaluations", 0) for f in row["fits"])
        records.append({**row, "method":METHOD, "status":"complete" if row["complete"] else "joint_fit_failure",
            "selection_status":selection_status, "action":action, "actual_action":action,
            "seconds":None if fit_count is None else fit_count["seconds"],
            "ac_calls":None if fit_count is None else fit_count["ac_calls"],
            "ac_calls_exact_including_failed_fits":fit_count is not None,
            "returned_fit_ac_calls":returned_calls, "returned_fit_calls_are_lower_subset":True,
            "true_covered":raw.get("true_index") is not None,
            "observation_hash":row["joint_observation_hash"]})
        search = selected.get("proxy_search", {})
        selection_rounds[str(r)] = dict(round=r, status=selection_status,
            total_selection_seconds=selector.get(r, {}).get("seconds"),
            selection_ac_predict_calls=selector.get(r, {}).get("ac_calls"),
            original_selection_index=index, proxy_lower=search.get("proxy_lower"), proxy_upper=search.get("proxy_upper"),
            absolute_gap=search.get("absolute_gap"), relative_gap=search.get("relative_gap"),
            proxy_stop_reason=search.get("status"), proxy_gap_met=search.get("status")=="proxy_gap_met",
            qp_count=search.get("qp_count"), nonlinear_infimum_computed=False)
    all_complete = all(r["status"] == "complete" for r in selection_rounds.values())
    method = dict(status="complete" if all_complete else "selection_failure_fallback", rounds=selection_rounds,
        total_selection_seconds=sum(x["seconds"] for x in metrics["selection_calls"]),
        selection_ac_predict_calls=sum(x["ac_calls"] for x in metrics["selection_calls"]),
        full_round_selection_location="top-level selections[round-1]; proxy_search holds frozen bounds/iteration records",
        rounds_are_separately_frozen_proxies=True, nonlinear_separation_certificate=False)
    result = {**raw, "records":records, "selection":dict(methods={METHOD:method},
        selector_total_seconds=method["total_selection_seconds"], selector_ac_predict_calls=method["selection_ac_predict_calls"]),
        "selection_uses_truth":False, "selection_uses_future_noise":False, "mandatory_rounds":2,
        "ac_call_metrics":metrics, "policy_id":config["policy_id"], "population_id":config["population_id"],
        "source_sha256":config["source_sha256"], "controller_policy_id":raw["policy_id"],
        "latent_archive_contract":ARCHIVE_CONTRACT, "seconds":metrics["whole_trial_seconds"],
        "observation_archive_expected":True, "execution_failure":False}
    return result


def _deadline(value):
    if value is None: return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("deadline requires explicit timezone")
    return parsed.astimezone(timezone.utc)


def _failure_trial(seed, config, error):
    # Annotation is evaluation-only after a failed trace. It does not seed a
    # selector. The predeclared four-model bank covers this generator class.
    path, policy_seed, topology, case = rolling._evaluation_path(seed, config["policy"], config["stress"])
    metrics = getattr(error, "metrics", {})
    partial_calls = {x["round"]:x for x in metrics.get("joint_fit_calls", [])}
    records = [dict(method=METHOD, round=r, status="joint_fit_failure", losses=[None]*4,
        success=[False]*4, fits=[], seconds=partial_calls.get(r, {}).get("seconds"),
        ac_calls=partial_calls.get(r, {}).get("ac_calls"), selection_status="selection_failure_fallback",
        action={}, actual_action={}, full_round_evidence_unavailable_due_to_trace_failure=True,
        joint_fit_call_was_entered=r in partial_calls,
        fit_failures={"trace":str(error), "semantics":"required full trajectory incomplete; placeholder is not a claim every optimizer failed"}) for r in ROUNDS]
    score = rolling.policy_trial_scores(np.full((1, 2, 4), np.inf), topology,
        success=np.zeros((1, 2, 4), bool), coverage_ok=True)
    return dict(seed=seed, history_id=str(seed), policy_seed=policy_seed, case=case, truth_topology=topology,
        true_index=topology, candidate_names=["star", "clade_01", "clade_02", "clade_12"], records=records,
        rounds=[], selections=[], score_record=asdict(score), independent_history_unit=True, mandatory_rounds=2,
        selection=dict(methods={METHOD:dict(status="selection_failure_fallback")}),
        selection_uses_truth=False, selection_uses_future_noise=False, execution_failure=True, error=str(error),
        policy_id=config["policy_id"], population_id=config["population_id"], source_sha256=config["source_sha256"],
        seconds=metrics.get("whole_trial_seconds", 0.), ac_call_metrics=metrics,
        observation_archive_expected=False, latent_archive_contract=ARCHIVE_CONTRACT,
        may_reuse_one_round_gates=False, truth=rolling.truth_metadata(path))


def _publish_archive(output, seed):
    source = output/f"rolling_{seed}_observations.npz"
    target = output/f"trial_{seed}_observations.npz"
    if not source.exists():
        raise ValueError("complete trace did not save its mandatory joint-state archive")
    if target.exists():
        if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
            raise ValueError("existing pending observation archive differs")
        source.unlink()
    else:
        source.rename(target)


def run_batch(output, *, phase, count, start_seed, policy=None, stress=None, deadline_utc=None, _now=None):
    now = _now or (lambda:datetime.now(timezone.utc))
    cutoff = _deadline(deadline_utc)
    config = batch_config(phase=phase, count=count, start_seed=start_seed, policy=policy, stress=stress)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    config_file = output/"config.json"
    if config_file.exists() and json.loads(config_file.read_text(encoding="utf-8")) != config:
        raise ValueError("refusing resume with changed policy, source, population, device or runtime header")
    dump(config_file, config)
    began = perf_counter(); rows=[]; stop="count_complete"
    dump(output/"progress.json", dict(config=config, completed=0, state="ready", deadline_utc=deadline_utc))
    for index in range(config["count"]):
        seed = config["start_seed"]+index; filename=output/f"trial_{seed}.json"
        if filename.exists():
            row = json.loads(filename.read_text(encoding="utf-8"))
            if any(row.get(key) != config[key] for key in ("policy_id", "population_id", "source_sha256")):
                raise ValueError("saved trial differs from the frozen batch configuration")
            if row.get("observation_archive_expected") and not (output/f"trial_{seed}_observations.npz").exists():
                raise ValueError("saved completed trace is missing its mandatory observation archive")
        else:
            if cutoff is not None and now() >= cutoff:
                stop="wall_clock_budget_checkpoint"; break
            if source_hashes() != config["source_sha256"]:
                stop="source_changed_checkpoint"; break
            dump(output/"progress.json", dict(config=config, completed=len(rows), in_flight_seed=seed,
                state="history_running", started_utc=now().isoformat(), deadline_utc=deadline_utc,
                in_flight_history_finishes=True))
            try:
                raw, metrics = _counted_trial(seed, policy=config["policy"], stress=stress, output=output)
                row = normalize_trial(raw, metrics, config)
                _publish_archive(output, seed)
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                row = _failure_trial(seed, config, error)
            if source_hashes() != config["source_sha256"]:
                dump(output/f"pending_source_drift_{seed}.json", row)
                stop="source_changed_during_history_checkpoint"; break
            dump(filename, row)
            raw_json = output/f"rolling_{seed}.json"
            if raw_json.exists() and not row["execution_failure"]:
                # Full raw rounds/selections already retained verbatim in row.
                raw_json.unlink()
        rows.append(row)
        print(json.dumps(dict(phase=config["phase"], trial=index+1, count=count, seed=seed,
            case=row["case"], truth_topology=row["truth_topology"], seconds=row["seconds"],
            complete=row["score_record"]["complete"], execution_failure=row.get("execution_failure", False)),
            ensure_ascii=False), flush=True)
        dump(output/"progress.json", dict(config=config, completed=len(rows), last_seed=seed, state="history_recorded"))
    summary = dict(config=config, history_count=len(rows), requested_history_count=count, stop_reason=stop,
        all_requested_histories_attempted=len(rows)==count, complete_history_count=sum(r["score_record"]["complete"] for r in rows),
        seconds=sum(r["seconds"] for r in rows), invocation_seconds=perf_counter()-began,
        selection_failure_counts={METHOD:sum(r["selection"]["methods"][METHOD]["status"]=="selection_failure_fallback" for r in rows)},
        joint_fit_failure_counts={METHOD:sum(any(x["status"]!="complete" for x in r["records"]) for r in rows)},
        execution_failure_count=sum(r.get("execution_failure", False) for r in rows),
        truth_topology_counts={str(i):sum(r["truth_topology"]==i for r in rows) for i in range(4)},
        mandatory_rounds=2, expansion_justified=False, may_reuse_one_round_gates=False)
    dump(output/"summary.json", summary)
    dump(output/"progress.json", dict(config=config, completed=len(rows), state="count_complete" if len(rows)==count else "checkpoint",
        stop_reason=stop, all_requested_histories_attempted=len(rows)==count))
    return config, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "calibration", "evaluation", "eval", "stress"), default="pilot")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--start-seed", type=int, default=910001)
    parser.add_argument("--output", type=Path, default=HERE/"outputs"/"adaptive_rolling"/"pilot_v1")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--deadline-utc", help="aware ISO UTC; checked before each independent history, in-flight history finishes")
    parser.add_argument("--stress", choices=("bias", "source_drift", "load_trend"))
    args = parser.parse_args()
    policy = None if args.policy is None else json.loads(args.policy.read_text(encoding="utf-8"))
    run_batch(args.output, phase=args.phase, count=args.count, start_seed=args.start_seed,
        policy=policy, stress=args.stress, deadline_utc=args.deadline_utc)


if __name__ == "__main__": main()
