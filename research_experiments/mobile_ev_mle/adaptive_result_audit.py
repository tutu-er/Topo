"""Independent history-unit collection and calibration for adaptive EV studies.

The collector never selects an action or refits a topology. It checks saved
outputs, applies frozen calibration thresholds, and keeps failed histories in
the denominator. Optional AC replay uses saved fitted states, never truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

try:
    from .adaptive_statistics import (policy_trial_scores, calibrate_policy_gates,
        apply_candidate_gates, history_event_summary, paired_binary_effects,
        paired_continuous_effects)
except ImportError:
    from adaptive_statistics import (policy_trial_scores, calibrate_policy_gates,
        apply_candidate_gates, history_event_summary, paired_binary_effects,
        paired_continuous_effects)


def _finite(value):
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, np.generic):
        return _finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite(v) for v in value]
    return value


def _dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_finite(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(_finite(v), ensure_ascii=False)
                             if isinstance(v, (list, tuple, dict)) else _finite(v)
                             for k, v in row.items()})


def _numbers(value):
    return np.asarray([np.nan if v is None else float(v) for v in value], float)


def _measurement_summary(values):
    array = _numbers(values)
    valid = np.isfinite(array)
    kept = array[valid]
    return dict(history_count=len(array), available_count=int(valid.sum()),
                missing_or_failed_count=int((~valid).sum()),
                mean=None if len(kept) == 0 else float(kept.mean()),
                median=None if len(kept) == 0 else float(np.median(kept)),
                minimum=None if len(kept) == 0 else float(kept.min()),
                maximum=None if len(kept) == 0 else float(kept.max()),
                descriptive_complete_subset=True)


def validate_numeric_bounds(value, *, tolerance=1e-7, path="root"):
    """Check bound direction without mixing proxy, QP, MILP, or AC quantities."""
    checked = 0
    if isinstance(value, dict):
        for lo_key, hi_key in (("q_lower", "q_upper"),
                               ("proxy_lower", "proxy_upper"),
                               ("none_proxy_lower", "none_proxy_upper")):
            if lo_key in value and hi_key in value:
                lo, hi = value[lo_key], value[hi_key]
                if lo is not None and hi is not None:
                    lo, hi = float(lo), float(hi)
                    if not math.isfinite(lo) or not math.isfinite(hi):
                        raise ValueError(f"nonfinite numerical bound at {path}")
                    if lo > hi + tolerance * max(1.0, abs(lo), abs(hi)):
                        raise ValueError(f"invalid lower > upper at {path}: {lo_key}, {hi_key}")
                    checked += 1
        # All stored solver records are maximization masters; best-bound is
        # above incumbent. A direction-lower master's bound only concerns that
        # finite-direction master and does not bound the full proxy above.
        if value.get("kind") in ("direction_lower", "secant_upper"):
            lo, hi = value.get("incumbent"), value.get("bound")
            if lo is not None and hi is not None:
                lo, hi = float(lo), float(hi)
                if lo > hi + tolerance * max(1.0, abs(lo), abs(hi)):
                    raise ValueError(f"invalid max-MILP incumbent > best-bound at {path}")
                checked += 1
        for key, child in value.items():
            checked += validate_numeric_bounds(child, tolerance=tolerance,
                                                path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            checked += validate_numeric_bounds(child, tolerance=tolerance,
                                                path=f"{path}[{index}]")
    return checked


def load_batch(folder):
    folder = Path(folder)
    config_path = folder / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("policy_id", "population_id", "source_sha256", "methods"):
        if key not in config:
            raise ValueError(f"missing frozen batch context: {key}")
    if not config["source_sha256"]:
        raise ValueError("source hash map must be nonempty")
    if config.get("policy", {}).get("gates_influence_shadow_policy", False):
        raise ValueError("collector requires the frozen shadow policy without gate-driven pruning")
    methods = tuple(config["methods"])
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("method list must be nonempty and unique")
    files = sorted(folder.glob("trial_[0-9]*.json"))
    rows = [json.loads(file.read_text(encoding="utf-8")) for file in files]
    ids = [str(row["history_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate independent history IDs")
    for row in rows:
        if not row.get("independent_history_unit", False):
            raise ValueError("one independent historical path must be the declared statistical unit")
        if row.get("selection_uses_truth") or row.get("selection_uses_future_noise"):
            raise ValueError("selector/evaluator separation was not maintained")
        records = row["records"]
        if set(record["method"] for record in records) != set(methods):
            raise ValueError("trial did not preserve all frozen methods")
        names = row["candidate_names"]
        if not names or len(set(names)) != len(names):
            raise ValueError("candidate names must be nonempty and unique")
        for record in records:
            if len(record["losses"]) != len(names) or len(record["success"]) != len(names):
                raise ValueError("candidate loss/success axes do not match names")
        validate_numeric_bounds(row)
    return dict(folder=folder, config=config, rows=rows,
                manifest={"config_sha256": _sha(config_path),
                          "trial_sha256": {file.name: _sha(file) for file in files},
                          "observation_archive_sha256": {file.name: _sha(file)
                              for file in sorted(folder.glob("trial_[0-9]*_observations.npz"))}},
                requested_history_count=config.get("count"),
                completed_history_count=len(rows))


def _records_by_method(row, methods):
    grouped = {method: [] for method in methods}
    for record in row["records"]:
        grouped[record["method"]].append(record)
    for method in methods:
        grouped[method].sort(key=lambda record: record.get("round", 0))
        rounds = [record.get("round", 0) for record in grouped[method]]
        if len(set(rounds)) != len(rounds):
            raise ValueError("duplicate method/round records")
    return grouped


def _score(row, methods):
    grouped = _records_by_method(row, methods)
    rounds = [len(grouped[method]) for method in methods]
    if len(set(rounds)) != 1:
        raise ValueError("all methods must preserve the same declared round count")
    losses = [[_numbers(record["losses"]) for record in grouped[method]] for method in methods]
    success = [[record["success"] for record in grouped[method]] for method in methods]
    return policy_trial_scores(losses, row.get("true_index"), success=success,
                               coverage_ok=row.get("true_index") is not None)


def _bind_batches(calibration, evaluation, *, allow_stress=False):
    a, b = calibration["config"], evaluation["config"]
    for key in ("policy_id", "population_id", "source_sha256", "methods"):
        if a[key] != b[key]:
            raise ValueError(f"calibration/evaluation frozen context differs: {key}")
    if a.get("stress") is not None:
        raise ValueError("calibration must come from the declared reference population")
    stressed = b.get("stress") is not None
    if stressed and not allow_stress:
        raise ValueError("stress evaluation is out of population; explicitly allow a descriptive stress report")
    overlap = set(str(row["history_id"]) for row in calibration["rows"]) & set(
        str(row["history_id"]) for row in evaluation["rows"])
    if overlap:
        raise ValueError("evaluation histories overlap calibration histories")
    # Identical labels alone cannot make one-round calibration valid for a
    # longer trace. The max-method/round score must cover the frozen trace.
    signatures = []
    for batch in (calibration, evaluation):
        for row in batch["rows"]:
            grouped = _records_by_method(row, a["methods"])
            signatures.append(tuple((method, tuple(record.get("round", 0)
                for record in grouped[method])) for method in a["methods"]))
    if signatures and any(signature != signatures[0] for signature in signatures):
        raise ValueError("calibration/evaluation method-round trace differs")
    return stressed


def replay_saved_fit(batch_folder, row, record, fit):
    """Recompute saved-state Gaussian loss using independent AC evaluation.

    Missing arrays are explicitly unavailable. No optimizer is called. The
    check certifies the saved objective evaluation, not attainment of an MLE.
    """
    filename = Path(batch_folder) / f"trial_{row['seed']}_observations.npz"
    if not filename.exists():
        return dict(status="unavailable", reason="observation archive absent")
    round_id=record.get('round',0)
    prefix = f"round_{round_id}__{fit['name']}__fit_" if round_id else f"{record['method']}__{fit['name']}__fit_"
    with np.load(filename, allow_pickle=False) as archive:
        keys = [prefix + key for key in ("p", "q", "source_p", "source_q")]
        if not all(key in archive for key in keys):
            return dict(status="unavailable", reason="fitted latent states not saved")
        states = {key: np.array(archive[prefix + key]) for key in ("p", "q", "source_p", "source_q")}
        observed = {}
        for key in ("p", "q", "v", "master_p", "master_q", "root_v"):
            if round_id:
                key_name=f'round_{round_id}_joint_{key}'
                if key_name not in archive:return dict(status='unavailable',reason=f'round channel {key} absent')
                observed[key]=np.array(archive[key_name]);continue
            h, f = f"history_{key}", f"{record['method']}_{key}"
            if h not in archive or f not in archive:
                return dict(status="unavailable", reason=f"channel {key} absent")
            observed[key] = np.concatenate((archive[h], archive[f]))
    try:
        from .likelihood import ac_predict
        from .candidates import RadialCandidate
        from .simulation import model
    except ImportError:
        from research_experiments.mobile_ev_mle.likelihood import ac_predict
        from research_experiments.mobile_ev_mle.candidates import RadialCandidate
        from research_experiments.mobile_ev_mle.simulation import model
    n = observed["p"].shape[1]
    candidate = RadialCandidate.from_supports(fit["name"], n, fit["supports"])
    meter = model(n)
    physical = ac_predict(candidate, fit["r_pu"], fit["x_pu"],
                          states["p"], states["q"], states["source_p"],
                          states["source_q"], observed["root_v"])
    residuals, normalizer = [], 0.0
    for key in ("p", "q", "v", "master_p", "master_q"):
        bias = getattr(meter, "bias_" + key) if key in ("p", "q", "v") else 0.0
        sigma = np.broadcast_to(getattr(meter, key + "_sigma"), observed[key].shape)
        residuals.append(((physical[key] + bias - observed[key]) / sigma).ravel())
        normalizer += float(np.log(2 * np.pi * sigma**2).sum())
    loss = float(np.concatenate(residuals) @ np.concatenate(residuals))
    loglik = -0.5 * (loss + normalizer)
    difference = loss - float(fit["loss"])
    loglik_difference = loglik - float(fit["loglik"])
    valid = abs(difference) <= 1e-6 * max(1.0, loss) and abs(loglik_difference) <= 1e-6 * max(1.0, abs(loglik))
    return dict(status="verified" if valid else "mismatch", replay_loss=loss,
                replay_loglik=loglik, loss_difference=difference,
                loglik_difference=loglik_difference,
                max_ac_equation_residual=physical["ac_equation_residual"],
                max_power_balance_residual=physical["power_balance_residual"],
                global_optimum_certified=False)


def _selection_for_record(row,record):
    selection=row.get('selection',{}).get('methods',{}).get(record['method'],{})
    round_id=record.get('round',0)
    if not round_id:return selection
    leaf=selection.get('rounds',{}).get(str(round_id),{})
    original_index=leaf.get('original_selection_index')
    originals=row.get('selections',[])
    if original_index is not None and 0<=original_index<len(originals):
        original=originals[original_index]
        return {**original,**original.get('proxy_search',{}),**leaf}
    return leaf


def _flat_evidence(batch, *, replay=True):
    fits, actions, bounds = [], [], []
    for row in batch["rows"]:
        common = dict(history_id=row["history_id"], seed=row["seed"],
                      truth_topology=row.get("truth_topology"), case=row.get("case"),
                      stress=row.get("stress"))
        for record in row["records"]:
            method = record["method"]
            base = dict(common, method=method, round=record.get("round", 0))
            selection = _selection_for_record(row,record)
            action = record.get("action", {})
            actions.append(dict(base, action_name=action.get("name"),
                energy_kwh=action.get("energy_kwh"), max_kw=action.get("max_kw"),
                sites=action.get("sites"), p=action.get("p"), q=action.get("q"),
                selection_status=record.get("selection_status"), fit_status=record.get("status"),
                actual_voltage_min=record.get("voltage_min"), actual_voltage_max=record.get("voltage_max"),
                actual_voltage_ok=record.get("operational_voltage_ok"),
                evaluation_seconds=record.get("seconds"), joint_fit_ac_calls=record.get("ac_calls"),
                selection_seconds=selection.get("total_selection_seconds"),
                selection_ac_calls=selection.get("selection_ac_predict_calls"),
                selector_total_seconds=row.get("selection", {}).get("selector_total_seconds"),
                selector_total_ac_calls=row.get("selection", {}).get("selector_ac_predict_calls"),
                impedance_relative_rmse=record.get("impedance_relative_rmse"),
                true_loss=record.get("true_loss"), true_gap=record.get("true_gap")))
            for fit in record.get("fits", []):
                verification = replay_saved_fit(batch["folder"], row, record, fit) if replay else dict(status="not_requested")
                fits.append(dict(base, candidate=fit["name"], loss=fit["loss"],
                    loglik=fit["loglik"], success=fit["success"], r_pu=fit.get("r_pu"),
                    x_pu=fit.get("x_pu"), supports=fit.get("supports"),
                    global_optimum_certified=False, **{"ac_audit_"+k: v for k, v in verification.items()}))
            if selection:
                rollback = selection.get("actual_rollback", [])
                errors = [x.get("joint_taylor_error_norm") for x in rollback if x.get("joint_taylor_error_norm") is not None]
                discrepancies = [x["actual_soft_cost"] - x["affine_soft_cost"] for x in rollback
                                 if x.get("actual_soft_cost") is not None and x.get("affine_soft_cost") is not None]
                valid_rollback = [x for x in rollback if x.get("affine_soft_cost") is not None]
                active = min(valid_rollback, key=lambda x:x["affine_soft_cost"]) if valid_rollback else {}
                bounds.append(dict(base, level="selected_proxy", lower=selection.get("proxy_lower"),
                    upper=selection.get("proxy_upper"), absolute_gap=selection.get("absolute_gap"),
                    relative_gap=selection.get("relative_gap"), proxy_stop_reason=selection.get("proxy_stop_reason"),
                    proxy_gap_met=selection.get("proxy_gap_met"), qp_count=selection.get("qp_count"),
                    scope="frozen affine proxy only", ac_taylor_norm_max=max(errors, default=None),
                    ac_cost_minus_affine_min=min(discrepancies, default=None),
                    ac_cost_minus_affine_max=max(discrepancies, default=None),
                    active_pair_name=active.get("pair",active.get("name")),
                    active_affine_witness_cost=active.get("affine_soft_cost"),
                    active_ac_witness_cost=active.get("actual_soft_cost"),
                    active_actual_history_costs=active.get("actual_history_costs"),
                    active_actual_future_distance_squared=active.get("actual_future_distance_squared"),
                    active_taylor_error_norm=active.get("joint_taylor_error_norm"),
                    minimum_ac_witness_cost=min((x["actual_soft_cost"] for x in rollback
                        if x.get("physically_feasible",False) and x.get("actual_soft_cost") is not None), default=None),
                    rollback_failure_count=sum(not x.get("physically_feasible", False) for x in rollback),
                    nonlinear_infimum_computed=False))
                for index, iteration in enumerate(selection.get("records", [])):
                    for pair in iteration.get("pair_values", []):
                        bounds.append(dict(base, level="inner_qp", iteration=index,
                            pair=pair.get("name"), lower=pair.get("q_lower"), upper=pair.get("q_upper"),
                            numeric_gap=pair.get("numeric_gap"), scope="box-QP at one fixed control"))
                for solver in selection.get("solver_records", []):
                    bounds.append(dict(base, level="MILP_master", kind=solver.get("kind"),
                        incumbent=solver.get("incumbent"), best_bound=solver.get("bound"),
                        mip_gap=solver.get("mip_gap"), solver_status=solver.get("status"),
                        scope="finite-direction master" if solver.get("kind")=="direction_lower" else "secant upper master"))
    return fits, actions, bounds


def _events_for_history(row, methods, calibration, config):
    grouped = _records_by_method(row, methods)
    names = row["candidate_names"]
    true = row.get("true_index")
    result = {}
    for method in methods:
        gates = [apply_candidate_gates(_numbers(record["losses"]), calibration,
            success=record["success"], history_id=row["history_id"],
            policy_id=config["policy_id"], population_id=config["population_id"],
            candidate_names=names) for record in grouped[method]]
        last = grouped[method][-1]
        retained = ~np.asarray(gates[-1]["combined_reject"], bool)
        relative_retained = ~np.asarray(gates[-1]["relative_reject"], bool)
        covered = true is not None
        false_reject = covered and any(gate["combined_reject"][true] for gate in gates)
        wrong = np.ones(len(names), bool)
        if covered:
            wrong[true] = False
        wrong_count = int(wrong.sum())
        eliminated = int(np.sum(wrong & ~retained))
        complete = all(gate["complete"] for gate in gates)
        physical = all(record.get("operational_voltage_ok", False) for record in grouped[method])
        selection_failure = any(record.get("selection_status")=="selection_failure_fallback" for record in grouped[method])
        correct = bool(covered and complete and physical and retained.sum()==1 and retained[true])
        result[method] = dict(gates=gates, retained=retained.tolist(),
            false_rejection=bool(false_reject), truth_omission=not covered,
            truth_loss_event=bool(not covered or false_reject),
            unique_correct_retention=correct,
            algorithm_unique_correct_retention=correct and not selection_failure,
            wrong_unique_retention=bool(retained.sum()==1 and (not covered or not retained[true])),
            unresolved=bool(retained.sum()>1), empty_retained=bool(retained.sum()==0),
            relative_unique_correct_retention=bool(covered and complete and physical and relative_retained.sum()==1 and relative_retained[true]),
            selection_failure=selection_failure, fit_failure=not complete,
            physical_failure=not physical,
            bank_adequacy_alarm=any(gate["absolute_alarm"] for gate in gates),
            any_wrong_candidate_eliminated=eliminated>0,
            all_wrong_candidates_eliminated=bool(wrong_count and eliminated==wrong_count),
            wrong_candidates_eliminated=eliminated, wrong_candidate_count=wrong_count,
            wrong_elimination_fraction=eliminated/wrong_count if wrong_count else None,
            true_loss=last.get("true_loss"), true_gap=last.get("true_gap"),
            impedance_relative_rmse=last.get("impedance_relative_rmse"),
            evaluation_seconds=None if any(record.get('seconds') is None for record in grouped[method]) else sum(record['seconds'] for record in grouped[method]),
            joint_fit_ac_calls=None if any(record.get('ac_calls') is None for record in grouped[method]) else sum(record['ac_calls'] for record in grouped[method]))
    return result


EVENTS = ("false_rejection", "truth_omission", "truth_loss_event", "unique_correct_retention",
    "algorithm_unique_correct_retention", "relative_unique_correct_retention", "wrong_unique_retention",
    "unresolved", "empty_retained", "selection_failure", "fit_failure", "physical_failure",
    "bank_adequacy_alarm", "any_wrong_candidate_eliminated", "all_wrong_candidates_eliminated")
METRICS = ("wrong_elimination_fraction", "true_loss", "true_gap", "impedance_relative_rmse",
           "evaluation_seconds", "joint_fit_ac_calls")


def _aggregate(rows, evaluated, methods):
    ids = [str(row["history_id"]) for row in rows]
    summary = {}
    for method in methods:
        summary[method] = dict(events={key: history_event_summary(ids,
            np.asarray([item[method][key] for item in evaluated], bool)) for key in EVENTS},
            measurements={key: _measurement_summary([item[method][key] for item in evaluated]) for key in METRICS},
            wrong_candidates_eliminated_total=sum(item[method]["wrong_candidates_eliminated"] for item in evaluated),
            wrong_candidate_count_total=sum(item[method]["wrong_candidate_count"] for item in evaluated),
            candidate_counts_are_not_independent_statistical_units=True)
    paired = {}
    for control in ("constant_fixed", "none"):
        if control not in methods:
            continue
        paired[control] = {}
        for method in methods:
            if method == control:
                continue
            events = {key: paired_binary_effects(ids,
                np.asarray([item[method][key] for item in evaluated], bool),
                np.asarray([item[control][key] for item in evaluated], bool))
                for key in ("unique_correct_retention", "false_rejection", "any_wrong_candidate_eliminated", "unresolved", "fit_failure")}
            measurements = {}
            for key in METRICS:
                a = _numbers([item[method][key] for item in evaluated])
                b = _numbers([item[control][key] for item in evaluated])
                valid = np.isfinite(a) & np.isfinite(b)
                measurements[key] = {**paired_continuous_effects(np.asarray(ids)[valid], a[valid], b[valid]),
                    "missing_pair_count": int((~valid).sum()), "full_history_count": len(ids),
                    "descriptive_complete_subset": True}
            paired[control][method] = dict(events=events, measurements=measurements)
    return dict(methods=summary, paired_against=paired)


def pilot_summary(input, out, *, replay=True):
    batch = load_batch(input)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fits, actions, bounds = _flat_evidence(batch, replay=replay)
    _csv(out/"fits.csv", fits); _csv(out/"actions.csv", actions); _csv(out/"numerical_bounds.csv", bounds)
    mismatch = [fit for fit in fits if fit.get("ac_audit_status")=="mismatch"]
    summary = dict(config=batch["config"], manifest=batch["manifest"],
        requested_history_count=batch["requested_history_count"], history_count=len(batch["rows"]),
        calibration_performed=False, inferential_gate_claim=False,
        topology_counts={str(k): sum(row.get("truth_topology")==k for row in batch["rows"]) for k in range(4)},
        ac_replay_status_counts={status: sum(fit.get("ac_audit_status")==status for fit in fits)
                                 for status in ("verified", "unavailable", "mismatch", "not_requested")},
        numerical_bound_check_count=sum(validate_numeric_bounds(row) for row in batch["rows"]),
        selection_failure_counts={m: sum(any(record.get("selection_status")=="selection_failure_fallback"
            for record in row["records"] if record["method"]==m) for row in batch["rows"]) for m in batch["config"]["methods"]},
        fit_failure_counts={m: sum(any(record.get("status")!="complete"
            for record in row["records"] if record["method"]==m) for row in batch["rows"]) for m in batch["config"]["methods"]},
        physical_failure_counts={m: sum(any(not record.get("operational_voltage_ok", False)
            for record in row["records"] if record["method"]==m) for row in batch["rows"]) for m in batch["config"]["methods"]},
        proxy_bounds_are_nonlinear_AC_bounds=False, local_fits_are_global_MLE=False)
    _dump(out/"pilot_audit.json", summary)
    if mismatch:
        raise ValueError("saved AC objective replay mismatch; inspect fits.csv")
    return summary


def run_audit(calibration, evaluation, out, alpha=.1, *, allow_stress=False, replay=True):
    cal, ev = load_batch(calibration), load_batch(evaluation)
    stressed = _bind_batches(cal, ev, allow_stress=allow_stress)
    methods = tuple(cal["config"]["methods"])
    if not cal["rows"] or not ev["rows"]:
        raise ValueError("calibration and evaluation require independent histories")
    for batch in (cal, ev):
        if batch["requested_history_count"] != batch["completed_history_count"]:
            raise ValueError("incomplete predeclared batch; resume it before inferential calibration/evaluation")
    calibrated = calibrate_policy_gates([_score(row, methods) for row in cal["rows"]],
        [str(row["history_id"]) for row in cal["rows"]],
        policy_id=cal["config"]["policy_id"], population_id=cal["config"]["population_id"],
        alpha_total=alpha, relative_fraction=.5)
    evaluated = [_events_for_history(row, methods, calibrated, ev["config"]) for row in ev["rows"]]
    summary = dict(calibration=calibrated.summary(), evaluation_config=ev["config"],
        collector_sha256=_sha(__file__),
        calibration_manifest=cal["manifest"], evaluation_manifest=ev["manifest"],
        evaluation_history_count=len(ev["rows"]),
        requested_calibration_history_count=cal["requested_history_count"],
        requested_evaluation_history_count=ev["requested_history_count"],
        out_of_population_stress=stressed, marginal_rank_gate_claim=not stressed,
        conditional_history_or_uniform_parameter_guarantee=False,
        gates_influence_shadow_policy=False, topology_correct_probability_reported=False,
        **_aggregate(ev["rows"], evaluated, methods))
    # Descriptive strata remain separate from the marginal calibration claim.
    strata = {}
    for topology in range(4):
        if topology==0:
            labels = [("star", [i for i, row in enumerate(ev["rows"]) if row.get("truth_topology")==0])]
        else:
            labels = [(f"cherry_{topology}_{case}", [i for i, row in enumerate(ev["rows"])
                if row.get("truth_topology")==topology and row.get("case")==case]) for case in ("weak", "strong")]
        for label, indices in labels:
            strata[label] = _aggregate([ev["rows"][i] for i in indices],
                                      [evaluated[i] for i in indices], methods)
    summary["descriptive_strata"] = strata
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fits, actions, bounds = _flat_evidence(ev, replay=replay)
    _csv(out/"fits.csv", fits); _csv(out/"actions.csv", actions); _csv(out/"numerical_bounds.csv", bounds)
    _dump(out/"history_gates.json", [dict(history_id=row["history_id"], methods=item)
                                    for row, item in zip(ev["rows"], evaluated)])
    _dump(out/"audit_summary.json", summary)
    if any(fit.get("ac_audit_status")=="mismatch" for fit in fits):
        raise ValueError("saved AC objective replay mismatch; inspect fits.csv")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=.1)
    parser.add_argument("--allow-stress", action="store_true")
    parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args()
    if args.pilot:
        result = pilot_summary(args.pilot, args.out, replay=not args.no_replay)
    elif args.calibration and args.evaluation:
        result = run_audit(args.calibration, args.evaluation, args.out, args.alpha,
                           allow_stress=args.allow_stress, replay=not args.no_replay)
    else:
        parser.error("provide --pilot or both --calibration and --evaluation")
    print(json.dumps(dict(history_count=result.get("history_count", result.get("evaluation_history_count")),
                         report=str(args.out)), ensure_ascii=False))


if __name__=="__main__":
    main()
