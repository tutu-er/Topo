"""Read-only, independent-history diagnostics for a frozen EV policy.

No optimization, action selection, parameter fitting, or threshold fitting is
performed here. Saved full-AC losses are local fit values, not certified MLEs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False)
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def load_batch(folder):
    """Reject partial evaluation and method/axis loss before describing it."""
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    for key in ("count", "methods", "policy_id", "population_id", "source_sha256"):
        if key not in config:
            raise ValueError(f"missing frozen batch context: {key}")
    if not config["source_sha256"]:
        raise ValueError("frozen source hash map is empty")
    if config.get("policy", {}).get("gates_influence_shadow_policy", False):
        raise ValueError("analysis requires the frozen shadow policy")
    methods = config["methods"]
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("method list must be nonempty and unique")
    files = sorted(folder.glob("trial_[0-9]*.json"))
    if len(files) != config["count"]:
        raise ValueError("incomplete predeclared batch; this module does not analyze a running batch")
    rows = [json.loads(file.read_text(encoding="utf-8")) for file in files]
    ids = [str(row["history_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate independent history IDs")
    for row in rows:
        if not row.get("independent_history_unit"):
            raise ValueError("independent history unit was not declared")
        if row.get("selection_uses_truth") or row.get("selection_uses_future_noise"):
            raise ValueError("selector/evaluator separation failed")
        names = row["candidate_names"]
        if not names or len(set(names)) != len(names):
            raise ValueError("invalid candidate names")
        records = row["records"]
        if set(record["method"] for record in records) != set(methods):
            raise ValueError("missing frozen method")
        axes = {}
        for record in records:
            if len(record["losses"]) != len(names) or len(record["success"]) != len(names):
                raise ValueError("loss/success candidate axis mismatch")
            key = (record["method"], record.get("round", 0))
            if key in axes:
                raise ValueError("duplicate method/round record")
            axes[key] = True
        rounds = [sorted(record.get("round", 0) for record in records
                         if record["method"] == method) for method in methods]
        if any(trace != rounds[0] for trace in rounds):
            raise ValueError("methods did not preserve the same round trace")
    return dict(folder=folder, config=config, rows=rows,
                manifest={"config_sha256": _sha(folder / "config.json"),
                          "trial_sha256": {file.name: _sha(file) for file in files}})


def load_collector(folder, batch):
    """Use an independently calibrated collector; never estimate gates here."""
    folder = Path(folder)
    summary = json.loads((folder / "audit_summary.json").read_text(encoding="utf-8"))
    gates = json.loads((folder / "history_gates.json").read_text(encoding="utf-8"))
    for key in ("policy_id", "population_id", "source_sha256", "methods"):
        if summary["evaluation_config"][key] != batch["config"][key]:
            raise ValueError(f"collector evaluation context mismatch: {key}")
    for key in ("config_sha256", "trial_sha256"):
        if summary["evaluation_manifest"][key] != batch["manifest"][key]:
            raise ValueError("collector was computed from different saved trials")
    ids = [str(row["history_id"]) for row in gates]
    if len(set(ids)) != len(ids) or set(ids) != set(str(row["history_id"]) for row in batch["rows"]):
        raise ValueError("collector history IDs differ")
    return {str(row["history_id"]): row["methods"] for row in gates}, {
        "audit_summary_sha256": _sha(folder / "audit_summary.json"),
        "history_gates_sha256": _sha(folder / "history_gates.json"),
        "calibration": summary["calibration"],
        "thresholds_fitted_by_this_module": False}


def wrong_margin(record, true_index, names):
    """Signed min-wrong minus truth loss, unavailable for any failed fit."""
    if true_index is None:
        return dict(wrong_margin=None, margin_missing_reason="truth_omitted")
    if not isinstance(true_index, int) or not 0 <= true_index < len(names):
        raise ValueError("invalid true_index")
    values = [_number(value) for value in record["losses"]]
    if len(values) < 2:
        return dict(wrong_margin=None, margin_missing_reason="no_competing_candidate")
    if not all(record["success"]) or any(value is None for value in values):
        return dict(wrong_margin=None, margin_missing_reason="necessary_fit_failed")
    wrong = [index for index in range(len(names)) if index != true_index]
    best = min(wrong, key=lambda index: values[index])
    margin = values[best] - values[true_index]
    return dict(wrong_margin=margin, margin_missing_reason=None,
                closest_wrong_candidate=names[best], true_fitted_loss=values[true_index],
                closest_wrong_fitted_loss=values[best],
                signed_half_loglik_margin=margin / 2,
                truth_strictly_best=margin > 0, true_gap_reconstructed=max(0.0, -margin))


def _interval(lo, hi):
    lo, hi = _number(lo), _number(hi)
    if lo is None or hi is None:
        return None
    if lo > hi + 1e-7 * max(1., abs(lo), abs(hi)):
        raise ValueError("invalid proxy interval")
    return dict(lower=lo, upper=hi, midpoint=(lo + hi) / 2,
                raw_numeric_gap=hi - lo, direction_conflict=lo > hi,
                ordered_bound_claim=lo <= hi, selected_QP_tolerance=1e-7)


def interval_gain(value, reference):
    if value is None or reference is None:
        return None
    if value.get("direction_conflict", False) or reference.get("direction_conflict", False):
        return dict(lower=None, upper=None, midpoint=value["midpoint"] - reference["midpoint"],
                    numerical_direction_conflict=True, ordered_bound_claim=False)
    return dict(lower=value["lower"] - reference["upper"],
                upper=value["upper"] - reference["lower"],
                midpoint=value["midpoint"] - reference["midpoint"])


def _action_kw(action, base_kva):
    if action is None or "p" not in action:
        return None
    value = np.asarray(action["p"], dtype=float) * base_kva
    return value if value.ndim == 2 and np.isfinite(value).all() else None


def _same_action(left, right):
    if left is None or right is None:
        return False
    right = np.asarray(right, dtype=float)
    return left.shape == right.shape and np.isfinite(right).all() and np.allclose(
        left, right, rtol=0, atol=2e-6)


def _matching_proxy_record(search, action_kw):
    matches = [record for record in search.get("records", [])
               if _same_action(action_kw, record.get("u_kw"))]
    if not matches:
        return None
    # Duplicate QP solves may have different numerical dual gaps. Their
    # intersections are valid, but this reader preserves one actual record.
    return min(matches, key=lambda record: float(record["q_upper"]) - float(record["q_lower"]))


def _selection_at(row, method, round_id):
    entry = row.get("selection", {}).get("methods", {}).get(method, {})
    if round_id and "rounds" in entry:
        leaf = entry["rounds"].get(str(round_id), entry["rounds"].get(round_id, {}))
        # Rolling batch stores the full controller output separately. It does
        # not make its planned eight-slot score a score of the executed prefix.
        full = row.get("selections", [])
        extra = full[round_id - 1].get("proxy_search", {}) if round_id <= len(full) else {}
        return {**extra, **leaf}
    return entry


def proxy_diagnostics(row, record):
    """Extract saved selected-action values, never an optimal-design UB."""
    method = record["method"]
    round_id = record.get("round", 0)
    selection = _selection_at(row, method, round_id)
    history = row.get("selection", {}).get("history", {})
    base_kva = float(history.get("config", {}).get("base_kva", 20.0))
    action_kw = _action_kw(record.get("action"), base_kva)
    out = dict(refit_proxy=None, own_proxy=None, proxy_origin=None,
               proxy_missing_reason="action not saved under the shared refit proxy",
               global_design_lower=_number(selection.get("proxy_lower")),
               global_design_upper=_number(selection.get("proxy_upper")),
               global_design_upper_is_selected_value=False,
               own_proxy_scope="fixed_MLE" if method == "fixed_mle_milp" else "open_competitor_refit")
    lo, hi = out["global_design_lower"], out["global_design_upper"]
    out["global_design_raw_numeric_gap"] = hi - lo if lo is not None and hi is not None else None
    out["global_design_direction_conflict"] = bool(lo is not None and hi is not None and lo > hi)
    out["global_design_conflict_within_frozen_tolerance"] = bool(out["global_design_direction_conflict"]
        and lo - hi <= 1e-6 * max(1., abs(lo), abs(hi)))
    out["global_design_bounds_strictly_ordered"] = bool(lo is not None and hi is not None and lo <= hi)
    if round_id:
        out["proxy_missing_reason"] = "rolling full-plan score is not the executed-prefix score"
        return out
    refit = row.get("selection", {}).get("methods", {}).get("refit_milp", {})
    matching = _matching_proxy_record(refit, action_kw)
    if matching is not None:
        out.update(refit_proxy=_interval(matching.get("q_lower"), matching.get("q_upper")),
                   proxy_origin="saved_refit_QP_at_same_action", proxy_missing_reason=None)
    library = row.get("selection", {}).get("methods", {}).get("legacy_library_proxy", {})
    if out["refit_proxy"] is None and library.get("criterion") == "same frozen soft-history proxy as refit MILP":
        library_match = next((entry for entry in library.get("actions", [])
            if entry.get("action") == record.get("action", {}).get("name") and entry.get("success")), None)
        if library_match is not None:
            out.update(refit_proxy=_interval(library_match.get("q_lower"), library_match.get("q_upper")),
                       proxy_origin="saved_library_same_frozen_refit_QP", proxy_missing_reason=None)
    matching_own = _matching_proxy_record(selection, action_kw)
    if matching_own is not None:
        out["own_proxy"] = _interval(matching_own.get("q_lower"), matching_own.get("q_upper"))
    rollback = selection.get("actual_rollback", [])
    if rollback and _same_action(action_kw, selection.get("selected_u_kw")):
        valid = [entry for entry in rollback
                 if _interval(entry.get("proxy_q_lower"), entry.get("proxy_q_upper")) is not None]
        if valid:
            active = min(valid, key=lambda entry: entry["proxy_q_upper"])
            own = _interval(min(entry["proxy_q_lower"] for entry in valid),
                            min(entry["proxy_q_upper"] for entry in valid))
            expected = len(selection.get("pair_hashes", []))
            complete_rollback = bool(len(valid) == len(rollback) and
                (len(valid) == expected if expected else matching_own is not None))
            out["rollback_pair_count"] = len(rollback)
            out["rollback_proxy_bounds_complete"] = complete_rollback
            out["proxy_active_pair_may_be_partial_rollback"] = not complete_rollback
            if complete_rollback:
                out["own_proxy"] = own
            if method == "refit_milp" and complete_rollback:
                out.update(refit_proxy=own, proxy_origin="saved_refit_QP_rollback_at_same_action",
                           proxy_missing_reason=None)
            history_costs = active.get("actual_history_costs", [])
            history_cost = sum(history_costs) if history_costs and all(
                _number(value) is not None for value in history_costs) else None
            future = _number(active.get("actual_future_distance_squared"))
            total = _number(active.get("actual_soft_cost"))
            pair_members = [re.sub(r"_centre_\d+$", "", piece)
                            for piece in active.get("pair_name", "").split("__")[:2]]
            true_name = row["candidate_names"][row["true_index"]] if row.get("true_index") is not None else None
            taylor_errors = [_number(entry.get("joint_taylor_error_norm")) for entry in rollback]
            out.update(proxy_active_pair=active.get("pair_name"),
                proxy_active_pair_contains_truth=true_name in pair_members if true_name else None,
                active_actual_history_cost=history_cost, active_actual_future_witness_cost=future,
                active_actual_soft_witness_cost=total,
                active_actual_future_share_of_witness=future / total if future is not None and total and total > 0 else None,
                active_joint_taylor_error_norm=_number(active.get("joint_taylor_error_norm")),
                worst_joint_taylor_error_norm=max(taylor_errors) if taylor_errors and all(
                    value is not None for value in taylor_errors) else None,
                rollback_physical_failure_count=sum(not entry.get("physically_feasible", False) for entry in rollback),
                actual_rollback_cost_is_nonlinear_infimum_upper_only=True)
    return out


def extract_history(row, methods, gates=None):
    by_method = {method: sorted((record for record in row["records"] if record["method"] == method),
                 key=lambda record: record.get("round", 0)) for method in methods}
    result = []
    stratum = "star" if row.get("truth_topology") == 0 else (
        f"{row.get('case', 'unknown')}_cherry" if row.get("truth_topology") in (1, 2, 3) else "unknown")
    for method in methods:
        trace = by_method[method]
        final = trace[-1]
        all_fit_success = all(all(record["success"]) and all(_number(value) is not None
            for value in record["losses"]) for record in trace)
        physical_failure = any(not record.get("operational_voltage_ok", False) for record in trace)
        selection_failure = any(record.get("selection_status") == "selection_failure_fallback" for record in trace)
        margin = wrong_margin(final, row.get("true_index"), row["candidate_names"])
        entry = dict(history_id=str(row["history_id"]), method=method, stratum=stratum,
                     truth_topology=row.get("truth_topology"), case=row.get("case"),
                     round_count=len(trace), truth_omission=row.get("true_index") is None,
                     fit_failure=not all_fit_success, selection_failure=selection_failure,
                     physical_failure=physical_failure, **margin, **proxy_diagnostics(row, final))
        entry.update(action_name=final.get("action", {}).get("name"),
                     action_grid_energy_kwh=_number(final.get("action", {}).get("energy_kwh")),
                     action_max_kw=_number(final.get("action", {}).get("max_kw")))
        entry["algorithm_margin_usable"] = bool(entry["wrong_margin"] is not None and all_fit_success
            and not physical_failure and not selection_failure)
        entry["strict_truth_winner_all_history_event"] = bool(entry["algorithm_margin_usable"]
            and entry["wrong_margin"] > 0)
        calls = [record.get("ac_calls") for record in trace]
        seconds = [record.get("seconds") for record in trace]
        entry["joint_fit_ac_calls"] = sum(calls) if all(_number(value) is not None for value in calls) else None
        entry["joint_fit_seconds"] = sum(seconds) if all(_number(value) is not None for value in seconds) else None
        selection = _selection_at(row, method, final.get("round", 0))
        entry["selected_route_travel_energy_kwh"] = _number(selection.get("operational", {}).get("travel_energy_kwh"))
        for output, key in (("selection_ac_calls", "selection_ac_predict_calls"),
                            ("selection_seconds", "total_selection_seconds"),
                            ("proxy_search_seconds", "seconds"), ("qp_count", "qp_count")):
            entry[output] = _number(selection.get(key))
        solvers = selection.get("solver_records", [])
        entry.update(milp_solve_count=len(solvers) if solvers else None,
                     milp_time_limit_count=sum(solver.get("status") == 1 for solver in solvers) if solvers else None,
                     upper_milp_missing_bound_count=sum(solver.get("kind") == "secant_upper"
                        and _number(solver.get("bound")) is None for solver in solvers) if solvers else None,
                     maximum_milp_variables=max((solver.get("variables", 0) for solver in solvers), default=None),
                     proxy_gap_met=selection.get("proxy_gap_met"), proxy_stop_reason=selection.get("proxy_stop_reason"))
        if gates is not None:
            if method not in gates:
                raise ValueError("collector omitted a frozen method")
            for key in ("unique_correct_retention", "algorithm_unique_correct_retention", "false_rejection",
                        "any_wrong_candidate_eliminated", "wrong_elimination_fraction", "unresolved"):
                entry[f"gate_{key}"] = gates[method].get(key)
        result.append(entry)
    lookup = {entry["method"]: entry for entry in result}
    for entry in result:
        for control in ("none", "constant_fixed"):
            other = lookup.get(control)
            if other is None:
                continue
            value, reference = entry["wrong_margin"], other["wrong_margin"]
            entry[f"margin_gain_vs_{control}"] = value - reference if value is not None and reference is not None else None
            entry[f"refit_proxy_gain_vs_{control}"] = interval_gain(entry["refit_proxy"], other["refit_proxy"])
            gate_value, gate_reference = entry.get("gate_wrong_elimination_fraction"), other.get("gate_wrong_elimination_fraction")
            if gates is not None:
                entry[f"gate_wrong_elimination_gain_vs_{control}"] = gate_value - gate_reference if (
                    gate_value is not None and gate_reference is not None) else None
    return result


def _array(values):
    return np.array([np.nan if value is None else value for value in values], dtype=float)


def _spearman(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return None
    a, b = rankdata(x[valid]), rankdata(y[valid])
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _percentile(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return None if not values else np.quantile(values, [.025, .975]).tolist()


def measurement_summary(values, *, draws=1000, seed=0):
    array = _array(values)
    valid = np.isfinite(array)
    result = dict(history_count=len(array), available_count=int(valid.sum()),
                  missing_count=int((~valid).sum()), mean=None, median=None,
                  mean_history_bootstrap_95_interval=None, median_history_bootstrap_95_interval=None,
                  descriptive_complete_subset=True, bootstrap_unit="independent history")
    if not valid.any():
        return result
    result.update(mean=float(array[valid].mean()), median=float(np.median(array[valid])))
    rng = np.random.default_rng(seed)
    means, medians = [], []
    for _ in range(draws):
        sample = array[rng.integers(len(array), size=len(array))]
        sample = sample[np.isfinite(sample)]
        if len(sample):
            means.append(float(sample.mean())); medians.append(float(np.median(sample)))
    result.update(mean_history_bootstrap_95_interval=_percentile(means),
                  median_history_bootstrap_95_interval=_percentile(medians))
    return result


def paired_gain_summary(values, *, draws=1000, seed=0, tie_tolerance=1e-8):
    array = _array(values)
    valid = np.isfinite(array)
    out = measurement_summary(values, draws=draws, seed=seed)
    out.update(win_count=int(np.sum(array[valid] > tie_tolerance)),
               loss_count=int(np.sum(array[valid] < -tie_tolerance)),
               tie_count=int(np.sum(np.abs(array[valid]) <= tie_tolerance)),
               win_fraction_full_history=float(np.sum(array[valid] > tie_tolerance) / len(array)) if len(array) else None,
               tie_tolerance=tie_tolerance)
    return out


def correlation_summary(xvalues, yvalues, *, draws=1000, seed=0):
    x, y = _array(xvalues), _array(yvalues)
    if len(x) != len(y):
        raise ValueError("correlation axes differ")
    valid = np.isfinite(x) & np.isfinite(y)
    value = _spearman(x, y)
    rng = np.random.default_rng(seed)
    sample_values = []
    if value is not None:
        for _ in range(draws):
            indices = rng.integers(len(x), size=len(x))
            sample_values.append(_spearman(x[indices], y[indices]))
    return dict(history_count=len(x), available_pair_count=int(valid.sum()),
                missing_pair_count=int((~valid).sum()), spearman=value,
                history_bootstrap_95_interval=_percentile(sample_values),
                undefined_bootstrap_count=sum(item is None for item in sample_values),
                undefined_reason="constant ranks or fewer than three available histories" if value is None else None,
                bootstrap_unit="independent history", association_is_not_causal_or_predictive_validation=True)


def summarize(entries, methods, *, draws=1000, seed=881001):
    ids = list(dict.fromkeys(entry["history_id"] for entry in entries))
    lookup = {(entry["history_id"], entry["method"]): entry for entry in entries}
    if len(lookup) != len(entries) or any((history, method) not in lookup for history in ids for method in methods):
        raise ValueError("history-method table is incomplete or duplicated")
    result = dict(history_count=len(ids), methods={}, paired_against={}, within_history_rank_associations={})
    for method in methods:
        rows = [lookup[history, method] for history in ids]
        events = ("truth_omission", "fit_failure", "selection_failure", "physical_failure",
                  "strict_truth_winner_all_history_event")
        metric = dict(events={key: dict(event_count=sum(bool(row[key]) for row in rows),
                history_count=len(ids)) for key in events},
            wrong_margin=measurement_summary([row["wrong_margin"] for row in rows], draws=draws, seed=seed),
            complexity={key: measurement_summary([row.get(key) for row in rows], draws=draws, seed=seed)
                for key in ("joint_fit_ac_calls", "joint_fit_seconds", "selection_ac_calls", "selection_seconds",
                            "qp_count", "milp_solve_count", "milp_time_limit_count", "maximum_milp_variables")},
            active_future_witness_share=measurement_summary([row.get("active_actual_future_share_of_witness")
                for row in rows], draws=draws, seed=seed),
            global_design_numeric_direction_conflict_count=sum(row["global_design_direction_conflict"] for row in rows),
            selected_refit_QP_numeric_direction_conflict_count=sum(row["refit_proxy"] is not None
                and row["refit_proxy"].get("direction_conflict", False) for row in rows),
            raw_refit_proxy_vs_signed_margin=correlation_summary([
                row["refit_proxy"]["midpoint"] if row["refit_proxy"] else None for row in rows],
                [row["wrong_margin"] for row in rows], draws=draws, seed=seed))
        for key in ("unique_correct_retention", "algorithm_unique_correct_retention", "false_rejection",
                    "any_wrong_candidate_eliminated", "unresolved"):
            if rows and f"gate_{key}" in rows[0]:
                metric["events"][f"gate_{key}"] = dict(event_count=sum(bool(row[f"gate_{key}"]) for row in rows),
                    history_count=len(ids))
        if rows and "gate_wrong_elimination_fraction" in rows[0]:
            score_values = [row["refit_proxy"]["midpoint"] if row["refit_proxy"] else None for row in rows]
            metric["refit_proxy_vs_calibrated_elimination_fraction"] = correlation_summary(score_values,
                [row["gate_wrong_elimination_fraction"] for row in rows], draws=draws, seed=seed)
            metric["refit_proxy_vs_calibrated_unique_correct_event"] = correlation_summary(score_values,
                [int(row["gate_algorithm_unique_correct_retention"]) for row in rows], draws=draws, seed=seed)
        result["methods"][method] = metric
    for control in ("none", "constant_fixed"):
        if control not in methods:
            continue
        result["paired_against"][control] = {}
        for method in methods:
            if method == control:
                continue
            rows = [lookup[history, method] for history in ids]
            gains = [row.get(f"margin_gain_vs_{control}") for row in rows]
            scores = [row.get(f"refit_proxy_gain_vs_{control}") for row in rows]
            result["paired_against"][control][method] = dict(
                signed_margin_gain=paired_gain_summary(gains, draws=draws, seed=seed),
                refit_proxy_gain_vs_actual_margin_gain=correlation_summary([
                    score["midpoint"] if score else None for score in scores], gains, draws=draws, seed=seed),
                refit_proxy_gain=measurement_summary([score["midpoint"] if score else None for score in scores], draws=draws, seed=seed),
                certified_positive_proxy_gain_count=sum(score is not None and score["lower"] is not None
                    and score["lower"] > 0 for score in scores),
                proxy_gain_numerical_direction_conflict_count=sum(score is not None
                    and score.get("numerical_direction_conflict", False) for score in scores))
            if rows and "gate_wrong_elimination_fraction" in rows[0]:
                result["paired_against"][control][method]["refit_proxy_gain_vs_calibrated_elimination_gain"] = correlation_summary([
                    score["midpoint"] if score else None for score in scores],
                    [row.get(f"gate_wrong_elimination_gain_vs_{control}") for row in rows], draws=draws, seed=seed)
        per_history = []
        for history in ids:
            choices = [lookup[history, method] for method in methods if method != control]
            proxy = [row.get(f"refit_proxy_gain_vs_{control}") for row in choices]
            actual = [row.get(f"margin_gain_vs_{control}") for row in choices]
            rho = _spearman(_array([score["midpoint"] if score else None for score in proxy]), _array(actual))
            per_history.append(rho)
        result["within_history_rank_associations"][control] = dict(
            statistic="mean of within-history Spearman correlations across frozen methods",
            **measurement_summary(per_history, draws=draws, seed=seed),
            method_rows_are_not_independent_samples=True)
    return result


def run_analysis(input, out, *, collector=None, draws=1000, seed=881001):
    batch = load_batch(input)
    if draws < 0:
        raise ValueError("bootstrap draws must be nonnegative")
    gates, collector_context = load_collector(collector, batch) if collector else ({}, None)
    methods = batch["config"]["methods"]
    entries = [entry for row in batch["rows"] for entry in extract_history(row, methods,
               gates.get(str(row["history_id"])))]
    summary = dict(config=batch["config"], input_manifest=batch["manifest"],
        analysis_source_sha256=_sha(__file__), collector_context=collector_context,
        actual_metric="min wrong candidate fitted full-AC loss minus true candidate fitted loss",
        global_MLE_claim=False, topology_correct_probability_reported=False,
        optimization_or_gate_tuning_performed=False, confidence_intervals_are_descriptive_bootstrap=True,
        bootstrap_seed=seed, bootstrap_draws=draws, missing_margins_not_imputed=True,
        common_refit_proxy_required_for_score_gains=True,
        fixed_MLE_own_criterion_kept_separate=True,
        nonlinear_rollback_witness_is_infimum_upper_only=True,
        **summarize(entries, methods, draws=draws, seed=seed))
    summary["strata"] = {label: summarize([entry for entry in entries if entry["stratum"] == label],
        methods, draws=draws, seed=seed) for label in dict.fromkeys(entry["stratum"] for entry in entries)}
    history_costs = []
    for row in batch["rows"]:
        selector = _number(row.get("selection", {}).get("selector_ac_predict_calls"))
        joint = [_number(record.get("ac_calls")) for record in row["records"]]
        history_costs.append(dict(history_id=str(row["history_id"]), history_seconds=_number(row.get("seconds")),
            selector_ac_calls=selector, joint_fit_ac_calls=sum(joint) if all(value is not None for value in joint) else None,
            instrumented_selector_and_joint_calls=selector + sum(joint) if selector is not None and all(value is not None for value in joint) else None,
            counter_scope="selector plus joint fitting, including failed calls; excludes uninstrumented truth simulation"))
    summary["complexity_accounting"] = dict(history_count=len(history_costs),
        instrumented_calls=measurement_summary([row["instrumented_selector_and_joint_calls"] for row in history_costs], draws=draws, seed=seed),
        history_seconds=measurement_summary([row["history_seconds"] for row in history_costs], draws=draws, seed=seed),
        global_selector_count_must_not_be_added_again_to_per_method_selection_counts=True,
        returned_fit_ac_calls_not_used_as_total=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _csv(out / "history_method_metrics.csv", entries)
    _csv(out / "history_complexity.csv", history_costs)
    _dump(out / "empirical_design_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--collector", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=881001)
    args = parser.parse_args()
    result = run_analysis(args.input, args.out, collector=args.collector,
                          draws=args.bootstrap_draws, seed=args.seed)
    print(json.dumps(dict(histories=result["history_count"], methods=list(result["methods"]),
                         optimization_or_gate_tuning_performed=False)))


if __name__ == "__main__":
    main()
