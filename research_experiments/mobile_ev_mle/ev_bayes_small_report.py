"""Read-only offline summary of frozen ev_bayes_small history records.

No AC solver, case generator or policy is imported.  Future truth is read only
from saved evaluation-only fields for offline diagnostics.  Every failure and
missing expected history remains visible; 32 conditional noise repeats never
become independent histories.  Outputs: summary.json, results.md, paired.csv.
"""
from __future__ import annotations

import argparse
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
from collections import Counter
from itertools import combinations
import numpy as np
from scipy.stats import t as student_t


DOMAINS = ("fixed_charge", "fixed_bidir", "mobile_charge", "mobile_bidir")
PARENTS = dict(fixed_charge=(), fixed_bidir=("fixed_charge",),
               mobile_charge=("fixed_charge",), mobile_bidir=("fixed_bidir", "mobile_charge"))
METHODS = ("none", "constant", "highlow") + tuple(f + "__" + d for f in ("square", "bayes") for d in DOMAINS)
DOMAIN_LABEL = dict(zip(DOMAINS, ("FC", "FD", "MC", "MD")))
ACTUAL_METRICS = ("error", "true_rejected", "unresolved", "unique_correct", "wrong_unique", "brier",
                  "negative_log_true_probability", "confidence", "true_probability", "abstention_loss")


def _read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _clean(value):
    if isinstance(value, np.ndarray): return _clean(value.tolist())
    if isinstance(value, np.generic): return _clean(value.item())
    if isinstance(value, dict): return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def _number(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError): return None


def estimate(values):
    """Descriptive t interval whose units are independent historical records."""
    values = np.asarray([v for v in values if _number(v) is not None], float)
    n = len(values)
    out = dict(n_histories=n, mean=float(values.mean()) if n else None,
               minimum=float(values.min()) if n else None, maximum=float(values.max()) if n else None,
               sd=None, se=None, t95=None, interval_scope="descriptive between-history t interval")
    if n > 1:
        sd = float(values.std(ddof=1)); se = sd / np.sqrt(n)
        half = float(student_t.ppf(.975, n - 1) * se)
        out.update(sd=sd, se=se, t95=[out["mean"] - half, out["mean"] + half])
    return out


def _actual(record, method):
    method_row = record.get("methods", {}).get(method, {})
    return record.get("actions", {}).get(method_row.get("action_key"), {}).get("actual", {})


def _sample_vector(actual, metric):
    samples = actual.get("samples", [])
    if not samples: return None
    if metric == "abstention_loss":
        if any("unresolved" not in s or "wrong_unique" not in s for s in samples): return None
        return np.asarray([.1 * s["unresolved"] + s["wrong_unique"] for s in samples], float)
    key = "correct" if metric == "error" else metric
    if any(key not in sample for sample in samples): return None
    values = np.asarray([sample[key] for sample in samples], float)
    if not np.isfinite(values).all(): return None
    return 1 - values if metric == "error" else values


def _history_actual(record, method, metric, regimes):
    actual = _actual(record, method)
    values = {}
    for regime in regimes:
        row = actual.get(regime, {})
        samples = _sample_vector(row, metric)
        value = float(samples.mean()) if samples is not None else _number(
            1 - row["correct"] if metric == "error" and "correct" in row else row.get(metric))
        if value is not None: values[regime] = value
    # Main paired summaries require the same complete set of declared regimes.
    return values, float(np.mean(list(values.values()))) if len(values) == len(regimes) else None


def _paired_noise_se(record, left, right, regimes):
    """Conditional noise SE, averaging paired regimes before taking variance."""
    a, b = _actual(record, left), _actual(record, right)
    differences = []
    for regime in regimes:
        x, y = a.get(regime, {}), b.get(regime, {})
        if x.get("noise_seed") != y.get("noise_seed") or "noise_seed" not in x: return None
        vx, vy = _sample_vector(x, "error"), _sample_vector(y, "error")
        if vx is None or vy is None or vx.shape != vy.shape: return None
        differences.append(vx - vy)
    if not differences or len({len(v) for v in differences}) != 1: return None
    # Both regimes are replayed with the same noise seed in this study.
    if len({a[r].get("noise_seed") for r in regimes}) != 1: return None
    delta = np.mean(differences, axis=0)
    return float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else None


def _contrasts():
    result = [("bayes__" + d, "square__" + d) for d in DOMAINS]
    result += [("bayes__" + d, "bayes__" + p) for d in DOMAINS for p in PARENTS[d]]
    result += [("highlow", "constant"), ("square__fixed_charge", "constant"),
               ("bayes__mobile_bidir", "constant")]
    return result


def _stored_paired(record, left, right):
    if left.startswith("bayes__") and right.startswith("square__") and left[7:] == right[8:]:
        key = "bayes_minus_square__" + left[7:]
    elif left.startswith("bayes__") and right.startswith("bayes__"):
        key = left + "_minus_" + right[7:]
    else: return {}
    return record.get("independent_paired", {}).get(key, {})


def paired_rows(records, phase, regimes):
    rows = []
    for record in records:
        methods = record.get("methods", {}); spec = record.get("spec", {})
        for left, right in _contrasts():
            if left not in methods or right not in methods: continue
            lm, rm = methods[left], methods[right]
            row = dict(phase=phase, history_index=spec.get("index"), seed=spec.get("seed"),
                true_name=record.get("true_name"), method=left, baseline=right,
                same_action=lm.get("action_key") == rm.get("action_key"))
            for score_name, prefix in (("planning_score", "planning"), ("independent_score", "predicted")):
                x, y = lm.get(score_name, {}), rm.get(score_name, {})
                vx, vy = _number(x.get("risk")), _number(y.get("risk"))
                row[prefix + "_risk_delta"] = None if vx is None or vy is None else vx - vy
            pair = _stored_paired(record, left, right)
            row["predicted_paired_mc_se"] = _number(pair.get("risk_se"))
            sa = _number(lm.get("independent_score", {}).get("risk_se"))
            sb = _number(rm.get("independent_score", {}).get("risk_se"))
            row["predicted_mc_se_upper_bound"] = None if sa is None or sb is None else sa + sb
            for metric in ACTUAL_METRICS:
                av, am = _history_actual(record, left, metric, regimes)
                bv, bm = _history_actual(record, right, metric, regimes)
                row["actual_" + metric + "_delta"] = None if am is None or bm is None else am - bm
                for regime in regimes:
                    row[regime + "_" + metric + "_delta"] = (av[regime] - bv[regime]
                        if regime in av and regime in bv else None)
            row["actual_conditional_paired_noise_se"] = _paired_noise_se(record, left, right, regimes)
            rows.append(row)
    return rows


def method_summary(records, method, expected, regimes, n_noise):
    planning, predicted, integration_se, actual_values = [], [], [], {m: [] for m in ACTUAL_METRICS}
    predicted_scores = {m: [] for m in ("risk", "eig", "brier", "brier_risk", "abstention_risk")}
    by_regime = {r: [] for r in regimes}; counts = Counter(); flag_counts = Counter()
    seen_histories = complete_histories = 0
    for record in records:
        row = record.get("methods", {}).get(method)
        if row is None: continue
        seen_histories += 1
        planning.append(row.get("planning_score", {}).get("risk"))
        predicted.append(row.get("independent_score", {}).get("risk"))
        integration_se.append(row.get("independent_score", {}).get("risk_se"))
        for metric in predicted_scores: predicted_scores[metric].append(row.get("independent_score", {}).get(metric))
        for flag in ("planning_success", "planning_improvement", "fallback_to_initial", "executable_plan_available"):
            if flag in row:
                flag_counts[flag + "_reported"] += 1
                flag_counts[flag + "_true"] += bool(row[flag])
        operation = row.get("operation")
        if operation is not None:
            flag_counts["operation_checked"] += 1
            flag_counts["operation_failed"] += not bool(operation.get("feasible"))
        actual = _actual(record, method)
        complete_histories += all(regime in actual for regime in regimes)
        for metric in ACTUAL_METRICS:
            reg, pooled = _history_actual(record, method, metric, regimes)
            actual_values[metric].append(pooled)
            if metric == "error":
                for regime, value in reg.items(): by_regime[regime].append(value)
        for regime in regimes:
            if regime not in actual: continue
            a = actual[regime]; counts["observed_conditions"] += 1
            samples = a.get("samples", []); counts["observed_noise_draws"] += len(samples)
            if "physical_voltage_feasible" in a:
                counts["physical_voltage_checked"] += 1
                counts["physical_voltage_failed"] += not bool(a["physical_voltage_feasible"])
            for metric in ("true_rejected", "unresolved", "unique_correct", "wrong_unique"):
                vector = _sample_vector(a, metric)
                if vector is not None:
                    counts[metric + "_numerator"] += int(np.rint(vector.sum()))
                    counts[metric + "_denominator"] += len(vector)
    counts.update(expected_conditions=expected * len(regimes), expected_noise_draws=expected * len(regimes) * n_noise)
    counts["missing_conditions"] = counts["expected_conditions"] - counts["observed_conditions"]
    return dict(expected_histories=expected, available_method_histories=seen_histories,
        complete_actual_histories=complete_histories, missing_method_histories=expected - seen_histories,
        planning_risk=estimate(planning), predicted_risk=estimate(predicted),
        predictive_scores={key: estimate(value) for key, value in predicted_scores.items()},
        independent_integral_se=estimate(integration_se), actual={k: estimate(v) for k, v in actual_values.items()},
        error_by_regime={r: estimate(v) for r, v in by_regime.items()}, counts=dict(counts),
        planning_flags=dict(flag_counts), same_task=method != "none")


def inheritance_and_solver(records):
    edges, proposals = [], []
    for record in records:
        index = record.get("spec", {}).get("index")
        for child in DOMAINS:
            cm = record.get("methods", {}).get("bayes__" + child, {})
            for parent in PARENTS[child]:
                pm = record.get("methods", {}).get("bayes__" + parent, {})
                c, p = _number(cm.get("planning_score", {}).get("risk")), _number(pm.get("planning_score", {}).get("risk"))
                ci, pi = _number(cm.get("independent_score", {}).get("risk")), _number(pm.get("independent_score", {}).get("risk"))
                edges.append(dict(history_index=index, child=child, parent=parent,
                    planning_risk_delta=None if c is None or p is None else c - p,
                    planning_violation=None if c is None or p is None else bool(c > p + 1e-10),
                    independent_risk_delta=None if ci is None or pi is None else ci - pi))
            for event in record.get("bayes_logs", {}).get(child, {}).get("events", []):
                m = event.get("milp", {})
                proposals.append(dict(history_index=index, domain=child, iteration=event.get("iteration"),
                    success=bool(m.get("success")), accepted=bool(event.get("accepted")),
                    status=m.get("status"), failure=m.get("failure"), stop_reason=m.get("stop_reason"),
                    full_AC_failure=event.get("full_AC_failure"),
                    solver_incumbent=m.get("solver_incumbent"), solver_best_bound=m.get("solver_best_bound"),
                    solver_relative_gap=m.get("solver_relative_gap"), proxy_absolute_gap=m.get("proxy_absolute_gap"),
                    proxy_uniform_error=m.get("proxy_uniform_error"), proxy_actual_error=m.get("proxy_actual_error"),
                    proxy_value=m.get("proxy_value"), frozen_C=m.get("frozen_C"),
                    frozen_C_anchor=m.get("frozen_C_anchor"),
                    full_gap_met=(_number(m.get("solver_relative_gap")) is not None and m["solver_relative_gap"] <= 1.0001e-5),
                    parent_risk=event.get("parent_risk"), proposed_risk=event.get("proposed_risk"),
                    risk_gain=event.get("risk_gain"), seconds=m.get("seconds")))
    known = [e for e in edges if e["planning_violation"] is not None]
    error_fields = {}
    for key in ("proxy_uniform_error", "proxy_actual_error", "proxy_absolute_gap", "solver_relative_gap"):
        values = [float(p[key]) for p in proposals if _number(p[key]) is not None]
        error_fields[key] = dict(count=len(values), minimum=min(values) if values else None,
            maximum=max(values) if values else None, mean=float(np.mean(values)) if values else None)
    return dict(edges=edges, known_edges=len(known), planning_violations=sum(e["planning_violation"] for e in known),
        unknown_edges=len(edges) - len(known), proposals=proposals, proposal_count=len(proposals),
        successful_proposals=sum(p["success"] for p in proposals), accepted_proposals=sum(p["accepted"] for p in proposals),
        failed_proposals=sum(not p["success"] for p in proposals),
        full_AC_failures=sum(p["full_AC_failure"] is not None for p in proposals),
        full_gap_met=sum(p["full_gap_met"] for p in proposals),
        proxy_error_summary=error_fields,
        stop_reasons=dict(Counter(str(p["stop_reason"] or p["failure"]) for p in proposals)))


def ranking(records, regimes):
    rows = []
    for record in records:
        seen, actions = set(), []
        for method, value in record.get("methods", {}).items():
            key = value.get("action_key")
            if method == "none" or key in seen or key is None: continue
            seen.add(key)
            score = value.get("independent_score", {})
            _, error = _history_actual(record, method, "error", regimes)
            if _number(score.get("risk")) is not None and error is not None:
                actions.append((method, score, error))
        for a, b in combinations(actions, 2):
            delta = a[1]["risk"] - b[1]["risk"]; actual = a[2] - b[2]
            if abs(delta) <= 1e-12: relation = "predicted_tie"
            elif abs(actual) <= 1e-12: relation = "actual_tie"
            else: relation = "concordant" if delta * actual > 0 else "reversed"
            se = _number(a[1].get("risk_se")); sb = _number(b[1].get("risk_se"))
            rows.append(dict(history_index=record.get("spec", {}).get("index"), a=a[0], b=b[0],
                predicted_risk_delta=delta, actual_error_delta=actual, relation=relation,
                conservative_mc_resolved=None if se is None or sb is None else abs(delta) > 1.96 * (se + sb)))
    return dict(counts=dict(Counter(r["relation"] for r in rows)), rows=rows,
        statistical_unit="correlated action pairs within histories; not independent replications")


def prediction_diagnostics(records, regimes):
    future, evidence, line_errors = [], [], []
    unavailable = Counter()
    for record in records:
        index = record.get("spec", {}).get("index"); true_name = record.get("true_name")
        prediction = record.get("prediction", {}); updates = prediction.get("candidate_updates", {})
        update = updates.get(true_name)
        if update is None:
            unavailable["true_reference_prediction"] += 1
            continue
        try:
            observed = np.asarray(update["evidence_observed_whitened"], float)
            mean = np.asarray(update["evidence_predicted_mean"], float)
            covariance = np.asarray(update["evidence_covariance"], float)
            residual = observed - mean
            V = int(prediction["evidence_count"]); n = (len(observed) // V - 2) // 3
            sizes = (V * n, V * n, V * n, V, V); start = 0; channels = {}
            for name, size in zip(("p", "q", "v", "master_p", "master_q"), sizes):
                sl = slice(start, start + size); start += size
                channels[name] = float(np.mean(residual[sl] ** 2 / np.diag(covariance)[sl]))
            evidence.append(dict(history_index=index, true_reference=true_name,
                mahalanobis_per_dimension=float(residual @ np.linalg.solve(covariance, residual) / len(residual)),
                marginal_standardized_mse=channels))
        except (KeyError, ValueError, np.linalg.LinAlgError): unavailable["evidence_innovation"] += 1
        for regime in regimes:
            truth = record.get("true_future_inputs_evaluation_only", {}).get(regime)
            if truth is None:
                unavailable["future_PQ_source_truth"] += 1
                continue
            try:
                target = np.c_[truth["p"], truth["q"], truth["source_p"], truth["source_q"]]
                mean = np.asarray(update["future_latent_mean"], float)
                covariance = np.asarray(update["future_latent_covariance"], float)
                if target.shape != mean.shape: raise ValueError("latent truth dimensions differ")
                width = target.shape[1]; n = (width - 2) // 2; base = float(prediction["base_kw"])
                error = target - mean; sd = np.sqrt(np.maximum(0., np.diag(covariance))).reshape(target.shape)
                groups = {}
                for name, sl in (("P", slice(0, n)), ("Q", slice(n, 2*n)), ("source_P", slice(-2, -1)), ("source_Q", slice(-1, None))):
                    difference, standard = error[:, sl], sd[:, sl]
                    groups[name] = dict(rmse_physical=float(base * np.sqrt(np.mean(difference ** 2))),
                        mean_sd_physical=float(base * np.mean(standard)),
                        marginal_95_coverage=float(np.mean(abs(difference) <= 1.96 * standard)),
                        units="kW" if name.endswith("P") else "kvar")
                vector = error.ravel()
                future.append(dict(history_index=index, regime=regime, true_reference=true_name,
                    groups=groups, latent_mahalanobis_per_dimension=float(
                        vector @ np.linalg.solve(covariance, vector) / len(vector)),
                    evaluation_only=True, n_independent_histories=1))
            except (KeyError, ValueError, np.linalg.LinAlgError): unavailable["future_prediction_diagnostic"] += 1
        fits = record.get("training_result", {}).get("fits", [])
        if isinstance(fits, dict): fits = list(fits.values())
        fitted = next((f for f in fits if f.get("candidate") == true_name), None)
        if fitted and "true_lines" in record:
            try:
                row = dict(history_index=index)
                for name in ("r", "x"):
                    true = np.asarray(record["true_lines"][name], float); got = np.asarray(fitted[name], float)
                    row["relative_" + name + "_error"] = float(np.linalg.norm(got - true) / np.linalg.norm(true))
                line_errors.append(row)
            except (KeyError, ValueError, ZeroDivisionError): unavailable["line_error"] += 1
    return dict(future=future, evidence=evidence, line_errors=line_errors, unavailable=dict(unavailable),
        scope="offline saved truth only; true-parameter uncertainty is not included in model covariance")


def legacy_summary(records, regimes):
    methods = ("constant", "square__mobile_bidir", "bayes__mobile_bidir")
    rows, output = [], {}
    for method in methods:
        margins = []; correct = completed = failed = 0
        for record in records:
            action = record.get("methods", {}).get(method, {}).get("action_key")
            per_history = []
            for regime in regimes:
                entry = record.get("legacy_joint_audit", {}).get(regime + "__" + str(action))
                if entry is None: continue
                row = dict(history_index=record.get("spec", {}).get("index"), method=method, regime=regime,
                    complete=bool(entry.get("complete")), M=entry.get("M"), selected=entry.get("selected"))
                rows.append(row)
                if row["complete"]:
                    completed += 1; correct += row["selected"] == record.get("true_name")
                    if _number(row["M"]) is not None: per_history.append(row["M"])
                else: failed += 1
            if len(per_history) == len(regimes): margins.append(float(np.mean(per_history)))
        output[method] = dict(complete_conditions=completed, failed_conditions=failed,
            raw_correct_winner_conditions=correct, full_SSE_margin=estimate(margins))
    return dict(methods=output, rows=rows,
        scope="prespecified full AC local joint-refit audit, one noise draw; not Bayesian calibration")


def physical_summary(records):
    counts = Counter(); max_equation = max_balance = 0.
    for record in records:
        for action in record.get("actions", {}).values():
            for physical in action.get("forecast", {}).get("physical", []):
                counts["predictive_candidate_blocks"] += 1
                eq = _number(physical.get("ac_equation_residual")); balance = _number(physical.get("power_balance_residual"))
                if eq is not None: max_equation = max(max_equation, abs(eq))
                if balance is not None: max_balance = max(max_balance, abs(balance))
                valid = (eq is not None and balance is not None and abs(eq) <= 1e-8 and abs(balance) <= 1e-8
                    and _number(physical.get("v_min")) is not None and _number(physical.get("v_max")) is not None
                    and .85 <= physical["v_min"] <= physical["v_max"] <= 1.15)
                counts["predictive_failed_or_unknown"] += not valid
            for actual in action.get("actual", {}).values():
                counts["actual_unique_action_conditions"] += 1
                counts["actual_voltage_failed_or_unknown"] += not bool(actual.get("physical_voltage_feasible"))
                eq = _number(actual.get("ac_equation_residual")); balance = _number(actual.get("power_balance_residual"))
                if eq is not None: max_equation = max(max_equation, abs(eq))
                if balance is not None: max_balance = max(max_balance, abs(balance))
                counts["actual_equation_failed_or_unknown"] += (eq is None or balance is None
                    or abs(eq) > 1e-8 or abs(balance) > 1e-8)
    return dict(counts=dict(counts), max_ac_equation_residual=max_equation,
        max_power_balance_residual=max_balance, scope="saved values, unique action cache; independent AC audit is separate")


def plot_data(records, phase, regimes):
    points, curves = [], []
    for record in records:
        index = record.get("spec", {}).get("index")
        for method in ("constant", "square__mobile_bidir", "bayes__mobile_bidir"):
            row = record.get("methods", {}).get(method, {})
            predicted = _number(row.get("independent_score", {}).get("risk"))
            actual, _ = _history_actual(record, method, "error", regimes)
            if predicted is not None:
                points.append(dict(history_index=index, method=method, predicted_risk=predicted, actual=actual))
        if phase == "evaluation" and index in (0, 4):
            for method in ("square__mobile_bidir", "bayes__mobile_bidir"):
                row = record.get("methods", {}).get(method, {})
                if "power_kw" in row:
                    curves.append(dict(history_index=index, method=method, power_kw=row["power_kw"],
                        selected_arcs=row.get("decoded", {}).get("selected_arcs", []),
                        dt_hours=record.get("operation_config", {}).get("dt_hours", .25)))
    return dict(points=points, predeclared_curves=curves, curve_indices_fixed_before_evaluation=[0, 4])


def summarize_phase(records, phase, expected_specs, config):
    expected = len(expected_specs) if expected_specs else len(records)
    regimes = config.get("regimes", ["regular", "shift"]); noise = int(config.get("actual_noise_repeats", 32))
    status = dict(Counter(r.get("status", "unknown") for r in records))
    rows = paired_rows(records, phase, regimes); contrasts = []
    for left, right in _contrasts():
        selected = [r for r in rows if r["method"] == left and r["baseline"] == right]
        values = [r["actual_error_delta"] for r in selected if r["actual_error_delta"] is not None]
        contrasts.append(dict(method=left, baseline=right, expected_histories=expected,
            predicted_risk_delta=estimate([r["predicted_risk_delta"] for r in selected]),
            actual_error_delta=estimate(values),
            actual_rejection_delta=estimate([r["actual_true_rejected_delta"] for r in selected]),
            error_delta_by_regime={reg: estimate([r.get(reg + "_error_delta") for r in selected]) for reg in regimes},
            wins_ties_losses=[sum(v < -1e-12 for v in values), sum(abs(v) <= 1e-12 for v in values), sum(v > 1e-12 for v in values)]))
    known_indices = {r.get("spec", {}).get("index") for r in records}
    return dict(phase=phase, expected_independent_histories=expected, found_histories=len(records),
        complete_histories=status.get("complete", 0), statuses=status,
        missing_history_indices=[s.get("index") for s in expected_specs if s.get("index") not in known_indices],
        failures=[dict(spec=r.get("spec"), status=r.get("status"), failure=r.get("failure"),
            training_failures=r.get("training_result", {}).get("failures")) for r in records if r.get("status") != "complete"],
        candidate_miss_count=sum(r.get("candidate_miss") is True for r in records),
        candidate_miss_unknown=sum("candidate_miss" not in r for r in records),
        candidate_miss_observed_denominator=sum("candidate_miss" in r for r in records),
        methods={m: method_summary(records, m, expected, regimes, noise) for m in METHODS},
        paired=contrasts, paired_history_rows=rows, inheritance_solver=inheritance_and_solver(records),
        prediction_actual_ranking=ranking(records, regimes), prediction_diagnostics=prediction_diagnostics(records, regimes),
        legacy_joint_audit=legacy_summary(records, regimes), physical_saved_audit=physical_summary(records),
        plot_data=plot_data(records, phase, regimes),
        runtime={**{k: estimate([r.get(k) for r in records]) for k in
                 ("seconds", "square_seconds", "bayes_seconds", "objective_seconds", "objective_calls", "AC_predictive_states")},
                 "bayes_with_square_initialization_seconds": estimate([
                     float(r["square_seconds"]) + float(r["bayes_seconds"]) for r in records
                     if _number(r.get("square_seconds")) is not None and _number(r.get("bayes_seconds")) is not None])},
        effective_provenance=[dict(history_index=r.get("spec", {}).get("index"), **r.get("effective_provenance", {})) for r in records],
        prior_underflow_histories=sum(r.get("prediction", {}).get("prior_display_underflow_count", 0) > 0 for r in records))


def analyze(directory):
    root = Path(directory).resolve()
    if not root.is_dir(): raise ValueError("input must be an existing run directory")
    frozen_path = root / "freeze.json"
    if not frozen_path.exists() and root.name in ("development", "evaluation"):
        frozen_path = root.parent / "freeze.json"
    frozen = _read(frozen_path) if frozen_path.exists() else {}
    config = frozen.get("config", {}); records_by_phase = {}; hashes = {}; mismatches = []
    for phase in ("development", "evaluation"):
        folder = root if root.name == phase else root / phase
        records = []
        for path in sorted(folder.glob("history*.json.gz")) if folder.exists() else []:
            hashes[str(path.relative_to(root))] = sha256(path.read_bytes()).hexdigest()
            try: record = _read(path)
            except Exception as exc:
                record = dict(status="unreadable_record", failure=type(exc).__name__ + ": " + str(exc),
                    spec=dict(phase=phase, index=int(path.name.split("_")[-1].split(".")[0])))
            if frozen.get("policy_id") is not None and record.get("policy_id") != frozen["policy_id"]:
                mismatches.append(dict(path=str(path), observed=record.get("policy_id"), expected=frozen["policy_id"]))
            records.append(record)
        if records or phase in frozen.get("specifications", {}): records_by_phase[phase] = records
    if not records_by_phase:
        raise ValueError("no history records or frozen phase specifications found")
    result = dict(schema="ev_bayes_small_offline_report_v1", input=str(root), config=config,
        policy_id=frozen.get("policy_id"), policy_identity_mismatches=mismatches, input_record_sha256=hashes,
        frozen_source_sha256=frozen.get("source_sha256", {}),
        report_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        interval_scope="95% t intervals are descriptive, history is the unit; paired regimes/noise do not add histories",
        probability_scope="conditional fixed-trained-R/X local Gaussian model, not line-parameter Bayes integration",
        phases={phase: summarize_phase(records, phase, frozen.get("specifications", {}).get(phase, []), config)
                for phase, records in records_by_phase.items()})
    return _clean(result)


def _fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}g}"


def _interval(summary):
    value = _fmt(summary.get("mean"))
    interval = summary.get("t95")
    return value if interval is None else value + " [" + ", ".join(_fmt(x) for x in interval) + "]"


def _label(method):
    if method == "none": return "无EV*"
    if method == "constant": return "恒充"
    if method == "highlow": return "高低充电"
    family, domain = method.split("__", 1)
    return ("Bayes " if family == "bayes" else "平方 ") + DOMAIN_LABEL[domain]


def markdown(summary):
    config = summary["config"]
    lines = ["# 条件贝叶斯EV小规模验证", "",
        "每拓扑固定一组训练R/X；80训练、20证据、6未来。报告保留失败和未决；概率属于局部Gaussian预测模型。",
        "FC/FD/MC/MD为固定只充、固定双向、可移只充、可移双向；移动为选项。无EV不完成电池任务，单列。",
        "实际错误率先对同一历史的未来条件和计量噪声平均，再按独立历史汇总。区间为小样本描述性95% t区间。", ""]
    for phase, data in summary["phases"].items():
        title = "开发" if phase == "development" else "独立评价"
        lines += ["## " + title, "", f"找到{data['found_histories']}/{data['expected_independent_histories']}条历史；"
            f"完成{data['complete_histories']}，状态{json.dumps(data['statuses'], ensure_ascii=False)}。"
            f"候选漏真{data['candidate_miss_count']}/{data['candidate_miss_observed_denominator']}，未知{data['candidate_miss_unknown']}。", ""]
        if not data["found_histories"]:
            lines += ["本阶段尚无结果。", ""]; continue
        lines += ["| 方法 | 独立积分风险 | 实际错误率 | 90%集合排真 | 未决 | 完整历史 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            m = data["methods"][method]; c = m["counts"]
            lines.append("| " + " | ".join((_label(method), _fmt(m["predicted_risk"]["mean"]),
                _fmt(m["actual"]["error"]["mean"]), f"{c.get('true_rejected_numerator', 0)}/{c.get('true_rejected_denominator', 0)}",
                f"{c.get('unresolved_numerator', 0)}/{c.get('unresolved_denominator', 0)}",
                f"{m['complete_actual_histories']}/{m['expected_histories']}")) + " |")
        lines += ["", "排真与未决分母是已观察的配对计量实现；每方法缺失条件数保存在summary.json，"
                  "每历史32次噪声不计作32条历史。90%为模型后验集合质量，实际排真频率单独记录。", "",
                  "| 配对：前者减后者 | 历史n | Δ预测风险 | Δ实际错误率 [区间] | 前者胜/平/负 |",
                  "|---|---:|---:|---:|---:|"]
        for row in data["paired"]:
            lines.append("| " + " | ".join((_label(row["method"]) + " − " + _label(row["baseline"]),
                str(row["actual_error_delta"]["n_histories"]), _fmt(row["predicted_risk_delta"]["mean"]),
                _interval(row["actual_error_delta"]), "/".join(map(str, row["wins_ties_losses"])))) + " |")
        maximum_se = max((m['independent_integral_se']['maximum'] for m in data['methods'].values()
                          if m['independent_integral_se']['maximum'] is not None), default=None)
        lines += ["", "独立预测积分的单方案风险SE最大" + _fmt(maximum_se)
            + "；配对积分SE逐历史保存在paired.csv。EIG、Brier集中度及允许未决风险另见summary.json，"
            "Brier集中度的补数才与实际平方评分损失对应。", ""]
        audit = data["inheritance_solver"]
        lines += ["", f"共同规划风险：{audit['known_edges']}条已知继承边有{audit['planning_violations']}次退步；"
            f"提案{audit['proposal_count']}，可行返回{audit['successful_proposals']}，接受{audit['accepted_proposals']}，"
            f"未成功{audit['failed_proposals']}，完整AC提案失败{audit['full_AC_failures']}，"
            f"达到代理相对间隙要求{audit['full_gap_met']}。统一代理误差最大"
            + _fmt(audit['proxy_error_summary']['proxy_uniform_error']['maximum']) + "，取得计划的实际代理误差最大"
            + _fmt(audit['proxy_error_summary']['proxy_actual_error']['maximum'])
            + "；MILP间隙闭合不等于原Bayes风险最优。",
            "动作排序（去除重复动作、只比较同任务方案）：" + json.dumps(data["prediction_actual_ranking"]["counts"], ensure_ascii=False)
            + "。这些动作对在同一历史内相关，不能作独立成功次数。", ""]
        diag = data["prediction_diagnostics"]
        if diag["future"]:
            lines += ["| 离线未来输入诊断 | P RMSE kW | Q RMSE kvar | 源P RMSE kW | 源Q RMSE kvar |",
                      "|---|---:|---:|---:|---:|"]
            for regime in config.get("regimes", ["regular", "shift"]):
                values = [r for r in diag["future"] if r["regime"] == regime]
                means = [estimate([r["groups"][name]["rmse_physical"] for r in values])["mean"]
                         for name in ("P", "Q", "source_P", "source_Q")]
                lines.append("| " + regime + " | " + " | ".join(_fmt(v) for v in means) + " |")
            lines += ["", "该表使用保存的真实轨迹，仅用于离线诊断；输入RMSE大小不等于动作收益排序误差的贡献。", ""]
        else: lines += ["未来P/Q、源预测失配：原记录缺相应离线字段，未反推误差来源。", ""]
        physical = data["physical_saved_audit"]
        lines += ["保存态物理核查：" + json.dumps(physical["counts"], ensure_ascii=False)
            + "；最大AC方程残差" + _fmt(physical["max_ac_equation_residual"])
            + "。该项重读保存值，独立回代审计另计。", ""]
        runtime = data["runtime"]
        lines += ["平均历史耗时：总计" + _fmt(runtime["seconds"]["mean"]) + "s，平方四域"
            + _fmt(runtime["square_seconds"]["mean"]) + "s，Bayes四域增量" + _fmt(runtime["bayes_seconds"]["mean"])
            + "s，计入所用平方初始化后" + _fmt(runtime["bayes_with_square_initialization_seconds"]["mean"])
            + "s；完整AC预测/积分" + _fmt(runtime["objective_seconds"]["mean"])
            + "s（属于已列费用，不重复相加）。历史总时还含离线评价与旧联合AC审计；小批耗时不作为固有加速。", "",
            "完整AC联合重拟合另作预定审计："]
        for method, value in data["legacy_joint_audit"]["methods"].items():
            lines.append(f"- {_label(method)}：完成{value['complete_conditions']}条件、失败{value['failed_conditions']}，"
                f"原始最小损失选对{value['raw_correct_winner_conditions']}；全SSE间隙均值{_fmt(value['full_SSE_margin']['mean'])}。")
        lines += ["", "该审计每条件仅一个计量实现，取得值来自局部拟合；与Gaussian后验结果分开解释。", ""]
    for figure in summary.get("figures", []):
        lines += ["![" + figure["caption"] + "](" + figure["relative_path"] + ")", "", figure["caption"], ""]
    lines += ["## 解释范围", "", "主目标是预期MAP错误率；MILP优化其Bhattacharyya上界的保守分段代理，"
        "再用完整AC均值和重新传播的协方差核查原风险。规划256样本与独立2048样本分开；逐历史配对积分SE保存在paired.csv。"
        "源和P/Q过程已积分，训练线路及过程超参数仍固定。报告不将求解器成功、后验集合质量或小批经验频率互相替代。", "",
        "服务公平性采用相同终端电池能量任务、运行约束和吞吐上限。充放电效率及驾驶耗电会改变网侧净电量，"
        "所以这些对照并非强制相同grid kWh。计算公平性采用相同单次求解上限；Bayes还使用平方方案作初始化，"
        "其取得费用已合计，完整流程并非相同walltime。", "",
        "输入记录哈希、失败列表、条件噪声SE、预测误差和求解界见summary.json；本汇总不选择或修改策略。", ""]
    return "\n".join(lines)


def make_figures(summary, directory):
    """Two prespecified static figures; no fits, selection or good-case search."""
    evaluation = summary["phases"].get("evaluation", {})
    data = evaluation.get("plot_data", {})
    points = data.get("points", [])
    if not points: return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = directory / "figures"; folder.mkdir(exist_ok=True)
    regimes = summary["config"].get("regimes", ["regular", "shift"])
    fig, axes = plt.subplots(1, len(regimes), figsize=(9, 4), squeeze=False, constrained_layout=True)
    labels = {"constant": "Constant", "square__mobile_bidir": "Square MD", "bayes__mobile_bidir": "Bayes MD"}
    colors = dict(zip(labels, ("#7a7a7a", "#176bb0", "#c05225")))
    markers = dict(zip(labels, ("o", "s", "^")))
    for ax, regime in zip(axes[0], regimes):
        for method in labels:
            rows = [r for r in points if r["method"] == method and regime in r["actual"]]
            ax.scatter([r["predicted_risk"] for r in rows], [r["actual"][regime] for r in rows],
                c=colors[method], marker=markers[method], label=labels[method], s=42, alpha=.7)
            # Keep points at their measured coordinates; combine near-overlap
            # labels at plot resolution rather than jittering the data.
            groups = {}
            for row in rows:
                key = (round(row["predicted_risk"], 2), round(row["actual"][regime], 3))
                groups.setdefault(key, []).append(row)
            offset = {"constant": (4, 4), "square__mobile_bidir": (6, 10), "bayes__mobile_bidir": (6, 20)}[method]
            for group in groups.values():
                x = float(np.mean([r["predicted_risk"] for r in group]))
                y = float(np.mean([r["actual"][regime] for r in group]))
                label = ",".join(str(r["history_index"]) for r in group)
                ax.annotate(label, (x, y), fontsize=7, xytext=offset,
                    textcoords="offset points", color=colors[method])
        ax.plot([0, 1], [0, 1], color="black", lw=.8, ls="--")
        ax.set(xlabel="Independent predicted error", ylabel="Actual conditional error (32 noises)",
               title=regime, xlim=(-.03, 1.03), ylim=(-.03, 1.14)); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{evaluation.get('complete_histories', 0)} independent evaluation histories; paired futures")
    risk_path = folder / "predicted_vs_actual.png"; fig.savefig(risk_path, dpi=180); plt.close(fig)
    result = [dict(relative_path="figures/predicted_vs_actual.png",
        caption="预定三方案的预测与实际错误率；数字为独立历史编号，近重合点按显示精度合并标签，数据坐标不移动。两种未来和32次噪声不增加历史分母。")]
    curves = data.get("predeclared_curves", [])
    if curves:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), squeeze=False, constrained_layout=True)
        for row_index, history_index in enumerate((0, 4)):
            for method, style in (("square__mobile_bidir", "--"), ("bayes__mobile_bidir", "-")):
                curve = next((r for r in curves if r["history_index"] == history_index and r["method"] == method), None)
                if curve is None: continue
                power = np.asarray(curve["power_kw"], float); T = len(power); time = np.arange(T + 1) * curve["dt_hours"]
                net = power.sum(axis=1)
                axes[row_index, 0].step(time, np.r_[net, net[-1]], where="post", ls=style,
                    c=colors[method], label=labels[method])
                site = np.full(T, -1.)
                for arc in curve["selected_arcs"]:
                    if arc.get("origin") == arc.get("destination") and arc.get("origin_connected") and arc.get("destination_connected"):
                        site[int(arc["departure"]):int(arc["arrival"])] = arc["origin"]
                axes[row_index, 1].step(time, np.r_[site, site[-1]], where="post", ls=style,
                    c=colors[method], label=labels[method])
            for column, ax in enumerate(axes[row_index]):
                ax.set(title=f"Prespecified history {history_index}", xlabel="Time (h)",
                    ylabel="Net EV power (kW)" if column == 0 else "Connected site")
                ax.grid(alpha=.2); ax.legend(fontsize=8)
            axes[row_index, 1].set_yticks([-1, 0, 1, 2], ["Travel / unplugged", "0", "1", "2"])
        path = folder / "predeclared_power_locations.png"; fig.savefig(path, dpi=180); plt.close(fig)
        result.append(dict(relative_path="figures/predeclared_power_locations.png",
            caption="评价前固定历史0和4的平方MD与Bayes MD曲线；允许完全相同，未按收益挑选。负功率为放电，位置−1表示移动或未接线。"))
    return result


def write_outputs(summary, output):
    path = Path(output).resolve()
    directory = path.parent if path.suffix.lower() in (".md", ".json") else path
    directory.mkdir(parents=True, exist_ok=True)
    if directory == Path(summary["input"]): raise ValueError("choose a report subdirectory, not the raw run root")
    try: summary["figures"] = make_figures(summary, directory)
    except Exception as exc:
        summary["figures"] = []; summary["figure_failure"] = type(exc).__name__ + ": " + str(exc)
    json_path = path if path.suffix.lower() == ".json" else directory / "summary.json"
    md_path = path if path.suffix.lower() == ".md" else directory / "results.md"
    json_path.write_text(json.dumps(_clean(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(markdown(summary), encoding="utf-8")
    rows = [row for phase in summary["phases"].values() for row in phase["paired_history_rows"]]
    fields = sorted({key for row in rows for key in row}) or ["phase", "history_index", "method", "baseline"]
    with (directory / "paired.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return dict(summary=str(json_path), markdown=str(md_path), paired=str(directory / "paired.csv"))


def self_test():
    """Small arithmetic/schema counterexamples; no files, physics or policy run."""
    assert estimate([])["mean"] is None and estimate([1.])["t95"] is None
    assert estimate([1., 3.])["n_histories"] == 2
    fake = dict(spec=dict(index=0), status="training_failure", methods={}, candidate_miss=False)
    phase = summarize_phase([fake], "evaluation", [dict(index=0), dict(index=1)], {})
    assert phase["expected_independent_histories"] == 2 and phase["complete_histories"] == 0
    assert phase["methods"]["constant"]["missing_method_histories"] == 2
    assert phase["methods"]["constant"]["counts"]["expected_noise_draws"] == 128
    assert phase["missing_history_indices"] == [1]
    # Two paired regimes of 32 noise draws remain one independent history.
    samples = [dict(correct=int(i % 2 == 0)) for i in range(32)]
    fake = dict(methods=dict(constant=dict(action_key="a")), actions=dict(a=dict(actual={
        r: dict(samples=samples, noise_seed=3) for r in ("regular", "shift")})))
    _, actual = _history_actual(fake, "constant", "error", ("regular", "shift"))
    assert actual == .5 and _paired_noise_se(fake, "constant", "constant", ("regular", "shift")) == 0
    print("report self-test passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input"); parser.add_argument("--output"); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test: self_test(); return
    if not args.input or not args.output: parser.error("--input and --output are required")
    result = analyze(args.input); print(json.dumps(write_outputs(result, args.output), ensure_ascii=False))


if __name__ == "__main__": main()
