"""Read-only replay and paired summaries for the frozen G-precision study.

This module never fits, plans, changes a policy, or regenerates observations.
Forward AC calls only replay saved fitted states.  The statistical unit is an
independent historical record, with regular/shift treated as paired tasks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics

import numpy as np

from .. import likelihood as lk
from .. import ev_mobile_v4_domain as nested
from ..ev_mobile_service_domain import MobileServiceConfig
from ..block_joint_margin.analysis.summarize import (
    task_row, summarize_cells, paired_outcomes,
)

ROOT = Path(__file__).resolve().parents[3]
METHODS = ("OBS", "CC", "HL", "P-SQ", "G-SQ", "G-AQ")
ROUTES = ("fixed", "joint")
LABELS = ("noisy", "clean")
REGIMES = ("regular", "shift")


def read_json(path):
    path = Path(path)
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def serial(value):
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2,
                                   sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def number(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def mean(values):
    values = [float(v) for v in values if number(v) is not None]
    return statistics.fmean(values) if values else None


def distribution(values):
    values = [float(v) for v in values if number(v) is not None]
    return dict(n=len(values), mean=mean(values), sum=sum(values),
                min=min(values) if values else None, max=max(values) if values else None,
                median=statistics.median(values) if values else None)


def sign(value, tol=1e-8):
    value = number(value)
    return "unavailable" if value is None else "positive" if value > tol else "negative" if value < -tol else "zero"


class Audit:
    def __init__(self):
        self.counts = Counter()
        self.issues = []
        self.maxima = defaultdict(float)
        self.ac_calls = 0

    def check(self, kind, ok, detail):
        self.counts[kind + ":checked"] += 1
        if not ok:
            self.counts[kind + ":failed"] += 1
            self.issues.append(dict(kind=kind, detail=serial(detail)))
        return bool(ok)

    def close(self, kind, actual, stored, detail, atol=1e-7, rtol=1e-8):
        try:
            a, b = np.asarray(actual, float), np.asarray(stored, float)
            error = float(np.max(np.abs(a - b))) if a.size else 0.
            self.maxima[kind + ":absolute"] = max(self.maxima[kind + ":absolute"], error)
            ok = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a, b, atol=atol, rtol=rtol)
        except Exception as exc:
            ok, error = False, None
            detail = dict(context=detail, exception=repr(exc))
        return self.check(kind, ok, dict(context=detail, absolute_error=error))

    def record(self):
        return dict(checks=dict(self.counts), maxima=dict(self.maxima), issues=self.issues,
                    passed=not self.issues, forward_AC_replay_calls=self.ac_calls,
                    fitting_calls=0, planning_calls=0,
                    scope="saved-state numerical/physical consistency, not a global optimality or probability-calibration certificate")


def expected_specs(config, phase):
    rows = []
    for i, seed in enumerate(config[phase + "_seeds"]):
        if phase == "development":
            bank, family, H, excitation = ("n3", 0, 32, "ordinary") if i == 0 else ("n4", 1, 96, "correlated")
        else:
            bank, family = ("n3" if i < 4 else "n4"), i % 4
            H = 32 if family % 2 == 0 else 96
            excitation = "ordinary" if (family + i // 4) % 2 == 0 else "correlated"
        rows.append(dict(phase=phase, index=i, seed=seed, bank=bank,
                         family_index=family, H=H, excitation=excitation, innovation="nominal"))
    return rows


def audit_freeze(frozen, audit):
    payload = {key: value for key, value in frozen.items() if key != "identity"}
    identity = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    audit.check("freeze_identity", identity == frozen.get("identity"), dict(computed=identity, stored=frozen.get("identity")))
    for name, digest in frozen.get("source_sha256", {}).items():
        path = ROOT / name
        audit.check("source_hash", path.exists() and sha256(path.read_bytes()).hexdigest() == digest,
                    dict(path=name, expected=digest))
    audit.check("selector_truth", frozen.get("truth_in_planning") is False, "frozen truth_in_planning flag")


def polynomial(task, power):
    power = np.asarray(power, float)
    return float(task["constant"] + np.sum(np.asarray(task["linear"], float) * power)
                 + np.sum(np.asarray(task["quadratic"], float) * power * power))


def task_scores(tasks, power):
    values = {task["name"]: polynomial(task, power) for task in tasks}
    return min(values.values()) if values else None, values


def audit_coefficients(record, audit):
    planning = record.get("planning", {})
    context = planning.get("context", {})
    responses = context.get("zero_responses", {})
    sigma = np.asarray(context.get("sigmas", []), float)
    if not responses:
        return
    names = sorted(responses)
    expected = []
    for ia, a in enumerate(names):
        for b in names[ia + 1:]:
            ma, mb = np.asarray(responses[a]["mean"], float), np.asarray(responses[b]["mean"], float)
            g = np.asarray(responses[a]["jacobian"], float) - np.asarray(responses[b]["jacobian"], float)
            d = ma - mb
            expected.append(dict(name=a + "__vs__" + b,
                                 constant=.5 * float(np.sum((d / sigma) ** 2)),
                                 linear=np.einsum("tm,tmj->tj", d / sigma ** 2, g),
                                 quadratic=.5 * np.sum((g / sigma[:, :, None]) ** 2, axis=1)))
    audit.check("all_pairs", len(expected) == len(names) * (len(names) - 1) // 2,
                dict(spec=record.get("spec"), pairs=len(expected)))
    for method in ("G-SQ", "G-AQ"):
        stored = planning.get("tasks", {}).get(method, [])
        by_name = {t["name"]: t for t in stored}
        audit.check("task_pair_set", set(by_name) == {t["name"] for t in expected},
                    dict(spec=record.get("spec"), method=method, names=list(by_name)))
        for task in expected:
            if task["name"] not in by_name:
                continue
            for channel in ("constant", "linear", "quadratic"):
                target = task[channel] if method == "G-AQ" or channel == "quadratic" else np.zeros_like(task[channel])
                audit.close("task_coefficient", target, by_name[task["name"]][channel],
                            dict(method=method, pair=task["name"], channel=channel))
        for domain, result in planning.get("families", {}).get(method, {}).items():
            power = result.get("power_kw")
            if power is None:
                continue
            score, values = task_scores(stored, power)
            audit.close("family_original_score", score, result.get("score"), dict(method=method, domain=domain))
            for name, value in values.items():
                audit.close("family_task_score", value, result.get("task_values", {}).get(name),
                            dict(method=method, domain=domain, pair=name))
            lower, upper = number(result.get("objective_lower")), number(result.get("objective_upper"))
            if lower is not None and upper is not None:
                audit.check("surrogate_bounds", lower <= score + 1e-6 and score <= upper + 1e-6,
                            dict(method=method, domain=domain, score=score, lower=lower, upper=upper))
    # P-SQ deliberately uses only public voltage precisions, not a topology.
    meter = record.get("future_meter", {})
    if "v_sigma" in meter:
        sigma_v = np.broadcast_to(np.asarray(meter["v_sigma"], float), sigma.shape[:1] + (record["candidate_structures"][0]["n"],))
        least = float(np.min(sigma_v))
        weights = ((least if least <= 1 else 1.) / sigma_v) ** 2
        for domain, result in planning.get("families", {}).get("P-SQ", {}).items():
            if "power_kw" in result:
                score = float(np.sum(weights * np.asarray(result["power_kw"], float) ** 2))
                audit.close("precision_original_score", score, result.get("score"), dict(domain=domain))


def audit_operations(record, audit):
    planning = record.get("planning", {})
    if "domain_config" not in planning:
        return
    domains = nested.build_nested_domains(MobileServiceConfig(**planning["domain_config"]))
    rows = []
    for method, plan in planning.get("plans", {}).items():
        if method == "OBS":
            audit.check("passive_zero_power", np.allclose(plan.get("power"), 0), record.get("spec"))
            continue
        rows.append(("plan:" + method, plan.get("domain"), plan.get("operation_x"), plan.get("power")))
    for method, family in planning.get("families", {}).items():
        for domain, result in family.items():
            rows.append(("family:" + method, domain, result.get("operation_x"), result.get("power_kw")))
            for label, parent in result.get("parents", {}).items():
                if parent.get("operation_x") is not None:
                    rows.append(("parent:" + method + ":" + label, domain, parent["operation_x"], parent.get("power_kw")))
    for label, name, x, power in rows:
        if x is None:
            continue
        try:
            domain = domains[name]
            result = domain.check(np.asarray(x, float))
            audit.check("operation_feasibility", result.get("feasible", False), dict(label=label, domain=name, result=result))
            if power is not None:
                actual = np.asarray(x, float)[domain.indices.u].reshape(np.asarray(power).shape)
                audit.close("operation_power", actual, power, dict(label=label, domain=name))
        except Exception as exc:
            audit.check("operation_replay", False, dict(label=label, domain=name, exception=repr(exc)))


def obs_arrays(observations, meter):
    p = np.asarray(observations["p"], float)
    t, n = p.shape
    obs = {name: np.asarray(observations[name], float) for name in ("p", "q", "v", "master_p", "master_q")}
    obs["root_v"] = np.broadcast_to(np.asarray(observations.get("root_v", 1.), float), (t,)).copy()
    mm = {}
    for name in ("bias_p", "bias_q", "bias_v", "v_sigma", "master_p_sigma", "master_q_sigma"):
        shape = (t,) if name.startswith("master") else (t, n)
        mm[name] = np.broadcast_to(np.asarray(meter.get(name, 0.), float), shape).copy()
    return obs, mm


def clean_observations(noisy, meter, white):
    result = dict(noisy)
    white = np.asarray(white, float)
    offset = 0
    for name in ("p", "q", "v", "master_p", "master_q"):
        values = np.asarray(noisy[name], float)
        sigma = np.broadcast_to(np.asarray(meter[name + "_sigma"], float), values.shape)
        stop = offset + values.size
        result[name] = values - sigma * white[offset:stop].reshape(values.shape)
        offset = stop
    if offset != white.size:
        raise ValueError("saved white-noise length differs from all metered channels")
    return result


def join_blocks(blocks):
    return ({name: np.concatenate([b[0][name] for b in blocks], axis=0) for name in blocks[0][0]},
            {name: np.concatenate([b[1][name] for b in blocks], axis=0) for name in blocks[0][1]})


def in_bounds(values, bounds, tolerance=1e-8):
    if bounds is None:
        return True
    values = np.asarray(values, float)
    return bool(np.isfinite(values).all() and np.all(values >= bounds[0] - tolerance) and np.all(values <= bounds[1] + tolerance))


def replay_entry(candidate, entry, observations, meters, history_t, options, audit, label):
    state = entry.get("state")
    if not state:
        return
    obs, mm = observations, meters
    audit.ac_calls += 1
    physical = lk.ac_predict(candidate, np.asarray(state["r"], float), np.asarray(state["x"], float),
        obs["p"] - mm["bias_p"], obs["q"] - mm["bias_q"],
        np.asarray(state["source_p"], float), np.asarray(state["source_q"], float), obs["root_v"])
    residual = dict(v=(physical["v"] + mm["bias_v"] - obs["v"]) / mm["v_sigma"],
                    master_p=(physical["master_p"] - obs["master_p"]) / mm["master_p_sigma"],
                    master_q=(physical["master_q"] - obs["master_q"]) / mm["master_q_sigma"])
    losses = {}
    for block, indices in (("history", slice(None, history_t)), ("future", slice(history_t, None))):
        channels = {name: float(np.sum(values[indices] ** 2)) for name, values in residual.items()}
        losses[block] = sum(channels.values())
        stored = entry.get("blocks", {}).get(block, {})
        audit.close("AC_block_SSE", losses[block], stored.get("loss"), dict(label=label, block=block))
        for name, value in channels.items():
            audit.close("AC_channel_SSE", value, stored.get("channel_loss", {}).get(name), dict(label=label, block=block, channel=name))
    total = sum(losses.values())
    audit.close("AC_total_SSE", total, entry.get("loss"), label)
    audit.check("AC_balance", physical["ac_equation_residual"] <= 1e-8 and physical["power_balance_residual"] <= 1e-8, label)
    audit.maxima["AC_equation_residual"] = max(audit.maxima["AC_equation_residual"], physical["ac_equation_residual"])
    audit.maxima["AC_power_balance_residual"] = max(audit.maxima["AC_power_balance_residual"], physical["power_balance_residual"])
    voltage = np.abs(physical["voltage"])
    physical_checks = dict(
        line_r=in_bounds(state["r"], options.get("r_bounds")),
        line_x=in_bounds(state["x"], options.get("x_bounds")),
        x_over_r=in_bounds(np.asarray(state["x"]) / np.asarray(state["r"]), options.get("ratio_bounds")),
        source_p=in_bounds(state["source_p"], options.get("source_p_bounds")),
        source_q=in_bounds(state["source_q"], options.get("source_q_bounds")),
        voltage=in_bounds(voltage, options.get("voltage_bounds")))
    # Saved unusable states remain in the denominator; their violated bounds
    # are evidence, rather than an unexplained postprocessing exception.
    if entry.get("usable", entry.get("success", False)):
        audit.check("AC_usable_physical_bounds", all(physical_checks.values()), dict(label=label, checks=physical_checks))
    return total


def audit_evaluation(record, audit):
    if not record.get("evaluation"):
        return
    structures = {}
    for item in record.get("candidate_structures", []):
        structures[item["name"]] = lk.RadialCandidate.from_supports(item["name"], item["n"], item["supports"])
    blocks = [obs_arrays(record[name], record[meter]) for name, meter in
              (("observed_training", "train_meter"), ("observed_evidence", "evidence_meter"))]
    historical, historical_meter = join_blocks(blocks)
    history_t = len(historical["p"])
    options = record.get("fit_options", {})
    for action_key, action in record["evaluation"].get("actions", {}).items():
        for regime, actual in action.get("regimes", {}).items():
            if actual.get("physical_generation_failed") or "observations" not in actual:
                continue
            for measurement in LABELS:
                if measurement + "_refit" not in actual:
                    continue
                try:
                    future = actual["observations"] if measurement == "noisy" else clean_observations(
                        actual["observations"], record["future_meter"], actual["actual_white_noise"])
                    observations, meters = join_blocks([(historical, historical_meter), obs_arrays(future, record["future_meter"])])
                    fit = actual[measurement + "_refit"]
                    for name, comparison in fit.get("comparisons", {}).items():
                        for route in ROUTES:
                            entry = comparison.get(route, {})
                            if not entry.get("state"):
                                continue
                            label = dict(spec=record["spec"], action=action_key, regime=regime,
                                         measurement=measurement, candidate=name, route=route)
                            try:
                                total = replay_entry(structures[name], entry, observations, meters, history_t, options, audit, label)
                                if name in fit.get("losses", {}).get(route, {}):
                                    audit.close("AC_loss_map", total, fit["losses"][route][name], label)
                            except Exception as exc:
                                audit.check("AC_replay_exception", False, dict(context=label, exception=repr(exc)))
                except Exception as exc:
                    audit.check("observation_reconstruction", False, dict(action=action_key, regime=regime, measurement=measurement, exception=repr(exc)))


def history_detail(record):
    planning, evaluation = record.get("planning", {}), record.get("evaluation", {})
    plans = planning.get("plans", {})
    plan_rows = {}
    for name, plan in plans.items():
        check = plan.get("physicalcheck", {})
        plan_rows[name] = dict(domain=plan.get("domain"), action_key=plan.get("action_key"),
            fallback_reason=plan.get("fallback_reason"), passive_control=plan.get("passive_control", False),
            feasible=check.get("feasible"), moves=check.get("moves"),
            throughput_kwh=check.get("throughput_kwh"), grid_energy_kwh=check.get("grid_energy_kwh"),
            travel_energy_kwh=check.get("travel_energy_kwh"), driver_cost=check.get("driver_cost"))
    solver_rows = []
    for method, family in planning.get("families", {}).items():
        for domain, result in family.items():
            solver_rows.append(dict(method=method, domain=domain, success=result.get("success"),
                score=result.get("score"), lower=result.get("objective_lower"), upper=result.get("objective_upper"),
                gap=result.get("gap"), target_gap=result.get("target_gap"), completed=result.get("completed"),
                unfinished=result.get("unfinished"), seconds=result.get("seconds"),
                iterations=result.get("n_iterations"), raw_failures=result.get("failures", []),
                selected_origin=result.get("selected_origin"), score_gain=result.get("score_gain"),
                inheritance_protected=result.get("inheritance_protected"),
                bound_scope=result.get("bound_scope")))
    counters = Counter()
    for key, value in planning.get("history", {}).get("counters", {}).items():
        if number(value) is not None:
            counters[key] += value
    counters["ac_calls"] += planning.get("context", {}).get("ac_calls", 0)
    evaluation_seconds = []
    physical_failures = 0
    for action in evaluation.get("actions", {}).values():
        for actual in action.get("regimes", {}).values():
            physical_failures += bool(actual.get("physical_generation_failed"))
            for label in LABELS:
                fitted = actual.get(label + "_refit", {})
                for key, value in fitted.get("counters", {}).items():
                    if number(value) is not None:
                        counters[key] += value
                evaluation_seconds.append(fitted.get("seconds"))
    pair_rows = []
    tasks = planning.get("tasks", {})
    base = plans.get("P-SQ", {})
    for method in ("G-SQ", "G-AQ"):
        new = plans.get(method, {})
        own_tasks = tasks.get(method, [])
        baseline_score = task_scores(own_tasks, base["power"])[0] if own_tasks and "power" in base else None
        new_score = task_scores(own_tasks, new["power"])[0] if own_tasks and "power" in new else None
        nominal_key = "incremental_score" if method == "G-SQ" else "total_score"
        nominal_base = number(planning.get("nominal_AC", {}).get("P-SQ", {}).get(nominal_key))
        nominal_new = number(planning.get("nominal_AC", {}).get(method, {}).get(nominal_key))
        surrogate_delta = new_score - baseline_score if new_score is not None and baseline_score is not None else None
        nominal_delta = nominal_new - nominal_base if nominal_new is not None and nominal_base is not None else None
        pair = dict(method=method, same_action_as_P_SQ=bool(new and base and new.get("action_key") == base.get("action_key")),
            fallback_reason=new.get("fallback_reason"), baseline_surrogate=baseline_score, new_surrogate=new_score,
            surrogate_delta=surrogate_delta, surrogate_sign=sign(surrogate_delta),
            baseline_nominal_AC=nominal_base, new_nominal_AC=nominal_new,
            nominal_AC_delta=nominal_delta, nominal_AC_sign=sign(nominal_delta), actual=[])
        for route in ROUTES:
            for label in LABELS:
                margins = []
                for regime in REGIMES:
                    a, b = (task_row(record, m, regime, route, label) for m in ("P-SQ", method))
                    left, right = a["true_margin_evaluation_only"], b["true_margin_evaluation_only"]
                    delta = right - left if right is not None and left is not None else None
                    margins.append(delta)
                    pair["actual"].append(dict(route=route, measurement=label, regime=regime,
                        baseline_half_SSE_margin=left, new_half_SSE_margin=right, delta=delta,
                        sign=sign(delta), both_available=a["available"] and b["available"],
                        same_prediction=a["predicted"] == b["predicted"] if a["available"] and b["available"] else None,
                        baseline_error=a["error"], new_error=b["error"]))
                pair.setdefault("history_mean_margin_delta", {})[route + "/" + label] = mean(margins) if all(v is not None for v in margins) else None
        pair_rows.append(pair)
    return dict(spec=record.get("spec"), status=record.get("status"), seconds=record.get("seconds"),
        failures=record.get("failures", {}), planning_failures=planning.get("failures", {}),
        plans=plan_rows, solver=solver_rows, cost=planning.get("cost", {}),
        scientific_counters=dict(counters), evaluation_seconds=sum(v for v in evaluation_seconds if number(v) is not None),
        unique_evaluated_actions=len(evaluation.get("actions", {})), physical_generation_failures=physical_failures,
        paired_G_vs_P_SQ=pair_rows)


def summarize_phase(records, expected, details):
    tasks = [task_row(record, method, regime, route, label) for record in records
             for method in METHODS for route in ROUTES for label in LABELS for regime in REGIMES]
    def select(method, route, label):
        return [r for r in tasks if r["method"] == method and r["route"] == route and r["measurement"] == label]
    tables = {method: {route: {label: summarize_cells(select(method, route, label), len(expected), len(records))
               for label in LABELS} for route in ROUTES} for method in METHODS}
    comparisons = {method: {route: {label: paired_outcomes(select("P-SQ", route, label), select(method, route, label))
                    for label in LABELS} for route in ROUTES} for method in METHODS if method != "P-SQ"}
    inference = {method: {label: paired_outcomes(select(method, "fixed", label), select(method, "joint", label))
                 for label in LABELS} for method in METHODS}
    diagnostic = {}
    for method in ("G-SQ", "G-AQ"):
        pairs = [p for d in details for p in d["paired_G_vs_P_SQ"] if p["method"] == method]
        changed = [p for p in pairs if not p["same_action_as_P_SQ"]]
        diagnostic[method] = dict(n_histories=len(pairs), changed_plan_histories=len(changed),
            fallback_histories=sum(p["fallback_reason"] is not None for p in pairs),
            surrogate_delta=distribution(p["surrogate_delta"] for p in pairs),
            surrogate_signs=dict(Counter(p["surrogate_sign"] for p in pairs)),
            nominal_AC_delta=distribution(p["nominal_AC_delta"] for p in pairs),
            nominal_AC_signs=dict(Counter(p["nominal_AC_sign"] for p in pairs)), actual={})
        for route in ROUTES:
            for label in LABELS:
                rows = [r for p in pairs for r in p["actual"] if r["route"] == route and r["measurement"] == label]
                changed_rows = [r for p in changed for r in p["actual"] if r["route"] == route and r["measurement"] == label]
                diagnostic[method]["actual"][route + "/" + label] = dict(
                    n_tasks=len(rows), margin_signs=dict(Counter(r["sign"] for r in rows)),
                    changed_task_margin_signs=dict(Counter(r["sign"] for r in changed_rows)),
                    same_prediction_tasks=sum(r["same_prediction"] is True for r in rows),
                    available_prediction_pairs=sum(r["same_prediction"] is not None for r in rows),
                    equal_history_margin_delta=distribution(p["history_mean_margin_delta"][route + "/" + label] for p in pairs))
    solver = [r for d in details for r in d["solver"]]
    solver_summary = {method: {domain: dict(
        n_conditions=len(rows := [r for r in solver if r["method"] == method and r["domain"] == domain]),
        completed=sum(r["completed"] is True for r in rows), unfinished=sum(r["unfinished"] is True for r in rows),
        success=sum(r["success"] is True for r in rows), raw_failure_conditions=sum(bool(r["raw_failures"]) for r in rows),
        gaps=distribution(r["gap"] for r in rows), seconds=distribution(r["seconds"] for r in rows))
        for domain in nested.DOMAIN_FLAGS} for method in ("P-SQ", "G-SQ", "G-AQ")}
    return dict(n_expected_independent_histories=len(expected), n_written_independent_histories=len(records),
        pending_indices=sorted(set(s["index"] for s in expected) - {r["spec"]["index"] for r in records}),
        record_status=dict(Counter(r.get("status", "missing") for r in records)),
        tables=tables, paired_vs_P_SQ=comparisons, paired_joint_vs_fixed=inference,
        score_and_margin_diagnostics=diagnostic, solver=solver_summary,
        plan_metrics={method: {key: distribution(d["plans"].get(method, {}).get(key) for d in details)
            for key in ("moves", "throughput_kwh", "travel_energy_kwh", "driver_cost")}
            for method in METHODS},
        costs={key: distribution(d["cost"].get(key) for d in details)
               for key in sorted({k for d in details for k in d["cost"]})},
        counters=dict(sum((Counter(d["scientific_counters"]) for d in details), Counter())),
        physical_generation_failures=sum(d["physical_generation_failures"] for d in details), task_rows=tasks)


def report_text(summary, audit):
    lines = ["# G-precision frozen study", "", "This report separates EV planning from final shared-line refitting. "
        "One history is one statistical unit; its two future regimes are paired tasks.", "",
        f"Status: **{summary['status']}**. Saved-state audit: **{'passed' if audit['passed'] else 'issues found'}**.", "",
        "|Phase|Method|History records|Joint noisy errors / available|Failed tasks|Unresolved incl. failure|Joint clean errors / available|",
        "|---|---|---:|---:|---:|---:|---:|"]
    for phase in ("development", "evaluation"):
        result = summary[phase]
        for method in METHODS:
            noisy, clean = (result["tables"][method]["joint"][label] for label in LABELS)
            lines.append(f"|{phase}|{method}|{result['n_written_independent_histories']}/{result['n_expected_independent_histories']}|"
                f"{noisy['error_tasks']}/{noisy['available_tasks']}|{noisy['failed_tasks']}|"
                f"{noisy['unresolved_tasks_including_failure']}|{clean['error_tasks']}/{clean['available_tasks']}|")
    lines.extend(["", "P-SQ maximizes precision-weighted power squares. G-SQ uses AC-response difference squares; "
        "G-AQ also retains the current response difference and its signed linear term. All three optimize the same nested EV domains. "
        "Their solver bounds apply to the frozen polynomial surrogate, not to nonlinear AC classification.", "",
        "For each G policy, the paired summary separately records surrogate improvement, nominal full-AC response distance, "
        "and the true-topology margin (half the SSE difference to its strongest competitor). They are different quantities. "
        "The true label is used only after plans are sealed.", "",
        "Clean removes only the new meter noise. Historical data, trained parameters and ordinary forecast errors remain unchanged. "
        "Unresolved means a numerical SSE tie or a retained failure; no probability calibration or confidence-set coverage is claimed.", "",
        "OBS keeps the same measurement duration but performs no battery service. CC, HL and the three MILP policies share the declared battery task. "
        "Failures and pending histories remain explicit. Development is diagnostic; evaluation is not used to tune the policy.", "",
        f"Audit forward-AC calls: {audit['forward_AC_replay_calls']}; fitting/planning calls: 0. "
        "Audit replay cost is reported separately from the experiment's saved computation costs.", ""])
    if audit["issues"]:
        lines.append(f"Audit issues: {len(audit['issues'])}; see audit.json before interpreting results.")
    return "\n".join(lines) + "\n"


def analyze(output):
    output = Path(output).resolve()
    audit = Audit()
    frozen = read_json(output / "freeze.json")
    audit_freeze(frozen, audit)
    summary = dict(freeze_identity=frozen.get("identity"), analysis_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        generated_utc=datetime.now(timezone.utc).isoformat(),
        statistical_unit="independent historical record; two paired future regimes",
        probability_calibrated=False, evaluation_tuned=False,
        clean_scope="new metering noise removed only; historical training remains noisy")
    all_details = {}
    pending, failed = 0, 0
    for phase in ("development", "evaluation"):
        expected = expected_specs(frozen["config"], phase)
        records, details = [], []
        for spec in expected:
            path = output / phase / f"history_{spec['index']:03d}.json.gz"
            if not path.exists():
                pending += 1
                continue
            try:
                record = read_json(path)
            except Exception as exc:
                audit.check("history_read", False, dict(path=str(path), exception=repr(exc)))
                pending += 1
                continue
            records.append(record)
            audit.check("record_spec", record.get("spec") == spec, dict(path=str(path), expected=spec, actual=record.get("spec")))
            audit.check("record_freeze", record.get("freeze_identity") == frozen.get("identity"), str(path))
            audit.check("record_source_unchanged", record.get("source_unchanged") is True, str(path))
            audit.check("record_config", record.get("config") == frozen.get("config"), str(path))
            for name, function in (("coefficients", audit_coefficients), ("operations", audit_operations), ("evaluation", audit_evaluation)):
                try:
                    function(record, audit)
                except Exception as exc:
                    audit.check(name + "_exception", False, dict(path=str(path), exception=repr(exc)))
            detail = history_detail(record)
            detail["record_path"] = str(path)
            detail["record_sha256"] = sha256(path.read_bytes()).hexdigest()
            details.append(detail)
            failed += record.get("status") != "complete"
        summary[phase] = summarize_phase(records, expected, details)
        all_details[phase] = details
    summary["status"] = "pending" if pending else "complete_with_failures" if failed else "complete"
    summary["pending_histories"] = pending
    summary["noncomplete_record_histories"] = failed
    audit_record = audit.record()
    audit_record["analysis_sha256"] = summary["analysis_sha256"]
    summary["audit_passed"] = audit_record["passed"]
    write_json(output / "summary.json", summary)
    write_json(output / "audit.json", audit_record)
    write_json(output / "history_pairs.json", all_details)
    (output / "report.md").write_text(report_text(summary, audit_record), encoding="utf-8")
    return summary, audit_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "g_precision_v1")
    args = parser.parse_args()
    summary, audit = analyze(args.output)
    print(json.dumps(dict(status=summary["status"], pending_histories=summary["pending_histories"],
        audit_passed=audit["passed"], audit_issues=len(audit["issues"]),
        forward_AC_replay_calls=audit["forward_AC_replay_calls"])))


if __name__ == "__main__":
    main()
