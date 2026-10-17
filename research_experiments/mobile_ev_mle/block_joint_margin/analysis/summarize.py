"""Read-only scientific postprocessing for the frozen block-margin pilot.

No project code is imported, fitted, optimized, simulated, or reconfigured.
The fixed historical record is the statistical unit; regular/shift are two
paired future tasks, and clean is a diagnostic reuse, not another history.

Optional recovery_manifest.json schema:
  {"phase":"evaluation", "mapping":{"2":"private/history_002.json.gz"},
   "operational_failures":[...], "scope":"I/O-only, same seed/source/config"}
Bare index->path and phase->{index:path} mappings are also accepted.  A
recovery cannot replace an existing public scientific result.  Effective
records are checked against frozen phase/index/seed/source hashes.
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
import struct


METHODS = ("constant", "SQMD", "joint_margin")
ROUTES = ("fixed", "joint")
LABELS = ("noisy", "clean")
REGIMES = ("regular", "shift")


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def mean(values):
    values = [number(v) for v in values if number(v) is not None]
    return None if not values else statistics.fmean(values)


def rate(count, denominator):
    return None if not denominator else count / denominator


def distribution(values):
    values = [number(v) for v in values if number(v) is not None]
    return dict(n=len(values), sum=sum(values), mean=mean(values),
                median=None if not values else statistics.median(values),
                min=None if not values else min(values),
                max=None if not values else max(values))


def counter_sum(records):
    result = Counter()
    for record in records:
        for key, value in (record or {}).items():
            if number(value) is not None:
                result[key] += value
    return dict(result)


def recovery_mapping(output):
    path = output / "recovery_manifest.json"
    if not path.exists():
        return {}, dict(present=False, source=None, operational_failures=[])
    manifest = read_json(path)
    mapping = manifest.get("mapping")
    if mapping is None:
        mapping = {key: value for key, value in manifest.items()
                   if key in ("development", "evaluation") or str(key).isdigit()}
    records = {}
    if any(str(key).isdigit() for key in mapping):
        phase = manifest.get("phase", "evaluation")
        mapping = {phase: mapping}
    for phase, indices in mapping.items():
        if phase not in ("development", "evaluation") or not isinstance(indices, dict):
            raise ValueError("Recovery mapping needs a development/evaluation phase")
        for index, location in indices.items():
            if isinstance(location, dict):
                location = location.get("effective_history_path", location.get("path"))
            target = Path(location)
            if not target.is_absolute():
                target = path.parent / target
            records[(phase, int(index))] = target.resolve()
    return records, dict(present=True, source=str(path), sha256=digest(path),
        operational_failures=manifest.get("operational_failures", []),
        retries=manifest.get("retries", []), scope=manifest.get("scope"),
        n_requested_effective_records=len(records))


def load_phase(output, phase, freeze, recovery):
    seeds = freeze["config"][phase + "_seeds"]
    loaded, manifests, pending, issues = [], [], [], []
    for index, seed in enumerate(seeds):
        public = output / phase / ("history_%03d.json.gz" % index)
        effective = recovery.get((phase, index), public).resolve()
        recovered = effective != public.resolve()
        metadata = dict(phase=phase, index=index, seed=seed,
                        public_path=str(public), effective_path=str(effective),
                        recovered_io_only=recovered, public_record_exists=public.exists())
        if recovered and public.exists():
            # Choosing the better of repeated numerical/scientific outcomes is
            # prohibited.  I/O-before-record failures have no public record.
            raise ValueError("Recovery would replace an existing scientific record: " + str(public))
        if not effective.exists():
            pending.append(dict(metadata, reason="effective record not written"))
            continue
        value = read_json(effective)
        spec = value.get("spec", {})
        checks = dict(
            phase=spec.get("phase") == phase,
            index=spec.get("index") == index,
            seed=spec.get("seed") == seed,
            freeze_identity=value.get("freeze_identity") == freeze["identity"],
            source_sha=value.get("source_sha") == freeze["source_sha"],
            source_unchanged_during_run=value.get("source_unchanged_during_run") is True)
        metadata.update(sha256=digest(effective), status=value.get("status"), checks=checks)
        if not all(checks.values()):
            issues.append(dict(index=index, checks=checks, path=str(effective)))
        loaded.append(value)
        manifests.append(metadata)
    expected_names = {"history_%03d.json.gz" % i for i in range(len(seeds))}
    unexpected = [str(p) for p in (output / phase).glob("history_*.json.gz")
                  if p.name not in expected_names]
    if unexpected:
        issues.append(dict(unexpected_public_records=unexpected))
    return loaded, manifests, pending, issues


def find_action(record, method):
    actions = record.get("evaluation", {}).get("actions", {})
    matches = [(key, action) for key, action in actions.items()
               if method in action.get("methods", [])]
    if len(matches) > 1:
        raise ValueError("A method is assigned to multiple evaluated actions")
    return matches[0] if matches else (None, {})


def task_row(record, method, regime, route, label):
    spec = record.get("spec", {})
    action_key, action = find_action(record, method)
    actual = action.get("regimes", {}).get(regime, {})
    fit = actual.get(label + "_refit", {})
    decision = actual.get(label + "_decisions", {}).get(route, {})
    available = bool(decision.get("available") and decision.get("error") is not None)
    error = bool(decision["error"]) if available else None
    failures = (decision.get("failures") or fit.get("failures") or fit.get("failure")
                or actual.get("failure") or record.get("failures") or "missing decision")
    losses = decision.get("losses", {})
    truth = record.get("evaluation", {}).get("true_name_evaluation_only")
    competitors = [number(value) for key, value in losses.items() if key != truth]
    competitors = [value for value in competitors if value is not None]
    true_loss = number(losses.get(truth))
    margin = ((min(competitors) - true_loss) / 2
              if true_loss is not None and competitors else None)
    lines = decision.get("line_error", {})
    return dict(phase=spec.get("phase"), index=spec.get("index"), seed=spec.get("seed"),
        regime=regime, method=method, route=route, measurement=label,
        available=available, failed=not available, error=error,
        predicted=decision.get("predicted"),
        unresolved=bool(decision.get("unresolved", True)) if available else True,
        retained=decision.get("retained", [c.get("name")
            for c in record.get("candidate_structures", [])]),
        failure=None if available else failures,
        action_key=action_key, true_margin_evaluation_only=margin,
        r_mean_absolute_relative=number(lines.get("r", {}).get("mean_absolute_relative")),
        x_mean_absolute_relative=number(lines.get("x", {}).get("mean_absolute_relative")),
        r_relative_l2=number(lines.get("r", {}).get("relative_l2")),
        x_relative_l2=number(lines.get("x", {}).get("relative_l2")))


def summarize_cells(rows, expected_histories, written_histories):
    total, available = len(rows), sum(row["available"] for row in rows)
    errors = sum(row["error"] is True for row in rows)
    correct = sum(row["error"] is False for row in rows)
    failed = total - available
    numerical_ties = sum(row["available"] and row["unresolved"] for row in rows)
    unique_correct = sum(row["available"] and row["error"] is False
                         and not row["unresolved"] for row in rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["index"]].append(row)
    complete = [group for group in groups.values()
                if len(group) == len(REGIMES) and all(r["available"] for r in group)]
    line_errors = {}
    for key in ("r_mean_absolute_relative", "x_mean_absolute_relative",
                "r_relative_l2", "x_relative_l2"):
        by_history = [mean([r[key] for r in group if r["available"]])
                      for group in groups.values()]
        full_history = [mean([r[key] for r in group]) for group in complete]
        line_errors[key] = dict(
            equal_history_mean=mean(by_history),
            n_histories=sum(value is not None for value in by_history),
            complete_history_mean=mean(full_history), n_complete_histories=len(complete))
    return dict(
        expected_independent_histories=expected_histories,
        written_independent_histories=written_histories,
        fully_available_histories=len(complete),
        expected_tasks=expected_histories * len(REGIMES), written_tasks=total,
        pending_tasks=(expected_histories - written_histories) * len(REGIMES),
        available_tasks=available, failed_tasks=failed,
        error_tasks=errors, correct_label_tasks=correct, unique_correct_tasks=unique_correct,
        wrong_decisive_tasks=sum(row["error"] is True and not row["unresolved"] for row in rows),
        numerical_tie_tasks=numerical_ties,
        unresolved_tasks_including_failure=numerical_ties + failed,
        error_rate_available=rate(errors, available),
        unique_correct_rate_all_written=rate(unique_correct, total),
        unresolved_rate_all_written=rate(numerical_ties + failed, total),
        error_rate_interval_without_guessing_failures=[rate(errors, total),
                                                     rate(errors + failed, total)],
        line_error=line_errors,
        unresolved_scope="numerical SSE tie or failure; no calibrated confidence set")


def paired_outcomes(base, new):
    a = {(r["index"], r["regime"]): r for r in base}
    b = {(r["index"], r["regime"]): r for r in new}
    if set(a) != set(b):
        raise ValueError("Paired task keys differ")
    counts = Counter()
    history = defaultdict(list)
    margin_deltas = []
    for key in sorted(a):
        left, right = a[key], b[key]
        if not left["available"] or not right["available"]:
            category = ("both_failed" if left["failed"] and right["failed"]
                        else "base_failed" if left["failed"] else "new_failed")
            counts[category] += 1
            continue
        delta = int(right["error"]) - int(left["error"])
        counts["new_wins" if delta < 0 else "new_losses" if delta > 0 else "ties"] += 1
        counts["both_wrong" if right["error"] and left["error"] else
               "both_correct" if not right["error"] and not left["error"] else "discordant"] += 1
        history[key[0]].append(delta)
        if left["true_margin_evaluation_only"] is not None and right["true_margin_evaluation_only"] is not None:
            margin_deltas.append(right["true_margin_evaluation_only"] - left["true_margin_evaluation_only"])
    full = {str(index): statistics.fmean(values) for index, values in history.items()
            if len(values) == len(REGIMES)}
    return dict(n_written_paired_tasks=len(a),
        n_jointly_available_tasks=sum(counts[k] for k in ("new_wins", "new_losses", "ties")),
        **{key: counts[key] for key in ("new_wins", "new_losses", "ties", "both_wrong",
            "both_correct", "base_failed", "new_failed", "both_failed")},
        n_complete_paired_histories=len(full),
        mean_error_delta_new_minus_base_complete_histories=mean(list(full.values())),
        history_error_deltas=full,
        mean_true_margin_delta_available_tasks=mean(margin_deltas),
        statistical_unit="history; two future regimes are paired tasks, not independent draws")


def power_key(power):
    """Match the frozen selector's time-major little-endian float64 key."""
    return sha256(b"".join(struct.pack("<d", float(value))
                           for row in power for value in row)).hexdigest()


def saved_forecast_diagnostics(record, source_path):
    """Posthoc truth-label slice of already sealed objective records only."""
    plan = record.get("planning", {}).get("plans", {})
    wanted = {method: power_key(plan[method]["power"])
              for method in ("SQMD", "joint_margin") if method in plan}
    steps = source_path.with_name(source_path.name.replace(".json.gz", ".steps"))
    sealed = {}
    for step in sorted(steps.glob("*objective_*.json.gz")):
        raw = read_json(step)
        details = raw.get("details", {})
        key, result = details.get("key"), details.get("result", {})
        if key not in wanted.values():
            continue
        if raw.get("freeze_identity") != record.get("freeze_identity"):
            raise ValueError("Saved objective step differs from the history freeze")
        if result.get("power_kw") is None or power_key(result["power_kw"]) != key:
            raise ValueError("Saved objective key and power differ")
        sealed[key] = (step, result)
    truth = record.get("evaluation", {}).get("true_name_evaluation_only")
    outputs = {}
    for method, key in wanted.items():
        if key not in sealed:
            outputs[method] = dict(available=False, reason="matching sealed objective step missing")
            continue
        step, result = sealed[key]
        by_competitor = {}
        for plane in result.get("planes", []):
            a, b = plane["pair"]
            if a != truth:
                continue
            value = number(plane.get("value"))
            if value is not None:
                by_competitor[b] = min(by_competitor.get(b, math.inf), value)
        values = list(by_competitor.values())
        cap = record["config"]["margin_cap"]
        outputs[method] = dict(available=bool(result.get("complete") and values),
            all_reference_capped_average_score=result.get("score"),
            true_reference_worst_competitor_margin=min(values) if values else None,
            true_reference_capped_average_margin=mean([min(cap, value) for value in values]),
            true_reference_competitor_margins=by_competitor,
            source_step=str(step), source_sha256=digest(step))
    return dict(methods=outputs, truth_label_used_only_in_postprocessing=True,
        scope="sealed hypothetical full-AC fits; true-reference slice is evaluation-only, not selector input")


def sign_comparison(predicted_delta, actual_delta, tolerance=1e-8):
    a, b = number(predicted_delta), number(actual_delta)
    if a is None or b is None:
        return "unavailable"
    sa = 0 if abs(a) <= tolerance else (1 if a > 0 else -1)
    sb = 0 if abs(b) <= tolerance else (1 if b > 0 else -1)
    if not sa or not sb:
        return "zero_in_one_or_both"
    return "same_sign" if sa == sb else "opposite_sign"


def ranking_diagnostics(detail, tasks):
    index = detail["spec"]["index"]
    methods = detail["sealed_forecast_diagnostics"]["methods"]
    sq, jm = methods.get("SQMD", {}), methods.get("joint_margin", {})
    p0, p1 = (number(m.get("true_reference_worst_competitor_margin")) for m in (sq, jm))
    conditional_delta = None if p0 is None or p1 is None else p1 - p0
    selected = {(row["method"], row["regime"], row["measurement"]): row
                for row in tasks if row["index"] == index and row["route"] == "joint"}
    results = []
    for regime in REGIMES:
        for label in LABELS:
            left, right = selected.get(("SQMD", regime, label), {}), selected.get(("joint_margin", regime, label), {})
            a, b = (number(r.get("true_margin_evaluation_only")) for r in (left, right))
            actual_delta = None if a is None or b is None else b - a
            results.append(dict(regime=regime, measurement=label,
                SQ_true_margin=a, JM_true_margin=b, actual_true_margin_delta=actual_delta,
                all_reference_score_delta=detail["score_gain"],
                true_reference_worst_margin_delta=conditional_delta,
                aggregate_score_vs_actual_sign=sign_comparison(detail["score_gain"], actual_delta),
                true_reference_forecast_vs_actual_sign=sign_comparison(conditional_delta, actual_delta)))
    return dict(all_reference_score_delta=detail["score_gain"],
        true_reference_worst_margin_delta=conditional_delta,
        aggregate_vs_true_reference_sign=sign_comparison(detail["score_gain"], conditional_delta),
        realized=results,
        scope="S is a capped all-reference average; realized margin concerns one true topology and its strongest competitor. Sign disagreement alone is not a pure response-prediction error. True-reference forecasts still use hypothetical support/forecasts, whereas realized margins use actual future meters.")


def history_details(record, source_path):
    planning, evaluation = record.get("planning", {}), record.get("evaluation", {})
    gate = planning.get("gradient_gate", {})
    logs = planning.get("planning_logs", {}).get("joint_margin", {})
    costs = planning.get("costs", {})
    checks = gate.get("checks", [])
    plans, service = planning.get("plans", {}), planning.get("operation_config", {})
    plan_rows = {}
    for method in METHODS:
        plan = plans.get(method, {})
        audit = plan.get("physicalcheck", {})
        energies = audit.get("energy_kwh", [])
        plan_rows[method] = dict(
            available=bool(plan), feasible=audit.get("feasible", False),
            action_key=plan.get("action_key"), domain=plan.get("domain"),
            moves=audit.get("moves"), grid_energy_kwh=audit.get("grid_energy_kwh"),
            throughput_kwh=audit.get("throughput_kwh"),
            terminal_energy_kwh=energies[-1] if energies else None,
            travel_energy_kwh=audit.get("travel_energy_kwh"),
            conversion_loss_kwh=audit.get("conversion_loss_kwh"),
            driver_cost=audit.get("driver_cost"),
            max_violation=audit.get("max_violation"))
    event_rows = []
    for event in logs.get("events", []):
        milp = event.get("milp", {})
        event_rows.append(dict(
            iteration=event.get("iteration"), accepted=event.get("accepted", False),
            parent_score=event.get("parent_score"), proposed_score=event.get("proposed_score"),
            retained_score=event.get("retained_score"),
            complete_AC_objective=event.get("complete_AC_objective", False),
            failure=event.get("failure"), objective_failures=event.get("objective_failures"),
            solver_status=milp.get("solver_status"), success=milp.get("success"),
            surrogate_incumbent=milp.get("surrogate_incumbent"),
            surrogate_best_bound=milp.get("surrogate_best_bound"),
            surrogate_absolute_gap=milp.get("surrogate_absolute_gap"),
            solver_mip_gap=milp.get("solver_mip_gap"),
            bound_scope=milp.get("bound_scope")))
    noises, future_lengths, eval_counters, eval_seconds = [], [], [], []
    simulator_failures = 0
    for action in evaluation.get("actions", {}).values():
        for actual in action.get("regimes", {}).values():
            if actual.get("physical_generation_failed"):
                simulator_failures += 1
            if "actual_white_noise" in actual:
                noises.append(sha256(json.dumps(actual["actual_white_noise"], separators=(",", ":")).encode()).hexdigest())
            if "observations" in actual:
                future_lengths.append(len(actual["observations"]["p"]))
            for label in LABELS:
                fitted = actual.get(label + "_refit", {})
                eval_counters.append(fitted.get("counters", {}))
                eval_seconds.append(fitted.get("seconds"))
    support = planning.get("scenario_support", {}).get("support", {})
    history_counters = planning.get("history", {}).get("counters", {})
    support_counters = counter_sum(x.get("counters", {}) for x in support.values())
    objective_counters = costs.get("objective", {}).get("counters", {})
    unique_evaluation_counters = counter_sum(eval_counters)
    counters = counter_sum([history_counters, support_counters, objective_counters,
                            unique_evaluation_counters])
    terminal = number(service.get("terminal_energy_kwh"))
    service_equal = (all(plan_rows[m]["available"] and plan_rows[m]["feasible"] for m in METHODS)
        and terminal is not None
        and all(number(plan_rows[m]["terminal_energy_kwh"]) is not None
                and abs(plan_rows[m]["terminal_energy_kwh"] - terminal) <= 1e-6
                for m in METHODS))
    initial, final = number(logs.get("initial_score")), number(logs.get("final_score"))
    monotone = None if initial is None or final is None else final >= initial - 1e-9
    event_monotone = all(number(e["parent_score"]) is not None and
        number(e["retained_score"]) is not None and
        e["retained_score"] >= e["parent_score"] - 1e-9 for e in event_rows)
    block = planning.get("block_adapter", {})
    return dict(
        spec=record.get("spec", {}), status=record.get("status"),
        failures=record.get("failures", {}),
        gradient_gate=dict(pass_gate=bool(gate.get("pass_gate")), reason=gate.get("reason"),
            checked_planes=len(checks), failed_planes=sum(not c.get("passed") for c in checks),
            max_absolute_error=max((c["absolute_error"] for c in checks), default=None),
            max_error_over_tolerance=max((c["absolute_error"] / c["tolerance"]
                for c in checks if c.get("tolerance", 0) > 0), default=None),
            scheme=gate.get("direction", {}).get("scheme"), seconds=gate.get("seconds")),
        new_metric_design_failure=record.get("new_metric_design_failure", True),
        design_failure_fallback=record.get("new_metric_fallback"),
        joint_margin_same_plan_as_square=(plans.get("joint_margin", {}).get("action_key") is not None
            and plans.get("joint_margin", {}).get("action_key") == plans.get("SQMD", {}).get("action_key")),
        initial_score=initial, final_score=final,
        score_gain=None if initial is None or final is None else final - initial,
        same_full_AC_score_nonregression=monotone,
        retained_event_scores_nonregression=event_monotone,
        logged_nonregression=logs.get("inherited_square_not_worse"),
        accepted_proposals=sum(bool(e["accepted"]) for e in event_rows),
        proposal_events=event_rows, plans=plan_rows,
        task_checks=dict(same_terminal_service=service_equal,
            target_terminal_energy_kwh=terminal,
            same_future_meter_budget=bool(future_lengths) and all(
                n == record.get("config", {}).get("horizon") for n in future_lengths),
            observed_future_lengths=sorted(set(future_lengths)),
            actual_noise_hashes_equal=bool(noises) and len(set(noises)) == 1,
            actual_noise_paired_flag=evaluation.get("actual_noise_paired_across_plans_and_regimes"),
            clean_not_used_for_selection=evaluation.get("clean_used_for_selection") is False,
            bayes_center_not_used=evaluation.get("bayes_center_used") is False,
            same_SQ_JM_operation_domain=(plans.get("SQMD", {}).get("domain") == "mobile_bidir"
                and plans.get("joint_margin", {}).get("domain") == "mobile_bidir")),
        block_adapter=dict(adds_information_beyond_candidate_bank=block.get("adds_information_beyond_candidate_bank"),
            accessible_disputed_nodes=block.get("accessible_disputed_nodes"),
            n_directed_pairs=len(block.get("directed_pairs", [])),
            extra_likelihood=block.get("extra_likelihood")),
        sealed_forecast_diagnostics=saved_forecast_diagnostics(record, source_path),
        history_support={key: dict(retained=value.get("retained"),
            alternative_available=value.get("alternative_support_available"),
            attempted=len(value.get("attempts", []))) for key, value in support.items()},
        cost=dict(wall_seconds=record.get("seconds"),
            planning_seconds=costs.get("planning_seconds"), square_seconds=costs.get("square_seconds"),
            history_seconds=planning.get("history", {}).get("seconds"),
            margin_objective_seconds=costs.get("objective", {}).get("seconds"),
            margin_setup_seconds=costs.get("objective", {}).get("setup_seconds"),
            gradient_gate_seconds=gate.get("seconds"),
            objective_action_calls=costs.get("objective", {}).get("calls"),
            unique_evaluation_seconds=sum(v for v in eval_seconds if number(v) is not None),
            history_counters=history_counters, support_counters=support_counters,
            objective_counters=objective_counters, unique_evaluation_counters=unique_evaluation_counters,
            logged_counters=counters,
            logged_ac_calls_including_direct=counters.get("ac_calls", 0) + counters.get("direct_ac_calls", 0),
            unique_evaluated_actions=len(evaluation.get("actions", {})),
            method_action_references=sum(len(a.get("methods", [])) for a in evaluation.get("actions", {}).values()),
            simulator_failure_actions_regimes=simulator_failures,
            counter_scope="logged fitting/direct AC counters; simulator/case generation not fully instrumented; unique cached evaluation charged once"))


def phase_summary(records, manifests, pending, issues, freeze, phase):
    expected = len(freeze["config"][phase + "_seeds"])
    tasks = [task_row(r, method, regime, route, label) for r in records
             for method in METHODS for regime in REGIMES for route in ROUTES for label in LABELS]
    details = [history_details(record, Path(manifest["effective_path"]))
               for record, manifest in zip(records, manifests)]
    for detail in details:
        detail["ranking_diagnostics"] = ranking_diagnostics(detail, tasks)
    table = {method: {route: {label: summarize_cells(
        [r for r in tasks if (r["method"], r["route"], r["measurement"]) == (method, route, label)],
        expected, len(records)) for label in LABELS} for route in ROUTES} for method in METHODS}
    paired_planning = {route: {label: paired_outcomes(
        [r for r in tasks if (r["method"], r["route"], r["measurement"]) == ("SQMD", route, label)],
        [r for r in tasks if (r["method"], r["route"], r["measurement"]) == ("joint_margin", route, label)])
        for label in LABELS} for route in ROUTES}
    paired_inference = {method: {label: paired_outcomes(
        [r for r in tasks if (r["method"], r["route"], r["measurement"]) == (method, "fixed", label)],
        [r for r in tasks if (r["method"], r["route"], r["measurement"]) == (method, "joint", label)])
        for label in LABELS} for method in METHODS}
    gates = [d["gradient_gate"] for d in details]
    events = [e for d in details for e in d["proposal_events"]]
    nonregression = [d["same_full_AC_score_nonregression"] for d in details]
    known_regression = sum(v is False for v in nonregression)
    task_keys = ("same_terminal_service", "same_future_meter_budget", "actual_noise_hashes_equal",
                 "clean_not_used_for_selection", "bayes_center_not_used", "same_SQ_JM_operation_domain")
    summary = dict(
        phase=phase, expected_independent_histories=expected, written_independent_histories=len(records),
        pending_independent_histories=len(pending), statistical_complete=len(records) == expected,
        future_tasks_per_history=2, expected_tasks_per_method=2 * expected,
        record_status_counts=dict(Counter(r.get("status", "missing") for r in records)),
        integrity_pass=not issues, integrity_issues=issues,
        input_records=manifests, pending_records=pending,
        method_results=table,
        planning_paired_SQMD_to_joint_margin=paired_planning,
        inference_paired_fixed_to_joint=paired_inference,
        planning=dict(
            gradient_gate_pass=sum(g["pass_gate"] for g in gates),
            gradient_gate_fail=sum(not g["pass_gate"] for g in gates),
            design_failure_fallback_to_SQ=sum(d["design_failure_fallback"] == "SQMD" for d in details),
            same_plan_as_SQ=sum(d["joint_margin_same_plan_as_square"] for d in details),
            gradient_failed_plane_count=sum(g["failed_planes"] for g in gates),
            n_proposals=len(events), accepted_proposals=sum(e["accepted"] for e in events),
            accepted_histories=sum(d["accepted_proposals"] > 0 for d in details),
            complete_AC_objective_failed_proposals=sum(not e["complete_AC_objective"] for e in events),
            same_score_nonregression_pass=sum(v is True for v in nonregression),
            same_score_nonregression_fail=known_regression,
            same_score_nonregression_unavailable=sum(v is None for v in nonregression),
            event_score_regression=sum(not d["retained_event_scores_nonregression"] for d in details),
            score_gain=distribution(d["score_gain"] for d in details),
            surrogate_bound_scope="frozen affine margin MILP, not original AC global bound",
            surrogate_incumbent=distribution(e["surrogate_incumbent"] for e in events),
            surrogate_best_bound=distribution(e["surrogate_best_bound"] for e in events),
            surrogate_absolute_gap=distribution(e["surrogate_absolute_gap"] for e in events),
            solver_mip_gap=distribution(e["solver_mip_gap"] for e in events),
            solver_status_counts=dict(Counter(str(e["solver_status"]) for e in events))),
        movement={method: dict(available_plans=sum(d["plans"][method]["available"] for d in details),
            moved_histories=sum((d["plans"][method]["moves"] or 0) > 0 for d in details),
            moves=distribution(d["plans"][method]["moves"] for d in details)) for method in METHODS},
        task_consistency={key: dict(passed=sum(d["task_checks"][key] is True for d in details),
            failed_or_unavailable=sum(d["task_checks"][key] is not True for d in details)) for key in task_keys},
        cost={key: distribution(d["cost"][key] for d in details) for key in (
            "wall_seconds", "planning_seconds", "square_seconds", "history_seconds",
            "margin_objective_seconds", "margin_setup_seconds", "gradient_gate_seconds",
            "unique_evaluation_seconds", "logged_ac_calls_including_direct", "objective_action_calls")},
        logged_counters=counter_sum(d["cost"]["logged_counters"] for d in details),
        WZZT_adapter_extra_information_histories=sum(
            d["block_adapter"]["adds_information_beyond_candidate_bank"] is True for d in details),
        score_to_realized_margin=dict(
            all_reference_vs_true_reference_forecast=dict(Counter(
                d["ranking_diagnostics"]["aggregate_vs_true_reference_sign"] for d in details)),
            by_measurement={label: dict(
                aggregate_score_sign_counts=dict(Counter(
                    item["aggregate_score_vs_actual_sign"] for d in details
                    for item in d["ranking_diagnostics"]["realized"] if item["measurement"] == label)),
                true_reference_forecast_sign_counts=dict(Counter(
                    item["true_reference_forecast_vs_actual_sign"] for d in details
                    for item in d["ranking_diagnostics"]["realized"] if item["measurement"] == label)),
                actual_true_margin_delta=distribution(
                    item["actual_true_margin_delta"] for d in details
                    for item in d["ranking_diagnostics"]["realized"] if item["measurement"] == label))
                for label in LABELS},
            interpretation="Global S and realized true-pair margin are different targets. Posthoc true-reference forecasts help separate aggregation mismatch; residual sign disagreement also includes forecasting, support and local fitting error."),
        no_calibrated_rejection_or_probability_claim=True)
    return summary, dict(histories=details, tasks=tasks,
                         paired_planning=paired_planning, paired_inference=paired_inference)


def summarize(output, phases=("development", "evaluation"), prefix=""):
    output = Path(output).resolve()
    frozen = read_json(output / "freeze.json")
    recovery, recovery_info = recovery_mapping(output)
    summaries, histories = {}, {}
    for phase in phases:
        records, manifest, pending, issues = load_phase(output, phase, frozen, recovery)
        summaries[phase], histories[phase] = phase_summary(
            records, manifest, pending, issues, frozen, phase)
    summary = dict(schema="block_joint_margin_postprocess_v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        output_directory=str(output), freeze_identity=frozen["identity"],
        freeze_sha256=digest(output / "freeze.json"),
        postprocessor_sha256=digest(__file__),
        read_only_of_frozen_science=True, no_solver_simulation_or_policy_update=True,
        scientific_units="one independent history; two paired future regimes; clean is diagnostic only",
        clean_diagnostic_changes="new meter noise set to zero; historical noise, unknown lines and forecast variation remain",
        method_names=dict(constant="constant charge", SQMD="same-domain power-square MILP",
                          joint_margin="local joint-profile-margin price MILP with inherited SQ plan"),
        route_names=dict(fixed="historical R/X fixed; per-time source fitted", joint="shared R/X fitted on history and future"),
        recovery=recovery_info, phases=summaries,
        safeguards=["failure kept in written denominator", "pending is not failure",
                    "unresolved is numerical tie/failure, not calibrated confidence coverage",
                    "available-case error and all-written unresolved reported separately",
                    "no development/evaluation pooling for outcome claims",
                    "cache evaluation cost counted once per unique action",
                    "same score nonregression does not ensure true error nonregression",
                    "no comparison with old 4.69% across changed seeds/starts as paired evidence"])
    target = output / (prefix + "summary.json")
    paired = output / (prefix + "history_pairs.json")
    write_json(paired, dict(freeze_identity=frozen["identity"], phases=histories))
    summary["history_pairs"] = dict(path=str(paired), sha256=digest(paired))
    write_json(target, summary)
    return summary


def self_test():
    def row(index, regime, available, error):
        return dict(index=index, regime=regime, available=available, failed=not available,
                    error=error, unresolved=not available, true_margin_evaluation_only=None,
                    r_mean_absolute_relative=.2 if available else None,
                    x_mean_absolute_relative=.3 if available else None,
                    r_relative_l2=.2 if available else None, x_relative_l2=.3 if available else None)
    rows = [row(0, "regular", True, True), row(0, "shift", False, None)]
    cell = summarize_cells(rows, 2, 1)
    assert cell["failed_tasks"] == 1 and cell["pending_tasks"] == 2
    assert cell["error_tasks"] == 1 and cell["error_rate_available"] == 1
    assert cell["unresolved_rate_all_written"] == .5
    newer = [row(0, "regular", True, False), row(0, "shift", True, False)]
    paired = paired_outcomes(rows, newer)
    assert paired["new_wins"] == 1 and paired["base_failed"] == 1
    assert paired["n_complete_paired_histories"] == 0
    both = paired_outcomes(newer, newer)
    assert both["ties"] == 2 and both["n_complete_paired_histories"] == 1
    assert both["mean_error_delta_new_minus_base_complete_histories"] == 0
    print("postprocessor self-test passed: failures, pending, and paired history units")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--phase", choices=("all", "development", "evaluation"), default="all")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    phases = ("development", "evaluation") if args.phase == "all" else (args.phase,)
    result = summarize(args.output, phases, args.prefix)
    brief = {}
    for phase, value in result["phases"].items():
        brief[phase] = dict(written=value["written_independent_histories"],
            expected=value["expected_independent_histories"], integrity=value["integrity_pass"],
            gradient_fail=value["planning"]["gradient_gate_fail"],
            accepted=value["planning"]["accepted_proposals"],
            noisy_joint={m: {key: value["method_results"][m]["joint"]["noisy"][key]
                for key in ("error_tasks", "available_tasks", "failed_tasks", "pending_tasks")}
                for m in METHODS})
    print(json.dumps(brief, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
