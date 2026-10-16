"""Independent saved-state and plan-inheritance audit for the v4 study.

This module calls no optimizer, selector, simulator generator, or production AC
predictor. AC states are replayed using the older independent branch-current
sweep. Predictive caches receive loss/aggregation checks; their nonlinear AC
states are outside this first audit's scope. Missing fixed-source states remain
explicit, even when their saved residual arithmetic can still be checked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import ev_mobile_v2_review as old
from . import ev_mobile_v4_domain as nested
from .ev_mobile_service_domain import MobileServiceConfig
from .likelihood import RadialCandidate
from . import gradient_price_study as io


HERE = Path(__file__).resolve().parent
FAMILIES = ("square", "aligned", "guarded")
METHODS = tuple(family + "__" + domain for family in FAMILIES for domain in nested.DOMAIN_FLAGS)
EXCEPTIONS = (ValueError, KeyError, TypeError, IndexError, AssertionError,
              FloatingPointError, np.linalg.LinAlgError)


def _bank(record):
    bank = record["spec"]["bank"]
    if bank == "n3":
        n, families = 3, ((), ((0, 1),), ((0, 2),), ((1, 2),))
    elif bank == "n4":
        n, families = 4, ((), ((0, 1),), ((2, 3),), ((0, 1), (2, 3)))
    else:
        raise ValueError("unknown predeclared physical bank")
    return {("star" if not family else "clade_" + "_".join("".join(map(str, s)) for s in family)):
        RadialCandidate.from_supports(
            "star" if not family else "clade_" + "_".join("".join(map(str, s)) for s in family), n, family)
        for family in families}


def _close(a, b, label, *, atol=2e-7, rtol=2e-8):
    old.close(a, b, label, atol=atol, rtol=rtol)


def _require(condition, label):
    old.require(condition, label)


def _gaps(profile, names):
    rows = profile.get("nearest", [])
    _require(len(rows) == len(names) and {r["a"] for r in rows} == set(names), "one nearest gap per reference required")
    result = {r["a"]: float(r["half_gap"]) for r in rows}
    _require(np.isfinite(list(result.values())).all(), "nearest gaps must be finite")
    return result


def check_predictive_profile(profile, names, *, key=None, objective_id=None):
    """Recompute all directed signed half-SSE gaps and mean-nearest score."""
    if key is not None:
        _require(nested.action_key(profile["power_kw"]) == key, "predictive action identity mismatch")
    if objective_id is not None:
        _require(profile.get("objective_id") == objective_id, "different common AC objective")
    for label in ("uses_truth", "uses_future_noise"):
        _require(profile.get(label) is False, "predictive cache must exclude " + label)
    _require(profile.get("source_interface") == 1, "predictive source must use interface1")
    if not profile.get("success"):
        return dict(success=False, failure="recorded incomplete predictive profile", pair_checks=0)
    _require(profile.get("uses_observed_D") is True, "predictive cache must use complete observed D")
    references = profile.get("references", [])
    _require(len(references) == len(names) and {r["a"] for r in references} == set(names),
             "all reference explanations required")
    by_reference = {r["a"]: r for r in references}
    for reference in references:
        _require(reference.get("success") is True, "successful profile has failed reference")
        _require(set(reference.get("fits", {})) == set(names), "K by K explanatory fits required")
        for fitted in reference["fits"].values():
            _require(fitted.get("success") is True and fitted.get("gradient_reliable") is True,
                     "successful profile has failed explanatory fit")
            _require(np.isfinite(float(fitted["loss"])), "finite explanatory loss required")
    pairs = profile.get("pairs", [])
    expected = {(a, b) for a in names for b in names if a != b}
    _require(len(pairs) == len(expected) and {(p["a"], p["b"]) for p in pairs} == expected,
             "all directed pairs exactly once required")
    values = {}
    for pair in pairs:
        a, b = pair["a"], pair["b"]
        own = float(by_reference[a]["fits"][a]["loss"])
        rival = float(by_reference[a]["fits"][b]["loss"])
        _require(pair.get("success") is True, "successful profile has failed pair")
        _close(pair["own_loss"], own, "pair own loss")
        _close(pair["rival_loss"], rival, "pair rival loss")
        _close(pair["half_gap"], .5 * (rival - own), "signed joint half-SSE gap")
        values[(a, b)] = .5 * (rival - own)
    gaps = _gaps(profile, names)
    aggregate = profile["aggregation"]
    _require(aggregate.get("success") is True, "successful profile must aggregate all tasks")
    aggregate_gaps = _gaps(aggregate, names)
    for a in names:
        nearest = min(values[(a, b)] for b in names if b != a)
        _close(gaps[a], nearest, "nearest competitor for " + a)
        _close(aggregate_gaps[a], nearest, "aggregate nearest competitor for " + a)
    mean = float(np.mean(list(gaps.values())))
    _close(profile["score"], mean, "mean nearest signed gap")
    _close(aggregate["score"], mean, "aggregate mean nearest signed gap")
    return dict(success=True, score=mean, pair_checks=len(pairs), reference_count=len(names))


def check_reference_guard(saved, before, after, names):
    """Recompute a guard relative to one selected anchor, not mixed parents."""
    b, a = _gaps(before, names), _gaps(after, names)
    tolerance = float(saved["tolerance"])
    _require(np.isfinite(tolerance) and tolerance >= 0., "invalid guard tolerance")
    _require(set(saved["expected_references"]) == set(names)
             and len(saved["expected_references"]) == len(names), "guard reference coverage")
    violated = []
    for name in names:
        slack = a[name] - b[name]
        _close(saved["slacks"][name], slack, "guard slack " + name)
        _close(saved["floor"][name], b[name], "guard selected-anchor floor " + name)
        if slack < -tolerance * (1. + abs(b[name])):
            violated.append(name)
    _require(set(saved["violated"]) == set(violated), "guard violated-reference list")
    _require(saved.get("complete") is True, "complete profiles require complete guard")
    _require(bool(saved["success"]) == (not violated), "guard acceptance flag")
    return not violated


def _profile_identity(profile):
    """Ignore wall times when two cold families reproduce the same objective."""
    references = []
    for reference in profile.get("references", []):
        fits = {name: {field: row.get(field) for field in
            ("success", "loss", "state", "gradient_per_kw", "gradient_reliable", "failure")}
            for name, row in reference.get("fits", {}).items()}
        references.append(dict(a=reference.get("a"), success=reference.get("success"), fits=fits,
                               failure=reference.get("failure")))
    selected = {field: profile.get(field) for field in
        ("objective_id", "power_kw", "success", "score", "nearest", "pairs")}
    selected["references"] = references
    return io.identity(io.serial(selected), {})


def check_family(family, rows, methods, cache, names, objective_id):
    """Replay parent-first full-AC acceptance from the saved log only."""
    guard = family == "guarded"
    checks, selected = 0, {}
    for domain in nested.DOMAIN_FLAGS:
        log = rows[domain]
        _require(bool(log["guard"]) == guard, "family guard flag")
        expected_parents = [p for p in nested.PARENTS[domain] if p in selected]
        parents = log["direct_parents"]
        _require([p["domain"] for p in parents] == expected_parents, "direct parent coverage/order")
        current = None
        for parent in parents:
            saved_parent = selected[parent["domain"]]
            _require(parent["action_key"] == saved_parent["action_key"], "parent action replaced")
            _close(parent["score"], saved_parent["score"], "parent common score replaced")
            profile = cache[parent["action_key"]]
            _require(profile.get("objective_id") == objective_id, "parent objective changed")
            if current is None or profile["score"] > current["score"]:
                current = profile
        for initial in log["initials"]:
            profile = cache[initial["action_key"]]
            _close(initial["score"], profile["score"], "initializer AC score")
            floor_ok = True
            if current is not None:
                floor_ok = check_reference_guard(initial["reference_floor"], current, profile, names)
                checks += 1
            else:
                _require(initial["reference_floor"] is None, "first initializer has no prior floor")
            # Dev2 logs include the per-domain quota and use the same relative
            # numerical threshold as proposal acceptance. Dev1 had only 1e-9.
            relative = 1e-10 if "domain_soft_quota_seconds" in log else 0.
            take = current is None or (profile["score"] > current["score"] + 1e-9
                + relative * abs(current["score"]) and (not guard or floor_ok))
            _require(bool(initial["accepted"]) == take, "initializer acceptance differs from common AC rule")
            if take:
                current = profile
        if current is None:
            _require(bool(log.get("planning_failure")), "missing initializer requires explicit planning failure")
            continue
        if "parent_action" in log:
            _require(log["parent_action"] == nested.action_key(current["power_kw"]), "proposal anchor action")
            _close(log["parent_score"], current["score"], "proposal anchor score")
        else:
            _require(bool(log.get("planning_failure")) and not log["steps"],
                     "missing proposal anchor must be an explicit early planning failure")
        for step in log["steps"]:
            if "AC_score" not in step:
                _require(not step.get("accepted"), "proposal cannot be accepted without complete AC score")
                continue
            proposed = cache[step["proposed_action"]]
            _close(step["previous_AC_score"], current["score"], "proposal previous AC score")
            _close(step["AC_score"], proposed["score"], "proposal complete AC score")
            floor_ok = check_reference_guard(step["reference_floor"], current, proposed, names)
            checks += 1
            improves = proposed["score"] > current["score"] + 1e-9 + 1e-10 * abs(current["score"])
            _require(bool(step["mean_improves"]) == improves, "proposal mean-improvement flag")
            take = improves and (not guard or floor_ok)
            _require(bool(step["accepted"]) == take, "proposal acceptance differs from common AC rule")
            if take:
                current = proposed
        method = methods[family + "__" + domain]
        _require(method["action_key"] == log["selected_action"] == nested.action_key(current["power_kw"]),
                 "selected method differs from accepted full path")
        _close(log["selected_score"], current["score"], "selected full AC score")
        _close(method["nominal"]["score"], current["score"], "reported nominal score")
        selected_gaps = _gaps(dict(nearest=log["selected_reference_gaps"]), names)
        current_gaps = _gaps(current, names)
        for name in names:
            _close(selected_gaps[name], current_gaps[name], "selected reference gap")
        parent_best = max((p["score"] for p in parents), default=None)
        if parent_best is not None:
            _close(log["parent_best_score"], parent_best, "best-parent score")
            _require(current["score"] >= parent_best - 1e-9, "same full AC objective fell below parent")
        else:
            _require(log["parent_best_score"] is None, "root domain has no parent score")
        _require(log.get("scalar_monotonicity_verified") is True, "parent monotonicity flag")
        selected[domain] = dict(action_key=method["action_key"], score=current["score"])
    return dict(domain_checks=len(selected), reference_guard_checks=checks)


def _training_predictions(saved, observation, meter, actual):
    T, n = np.asarray(observation["p"]).shape
    for name in ("p", "q"):
        calibrated = np.asarray(observation[name]) - old._array(meter["bias_" + name], (T, n))
        _close(saved[name], calibrated, name + " is observed input once", atol=0., rtol=0.)
        _close(saved["predicted"][name], observation[name], name + " measurement prediction", atol=2e-10, rtol=0.)
    residual = actual["blocks"]["observed"]["residual"]
    start = 0
    for name, width in (("v", n), ("master_p", 1), ("master_q", 1)):
        shape = (T, n) if name == "v" else (T,)
        white = residual[:, start:start + width].reshape(shape)
        start += width
        measurement = np.asarray(observation[name]) + white * old._array(meter[name + "_sigma"], shape)
        _close(saved["predicted"][name], measurement, name + " saved measurement prediction", atol=2e-10, rtol=0.)


def _score_arithmetic(saved, blocks, n):
    """Check score fields without reconstructing missing source coordinates."""
    total, normalizer = 0., 0.
    _require(set(saved["blocks"]) == {label for label, _, _ in blocks}, "score block coverage")
    for label, obs, meter in blocks:
        T = len(obs["p"])
        row = saved["blocks"][label]
        residual = np.asarray(row["residual"], float)
        _require(residual.shape == (T, n + 2) and np.isfinite(residual).all(), "residual block dimensions")
        loss = float(np.sum(residual ** 2))
        _close(row["loss"], loss, "saved block residual SSE")
        _require(int(row["time_count"]) == T, "block time count")
        for name, part in (("v", residual[:, :n]), ("master_p", residual[:, n]), ("master_q", residual[:, n + 1])):
            _close(row["channel_loss"][name], np.sum(part ** 2), "saved channel residual SSE")
            shape = (T, n) if name == "v" else (T,)
            sigma = old._array(meter[name + "_sigma"], shape)
            _require(np.all(sigma > 0.), "positive meter precision")
            normalizer += float(np.sum(np.log(2. * np.pi * sigma ** 2)))
        total += loss
    _close(saved["loss"], total, "sum of D and future SSE")
    _close(saved["normalizer"], normalizer, "D and future normalizer once")
    _close(saved["loglik"], -.5 * (total + normalizer), "conditional Gaussian log likelihood")
    return total


def audit_record(record, *, expected_methods=METHODS):
    began = perf_counter()
    errors, missing, failures, states, method_rows = [], [], [], [], []
    metrics = dict(saved_states_replayed=0, AC_block_replays=0, max_whitened_residual_error=0.,
        max_block_SSE_error=0., max_total_SSE_error=0., max_AC_equation=0., max_power_balance=0.,
        operation_checks=0, max_operation_violation=0., training_states=0,
        actual_joint_unique_states=0, actual_fixed_unique_states=0, fixed_missing_state_slots=0,
        fixed_residual_arithmetic_checks=0, predictive_profiles_checked=0, predictive_pairs_checked=0,
        reference_guard_checks=0, family_domain_checks=0, square_parent_checks=0,
        actual_method_candidate_slots=0, recorded_joint_fit_failure_slots=0,
        recorded_fixed_source_failure_slots=0,
        recorded_inference_failure_method_slots=0, recorded_planning_failure_slots=0,
        recorded_operation_failure_slots=0, PQ_once_checks=0)

    def attempt(label, work):
        try:
            return work()
        except EXCEPTIONS as exc:
            errors.append(dict(check=label, reason=type(exc).__name__ + ": " + str(exc)))
            return None

    def absent(label, reason="saved state unavailable"):
        missing.append(dict(check=label, reason=reason))

    def replay(label, candidate, saved, state, blocks):
        actual = old.replay_state(candidate, state, blocks, record["config"])
        old._compare_score(saved, actual, label, metrics, selected=bool(saved.get("success", saved.get("usable", True))))
        metrics["max_total_SSE_error"] = max(metrics["max_total_SSE_error"], abs(float(saved["loss"]) - actual["loss"]))
        states.append(dict(check=label, recorded_fit_success=saved.get("success"),
            saved_loss=saved["loss"], replay_loss=actual["loss"],
            line_domain_feasible=actual["line_domain_feasible"], source_box_feasible=actual["source_box_feasible"],
            physical_voltage_feasible=actual["physical_voltage_feasible"]))
        return actual

    bank = attempt("candidate bank", lambda: _bank(record))
    names = list(bank) if bank else ["missing_" + str(i) for i in range(4)]
    regimes = list(record.get("spec", {}).get("regimes", []))
    expected = dict(method_slots=len(expected_methods), training_candidate_states=len(names),
        future_regime_slots=len(regimes), actual_method_slots=len(expected_methods) * len(regimes),
        actual_method_candidate_slots=len(expected_methods) * len(regimes) * len(names))
    required = ("observed", "observed_meter", "future_meter", "training_result", "config", "operation_config")
    for key in required:
        if key not in record:
            absent(key, record.get("failure", record.get("status", "unavailable")))
    if missing or not bank:
        return dict(success=False, available_checks_passed=not errors, complete_physical_coverage=False,
            spec=record.get("spec"), recorded_status=record.get("status"), recorded_failure=record.get("failure"),
            expected=expected, metrics=metrics, errors=errors, missing=missing, recorded_failures=failures,
            states=states, methods=method_rows, seconds=perf_counter() - began)
    D, dm, fm, cfg = record["observed"], record["observed_meter"], record["future_meter"], record["config"]
    n = len(D["p"][0]); T = int(record["spec"]["T"])
    attempt("source interface", lambda: _require(cfg.get("source_interface") == 1, "fixed physical source interface1"))
    attempt("candidate dimensions", lambda: _require(all(c.n == n for c in bank.values()), "candidate/meter dimensions"))
    observed_blocks = (("observed", D, dm),)
    trained = {r.get("candidate", r.get("name")): r for r in record["training_result"].get("fits", [])}
    for name, candidate in bank.items():
        saved = trained.get(name)
        if saved is None:
            absent("observed_fit/" + name, str(record["training_result"].get("failures", {})))
            continue
        if not saved.get("success"):
            failures.append(dict(kind="training_fit_failure", candidate=name))
        def training_work(c=candidate, row=saved):
            _require(tuple(map(tuple, row["supports"])) == c.supports, "training edge ordering")
            actual = replay("observed_fit/" + c.name, c, row, row, observed_blocks)
            _training_predictions(row, D, dm, actual)
            metrics["training_states"] += 1
        attempt("observed_fit/" + name, training_work)

    domains = attempt("operation domains", lambda: nested.build_nested_domains(MobileServiceConfig(**record["operation_config"])))
    methods = record.get("methods", {})
    attempt("method coverage", lambda: _require(set(methods) == set(expected_methods), "all predeclared method slots required"))
    by_action = {}
    for name in expected_methods:
        method = methods.get(name)
        method_rows.append(dict(name=name, present=method is not None,
            planning_success=False if method is None else method.get("planning_success")))
        if method is None:
            absent("method/" + name, record.get("failure", "method unavailable")); continue
        if method.get("planning_failure") or method.get("planning_success") is False:
            metrics["recorded_planning_failure_slots"] += 1
            failures.append(dict(kind="planning_failure", method=name, reason=method.get("planning_failure")))
        domain = method["domain"]
        attempt("method family/domain/" + name, lambda: _require(name == method["family"] + "__" + domain, "method label mismatch"))
        def operation_work(row=method):
            _require(domains is not None and row["domain"] in domains, "valid saved operation domain required")
            physical = domains[row["domain"]].check(np.asarray(row["operation_x"], float))
            _require(physical["feasible"], "saved full operation is physically infeasible")
            for field in ("u_kw", "energy_kwh", "soc", "moves", "travel_energy_kwh", "driver_cost", "grid_energy_kwh", "throughput_kwh"):
                _close(row["operation"][field], physical[field], "operation replay/" + field, atol=1e-7, rtol=0.)
            _require(row["operation"]["route"] == physical["route"], "full route timing changed")
            _close(row["power_kw"], physical["u_kw"], "full x to executed powers", atol=1e-7, rtol=0.)
            key = nested.action_key(row["power_kw"])
            _require(key == row["action_key"], "method power identity")
            metrics["operation_checks"] += 1
            metrics["max_operation_violation"] = max(metrics["max_operation_violation"], physical["max_violation"])
            return key
        key = attempt("operation/" + name, operation_work)
        if key is None:
            metrics["recorded_operation_failure_slots"] += 1
        else:
            by_action.setdefault(key, np.asarray(method["power_kw"], float))

    all_profiles = []
    objective_ids = set()
    for family in ("aligned", "guarded"):
        row = record.get("families", {}).get(family)
        if row is None:
            absent("planning_family/" + family); continue
        objective_ids.add(row["objective_id"])
        for key, profile in row.get("cache", {}).items():
            all_profiles.append((family, key, profile, row["objective_id"]))
    if "reporting_cache" not in record:
        absent("reporting_cache")
    else:
        objective_id = next(iter(objective_ids)) if len(objective_ids) == 1 else None
        all_profiles.extend(("reporting", key, profile, objective_id) for key, profile in record["reporting_cache"].items())
    attempt("same AC objective across domains/families", lambda: _require(len(objective_ids) == 1, "families use different common objectives"))
    checked_profiles = {}
    for family, key, profile, identity in all_profiles:
        label = "predictive/" + family + "/" + key
        if not profile.get("success"):
            failures.append(dict(kind="predictive_fit_failure", family=family, action_key=key))
        digest = _profile_identity(profile)
        if key in checked_profiles:
            attempt(label + "/cache reuse", lambda: _require(checked_profiles[key] == digest, "same action has different cached objective evidence"))
            continue
        checked_profiles[key] = digest
        check = attempt(label, lambda p=profile, k=key, obj=identity: check_predictive_profile(p, names, key=k, objective_id=obj))
        if check is not None:
            metrics["predictive_profiles_checked"] += 1
            metrics["predictive_pairs_checked"] += check["pair_checks"]
    for family in ("aligned", "guarded"):
        row = record.get("families", {}).get(family)
        if row is None:
            continue
        check = attempt("parent acceptance/" + family, lambda f=family, r=row:
            check_family(f, r["logs"], methods, r["cache"], names, r["objective_id"]))
        if check is not None:
            metrics["family_domain_checks"] += check["domain_checks"]
            metrics["reference_guard_checks"] += check["reference_guard_checks"]
    def square_work():
        logs = record["families"]["square"]["logs"]
        sigma = old._array(fm["v_sigma"], (T, n))
        minimum = float(sigma.min())
        weights = (minimum / sigma) ** 2 if minimum <= 1. else (1. / sigma) ** 2
        for domain in nested.DOMAIN_FLAGS:
            method = methods["square__" + domain]
            Q = float(np.sum(weights * np.asarray(method["power_kw"]) ** 2))
            _close(logs[domain]["Q"], Q, "square exact physical-power objective")
            parents = [float(np.sum(weights * np.asarray(methods["square__" + p]["power_kw"]) ** 2))
                       for p in nested.PARENTS[domain]]
            _close(logs[domain]["parent_Q"], parents, "square parent objective list")
            _require(all(Q >= value - 1e-9 for value in parents), "square objective decreased from legal parent")
            _require(logs[domain]["scalar_monotonicity_verified"] is True, "square monotonicity flag")
            metrics["square_parent_checks"] += len(parents)
    attempt("square-family parent inheritance", square_work)
    for name, method in methods.items():
        def nominal_work(row=method):
            profile = record["reporting_cache"][row["action_key"]]
            _require(bool(row["nominal"]["success"]) == bool(profile["success"]), "nominal complete flag")
            if profile["success"]:
                _close(row["nominal"]["score"], profile["score"], "method nominal common objective")
                saved, expected_gaps = _gaps(row["nominal"], names), _gaps(profile, names)
                for reference in names:
                    _close(saved[reference], expected_gaps[reference], "method nominal per-reference gap")
        attempt("nominal/" + name, nominal_work)

    actual = record.get("actual", {})
    attempt("future regime coverage", lambda: _require(set(actual) == set(regimes), "all declared future regimes required"))
    for regime in regimes:
        if regime not in actual:
            absent("actual/" + regime); continue
        actions = actual[regime].get("actions", {})
        attempt("action coverage/" + regime, lambda: _require(set(actions) == set(by_action), "actual action cache differs from executable method actions"))
        for name in expected_methods:
            metrics["actual_method_candidate_slots"] += len(names)
            method = methods.get(name)
            if method is None or method.get("action_key") not in actions:
                absent("actual_method/" + regime + "/" + name); continue
            evaluation = actions[method["action_key"]]["evaluation"]
            metrics["recorded_inference_failure_method_slots"] += not bool(evaluation.get("complete"))
            for candidate_name in names:
                comparison = evaluation.get("comparisons", {}).get(candidate_name, {})
                row = comparison.get("joint", {})
                metrics["recorded_joint_fit_failure_slots"] += not bool(row.get("success"))
                metrics["recorded_fixed_source_failure_slots"] += not bool(comparison.get("fixed_source_success"))
        paired = None
        for key, action in actions.items():
            if key not in by_action:
                continue
            future, evaluation = action["future"], action["evaluation"]
            blocks = (*observed_blocks, ("future", future, fm))
            power = by_action[key]
            baseline = dict(p=np.asarray(future["p"]) - power / float(cfg["base_kw"]),
                            q=np.asarray(future["q"]), root_v=np.asarray(future["root_v"]))
            if paired is None:
                paired = baseline
            else:
                def pq_work():
                    for channel in baseline:
                        _close(baseline[channel], paired[channel], "paired background / EV once " + channel,
                               atol=3e-15, rtol=2e-13)
                    metrics["PQ_once_checks"] += 1
                attempt("EV once/" + regime + "/" + key, pq_work)
            losses, complete = {}, len(evaluation.get("comparisons", {})) == len(names)
            for name, candidate in bank.items():
                comparison = evaluation.get("comparisons", {}).get(name)
                if comparison is None:
                    reason = str(evaluation.get("failures", {}).get(name, "candidate unavailable"))
                    absent("actual/" + regime + "/" + key + "/" + name,
                           reason)
                    failures.append(dict(kind="actual_candidate_exception", regime=regime,
                                         action_key=key, candidate=name, reason=reason))
                    complete = False; continue
                fixed, joint = comparison.get("fixed"), comparison.get("joint")
                if not comparison.get("fixed_source_success"):
                    failures.append(dict(kind="actual_fixed_source_fit_failure", regime=regime,
                                         action_key=key, candidate=name))
                if fixed is None:
                    absent("actual_fixed/" + regime + "/" + key + "/" + name)
                else:
                    def fixed_work(row=fixed, c=candidate):
                        _score_arithmetic(row, blocks, n)
                        metrics["fixed_residual_arithmetic_checks"] += 1
                        observed = row["blocks"]["observed"]
                        _close(observed["loss"], trained[c.name]["loss"], "fixed observed D appears once")
                        if row.get("state") is not None:
                            for line in ("r", "x"):
                                _close(row["state"][line], trained[c.name][line], "fixed trained line " + line, atol=0., rtol=0.)
                            for channel in ("source_p", "source_q"):
                                _close(np.asarray(row["state"][channel])[:len(D["p"])], trained[c.name][channel],
                                       "fixed observed source " + channel, atol=0., rtol=0.)
                            replay("actual_fixed/" + regime + "/" + key + "/" + c.name,
                                   c, row, row["state"], blocks)
                            metrics["actual_fixed_unique_states"] += 1
                        else:
                            metrics["fixed_missing_state_slots"] += 1
                            absent("actual_fixed_state/" + regime + "/" + key + "/" + c.name,
                                   "fixed-source coordinates not serialized; residual arithmetic only")
                    attempt("fixed score/" + regime + "/" + key + "/" + name, fixed_work)
                if joint is None or joint.get("state") is None:
                    absent("actual_joint_state/" + regime + "/" + key + "/" + name)
                    complete = False; continue
                if not joint.get("success"):
                    failures.append(dict(kind="actual_joint_fit_failure", regime=regime, action_key=key, candidate=name,
                        selected_origin=joint.get("selected_origin"), optimizer_error=joint.get("optimizer_error")))
                complete = complete and bool(joint.get("success"))
                def joint_work(row=joint, c=candidate, fixed_score=fixed):
                    _score_arithmetic(row, blocks, n)
                    checked = replay("actual_joint/" + regime + "/" + key + "/" + c.name,
                                     c, row, row["state"], blocks)
                    metrics["actual_joint_unique_states"] += 1
                    losses[c.name] = checked["loss"]
                    _close(evaluation["losses"][c.name], checked["loss"], "actual reported profile loss")
                    if fixed_score is not None:
                        _require(row["loss"] <= fixed_score["loss"] + 1e-8 * (1. + abs(fixed_score["loss"])),
                                 "retained joint state worse than its fixed feasible witness")
                        _close(row["feasible_witness"]["loss"], fixed_score["loss"], "same fixed feasible witness loss")
                attempt("joint score/" + regime + "/" + key + "/" + name, joint_work)
            attempt("actual completeness/" + regime + "/" + key, lambda: _require(bool(evaluation["complete"]) == complete,
                    "actual completeness flag differs from all required candidate optimizers"))
            if complete and len(losses) == len(names) and evaluation.get("M") is not None:
                true_name = record["true_name"]  # Saved-loss arithmetic only; no selection.
                M = min(value for name, value in losses.items() if name != true_name) - losses[true_name]
                attempt("actual M/" + regime + "/" + key, lambda: _close(evaluation["M"], M, "saved actual margin"))
    metrics["unique_actual_actions"] = sum(len(row.get("actions", {})) for row in actual.values())
    joint_expected = len(names) * metrics["unique_actual_actions"]
    return dict(success=not errors and not missing, available_checks_passed=not errors,
        complete_physical_coverage=not missing,
        complete_joint_physical_coverage=metrics["actual_joint_unique_states"] == joint_expected
            and metrics["training_states"] == len(names),
        spec=record.get("spec"), recorded_status=record.get("status"), expected=expected,
        metrics=metrics, errors=errors, missing=missing, recorded_failures=failures,
        states=states, methods=method_rows, seconds=perf_counter() - began,
        optimizer_calls=0, production_AC_calls=0, new_simulation_calls=0,
        predictive_scope="cache pair/nearest/mean and complete-history acceptance arithmetic; no predictive AC state replay",
        physical_scope="independent branch-current replay of all available retained observed, actual fixed and actual joint states",
        guarantee_scope="saved-state consistency and legal parent inheritance; no local/global optimum or statistical calibration certificate")


def run(directory, output, *, phase="development"):
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError(output)
    frozen = old.read(directory / "freeze.json")
    expected = frozen["specifications"][phase]
    rows = []
    for file in sorted((directory / phase).glob("*.json.gz")):
        record = old.read(file)
        try:
            _require(record.get("policy_id") == frozen["policy_id"], "record/freeze policy mismatch")
            audit = audit_record(record)
        except EXCEPTIONS as exc:
            audit = dict(success=False, available_checks_passed=False, spec=record.get("spec"),
                errors=[dict(check="record", reason=type(exc).__name__ + ": " + str(exc))], missing=[],
                metrics={}, recorded_failures=[])
        audit.update(input_file=str(file.resolve()), input_sha256=sha256(file.read_bytes()).hexdigest())
        rows.append(audit)
        print("AUDIT", file.name, audit["success"], "errors", len(audit["errors"]),
              "missing", len(audit["missing"]), flush=True)
    identity = lambda value: io.identity(io.serial(value), {})
    expected_ids = {identity(spec) for spec in expected}
    actual_ids = [identity(row.get("spec")) for row in rows]
    coverage = len(actual_ids) == len(set(actual_ids)) and set(actual_ids) == expected_ids
    totals, maxima = Counter(), {}
    for row in rows:
        for key, value in row.get("metrics", {}).items():
            if key.startswith("max_"):
                maxima[key] = max(maxima.get(key, 0.), value)
            else:
                totals[key] += value
    source_status = {}
    for name, expected_hash in frozen["source_sha256"].items():
        file = HERE.parents[1] / name if name.startswith("rnj_wzzt_core/") else HERE / name
        source_status[name] = file.is_file() and sha256(file.read_bytes()).hexdigest() == expected_hash
    result = dict(success=coverage and all(row["success"] for row in rows) and all(source_status.values()),
        available_checks_passed=all(row.get("available_checks_passed") for row in rows),
        phase=phase, policy_id=frozen["policy_id"], exact_case_coverage=coverage,
        expected_case_slots=len(expected), saved_case_slots=len(rows), case_results=rows,
        totals=dict(totals), maxima=maxima,
        recorded_failure_counts=dict(Counter(failure["kind"] for row in rows for failure in row["recorded_failures"])),
        all_frozen_scientific_sources_unchanged=all(source_status.values()), source_hash_checks=source_status,
        original_source_sha256=frozen["source_sha256"],
        input_freeze_sha256=sha256((directory / "freeze.json").read_bytes()).hexdigest(),
        audit_source_sha256={file.name: sha256(file.read_bytes()).hexdigest() for file in
            (Path(__file__), HERE / "ev_mobile_v2_review.py", HERE / "ev_original_review.py", HERE / "ev_mobile_v4_domain.py")},
        optimizer_calls=0, production_AC_calls=0, new_simulation_calls=0,
        denominator="independent history and all predeclared method/regime/candidate slots retained; only repeated actions deduplicated")
    old.write(output, io.serial(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("development", "evaluation"), default="development")
    args = parser.parse_args()
    result = run(args.directory, args.output, phase=args.phase)
    print(json.dumps({key: result[key] for key in
        ("success", "available_checks_passed", "expected_case_slots", "saved_case_slots", "totals", "maxima")}))


if __name__ == "__main__":
    main()
