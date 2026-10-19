"""Keep an audited executable baseline unless the frozen proxy improves.

This opt-in wrapper leaves the frozen C/D policy and original search intact.
It uses the existing operation MILP to realize a specified baseline control;
it never enumerates routes or power profiles. All bounds below concern the
same fixed affine proxy and ordinary floating-point solvers, not nonlinear
AC separation, statistical rejection, or expected information gain.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping
import numpy as np

from research_experiments.mobile_ev_mle.ev_operations_milp import (
    EVConfig, operation_model, decode_plan, check_plan,
)
from research_experiments.mobile_ev_mle.ev_information_milp import (
    FrozenPair, adaptive_search, solve_pair,
)

_NUMERICAL_ERRORS = (ValueError, RuntimeError, np.linalg.LinAlgError,
                     FloatingPointError)
_INPUT_ERRORS = _NUMERICAL_ERRORS + (TypeError, OverflowError)
_SEARCH_OPTIONS = {
    "initial_us", "max_iterations", "segments", "max_segments",
    "solver_seconds", "total_seconds", "absolute_gap", "relative_gap", "seed",
}
_SCOPE = "same frozen affine proxy; engineering floating-point bounds only"


def _positive(value, name, *, allow_zero=False):
    value = float(value)
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(name + " must be finite and " +
                         ("nonnegative" if allow_zero else "positive"))
    return value


def _control(config, value):
    if value is None:
        raise ValueError("baseline control is unspecified")
    u = np.asarray(value, dtype=float)
    shape = (config.n_slots, config.n_sites)
    if u.shape == (config.n_slots * config.n_sites,):
        u = u.reshape(shape)
    if u.shape != shape or not np.isfinite(u).all():
        raise ValueError("baseline requires a finite time-major control of shape " + str(shape))
    return np.array(u, copy=True)


def realize_baseline(config: EVConfig, baseline_u_kw, *, time_limit=10.0,
                     mip_rel_gap=1e-8, feasibility_tolerance=2e-6):
    """Fix u in the existing operation MILP and audit its actual incumbent.

    A feasible incumbent suffices: this is a feasibility problem, so solver
    optimality is not required. Requested and returned controls are retained
    separately and only the returned, audited control is scored or executed.
    """
    result = dict(success=False, status="baseline_input_failure", plan=None,
                  requested_u_kw=None, actual_u_kw=None, solver=None,
                  operation_audit=None, failures=[], enumerated_complete_actions=0)
    try:
        time_limit = _positive(time_limit, "baseline time limit")
        mip_rel_gap = _positive(mip_rel_gap, "baseline MIP gap", allow_zero=True)
        tolerance = _positive(feasibility_tolerance, "feasibility tolerance")
        result["settings"] = dict(time_limit=time_limit, mip_rel_gap=mip_rel_gap,
                                  feasibility_tolerance=tolerance)
        requested = _control(config, baseline_u_kw)
        result["requested_u_kw"] = requested.tolist()
        model, indices = operation_model(config)
        for index, value in zip(indices.u_index.ravel(), requested.ravel()):
            model.add_row({int(index): 1.0}, float(value), float(value))
        solved = model.solve(time_limit=time_limit, mip_rel_gap=mip_rel_gap)
        result["solver"] = dict(
            status=int(solved.status), success=bool(solved.success),
            message=str(solved.message), incumbent_available=solved.x is not None,
            mip_gap=getattr(solved, "mip_gap", None),
            objective_value=getattr(solved, "objective_value", None),
            objective_bound=getattr(solved, "objective_bound", None),
        )
        if solved.x is None:
            result["status"] = "baseline_infeasible_or_solver_failed"
            result["failures"].append(str(solved.message))
            return result
        plan = decode_plan(solved, indices)
        result["actual_u_kw"] = np.asarray(plan.u_kw).tolist()
        audit = check_plan(plan, config, tolerance=tolerance)
        mismatch = float(np.max(np.abs(plan.u_kw - requested), initial=0.0))
        result.update(operation_audit=audit, max_fixed_control_violation=mismatch)
        if not audit["feasible"] or mismatch > tolerance:
            result["status"] = "baseline_incumbent_audit_failed"
            result["failures"].append("baseline incumbent fails operation/fixed-control checks")
            return result
        result.update(success=True, status="baseline_realized", plan=plan)
    except _INPUT_ERRORS as error:
        result["failures"].append(str(error))
    return result


def _clone_pairs(pairs, dimension):
    pairs = list(pairs)
    if not pairs or any(not isinstance(p, FrozenPair) for p in pairs):
        raise ValueError("nonempty FrozenPair collection required")
    if len({p.name for p in pairs}) != len(pairs):
        raise ValueError("pair names must be unique")
    if any(p.control_dimension != dimension for p in pairs):
        raise ValueError("pair and operation control dimensions differ")
    # Existing QP acceleration writes internal caches. Use independent copies
    # so this wrapper also preserves the caller's metadata and solver caches.
    copied = [FrozenPair(p.name, p.c, p.B, p.A, p.F, p.lower, p.upper,
                         p.scale, deepcopy(p.metadata)) for p in pairs]
    return pairs, copied


def _score(pairs, u):
    rows, failures = [], []
    for pair in pairs:
        try:
            solved = solve_pair(pair, u)
            lo, hi = float(solved["q_lower"]), float(solved["q_upper"])
            valid = (bool(solved["success"]) and np.isfinite([lo, hi]).all()
                     and 0 <= lo <= hi)
            rows.append(dict(name=pair.name, valid=bool(valid), solution=solved))
            if not valid:
                failures.append(pair.name + ": QP failed or bounds inconsistent")
        except _NUMERICAL_ERRORS as error:
            rows.append(dict(name=pair.name, valid=False, error=str(error)))
            failures.append(pair.name + ": " + str(error))
    valid = not failures
    return dict(success=valid, rows=rows, failures=failures,
                q_lower=min(row["solution"]["q_lower"] for row in rows) if valid else None,
                q_upper=min(row["solution"]["q_upper"] for row in rows) if valid else None,
                scope=_SCOPE)


def guarded_adaptive_search(config: EVConfig, pairs, baseline_u_kw, *,
                            search_options: Mapping | None = None,
                            baseline_seconds=10.0, baseline_mip_rel_gap=1e-8,
                            feasibility_tolerance=2e-6, minimum_proxy_gain=1e-8):
    """Run the original MILP search, then compare audited plans fairly.

    Q(u)=min_k q_k(u). Independent point QP bounds give
    L_selected-U_baseline <= Q(selected)-Q(baseline)
                          <= U_selected-L_baseline.
    The default policy adopts the search plan only when its guarded lower
    gain exceeds minimum_proxy_gain; otherwise it returns the executable
    baseline and retains every search/point-evaluation failure. There is no
    exploration policy in this version. Future truth/noise are not API inputs.
    """
    answer = dict(success=False, status="guard_input_failure", plan=None,
                  selected_origin=None, guard_passed=False, fallback=False,
                  baseline=None, baseline_score=None, search=None,
                  search_operation_audit=None, search_score=None,
                  proxy_gain_interval=None, failures=[],
                  decision_policy="conservative_baseline_guard",
                  scope=_SCOPE, pair_hashes={},
                  selector_uses_truth=False, selector_uses_future_noise=False,
                  enumerated_complete_actions=0)
    try:
        margin = _positive(minimum_proxy_gain, "minimum proxy gain", allow_zero=True)
        tolerance = _positive(feasibility_tolerance, "feasibility tolerance")
        options = dict(search_options or {})
        unknown = set(options) - _SEARCH_OPTIONS
        if unknown:
            raise ValueError("unknown search options: " + ", ".join(sorted(unknown)))
        original, copied = _clone_pairs(pairs, config.n_slots * config.n_sites)
        answer["pair_hashes"] = {p.name: p.fingerprint() for p in original}
        answer.update(minimum_proxy_gain=margin, feasibility_tolerance=tolerance,
                      search_options=deepcopy(options))
    except _INPUT_ERRORS as error:
        answer["failures"].append(str(error))
        return answer

    baseline = realize_baseline(config, baseline_u_kw, time_limit=baseline_seconds,
        mip_rel_gap=baseline_mip_rel_gap, feasibility_tolerance=tolerance)
    answer["baseline"] = baseline
    if not baseline["success"]:
        answer.update(status="baseline_unavailable", failures=list(baseline["failures"]))
        return answer
    baseline_plan = baseline["plan"]
    answer.update(success=True, plan=baseline_plan, selected_origin="baseline", fallback=True)
    baseline_score = _score(copied, baseline_plan.u_kw)
    answer["baseline_score"] = baseline_score
    answer["failures"].extend("baseline: " + f for f in baseline_score["failures"])

    try:
        searched = adaptive_search(config, copied, **options)
        # Keep the entire object, including unsuccessful incumbents, raw bounds,
        # failures, solver records, cuts, timing and stopping status.
        answer["search"] = searched
    except _NUMERICAL_ERRORS as error:
        answer.update(status="baseline_search_exception")
        answer["failures"].append("search: " + str(error))
        return answer

    answer["failures"].extend("search: " + str(f) for f in searched.get("failures", []))
    search_plan = searched.get("plan")
    if search_plan is None:
        answer["status"] = "baseline_search_plan_unavailable"
        answer["failures"].append("search returned no executable plan")
        return answer
    try:
        audit = check_plan(search_plan, config, tolerance=tolerance)
        answer["search_operation_audit"] = audit
        if not audit["feasible"]:
            raise RuntimeError("search plan fails independent operation checks")
    except _NUMERICAL_ERRORS as error:
        answer["status"] = "baseline_search_plan_audit_failed"
        answer["failures"].append(str(error))
        return answer

    search_score = _score(copied, search_plan.u_kw)
    answer["search_score"] = search_score
    answer["failures"].extend("search point: " + f for f in search_score["failures"])
    if baseline_score["success"] and search_score["success"]:
        raw_lower = float(search_score["q_lower"] - baseline_score["q_upper"])
        raw_upper = float(search_score["q_upper"] - baseline_score["q_lower"])
        guard = 128 * np.finfo(float).eps * (1 + abs(search_score["q_lower"])
            + abs(search_score["q_upper"]) + abs(baseline_score["q_lower"])
            + abs(baseline_score["q_upper"]))
        interval = dict(raw_lower=raw_lower, raw_upper=raw_upper,
                        lower=raw_lower-guard, upper=raw_upper+guard,
                        subtraction_roundoff_guard=float(guard), scope=_SCOPE)
        answer["proxy_gain_interval"] = interval
        if bool(searched.get("success")) and not searched.get("failures") and interval["lower"] > margin:
            answer.update(status="search_proxy_gain_supported", plan=search_plan,
                          selected_origin="milp_search", guard_passed=True, fallback=False)
        else:
            answer["status"] = ("baseline_search_failed" if not searched.get("success") or searched.get("failures")
                                else "baseline_proxy_gain_unproven")
    else:
        answer["status"] = "baseline_point_evaluation_failed"
    if not bool(searched.get("success")) and not searched.get("failures"):
        answer["failures"].append("search reports failure: " + str(searched.get("status")))
    return answer
