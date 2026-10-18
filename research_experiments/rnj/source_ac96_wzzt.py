"""Auditable, zero-intercept WZZT search for the one-day Source comparison.

The core Source builders jointly refit every atom coefficient and the entire
unknown source path in each solve.  This experiment wrapper adds a bounded
forward search, independent prediction/feasibility checks and checkpoints;
it does not alter core estimators or claim a global topology optimum.
"""

from __future__ import annotations

from time import perf_counter
from typing import Callable, Iterable, Sequence

import numpy as np

from rnj_wzzt.estimation import laminar_l1_milp as core
from rnj_wzzt.estimation import laminar_l1_source as source
from rnj_wzzt.estimation.unmetered_inputs import (
    PreparedSourceInputs,
    SourceInputs,
    prepare_source_inputs,
)


_FEASIBILITY_TOLERANCE = 1e-6
_GAIN_ABSOLUTE_TOLERANCE = 1e-10
_GAIN_RELATIVE_TOLERANCE = 1e-8


def _prepare_source(scenarios, prepared, source_inputs):
    if source_inputs is None:
        return None, None
    if not isinstance(source_inputs, PreparedSourceInputs):
        if isinstance(scenarios, core._PreparedScenarios):
            raise ValueError("prepared scenarios require PreparedSourceInputs")
        source_inputs = prepare_source_inputs(scenarios, source_inputs)
    time_indices = None if isinstance(scenarios, core._PreparedScenarios) else tuple(
        item["P_terminal"].index
        if hasattr(item["P_terminal"], "columns") else None
        for item in scenarios
    )
    amplitudes = source._copy_source_amplitudes(
        source_inputs,
        expected_lengths=tuple(item.shape[0] for item in prepared.p),
        time_indices=time_indices,
    )
    if amplitudes is None:
        return None, None
    # Builders receive prepared arrays, so retain exactly their observation
    # order rather than aligning a second time against the original source.
    aligned = PreparedSourceInputs(
        p=amplitudes[0], q=amplitudes[1],
        time_indices=tuple(
            (source_inputs.time_indices[k] if time_indices is None or time_indices[k] is None
             else time_indices[k]).copy()
            for k in range(len(prepared.p))
        ),
        provenance=source_inputs.provenance,
    )
    return aligned, amplitudes


def _violation(values: np.ndarray, lower, upper) -> float:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        return float("inf")
    return float(max(
        np.max(np.maximum(np.asarray(lower) - values, 0.0), initial=0.0),
        np.max(np.maximum(values - np.asarray(upper), 0.0), initial=0.0),
    ))


def _audit_solution(model, blocks, result, prepared, supports, amplitudes, bound):
    """Recompute the physical Source prediction and every solver constraint.

    This also works for a backend incumbent that has no optimality certificate;
    the caller keeps its diagnostics but never accepts it as a path point.
    """
    values = getattr(result, "x", None)
    objective = getattr(result, "fun", None)
    if values is None or objective is None:
        return {"has_incumbent": False, "valid": False}
    values = np.asarray(values, dtype=float)
    if values.shape != (model.variables.size,) or not np.isfinite(values).all() or not np.isfinite(objective):
        return {"has_incumbent": False, "valid": False,
                "error": "nonfinite_or_wrong_shape_incumbent"}
    constraints = model.rows.build(model.variables.size)
    row_violation = _violation(constraints.A @ values, constraints.lb, constraints.ub)
    bound_violation = _violation(values, model.variables.lower, model.variables.upper)
    integer_columns = np.asarray(model.variables.integrality, dtype=bool)
    integral_violation = float(np.max(
        np.abs(values[integer_columns] - np.rint(values[integer_columns])), initial=0.0,
    ))
    audit = {
        "has_incumbent": True, "valid": False,
        "constraint_violation": row_violation,
        "bound_violation": bound_violation,
        "integrality_violation": integral_violation,
        "solver_objective": float(objective),
    }
    try:
        family = supports
        if model.z is not None:
            new_support = core._round_and_validate_support(values[model.z], supports, prepared.n)
            family = (*supports, new_support)
        # Core extension columns new_r and new_x are deliberately separate
        # from the old coefficient blocks; never take a combined slice.
        r_values = np.asarray(values[model.r], dtype=float)
        x_values = np.asarray(values[model.x], dtype=float)
        if model.new_r is not None:
            r_values = np.r_[r_values, values[model.new_r]]
            x_values = np.r_[x_values, values[model.new_x]]
        r_values = core._bounded_weights(r_values, bound)
        x_values = core._bounded_weights(x_values, bound)
        R, X = core.build_matrices_from_atoms(prepared.n, family, r_values, x_values)
        h = np.zeros(len(family), dtype=int)
        gR, gX = np.zeros(prepared.n), np.zeros(prepared.n)
        if blocks is not None:
            raw_h = values[blocks.h]
            if np.max(np.abs(raw_h - np.rint(raw_h)), initial=0.0) > _FEASIBILITY_TOLERANCE:
                raise RuntimeError("source path violates integrality")
            h = np.rint(raw_h).astype(int)
            # Independently verify that h selects all ancestors of one
            # non-root atom-tree node.  No physical source site enters this.
            masks = [np.asarray([int(set(node) <= set(edge)) for edge in family])
                     for node in family]
            if not any(np.array_equal(h, mask) for mask in masks):
                raise RuntimeError("source path is not a complete non-root node path")
            z = np.zeros((len(family), prepared.n))
            for k, support in enumerate(family):
                z[k, list(support)] = 1.0
            gR, gX = (r_values * h) @ z, (x_values * h) @ z
            product_error = max(
                float(np.max(np.abs(values[blocks.w_r] - r_values * h), initial=0.0)),
                float(np.max(np.abs(values[blocks.w_x] - x_values * h), initial=0.0)),
            )
            if model.z is not None:
                new_z = z[-1]
                product_error = max(product_error,
                    float(np.max(np.abs(values[blocks.v_r] - r_values[-1] * h[-1] * new_z), initial=0.0)),
                    float(np.max(np.abs(values[blocks.v_x] - x_values[-1] * h[-1] * new_z), initial=0.0)),
                )
            audit["source_product_violation"] = product_error
        loss = 0.0
        for k, (p, q, target) in enumerate(zip(prepared.p, prepared.q, prepared.target, strict=True)):
            prediction = p @ R.T + q @ X.T
            if amplitudes is not None:
                prediction += amplitudes[0][k][:, None] * gR + amplitudes[1][k][:, None] * gX
            loss += float(np.abs(target - prediction).sum())
        loss /= prepared.observation_count
        objective_difference = abs(loss - float(objective))
        audit.update({
            "supports": [list(item) for item in family],
            "r_weights": r_values.tolist(), "x_weights": x_values.tolist(),
            "h": h.tolist(), "source_gR": gR.tolist(), "source_gX": gX.tolist(),
            "train_mae": loss, "objective_difference": objective_difference,
            "coefficient_bound_active": bool(np.any(r_values >= 0.995 * bound)
                                              or np.any(x_values >= 0.995 * bound)),
        })
        audit["valid"] = bool(
            max(row_violation, bound_violation, integral_violation,
                audit.get("source_product_violation", 0.0)) <= _FEASIBILITY_TOLERANCE
            and core._objectives_agree(loss, float(objective))
        )
    except (RuntimeError, ValueError) as exc:
        audit["error"] = str(exc)
    return audit


def fit_wzzt(
    scenarios: Sequence[dict] | core._PreparedScenarios,
    source_inputs: PreparedSourceInputs | Sequence[SourceInputs] | None = None,
    initial_supports: Iterable[Iterable[int]] | None = None,
    *,
    per_solve_seconds: float = 60,
    wall_seconds: float = 300,
    coefficient_bound: float = 2,
    solver: str = "highs",
    checkpoint: Callable[[dict], None] | None = None,
) -> dict:
    """Fit a training-only, bounded greedy laminar path on the same AC day.

    All terminal singletons are fixed from initialization.  Additional initial
    supports (for RNJ+WZZT) constrain subsequent free support extensions. Every
    accepted point has a strict numerical optimum certificate for its own
    fixed-support or one-extension subproblem plus an independent Source-aware
    objective/constraint audit.  An uncertified incumbent is retained only in
    ``attempts``.  There is no intercept, validation day, or pruning.

    ``checkpoint`` receives a complete serializable snapshot after every
    attempted solve, including accepted points and failures.  Wall budget
    includes Python model construction and all fixed/extension solves; each
    solver receives the smaller of remaining wall time and per-solve budget.
    """
    started = perf_counter()
    per_solve_seconds = core._require_positive_finite("per_solve_seconds", per_solve_seconds)
    wall_seconds = core._require_positive_finite("wall_seconds", wall_seconds)
    coefficient_bound = core._require_positive_finite("coefficient_bound", coefficient_bound)
    core._validate_solver(solver)
    if checkpoint is not None and not callable(checkpoint):
        raise ValueError("checkpoint must be callable or None")
    prepared = core._prepare_scenarios(scenarios)
    prepared_source, amplitudes = _prepare_source(scenarios, prepared, source_inputs)
    extras = core.normalize_supports(() if initial_supports is None else initial_supports, prepared.n)
    singleton_family = tuple((k,) for k in range(prepared.n))
    supports = core.normalize_supports(
        (*singleton_family, *(item for item in extras if len(item) > 1)), prepared.n,
    )
    attempts, path = [], []
    status, stop_reason = "running", "initializing"
    current = None

    def snapshot():
        chosen = current or {}
        return {
            "labels": list(prepared.labels),
            "supports": chosen.get("supports", [list(item) for item in supports]),
            "support_labels": [
                [prepared.labels[k] for k in item]
                for item in chosen.get("supports", supports)
            ],
            "r_weights": chosen.get("r_weights"), "x_weights": chosen.get("x_weights"),
            "h": chosen.get("h"), "R": chosen.get("R"), "X": chosen.get("X"),
            "source_gR": chosen.get("source_gR"), "source_gX": chosen.get("source_gX"),
            "train_mae": chosen.get("train_mae"),
            "source_enabled": amplitudes is not None,
            "initial_supports": [list(item) for item in (*singleton_family, *(item for item in extras if len(item) > 1))],
            "coefficient_bound": coefficient_bound,
            "coefficient_bound_active": chosen.get("coefficient_bound_active"),
            "path": list(path), "attempts": list(attempts),
            "stop_reason": stop_reason, "status": status,
            "seconds": perf_counter() - started,
            "per_solve_seconds": per_solve_seconds, "wall_seconds": wall_seconds,
            "solver": solver, "intercept": False, "pruning": False,
            "selection": "last_certified_positive_training_gain",
            "certificate_scope": "fixed-support or one-extension bounded subproblem; greedy whole path",
        }

    def notify():
        if checkpoint is not None:
            checkpoint(snapshot())

    while True:
        extension = current is not None
        if extension and len(supports) >= 2 * prepared.n - 1:
            status, stop_reason = "complete", "laminar_family_limit"
            break
        if perf_counter() - started >= wall_seconds:
            status, stop_reason = "incomplete", "wall_time_limit"
            break
        attempt = {"kind": "extension" if extension else "initial_fixed",
                   "support_count_before": len(supports), "accepted": False}
        model_started = perf_counter()
        try:
            builder = source.build_extension_source_model if extension else source.build_fixed_source_model
            model, blocks = builder(
                prepared, supports, prepared_source,
                r_upper_bound=coefficient_bound, x_upper_bound=coefficient_bound,
            )
            remaining = wall_seconds - (perf_counter() - started)
            attempt["model_build_seconds"] = perf_counter() - model_started
            if remaining <= 0.0:
                attempt["error"] = "wall_time_limit_during_model_build"
                attempts.append(attempt)
                status, stop_reason = "incomplete", "wall_time_limit"
                notify()
                break
            limit = min(per_solve_seconds, remaining)
            attempt["time_limit_seconds"] = limit
            result, diagnostics = core._run_milp(
                model.variables, model.rows, time_limit=limit,
                mip_rel_gap=0.0, presolve=True, disp=False, solver=solver,
            )
            attempt["diagnostics"] = diagnostics.to_dict()
            audit = _audit_solution(model, blocks, result, prepared, supports, amplitudes, coefficient_bound)
            attempt["incumbent"] = audit
            certified = core.solver_diagnostics_prove_optimality(diagnostics)
            attempt["optimality_certified"] = certified
            attempts.append(attempt)
            if not certified:
                if extension and diagnostics.status == 2:
                    status, stop_reason = "complete", "no_feasible_extension"
                else:
                    status = "incomplete" if extension else "failed"
                    stop_reason = "extension_not_proven_optimal" if extension else "initial_fixed_not_proven_optimal"
                notify()
                break
            if not audit["valid"]:
                status, stop_reason = "failed", "physical_objective_or_constraint_audit_failed"
                notify()
                break
            gain = None if current is None else current["train_mae"] - audit["train_mae"]
            attempt["gain"] = gain
            if current is not None:
                tolerance = _GAIN_ABSOLUTE_TOLERANCE + _GAIN_RELATIVE_TOLERANCE * max(
                    abs(current["train_mae"]), np.finfo(float).eps,
                )
                if gain <= tolerance:
                    status, stop_reason = "complete", "no_significant_one_atom_gain"
                    notify()
                    break
            supports = tuple(tuple(item) for item in audit["supports"])
            R, X = core.build_matrices_from_atoms(
                prepared.n, supports, audit["r_weights"], audit["x_weights"],
            )
            current = dict(audit, R=R.tolist(), X=X.tolist(), gain=gain,
                           diagnostics=diagnostics.to_dict())
            attempt["accepted"] = True
            path.append(dict(current, iteration=len(path)))
            status, stop_reason = "running", "searching"
            notify()
        except Exception as exc:
            # Keep solver and modeling failures in the same experiment
            # denominator; the last certified model remains available.
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            if not attempts or attempts[-1] is not attempt:
                attempts.append(attempt)
            status, stop_reason = "failed", "solver_or_model_exception"
            notify()
            break
    notify()
    return snapshot()
