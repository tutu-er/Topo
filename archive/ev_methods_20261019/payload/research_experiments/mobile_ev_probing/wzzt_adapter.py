"""Research-only source-aware WZZT search and fixed-family competing fits.

Each call rebuilds from singleton atoms and externally certified supports.
Every solve reoptimizes the complete hidden-source path. The forward search
is greedy; solver optimality concerns only its bounded fixed/one-atom domain.

Optional source amplitude bounds use |e| <= bP*gR + bQ*gX. This is an
output-wise outer approximation of a shared source-amplitude uncertainty
set, not an exact robust source model and not a confidence guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping
from typing import Iterable, Sequence

import numpy as np

from rnj_wzzt.estimation import laminar_l1_milp as core
from rnj_wzzt.estimation import laminar_l1_source as source
from rnj_wzzt.estimation.unmetered_inputs import PreparedSourceInputs


# The normalized drop-MAE objective can be only 1e-7--1e-5. Keeping its
# solver costs near order one avoids premature termination from native
# absolute objective tolerances. Scaling changes no feasible set/optimum.
_OBJECTIVE_SCALE = 1e5


@dataclass(frozen=True)
class SourceSolverDiagnostics(core.SolverDiagnostics):
    """Native diagnostics plus every same-domain numerical recovery solve."""
    diagnostics_history: tuple[core.SolverDiagnostics, ...] = ()
    presolve_history: tuple[bool, ...] = ()


@dataclass(frozen=True)
class SourceFitSolution:
    terminal_labels: tuple
    support_indices: tuple[tuple[int, ...], ...]
    r_values: np.ndarray
    x_values: np.ndarray
    r_matrix: np.ndarray
    x_matrix: np.ndarray
    h: np.ndarray
    g_r: np.ndarray
    g_x: np.ndarray
    train_mae: float
    adjusted_train_mae: float
    diagnostics: core.SolverDiagnostics
    validation_mae: float | None = None
    adjusted_validation_mae: float | None = None
    r_path_upper_bound: tuple[float, ...] | None = None
    x_path_upper_bound: tuple[float, ...] | None = None
    certified_x_lower_bounds: tuple[tuple[tuple[int, ...], float], ...] = ()
    certified_r_lower_bounds: tuple[tuple[tuple[int, ...], float], ...] = ()
    r_upper_bound: float = 1.0
    x_upper_bound: float = 1.0

    @property
    def supports(self):
        return self.support_indices

    @property
    def support_labels(self):
        return tuple(tuple(self.terminal_labels[i] for i in s) for s in self.support_indices)

    @property
    def objective(self):
        return self.adjusted_train_mae

    @property
    def diagnostics_history(self):
        return getattr(self.diagnostics, "diagnostics_history", ()) or (self.diagnostics,)


@dataclass(frozen=True)
class SourceSolveAttempt:
    stage: str
    support_indices: tuple[tuple[int, ...], ...]
    added_support: tuple[int, ...] | None
    diagnostics: core.SolverDiagnostics
    accepted: bool

    @property
    def diagnostics_history(self):
        return getattr(self.diagnostics, "diagnostics_history", ()) or (self.diagnostics,)


@dataclass(frozen=True)
class SourceFitResult:
    selected: SourceFitSolution | None
    path: tuple[SourceFitSolution, ...]
    attempted_solves: tuple[SourceSolveAttempt, ...]
    stop_reason: str
    mandatory_supports: tuple[tuple[int, ...], ...]
    r_upper_bound: float
    x_upper_bound: float
    bound_active: bool
    certified_r_lower_bounds: tuple[tuple[tuple[int, ...], float], ...] = ()
    r_path_upper_bound: tuple[float, ...] | None = None
    x_path_upper_bound: tuple[float, ...] | None = None
    certified_x_lower_bounds: tuple[tuple[tuple[int, ...], float], ...] = ()

    @property
    def support_indices(self):
        return () if self.selected is None else self.selected.support_indices

    @property
    def r_matrix(self):
        return None if self.selected is None else self.selected.r_matrix

    @property
    def x_matrix(self):
        return None if self.selected is None else self.selected.x_matrix

    @property
    def h(self):
        return None if self.selected is None else self.selected.h

    @property
    def g_r(self):
        return None if self.selected is None else self.selected.g_r

    @property
    def g_x(self):
        return None if self.selected is None else self.selected.g_x

    @property
    def train_mae(self):
        return None if self.selected is None else self.selected.train_mae

    def summary(self):
        return {
            "support_indices": [list(s) for s in self.support_indices],
            "train_mae": self.train_mae,
            "adjusted_train_mae": None if self.selected is None else self.selected.objective,
            "path_length": len(self.path), "stop_reason": self.stop_reason,
            "r_upper_bound": self.r_upper_bound, "x_upper_bound": self.x_upper_bound,
            "bound_active": self.bound_active,
            "solver_objective_scale": _OBJECTIVE_SCALE,
            "certified_r_lower_bounds": [{"support": list(s), "lower": v}
                                          for s, v in self.certified_r_lower_bounds],
            "r_path_upper_bound": self.r_path_upper_bound,
            "x_path_upper_bound": self.x_path_upper_bound,
            "certified_x_lower_bounds": [{"support": list(s), "lower": v}
                                          for s, v in self.certified_x_lower_bounds],
            "solver_attempts": [{"stage": a.stage, "accepted": a.accepted,
                                 "added_support": a.added_support,
                                 **a.diagnostics.to_dict()} for a in self.attempted_solves],
        }


class SourceSolveError(RuntimeError):
    """A failed/unproved solve, with full diagnostics and no accepted incumbent."""

    def __init__(self, stage, diagnostics):
        self.stage, self.diagnostics = stage, diagnostics
        self.diagnostics_history = getattr(diagnostics, "diagnostics_history", ()) or (diagnostics,)
        super().__init__(f"{stage} not proven optimal: status={diagnostics.status}, "
                         f"gap={diagnostics.mip_gap}, {diagnostics.message}")


def _error_blocks(value, prepared, name):
    lengths = tuple(len(p) for p in prepared.p)
    if value is None:
        return tuple(np.zeros(t) for t in lengths)
    if np.isscalar(value):
        blocks = tuple(np.full(t, float(value)) for t in lengths)
    elif len(lengths) == 1 and np.asarray(value).shape == (lengths[0],):
        blocks = (np.asarray(value, dtype=float).copy(),)
    else:
        if len(value) != len(lengths):
            raise ValueError(f"{name} must have one block per scenario")
        blocks = tuple(np.asarray(v, dtype=float).copy() for v in value)
    if any(b.shape != (t,) or not np.isfinite(b).all() or (b < 0).any()
           for b, t in zip(blocks, lengths, strict=True)):
        raise ValueError(f"{name} must contain finite nonnegative (T,) blocks")
    return blocks


def _prepare(scenarios, source_inputs, source_p_error, source_q_error, r_bound, x_bound,
             *, existing_source_response=False):
    prepared, _, amplitudes, _, _ = source._prepare_source_model_inputs(
        scenarios, (), source_inputs, r_upper_bound=r_bound, x_upper_bound=x_bound)
    aligned = None
    if amplitudes is not None:
        aligned = PreparedSourceInputs(amplitudes[0], amplitudes[1],
                                       source_inputs.time_indices, source_inputs.provenance)
    b_p = _error_blocks(source_p_error, prepared, "source_p_error")
    b_q = _error_blocks(source_q_error, prepared, "source_q_error")
    if (aligned is None and not existing_source_response
            and any(np.any(b) for b in (*b_p, *b_q))):
        raise ValueError("nonzero source uncertainty requires active nonzero source amplitudes")
    return prepared, aligned, b_p, b_q


def _add_response_allowance(model, blocks, supports, prepared, b_p, b_q):
    """Add +/- allowance to the existing final paired residual rows."""
    if blocks is None:
        return
    first = model.rows.size - 2 * prepared.observation_count
    if first < 0:
        raise RuntimeError("core residual row layout changed")
    obs = 0
    for ps, bp, bq in zip(prepared.p, b_p, b_q, strict=True):
        for t in range(len(ps)):
            for i in range(prepared.n):
                r_cols = [blocks.w_r.start + k for k, s in enumerate(supports) if i in s]
                x_cols = [blocks.w_x.start + k for k, s in enumerate(supports) if i in s]
                if isinstance(blocks, source.SourceExtensionBlocks):
                    r_cols.append(blocks.v_r.start + i)
                    x_cols.append(blocks.v_x.start + i)
                for side, sign in ((0, 1.0), (1, -1.0)):
                    row = first + 2 * obs + side
                    for cols, amount in ((r_cols, bp[t]), (x_cols, bq[t])):
                        for col in cols:
                            if amount:
                                model.rows.row_index.append(row)
                                model.rows.col_index.append(col)
                                model.rows.data.append(sign * float(amount))
                obs += 1


def _checked_r_lower_bounds(values, supports, r_upper_bound, *, name="certified_r_lower_bounds"):
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a support-to-bound mapping")
    checked = {}
    for support, value in values.items():
        item = tuple(sorted(set(int(i) for i in support)))
        if item not in supports:
            raise ValueError(f"{name} must refer to retained certified supports")
        bound = core._require_nonnegative_finite(name, value)
        if bound > r_upper_bound:
            raise ValueError(f"{name} exceeds corresponding atom upper bound")
        checked[item] = max(checked.get(item, 0.0), bound)
    return checked


def _checked_path_upper_bound(value, n, name):
    """An engineering bound on the aggregate root-to-terminal coefficient."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite nonnegative scalar or aligned (n,) vector")
    bounds = np.asarray(value, dtype=float)
    if bounds.ndim == 0:
        bounds = np.full(n, float(bounds))
    if bounds.shape != (n,) or not np.isfinite(bounds).all() or (bounds < 0).any():
        raise ValueError(f"{name} must be a finite nonnegative scalar or aligned (n,) vector")
    return tuple(float(v) for v in bounds)


def _add_path_upper_bounds(model, supports, n, r_path, x_path, *, extension):
    # u[ii] = new_weight*z[i]*z[i] = new_weight*z[i]. This reuses the
    # existing exact binary-product formulation, including zero-source fits.
    for block, products, bounds in ((model.r, "u_r", r_path),
                                    (model.x, "u_x", x_path)):
        if bounds is None:
            continue
        for i, upper in enumerate(bounds):
            entries = {block.start + k: 1.0 for k, s in enumerate(supports) if i in s}
            if extension:
                diagonal_index = i * n - i * (i - 1) // 2
                entries[model.variables.slices[products].start + diagonal_index] = 1.0
            model.rows.add(entries, upper=upper)


def _path_bound_active(fit, r_path, x_path):
    return any(bounds is not None and np.any(np.diag(matrix) >= .995 * np.asarray(bounds))
               for matrix, bounds in ((fit.r_matrix, r_path), (fit.x_matrix, x_path)))


def _predict_metrics(prepared, fit, amplitudes, b_p, b_q):
    raw, adjusted = 0.0, 0.0
    for k, (p, q, y) in enumerate(zip(prepared.p, prepared.q, prepared.target, strict=True)):
        prediction = p @ fit.r_matrix.T + q @ fit.x_matrix.T
        if amplitudes is not None:
            prediction += amplitudes.p[k][:, None] * fit.g_r
            prediction += amplitudes.q[k][:, None] * fit.g_x
        residual = np.abs(y - prediction)
        allowance = b_p[k][:, None] * fit.g_r + b_q[k][:, None] * fit.g_x
        raw += residual.sum()
        adjusted += np.maximum(residual - allowance, 0.0).sum()
    return float(raw / prepared.observation_count), float(adjusted / prepared.observation_count)


def _solve(prepared, supports, amplitudes, b_p, b_q, *, extension, r_bound, x_bound,
           time_limit, mip_rel_gap, presolve, disp, solver, r_lower_bounds=None,
           r_path_upper_bound=None, x_path_upper_bound=None, x_lower_bounds=None):
    options = dict(extension=extension, r_bound=r_bound, x_bound=x_bound,
        time_limit=time_limit, mip_rel_gap=mip_rel_gap, disp=disp, solver=solver,
        r_lower_bounds=r_lower_bounds, x_lower_bounds=x_lower_bounds,
        r_path_upper_bound=r_path_upper_bound, x_path_upper_bound=x_path_upper_bound)
    fit, diagnostics, added = _solve_once(
        prepared, supports, amplitudes, b_p, b_q, presolve=presolve, **options)
    if not (presolve and diagnostics.message.startswith("physical_postcheck_failed:")):
        return fit, diagnostics, added
    # One recovery attempt changes only the solver presolve option. The
    # prepared data, source semantics, engineering domain and proof checks
    # remain identical. A time-limit/nonoptimal solve is never retried here.
    recovered, retry_diagnostics, retry_added = _solve_once(
        prepared, supports, amplitudes, b_p, b_q, presolve=False, **options)
    final = SourceSolverDiagnostics(**retry_diagnostics.to_dict(),
        diagnostics_history=(diagnostics, retry_diagnostics),
        presolve_history=(bool(presolve), False))
    if recovered is not None:
        recovered = replace(recovered, diagnostics=final)
    return recovered, final, retry_added


def _solve_once(prepared, supports, amplitudes, b_p, b_q, *, extension, r_bound, x_bound,
                time_limit, mip_rel_gap, presolve, disp, solver, r_lower_bounds=None,
                r_path_upper_bound=None, x_path_upper_bound=None, x_lower_bounds=None):
    builder = source.build_extension_source_model if extension else source.build_fixed_source_model
    model, blocks = builder(prepared, supports, amplitudes,
                            r_upper_bound=r_bound, x_upper_bound=x_bound)
    _add_response_allowance(model, blocks, supports, prepared, b_p, b_q)
    _add_path_upper_bounds(model, supports, prepared.n,
                          r_path_upper_bound, x_path_upper_bound, extension=extension)
    for k, support in enumerate(supports):
        if r_lower_bounds is not None and support in r_lower_bounds:
            model.variables.lower[model.r.start + k] = r_lower_bounds[support]
        if x_lower_bounds is not None and support in x_lower_bounds:
            model.variables.lower[model.x.start + k] = x_lower_bounds[support]
    model.variables.objective[:] = [float(c) * _OBJECTIVE_SCALE
                                   for c in model.variables.objective]
    result, diagnostics = core._run_milp(
        model.variables, model.rows, time_limit=time_limit,
        mip_rel_gap=mip_rel_gap if extension else 0.0,
        presolve=presolve, disp=disp, solver=solver)
    # Diagnostics and the accepted physical objective use the original MAE
    # units; retain the native relative gap/status without weakening proof.
    if result.fun is not None:
        result.fun = float(result.fun) / _OBJECTIVE_SCALE
    diagnostics = replace(diagnostics,
        objective=None if diagnostics.objective is None else diagnostics.objective / _OBJECTIVE_SCALE,
        dual_bound=None if diagnostics.dual_bound is None else diagnostics.dual_bound / _OBJECTIVE_SCALE)
    if not (core.solver_diagnostics_prove_optimality(diagnostics)
            and result.x is not None and result.fun is not None):
        return None, diagnostics, None
    try:
        return _read_solution(prepared, supports, amplitudes, b_p, b_q,
            model, blocks, result, diagnostics, extension=extension,
            r_bound=r_bound, x_bound=x_bound,
            r_lower_bounds=r_lower_bounds, x_lower_bounds=x_lower_bounds,
            r_path_upper_bound=r_path_upper_bound, x_path_upper_bound=x_path_upper_bound)
    except (RuntimeError, ValueError, IndexError) as error:
        # A native optimal status does not override an invalid physical
        # incumbent. Keep its objective/message for audit, not its proof.
        failed = replace(diagnostics, status=4, success=False,
            raw_status=diagnostics.raw_status if diagnostics.raw_status is not None else diagnostics.status,
            dual_bound=None, mip_gap=None,
            message=f"physical_postcheck_failed: {error}; native solver: {diagnostics.message}")
        return None, failed, None


def _read_solution(prepared, supports, amplitudes, b_p, b_q,
                   model, blocks, result, diagnostics, *, extension,
                   r_bound, x_bound, r_lower_bounds, x_lower_bounds,
                   r_path_upper_bound, x_path_upper_bound):
    tolerance = core._COEFFICIENT_BOUND_TOLERANCE
    vector = np.asarray(result.x, dtype=float)
    if vector.shape != (model.variables.size,):
        raise RuntimeError("solver vector does not match the model variable count")
    if not np.isfinite(vector).all() or not np.isfinite(result.fun):
        raise RuntimeError("solver vector/objective contains nonfinite values")
    # Physical coefficients use the core's strict 1e-8 guard. Do not impose
    # that unrelated absolute scale on auxiliary residual/product variables;
    # their native feasibility and reconstructed objective are checked below.
    added = None
    if extension:
        rounded_z = np.rint(result.x[model.z])
        if np.any((rounded_z != 0) & (rounded_z != 1)):
            raise RuntimeError("invalid binary extension membership")
        added = core._round_and_validate_support(result.x[model.z], supports, prepared.n)
        family = (*supports, added)
        r = np.r_[result.x[model.r], result.x[model.new_r]]
        x = np.r_[result.x[model.x], result.x[model.new_x]]
    else:
        family, r, x = supports, result.x[model.r], result.x[model.x]
    coefficient_violation = max(np.max(-r, initial=0.), np.max(r - r_bound, initial=0.),
                                np.max(-x, initial=0.), np.max(x - x_bound, initial=0.))
    if coefficient_violation > tolerance:
        raise RuntimeError(f"physical atom coefficients violate bounds: "
                           f"max_violation={coefficient_violation:.6g}, tolerance={tolerance:.6g}")
    r, x = core._bounded_weights(r, r_bound), core._bounded_weights(x, x_bound)
    h = np.zeros(len(family), dtype=int)
    if blocks is not None:
        actual_h = np.asarray(result.x[blocks.h])
        h = np.rint(actual_h).astype(int)
        if (np.max(np.abs(actual_h - h), initial=0) > 1e-6
                or np.any((h != 0) & (h != 1)) or not np.any(h)):
            raise RuntimeError("invalid solved source path")
        active = [set(s) for s, on in zip(family, h, strict=True) if on]
        if any(not (a <= b or b <= a) for a in active for b in active):
            raise RuntimeError("source path contains disjoint atoms")
        if any(on and not h[k] for s, on in zip(family, h, strict=True)
               for k, ancestor in enumerate(family) if set(s) <= set(ancestor)):
            raise RuntimeError("source path omits an ancestor")
    z = np.zeros((len(family), prepared.n))
    for k, s in enumerate(family):
        z[k, list(s)] = 1.0
    r_matrix, x_matrix = core.build_matrices_from_atoms(prepared.n, family, r, x)
    for matrix, bounds, name in ((r_matrix, r_path_upper_bound, "R"),
                                (x_matrix, x_path_upper_bound, "X")):
        if bounds is not None and np.max(np.diag(matrix) - bounds, initial=0.) > tolerance:
            raise RuntimeError(f"physical {name} path exceeds its engineering upper bound")
    for values, bounds, name in ((r, r_lower_bounds, "R"), (x, x_lower_bounds, "X")):
        for support, bound in (bounds or {}).items():
            if values[family.index(support)] < bound - tolerance:
                raise RuntimeError(f"physical certified {name} atom is below its lower bound")
    fit = SourceFitSolution(prepared.labels, family, r, x, r_matrix, x_matrix,
                            h, z.T @ (r * h), z.T @ (x * h), 0.0, 0.0, diagnostics,
                            r_path_upper_bound=r_path_upper_bound,
                            x_path_upper_bound=x_path_upper_bound,
                            certified_r_lower_bounds=tuple((r_lower_bounds or {}).items()),
                            certified_x_lower_bounds=tuple((x_lower_bounds or {}).items()),
                            r_upper_bound=r_bound, x_upper_bound=x_bound)
    raw, adjusted = _predict_metrics(prepared, fit, amplitudes, b_p, b_q)
    if not core._objectives_agree(adjusted, float(result.fun)):
        raise RuntimeError(f"source physical objective mismatch: {adjusted} vs {result.fun}")
    return replace(fit, train_mae=raw, adjusted_train_mae=adjusted), diagnostics, added


def fit_fixed_source_supports(
    scenarios: Sequence[dict], supports: Iterable[Iterable[int]], *,
    source_inputs: PreparedSourceInputs | None = None,
    source_p_error=None, source_q_error=None,
    certified_r_lower_bounds: Mapping | None = None,
    certified_x_lower_bounds: Mapping | None = None,
    r_upper_bound: float = 1.0, x_upper_bound: float = 1.0,
    r_path_upper_bound=None, x_path_upper_bound=None,
    time_limit: float | None = None, presolve=True, disp=False, solver="highs",
) -> SourceFitSolution:
    """Globally solve one bounded fixed family, reselecting its source path.

    No incumbent is returned unless the solver proves optimality. Competing
    families may be fitted with this API, but failure to find a competing
    family is not an exclusion certificate. Optional certified R lower
    and X lower bounds are externally justified aggregate clade-weight
    bounds. Path upper bounds constrain diag(R), diag(X), in squared-voltage
    coefficient units, and must come from external engineering information.
    """
    core._validate_solver(solver)
    r_bound, x_bound, limit, _ = core._validate_solver_parameters(
        r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound, time_limit=time_limit)
    prepared, amplitudes, bp, bq = _prepare(
        scenarios, source_inputs, source_p_error, source_q_error, r_bound, x_bound)
    family = core.normalize_supports(supports, prepared.n)
    lower_bounds = _checked_r_lower_bounds(certified_r_lower_bounds, family, r_bound)
    x_lower_bounds = _checked_r_lower_bounds(certified_x_lower_bounds, family, x_bound,
                                            name="certified_x_lower_bounds")
    r_path = _checked_path_upper_bound(r_path_upper_bound, prepared.n, "r_path_upper_bound")
    x_path = _checked_path_upper_bound(x_path_upper_bound, prepared.n, "x_path_upper_bound")
    fit, diagnostics, _ = _solve(prepared, family, amplitudes, bp, bq,
        extension=False, r_bound=r_bound, x_bound=x_bound, time_limit=limit,
        mip_rel_gap=0.0, presolve=presolve, disp=disp, solver=solver,
        r_lower_bounds=lower_bounds, x_lower_bounds=x_lower_bounds,
        r_path_upper_bound=r_path, x_path_upper_bound=x_path)
    if fit is None:
        raise SourceSolveError("fixed_support", diagnostics)
    return fit


def fit_source_aware_wzzt(
    scenarios: Sequence[dict], *, source_inputs: PreparedSourceInputs | None = None,
    certified_supports: Iterable[Iterable[int]] = (),
    certified_r_lower_bounds: Mapping | None = None,
    certified_x_lower_bounds: Mapping | None = None,
    source_p_error=None, source_q_error=None,
    validation_scenarios: Sequence[dict] | None = None,
    validation_source_inputs: PreparedSourceInputs | None = None,
    validation_source_p_error=None, validation_source_q_error=None,
    max_atoms: int | None = None, r_upper_bound: float = 1.0, x_upper_bound: float = 1.0,
    r_path_upper_bound=None, x_path_upper_bound=None,
    improvement_abs_tol: float = 1e-10, improvement_rel_tol: float = 1e-8,
    zero_tolerance: float = 1e-9, time_limit: float | None = None,
    mip_rel_gap: float = 0.0, presolve=True, disp=False, solver="highs",
) -> SourceFitResult:
    """Restart a fully corrective greedy path from mandatory/certified atoms.

    Certificates are supplied by the caller; this fitter does not certify
    them. A supplied certified_r_lower_bounds mapping additionally prevents
    the corresponding certified atom being given zero R weight. Held-out
    scores retain the training source path and weights.
    Nonoptimal initial/refit/extension solves stop without accepting them.
    Bounds are explicit and fixed, and any active bound is reported.
    Optional root-to-terminal path bounds constrain aggregate coefficients;
    they also imply g_R <= diag(R) and g_X <= diag(X) for the source response.
    """
    core._validate_solver(solver)
    rb, xb, limit, gap = core._validate_solver_parameters(
        r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound,
        time_limit=time_limit, mip_rel_gap=mip_rel_gap)
    prepared, amplitudes, bp, bq = _prepare(
        scenarios, source_inputs, source_p_error, source_q_error, rb, xb)
    cert = core.normalize_supports(certified_supports, prepared.n)
    mandatory = core.normalize_supports(
        tuple((i,) for i in range(prepared.n)) + tuple(s for s in cert if len(s) > 1), prepared.n)
    lower_bounds = _checked_r_lower_bounds(certified_r_lower_bounds, cert, rb)
    x_lower_bounds = _checked_r_lower_bounds(certified_x_lower_bounds, cert, xb,
                                            name="certified_x_lower_bounds")
    r_path = _checked_path_upper_bound(r_path_upper_bound, prepared.n, "r_path_upper_bound")
    x_path = _checked_path_upper_bound(x_path_upper_bound, prepared.n, "x_path_upper_bound")
    if max_atoms is None:
        max_atoms = 2 * prepared.n - 1
    if isinstance(max_atoms, (bool, np.bool_)) or not isinstance(max_atoms, (int, np.integer)):
        raise ValueError("max_atoms must be an integer")
    if not len(mandatory) <= max_atoms <= 2 * prepared.n - 1:
        raise ValueError("max_atoms must cover mandatory atoms and be <= 2*n-1")
    atol = core._require_nonnegative_finite("improvement_abs_tol", improvement_abs_tol)
    rtol = core._require_nonnegative_finite("improvement_rel_tol", improvement_rel_tol)
    ztol = core._require_nonnegative_finite("zero_tolerance", zero_tolerance)
    validation = None
    if validation_scenarios is not None:
        validation = _prepare(validation_scenarios, validation_source_inputs,
            validation_source_p_error, validation_source_q_error, rb, xb,
            existing_source_response=amplitudes is not None)
        if validation[0].labels != prepared.labels:
            raise ValueError("training and validation labels must agree")
        if ((amplitudes is None and validation[1] is not None)
                or (amplitudes is not None and validation_source_inputs is None)):
            raise ValueError("training and validation must use the same source model")
    elif validation_source_inputs is not None:
        raise ValueError("validation_source_inputs requires validation_scenarios")
    attempts, path = [], []

    def solve(family, stage, extension=False):
        fit, diagnostics, added = _solve(prepared, family, amplitudes, bp, bq,
            extension=extension, r_bound=rb, x_bound=xb, time_limit=limit,
            mip_rel_gap=float(gap), presolve=presolve, disp=disp, solver=solver,
            r_lower_bounds=lower_bounds, x_lower_bounds=x_lower_bounds,
            r_path_upper_bound=r_path, x_path_upper_bound=x_path)
        attempts.append(SourceSolveAttempt(stage, family, added, diagnostics, False))
        return fit

    def append(fit):
        if validation is not None:
            raw, adjusted = _predict_metrics(validation[0], fit, *validation[1:])
            fit = replace(fit, validation_mae=raw, adjusted_validation_mae=adjusted)
        path.append(fit)
        return fit

    current = solve(mandatory, "initial")
    stop = "initial_not_proven_optimal" if current is None else "max_atoms"
    if current is not None:
        attempts[-1] = replace(attempts[-1], accepted=True)
        current = append(current)
    seen = {frozenset(mandatory)}
    while current is not None and len(current.support_indices) < max_atoms:
        tolerance = atol + rtol * max(abs(current.objective), np.finfo(float).eps)
        candidate = solve(current.support_indices, "extension", extension=True)
        if candidate is None:
            diag = attempts[-1].diagnostics
            if diag.status == 2:
                stop = "no_feasible_extension"
            elif (diag.dual_bound is not None and np.isfinite(diag.dual_bound)
                  and current.objective - diag.dual_bound <= tolerance):
                stop = "no_significant_one_atom_gain_certified_by_dual_bound"
            else:
                stop = "extension_not_proven_optimal"
            break
        extension_attempt = len(attempts) - 1
        if current.objective - candidate.objective <= tolerance:
            stop = "no_significant_one_atom_gain"
            break
        # Fixed-family solve deliberately reconstructs and reselects all h.
        candidate = solve(candidate.support_indices, "refit")
        if candidate is None:
            stop = "refit_not_proven_optimal"
            break
        attempts[-1] = replace(attempts[-1], accepted=True)
        keep = tuple(s for s, r, x in zip(candidate.support_indices,
            candidate.r_values, candidate.x_values, strict=True)
            if s in mandatory or r > ztol or x > ztol)
        if len(keep) < len(candidate.support_indices):
            pruned = solve(keep, "pruning_refit")
            if pruned is None:
                stop = "pruning_refit_not_proven_optimal"
                break
            if pruned.objective <= candidate.objective + tolerance:
                candidate = pruned
                attempts[-1] = replace(attempts[-1], accepted=True)
        if current.objective - candidate.objective <= tolerance:
            stop = "no_significant_gain_after_refit"
            break
        signature = frozenset(candidate.support_indices)
        if signature in seen:
            stop = "repeated_family_after_pruning"
            break
        seen.add(signature)
        attempts[extension_attempt] = replace(attempts[extension_attempt], accepted=True)
        current = append(candidate)
    selected = None
    if path:
        selected = min(path, key=lambda f: (f.adjusted_validation_mae, len(f.support_indices))) \
            if validation is not None else path[-1]
    active = bool(selected is not None and
        (np.any(selected.r_values >= .995 * rb) or np.any(selected.x_values >= .995 * xb)
         or _path_bound_active(selected, r_path, x_path)))
    return SourceFitResult(selected, tuple(path), tuple(attempts), stop, mandatory, rb, xb, active,
                           tuple(lower_bounds.items()), r_path, x_path,
                           tuple(x_lower_bounds.items()))


__all__ = ["SourceSolverDiagnostics", "SourceFitSolution", "SourceSolveAttempt", "SourceFitResult", "SourceSolveError",
           "fit_fixed_source_supports", "fit_source_aware_wzzt"]
