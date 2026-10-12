"""Exact inactive-history-box acceleration for a frozen convex box QP.

The original residual is d + C z, with all intercepts, fixed coordinates,
and historical observations retained. After coordinate scaling z=S y and
substitution of fixed coordinates, partition the free columns as D_H,D_R.
H denotes historical latent P/Q/source states, never line parameters. These
columns have exactly zero future-response rows in the Stage-A construction.

For full-column-rank D_H=Q R (possibly pivoted), minimizing over unrestricted
y_H gives y_H=-R^{-1} Q^T (d_adjusted + D_R y_R), and leaves residual
(I-QQ^T)(d_adjusted + D_R y_R). Solve this smaller problem with the ORIGINAL
remaining-coordinate box. If the recovered y_H satisfies its ORIGINAL box,
the relaxed minimizer is feasible for the full problem; equality of the
relaxed lower optimum and feasible full objective proves full-box optimality
in exact arithmetic. If it violates any history bound, do not clip it:
return None so the caller solves the full bounded problem. Rank failure,
missing metadata, nonzero future history columns, or reduced-solver failure
also returns None. This neither widens a domain nor changes the criterion.

Floating solver status is not the optimality certificate: the caller must
continue evaluating the primal objective, ORIGINAL full gradient and convex
box tangent lower bound, as solve_pair already does. These statements concern
the frozen linear residual QP, not the nonlinear AC likelihood or its MLE.

Matrices/factorizations are cached only in pair._solver_cache when it is a
dict. Cache keys hash the actual supplied C, scaling and column partition;
there is no global cache and no ndarray inserted in public metadata.
"""
from __future__ import annotations

from hashlib import sha256

import numpy as np
from scipy.linalg import qr, solve_triangular, svdvals
from scipy.optimize import lsq_linear


def _value(pair, name, default=None):
    return pair.get(name, default) if isinstance(pair, dict) else getattr(pair, name, default)


def _history_columns(metadata, p):
    """Map full pre-elimination history-state slices to current pair columns."""
    if not isinstance(metadata, dict) or not metadata.get("variable_layout"):
        return None
    original_indices = metadata.get("free_indices", metadata.get("fullfree_indices"))
    if original_indices is None:
        return None
    ids = np.asarray(original_indices)
    if (ids.shape != (p,) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or len(np.unique(ids)) != p):
        return None
    history = np.zeros(p, dtype=bool)
    for block in metadata["variable_layout"]:
        if not isinstance(block, dict) or "history_state" not in block:
            return None
        interval = np.asarray(block["history_state"])
        if (interval.shape != (2,) or not np.issubdtype(interval.dtype, np.integer)
                or interval[0] < 0 or interval[1] < interval[0]):
            return None
        history |= (ids >= interval[0]) & (ids < interval[1])
    return history


def _factorization(pair, C, scale, history_indices, rest_indices, future_start):
    digest = sha256()
    for array in (C, scale, history_indices, rest_indices):
        array = np.ascontiguousarray(array)
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    digest.update(str(future_start).encode())
    key = digest.hexdigest()
    cache = _value(pair, "_solver_cache")
    storage = cache.setdefault("history_qp_qr", {}) if isinstance(cache, dict) else None
    if storage is not None and key in storage:
        return storage[key]
    # The construction has no historical-state contribution to future/F rows.
    # Require literal zero: silently ignoring a small nonzero column would
    # change the QP, whereas returning None leaves the full solver in control.
    if (future_start < len(history_indices)
            or np.any(C[future_start:, history_indices] != 0.)):
        result = None
    else:
        hist = C[:future_start, history_indices]*scale[history_indices]
        q, r, pivot = qr(hist, mode="economic", pivoting=True, check_finite=False)
        singular = svdvals(r, check_finite=False)
        rank_tolerance = 32*np.finfo(float).eps*max(hist.shape)*singular[0] if len(singular) else 0.
        if (r.shape != (len(history_indices), len(history_indices))
                or len(singular) == 0 or singular[-1] <= rank_tolerance):
            result = None
        else:
            rest = C[:, rest_indices]*scale[rest_indices]
            projected_rest = rest.copy()
            q_rest = q.T@rest[:future_start]
            projected_rest[:future_start] -= q@q_rest
            result = dict(q=q, r=r, pivot=pivot, rest=rest,
                q_rest=q_rest, projected_rest=projected_rest,
                singular_min=float(singular[-1]), rank_tolerance=float(rank_tolerance))
    if storage is not None:
        storage[key] = result
    return result


def accelerate_history_qp(pair, C, d, z_fixed, free_mask, tolerance=1e-10, max_iter=1000):
    """Return a full-coordinate feasible reduced-QP solution, or None.

    C and d are UNSCALED original full residual data. z_fixed is a full-length
    original-coordinate vector; its fixed entries must equal the declared
    fixed box values. free_mask must be exactly lower<upper. The returned z is
    in original coordinates and preserves every fixed coordinate. Caller
    should always compute primal/gradient/tangent lower bound using C,d,z.
    """
    C, d = np.asarray(C, float), np.asarray(d, float)
    z = np.asarray(z_fixed, float).copy()
    free = np.asarray(free_mask)
    lower = np.asarray(_value(pair, "lower"), float)
    upper = np.asarray(_value(pair, "upper"), float)
    scale = np.asarray(_value(pair, "scale"), float)
    p = len(lower) if lower.ndim == 1 else -1
    if (p < 0 or upper.shape != (p,) or scale.shape != (p,) or z.shape != (p,)
            or free.shape != (p,) or free.dtype != np.dtype(bool)
            or C.ndim != 2 or C.shape[1] != p or d.shape != (len(C),)
            or not all(np.isfinite(a).all() for a in (C, d, z, lower, upper, scale))
            or np.any(scale <= 0.) or np.any(lower > upper)
            or not np.array_equal(free, lower < upper)
            or np.any(z[~free] != lower[~free])
            or not np.isfinite(tolerance) or tolerance <= 0 or max_iter < 1):
        return None
    metadata = _value(pair, "metadata")
    history_mask = _history_columns(metadata, p)
    future_start = metadata.get("future_row_start") if isinstance(metadata, dict) else None
    if (history_mask is None or not isinstance(future_start, (int, np.integer))
            or not 0 <= future_start <= len(C)):
        return None
    history_indices = np.flatnonzero(free & history_mask)
    rest_indices = np.flatnonzero(free & ~history_mask)
    if not len(history_indices):
        return None
    factors = _factorization(pair, C, scale, history_indices, rest_indices, future_start)
    if factors is None:
        return None
    adjusted = d+C[:, ~free]@z[~free]
    q, r, pivot = factors["q"], factors["r"], factors["pivot"]
    q_d = q.T@adjusted[:future_start]
    projected_d = adjusted.copy()
    projected_d[:future_start] -= q@q_d
    if len(rest_indices):
        result = lsq_linear(factors["projected_rest"], -projected_d,
            bounds=(lower[rest_indices]/scale[rest_indices], upper[rest_indices]/scale[rest_indices]),
            method="bvls", lsq_solver="exact", tol=tolerance, max_iter=max_iter)
        if not result.success or not np.isfinite(result.x).all():
            return None
        z[rest_indices] = np.clip(result.x*scale[rest_indices], lower[rest_indices], upper[rest_indices])
        y_rest = z[rest_indices]/scale[rest_indices]
        status, message = int(result.status), str(result.message)
        reduced_optimality = float(result.optimality)
    else:
        y_rest = np.zeros(0)
        status, message, reduced_optimality = 3, "no remaining free coordinates", 0.
    rhs = -(q_d+factors["q_rest"]@y_rest)
    y_hist = np.empty(len(history_indices))
    y_hist[pivot] = solve_triangular(r, rhs, lower=False, check_finite=False)
    z[history_indices] = y_hist*scale[history_indices]
    # No bound tolerance, no projection of recovered H, no expanded box.
    if (not np.isfinite(z).all() or np.any(z < lower) or np.any(z > upper)):
        return None
    residual = d+C@z
    scaled_gradient = (C*scale).T@residual
    active_lower, active_upper = z == lower, z == upper
    violation = np.where(active_lower, np.minimum(scaled_gradient, 0.),
        np.where(active_upper, np.maximum(scaled_gradient, 0.), scaled_gradient))
    violation[~free] = 0.
    return dict(z=z, success=True, status=status,
        message="QR history elimination; recovered original H box feasible; "+message,
        optimality=float(np.max(abs(violation), initial=0.)), reduced_optimality=reduced_optimality,
        acceleration="history_QR_inactive_box_exact_conditional",
        eliminated_history_count=len(history_indices), remaining_free_count=len(rest_indices),
        recovered_history_box_feasible=True, qr_singular_min=factors["singular_min"],
        mathematical_scope="same frozen convex box QP; original full-gradient gap still required")
