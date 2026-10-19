"""Bounded damped arrow least squares using QR elimination of local sources.

The inputs already use scaled step coordinates.  The problem solved is

    .5 sum_t ||r_t + G_t p_shared + S_t p_source,t||^2
        + .5 damping ||p||^2,    lower <= p <= upper.

Every source block has two columns.  A feasible BVLS active set treats shared
and source bounds together; QR only replaces its unconstrained free solve.
There are no normal equations and no clipping of an unconstrained optimum.
Clipping is used only to construct a feasible initial point and to snap
roundoff-sized boundary violations.  This module does not evaluate AC states.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq, qr, solve_triangular, svd


def _validate(residual, line_jac, source_jac, lower, upper, damping, kkt_tol, max_iter):
    r, g, s = (np.asarray(value, dtype=float) for value in (residual, line_jac, source_jac))
    if r.ndim != 2 or min(r.shape) < 1:
        raise ValueError("residual must have positive (time, channel) dimensions")
    t, k = r.shape
    if g.ndim != 3 or g.shape[:2] != (t, k):
        raise ValueError("line_jac must have shape (time, channel, shared)")
    if s.shape != (t, k, 2):
        raise ValueError("source_jac must have shape (time, channel, 2)")
    width = g.shape[2]+2*t
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if lo.shape != (width,) or hi.shape != (width,):
        raise ValueError("bounds must match shared plus two source steps per time")
    if not all(np.isfinite(a).all() for a in (r, g, s)):
        raise ValueError("residuals and Jacobians must be finite")
    if np.isnan(lo).any() or np.isnan(hi).any() or np.any(lo > hi):
        raise ValueError("bounds must be ordered and contain no NaN")
    if np.any(np.isposinf(lo)) or np.any(np.isneginf(hi)):
        raise ValueError("bounds must contain at least one finite point")
    if not np.isfinite(damping) or damping < 0:
        raise ValueError("damping must be finite and nonnegative")
    if not np.isfinite(kkt_tol) or kkt_tol <= 0:
        raise ValueError("kkt_tol must be finite and positive")
    if max_iter is not None and (int(max_iter) != max_iter or max_iter < 1):
        raise ValueError("max_iter must be a positive integer")
    return r, g, s, lo, hi


def _residual_gradient(r, g, s, step, damping):
    d = g.shape[2]
    residual = r+np.einsum('tkd,d->tk', g, step[:d])+np.einsum(
        'tks,ts->tk', s, step[d:].reshape(len(r), 2))
    gradient = np.r_[np.einsum('tkd,tk->d', g, residual),
                     np.einsum('tks,tk->ts', s, residual).ravel()]+damping*step
    linear_objective = .5*float(np.sum(residual*residual))
    objective = linear_objective+.5*damping*float(step@step)
    return objective, linear_objective, gradient


def _normal_residual(gradient, active):
    answer = gradient.copy()
    answer[active == -1] = np.minimum(answer[active == -1], 0.)
    answer[active == 1] = np.maximum(answer[active == 1], 0.)
    answer[active == 2] = 0.
    return answer


def _free_solution(r, g, s, step, active, damping, counters, factor_cache):
    """Exact free-set LS minimizer, up to rank-aware floating-point solves."""
    t, k, d = g.shape
    line_free = np.flatnonzero(active[:d] == 0)
    line_active = np.flatnonzero(active[:d] != 0)
    p = len(line_free)
    root_damping = np.sqrt(damping)
    reduced_matrices, reduced_targets, local = [], [], []
    for time in range(t):
        ids = d+2*time+np.arange(2)
        source_free = np.flatnonzero(active[ids] == 0)
        source_active = np.flatnonzero(active[ids] != 0)
        target = -r[time].copy()
        if len(line_active):
            target -= g[time][:, line_active]@step[line_active]
        if len(source_active):
            target -= s[time][:, source_active]@step[ids[source_active]]
        matrix = g[time][:, line_free]
        q = len(source_free)
        if not q:
            reduced_matrices.append(matrix)
            reduced_targets.append(target)
            continue
        nuisance = s[time][:, source_free]
        if damping:
            nuisance = np.vstack((nuisance, root_damping*np.eye(q)))
            matrix = np.vstack((matrix, np.zeros((q, p))))
            target = np.r_[target, np.zeros(q)]
        key = (time, tuple(source_free))
        if key in factor_cache:
            factor = factor_cache[key]
            counters['factor_cache_hits'] += 1
        else:
            use_qr = False
            if damping:
                orthogonal, triangular = qr(nuisance, mode='full', check_finite=False)
                counters['qr_calls'] += 1
                diag = abs(np.diag(triangular[:q]))
                threshold = (np.finfo(float).eps*max(nuisance.shape)
                             *np.linalg.norm(nuisance, 'fro'))
                use_qr = bool(np.min(diag) > threshold)
            if use_qr:
                factor = ('qr', orthogonal, triangular[:q, :], q)
            else:
                # At zero damping source columns can be dependent or exactly
                # zero. SVD supplies their column space and min-norm backsolve.
                u, singular, vh = svd(nuisance, full_matrices=True, check_finite=False)
                counters['svd_calls'] += 1
                threshold = (np.finfo(float).eps*max(nuisance.shape)*singular[0]
                             if len(singular) else 0.)
                rank = int(np.count_nonzero(singular > threshold))
                factor = ('svd', u, (vh[:rank].T, singular[:rank]), rank)
            factor_cache[key] = factor
        method, orthogonal, factors, rank = factor
        use_qr = method == 'qr'
        if use_qr:
            transformed_matrix = orthogonal.T@matrix
            transformed_target = orthogonal.T@target
            reduced_matrices.append(transformed_matrix[q:])
            reduced_targets.append(transformed_target[q:])
            local.append((ids[source_free], 'qr', factors,
                          transformed_matrix[:q], transformed_target[:q]))
        else:
            transformed_matrix = orthogonal.T@matrix
            transformed_target = orthogonal.T@target
            reduced_matrices.append(transformed_matrix[rank:])
            reduced_targets.append(transformed_target[rank:])
            local.append((ids[source_free], 'svd', factors,
                          transformed_matrix[:rank], transformed_target[:rank]))
    if p:
        matrix = np.vstack(reduced_matrices)
        target = np.concatenate(reduced_targets)
        if damping:
            matrix = np.vstack((matrix, root_damping*np.eye(p)))
            target = np.r_[target, np.zeros(p)]
        shared = lstsq(matrix, target, cond=None, lapack_driver='gelsy',
                       check_finite=False)[0]
        counters['shared_solves'] += 1
    else:
        shared = np.empty(0)
    answer = step.copy()
    answer[line_free] = shared
    for ids, method, factors, matrix, target in local:
        rhs = target-matrix@shared
        if method == 'qr':
            answer[ids] = solve_triangular(factors, rhs, check_finite=False)
        else:
            right, singular = factors
            answer[ids] = right@(rhs/singular) if len(singular) else 0.
    return answer


def solve_block_step(residual, line_jac, source_jac, lower, upper, *, damping=1e-4,
                     kkt_tol=1e-10, max_iter=None):
    """Solve the bounded scaled GN/LM model and return an auditable result.

    ``step`` and ``vector`` have shared coordinates first, then time-major
    source P/Q. ``active_mask`` is -1/0/+1 at lower/free/upper, and 2 for fixed
    coordinates. ``kkt`` is an absolute infinity-norm stationarity residual in
    the supplied scaled coordinates. Failure returns the retained feasible
    step, never an uncertified optimal status. Bounds may be infinite.
    """
    r, g, s, lo, hi = _validate(residual, line_jac, source_jac, lower, upper,
                               damping, kkt_tol, max_iter)
    width = len(lo)
    limit = int(max_iter) if max_iter is not None else max(100, 8*width)
    # A feasible starting point, not a clipped solution of the LS objective.
    step = np.maximum(lo, np.minimum(np.zeros(width), hi))
    fixed = lo == hi
    active = np.zeros(width, dtype=np.int8)
    active[step == lo] = -1
    active[step == hi] = 1
    active[fixed] = 2
    counters = dict(qr_calls=0, svd_calls=0, shared_solves=0, free_solves=0,
                    bound_hits=0, bound_releases=0, roundoff_snaps=0, factor_cache_hits=0)
    factor_cache = {}
    objective, linear_objective, gradient = _residual_gradient(r, g, s, step, damping)
    history = [objective]
    success, message = False, 'active-set iteration budget exhausted'
    iterations = 0
    eps = np.finfo(float).eps
    for iterations in range(1, limit+1):
        candidate = _free_solution(r, g, s, step, active, damping, counters, factor_cache)
        counters['free_solves'] += 1
        if not np.isfinite(candidate).all():
            message = 'free least-squares solve returned nonfinite values'
            break
        free = active == 0
        finite_bounds = np.maximum(np.where(np.isfinite(lo), abs(lo), 0.),
                                   np.where(np.isfinite(hi), abs(hi), 0.))
        snap_tol = 32*eps*np.maximum(1., finite_bounds)
        large_lower = free & (candidate < lo-snap_tol)
        large_upper = free & (candidate > hi+snap_tol)
        if np.any(large_lower | large_upper):
            direction = candidate-step
            ratios = np.full(width, np.inf)
            ratios[large_lower] = (lo[large_lower]-step[large_lower])/direction[large_lower]
            ratios[large_upper] = (hi[large_upper]-step[large_upper])/direction[large_upper]
            alpha = float(np.min(ratios))
            if not np.isfinite(alpha) or alpha < -64*eps or alpha > 1.+64*eps:
                message = 'invalid first-boundary step length'
                break
            alpha = min(1., max(0., alpha))
            step[free] += alpha*direction[free]
            # Activate ties together; every hit is explicitly placed on its box.
            hits = ratios <= alpha+32*eps*max(1., abs(alpha))
            low_hits, high_hits = hits & large_lower, hits & large_upper
            step[low_hits], step[high_hits] = lo[low_hits], hi[high_hits]
            active[low_hits], active[high_hits] = -1, 1
            counters['bound_hits'] += int(np.count_nonzero(hits))
        else:
            step = candidate
            # Only machine-roundoff violations are snapped here. They have
            # already passed the strict substantive-boundary branch above.
            low_snap, high_snap = free & (step < lo), free & (step > hi)
            step[low_snap], step[high_snap] = lo[low_snap], hi[high_snap]
            active[low_snap], active[high_snap] = -1, 1
            counters['roundoff_snaps'] += int(np.count_nonzero(low_snap | high_snap))
        objective_new, linear_objective, gradient = _residual_gradient(r, g, s, step, damping)
        history.append(objective_new)
        if objective_new > objective+512*eps*max(1., abs(objective)):
            message = 'free-set step increased the convex model beyond roundoff'
            objective = objective_new
            break
        objective = objective_new
        if np.any(large_lower | large_upper):
            continue
        normal_residual = _normal_residual(gradient, active)
        if float(np.max(abs(normal_residual))) <= kkt_tol:
            success, message = True, 'box feasibility and full model KKT satisfied'
            break
        violations = np.zeros(width)
        violations[active == -1] = np.maximum(-gradient[active == -1], 0.)
        violations[active == 1] = np.maximum(gradient[active == 1], 0.)
        if float(np.max(violations)) <= kkt_tol:
            message = 'free linear solve stationarity exceeds the requested KKT tolerance'
            break
        # Deterministic first-index tie rule. A hard budget preserves any
        # numerical degeneracy instead of reporting false convergence.
        active[int(np.argmax(violations))] = 0
        counters['bound_releases'] += 1
    objective, linear_objective, gradient = _residual_gradient(r, g, s, step, damping)
    normal_residual = _normal_residual(gradient, active)
    kkt = float(np.max(abs(normal_residual)))
    feasibility = float(max(0., np.max(lo-step), np.max(step-hi)))
    gap = float(normal_residual@normal_residual)/(2*damping) if damping > 0 else None
    return dict(step=step, vector=step.copy(), success=bool(success and feasibility == 0.),
        kkt=kkt, active_mask=active, iterations=iterations, objective=objective,
        linear_objective=linear_objective, regularization=objective-linear_objective,
        gradient=gradient, normal_residual=normal_residual,
        max_feasibility_violation=feasibility, qp_gap_upper_bound=gap,
        error_bound=gap,
        objective_lower_bound=max(0., objective-gap) if gap is not None else None,
        message=message, objective_history=history, **counters)


__all__ = ['solve_block_step']
