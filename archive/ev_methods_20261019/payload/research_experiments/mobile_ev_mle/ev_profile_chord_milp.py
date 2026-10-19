"""Continuous-action MILP bounds for max_u min_k (||d_k + T_k u||^2-offset_k).

Each scalar residual has a finite box-enclosing interval. An interval binary
and a continuous within-interval weight give its piecewise linear square
chord exactly. Full action curves are not enumerated. If a chord interval
has width h, 0 <= chord(x)-x*x <= h*h/4. Consequently the MILP global best
bound is an upper bound on the original frozen quadratic problem, while
the ORIGINAL score of any feasible returned control is a lower bound.

All certificates below use floating-point arithmetic and solver tolerances;
they concern the supplied frozen quadratics, not a nonlinear AC model.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csc_matrix, issparse


@dataclass(frozen=True)
class FrozenQuadratic:
    name: str
    d: np.ndarray
    T: np.ndarray
    offset: float = 0.0

    def __post_init__(self):
        d, matrix = np.array(self.d, float, copy=True), np.array(self.T, float, copy=True)
        if (d.ndim != 1 or matrix.ndim != 2 or matrix.shape[0] != len(d)
                or matrix.shape[1] < 1 or not np.isfinite(d).all()
                or not np.isfinite(matrix).all() or not np.isfinite(self.offset)):
            raise ValueError('finite d/T/offset with matching residual dimensions required')
        d.setflags(write=False); matrix.setflags(write=False)
        object.__setattr__(self, 'd', d)
        object.__setattr__(self, 'T', matrix)
        object.__setattr__(self, 'offset', float(self.offset))

    @property
    def control_dimension(self):
        return self.T.shape[1]

    def value(self, u):
        u = np.asarray(u, float)
        if u.shape != (self.control_dimension,) or not np.isfinite(u).all():
            raise ValueError('finite control of matching dimension required')
        residual = self.d + self.T @ u
        value = float(residual @ residual - self.offset)
        if not np.isfinite(value):
            raise ValueError('quadratic evaluation overflow')
        return value


def _constraints(constraints, dimension):
    if constraints is None:
        return []
    group = [constraints] if isinstance(constraints, LinearConstraint) else list(constraints)
    normalized = []
    for constraint in group:
        if not isinstance(constraint, LinearConstraint):
            raise TypeError('constraints must be scipy LinearConstraint objects')
        A = csc_matrix(constraint.A) if issparse(constraint.A) else np.atleast_2d(np.asarray(constraint.A, float))
        if A.ndim != 2 or A.shape[1] != dimension:
            raise ValueError('linear constraint control dimension mismatch')
        values = A.data if issparse(A) else A
        lo = np.broadcast_to(np.asarray(constraint.lb, float), (A.shape[0],)).copy()
        hi = np.broadcast_to(np.asarray(constraint.ub, float), (A.shape[0],)).copy()
        if (not np.isfinite(values).all() or np.isnan(lo).any() or np.isnan(hi).any()
                or np.any(lo > hi) or np.isposinf(lo).any() or np.isneginf(hi).any()):
            raise ValueError('invalid linear constraint coefficients or bounds')
        normalized.append(LinearConstraint(A, lo, hi))
    return normalized


def check_control(u, lower, upper, constraints=None, *, tolerance=1e-7):
    """Report box/linear constraint violations without modifying the control."""
    lower, upper, u = np.asarray(lower, float), np.asarray(upper, float), np.asarray(u, float)
    if lower.ndim != 1 or upper.shape != lower.shape:
        raise ValueError('matching one-dimensional box bounds required')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('finite nonnegative feasibility tolerance required')
    if u.shape != lower.shape or not np.isfinite(u).all():
        return dict(feasible=False, max_box_violation=float('inf'), max_linear_violation=float('inf'), tolerance=tolerance)
    group = _constraints(constraints, len(lower))
    box = max(0.0, float(np.max(lower-u)), float(np.max(u-upper)))
    linear = 0.0
    for c in group:
        values = np.asarray(c.A @ u).ravel()
        linear = max(linear, float(np.max(c.lb-values, initial=0)), float(np.max(values-c.ub, initial=0)))
    return dict(feasible=max(box, linear) <= tolerance,
                max_box_violation=box, max_linear_violation=linear, tolerance=float(tolerance))


def residual_intervals(quadratic, lower, upper):
    """Box enclosure, outward padded for floating arithmetic; constraints may tighten it."""
    matrix = quadratic.T.astype(np.longdouble)
    lo, hi = np.asarray(lower, np.longdouble), np.asarray(upper, np.longdouble)
    a = quadratic.d.astype(np.longdouble) + np.minimum(matrix*lo, matrix*hi).sum(axis=1)
    b = quadratic.d.astype(np.longdouble) + np.maximum(matrix*lo, matrix*hi).sum(axis=1)
    magnitude = 1 + abs(quadratic.d.astype(np.longdouble)) + (abs(matrix)*np.maximum(abs(lo), abs(hi))).sum(axis=1)
    pad = 64*np.finfo(float).eps*magnitude
    variable = np.any((quadratic.T != 0) & (np.asarray(lower) != np.asarray(upper)), axis=1)
    a = np.asarray(a - np.where(variable, pad, 0), float)
    b = np.asarray(b + np.where(variable, pad, 0), float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('residual interval overflow')
    return a, b, variable


def chord_value(residual, knots):
    nodes = np.asarray(knots, float)
    if len(nodes) == 1:
        return float(nodes[0]**2)
    k = min(max(int(np.searchsorted(nodes, residual, side='right'))-1, 0), len(nodes)-2)
    a, b = nodes[k:k+2]
    return float((a+b)*residual-a*b)


def build_profile_chord_milp(quadratics, lower, upper, constraints=None, *, segments=4):
    """Return a scipy MILP model and interval/error metadata, with continuous controls."""
    quadratics = list(quadratics)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if (not quadratics or lower.ndim != 1 or len(lower) < 1 or upper.shape != lower.shape
            or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper)):
        raise ValueError('nonempty quadratics and finite ordered control box required')
    if any(not isinstance(q, FrozenQuadratic) or q.control_dimension != len(lower) for q in quadratics):
        raise ValueError('quadratic control dimension mismatch')
    if len({q.name for q in quadratics}) != len(quadratics):
        raise ValueError('unique quadratic names required')
    if isinstance(segments, bool) or int(segments) != segments or segments < 1:
        raise ValueError('positive integer segments required')
    segments = int(segments); dimension = len(lower)
    group = _constraints(constraints, dimension)
    intervals = [residual_intervals(q, lower, upper) for q in quadratics]
    pair_lower, pair_upper, magnitudes = [], [], []
    for q, (a, b, _) in zip(quadratics, intervals):
        mins = np.where((a <= 0) & (b >= 0), 0, np.minimum(a*a, b*b))
        maxima = np.maximum(a*a, b*b)
        pair_lower.append(float(mins.sum()-q.offset))
        pair_upper.append(float(maxima.sum()-q.offset))
        magnitudes.append(float(maxima.sum()+abs(q.offset)))
    if not np.isfinite(pair_lower+pair_upper+magnitudes).all():
        raise ValueError('quadratic box bounds overflow')
    scale = max(1e-12, max(magnitudes))
    lower_score, upper_score = min(pair_lower), min(pair_upper)
    guard = 256*np.finfo(float).eps*max(scale, abs(lower_score), abs(upper_score), 1e-12)
    lbs, ubs, integer = list(lower), list(upper), [0]*dimension
    score_index = dimension
    lbs.append((lower_score-guard)/scale); ubs.append((upper_score+guard)/scale); integer.append(0)
    rows, row_lower, row_upper = [], [], []

    def add_variables(count, integral=False):
        start = len(lbs)
        lbs.extend([0.0]*count); ubs.extend([1.0]*count); integer.extend([int(integral)]*count)
        return np.arange(start, start+count)

    def add_row(row, lo=-np.inf, hi=np.inf):
        rows.append({int(k): float(v) for k, v in row.items() if v != 0})
        row_lower.append(float(lo)); row_upper.append(float(hi))

    for c in group:
        A = c.A.tocsr() if issparse(c.A) else c.A
        for j in range(A.shape[0]):
            if issparse(A):
                r = A.getrow(j); row = dict(zip(r.indices, r.data))
            else:
                row = dict(enumerate(A[j]))
            add_row(row, c.lb[j], c.ub[j])
    all_knots, pair_errors = [], []
    for q, (a, b, variable) in zip(quadratics, intervals):
        score_row = {score_index: 1.0}; constant = -q.offset; error = 0.0; knots_group = []
        for j, (lo, hi, varies) in enumerate(zip(a, b, variable)):
            if not varies:
                knots_group.append(np.array([lo])); constant += lo*lo
                continue
            knots = np.linspace(lo, hi, segments+1); knots_group.append(knots)
            if np.any(np.diff(knots) <= 0):
                raise ValueError('floating-point residual intervals cannot support requested segments')
            selected, fraction = add_variables(segments, True), add_variables(segments)
            add_row(dict(zip(selected, np.ones(segments))), 1, 1)
            residual_row = dict(enumerate(q.T[j]))
            for k, (left, right) in enumerate(zip(knots[:-1], knots[1:])):
                add_row({fraction[k]: 1, selected[k]: -1}, hi=0)
                residual_row[selected[k]] = -left
                residual_row[fraction[k]] = -(right-left)
                score_row[selected[k]] = -left*left/scale
                score_row[fraction[k]] = -(right*right-left*left)/scale
            add_row(residual_row, -q.d[j], -q.d[j])
            error += float(np.max(np.diff(knots))**2/4)
        add_row(score_row, hi=(constant+guard)/scale)
        all_knots.append(knots_group); pair_errors.append(error)
    rr, cc, vv = [], [], []
    for i, row in enumerate(rows):
        for j, value in row.items():
            rr.append(i); cc.append(j); vv.append(value)
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(rows), len(lbs))).tocsc()
    objective = np.zeros(len(lbs)); objective[score_index] = -1
    return dict(c=objective, integrality=np.asarray(integer), bounds=Bounds(lbs, ubs),
                constraints=LinearConstraint(matrix, row_lower, row_upper),
                quadratics=quadratics, control_lower=lower.copy(), control_upper=upper.copy(),
                physical_constraints=group, control_dimension=dimension, score_index=score_index,
                scale=scale, numeric_guard=guard, box_upper=upper_score+guard,
                knots=all_knots, secant_error_by_quadratic=np.asarray(pair_errors),
                secant_error_max=max(pair_errors), segments=segments,
                n_variables=len(lbs), n_binary=int(sum(integer)), n_rows=len(rows))


def solve_profile_chord_milp(quadratics, lower, upper, constraints=None, *, segments=4,
                             time_limit=30.0, mip_rel_gap=1e-8, feasible_incumbents=(),
                             feasibility_tolerance=1e-7, presolve=True):
    """Optimize the chord surrogate and return ORIGINAL-score bounds and a feasible u.

    feasible_incumbents are retained feasible controls, not solver warm starts.
    No returned solver status alone certifies exact optimality of the quadratic.
    """
    began = perf_counter()
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('positive finite time limit required')
    if not np.isfinite(mip_rel_gap) or mip_rel_gap < 0:
        raise ValueError('finite nonnegative MIP gap required')
    built = build_profile_chord_milp(quadratics, lower, upper, constraints, segments=segments)
    qs = built['quadratics']; n = built['control_dimension']; candidates = []

    def evaluate(u, origin):
        audit = check_control(u, built['control_lower'], built['control_upper'],
                              built['physical_constraints'], tolerance=feasibility_tolerance)
        if not audit['feasible']:
            return None, audit
        values = np.asarray([q.value(u) for q in qs])
        return dict(u=np.asarray(u, float).copy(), origin=origin, quadratic_values=values,
                    objective=float(min(values)), physical_validation=audit), audit

    for u in feasible_incumbents:
        evaluated, audit = evaluate(u, 'retained_feasible_incumbent')
        if evaluated is None:
            raise ValueError('supplied incumbent violates physical constraints: '+str(audit))
        candidates.append(evaluated)
    result = milp(built['c'], integrality=built['integrality'], bounds=built['bounds'],
                  constraints=built['constraints'],
                  options=dict(time_limit=float(time_limit), mip_rel_gap=float(mip_rel_gap), presolve=bool(presolve)))
    raw_u = None if result.x is None else np.asarray(result.x[:n], float)
    solver_audit = None
    if raw_u is not None:
        evaluated, solver_audit = evaluate(raw_u, 'milp_incumbent')
        if evaluated is not None:
            candidates.append(evaluated)
    best = max(candidates, key=lambda r: r['objective']) if candidates else None
    fun = getattr(result, 'fun', None)
    chord_incumbent = None if fun is None or not np.isfinite(fun) else float(-fun*built['scale'])
    dual = getattr(result, 'mip_dual_bound', None)
    # HiGHS omits MIP fields for a degenerate all-continuous (constant-score) model.
    if dual is None and result.status == 0 and built['n_binary'] == 0:
        dual = fun
    solver_upper = None if dual is None or not np.isfinite(dual) else float(-dual*built['scale']+built['numeric_guard'])
    upper = min(built['box_upper'], solver_upper) if solver_upper is not None else built['box_upper']
    lower_value = None if best is None else best['objective']
    bound_consistent = lower_value is None or upper+10*built['numeric_guard'] >= lower_value
    if not bound_consistent:
        upper = built['box_upper']
    # A tiny negative numerical gap is retained in raw_gap and clipped for reporting.
    raw_gap = None if lower_value is None else float(upper-lower_value)
    point_chord = None
    if best is not None:
        point_chord = min(sum(chord_value(r, nodes) for r, nodes in zip(q.d+q.T@best['u'], knots))-q.offset
                          for q, knots in zip(qs, built['knots']))
    return dict(u=None if best is None else best['u'],
                quadratic_values=None if best is None else best['quadratic_values'],
                objective_lower=lower_value, objective_upper=float(upper),
                original_gap=None if raw_gap is None else max(0.0, raw_gap), raw_gap=raw_gap,
                relative_gap=None if raw_gap is None else max(0.0, raw_gap)/max(1e-12, abs(lower_value), abs(upper)),
                secant_error_max=float(built['secant_error_max']),
                secant_error_by_quadratic=built['secant_error_by_quadratic'],
                chord_value_at_returned_u=point_chord,
                chord_incumbent=chord_incumbent, solver_best_bound=solver_upper,
                box_upper=float(built['box_upper']), numeric_guard=float(built['numeric_guard']),
                status=int(result.status), message=str(result.message), solver_success=bool(result.success),
                time_limit_reached=bool(result.status == 1 and 'time limit' in str(result.message).lower()),
                solver_mip_gap=getattr(result, 'mip_gap', None), solver_nodes=getattr(result, 'mip_node_count', None),
                physical_validation=None if best is None else best['physical_validation'],
                solver_physical_validation=solver_audit, raw_solver_u=raw_u,
                selected_origin=None if best is None else best['origin'],
                bound_consistent=bool(bound_consistent),
                segments=int(segments), n_variables=built['n_variables'], n_binary=built['n_binary'],
                n_rows=built['n_rows'], seconds=perf_counter()-began,
                bound_scope='numerical floating bounds for supplied frozen quadratics and linear feasible set; not nonlinear AC',
                original_global_optimum_certified=False)
