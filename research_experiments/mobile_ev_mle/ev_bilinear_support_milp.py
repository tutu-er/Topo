"""Finite-direction MILP lower master preserving control-times-parameter terms.

Residual: r(u,z)=c+B@u+(A0+sum_t u[t]*At[t])@z, z in a fixed box.
The master maximizes a guaranteed directional distance LOWER approximation.
Its global best bound is not an upper bound for the true distance optimum.
This independent analytic prototype does not call or certify nonlinear AC.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, milp
from scipy.sparse import coo_matrix, issparse
from .ev_profile_chord_milp import _constraints, check_control


@dataclass(frozen=True)
class BilinearPair:
    name: str
    c: np.ndarray
    B: np.ndarray
    A0: np.ndarray
    At: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        for key in ('c', 'B', 'A0', 'At', 'lower', 'upper'):
            value = np.array(getattr(self, key), float, copy=True)
            if not np.isfinite(value).all():
                raise ValueError('finite bilinear arrays required')
            value.setflags(write=False); object.__setattr__(self, key, value)
        if (self.c.ndim != 1 or self.B.ndim != 2 or self.A0.ndim != 2
                or self.B.shape[0] != len(self.c) or self.A0.shape[0] != len(self.c)
                or self.B.shape[1] < 1 or self.At.shape != (self.B.shape[1], *self.A0.shape)
                or self.lower.shape != (self.A0.shape[1],) or self.upper.shape != self.lower.shape
                or np.any(self.lower > self.upper)):
            raise ValueError('inconsistent bilinear dimensions or nuisance bounds')

    @property
    def control_dimension(self):
        return self.B.shape[1]

    def matrix(self, u):
        u = np.asarray(u, float)
        if u.shape != (self.control_dimension,) or not np.isfinite(u).all():
            raise ValueError('finite control vector of matching dimension required')
        return self.A0 + np.einsum('t,tmn->mn', u, self.At)

    def residual(self, u, z):
        return self.c+self.B@u+self.matrix(u)@z


def unit_direction(v):
    v = np.asarray(v, float)
    if v.ndim != 1 or not np.isfinite(v).all():
        raise ValueError('finite one-dimensional direction required')
    norm = float(np.linalg.norm(v))
    return v.copy() if norm == 0 else v/norm


def support_coefficients(pair, direction):
    v = np.asarray(direction, float)
    if v.shape != pair.c.shape or not np.isfinite(v).all():
        raise ValueError('direction dimension or finiteness invalid')
    norm = float(np.linalg.norm(v))
    if norm > 1+1e-12:
        raise ValueError('direction norm must be at most one; use unit_direction explicitly')
    if norm > 1:
        v = v/norm
    middle, radius = (pair.lower+pair.upper)/2, (pair.upper-pair.lower)/2
    a0 = v@pair.A0
    au = np.einsum('m,tmn->nt', v, pair.At)
    intercept = float(v@pair.c+a0@middle)
    slope = v@pair.B+middle@au
    return dict(direction=v, a0=a0, au=au, intercept=intercept, slope=slope, radius=radius)


def support_value(pair, direction, u):
    x = support_coefficients(pair, direction); u = np.asarray(u, float)
    return float(x['intercept']+x['slope']@u-x['radius']@np.abs(x['a0']+x['au']@u))


def _range(c, B, lower, upper):
    return c+np.minimum(B*lower, B*upper).sum(axis=-1), c+np.maximum(B*lower, B*upper).sum(axis=-1)


def fixed_witness_upper(pair, lower, upper, z=None):
    """A true-distance outer upper bound from one globally feasible fixed witness."""
    z = (pair.lower+pair.upper)/2 if z is None else np.asarray(z, float)
    if z.shape != pair.lower.shape or not np.isfinite(z).all() or np.any(z<pair.lower) or np.any(z>pair.upper):
        raise ValueError('fixed nuisance witness must be feasible throughout control domain')
    c = pair.c+pair.A0@z
    B = pair.B+np.einsum('tmn,n->mt', pair.At, z)
    a, b = _range(c, B, lower, upper)
    value = float(np.sqrt(np.sum(np.maximum(a*a, b*b))))
    guard = 128*np.finfo(float).eps*(1+value)
    return value+guard


def evaluate_qp(pair, u, *, tolerance=1e-11):
    """Full fixed-u convex box QP, primal upper and tangent lower numerical bounds."""
    A, c = pair.matrix(u), pair.c+pair.B@u
    fixed = pair.lower == pair.upper
    z = np.clip(np.zeros(len(pair.lower)), pair.lower, pair.upper)
    free = ~fixed
    if free.any():
        solved = lsq_linear(A[:, free], -(c+A[:, fixed]@z[fixed]),
            bounds=(pair.lower[free], pair.upper[free]), method='bvls', tol=tolerance, max_iter=1000)
        z[free] = np.clip(solved.x, pair.lower[free], pair.upper[free])
        success, status, optimality = bool(solved.success), int(solved.status), float(solved.optimality)
    else:
        success, status, optimality = True, 3, 0.
    r = c+A@z; primal = float(r@r); gradient = 2*A.T@r
    correction = float(np.minimum(gradient*(pair.lower-z), gradient*(pair.upper-z)).sum())
    guard = 256*np.finfo(float).eps*(1+abs(primal)+abs(correction))
    lower = max(0., min(primal, primal+correction-guard))
    return dict(success=success, status=status, z=z, residual=r, q_primal_upper=primal,
                q_tangent_lower=lower, numerical_gap=primal-lower,
                distance_primal_upper=float(np.sqrt(primal)), distance_tangent_lower=float(np.sqrt(lower)),
                optimality=optimality, numeric_guard=guard)


def solve_bilinear_support_milp(pairs, directions, control_lower, control_upper,
                               constraints=None, *, time_limit=10., mip_rel_gap=1e-8,
                               feasibility_tolerance=1e-7):
    began = perf_counter(); pairs = list(pairs)
    low, high = np.asarray(control_lower, float), np.asarray(control_upper, float)
    if (not pairs or low.ndim != 1 or len(low) < 1 or high.shape != low.shape
            or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low>high)
            or any(p.control_dimension != len(low) for p in pairs) or len(directions) != len(pairs)):
        raise ValueError('matching pairs/directions and finite ordered control box required')
    if not np.isfinite(time_limit) or time_limit <= 0 or not np.isfinite(mip_rel_gap) or mip_rel_gap < 0:
        raise ValueError('invalid MILP time or gap budget')
    physical = _constraints(constraints, len(low))
    groups = []
    for pair, raw in zip(pairs, directions):
        group = [support_coefficients(pair, v) for v in raw]
        if not any(np.all(x['direction'] == 0) for x in group):
            group.append(support_coefficients(pair, np.zeros_like(pair.c)))
        groups.append(group)
    witness_uppers = [fixed_witness_upper(p, low, high) for p in pairs]
    tupper = min(witness_uppers)
    scale = max(1e-8, tupper)
    n = len(low); tid = n
    lbs, ubs, integer = list(low)+[0.], list(high)+[tupper/scale], [0]*(n+1)
    rows, row_low, row_high, audits = [], [], [], []
    def variables(count, upper=1., discrete=False):
        start = len(lbs); lbs.extend([0.]*count); ubs.extend(np.broadcast_to(upper, (count,)).tolist())
        integer.extend([int(discrete)]*count)
        return np.arange(start, start+count)
    def row(values, lo=-np.inf, hi=np.inf):
        rows.append({int(k):float(v) for k,v in values.items() if v != 0}); row_low.append(lo); row_high.append(hi)
    for c in physical:
        A = c.A.tocsr() if issparse(c.A) else c.A
        for j in range(A.shape[0]):
            if issparse(A):
                r = A.getrow(j); values = dict(zip(r.indices, r.data))
            else: values = dict(enumerate(A[j]))
            row(values, c.lb[j], c.ub[j])
    for pair, group in zip(pairs, groups):
        chosen = variables(len(group), discrete=True)
        row(dict(zip(chosen, np.ones(len(group)))), 1, 1)
        for j, x in enumerate(group):
            a_lo, a_hi = _range(x['a0'], x['au'], low, high)
            wupper = np.maximum(abs(a_lo), abs(a_hi))
            zero_direction = bool(np.all(x['direction'] == 0))
            if not zero_direction:
                wupper = wupper + 128*np.finfo(float).eps*(1+wupper)
            base_lo, _ = _range(x['intercept'], x['slope'], low, high)
            ell_min = float(base_lo-x['radius']@wupper)
            guard = 0. if zero_direction else 256*np.finfo(float).eps*(1+abs(ell_min)+tupper+float(wupper.sum()))
            M = max(0., tupper-ell_min)+guard
            w = variables(len(wupper), upper=wupper)
            for i in range(len(w)):
                r = {int(k):float(v) for k,v in enumerate(x['au'][i])}; r[int(w[i])] = -1.
                row(r, hi=-x['a0'][i])
                r = {int(k):-float(v) for k,v in enumerate(x['au'][i])}; r[int(w[i])] = -1.
                row(r, hi=x['a0'][i])
            r = {tid:1., int(chosen[j]):M/scale}
            r.update({int(k):-float(v)/scale for k,v in enumerate(x['slope']) if v})
            r.update({int(k):float(v)/scale for k,v in zip(w, x['radius']) if v})
            # Downward support guard retains the lower-bound direction.
            row(r, hi=(x['intercept']+M-guard)/scale)
            audits.append(dict(pair=pair.name, direction=x['direction'], ell_box_lower=ell_min,
                               abs_upper=wupper, big_m=M, numeric_guard=guard))
    ri, ci, data = [], [], []
    for i, r in enumerate(rows):
        for j,v in r.items():ri.append(i);ci.append(j);data.append(v)
    A = coo_matrix((data,(ri,ci)),shape=(len(rows),len(lbs))).tocsc()
    objective = np.zeros(len(lbs)); objective[tid] = -1.
    result = milp(objective, integrality=integer, bounds=Bounds(lbs,ubs),
                  constraints=LinearConstraint(A,row_low,row_high),
                  options=dict(time_limit=float(time_limit),mip_rel_gap=float(mip_rel_gap)))
    u = None if result.x is None else np.asarray(result.x[:n],float)
    physical_audit = None if u is None else check_control(u,low,high,physical,tolerance=feasibility_tolerance)
    if physical_audit is not None and not physical_audit['feasible']:u=None
    evaluated = [] if u is None else [evaluate_qp(p,u) for p in pairs]
    pair_lower = [] if u is None else [max(0.,max(support_value(p,x['direction'],u) for x in group)) for p,group in zip(pairs,groups)]
    bestbound = getattr(result,'mip_dual_bound',None)
    master_bound = None if bestbound is None or not np.isfinite(bestbound) else float(-bestbound*scale)
    return dict(u=u,status=int(result.status),message=str(result.message),solver_success=bool(result.success),
        distance_lower=None if u is None else float(min(pair_lower)), pair_distance_lower=pair_lower,
        qp_evaluations=evaluated,
        evaluated_distance_primal_upper=None if u is None else min(x['distance_primal_upper'] for x in evaluated),
        lower_master_incumbent=None if result.fun is None else float(-result.fun*scale),
        lower_master_best_bound=master_bound,
        lower_master_best_bound_is_true_distance_upper=False,
        fixed_witness_global_upper=tupper, fixed_witness_pair_uppers=witness_uppers,
        physical_validation=physical_audit, direction_audits=audits,
        n_variables=len(lbs),n_binary=int(sum(integer)),n_rows=len(rows),seconds=perf_counter()-began,
        scope='finite-direction lower master for declared bilinear residual and fixed nuisance box; numerical floating bounds; no AC certificate')
