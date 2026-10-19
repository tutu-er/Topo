"""Bounded shared-parameter least squares and independently checked bounds.

For one frozen AC expansion this module solves
    q(u) = min_{z in Z} .5 ||a + B (u-u0) + J z||^2.
Z is the original physical parameter polytope, translated to increments.  It
does not depend on u.  A feasible z gives an UPPER bound.  A residual vector v
and a support-LP dual give a LOWER bound, even if the QP has not converged.
These are numerical bounds on this declared linear model, not on full AC.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from scipy.sparse import csr_matrix


def _finite(value, name, ndim=None):
    value = np.asarray(value, dtype=float)
    if (ndim is not None and value.ndim != ndim) or not np.isfinite(value).all():
        raise ValueError(name + ' must be finite with the declared dimension')
    return value


@dataclass(frozen=True)
class ParameterDomain:
    lower: np.ndarray
    upper: np.ndarray
    A: np.ndarray
    row_lower: np.ndarray
    row_upper: np.ndarray
    anchor: np.ndarray
    metadata: dict

    @property
    def dimension(self):
        return len(self.lower)

    def audit(self, z, tolerance=1e-8):
        z = np.asarray(z, float)
        if z.shape != self.lower.shape or not np.isfinite(z).all():
            return dict(feasible=False, reason='shape_or_nonfinite')
        box = float(max(0., np.max(self.lower-z, initial=0.),
                        np.max(z-self.upper, initial=0.)))
        Az = self.A @ z
        row = float(max(0., np.max(self.row_lower-Az, initial=0.),
                        np.max(Az-self.row_upper, initial=0.)))
        return dict(feasible=max(box, row) <= tolerance, box_violation=box,
                    linear_violation=row, tolerance=float(tolerance))

    def inequalities(self):
        """Return G z <= h, keeping finite sides only; boxes are separate."""
        high, low = np.isfinite(self.row_upper), np.isfinite(self.row_lower)
        return (np.vstack((self.A[high], -self.A[low])),
                np.r_[self.row_upper[high], -self.row_lower[low]])

    def as_dict(self):
        return dict(lower=self.lower.copy(), upper=self.upper.copy(),
                    A=self.A.copy(), row_lower=self.row_lower.copy(),
                    row_upper=self.row_upper.copy(), anchor=self.anchor.copy(),
                    metadata=dict(self.metadata))


def make_domain(lower, upper, constraints=(), *, anchor=None, metadata=None):
    """Make a nonempty finite box/intersection domain, independent of control."""
    lo, hi = _finite(lower, 'lower', 1), _finite(upper, 'upper', 1)
    if lo.shape != hi.shape or np.any(lo > hi):
        raise ValueError('matching ordered finite bounds required')
    if isinstance(constraints, LinearConstraint):
        constraints = [constraints]
    matrices, lbs, ubs = [], [], []
    for con in constraints:
        if not isinstance(con, LinearConstraint):
            raise TypeError('constraints must contain LinearConstraint')
        A = csr_matrix(con.A).toarray()
        if A.ndim != 2 or A.shape[1] != len(lo) or not np.isfinite(A).all():
            raise ValueError('finite parameter constraint matrix dimension required')
        lb = np.broadcast_to(np.asarray(con.lb, float), (len(A),)).copy()
        ub = np.broadcast_to(np.asarray(con.ub, float), (len(A),)).copy()
        if (np.isnan(lb).any() or np.isnan(ub).any() or np.isposinf(lb).any()
                or np.isneginf(ub).any() or np.any(lb > ub)):
            raise ValueError('valid parameter row bounds required')
        matrices.append(A); lbs.append(lb); ubs.append(ub)
    A = np.vstack(matrices) if matrices else np.zeros((0, len(lo)))
    lb = np.concatenate(lbs) if lbs else np.zeros(0)
    ub = np.concatenate(ubs) if ubs else np.zeros(0)
    provisional = ParameterDomain(lo.copy(), hi.copy(), A, lb, ub,
                                  np.zeros(len(lo)), dict(metadata or {}))
    if anchor is None:
        anchor = .5*lo + .5*hi
        if not provisional.audit(anchor, 1e-10)['feasible']:
            G, h = provisional.inequalities()
            result = linprog(np.zeros(len(lo)), A_ub=G if len(G) else None,
                             b_ub=h if len(G) else None, bounds=list(zip(lo, hi)),
                             method='highs')
            if not result.success or result.x is None:
                raise ValueError('physical parameter domain is empty or feasibility solve failed')
            anchor = np.clip(result.x, lo, hi)
    anchor = _finite(anchor, 'anchor', 1)
    if not provisional.audit(anchor, 1e-9)['feasible']:
        raise ValueError('anchor must be feasible in the original parameter domain')
    return ParameterDomain(lo.copy(), hi.copy(), A, lb, ub, anchor.copy(),
                           dict(metadata or {}))


def _broadcast_bounds(bounds, shape, name):
    if len(bounds) != 2:
        raise ValueError(name + ' must be a lower/upper pair')
    lo = np.broadcast_to(_finite(bounds[0], name+' lower'), shape).copy()
    hi = np.broadcast_to(_finite(bounds[1], name+' upper'), shape).copy()
    if np.any(lo > hi):
        raise ValueError(name + ' must be ordered')
    return lo, hi


def physical_delta_domain(r, x, source_center, *, r_bounds, x_bounds,
                          source_bounds, ratio_bounds=None):
    """Translate absolute R/X and same-interface source bounds to increments.

    Coordinate order is [all delta R, all delta X, source.ravel(C)].  Usually
    source_center is (H+E, 2), with time-major [P,Q].  No source location is
    introduced or moved here; the supplied J owns that physical interface.
    Source bounds can be scalar endpoints or arrays broadcast to its shape.
    """
    r, x = _finite(r, 'R', 1), _finite(x, 'X', 1)
    source = _finite(source_center, 'source_center')
    if x.shape != r.shape or not len(r):
        raise ValueError('nonempty matching R/X vectors required')
    rlo, rhi = _broadcast_bounds(r_bounds, r.shape, 'R bounds')
    xlo, xhi = _broadcast_bounds(x_bounds, x.shape, 'X bounds')
    slo, shi = _broadcast_bounds(source_bounds, source.shape, 'source bounds')
    center = np.r_[r, x, source.ravel(order='C')]
    lower = np.r_[rlo, xlo, slo.ravel(order='C')]-center
    upper = np.r_[rhi, xhi, shi.ravel(order='C')]-center
    constraints = []
    if ratio_bounds is not None:
        kmin, kmax = map(float, ratio_bounds)
        if not np.isfinite([kmin, kmax]).all() or not 0 <= kmin <= kmax:
            raise ValueError('finite nonnegative ordered X/R bounds required')
        if np.any(rlo <= 0):
            raise ValueError('positive absolute R lower bound required for X/R constraints')
        m = len(r); A = np.zeros((2*m, len(center)))
        for e in range(m):
            A[e, e] = -kmin; A[e, m+e] = 1.
            A[m+e, e] = kmax; A[m+e, m+e] = -1.
        constraints = [LinearConstraint(A, -A@center, np.inf)]
    metadata = dict(coordinate_order='all delta R, all delta X, time-major delta source(P,Q)',
                    absolute_center=center, line_count=len(r), source_shape=source.shape,
                    ratio_bounds=ratio_bounds, scope='original physical bounds, not a confidence set')
    # A legal trained point gives the natural zero-increment anchor.  If it
    # was at a numerical fitting boundary, make_domain checks rather than
    # silently altering its original physical box.
    zero = np.zeros(len(center))
    try:
        return make_domain(lower, upper, constraints, anchor=zero, metadata=metadata)
    except ValueError:
        return make_domain(lower, upper, constraints, metadata=metadata)


def support_lower(domain, coefficient, *, numerical_guard=1e-10):
    """Return a valid weak-dual lower bound on min_{z in Z} g.T z.

    For Gz<=h and ANY lambda>=0 the bound is
      -lambda.T h + min_box (g+G.T lambda).T z.
    LP marginals merely choose lambda.  Clipping its sign and explicitly
    minimizing the stationarity remainder over the finite box retains weak
    dual validity even when LP status/multipliers are inexact.  The LP primal
    objective is never used as a lower bound.
    """
    began = perf_counter()
    g = _finite(coefficient, 'support coefficient', 1)
    if g.shape != domain.lower.shape or numerical_guard < 0:
        raise ValueError('matching support dimension and nonnegative guard required')
    G, h = domain.inequalities()
    multipliers = np.zeros(len(h)); result = None
    error = None
    if len(h) and len(g):
        scale = max(1., float(np.max(abs(g), initial=0.)))
        try:
            result = linprog(g/scale, A_ub=G, b_ub=h,
                             bounds=list(zip(domain.lower, domain.upper)), method='highs')
            raw = getattr(getattr(result, 'ineqlin', None), 'marginals', None)
            if raw is not None and np.asarray(raw).shape == h.shape and np.isfinite(raw).all():
                multipliers = np.maximum(0., -np.asarray(raw, float)*scale)
        except Exception as exc:  # Box-only lambda=0 remains valid.
            error = type(exc).__name__+': '+str(exc)
    remainder = g + G.T@multipliers
    products = np.where(remainder >= 0, remainder*domain.lower, remainder*domain.upper)
    dual_terms = multipliers*h
    value = float(products.sum()-dual_terms.sum())
    guard = float(numerical_guard)*(1.+float(np.sum(abs(products)))+float(np.sum(abs(dual_terms))))
    return dict(lower=value-guard, raw_lower=value, numerical_guard=guard,
                multipliers=multipliers, stationarity_remainder=remainder,
                LP_status=None if result is None else int(result.status),
                LP_success=None if result is None else bool(result.success),
                LP_primal=None if result is None or result.fun is None else float(result.fun)*scale,
                LP_exception=error, seconds=perf_counter()-began,
                proof='nonnegative inequality multipliers plus exact finite-box remainder support')


@dataclass(frozen=True)
class BoundedPair:
    a_label: str
    b_label: str
    residual: np.ndarray
    B: np.ndarray
    J: np.ndarray
    reference_u: np.ndarray
    domain: ParameterDomain
    self_interval: tuple
    affine_offset: float
    affine_control: np.ndarray
    affine_reference: np.ndarray

    @property
    def control_shape(self):
        return self.reference_u.shape

    @property
    def control_dimension(self):
        return self.reference_u.size


def build_pair(a_label, b_label, residual, B, J, reference_u, domain, *,
               self_interval=(0., 0.), affine_offset=0., affine_control=None,
               affine_reference=None):
    if not all(isinstance(label, str) and label for label in (a_label, b_label)):
        raise ValueError('nonempty string candidate labels required')
    a, B, J = _finite(residual, 'residual', 1), _finite(B, 'B', 2), _finite(J, 'J', 2)
    u = _finite(reference_u, 'reference_u')
    if (u.ndim not in (1, 2) or not u.size or B.shape != (len(a), u.size)
            or J.shape != (len(a), domain.dimension)):
        raise ValueError('residual/B/J/reference/domain dimensions disagree')
    own = tuple(map(float, self_interval))
    if len(own) != 2 or not np.isfinite(own).all() or own[0] > own[1]:
        raise ValueError('ordered finite reference-a self interval required')
    beta = np.zeros(u.size) if affine_control is None else _finite(affine_control, 'affine_control').ravel(order='C')
    aref = u.ravel(order='C') if affine_reference is None else _finite(affine_reference, 'affine_reference').ravel(order='C')
    if beta.shape != (u.size,) or aref.shape != (u.size,) or not np.isfinite(affine_offset):
        raise ValueError('finite matching affine correction coordinates required')
    return BoundedPair(a_label, b_label, a.copy(), B.copy(), J.copy(), u.copy(), domain, own,
                       float(affine_offset), beta.copy(), aref.copy())


def _control(pair, u):
    flat = _finite(u, 'control').ravel(order='C')
    if flat.shape != (pair.control_dimension,):
        raise ValueError('matching time-major control dimension required')
    return flat


def _solve_scipy(H, f, y0, G, h, *, max_iterations, tolerance):
    scale = max(1., float(np.max(abs(H), initial=0.)), float(np.max(abs(f), initial=0.)))
    constraint = () if not len(G) else [LinearConstraint(G, -np.inf, h)]
    result = minimize(lambda y: (.5*y@H@y+f@y)/scale, y0,
                      jac=lambda y: (H@y+f)/scale, method='SLSQP',
                      bounds=Bounds(-np.ones(len(y0)), np.ones(len(y0))),
                      constraints=constraint,
                      options=dict(maxiter=int(max_iterations), ftol=float(tolerance), disp=False))
    return result.x, dict(status=int(result.status), success=bool(result.success),
                          message=str(result.message), iterations=int(result.nit),
                          backend_used='scipy_SLSQP')


def _solve_gurobi(H, f, y0, G, h, *, seconds, tolerance):
    import gurobipy as gp
    from gurobipy import GRB
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0); env.start()
        with gp.Model(env=env) as model:
            model.Params.OutputFlag = 0; model.Params.Threads = 1
            model.Params.Seed = 0; model.Params.TimeLimit = float(seconds)
            model.Params.FeasibilityTol = max(1e-9, min(1e-6, float(tolerance)))
            model.Params.OptimalityTol = max(1e-9, min(1e-6, float(tolerance)))
            model.Params.BarConvTol = max(1e-12, min(1e-8, float(tolerance)))
            y = model.addMVar(len(y0), lb=-1., ub=1.)
            y.Start = y0
            if len(G): model.addMConstr(G, y, '<', h)
            model.setObjective(.5*y@H@y+f@y, GRB.MINIMIZE)
            model.optimize()
            point = None if not model.SolCount else np.asarray(y.X, float).copy()
            return point, dict(status=int(model.Status), success=model.Status == GRB.OPTIMAL,
                               message='Gurobi status '+str(model.Status),
                               iterations=float(model.IterCount),
                               barrier_iterations=float(model.BarIterCount),
                               runtime=float(model.Runtime), backend_used='gurobi')


def solve_pair(pair, u, *, initial_z=None, backend='scipy', seconds=5.,
               max_iterations=500, tolerance=1e-11, absolute_gap=1e-7,
               relative_gap=1e-7, feasibility_tolerance=1e-8):
    """Compute a primal/weak-dual interval and a globally valid affine support.

    The residual price B.T r is an exact gradient only when r is optimal.
    Irrespective of optimality, the returned support_affine is a lower bound
    at every control, because its intercept includes the support LP.
    """
    began = perf_counter(); flat = _control(pair, u)
    if backend not in ('scipy', 'gurobi', 'auto'):
        raise ValueError('backend must be scipy, gurobi or auto')
    if seconds <= 0 or max_iterations < 1 or min(tolerance, absolute_gap, relative_gap,
                                                feasibility_tolerance) < 0:
        raise ValueError('valid positive solve budget and nonnegative tolerances required')
    domain = pair.domain
    c = pair.residual+pair.B@(flat-pair.reference_u.ravel(order='C'))
    z0 = domain.anchor.copy() if initial_z is None else _finite(initial_z, 'initial_z', 1).copy()
    if not domain.audit(z0, feasibility_tolerance)['feasible']:
        raise ValueError('initial_z must be legal in the original parameter domain')
    mid = .5*domain.lower+.5*domain.upper
    span = .5*(domain.upper-domain.lower); free = span > 0
    z = z0.copy(); solver = dict(status=0, success=True, message='no free coordinates', backend_used='none')
    failures = []
    if np.any(free):
        K = pair.J[:, free]*span[free]
        r0 = c+pair.J@mid
        H, f = K.T@K, K.T@r0
        y0 = np.clip((z0[free]-mid[free])/span[free], -1., 1.)
        G0, h0 = domain.inequalities(); G = G0[:, free]*span[free]
        h = h0-G0@mid
        y = None
        if backend in ('gurobi', 'auto'):
            try:
                y, solver = _solve_gurobi(H, f, y0, G, h, seconds=seconds, tolerance=tolerance)
            except Exception as exc:
                failures.append(dict(kind='gurobi_QP_exception', message=type(exc).__name__+': '+str(exc)))
                solver = dict(status=4, success=False, message=str(exc), backend_used='gurobi_failed')
        if backend == 'scipy' or (backend == 'auto' and y is None):
            try:
                y, solver = _solve_scipy(H, f, y0, G, h, max_iterations=max_iterations, tolerance=tolerance)
            except Exception as exc:
                failures.append(dict(kind='scipy_QP_exception', message=type(exc).__name__+': '+str(exc)))
                solver = dict(status=4, success=False, message=str(exc), backend_used='scipy_failed')
        if y is not None and np.isfinite(y).all():
            proposed = mid.copy(); proposed[free] += span[free]*np.clip(y, -1., 1.)
            audit = domain.audit(proposed, feasibility_tolerance)
            if audit['feasible']:
                # Retain the initial legal witness if a stopped solver worsens it.
                if np.linalg.norm(c+pair.J@proposed) <= np.linalg.norm(c+pair.J@z0): z = proposed
            else:
                failures.append(dict(kind='QP_infeasible_explanation_keep_anchor', audit=audit))
        elif np.any(free):
            failures.append(dict(kind='QP_no_incumbent_keep_anchor'))
    residual = c+pair.J@z
    primal = float(.5*(residual@residual))
    support = support_lower(domain, pair.J.T@residual)
    dual_raw = float(residual@c-.5*(residual@residual)+support['lower'])
    cancellation_guard = 64*np.finfo(float).eps*(1.+abs(residual@c)+.5*(residual@residual)+abs(support['lower']))
    dual = max(0., dual_raw-cancellation_guard)
    upper = primal+64*np.finfo(float).eps*(1.+primal)
    gap = upper-dual
    if gap < -1e-8*(1.+upper):
        failures.append(dict(kind='dual_above_feasible_primal', lower=dual, upper=upper))
        dual, gap = 0., upper
    completed = gap <= absolute_gap+relative_gap*max(1., abs(upper))
    raw_price = pair.B.T@residual
    price = raw_price+pair.affine_control
    # Projection onto the convex image JZ gives
    # .5||r-r*||^2 <= primal-q <= primal-dual.  This also covers nonunique z.
    price_error = float(np.linalg.norm(pair.B, 2)*np.sqrt(2*max(0., gap)))
    correction = float(pair.affine_offset+pair.affine_control@(flat-pair.affine_reference))
    affine_intercept = float(residual@(pair.residual-pair.B@pair.reference_u.ravel())
                             -.5*(residual@residual)+support['lower']-cancellation_guard)
    return dict(success=True, completed=bool(completed), unfinished=not completed,
                a=pair.a_label, b=pair.b_label, z=z, residual=residual,
                primal=primal, primal_upper=upper, dual_lower=dual, inner_gap=float(gap),
                score_lower=float(dual-pair.self_interval[1]+correction),
                score_upper=float(upper-pair.self_interval[0]+correction),
                reference_a_self_interval=pair.self_interval,
                price=price, price_exact_within_inner_gap=bool(completed),
                price_error_norm_bound=price_error,
                price_error_scope='||B||_2 sqrt(2 inner_gap), under the fixed-Z numerical bounds',
                raw_price=raw_price, affine_correction=correction,
                raw_support_affine=dict(intercept=affine_intercept, coefficient=raw_price,
                    scope='global affine lower bound for raw q in this frozen fixed-Z model'),
                support_affine=dict(intercept=float(affine_intercept-pair.self_interval[1]
                    +pair.affine_offset-pair.affine_control@pair.affine_reference), coefficient=price,
                    scope='global affine lower bound for corrected q-q_aa in this frozen model'),
                support_LP=support, explanation_audit=domain.audit(z, feasibility_tolerance),
                solver=solver, failures=failures, seconds=perf_counter()-began,
                scope='bounded shared-parameter linear residual QP; not a full AC bound')


def validate_pairs(pairs, references=None):
    pairs = tuple(pairs)
    if not pairs or not all(isinstance(pair, BoundedPair) for pair in pairs):
        raise ValueError('nonempty BoundedPair sequence required')
    labels = tuple(dict.fromkeys(pair.a_label for pair in pairs)) if references is None else tuple(references)
    expected = {(a, b) for a in labels for b in labels if a != b}
    actual = [(p.a_label, p.b_label) for p in pairs]
    if len(labels) < 2 or len(set(labels)) != len(labels) or set(actual) != expected or len(actual) != len(expected):
        raise ValueError('all K(K-1) directed pairs must be present exactly once')
    if any(p.control_shape != pairs[0].control_shape for p in pairs):
        raise ValueError('all pair controls must share shape/order')
    return pairs, labels


def aggregate_solutions(solutions, labels):
    """Order-preserving interval for mean_a min_{b!=a}(q_ab-q_aa)."""
    nearest = []
    for a in labels:
        relevant = [s for s in solutions if s['a'] == a]
        nearest.append(dict(a=a,
            lower=min(s['score_lower'] for s in relevant),
            upper=min(s['score_upper'] for s in relevant),
            lower_competitor=min(relevant, key=lambda s: s['score_lower'])['b'],
            upper_competitor=min(relevant, key=lambda s: s['score_upper'])['b']))
    return dict(lower=float(np.mean([r['lower'] for r in nearest])),
                upper=float(np.mean([r['upper'] for r in nearest])),
                nearest=nearest, completed=all(s['completed'] for s in solutions))


def evaluate_pairs(pairs, u, *, references=None, **options):
    pairs, labels = validate_pairs(pairs, references)
    solutions = [solve_pair(pair, u, **options) for pair in pairs]
    return dict(**aggregate_solutions(solutions, labels), solutions=solutions)
