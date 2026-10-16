"""Shared-line tangent projection and a square-chord EV operation MILP.

All bounds refer to a *frozen linear residual model*, not nonlinear AC.  The
assembler owns the history/future Jacobians and their physical interpretation.
Controls are flattened in C order: time first, site second.  No route or action
library, future measurement, or true model is accessed by this module.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import coo_matrix, csr_matrix

from .ev_original_mip_backend import solve_linear_mip


def _finite(value, name, ndim=None):
    a = np.asarray(value, dtype=float)
    if (ndim is not None and a.ndim != ndim) or not np.isfinite(a).all():
        raise ValueError(name + ' must have finite values and the required dimensions')
    return a


def _fingerprint(a):
    return sha256(str(a.shape).encode('ascii') + np.ascontiguousarray(a, dtype='<f8').tobytes()).hexdigest()


def _rank(singular, shape, rtol, atol):
    relative = max(shape, default=1) * np.finfo(float).eps if rtol is None else float(rtol)
    if not np.isfinite([relative, atol]).all() or min(relative, atol) < 0:
        raise ValueError('rank tolerances must be finite and nonnegative')
    threshold = max(float(atol), relative * (float(singular[0]) if len(singular) else 0.))
    return int(np.count_nonzero(singular > threshold)), threshold


@dataclass(frozen=True)
class NuisanceProjector:
    """Implicit P=I-U U.T; columns of U span the declared nuisance matrix."""
    basis: np.ndarray
    matrix_fingerprint: str
    singular_values: np.ndarray
    column_scales: np.ndarray
    rank_threshold: float

    @property
    def rank(self):
        return self.basis.shape[1]

    @property
    def rows(self):
        return self.basis.shape[0]

    def project(self, value):
        value = _finite(value, 'projected value')
        if value.ndim not in (1, 2) or value.shape[0] != self.rows:
            raise ValueError('projected value must share the nuisance row dimension')
        if self.rank == self.rows:
            return np.zeros_like(value)
        return value - self.basis @ (self.basis.T @ value)

    def diagnostics(self):
        return dict(rows=self.rows, rank=self.rank, rank_threshold=self.rank_threshold,
                    singular_values=self.singular_values.copy(), column_scales=self.column_scales.copy(),
                    column_normalization='nonzero J columns normalized before numerical rank selection',
                    scope='projection onto the numerically retained column span of the supplied J')


def build_projector(J, *, rtol=None, atol=0.0):
    """Factor a nuisance matrix once, reusable for every reference against b.

    Column normalization preserves the exact span and avoids ranking R/X and
    source columns solely by their physical units.  Numerical rank is reported;
    a dropped nuisance direction is not an AC identifiability statement.
    """
    J = _finite(J, 'J', 2)
    scales = np.linalg.norm(J, axis=0)
    nonzero = scales > 0
    normalized = J[:, nonzero] / scales[nonzero]
    if normalized.shape[1]:
        U, s, _ = np.linalg.svd(normalized, full_matrices=False)
    else:
        U, s = np.zeros((J.shape[0], 0)), np.zeros(0)
    rank, threshold = _rank(s, normalized.shape, rtol, float(atol))
    return NuisanceProjector(U[:, :rank].copy(), _fingerprint(J), s, scales, threshold)


def project_residual(projector, residual):
    return projector.project(residual)


@dataclass(frozen=True)
class ProjectedPair:
    a_label: str
    b_label: str
    reference_u: np.ndarray
    projected_residual: np.ndarray
    projected_control: np.ndarray
    factor_offset: np.ndarray
    factor_control: np.ndarray
    constant: float
    const_minus: float
    control_remainder: np.ndarray
    control_singular_values: np.ndarray
    control_rank_threshold: float
    projector: NuisanceProjector

    @property
    def control_dimension(self):
        return self.reference_u.size

    @property
    def rank(self):
        return len(self.factor_offset)

    @property
    def control_shape(self):
        return self.reference_u.shape


def build_projected_pair(a_label, b_label, residual, B, J, reference_u, *,
                         const_minus=0.0, projector=None,
                         control_rtol=None, control_atol=0.0):
    """Build .5||P[residual+B(u-uF)]||^2 - const_minus.

    The same cached ``projector`` requires exactly the same supplied J.  The
    control SVD supplies a small affine-square representation.  Exact scoring
    retains full projected B; any dropped control remainder is bounded later.
    Self pairs are allowed here for computing the constant baseline, and are
    excluded from the competing-pair master.
    """
    if not all(isinstance(x, str) and x for x in (a_label, b_label)):
        raise ValueError('nonempty string candidate labels required')
    a, B, J = _finite(residual, 'residual', 1), _finite(B, 'B', 2), _finite(J, 'J', 2)
    u = _finite(reference_u, 'reference_u')
    if u.ndim not in (1, 2) or not u.size or B.shape != (len(a), u.size) or J.shape[0] != len(a):
        raise ValueError('residual, B, J and time-major reference_u dimensions disagree')
    if not np.isfinite(const_minus):
        raise ValueError('const_minus must be finite')
    if projector is None:
        projector = build_projector(J)
    elif not isinstance(projector, NuisanceProjector) or projector.matrix_fingerprint != _fingerprint(J):
        raise ValueError('cached projector must have been built from this J')
    d, C = projector.project(a), projector.project(B)
    U, s, Vh = np.linalg.svd(C, full_matrices=False)
    # Projection roundoff should not manufacture control modes when B lies in
    # the nuisance span.  Any discarded full-C remainder is still bounded in
    # pair_chord_data, and exact scoring continues to use full C.
    projection_floor = np.finfo(float).eps * max(J.shape + B.shape) * float(np.linalg.norm(B))
    rank, threshold = _rank(s, C.shape, control_rtol, max(float(control_atol), projection_floor))
    basis = U[:, :rank]
    alpha = basis.T @ d
    beta = s[:rank, None] * Vh[:rank]
    remainder = C - basis @ beta
    constant = float(.5 * np.sum((d - basis @ alpha) ** 2))
    return ProjectedPair(a_label, b_label, u.copy(), d, C, alpha, beta, constant,
                         float(const_minus), remainder, s, threshold, projector)


def _control(pair, u):
    u = _finite(u, 'control')
    if u.shape not in (pair.control_shape, (pair.control_dimension,)):
        raise ValueError('control must match the time-major reference shape')
    return u.reshape(-1, order='C')


def evaluate_pair(pair, u):
    """Exact attained value for this frozen numerical projector (half SSE)."""
    x = _control(pair, u) - pair.reference_u.ravel(order='C')
    r = pair.projected_residual + pair.projected_control @ x
    return float(.5 * (r @ r) - pair.const_minus)


def _pairs_and_labels(pairs, references=None):
    pairs = tuple(pairs)
    if not pairs or not all(isinstance(p, ProjectedPair) for p in pairs):
        raise ValueError('nonempty ProjectedPair sequence required')
    labels = tuple(dict.fromkeys(p.a_label for p in pairs)) if references is None else tuple(references)
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError('at least two distinct reference candidates required')
    expected = {(a, b) for a in labels for b in labels if a != b}
    actual = [(p.a_label, p.b_label) for p in pairs]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('all K(K-1) directed pairs must be present exactly once')
    shape = pairs[0].control_shape
    if any(p.control_shape != shape for p in pairs):
        raise ValueError('all pairs must share the time-major control shape')
    return pairs, labels


def aggregate_pairs(pairs, u, *, references=None):
    """Mean, across reference models, of their nearest competing half-SSE gap."""
    pairs, labels = _pairs_and_labels(pairs, references)
    values = {(p.a_label, p.b_label): evaluate_pair(p, u) for p in pairs}
    nearest = [dict(a=a, b=min((b for b in labels if b != a), key=lambda b: values[a, b]),
                    half_gap=min(values[a, b] for b in labels if b != a)) for a in labels]
    return dict(score=float(np.mean([r['half_gap'] for r in nearest])), nearest=nearest,
                pairs=[dict(a=a, b=b, half_gap=v) for (a, b), v in values.items()])


def _operation(domain, ncontrol):
    raw = domain.as_dict() if hasattr(domain, 'as_dict') else domain
    if not isinstance(raw, Mapping):
        raise ValueError('operation domain must be as_dict() compatible')
    lo, hi = np.asarray(raw['lower'], float), np.asarray(raw['upper'], float)
    raw_it = np.asarray(raw['integrality'], float)
    if lo.ndim != 1 or not len(lo) or hi.shape != lo.shape or raw_it.shape != lo.shape:
        raise ValueError('operation bound/integrality shapes disagree')
    if (np.isnan(lo).any() or np.isnan(hi).any() or np.isposinf(lo).any() or np.isneginf(hi).any()
            or np.any(lo > hi) or not np.isin(raw_it, [0, 1]).all()):
        raise ValueError('valid operation bounds and continuous/integer types required')
    it = raw_it.astype(int)
    U = csr_matrix(raw['u_map'], dtype=float)
    if U.shape != (ncontrol, len(lo)) or not np.isfinite(U.data).all():
        raise ValueError('finite operation u_map with matching shape required')
    constraints = raw.get('constraints', ())
    if isinstance(constraints, LinearConstraint):
        constraints = [constraints]
    groups = []
    for con in constraints:
        if not isinstance(con, LinearConstraint):
            raise ValueError('operation constraints must be LinearConstraint instances')
        A = csr_matrix(con.A, dtype=float)
        if A.shape[1] != len(lo) or not np.isfinite(A.data).all():
            raise ValueError('operation constraint matrix shape/values invalid')
        lb = np.broadcast_to(np.asarray(con.lb, float), (A.shape[0],)).copy()
        ub = np.broadcast_to(np.asarray(con.ub, float), (A.shape[0],)).copy()
        if (np.isnan(lb).any() or np.isnan(ub).any() or np.isposinf(lb).any()
                or np.isneginf(ub).any() or np.any(lb > ub)):
            raise ValueError('operation constraint bounds invalid')
        groups.append(LinearConstraint(A, lb, ub))
    if 'control_lower' in raw and 'control_upper' in raw:
        low, high = _finite(raw['control_lower'], 'control_lower', 1), _finite(raw['control_upper'], 'control_upper', 1)
    else:
        # Bound each linear control over the operation variable box.  Infinite
        # unrelated coordinates do not contribute through zero coefficients.
        lows, highs = [], []
        for row in range(ncontrol):
            r = U.getrow(row)
            xlo, xhi = lo[r.indices], hi[r.indices]
            lows.append(np.sum(np.where(r.data >= 0, r.data * xlo, r.data * xhi)))
            highs.append(np.sum(np.where(r.data >= 0, r.data * xhi, r.data * xlo)))
        low, high = np.asarray(lows), np.asarray(highs)
    if low.shape != (ncontrol,) or high.shape != low.shape or not np.isfinite([low, high]).all() or np.any(low > high):
        raise ValueError('finite matching control bounds required for square chords')
    # Declared tightened control bounds are part of the actual master domain.
    groups.append(LinearConstraint(U, low, high))
    return dict(lower=lo, upper=hi, integrality=it, U=U, constraints=groups,
                control_lower=low, control_upper=high)


def _audit(x, op, tolerance):
    x = np.asarray(x, float)
    if x.shape != op['lower'].shape or not np.isfinite(x).all():
        return dict(feasible=False, reason='operation_shape_or_nonfinite')
    box = float(max(0., np.max(op['lower'] - x, initial=0.), np.max(x - op['upper'], initial=0.)))
    linear, worst = 0., None
    for group, con in enumerate(op['constraints']):
        v = np.asarray(con.A @ x).ravel()
        violation = np.maximum(np.maximum(con.lb - v, v - con.ub), 0.)
        local = float(np.max(violation, initial=0.))
        if local > linear:
            i = int(np.argmax(violation))
            linear = local
            worst = dict(group=group, row=i, value=float(v[i]), lower=float(con.lb[i]),
                         upper=float(con.ub[i]), violation=local)
    discrete = float(np.max(abs(x[op['integrality'] == 1] - np.rint(x[op['integrality'] == 1])), initial=0.))
    return dict(feasible=max(box, linear, discrete) <= tolerance, box_violation=box,
                linear_violation=linear, integer_violation=discrete, worst_linear=worst,
                tolerance=float(tolerance))


def _named(values, shape, prefix):
    if isinstance(values, Mapping):
        items = list(values.items())
    elif isinstance(values, np.ndarray) and values.shape in (shape, (int(np.prod(shape)),)):
        items = [(prefix + '0', values)]
    else:
        items = [(prefix + str(i), v) for i, v in enumerate(values)]
    if any(not isinstance(name, str) or not name for name, _ in items):
        raise ValueError('initial plan names must be nonempty strings')
    return items


def pair_chord_data(pair, low, high, *, segments=4):
    """Affine-factor ranges and rigorous algebraic square-chord error terms.

    The rank remainder allowance bounds the absolute score change caused by
    control-SVD truncation, uniformly over the declared control box.
    """
    if isinstance(segments, bool) or int(segments) != segments or segments < 1:
        raise ValueError('segments must be a positive integer')
    low, high = _finite(low, 'control lower', 1), _finite(high, 'control upper', 1)
    if low.shape != (pair.control_dimension,) or high.shape != low.shape or np.any(low > high):
        raise ValueError('matching ordered control bounds required')
    center = pair.reference_u.ravel(order='C')
    factors = []
    for alpha, beta in zip(pair.factor_offset, pair.factor_control):
        intercept = float(alpha - beta @ center)
        lower = float(intercept + np.where(beta >= 0, beta * low, beta * high).sum())
        upper = float(intercept + np.where(beta >= 0, beta * high, beta * low).sum())
        grid = np.unique(np.linspace(lower, upper, int(segments) + 1)) if upper > lower else np.array([lower])
        factors.append(dict(intercept=intercept, coefficient=beta.copy(), lower=lower, upper=upper,
                            grid=grid, chord_error=float(np.max(np.diff(grid) ** 2, initial=0.) / 8.)))
    radius = float(np.linalg.norm(np.maximum(abs(low - center), abs(high - center))))
    dropped = float(np.linalg.norm(pair.control_remainder, 2)) if pair.control_remainder.size else 0.
    retained_norm = float(np.linalg.norm(pair.factor_control, 2)) if pair.rank else 0.
    delta = dropped * radius
    rank_error = float((np.linalg.norm(pair.projected_residual) + retained_norm * radius) * delta + .5 * delta ** 2)
    # Upper surrogate = retained-square chord + E_rank.  Since the exact
    # retained-vs-full difference can have either sign, its excess above the
    # full quadratic is bounded by chord_error + 2*E_rank, not just +E_rank.
    error = float(sum(f['chord_error'] for f in factors) + 2. * rank_error)
    box_upper = float(pair.constant - pair.const_minus + rank_error +
                      .5 * sum(max(f['lower'] ** 2, f['upper'] ** 2) for f in factors))
    return dict(a=pair.a_label, b=pair.b_label, constant=pair.constant - pair.const_minus,
                factors=factors, rank=pair.rank, rank_threshold=pair.control_rank_threshold,
                control_singular_values=pair.control_singular_values.copy(),
                nuisance_rank=pair.projector.rank, nuisance_rows=pair.projector.rows,
                nuisance_rank_threshold=pair.projector.rank_threshold,
                rank_remainder_norm=dropped, control_radius=radius, rank_error=rank_error,
                chord_error=sum(f['chord_error'] for f in factors), total_error=error, box_upper=box_upper)


def chord_upper_value(pair, u, chord_data):
    """Evaluate the piecewise linear upper surrogate without a solver."""
    flat = _control(pair, u)
    total = chord_data['constant'] + chord_data['rank_error']
    for f in chord_data['factors']:
        z = float(f['intercept'] + f['coefficient'] @ flat)
        if z < f['lower'] - 1e-9 or z > f['upper'] + 1e-9:
            raise ValueError('control lies outside a declared factor range')
        total += float(np.interp(z, f['grid'], .5 * f['grid'] ** 2))
    return float(total)


def _master(pairs, labels, op, segments):
    n = len(op['lower'])
    lower, upper, integer = list(op['lower']), list(op['upper']), list(op['integrality'])
    rows, rowlow, rowhigh, row_kinds = [], [], [], []

    def var(lo, hi, it=0):
        index = len(lower)
        lower.append(float(lo)); upper.append(float(hi)); integer.append(int(it))
        return index

    def row(coefficients, lo=-np.inf, hi=np.inf, kind='auxiliary'):
        rows.append(coefficients); rowlow.append(float(lo)); rowhigh.append(float(hi)); row_kinds.append(kind)

    for con in op['constraints']:
        A = con.A.tocsr()
        for i in range(A.shape[0]):
            indices = slice(A.indptr[i], A.indptr[i + 1])
            row(dict(zip(A.indices[indices], A.data[indices])), con.lb[i], con.ub[i], kind='operation')
    eta = {a: var(-np.inf, np.inf) for a in labels}
    audits, segment_binaries, mode_variables = [], 0, []
    for pair in pairs:
        audit = pair_chord_data(pair, op['control_lower'], op['control_upper'], segments=segments)
        coefficients = {eta[pair.a_label]: 1.}
        right = audit['constant'] + audit['rank_error']
        for factor in audit['factors']:
            grid = factor['grid']
            if len(grid) == 1:
                right += .5 * float(grid[0] ** 2)
                continue
            lam = [var(0., 1.) for _ in grid]
            select = []
            row({j: 1. for j in lam}, 1., 1.)
            r = np.asarray(factor['coefficient'] @ op['U']).ravel()
            equation = {int(j): float(v) for j, v in enumerate(r) if v}
            equation.update({j: -float(g) for j, g in zip(lam, grid)})
            row(equation, -factor['intercept'], -factor['intercept'], kind='affine_factor_equation')
            if len(grid) > 2:
                select = [var(0., 1., 1) for _ in range(len(grid) - 1)]
                segment_binaries += len(select)
                row({j: 1. for j in select}, 1., 1.)
                for i, j in enumerate(lam):
                    adj = {j: 1.}
                    if i: adj[select[i - 1]] = -1.
                    if i < len(select): adj[select[i]] = -1.
                    row(adj, hi=0.)
            for j, g in zip(lam, grid):
                coefficients[j] = -.5 * float(g ** 2)
            mode_variables.append(dict(factor=factor, lambda_indices=lam, segment_indices=select))
        row(coefficients, hi=right)
        audits.append(audit)
    ri, ci, values = [], [], []
    for i, coefficients in enumerate(rows):
        for j, v in coefficients.items():
            if v: ri.append(i); ci.append(j); values.append(v)
    A = coo_matrix((values, (ri, ci)), shape=(len(rows), len(lower))).tocsr()
    objective = np.zeros(len(lower))
    for j in eta.values(): objective[j] = -1. / len(labels)
    return dict(objective=objective, integrality=np.asarray(integer), bounds=Bounds(lower, upper),
                constraints=LinearConstraint(A, rowlow, rowhigh), pair_audits=audits,
                operation_dimension=n, eta=eta, segment_binaries=segment_binaries, row_kinds=row_kinds,
                mode_variables=mode_variables)


def _reconstruct_auxiliaries(master, op, pairs, x):
    """For a fixed operation x, reconstruct a contiguous chord representation.

    This changes no physical coordinate.  It is useful both for full parent
    starts and for inspecting solver auxiliary residuals.  A separate complete
    matrix audit decides whether the reconstructed representation is feasible.
    """
    point = np.zeros(len(master['objective']))
    point[:master['operation_dimension']] = x
    power = np.asarray(op['U'] @ x).ravel()
    for mode in master['mode_variables']:
        factor, lam, select = mode['factor'], mode['lambda_indices'], mode['segment_indices']
        grid = factor['grid']
        z = float(factor['intercept'] + factor['coefficient'] @ power)
        # Endpoint clipping only repairs arithmetic roundoff in auxiliaries;
        # the subsequent equality audit still checks the unchanged actual z.
        segment = min(max(int(np.searchsorted(grid, z, side='right')) - 1, 0), len(grid) - 2)
        weight = float(np.clip((z - grid[segment]) / (grid[segment + 1] - grid[segment]), 0., 1.))
        point[lam[segment]], point[lam[segment + 1]] = 1. - weight, weight
        if select: point[select[segment]] = 1.
    values = {}
    for pair, audit in zip(pairs, master['pair_audits']):
        values[pair.a_label, pair.b_label] = chord_upper_value(pair, power, audit)
    for a, index in master['eta'].items():
        point[index] = min(v for (reference, _), v in values.items() if reference == a)
    return point


def solve_projected_milp(pairs, domain, initial_powers=(), *, initial_operations=(),
                         references=None, segments=4, solver_seconds=15., total_seconds=None,
                         mip_rel_gap=1e-4, absolute_gap=1e-6, relative_gap=1e-4,
                         feasibility_tolerance=1e-7, backend='gurobi'):
    """Maximize mean nearest gap over one complete EV operation MILP.

    ``initial_powers`` is a name->power mapping or a sequence of time/site
    arrays.  A small feasibility MILP completes each curve to an operation x.
    ``initial_operations`` supplies already known x vectors and avoids that
    cost.  Exact quadratic comparison protects every checked parent.  A solver
    limit may return a valid plan with an unfinished proxy-optimality gap.
    """
    began = perf_counter()
    pairs, labels = _pairs_and_labels(pairs, references)
    if total_seconds is None: total_seconds = solver_seconds
    if not np.isfinite([solver_seconds, total_seconds, mip_rel_gap, absolute_gap, relative_gap,
                        feasibility_tolerance]).all() or min(solver_seconds, total_seconds) <= 0 or min(
                        mip_rel_gap, absolute_gap, relative_gap, feasibility_tolerance) < 0:
        raise ValueError('positive finite budgets and nonnegative tolerances required')
    op = _operation(domain, pairs[0].control_dimension)
    candidates, initials, failures = [], [], []
    shape = pairs[0].control_shape

    def keep(x, origin):
        audit = _audit(x, op, feasibility_tolerance)
        if not audit['feasible']: raise ValueError('infeasible operation: ' + origin)
        x = np.asarray(x, float).copy()
        power = np.asarray(op['U'] @ x).reshape(shape)
        aggregation = aggregate_pairs(pairs, power, references=labels)
        candidate = dict(origin=origin, x=x, power_kw=power, score=aggregation['score'],
                         audit=audit, aggregation=aggregation)
        candidates.append(candidate)
        return candidate

    for name, x in _named(initial_operations, op['lower'].shape, 'operation_'):
        try:
            candidate = keep(x, 'initial_operation:' + name)
            initials.append(dict(origin=candidate['origin'], success=True, exact_score=candidate['score'], solver_calls=0))
        except (ValueError, TypeError) as exc:
            initials.append(dict(origin='initial_operation:' + name, success=False, reason=str(exc), solver_calls=0))
    for name, power in _named(initial_powers, shape, 'power_'):
        record = dict(origin='initial_power:' + name, success=False, solver_calls=0)
        initials.append(record)
        if power is None:
            record['reason'] = 'unavailable_parent'; continue
        flat = _control(pairs[0], power)
        existing = next((c for c in candidates if np.max(abs(c['power_kw'].ravel() - flat), initial=0.) <= 1e-12), None)
        if existing is not None:
            record.update(success=True, exact_score=existing['score'], reason='already_completed_same_power'); continue
        remaining = total_seconds - (perf_counter() - began)
        if remaining <= 0:
            record['reason'] = 'total_budget_exhausted'; continue
        # Identity mapping needs no solver to recover the operation variables.
        identity = op['U'].shape[0] == op['U'].shape[1] and (op['U'] != csr_matrix(np.eye(len(flat)))).nnz == 0
        if identity:
            x = flat
        else:
            completion = solve_linear_mip(np.zeros(len(op['lower'])), op['integrality'],
                Bounds(op['lower'], op['upper']), op['constraints'] + [LinearConstraint(op['U'], flat, flat)],
                time_limit=min(solver_seconds, remaining), mip_rel_gap=mip_rel_gap, backend=backend)
            record.update(solver_calls=1, status=int(completion.status), message=str(completion.message))
            x = completion.x
        if x is None:
            record['reason'] = 'operation_completion_failed'; continue
        try:
            candidate = keep(x, record['origin'])
            difference = float(np.max(abs(candidate['power_kw'].ravel() - flat), initial=0.))
            if difference > feasibility_tolerance:
                candidates.pop(); raise ValueError('completed power differs from the requested parent')
            record.update(success=True, exact_score=candidate['score'], power_difference=difference)
        except ValueError as exc: record['reason'] = str(exc)

    master = _master(pairs, labels, op, segments)
    pair_audits = master['pair_audits']
    interpolation_error = float(np.mean([max(a['total_error'] for a in pair_audits if a['a'] == label) for label in labels]))
    box_upper = float(np.mean([min(a['box_upper'] for a in pair_audits if a['a'] == label) for label in labels]))
    best_initial = max((c['score'] for c in candidates), default=None)
    remaining = total_seconds - (perf_counter() - began)
    result, checked_master_score, raw_master_score = None, None, None
    model_audit, repaired_audit, operation_incumbent_audit = None, None, None
    if remaining > 0:
        start = None
        if candidates:
            start = _reconstruct_auxiliaries(master, op, pairs, max(candidates, key=lambda c: c['score'])['x'])
        result = solve_linear_mip(master['objective'], master['integrality'], master['bounds'],
            master['constraints'], time_limit=min(solver_seconds, remaining), mip_rel_gap=mip_rel_gap,
            start=start, backend=backend)
        if int(result.status) >= 2:
            failures.append(dict(kind='master_solver_failure', status=int(result.status), message=str(result.message)))
        if result.x is not None:
            try:
                # The auxiliaries establish a chord representation, whereas
                # an exact-quadratic attained lower needs only feasible x.
                # Keep both audits: a chord residual never relaxes the physical
                # operation tolerance, and is not called a feasible surrogate.
                model_op = dict(lower=np.asarray(master['bounds'].lb), upper=np.asarray(master['bounds'].ub),
                                integrality=master['integrality'], constraints=[master['constraints']])
                model_audit = _audit(result.x, model_op, feasibility_tolerance)
                if model_audit.get('worst_linear') is not None:
                    worst = model_audit['worst_linear']
                    worst['kind'] = master['row_kinds'][worst['row']]
                    coefficients = master['constraints'].A.getrow(worst['row']).data
                    worst['max_abs_coefficient'] = float(np.max(abs(coefficients), initial=0.))
                operation_incumbent_audit = _audit(result.x[:master['operation_dimension']], op, feasibility_tolerance)
                raw_master_score = float(-master['objective'] @ result.x)
                if model_audit['feasible']:
                    checked_master_score = raw_master_score
                else:
                    failures.append(dict(kind='master_auxiliary_constraint_violation' if operation_incumbent_audit['feasible']
                        else 'master_operation_constraint_violation', audit=model_audit,
                        operation_audit=operation_incumbent_audit))
                if not operation_incumbent_audit['feasible']:
                    raise ValueError('master operation fails its unchanged physical-domain audit')
                repaired = _reconstruct_auxiliaries(master, op, pairs, result.x[:master['operation_dimension']])
                repaired_audit = _audit(repaired, model_op, feasibility_tolerance)
                if repaired_audit['feasible']:
                    checked_master_score = float(-master['objective'] @ repaired)
                elif not model_audit['feasible']:
                    failures.append(dict(kind='auxiliary_reconstruction_failed', audit=repaired_audit))
                keep(result.x[:master['operation_dimension']], 'milp_incumbent')
            except ValueError as exc: failures.append(dict(kind='master_incumbent_rejected', reason=str(exc)))
    else: failures.append(dict(kind='master_skipped_total_budget'))
    winner = max(candidates, key=lambda c: c['score']) if candidates else None
    dual = None if result is None else getattr(result, 'mip_dual_bound', None)
    mip_upper = -float(dual) if dual is not None and np.isfinite(dual) else None
    upper = box_upper if mip_upper is None else min(box_upper, mip_upper)
    lower, gap, completed = None, None, False
    if winner is not None:
        numerical_guard = 512 * np.finfo(float).eps * (1. + abs(winner['score']) + abs(upper))
        lower = winner['score'] - numerical_guard
        upper += numerical_guard
        if upper < lower:
            failures.append(dict(kind='numerical_upper_below_attained_lower', upper=upper, lower=lower))
            upper = None
        else:
            gap = float(upper - lower)
            completed = gap <= absolute_gap + relative_gap * max(1., abs(upper), abs(lower))
    solver = dict(status=None if result is None else int(result.status), message=None if result is None else str(result.message),
                  incumbent_surrogate=checked_master_score, best_bound=mip_upper,
                  raw_incumbent_surrogate=raw_master_score, master_incumbent_audit=model_audit,
                  reconstructed_incumbent_audit=repaired_audit,
                  operation_incumbent_audit=operation_incumbent_audit,
                  backend_requested=backend, backend_used=None if result is None else getattr(result, 'backend_used', None),
                  mip_gap=None if result is None else getattr(result, 'mip_gap', None),
                  start_supplied=False if result is None else bool(getattr(result, 'start_supplied', False)),
                  seconds_available=max(0., remaining))
    return dict(success=winner is not None, power_kw=None if winner is None else winner['power_kw'],
                operation_x=None if winner is None else winner['x'],
                operation=dict(success=False, x=None, audit=dict(feasible=False)) if winner is None else
                    dict(success=True, x=winner['x'], audit=winner['audit']),
                exact_score=None if winner is None else winner['score'],
                aggregation=None if winner is None else winner['aggregation'],
                objective_lower=lower, objective_upper=upper, best_bound=upper, gap=gap,
                mip_best_bound=mip_upper, box_upper=box_upper,
                completed=bool(completed), unfinished=not completed, segments=int(segments),
                selected_origin=None if winner is None else winner['origin'], best_initial_score=best_initial,
                inheritance_protected=winner is not None and (best_initial is None or winner['score'] >= best_initial),
                initials=initials, failures=failures, solver=solver, factor_ranges=pair_audits,
                evaluations=[dict(origin=c['origin'], exact_score=c['score'], power_kw=c['power_kw'],
                                  operation_audit=c['audit']) for c in candidates],
                interpolation_error_bound=interpolation_error,
                n_operation_variables=len(op['lower']), n_master_variables=len(master['objective']),
                n_operation_integer=int(np.count_nonzero(op['integrality'])),
                n_segment_binary=master['segment_binaries'], all_pairs_count=len(pairs),
                reference_labels=list(labels), seconds=perf_counter() - began,
                scope='numerical bounds for the declared frozen shared-nuisance linear projection proxy; not a global AC bound',
                control_order='time-major C order', uses_truth=False, uses_future_noise=False,
                full_curve_enumeration=False)


# Short alternate spelling retained for callers that used the draft API.
solve_projection_milp = solve_projected_milp
