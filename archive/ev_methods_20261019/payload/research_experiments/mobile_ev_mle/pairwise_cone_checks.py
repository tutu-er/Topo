"""Gate 1 only: exact small-matrix audits of pairwise nonnegative cones.

No AC, candidate selection, EV action enumeration, or policy imports. The four
faces of a two-coordinate nonnegative cone are enumerated to audit a fixed
matrix problem. Float inputs are treated as exact binary rational numbers;
this is an analytic audit implementation, not a deployment optimizer.

For t = A_s zeta - A_r lower and x = eta - lower >= 0,
    min ||t - A_r x + B delta||^2.
A dual q is feasible iff A_r.T q <= 0 and B.T q = 0. Its lower bound is
    2 q.T t - ||q||^2.
Zero q always gives 0. A tolerance-sized positive violation is not feasible
because x and delta have no finite upper bound. Certificates below use exact
rational checks and outward-rounded float bounds, not tolerance feasibility.
"""
from __future__ import annotations

from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import numpy as np


def _array(value, name, ndim):
    if np.iscomplexobj(value):
        raise ValueError(f'{name} must be real')
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if a.ndim != ndim or not np.isfinite(a).all():
        raise ValueError(f'{name} has wrong dimension or nonfinite values')
    return a


def _matrix(value, name, rows=None, cols=None):
    a = _array(value, name, 2)
    if a.shape[0] == 0 or (rows is not None and a.shape[0] != rows) or (cols is not None and a.shape[1] != cols):
        raise ValueError(f'{name} has incompatible shape')
    return a


def _vector(value, name, size, nonnegative=False):
    a = _array(value, name, 1)
    if a.shape != (size,) or (nonnegative and (a < 0).any()):
        raise ValueError(f'{name} has incompatible shape or negative entry')
    return a


def _f(value):
    return F.from_float(float(value))


def _v(a):
    return [_f(x) for x in a]


def _cols(a):
    return [_v(a[:, i]) for i in range(a.shape[1])]


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def _add(a, b, scale=F(1)):
    return [x+scale*y for x, y in zip(a, b)]


def _linear(cols, coeff, n):
    out = [F(0)]*n
    for col, c in zip(cols, coeff):
        out = _add(out, col, c)
    return out


def _solve(matrix, rhs):
    """Exact Gaussian elimination; callers supply an independent Gram matrix."""
    n = len(rhs)
    aug = [list(row)+[v] for row, v in zip(matrix, rhs)]
    for i in range(n):
        pivot = next((k for k in range(i, n) if aug[k][i]), None)
        if pivot is None:
            raise ValueError('singular exact linear system')
        aug[i], aug[pivot] = aug[pivot], aug[i]
        scale = aug[i][i]
        aug[i] = [v/scale for v in aug[i]]
        for k in range(n):
            if k != i:
                scale = aug[k][i]
                aug[k] = [x-scale*y for x, y in zip(aug[k], aug[i])]
    return [row[-1] for row in aug]


def _basis_indices(cols):
    echelon, selected = [], []
    for index, col in enumerate(cols):
        v = list(col)
        for pivot, basis in echelon:
            v = _add(v, basis, -v[pivot])
        pivot = next((i for i, x in enumerate(v) if x), None)
        if pivot is not None:
            selected.append(index)
            echelon.append((pivot, [x/v[pivot] for x in v]))
    return selected


class _Projector:
    """Orthogonal complement of B, with exact rank and no truncated columns."""
    def __init__(self, cols, n):
        self.cols, self.n = cols, n
        self.indices = _basis_indices(cols)
        self.basis = [cols[i] for i in self.indices]
        self.gram = [[_dot(a, b) for b in self.basis] for a in self.basis]

    def project(self, v):
        if not self.basis:
            return list(v), [F(0)]*len(self.cols)
        coefficients = _solve(self.gram, [_dot(b, v) for b in self.basis])
        delta = [F(0)]*len(self.cols)
        for index, coefficient in zip(self.indices, coefficients):
            delta[index] = -coefficient
        return _add(v, _linear(self.cols, delta, self.n)), delta


def _float(value):
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError('exact audit result is outside finite float display range') from exc
    if not math.isfinite(result):
        raise ValueError('exact audit result is outside finite float display range')
    return result


def _rounded(value, upward):
    result = _float(value)
    represented = _f(result)
    if (upward and represented < value) or (not upward and represented > value):
        result = float(np.nextafter(result, math.inf if upward else -math.inf))
    if not math.isfinite(result):
        raise ValueError('outward-rounded bound is nonfinite')
    return result


def _display(v):
    return [_float(x) for x in v]


def _exact(v):
    return [str(x) for x in v]


def _dual(source, ar_cols, b_cols, lower, q):
    t = _add(source, _linear(ar_cols, lower, len(source)), -F(1))
    ar_dot = [_dot(c, q) for c in ar_cols]
    b_dot = [_dot(c, q) for c in b_cols]
    feasible = all(v <= 0 for v in ar_dot) and all(v == 0 for v in b_dot)
    value = 2*_dot(q, t)-_dot(q, q)
    return dict(candidate_feasible_exact=feasible,
        competition_dot=_display(ar_dot), competition_dot_exact=_exact(ar_dot),
        competition_slack=_display([-v for v in ar_dot]),
        nuisance_dot=_display(b_dot), nuisance_dot_exact=_exact(b_dot),
        candidate_value_exact=str(value), candidate_value=_float(value),
        valid_lower_bound=_rounded(value, False) if feasible else 0.,
        valid_lower_bound_source='exact_rational_candidate' if feasible else 'unconditional_zero_dual',
        exact_dual_vector=_exact(q),
        dual_vector_for_valid_bound_exact=_exact(q if feasible else [F(0)]*len(q)),
        valid_lower_bound_exact=str(value) if feasible else '0')


def check_dual_candidate(ar, b, source, q, eta_lower=None):
    """Reject any positive dot or nonzero nuisance dot, however small."""
    ar = _matrix(ar, 'A_r', cols=2);n = ar.shape[0]
    b = _matrix(b, 'B', rows=n)
    source = _vector(source, 'source', n)
    q = _vector(q, 'dual', n)
    lower = _vector(np.zeros(2) if eta_lower is None else eta_lower, 'eta_lower', 2, True)
    return _dual(_v(source), _cols(ar), _cols(b), _v(lower), _v(q))


def pairwise_cone_distance(a_s, a_r, b, zeta, *, eta_lower=None):
    """Exact fixed-matrix optimum, legal float primal, and rigorous dual bound.

    eta_lower>0 is a translated cone with a finite nonnegative lower bound.
    The closure eta>=0 is not interchangeable with this physical lower bound.
    Exact rank is the rank of the given binary-rational input, with no SVD
    truncation or ridge. Very near dependence can produce large compensation;
    the returned coefficients and rank make that issue visible.
    """
    a_s = _matrix(a_s, 'A_s', cols=2);n = a_s.shape[0]
    a_r = _matrix(a_r, 'A_r', rows=n, cols=2)
    b = _matrix(b, 'B', rows=n)
    zeta = _vector(zeta, 'zeta', 2, True)
    lower_float = _vector(np.zeros(2) if eta_lower is None else eta_lower, 'eta_lower', 2, True)
    source = _linear(_cols(a_s), _v(zeta), n)
    ar_cols, b_cols, lower = _cols(a_r), _cols(b), _v(lower_float)
    projection = _Projector(b_cols, n)
    y, _ = projection.project(source)
    c = [projection.project(col)[0] for col in ar_cols]
    shifted = _add(y, _linear(c, lower, n), -F(1))
    candidates = []
    skipped = []
    for count in range(3):
        for free in combinations(range(2), count):
            columns = [c[i] for i in free]
            if len(_basis_indices(columns)) < len(free):
                skipped.append(list(free));continue
            x = [F(0), F(0)]
            values = _solve([[_dot(a, z) for z in columns] for a in columns], [_dot(a, shifted) for a in columns]) if free else []
            if any(v < 0 for v in values):
                continue
            for index, value in zip(free, values):
                x[index] = value
            e = _add(shifted, _linear(c, x, n), -F(1))
            candidates.append((_dot(e, e), x, list(free)))
    optimum, x, free = min(candidates, key=lambda item: item[0])
    eta_exact = _add(lower, x)
    e, delta_exact = projection.project(_add(source, _linear(ar_cols, eta_exact, n), -F(1)))
    dual = _dual(source, ar_cols, b_cols, lower, e)
    assert dual['candidate_feasible_exact'] and _dot(e, e) == optimum
    assert F(dual['candidate_value_exact']) == optimum
    eta, delta = _display(eta_exact), _display(delta_exact)
    # Certify the actually returned binary floats, not just hidden exact values.
    eta_f, delta_f = _v(eta), _v(delta)
    assert all(v >= lo for v, lo in zip(eta_f, lower))
    primal_e = _add(_add(source, _linear(ar_cols, eta_f, n), -F(1)), _linear(b_cols, delta_f, n))
    upper_exact = _dot(primal_e, primal_e)
    upper, lower_bound = _rounded(upper_exact, True), dual['valid_lower_bound']
    slack = [-_dot(col, e) for col in ar_cols]
    x_f = _add(eta_f, lower, -F(1))
    gap_identity = _dot(_add(primal_e, e, -F(1)), _add(primal_e, e, -F(1)))+2*_dot(x_f, slack)
    assert upper_exact-optimum == gap_identity >= 0
    free_projection = _Projector(c, n).project(y)[0]
    binding = [i for i in range(2) if x[i] == 0]
    nonbinding = [i for i in range(2) if x[i] > 0]
    return dict(distance_squared=_float(optimum), distance_squared_exact=str(optimum),
        eta=eta, delta_nu=delta, residual=_display(primal_e),
        exact_eta=_exact(eta_exact), exact_delta_nu=_exact(delta_exact), exact_optimal_residual=_exact(e),
        eta_lower=lower_float.tolist(), domain='closure_nonnegative_cone' if all(v == 0 for v in lower) else 'finite_lower_bound_translated_cone',
        nuisance_rank=len(projection.indices), nuisance_basis_columns=projection.indices,
        projected_competitor_rank=len(_basis_indices(c)), active_face_free_coordinates=free,
        binding_coordinates=binding, nonbinding_coordinates=nonbinding,
        strict_complementarity_exact=all(slack[i] > 0 for i in binding),
        projected_free_competitor_rank=len(_basis_indices([c[i] for i in nonbinding])),
        nuisance_plus_free_rank=len(_basis_indices(b_cols+[ar_cols[i] for i in nonbinding])),
        stable_face_price_requires='constant binding face; fixed zeta/lower; locally constant nuisance and projected-free-column ranks (equivalently [B,A_r,F]); smooth selected optimizer',
        current_free_projection_distance_squared=_float(_dot(y, y)),
        unconstrained_competitor_distance_squared=_float(_dot(free_projection, free_projection)),
        feasible_faces_considered=len(candidates), singular_faces_skipped=skipped,
        primal_upper_bound=upper, primal_upper_bound_exact=str(upper_exact),
        dual_lower_bound=lower_bound, primal_dual_gap=_rounded(_f(upper)-_f(lower_bound), True),
        primal_dual_gap_display=upper-lower_bound,
        exact_primal_dual_gap=str(upper_exact-optimum), exact_gap_identity=str(gap_identity),
        dual=dual, unconditional_zero_dual_lower_bound=0.,
        certificate_scope='Exact binary-rational frozen matrices and unconstrained nuisance; not AC, deployment, or interval uncertainty on input matrices',
        rank_policy='exact rational rank; no nonzero singular value truncated')


def envelope_price(a_s, a_r, b, zeta, d_a_s, d_a_r, d_b, *, eta_lower=None):
    """dD/du = 2 e.T(dA_s zeta - dA_r eta + dB delta_nu).

    Formula is conditional on a stable active face and rank plus a smooth
    optimizer selection. At rank/cone changes this number is not a guarantee
    of a derivative. zeta, lower and the whitening covariance remain fixed.
    For a half-squared-distance objective use half the returned price.
    """
    a_s = _matrix(a_s, 'A_s', cols=2);n = a_s.shape[0]
    a_r = _matrix(a_r, 'A_r', rows=n, cols=2);b = _matrix(b, 'B', rows=n)
    zeta = _vector(zeta, 'zeta', 2, True)
    da = _matrix(d_a_s, 'dA_s', rows=n, cols=2)
    dr = _matrix(d_a_r, 'dA_r', rows=n, cols=2)
    db = _matrix(d_b, 'dB', rows=n, cols=b.shape[1])
    fit = pairwise_cone_distance(a_s, a_r, b, zeta, eta_lower=eta_lower)
    e = [F(x) for x in fit['exact_optimal_residual']]
    eta = [F(x) for x in fit['exact_eta']];delta = [F(x) for x in fit['exact_delta_nu']]
    source_term = 2*_dot(e, _linear(_cols(da), _v(zeta), n))
    competitor_term = -2*_dot(e, _linear(_cols(dr), eta, n))
    nuisance_term = 2*_dot(e, _linear(_cols(db), delta, n))
    price = source_term+competitor_term+nuisance_term
    return dict(price=_float(price), price_exact=str(price), source_term=_float(source_term),
                competitor_term=_float(competitor_term), nuisance_term=_float(nuisance_term),
                fit=fit, derivative_status='conditional_stable_face_formula_only')


def _smooth_case(u):
    a_s = np.array([[1+.2*u, 0.], [1+.1*u, 0.], [2+.3*u, 0.]])
    a_r = np.array([[1., 0.], [.1*u, -1.], [0., 0.]])
    b = np.array([[0.], [.2*u], [1.]])
    return a_s, a_r, b


def run_analytic_checks():
    empty1, empty2 = np.empty((1, 0)), np.empty((2, 0))
    cases = dict(
        positive_cherry_can_copy_signal=pairwise_cone_distance([[1., 0.]], [[1., 0.]], empty1, [1., 0.]),
        positive_competitor_reduces_distance=pairwise_cone_distance([[1., 0.], [1., 0.]], [[1., 0.], [0., 0.]], empty2, [1., 0.]),
        negative_column_must_not_be_free=pairwise_cone_distance([[1., 0.], [1., 0.]], [[-1., 0.], [0., 0.]], empty2, [1., 0.]),
        common_rank_deficient_compensation=pairwise_cone_distance(np.eye(2), [[1., 0.], [0., 0.]], [[0., 0.], [1., 2.]], [1., 1.]),
        closure=pairwise_cone_distance(np.eye(2), np.eye(2), empty2, [0., 1.]),
        strictly_positive_copy=pairwise_cone_distance(np.eye(2), np.eye(2), empty2, [.003, .002], eta_lower=[1e-6, 1e-6]),
        finite_positive_lower_bound=pairwise_cone_distance(np.eye(2), np.eye(2), empty2, [0., 1.], eta_lower=[.2, .3]),
        zero_competitor_columns=pairwise_cone_distance(np.eye(2), np.zeros((2, 2)), empty2, [1., 2.]),
        opposite_competitor_columns=pairwise_cone_distance([[-1., 0.], [1., 0.]], [[1., -1.], [0., 0.]], empty2, [1., 0.]),
        tiny_nonzero_nuisance_is_kept=pairwise_cone_distance(np.eye(2), np.zeros((2, 2)), [[1e-14, 0.], [0., 1.]], [1., 2.]))
    expected = {'positive_cherry_can_copy_signal': 0., 'positive_competitor_reduces_distance': 1., 'negative_column_must_not_be_free': 2., 'common_rank_deficient_compensation': 0., 'closure': 0., 'strictly_positive_copy': 0., 'finite_positive_lower_bound': .04, 'zero_competitor_columns': 5., 'opposite_competitor_columns': 1., 'tiny_nonzero_nuisance_is_kept': 0.}
    for name, value in expected.items():
        assert abs(cases[name]['distance_squared']-value) < 1e-14
    a_s, a_r, b = _smooth_case(0.)
    price = envelope_price(a_s, a_r, b, [1., 0.], [[.2, 0.], [.1, 0.], [.3, 0.]], [[0., 0.], [.1, 0.], [0., 0.]], [[0.], [.2], [0.]])
    finite = []
    for h in (1e-3, 5e-4, 1e-4):
        minus = pairwise_cone_distance(*_smooth_case(-h), [1., 0.])
        plus = pairwise_cone_distance(*_smooth_case(h), [1., 0.])
        fd = (plus['distance_squared']-minus['distance_squared'])/(2*h)
        finite.append(dict(step=h, finite_difference=fd, price_error=fd-price['price'], minus_binding=minus['binding_coordinates'], plus_binding=plus['binding_coordinates'], minus_nuisance_rank=minus['nuisance_rank'], plus_nuisance_rank=plus['nuisance_rank']))
        assert abs(fd-price['price']) < 2e-7 and minus['binding_coordinates'] == plus['binding_coordinates'] == [1]
    invalid = check_dual_candidate([[1., 0.]], empty1, [1.], [1e-12])
    assert not invalid['candidate_feasible_exact'] and invalid['candidate_value'] > 0 and invalid['valid_lower_bound'] == 0
    return dict(stage='gate_1_analytic_only', ac_calls=0, actual_EV_actions_selected=0,
        fixed_two_coordinate_QP_faces_only=True, cases=cases,
        counterexample_summary=dict(current_positive_but_competitor_zero=[1., 0.], unconstrained_competitor_wrongly_eliminates_negative_direction=[2., 1.]),
        invalid_tolerance_dual=invalid, smooth_price=price, finite_differences=finite,
        derivative_scope='stable binding face and relevant ranks only; rank/cone changes audited separately in tests',
        bound_scope='exact rational primal/dual for fixed binary-rational input; not nonlinear AC or outer MILP bounds',
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('pairwise_cone_checks.py', 'test_pairwise_cone_checks.py')},
        runtime=dict(python=platform.python_version(), numpy=np.__version__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run_analytic_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(stage=result['stage'], checks=len(result['cases']), output=str(args.output), ac_calls=0), ensure_ascii=False))
