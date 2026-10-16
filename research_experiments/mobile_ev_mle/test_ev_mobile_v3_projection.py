"""Independent small algebra/domain audits; no power flow or future data."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import LinearConstraint

from . import ev_mobile_v3_projection as v3


def _pair(a, b, residual, B, *, J=None, center=None, minus=0., **options):
    B = np.asarray(B, float)
    if J is None: J = np.zeros((B.shape[0], 0))
    if center is None: center = np.zeros((1, B.shape[1]))
    return v3.build_projected_pair(a, b, residual, B, J, center, const_minus=minus, **options)


def _box(low, high, *, integer=None, constraints=()):
    low, high = np.asarray(low, float), np.asarray(high, float)
    return dict(lower=low, upper=high, integrality=np.zeros(len(low), int) if integer is None else integer,
                constraints=constraints, u_map=np.eye(len(low)), control_lower=low, control_upper=high)


def test_projection_equals_direct_joint_least_squares_and_cache():
    rng = np.random.default_rng(4)
    J = rng.normal(size=(9, 5))
    J[:, 4] = J[:, 0] + J[:, 1]
    # Physical column units should not change the projected span.
    J = J * np.array([1e-5, 2e3, .2, 4., 5.])
    residual, B, center = rng.normal(size=9), rng.normal(size=(9, 4)), rng.normal(size=(2, 2))
    projector = v3.build_projector(J)
    pair = v3.build_projected_pair('a', 'b', residual, B, J, center, projector=projector, const_minus=.7)
    assert pair.projector is projector
    assert projector.rank == 4
    for _ in range(10):
        u = rng.normal(size=(2, 2))
        target = residual + B @ (u - center).ravel()
        scales = np.linalg.norm(J, axis=0)
        z = np.linalg.lstsq(J / scales, -target, rcond=None)[0]
        direct = .5 * np.linalg.norm(target + (J / scales) @ z) ** 2 - .7
        assert v3.evaluate_pair(pair, u) == pytest.approx(direct, abs=2e-12)
    with pytest.raises(ValueError, match='cached projector'):
        v3.build_projected_pair('a', 'b', residual, B, J + .01, center, projector=projector)


def test_shared_line_preserves_temporal_excitation_independent_lines_erase_it():
    shared = _pair('a', 'b', [0., 0.], np.eye(2), J=-np.ones((2, 1)))
    independent = _pair('a', 'b', [0., 0.], np.eye(2), J=-np.eye(2))
    alternating, flat = np.array([[1., -1.]]), np.array([[0., 0.]])
    assert v3.evaluate_pair(shared, alternating) == pytest.approx(1.)
    assert v3.evaluate_pair(shared, flat) == 0.
    assert abs(v3.evaluate_pair(independent, alternating)) < 1e-28


def test_self_constant_and_signed_nearest_all_competitors():
    own = _pair('a', 'a', [1., 2., 3.], np.zeros((3, 2)), J=np.array([[1.], [1.], [0.]]))
    assert own.rank == 0
    assert v3.evaluate_pair(own, [[100., -30.]]) == pytest.approx(v3.evaluate_pair(own, [[0., 0.]]))
    pairs = [_pair(a, b, [0.], [[1.]], minus=3. if a == 'a' else 2.)
             for a in ['a', 'b', 'c'] for b in ['a', 'b', 'c'] if a != b]
    result = v3.aggregate_pairs(pairs, [[0.]])
    assert result['score'] == pytest.approx(-7 / 3)
    assert len(result['nearest']) == 3
    with pytest.raises(ValueError, match=r'K\(K-1\)'):
        v3.aggregate_pairs(pairs[:-1], [[0.]])
    with pytest.raises(ValueError, match=r'K\(K-1\)'):
        v3.aggregate_pairs(pairs + pairs[:1], [[0.]])


@pytest.mark.parametrize('constant', [-4., 2.])
def test_chords_with_both_residual_signs_and_negative_offsets(constant):
    pair = _pair('a', 'b', [constant, -1.], [[2., -.3], [0., .8]], minus=12.)
    low, high = np.array([-2., -.5]), np.array([1., 2.])
    data = v3.pair_chord_data(pair, low, high, segments=4)
    for x in np.linspace(low[0], high[0], 11):
        for y in np.linspace(low[1], high[1], 11):
            u = np.array([[x, y]])
            exact, upper = v3.evaluate_pair(pair, u), v3.chord_upper_value(pair, u, data)
            assert upper >= exact - 1e-11
            assert upper - exact <= data['total_error'] + 1e-11


def test_control_svd_truncation_allowance_is_an_upper_bound():
    # Deliberately drop the small second direction; d has a component along it.
    pair = _pair('a', 'b', [2., 100.], [[3., 0.], [0., .02]], control_atol=.03)
    assert pair.rank == 1
    data = v3.pair_chord_data(pair, [-2., -4.], [2., 3.], segments=3)
    assert data['rank_error'] > 0
    for u in np.random.default_rng(2).uniform([-2., -4.], [2., 3.], size=(50, 2)):
        assert v3.chord_upper_value(pair, u, data) >= v3.evaluate_pair(pair, u) - 1e-9


def test_zero_gradient_origin_does_not_stop_the_global_proxy_milp():
    pairs = [_pair(a, b, [0.], [[np.sqrt(2.)]]) for a, b in [('a', 'b'), ('b', 'a')]]
    out = v3.solve_projected_milp(pairs, _box([-1.], [1.]), {'parent': np.array([[0.]])},
        segments=4, solver_seconds=5., backend='scipy')
    assert out['success']
    assert out['exact_score'] == pytest.approx(1.)
    assert abs(out['power_kw'][0, 0]) == pytest.approx(1.)
    assert out['best_bound'] >= out['exact_score'] - 1e-10
    assert out['gap'] < 1e-8
    assert out['completed']
    assert out['inheritance_protected']


def test_tiny_binary_audit_all_pairs_mean_nearest_and_negative_eta():
    pairs = [
        _pair('a', 'b', [-1.], [[2., -.2]], minus=3.),
        _pair('a', 'c', [.5], [[-.5, .8]], minus=3.),
        _pair('b', 'a', [-.7], [[.2, -.8]], minus=1.),
        _pair('b', 'c', [.2], [[.6, .2]], minus=1.),
        _pair('c', 'a', [-1.], [[1., .1]], minus=2.),
        _pair('c', 'b', [2.], [[-.4, .5]], minus=2.),
    ]
    domain = _box([0., 0.], [1., 1.], integer=[1, 1], constraints=[LinearConstraint([[1., 1.]], 1., 1.)])
    possible = [np.array([[0., 1.]]), np.array([[1., 0.]])]
    audited = max(v3.aggregate_pairs(pairs, u)['score'] for u in possible)
    out = v3.solve_projected_milp(pairs, domain, {'parent': possible[0]}, segments=8, backend='scipy', solver_seconds=5.)
    assert out['success'] and out['all_pairs_count'] == 6
    assert out['exact_score'] == pytest.approx(audited)
    assert out['best_bound'] >= audited - 1e-10
    assert out['best_bound'] - out['exact_score'] <= out['interpolation_error_bound'] + 1e-8
    assert out['exact_score'] < 0


def test_nonidentity_operation_lifting_and_time_major_power():
    # x contains a non-control integer plus four time-major control variables.
    U = np.column_stack([np.zeros(4), np.eye(4)])
    domain = dict(lower=np.r_[0., -np.ones(4)], upper=np.ones(5), integrality=[1, 0, 0, 0, 0],
                  u_map=U, control_lower=-np.ones(4), control_upper=np.ones(4),
                  constraints=[LinearConstraint([[1., 0., 0., 0., 0.]], 1., 1.)])
    center = np.zeros((2, 2))
    pairs = [_pair(a, b, [0.], [[1., 2., 3., 4.]], center=center) for a, b in [('a', 'b'), ('b', 'a')]]
    parent = np.array([[.1, .2], [.3, .4]])
    out = v3.solve_projected_milp(pairs, domain, {'parent': parent}, segments=1, backend='scipy', solver_seconds=5.)
    assert out['success'] and out['initials'][0]['success']
    assert out['initials'][0]['exact_score'] == pytest.approx(.5 * 3. ** 2)
    assert np.array_equal(out['operation_x'][1:].reshape(2, 2), out['power_kw'])
    assert out['operation_x'][0] == 1.


def test_parent_is_protected_if_backend_fails(monkeypatch):
    pairs = [_pair(a, b, [0.], [[np.sqrt(2.)]]) for a, b in [('a', 'b'), ('b', 'a')]]
    monkeypatch.setattr(v3, 'solve_linear_mip', lambda *a, **k:
        SimpleNamespace(x=None, status=4, message='mock unavailable', mip_dual_bound=None))
    out = v3.solve_projected_milp(pairs, _box([-1.], [1.]),
        initial_operations={'parent': np.array([.7])}, backend='scipy', solver_seconds=5.)
    assert out['success'] and out['inheritance_protected']
    assert out['exact_score'] == pytest.approx(.49)
    assert out['selected_origin'] == 'initial_operation:parent'
    assert out['solver']['status'] == 4
    assert out['failures'][0]['kind'] == 'master_solver_failure'
    assert out['best_bound'] >= .49


def test_constant_signed_master_has_no_segment_variables():
    pairs = [_pair(a, b, [1.], [[0.]], minus=2.) for a, b in [('a', 'b'), ('b', 'a')]]
    out = v3.solve_projected_milp(pairs, _box([-1.], [1.]), segments=4, backend='scipy', solver_seconds=5.)
    assert out['success'] and out['exact_score'] == pytest.approx(-1.5)
    assert out['n_segment_binary'] == 0
    assert out['gap'] < 1e-8


def test_invalid_dimensions_domains_and_segments_fail_explicitly():
    with pytest.raises(ValueError, match='dimensions disagree'):
        _pair('a', 'b', [0., 1.], [[1.]])
    pair = _pair('a', 'b', [0.], [[1.]])
    with pytest.raises(ValueError, match='positive integer'):
        v3.pair_chord_data(pair, [-1.], [1.], segments=0)
    with pytest.raises(ValueError, match='outside'):
        v3.chord_upper_value(pair, [[2.]], v3.pair_chord_data(pair, [-1.], [1.]))
    pairs = [pair, _pair('b', 'a', [0.], [[1.]])]
    with pytest.raises(ValueError, match='finite'):
        v3.solve_projected_milp(pairs, _box([-np.inf], [np.inf]), backend='scipy')
    with pytest.raises(ValueError, match='continuous/integer'):
        v3.solve_projected_milp(pairs, _box([-1.], [1.], integer=[.5]), backend='scipy')


def test_truncated_square_error_bound_covers_upper_excess_in_both_directions():
    pair = _pair('a', 'b', [2., 8.], [[3., 0.], [0., .02]], control_atol=.03)
    data = v3.pair_chord_data(pair, [-2., -4.], [2., 3.], segments=4)
    for x in np.linspace(-2., 2., 13):
        for y in np.linspace(-4., 3., 13):
            u = [[x, y]]
            excess = v3.chord_upper_value(pair, u, data) - v3.evaluate_pair(pair, u)
            assert excess >= -1e-10
            assert excess <= data['total_error'] + 1e-10


def test_operation_bound_completion_can_use_a_supplied_parent_without_extra_solve(monkeypatch):
    pairs = [_pair(a, b, [0.], [[1.]]) for a, b in [('a', 'b'), ('b', 'a')]]
    calls = []
    def no_incumbent(*a, **k):
        calls.append(k)
        return SimpleNamespace(x=None, status=1, message='mock limit', mip_dual_bound=-1.)
    monkeypatch.setattr(v3, 'solve_linear_mip', no_incumbent)
    domain = dict(lower=[0., -1.], upper=[1., 1.], integrality=[1, 0], u_map=[[0., 1.]],
                  control_lower=[-1.], control_upper=[1.], constraints=[])
    out = v3.solve_projected_milp(pairs, domain, {'power': np.array([[.5]])},
        initial_operations={'known_operation': np.array([1., .5])}, backend='scipy', solver_seconds=5.)
    assert out['success'] and len(calls) == 1
    assert out['initials'][1]['reason'] == 'already_completed_same_power'
    assert out['best_bound'] >= out['exact_score']


def test_auxiliary_violation_keeps_strictly_feasible_exact_quadratic_plan(monkeypatch):
    pairs = [_pair(a, b, [0.], [[np.sqrt(2.)]]) for a, b in [('a', 'b'), ('b', 'a')]]
    real = v3.solve_linear_mip
    def imperfect_auxiliary(*args, **kwargs):
        result = real(*args, **kwargs)
        # One lambda row is imperfect; the complete physical x is unchanged.
        result.x[3] += 5e-5
        return result
    monkeypatch.setattr(v3, 'solve_linear_mip', imperfect_auxiliary)
    out = v3.solve_projected_milp(pairs, _box([-1.], [1.]),
        initial_operations={'parent': np.array([0.])}, segments=4, backend='scipy', solver_seconds=5.)
    assert out['exact_score'] == pytest.approx(1.)
    assert out['solver']['incumbent_surrogate'] is not None
    assert out['solver']['master_incumbent_audit']['linear_violation'] > 1e-7
    assert out['solver']['operation_incumbent_audit']['feasible']
    assert out['solver']['reconstructed_incumbent_audit']['feasible']
    assert out['failures'][0]['kind'] == 'master_auxiliary_constraint_violation'


def test_physical_violation_is_rejected_at_the_original_tolerance(monkeypatch):
    pairs = [_pair(a, b, [0.], [[np.sqrt(2.)]]) for a, b in [('a', 'b'), ('b', 'a')]]
    real = v3.solve_linear_mip
    def imperfect_operation(*args, **kwargs):
        result = real(*args, **kwargs)
        result.x[0] = 1. + 5e-5
        return result
    monkeypatch.setattr(v3, 'solve_linear_mip', imperfect_operation)
    out = v3.solve_projected_milp(pairs, _box([-1.], [1.]),
        initial_operations={'parent': np.array([.2])}, segments=4, backend='scipy', solver_seconds=5.)
    assert out['exact_score'] == pytest.approx(.04)
    assert out['selected_origin'] == 'initial_operation:parent'
    assert not out['solver']['operation_incumbent_audit']['feasible']
