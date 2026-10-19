"""Independent analytic and small numerical audits of frozen-proxy MILPs.

The only discretized outer comparison is a deliberately tiny test. The
actual selector receives matrices and an operations model, not an action
library, truth topology, or future measurement noise.
"""
import importlib
import numpy as np
import pytest
from scipy.optimize import minimize

try:
    from .ev_operations_milp import EVConfig, check_plan
    from .ev_information_milp import (
        FrozenPair, adaptive_search, cut_from_solution, lower_master,
        solve_pair, support_cut, upper_master,
    )
except ImportError:
    from ev_operations_milp import EVConfig, check_plan
    from ev_information_milp import (
        FrozenPair, adaptive_search, cut_from_solution, lower_master,
        solve_pair, support_cut, upper_master,
    )


def pure_pair(name, B, c=None):
    B = np.asarray(B, float)
    return FrozenPair(name, np.zeros(len(B)) if c is None else np.asarray(c),
                      B, np.zeros((len(B), 0)), np.zeros((0, 0)),
                      np.zeros(0), np.zeros(0))


def analytic_pair():
    # min_z (2u+sqrt(2)z)^2+z^2 = 4u^2/3, z*=-2sqrt(2)u/3.
    return FrozenPair('analytic', np.zeros(1), np.array([[2.]]),
                      np.array([[np.sqrt(2)]]), np.ones((1, 1)),
                      np.array([-20.]), np.array([20.]))


@pytest.mark.parametrize('u', [-4., 0., .7, 3.])
def test_analytic_soft_history_profile_and_tight_direction(u):
    pair = analytic_pair()
    result = solve_pair(pair, [u])
    truth = 4 * u * u / 3
    assert result['success']
    assert result['q_lower'] <= truth + 1e-10
    assert result['q_upper'] == pytest.approx(truth, abs=1e-10)
    assert result['numeric_gap'] < 1e-9
    assert result['z'][0] == pytest.approx(-2 * np.sqrt(2) * u / 3)
    cut = cut_from_solution(pair, [u], result, [-6], [6])
    assert cut.value([u]) == pytest.approx(np.sqrt(truth), abs=1e-10)


def test_fixed_coordinates_and_empty_nuisance_dimension():
    pair = FrozenPair('fixed', np.array([1.]), np.array([[1.]]),
                      np.array([[2., 1.]]), np.zeros((0, 2)),
                      np.array([.5, -10]), np.array([.5, 10]))
    result = solve_pair(pair, [3])
    assert result['z'] == pytest.approx([.5, -5])
    assert result['q_upper'] < 1e-20
    pair = pure_pair('none', [[2.]], [1])
    result = solve_pair(pair, [3])
    assert result['success']
    assert result['q_upper'] == pytest.approx(49)
    assert result['q_lower'] <= 49 <= result['q_upper']
    assert result['numeric_gap'] < 1e-10


def test_rank_deficient_nuisance_matrix_and_scaling_do_not_create_information():
    pair = FrozenPair('rank_def', np.zeros(1), np.ones((1, 1)),
                      np.array([[1., 1.]]), np.zeros((0, 2)),
                      np.array([-10., -10.]), np.array([10., 10.]),
                      scale=np.array([.01, 100.]))
    result = solve_pair(pair, [4.])
    assert result['success']
    assert result['q_lower'] == pytest.approx(0)
    assert result['q_upper'] < 1e-20
    assert np.sum(result['z']) == pytest.approx(-4)


def test_active_box_bound_profile_is_not_unconstrained_formula():
    pair = FrozenPair('active', np.zeros(1), np.array([[2.]]),
                      np.ones((1, 1)), np.ones((1, 1)), np.zeros(1), np.ones(1))
    result = solve_pair(pair, [2])
    assert result['z'] == pytest.approx([0])
    assert result['q_upper'] == pytest.approx(16)
    assert result['q_lower'] <= 16
    assert result['numeric_gap'] < 1e-9


def test_arbitrary_unit_directions_are_global_profile_distance_lower_bounds():
    pair = FrozenPair('generic', np.array([.4, -.7]), np.array([[1., .2], [-.3, 2.]]),
                      np.array([[1., 2.], [.5, -1.]]), np.diag([.3, .6]),
                      np.array([-.2, -1]), np.array([.4, .5]))
    rng = np.random.default_rng(91831)
    us = rng.uniform(-2, 3, (25, 2))
    distances = [np.sqrt(solve_pair(pair, u)['q_upper']) for u in us]
    for _ in range(15):
        v = rng.normal(size=4)
        v /= max(np.linalg.norm(v), 1.)
        cut = support_cut(pair, v, [-2, -2], [3, 3])
        assert all(cut.value(u) <= d + 1e-9 for u, d in zip(us, distances))
    with pytest.raises(ValueError, match='norm'):
        support_cut(pair, [2, 0, 0, 0])


def test_support_constant_uses_box_minimum_not_an_arbitrary_qp_witness():
    # For min_{z in [-1,1]} (u+z)^2, distance at u=0 is zero.
    # A witness z=1 would produce the invalid supposed lower plane 1+u.
    pair = FrozenPair('support', np.zeros(1), np.ones((1, 1)),
                      np.ones((1, 1)), np.zeros((0, 1)),
                      np.array([-1.]), np.array([1.]))
    cut = support_cut(pair, [1.])
    assert cut.value([0]) == pytest.approx(-1)
    fake_solution = {'residual': np.array([2.]), 'z': np.array([1.])}
    regenerated = cut_from_solution(pair, [0], fake_solution)
    assert regenerated.value([0]) <= 0
    assert regenerated.intercept == pytest.approx(-1)


def test_direction_master_uses_max_directions_and_has_correct_max_bound_sign():
    cfg = EVConfig(n_slots=2, n_sites=1, net_grid_energy_kwh=1.5)
    pair = pure_pair('difference', [[1., -1.]])
    cuts = [[support_cut(pair, [0]), support_cut(pair, [1]), support_cut(pair, [-1])]]
    result = lower_master(cfg, [pair], cuts)
    assert result['result'].success
    assert result['distance_incumbent'] == pytest.approx(6, abs=1e-7)
    assert result['distance_master_upper'] >= result['distance_incumbent'] - 1e-7
    assert result['distance_master_upper'] == pytest.approx(6, abs=1e-7)
    assert check_plan(result['plan'], cfg)['feasible']


def test_zero_direction_master_is_flat_but_upper_master_finds_positive_score():
    cfg = EVConfig(n_slots=2, n_sites=1, net_grid_energy_kwh=1.5)
    pair = pure_pair('difference', [[1., -1.]])
    solution = solve_pair(pair, [3, 3])
    assert solution['q_upper'] == pytest.approx(0)
    zero_cut = cut_from_solution(pair, [3, 3], solution)
    lower = lower_master(cfg, [pair], [[zero_cut]])
    assert lower['distance_incumbent'] == pytest.approx(0)
    upper = upper_master(cfg, [pair], [[np.zeros(0)]], segments=4)
    assert upper['result'].success
    assert solve_pair(pair, upper['plan'].u_kw)['q_upper'] == pytest.approx(36, abs=1e-6)
    search = adaptive_search(cfg, [pair], initial_us=([3, 3],), max_iterations=4,
                             absolute_gap=1e-6, relative_gap=1e-6)
    assert search['success']
    assert search['proxy_lower'] == pytest.approx(36, abs=1e-5)
    assert search['proxy_upper'] >= 36 - 1e-5
    assert search['absolute_gap'] < 1e-4
    assert search['enumerated_complete_actions'] == 0


def test_two_slot_fixed_energy_square_objective_has_analytic_bang_bang_optimum():
    # u0+u1=6 and 0<=u<=6, hence u0²+u1²<=36 with equality at endpoints.
    cfg = EVConfig(n_slots=2, n_sites=1, net_grid_energy_kwh=1.5)
    pair = pure_pair('squares', np.eye(2))
    result = adaptive_search(cfg, [pair], max_iterations=4, absolute_gap=1e-6,
                             relative_gap=1e-6)
    assert result['success']
    assert result['proxy_lower'] == pytest.approx(36, abs=1e-5)
    assert result['proxy_upper'] == pytest.approx(36, abs=1e-5)
    assert sorted(result['plan'].u_kw.ravel()) == pytest.approx([0, 6], abs=1e-6)
    assert result['selector_uses_truth'] is False
    assert result['selector_uses_future_noise'] is False


def test_two_pair_max_min_matches_tiny_grid_and_independent_nuisance_qps():
    cfg = EVConfig(n_slots=2, n_sites=1, net_grid_energy_kwh=1.5)
    pairs = [FrozenPair(name, np.zeros(1), np.array([B]), np.ones((1, 1)),
                        np.array([[np.sqrt(.2)]]), np.array([-.5]), np.array([.5]))
             for name, B in [('first', [1., 0.]), ('second', [0., 1.])]]
    # Independently solve each scalar nuisance QP. A 13-point outer grid
    # contains the analytic symmetry optimum u=(3,3); each profile is monotone
    # increasing in its nonnegative selected power, so the true max-min is there.
    scores = []
    for u0 in np.linspace(0, 6, 13):
        inner = []
        for value in [u0, 6-u0]:
            solved = minimize(lambda z: (value+z[0])**2 + .2*z[0]**2,
                              [0.], jac=lambda z: np.array([2*(value+z[0])+.4*z[0]]),
                              bounds=[(-.5, .5)], method='SLSQP',
                              options={'ftol': 1e-12, 'maxiter': 200})
            assert solved.success
            inner.append(solved.fun)
        scores.append(min(inner))
    truth = max(scores)
    assert truth == pytest.approx(6.3)
    result = adaptive_search(cfg, pairs, max_iterations=6, segments=2,
                             absolute_gap=1e-5, relative_gap=1e-5)
    assert result['success'], result['failures']
    assert result['proxy_lower'] <= truth + 1e-5
    assert result['proxy_upper'] >= truth - 1e-5
    assert result['proxy_lower'] == pytest.approx(truth, abs=1e-4)
    assert result['plan'].u_kw.ravel() == pytest.approx([3, 3], abs=1e-4)


def test_square_secant_overestimate_has_declared_error_and_refines():
    cfg = EVConfig(n_slots=1, n_sites=1, net_grid_energy_kwh=.325)
    pair = pure_pair('square', [[1.]])
    true_q = 1.3**2
    errors = []
    for segments in [2, 4, 8]:
        result = upper_master(cfg, [pair], [[np.zeros(0)]], segments=segments)
        assert result['result'].success
        error = result['q_incumbent'] - true_q
        declared = result['secant_errors'][0]['secant_error_upper']
        assert -1e-7 <= error <= declared + 1e-7
        assert declared == pytest.approx(6**2/(4*segments**2), abs=1e-8)
        assert result['q_proxy_upper'] >= true_q - 1e-7
        errors.append(error)
    assert errors[2] <= errors[1] <= errors[0]


def test_cached_declared_vector_can_later_become_a_valid_outer_incumbent():
    # A declared vector initially provides only cuts. When the MILP later
    # produces a physical plan with that same u, caching must not prevent it
    # from becoming the incumbent or bypass the physical validation.
    cfg = EVConfig(n_slots=2, n_sites=1, net_grid_energy_kwh=1.5)
    pairs = [pure_pair('first', [[1., 0.]]), pure_pair('second', [[0., 1.]])]
    result = adaptive_search(cfg, pairs, initial_us=([3., 3.],), max_iterations=3)
    assert result['success']
    assert result['proxy_lower'] == pytest.approx(9, abs=1e-5)
    assert result['plan'].u_kw.ravel() == pytest.approx([3, 3], abs=1e-5)


def test_feasible_poor_witness_is_an_upper_bound_not_distance_lower():
    cfg = EVConfig(n_slots=1, n_sites=1, net_grid_energy_kwh=.75)
    pair = FrozenPair('poor_witness', np.zeros(1), np.ones((1, 1)),
                      np.ones((1, 1)), np.ones((1, 1)),
                      np.array([-10.]), np.array([10.]))
    truth = 4.5  # u=3, optimum z=-1.5.
    assert solve_pair(pair, [3])['q_upper'] == pytest.approx(truth)
    upper = upper_master(cfg, [pair], [[np.array([5.])]], segments=2)
    assert upper['q_proxy_upper'] == pytest.approx(89, abs=1e-5)
    assert upper['q_proxy_upper'] - truth > 80
    with pytest.raises(ValueError, match='infeasible'):
        upper_master(cfg, [pair], [[np.array([11.])]])


def test_operations_infeasible_is_retained_without_a_spurious_design():
    cfg = EVConfig(n_slots=1, n_sites=1, net_grid_energy_kwh=2)
    pair = pure_pair('infeasible', [[1.]])
    result = adaptive_search(cfg, [pair])
    assert not result['success']
    assert result['status'] == 'operations_infeasible_or_failed'
    assert result['plan'] is None
    assert result['qp_count'] == 0
    assert result['failures']


def test_lower_direction_master_best_bound_is_not_full_proxy_upper_bound():
    cfg = EVConfig(n_slots=2, n_sites=1, net_grid_energy_kwh=1.5)
    pair = pure_pair('squares', np.eye(2))
    result = lower_master(cfg, [pair], [[support_cut(pair, np.zeros(2))]])
    assert result['result'].success
    assert result['distance_master_upper'] == pytest.approx(0)
    assert solve_pair(pair, [6, 0])['q_upper'] == pytest.approx(36)
    assert 'not full proxy upper' in result['bound_scope']


def test_inner_failure_retains_difficult_pair_instead_of_increasing_score(monkeypatch):
    cfg = EVConfig(n_slots=2, n_sites=1, net_grid_energy_kwh=1.5)
    pairs = [pure_pair('easy', np.eye(2)), pure_pair('difficult', [[1., -1.]])]
    module = importlib.import_module(solve_pair.__module__)
    original = solve_pair

    def failed(pair, u, **kwargs):
        result = original(pair, u, **kwargs)
        if pair.name == 'difficult':
            result['success'] = False
        return result

    monkeypatch.setattr(module, 'solve_pair', failed)
    result = adaptive_search(cfg, pairs, max_iterations=1)
    assert not result['success']
    assert result['status'] == 'failure_preserved'
    assert result['plan'] is None
    assert set(result['pair_hashes']) == {'easy', 'difficult'}
    assert result['qp_count'] == 2
    assert 'no pair removed' in result['failures'][0]


def test_frozen_pair_shape_and_scale_validation():
    with pytest.raises(ValueError, match='dimensions'):
        FrozenPair('bad', np.zeros(2), np.ones((1, 1)), np.ones((1, 1)),
                   np.ones((1, 1)), np.zeros(1), np.ones(1))
    with pytest.raises(ValueError, match='scale'):
        FrozenPair('bad', np.zeros(1), np.ones((1, 1)), np.ones((1, 1)),
                   np.ones((1, 1)), np.zeros(1), np.ones(1), scale=np.zeros(1))
