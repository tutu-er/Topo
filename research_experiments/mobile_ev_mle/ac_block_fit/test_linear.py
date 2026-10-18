"""Independent dense-oracle checks of the bounded QR block solver."""
import numpy as np
import pytest
from scipy.optimize import lsq_linear

from .linear import solve_block_step


def dense_problem(residual, line, source, damping):
    t, k, d = line.shape
    matrix = np.zeros((t*k, d+2*t))
    matrix[:, :d] = line.reshape(t*k, d)
    for time in range(t):
        matrix[time*k:(time+1)*k, d+2*time:d+2*time+2] = source[time]
    if damping:
        matrix = np.vstack((matrix, np.sqrt(damping)*np.eye(d+2*t)))
    target = np.r_[-residual.ravel(), np.zeros(d+2*t)] if damping else -residual.ravel()
    return matrix, target


def assert_oracle(residual, line, source, lower, upper, damping, *, kkt_tol=1e-9):
    result = solve_block_step(residual, line, source, lower, upper,
                             damping=damping, kkt_tol=kkt_tol)
    matrix, target = dense_problem(residual, line, source, damping)
    lower, upper = np.asarray(lower), np.asarray(upper)
    fixed = lower == upper
    oracle = lower.copy()
    if np.any(~fixed):
        fit = lsq_linear(matrix[:, ~fixed], target-matrix[:, fixed]@lower[fixed],
                         bounds=(lower[~fixed], upper[~fixed]), method='bvls',
                         tol=1e-12, max_iter=1000)
        assert fit.success
        oracle[~fixed] = fit.x
    cost = .5*np.sum((matrix@oracle-target)**2)
    assert result['success'], result['message']
    assert result['max_feasibility_violation'] == 0.
    assert result['kkt'] <= kkt_tol
    assert result['objective'] == pytest.approx(cost, rel=2e-9, abs=2e-10)
    if damping:
        np.testing.assert_allclose(result['step'], oracle, atol=3e-7, rtol=2e-7)
        assert result['objective_lower_bound'] <= cost+1e-10
        assert result['objective']-cost <= result['qp_gap_upper_bound']+1e-9
    history = np.asarray(result['objective_history'])
    assert np.all(np.diff(history) <= 2e-12*np.maximum(1., abs(history[:-1])))
    return result


@pytest.mark.parametrize('seed', range(12))
@pytest.mark.parametrize('damping', [0., 1e-4, .2])
def test_random_arrow_blocks_match_independent_dense_bvls(seed, damping):
    rng = np.random.default_rng(seed)
    t, k, d = 3, 5, 4
    r, g, s = rng.normal(size=(t, k)), rng.normal(size=(t, k, d)), rng.normal(size=(t, k, 2))
    width = d+2*t
    lower, upper = -rng.uniform(.01, .7, width), rng.uniform(.01, .7, width)
    assert_oracle(r, g, s, lower, upper, damping)


def test_active_source_shared_bounds_and_release_are_retained():
    # The source begins on a lower bound but the constrained optimum releases
    # it; other source and shared coordinates hit opposing limits.
    r = np.array([[-2., 3., -1.], [1., -4., 2.]])
    g = np.array([[[1.], [.2], [1.]], [[.5], [-1.], [.3]]])
    s = np.array([[[1., 0.], [0., 1.], [.5, -.4]],
                  [[1., .1], [.3, 1.], [-.6, .2]]])
    lo, hi = np.array([-.1, 0., -.3, -.2, -.2]), np.array([.15, .5, .3, .2, .2])
    out = assert_oracle(r, g, s, lo, hi, .03)
    assert out['bound_releases'] >= 1
    assert np.count_nonzero(out['active_mask']) >= 2


@pytest.mark.parametrize('damping', [0., .01])
def test_rank_deficient_lines_and_sources_match_objective(damping):
    rng = np.random.default_rng(511)
    base = rng.normal(size=(3, 4, 1))
    g = np.concatenate((base, base, np.zeros_like(base)), axis=2)
    source = rng.normal(size=(3, 4, 1))
    s = np.concatenate((source, 2*source), axis=2)
    r = rng.normal(size=(3, 4))
    out = assert_oracle(r, g, s, -np.ones(9), np.ones(9), damping)
    if not damping:
        assert out['svd_calls'] > 0


@pytest.mark.parametrize('damping', [0., .1])
def test_fixed_lines_equals_independent_two_dimensional_sources(damping):
    rng = np.random.default_rng(7)
    r, g, s = rng.normal(size=(5, 4)), np.empty((5, 4, 0)), rng.normal(size=(5, 4, 2))
    out = assert_oracle(r, g, s, -.2*np.ones(10), .3*np.ones(10), damping)
    assert out['shared_solves'] == 0


def test_all_sources_fixed_and_one_shared_coordinate_free():
    r = np.array([[1., 2.], [-1., 3.]])
    g = np.ones((2, 2, 1))
    s = np.ones((2, 2, 2))
    lo, hi = np.array([-3., .1, -.2, .3, .4]), np.array([3., .1, -.2, .3, .4])
    out = assert_oracle(r, g, s, lo, hi, .1)
    assert np.all(out['active_mask'][1:] == 2)
    assert out['qr_calls'] == 0


def test_one_free_source_coordinate_and_no_shared_coordinates():
    r = np.array([[1., -2., 4.]])
    g = np.empty((1, 3, 0))
    s = np.array([[[1., 3.], [.2, -1.], [.4, 2.]]])
    assert_oracle(r, g, s, np.array([-.3, .2]), np.array([.1, .2]), .1)


def test_zero_gradient_zero_columns_unbounded_and_no_unique_parameter():
    r, g, s = np.ones((2, 2)), np.zeros((2, 2, 0)), np.zeros((2, 2, 2))
    out = assert_oracle(r, g, s, -np.inf*np.ones(4), np.inf*np.ones(4), 0.)
    assert out['iterations'] == 1
    assert out['kkt'] == 0.


def test_one_channel_and_two_undetermined_source_coordinates():
    r = np.array([[2.], [-3.]])
    g = np.array([[[1., 1.]], [[2., 2.]]])
    s = np.array([[[1., 2.]], [[-.5, 1.]]])
    assert_oracle(r, g, s, -np.ones(6), np.ones(6), 0.)


def test_source_qr_factors_are_reused_across_shared_active_sets():
    rng = np.random.default_rng(231)
    r, g, s = rng.normal(size=(3, 5)), rng.normal(size=(3, 5, 5)), rng.normal(size=(3, 5, 2))
    out = assert_oracle(r, g, s, -.03*np.ones(11), .03*np.ones(11), .1)
    # There are only three nonempty free masks of two local source columns.
    assert out['qr_calls'] <= 3*len(r)
    assert out['factor_cache_hits'] > 0


def test_zero_residual_is_not_a_false_bound_release():
    rng = np.random.default_rng(31)
    out = assert_oracle(np.zeros((2, 3)), rng.normal(size=(2, 3, 2)),
        rng.normal(size=(2, 3, 2)), -np.ones(6), np.ones(6), .1)
    assert out['objective'] == 0.
    assert out['bound_releases'] == 0


def test_narrow_and_fixed_boxes_with_feasible_nonzero_initial_point():
    rng = np.random.default_rng(5)
    r, g, s = rng.normal(size=(2, 4)), rng.normal(size=(2, 4, 2)), rng.normal(size=(2, 4, 2))
    lo = np.array([.2, -.1, -1e-12, -.3, .04, .7])
    hi = np.array([.2+1e-12, -.1, 1e-12, .3, .04, .9])
    assert_oracle(r, g, s, lo, hi, .1)


def test_budget_exhaustion_returns_feasible_failure():
    rng = np.random.default_rng(99)
    r, g, s = rng.normal(size=(3, 5)), rng.normal(size=(3, 5, 4)), rng.normal(size=(3, 5, 2))
    out = solve_block_step(r, g, s, -.01*np.ones(10), .01*np.ones(10),
                           damping=.1, max_iter=1)
    assert not out['success']
    assert out['max_feasibility_violation'] == 0.
    assert out['iterations'] == 1


@pytest.mark.parametrize('kwargs', [dict(damping=-1), dict(kkt_tol=0), dict(max_iter=0)])
def test_invalid_solver_settings_raise(kwargs):
    with pytest.raises(ValueError):
        solve_block_step(np.ones((1, 2)), np.ones((1, 2, 0)), np.ones((1, 2, 2)),
                         -np.ones(2), np.ones(2), **kwargs)


def test_nonfinite_or_reversed_data_raise():
    with pytest.raises(ValueError):
        solve_block_step(np.ones((1, 2)), np.ones((1, 2, 0)), np.ones((1, 2, 2)),
                         np.ones(2), -np.ones(2))
    with pytest.raises(ValueError):
        solve_block_step(np.array([[np.nan, 0.]]), np.ones((1, 2, 0)), np.ones((1, 2, 2)),
                         -np.ones(2), np.ones(2))
