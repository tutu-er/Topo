"""Independent tests of branch-information formulas, bounds and failures."""
import numpy as np
import pytest

from .structured_information_checks import (
    ENGINEERING_SCALE, OMEGA, projected_information, information_half_value,
    compensation_price, four_slot_information, four_slot_gradient,
    four_slot_global_shape, operational_price_step, tiny_grid_audit, run_audit)


def test_projection_is_best_compensation_and_keeps_physical_units():
    a = np.array([[1., .2], [2., .5], [0., 1.]])
    b = np.array([[1.], [2.], [-1.]])
    direction = np.array([.003, .002])
    info = projected_information(a, b)['information']
    fitted = np.linalg.lstsq(b, -a@direction, rcond=None)[0]
    residual = a@direction+b@fitted
    np.testing.assert_allclose(direction@info@direction, residual@residual, atol=1e-20)
    np.testing.assert_allclose(b.T@residual, 0., atol=1e-17)
    assert float(direction@info@direction) >= 0


def test_positive_R_X_second_moment_is_not_covariance():
    directions = np.array([[ENGINEERING_SCALE, 0.], [0., ENGINEERING_SCALE]])
    second = directions.T@directions/2
    covariance = second-np.outer(directions.mean(0), directions.mean(0))
    np.testing.assert_allclose(second, OMEGA, atol=0, rtol=1e-14)
    assert np.linalg.norm(second-covariance) > 1e-6
    a, b = np.array([[1., 2.], [3., 1.], [0., 2.]]), np.ones((3, 1))
    info = projected_information(a, b)['information']
    expected = .5*np.mean([v@info@v for v in directions])
    np.testing.assert_allclose(information_half_value(a, b), expected, atol=1e-20)


def test_price_includes_changing_compensation_and_matches_direction_fd():
    a = np.array([[.4], [1.]])
    b = np.array([[1.], [.4]])
    da, db = np.array([[1.], [0.]]), np.array([[0.], [1.]])
    price = compensation_price(a, b, da, db, np.ones((1, 1)))
    step = 1e-6
    finite = (information_half_value(a+step*da, b+step*db, np.ones((1, 1)))-
              information_half_value(a-step*da, b-step*db, np.ones((1, 1))))/(2*step)
    np.testing.assert_allclose(price['price'], finite, atol=2e-10)
    assert abs(price['compensation_term']) > .1
    assert abs(price['signal_term']-finite) > .1


def test_rank_deficient_nuisance_is_projected_by_actual_rank():
    a = np.array([[1., 2.], [2., 1.], [3., 2.]])
    column = np.array([1., -1., 2.])
    b = np.c_[column, 2*column, np.zeros(3)]
    fit = projected_information(a, b)
    independent = a+b@np.linalg.lstsq(b, -a, rcond=None)[0]
    assert fit['nuisance_rank'] == 1
    np.testing.assert_allclose(fit['residual'], independent, atol=1e-14)
    np.testing.assert_allclose(b.T@fit['residual'], 0., atol=1e-13)


def test_empty_nuisance_preserves_signal_information():
    a = np.array([[1., 2.], [3., 4.]])
    fit = projected_information(a, np.empty((2, 0)))
    np.testing.assert_allclose(fit['information'], a.T@a)
    assert fit['nuisance_rank'] == 0


def test_history_increment_and_two_stage_projection():
    # One shared nuisance column, one historical state and one future state.
    ah = np.array([[1., 0.], [1., 1.], [0., 1.]])
    an = np.array([[2., 1.], [1., 2.], [0., 3.]])
    sh, sn = np.array([[1.], [2.], [1.]]), np.array([[1.], [1.], [2.]])
    lh, ln = np.array([[1.], [0.], [0.]]), np.array([[0.], [0.], [1.]])
    joint_a = np.r_[ah, an]
    joint_b = np.r_[np.c_[sh, lh, np.zeros((3, 1))], np.c_[sn, np.zeros((3, 1)), ln]]
    ih = projected_information(ah, np.c_[sh, lh])['information']
    joint = projected_information(joint_a, joint_b)['information']
    assert np.linalg.eigvalsh(joint-ih).min() >= -1e-13
    h1, n1 = projected_information(np.c_[ah, sh], lh)['residual'], projected_information(np.c_[an, sn], ln)['residual']
    staged = projected_information(np.r_[h1[:, :2], n1[:, :2]], np.r_[h1[:, 2:], n1[:, 2:]])['information']
    np.testing.assert_allclose(staged, joint, atol=1e-13)


def test_EV_increment_against_no_EV_can_be_indefinite():
    history = .1*np.eye(2)
    base = projected_information(np.r_[history, np.eye(2)], np.empty((4, 0)))['information']
    ev = projected_information(np.r_[history, np.diag([1.5, .5])], np.empty((4, 0)))['information']
    np.testing.assert_allclose(ev-base, np.diag([1.25, -.75]))
    assert np.linalg.eigvalsh(ev-base).min() < 0
    assert np.linalg.eigvalsh(ev-history.T@history).min() >= 0


def test_active_boundary_and_pure_X_weak_direction_are_retained():
    free = projected_information(np.ones((1, 1)), np.ones((1, 1)))['information'][0, 0]
    # Compensation eta>=0 cannot cancel a positive signal .5.
    cone_distance = .5**2
    assert free == 0.
    assert cone_distance > free
    info = projected_information(np.array([[1., 0.], [2., 0.]]), np.ones((2, 1)))['information']
    assert np.trace(info@OMEGA) > 0
    assert info[1, 1] == 0.


def test_nonlinear_compensation_can_defeat_linear_information():
    for zeta in (.1, .01, .001):
        true_mean = zeta
        competing_mean = np.sqrt(zeta)**2
        np.testing.assert_allclose(true_mean, competing_mean, atol=1e-16)
        linear_squared = zeta*zeta
        assert linear_squared > (true_mean-competing_mean)**2
        assert np.sqrt(zeta)/zeta >= np.sqrt(10)-1e-14


@pytest.mark.parametrize('h,a', [(0., 4.), (.2, 2.), (3., 4.), (10., 4.), (10., 7.)])
def test_four_slot_global_class_and_independent_grid(h, a):
    shape = four_slot_global_shape(h, a)
    grid = tiny_grid_audit(h, a, shape['information'])
    assert len(grid) == 19
    assert all(row['below_analytic_global'] for row in grid)
    np.testing.assert_allclose(max(row['information'] for row in grid), shape['information'], atol=1e-12)
    classes = shape['vertex_class_values']
    np.testing.assert_allclose(classes['high_high']-classes['low_high'],
                               (a-1)*(h+1-a)/(h+2+2*a), atol=1e-12)
    assert shape['optimal_class'] == ('high_high' if h > a-1 else 'low_high' if h < a-1 else 'tie')


def test_four_slot_price_matches_direct_finite_difference_and_closed_values():
    power = np.array([.2, .7, .8, .3])
    transfer = np.array([1., -1., 0., 0.])
    step = 1e-6
    finite = (four_slot_information(power+step*transfer)-four_slot_information(power-step*transfer))/(2*step)
    np.testing.assert_allclose(four_slot_gradient(power)@transfer, finite, atol=1e-9)
    np.testing.assert_allclose([four_slot_information(np.full(4, .5)),
        four_slot_information([1., 0., 1., 0.]), four_slot_information([0., 1., 1., 0.])],
        [1.25, 3.75, 4.8], atol=1e-14)


def test_same_operation_MILP_selects_better_curve_from_constant():
    result = operational_price_step(np.full(4, .5))
    assert result['success'] and result['validation']['feasible']
    np.testing.assert_allclose(result['power'], [0., 1., 1., 0.], atol=1e-8)
    np.testing.assert_allclose(result['actual_information'], 4.8, atol=1e-10)
    assert result['incumbent'] <= result['best_bound']+1e-8
    assert result['absolute_gap'] < 1e-7
    assert result['bound_scope'] == 'frozen affine operational price problem'


def test_local_alternating_seed_can_stall_and_weak_precision_rule_fails():
    alternating = np.array([1., 0., 1., 0.])
    step = operational_price_step(alternating)
    assert step['success']
    np.testing.assert_allclose(step['power'], alternating, atol=1e-8)
    assert step['actual_information'] < four_slot_global_shape()['information']
    assert four_slot_information([0., 1., 1., 0.], 0., 4.) < four_slot_information(alternating, 0., 4.)
    np.testing.assert_allclose(four_slot_gradient(np.full(4, .5), 0., 4.), 0., atol=0.)
    assert four_slot_global_shape(0., 4.)['information'] > 0


def test_expected_random_unit_improvement_approaches_price():
    base, derivative = np.array([[1., 2.], [3., 1.], [0., 2.]]), np.array([[.4, .2], [0., .3], [.1, 0.]])
    b = np.ones((3, 1))
    eps = 1e-7
    positive = np.array([[ENGINEERING_SCALE, 0.], [0., ENGINEERING_SCALE]])
    expected_increment = np.mean([.5*(v@projected_information(base+eps*derivative, b)['information']@v-
        v@projected_information(base, b)['information']@v)/eps for v in positive])
    price = compensation_price(base, b, derivative, np.zeros_like(b))['price']
    np.testing.assert_allclose(expected_increment, price, rtol=1e-7, atol=1e-12)


def test_audit_passes_and_explicitly_retains_failure_examples():
    report = run_audit()
    assert report['status'] == 'passed'
    assert report['passed_count'] == report['check_count'] == 21
    assert report['strong']['local_seed_step']['actual_information'] < report['strong']['global_information']
    assert report['weak']['precision_heuristic_information'] < report['weak']['global_information']
    assert report['nonlinear_counterexample'][0]['linear_squared_distance'] > 0
    assert not report['full_AC_global_optimality_certified']


@pytest.mark.parametrize('call', [
    lambda: projected_information(np.ones((2, 2)), np.ones((3, 1))),
    lambda: projected_information(np.array([[np.nan]]), np.ones((1, 1))),
    lambda: information_half_value(np.ones((2, 2)), np.ones((2, 1)), np.diag([1., -1.])),
    lambda: compensation_price(np.ones((2, 2)), np.ones((2, 1)), np.ones((1, 2)), np.ones((2, 1))),
    lambda: four_slot_information(np.ones(3)),
    lambda: four_slot_global_shape(-1., 4.),
    lambda: four_slot_global_shape(1., 1.)])
def test_invalid_math_inputs_are_explicit(call):
    with pytest.raises(ValueError):
        call()