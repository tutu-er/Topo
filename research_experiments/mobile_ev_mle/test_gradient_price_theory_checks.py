"""Analytic and independent sampling checks; no experimental policy input."""
import numpy as np
import pytest

from research_experiments.mobile_ev_mle.gradient_price_theory_checks import (
    active_boundary_expected_loss, active_boundary_expected_price,
    energy_transfer_price, fixed_gaussian_llr_moments,
    gaussian_quadratic_moments, projection_loss_moments,
    projection_margin_moments, projection_margin_price, run_theory_checks)


def test_projection_mean_and_variance_from_scalar_noncentral_square():
    result = projection_loss_moments([2., 3.], [0., 0.], np.diag([1., 0.]), np.diag([4., 9.]))
    assert result["mean_prediction_loss"] == 2.
    assert result["noise_offset"] == 2.
    assert result["mean"] == 4.
    assert result["variance"] == 24.
    assert result["linear_variance"] == 16.
    assert result["quadratic_variance"] == 8.


def test_indefinite_quadratic_margin_variance_and_zero_history_covariance():
    b, c, k = np.array([.2, .4]), np.diag([1., -1.]), np.diag([0., 3.])
    result = gaussian_quadratic_moments(2., b, c, k)
    assert result["mean"] == .5
    assert result["variance"] == pytest.approx(.48+4.5)
    assert result["linear_variance"] == pytest.approx(.48)


def test_fixed_projection_expected_and_mean_margin_gradients_are_equal():
    m, j = np.array([.3, 1.]), np.array([[0.], [2.]])
    ar, ac = np.zeros(2), np.array([.1, .2])
    pr, pc, k = np.diag([0., 1.]), np.eye(2), np.diag([0., 1.])
    price = projection_margin_price(m, j, ar, ac, pr, pc)[0]
    step = 1e-5
    plus = projection_margin_moments(m+step*j[:, 0], ar, ac, pr, pc, k)
    minus = projection_margin_moments(m-step*j[:, 0], ar, ac, pr, pc, k)
    assert price == pytest.approx((plus["mean"]-minus["mean"])/(2*step), abs=1e-10)
    assert price == pytest.approx((plus["mean_prediction_margin"]-minus["mean_prediction_margin"])/(2*step), abs=1e-10)


def test_active_boundary_has_positive_expected_price_at_zero_mean_price():
    assert active_boundary_expected_loss(0.) == .25
    assert active_boundary_expected_price(0.) == pytest.approx(1/np.sqrt(2*np.pi))
    step = 1e-5
    numerical = (active_boundary_expected_loss(step)-active_boundary_expected_loss(-step))/(2*step)
    assert numerical == pytest.approx(active_boundary_expected_price(0.), abs=1e-10)
    assert active_boundary_expected_loss(0., sigma=2.) == 1.
    assert active_boundary_expected_price(0., sigma=2.) == pytest.approx(2/np.sqrt(2*np.pi))


def test_fixed_gaussian_llr_mean_variance_kl_and_tie_case():
    result = fixed_gaussian_llr_moments([.8, -.6], [0., 0.], np.eye(2))
    assert result["gamma"] == 1.
    assert result["llr_mean"] == result["KL"] == .5
    assert result["llr_variance"] == 2*result["KL"]
    assert result["equal_prior_error"] == pytest.approx(.3085375387259869)
    assert fixed_gaussian_llr_moments([0.], [0.], [[1.]])["equal_prior_error"] == .5


def test_equal_energy_marginal_price_uses_slot_duration():
    assert energy_transfer_price([2., 3.], [.25, .5], 0, 1) == -2.
    assert energy_transfer_price([2., 3.], .25, 0, 1) == 4.
    delta_energy = .01
    delta_power = np.array([-delta_energy/.25, delta_energy/.5])
    assert np.array([.25, .5])@delta_power == pytest.approx(0.)
    assert np.array([2., 3.])@delta_power == pytest.approx(-2*delta_energy)


@pytest.mark.parametrize("bad_covariance", [np.array([[1., .2], [0., 1.]]), np.diag([1., -1.])])
def test_invalid_covariance_is_rejected(bad_covariance):
    with pytest.raises(ValueError):
        gaussian_quadratic_moments(0., np.zeros(2), np.eye(2), bad_covariance)


def test_nonprojection_and_invalid_durations_are_rejected():
    with pytest.raises(ValueError, match="idempotent"):
        projection_loss_moments([0.], [0.], [[.5]], [[1.]])
    with pytest.raises(ValueError, match="positive"):
        energy_transfer_price([1., 2.], [0., .25], 0, 1)
    with pytest.raises(ValueError, match="positive definite"):
        fixed_gaussian_llr_moments([0.], [1.], [[0.]])


def test_independent_monte_carlo_audit_and_crn_unit_increment_limit():
    evidence = run_theory_checks(sample_count=20000, seed=17031013)
    assert evidence["passed"], evidence
    assert evidence["monte_carlo_failure_count"] == 0
    assert not evidence["affects_policy_or_thresholds"]
    assert evidence["config"]["history_covariance"] == "zero"
    increments = evidence["unit_random_increments"]
    errors = [abs(i["finite_to_price_error"]) for i in increments]
    assert all(b < a for a, b in zip(errors, errors[1:]))
    assert all(i["common_random_numbers"] for i in increments)
    at_zero = next(b for b in evidence["active_boundary"] if b["u"] == 0.)
    assert at_zero["mean_prediction_price"] == 0.
    assert at_zero["expected_price"] > .39


def test_sampling_budget_is_explicit_and_restricted():
    with pytest.raises(ValueError, match="200000"):
        run_theory_checks(sample_count=200001)
