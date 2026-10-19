"""Three independent checks of the stationary affine design theorem."""
import numpy as np
from .stationary_price_check import (
    extreme_power_vector, slot_half_cost, continuous_slope_half_distance, run_audit)


def test_extreme_majorizes_grid_before_and_after_shared_competitor_refit():
    evidence = run_audit()
    assert evidence["status"] == "passed"
    assert len(evidence["grid_rows"]) == 16
    assert evidence["checks"]["majorization"]
    assert evidence["checks"]["per_fixed_theta_order"]
    assert evidence["checks"]["min_theta_order"]
    np.testing.assert_allclose(evidence["extreme_power"], [.3, .15, 0., 0.])
    assert not evidence["full_AC_global_optimality_certified"]


def test_open_true_power_compensation_matches_independent_numerical_least_squares():
    for slope, intercept in ((.7, .02), (1.2, -.03), (1.8, .01)):
        for power in (0., .15, .30):
            mismatch = (slope-1.4)*power+intercept
            matrix = np.array([[1/.08], [slope/.03]])
            target = np.array([0., -mismatch/.03])
            delta = np.linalg.lstsq(matrix, target, rcond=None)[0]
            residual = matrix@delta-target
            np.testing.assert_allclose(.5*(residual@residual),
                slot_half_cost(power, slope, 1.4, intercept), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(matrix.T@residual, 0., atol=1e-10)
            assert slot_half_cost(power, slope, 1.4, intercept) <= .5*mismatch**2/.03**2+1e-12


def test_continuous_shared_slope_closed_minimum_and_saturated_design():
    constant = np.full(4, .45/4)
    extreme = extreme_power_vector(.45, .30, 4)
    values = []
    for power in (constant, extreme):
        closed, slope = continuous_slope_half_distance(power)
        design = np.r_[np.sqrt(2.), power/.03][:, None]
        target = np.r_[np.sqrt(2.)*.7, power*1.4/.03]
        fitted = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = design@fitted-target
        np.testing.assert_allclose(closed, .5*(residual@residual), atol=1e-12)
        np.testing.assert_allclose(slope, fitted[0], atol=1e-12)
        values.append(closed)
    assert values[1] > values[0]
