"""Independent arithmetic checks of the declared common-factor price model.

These are model checks, not observations of topology-identification benefit.
The Monte Carlo sample count is not a count of independent research cases.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from research_experiments.wzzt_dr_selection.dr_conic_pricing.model import (
    PriceProblem, evaluate_plan as _evaluate_plan,
)


def evaluate_plan(*args, **kwargs):
    # Dense matrices are intentional only for these one-/two-coordinate checks.
    return _evaluate_plan(*args, include_covariance=True, **kwargs)


def problem_data(**updates):
    data = dict(
        mu0=[0.], background_factors=[[0.]], profiles=[[1.]],
        response_factors=[[0.]], elasticities=[1.], reduction_hours=[1.],
        price_upper=[10.], actual_amplitude_upper=[10.],
        benefit_directions=[[1.]], benefit_offsets=[0.],
        risk_allocations=[.1], confidence=.9, budget=1.,
        device_matrix=[], device_rhs=[], contract_ids=["one"],
        exclusive_groups=[], benefit_mode="total",
        metadata={"source": "declared synthetic unit-test instance"},
    )
    data.update(updates)
    return data


def test_round_trip_copies_input_and_keeps_immutable_numeric_data():
    data = problem_data()
    problem = PriceProblem.from_dict(data)
    data["profiles"][0][0] = 99.
    assert problem.profiles[0, 0] == 1.
    with pytest.raises(ValueError):
        problem.profiles[0, 0] = 99.
    assert PriceProblem.from_dict(problem.to_dict()).to_dict() == problem.to_dict()


@pytest.mark.parametrize("change", [
    {"profiles": [[1., 2.]]},
    {"background_factors": [[0.], [1.]]},
    {"response_factors": [[1.01]]},
    {"elasticities": [0.]},
    {"price_upper": [-1.]},
    {"budget": -1.},
    {"risk_allocations": [.1001]},
    {"risk_allocations": [0.]},
    {"mu0": [float("nan")]},
    {"benefit_mode": "incremental", "benefit_offsets": [1.]},
    {"exclusive_groups": [[0], [0]]},
    {"surprise_field": 1},
])
def test_malformed_planning_inputs_fail_before_solver(change):
    with pytest.raises((ValueError, TypeError)):
        PriceProblem.from_dict(problem_data(**change))


def test_response_price_cost_and_cantelli_coefficients_by_hand():
    p = PriceProblem.from_dict(problem_data(
        mu0=[2.], background_factors=[[.3]], profiles=[[2.]],
        response_factors=[[.2]], elasticities=[2.], reduction_hours=[3.],
    ))
    actual = evaluate_plan(p, [.5])
    # E(Y)=2+2*.5=3; random coefficient .3+2*.5*.2=.5.
    assert actual["prices"] == [.25]
    assert actual["net_mean"] == [3.]
    assert actual["net_covariance"][0][0] == pytest.approx(.25/3)
    assert actual["benefit_standard_deviations"] == pytest.approx([.5/np.sqrt(3)])
    assert actual["reliable_benefit"] == pytest.approx(3-3*.5/np.sqrt(3))
    assert actual["expected_cost"] == pytest.approx(3/2*.5**2)
    assert actual["hard_budget_bound"] == pytest.approx(3/2*1.2*.5**2)
    assert actual["actual_amplitude_lower"] == pytest.approx([.4])
    assert actual["actual_amplitude_upper"] == pytest.approx([.6])


def test_shared_factors_can_cancel_background_variation_exactly():
    p = PriceProblem.from_dict(problem_data(
        background_factors=[[.4]], response_factors=[[-.4]], budget=2.,
    ))
    actual = evaluate_plan(p, [1.])
    assert actual["net_covariance"] == [[0.]]
    assert actual["benefit_standard_deviations"] == [0.]
    assert actual["reliable_benefit"] == pytest.approx(1.)
    # Independently adding both marginal variances would give .32/3 > 0.
    assert .4**2/3 + (-.4)**2/3 > actual["net_covariance"][0][0]


def test_zero_price_has_exactly_zero_response_and_payment():
    p = PriceProblem.from_dict(problem_data(
        mu0=[4.], background_factors=[[.4]], response_factors=[[.8]],
    ))
    actual = evaluate_plan(p, [0.])
    assert actual["response_mean"] == [0.]
    assert actual["response_covariance"] == [[0.]]
    assert actual["expected_cost"] == actual["hard_budget_bound"] == 0.
    assert actual["net_mean"] == [4.]
    assert actual["net_covariance"][0][0] == pytest.approx(.16/3)


def test_incremental_linear_benefit_cancels_background_and_pv_terms():
    data = problem_data(
        response_factors=[[.3]], benefit_mode="incremental",
        mu0=[8.], background_factors=[[.9]],
    )
    p = PriceProblem.from_dict(data)
    p_other = PriceProblem.from_dict(dict(data, mu0=[-123.], background_factors=[[-72.]]))
    first, second = evaluate_plan(p, [.4]), evaluate_plan(p_other, [.4])
    for key in ("benefit_means", "benefit_standard_deviations", "benefit_lower_bounds"):
        assert first[key] == second[key]
    assert first["benefit_means"] == pytest.approx([.4])
    assert first["benefit_standard_deviations"] == pytest.approx([.12/np.sqrt(3)])
    # Physical net load has changed even though paired linear benefit has not.
    assert first["net_mean"] != second["net_mean"]
    assert first["net_covariance"] != second["net_covariance"]


def test_joint_moments_match_direct_independent_sampling():
    mu = np.array([1., 2.])
    f = np.array([[.3, -.2], [.1, .4]])
    h = np.array([[1., -.5], [.2, 1.]])
    d = np.array([[.3, .1], [-.2, .2]])
    x = np.array([.4, .6])
    p = PriceProblem.from_dict(problem_data(
        mu0=mu, background_factors=f, profiles=h, response_factors=d,
        elasticities=[1., 1.], reduction_hours=[1., 1.],
        price_upper=[2., 2.], actual_amplitude_upper=[2., 2.],
        benefit_directions=[[1., -1.]], contract_ids=["a", "b"], budget=3.,
    ))
    rng = np.random.default_rng(720193)
    z = rng.uniform(-1., 1., size=(120_000, 2))
    realized = x[None, :]*(1+z@d.T)
    samples = mu[None, :]+z@f.T+realized@h.T
    actual = evaluate_plan(p, x)
    np.testing.assert_allclose(samples.mean(axis=0), actual["net_mean"], atol=.003, rtol=0)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), actual["net_covariance"], atol=.0015, rtol=0)
    np.testing.assert_allclose(np.cov(realized, rowvar=False), actual["response_covariance"], atol=.0003, rtol=0)


def test_device_box_bound_equals_direct_vertex_maximum():
    d = np.array([[.2, .1], [-.2, .1]])
    a = np.array([[1., -.7], [-1., .2], [.3, 1.]])
    x = np.array([.3, .4])
    p = PriceProblem.from_dict(problem_data(
        mu0=[0., 0.], background_factors=np.zeros((2, 2)), profiles=np.eye(2),
        response_factors=d, elasticities=[1., 1.], reduction_hours=[1., 1.],
        price_upper=[1., 1.], actual_amplitude_upper=[1., 1.],
        benefit_directions=[[1., 1.]], contract_ids=["a", "b"],
        device_matrix=a, device_rhs=[1., 1., 1.],
    ))
    values = np.array([a@(x*(1+d@np.array(z))) for z in itertools.product((-1., 1.), repeat=2)])
    actual = evaluate_plan(p, x)
    np.testing.assert_allclose(values.max(axis=0), actual["device_worst_values"], atol=1e-14, rtol=0)


def test_shared_factor_payment_cap_is_safe_but_can_be_conservative():
    p = PriceProblem.from_dict(problem_data(
        mu0=[0.], background_factors=[[0.]], profiles=[[1., 1.]],
        response_factors=[[.5], [-.5]], elasticities=[1., 1.],
        reduction_hours=[1., 1.], price_upper=[2., 2.],
        actual_amplitude_upper=[2., 2.], contract_ids=["a", "b"], budget=3.,
    ))
    actual = evaluate_plan(p, [1., 1.])
    # C=(1+.5z)+(1-.5z)=2 for every z, while sum of marginal worst bounds is3.
    assert actual["exact_box_worst_cost"] == pytest.approx(2.)
    assert actual["hard_budget_bound"] == pytest.approx(3.)


def test_distinct_module_effects_do_not_impose_artificial_spatial_dispersion():
    p = PriceProblem.from_dict(problem_data(
        mu0=[0., 0.], background_factors=np.zeros((2, 1)), profiles=np.eye(2),
        response_factors=np.zeros((2, 1)), elasticities=[1., 1.],
        reduction_hours=[1., 1.], price_upper=[2., 2.],
        actual_amplitude_upper=[2., 2.], contract_ids=["a", "b"],
        benefit_directions=[[1., 0.]],
    ))
    assert evaluate_plan(p, [1., 0.])["numerical_feasible"]
    assert not p.exclusive_groups


def test_zero_factor_dimension_is_a_valid_deterministic_problem():
    p = PriceProblem.from_dict(problem_data(
        background_factors=np.zeros((1, 0)), response_factors=np.zeros((1, 0)),
    ))
    actual = evaluate_plan(p, [.25])
    assert actual["benefit_standard_deviations"] == [0.]
    assert actual["benefit_lower_bounds"] == [.25]
