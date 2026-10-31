"""Small mathematical references for both real conic solver variants.

The discrete reference enumerates every fixed type assignment, rather than
comparing two instances of the same binary implementation.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

pytest.importorskip("gurobipy")

from research_experiments.wzzt_dr_selection.dr_conic_pricing.model import PriceProblem
from research_experiments.wzzt_dr_selection.dr_conic_pricing.solver import solve_price
from .test_independent_model import problem_data


def solve(data, **options):
    return solve_price(PriceProblem.from_dict(data), time_limit=30., **options)


def independent_lcb(data, x):
    x = np.asarray(x, float)
    h, d = np.asarray(data["profiles"]), np.asarray(data["response_factors"])
    factor = (h*x)@d
    if data.get("benefit_mode", "total") == "incremental":
        mean = h@x
        offsets = np.zeros(len(data["benefit_offsets"]))
    else:
        mean = np.asarray(data["mu0"])+h@x
        factor = factor+np.asarray(data["background_factors"])
        offsets = np.asarray(data["benefit_offsets"])
    g = np.asarray(data["benefit_directions"])
    eps = np.asarray(data["risk_allocations"])
    return float(np.min(offsets+g@mean-np.sqrt((1-eps)/eps)*np.linalg.norm(g@factor, axis=1)/np.sqrt(3)))


def assert_incumbent(result):
    assert result["amplitudes"] is not None, result
    assert result["prices"] is not None, result
    assert result["primary"]["status_code"] == 2, result
    assert np.isfinite(result["amplitudes"]).all()


def test_deterministic_one_contract_matches_closed_form_budget_solution():
    data = problem_data(elasticities=[2.], budget=8., actual_amplitude_upper=[100.])
    result = solve(data)
    assert_incumbent(result)
    assert result["amplitudes"] == pytest.approx([4.], abs=5e-5)
    assert result["prices"] == pytest.approx([2.], abs=3e-5)
    assert independent_lcb(data, result["amplitudes"]) == pytest.approx(4., abs=5e-5)


def test_nonzero_variance_solution_matches_cantelli_and_hard_budget():
    data = problem_data(response_factors=[[.2]], budget=1.2)
    result = solve(data)
    assert_incumbent(result)
    assert result["amplitudes"] == pytest.approx([1.], abs=3e-5)
    assert independent_lcb(data, result["amplitudes"]) == pytest.approx(1-.2*np.sqrt(3), abs=3e-5)
    assert 1.2*result["amplitudes"][0]**2 <= 1.2+1e-5


def test_optimizer_retains_common_factor_cancellation():
    data = problem_data(background_factors=[[.4]], response_factors=[[-.4]], budget=1.4)
    result = solve(data)
    assert_incumbent(result)
    assert result["amplitudes"] == pytest.approx([1.], abs=3e-5)
    assert independent_lcb(data, result["amplitudes"]) == pytest.approx(1., abs=5e-5)


def test_multiple_excitation_requirements_balance_without_forced_spread():
    data = problem_data(
        mu0=[0., 0.], background_factors=np.zeros((2, 1)).tolist(),
        profiles=np.eye(2).tolist(), response_factors=np.zeros((2, 1)).tolist(),
        elasticities=[1., 1.], reduction_hours=[1., 1.], price_upper=[4., 4.],
        actual_amplitude_upper=[4., 4.], benefit_directions=np.eye(2).tolist(),
        benefit_offsets=[0., 0.], risk_allocations=[.05, .05], contract_ids=["a", "b"], budget=2.,
    )
    result = solve(data)
    assert_incumbent(result)
    assert result["amplitudes"] == pytest.approx([1., 1.], abs=4e-5)
    assert independent_lcb(data, result["amplitudes"]) == pytest.approx(1., abs=4e-5)


def test_secondary_fee_objective_removes_useless_payments():
    data = problem_data(
        mu0=[2.], profiles=[[0., 1.]], response_factors=[[0.], [0.]],
        elasticities=[1., 1.], reduction_hours=[1., 1.], price_upper=[1., 0.],
        actual_amplitude_upper=[2., 2.], contract_ids=["redundant", "disabled"],
    )
    result = solve(data)
    assert_incumbent(result)
    assert result["amplitudes"] == pytest.approx([0., 0.], abs=1e-5)
    assert result["secondary"] is not None
    assert result["secondary"]["status_code"] == 2


def test_negative_baseline_lower_bound_does_not_force_buying():
    data = problem_data(mu0=[-2.], profiles=[[-1.]])
    result = solve(data)
    assert_incumbent(result)
    assert result["amplitudes"] == pytest.approx([0.], abs=1e-5)
    assert independent_lcb(data, result["amplitudes"]) == pytest.approx(-2., abs=1e-5)


def test_zero_budget_and_zero_price_cap_preserve_no_dr_solution():
    for change in ({"budget": 0.}, {"price_upper": [0.]}):
        result = solve(problem_data(**change))
        assert_incumbent(result)
        assert result["amplitudes"] == pytest.approx([0.], abs=1e-5)


def test_device_robust_bound_limits_realized_amplitude():
    data = problem_data(response_factors=[[.5]], budget=10., device_matrix=[[1.]], device_rhs=[.9])
    result = solve(data)
    assert_incumbent(result)
    x = result["amplitudes"][0]
    assert x == pytest.approx(.6, abs=4e-5)
    assert max(x*(1+.5*z) for z in (-1., 1.)) <= .9+1e-5


def test_misocp_matches_all_fixed_type_socp_assignments():
    data = problem_data(
        profiles=[[1., .8, .6, 1.4]], response_factors=[[0.]]*4,
        elasticities=[1.]*4, reduction_hours=[1.]*4,
        price_upper=[1.]*4, actual_amplitude_upper=[1.]*4,
        contract_ids=["a0", "a1", "b0", "b1"], exclusive_groups=[[0, 1], [2, 3]],
    )
    # Optional none choice is retained for each device.
    assignments = [list(i for i in pair if i is not None)
                   for pair in itertools.product((None, 0, 1), (None, 2, 3))]
    references = [solve(data, mode="socp", allowed_contracts=allowed) for allowed in assignments]
    assert all(r["amplitudes"] is not None for r in references)
    best = max(independent_lcb(data, r["amplitudes"]) for r in references)
    result = solve(data, mode="misocp")
    assert_incumbent(result)
    assert independent_lcb(data, result["amplitudes"]) == pytest.approx(best, abs=7e-5)
    assert best == pytest.approx(np.hypot(1., 1.4), abs=7e-5)
    x = np.asarray(result["amplitudes"])
    assert np.count_nonzero(x[[0, 1]] > 1e-5) <= 1
    assert np.count_nonzero(x[[2, 3]] > 1e-5) <= 1


def test_socp_cannot_silently_relax_same_device_type_exclusivity():
    data = problem_data(
        profiles=[[1., 1.]], response_factors=[[0.], [0.]],
        elasticities=[1., 1.], reduction_hours=[1., 1.], price_upper=[1., 1.],
        actual_amplitude_upper=[1., 1.], contract_ids=["a", "b"], exclusive_groups=[[0, 1]],
    )
    with pytest.raises(ValueError):
        solve(data, mode="socp")
    with pytest.raises(ValueError):
        solve(data, mode="socp", allowed_contracts=[0, 1])


def test_infeasibility_is_not_a_zero_fee_success():
    data = problem_data(device_matrix=[[0.]], device_rhs=[-1.])
    result = solve(data)
    assert result["amplitudes"] is None
    assert result["prices"] is None
    assert result["primary"]["status_code"] in (3, 4)


def test_all_contracts_inactive_reports_baseline_instead_of_failure():
    data = problem_data(mu0=[2.], background_factors=[[.1]])
    result = solve(data, allowed_contracts=[])
    assert_incumbent(result)
    assert result["amplitudes"] == [0.]
    assert independent_lcb(data, result["amplitudes"]) == pytest.approx(2-.1*np.sqrt(3))


def test_incremental_optimizer_is_unchanged_by_background_moments():
    data = problem_data(benefit_mode="incremental", response_factors=[[.2]], budget=1.2)
    a = solve(data)
    other = dict(data, mu0=[73.], background_factors=[[-51.]])
    b = solve(other)
    assert_incumbent(a)
    assert_incumbent(b)
    assert a["amplitudes"] == pytest.approx(b["amplitudes"], abs=2e-5)
    assert independent_lcb(data, a["amplitudes"]) == pytest.approx(
        independent_lcb(other, b["amplitudes"]), abs=2e-5)


def test_tiny_time_budget_preserves_unresolved_status_instead_of_throwing():
    p = PriceProblem.from_dict(problem_data())
    result = solve_price(p, time_limit=1e-6)
    assert "primary" in result
    assert result["primary"]["status_code"] in (2, 9, 11)
    if result["amplitudes"] is None:
        assert result["prices"] is None
    else:
        assert np.isfinite(result["amplitudes"]).all()
