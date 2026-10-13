"""Closed-form audits independent of the toy's QP/MILP implementation."""
from functools import lru_cache
import inspect

import numpy as np
import pytest

try:
    from research_experiments.mobile_ev_mle import future_power_metric_check as toy
except ImportError:
    import future_power_metric_check as toy


@pytest.mark.parametrize("u", [(0., 0.), (3., 3.), (6., 0.), (0., 6.), (1.7, 4.3)])
def test_future_power_profile_matches_completion_of_squares(u):
    # Independently complete each square:
    # z^2+(a*u+b*z)^2 = (1+b^2)(z+a*b*u/(1+b^2))^2
    #                    + a^2*u^2/(1+b^2).
    expected = 4*u[0]**2/17 + 2.25*u[1]**2/1.01
    expected_z = [-8*u[0]/17, -.15*u[1]/1.01]
    result = toy.solve_pair(toy.free_power_pair(), u)
    assert result["success"]
    assert result["q_lower"] <= expected + 1e-9
    assert result["q_upper"] >= expected - 1e-9
    assert result["q_upper"] == pytest.approx(expected, abs=1e-9)
    assert result["z"] == pytest.approx(expected_z, abs=1e-9)
    assert result["numeric_gap"] < 1e-8
    assert max(abs(value) for value in expected_z) < 10


@pytest.mark.parametrize("u", [(0., 0.), (3., 3.), (6., 0.), (0., 6.), (1.7, 4.3)])
def test_shared_line_history_penalty_uses_one_common_adjustment(u):
    # z^2+(u0+z)^2+(u1+z)^2 has z*=-(u0+u1)/3.
    expected = u[0]**2 + u[1]**2 - (u[0]+u[1])**2/3
    result = toy.solve_pair(toy.shared_line_pair(), u)
    assert result["q_upper"] == pytest.approx(expected, abs=1e-10)
    assert result["q_lower"] <= expected + 1e-10
    assert result["z"][0] == pytest.approx(-(u[0]+u[1])/3, abs=1e-10)


@lru_cache(maxsize=1)
def evidence():
    return toy.run()


def test_milp_ranking_reversal_matches_analytic_global_maxima():
    result = evidence()
    fixed, free = result["fixed_milp"], result["profiled_milp"]
    # With u0+u1=6, both independent closed-form objectives are convex
    # quadratics on [0,6]. Their maxima are at endpoints; strict inequality
    # of endpoint values identifies a unique maximizing timing.
    assert 4 > 2.25
    assert 4/17 < 2.25/1.01
    fixed_optimum, free_optimum = 144., 81/1.01
    for actual, expected in ((fixed, fixed_optimum), (free, free_optimum)):
        assert actual["success"], actual["failures"]
        assert actual["proxy_lower"] <= expected + 1e-6
        assert actual["proxy_upper"] >= expected - 1e-6
        assert actual["proxy_lower"] == pytest.approx(expected, abs=1e-5)
        assert actual["operations_audit"]["feasible"]
        assert actual["enumerated_complete_actions"] == 0
        assert actual["selector_uses_truth"] is False
        assert actual["selector_uses_future_noise"] is False
    assert np.asarray(fixed["plan"]["u_kw"]).ravel() == pytest.approx([6., 0.], abs=1e-6)
    assert np.asarray(free["plan"]["u_kw"]).ravel() == pytest.approx([0., 6.], abs=1e-6)
    assert result["cross_evaluation"]["fixed_selected"]["profiled_metric"]["q_upper"] == pytest.approx(144/17, abs=1e-6)
    assert result["cross_evaluation"]["free_selected"]["fixed_metric"]["q_upper"] == pytest.approx(81., abs=1e-6)


def test_multiple_centers_make_balanced_timing_optimal():
    result = evidence()["multi_center_milp"]
    # For d=u0-u1 in [-6,6], min((d-6)^2,(d+6)^2)=(6-|d|)^2.
    # Its unique maximum is 36 at d=0, hence u=(3,3).
    assert result["success"], result["failures"]
    assert result["proxy_lower"] == pytest.approx(36., abs=1e-5)
    assert result["proxy_upper"] >= 36 - 1e-6
    assert np.asarray(result["plan"]["u_kw"]).ravel() == pytest.approx([3., 3.], abs=1e-6)
    assert result["operations_audit"]["feasible"]
    assert result["enumerated_complete_actions"] == 0


def test_gain_and_operations_baseline_have_same_energy_and_sse_scale():
    result = evidence()
    baseline = result["constant_charge_baseline"]
    assert baseline["operations_audit"]["feasible"]
    assert baseline["operations_audit"]["grid_energy_kwh"] == pytest.approx(6.)
    assert baseline["fixed_metric"]["q_upper"] == pytest.approx(56.25)
    assert baseline["profiled_metric"]["q_upper"] == pytest.approx(36/17 + 20.25/1.01)
    assert result["gain_vs_constant_charge"]["fixed_metric"]["gain_lower"] == pytest.approx(87.75, abs=1e-5)
    profiled_gain = 81/1.01 - (36/17 + 20.25/1.01)
    assert result["gain_vs_constant_charge"]["profiled_metric"]["gain_lower"] == pytest.approx(profiled_gain, abs=1e-5)
    fixed_profiled_gain = 144/17 - (36/17 + 20.25/1.01)
    assert fixed_profiled_gain < 0
    assert result["gain_vs_constant_charge"]["fixed_selected_under_profiled_metric"]["gain_lower"] == pytest.approx(fixed_profiled_gain, abs=1e-5)
    assert result["shared_line_audit"]["balanced"]["q_upper"] == pytest.approx(6.)
    assert result["shared_line_audit"]["pulse"]["q_upper"] == pytest.approx(24.)


def test_artifact_metadata_preserves_matrix_source_and_bound_evidence():
    result = evidence()
    assert all(len(value) == 64 for value in result["matrix_hashes"].values())
    assert all(len(value) == 64 for value in result["source_sha256"].values())
    assert "SSE" in result["score_definition"]
    assert result["selector_enumerated_complete_actions"] == 0
    assert result["selector_inputs_exclude_truth_and_future_noise"] is True
    assert tuple(inspect.signature(toy.select).parameters) == ("config", "pairs")
    for name in ("fixed_milp", "profiled_milp", "multi_center_milp"):
        assert result[name]["solver_records"]
        assert result[name]["records"]
