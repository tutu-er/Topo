"""Stage-A checks: intercepts, H-only information, units and joint Taylor model."""
from types import SimpleNamespace

import numpy as np
import pytest

try:
    from .adaptive_history import build_history_context, build_frozen_pairs, evaluate_frozen_pair_actual
    from .likelihood import MeasurementModel, Observations, RadialCandidate, ac_predict
    from .symmetric_ac_local import mean_and_jacobian, whiten_observations
except ImportError:
    from adaptive_history import build_history_context, build_frozen_pairs, evaluate_frozen_pair_actual
    from research_experiments.mobile_ev_mle.likelihood import MeasurementModel, Observations, RadialCandidate, ac_predict
    from research_experiments.mobile_ev_mle.symmetric_ac_local import mean_and_jacobian, whiten_observations


def _fixture(count=6):
    candidates = [RadialCandidate.from_supports("clade01", 3, ((0, 1),)),
                  RadialCandidate.from_supports("clade02", 3, ((0, 2),))]
    meter = MeasurementModel(p_sigma=.0006, q_sigma=.0004, v_sigma=.00012,
        master_p_sigma=.001, master_q_sigma=.001,
        bias_p=np.array([.0001, -.0002, .0003]),
        bias_q=np.array([-.0001, .0002, -.0003]), bias_v=np.array([.00003, -.00004, .00002]))
    r, x = np.array([.005, .006, .028, .031, .034]), np.array([.0035, .004, .011, .013, .015])
    time = np.arange(count, dtype=float)
    p = .12+.006*np.sin(.65*time[:, None]+np.arange(3))
    q = .04+.005*np.cos(.9*time[:, None]+np.arange(3))
    sp, sq = .10+.012*np.sin(.55*time), .035+.007*np.cos(.61*time)
    physical = ac_predict(candidates[0], r, x, p, q, sp, sq)
    # Nonzero historical measurement residual is intentional. Feasible
    # supplied expansion points need not be stationary: the stack must retain it.
    history = Observations(p=p+meter.bias_p+.2*meter.p_sigma,
        q=q+meter.bias_q-.3*meter.q_sigma, v=physical["v"]+meter.bias_v+.4*meter.v_sigma,
        master_p=physical["master_p"]+.5*meter.master_p_sigma,
        master_q=physical["master_q"]-.2*meter.master_q_sigma)
    fits = {}
    observed = whiten_observations(history, meter)
    for candidate in candidates:
        geometry = mean_and_jacobian(candidate, r, x, p, q, sp, sq, 1., meter)
        loss = float(np.sum((geometry["mean"]-observed)**2))
        fits[candidate.name] = SimpleNamespace(r=r.copy(), x=x.copy(), p=p.copy(), q=q.copy(),
            source_p=sp.copy(), source_q=sq.copy(), success=True, loss=loss, status=2,
            message="feasible nonstationary historical expansion fixture", nfev=0,
            optimality=1., diagnostics={"test_expansion_fixture": True})
    return history, candidates, meter, fits


def _context(**kwargs):
    history, candidates, meter, fits = _fixture()
    return build_history_context(history, candidates, meter, historical_fits=fits,
        horizon=3, max_centers=1, max_attempts_per_candidate=1, **kwargs)


def test_full_history_residual_intercepts_are_retained_at_nonstationary_points():
    context = _context()
    pair = build_frozen_pairs(context, source_mode="fixed")[0]
    md = pair["metadata"]
    d = sum(md["history_row_widths"])
    assert np.linalg.norm(pair["c"][:d]) > 0
    assert not pair["B"][:d].any()
    actual = evaluate_frozen_pair_actual(context, pair, np.zeros(9), np.zeros(len(pair["lower"])))
    np.testing.assert_allclose(actual["actual_residual"], pair["c"], atol=1e-9)
    assert actual["actual_soft_cost"] == pytest.approx(actual["affine_soft_cost"], abs=1e-8)
    assert sum(actual["actual_history_costs"]) == pytest.approx(md["historical_center_cost"])
    assert pair["F"].shape == (0, len(pair["lower"]))
    assert context["summary"]["centres"]["clade01"][0]["reconstructed_kkt_violation"] > 0


def test_kw_control_jacobian_matches_original_ac_central_difference():
    context = _context()
    nominal = np.full(9, .2)
    pair = build_frozen_pairs(context, nominal, source_mode="fixed")[0]
    zeros = np.zeros(len(pair["lower"]))
    epsilon = .001
    for column in (0, 2, 4, 8):
        plus, minus = nominal.copy(), nominal.copy()
        plus[column] += epsilon; minus[column] -= epsilon
        gp = evaluate_frozen_pair_actual(context, pair, plus, zeros)["actual_residual"]
        gm = evaluate_frozen_pair_actual(context, pair, minus, zeros)["actual_residual"]
        np.testing.assert_allclose(pair["B"][:, column], (gp-gm)/(2*epsilon), rtol=3e-6, atol=1e-6)
    assert pair["metadata"]["base_kva"] == 20.
    assert pair["metadata"]["control_order"] == "time-major"


def test_history_and_future_source_columns_match_joint_ac_difference():
    context = _context()
    pair = build_frozen_pairs(context)[0]
    md = pair["metadata"]
    full_indices = [md["variable_layout"][0]["history_state"][0]+6,
                    md["variable_layout"][0]["future_source"][0]+1,
                    md["variable_layout"][1]["future_source"][0]]
    epsilon = 2e-7
    for full_index in full_indices:
        column = md["free_indices"].index(full_index)
        plus = np.zeros(len(pair["lower"])); minus = plus.copy()
        plus[column] += epsilon; minus[column] -= epsilon
        gp = evaluate_frozen_pair_actual(context, pair, np.zeros(9), plus)["actual_residual"]
        gm = evaluate_frozen_pair_actual(context, pair, np.zeros(9), minus)["actual_residual"]
        np.testing.assert_allclose(pair["A"][:, column], (gp-gm)/(2*epsilon), rtol=3e-6, atol=3e-4)


def test_joint_control_and_line_taylor_error_decreases_quadratically():
    context = _context()
    pair = build_frozen_pairs(context, source_mode="fixed")[0]
    md = pair["metadata"]
    # Simultaneously change a physical line and commanded EV power. This
    # checks the previously omitted u*delta_beta cross/Taylor discrepancy.
    full_index = md["variable_layout"][0]["beta"][0]+2
    column = md["free_indices"].index(full_index)
    z = np.zeros(len(pair["lower"])); z[column] = .0015
    u = np.zeros(9); u[0] = .6; u[4] = .3
    full = evaluate_frozen_pair_actual(context, pair, u, z)
    half = evaluate_frozen_pair_actual(context, pair, u/2, z/2)
    assert full["physically_feasible"] and half["physically_feasible"]
    assert full["joint_taylor_error_norm"] > 1e-7
    assert half["joint_taylor_error_norm"] < .30*full["joint_taylor_error_norm"]


def test_known_bias_and_past_ev_are_removed_only_from_forecast_baseline():
    history, candidates, meter, fits = _fixture()
    previous_kw = np.full(history.p.shape, .4)
    context = build_history_context(history, candidates, meter, historical_fits=fits,
        horizon=3, max_centers=1, max_attempts_per_candidate=1, known_history_ev_kw=previous_kw)
    expected = (history.p-meter.bias_p-previous_kw/20).mean(axis=0)
    np.testing.assert_allclose(context["scenarios"][0]["p0"], np.tile(expected, (3, 1)))
    centre = context["centres"]["clade01"][0]
    # Historical fitting always uses total meters; no subtraction in r_H.
    np.testing.assert_allclose(centre["history_residual"],
        centre["geometry"]["mean"]-whiten_observations(history, meter))


def test_history_only_trend_and_declared_tracking_scenarios_are_explicit():
    support = dict(scenarios=[dict(name="trend", baseline="ols_trend", source_baseline="ols_trend",
        tracking_gain=1.03, source_shift_p=.002, source_shift_q=-.001,
        provenance="declared_device_gain_plus_H_OLS")])
    context = _context(forecast_support=support)
    scenario = context["scenarios"][0]
    assert not np.allclose(scenario["p0"][0], scenario["p0"][-1])
    assert context["summary"]["constant_prediction_time_symmetry"] is False
    pair = build_frozen_pairs(context)[0]
    assert pair["metadata"]["tracking_gain"] == 1.03
    assert pair["metadata"]["scenario_provenance"] == "declared_device_gain_plus_H_OLS"
    assert pair["metadata"]["source_node"] == 1


def test_constant_prediction_preserves_time_permutation_symmetry_of_nominal_mean_difference():
    context = _context()
    pair = build_frozen_pairs(context, source_mode="fixed")[0]
    u = np.zeros((3, 3)); u[0, 0] = .5; u[2, 1] = .2
    one = evaluate_frozen_pair_actual(context, pair, u, np.zeros(len(pair["lower"])))
    two = evaluate_frozen_pair_actual(context, pair, u[::-1], np.zeros(len(pair["lower"])))
    assert one["actual_soft_cost"] == pytest.approx(two["actual_soft_cost"], abs=1e-9)
    assert context["summary"]["constant_prediction_time_symmetry"]


def test_zero_width_coordinates_are_eliminated_without_losing_constant_rows():
    context = _context(line_trust_fraction=0., history_state_radius=0.)
    pair = build_frozen_pairs(context, source_mode="fixed")[0]
    assert pair["A"].shape[1] == 0
    assert len(pair["lower"]) == 0
    actual = evaluate_frozen_pair_actual(context, pair, np.zeros(9), np.zeros(0))
    assert actual["actual_soft_cost"] == pytest.approx(float(pair["c"]@pair["c"]))


def test_forecast_truth_and_h_plus_future_fit_are_rejected():
    history, candidates, meter, fits = _fixture()
    with pytest.raises(ValueError, match="truth or future noise"):
        build_history_context(history, candidates, meter, forecast_support=[dict(true_atom=1)])
    bad = fits["clade01"]
    bad.p = np.r_[bad.p, bad.p[:1]]; bad.q = np.r_[bad.q, bad.q[:1]]
    context = build_history_context(history, candidates, meter, historical_fits=fits,
        max_centers=1, max_attempts_per_candidate=1)
    assert context["status"] == "history_fit_failure"
    assert any("H only" in row.get("error", "") for row in context["attempts"])
    with pytest.raises(ValueError, match="failed historical"):
        build_frozen_pairs(context)


def test_invalid_required_scenario_is_not_silently_removed():
    context = _context(forecast_support=[dict(name="bad", source_shift_p=10.)])
    with pytest.raises((ValueError, RuntimeError), match="domain|voltage|converge|AC"):
        build_frozen_pairs(context)


def test_flat_seed_search_uses_actual_refits_and_retains_all_attempts():
    history, candidates, meter, _ = _fixture()
    context = build_history_context(history, candidates, meter, horizon=3,
        max_centers=2, max_attempts_per_candidate=3, multistart=1, max_nfev=150, seed=271)
    assert context["status"] == "ready"
    assert len(context["attempts"]) == 10
    assert sum(row["kind"] == "flat_direction_actual_refit" for row in context["attempts"]) == 4
    assert all(row["nfev"] > 0 for row in context["attempts"] if row["success"] and row["optimized"])
    assert all(1 <= count <= 2 for count in context["summary"]["centre_counts"].values())
    assert context["summary"]["wzzt_audit"]["wzzt_default_source_variables_active"] is False
    assert context["summary"]["wzzt_audit"]["wzzt_is_independent_extra_history_evidence"] is False
    pairs = build_frozen_pairs(context, source_mode="fixed")
    expected = np.prod(list(context["summary"]["centre_counts"].values()))
    assert len(pairs) == expected


def test_actual_nonstationary_flat_seeds_are_not_called_new_mles():
    history, candidates, meter, _ = _fixture()
    context = build_history_context(history, candidates, meter, horizon=3,
        max_centers=2, max_attempts_per_candidate=3, multistart=1, seed=271)
    kinds = [c["centre_kind"] for cs in context["centres"].values() for c in cs]
    assert "optimizer_fit" in kinds
    assert "feasible_low_loss_seed" in kinds
    for cs in context["centres"].values():
        minimum = min(row["actual_recomputed_history_loss"] for row in context["attempts"]
                      if row["candidate"] == cs[0]["candidate"].name and row["success"])
        for centre in cs:
            assert centre["history_loss"] <= minimum+12.
            if centre["centre_kind"] == "feasible_low_loss_seed":
                assert not centre["optimized"] and centre["fit"].status == 0
                assert centre["reconstructed_kkt_violation"] > 0


def test_additional_centres_cannot_increase_frozen_family_minimum():
    # Preserve every old pair matrix; append additional interpreted centres.
    # This checks finite proxy-family expansion, not recentered confidence domains.
    from copy import copy
    from scipy.optimize import lsq_linear
    history, candidates, meter, _ = _fixture()
    expanded = build_history_context(history, candidates, meter, horizon=3,
        max_centers=2, max_attempts_per_candidate=3, multistart=1, seed=271)
    restricted = copy(expanded)
    restricted["centres"] = {k:[v[0]] for k,v in expanded["centres"].items()}
    one = build_frozen_pairs(restricted, source_mode="fixed")
    many = build_frozen_pairs(expanded, source_mode="fixed")
    assert len(many) > len(one)
    assert many[0]["name"] == one[0]["name"]
    np.testing.assert_array_equal(many[0]["A"], one[0]["A"])
    def value(pair):
        residual = pair["c"]+pair["B"]@np.full(9, .2)
        solution = lsq_linear(pair["A"]*pair["scale"], -residual,
            bounds=(pair["lower"]/pair["scale"], pair["upper"]/pair["scale"]), method="bvls", tol=1e-10)
        return float(2*solution.cost)
    assert min(value(p) for p in many) <= min(value(p) for p in one)+1e-7
