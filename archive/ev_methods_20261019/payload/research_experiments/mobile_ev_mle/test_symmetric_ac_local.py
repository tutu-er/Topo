"""Independent bounded-QP and AC finite-difference checks of the local stage."""
import inspect

import numpy as np
import pytest

try:
    from .symmetric_ac_local import (baseline_from_history, historical_profile_geometry,
        local_symmetric_scores, mean_and_jacobian, solve_bounded_soft_qp)
    from .likelihood import MeasurementModel, Observations, RadialCandidate, ac_predict
    from .design import EVAction
except ImportError:
    from symmetric_ac_local import (baseline_from_history, historical_profile_geometry,
        local_symmetric_scores, mean_and_jacobian, solve_bounded_soft_qp)
    from research_experiments.mobile_ev_mle.likelihood import MeasurementModel, Observations, RadialCandidate, ac_predict
    from research_experiments.mobile_ev_mle.design import EVAction


def _model():
    return MeasurementModel(p_sigma=.0006, q_sigma=.0004, v_sigma=.00012,
        master_p_sigma=.001, master_q_sigma=.001,
        bias_p=np.array([.0001, -.0002, .0003]),
        bias_q=np.array([-.0001, .0002, -.0003]), bias_v=np.array([.00003, -.00004, .00002]))


def _inputs(t=3):
    candidate = RadialCandidate.from_supports("a", 3, ((0, 1),))
    r = np.array([.005, .006, .028, .031, .034])
    x = np.array([.0035, .004, .011, .013, .015])
    time = np.arange(t, dtype=float)
    p = .12 + .015*np.sin(time[:, None]*.61+np.arange(3))
    q = .04 + .009*np.cos(time[:, None]*.93+np.arange(3)*.83)
    sp = .10+.020*np.sin(time*.71)
    sq = .035+.010*np.cos(time*.51)
    return candidate, r, x, p, q, sp, sq


def test_bounded_scalar_qp_optimum_is_hand_calculated_active_bound():
    # Unconstrained min (2+z)^2+z^2 is z=-1. Box forces z=-.25.
    result = solve_bounded_soft_qp([2.], [[1.]], [[1.]], [-.25], [.5])
    assert result["success"]
    assert result["step"] == pytest.approx([-.25])
    assert result["proxy_score"] == pytest.approx(3.125)
    assert result["active_mask"] == pytest.approx([-1])
    assert result["optimality"] < 1e-10
    assert result["qp_tangent_lower_bound_numeric"] == pytest.approx(3.125)
    assert result["qp_gap_numeric"] < 1e-12


def test_unbounded_inside_box_matches_two_nuisance_closed_form():
    result = solve_bounded_soft_qp([2.], [[1., -1.]], np.eye(2), [-10., -10.], [10., 10.])
    assert result["step"] == pytest.approx([-2/3, 2/3])
    assert result["proxy_score"] == pytest.approx(4/3)
    assert result["proxy_future_distance_squared"] == pytest.approx(4/9)
    assert result["proxy_history_penalty"] == pytest.approx(8/9)


def test_source_box_contains_fixed_source_and_cannot_increase_proxy_minimum():
    fixed = solve_bounded_soft_qp([2.], [[1.]], [[1.]], [-.2], [.2])
    free = solve_bounded_soft_qp([2.], [[1., 1.]], [[1., 0.]], [-.2, -.2], [.2, .2])
    assert free["proxy_score"] <= fixed["proxy_score"]
    assert free["proxy_score"] == pytest.approx(2.6)


def test_coordinate_scaling_changes_no_physical_qp_solution():
    args = ([2., -1.], [[1., 2.], [.4, -.5]], [[.3, 0.], [0., .7]], [-.5, -.8], [.9, .3])
    ordinary = solve_bounded_soft_qp(*args)
    scaled = solve_bounded_soft_qp(*args, scale=[.01, 20.])
    assert scaled["step"] == pytest.approx(ordinary["step"], abs=1e-10)
    assert scaled["proxy_score"] == pytest.approx(ordinary["proxy_score"], abs=1e-10)


def test_profile_history_geometry_matches_independent_nuisance_least_squares():
    jb = np.array([[1., 0.], [0., 2.], [2., 1.], [-1., .5]])
    # Duplicate columns explicitly exercise rank-deficient nuisance projection.
    jw = np.array([[1., 2.], [0., 0.], [1., 2.], [0., 0.]])
    db = np.array([.3, -.8])
    result = historical_profile_geometry(jb, jw)
    nuisance = np.linalg.lstsq(jw, -jb@db, rcond=None)[0]
    independent = np.linalg.norm(jb@db+jw@nuisance)**2
    assert db@result["precision"]@db == pytest.approx(independent)
    assert np.linalg.norm(result["factor"]@db)**2 == pytest.approx(independent)
    assert result["state_rank"] == 1
    assert result["state_compensation"]@db == pytest.approx(nuisance)


def test_complete_ac_line_jacobian_matches_central_finite_differences():
    candidate, r, x, p, q, sp, sq = _inputs()
    model = _model()
    analytic = mean_and_jacobian(candidate, r, x, p, q, sp, sq, 1., model)
    beta = np.r_[r, x]
    epsilon = 2e-7
    numerical = []
    for column in range(len(beta)):
        plus, minus = beta.copy(), beta.copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        gp = mean_and_jacobian(candidate, plus[:candidate.m], plus[candidate.m:], p, q, sp, sq, 1., model, with_jacobian=False)
        gm = mean_and_jacobian(candidate, minus[:candidate.m], minus[candidate.m:], p, q, sp, sq, 1., model, with_jacobian=False)
        numerical.append((gp["mean"]-gm["mean"])/(2*epsilon))
    np.testing.assert_allclose(analytic["line_jacobian"], np.column_stack(numerical), rtol=2e-6, atol=2e-4)
    # Terminal P/Q have zero line derivatives; master P/Q are retained and informative.
    assert not analytic["line_jacobian"][:2*p.size].any()
    assert np.linalg.norm(analytic["line_jacobian"][-2*len(p):]) > 0


def test_complete_ac_source_jacobian_matches_per_time_finite_differences():
    candidate, r, x, p, q, sp, sq = _inputs()
    model = _model()
    geometry = mean_and_jacobian(candidate, r, x, p, q, sp, sq, 1., model)
    epsilon, local = 2e-7, 2*candidate.n+2
    for time in range(len(p)):
        for reactive in (False, True):
            xp, xm = (sq.copy(), sq.copy()) if reactive else (sp.copy(), sp.copy())
            xp[time] += epsilon
            xm[time] -= epsilon
            plus = mean_and_jacobian(candidate, r, x, p, q, sp if reactive else xp,
                xp if reactive else sq, 1., model, with_jacobian=False)
            minus = mean_and_jacobian(candidate, r, x, p, q, sp if reactive else xm,
                xm if reactive else sq, 1., model, with_jacobian=False)
            column = time*local+2*candidate.n+int(reactive)
            np.testing.assert_allclose(geometry["state_jacobian"][:, column],
                (plus["mean"]-minus["mean"])/(2*epsilon), rtol=2e-6, atol=2e-4)


def test_common_forecast_corrects_bias_and_adds_no_ev_implicitly():
    model = _model()
    p = np.array([[.1, .2, .3], [.2, .3, .4]])
    q = np.array([[.04, .05, .06], [.05, .06, .07]])
    history = Observations(p=p+model.bias_p, q=q+model.bias_q, v=np.ones_like(p),
        master_p=np.ones(2), master_q=np.ones(2))
    forecast = baseline_from_history(history, model, 4)
    np.testing.assert_allclose(forecast["p"], np.tile(p.mean(axis=0), (4, 1)))
    np.testing.assert_allclose(forecast["q"], np.tile(q.mean(axis=0), (4, 1)))
    assert "truth" not in inspect.signature(local_symmetric_scores).parameters
    assert "future_y" not in inspect.signature(local_symmetric_scores).parameters


def test_history_only_pipeline_keeps_all_pairs_and_backsubstitutes_ac():
    candidate, r, x, p, q, sp, sq = _inputs(t=12)
    rival = RadialCandidate.from_supports("b", 3, ((0, 2),))
    model = _model()
    predicted = ac_predict(candidate, r, x, p, q, sp, sq)
    history = Observations(p=p+model.bias_p, q=q+model.bias_q, v=predicted["v"]+model.bias_v,
        master_p=predicted["master_p"], master_q=predicted["master_q"])
    charge = np.zeros((4, 3)); charge[:, 0] = .15
    actions = [EVAction("none", np.zeros_like(charge), np.zeros_like(charge), np.zeros(4, int), "none"),
               EVAction("charge", charge, np.zeros_like(charge), np.zeros(4, int), "constant")]
    result = local_symmetric_scores([candidate, rival], history, model, actions, multistart=1, max_nfev=150)
    assert result["status"] == "completed"
    assert result["future_noise_observations_used"] is False
    assert result["covariance_channels"] == ["p", "q", "v", "master_p", "master_q"]
    np.testing.assert_allclose(result["forecast"]["baseline_p"], np.tile(p.mean(axis=0), (4, 1)))
    fixed = result["variants"]["fixed"]["actions"]["charge"]["pairwise"][0]
    source = result["variants"]["local_refit"]["actions"]["charge"]["pairwise"][0]
    assert fixed["success"] and source["success"]
    assert source["proxy_score"] <= fixed["proxy_score"]+1e-7
    assert all(fixed["nonlinear_physical_feasible"])
    assert fixed["max_ac_equation_residual"] < 2e-10
    assert fixed["max_power_balance_residual"] < 2e-10
    assert fixed["nonlinear_global_certificate"] is False
    assert np.max(np.abs(np.asarray(source["source_steps"]))) <= .02+1e-12


def test_single_mode_interface_and_invalid_historical_fit_are_explicit():
    from types import SimpleNamespace
    candidate, r, x, p, q, sp, sq = _inputs(t=4)
    rival = RadialCandidate.from_supports("b", 3, ((0, 2),))
    model = _model()
    predictions = ac_predict(candidate, r, x, p, q, sp, sq)
    history = Observations(p=p+model.bias_p, q=q+model.bias_q, v=predictions["v"]+model.bias_v,
        master_p=predictions["master_p"], master_q=predictions["master_q"])
    zero = np.zeros((4, 3))
    action = EVAction("none", zero, zero, np.zeros(4, int), "none")
    bad = SimpleNamespace(success=False, loss=np.inf, message="known computational failure")
    result = local_symmetric_scores([candidate, rival], history, model, [action],
        historical_fits={"a": bad, "b": bad}, source_mode="fixed")
    assert result["source_mode"] == "fixed"
    assert result["status"] == "historical_fit_failure"
    assert result["actions"] == {} and result["selected"] is None
    assert len(result["historical_failures"]) == 2


def test_missing_master_anchors_cannot_be_silently_accepted():
    candidate, _, _, p, q, _, _ = _inputs()
    history = Observations(p=p, q=q, v=np.ones_like(p))
    with pytest.raises(ValueError, match="master P/Q"):
        local_symmetric_scores([candidate, candidate], history, _model(), [])
