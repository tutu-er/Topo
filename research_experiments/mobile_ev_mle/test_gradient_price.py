"""Independent full-response, profile-price and failure-retention checks."""
from dataclasses import replace

import numpy as np
import pytest

from research_experiments.mobile_ev_mle.gradient_price import (
    GradientPriceEvaluator, profile_margin_gradient, whitened_residual)
from research_experiments.mobile_ev_mle.likelihood import (
    MeasurementModel, Observations, RadialCandidate, ac_predict, fit_candidate)


@pytest.fixture(scope="module")
def fitted_case():
    cherry = RadialCandidate.from_supports("gradient_cherry", 3, [(0, 1)])
    star = RadialCandidate.from_supports("gradient_star", 3, [])
    rng = np.random.default_rng(51)
    t = 12
    r = np.array([.012, .010, .017, .018, .016])
    x = np.array([.006, .005, .009, .011, .007])
    p = rng.uniform(.03, .13, (t, 3)); q = rng.uniform(.015, .07, (t, 3))
    sp = rng.uniform(.01, .04, t); sq = rng.uniform(-.01, .01, t)
    raw = ac_predict(cherry, r, x, p, q, sp, sq)
    bp, bq, bv = np.array([.0002, -.0001, .0003]), .0001, -.00005
    model = MeasurementModel(.001, .001, .0003, .001, .001,
                             bias_p=bp, bias_q=bq, bias_v=bv)
    history = Observations(p+bp+rng.normal(0, .001, p.shape),
        q+bq+rng.normal(0, .001, q.shape), raw["v"]+bv+rng.normal(0, .0003, p.shape),
        1., raw["master_p"]+rng.normal(0, .001, t),
        raw["master_q"]+rng.normal(0, .001, t))
    candidates = (cherry, star)
    fits = {c.name:fit_candidate(c, history, model, multistart=1, max_nfev=300,
                                linear_solver="exact", tol=1e-10, seed=12)
            for c in candidates}
    assert all(f.success for f in fits.values())
    return history, candidates, fits, model


def evaluator(fitted_case, **options):
    history, candidates, fits, model = fitted_case
    defaults = dict(horizon=2, q_over_p=.15,
                    fit_options=dict(multistart=1, max_nfev=300))
    defaults.update(options)
    return GradientPriceEvaluator(history, candidates, fits, model, **defaults)


def _white_readings(observations, meter):
    blocks = []
    for name in ("p", "q", "v", "master_p", "master_q"):
        value = np.asarray(getattr(observations, name))
        blocks.append((value/getattr(meter, name+"_sigma")).reshape(observations.t, -1))
    return np.concatenate(blocks, axis=1).ravel()


def test_code_residual_sign_and_half_sse_are_exact(fitted_case):
    history, _, fits, model = fitted_case
    for fit in fits.values():
        residual = whitened_residual(fit, history, model)
        assert residual@residual == pytest.approx(fit.loss, rel=1e-12)
        expected = (fit.predicted["p"]-history.p)/model.p_sigma
        np.testing.assert_allclose(residual.reshape(history.t, 11)[:, :3], expected)


def test_local_nonzero_residual_p_price_cancels_along_physical_response():
    # Independent two-output counterexample after fitting a free true P.
    eps = .2
    ra = np.array([eps/2, -eps/2])  # code sign: fitted minus measured
    rb = -ra
    assert .5*ra@ra > 0 and .5*rb@rb == pytest.approx(.5*ra@ra)
    physical = profile_margin_gradient(np.array([[1.], [1.]]), ra, rb)
    pq_only = profile_margin_gradient(np.array([[1.], [0.]]), ra, rb)
    np.testing.assert_allclose(physical, 0.)
    np.testing.assert_allclose(pq_only, eps)


def test_complete_ac_meter_response_jacobian_matches_independent_difference(fitted_case):
    ev = evaluator(fitted_case)
    name = ev.candidates[0].name
    u, step = np.array([2., 3.]), 1e-4
    response = ev.future_response(name, 0, u)
    numerical = np.zeros_like(response["jacobian"])
    for t in range(2):
        direction = np.eye(2)[t]*step
        plus = ev.future_response(name, 0, u+direction)["observations"]
        minus = ev.future_response(name, 0, u-direction)["observations"]
        numerical[:, t] = (_white_readings(plus, ev.future_model)-
                            _white_readings(minus, ev.future_model))/(2*step)
    np.testing.assert_allclose(response["jacobian"], numerical, rtol=3e-7, atol=5e-7)
    # Master P includes the direct EV consumption and its AC loss response.
    assert response["jacobian"].reshape(2, 11, 2)[0, 9, 0] > 50.
    baseline = ev.forecasts[name].p
    np.testing.assert_allclose(response["physical"]["p"][:, 0]-baseline[:, 0], u/20.)


def test_profile_price_matches_full_ac_joint_refit_difference(fitted_case):
    ev = evaluator(fitted_case)
    u, h = np.array([2., 3.]), .001
    center = ev.evaluate(0, u)
    assert center["success"], center["failures"]
    assert center["half_SSE"] and not center["global_optimality_certified"]
    for t in range(2):
        step = np.eye(2)[t]*h
        plus, minus = ev.evaluate(0, u+step), ev.evaluate(0, u-step)
        assert plus["success"] and minus["success"]
        for pair, p, m in zip(center["pairs"], plus["pairs"], minus["pairs"]):
            numerical = (p["score"]-m["score"])/(2*h)
            assert pair["gradient_kw"][t] == pytest.approx(numerical, rel=3e-4, abs=3e-4)


def test_ordered_margins_decomposition_and_latent_future_power_are_retained(fitted_case):
    history, _, _, _ = fitted_case
    before = {k:np.asarray(getattr(history, k)).copy() for k in ("p", "q", "v", "master_p", "master_q")}
    ev = evaluator(fitted_case)
    result = ev.evaluate(0, [2., 3.])
    assert result["success"] and len(result["pairs"]) == 2
    assert result["score"] == min(p["score"] for p in result["pairs"])
    assert result["fit_calls"] == 4
    assert result["total_ac_calls"] == result["ac_evaluations"]+result["response_ac_calls"]
    assert all(r["initial_source"] == "historical_fit_and_new_meters" for r in result["fit_records"])
    for pair in result["pairs"]:
        np.testing.assert_allclose(pair["gradient_kw"], pair["gradient_pq"]+
                                   pair["gradient_voltage"]+pair["gradient_master"], atol=1e-12)
        sample = pair["samples"][0]
        assert sample["score"] == pytest.approx(sample["history_margin_change"]+sample["future_margin"], abs=1e-10)
        assert pair["joint_half_cost_competitor"] == pytest.approx(
            pair["joint_history_half_cost_competitor"]+pair["future_half_cost_competitor"], abs=1e-10)
    for k, array in before.items():
        np.testing.assert_array_equal(getattr(history, k), array)
    reference = ev.candidates[0].name
    future = ev.future_response(reference, 0, [2., 3.])["observations"]
    rival = result["fit_results"][(reference, 0, ev.candidates[1].name)]
    assert rival.p.shape[0] == history.t+2
    assert np.max(abs(rival.p[-2:]-future.p)) > 1e-6
    assert rival.diagnostics["source_spatial_location_shared"]


def test_fixed_sample_average_then_min_and_reproducible_initialization(fitted_case):
    noise = np.random.default_rng(1703).normal(size=(2, 2, 11))
    ev = evaluator(fitted_case, design_noise_samples=noise)
    first, second = ev.evaluate(0, [2., 3.]), ev.evaluate(0, [2., 3.])
    assert first["success"] and second["success"]
    assert first["prediction_mode"] == "fixed_sample_average"
    assert first["score"] == pytest.approx(second["score"], abs=1e-12)
    for pair in first["pairs"]:
        assert pair["score"] == pytest.approx(np.mean([s["score"] for s in pair["samples"]]))
        np.testing.assert_allclose(pair["gradient_kw"], np.mean([s["gradient"] for s in pair["samples"]], axis=0))
    assert first["score"] == min(p["score"] for p in first["pairs"])
    noise[:] = 100.
    np.testing.assert_allclose(ev.design_noise_samples, np.random.default_rng(1703).normal(size=(2, 2, 11)))


def test_failed_fits_are_counted_and_never_removed_from_competition(fitted_case):
    ev = evaluator(fitted_case, fit_options=dict(multistart=1, max_nfev=1))
    result = ev.evaluate(0, [2., 3.])
    assert not result["success"] and result["score"] is None
    assert len(result["pairs"]) == 2 and not any(p["success"] for p in result["pairs"])
    assert result["fit_calls"] == 4 and result["ac_evaluations"] >= 4
    assert len(result["fit_records"]) == 4 and result["failures"]


def test_prediction_failure_keeps_pairs_and_counts_attempted_ac(fitted_case):
    ev = evaluator(fitted_case, voltage_limits=(.9999, 1.0001))
    result = ev.evaluate(0, [2., 3.])
    assert not result["success"] and len(result["pairs"]) == 2
    assert result["fit_calls"] == 0 and result["response_ac_calls"] == 2
    assert all(f["kind"] == "prediction" for f in result["failures"])


def test_incompatible_time_sigma_requires_explicit_future_model(fitted_case):
    history, candidates, fits, model = fitted_case
    time_model = replace(model, p_sigma=np.full(history.p.shape, model.p_sigma))
    with pytest.raises(ValueError, match="future.p_sigma"):
        GradientPriceEvaluator(history, candidates, fits, time_model, horizon=2)
    ev = GradientPriceEvaluator(history, candidates, fits, time_model, horizon=2, future_model=model,
                               fit_options=dict(multistart=1, max_nfev=300))
    assert ev.joint_model.p_sigma.shape == (history.t+2, history.n)
    assert ev.evaluate(0, [2., 3.])["success"]


def test_historical_fit_and_fixed_history_consistency_checks(fitted_case):
    history, candidates, fits, model = fitted_case
    wrong = dict(fits)
    wrong[candidates[0].name] = replace(fits[candidates[0].name], loss=fits[candidates[0].name].loss+1.)
    with pytest.raises(ValueError, match="does not match"):
        GradientPriceEvaluator(history, candidates, wrong, model, horizon=2)
    ev = evaluator(fitted_case)
    ev.history.p[0, 0] += .1
    with pytest.raises(RuntimeError, match="historical observations were modified"):
        ev.evaluate(0, [2., 3.])
