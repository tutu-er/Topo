"""Independent checks for public quality and H-only AC center construction."""
from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest
from .candidates import RadialCandidate
from .likelihood import Observations, ac_predict, FitResult
from .simulation import model
from .single_site_milp import SingleSiteConstraints, check_single_site_plan
from . import information_context as context


def fixture():
    star = RadialCandidate.from_supports("star", 3, ())
    candidates = [star] + [RadialCandidate.from_supports("pair_"+"".join(map(str, support)), 3, (support,))
                          for support in context.TARGET_SUPPORTS]
    t = 5
    rng = np.random.default_rng(20)
    p = .12+rng.normal(0, .002, (t, 3))
    q = .04+rng.normal(0, .001, (t, 3))
    sp = .1+rng.normal(0, .003, t)
    sq = .035+rng.normal(0, .002, t)
    r = np.array([.005, .028, .031, .034])
    x = np.array([.0035, .011, .013, .015])
    meter = model(3)
    physical = ac_predict(star, r, x, p, q, sp, sq)
    observed = Observations(p=p+meter.bias_p, q=q+meter.bias_q,
        v=physical["v"]+meter.bias_v, master_p=physical["master_p"], master_q=physical["master_q"])
    fit = FitResult(star.name, r, x, p, q, sp, sq, meter.bias_p, meter.bias_q, meter.bias_v,
        loss=0., loglik=0., success=True, status=1, message="test exact", nfev=0,
        optimality=0., predicted={**physical, "p": observed.p, "q": observed.q, "v": observed.v},
        diagnostics=dict(joint_parameter_count=2*star.m+8*t))
    fits = {star.name: fit}
    for candidate in candidates[1:]:
        fits[candidate.name] = SimpleNamespace(success=True, loss=0.)
    return observed, meter, candidates, fits


@pytest.mark.parametrize("kind", ["uniform", "temporal", "spatial"])
def test_quality_budget_bias_joint_and_json(kind):
    history, meter, _, _ = fixture()
    future, joint, record = context.quality_models(history, meter, kind, row_permutation=[3, 1, 0, 2])
    counts = np.asarray(record["v_repeat_counts"])
    assert counts.sum() == 60
    np.testing.assert_allclose(future.v_sigma, meter.v_sigma/np.sqrt(counts))
    np.testing.assert_allclose(future.p_sigma, meter.p_sigma/np.sqrt(5))
    np.testing.assert_allclose(future.master_p_sigma, meter.master_p_sigma/np.sqrt(5))
    for channel in ("p", "q", "v"):
        np.testing.assert_array_equal(getattr(future, "bias_"+channel), getattr(meter, "bias_"+channel))
    np.testing.assert_array_equal(joint.v_sigma[:history.t], np.full((history.t, 3), meter.v_sigma))
    np.testing.assert_array_equal(joint.v_sigma[history.t:], future.v_sigma)
    saved = json.loads(json.dumps(context.meter_to_record(joint)))
    restored = context.meter_from_record(saved)
    for name in context.SIGMA_CHANNELS:
        if getattr(joint, name+"_sigma") is None:
            assert getattr(restored, name+"_sigma") is None
        else:
            np.testing.assert_array_equal(getattr(restored, name+"_sigma"), getattr(joint, name+"_sigma"))


def test_quality_known_row_permutation_and_validation():
    history, meter, _, _ = fixture()
    _, _, natural = context.quality_models(history, meter, "spatial")
    _, _, shuffled = context.quality_models(history, meter, "spatial", row_permutation=[2, 0, 3, 1])
    np.testing.assert_array_equal(shuffled["v_repeat_counts"], np.asarray(natural["v_repeat_counts"])[[2, 0, 3, 1]])
    with pytest.raises(ValueError, match="permutation"):
        context.quality_models(history, meter, row_permutation=[0, 1, 1, 3])
    with pytest.raises(ValueError, match="time-constant"):
        context.quality_models(history, replace(meter, p_sigma=np.linspace(.0005, .0007, history.t)[:, None]))


def test_actual_noise_matches_declared_variance():
    history, meter, _, _ = fixture()
    future, _, _ = context.quality_models(history, meter, "spatial")
    rng = np.random.default_rng(827)
    # Direct independent draws under known sigma; no topology/future input.
    draws = np.stack([context.draw_measurement_noise(future, 4, 3, rng)["v"] for _ in range(16000)])
    ratios = draws.var(axis=0, ddof=1)/future.v_sigma**2
    assert np.max(abs(ratios-1)) < .04
    assert np.max(abs(draws.mean(axis=0)/future.v_sigma)) < .03


def test_contracted_embedding_preserves_full_ac():
    history, _, candidates, fits = fixture()
    star, fit = candidates[0], fits["star"]
    original = ac_predict(star, fit.r, fit.x, fit.p, fit.q, fit.source_p, fit.source_q)
    for cherry in candidates[1:]:
        r, x, support = context.embed_star(star, fit, cherry)
        index = list(cherry.supports).index(support)
        assert r[index] == x[index] == 0.
        for j, old_support in enumerate(star.supports):
            target = list(cherry.supports).index(old_support)
            assert r[target] == fit.r[j]
            assert x[target] == fit.x[j]
        contracted = ac_predict(cherry, r, x, fit.p, fit.q, fit.source_p, fit.source_q)
        for channel in ("v", "master_p", "master_q"):
            np.testing.assert_allclose(contracted[channel], original[channel], rtol=0., atol=2e-14)


def test_full_history_white_jacobian_finite_difference():
    history, meter, candidates, fits = fixture()
    star, fit = candidates[0], fits["star"]
    jac, _ = context.history_white_jacobian(star, fit, history, meter)
    state = np.c_[fit.p, fit.q, fit.source_p, fit.source_q]
    z = np.r_[fit.r, fit.x, state.ravel()]
    direction = np.random.default_rng(920).normal(size=len(z))
    direction /= np.max(abs(direction))
    def white(value):
        local = value[2*star.m:].reshape(history.t, 8)
        response = ac_predict(star, value[:star.m], value[star.m:2*star.m], local[:, :3], local[:, 3:6], local[:, 6], local[:, 7])
        return np.concatenate([((response[name]+(getattr(meter, "bias_"+name) if name in ("p", "q", "v") else 0)) /
            getattr(meter, name+"_sigma")).reshape(history.t, -1) for name in context.measurement_channels(history)], axis=1).ravel()
    for eps in (2e-7, 1e-7):
        actual = (white(z+eps*direction)-white(z-eps*direction))/(2*eps)
        np.testing.assert_allclose(actual, jac@direction, rtol=2e-6, atol=3e-5)


@pytest.mark.parametrize("kind", ["global", "local", "random"])
def test_cheap_precision_rule_operationally_feasible_with_bounds(kind):
    history, meter, _, _ = fixture()
    _, _, quality = context.quality_models(history, meter, "spatial")
    constraints = SingleSiteConstraints(availability=np.ones((4, 3), bool))
    selected = context.cheap_quality_plan(kind, quality, constraints, random_seed=99)
    assert selected["success"]
    assert check_single_site_plan(selected["plan"], constraints)["feasible"]
    assert selected["best_bound"] >= selected["incumbent"]-1e-7
    assert selected["uses_truth"] is selected["uses_future_noise"] is False
    if kind == "random":
        again = context.cheap_quality_plan(kind, quality, constraints, random_seed=99)
        assert selected["selected_site"] == again["selected_site"]
        assert selected["incumbent"] == 0.


def test_star_centers_h_only_and_support_gate(monkeypatch):
    history, meter, candidates, fits = fixture()
    centers, record = context.build_centres(history, candidates, fits, meter, refit=False)
    assert record["activated"] and len(centers) == 3 and record["anchor_count"] == 1
    future_truth = np.full((4, 3), 1000.)
    future_truth[:] = -1000.  # No such object is accepted or read by the builder.
    again, repeated = context.build_centres(history, candidates, fits, meter, refit=False)
    for first, second in zip(centers, again):
        np.testing.assert_array_equal(first.r, second.r)
        np.testing.assert_array_equal(first.future.p, second.future.p)
        np.testing.assert_array_equal(first.future.source_p, second.future.source_p)
        assert first.target_support in context.TARGET_SUPPORTS
    fits["star"].loss = 20.
    outside, why = context.build_centres(history, candidates, fits, meter, refit=False)
    assert not outside and why["reason"] == "history_star_outside_computational_support"


def test_two_refit_attempts_keep_duplicates_and_failures(monkeypatch):
    history, meter, candidates, fits = fixture()
    calls = []
    def fake_fit(candidate, observed, noise, **options):
        calls.append(options)
        if len(calls) == 2:
            raise RuntimeError("intentional solver failure")
        return fits["star"]
    monkeypatch.setattr(context.likelihood, "fit_candidate", fake_fit)
    centers, record = context.build_centres(history, candidates, fits, meter)
    assert len(calls) == 2 and len(record["attempts"]) == 2 and len(centers) == 3
    assert record["attempts"][0]["reason"] == "duplicate_explanation"
    assert record["attempts"][1]["reason"] == "refit_exception"
    assert record["attempts"][1]["success"] is False
    assert all(options["multistart"] == 1 for options in calls)
    assert record["original_parameter_box_shrunk"] is False


def test_distinct_supported_refit_can_supply_second_anchor(monkeypatch):
    history, meter, candidates, fits = fixture()
    original = fits["star"]
    alternative = replace(original, r=original.r+np.array([0., .0001, 0., 0.]))
    physical = ac_predict(candidates[0], alternative.r, alternative.x, alternative.p, alternative.q,
                          alternative.source_p, alternative.source_q)
    blocks = []
    for channel in context.measurement_channels(history):
        value = physical[channel]+(getattr(meter, "bias_"+channel) if channel in ("p", "q", "v") else 0)
        blocks.append(((value-getattr(history, channel))/getattr(meter, channel+"_sigma")).ravel())
    residual = np.concatenate(blocks)
    alternative.loss = float(residual@residual)
    monkeypatch.setattr(context.likelihood, "fit_candidate", lambda *a, **k: alternative)
    centers, record = context.build_centres(history, candidates, fits, meter)
    assert len(centers) == 6 and record["anchor_count"] == 2
    assert record["attempts"][0]["retained"] is True
    assert record["attempts"][1]["reason"] == "duplicate_explanation"
    assert all(0 <= row["history_gap_half_sse"] <= 6 for row in record["anchor_records"])

def test_refit_inconsistent_loss_is_preserved_as_failure(monkeypatch):
    history, meter, candidates, fits = fixture()
    bad = replace(fits["star"], r=fits["star"].r+.01, loss=0.)
    monkeypatch.setattr(context.likelihood, "fit_candidate", lambda *a, **k: bad)
    centers, record = context.build_centres(history, candidates, fits, meter)
    assert len(centers) == 3 and len(record["attempts"]) == 2
    assert all(item["reason"] == "refit_exception" for item in record["attempts"])
    assert all("recomputation" in item["error"] for item in record["attempts"])


@pytest.mark.parametrize("fail_second_verification", [False, True])
def test_attempted_ac_call_count_matches_independent_alias_and_module_instrumentation(monkeypatch, fail_second_verification):
    history, meter, candidates, fits = fixture()
    fit = fits["star"]
    physical_ac = ac_predict
    attempted = {"alias": 0, "module": 0}
    def alias_ac(*args, **kwargs):
        attempted["alias"] += 1
        return physical_ac(*args, **kwargs)
    def module_ac(*args, **kwargs):
        attempted["module"] += 1
        # fake refit and independent verification each enter this endpoint.
        if fail_second_verification and attempted["module"] == 4:
            raise RuntimeError("second independent AC verification fails")
        return physical_ac(*args, **kwargs)
    def fake_fit(candidate, observed, noise, **options):
        context.likelihood.ac_predict(candidate, fit.r, fit.x, fit.p, fit.q,
                                      fit.source_p, fit.source_q, observed.root_v)
        return fit
    monkeypatch.setattr(context, "ac_predict", alias_ac)
    monkeypatch.setattr(context.likelihood, "ac_predict", module_ac)
    monkeypatch.setattr(context.likelihood, "fit_candidate", fake_fit)
    centers, record = context.build_centres(history, candidates, fits, meter)
    assert len(centers) == 3
    assert attempted == {"alias": 1, "module": 4}
    assert record["ac_calls"] == sum(attempted.values()) == 5
    assert [item["ac_calls"] for item in record["attempts"]] == [2, 2]
    if fail_second_verification:
        assert record["attempts"][1]["success"] is False
        assert record["attempts"][1]["reason"] == "refit_exception"


def test_nonstationary_flat_point_is_retained_by_actual_H_loss_not_refit(monkeypatch):
    history, meter, candidates, fits = fixture()
    monkeypatch.setattr(context.likelihood, 'fit_candidate', lambda *a, **k: fits['star'])
    centers, record = context.build_centres(history, candidates, fits, meter, retain_perturbed_points=True)
    assert record['anchor_count'] == 2 and len(centers) == 6
    point = record['attempts'][0]['perturbed_point']
    assert point['retained'] and 0 < point['history_gap_half_sse'] <= 6
    assert record['anchor_records'][1]['kind'] == 'feasible_perturbed_H_point'
    assert record['attempts'][0]['reason'] == 'duplicate_explanation'
    center = centers[3]
    actual = ac_predict(center.candidate, center.r, center.x, center.history_p, center.history_q,
        center.history_source_p, center.history_source_q, history.root_v)
    residual = np.concatenate([((actual[name]+(getattr(meter,'bias_'+name) if name in ('p','q','v') else 0)
        -getattr(history,name))/getattr(meter,name+'_sigma')).ravel() for name in context.measurement_channels(history)])
    assert float(residual@residual) == pytest.approx(point['loss_full_SSE'], rel=1e-9, abs=1e-12)
    assert record['ac_calls'] == 5  # H Jacobian + 2 checked points + 2 duplicate-fit verification calls.


def test_flat_point_outside_H_support_is_kept_as_rejection(monkeypatch):
    history, meter, candidates, fits = fixture()
    monkeypatch.setattr(context.likelihood, 'fit_candidate', lambda *a, **k: fits['star'])
    centers, record = context.build_centres(history, candidates, fits, meter,
        retain_perturbed_points=True, gate_half_sse=.0001)
    assert len(centers) == 3 and record['anchor_count'] == 1
    assert all(a['perturbed_point']['reason'] == 'outside_history_gate' for a in record['attempts'])
    assert all(not a['perturbed_point']['retained'] for a in record['attempts'])
