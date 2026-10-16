"""Case independence, physical scale, and history-only forecast contracts."""
from copy import deepcopy
from dataclasses import fields
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_mobile_cases_v2 as old
from . import ev_mobile_v4_cases as cases
from . import likelihood as lk
from .simulation import model


def analytic_values(times, n=3):
    times = np.asarray(times, float)
    phase = np.arange(n) * .5
    p = .12 + .01 * np.sin(2 * np.pi * times[:, None] / 96 + phase)
    ratio = .32 + .04 * np.cos(2 * np.pi * times[:, None] / 32 + phase)
    source_p = .10 + .007 * np.sin(2 * np.pi * times / 24 + .4)
    source_q = .035 + .002 * np.cos(2 * np.pi * times / 12 + .3)
    return p, ratio, source_p, source_q


class HistoricalFit:
    success = True

    def __init__(self, source_p, source_q):
        self.source_p = source_p
        self.source_q = source_q

    @property
    def r(self):
        raise AssertionError("forecast must not inspect line parameters")

    @property
    def x(self):
        raise AssertionError("forecast must not inspect line parameters")


def fake_inputs(count=100, n=3):
    p, ratio, sp, sq = analytic_values(np.arange(count), n)
    meter = SimpleNamespace(bias_p=np.linspace(-.00015, .00015, n),
                            bias_q=np.linspace(.0001, -.0001, n))
    observed = SimpleNamespace(p=p + meter.bias_p, q=p * ratio + meter.bias_q)
    fits = {"star": HistoricalFit(sp.copy(), sq.copy()),
            "clade_01": HistoricalFit(sp.copy() + .003, sq.copy() - .001)}
    return observed, meter, fits


@pytest.fixture(scope="module", params=["n3", "n4"])
def paired_case(request):
    return tuple(cases.make_case(request.param, 2026102107, 1, regime)
                 for regime in ("regular", "shift"))


def test_regimes_share_all_training_observed_meters_assets_and_future_noise(paired_case):
    regular, shift = paired_case
    for key in ("p", "q", "source_p", "source_q", "root_v"):
        np.testing.assert_array_equal(regular[key][:100], shift[key][:100])
    for section in ("train", "validation"):
        for field in fields(lk.Observations):
            name = field.name
            a, b = getattr(regular[section], name), getattr(shift[section], name)
            if a is None or b is None:
                assert a is b
            else:
                np.testing.assert_array_equal(a, b)
    for name in regular["white"]:
        np.testing.assert_array_equal(regular["white"][name], shift["white"][name])
    for key in ("r", "x"):
        np.testing.assert_array_equal(regular["truth"][key], shift["truth"][key])
    assert regular["truth"]["candidate"].supports == shift["truth"]["candidate"].supports
    assert regular["train"].t == 96 and regular["validation"].t == 4
    assert regular["p"].shape == (112, regular["train"].n)
    for key in ("p", "q", "source_p", "source_q"):
        assert np.max(abs(regular[key][100:] - shift[key][100:])) > 0.


def test_ordinary_variation_same_impedance_box_and_legal_base_ac(paired_case):
    regular, _ = paired_case
    n, base_kw = regular["train"].n, regular["base_kw"]
    p = regular["p"] * base_kw
    ratio = regular["q"] / regular["p"]
    assert np.min(p) > 1.7 and np.max(p) < 3.2
    assert np.min(ratio) >= .2 - 1e-12 and np.max(ratio) <= .45 + 1e-12
    assert np.all(np.std(p[:100], axis=0) > .10)
    assert np.all(np.std(ratio[:100], axis=0) > .03)
    # Reactive variation is not a fixed multiple of each active-load series.
    centered_p = p[:100] - p[:100].mean(axis=0)
    q = regular["q"][:100] * base_kw
    residual = q - q.mean(axis=0) - centered_p * (
        np.sum(centered_p * (q - q.mean(axis=0)), axis=0) / np.sum(centered_p ** 2, axis=0))
    assert np.all(np.std(residual, axis=0) > .04)
    original = old.case_config(regular["case_id"], training_points=96)
    assert regular["config"]["training"]["r_bounds"] == original["training"]["r_bounds"]
    assert regular["config"]["training"]["x_bounds"] == original["training"]["x_bounds"]
    assert regular["config"]["ratio_bounds"] == original["ratio_bounds"]
    truth = regular["truth"]
    physical = lk.ac_predict(truth["candidate"], truth["r"], truth["x"],
        regular["p"], regular["q"], regular["source_p"], regular["source_q"], regular["root_v"])
    assert np.min(physical["v"]) > .85 and np.max(physical["v"]) < 1.15
    assert np.max(abs(np.asarray(physical["ac_equation_residual"]))) < 1e-10
    assert regular["provenance"]["source_interface"] == 1
    assert len(regular["bank"]) == 4
    assert all(c.n == n for c in regular["bank"])
    assert regular["provenance"]["field_PQV_available"] is False


def test_total_future_precision_is_uniform_and_unchanged(paired_case):
    regular, _ = paired_case
    meter = model(regular["train"].n)
    for channel in ("p", "q", "v", "master_p", "master_q"):
        sigma = np.asarray(getattr(regular["fm"], channel + "_sigma"))
        expected = np.asarray(getattr(meter, channel + "_sigma")) / np.sqrt(30. / 12.)
        np.testing.assert_allclose(sigma, np.broadcast_to(expected, sigma.shape))
    assert regular["config"]["public_voltage_repeat_counts"] == [2.5] * 12
    assert sum(regular["config"]["public_voltage_repeat_counts"]) == 30.


def test_historical_OLS_recovers_unknown_clean_coefficients_with_fixed_lines():
    observed, meter, fits = fake_inputs()
    before = {name: (fit.source_p.copy(), fit.source_q.copy()) for name, fit in fits.items()}
    forecast = cases.forecast_for(observed, meter, fits, T=12, base_kw=20.)
    p, ratio, sp, sq = analytic_values(np.arange(100, 112))
    np.testing.assert_allclose(forecast["p"], p, atol=1e-14)
    np.testing.assert_allclose(forecast["q"], p * ratio, atol=1e-14)
    np.testing.assert_allclose(forecast["source_by_candidate"]["star"]["source_p"], sp, atol=1e-14)
    np.testing.assert_allclose(forecast["source_by_candidate"]["star"]["source_q"], sq, atol=1e-14)
    for name, fit in fits.items():
        np.testing.assert_array_equal(fit.source_p, before[name][0])
        np.testing.assert_array_equal(fit.source_q, before[name][1])
    assert forecast["metadata"]["uses_regime"] is False
    assert forecast["metadata"]["uses_truth"] is False
    assert forecast["metadata"]["line_parameters_read_or_changed"] is False
    assert forecast["metadata"]["coefficients"]["p"]["rank"] == 9
    assert forecast["metadata"]["no_calibrated_prediction_intervals"]


def test_forecast_cannot_receive_future_or_regime_and_keeps_candidate_sources_separate():
    observed, meter, fits = fake_inputs()
    initial = cases.forecast_for(observed, meter, fits)
    assert set(inspect.signature(cases.forecast_for).parameters) == {"observedD", "Dmeter", "fits", "T", "base_kw"}
    # Objects carrying unrelated/poisoned future attributes cannot influence it.
    observed.truth = object()
    observed.future = np.full((12, 3), np.nan)
    observed.regime = "invalid-simulator-label"
    observed.white = object()
    second = cases.forecast_for(observed, meter, fits)
    for key in ("p", "q", "root"):
        np.testing.assert_array_equal(initial[key], second[key])
    assert initial["metadata"]["calibrated_PQ_sha256"] == second["metadata"]["calibrated_PQ_sha256"]
    a, b = initial["source_by_candidate"]["star"], initial["source_by_candidate"]["clade_01"]
    np.testing.assert_allclose(b["source_p"] - a["source_p"], .003, atol=1e-14)
    np.testing.assert_allclose(b["source_q"] - a["source_q"], -.001, atol=1e-14)


def test_forecasts_are_identical_for_regular_and_shift_observed_data(paired_case):
    regular, shift = paired_case
    def forecast(case):
        n = case["train"].n
        observed = SimpleNamespace(p=np.vstack((case["train"].p, case["validation"].p)),
                                   q=np.vstack((case["train"].q, case["validation"].q)))
        # Deliberately artificial historical fits: no AC estimation is needed
        # to verify the predictor's information boundary and candidate schema.
        fits = {candidate.name: HistoricalFit(np.full(100, .10), np.full(100, .035))
                for candidate in case["bank"]}
        return cases.forecast_for(observed, model(n), fits)
    a, b = forecast(regular), forecast(shift)
    np.testing.assert_array_equal(a["p"], b["p"])
    np.testing.assert_array_equal(a["q"], b["q"])
    assert a["metadata"]["observed_PQ_sha256"] == b["metadata"]["observed_PQ_sha256"]


def test_forecast_scales_and_hashes_follow_actual_historical_inputs():
    observed, meter, fits = fake_inputs()
    a = cases.forecast_for(observed, meter, fits)
    changed = deepcopy(observed); changed.p[-1, 0] += .005
    b = cases.forecast_for(changed, meter, fits)
    assert a["metadata"]["observed_PQ_sha256"] != b["metadata"]["observed_PQ_sha256"]
    assert np.max(abs(a["p"] - b["p"])) > 1e-6
    np.testing.assert_array_equal(a["source_by_candidate"]["star"]["source_p"],
                                  b["source_by_candidate"]["star"]["source_p"])
    fits["star"].source_p += .01
    c = cases.forecast_for(observed, meter, fits)
    assert a["metadata"]["fitted_source_sha256"] != c["metadata"]["fitted_source_sha256"]
    np.testing.assert_allclose(c["source_by_candidate"]["star"]["source_p"] -
                               a["source_by_candidate"]["star"]["source_p"], .01)


def test_clean_future_adds_ev_once_and_keeps_other_terminal_q_unchanged(paired_case):
    regular, _ = paired_case
    n = regular["train"].n
    power = np.zeros((12, n)); power[3:6, 1] = 2.
    base = cases.observe_future(regular, np.zeros_like(power), clean=True)
    ev = cases.observe_future(regular, power, clean=True)
    np.testing.assert_allclose(ev.p - base.p, power / regular["base_kw"], atol=1e-15)
    np.testing.assert_array_equal(ev.q, base.q)


@pytest.mark.parametrize("kwargs", [dict(bank="n5"), dict(family_index=4), dict(regime="oracle"), dict(H=8)])
def test_invalid_case_inputs_rejected(kwargs):
    args = dict(bank="n3", seed=2026102107, family_index=0, regime="regular", H=96)
    args.update(kwargs)
    with pytest.raises(ValueError):
        cases.make_case(**args)


@pytest.mark.parametrize("change", ["nan_p", "bad_source_length", "failed_fit", "missing_sources"])
def test_invalid_historical_forecast_inputs_rejected(change):
    observed, meter, fits = fake_inputs()
    if change == "nan_p":
        observed.p[0, 0] = np.nan
    elif change == "bad_source_length":
        fits["star"].source_p = np.ones(96)
    elif change == "failed_fit":
        fits["star"].success = False
    elif change == "missing_sources":
        fits = {}
    with pytest.raises(ValueError):
        cases.forecast_for(observed, meter, fits)
