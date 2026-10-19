"""Forecast timing/units, known hand solutions and causal rolling validation."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from .forecast import ForecastConfig, forecast_series, forecast_for, select_on_development


def stamps(count):
    return pd.date_range("2016-01-01", periods=count, freq="15min", tz="UTC")


def test_persistence_and_seasonal_naive_hand_solution():
    values = np.arange(100, dtype=float)
    np.testing.assert_array_equal(forecast_series(values, 3)["values"], [99., 99., 99.])
    pred = forecast_series(values, 3, config=ForecastConfig(method="seasonal_naive"))["values"]
    np.testing.assert_array_equal(pred, [4., 5., 6.])
    short = forecast_series(values[:10], 3, config=ForecastConfig(method="seasonal_naive"))
    assert short["metadata"]["effective_method"] == "persistence"
    assert short["metadata"]["fallback"]


@pytest.mark.parametrize("method", ["persistence", "seasonal_naive", "ridge"])
def test_constant_history_stays_constant(method):
    result = forecast_series(np.full((96, 2), 3.5), 12, timestamps=stamps(96), config=ForecastConfig(method=method))
    np.testing.assert_allclose(result["values"], 3.5, atol=1e-10)
    assert result["metadata"]["future_values_used"] is False


def test_ridge_coefficients_and_scales_come_from_past():
    past = np.column_stack((np.linspace(1., 2., 80), np.linspace(3., 2., 80)))
    config = ForecastConfig(method="ridge", ridge_alpha=3.)
    a = forecast_series(past, 3, timestamps=stamps(80), config=config)
    # No future argument exists; changing an unpassed external future has no
    # effect. Every feature scale is independently checkable on past lag rows.
    b = forecast_series(past.copy(), 3, timestamps=stamps(80), config=config)
    np.testing.assert_allclose(a["values"], b["values"])
    lag = max(config.ar_lags)
    expected = [np.mean(past[lag - k:len(past) - k, 0]) for k in config.ar_lags]
    np.testing.assert_allclose(a["metadata"]["parameters"][0]["feature_center"][1:5], expected)
    assert a["metadata"]["historical_cutoff_utc"] == stamps(80)[-1].isoformat()


def test_exact_AR_one_step_and_recursive_future():
    values = [.5]
    for _ in range(35):
        values.append(.8 * values[-1] + 1.)
    result = forecast_series(values[:30], 6,
        config=ForecastConfig(method="ridge", ridge_alpha=0., ar_lags=(1,), calendar=False))
    np.testing.assert_allclose(result["values"], values[30:36], atol=1e-10)


def test_causal_missing_history_and_irregular_dropped_rows():
    values = np.ones(30); values[10] = np.nan
    out = forecast_series(values, 3)
    assert out["metadata"]["causal_filled_values"] == 1
    values[0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        forecast_series(values, 3)
    with pytest.raises(ValueError, match="regular clock"):
        forecast_series(np.ones(29), 3, timestamps=stamps(30).delete(10))


def test_compatibility_units_bias_Q_and_candidate_source():
    observed = SimpleNamespace(p=np.full((96, 2), .2), q=np.full((96, 2), .08), root_v=np.full(96, 1.01))
    meter = SimpleNamespace(bias_p=.01, bias_q=.005)
    states = {"a": dict(source_p=np.full(96, .03), source_q=np.full(96, -.01)),
              "b": dict(source_p=np.full(96, .05), source_q=np.full(96, .02))}
    result = forecast_for(observed, meter, states, T=4, base_kw=100., timestamps=stamps(96))
    assert result["p"].shape == (4, 2)
    np.testing.assert_allclose(result["p"], .19)
    np.testing.assert_allclose(result["q"], .075)
    np.testing.assert_allclose(result["root_v"], 1.01)
    np.testing.assert_allclose(result["source_by_candidate"]["a"]["source_q"], -.01)
    np.testing.assert_allclose(result["source_by_candidate"]["b"]["source_p"], .05)
    assert result["metadata"]["observed_only"]


def test_unpassed_simulator_future_cannot_change_compatibility_output():
    observed = dict(p=np.ones((48, 1)), q=np.ones((48, 1)) * .2,
                    root_v=1., future=np.ones((12, 1)) * 999)
    state = {"a": dict(source_p=np.zeros(48), source_q=np.zeros(48))}
    first = forecast_for(observed, {}, state)
    observed["future"] *= -1e6
    second = forecast_for(observed, {}, state)
    np.testing.assert_array_equal(first["p"], second["p"])
    np.testing.assert_array_equal(first["source_by_candidate"]["a"]["source_p"],
                                  second["source_by_candidate"]["a"]["source_p"])


def test_sparse_source_fit_requires_real_clock_and_causal_alignment():
    observed = SimpleNamespace(p=np.ones((96, 1)), q=np.ones((96, 1)) * .2, root_v=1.)
    chosen = np.r_[np.arange(0, 94, 2), 95]
    states = {"a": dict(source_p=np.arange(len(chosen), dtype=float) / 100,
                         source_q=np.ones(len(chosen)) * .02)}
    with pytest.raises(ValueError, match="source_times"):
        forecast_for(observed, {}, states, timestamps=stamps(96))
    result = forecast_for(observed, {}, states, timestamps=stamps(96), source_times=stamps(96)[chosen])
    assert result["metadata"]["source"]["a"]["alignment"]["causal_forward_fill"]
    np.testing.assert_allclose(result["source_by_candidate"]["a"]["source_p"], states["a"]["source_p"][-1])
    with pytest.raises(ValueError, match="cutoff"):
        forecast_for(observed, {}, {"a": dict(source_p=np.ones(95), source_q=np.zeros(95))},
                     timestamps=stamps(96), source_times=stamps(96)[:95])


def test_rolling_development_selection_and_evaluation_rejection():
    values = np.ones((96, 2))
    candidates = (ForecastConfig(method="persistence"), ForecastConfig(method="ridge"))
    config, receipt = select_on_development([dict(values=values, timestamps=stamps(96), phase="development")],
        candidates=candidates, origins=(48, 60, 72), horizon=12)
    assert config.method == "persistence"
    assert not receipt["evaluation_used"]
    assert all(row["available_origins"] == 3 for row in receipt["results"])
    with pytest.raises(ValueError, match="cannot select"):
        select_on_development([dict(values=values, phase="evaluation")], candidates=candidates)
    with pytest.raises(ValueError, match="development-only"):
        select_on_development([dict(values=values)], phase="evaluation")


@pytest.mark.parametrize("setting", [dict(method="oracle"), dict(ridge_alpha=-1), dict(ar_lags=(0,)), dict(ar_lags=(1, 1))])
def test_bad_configuration_rejected(setting):
    with pytest.raises(ValueError):
        ForecastConfig(**setting)
