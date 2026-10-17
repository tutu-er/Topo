"""Observed-only persistence, seasonal-naive and causal ridge-AR forecasts.

Calendar variables are known clock features, not the simulated process's
harmonic periods. Every coefficient/scale is estimated on a historical prefix.
No true lines, future loads, future source paths, noise or regime labels enter
the planning API. Forecasts are points; no coverage claim is attached.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from collections.abc import Mapping
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from .data import causal_forward_fill, utc_string


@dataclass(frozen=True)
class ForecastConfig:
    method: str = "persistence"
    ridge_alpha: float = 10.
    ar_lags: tuple[int, ...] = (1, 2, 4, 8)
    seasonal_slots: int = 96
    dt_minutes: int = 15
    calendar: bool = True
    max_causal_gap: int = 4

    def __post_init__(self):
        object.__setattr__(self, "ar_lags", tuple(self.ar_lags))
        if self.method not in ("persistence", "seasonal_naive", "ridge"):
            raise ValueError("registered causal forecast method required")
        if (not np.isfinite(self.ridge_alpha) or self.ridge_alpha < 0 or
                not self.ar_lags or any(int(k) != k or k < 1 for k in self.ar_lags) or
                len(set(self.ar_lags)) != len(self.ar_lags) or
                self.seasonal_slots < 1 or self.dt_minutes < 1 or self.max_causal_gap < 0):
            raise ValueError("valid fixed ridge/lags/calendar/gap settings required")


DEFAULT_CANDIDATES = (
    ForecastConfig(method="persistence"), ForecastConfig(method="seasonal_naive"),
    ForecastConfig(method="ridge", ridge_alpha=1.),
    ForecastConfig(method="ridge", ridge_alpha=10.),
    ForecastConfig(method="ridge", ridge_alpha=100.),
)


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _matrix(values):
    values = np.asarray(values, float)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[:, None]
    if values.ndim != 2 or not len(values) or not values.shape[1] or np.isinf(values).any():
        raise ValueError("nonempty time-by-channel historical values required")
    return values, one_dimensional


def _stamps(timestamps, count, config):
    if timestamps is None:
        return None
    stamps = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    if len(stamps) != count or stamps.hasnans or not stamps.is_monotonic_increasing or stamps.has_duplicates:
        raise ValueError("matching strictly increasing observed timestamps required")
    clock = stamps.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    if len(stamps) > 1 and not np.all(np.diff(clock) == np.timedelta64(config.dt_minutes, "m")):
        raise ValueError("forecast observations must retain the regular clock; fill missing values causally, not by dropping rows")
    return stamps


def _clock(stamp):
    hour = stamp.hour + stamp.minute / 60.
    daily, weekly = 2 * np.pi * hour / 24., 2 * np.pi * (stamp.dayofweek + hour / 24.) / 7.
    return [np.sin(daily), np.cos(daily), np.sin(weekly), np.cos(weekly)]


def _feature(values, index, channel, config, stamps):
    row = [1.] + [values[index - lag, channel] for lag in config.ar_lags]
    if config.calendar and stamps is not None:
        row.extend(_clock(stamps[index]))
    return np.asarray(row, float)


def forecast_series(values, horizon, *, timestamps=None, config=None, nonnegative=False):
    """Forecast using only supplied past values, in their unchanged units.

    Ridge AR is fitted separately per channel and recursively propagated.
    Feature centering/scales use fitted past rows only. With too few past rows
    or insufficient seasonal history, fallback is explicitly persistence.
    """
    config = config or ForecastConfig()
    if isinstance(config, Mapping):
        config = ForecastConfig(**config)
    if int(horizon) != horizon or horizon < 1:
        raise ValueError("positive integer future horizon required")
    raw, one_dimensional = _matrix(values)
    data, filled = causal_forward_fill(raw, max_gap=config.max_causal_gap)
    if not np.isfinite(data).all():
        raise ValueError("historical missing values lack a sufficiently recent past reading")
    stamps = _stamps(timestamps, len(data), config)
    if stamps is not None:
        future_stamps = pd.date_range(stamps[-1] + pd.Timedelta(minutes=config.dt_minutes),
                                     periods=horizon, freq=f"{config.dt_minutes}min")
        all_stamps = stamps.append(future_stamps)
    else:
        future_stamps, all_stamps = None, None
    output = np.empty((horizon, data.shape[1]))
    fallback, clipped = None, 0
    method = config.method
    if method == "seasonal_naive" and len(data) < config.seasonal_slots:
        method, fallback = "persistence", "insufficient seasonal history"
    if method == "ridge" and len(data) - max(config.ar_lags) < max(12, 2 * len(config.ar_lags)):
        method, fallback = "persistence", "insufficient ridge training rows"
    parameters = []
    extended = np.vstack((data, np.zeros_like(output)))
    if method == "ridge":
        begin = max(config.ar_lags)
        for j in range(data.shape[1]):
            X = np.vstack([_feature(extended, t, j, config, all_stamps) for t in range(begin, len(data))])
            y = data[begin:, j]
            center = np.mean(X, axis=0)
            scale = np.std(X, axis=0)
            center[0], scale[0] = 0., 1.
            scale[scale < 1e-12] = 1.
            Z = (X - center) / scale
            penalty = config.ridge_alpha * np.eye(Z.shape[1]); penalty[0, 0] = 0.
            coefficient = np.linalg.lstsq(Z.T @ Z + penalty, Z.T @ y, rcond=None)[0]
            parameters.append(dict(coefficient=coefficient.tolist(), feature_center=center.tolist(),
                                   feature_scale=scale.tolist(), training_rows=len(y),
                                   training_rmse=float(np.sqrt(np.mean((Z @ coefficient - y) ** 2)))))
            for step in range(horizon):
                t = len(data) + step
                row = (_feature(extended, t, j, config, all_stamps) - center) / scale
                prediction = float(row @ coefficient)
                if nonnegative and prediction < 0:
                    prediction, clipped = 0., clipped + 1
                if not np.isfinite(prediction):
                    raise RuntimeError("causal ridge produced a nonfinite prediction")
                extended[t, j], output[step, j] = prediction, prediction
    else:
        for step in range(horizon):
            row = extended[len(data) + step - config.seasonal_slots] if method == "seasonal_naive" else data[-1]
            output[step] = np.maximum(row, 0.) if nonnegative else row
            clipped += int(np.count_nonzero(output[step] != row))
            extended[len(data) + step] = output[step]
    if not np.isfinite(output).all():
        raise RuntimeError("causal forecast is not finite")
    metadata = dict(config=asdict(config), effective_method=method, fallback=fallback,
        historical_rows=len(data), channels=data.shape[1], causal_filled_values=int(filled.sum()),
        nonnegative_clips=clipped, normalization="ridge feature statistics use historical fitted rows only; output unit unchanged",
        historical_cutoff_utc=stamps[-1].isoformat() if stamps is not None else None,
        future_timestamps_utc=[t.isoformat() for t in future_stamps] if future_stamps is not None else None,
        calendar_features="clock daily and weekly sin/cos" if config.calendar and stamps is not None else "none",
        true_process_periods_used=False, future_values_used=False, parameters=parameters,
        uncertainty="point forecast; prediction intervals not calibrated")
    return dict(values=output[:, 0] if one_dimensional else output, metadata=metadata)


def forecast_for(observed, meter, states, T=12, base_kw=20., *, timestamps=None,
                 source_times=None, config=None):
    """Compatibility interface for FixedResponses, accepting observed history.

    observed P/Q and fitted source P/Q are already in per-unit; base_kw is
    recorded for unit clarity and is not used for data-dependent rescaling.
    source_times may be one shared vector or a mapping by candidate name when
    source fits use a predeclared subset of observed historical timestamps.
    """
    config = config or ForecastConfig()
    if not np.isfinite(base_kw) or base_kw <= 0:
        raise ValueError("fixed positive kW power base required")
    p, _ = _matrix(_get(observed, "p"))
    q, _ = _matrix(_get(observed, "q"))
    if p.shape != q.shape:
        raise ValueError("historical P and Q shapes differ")
    try:
        p = p - np.broadcast_to(np.asarray(_get(meter, "bias_p", 0.), float), p.shape)
        q = q - np.broadcast_to(np.asarray(_get(meter, "bias_q", 0.), float), q.shape)
    except ValueError as exc:
        raise ValueError("calibrated historical P/Q biases must match observations") from exc
    forecasts = {"p": forecast_series(p, T, timestamps=timestamps, config=config, nonnegative=True),
                 "q": forecast_series(q, T, timestamps=timestamps, config=config)}
    root = np.asarray(_get(observed, "root_v", 1.), float)
    if root.ndim == 0:
        last_root = float(root)
    else:
        if root.shape != (len(p),) or not np.isfinite(root).all():
            raise ValueError("known historical root voltage must match history")
        last_root = float(root[-1])
    if not np.isfinite(last_root) or last_root <= 0:
        raise ValueError("positive known historical root voltage required")
    source, source_metadata = {}, {}
    for name, state in states.items():
        sp = np.asarray(_get(state, "source_p"), float)
        sq = np.asarray(_get(state, "source_q"), float)
        if sp.ndim != 1 or sq.shape != sp.shape or not len(sp):
            raise ValueError("each candidate needs historical source P and Q sequences")
        stamps = source_times.get(name) if isinstance(source_times, Mapping) else source_times
        if stamps is None:
            if len(sp) != len(p):
                raise ValueError("compressed source fit requires explicit source_times; do not relabel its clock")
            stamps = timestamps
        source_alignment = dict(original_fit_rows=len(sp), causal_forward_fill=False)
        if stamps is not None:
            fit_stamps = pd.DatetimeIndex(pd.to_datetime(stamps, utc=True))
            if len(fit_stamps) != len(sp) or fit_stamps.hasnans or fit_stamps.has_duplicates or not fit_stamps.is_monotonic_increasing:
                raise ValueError("source_times must match strictly ordered fitted source rows")
            cfg = ForecastConfig(**config) if isinstance(config, Mapping) else config
            regular = pd.date_range(fit_stamps[0], fit_stamps[-1], freq=f"{cfg.dt_minutes}min")
            if not fit_stamps.isin(regular).all():
                raise ValueError("source fit timestamps do not lie on the declared prediction clock")
            if len(regular) != len(fit_stamps):
                # Retain gaps on the physical clock. The forecaster fills only
                # from the last fitted past source, never by interpolation.
                sp = pd.Series(sp, index=fit_stamps).reindex(regular).to_numpy()
                sq = pd.Series(sq, index=fit_stamps).reindex(regular).to_numpy()
                stamps = regular
                source_alignment["causal_forward_fill"] = True
                source_alignment["regular_clock_rows"] = len(regular)
        sp_forecast = forecast_series(sp, T, timestamps=stamps, config=config)
        sq_forecast = forecast_series(sq, T, timestamps=stamps, config=config)
        if timestamps is not None and stamps is not None:
            source_cutoff = pd.Timestamp(pd.to_datetime(stamps, utc=True)[-1])
            observed_cutoff = pd.Timestamp(pd.to_datetime(timestamps, utc=True)[-1])
            if source_cutoff != observed_cutoff:
                raise ValueError("source subset must include the current observed cutoff; gap forecast is not implicit")
        source[name] = dict(source_p=sp_forecast["values"], source_q=sq_forecast["values"])
        source_metadata[name] = dict(p=sp_forecast["metadata"], q=sq_forecast["metadata"], alignment=source_alignment)
    return dict(p=forecasts["p"]["values"], q=forecasts["q"]["values"], root_v=np.full(T, last_root),
        source_by_candidate=source,
        metadata=dict(version="observed_causal_forecast_v1", config=asdict(config) if not isinstance(config, Mapping) else dict(config),
            input_units="per-unit P/Q and fitted source; fixed base_kw only recorded", base_kw=float(base_kw),
            P=forecasts["p"]["metadata"], Q=forecasts["q"]["metadata"], source=source_metadata,
            root_rule="persistence of last calibrated observed root", source_interface=1,
            observed_only=True, future_regime_used=False, truth_used=False,
            process_harmonic_periods_used=False))


def select_on_development(development, *, candidates=DEFAULT_CANDIDATES, horizon=12,
                          origins=(48, 60, 72, 84), phase="development", nonnegative=False):
    """Freeze one configuration using rolling prefixes of development data.

    Each development item has values (time,channel) and optional timestamps.
    Targets stay inside that development record. Error normalization uses the
    corresponding past prefix standard deviation, never a whole-dataset scale.
    """
    if phase != "development" or not development:
        raise ValueError("nonempty development-only records required for forecast selection")
    results = []
    for config in candidates:
        squared_errors, raw_errors, failures, n_origins = [], [], [], 0
        for record_index, record in enumerate(development):
            if record.get("phase", "development") != "development":
                raise ValueError("calibration/evaluation records cannot select forecast hyperparameters")
            values, _ = _matrix(record["values"])
            stamps = record.get("timestamps")
            for origin in origins:
                if origin < 1 or origin + horizon > len(values):
                    continue
                n_origins += 1
                try:
                    predicted = forecast_series(values[:origin], horizon,
                        timestamps=None if stamps is None else stamps[:origin], config=config,
                        nonnegative=nonnegative)["values"]
                    target = values[origin:origin + horizon]
                    if not np.isfinite(target).all():
                        raise ValueError("development target missing")
                    prefix, _ = causal_forward_fill(values[:origin], max_gap=config.max_causal_gap)
                    if not np.isfinite(prefix).all():
                        raise ValueError("development prefix remains missing")
                    scale = np.maximum(np.std(prefix, axis=0), 1e-3)
                    squared_errors.append(float(np.mean(((predicted - target) / scale) ** 2)))
                    raw_errors.append(float(np.mean((predicted - target) ** 2)))
                except Exception as exc:
                    failures.append(dict(record=record_index, origin=int(origin), failure=repr(exc)))
        score = float(np.mean(squared_errors)) if squared_errors and not failures else None
        results.append(dict(config=asdict(config), normalized_MSE=score,
            raw_MSE=float(np.mean(raw_errors)) if raw_errors else None,
            n_rolling_origins=n_origins, available_origins=len(squared_errors), failures=failures))
    eligible = [(row["normalized_MSE"], i) for i, row in enumerate(results) if row["normalized_MSE"] is not None]
    if not eligible:
        raise RuntimeError("no declared forecaster passed development rolling validation")
    _, index = min(eligible)
    selected = candidates[index]
    record = dict(selected_config=asdict(selected), results=results,
        selection_phase="development", origins=list(origins), horizon=horizon,
        target="mean rolling-prefix-standardized squared error", ties="preregistered candidate order",
        nonnegative=bool(nonnegative), evaluation_used=False)
    record["selection_sha256"] = sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return selected, record
