"""Declared ordinary-load cases and forecasts using observed history only.

The regular regime uses public harmonic periods with unknown coefficients and
small AR innovations. The shift regime has the exact same first 100 physical
and measured rows, followed by an unannounced level/low-frequency change. This
is a synthetic matched-model test plus a prediction stress test, not field data.
No impedance bound, candidate bank, or source interface is changed from v2/v3.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import numpy as np

from . import ev_mobile_cases_v2 as old
from . import ev_original_study as base
from . import likelihood as lk
from .ev_fixed_train_fit import subset_model, subset_observations
from .simulation import model


PERIODS = (96., 32., 24., 12.)  # 15-minute slots; 24 h, 8 h, 6 h, 3 h.
REGIMES = ("regular", "shift")
TRAINING_END = 96
FUTURE_START = 100
HORIZON = 12
TRAJECTORY_COUNT = FUTURE_START + HORIZON
GENERATOR_CONFIG = dict(
    version="ordinary_harmonic_AR_v4", periods_slots=PERIODS,
    dt_hours=.25, p_mean_kw=[2.4, 2.55, 2.3, 2.5],
    p_fractional_amplitudes=[.085, .040, .025],
    p_periods_slots=[96., 24., 12.], p_innovation_stationary_sd_kw=.025,
    q_over_p_mean=.325, q_over_p_amplitudes=[.075, .035],
    q_over_p_periods_slots=[32., 12.], q_over_p_stationary_sd=.004,
    q_over_p_bounds=[.2, .45], source_p_mean_kw=2.,
    source_p_amplitudes_kw=[.18, .055], source_p_periods_slots=[96., 24.],
    source_p_stationary_sd_kw=.015, source_q_mean_kvar=.7,
    source_q_amplitudes_kvar=[.065, .025], source_q_periods_slots=[32., 12.],
    source_q_stationary_sd_kvar=.006, root_v=1., source_interface=1,
    unannounced_shift_start=FUTURE_START,
    shift_p_level_kw=[.20, -.16, .12, -.22], shift_p_amplitude_kw=.18,
    shift_p_period_slots=48., shift_q_over_p_level=[.020, -.020, .025, -.025],
    shift_q_over_p_amplitude=.025, shift_q_over_p_period_slots=36.,
    shift_source_p_level_kw=.12, shift_source_p_amplitude_kw=.15,
    shift_source_q_level_kvar=-.04, shift_source_q_amplitude_kvar=.04,
    synthetic=True, has_PV=False, has_wind=False)
FORECAST_CONFIG = dict(
    version="observed_history_harmonic_OLS_v4", periods_slots=PERIODS,
    fitted_channels="calibrated P, Q/P, and each candidate fitted source P/Q",
    solver="columnwise OLS by SVD least squares", source_interface=1,
    p_bounds_kw=[.05, 12.], q_over_p_bounds=[.15, .55],
    source_bounds_pu=[-.5, .5], fixed_root_v=1.,
    interpretation="point forecast; no calibrated prediction interval",
    public_periods_known=True, true_coefficients_used=False,
    regime_used=False, root_fixed=True)


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return int(value)


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _hash(record):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    value = json.dumps(record, default=convert, sort_keys=True,
                       separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def harmonic_features(times):
    """Public period features; every coefficient is estimated from history."""
    times = np.asarray(times, float)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("times must be a finite vector of slot indices")
    columns = [np.ones(len(times))]
    for period in PERIODS:
        angle = 2. * np.pi * times / period
        columns.extend((np.sin(angle), np.cos(angle)))
    return np.column_stack(columns)


def _ar(rng, count, width, sd, phi):
    values = np.empty((count, width))
    values[0] = sd * rng.standard_normal(width)
    innovation = sd * np.sqrt(1. - phi ** 2) * rng.standard_normal((count - 1, width))
    for t in range(1, count):
        values[t] = phi * values[t - 1] + innovation[t - 1]
    return values


def generate_trajectory(seed, n, *, regime="regular"):
    """Generate 96 training + 4 observed + 12 future rows, with no EV load."""
    seed, n = _integer(seed, "seed"), _integer(n, "n", 1)
    if n not in (3, 4) or regime not in REGIMES:
        raise ValueError("declared n3/n4 and regular/shift required")
    names = ("phases", "p", "ratio", "source_p", "source_q", "shift")
    sequences = np.random.SeedSequence(seed).spawn(len(names))
    seeds = {name: int(sequence.generate_state(1)[0])
             for name, sequence in zip(names, sequences)}
    rng = {name: np.random.default_rng(value) for name, value in seeds.items()}
    cfg, N = GENERATOR_CONFIG, TRAJECTORY_COUNT
    t = np.arange(N, dtype=float)
    p_phase = rng["phases"].uniform(0., 2. * np.pi, (3, n))
    ratio_phase = rng["phases"].uniform(0., 2. * np.pi, (2, n))
    source_phase = rng["phases"].uniform(0., 2. * np.pi, 4)
    means = np.asarray(cfg["p_mean_kw"][:n]) + rng["phases"].uniform(-.03, .03, n)
    modulation = np.zeros((N, n))
    for amplitude, period, phase in zip(cfg["p_fractional_amplitudes"], cfg["p_periods_slots"], p_phase):
        modulation += amplitude * np.sin(2. * np.pi * t[:, None] / period + phase)
    p_raw = means * (1. + modulation) + _ar(rng["p"], N, n, cfg["p_innovation_stationary_sd_kw"], .65)
    ratio_raw = np.full((N, n), cfg["q_over_p_mean"])
    for amplitude, period, phase in zip(cfg["q_over_p_amplitudes"], cfg["q_over_p_periods_slots"], ratio_phase):
        ratio_raw += amplitude * np.sin(2. * np.pi * t[:, None] / period + phase)
    ratio_raw += _ar(rng["ratio"], N, n, cfg["q_over_p_stationary_sd"], .65)
    source_p_raw = np.full(N, cfg["source_p_mean_kw"])
    source_q_raw = np.full(N, cfg["source_q_mean_kvar"])
    for amplitude, period, phase in zip(cfg["source_p_amplitudes_kw"], cfg["source_p_periods_slots"], source_phase[:2]):
        source_p_raw += amplitude * np.sin(2. * np.pi * t / period + phase)
    for amplitude, period, phase in zip(cfg["source_q_amplitudes_kvar"], cfg["source_q_periods_slots"], source_phase[2:]):
        source_q_raw += amplitude * np.sin(2. * np.pi * t / period + phase)
    source_p_raw += _ar(rng["source_p"], N, 1, cfg["source_p_stationary_sd_kw"], .7)[:, 0]
    source_q_raw += _ar(rng["source_q"], N, 1, cfg["source_q_stationary_sd_kvar"], .6)[:, 0]
    # Draw these phases in both regimes. The regime changes no historical RNG.
    shift_phase = rng["shift"].uniform(0., 2. * np.pi, (2, n))
    shift_source_phase = rng["shift"].uniform(0., 2. * np.pi, 2)
    if regime == "shift":
        h = np.arange(HORIZON, dtype=float)
        rows = slice(FUTURE_START, None)
        p_raw[rows] += np.asarray(cfg["shift_p_level_kw"][:n]) + cfg["shift_p_amplitude_kw"] * np.sin(
            2. * np.pi * h[:, None] / cfg["shift_p_period_slots"] + shift_phase[0])
        ratio_raw[rows] += np.asarray(cfg["shift_q_over_p_level"][:n]) + cfg["shift_q_over_p_amplitude"] * np.sin(
            2. * np.pi * h[:, None] / cfg["shift_q_over_p_period_slots"] + shift_phase[1])
        source_p_raw[rows] += cfg["shift_source_p_level_kw"] + cfg["shift_source_p_amplitude_kw"] * np.sin(
            2. * np.pi * h / cfg["shift_p_period_slots"] + shift_source_phase[0])
        source_q_raw[rows] += cfg["shift_source_q_level_kvar"] + cfg["shift_source_q_amplitude_kvar"] * np.sin(
            2. * np.pi * h / 32. + shift_source_phase[1])
    p = np.maximum(p_raw, .05)
    ratio = np.clip(ratio_raw, *cfg["q_over_p_bounds"])
    source_p, source_q = np.maximum(source_p_raw, 0.), np.maximum(source_q_raw, 0.)
    return dict(p_kw=p, q_kvar=p * ratio, q_over_p=ratio,
        source_p_kw=source_p, source_q_kvar=source_q, root_v=np.ones(N),
        metadata=dict(config=deepcopy(cfg), seed=seed, child_seeds=seeds,
            n=n, count=N, regime=regime, EV_included=False, source_interface=1,
            clips=dict(p=int(np.count_nonzero(p != p_raw)),
                q_over_p=int(np.count_nonzero(ratio != ratio_raw)),
                source_p=int(np.count_nonzero(source_p != source_p_raw)),
                source_q=int(np.count_nonzero(source_q != source_q_raw))),
            generation_only=True, field_data=False))


def make_case(bank, seed, family_index, regime="regular", H=96):
    """Truth/meters only. Selectors receive separate observed-history objects."""
    seed, family_index = _integer(seed, "seed"), _integer(family_index, "family_index")
    if bank not in ("n3", "n4") or family_index not in range(4) or H != 96 or regime not in REGIMES:
        raise ValueError("declared bank n3/n4, family 0..3, H96, regular/shift required")
    case_id = "remote8" if bank == "n3" else "twinblock12"
    cfg = old.case_config(case_id, training_points=H)
    cfg.update(version="mobile_v4_harmonic_gap_audit", horizon=HORIZON,
        forecast="observed_history_harmonic_OLS_v4", forecast_gap=0,
        public_voltage_repeat_counts=[30. / HORIZON] * HORIZON,
        future_metering_integration_seconds_per_slot=3000. / HORIZON,
        regime=regime, observed_before_EV=FUTURE_START,
        forecast_information="all supplied observed pre-EV history; no future rows",
        synthetic_load_model="public harmonic periods, unknown fitted coefficients, small AR innovations")
    candidate, r, x, seeds, assets = old._assets(seed, case_id, family_index)
    n = candidate.n
    bg = generate_trajectory(seeds["background"], n, regime=regime)
    p, q = bg["p_kw"] / base.BASE_KW, bg["q_kvar"] / base.BASE_KW
    sp, sq = bg["source_p_kw"] / base.BASE_KW, bg["source_q_kvar"] / base.BASE_KW
    root, meter = bg["root_v"], model(n)
    noise_seeds = np.random.SeedSequence(seeds["noise"]).spawn(len(base.CHANNELS))
    white = {key: np.random.default_rng(s).standard_normal(
        (TRAJECTORY_COUNT, n) if key in ("p", "q", "v") else TRAJECTORY_COUNT)
        for key, s in zip(base.CHANNELS, noise_seeds)}
    physical = lk.ac_predict(candidate, r, x, p, q, sp, sq, root)
    full = base.observation(physical, meter, white)
    train_rows, observed_rows = slice(0, TRAINING_END), slice(TRAINING_END, FUTURE_START)
    precision = 30. / HORIZON
    fm = replace(meter, v_sigma=np.broadcast_to(meter.v_sigma / np.sqrt(precision), (HORIZON, n)),
        **{key + "_sigma": getattr(meter, key + "_sigma") / np.sqrt(precision)
           for key in ("p", "q", "master_p", "master_q")})
    provenance = dict(data_kind="declared_synthetic_LV_harmonic_and_prediction_shift",
        asset=deepcopy(base.ASSET), assets=assets, seeds=seeds, background=bg["metadata"],
        source_interface=1, H=H, T=HORIZON,
        source_interface_meaning="one common physical interface upstream of terminal subtrees",
        candidate_scope="four predeclared local structures, no truth filtering",
        total_effective_precision=30., precision_uniform_in_time=True,
        metering=dict(total_effective_precision_weight_per_channel=30.,
            per_slot_effective_weight=precision, integration_seconds_per_slot=3000. / HORIZON,
            integration_seconds_total=3000., weights_are_integer_sample_counts=False),
        field_PQV_available=False, true_labels_used_only_for_generation_and_evaluation=True,
        regular_scope="known feature periods with unknown historical coefficients; synthetic matched forecasting structure",
        shift_scope="same first 100 rows; unannounced future level and low-frequency changes")
    return dict(case_id=case_id, seed=seed, family_index=family_index, regime=regime,
        config=cfg, train=subset_observations(full, train_rows),
        validation=subset_observations(full, observed_rows),
        tm=subset_model(meter, full, train_rows), vm=subset_model(meter, full, observed_rows), fm=fm,
        p=p, q=q, source_p=sp, source_q=sq, root_v=root, white=white,
        future_start=FUTURE_START, base_kw=base.BASE_KW,
        truth=dict(candidate=candidate, r=r, x=x), bank=old.candidate_bank(case_id),
        provenance=provenance, truth_used_only_for_generation_and_evaluation=True)


def _array(value, shape, name):
    try:
        result = np.broadcast_to(np.asarray(value, float), shape).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(name + " must broadcast to " + str(shape)) from exc
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite")
    return result


def _fit_predict(values, horizon):
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    count = len(values)
    features = harmonic_features(np.arange(count, dtype=float))
    future_features = harmonic_features(np.arange(count, count + horizon, dtype=float))
    coefficient, _, rank, singular = np.linalg.lstsq(features, values, rcond=None)
    if rank != features.shape[1]:
        raise ValueError("historical harmonic feature matrix is rank deficient")
    residual = values - features @ coefficient
    prediction = future_features @ coefficient
    return prediction, dict(coefficients=coefficient, rank=int(rank), singular_values=singular,
        condition_number=float(singular[0] / singular[-1]),
        historical_residual_rms=np.sqrt(np.mean(residual ** 2, axis=0)),
        estimation_rows=count, no_generator_coefficients=True)


def _clip(values, bounds):
    result = np.clip(values, *bounds)
    return result, dict(count=int(np.count_nonzero(values != result)), bounds=np.asarray(bounds).tolist(),
                        raw_min=float(np.min(values)), raw_max=float(np.max(values)))


def forecast_for(observedD, Dmeter, fits, T=12, base_kw=20.):
    """Estimate public-feature coefficients from observed P/Q and fitted sources.

    This function has no case, regime, generating coefficient, true topology,
    true line, or future-noise argument. Sources are forecast separately for
    every already fitted candidate, at the same fixed physical interface 1.
    R/X arrays in those fits are not read or modified.
    """
    T = _integer(T, "T", 1)
    if not np.isscalar(base_kw) or not np.isfinite(base_kw) or base_kw <= 0.:
        raise ValueError("base_kw must be finite and positive")
    p_read = np.asarray(_get(observedD, "p"), float)
    if p_read.ndim != 2 or p_read.shape[0] < 2 * len(PERIODS) + 2 or p_read.shape[1] < 1:
        raise ValueError("observed P/Q require at least 10 times and one terminal")
    shape, count = p_read.shape, len(p_read)
    p_read, q_read = _array(p_read, shape, "observed p"), _array(_get(observedD, "q"), shape, "observed q")
    bp = _array(_get(Dmeter, "bias_p", 0.), shape, "bias_p")
    bq = _array(_get(Dmeter, "bias_q", 0.), shape, "bias_q")
    p, q = p_read - bp, q_read - bq
    # Forecasting clips are wider than generator ratios; record every correction.
    positive_p = np.maximum(p, FORECAST_CONFIG["p_bounds_kw"][0] / base_kw)
    raw_ratio = q / positive_p
    ratio, ratio_training_clip = _clip(raw_ratio, FORECAST_CONFIG["q_over_p_bounds"])
    pred_p_raw, p_fit = _fit_predict(p, T)
    pred_ratio_raw, ratio_fit = _fit_predict(ratio, T)
    pred_p, p_clip = _clip(pred_p_raw, np.asarray(FORECAST_CONFIG["p_bounds_kw"]) / base_kw)
    pred_ratio, ratio_clip = _clip(pred_ratio_raw, FORECAST_CONFIG["q_over_p_bounds"])
    pred_q = pred_p * pred_ratio
    if not isinstance(fits, Mapping) or not fits:
        raise ValueError("fits must map candidate names to successful historical fits")
    sources, source_fits, source_clips, source_inputs = {}, {}, {}, {}
    for name, fit in fits.items():
        if not isinstance(name, str) or not name or not bool(_get(fit, "success", True)):
            raise ValueError("each candidate must have a successful named historical fit")
        sources[name], source_fits[name], source_clips[name], source_inputs[name] = {}, {}, {}, {}
        for key in ("source_p", "source_q"):
            values = np.asarray(_get(fit, key), float)
            if values.shape != (count,) or not np.isfinite(values).all():
                raise ValueError(name + " " + key + " must match observed history length")
            prediction, fitted = _fit_predict(values, T)
            prediction, clipped = _clip(prediction[:, 0], FORECAST_CONFIG["source_bounds_pu"])
            sources[name][key], source_fits[name][key] = prediction, fitted
            source_clips[name][key], source_inputs[name][key] = clipped, values
    root = np.ones(T)
    metadata = dict(config=deepcopy(FORECAST_CONFIG), observed_history_only=True,
        training_only=True, uses_validation=count > TRAINING_END,
        uses_truth=False, uses_future_noise=False, uses_regime=False,
        observed_count=count, n=shape[1], horizon=T, gap=0,
        forecast_steps=list(range(1, T + 1)), base_kw=float(base_kw),
        source_interface=1, root_fixed=True, line_parameters_read_or_changed=False,
        information_scope="only supplied observed meters P/Q and candidate fitted historical source sequences",
        observed_PQ_sha256=_hash(dict(p=p_read, q=q_read, bias_p=bp, bias_q=bq)),
        calibrated_PQ_sha256=_hash(dict(p=p, q=q)), fitted_source_sha256=_hash(source_inputs),
        coefficients=dict(p=p_fit, q_over_p=ratio_fit, source_by_candidate=source_fits),
        clips=dict(p=p_clip, q_over_p=ratio_clip, historical_q_over_p=ratio_training_clip,
                   source_by_candidate=source_clips),
        no_calibrated_prediction_intervals=True, generator_config_used_by_forecast=False,
        units=dict(p="pu", q="pu", source_p="pu", source_q="pu", root="pu"))
    return dict(p=pred_p, q=pred_q, root=root, root_v=root.copy(),
                source_by_candidate=sources, metadata=metadata)


observe_future = old.observe_future
