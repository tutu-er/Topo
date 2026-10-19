"""Declared ordinary-load dynamics and training-only point forecasts.

The generator is a synthetic stationary Gaussian AR(1) construction, followed
by the explicitly reported positivity/PF clips. It is not a measured profile.
Its physical outputs never enter ``fit_forecast``. Forecast P and source powers
are in per unit; the generator uses kW/kvar. A gap of four means that the first
returned forecast is five steps after the final training observation.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json

import numpy as np


CONFIG = dict(
    version="ordinary_stationary_ar1_pf_v1", classification="declared_synthetic_loads",
    mean_p_kw=[2.4, 2.8, 2.1], common_sd_kw=.12, common_phi=.9,
    individual_sd_kw=.25, individual_phi=.8,
    pf_mean=.97, pf_sd=.004, pf_phi=.85, pf_bounds=[.94, .99],
    source_p_mean_kw=2., source_p_sd_kw=.15, source_p_phi=.9,
    source_q_mean_kvar=.7, source_q_sd_kvar=.06, source_q_phi=.85,
    positive_floor_kw=.05, root_v=1., source_interface=1,
    stationary_initialization=True, independent_innovation_streams=True,
    has_PV=False, has_wind=False, field_data=False)
FORECAST_CONFIG = dict(
    version="training_only_P_PF_OLS_AR1_v1", phi_bounds=[-.95, .98],
    p_bounds_kw=[.05, 12.], pf_bounds=[.94, .99], source_bounds_pu=[-.5, .5],
    source_interface=1, fixed_root_v=1.,
    interpretation="point predictions; no calibrated predictive interval",
    forecasting_model="per-household AR1 approximation to common plus individual dynamics",
    estimation="OLS with intercept; input measurement/fit error is not corrected")


def _integer(value, name, *, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _array(value, shape, name):
    try:
        result = np.broadcast_to(np.asarray(value, float), shape).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must broadcast to {shape}") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _hash(record):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    content = json.dumps(record, default=convert, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _stationary_ar1(rng, count, mean, sd, phi, width=1):
    mean = np.broadcast_to(np.asarray(mean, float), (width,))
    values = np.empty((count, width))
    values[0] = mean + sd * rng.standard_normal(width)
    innovations = sd * np.sqrt(1. - phi ** 2) * rng.standard_normal((count - 1, width))
    for t in range(1, count):
        values[t] = mean + phi * (values[t - 1] - mean) + innovations[t - 1]
    return values


def generate_background(seed, count, n=3):
    """Generate one continuous train/validation/future physical trajectory.

    Means repeat the declared three-household pattern when n exceeds three.
    Independent named streams make every shorter trajectory an exact prefix.
    The source is a single time-varying load at physical interface 1, not an
    additional independent source at every household.
    """
    seed = _integer(seed, "seed", minimum=0)
    count, n = _integer(count, "count"), _integer(n, "n")
    names = ("common", "individual", "pf", "source_p", "source_q")
    sequences = np.random.SeedSequence(seed).spawn(len(names))
    child_seeds = {k: int(s.generate_state(1)[0]) for k, s in zip(names, sequences)}
    rng = {k: np.random.default_rng(v) for k, v in child_seeds.items()}
    means = np.resize(np.asarray(CONFIG["mean_p_kw"]), n)
    common = _stationary_ar1(rng["common"], count, 0., CONFIG["common_sd_kw"], CONFIG["common_phi"])
    individual = _stationary_ar1(rng["individual"], count, 0., CONFIG["individual_sd_kw"], CONFIG["individual_phi"], n)
    p_raw = means + common + individual
    p = np.maximum(p_raw, CONFIG["positive_floor_kw"])
    pf_raw = _stationary_ar1(rng["pf"], count, CONFIG["pf_mean"], CONFIG["pf_sd"], CONFIG["pf_phi"], n)
    pf = np.clip(pf_raw, *CONFIG["pf_bounds"])
    q = p * np.tan(np.arccos(pf))
    source_p_raw = _stationary_ar1(rng["source_p"], count, CONFIG["source_p_mean_kw"],
        CONFIG["source_p_sd_kw"], CONFIG["source_p_phi"])[:, 0]
    source_q_raw = _stationary_ar1(rng["source_q"], count, CONFIG["source_q_mean_kvar"],
        CONFIG["source_q_sd_kvar"], CONFIG["source_q_phi"])[:, 0]
    source_p, source_q = np.maximum(source_p_raw, 0.), np.maximum(source_q_raw, 0.)
    clips = dict(p=int(np.count_nonzero(p != p_raw)), pf=int(np.count_nonzero(pf != pf_raw)),
                 source_p=int(np.count_nonzero(source_p != source_p_raw)),
                 source_q=int(np.count_nonzero(source_q != source_q_raw)))
    return dict(p_kw=p, q_kvar=q, pf=pf, source_p_kw=source_p, source_q_kvar=source_q,
        root_v=np.ones(count), metadata=dict(config=deepcopy(CONFIG), seed=seed,
            count=count, n=n, child_seeds=child_seeds, clips=clips,
            source_interface=1, generation_only=True,
            note="AR(1) is stationary before the recorded physical clips; no EV in background"))


def _fit_ar1(values):
    """Columnwise OLS; refit the intercept after clipping the AR coefficient."""
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    left, right = values[:-1], values[1:]
    xmean, ymean = left.mean(axis=0), right.mean(axis=0)
    dx, dy = left - xmean, right - ymean
    denominator = (dx * dx).sum(axis=0)
    threshold = np.finfo(float).eps * np.maximum(1., (left * left).sum(axis=0))
    raw = np.divide((dx * dy).sum(axis=0), denominator, out=np.zeros_like(xmean),
                    where=denominator > threshold)
    phi = np.clip(raw, *FORECAST_CONFIG["phi_bounds"])
    intercept = ymean - phi * xmean
    residual = right - intercept - left * phi
    return dict(phi_raw=raw.tolist(), phi=phi.tolist(),
        phi_clipped=(phi != raw).tolist(), intercept=intercept.tolist(),
        mean=values.mean(axis=0).tolist(), last=values[-1].tolist(),
        residual_sd=np.sqrt((residual * residual).sum(axis=0) / max(len(left) - 2, 1)).tolist(),
        constant_columns=(denominator <= threshold).tolist(), pairs=len(left))


def _predict(values, estimate, kind, horizon, gap):
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    if kind == "mean":
        return np.broadcast_to(values.mean(axis=0), (horizon, values.shape[1])).copy()
    state = values[-1].copy()
    phi, intercept = np.asarray(estimate["phi"]), np.asarray(estimate["intercept"])
    steps = []
    for step in range(1, gap + horizon + 1):
        state = intercept + phi * state
        if step > gap:
            steps.append(state.copy())
    return np.asarray(steps)


def _clip(values, bounds):
    clipped = np.clip(values, *bounds)
    return clipped, dict(count=int(np.count_nonzero(clipped != values)), bounds=list(bounds),
                         raw_min=float(np.min(values)), raw_max=float(np.max(values)))


def fit_forecast(training, train_meter, train_fits, horizon=6, gap=4,
                 base_kw=20., kind="ar1"):
    """Predict from calibrated training meters and candidate training sources.

    Only P/Q, their known offset corrections, and fitted training source arrays
    affect the predictions. AR1 uses P and implied household PF; Q is rebuilt
    from them. The mean comparator uses the calibrated P/Q means, with any
    physical-range correction reported. Missing validation rows are forecast
    through, never filled with held-out measurements. Roots are fixed at 1 pu.
    Candidate source predictions retain the original [-.5,.5] pu fit domain,
    including negative source values. Positive source draws in the generator
    impose no additional restriction on a competing candidate's explanation.
    """
    horizon, gap = _integer(horizon, "horizon"), _integer(gap, "gap", minimum=0)
    if kind not in ("ar1", "mean"):
        raise ValueError("kind must be 'ar1' or 'mean'")
    if not np.isscalar(base_kw) or not np.isfinite(base_kw) or base_kw <= 0:
        raise ValueError("base_kw must be finite and positive")
    observed_p = np.asarray(_get(training, "p"), float)
    if observed_p.ndim != 2 or observed_p.shape[0] < 3 or not observed_p.shape[1]:
        raise ValueError("training P/Q must have shape (at least 3 times, terminals)")
    shape = observed_p.shape
    observed_p = _array(observed_p, shape, "training p")
    observed_q = _array(_get(training, "q"), shape, "training q")
    bp = _array(_get(train_meter, "bias_p", 0.), shape, "bias_p")
    bq = _array(_get(train_meter, "bias_q", 0.), shape, "bias_q")
    p, q = observed_p - bp, observed_q - bq
    p_pf, q_pf = np.maximum(p, 0.), np.maximum(q, 0.)
    ratio = np.divide(p_pf, np.hypot(p_pf, q_pf), out=np.full(shape, .97), where=p_pf > 0.)
    pf, pf_training_clip = _clip(ratio, FORECAST_CONFIG["pf_bounds"])
    p_fit, pf_fit = _fit_ar1(p), _fit_ar1(pf)
    pred_p_raw = _predict(p, p_fit, kind, horizon, gap)
    pred_p, p_clip = _clip(pred_p_raw, np.asarray(FORECAST_CONFIG["p_bounds_kw"]) / base_kw)
    if kind == "ar1":
        pred_pf, pf_clip = _clip(_predict(pf, pf_fit, kind, horizon, gap), FORECAST_CONFIG["pf_bounds"])
        pred_q = pred_p * np.tan(np.arccos(pred_pf))
        q_clip = dict(count=0, note="Q reconstructed from positive P and clipped PF")
    else:
        # Preserve the old PQ-mean comparator whenever its implied PF is legal.
        mean_q = np.broadcast_to(q.mean(axis=0), pred_p.shape).copy()
        nonnegative_q = np.maximum(mean_q, 0.)
        implied_pf = pred_p / np.hypot(pred_p, nonnegative_q)
        pred_pf, pf_clip = _clip(implied_pf, FORECAST_CONFIG["pf_bounds"])
        rebuilt_q = pred_p * np.tan(np.arccos(pred_pf))
        pred_q = np.where((implied_pf == pred_pf) & (mean_q >= 0.), mean_q, rebuilt_q)
        q_clip = dict(count=int(np.count_nonzero(pred_q != mean_q)),
                      note="PQ means retained unless physical P/PF bounds require correction")
    if not isinstance(train_fits, Mapping):
        raise ValueError("train_fits must map candidate names to training fits")
    sources, source_fits, source_clips, source_inputs = {}, {}, {}, {}
    for name, fit in train_fits.items():
        if not isinstance(name, str) or not name:
            raise ValueError("candidate names must be nonempty strings")
        if not bool(_get(fit, "success", True)):
            raise ValueError(f"candidate {name} has no successful training fit")
        inputs, estimates, clips, predictions = {}, {}, {}, {}
        for channel in ("source_p", "source_q"):
            values = np.asarray(_get(fit, channel), float)
            if values.shape != (shape[0],) or not np.isfinite(values).all():
                raise ValueError(f"{name} {channel} must have one finite value per training time")
            estimate = _fit_ar1(values)
            predicted = _predict(values, estimate, kind, horizon, gap)[:, 0]
            predicted, clip = _clip(predicted, FORECAST_CONFIG["source_bounds_pu"])
            inputs[channel], estimates[channel], clips[channel] = values, estimate, clip
            predictions[channel] = predicted
        sources[name], source_fits[name], source_clips[name], source_inputs[name] = predictions, estimates, clips, inputs
    raw_training = {}
    for channel in ("p", "q", "v", "root_v", "master_p", "master_q", "source_p", "source_q"):
        value = _get(training, channel)
        if value is not None:
            array = np.asarray(value, float)
            if not np.isfinite(array).all():
                raise ValueError(f"training {channel} must be finite for provenance")
            raw_training[channel] = array
    metadata = dict(config=deepcopy(FORECAST_CONFIG), kind=kind, training_count=shape[0],
        n=shape[1], horizon=horizon, gap=gap, forecast_steps=list(range(gap + 1, gap + horizon + 1)),
        base_kw=float(base_kw), training_only=True, source_interface=1,
        uses_validation=False, uses_truth=False, uses_future_noise=False,
        training_sha256=_hash(dict(observations=raw_training, bias_p=bp, bias_q=bq)),
        calibrated_training_PQ_sha256=_hash(dict(p=p, q=q)),
        training_source_fits_sha256=_hash(source_inputs),
        ar1=dict(p=p_fit, pf=pf_fit, source_by_candidate=source_fits),
        clips=dict(p=p_clip, pf=pf_clip, q=q_clip, source_by_candidate=source_clips,
            training_implied_pf=pf_training_clip,
            training_negative_p=int(np.count_nonzero(p < 0.)),
            training_negative_q=int(np.count_nonzero(q < 0.))),
        generator_config_used_by_forecast=False,
        output_units=dict(p="pu", q="pu", source_p="pu", source_q="pu", root="pu"))
    root = np.ones(horizon)
    return dict(p=pred_p, q=pred_q, root=root, root_v=root.copy(),
                source_by_candidate=sources, metadata=metadata)
