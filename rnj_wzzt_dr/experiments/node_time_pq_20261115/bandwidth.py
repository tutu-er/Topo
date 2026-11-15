"""Fit and seal a node/hour action band using training P/Q and time only.

Inputs P and Q use the existing common per-unit power base.  The fitted
intercept, residual range and resulting action band use kW/kvar.  This module
does not load a forecast, validation observation, network or true parameter.
The residual range is an empirical operating envelope, not a probability bound.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone as datetime_timezone
import hashlib
import json
from numbers import Integral

import numpy as np
import pandas as pd


def array_hash(value, *, integer=False):
    """Stable content hash including canonical numeric dtype and shape."""
    array = np.ascontiguousarray(value, dtype='<i8' if integer else '<f8')
    digest = hashlib.sha256(json.dumps(
        {'shape': list(array.shape), 'dtype': array.dtype.str},
        sort_keys=True, separators=(',', ':')).encode('ascii'))
    digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _training_clock(timestamps, count, local_timezone):
    values = list(timestamps)
    if len(values) != count or not values:
        raise ValueError('One explicit UTC timestamp is required per training row')
    parsed = []
    for text in values:
        if not isinstance(text, str):
            raise ValueError('Training timestamps must be ISO UTC strings')
        try:
            stamp = datetime.fromisoformat(text.replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError('Invalid training timestamp') from exc
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError('Naive training timestamps are not accepted')
        if stamp.utcoffset().total_seconds() != 0:
            raise ValueError('Training timestamp source must explicitly use UTC')
        parsed.append(stamp.astimezone(datetime_timezone.utc))
    if any(right <= left for left, right in zip(parsed, parsed[1:])):
        raise ValueError('Training timestamps must be strictly increasing and unique')
    try:
        local = pd.DatetimeIndex(parsed).tz_convert(local_timezone)
    except Exception as exc:
        raise ValueError('Invalid or unavailable local clock timezone') from exc
    offsets = {stamp.utcoffset().total_seconds() for stamp in local.to_pydatetime()}
    if len(offsets) != 1:
        raise ValueError('Training window crosses a local UTC-offset/DST change')
    canonical = [stamp.isoformat().replace('+00:00', 'Z') for stamp in parsed]
    return canonical, np.asarray(local.hour, dtype=np.int64), local


def build_bandwidth(P, Q, labels, timestamps, base_kw, *,
                    timezone='Europe/Berlin', start_clock_hour=8, source=None):
    """Return (arrays, metadata), fitting only the explicitly supplied training rows.

    Per node: kappa = sum((P-Pmean)*(Q-Qmean))/sum((P-Pmean)**2),
    a = Qmean-kappa*Pmean, r = Q-a-kappa*P.  For local clock hour h,
    band[h] = max(r at h)-min(r at h).  No multiplier, floor or smoothing.
    At least two training observations at every local hour are required.
    """
    p, q = np.asarray(P, dtype=float), np.asarray(Q, dtype=float)
    if (p.ndim != 2 or p.shape != q.shape or not p.size
            or not np.isfinite(p).all() or not np.isfinite(q).all()):
        raise ValueError('Finite aligned two-dimensional training P and Q required')
    raw_labels = list(labels)
    if (len(raw_labels) != p.shape[1] or not all(
            isinstance(label, Integral) and not isinstance(label, bool) for label in raw_labels)
            or len(set(raw_labels)) != len(raw_labels)):
        raise ValueError('Unique integer terminal labels must match training columns')
    node_labels = np.asarray(raw_labels, dtype=np.int64)
    base = float(base_kw)
    if not np.isfinite(base) or base <= 0:
        raise ValueError('Positive finite common power base required')
    if (not isinstance(start_clock_hour, Integral) or isinstance(start_clock_hour, bool)
            or not 0 <= start_clock_hour < 24):
        raise ValueError('Decision start hour must be an integer in [0,23]')
    utc, clock, local = _training_clock(timestamps, p.shape[0], timezone)
    hours = (int(start_clock_hour)+np.arange(24, dtype=np.int64)) % 24
    hourly_counts = np.asarray([np.count_nonzero(clock == hour) for hour in hours], dtype=np.int64)
    if np.any(hourly_counts < 2):
        missing = hours[hourly_counts < 2].tolist()
        raise ValueError(f'At least two training rows per local hour required; insufficient hours: {missing}')

    p_native, q_native = p*base, q*base
    if not np.isfinite(p_native).all() or not np.isfinite(q_native).all():
        raise ValueError('Power-base conversion overflow')
    pc, qc = p_native-p_native.mean(axis=0), q_native-q_native.mean(axis=0)
    denominator = np.sum(pc*pc, axis=0)
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        bad = node_labels[(denominator <= 0) | ~np.isfinite(denominator)].tolist()
        raise ValueError(f'Constant/invalid training P prevents centered OLS at nodes {bad}')
    kappa = np.sum(pc*qc, axis=0)/denominator
    intercept = q_native.mean(axis=0)-kappa*p_native.mean(axis=0)
    residual = q_native-intercept[None, :]-p_native*kappa[None, :]
    if not np.isfinite(residual).all() or not np.isfinite(kappa).all():
        raise ValueError('Nonfinite OLS fit; no numerical floor or replacement applied')
    band = np.stack([np.ptp(residual[clock == hour], axis=0) for hour in hours])
    counts = np.repeat(hourly_counts[:, None], p.shape[1], axis=1)
    arrays = dict(labels=node_labels.copy(), kappa=kappa.copy(),
        intercept_kvar=intercept.copy(), band_kvar=band, counts=counts,
        decision_hours=hours)
    metadata = dict(schema='node_time_pq_band_v1',
        information_scope='Only supplied training P/Q, terminal labels and training UTC timestamps; no forecast, validation, future outcome or true network',
        interpretation='Empirical same-clock-hour residual change range; not a confidence interval or total response-error guarantee',
        formulas=dict(kappa='sum((p-mean(p))*(q-mean(q)))/sum((p-mean(p))^2)',
            intercept='mean(q)-kappa*mean(p)', residual='q-intercept-kappa*p',
            band='max(training residual at local clock hour)-min(training residual at local clock hour)',
            action_constraint='-band[h,i] <= delta_q[h,i]-kappa[i]*delta_p[h,i] <= band[h,i]'),
        labels=node_labels.tolist(), training_rows=int(p.shape[0]), nodes=int(p.shape[1]),
        input_units='P and Q per unit on the same common apparent-power base',
        common_power_base_kva=base, coefficient_units='kvar/kW',
        intercept_and_band_units='kvar', local_timezone=str(timezone),
        start_clock_hour=int(start_clock_hour), decision_hours=hours.tolist(),
        training_timestamps_utc=utc, training_local_hour=clock.tolist(),
        training_start_local=local[0].isoformat(), training_end_local=local[-1].isoformat(),
        minimum_count_per_hour=2, counts=counts.tolist(),
        kappa=kappa.tolist(), intercept_kvar=intercept.tolist(), band_kvar=band.tolist(),
        centered_p_sum_squares_kw2=denominator.tolist(),
        multiplier=1., bandwidth_floor=None, zero_band_allowed=True,
        missing_policy='Reject nonfinite P/Q, invalid/duplicate timestamps, insufficient local-hour count, constant P or DST-offset change; no imputation',
        input_content_sha256=dict(P_pu=array_hash(p), Q_pu=array_hash(q),
            labels=array_hash(node_labels, integer=True), timestamps_utc=_json_hash(utc)),
        source=deepcopy(source))
    metadata['array_content_sha256'] = {key: array_hash(value,
        integer=key in ('labels', 'counts', 'decision_hours')) for key, value in arrays.items()}
    metadata['fit_receipt_sha256'] = _json_hash(metadata)
    for value in arrays.values():
        value.setflags(write=False)
    return arrays, metadata
