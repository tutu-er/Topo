"""Strict observation alignment and physical-day grouping, without topology inputs."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd
from research_experiments.wzzt_dr_selection.end_to_end import wzzt_adapter as observed


def normalize(scenarios, split):
    clean = observed._normalize_observed(scenarios, split)
    frames = [[], [], [], []]
    all_days = []
    for source, scene in zip(scenarios, clean, strict=True):
        index = scene["P_terminal"].index
        if isinstance(index, pd.DatetimeIndex):
            if not index.is_monotonic_increasing:
                raise ValueError(f"{split} timestamps must already be chronological")
            days = [str(v) for v in index.normalize()]
            explicit = source.get("day_ids")
            if explicit is not None:
                if len(explicit) != len(index):
                    raise ValueError("day_ids must match the exact time rows")
                # Explicit IDs may name dates differently but cannot split or combine dates.
                for a in range(len(index)):
                    for b in range(a):
                        if (explicit[a] == explicit[b]) != (days[a] == days[b]):
                            raise ValueError("day_ids disagree with timestamp date boundaries")
        else:
            explicit = source.get("day_ids")
            if explicit is None or len(explicit) != len(index):
                raise ValueError("non-datetime observations require aligned explicit day_ids")
            days = [str(v) for v in explicit]
        for parts, key in zip(frames, ("P_terminal", "Q_terminal", "V_terminal", "V_root_observed"), strict=True):
            parts.append(scene[key])
        all_days.extend(days)
    obs = tuple(pd.concat(parts, axis=0) for parts in frames)
    if isinstance(obs[0].index, pd.DatetimeIndex):
        if not obs[0].index.is_unique or not obs[0].index.is_monotonic_increasing:
            raise ValueError(f"{split} days must be disjoint and chronological")
    unique = list(dict.fromkeys(all_days))
    for day in unique:
        positions = np.flatnonzero(np.asarray(all_days) == day)
        if len(positions) < 2:
            raise ValueError("each physical day requires at least two observed rows")
        if positions[-1] - positions[0] + 1 != len(positions):
            raise ValueError("physical-day rows must be contiguous")
    return obs, all_days, {day: all_days.count(day) for day in unique}


def arrays(obs, preprocess="raw", *, day_ids=None):
    p, q, v, root = obs
    values = [p.to_numpy(dtype=float).copy(), q.to_numpy(dtype=float).copy(),
              root.to_numpy(dtype=float)[:, None]**2-v.to_numpy(dtype=float)**2]
    if preprocess == "demean":
        if day_ids is None:
            if not isinstance(p.index, pd.DatetimeIndex):
                raise ValueError("daily demeaning requires real day IDs")
            day_ids = [str(v) for v in p.index.normalize()]
        if len(day_ids) != len(p):
            raise ValueError("day_ids length mismatch")
        groups = np.asarray(day_ids)
        for day in dict.fromkeys(day_ids):
            rows = np.flatnonzero(groups == day)
            if len(rows) < 2:
                raise ValueError("daily demeaning requires at least two rows per day")
            for value in values:
                value[rows] -= value[rows].mean(axis=0)
    elif preprocess != "raw":
        raise ValueError("unknown preprocessing")
    return values


def matrices(n, supports, r, x):
    result = []
    for weights in (r, x):
        m = np.zeros((n, n))
        for support, weight in zip(supports, weights, strict=True):
            m[np.ix_(support, support)] += weight
        result.append(m)
    return result


def loss(obs, r, x):
    p, q, y = arrays(obs)
    residual = p @ np.asarray(r).T + q @ np.asarray(x).T-y
    return {"mae": float(np.mean(np.abs(residual))), "rmse": float(np.sqrt(np.mean(residual**2)))}


def input_hash(obs, day_ids):
    h = hashlib.sha256()
    for value in obs:
        h.update(pd.util.hash_pandas_object(value, index=True).values.tobytes())
    h.update(json.dumps(list(obs[0].columns), default=str).encode())
    h.update(json.dumps(day_ids).encode())
    return h.hexdigest()
