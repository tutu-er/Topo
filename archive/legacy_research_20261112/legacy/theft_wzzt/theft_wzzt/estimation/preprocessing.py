"""Preprocessing helpers for terminal smart-meter data."""

from __future__ import annotations

import pandas as pd


RECIPE = {"name": "daily_demean", "kind": "demean"}


def squared_voltage_drop_from_observed_root(
    V_terminal: pd.DataFrame,
    root_voltage: pd.Series,
) -> pd.DataFrame:
    """Return ``V_root(t)^2 - V_i(t)^2`` for each terminal meter.

    This is the preferred target when the transformer/root bus voltage magnitude
    is directly observed, because root voltage fluctuations are removed before
    sensitivity fitting.
    """

    common_index = V_terminal.index.intersection(root_voltage.index)
    V_sq = V_terminal.loc[common_index].pow(2)
    root_sq = root_voltage.loc[common_index].pow(2)
    return pd.DataFrame(
        root_sq.to_numpy()[:, None] - V_sq.to_numpy(),
        index=common_index,
        columns=V_sq.columns,
    )


def daily_demean(data: pd.DataFrame | pd.Series, samples_per_day: int | None = None) -> pd.DataFrame | pd.Series:
    """Subtract the mean of each day independently."""

    is_series = isinstance(data, pd.Series)
    frame = data.to_frame() if is_series else data.copy()
    n = len(frame)
    if n == 0:
        return data.copy()
    day = int(samples_per_day or n)
    parts = []
    for start in range(0, n, day):
        segment = frame.iloc[start : start + day]
        parts.append(segment - segment.mean(axis=0))
    result = pd.concat(parts, axis=0)
    return result.iloc[:, 0] if is_series else result


def rolling_highpass(data: pd.DataFrame | pd.Series, window: int) -> pd.DataFrame | pd.Series:
    """Subtract a centered rolling mean from a signal."""

    is_series = isinstance(data, pd.Series)
    frame = data.to_frame() if is_series else data.copy()
    smooth = frame.rolling(window=window, min_periods=1, center=True).mean()
    result = frame - smooth
    return result.iloc[:, 0] if is_series else result


def apply_preprocessing_recipe(
    frame: pd.DataFrame,
    recipe: dict,
    samples_per_day: int,
) -> pd.DataFrame:
    """Apply one temporal recipe, shared by P, Q, and voltage-drop observations."""

    kind = str(recipe["kind"])
    if kind == "raw":
        return frame.copy()
    if kind == "demean":
        return daily_demean(frame, samples_per_day=samples_per_day)
    if kind == "rolling_highpass":
        return rolling_highpass(frame, window=int(recipe["window"]))
    if kind == "difference":
        return frame.diff().dropna()
    if kind == "chain":
        result = frame.copy()
        for step in recipe["steps"]:
            result = apply_preprocessing_recipe(result, step, samples_per_day)
        return result
    raise ValueError(f"unknown preprocessing kind: {kind}")
