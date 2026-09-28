"""Temporal preprocessing retained for historical research experiments."""

from __future__ import annotations

import pandas as pd

from rnj_wzzt.estimation.multiscenario import align_scenarios


def daily_demean(data: pd.DataFrame | pd.Series, samples_per_day: int | None = None) -> pd.DataFrame | pd.Series:
    """Subtract the mean of each day independently."""

    n = len(data)
    if n == 0:
        return data.copy()
    day = int(samples_per_day or n)
    parts = []
    for start in range(0, n, day):
        segment = data.iloc[start : start + day]
        parts.append(segment - segment.mean(axis=0))
    return pd.concat(parts, axis=0)


def rolling_highpass(data: pd.DataFrame | pd.Series, window: int) -> pd.DataFrame | pd.Series:
    """Subtract a centered rolling mean from a signal."""

    return data - data.rolling(window=window, min_periods=1, center=True).mean()


def apply_preprocessing_recipe(
    frame: pd.DataFrame,
    recipe: dict,
    samples_per_day: int,
) -> pd.DataFrame:
    """Apply an explicit temporal recipe in a research experiment."""

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


def preprocess_scenarios(scenarios: list[dict], recipe: dict) -> list[dict]:
    """Align observations, then apply one temporal recipe to P, Q, and drop."""

    fitted = []
    for scenario in align_scenarios(scenarios):
        index = scenario["P_terminal"].index
        transformed = [
            apply_preprocessing_recipe(scenario[key], recipe, len(index))
            for key in ("P_terminal", "Q_terminal", "drop_target")
        ]
        p, q, drop = transformed
        index = p.index.intersection(q.index).intersection(drop.index)
        fitted.append({
            "name": scenario["name"], "P_terminal": p.loc[index],
            "Q_terminal": q.loc[index], "drop_target": drop.loc[index],
        })
    return fitted


__all__ = [
    "daily_demean", "rolling_highpass", "apply_preprocessing_recipe",
    "preprocess_scenarios",
]
