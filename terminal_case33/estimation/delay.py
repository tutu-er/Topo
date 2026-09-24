"""Simple meter-delay correction helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def integer_delay_correction(reference: pd.DataFrame, target: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    """Align each target column to the reference by maximum absolute correlation."""

    corrected = target.copy()
    for col in target.columns:
        best_lag = 0
        best_score = -np.inf
        ref = reference[col].to_numpy()
        for lag in range(-max_lag, max_lag + 1):
            rolled = np.roll(target[col].to_numpy(), -lag)
            score = abs(float(np.corrcoef(ref, rolled)[0, 1])) if np.std(rolled) > 0 and np.std(ref) > 0 else 0.0
            if score > best_score:
                best_score = score
                best_lag = lag
        corrected[col] = np.roll(target[col].to_numpy(), -best_lag)
    return corrected

