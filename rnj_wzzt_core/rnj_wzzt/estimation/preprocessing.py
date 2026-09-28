"""Construct observed-root squared-voltage-drop targets."""

from __future__ import annotations

import pandas as pd


def squared_voltage_drop_from_observed_root(
    V_terminal: pd.DataFrame,
    root_voltage: pd.Series,
) -> pd.DataFrame:
    """Return ``V_root(t)^2 - V_i(t)^2`` for each terminal meter.

    Subtracting the simultaneously observed root level removes its direct
    additive contribution. Root-meter error remains in the target.
    """

    common_index = V_terminal.index.intersection(root_voltage.index)
    V_sq = V_terminal.loc[common_index].pow(2)
    root_sq = root_voltage.loc[common_index].pow(2)
    return pd.DataFrame(
        root_sq.to_numpy()[:, None] - V_sq.to_numpy(),
        index=common_index,
        columns=V_sq.columns,
    )
