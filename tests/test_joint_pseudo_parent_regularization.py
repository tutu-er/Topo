"""Tests for regularized joint pseudo-parent voltage estimation."""

import numpy as np
import pandas as pd

from terminal_case33.pipeline.pseudo_parent_voltage import (
    fit_joint_pseudo_parent_voltage,
    regularized_joint_fit_options,
)


def test_regularized_joint_preset_is_explicit_and_overrideable() -> None:
    options = regularized_joint_fit_options()

    assert options["gauge_quantile"] == 0.05
    assert options["optimization_gauge_quantile"] == 0.0
    assert options["anchor_strength"] == 0.10
    assert options["prior_weight"] == 1.0
    assert options["smoothness_weight"] == 1.0
    assert options["common_mode_method"] == "huber"


def test_deembedded_prior_reduces_noisy_common_mode_error() -> None:
    terminals = [1, 2, 3]
    count = 120
    rng = np.random.default_rng(83)
    p = rng.uniform(0.002, 0.018, (count, 3))
    q = rng.uniform(0.001, 0.009, (count, 3))
    r_true = np.array(
        [[0.90, 0.35, 0.00], [0.35, 0.85, 0.00], [0.00, 0.00, 0.65]]
    )
    x_true = 0.70 * r_true
    root = 1.02 + 0.001 * np.sin(np.linspace(0.0, 6.0 * np.pi, count))
    common_drop = 0.010 + 0.003 * np.sin(np.linspace(0.0, 3.0 * np.pi, count))
    target = common_drop[:, None] + p @ r_true.T + q @ x_true.T
    target += rng.normal(0.0, 8e-4, target.shape)
    target[::17, 0] += 4e-3
    voltage = np.sqrt(root[:, None] ** 2 - target)
    scenario = {
        "name": "regularized_joint",
        "P_terminal": pd.DataFrame(p, columns=terminals),
        "Q_terminal": pd.DataFrame(q, columns=terminals),
        "V_terminal": pd.DataFrame(voltage, columns=terminals),
        "root_voltage": pd.Series(root),
    }
    initial_r = r_true + 0.20 * np.ones_like(r_true)
    initial_x = x_true + 0.10 * np.ones_like(x_true)
    free = fit_joint_pseudo_parent_voltage(
        [scenario],
        terminals,
        900000,
        initial_r,
        initial_x,
        ridge=0.0,
    )
    prior_vsq = pd.Series(root**2 - common_drop)
    regularized = fit_joint_pseudo_parent_voltage(
        [scenario],
        terminals,
        900000,
        initial_r,
        initial_x,
        ridge=0.0,
        gauge_quantile=0.20,
        common_mode_method="huber",
        prior_pseudo_voltage_squared=[prior_vsq],
        prior_weight=2.0,
        smoothness_weight=1.0,
        anchor_strength=0.10,
    )
    truth = root**2 - common_drop
    free_rmse = np.sqrt(np.mean((free.pseudo_voltage_squared[0] - truth) ** 2))
    regularized_rmse = np.sqrt(
        np.mean((regularized.pseudo_voltage_squared[0] - truth) ** 2)
    )

    assert regularized_rmse < free_rmse
    assert regularized.prior_weight == 2.0
    assert regularized.common_mode_method == "huber"
