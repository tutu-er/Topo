"""Tests for physical pseudo data and latent layer-common voltage modes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.run_layered_voltage_mode_comparison import fit_latent_common_mode_sensitivity


def test_latent_common_mode_recovers_exact_synthetic_model() -> None:
    rng = np.random.default_rng(7)
    sample_count = 120
    buses = [10, 11, 12]
    p = pd.DataFrame(rng.normal(0.0, 0.2, (sample_count, 3)), columns=buses)
    q = pd.DataFrame(rng.normal(0.0, 0.1, (sample_count, 3)), columns=buses)
    r_true = np.asarray(
        [
            [0.80, 0.30, 0.20],
            [0.30, 0.90, 0.25],
            [0.20, 0.25, 1.00],
        ]
    )
    x_true = 0.4 * r_true
    common = 0.05 * np.sin(np.arange(sample_count) / 9.0)
    target = p.to_numpy() @ r_true.T + q.to_numpy() @ x_true.T + common[:, None]
    scenario = {
        "name": "synthetic",
        "P_terminal": p,
        "Q_terminal": q,
        "drop_target": pd.DataFrame(target, columns=buses),
    }

    fit = fit_latent_common_mode_sensitivity(
        [scenario],
        buses,
        [pd.Series(common)],
        prior_weight=5.0,
        smoothness_weight=0.1,
        iterations=12,
        alpha=1e-8,
    )

    expected_mode = common - common.mean()
    assert fit.r2_score > 0.99999
    assert np.sqrt(np.mean((fit.common_modes[0].to_numpy() - expected_mode) ** 2)) < 2e-5
    assert np.allclose(fit.r_matrix, r_true, atol=2e-5)
    assert np.allclose(fit.x_matrix, x_true, atol=3e-5)
