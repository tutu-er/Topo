"""Tests for separable generalized total least squares."""

from __future__ import annotations

import numpy as np
import pandas as pd

from terminal_case33.estimation.gtls import fit_separable_gtls_sensitivity


def test_gtls_reduces_errors_in_variables_attenuation() -> None:
    rng = np.random.default_rng(20260714)
    terminals = [10, 11, 12]
    r_true = np.array(
        [
            [0.80, 0.30, 0.15],
            [0.30, 0.90, 0.15],
            [0.15, 0.15, 0.70],
        ]
    )
    x_true = 0.55 * r_true
    sample_count = 4000
    p_true = rng.normal(0.0, 0.20, size=(sample_count, len(terminals)))
    q_true = rng.normal(0.0, 0.12, size=(sample_count, len(terminals)))
    drop_true = p_true @ r_true.T + q_true @ x_true.T
    pq_noise = 0.20
    voltage_noise = 0.02
    p_measured = p_true + rng.normal(0.0, pq_noise * p_true.std(axis=0), p_true.shape)
    q_measured = q_true + rng.normal(0.0, pq_noise * q_true.std(axis=0), q_true.shape)
    drop_measured = drop_true + rng.normal(
        0.0,
        2.0 * voltage_noise,
        drop_true.shape,
    )
    scenario = {
        "name": "synthetic_eiv",
        "P_terminal": pd.DataFrame(p_measured, columns=terminals),
        "Q_terminal": pd.DataFrame(q_measured, columns=terminals),
        "V_terminal": pd.DataFrame(np.ones_like(drop_true), columns=terminals),
        "drop_target": pd.DataFrame(drop_measured, columns=terminals),
    }

    estimate = fit_separable_gtls_sensitivity(
        [scenario],
        pq_noise_relative_std=pq_noise,
        voltage_noise_relative_std=voltage_noise,
    )
    design = np.hstack([p_measured, q_measured])
    ols, *_ = np.linalg.lstsq(design, drop_measured, rcond=None)
    ols_error = np.linalg.norm(ols[:3].T - r_true) + np.linalg.norm(ols[3:].T - x_true)
    gtls_error = np.linalg.norm(estimate.r_matrix - r_true) + np.linalg.norm(
        estimate.x_matrix - x_true
    )

    assert gtls_error < ols_error
    assert np.allclose(estimate.r_matrix, estimate.r_matrix.T)
    assert np.all(estimate.r_matrix >= 0.0)
