"""Meter-error propagation for squared-voltage sensitivity models."""

from __future__ import annotations

import numpy as np


def relative_noise_scale(value: np.ndarray, relative_std: float) -> np.ndarray:
    """Return sample-dependent relative noise scales with a stable floor."""

    if relative_std < 0.0:
        raise ValueError("relative_std must be nonnegative")
    magnitude = np.abs(np.asarray(value, dtype=float))
    positive = magnitude[magnitude > 0.0]
    floor = 0.01 * float(np.median(positive)) if positive.size else 1e-12
    return relative_std * np.maximum(magnitude, max(floor, 1e-12))


def propagated_squared_voltage_variance(
    p: np.ndarray,
    q: np.ndarray,
    voltage: np.ndarray,
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    pq_noise_relative_std: float,
    voltage_noise_relative_std: float,
) -> np.ndarray:
    """Propagate independent P/Q/V meter errors into squared-voltage drop."""

    sigma_p = relative_noise_scale(p, pq_noise_relative_std)
    sigma_q = relative_noise_scale(q, pq_noise_relative_std)
    sigma_voltage = relative_noise_scale(voltage, voltage_noise_relative_std)
    sigma_drop = 2.0 * np.maximum(np.abs(voltage), 1e-12) * sigma_voltage
    return (
        sigma_drop**2
        + sigma_p**2 @ (np.asarray(r_matrix, dtype=float) ** 2).T
        + sigma_q**2 @ (np.asarray(x_matrix, dtype=float) ** 2).T
    )
