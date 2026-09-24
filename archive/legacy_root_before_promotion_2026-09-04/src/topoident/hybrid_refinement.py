"""Optional joint Gauss--Newton polishing from a pseudo-PF warm start."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .zhang2020 import FineEstimate, _ybus_from_parameters


def polish_jointly(
    p: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    warm: FineEstimate,
    *,
    samples: int = 40,
    max_nfev: int = 60,
    prune_ratio: float = 0.04,
) -> FineEstimate:
    p, q, v = p[-samples:], q[-samples:], v[-samples:]
    n_samples, n_bus = p.shape
    magnitude = np.hypot(warm.g, warm.b)
    keep = magnitude >= prune_ratio * np.median(magnitude)
    edges = [edge for edge, active in zip(warm.edges, keep) if active]
    g0 = np.maximum(warm.g[keep], 1e-5)
    h0 = np.maximum(-warm.b[keep], 1e-5)
    theta0 = warm.theta[-samples:]
    m = len(edges)
    block = n_bus - 1
    angle_count = n_samples * block
    x0 = np.r_[np.log(g0), np.log(h0), theta0[:, 1:].ravel()]
    p_scale = max(float(np.std(p[:, 1:])), 1e-4)
    q_scale = max(float(np.std(q[:, 1:])), 1e-4)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        g, h = np.exp(x[:m]), np.exp(x[m : 2 * m])
        theta = np.zeros((n_samples, n_bus))
        theta[:, 1:] = x[2 * m :].reshape(n_samples, block)
        return g, h, theta

    def residual(x: np.ndarray) -> np.ndarray:
        g, h, theta = unpack(x)
        ybus = _ybus_from_parameters(n_bus, edges, g, h)
        voltage = v * np.exp(1j * theta)
        power = voltage * np.conj(voltage @ ybus.T)
        return np.r_[
            ((power.real[:, 1:] - p[:, 1:]) / p_scale).ravel(),
            ((power.imag[:, 1:] - q[:, 1:]) / q_scale).ravel(),
        ]

    rows = 2 * n_samples * block
    sparsity = lil_matrix((rows, 2 * m + angle_count), dtype=int)
    sparsity[:, : 2 * m] = 1
    for t in range(n_samples):
        cols = slice(2 * m + t * block, 2 * m + (t + 1) * block)
        sparsity[t * block : (t + 1) * block, cols] = 1
        qrow = n_samples * block + t * block
        sparsity[qrow : qrow + block, cols] = 1
    lower = np.r_[np.full(2 * m, np.log(1e-6)), np.full(angle_count, -0.35)]
    upper = np.r_[np.full(2 * m, np.log(500.0)), np.full(angle_count, 0.35)]
    fit = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        jac_sparsity=sparsity.tocsr(),
        x_scale="jac",
        max_nfev=max_nfev,
    )
    g, h, theta = unpack(fit.x)
    return FineEstimate(edges, g, -h, theta, float(2 * fit.cost), fit.nfev, fit.success)
