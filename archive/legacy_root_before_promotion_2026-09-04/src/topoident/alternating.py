"""Alternating pseudo-power-flow refinement for Zhang et al. Step 2."""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from .zhang2020 import BasicEstimate, FineEstimate, _initial_angles


def fine_identification_alternating(
    p: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    basic: BasicEstimate,
    *,
    samples: int = 40,
    iterations: int = 12,
    ridge: float = 1e-8,
) -> FineEstimate:
    """Alternate angle recovery and constrained linear parameter updates."""
    p = p[-samples:]
    q = q[-samples:]
    v = v[-samples:]
    n_samples, n_bus = p.shape
    edges = basic.edges
    m = len(edges)
    raw_g = np.array([max(-basic.g_matrix[u, w], 0.0) for u, w in edges])
    raw_h = np.array([max(basic.b_matrix[u, w], 0.0) for u, w in edges])
    positive = np.r_[raw_g[raw_g > 0], raw_h[raw_h > 0]]
    floor = max(float(np.median(positive)) * 0.01, 1e-3)
    g = np.clip(np.maximum(raw_g, floor), 1e-3, 300.0)
    h = np.clip(np.maximum(raw_h, floor), 1e-3, 300.0)
    theta = np.zeros((n_samples, n_bus))
    previous_objective = np.inf
    objective = np.inf
    success = False

    for iteration in range(iterations):
        theta = _initial_angles(p, q, v, edges, g, h)
        voltage = v * np.exp(1j * theta)
        rows = n_samples * (n_bus - 1)
        design = np.zeros((2 * rows, 2 * m))

        for edge_idx, (u, w) in enumerate(edges):
            for col, y in ((edge_idx, 1.0 + 0.0j), (m + edge_idx, -1.0j)):
                y_basis = np.zeros((n_bus, n_bus), dtype=complex)
                y_basis[u, u] += y
                y_basis[w, w] += y
                y_basis[u, w] -= y
                y_basis[w, u] -= y
                contribution = voltage * np.conj(voltage @ y_basis.T)
                design[:rows, col] = contribution.real[:, 1:].ravel()
                design[rows:, col] = contribution.imag[:, 1:].ravel()

        target = np.r_[p[:, 1:].ravel(), q[:, 1:].ravel()]
        if ridge > 0:
            scale = np.sqrt(ridge)
            design_fit = np.vstack([design, scale * np.eye(2 * m)])
            target_fit = np.r_[target, scale * np.r_[g, h]]
        else:
            design_fit, target_fit = design, target
        update = lsq_linear(
            design_fit,
            target_fit,
            bounds=(0.0, 500.0),
            lsmr_tol="auto",
            max_iter=250,
        )
        g, h = update.x[:m], update.x[m:]
        objective = float(np.sum((design @ update.x - target) ** 2))
        relative_change = abs(previous_objective - objective) / max(previous_objective, 1e-12)
        if objective < 1e-10 or (iteration > 1 and relative_change < 1e-5):
            success = True
            break
        previous_objective = objective

    theta = _initial_angles(p, q, v, edges, g, h)
    return FineEstimate(edges, g, -h, theta, objective, iteration + 1, success)
