"""Step-2 refinement with the paper's pseudo-power-flow angle reset."""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from .powerflow import solve_power_flow
from .zhang2020 import BasicEstimate, FineEstimate, _initial_angles, _ybus_from_parameters


def _pseudo_power_flow_angles(
    p: np.ndarray,
    q: np.ndarray,
    edges: list[tuple[int, int]],
    g: np.ndarray,
    h: np.ndarray,
    measured_v: np.ndarray,
) -> np.ndarray:
    n_samples, n_bus = p.shape
    ybus = _ybus_from_parameters(n_bus, edges, g, h)
    theta = np.zeros((n_samples, n_bus))
    previous = None
    try:
        for t in range(n_samples):
            voltage = solve_power_flow(
                ybus, p[t], q[t], initial=previous, tolerance=1e-8, max_nfev=120
            )
            theta[t] = np.angle(voltage)
            previous = voltage
    except RuntimeError:
        return _initial_angles(p, q, measured_v, edges, g, h)
    return theta


def fine_identification_pseudo(
    p: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    basic: BasicEstimate,
    *,
    samples: int = 40,
    iterations: int = 16,
    ridge: float = 1e-9,
) -> FineEstimate:
    """Alternate pseudo PF and nonnegative branch-parameter regression."""
    p = p[-samples:]
    q = q[-samples:]
    v = v[-samples:]
    n_samples, n_bus = p.shape
    edges = basic.edges
    m = len(edges)
    raw_g = np.array([max(-basic.g_matrix[u, w], 0.0) for u, w in edges])
    raw_h = np.array([max(basic.b_matrix[u, w], 0.0) for u, w in edges])
    positive = np.r_[raw_g[raw_g > 0], raw_h[raw_h > 0]]
    floor = max(float(np.median(positive)) * 0.005, 1e-3)
    g = np.clip(np.maximum(raw_g, floor), 1e-3, 300.0)
    h = np.clip(np.maximum(raw_h, floor), 1e-3, 300.0)
    previous_objective = np.inf
    objective = np.inf
    theta = np.zeros((n_samples, n_bus))
    success = False

    for iteration in range(iterations):
        theta = _pseudo_power_flow_angles(p, q, edges, g, h, v)
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
        scale = np.sqrt(ridge)
        design_fit = np.vstack([design, scale * np.eye(2 * m)])
        target_fit = np.r_[target, scale * np.r_[g, h]]
        update = lsq_linear(design_fit, target_fit, bounds=(0.0, 500.0), max_iter=300)
        new_g, new_h = update.x[:m], update.x[m:]
        # Damping is important when a bad extra edge is nearly collinear with a true one.
        g = 0.35 * g + 0.65 * new_g
        h = 0.35 * h + 0.65 * new_h
        objective = float(np.sum((design @ np.r_[g, h] - target) ** 2))
        relative_change = abs(previous_objective - objective) / max(previous_objective, 1e-12)
        if iteration > 2 and relative_change < 1e-5:
            success = True
            break
        previous_objective = objective

    theta = _pseudo_power_flow_angles(p, q, edges, g, h, v)
    return FineEstimate(edges, g, -h, theta, objective, iteration + 1, success)
