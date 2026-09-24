"""Numerical reproduction of Zhang et al. (IEEE TSG, 2020).

Step 1 follows the paper's angle-free matrix regression. Step 2 uses bounded
Gauss--Newton least squares, numerically equivalent to the paper's specialized
Newton update but more stable than explicitly forming a Moore--Penrose inverse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .powerflow import Branch


@dataclass
class BasicEstimate:
    g_matrix: np.ndarray
    b_matrix: np.ndarray
    edges: list[tuple[int, int]]
    scores: dict[tuple[int, int], float]


@dataclass
class FineEstimate:
    edges: list[tuple[int, int]]
    g: np.ndarray
    b: np.ndarray
    theta: np.ndarray
    objective: float
    nfev: int
    success: bool


def basic_identification(
    p: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    *,
    gamma_top: float = 0.05,
    max_edges: int = 60,
    rcond: float = 1e-12,
) -> BasicEstimate:
    """Paper equations (11)--(21), with a deterministic edge-count safeguard.

    The unregularized regression is ill-conditioned because all voltage
    magnitudes are close to one. The paper thresholds relative contribution;
    if that leaves too many entries, ``max_edges`` retains the strongest
    physically signed entries so Step 2 remains computationally tractable.
    """
    if p.shape != q.shape or p.shape != v.shape:
        raise ValueError("p, q, and v must have the same [sample, bus] shape")
    p_over_v = p / v
    q_over_v = q / v
    # V C = P/V, followed by the symmetry operations in eqs. (15)--(16).
    g_matrix = np.linalg.lstsq(v, p_over_v, rcond=rcond)[0].T
    b_matrix = -np.linalg.lstsq(v, q_over_v, rcond=rcond)[0].T
    g_matrix = 0.5 * (g_matrix + g_matrix.T)
    b_matrix = 0.5 * (b_matrix + b_matrix.T)

    n_bus = v.shape[1]
    candidates: list[tuple[float, float, tuple[int, int]]] = []
    scores: dict[tuple[int, int], float] = {}
    for i in range(n_bus):
        for j in range(i + 1, n_bus):
            gamma_g = abs(g_matrix[i, j]) / max(abs(g_matrix[i, i]), 1e-12)
            gamma_b = abs(b_matrix[i, j]) / max(abs(b_matrix[i, i]), 1e-12)
            gamma = max(gamma_g, gamma_b)
            # Off-diagonal physical signs are G_ij < 0 and B_ij > 0.
            physical_strength = max(-g_matrix[i, j], 0.0) + max(b_matrix[i, j], 0.0)
            scores[(i, j)] = gamma
            if gamma > gamma_top and physical_strength > 0.0:
                candidates.append((physical_strength, gamma, (i, j)))
    candidates.sort(reverse=True)
    edges = [edge for _, _, edge in candidates[:max_edges]]
    return BasicEstimate(g_matrix, b_matrix, edges, scores)


def _ybus_from_parameters(
    n_bus: int, edges: list[tuple[int, int]], g: np.ndarray, h: np.ndarray
) -> np.ndarray:
    """Build Y with branch admittance g-jh (h=-b > 0)."""
    ybus = np.zeros((n_bus, n_bus), dtype=complex)
    for (u, w), ge, he in zip(edges, g, h):
        y = complex(ge, -he)
        ybus[u, u] += y
        ybus[w, w] += y
        ybus[u, w] -= y
        ybus[w, u] -= y
    return ybus


def _initial_angles(
    p: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    edges: list[tuple[int, int]],
    g: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    n_samples, n_bus = p.shape
    ybus = _ybus_from_parameters(n_bus, edges, g, h)
    theta = np.zeros((n_samples, n_bus))
    for t in range(n_samples):

        def residual(angle: np.ndarray) -> np.ndarray:
            full = np.r_[0.0, angle]
            voltage = v[t] * np.exp(1j * full)
            power = voltage * np.conj(ybus @ voltage)
            return np.r_[power.real[1:] - p[t, 1:], power.imag[1:] - q[t, 1:]]

        fit = least_squares(residual, np.zeros(n_bus - 1), max_nfev=30)
        theta[t, 1:] = fit.x
    return theta


def fine_identification(
    p: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    basic: BasicEstimate,
    *,
    samples: int = 20,
    max_nfev: int = 35,
    prior_weight: float = 2e-5,
) -> FineEstimate:
    """Jointly estimate branch parameters and unmeasured voltage angles."""
    p = p[-samples:]
    q = q[-samples:]
    v = v[-samples:]
    n_samples, n_bus = p.shape
    edges = basic.edges
    if not edges:
        raise ValueError("basic identification returned no candidate edges")

    raw_g = np.array([max(-basic.g_matrix[u, w], 0.0) for u, w in edges])
    raw_h = np.array([max(basic.b_matrix[u, w], 0.0) for u, w in edges])
    positive = np.r_[raw_g[raw_g > 0], raw_h[raw_h > 0]]
    floor = max(float(np.median(positive)) * 0.01, 1e-3)
    g0 = np.clip(np.maximum(raw_g, floor), 1e-3, 300.0)
    h0 = np.clip(np.maximum(raw_h, floor), 1e-3, 300.0)
    theta0 = _initial_angles(p, q, v, edges, g0, h0)
    m = len(edges)
    k_angle = n_samples * (n_bus - 1)
    x0 = np.r_[np.log(g0), np.log(h0), theta0[:, 1:].ravel()]

    p_scale = max(float(np.std(p[:, 1:])), 1e-4)
    q_scale = max(float(np.std(q[:, 1:])), 1e-4)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = np.exp(x[:m])
        h = np.exp(x[m : 2 * m])
        theta = np.zeros((n_samples, n_bus))
        theta[:, 1:] = x[2 * m :].reshape(n_samples, n_bus - 1)
        return g, h, theta

    def residual(x: np.ndarray) -> np.ndarray:
        g, h, theta = unpack(x)
        ybus = _ybus_from_parameters(n_bus, edges, g, h)
        voltage = v * np.exp(1j * theta)
        power = voltage * np.conj(voltage @ ybus.T)
        mismatch = np.r_[
            ((power.real[:, 1:] - p[:, 1:]) / p_scale).ravel(),
            ((power.imag[:, 1:] - q[:, 1:]) / q_scale).ravel(),
        ]
        prior = np.sqrt(prior_weight) * (x[: 2 * m] - x0[: 2 * m])
        return np.r_[mismatch, prior]

    rows_data = 2 * n_samples * (n_bus - 1)
    sparsity = lil_matrix((rows_data + 2 * m, 2 * m + k_angle), dtype=int)
    sparsity[:rows_data, : 2 * m] = 1
    block = n_bus - 1
    for t in range(n_samples):
        angle_slice = slice(2 * m + t * block, 2 * m + (t + 1) * block)
        sparsity[t * block : (t + 1) * block, angle_slice] = 1
        q_start = n_samples * block + t * block
        sparsity[q_start : q_start + block, angle_slice] = 1
    sparsity[rows_data:, : 2 * m] = np.eye(2 * m, dtype=int)

    lower = np.r_[np.full(2 * m, np.log(1e-5)), np.full(k_angle, -0.35)]
    upper = np.r_[np.full(2 * m, np.log(500.0)), np.full(k_angle, 0.35)]
    fit = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        jac_sparsity=sparsity.tocsr(),
        x_scale="jac",
        max_nfev=max_nfev,
        verbose=0,
    )
    g, h, theta = unpack(fit.x)
    return FineEstimate(edges, g, -h, theta, float(2.0 * fit.cost), fit.nfev, fit.success)


def evaluate(
    estimate: FineEstimate, true_branches: list[Branch], *, prune_ratio: float = 0.04
) -> dict[str, float | int]:
    true = {tuple(sorted((e.u, e.v))): e for e in true_branches}
    magnitude = np.hypot(estimate.g, estimate.b)
    cutoff = prune_ratio * float(np.median(magnitude))
    retained = {edge: idx for idx, edge in enumerate(estimate.edges) if magnitude[idx] >= cutoff}
    true_edges = set(true)
    pred_edges = set(retained)
    matched = true_edges & pred_edges
    precision = len(matched) / max(len(pred_edges), 1)
    recall = len(matched) / len(true_edges)
    if matched:
        g_errors = []
        b_errors = []
        for edge in matched:
            idx = retained[edge]
            y = true[edge].y
            g_errors.append(abs(estimate.g[idx] - y.real) / abs(y.real))
            b_errors.append(abs(estimate.b[idx] - y.imag) / abs(y.imag))
        g_mape = 100.0 * float(np.mean(g_errors))
        b_mape = 100.0 * float(np.mean(b_errors))
    else:
        g_mape = b_mape = float("nan")
    return {
        "candidate_edges": len(estimate.edges),
        "retained_edges": len(pred_edges),
        "true_edges": len(true_edges),
        "matched_edges": len(matched),
        "precision": precision,
        "recall": recall,
        "g_mape_percent": g_mape,
        "b_mape_percent": b_mape,
        "objective": estimate.objective,
        "nfev": estimate.nfev,
    }
