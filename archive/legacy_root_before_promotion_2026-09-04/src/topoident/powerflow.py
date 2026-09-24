"""Balanced AC power-flow utilities used by both reproductions.

This dependency-light implementation is a deterministic experiment simulator,
not a replacement for MATPOWER or OpenDSS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Branch:
    """Series branch in per unit; bus indices are zero based."""

    u: int
    v: int
    r: float
    x: float

    @property
    def y(self) -> complex:
        return 1.0 / complex(self.r, self.x)


def build_ybus(n_bus: int, branches: Iterable[Branch]) -> np.ndarray:
    """Build a series-only nodal admittance matrix."""
    ybus = np.zeros((n_bus, n_bus), dtype=complex)
    for branch in branches:
        y = branch.y
        u, v = branch.u, branch.v
        ybus[u, u] += y
        ybus[v, v] += y
        ybus[u, v] -= y
        ybus[v, u] -= y
    return ybus


def injections(ybus: np.ndarray, voltage: np.ndarray) -> np.ndarray:
    """Return complex nodal injections S = V conj(YV)."""
    return voltage * np.conj(ybus @ voltage)


def solve_power_flow(
    ybus: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    *,
    slack: int = 0,
    v_slack: complex = 1.0 + 0.0j,
    initial: np.ndarray | None = None,
    tolerance: float = 1e-10,
    max_nfev: int = 200,
) -> np.ndarray:
    """Solve a balanced PQ power flow with one fixed slack voltage.

    Voltage magnitudes are parameterized logarithmically, so an iterate can
    never produce a negative magnitude.
    """
    n_bus = len(p)
    pq = np.array([i for i in range(n_bus) if i != slack], dtype=int)
    if initial is None:
        theta0 = np.zeros(len(pq))
        logv0 = np.zeros(len(pq))
    else:
        theta0 = np.angle(initial[pq])
        logv0 = np.log(np.maximum(np.abs(initial[pq]), 0.5))
    x0 = np.r_[theta0, logv0]

    def unpack(x: np.ndarray) -> np.ndarray:
        voltage = np.empty(n_bus, dtype=complex)
        voltage[slack] = v_slack
        k = len(pq)
        voltage[pq] = np.exp(x[k:]) * np.exp(1j * x[:k])
        return voltage

    def mismatch(x: np.ndarray) -> np.ndarray:
        s = injections(ybus, unpack(x))
        return np.r_[s.real[pq] - p[pq], s.imag[pq] - q[pq]]

    result = least_squares(
        mismatch,
        x0,
        xtol=tolerance,
        ftol=tolerance,
        gtol=tolerance,
        max_nfev=max_nfev,
    )
    max_error = float(np.max(np.abs(mismatch(result.x))))
    if not result.success or max_error > 2e-7:
        raise RuntimeError(
            f"power flow did not converge: {result.message}; max mismatch={max_error:.3e}"
        )
    return unpack(result.x)


def is_connected(n_bus: int, edges: Iterable[tuple[int, int]]) -> bool:
    adjacency = [[] for _ in range(n_bus)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for nxt in adjacency[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == n_bus


def path_in_tree(
    n_bus: int, edges: list[tuple[int, int]], source: int, target: int
) -> list[tuple[int, int]]:
    """Return the unique edge path in a tree."""
    adjacency: list[list[int]] = [[] for _ in range(n_bus)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    parent = {source: -1}
    stack = [source]
    while stack:
        node = stack.pop()
        if node == target:
            break
        for nxt in adjacency[node]:
            if nxt not in parent:
                parent[nxt] = node
                stack.append(nxt)
    if target not in parent:
        raise ValueError("source and target are disconnected")
    nodes = [target]
    while nodes[-1] != source:
        nodes.append(parent[nodes[-1]])
    nodes.reverse()
    return [tuple(sorted((a, b))) for a, b in zip(nodes[:-1], nodes[1:])]
