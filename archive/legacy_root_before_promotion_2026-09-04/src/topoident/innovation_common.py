"""Shared simulation helpers for the five innovation prototypes."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .powerflow import Branch, build_ybus, solve_power_flow
from .terminal_pmu import candidate_configurations


def benchmark():
    return candidate_configurations()


def load_scenarios(
    n_bus: int,
    terminals: list[int],
    samples: int,
    rng: np.random.Generator,
    *,
    hidden_nodes: Iterable[int] = (),
    hidden_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate known terminal loads and optional unknown hidden loads."""
    p = np.zeros((samples, n_bus))
    q = np.zeros_like(p)
    nominal = np.array([0.055, 0.070, 0.060, 0.075])
    for t in range(samples):
        scale = rng.lognormal(mean=-0.5 * 0.25**2, sigma=0.25, size=len(terminals))
        p[t, terminals] = -nominal * scale
        q[t, terminals] = 0.42 * p[t, terminals]
        for node in hidden_nodes:
            draw = hidden_scale * rng.lognormal(mean=-0.5 * 0.35**2, sigma=0.35)
            p[t, node] = -draw
            q[t, node] = -0.36 * draw
        p[t, 0] = -p[t, 1:].sum()
        q[t, 0] = -q[t, 1:].sum()
    return p, q


def perturb_library(
    library: dict[tuple[int, int], Branch],
    rng: np.random.Generator,
    *,
    sigma: float = 0.15,
) -> dict[tuple[int, int], Branch]:
    perturbed = {}
    for edge, branch in library.items():
        r_scale = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma)
        x_scale = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma)
        perturbed[edge] = Branch(branch.u, branch.v, branch.r * r_scale, branch.x * x_scale)
    return perturbed


def simulate_library(
    configurations: list[list[tuple[int, int]]],
    library: dict[tuple[int, int], Branch],
    p: np.ndarray,
    q: np.ndarray,
    n_bus: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return voltage-magnitude and angle arrays [topology, sample, bus]."""
    vm = np.zeros((len(configurations), len(p), n_bus))
    theta = np.zeros_like(vm)
    for config_idx, edges in enumerate(configurations):
        ybus = build_ybus(n_bus, [library[edge] for edge in edges])
        previous = None
        for t in range(len(p)):
            voltage = solve_power_flow(ybus, p[t], q[t], initial=previous)
            previous = voltage
            vm[config_idx, t] = np.abs(voltage)
            theta[config_idx, t] = np.angle(voltage)
    return vm, theta


def signature(
    vm: np.ndarray,
    theta: np.ndarray,
    terminals: list[int],
    sensors: tuple[int, ...],
) -> np.ndarray:
    """Concatenate terminal SM magnitudes and micro-PMU phasor channels."""
    parts = [vm[..., terminals]]
    if sensors:
        idx = list(sensors)
        parts.extend([vm[..., idx], theta[..., idx]])
    return np.concatenate(parts, axis=-1)


def nearest_signature_accuracy(
    sig: np.ndarray,
    rng: np.random.Generator,
    *,
    noise: np.ndarray | float,
    trials: int = 10,
) -> float:
    """Classify each topology/sample using normalized nearest signatures."""
    noise_array = np.asarray(noise, dtype=float)
    correct = 0
    total = 0
    for truth in range(sig.shape[0]):
        for t in range(sig.shape[1]):
            for _ in range(trials):
                observed = sig[truth, t] + rng.normal(0.0, noise_array, sig.shape[-1])
                scores = np.sum(((sig[:, t] - observed) / noise_array) ** 2, axis=1)
                correct += int(int(np.argmin(scores)) == truth)
                total += 1
    return correct / total


def edge_edit_distance(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    return len(set(a).symmetric_difference(set(b)))


def maximum_spanning_tree(
    n_bus: int, edges: list[tuple[int, int]], weights: np.ndarray
) -> list[tuple[int, int]]:
    """Kruskal maximum spanning tree without an external graph package."""
    parent = list(range(n_bus))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    result = []
    for idx in np.argsort(weights)[::-1]:
        u, v = edges[int(idx)]
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            result.append((u, v))
            if len(result) == n_bus - 1:
                break
    return result
