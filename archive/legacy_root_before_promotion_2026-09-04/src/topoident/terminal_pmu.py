"""Terminal-smart-meter topology detection and robust micro-PMU placement."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .powerflow import Branch, build_ybus, path_in_tree, solve_power_flow


@dataclass
class PlacementResult:
    terminal_nodes: list[int]
    configurations: list[list[tuple[int, int]]]
    optimal_by_budget: dict[int, tuple[int, ...]]
    separation_by_budget: dict[int, float]
    accuracy_by_scheme: dict[str, float]


def terminal_test_network() -> tuple[
    int, list[tuple[int, int]], dict[tuple[int, int], Branch], list[int]
]:
    """Return a 12-bus radial network with four metered end users."""
    n_bus = 12
    base = [
        (0, 1),
        (1, 2),
        (1, 3),
        (2, 4),
        (2, 5),
        (3, 6),
        (3, 7),
        (4, 8),
        (5, 9),
        (6, 10),
        (7, 11),
    ]
    ties = [(2, 3), (4, 5), (6, 7), (4, 6), (5, 7)]
    library: dict[tuple[int, int], Branch] = {}
    for idx, edge in enumerate(base + ties):
        u, v = tuple(sorted(edge))
        r = 0.010 + 0.0025 * (idx % 5)
        x = 0.014 + 0.0020 * ((2 * idx + 1) % 5)
        library[(u, v)] = Branch(u, v, r, x)
    return n_bus, [tuple(sorted(e)) for e in base], library, [8, 9, 10, 11]


def candidate_configurations() -> tuple[
    int, list[list[tuple[int, int]]], dict[tuple[int, int], Branch], list[int]
]:
    """Create feasible radial switch states by one branch exchange."""
    n_bus, base, library, terminals = terminal_test_network()
    base_set = set(base)
    ties = [edge for edge in library if edge not in base_set]
    configurations = {tuple(sorted(base))}
    for tie in ties:
        cycle_path = path_in_tree(n_bus, base, tie[0], tie[1])
        for opened in cycle_path:
            # Keep every physical end user terminal under all switch states.
            if opened[0] in terminals or opened[1] in terminals:
                continue
            changed = (base_set | {tie}) - {opened}
            configurations.add(tuple(sorted(changed)))
    return n_bus, [list(c) for c in sorted(configurations)], library, terminals


def _load_scenarios(
    n_bus: int, terminals: list[int], samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    p = np.zeros((samples, n_bus))
    q = np.zeros((samples, n_bus))
    nominal = np.array([0.055, 0.070, 0.060, 0.075])
    for t in range(samples):
        scale = rng.lognormal(mean=-0.5 * 0.25**2, sigma=0.25, size=len(terminals))
        p[t, terminals] = -nominal * scale
        q[t, terminals] = 0.42 * p[t, terminals]
        p[t, 0] = -p[t].sum()
        q[t, 0] = -q[t].sum()
    return p, q


def _signature_library(
    configurations: list[list[tuple[int, int]]],
    branch_library: dict[tuple[int, int], Branch],
    p: np.ndarray,
    q: np.ndarray,
    n_bus: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_config = len(configurations)
    samples = len(p)
    vm = np.zeros((n_config, samples, n_bus))
    theta = np.zeros_like(vm)
    for config_idx, edges in enumerate(configurations):
        branches = [branch_library[e] for e in edges]
        ybus = build_ybus(n_bus, branches)
        previous = None
        for t in range(samples):
            voltage = solve_power_flow(ybus, p[t], q[t], initial=previous)
            previous = voltage
            vm[config_idx, t] = np.abs(voltage)
            theta[config_idx, t] = np.angle(voltage)
    return vm, theta


def topology_separation(
    sensors: tuple[int, ...],
    vm: np.ndarray,
    theta: np.ndarray,
    terminals: list[int],
    *,
    sigma_sm_v: float = 8e-4,
    sigma_pmu_v: float = 1e-4,
    sigma_pmu_theta: float = 2e-4,
) -> float:
    """Worst topology-pair average squared Mahalanobis separation."""
    values = []
    for a, b in combinations(range(vm.shape[0]), 2):
        distance = np.sum(((vm[a, :, terminals] - vm[b, :, terminals]) / sigma_sm_v) ** 2)
        if sensors:
            idx = list(sensors)
            distance += np.sum(((vm[a, :, idx] - vm[b, :, idx]) / sigma_pmu_v) ** 2)
            distance += np.sum(((theta[a, :, idx] - theta[b, :, idx]) / sigma_pmu_theta) ** 2)
        values.append(distance / vm.shape[1])
    return float(min(values))


def optimize_placement(
    vm: np.ndarray,
    theta: np.ndarray,
    terminals: list[int],
    *,
    max_budget: int = 3,
) -> tuple[dict[int, tuple[int, ...]], dict[int, float]]:
    """Exact small-network max-min placement optimization."""
    candidate_buses = range(1, vm.shape[2])
    placements: dict[int, tuple[int, ...]] = {}
    scores: dict[int, float] = {}
    for budget in range(max_budget + 1):
        best_set: tuple[int, ...] = ()
        best_score = -np.inf
        for sensors in combinations(candidate_buses, budget):
            score = topology_separation(sensors, vm, theta, terminals)
            if score > best_score:
                best_score, best_set = score, sensors
        placements[budget] = best_set
        scores[budget] = best_score
    return placements, scores


def identification_accuracy(
    sensors: tuple[int, ...],
    vm: np.ndarray,
    theta: np.ndarray,
    terminals: list[int],
    rng: np.random.Generator,
    *,
    trials: int = 20,
    sigma_sm_v: float = 8e-4,
    sigma_pmu_v: float = 1e-4,
    sigma_pmu_theta: float = 2e-4,
) -> float:
    """Finite-hypothesis weighted least-squares topology identification."""
    correct = 0
    total = 0
    sensor_idx = list(sensors)
    for truth in range(vm.shape[0]):
        for sample in range(vm.shape[1]):
            for _ in range(trials):
                observed_sm = vm[truth, sample, terminals] + rng.normal(
                    0.0, sigma_sm_v, len(terminals)
                )
                scores = np.sum(
                    ((vm[:, sample, terminals] - observed_sm) / sigma_sm_v) ** 2,
                    axis=1,
                )
                if sensors:
                    observed_v = vm[truth, sample, sensor_idx] + rng.normal(
                        0.0, sigma_pmu_v, len(sensors)
                    )
                    observed_t = theta[truth, sample, sensor_idx] + rng.normal(
                        0.0, sigma_pmu_theta, len(sensors)
                    )
                    scores += np.sum(
                        ((vm[:, sample, sensor_idx] - observed_v) / sigma_pmu_v) ** 2,
                        axis=1,
                    )
                    scores += np.sum(
                        ((theta[:, sample, sensor_idx] - observed_t) / sigma_pmu_theta) ** 2,
                        axis=1,
                    )
                correct += int(np.argmin(scores) == truth)
                total += 1
    return correct / total


def run_terminal_pmu_experiment(
    *, seed: int = 17, design_samples: int = 18, trials: int = 8
) -> PlacementResult:
    rng = np.random.default_rng(seed)
    n_bus, configurations, library, terminals = candidate_configurations()
    p, q = _load_scenarios(n_bus, terminals, design_samples, rng)
    vm, theta = _signature_library(configurations, library, p, q, n_bus)
    placements, separation = optimize_placement(vm, theta, terminals, max_budget=3)

    degree = np.zeros(n_bus, dtype=int)
    for u, v in configurations[0]:
        degree[u] += 1
        degree[v] += 1
    degree_based = tuple(np.argsort(degree[1:])[-2:] + 1)
    accuracy = {
        "smart_meter_only": identification_accuracy((), vm, theta, terminals, rng, trials=trials),
        "degree_based_2_pmu": identification_accuracy(
            degree_based, vm, theta, terminals, rng, trials=trials
        ),
        "optimized_2_pmu": identification_accuracy(
            placements[2], vm, theta, terminals, rng, trials=trials
        ),
    }
    return PlacementResult(terminals, configurations, placements, separation, accuracy)
