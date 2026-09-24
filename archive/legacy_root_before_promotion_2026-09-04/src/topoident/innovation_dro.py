"""Innovation 5: distributionally robust CVaR micro-PMU placement."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .innovation_common import (
    benchmark,
    load_scenarios,
    perturb_library,
    signature,
    simulate_library,
)


def _pair_distances(sig: np.ndarray, noise: np.ndarray) -> np.ndarray:
    values = []
    for a, b in combinations(range(sig.shape[0]), 2):
        values.append(float(np.mean(np.sum(((sig[a] - sig[b]) / noise) ** 2, axis=-1))))
    return np.asarray(values)


def _lower_cvar(values: np.ndarray, fraction: float = 0.15) -> float:
    count = max(1, int(np.ceil(fraction * len(values))))
    return float(np.mean(np.sort(values)[:count]))


def _profile_accuracy(
    sensors: tuple[int, ...],
    design_vm: np.ndarray,
    design_theta: np.ndarray,
    test_vm: np.ndarray,
    test_theta: np.ndarray,
    terminals: list[int],
    rng: np.random.Generator,
) -> float:
    design_sig = np.stack(
        [signature(v, t, terminals, sensors) for v, t in zip(design_vm, design_theta)]
    )
    test_sig = np.stack([signature(v, t, terminals, sensors) for v, t in zip(test_vm, test_theta)])
    noise = np.r_[
        np.full(len(terminals), 8e-4),
        np.full(len(sensors), 1e-4),
        np.full(len(sensors), 2e-4),
    ]
    correct = 0
    total = 0
    for draw in range(test_sig.shape[0]):
        for truth in range(test_sig.shape[1]):
            for t in range(test_sig.shape[2]):
                observed = test_sig[draw, truth, t] + rng.normal(0.0, noise)
                scores = np.zeros(test_sig.shape[1])
                for topology in range(test_sig.shape[1]):
                    delta = (design_sig[:, topology, t] - observed) / noise
                    # Profile likelihood over the ambiguity set.
                    scores[topology] = float(np.min(np.sum(delta**2, axis=1)))
                correct += int(np.argmin(scores) == truth)
                total += 1
    return correct / total


def run_dro_placement(
    *, seed: int = 505, operations: int = 5, design_draws: int = 8, test_draws: int = 6
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n_bus, configs, library, terminals = benchmark()
    p, q = load_scenarios(n_bus, terminals, operations, rng)
    nominal_vm, nominal_theta = simulate_library(configs, library, p, q, n_bus)

    design_vm = []
    design_theta = []
    for _ in range(design_draws):
        perturbed = perturb_library(library, rng, sigma=0.16)
        vm, theta = simulate_library(configs, perturbed, p, q, n_bus)
        design_vm.append(vm)
        design_theta.append(theta)
    design_vm = np.asarray(design_vm)
    design_theta = np.asarray(design_theta)

    nominal_best = ()
    nominal_score = -np.inf
    robust_best = ()
    robust_score = -np.inf
    for sensors in combinations(range(1, n_bus), 2):
        nominal_sig = signature(nominal_vm, nominal_theta, terminals, sensors)
        noise = np.r_[
            np.full(len(terminals), 8e-4),
            np.full(len(sensors), 1e-4),
            np.full(len(sensors), 2e-4),
        ]
        nominal_value = float(np.min(_pair_distances(nominal_sig, noise)))
        uncertain_values = []
        for draw in range(design_draws):
            sig = signature(design_vm[draw], design_theta[draw], terminals, sensors)
            uncertain_values.extend(_pair_distances(sig, noise))
        robust_value = _lower_cvar(np.asarray(uncertain_values), fraction=0.12)
        if nominal_value > nominal_score:
            nominal_best, nominal_score = sensors, nominal_value
        if robust_value > robust_score:
            robust_best, robust_score = sensors, robust_value

    test_vm = []
    test_theta = []
    for _ in range(test_draws):
        shifted = perturb_library(library, rng, sigma=0.28)
        vm, theta = simulate_library(configs, shifted, p, q, n_bus)
        test_vm.append(vm)
        test_theta.append(theta)
    test_vm = np.asarray(test_vm)
    test_theta = np.asarray(test_theta)
    nominal_accuracy = _profile_accuracy(
        nominal_best, design_vm, design_theta, test_vm, test_theta, terminals, rng
    )
    robust_accuracy = _profile_accuracy(
        robust_best, design_vm, design_theta, test_vm, test_theta, terminals, rng
    )

    return {
        "innovation": "DRO-CVaR placement over line-parameter ambiguity sets",
        "candidate_topologies": len(configs),
        "nominal_buses_1_based": [node + 1 for node in nominal_best],
        "robust_buses_1_based": [node + 1 for node in robust_best],
        "nominal_design_score": nominal_score,
        "robust_lower_cvar_score": robust_score,
        "shifted_test_accuracy_nominal_placement": nominal_accuracy,
        "shifted_test_accuracy_robust_placement": robust_accuracy,
        "design_parameter_sigma": 0.16,
        "test_parameter_sigma": 0.28,
    }
