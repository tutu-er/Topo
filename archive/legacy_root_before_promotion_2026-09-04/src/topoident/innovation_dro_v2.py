"""DRO placement using separation between entire topology ambiguity sets."""

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
from .innovation_dro import _pair_distances, _profile_accuracy


def _ambiguity_margin(ensemble: np.ndarray, noise: np.ndarray, alpha: float = 0.10) -> float:
    """Lower-CVaR distance between independently perturbed topology sets."""
    values = []
    for a, b in combinations(range(ensemble.shape[1]), 2):
        for draw_a in range(ensemble.shape[0]):
            for draw_b in range(ensemble.shape[0]):
                delta = (ensemble[draw_a, a] - ensemble[draw_b, b]) / noise
                values.append(float(np.mean(np.sum(delta**2, axis=-1))))
    values = np.sort(values)
    count = max(1, int(np.ceil(alpha * len(values))))
    return float(np.mean(values[:count]))


def run_dro_placement_v2(
    *, seed: int = 506, operations: int = 5, design_draws: int = 10, test_draws: int = 8
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n_bus, configs, library, terminals = benchmark()
    p, q = load_scenarios(n_bus, terminals, operations, rng)
    nominal_vm, nominal_theta = simulate_library(configs, library, p, q, n_bus)
    design_vm, design_theta = [], []
    for _ in range(design_draws):
        vm, theta = simulate_library(
            configs, perturb_library(library, rng, sigma=0.24), p, q, n_bus
        )
        design_vm.append(vm)
        design_theta.append(theta)
    design_vm, design_theta = np.asarray(design_vm), np.asarray(design_theta)

    nominal_best, robust_best = (), ()
    nominal_score = robust_score = -np.inf
    for sensors in combinations(range(1, n_bus), 2):
        noise = np.r_[np.full(len(terminals), 8e-4), [1e-4] * 2, [2e-4] * 2]
        nominal_sig = signature(nominal_vm, nominal_theta, terminals, sensors)
        nscore = float(np.min(_pair_distances(nominal_sig, noise)))
        ensemble = np.stack(
            [signature(v, t, terminals, sensors) for v, t in zip(design_vm, design_theta)]
        )
        rscore = _ambiguity_margin(ensemble, noise)
        if nscore > nominal_score:
            nominal_best, nominal_score = sensors, nscore
        if rscore > robust_score:
            robust_best, robust_score = sensors, rscore

    test_vm, test_theta = [], []
    for _ in range(test_draws):
        vm, theta = simulate_library(
            configs, perturb_library(library, rng, sigma=0.28), p, q, n_bus
        )
        test_vm.append(vm)
        test_theta.append(theta)
    test_vm, test_theta = np.asarray(test_vm), np.asarray(test_theta)
    nominal_accuracy = _profile_accuracy(
        nominal_best, design_vm, design_theta, test_vm, test_theta, terminals, rng
    )
    robust_accuracy = _profile_accuracy(
        robust_best, design_vm, design_theta, test_vm, test_theta, terminals, rng
    )
    return {
        "innovation": "DRO lower-CVaR separation between topology ambiguity sets",
        "candidate_topologies": len(configs),
        "nominal_buses_1_based": [node + 1 for node in nominal_best],
        "robust_buses_1_based": [node + 1 for node in robust_best],
        "nominal_design_score": nominal_score,
        "robust_ambiguity_margin": robust_score,
        "shifted_test_accuracy_nominal_placement": nominal_accuracy,
        "shifted_test_accuracy_robust_placement": robust_accuracy,
        "design_parameter_sigma": 0.24,
        "test_parameter_sigma": 0.28,
    }
