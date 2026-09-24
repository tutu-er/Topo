"""Innovation 1: Bayesian topology inference with hidden-load marginalization."""

from __future__ import annotations

import numpy as np

from .innovation_common import benchmark, load_scenarios, simulate_library


def _selected_signature(
    vm: np.ndarray,
    theta: np.ndarray,
    terminals: list[int],
    sensors: tuple[int, ...],
) -> np.ndarray:
    # Terminal SM voltage magnitudes plus micro-PMU phase angles.
    return np.concatenate([vm[..., terminals], theta[..., list(sensors)]], axis=-1)


def run_bayesian_hidden_loads(
    *, seed: int = 101, operations: int = 6, prior_draws: int = 12, test_draws: int = 8
) -> dict[str, object]:
    """Compare zero-injection WLS against marginalized Gaussian likelihood."""
    rng = np.random.default_rng(seed)
    n_bus, configs, library, terminals = benchmark()
    sensors = (8, 10)
    hidden_nodes = (4, 6)
    p_terminal, q_terminal = load_scenarios(n_bus, terminals, operations, rng)
    vm0, th0 = simulate_library(configs, library, p_terminal, q_terminal, n_bus)
    baseline = _selected_signature(vm0, th0, terminals, sensors)
    dim = baseline.shape[-1]
    noise_std = np.r_[np.full(len(terminals), 8e-4), np.full(len(sensors), 2e-4)]

    training = np.zeros((prior_draws, len(configs), operations, dim))
    for draw in range(prior_draws):
        p, q = p_terminal.copy(), q_terminal.copy()
        for t in range(operations):
            for node in hidden_nodes:
                value = 0.018 * rng.lognormal(mean=-0.5 * 0.35**2, sigma=0.35)
                p[t, node] = -value
                q[t, node] = -0.36 * value
            p[t, 0] = -p[t, 1:].sum()
            q[t, 0] = -q[t, 1:].sum()
        vm, theta = simulate_library(configs, library, p, q, n_bus)
        training[draw] = _selected_signature(vm, theta, terminals, sensors)

    means = np.mean(training, axis=0)
    covariances = np.zeros((len(configs), operations, dim, dim))
    noise_cov = np.diag(noise_std**2)
    for k in range(len(configs)):
        for t in range(operations):
            centered = training[:, k, t] - means[k, t]
            covariances[k, t] = centered.T @ centered / max(prior_draws - 1, 1) + noise_cov

    wls_correct = 0
    bayes_correct = 0
    total = 0
    for _ in range(test_draws):
        p, q = p_terminal.copy(), q_terminal.copy()
        for t in range(operations):
            for node in hidden_nodes:
                value = 0.018 * rng.lognormal(mean=-0.5 * 0.35**2, sigma=0.35)
                p[t, node] = -value
                q[t, node] = -0.36 * value
            p[t, 0] = -p[t, 1:].sum()
            q[t, 0] = -q[t, 1:].sum()
        vm, theta = simulate_library(configs, library, p, q, n_bus)
        actual = _selected_signature(vm, theta, terminals, sensors)
        for truth in range(len(configs)):
            for t in range(operations):
                observed = actual[truth, t] + rng.normal(0.0, noise_std)
                wls_scores = np.sum(((baseline[:, t] - observed) / noise_std) ** 2, axis=1)
                bayes_scores = np.zeros(len(configs))
                for k in range(len(configs)):
                    delta = observed - means[k, t]
                    sign, logdet = np.linalg.slogdet(covariances[k, t])
                    bayes_scores[k] = delta @ np.linalg.solve(covariances[k, t], delta) + logdet
                    if sign <= 0:
                        bayes_scores[k] = np.inf
                wls_correct += int(np.argmin(wls_scores) == truth)
                bayes_correct += int(np.argmin(bayes_scores) == truth)
                total += 1

    return {
        "innovation": "hidden-load marginalized Bayesian topology likelihood",
        "candidate_topologies": len(configs),
        "hidden_nodes_1_based": [node + 1 for node in hidden_nodes],
        "pmu_buses_1_based": [node + 1 for node in sensors],
        "zero_injection_wls_accuracy": wls_correct / total,
        "marginalized_bayes_accuracy": bayes_correct / total,
        "prior_draws": prior_draws,
        "test_cases": total,
    }
