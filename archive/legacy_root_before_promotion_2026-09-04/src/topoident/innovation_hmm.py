"""Innovation 3: graph-constrained HMM for dynamic switch-state tracking."""

from __future__ import annotations

import numpy as np

from .innovation_common import benchmark, edge_edit_distance, load_scenarios, simulate_library


def _switch_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    true_switch = truth[1:] != truth[:-1]
    pred_switch = prediction[1:] != prediction[:-1]
    tp = int(np.sum(true_switch & pred_switch))
    fp = int(np.sum(~true_switch & pred_switch))
    fn = int(np.sum(true_switch & ~pred_switch))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def run_dynamic_hmm(
    *, seed: int = 303, horizon: int = 100, missing_probability: float = 0.35
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n_bus, configs, library, terminals = benchmark()
    p, q = load_scenarios(n_bus, terminals, horizon, rng)
    vm, theta = simulate_library(configs, library, p, q, n_bus)
    sensor = 9
    sig = np.concatenate([vm[..., terminals], theta[..., [sensor]]], axis=-1)
    noise = np.r_[np.full(len(terminals), 1.3e-3), 4e-4]

    adjacency = np.zeros((len(configs), len(configs)), dtype=bool)
    for i in range(len(configs)):
        for j in range(len(configs)):
            adjacency[i, j] = i == j or edge_edit_distance(configs[i], configs[j]) == 2

    truth = np.zeros(horizon, dtype=int)
    truth[0] = int(rng.integers(len(configs)))
    for t in range(1, horizon):
        if rng.random() < 0.10:
            neighbors = np.flatnonzero(adjacency[truth[t - 1]])
            neighbors = neighbors[neighbors != truth[t - 1]]
            truth[t] = int(rng.choice(neighbors)) if len(neighbors) else truth[t - 1]
        else:
            truth[t] = truth[t - 1]

    observed = np.zeros((horizon, sig.shape[-1]))
    mask = rng.random(observed.shape) > missing_probability
    # Keep at least one channel per timestamp.
    for t in range(horizon):
        if not np.any(mask[t]):
            mask[t, int(rng.integers(sig.shape[-1]))] = True
        observed[t] = sig[truth[t], t] + rng.normal(0.0, noise)

    emissions = np.zeros((horizon, len(configs)))
    for t in range(horizon):
        delta = (sig[:, t] - observed[t]) / noise
        emissions[t] = np.sum(delta[:, mask[t]] ** 2, axis=1)
    independent = np.argmin(emissions, axis=1)

    transition_cost = np.full((len(configs), len(configs)), 25.0)
    transition_cost[adjacency] = 4.0
    np.fill_diagonal(transition_cost, 0.0)
    dp = np.zeros_like(emissions)
    back = np.zeros((horizon, len(configs)), dtype=int)
    dp[0] = emissions[0]
    for t in range(1, horizon):
        candidates = dp[t - 1][:, None] + transition_cost
        back[t] = np.argmin(candidates, axis=0)
        dp[t] = emissions[t] + np.min(candidates, axis=0)
    viterbi = np.zeros(horizon, dtype=int)
    viterbi[-1] = int(np.argmin(dp[-1]))
    for t in range(horizon - 1, 0, -1):
        viterbi[t - 1] = back[t, viterbi[t]]

    return {
        "innovation": "switch-graph constrained HMM with missing asynchronous channels",
        "candidate_topologies": len(configs),
        "horizon": horizon,
        "missing_probability": missing_probability,
        "true_switches": int(np.sum(truth[1:] != truth[:-1])),
        "independent_accuracy": float(np.mean(independent == truth)),
        "viterbi_accuracy": float(np.mean(viterbi == truth)),
        "independent_switch_f1": _switch_f1(truth, independent),
        "viterbi_switch_f1": _switch_f1(truth, viterbi),
        "pmu_bus_1_based": sensor + 1,
    }
