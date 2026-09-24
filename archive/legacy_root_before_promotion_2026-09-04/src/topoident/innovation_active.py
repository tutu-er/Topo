"""Innovation 2: joint active probing and micro-PMU placement."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .innovation_common import benchmark, load_scenarios, simulate_library


def _design_signature(
    vm: np.ndarray,
    theta: np.ndarray,
    terminals: list[int],
    sensor: tuple[int, ...],
    probes: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    parts = []
    noise = []
    for probe in probes:
        parts.append(vm[:, probe, terminals])
        noise.extend([8e-4] * len(terminals))
        if sensor:
            parts.append(theta[:, probe, list(sensor)])
            noise.extend([2e-4] * len(sensor))
    return np.concatenate(parts, axis=1), np.asarray(noise)


def _separation(signatures: np.ndarray, noise: np.ndarray) -> float:
    values = []
    for a, b in combinations(range(len(signatures)), 2):
        values.append(float(np.sum(((signatures[a] - signatures[b]) / noise) ** 2)))
    return min(values)


def _accuracy(
    signatures: np.ndarray,
    noise: np.ndarray,
    rng: np.random.Generator,
    trials: int,
) -> float:
    correct = 0
    total = 0
    for truth in range(len(signatures)):
        for _ in range(trials):
            observed = signatures[truth] + rng.normal(0.0, noise)
            score = np.sum(((signatures - observed) / noise) ** 2, axis=1)
            correct += int(np.argmin(score) == truth)
            total += 1
    return correct / total


def run_active_probe_design(*, seed: int = 202, trials: int = 200) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n_bus, configs, library, terminals = benchmark()
    p0, q0 = load_scenarios(n_bus, terminals, 1, rng)
    actions = [(None, 0.0)]
    for terminal in terminals:
        actions.extend([(terminal, 0.015), (terminal, -0.015)])
    p = np.repeat(p0, len(actions), axis=0)
    q = np.repeat(q0, len(actions), axis=0)
    for action_idx, (node, delta) in enumerate(actions):
        if node is not None:
            p[action_idx, node] += delta
            q[action_idx, node] += 0.15 * delta
            p[action_idx, 0] = -p[action_idx, 1:].sum()
            q[action_idx, 0] = -q[action_idx, 1:].sum()
    vm, theta = simulate_library(configs, library, p, q, n_bus)

    passive_sig, passive_noise = _design_signature(vm, theta, terminals, (), (0,))
    passive_score = _separation(passive_sig, passive_noise)

    best_sensor = ()
    best_sensor_score = -np.inf
    for sensor in range(1, n_bus):
        sig, noise = _design_signature(vm, theta, terminals, (sensor,), (0,))
        score = _separation(sig, noise)
        if score > best_sensor_score:
            best_sensor, best_sensor_score = (sensor,), score

    best_probes = ()
    best_probe_score = -np.inf
    for probes in combinations(range(1, len(actions)), 2):
        sig, noise = _design_signature(vm, theta, terminals, (), probes)
        score = _separation(sig, noise)
        if score > best_probe_score:
            best_probes, best_probe_score = probes, score

    joint_sensor = ()
    joint_probes = ()
    joint_score = -np.inf
    for sensor in range(1, n_bus):
        for probes in combinations(range(1, len(actions)), 2):
            sig, noise = _design_signature(vm, theta, terminals, (sensor,), probes)
            score = _separation(sig, noise)
            if score > joint_score:
                joint_sensor, joint_probes, joint_score = (sensor,), probes, score

    designs = {
        "passive_sm": ((), (0,)),
        "passive_best_pmu": (best_sensor, (0,)),
        "active_sm": ((), best_probes),
        "joint_active_pmu": (joint_sensor, joint_probes),
    }
    accuracy = {}
    for name, (sensor, probes) in designs.items():
        sig, noise = _design_signature(vm, theta, terminals, sensor, probes)
        accuracy[name] = _accuracy(sig, noise, rng, trials)

    def describe(probe: int) -> dict[str, float | int]:
        node, delta = actions[probe]
        return {"bus_1_based": 0 if node is None else node + 1, "delta_p_pu": delta}

    return {
        "innovation": "joint max-min active probing and micro-PMU placement",
        "candidate_topologies": len(configs),
        "best_passive_pmu_bus_1_based": best_sensor[0] + 1,
        "best_active_probes": [describe(i) for i in best_probes],
        "joint_pmu_bus_1_based": joint_sensor[0] + 1,
        "joint_probes": [describe(i) for i in joint_probes],
        "separation": {
            "passive_sm": passive_score,
            "passive_best_pmu": best_sensor_score,
            "active_sm": best_probe_score,
            "joint_active_pmu": joint_score,
        },
        "identification_accuracy": accuracy,
    }
