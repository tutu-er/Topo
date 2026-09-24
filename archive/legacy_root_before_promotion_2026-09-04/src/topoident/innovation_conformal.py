"""Innovation 5: conformal topology sets with adaptive micro-PMU querying."""

from __future__ import annotations

import numpy as np

from .innovation_common import benchmark, load_scenarios, perturb_library, simulate_library


def run_conformal_adaptive_pmu(
    *,
    seed: int = 606,
    operations: int = 6,
    calibration_draws: int = 20,
    test_draws: int = 12,
    alpha: float = 0.05,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n_bus, configs, library, terminals = benchmark()
    p, q = load_scenarios(n_bus, terminals, operations, rng)
    nominal_vm, nominal_theta = simulate_library(configs, library, p, q, n_bus)
    sm_sigma = 1.0e-3
    pmu_sigma = 2.0e-4

    calibration_scores = [[] for _ in configs]
    angle_errors = np.zeros((calibration_draws, len(configs), operations, n_bus))
    for draw in range(calibration_draws):
        vm, theta = simulate_library(
            configs, perturb_library(library, rng, sigma=0.18), p, q, n_bus
        )
        angle_errors[draw] = theta - nominal_theta
        for truth in range(len(configs)):
            for t in range(operations):
                observed = vm[truth, t, terminals] + rng.normal(0.0, sm_sigma, len(terminals))
                score = np.sum(((observed - nominal_vm[truth, t, terminals]) / sm_sigma) ** 2)
                calibration_scores[truth].append(float(score))

    thresholds = np.array(
        [
            np.quantile(
                scores, min(1.0, (1.0 - alpha) * (len(scores) + 1) / len(scores)), method="higher"
            )
            for scores in calibration_scores
        ]
    )
    angle_scale = np.std(angle_errors, axis=0, ddof=1) + pmu_sigma

    sm_top1_correct = 0
    conformal_covered = 0
    adaptive_correct = 0
    set_sizes = []
    query_count = 0
    total = 0
    selected_buses = []
    for _ in range(test_draws):
        vm, theta = simulate_library(
            configs, perturb_library(library, rng, sigma=0.18), p, q, n_bus
        )
        for truth in range(len(configs)):
            for t in range(operations):
                observed_sm = vm[truth, t, terminals] + rng.normal(0.0, sm_sigma, len(terminals))
                sm_scores = np.sum(
                    ((nominal_vm[:, t, terminals] - observed_sm) / sm_sigma) ** 2,
                    axis=1,
                )
                prediction_set = np.flatnonzero(sm_scores <= thresholds)
                if len(prediction_set) == 0:
                    prediction_set = np.array([int(np.argmin(sm_scores))])
                sm_top1_correct += int(np.argmin(sm_scores) == truth)
                conformal_covered += int(truth in prediction_set)
                set_sizes.append(len(prediction_set))

                if len(prediction_set) > 1:
                    query_count += 1
                    best_bus = 1
                    best_margin = -np.inf
                    for bus in range(1, n_bus):
                        margins = []
                        for idx, a in enumerate(prediction_set):
                            for b in prediction_set[idx + 1 :]:
                                denom = angle_scale[a, t, bus] + angle_scale[b, t, bus]
                                margins.append(
                                    abs(nominal_theta[a, t, bus] - nominal_theta[b, t, bus])
                                    / max(denom, 1e-8)
                                )
                        margin = min(margins) if margins else 0.0
                        if margin > best_margin:
                            best_bus, best_margin = bus, margin
                    selected_buses.append(best_bus)
                    observed_angle = theta[truth, t, best_bus] + rng.normal(0.0, pmu_sigma)
                    combined = (
                        sm_scores[prediction_set]
                        + (
                            (nominal_theta[prediction_set, t, best_bus] - observed_angle)
                            / angle_scale[prediction_set, t, best_bus]
                        )
                        ** 2
                    )
                    chosen = int(prediction_set[int(np.argmin(combined))])
                else:
                    chosen = int(prediction_set[0])
                adaptive_correct += int(chosen == truth)
                total += 1

    bus_histogram = {
        str(bus + 1): int(np.sum(np.asarray(selected_buses) == bus))
        for bus in sorted(set(selected_buses))
    }
    return {
        "innovation": "class-conditional conformal topology sets with adaptive PMU query",
        "candidate_topologies": len(configs),
        "target_miscoverage": alpha,
        "sm_top1_accuracy": sm_top1_correct / total,
        "conformal_set_coverage": conformal_covered / total,
        "average_prediction_set_size": float(np.mean(set_sizes)),
        "adaptive_one_pmu_accuracy": adaptive_correct / total,
        "pmu_query_rate": query_count / total,
        "adaptive_bus_query_histogram_1_based": bus_histogram,
        "test_cases": total,
    }
