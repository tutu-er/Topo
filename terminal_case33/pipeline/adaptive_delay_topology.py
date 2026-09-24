"""Adaptive rooted topology recovery from uncertain reduced sensitivities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from terminal_case33.estimation.multiscenario import fit_projected_sensitivity
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.graph.rooted_neighbor_joining import (
    RootedTreeResult,
    rooted_neighbor_joining,
    shared_paths_from_distances,
)
from terminal_case33.models.lin_distflow import impedance_distance_from_reduced_R


@dataclass(frozen=True)
class AdaptiveTopologyResult:
    """RNJ result with uncertainty-weighted distance and tolerance-path support."""

    r_matrix: np.ndarray
    x_matrix: np.ndarray
    r_weight: float
    distance_matrix: np.ndarray
    root_depths: np.ndarray
    selected_tolerance_factor: float
    selected_tree: RootedTreeResult
    raw_clades: frozenset[frozenset[int]]
    pruned_clades: frozenset[frozenset[int]]
    clade_support: dict[frozenset[int], float]
    candidate_scores: dict[float, float]


def _normalized_distance(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    distance = impedance_distance_from_reduced_R(matrix)
    positive = distance[distance > 0.0]
    scale = max(float(np.mean(positive)) if positive.size else 1.0, 1e-12)
    return distance / scale, np.diag(matrix) / scale, scale


def _reliability_weight(
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    jackknife_r: list[np.ndarray],
    jackknife_x: list[np.ndarray],
) -> float:
    """Return a scalar inverse-variance weight on R distance."""

    if len(jackknife_r) < 2:
        return 0.75
    r_distances = np.stack([_normalized_distance(value)[0] for value in jackknife_r])
    x_distances = np.stack([_normalized_distance(value)[0] for value in jackknife_x])
    upper = np.triu_indices(r_matrix.shape[0], 1)
    variance_r = float(np.median(np.var(r_distances[:, upper[0], upper[1]], axis=0)))
    variance_x = float(np.median(np.var(x_distances[:, upper[0], upper[1]], axis=0)))
    denominator = variance_r + variance_x
    if denominator <= 1e-15:
        return 0.75
    weight_r = variance_x / denominator
    return float(np.clip(weight_r, 0.50, 0.95))


def _mixed_distance(
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    r_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    d_r, h_r, _ = _normalized_distance(r_matrix)
    d_x, h_x, _ = _normalized_distance(x_matrix)
    return (
        r_weight * d_r + (1.0 - r_weight) * d_x,
        r_weight * h_r + (1.0 - r_weight) * h_x,
    )


def _jaccard(left: set[frozenset[int]], right: set[frozenset[int]]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


def _tree_for(
    distance: np.ndarray,
    depths: np.ndarray,
    terminals: list[int],
    root_bus: int,
    tolerance_factor: float,
) -> RootedTreeResult:
    tolerance = tolerance_factor * max(float(np.median(depths)), 1e-12)
    return rooted_neighbor_joining(
        shared_paths_from_distances(distance, depths),
        depths,
        terminals,
        root_bus,
        tolerance,
    )


def fit_adaptive_rooted_topology(
    scenarios: list[dict],
    root_bus: int,
    tolerance_candidates: tuple[float, ...] = (0.12, 0.16, 0.20, 0.24, 0.28, 0.32),
    persistence_threshold: float = 0.70,
) -> AdaptiveTopologyResult:
    """Fit R/X, select a tolerance-path medoid, and contract weak clades.

    R/X distance weights are determined from scenario-jackknife variance. RNJ
    is run over a tolerance path. The path medoid maximizes mean clade Jaccard
    agreement. Clade support additionally includes scenario jackknife and
    R/RX75 distance perturbations before low-support hidden edges are removed.
    """

    if not scenarios:
        raise ValueError("at least one scenario is required")
    if not tolerance_candidates:
        raise ValueError("at least one tolerance candidate is required")
    if not 0.0 <= persistence_threshold <= 1.0:
        raise ValueError("persistence_threshold must be in [0, 1]")
    terminals = [int(column) for column in scenarios[0]["P_terminal"].columns]
    r_matrix, x_matrix, _r2, _condition = fit_projected_sensitivity(
        scenarios,
        constraint_mode="ordered",
    )
    jackknife_r: list[np.ndarray] = []
    jackknife_x: list[np.ndarray] = []
    if len(scenarios) >= 3:
        for omitted in range(len(scenarios)):
            subset = [item for index, item in enumerate(scenarios) if index != omitted]
            r_jack, x_jack, _r2, _condition = fit_projected_sensitivity(
                subset,
                constraint_mode="ordered",
            )
            jackknife_r.append(r_jack)
            jackknife_x.append(x_jack)
    r_weight = _reliability_weight(r_matrix, x_matrix, jackknife_r, jackknife_x)
    distance, depths = _mixed_distance(r_matrix, x_matrix, r_weight)

    path_trees: dict[float, RootedTreeResult] = {}
    path_clades: dict[float, set[frozenset[int]]] = {}
    for factor in sorted(set(float(value) for value in tolerance_candidates)):
        tree = _tree_for(distance, depths, terminals, root_bus, factor)
        path_trees[factor] = tree
        path_clades[factor] = rooted_clades(tree.edges, root_bus, terminals)
    candidate_scores = {
        factor: float(
            np.mean([_jaccard(clades, comparison) for comparison in path_clades.values()])
        )
        for factor, clades in path_clades.items()
    }
    center = float(np.median(list(path_clades)))
    selected_factor = max(
        path_clades,
        key=lambda factor: (
            candidate_scores[factor],
            -abs(factor - center),
            factor,
        ),
    )
    selected_tree = path_trees[selected_factor]
    selected_clades = path_clades[selected_factor]

    ensemble_clades = list(path_clades.values())
    for r_jack, x_jack in zip(jackknife_r, jackknife_x, strict=True):
        jack_distance, jack_depths = _mixed_distance(r_jack, x_jack, r_weight)
        jack_tree = _tree_for(
            jack_distance,
            jack_depths,
            terminals,
            root_bus,
            selected_factor,
        )
        ensemble_clades.append(rooted_clades(jack_tree.edges, root_bus, terminals))
    for alternative_weight in (1.0, 0.75):
        alt_distance, alt_depths = _mixed_distance(r_matrix, x_matrix, alternative_weight)
        alt_tree = _tree_for(
            alt_distance,
            alt_depths,
            terminals,
            root_bus,
            selected_factor,
        )
        ensemble_clades.append(rooted_clades(alt_tree.edges, root_bus, terminals))

    counts: Counter = Counter()
    for clades in ensemble_clades:
        counts.update(clades)
    denominator = max(len(ensemble_clades), 1)
    support = {
        clade: counts[clade] / denominator
        for clade in set().union(*ensemble_clades)
    }
    pruned = {
        clade
        for clade in selected_clades
        if support.get(clade, 0.0) >= persistence_threshold
    }
    return AdaptiveTopologyResult(
        r_matrix=r_matrix,
        x_matrix=x_matrix,
        r_weight=r_weight,
        distance_matrix=distance,
        root_depths=depths,
        selected_tolerance_factor=selected_factor,
        selected_tree=selected_tree,
        raw_clades=frozenset(selected_clades),
        pruned_clades=frozenset(pruned),
        clade_support=support,
        candidate_scores=candidate_scores,
    )