"""Bootstrap uncertainty propagation for rooted neighbor joining."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from terminal_case33.estimation.multiscenario import (
    fit_projected_sensitivity,
    preprocess_scenarios,
)
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.graph.rooted_neighbor_joining import (
    RootedTreeResult,
    rooted_neighbor_joining,
)
from terminal_case33.graph.sensitivity_geometry import sensitivity_geometry
from terminal_case33.graph.uncertainty_rnj import (
    UncertainRNJResult,
    rooted_neighbor_joining_with_uncertainty,
)


Clade = frozenset[int]


@dataclass(frozen=True)
class UncertaintyAwareRNJEstimate:
    """Point and uncertainty-aware RNJ trees from the same noisy data."""

    terminals: tuple[int, ...]
    base_tree: RootedTreeResult
    base_clades: frozenset[Clade]
    uncertain: UncertainRNJResult
    uncertain_clades: frozenset[Clade]
    bootstrap_clade_support: dict[Clade, float]
    shared_path_median: np.ndarray
    shared_path_standard_error: np.ndarray
    tolerance: float


def _circular_block_indices(
    count: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one length-preserving circular moving-block bootstrap sample."""

    if count <= 0 or block_length <= 0:
        raise ValueError("count and block_length must be positive")
    indices: list[int] = []
    while len(indices) < count:
        start = int(rng.integers(0, count))
        indices.extend((start + offset) % count for offset in range(block_length))
    return np.asarray(indices[:count], dtype=int)


def _resample_scenario(
    scenario: dict,
    indices: np.ndarray,
    replicate: int,
) -> dict:
    """Resample all synchronized meter channels with identical block indices."""

    def frame(name: str) -> pd.DataFrame:
        return scenario[name].iloc[indices].reset_index(drop=True).copy()

    def series(name: str) -> pd.Series:
        return scenario[name].iloc[indices].reset_index(drop=True).copy()

    return {
        "name": f"{scenario['name']}__bootstrap_{replicate}",
        "P_terminal": frame("P_terminal"),
        "Q_terminal": frame("Q_terminal"),
        "V_terminal": frame("V_terminal"),
        "root_voltage": series("root_voltage"),
        "drop_target": frame("drop_target"),
    }


def _fit_shared_paths(
    scenarios: list[dict],
    distance_mode: str,
    preprocessing: str,
) -> tuple[np.ndarray, np.ndarray]:
    recipe = {
        "name": preprocessing,
        "kind": "demean" if preprocessing == "daily_demean" else preprocessing,
    }
    prepared = preprocess_scenarios(scenarios, recipe)
    r_matrix, x_matrix, _r2, _condition = fit_projected_sensitivity(
        prepared,
        constraint_mode="ordered",
    )
    geometry = sensitivity_geometry(r_matrix, x_matrix, distance_mode)
    return geometry.shared_paths, geometry.root_depths


def estimate_uncertainty_aware_rnj(
    scenarios: list[dict],
    root_bus: int,
    distance_mode: str = "RX_75R_25X",
    preprocessing: str = "daily_demean",
    tolerance_factor: float = 0.16,
    bootstrap_replicates: int = 16,
    block_length: int = 12,
    confidence_level: float = 0.90,
    seed: int = 0,
) -> UncertaintyAwareRNJEstimate:
    """Estimate RNJ topology and propagate moving-block sampling uncertainty."""

    if not scenarios:
        raise ValueError("at least one scenario is required")
    if bootstrap_replicates < 2:
        raise ValueError("bootstrap_replicates must be at least two")
    if tolerance_factor < 0.0:
        raise ValueError("tolerance_factor must be nonnegative")
    terminals = tuple(int(column) for column in scenarios[0]["P_terminal"].columns)
    base_shared, base_depths = _fit_shared_paths(
        scenarios,
        distance_mode,
        preprocessing,
    )
    tolerance = tolerance_factor * max(float(np.median(base_depths)), 1e-12)
    base_tree = rooted_neighbor_joining(
        base_shared,
        base_depths,
        list(terminals),
        int(root_bus),
        tolerance,
    )

    rng = np.random.default_rng(seed)
    shared_samples = []
    depth_samples = []
    replicate_clades: list[set[Clade]] = []
    for replicate in range(bootstrap_replicates):
        resampled = []
        for scenario in scenarios:
            count = len(scenario["P_terminal"])
            indices = _circular_block_indices(count, min(block_length, count), rng)
            resampled.append(_resample_scenario(scenario, indices, replicate))
        shared, depths = _fit_shared_paths(
            resampled,
            distance_mode,
            preprocessing,
        )
        shared_samples.append(shared)
        depth_samples.append(depths)
        replicate_tolerance = tolerance_factor * max(float(np.median(depths)), 1e-12)
        tree = rooted_neighbor_joining(
            shared,
            depths,
            list(terminals),
            int(root_bus),
            replicate_tolerance,
        )
        replicate_clades.append(
            rooted_clades(tree.edges, int(root_bus), list(terminals))
        )

    shared_array = np.asarray(shared_samples, dtype=float)
    depth_array = np.asarray(depth_samples, dtype=float)
    uncertain = rooted_neighbor_joining_with_uncertainty(
        shared_array,
        depth_array,
        list(terminals),
        int(root_bus),
        tolerance,
        confidence_level=confidence_level,
    )
    all_clades = set().union(*replicate_clades) if replicate_clades else set()
    support = {
        clade: float(np.mean([clade in sample for sample in replicate_clades]))
        for clade in all_clades
    }
    return UncertaintyAwareRNJEstimate(
        terminals=terminals,
        base_tree=base_tree,
        base_clades=frozenset(rooted_clades(base_tree.edges, int(root_bus), list(terminals))),
        uncertain=uncertain,
        uncertain_clades=frozenset(
            rooted_clades(uncertain.tree.edges, int(root_bus), list(terminals))
        ),
        bootstrap_clade_support=support,
        shared_path_median=np.median(shared_array, axis=0),
        shared_path_standard_error=np.std(shared_array, axis=0, ddof=1),
        tolerance=float(tolerance),
    )
