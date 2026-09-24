"""Stable peripheral-clade proposals from unrooted NJ and rooted RG."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from terminal_case33.estimation.multiscenario import (
    fit_projected_sensitivity,
    preprocess_scenarios,
)
from terminal_case33.graph.diagnostics import locate_root_from_terminal_depths
from terminal_case33.graph.latent_tree import neighbor_joining, recursive_grouping
from terminal_case33.graph.rooted_hierarchy import PseudoCluster
from terminal_case33.graph.rooted_neighbor_joining import shared_paths_from_distances
from terminal_case33.models.lin_distflow import impedance_distance_from_reduced_R
from terminal_case33.estimation.preprocessing import (
    squared_voltage_drop_from_observed_root,
)


Clade = frozenset[int]
WeightedEdge = tuple[int, int, float]


@dataclass(frozen=True)
class PeripheralClusterEvidence:
    """Label-free stability and separation diagnostics for one clade."""

    clade: Clade
    in_base_nj: bool
    in_base_rg: bool
    nj_support: float
    rg_support: float
    consensus_support: float
    boundary_margin: float
    normalized_edge_length: float


@dataclass(frozen=True)
class PeripheralProposalResult:
    """Compatible NJ and NJ/RG cluster proposals before external confirmation."""

    terminals: tuple[int, ...]
    distance: np.ndarray
    root_depths: np.ndarray
    evidence: dict[Clade, PeripheralClusterEvidence]
    clusters_by_mode: dict[str, tuple[PseudoCluster, ...]]

    @property
    def candidate_clades(self) -> frozenset[Clade]:
        """Return every clade selected by at least one detector mode."""

        return frozenset(
            cluster.members for clusters in self.clusters_by_mode.values() for cluster in clusters
        )


def _distance_and_depth(
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    d_r = impedance_distance_from_reduced_R(r_matrix)
    d_x = impedance_distance_from_reduced_R(x_matrix)

    def scale(value: np.ndarray) -> float:
        positive = value[value > 0.0]
        return max(float(np.mean(positive)) if positive.size else 1.0, 1e-12)

    if mode == "R":
        return d_r, np.diag(r_matrix)
    if mode == "X":
        return d_x, np.diag(x_matrix)
    weights = {
        "RX_equal_normalized": (0.5, 0.5),
        "RX_75R_25X": (0.75, 0.25),
        "RX_25R_75X": (0.25, 0.75),
    }
    if mode not in weights:
        raise ValueError(f"unknown distance mode {mode!r}")
    weight_r, weight_x = weights[mode]
    scale_r, scale_x = scale(d_r), scale(d_x)
    return (
        weight_r * d_r / scale_r + weight_x * d_x / scale_x,
        weight_r * np.diag(r_matrix) / scale_r + weight_x * np.diag(x_matrix) / scale_x,
    )


def _fit_distance(
    scenarios: list[dict],
    recipe: dict,
    distance_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    prepared = preprocess_scenarios(scenarios, recipe)
    r_matrix, x_matrix, _r2, _condition = fit_projected_sensitivity(
        prepared,
        constraint_mode="ordered",
    )
    return _distance_and_depth(r_matrix, x_matrix, distance_mode)


def _root_nj_edges(
    edges: tuple[WeightedEdge, ...],
    terminals: list[int],
    root_depths: np.ndarray,
    root_label: int,
) -> tuple[tuple[WeightedEdge, ...], int]:
    """Place the known root on an unrooted NJ tree without discarding edge length."""

    placement = locate_root_from_terminal_depths(edges, terminals, root_depths)
    if placement.kind == "node":
        if placement.node is None:
            raise RuntimeError("node root placement did not return a node")
        return edges, int(placement.node)
    if placement.edge is None:
        raise RuntimeError("edge root placement did not return an edge")
    left, right = placement.edge
    original = next(edge for edge in edges if {int(edge[0]), int(edge[1])} == {left, right})
    length = float(original[2])
    offset = float(np.clip(placement.offset_from_edge_start, 0.0, length))
    if int(original[0]) != left:
        offset = length - offset
    synthetic_root = int(root_label)
    if synthetic_root in {node for edge in edges for node in edge[:2]}:
        synthetic_root = min(-1_000_000, min(node for edge in edges for node in edge[:2]) - 1)
    rooted = tuple(edge for edge in edges if edge is not original) + (
        (synthetic_root, left, offset),
        (synthetic_root, right, max(0.0, length - offset)),
    )
    return rooted, synthetic_root


def _edge_clades(
    edges: tuple[WeightedEdge, ...],
    root: int,
    terminals: list[int],
    maximum_cluster_size: int,
) -> dict[Clade, float]:
    graph = nx.Graph()
    for left, right, length in edges:
        graph.add_edge(int(left), int(right), weight=max(float(length), 0.0))
    if root not in graph or not nx.is_tree(graph):
        raise ValueError("rooted latent-tree edges must form a tree containing the root")
    directed = nx.bfs_tree(graph, root)
    terminal_set = set(terminals)
    result: dict[Clade, float] = {}
    for parent, child in directed.edges():
        descendants = nx.descendants(directed, child) | {child}
        clade = frozenset(terminal_set.intersection(descendants))
        if 2 <= len(clade) <= maximum_cluster_size:
            result[clade] = float(graph[parent][child]["weight"])
    return result


def _detector_groups(
    distance: np.ndarray,
    root_depths: np.ndarray,
    terminals: list[int],
    root: int,
    rg_tolerance: float,
    maximum_cluster_size: int,
) -> tuple[dict[Clade, float], dict[Clade, float]]:
    nj_tree = neighbor_joining(distance, terminals)
    nj_edges, nj_root = _root_nj_edges(
        nj_tree.edges,
        terminals,
        root_depths,
        root,
    )
    nj_groups = _edge_clades(
        nj_edges,
        nj_root,
        terminals,
        maximum_cluster_size,
    )

    augmented = np.zeros((len(terminals) + 1, len(terminals) + 1), dtype=float)
    augmented[:-1, :-1] = distance
    augmented[:-1, -1] = root_depths
    augmented[-1, :-1] = root_depths
    rg_tree = recursive_grouping(
        augmented,
        [*terminals, int(root)],
        tolerance=rg_tolerance,
    )
    rg_groups = _edge_clades(
        rg_tree.edges,
        int(root),
        terminals,
        maximum_cluster_size,
    )
    return nj_groups, rg_groups


def _relative_noise(
    frame: pd.DataFrame,
    relative_std: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    positive = np.abs(values[np.abs(values) > 0.0])
    floor = 0.01 * float(np.median(positive)) if positive.size else 1e-12
    scale = relative_std * np.maximum(np.abs(values), floor)
    return pd.DataFrame(
        rng.normal(0.0, scale, values.shape),
        index=frame.index,
        columns=frame.columns,
    )


def _perturb_blocks(
    scenarios: list[dict],
    rng: np.random.Generator,
    block_fraction: float,
    pq_extra_noise_rel: float,
    voltage_extra_noise_rel: float,
) -> list[dict]:
    sampled = []
    for scenario in scenarios:
        count = len(scenario["P_terminal"])
        block_size = max(4, min(count, int(round(block_fraction * count))))
        start = int(rng.integers(0, count))
        positions = (start + np.arange(block_size)) % count

        def take(key: str) -> pd.DataFrame:
            return scenario[key].iloc[positions].reset_index(drop=True).copy()

        p = take("P_terminal")
        q = take("Q_terminal")
        voltage = take("V_terminal")
        p += _relative_noise(p, pq_extra_noise_rel, rng)
        q += _relative_noise(q, pq_extra_noise_rel, rng)
        voltage += _relative_noise(voltage, voltage_extra_noise_rel, rng)
        root_voltage = scenario["root_voltage"].iloc[positions].reset_index(drop=True).copy()
        sampled.append(
            {
                "name": scenario.get("name", "scenario"),
                "P_terminal": p,
                "Q_terminal": q,
                "V_terminal": voltage,
                "root_voltage": root_voltage,
                "drop_target": squared_voltage_drop_from_observed_root(
                    voltage,
                    root_voltage,
                ),
            }
        )
    return sampled


def _boundary_margin(
    clade: Clade,
    distance: np.ndarray,
    root_depths: np.ndarray,
    terminals: list[int],
) -> float:
    shared = shared_paths_from_distances(distance, root_depths)
    position = {terminal: index for index, terminal in enumerate(terminals)}
    inside = [position[terminal] for terminal in sorted(clade)]
    outside = [index for index, terminal in enumerate(terminals) if terminal not in clade]
    within = shared[np.ix_(inside, inside)][np.triu_indices(len(inside), 1)]
    if not within.size or not outside:
        return 0.0
    boundary = float(np.quantile(within, 0.20))
    competitor = max(float(np.median(shared[np.ix_(inside, [index])])) for index in outside)
    return (boundary - competitor) / max(float(np.median(root_depths)), 1e-12)


def _compatible_clusters(
    evidence: dict[Clade, PeripheralClusterEvidence],
    mode: str,
    minimum_support: float,
    minimum_boundary_margin: float,
    minimum_edge_length_ratio: float,
    pseudo_id_start: int,
) -> tuple[PseudoCluster, ...]:
    if mode == "nj_stable":

        def support(item: PeripheralClusterEvidence) -> float:
            return item.nj_support

        def eligible(item: PeripheralClusterEvidence) -> bool:
            return item.in_base_nj or item.nj_support >= minimum_support

    elif mode == "nj_rg_consensus":

        def support(item: PeripheralClusterEvidence) -> float:
            return item.consensus_support

        def eligible(item: PeripheralClusterEvidence) -> bool:
            return bool(
                (item.in_base_nj and item.in_base_rg) or item.consensus_support >= minimum_support
            )

    else:
        raise ValueError(f"unknown peripheral proposal mode {mode!r}")
    candidates = [
        item
        for item in evidence.values()
        if eligible(item)
        and support(item) >= minimum_support
        and item.boundary_margin >= minimum_boundary_margin
        and item.normalized_edge_length >= minimum_edge_length_ratio
    ]
    selected: list[PeripheralClusterEvidence] = []
    occupied: set[int] = set()
    for item in sorted(
        candidates,
        key=lambda value: (
            -support(value),
            -value.boundary_margin,
            -len(value.clade),
            tuple(sorted(value.clade)),
        ),
    ):
        if occupied.isdisjoint(item.clade):
            selected.append(item)
            occupied.update(item.clade)
    return tuple(
        PseudoCluster(
            pseudo_id=pseudo_id_start + index,
            members=item.clade,
            confidence=float(support(item)),
            frozen_clades=tuple(),
            frozen_sibling_pairs=tuple(),
        )
        for index, item in enumerate(selected)
    )


def propose_peripheral_clusters(
    scenarios: list[dict],
    root_bus: int,
    preprocessing_recipe: dict | None = None,
    distance_mode: str = "RX_75R_25X",
    bootstrap_replicates: int = 8,
    rg_tolerance: float = 0.03,
    maximum_cluster_size: int = 5,
    minimum_support: float = 0.875,
    minimum_boundary_margin: float = 0.10,
    minimum_edge_length_ratio: float = 0.02,
    block_fraction: float = 0.70,
    pq_extra_noise_rel: float = 0.0025,
    voltage_extra_noise_rel: float = 0.0001,
    seed: int = 0,
) -> PeripheralProposalResult:
    """Generate stable NJ and NJ/RG peripheral-clade proposals.

    The input measurements are already noisy. Bootstrap replicates add one
    smaller perturbation layer and use circular time blocks. Only region
    membership is proposed; no NJ/RG internal edge is frozen.
    """

    if not scenarios:
        raise ValueError("at least one scenario is required")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0.5 <= block_fraction <= 1.0:
        raise ValueError("block_fraction must lie in [0.5, 1]")
    if not 0.0 <= minimum_support <= 1.0:
        raise ValueError("minimum_support must lie in [0, 1]")
    if maximum_cluster_size < 2:
        raise ValueError("maximum_cluster_size must be at least two")
    recipe = preprocessing_recipe or {
        "name": "daily_demean",
        "kind": "demean",
    }
    terminals = [int(column) for column in scenarios[0]["P_terminal"].columns]
    distance, root_depths = _fit_distance(scenarios, recipe, distance_mode)
    base_nj, base_rg = _detector_groups(
        distance,
        root_depths,
        terminals,
        int(root_bus),
        rg_tolerance,
        maximum_cluster_size,
    )
    universe = set(base_nj) | set(base_rg)
    counts = {clade: {"nj": 0, "rg": 0, "consensus": 0} for clade in universe}
    sampled_lengths: dict[Clade, dict[str, list[float]]] = {
        clade: {"nj": [], "rg": []} for clade in universe
    }
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap_replicates):
        sampled = _perturb_blocks(
            scenarios,
            rng,
            block_fraction,
            pq_extra_noise_rel,
            voltage_extra_noise_rel,
        )
        sampled_distance, sampled_depths = _fit_distance(
            sampled,
            recipe,
            distance_mode,
        )
        sampled_nj, sampled_rg = _detector_groups(
            sampled_distance,
            sampled_depths,
            terminals,
            int(root_bus),
            rg_tolerance,
            maximum_cluster_size,
        )
        for clade in set(sampled_nj) | set(sampled_rg):
            counts.setdefault(clade, {"nj": 0, "rg": 0, "consensus": 0})
            sampled_lengths.setdefault(clade, {"nj": [], "rg": []})
        for clade, value in counts.items():
            in_nj = clade in sampled_nj
            in_rg = clade in sampled_rg
            value["nj"] += int(in_nj)
            value["rg"] += int(in_rg)
            value["consensus"] += int(in_nj and in_rg)
            if in_nj:
                sampled_lengths[clade]["nj"].append(sampled_nj[clade])
            if in_rg:
                sampled_lengths[clade]["rg"].append(sampled_rg[clade])
    universe = set(counts)
    depth_scale = max(float(np.median(root_depths)), 1e-12)

    def mean_length(clade: Clade, detector: str) -> float:
        values = sampled_lengths[clade][detector]
        return float(np.mean(values)) if values else 0.0

    evidence = {
        clade: PeripheralClusterEvidence(
            clade=clade,
            in_base_nj=clade in base_nj,
            in_base_rg=clade in base_rg,
            nj_support=counts[clade]["nj"] / bootstrap_replicates,
            rg_support=counts[clade]["rg"] / bootstrap_replicates,
            consensus_support=counts[clade]["consensus"] / bootstrap_replicates,
            boundary_margin=_boundary_margin(
                clade,
                distance,
                root_depths,
                terminals,
            ),
            normalized_edge_length=max(
                base_nj.get(clade, 0.0),
                base_rg.get(clade, 0.0),
                mean_length(clade, "nj"),
                mean_length(clade, "rg"),
            )
            / depth_scale,
        )
        for clade in universe
    }
    clusters_by_mode = {
        "nj_stable": _compatible_clusters(
            evidence,
            "nj_stable",
            minimum_support,
            minimum_boundary_margin,
            minimum_edge_length_ratio,
            910_000,
        ),
        "nj_rg_consensus": _compatible_clusters(
            evidence,
            "nj_rg_consensus",
            minimum_support,
            minimum_boundary_margin,
            minimum_edge_length_ratio,
            920_000,
        ),
    }
    return PeripheralProposalResult(
        terminals=tuple(terminals),
        distance=distance,
        root_depths=root_depths,
        evidence=evidence,
        clusters_by_mode=clusters_by_mode,
    )
