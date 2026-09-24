"""Rooted-neighbor-joining candidate generation from reduced R/X matrices."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.graph.rooted_neighbor_joining import (
    RootedTreeResult,
    rooted_neighbor_joining,
)
from terminal_case33.graph.sensitivity_geometry import sensitivity_geometry


@dataclass(frozen=True)
class RNJCandidate:
    """One unique rooted topology proposed by an RNJ parameter setting."""

    tree: RootedTreeResult
    distance_mode: str
    tolerance_factor: float
    clades: frozenset[frozenset[int]]


def _distance_and_depth(
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    distance_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    geometry = sensitivity_geometry(r_matrix, x_matrix, distance_mode)
    return geometry.distance, geometry.root_depths


def generate_rnj_candidates(
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    terminals: list[int],
    root_bus: int,
    distance_modes: tuple[str, ...] = (
        "R",
        "X",
        "RX_equal_normalized",
        "RX_75R_25X",
    ),
    tolerance_factors: tuple[float, ...] = (0.12, 0.16, 0.20, 0.24, 0.28, 0.32),
) -> list[RNJCandidate]:
    """Generate unique rooted topologies over an R/X and tolerance grid."""

    if not distance_modes or not tolerance_factors:
        raise ValueError("distance_modes and tolerance_factors must be nonempty")
    unique: dict[frozenset[frozenset[int]], RNJCandidate] = {}
    for mode in distance_modes:
        geometry = sensitivity_geometry(r_matrix, x_matrix, mode)
        depths = geometry.root_depths
        scale = max(float(np.median(depths)), 1e-12)
        for tolerance_factor in tolerance_factors:
            tree = rooted_neighbor_joining(
                geometry.shared_paths,
                depths,
                terminals,
                root_bus,
                float(tolerance_factor) * scale,
            )
            clades = frozenset(rooted_clades(tree.edges, root_bus, terminals))
            candidate = RNJCandidate(
                tree=tree,
                distance_mode=mode,
                tolerance_factor=float(tolerance_factor),
                clades=clades,
            )
            unique.setdefault(clades, candidate)
    return list(unique.values())


def rooted_path_incidence(
    tree: RootedTreeResult,
) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    """Return root-oriented edges and terminal-to-edge path incidence."""

    graph = nx.Graph()
    graph.add_weighted_edges_from(tree.edges)
    if not nx.is_tree(graph) or tree.root not in graph:
        raise ValueError("candidate must be a connected rooted tree")
    oriented = tuple((int(parent), int(child)) for parent, child in nx.bfs_edges(graph, tree.root))
    edge_position = {edge: index for index, edge in enumerate(oriented)}
    incidence = np.zeros((len(tree.terminals), len(oriented)), dtype=float)
    for terminal_index, terminal in enumerate(tree.terminals):
        path = nx.shortest_path(graph, tree.root, terminal)
        for parent, child in zip(path[:-1], path[1:]):
            incidence[terminal_index, edge_position[(int(parent), int(child))]] = 1.0
    return oriented, incidence
