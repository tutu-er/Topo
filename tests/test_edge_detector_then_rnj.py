"""Tests for one-layer latent-edge detection before RNJ reconstruction."""

from __future__ import annotations

import networkx as nx
import numpy as np

from experiments.run_edge_detector_then_rnj import (
    _nj_groups,
    _normalized_boundary_margin,
)


def _additive_example() -> tuple[np.ndarray, np.ndarray, list[int]]:
    terminals = [10, 11, 12, 13]
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [
            (1, -1, 1.0),
            (-1, -2, 0.5),
            (-2, 10, 0.2),
            (-2, 11, 0.3),
            (-1, -3, 0.6),
            (-3, 12, 0.2),
            (-3, 13, 0.25),
        ]
    )
    distance = np.asarray(
        [
            [
                nx.shortest_path_length(graph, left, right, weight="weight")
                for right in terminals
            ]
            for left in terminals
        ],
        dtype=float,
    )
    depth = np.asarray(
        [nx.shortest_path_length(graph, 1, terminal, weight="weight") for terminal in terminals],
        dtype=float,
    )
    return distance, depth, terminals


def test_nj_first_layer_recovers_rooted_edge_clades() -> None:
    distance, depth, terminals = _additive_example()

    groups = _nj_groups(distance, depth, terminals, max_cluster_size=2)

    assert groups == {frozenset({10, 11}), frozenset({12, 13})}


def test_boundary_margin_rejects_cross_branch_group() -> None:
    distance, depth, terminals = _additive_example()

    true_margin = _normalized_boundary_margin(
        frozenset({10, 11}),
        distance,
        depth,
        terminals,
    )
    false_margin = _normalized_boundary_margin(
        frozenset({10, 12}),
        distance,
        depth,
        terminals,
    )

    assert true_margin > 0.1
    assert false_margin < 0.0
