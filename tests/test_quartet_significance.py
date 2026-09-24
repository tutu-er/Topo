"""Tests for bootstrap-ready quartet clade significance."""

from __future__ import annotations

import networkx as nx
import numpy as np

from terminal_case33.graph.quartet_significance import evaluate_quartet_clades


def _tree_metric() -> tuple[list[int], np.ndarray, np.ndarray]:
    root = 0
    terminals = [1, 2, 3, 4]
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [
            (root, 10, 1.0),
            (10, 1, 0.7),
            (10, 2, 1.1),
            (root, 20, 1.4),
            (20, 3, 0.8),
            (20, 4, 1.2),
        ]
    )
    distance = np.array(
        [
            [nx.shortest_path_length(graph, left, right, weight="weight") for right in terminals]
            for left in terminals
        ],
        dtype=float,
    )
    depths = np.array(
        [nx.shortest_path_length(graph, root, node, weight="weight") for node in terminals],
        dtype=float,
    )
    return terminals, distance, depths


def test_quartet_significance_accepts_true_splits_and_rejects_false_split() -> None:
    terminals, distance, depths = _tree_metric()
    rng = np.random.default_rng(7)
    distances = []
    depth_samples = []
    for _ in range(40):
        noise = rng.normal(0.0, 0.005, distance.shape)
        noise = 0.5 * (noise + noise.T)
        np.fill_diagonal(noise, 0.0)
        distances.append(np.maximum(distance + noise, 0.0))
        depth_samples.append(np.maximum(depths + rng.normal(0.0, 0.003, len(depths)), 0.0))
    true_left = frozenset({1, 2})
    true_right = frozenset({3, 4})
    false_mixed = frozenset({1, 3})
    accepted, evidence = evaluate_quartet_clades(
        {true_left, true_right, false_mixed},
        terminals,
        distances,
        depth_samples,
        confidence_level=0.90,
        minimum_positive_support=0.80,
        minimum_effect=0.01,
    )
    assert true_left in accepted
    assert true_right in accepted
    assert false_mixed not in accepted
    assert evidence[true_left].lower_confidence_bound > 0.0
    assert evidence[false_mixed].estimate < 0.0
