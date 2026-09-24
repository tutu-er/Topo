import networkx as nx
import numpy as np

from terminal_case33.graph.latent_tree import neighbor_joining, recursive_grouping, terminal_splits


def _additive_distance(edges, terminals):
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)
    return np.asarray([[nx.shortest_path_length(graph, i, j, weight="weight") for j in terminals] for i in terminals])


def test_neighbor_joining_recovers_exact_binary_tree_splits():
    terminals = [1, 2, 3, 4, 5, 6]
    true_edges = [
        (-1, 1, 0.2), (-1, 2, 0.3), (-1, -3, 0.4),
        (-2, 3, 0.2), (-2, 4, 0.4), (-2, -3, 0.5),
        (-3, -4, 0.3), (-4, 5, 0.2), (-4, 6, 0.3),
    ]
    distance = _additive_distance(true_edges, terminals)
    result = neighbor_joining(distance, terminals)
    assert terminal_splits(result.edges, terminals) == terminal_splits(true_edges, terminals)


def test_recursive_grouping_recovers_multifurcating_tree_splits():
    terminals = [1, 2, 3, 4, 5, 6, 7]
    true_edges = [
        (-1, 1, 0.2), (-1, 2, 0.3), (-1, 3, 0.4), (-1, -3, 0.5),
        (-2, 4, 0.2), (-2, 5, 0.3), (-2, -3, 0.4),
        (-3, 6, 0.2), (-3, 7, 0.3),
    ]
    distance = _additive_distance(true_edges, terminals)
    result = recursive_grouping(distance, terminals, tolerance=1e-8)
    assert result.forced_merges == 0
    assert terminal_splits(result.edges, terminals) == terminal_splits(true_edges, terminals)


def test_latent_algorithms_are_deterministic_under_small_noise():
    terminals = [1, 2, 3, 4]
    true_edges = [(-1, 1, 0.2), (-1, 2, 0.3), (-1, -2, 0.5), (-2, 3, 0.2), (-2, 4, 0.4)]
    distance = _additive_distance(true_edges, terminals)
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 1e-4, size=distance.shape)
    noisy = np.maximum(0.0, distance + 0.5 * (noise + noise.T))
    np.fill_diagonal(noisy, 0.0)
    assert neighbor_joining(noisy, terminals) == neighbor_joining(noisy, terminals)
    assert recursive_grouping(noisy, terminals) == recursive_grouping(noisy, terminals)
