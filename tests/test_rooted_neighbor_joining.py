import networkx as nx
import numpy as np

from terminal_case33.graph.diagnostics import root_partition_is_correct, terminal_sibling_pairs
from terminal_case33.graph.latent_tree import terminal_splits
from terminal_case33.graph.rooted_neighbor_joining import rooted_neighbor_joining, shared_paths_from_distances


def test_rnj_recovers_exact_rooted_multifurcating_tree():
    root = 10
    terminals = [1, 2, 3, 4, 5, 6]
    true_edges = [
        (10, 11, 0.4), (10, 12, 0.5),
        (11, 1, 0.2), (11, 2, 0.3), (11, 3, 0.4),
        (12, 4, 0.2), (12, 5, 0.3), (12, 6, 0.4),
    ]
    graph = nx.Graph()
    graph.add_weighted_edges_from(true_edges)
    distance = np.asarray([[nx.shortest_path_length(graph, i, j, weight="weight") for j in terminals] for i in terminals])
    depth = np.asarray([nx.shortest_path_length(graph, root, i, weight="weight") for i in terminals])
    shared = shared_paths_from_distances(distance, depth)
    result = rooted_neighbor_joining(shared, depth, terminals, root, group_tolerance=1e-10)
    assert terminal_splits(result.edges, terminals) == terminal_splits(true_edges, terminals)
    assert terminal_sibling_pairs(result.edges, terminals) == terminal_sibling_pairs(true_edges, terminals)
    placement = type("Placement", (), {"kind": "node", "node": root, "edge": None})()
    assert root_partition_is_correct(placement, result.edges, true_edges, root, terminals)
