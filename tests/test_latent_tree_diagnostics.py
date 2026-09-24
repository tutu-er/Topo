import networkx as nx
import numpy as np

from terminal_case33.graph.diagnostics import locate_root_from_terminal_depths, root_partition_is_correct
from terminal_case33.graph.latent_tree import neighbor_joining, recursive_grouping


def _distance_and_depth(edges, root, terminals):
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)
    distance = np.asarray([[nx.shortest_path_length(graph, i, j, weight="weight") for j in terminals] for i in terminals])
    depth = np.asarray([nx.shortest_path_length(graph, root, i, weight="weight") for i in terminals])
    return distance, depth


def test_exact_depths_locate_the_true_root_partition_for_nj_and_rg():
    root = 10
    terminals = [1, 2, 3, 4, 5, 6]
    true_edges = [
        (10, 11, 0.4), (10, 12, 0.5), (10, 13, 0.6),
        (11, 1, 0.2), (11, 2, 0.3),
        (12, 3, 0.2), (12, 4, 0.3),
        (13, 5, 0.2), (13, 6, 0.3),
    ]
    distance, depth = _distance_and_depth(true_edges, root, terminals)
    for result in (neighbor_joining(distance, terminals), recursive_grouping(distance, terminals, 1e-8)):
        placement = locate_root_from_terminal_depths(result.edges, terminals, depth)
        assert placement.rmse < 1e-10
        assert root_partition_is_correct(placement, result.edges, true_edges, root, terminals)
