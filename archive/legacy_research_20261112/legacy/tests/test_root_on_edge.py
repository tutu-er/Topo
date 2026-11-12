import networkx as nx
import numpy as np

from terminal_case33.graph.diagnostics import locate_root_from_terminal_depths, root_partition_is_correct
from terminal_case33.graph.latent_tree import neighbor_joining


def test_degree_two_root_is_recovered_as_a_point_on_latent_edge():
    root = 10
    terminals = [1, 2, 3, 4]
    true_edges = [
        (10, 11, 0.4), (10, 12, 0.5),
        (11, 1, 0.2), (11, 2, 0.3),
        (12, 3, 0.2), (12, 4, 0.3),
    ]
    graph = nx.Graph()
    graph.add_weighted_edges_from(true_edges)
    distance = np.asarray([[nx.shortest_path_length(graph, i, j, weight="weight") for j in terminals] for i in terminals])
    depth = np.asarray([nx.shortest_path_length(graph, root, i, weight="weight") for i in terminals])
    result = neighbor_joining(distance, terminals)
    placement = locate_root_from_terminal_depths(result.edges, terminals, depth)
    assert placement.kind == "edge"
    assert placement.rmse < 1e-10
    assert root_partition_is_correct(placement, result.edges, true_edges, root, terminals)
