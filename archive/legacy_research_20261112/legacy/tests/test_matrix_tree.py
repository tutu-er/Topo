import networkx as nx

from terminal_case33.graph.matrix_tree import matrix_tree_edge_marginals


def test_triangle_equal_weights_marginals():
    nodes = [1, 2, 3]
    edges = [(1, 2), (2, 3), (1, 3)]
    posterior = matrix_tree_edge_marginals(nodes, edges, {e: 1.0 for e in edges})
    assert abs(posterior.sum_marginals - 2.0) < 1e-10
    assert all(abs(v - 2 / 3) < 1e-10 for v in posterior.edge_marginals["edge_marginal"])


def test_chain_edges_have_marginal_one_and_map_tree_valid():
    nodes = [1, 2, 3, 4]
    edges = [(1, 2), (2, 3), (3, 4)]
    posterior = matrix_tree_edge_marginals(nodes, edges, {e: 1.0 for e in edges})
    assert all(abs(v - 1.0) < 1e-10 for v in posterior.edge_marginals["edge_marginal"])
    assert abs(posterior.sum_marginals - (len(nodes) - 1)) < 1e-10
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(posterior.map_tree)
    assert graph.number_of_edges() == len(nodes) - 1
    assert nx.is_tree(graph)

