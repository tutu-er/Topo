import networkx as nx
import numpy as np

from terminal_case33.data.case33bw_raw import load_raw_case33bw
from terminal_case33.models.lin_distflow import build_reduced_sensitivity_matrices, impedance_distance_from_reduced_R, ohm_to_pu
from terminal_case33.scenario.terminalize import terminalize_case33


def test_reduced_sensitivity_distance_matches_path_sum():
    net = terminalize_case33(load_raw_case33bw(), seed=3)
    terminals = net.buses[net.buses["bus_type"].eq("observed_terminal")]["bus_id"].astype(int).tolist()
    R, X = build_reduced_sensitivity_matrices(net, terminals)
    assert R.shape == (len(terminals), len(terminals))
    assert X.shape == R.shape
    assert np.allclose(R, R.T)
    assert np.allclose(X, X.T)
    dR = impedance_distance_from_reduced_R(R)
    assert (dR >= -1e-12).all()
    i, j = 0, min(4, len(terminals) - 1)
    graph = net.to_networkx_graph()
    path = nx.shortest_path(graph, terminals[i], terminals[j])
    edge_sum = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge_sum += ohm_to_pu(net, graph.edges[u, v]["r_ohm"])
    assert abs(dR[i, j] - edge_sum) < 1e-10

