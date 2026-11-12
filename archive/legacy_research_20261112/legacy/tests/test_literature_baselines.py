import networkx as nx
import numpy as np
import pandas as pd

from terminal_case33.estimation.literature_sensitivity import fit_current_sensitivity
from terminal_case33.graph.cl_grouping import cl_grouping
from terminal_case33.graph.enhanced_recursive_grouping import enhanced_recursive_grouping
from terminal_case33.graph.latent_tree import terminal_splits
from terminal_case33.graph.partial_meter import insert_interval_meters
from terminal_case33.graph.prufer import (
    prufer_decode,
    prufer_encode,
    soumalas_prufer_reconstruction,
)


def _distance(edges, nodes):
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)
    return np.asarray(
        [[nx.shortest_path_length(graph, left, right, weight="weight") for right in nodes] for left in nodes]
    )


def test_prufer_round_trip_and_integer_reconstruction():
    edges = [(-1, 1), (-1, 2), (-1, -2), (-2, 3), (-2, 4)]
    labels = sorted({node for edge in edges for node in edge})
    assert set(map(frozenset, prufer_decode(prufer_encode(edges), labels))) == set(map(frozenset, edges))

    terminals = [1, 2, 3, 4]
    weighted = [(left, right, 0.2) for left, right in edges]
    result = soumalas_prufer_reconstruction(_distance(weighted, terminals), terminals, [0.1, 0.2, 0.3])
    assert terminal_splits(result.edges, terminals) == terminal_splits(edges, terminals)


def test_enhanced_recursive_grouping_recovers_rooted_additive_tree():
    terminals = [1, 2, 3, 4]
    root = 0
    true_edges = [
        (root, -1, 0.5),
        (-1, 1, 0.2),
        (-1, 2, 0.3),
        (-1, -2, 0.4),
        (-2, 3, 0.2),
        (-2, 4, 0.3),
    ]
    nodes = [*terminals, root]
    result = enhanced_recursive_grouping(
        _distance(true_edges, nodes),
        nodes,
        observed_leaves=set(terminals),
        root=root,
        epsilon=1e-10,
    )
    graph = nx.Graph()
    graph.add_weighted_edges_from(result.best.edges)
    assert nx.is_tree(graph)
    assert all(graph.degree(node) == 1 for node in terminals)
    assert terminal_splits(result.best.edges, terminals) == terminal_splits(true_edges, terminals)
    assert result.best.distance_stress < 1e-10


def test_cl_grouping_returns_complete_latent_tree():
    terminals = [1, 2, 3, 4, 5]
    true_edges = [
        (-1, 1, 0.2),
        (-1, 2, 0.3),
        (-1, -2, 0.4),
        (-2, 3, 0.2),
        (-2, 4, 0.3),
        (-2, 5, 0.4),
    ]
    result = cl_grouping(_distance(true_edges, terminals), terminals, tolerance=1e-8)
    graph = nx.Graph()
    graph.add_weighted_edges_from(result.edges)
    assert nx.is_tree(graph)
    assert set(terminals).issubset(graph.nodes)


def test_flynn_common_mode_improves_fit():
    rng = np.random.default_rng(12)
    samples = 180
    nodes = [1, 2, 3]
    r = np.asarray([[0.40, 0.20, 0.10], [0.20, 0.45, 0.10], [0.10, 0.10, 0.35]])
    x = 0.6 * r
    current_r = rng.normal(0.02, 0.004, size=(samples, 3))
    current_x = rng.normal(0.008, 0.002, size=(samples, 3))
    common = 0.003 * np.sin(np.linspace(0.0, 8.0 * np.pi, samples))
    voltage = 1.0 + common[:, None] - current_r @ r.T - current_x @ x.T
    active = current_r * voltage
    reactive = current_x * voltage
    frames = {
        "V": pd.DataFrame(voltage, columns=nodes),
        "P": pd.DataFrame(active, columns=nodes),
        "Q": pd.DataFrame(reactive, columns=nodes),
    }
    constant = fit_current_sensitivity(frames["V"], frames["P"], frames["Q"], "constant")
    flynn = fit_current_sensitivity(frames["V"], frames["P"], frames["Q"], "free_regularized")
    assert flynn.r2_score > constant.r2_score
    assert np.allclose(flynn.R, flynn.R.T)
    assert np.min(flynn.R) >= -1e-12


def test_pengwah_interval_meter_attaches_at_common_path_depth():
    root, smart, interval = 0, [1, 2], [3]
    smart_edges = [(root, -1, 0.5), (-1, 1, 0.2), (-1, 2, 0.3)]
    result = insert_interval_meters(smart_edges, root, smart, interval, np.asarray([[0.5], [0.5]]))
    graph = nx.Graph()
    graph.add_weighted_edges_from(result.edges)
    assert nx.is_tree(graph)
    assert set(graph.neighbors(3)) == {-1}
    assert result.placements[0].mode == "existing_node"
