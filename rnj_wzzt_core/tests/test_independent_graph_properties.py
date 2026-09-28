"""Independent tree, resampling, and aggregation oracles.

These fixtures do not use production feeder builders, graph truth helpers,
matrix generators, or reconstructed hidden-node IDs to construct expectations.
The edge-incidence identity S = B diag(w) B.T supplies a separate oracle for
shared-path geometry. Deterministic random cases are falsification tests, not a
proof of recovery under arbitrary noise or topology distributions.
"""

from collections import defaultdict, deque
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from rnj_wzzt.graph.bootstrap import (
    _boundary_cherries,
    _moving_block_bootstrap_copy,
    _select_disjoint,
)
from rnj_wzzt.graph.rooted_hierarchy import (
    PseudoCluster,
    aggregate_rooted_scenarios,
    rooted_clades,
    rooted_tree_from_clades,
)
from rnj_wzzt.graph.rooted_neighbor_joining import (
    rooted_neighbor_joining,
    shared_paths_from_distances,
)
from rnj_wzzt.graph.sensitivity_geometry import sensitivity_geometry
from rnj_wzzt.models.lin_distflow import (
    build_reduced_sensitivity_matrices,
    impedance_distance_from_reduced_R,
    impedance_distance_from_reduced_X,
    ohm_to_pu,
)
from rnj_wzzt.models.network import TerminalizedNetwork


def _edge_incidence_oracle(edges, terminals):
    """Compute descendants on an independently specified directed tree."""
    children = defaultdict(list)
    for parent, child, _ in edges:
        children[parent].append(child)
    terminal_set = set(terminals)

    def below(node):
        result = {node} if node in terminal_set else set()
        for child in children[node]:
            result.update(below(child))
        return result

    supports = [below(child) for _, child, _ in edges]
    incidence = np.array(
        [[float(terminal in support) for support in supports] for terminal in terminals]
    )
    weights = np.array([weight for _, _, weight in edges])
    shared = (incidence * weights) @ incidence.T
    clades = {frozenset(support) for support in supports if 1 < len(support) < len(terminals)}
    return shared, clades


def _inspect_result(result):
    """Orient returned edges independently and check the tree contract."""
    adjacency = defaultdict(list)
    for left, right, weight in result.edges:
        assert left != right
        assert np.isfinite(weight) and weight >= 0.0
        adjacency[left].append((right, weight))
        adjacency[right].append((left, weight))
    queue = deque([result.root])
    seen = {result.root}
    directed = []
    while queue:
        parent = queue.popleft()
        for child, weight in adjacency[parent]:
            if child in seen:
                continue
            seen.add(child)
            directed.append((parent, child, weight))
            queue.append(child)
    assert len(seen) == len(adjacency)
    assert len(result.edges) == len(seen) - 1
    assert set(result.terminals) <= seen
    assert all(len(adjacency[node]) == 1 for node in result.terminals)
    return _edge_incidence_oracle(directed, result.terminals)


def _fixture_tree(kind, count=12, seed=0):
    """Make reduced directed trees by partitioning leaf labels, with dyadic weights."""
    rng = np.random.default_rng(seed)
    terminals = list(range(101, 101 + count))
    edges = []
    next_hidden = 10000

    def weight():
        return float(rng.integers(2, 25)) / 8.0

    def append_group(parent, members):
        nonlocal next_hidden
        if len(members) == 1:
            edges.append((parent, members[0], weight()))
            return
        hidden = next_hidden
        next_hidden += 1
        edges.append((parent, hidden, weight()))
        partition(hidden, members)

    def partition(parent, members):
        if kind == "star":
            groups = [[terminal] for terminal in members]
        elif kind == "comb":
            groups = [members[:1], members[1:]]
        elif kind == "random":
            shuffled = rng.permutation(members).tolist()
            fanout = int(rng.integers(2, min(5, len(members)) + 1))
            groups = [list(group) for group in np.array_split(shuffled, fanout)]
        else:
            fanout = min(3 if kind == "multifurcating" else 2, len(members))
            groups = [list(group) for group in np.array_split(members, fanout)]
        for group in groups:
            append_group(parent, group)

    if kind == "stem":
        edges.append((0, next_hidden, 1.25))
        next_hidden += 1
        partition(10000, terminals)
    else:
        partition(0, terminals)
    return edges, terminals


@pytest.mark.parametrize("kind", ["star", "balanced", "comb", "multifurcating", "stem"])
@pytest.mark.parametrize("count", [2, 3, 12, 32])
def test_exact_rnj_recovers_independent_tree_geometry_and_clades(kind, count):
    edges, terminals = _fixture_tree(kind, count)
    shared, expected_clades = _edge_incidence_oracle(edges, terminals)
    result = rooted_neighbor_joining(shared, shared.diagonal(), terminals, 0)
    reconstructed, actual_clades = _inspect_result(result)
    np.testing.assert_allclose(reconstructed, shared, rtol=0, atol=1e-12)
    assert actual_clades == expected_clades
    assert rooted_clades(result.edges, 0, terminals) == expected_clades


@pytest.mark.parametrize("seed", range(20))
def test_random_multifurcating_trees_with_independent_oracle(seed):
    edges, terminals = _fixture_tree("random", 5 + seed * 2, seed)
    shared, expected_clades = _edge_incidence_oracle(edges, terminals)
    result = rooted_neighbor_joining(shared, shared.diagonal(), terminals, 0)
    actual, clades = _inspect_result(result)
    np.testing.assert_allclose(actual, shared, rtol=0, atol=1e-12)
    assert clades == expected_clades


@pytest.mark.parametrize("kind", ["balanced", "comb", "multifurcating", "stem"])
@pytest.mark.parametrize("scale", [1e-6, 0.5, 1.0, 1e6])
def test_rnj_scale_permutation_and_positive_relabeling_equivariance(kind, scale):
    edges, terminals = _fixture_tree(kind, 15, 7)
    shared, expected_clades = _edge_incidence_oracle(edges, terminals)
    order = np.random.default_rng(781).permutation(len(terminals))
    relabel = {terminal: 5000 + 11 * index for index, terminal in enumerate(terminals)}
    labels = [relabel[terminals[index]] for index in order]
    scaled = scale * shared[np.ix_(order, order)]
    tolerance = scale * 0.01
    result = rooted_neighbor_joining(scaled, scaled.diagonal(), labels, 6000, tolerance)
    actual, clades = _inspect_result(result)
    np.testing.assert_allclose(actual, scaled, rtol=5e-15, atol=1e-14 * scale)
    assert clades == {frozenset(relabel[terminal] for terminal in clade) for clade in expected_clades}
    assert result.root == 6000
    assert result.terminals == tuple(labels)


def test_distance_adapter_against_hand_calculated_three_terminal_values():
    depths = np.array([7.0, 9.0, 5.0])
    distances = np.array([[0.0, 12.0, 12.0], [12.0, 0.0, 14.0], [12.0, 14.0, 0.0]])
    expected = np.array([[7.0, 2.0, 0.0], [2.0, 9.0, 0.0], [0.0, 0.0, 5.0]])
    np.testing.assert_array_equal(shared_paths_from_distances(distances, depths), expected)
    np.testing.assert_array_equal(impedance_distance_from_reduced_R(expected), distances)
    np.testing.assert_array_equal(impedance_distance_from_reduced_X(expected), distances)


@pytest.mark.parametrize("internal_weight, expected_cherry", [(0.0, False), (1e-12, False), (1e-8, True)])
def test_zero_and_short_internal_edges_expose_absolute_contraction_resolution(internal_weight, expected_cherry):
    # An unrelated deeper cherry forces this edge to be internal-to-internal.
    # The implementation contracts such lengths <= 1e-10 even with tolerance 0.
    edges = [(0, 10, 1.0), (10, 11, internal_weight), (11, 1, 1.0), (11, 2, 1.5),
             (10, 3, 2.0), (0, 12, 3.0), (12, 4, 1.0), (12, 5, 1.0)]
    shared, _ = _edge_incidence_oracle(edges, [1, 2, 3, 4, 5])
    result = rooted_neighbor_joining(shared, shared.diagonal(), [1, 2, 3, 4, 5], 0)
    actual, clades = _inspect_result(result)
    assert (frozenset({1, 2}) in clades) is expected_cherry
    np.testing.assert_allclose(actual, shared, rtol=0, atol=2e-12)


def test_unobserved_degree_two_nodes_are_only_identifiable_after_reduction():
    # The two consecutive hidden edges can be redistributed without changing data.
    first = [(0, 10, 0.5), (10, 11, 1.5), (11, 1, 1.0), (11, 2, 2.0)]
    second = [(0, 20, 2.0), (20, 1, 1.0), (20, 2, 2.0)]
    shared, _ = _edge_incidence_oracle(first, [1, 2])
    alternative, _ = _edge_incidence_oracle(second, [1, 2])
    np.testing.assert_array_equal(shared, alternative)
    result = rooted_neighbor_joining(shared, shared.diagonal(), [1, 2], 0)
    reconstructed, _ = _inspect_result(result)
    np.testing.assert_array_equal(reconstructed, shared)
    assert len(result.edges) == 3


@pytest.mark.parametrize("gap,tolerance,expected", [(0.25, 0.0, True), (0.25, 0.249, True),
                                                    (0.25, 0.25, False), (0.25, 0.251, False)])
def test_group_tolerance_resolution_has_an_explicit_topology_boundary(gap, tolerance, expected):
    edges = [(0, 10, 1.0), (10, 11, gap), (11, 1, 1.0), (11, 2, 1.0), (10, 3, 1.5)]
    shared, _ = _edge_incidence_oracle(edges, [1, 2, 3])
    result = rooted_neighbor_joining(shared, shared.diagonal(), [1, 2, 3], 0, tolerance)
    _, clades = _inspect_result(result)
    assert (frozenset({1, 2}) in clades) is expected


@pytest.mark.parametrize("mode", ["R", "X", "RX_equal_normalized", "RX_75R_25X", "RX_25R_75X"])
def test_channel_geometry_agrees_with_independent_edge_incidence(mode):
    edges, terminals = _fixture_tree("multifurcating", 12, 14)
    r, expected = _edge_incidence_oracle(edges, terminals)
    x_edges = [(parent, child, 0.5 + 0.125 * index) for index, (parent, child, _) in enumerate(edges)]
    x, _ = _edge_incidence_oracle(x_edges, terminals)
    geometry = sensitivity_geometry(r, x, mode)
    # Independent normalization counts unordered pairs exactly once.
    pairs = list(combinations(range(len(terminals)), 2))
    r_scale = np.mean([r[i, i] + r[j, j] - 2 * r[i, j] for i, j in pairs])
    x_scale = np.mean([x[i, i] + x[j, j] - 2 * x[i, j] for i, j in pairs])
    weights = {"R": (1.0, 0.0), "X": (0.0, 1.0),
               "RX_equal_normalized": (1 / r_scale, 1 / x_scale),
               "RX_75R_25X": (0.75 / r_scale, 0.25 / x_scale),
               "RX_25R_75X": (0.25 / r_scale, 0.75 / x_scale)}[mode]
    combined_edges = [(u, v, weights[0] * w + weights[1] * x_edge[2])
                      for (u, v, w), x_edge in zip(edges, x_edges, strict=True)]
    expected_shared, _ = _edge_incidence_oracle(combined_edges, terminals)
    np.testing.assert_allclose(geometry.shared_paths, expected_shared, rtol=1e-14)
    result = rooted_neighbor_joining(geometry.shared_paths, geometry.root_depths, terminals, 0)
    assert _inspect_result(result)[1] == expected


@pytest.mark.parametrize("r_scale,x_scale", [(1e-4, 1e4), (20.0, 0.25), (2.0, 2.0)])
def test_normalized_rx_geometry_is_invariant_to_independent_channel_units(r_scale, x_scale):
    r = np.array([[7.0, 2.0, 0.0], [2.0, 9.0, 0.0], [0.0, 0.0, 5.0]])
    x = np.array([[3.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 4.0]])
    original = sensitivity_geometry(r, x, "RX_75R_25X")
    changed = sensitivity_geometry(r_scale * r, x_scale * x, "RX_75R_25X")
    np.testing.assert_allclose(changed.shared_paths, original.shared_paths, rtol=1e-14)
    np.testing.assert_allclose(changed.distance, original.distance, rtol=1e-14)


def _small_network(base_kv=10.0, base_mva=2.0, reverse=False):
    r_edges = [(0, 10, 2.0), (10, 1, 5.0), (10, 2, 7.0), (0, 3, 5.0)]
    x_values = [1.0, 2.0, 4.0, 3.0]
    records = []
    for (parent, child, resistance), reactance in zip(r_edges, x_values, strict=True):
        records.append({"from_bus": child if reverse else parent,
                        "to_bus": parent if reverse else child,
                        "r_ohm": resistance, "x_ohm": reactance,
                        "branch_type": "line", "is_candidate": True, "is_true_closed": True})
    records.append({"from_bus": 1, "to_bus": 3, "r_ohm": 0.01, "x_ohm": 0.02,
                    "branch_type": "tie", "is_candidate": True, "is_true_closed": False})
    buses = pd.DataFrame({"bus_id": [0, 10, 1, 2, 3],
                          "bus_type": ["root", "hidden_internal", *["observed_terminal"] * 3],
                          "is_observed": [True, False, True, True, True],
                          "has_load": [False, False, True, True, True]})
    return TerminalizedNetwork(buses, pd.DataFrame(records), 0, base_kv, base_mva, {}, {}, {})


@pytest.mark.parametrize("voltage_model,factor", [("magnitude", 1.0), ("squared-voltage", 2.0)])
@pytest.mark.parametrize("convention", ["load_positive", "net_injection"])
@pytest.mark.parametrize("reverse", [False, True])
def test_lindistflow_against_hand_matrix_with_open_tie_and_reversed_edges(voltage_model, factor, convention, reverse):
    net = _small_network(reverse=reverse)
    r, x = build_reduced_sensitivity_matrices(net, [3, 1, 2], convention=convention,
                                             voltage_model=voltage_model)
    scale = factor / 50.0  # Z_base = 10^2 / 2 = 50 ohm.
    expected_r = scale * np.array([[5.0, 0.0, 0.0], [0.0, 7.0, 2.0], [0.0, 2.0, 9.0]])
    expected_x = scale * np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 1.0], [0.0, 1.0, 5.0]])
    np.testing.assert_allclose(r, expected_r, rtol=1e-15)
    np.testing.assert_allclose(x, expected_x, rtol=1e-15)
    assert net.get_leaf_buses() == [1, 2, 3]
    assert net.hidden_buses() == [10]
    assert net.observed_buses() == [0, 1, 2, 3]


@pytest.mark.parametrize("voltage_model,factor", [("magnitude", 1.0), ("squared-voltage", 2.0)])
@pytest.mark.parametrize("reverse", [False, True])
def test_lindistflow_rectangular_observation_injection_paths(voltage_model, factor, reverse):
    """A hidden-node injection shares the stem, but never a terminal's service edge."""
    net = _small_network(reverse=reverse)
    r, x = build_reduced_sensitivity_matrices(net, [3, 1], [2, 10, 3], voltage_model=voltage_model)
    scale = factor / 50.0
    np.testing.assert_allclose(r, scale * np.array([[0.0, 0.0, 5.0], [2.0, 2.0, 0.0]]), rtol=1e-15)
    np.testing.assert_allclose(x, scale * np.array([[0.0, 0.0, 3.0], [1.0, 1.0, 0.0]]), rtol=1e-15)


@pytest.mark.parametrize("base_kv,base_mva,expected", [(10.0, 2.0, 0.1), (0.4, 0.1, 3.125), (20.0, 8.0, 0.1)])
def test_impedance_base_conversion_has_independent_units(base_kv, base_mva, expected):
    assert ohm_to_pu(_small_network(base_kv, base_mva), 5.0) == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["star", "balanced", "comb", "multifurcating", "stem"])
@pytest.mark.parametrize("include_stem", [False, True])
def test_clade_constructor_roundtrip_uses_independent_descendants(kind, include_stem):
    edges, terminals = _fixture_tree(kind, 12, 3)
    _, expected_clades = _edge_incidence_oracle(edges, terminals)
    result = rooted_tree_from_clades(expected_clades, list(reversed(terminals)), 0, include_stem)
    shared, actual_clades = _inspect_result(result)
    assert actual_clades == expected_clades
    assert rooted_clades(result.edges, 0, result.terminals) == expected_clades
    assert float(np.min(shared)) == float(include_stem)


@pytest.mark.parametrize("clades", [
    {frozenset({1, 2}), frozenset({2, 3})},
    {frozenset({1, 9})},
    {frozenset({9})},
    {frozenset({1, 2, 3, 9})},
    {frozenset({1, 2, 3, 4, 9})},
])
def test_clade_constructor_rejects_crossing_or_foreign_supports(clades):
    with pytest.raises(ValueError):
        rooted_tree_from_clades(clades, [1, 2, 3, 4], 0)


def _bootstrap_scenario(count, offset=0):
    rows = np.arange(count) + offset
    index = pd.Index(np.arange(count) * 3 + 100, name="time")
    p = pd.DataFrame({1: rows, 2: rows + 1000}, index=index)
    return {"name": f"scenario_{offset}", "P_terminal": p, "Q_terminal": 2 * p + 7,
            "drop_target": -3 * p + 11}


class _FixedStarts:
    def __init__(self, starts):
        self.starts = np.array(starts)

    def integers(self, low, high, size):
        assert low == 0
        assert len(self.starts) == size
        assert all(0 <= value < high for value in self.starts)
        return self.starts


@pytest.mark.parametrize("count,length,starts,expected", [
    (7, 3, [5, 2, 6], [5, 6, 0, 2, 3, 4, 6]),
    (5, 5, [3], [3, 4, 0, 1, 2]),
    (5, 100, [3], [3, 4, 0, 1, 2]),
    (4, 1, [3, 3, 0, 2], [3, 3, 0, 2]),
    (2, 2, [1], [1, 0]),
])
def test_circular_bootstrap_indices_match_hand_sequences(count, length, starts, expected):
    original = _bootstrap_scenario(count)
    replicate = _moving_block_bootstrap_copy([original], _FixedStarts(starts), length)[0]
    assert replicate["P_terminal"][1].tolist() == expected
    assert replicate["P_terminal"].index.tolist() == list(range(count))
    pd.testing.assert_frame_equal(replicate["Q_terminal"], 2 * replicate["P_terminal"] + 7)
    pd.testing.assert_frame_equal(replicate["drop_target"], -3 * replicate["P_terminal"] + 11)
    assert replicate["name"] == original["name"]


@pytest.mark.parametrize("seed", [0, 1, 19, 71])
@pytest.mark.parametrize("length", [1, 4, 30])
def test_bootstrap_preserves_scenario_boundaries_alignment_and_input_ownership(seed, length):
    scenarios = [_bootstrap_scenario(19, 0), _bootstrap_scenario(11, 10000)]
    snapshots = [{key: value.copy(deep=True) for key, value in scenario.items() if key != "name"}
                 for scenario in scenarios]
    first = _moving_block_bootstrap_copy(scenarios, np.random.default_rng(seed), length)
    second = _moving_block_bootstrap_copy(scenarios, np.random.default_rng(seed), length)
    for original, snapshot, sample, repeated in zip(scenarios, snapshots, first, second, strict=True):
        assert len(sample["P_terminal"]) == len(original["P_terminal"])
        assert set(sample["P_terminal"][1]) <= set(original["P_terminal"][1])
        for key in ("P_terminal", "Q_terminal", "drop_target"):
            pd.testing.assert_frame_equal(sample[key], repeated[key])
        pd.testing.assert_frame_equal(sample["Q_terminal"], 2 * sample["P_terminal"] + 7)
        pd.testing.assert_frame_equal(sample["drop_target"], -3 * sample["P_terminal"] + 11)
        sample["P_terminal"].iloc[0, 0] = -999
        for key, expected in snapshot.items():
            pd.testing.assert_frame_equal(original[key], expected)


def test_bootstrap_rejects_single_row_scenario():
    with pytest.raises(ValueError, match="at least two rows"):
        _moving_block_bootstrap_copy([_bootstrap_scenario(1)], np.random.default_rng(0), 4)


def test_bootstrap_boundary_cherries_are_minimal_clades_including_multifurcations():
    cherry = frozenset({1, 2})
    trifurcation = frozenset({4, 5, 6})
    clades = {cherry, trifurcation, frozenset({1, 2, 3}), frozenset({1, 2, 3, 4, 5, 6})}
    assert _boundary_cherries(clades) == {cherry, trifurcation}
    assert _boundary_cherries(set()) == set()


@pytest.mark.parametrize("maximum,expected", [(0, []), (1, [{1, 2, 3}]), (2, [{1, 2, 3}, {4, 5}]),
                                               (10, [{1, 2, 3}, {4, 5}, {6, 7}])])
def test_disjoint_selection_respects_confidence_size_and_lexical_ties(maximum, expected):
    confidence = {frozenset({1, 2}): 0.9, frozenset({1, 2, 3}): 0.9,
                  frozenset({3, 4}): 0.8, frozenset({4, 5}): 0.8, frozenset({6, 7}): 0.8}
    assert _select_disjoint(set(confidence), confidence, maximum) == [frozenset(item) for item in expected]


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("order", [[1, 2, 3], [3, 2, 1]])
def test_cherry_deembedding_matches_independently_calculated_boundary_voltage(weight, order):
    # Root -- stem -- {cherry boundary, terminal 3}; external power contributes
    # to the stem drop but must disappear from the cherry's differential drop.
    p = np.array([[0.1, 0.2, 0.3], [0.4, -0.1, 0.5], [-0.2, 0.3, -0.1]])
    q = np.array([[0.02, 0.03, 0.01], [-0.01, 0.02, 0.03], [0.03, -0.02, 0.01]])
    r = np.array([[0.10, 0.06, 0.02], [0.06, 0.13, 0.02], [0.02, 0.02, 0.09]])
    x = np.array([[0.06, 0.04, 0.01], [0.04, 0.09, 0.01], [0.01, 0.01, 0.07]])
    root_sq = np.array([1.0, 1.02, 0.98])
    boundary_sq = root_sq - 0.02 * p.sum(axis=1) - 0.01 * q.sum(axis=1)
    boundary_sq -= 0.04 * p[:, :2].sum(axis=1) + 0.03 * q[:, :2].sum(axis=1)
    terminal_sq = root_sq[:, None] - p @ r.T - q @ x.T
    index = pd.Index([7, 13, 19], name="timestamp")
    p_frame = pd.DataFrame(p, columns=[1, 2, 3], index=index)
    q_frame = pd.DataFrame(q, columns=[1, 2, 3], index=index)
    voltage = pd.DataFrame(np.sqrt(terminal_sq), columns=[1, 2, 3], index=index)
    scenario = {"name": "independent", "P_terminal": p_frame, "Q_terminal": q_frame,
                "V_terminal": voltage, "root_voltage": pd.Series(np.sqrt(root_sq), index=index)}
    cluster = PseudoCluster(900001, frozenset({1, 2}), 1.0, (frozenset({1, 2}),), ((1, 2),))
    positions = [terminal - 1 for terminal in order]
    aggregated, mapping = aggregate_rooted_scenarios(
        [scenario], order, r[np.ix_(positions, positions)], x[np.ix_(positions, positions)],
        [cluster], "deembedded_vsq", weight)
    actual = aggregated[0]
    mean_sq = terminal_sq[:, :2].mean(axis=1)
    expected_sq = (1 - weight) * mean_sq + weight * boundary_sq
    np.testing.assert_allclose(actual["V_terminal"][900001] ** 2, expected_sq, rtol=1e-14)
    np.testing.assert_allclose(actual["P_terminal"][900001], p[:, :2].sum(axis=1), rtol=1e-14)
    np.testing.assert_allclose(actual["Q_terminal"][900001], q[:, :2].sum(axis=1), rtol=1e-14)
    singleton = next(pseudo for pseudo, members in mapping.items() if members == frozenset({3}))
    np.testing.assert_array_equal(actual["V_terminal"][singleton], voltage[3])
    np.testing.assert_allclose(actual["drop_target"][900001], root_sq - expected_sq, atol=1e-15)
    assert actual["V_terminal"].index.equals(index)
    assert not np.allclose(mean_sq, boundary_sq)


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("order", [[2, 7, 11, 20], [20, 11, 7, 2]])
def test_deembedding_matches_scalar_formula_for_asymmetric_estimates(weight, order):
    # Imperfect estimates need not be symmetric. Distinct row coefficients
    # expose a missing transpose; the last terminal is outside the clade.
    labels = [2, 7, 11, 20]
    r = np.array([[0.7, 0.1, 0.4, 0.9], [-0.2, 0.8, 0.2, 0.3],
                  [0.3, 0.05, 0.9, 0.4], [0.1, 0.2, 0.3, 0.8]])
    x = np.array([[0.4, 0.05, 0.1, 0.1], [0.08, 0.5, 0.2, 0.3],
                  [0.12, 0.09, 0.6, 0.1], [0.1, 0.2, 0.1, 0.4]])
    p = np.array([[0.1, 0.2, 0.3, 7.0], [-0.2, 0.3, 0.1, -4.0]])
    q = np.array([[0.02, 0.01, -0.03, -2.0], [-0.01, 0.03, 0.02, 5.0]])
    v = np.array([[0.95, 0.96, 0.94, 0.99], [0.97, 0.95, 0.93, 0.98]])
    scenario = {
        "name": "asymmetric",
        "P_terminal": pd.DataFrame(p, columns=labels),
        "Q_terminal": pd.DataFrame(q, columns=labels),
        "V_terminal": pd.DataFrame(v, columns=labels),
        "root_voltage": pd.Series([1.01, 1.02]),
    }
    cluster = PseudoCluster(900000, frozenset(labels[:3]), 1.0, (), ())
    positions = [labels.index(label) for label in order]
    actual, _ = aggregate_rooted_scenarios(
        [scenario], order, r[np.ix_(positions, positions)], x[np.ix_(positions, positions)],
        [cluster], "deembedded_vsq", weight,
    )
    # Upper-triangle 20th percentiles are 0.14 for R and 0.07 for X.
    recovered = [[v[t, j] ** 2 + sum(
        p[t, i] * max(r[j, i] - 0.14, 0.0)
        + q[t, i] * max(x[j, i] - 0.07, 0.0)
        for i in range(3)
    ) for j in range(3)] for t in range(2)]
    mean_sq = np.mean(v[:, :3] ** 2, axis=1)
    expected = mean_sq + weight * (np.median(recovered, axis=1) - mean_sq)
    np.testing.assert_allclose(actual[0]["V_terminal"][900000] ** 2, expected, atol=1e-14)


@pytest.mark.parametrize("kind", ["star", "balanced", "comb", "multifurcating"])
@pytest.mark.parametrize("count", [64, 128])
def test_large_exact_trees_stress_merging_and_hidden_depth(kind, count):
    edges, terminals = _fixture_tree(kind, count, 19)
    shared, expected_clades = _edge_incidence_oracle(edges, terminals)
    result = rooted_neighbor_joining(shared, shared.diagonal(), terminals, 0)
    actual, clades = _inspect_result(result)
    np.testing.assert_allclose(actual, shared, rtol=0, atol=1e-12)
    assert clades == expected_clades


@pytest.mark.parametrize("kind", ["balanced", "comb", "multifurcating"])
@pytest.mark.parametrize("seed", range(10))
def test_small_bounded_score_noise_preserves_resolved_clades(kind, seed):
    # All internal edges >= 0.25: a finite stress grid, not a recovery theorem.
    edges, terminals = _fixture_tree(kind, 16, seed)
    shared, expected_clades = _edge_incidence_oracle(edges, terminals)
    rng = np.random.default_rng(10000 + seed)
    perturbation = rng.uniform(-0.01, 0.01, size=shared.shape)
    noisy = shared + 0.5 * (perturbation + perturbation.T)
    result = rooted_neighbor_joining(noisy, noisy.diagonal(), terminals, 0, 0.02)
    _, clades = _inspect_result(result)
    assert clades == expected_clades


def test_pendant_depth_changes_leave_clades_unchanged_at_fixed_tolerance():
    edges, terminals = _fixture_tree("comb", 16, 3)
    shared, expected_clades = _edge_incidence_oracle(edges, terminals)
    changed_depths = shared.diagonal() + np.linspace(0.0, 20.0, len(terminals))
    expected = shared.copy()
    np.fill_diagonal(expected, changed_depths)
    result = rooted_neighbor_joining(shared, changed_depths, terminals, 0, 0.01)
    actual, clades = _inspect_result(result)
    assert clades == expected_clades
    np.testing.assert_allclose(actual, expected, rtol=1e-15)


def test_common_stem_is_visible_in_root_depths_but_not_terminal_distances_or_clades():
    edges, terminals = _fixture_tree("balanced", 8, 2)
    shared, expected_clades = _edge_incidence_oracle(edges, terminals)
    shifted = shared + 4.0
    # Pairwise distances alone cannot reveal an added common root path.
    original_distance = impedance_distance_from_reduced_R(shared)
    shifted_distance = impedance_distance_from_reduced_R(shifted)
    np.testing.assert_array_equal(original_distance, shifted_distance)
    result = rooted_neighbor_joining(shifted, shifted.diagonal(), terminals, 0)
    actual, clades = _inspect_result(result)
    np.testing.assert_array_equal(actual, shifted)
    assert clades == expected_clades


def test_rnj_symmetrization_and_read_only_input_contract():
    shared = np.array([[7.0, 2.0, 0.0], [2.0, 9.0, 0.0], [0.0, 0.0, 5.0]])
    skew = np.array([[0.0, 0.25, -0.5], [-0.25, 0.0, 0.75], [0.5, -0.75, 0.0]])
    observed = shared + skew
    before = observed.copy()
    depths = shared.diagonal().copy()
    observed.setflags(write=False)
    depths.setflags(write=False)
    result = rooted_neighbor_joining(observed, depths, [1, 2, 3], 0)
    actual, clades = _inspect_result(result)
    np.testing.assert_array_equal(actual, shared)
    np.testing.assert_array_equal(observed, before)
    assert clades == {frozenset({1, 2})}


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("bad_field", ["shared", "depths"])
def test_rnj_rejects_nonfinite_data_before_producing_a_tree(bad_value, bad_field):
    shared = np.array([[2.0, 1.0], [1.0, 3.0]])
    depths = shared.diagonal().copy()
    if bad_field == "shared":
        shared[0, 1] = bad_value
    else:
        depths[0] = bad_value
    with pytest.raises(ValueError, match="finite"):
        rooted_neighbor_joining(shared, depths, [1, 2], 0)


@pytest.mark.parametrize("shared,depths,tolerance", [
    (np.eye(3), np.ones(2), 0.0),
    (np.eye(2), np.ones((2, 1)), 0.0),
    (np.eye(2), np.ones(3), 0.0),
    (np.eye(2), np.ones(2), -0.01),
])
def test_rnj_rejects_invalid_shapes_or_negative_tolerance(shared, depths, tolerance):
    with pytest.raises(ValueError):
        rooted_neighbor_joining(shared, depths, [1, 2], 0, tolerance)


@pytest.mark.parametrize("labels,root", [
    ([-1, -2, -3], -4), ([-1, -2, -3], 0), ([1, 2, 3], -1),
    ([-1, 20, -3], -2), ([1, 2, 3], -20),
])
def test_rnj_negative_labels_never_collide_with_allocated_hidden_nodes(labels, root):
    shared = np.array([[4.0, 2.0, 1.0], [2.0, 5.0, 1.0], [1.0, 1.0, 3.0]])
    result = rooted_neighbor_joining(shared, shared.diagonal(), labels, root)
    actual, clades = _inspect_result(result)
    np.testing.assert_array_equal(actual, shared)
    assert clades == {frozenset(labels[:2])}
    assert result.root == root
    assert result.terminals == tuple(labels)


@pytest.mark.parametrize("labels,root", [
    ([-1, -2, -3], -4), ([-1, -2, -3], 0), ([1, 2, 3], -1),
    ([-1, 20, -3], -2), ([1, 2, 3], -20),
])
@pytest.mark.parametrize("include_stem", [False, True])
def test_clade_constructor_negative_labels_keep_terminals_as_leaves(labels, root, include_stem):
    clades = {frozenset(labels[:2])}
    result = rooted_tree_from_clades(clades, labels, root, include_stem)
    shared, actual_clades = _inspect_result(result)
    expected = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    if include_stem:
        expected += 1.0
    np.testing.assert_array_equal(shared, expected)
    assert actual_clades == clades


@pytest.mark.parametrize("labels,root", [([1, 1], 0), ([1, 2], 1), ([-1, -1], 0), ([-1, 2], -1)])
@pytest.mark.parametrize("constructor", ["rnj", "clades"])
def test_tree_constructors_reject_repeated_or_root_overlapping_terminal_labels(labels, root, constructor):
    with pytest.raises(ValueError, match="unique|root"):
        if constructor == "rnj":
            rooted_neighbor_joining(np.eye(2), np.ones(2), labels, root)
        else:
            rooted_tree_from_clades(set(), labels, root)


@pytest.mark.parametrize("tolerance", [np.nan, np.inf, -np.inf])
def test_rnj_rejects_nonfinite_group_tolerance(tolerance):
    with pytest.raises(ValueError, match="finite"):
        rooted_neighbor_joining(np.eye(2), np.ones(2), [1, 2], 0, tolerance)


@pytest.mark.parametrize("count,exact,raw_tolerance", [(11, True, 0.96), (12, False, 1.04), (32, False, 2.64)])
def test_default_rx75_tolerance_loses_unit_caterpillar_clades_as_depth_grows(count, exact, raw_tolerance):
    """Closed-form depth failure: every edge is one, but median depth grows.

    Terminal i has depth i+1 except the last, which shares its predecessor's
    depth. Terminal i and j share min(i,j) root-path edges. The true internal
    clades are proper suffixes; no production tree builder supplies the oracle.
    """
    labels = list(range(101, 101 + count))
    depths = np.r_[np.arange(1, count), count - 1].astype(float)
    shared = np.minimum.outer(np.arange(count), np.arange(count)).astype(float)
    np.fill_diagonal(shared, depths)
    expected = {frozenset(labels[start:]) for start in range(1, count - 1)}
    geometry = sensitivity_geometry(shared, shared, "RX_75R_25X")
    pair_distances = [depths[i] + depths[j] - 2 * min(i, j)
                      for i, j in combinations(range(count), 2)]
    scale = 1.0 / np.mean(pair_distances)
    np.testing.assert_allclose(geometry.shared_paths, scale * shared, rtol=1e-14)
    tolerance = 0.16 * float(np.median(geometry.root_depths))
    assert tolerance / scale == pytest.approx(raw_tolerance)
    result = rooted_neighbor_joining(
        geometry.shared_paths, geometry.root_depths, labels, 0, tolerance)
    _, actual = _inspect_result(result)
    assert (actual == expected) is exact
    if not exact:
        # The deepest binary cherry has a one-unit separation from its next
        # neighbor; tau >= one unit merges that neighbor into the same family.
        assert frozenset(labels[-2:]) not in actual
    resolved = rooted_neighbor_joining(
        geometry.shared_paths, geometry.root_depths, labels, 0, 0.01 * scale)
    assert _inspect_result(resolved)[1] == expected
