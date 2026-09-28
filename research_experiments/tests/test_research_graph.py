"""Independent coverage of graph adapters and alternative research geometries.

The migrated fixtures preserve independent incidence/descendant oracles; the
core retains its own RNJ, RX75 and bootstrap recovery tests.
"""

from collections import defaultdict, deque
from itertools import combinations

import numpy as np
import pytest

from research_experiments.rnj.graph_adapters import (
    rooted_tree_from_clades,
    shared_paths_from_distances,
)
from research_experiments.rnj.sensitivity_geometry import sensitivity_geometry
from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
from rnj_wzzt.graph.rooted_neighbor_joining import rooted_neighbor_joining
from rnj_wzzt.graph.sensitivity_geometry import sensitivity_geometry as core_geometry
from rnj_wzzt.models.lin_distflow import (
    impedance_distance_from_reduced_R,
    impedance_distance_from_reduced_X,
)


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


def test_distance_adapter_against_hand_calculated_three_terminal_values():
    depths = np.array([7.0, 9.0, 5.0])
    distances = np.array([[0.0, 12.0, 12.0], [12.0, 0.0, 14.0], [12.0, 14.0, 0.0]])
    expected = np.array([[7.0, 2.0, 0.0], [2.0, 9.0, 0.0], [0.0, 0.0, 5.0]])
    np.testing.assert_array_equal(shared_paths_from_distances(distances, depths), expected)
    np.testing.assert_array_equal(impedance_distance_from_reduced_R(expected), distances)
    np.testing.assert_array_equal(impedance_distance_from_reduced_X(expected), distances)


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


@pytest.mark.parametrize("labels,root", [([1, 1], 0), ([1, 2], 1), ([-1, -1], 0), ([-1, 2], -1)])
def test_clade_constructor_rejects_repeated_or_root_overlapping_terminal_labels(labels, root):
    with pytest.raises(ValueError, match="unique|root"):
        rooted_tree_from_clades(set(), labels, root)


@pytest.mark.parametrize("kind", ["tree", "asymmetric", "imperfect", "zero", "common_stem", "singleton"])
@pytest.mark.parametrize("r_scale,x_scale", [(1.0, 1.0), (1e-4, 1e4), (20.0, 0.25)])
def test_research_and_core_rx75_inputs_are_identical(kind, r_scale, x_scale):
    r = np.array([[7.0, 2.0, 0.0], [2.0, 9.0, 0.0], [0.0, 0.0, 5.0]])
    x = np.array([[3.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 4.0]])
    if kind == "asymmetric":
        r += np.array([[0.0, 0.25, -0.5], [-0.25, 0.0, 0.75], [0.5, -0.75, 0.0]])
    elif kind == "imperfect":
        r[0, 0], r[0, 1], r[1, 0] = -2.0, 20.0, 20.0
        x[1, 2], x[2, 1] = -3.0, -3.0
    elif kind == "zero":
        r, x = np.zeros((3, 3)), np.zeros((3, 3))
    elif kind == "common_stem":
        r, x = np.full((3, 3), 2.0), np.full((3, 3), 0.5)
    elif kind == "singleton":
        r, x = np.array([[2.0]]), np.array([[0.5]])
    r, x = r_scale * r, x_scale * x
    r.setflags(write=False)
    x.setflags(write=False)
    mainline = core_geometry(r, x)
    mainline_explicit = core_geometry(r, x, "RX_75R_25X")
    historical = sensitivity_geometry(r, x, "RX_75R_25X")
    for field in ("shared_paths", "root_depths"):
        np.testing.assert_array_equal(getattr(mainline, field), getattr(historical, field))
        np.testing.assert_array_equal(getattr(mainline, field), getattr(mainline_explicit, field))
