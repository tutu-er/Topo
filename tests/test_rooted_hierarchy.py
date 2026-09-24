import numpy as np
import pandas as pd

from terminal_case33.graph.rooted_hierarchy import (
    PseudoCluster,
    aggregate_rooted_scenarios,
    build_local_cluster_scenarios,
    expand_pseudo_clades_with_local_reidentification,
    reidentify_local_clades_from_shared_paths,
    rooted_clades,
    rooted_tree_from_clades,
)


def test_deembedded_voltage_recovers_linear_boundary_voltage():
    terminals = [1, 2]
    r_matrix = np.array([[1.2, 1.0], [1.0, 1.3]])
    x_matrix = np.array([[0.6, 0.5], [0.5, 0.7]])
    p = pd.DataFrame({1: [0.1, 0.2], 2: [0.2, 0.1]})
    q = pd.DataFrame({1: [0.02, 0.03], 2: [0.03, 0.02]})
    root_sq = np.array([1.04, 1.04])
    drop = p.to_numpy() @ r_matrix.T + q.to_numpy() @ x_matrix.T
    voltage = pd.DataFrame(np.sqrt(root_sq[:, None] - drop), columns=terminals)
    scenario = {
        "name": "linear",
        "P_terminal": p,
        "Q_terminal": q,
        "V_terminal": voltage,
        "root_voltage": pd.Series(np.sqrt(root_sq)),
    }
    cluster = PseudoCluster(900000, frozenset(terminals), 1.0, tuple(), tuple())
    aggregated, _ = aggregate_rooted_scenarios(
        [scenario], terminals, r_matrix, x_matrix, [cluster], "deembedded_vsq"
    )
    expected = root_sq - p.sum(axis=1).to_numpy() - 0.5 * q.sum(axis=1).to_numpy()
    recovered = aggregated[0]["V_terminal"][900000].pow(2).to_numpy()
    assert np.allclose(recovered, expected)


def test_deembedding_ignores_outside_clade_injection_error():
    terminals = [1, 2, 3]
    r_matrix = np.array(
        [
            [1.2, 1.0, 0.4],
            [1.0, 1.3, 0.2],
            [0.3, 0.3, 0.9],
        ]
    )
    x_matrix = np.zeros_like(r_matrix)
    p = pd.DataFrame({1: [0.1, 0.1], 2: [0.2, 0.2], 3: [0.0, 10.0]})
    q = pd.DataFrame(0.0, index=p.index, columns=terminals)
    voltage = pd.DataFrame({1: [0.98, 0.98], 2: [0.97, 0.97], 3: [0.99, 0.99]})
    scenario = {
        "name": "outside_injection",
        "P_terminal": p,
        "Q_terminal": q,
        "V_terminal": voltage,
        "root_voltage": pd.Series([1.02, 1.02]),
    }
    cluster = PseudoCluster(900000, frozenset({1, 2}), 1.0, tuple(), tuple())

    aggregated, _ = aggregate_rooted_scenarios(
        [scenario],
        terminals,
        r_matrix,
        x_matrix,
        [cluster],
        "deembedded_vsq",
    )

    recovered = aggregated[0]["V_terminal"][900000].pow(2).to_numpy()
    assert np.allclose(recovered[0], recovered[1])


def test_local_cluster_scenario_uses_pseudo_voltage_as_root():
    terminals = [1, 2]
    p = pd.DataFrame({1: [0.1, 0.2], 2: [0.2, 0.1]})
    q = 0.2 * p
    voltage = pd.DataFrame({1: [0.98, 0.97], 2: [0.975, 0.965]})
    scenario = {
        "name": "local",
        "P_terminal": p,
        "Q_terminal": q,
        "V_terminal": voltage,
        "root_voltage": pd.Series([1.02, 1.02]),
    }
    pseudo_voltage = pd.DataFrame({900000: [1.0, 0.995]})
    pseudo = {"V_terminal": pseudo_voltage}
    cluster = PseudoCluster(900000, frozenset(terminals), 1.0, tuple(), tuple())

    local = build_local_cluster_scenarios([scenario], [pseudo], cluster)[0]

    expected_drop = pseudo_voltage[900000].pow(2).to_numpy()[:, None] - voltage.pow(2).to_numpy()
    assert list(local["P_terminal"].columns) == terminals
    assert np.allclose(local["root_voltage"], pseudo_voltage[900000])
    assert np.allclose(local["drop_target"], expected_drop)


def test_local_reidentification_replaces_frozen_subclades():
    cluster = PseudoCluster(
        900000,
        frozenset({1, 2, 3}),
        1.0,
        (frozenset({1, 2}),),
        tuple(),
    )
    pseudo_members = {900000: cluster.members, 910000: frozenset({4})}
    pseudo_edges = ((0, 900000, 1.0), (0, 910000, 1.0))
    local_clades = {900000: {frozenset({2, 3})}}

    expanded = expand_pseudo_clades_with_local_reidentification(
        pseudo_edges,
        0,
        pseudo_members,
        [cluster],
        local_clades,
    )

    assert frozenset({1, 2, 3}) in expanded
    assert frozenset({2, 3}) in expanded
    assert frozenset({1, 2}) not in expanded


def test_shared_path_local_reroot_recovers_nested_clade():
    shared_paths = np.array(
        [
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    depths = np.array([2.0, 2.0, 1.0])
    cluster = PseudoCluster(900000, frozenset({1, 2, 3}), 1.0, tuple(), tuple())

    local = reidentify_local_clades_from_shared_paths(
        shared_paths,
        depths,
        [1, 2, 3],
        [cluster],
        tolerance_factor=0.1,
        boundary_quantile=0.0,
    )

    assert local[900000] == {frozenset({1, 2})}


def test_laminar_clades_reconstruct_the_same_reduced_rooted_tree():
    terminals = [1, 2, 3, 4, 5]
    expected = {
        frozenset({1, 2}),
        frozenset({1, 2, 3}),
        frozenset({4, 5}),
    }

    tree = rooted_tree_from_clades(expected, terminals, root=0)

    assert tree.root == 0
    assert set(tree.terminals) == set(terminals)
    assert rooted_clades(tree.edges, tree.root, terminals) == expected


def test_crossing_clades_are_rejected():
    with np.testing.assert_raises(ValueError):
        rooted_tree_from_clades(
            {frozenset({1, 2}), frozenset({2, 3})},
            [1, 2, 3, 4],
            root=0,
        )


def test_clade_reconstruction_can_preserve_common_root_stem():
    terminals = [1, 2, 3, 4]
    expected = {frozenset({1, 2}), frozenset({3, 4})}

    tree = rooted_tree_from_clades(
        expected,
        terminals,
        root=0,
        include_hidden_root_stem=True,
    )

    root_neighbors = [
        right if left == tree.root else left
        for left, right, _weight in tree.edges
        if tree.root in {left, right}
    ]
    assert len(root_neighbors) == 1
    assert root_neighbors[0] not in terminals
    assert rooted_clades(tree.edges, tree.root, terminals) == expected
