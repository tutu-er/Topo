"""Focused tests for uncertainty RNJ, joint pseudo voltage, and cross-fitting."""

import networkx as nx
import numpy as np
import pandas as pd

from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.graph.rooted_neighbor_joining import shared_paths_from_distances
from terminal_case33.graph.uncertainty_rnj import (
    rooted_neighbor_joining_with_uncertainty,
)
from terminal_case33.pipeline.pseudo_parent_voltage import (
    fit_joint_pseudo_parent_voltage,
)
from terminal_case33.pipeline.three_stage_crossfit import three_stage_splits


def test_uncertainty_rnj_recovers_tree_under_small_distance_error() -> None:
    root = 0
    terminals = [1, 2, 3, 4]
    true_edges = [
        (0, 10, 0.30),
        (0, 11, 0.25),
        (10, 1, 0.20),
        (10, 2, 0.22),
        (11, 3, 0.18),
        (11, 4, 0.21),
    ]
    graph = nx.Graph()
    graph.add_weighted_edges_from(true_edges)
    distance = np.asarray(
        [
            [nx.shortest_path_length(graph, left, right, weight="weight") for right in terminals]
            for left in terminals
        ]
    )
    depths = np.asarray(
        [nx.shortest_path_length(graph, root, node, weight="weight") for node in terminals]
    )
    shared = shared_paths_from_distances(distance, depths)
    rng = np.random.default_rng(17)
    shared_samples = []
    depth_samples = []
    for _ in range(40):
        noise = rng.normal(0.0, 0.002, shared.shape)
        noise = 0.5 * (noise + noise.T)
        sample_depth = depths + rng.normal(0.0, 0.001, depths.shape)
        sample = np.maximum(shared + noise, 0.0)
        np.fill_diagonal(sample, sample_depth)
        shared_samples.append(sample)
        depth_samples.append(sample_depth)

    result = rooted_neighbor_joining_with_uncertainty(
        np.asarray(shared_samples),
        np.asarray(depth_samples),
        terminals,
        root,
        group_tolerance=0.01,
        confidence_level=0.90,
    )

    assert rooted_clades(result.tree.edges, root, terminals) == rooted_clades(
        true_edges, root, terminals
    )
    assert len(result.decisions) == 2


def test_joint_pseudo_parent_recovers_free_common_voltage_mode() -> None:
    terminals = [1, 2, 3]
    count = 160
    rng = np.random.default_rng(29)
    p = rng.uniform(0.002, 0.018, (count, 3))
    q = rng.uniform(0.001, 0.009, (count, 3))
    r_true = np.array(
        [[0.90, 0.35, 0.00], [0.35, 0.85, 0.00], [0.00, 0.00, 0.65]]
    )
    x_true = 0.70 * r_true
    root = 1.02 + 0.001 * np.sin(np.linspace(0.0, 6.0 * np.pi, count))
    common_drop = 0.010 + 0.003 * np.sin(np.linspace(0.0, 3.0 * np.pi, count))
    target = common_drop[:, None] + p @ r_true.T + q @ x_true.T
    voltage = np.sqrt(root[:, None] ** 2 - target)
    scenario = {
        "name": "joint_mode",
        "P_terminal": pd.DataFrame(p, columns=terminals),
        "Q_terminal": pd.DataFrame(q, columns=terminals),
        "V_terminal": pd.DataFrame(voltage, columns=terminals),
        "root_voltage": pd.Series(root),
    }
    fit = fit_joint_pseudo_parent_voltage(
        [scenario],
        terminals,
        900000,
        r_true + 0.20 * np.ones_like(r_true),
        x_true + 0.10 * np.ones_like(x_true),
        ridge=0.0,
    )

    expected_parent_vsq = root**2 - common_drop
    assert np.sqrt(np.mean((fit.pseudo_voltage_squared[0] - expected_parent_vsq) ** 2)) < 1e-8
    assert np.max(np.abs(fit.r_local - r_true)) < 1e-8
    assert np.max(np.abs(fit.x_local - x_true)) < 1e-8
    assert fit.residual_nrmse < 1e-8


def test_three_stage_splits_rotate_each_day_without_leakage() -> None:
    splits = three_stage_splits(3)

    assert len(splits) == 3
    for split in splits:
        discovery = set(split.discovery_indices)
        estimation = set(split.estimation_indices)
        validation = set(split.validation_indices)
        assert discovery | estimation | validation == {0, 1, 2}
        assert discovery.isdisjoint(estimation)
        assert discovery.isdisjoint(validation)
        assert estimation.isdisjoint(validation)
    assert {split.discovery_indices[0] for split in splits} == {0, 1, 2}
