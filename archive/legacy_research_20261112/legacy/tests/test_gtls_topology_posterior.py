"""Tests for finite-candidate GTLS topology inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.graph.rooted_neighbor_joining import RootedTreeResult
from terminal_case33.pipeline.rnj_candidates import RNJCandidate, rooted_path_incidence
from terminal_case33.pipeline.gtls_topology_posterior import (
    infer_gtls_topology_posterior,
    project_gtls_to_candidate_tree,
)


def _exact_tree() -> RootedTreeResult:
    return RootedTreeResult(
        root=1,
        terminals=(10, 11, 12, 13),
        edges=(
            (1, -1, 0.04),
            (-1, 10, 0.03),
            (-1, 11, 0.05),
            (1, -2, 0.02),
            (-2, 12, 0.04),
            (-2, 13, 0.06),
        ),
        group_tolerance=0.0,
    )


def _exact_matrices(tree: RootedTreeResult) -> tuple[np.ndarray, np.ndarray]:
    _edges, incidence = rooted_path_incidence(tree)
    r_edge = np.array([0.08, 0.04, 0.05, 0.07, 0.03, 0.06])
    x_edge = 0.65 * r_edge
    return (
        (incidence * r_edge[None, :]) @ incidence.T,
        (incidence * x_edge[None, :]) @ incidence.T,
    )


def _scenarios(r_matrix: np.ndarray, x_matrix: np.ndarray) -> list[dict]:
    rng = np.random.default_rng(20260714)
    terminals = [10, 11, 12, 13]
    scenarios = []
    for scenario_index in range(3):
        p = 0.025 + rng.normal(0.0, 0.008, size=(400, 4))
        q = 0.010 + rng.normal(0.0, 0.004, size=(400, 4))
        drop = p @ r_matrix.T + q @ x_matrix.T
        drop += 0.001 * scenario_index
        voltage = np.sqrt(np.maximum(1.02**2 - drop, 0.8))
        scenarios.append(
            {
                "name": f"scenario_{scenario_index}",
                "P_terminal": pd.DataFrame(p, columns=terminals),
                "Q_terminal": pd.DataFrame(q, columns=terminals),
                "V_terminal": pd.DataFrame(voltage, columns=terminals),
                "drop_target": pd.DataFrame(drop, columns=terminals),
            }
        )
    return scenarios


def test_tree_projection_recovers_exact_additive_matrices() -> None:
    tree = _exact_tree()
    r_matrix, x_matrix = _exact_matrices(tree)
    clades = frozenset(rooted_clades(tree.edges, tree.root, tree.terminals))
    candidate = RNJCandidate(tree, "R", 0.0, clades)

    projection = project_gtls_to_candidate_tree(
        candidate,
        frozenset({"gtls"}),
        r_matrix,
        x_matrix,
    )

    assert projection.relative_matrix_error < 1e-8
    assert np.allclose(projection.r_matrix, r_matrix, atol=1e-8)
    assert np.allclose(projection.x_matrix, x_matrix, atol=1e-8)


def test_gtls_posterior_selects_clean_rooted_topology() -> None:
    tree = _exact_tree()
    r_matrix, x_matrix = _exact_matrices(tree)
    truth = frozenset(rooted_clades(tree.edges, tree.root, tree.terminals))

    posterior = infer_gtls_topology_posterior(
        _scenarios(r_matrix, x_matrix),
        root_bus=tree.root,
        pq_noise_relative_std=0.005,
        voltage_noise_relative_std=0.0002,
        validation_scenario_count=1,
        temporal_block_length=12,
    )

    assert posterior.selected.projection.candidate.clades == truth
    assert np.isclose(
        sum(score.posterior_probability for score in posterior.ranked),
        1.0,
    )
    assert posterior.effective_candidate_count >= 1.0
    assert posterior.consensus_clades == truth
    assert all(0.0 <= value <= 1.0 for value in posterior.clade_marginals.values())
