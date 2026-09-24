"""Tests for stable NJ/RG peripheral-clade proposals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from terminal_case33.pipeline.peripheral_edge_proposals import (
    propose_peripheral_clusters,
)


def _scenarios() -> list[dict]:
    terminals = [1, 2, 3, 4, 5]
    r_matrix = np.array(
        [
            [0.70, 0.40, 0.00, 0.00, 0.00],
            [0.40, 0.75, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.45, 0.20, 0.00],
            [0.00, 0.00, 0.20, 0.50, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.45],
        ]
    )
    x_matrix = 0.75 * r_matrix
    rng = np.random.default_rng(71)
    scenarios = []
    for index in range(4):
        p = rng.normal(0.08, 0.018, size=(96, len(terminals)))
        q = rng.normal(0.025, 0.009, size=(96, len(terminals)))
        drop = p @ r_matrix.T + q @ x_matrix.T
        root = pd.Series(np.full(len(p), 1.04), name="root_voltage")
        voltage = np.sqrt(np.maximum(root.to_numpy()[:, None] ** 2 - drop, 0.5))
        columns = pd.Index(terminals)
        scenarios.append(
            {
                "name": f"scenario_{index}",
                "P_terminal": pd.DataFrame(p, columns=columns),
                "Q_terminal": pd.DataFrame(q, columns=columns),
                "V_terminal": pd.DataFrame(voltage, columns=columns),
                "root_voltage": root,
                "drop_target": pd.DataFrame(drop, columns=columns),
            }
        )
    return scenarios


def test_nj_proposals_find_peripheral_clades_without_freezing_internals() -> None:
    result = propose_peripheral_clusters(
        _scenarios(),
        root_bus=0,
        bootstrap_replicates=4,
        maximum_cluster_size=3,
        minimum_support=0.75,
        minimum_boundary_margin=0.02,
        minimum_edge_length_ratio=0.01,
        block_fraction=0.75,
        pq_extra_noise_rel=0.0001,
        voltage_extra_noise_rel=0.00001,
        seed=9,
    )

    nj_clades = {cluster.members for cluster in result.clusters_by_mode["nj_stable"]}
    assert frozenset({1, 2}) in nj_clades
    assert frozenset({3, 4}) in nj_clades
    assert all(
        not cluster.frozen_clades and not cluster.frozen_sibling_pairs
        for clusters in result.clusters_by_mode.values()
        for cluster in clusters
    )
    assert result.evidence[frozenset({1, 2})].nj_support >= 0.75


def test_nj_proposals_are_seed_reproducible() -> None:
    arguments = {
        "root_bus": 0,
        "bootstrap_replicates": 3,
        "minimum_support": 2.0 / 3.0,
        "minimum_boundary_margin": 0.02,
        "minimum_edge_length_ratio": 0.01,
        "seed": 12,
    }
    left = propose_peripheral_clusters(_scenarios(), **arguments)
    right = propose_peripheral_clusters(_scenarios(), **arguments)
    assert left.clusters_by_mode == right.clusters_by_mode
    assert left.evidence == right.evidence
