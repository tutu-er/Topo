"""Tests for the complete constrained RNJ baseline API."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from terminal_case33.estimation.baseline import fit_complete_rnj_baseline


def test_complete_baseline_recovers_exact_sensitivity_and_residual() -> None:
    rng = np.random.default_rng(17)
    terminals = [10, 11, 12]
    sample_count = 240
    p = pd.DataFrame(rng.normal(0.0, 0.25, (sample_count, 3)), columns=terminals)
    q = pd.DataFrame(rng.normal(0.0, 0.12, (sample_count, 3)), columns=terminals)
    r_true = np.asarray(
        [
            [0.90, 0.35, 0.20],
            [0.35, 1.00, 0.25],
            [0.20, 0.25, 1.10],
        ]
    )
    x_true = 0.45 * r_true
    intercept = np.asarray([0.01, 0.02, 0.03])
    target = p.to_numpy() @ r_true.T + q.to_numpy() @ x_true.T + intercept
    scenario = {
        "name": "exact",
        "P_terminal": p,
        "Q_terminal": q,
        "drop_target": pd.DataFrame(target, columns=terminals),
    }

    result = fit_complete_rnj_baseline(
        [scenario],
        root_bus=1,
        preprocessing="raw",
        alpha=0.0,
        tolerance_factor=1e-8,
    )

    assert np.allclose(result.r_matrix.to_numpy(), r_true, atol=1e-8)
    assert np.allclose(result.x_matrix.to_numpy(), x_true, atol=1e-8)
    assert result.residual_rmse < 1e-10
    assert result.objective_value < 1e-16
    assert result.r2_score > 1.0 - 1e-12
    assert np.allclose(result.scenario_intercepts["exact"].to_numpy(), intercept, atol=1e-10)
    assert np.max(np.abs(result.delta_v_residuals["exact"].to_numpy())) < 1e-9

    graph = nx.Graph()
    graph.add_edges_from((left, right) for left, right, _ in result.tree.edges)
    assert nx.is_tree(graph)
    assert set(terminals) <= set(graph)
