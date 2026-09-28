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
    target = p.to_numpy() @ r_true.T + q.to_numpy() @ x_true.T
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
    assert not hasattr(result, "scenario_intercepts")
    assert np.max(np.abs(result.delta_v_residuals["exact"].to_numpy())) < 1e-9

    graph = nx.Graph()
    graph.add_edges_from((left, right) for left, right, _ in result.tree.edges)
    assert nx.is_tree(graph)
    assert set(terminals) <= set(graph)


def test_complete_baseline_keeps_unexplained_terminal_offsets_in_residuals() -> None:
    rng = np.random.default_rng(28)
    terminals = [10, 11]
    p = rng.normal(size=(80, 2))
    q = rng.normal(size=(80, 2))
    p -= p.mean(axis=0)
    q -= q.mean(axis=0)
    offset = np.asarray([0.03, -0.02])
    target = p + 0.5 * q + offset
    scenario = {"name": "offset", "P_terminal": pd.DataFrame(p, columns=terminals),
                "Q_terminal": pd.DataFrame(q, columns=terminals),
                "drop_target": pd.DataFrame(target, columns=terminals)}

    result = fit_complete_rnj_baseline([scenario], root_bus=1)

    assert result.preprocessing == "raw"
    residual = result.delta_v_residuals["offset"].to_numpy()
    np.testing.assert_allclose(residual.mean(axis=0), offset, atol=1e-12)
    assert result.residual_rmse >= np.sqrt(np.mean(offset**2)) - 1e-12
    np.testing.assert_allclose(result.objective_value, 0.5 * np.sum(residual**2))
