"""Tests for physics-informed sensitivity matrix constraints."""

import numpy as np
import pandas as pd
from terminal_case33.data.small_terminal_lv import build_small_terminal_lv_case
from terminal_case33.models.lin_distflow import build_reduced_sensitivity_matrices

from terminal_case33.estimation.matrix_constraints import (
    four_point_violation_summary,
    project_tree_covariance_matrix,
    sensitivity_matrix_diagnostics,
)
from terminal_case33.estimation.multiscenario import fit_projected_sensitivity


def test_tree_covariance_projection_enforces_matrix_properties() -> None:
    raw = np.array(
        [
            [0.4, 0.9, -0.2, 0.6],
            [0.1, 0.3, 0.8, -0.4],
            [0.7, -0.1, 0.2, 0.5],
            [0.2, 0.6, 0.4, 0.1],
        ]
    )
    projected = project_tree_covariance_matrix(raw, margin_ratio=1e-5)
    diagnostics = sensitivity_matrix_diagnostics(projected)
    assert np.allclose(projected, projected.T)
    assert diagnostics.minimum_entry >= -1e-10
    assert diagnostics.minimum_eigenvalue >= -1e-9
    assert diagnostics.minimum_diagonal_gap > 0.0
    assert diagnostics.minimum_distance > 0.0


def test_constrained_fit_has_ordered_diagonal() -> None:
    rng = np.random.default_rng(7)
    columns = [10, 20, 30]
    p = pd.DataFrame(rng.normal(size=(600, 3)), columns=columns)
    q = pd.DataFrame(rng.normal(size=(600, 3)), columns=columns)
    r_true = np.array([[1.8, 0.7, 0.2], [0.7, 2.0, 0.2], [0.2, 0.2, 1.5]])
    x_true = np.array([[1.2, 0.4, 0.1], [0.4, 1.4, 0.1], [0.1, 0.1, 1.0]])
    drop = pd.DataFrame(
        p.to_numpy() @ r_true.T + q.to_numpy() @ x_true.T,
        columns=columns,
    )
    scenario = {
        "name": "full_rank",
        "P_terminal": p,
        "Q_terminal": q,
        "drop_target": drop,
    }
    r_hat, x_hat, r2_score, _ = fit_projected_sensitivity([scenario])
    for matrix in [r_hat, x_hat]:
        diagnostics = sensitivity_matrix_diagnostics(matrix)
        assert diagnostics.minimum_diagonal_gap > 0.0
        assert diagnostics.minimum_eigenvalue >= -1e-9
        assert diagnostics.minimum_entry >= -1e-10
    assert np.allclose(r_hat, r_true, atol=1e-9)
    assert np.allclose(x_hat, x_true, atol=1e-9)
    assert r2_score > 1.0 - 1e-12


def test_four_point_summary_detects_nonadditive_distance() -> None:
    additive = np.array(
        [
            [0.0, 2.0, 4.0, 4.0],
            [2.0, 0.0, 4.0, 4.0],
            [4.0, 4.0, 0.0, 2.0],
            [4.0, 4.0, 2.0, 0.0],
        ]
    )
    nonadditive = additive.copy()
    nonadditive[0, 2] = nonadditive[2, 0] = 5.0
    assert four_point_violation_summary(additive)["max_four_point_violation"] == 0.0
    assert four_point_violation_summary(nonadditive)["max_four_point_violation"] > 0.0



def test_true_terminal_sensitivity_has_inverse_m_matrix_structure() -> None:
    net = build_small_terminal_lv_case()
    terminals = net.load_buses()
    r_matrix, x_matrix = build_reduced_sensitivity_matrices(
        net,
        terminals,
        terminals,
        voltage_model="squared-voltage",
    )
    for matrix in [r_matrix, x_matrix]:
        diagnostics = sensitivity_matrix_diagnostics(matrix)
        assert diagnostics.minimum_diagonal_gap > 0.0
        assert diagnostics.minimum_eigenvalue > 0.0
        assert diagnostics.precision_positive_offdiag_fraction == 0.0
        assert diagnostics.precision_minimum_row_sum >= 0.0

def test_projection_keeps_sparse_entries_nonnegative() -> None:
    """Final feasibility cleanup must not create negative off-diagonals."""

    raw = np.diag([1.0, 1e-12, 0.5])
    projected = project_tree_covariance_matrix(raw, margin_ratio=1e-3)
    diagnostics = sensitivity_matrix_diagnostics(projected)
    assert diagnostics.minimum_entry >= 0.0
    assert diagnostics.minimum_diagonal_gap >= 4.9e-4
    assert diagnostics.minimum_eigenvalue >= -1e-12
