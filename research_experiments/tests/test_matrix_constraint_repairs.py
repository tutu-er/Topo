"""Numerical checks for the legacy matrix feasibility repairs."""

import numpy as np
import pytest

from terminal_case33.estimation.matrix_constraints import (
    _project_diagonal_order,
    project_ordered_sensitivity_matrix,
    project_tree_covariance_matrix,
)


REPAIRS = [project_ordered_sensitivity_matrix, project_tree_covariance_matrix]


def test_single_half_space_uses_symmetric_frobenius_projection():
    # Only the first row violates a bound. The off-diagonal contributes twice
    # to the squared Frobenius distance, giving changes +2/3 and -1/3.
    matrix = np.array([[0.0, 1.0], [1.0, 3.0]])
    repaired = _project_diagonal_order(matrix, margin=0.0, sweeps=1)
    expected = np.array([[2.0 / 3.0, 2.0 / 3.0], [2.0 / 3.0, 3.0]])
    np.testing.assert_allclose(repaired, expected, atol=1e-15, rtol=0.0)
    np.testing.assert_array_equal(matrix, [[0.0, 1.0], [1.0, 3.0]])


@pytest.mark.parametrize("repair", REPAIRS)
def test_tiny_positive_diagonal_does_not_erase_nonzero_matrix(repair):
    matrix = np.array([[1e-16, 1.0], [1.0, 1e-16]])
    repaired = repair(matrix)
    assert np.max(np.abs(repaired)) > 0.1
    np.testing.assert_allclose(repaired, repaired.T, atol=1e-14, rtol=0.0)
    assert np.min(repaired) >= 0.0
    assert repaired[0, 0] >= repaired[0, 1] - 1e-14
    assert repaired[1, 1] >= repaired[0, 1] - 1e-14
    if repair is project_tree_covariance_matrix:
        assert np.linalg.eigvalsh(repaired).min() >= -1e-12


@pytest.mark.parametrize("repair", REPAIRS)
def test_repair_preserves_input_scale_margin_and_returns_feasible_matrix(repair):
    matrix = np.array([[2.0, 4.0, -2.0], [1.0, -3.0, 5.0], [3.0, 6.0, 0.5]])
    repaired = repair(matrix, margin_ratio=0.01)
    # The existing scale convention is the median positive input diagonal.
    margin = 0.01 * np.median([2.0, 0.5])
    mask = ~np.eye(3, dtype=bool)
    gaps = np.diag(repaired)[:, None] - repaired
    np.testing.assert_allclose(repaired, repaired.T, atol=1e-14, rtol=0.0)
    assert repaired.min() >= 0.0
    assert gaps[mask].min() >= margin - 1e-12
    if repair is project_tree_covariance_matrix:
        assert np.linalg.eigvalsh(repaired).min() >= -1e-12


@pytest.mark.parametrize("repair", REPAIRS)
def test_numerically_zero_matrix_retains_existing_zero_behavior(repair):
    np.testing.assert_array_equal(repair(np.full((2, 2), 1e-16)), np.zeros((2, 2)))


@pytest.mark.parametrize("repair", REPAIRS)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"margin_ratio": np.nan},
        {"max_iterations": -1},
        {"max_iterations": 1.5},
        {"tolerance": 0.0},
    ],
)
def test_invalid_parameters_are_rejected(repair, kwargs):
    with pytest.raises(ValueError):
        repair(np.eye(2), **kwargs)


@pytest.mark.parametrize("repair", REPAIRS)
def test_nonfinite_matrix_is_rejected(repair):
    with pytest.raises(ValueError, match="finite"):
        repair(np.array([[1.0, np.nan], [np.nan, 1.0]]))
