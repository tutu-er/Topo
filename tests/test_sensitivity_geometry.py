import numpy as np
import pytest

from terminal_case33.graph.rooted_neighbor_joining import shared_paths_from_distances
from terminal_case33.graph.sensitivity_geometry import sensitivity_geometry


@pytest.mark.parametrize(
    "mode",
    ["R", "X", "RX_equal_normalized", "RX_75R_25X", "RX_25R_75X"],
)
def test_direct_shared_path_score_matches_distance_round_trip(mode):
    r_matrix = np.array(
        [[1.4, 0.8, 0.2], [0.8, 1.6, 0.2], [0.2, 0.2, 1.1]]
    )
    x_matrix = np.array(
        [[1.0, 0.5, 0.1], [0.5, 1.3, 0.1], [0.1, 0.1, 0.9]]
    )

    geometry = sensitivity_geometry(r_matrix, x_matrix, mode)
    reconstructed = shared_paths_from_distances(geometry.distance, geometry.root_depths)

    np.testing.assert_allclose(geometry.shared_paths, reconstructed, rtol=0.0, atol=2e-15)


def test_direct_score_is_weighted_reduced_matrix_entry():
    r_matrix = np.array([[1.0, 0.4], [0.4, 1.5]])
    x_matrix = np.array([[0.8, 0.3], [0.3, 1.2]])
    geometry = sensitivity_geometry(r_matrix, x_matrix, "RX_75R_25X")

    expected = geometry.r_coefficient * r_matrix + geometry.x_coefficient * x_matrix
    np.testing.assert_allclose(geometry.shared_paths, expected, rtol=0.0, atol=1e-15)


def test_sensitivity_geometry_rejects_unknown_mode():
    matrix = np.eye(2)
    with pytest.raises(ValueError, match="unknown distance mode"):
        sensitivity_geometry(matrix, matrix, "unknown")


def test_parent_geometry_exports_research_implementation():
    from research_experiments.rnj.sensitivity_geometry import (
        sensitivity_geometry as research_geometry,
    )

    assert sensitivity_geometry is research_geometry


def test_parent_distance_adapter_exports_research_implementation():
    from research_experiments.rnj.graph_adapters import (
        shared_paths_from_distances as research_adapter,
    )

    assert shared_paths_from_distances is research_adapter
