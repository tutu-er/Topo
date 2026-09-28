"""RX75 shared-path geometry used by the RNJ + MILP mainline.

Distances are computed only to normalize the R and X channels. Alternative
channel weights and distance diagnostics belong to research_experiments.rnj.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rnj_wzzt.models.lin_distflow import impedance_distance_from_reduced_R


@dataclass(frozen=True)
class SensitivityGeometry:
    """Root depths and shared-path scores consumed by rooted neighbor joining."""

    root_depths: np.ndarray
    shared_paths: np.ndarray


def _validated_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return 0.5 * (matrix + matrix.T)



def _positive_mean(distance: np.ndarray) -> float:
    positive = distance[distance > 0.0]
    return max(float(np.mean(positive)) if positive.size else 1.0, 1e-12)



def _shared_path_geometry(
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    r_coefficient: float,
    x_coefficient: float,
) -> SensitivityGeometry:
    """Combine validated matrices and enforce RNJ's shared-path bounds."""

    score = r_coefficient * r_matrix + x_coefficient * x_matrix
    root_depths = np.maximum(np.diag(score), 0.0)
    score = np.maximum(score, 0.0)
    score = np.minimum(score, np.minimum(root_depths[:, None], root_depths[None, :]))
    np.fill_diagonal(score, root_depths)
    return SensitivityGeometry(root_depths=root_depths, shared_paths=score)


def sensitivity_geometry(
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    mode: str = "RX_75R_25X",
) -> SensitivityGeometry:
    """Return the mainline RX75 score without a distance-to-score round trip.

    Keep the RX75 mode argument for existing callers. Alternative modes are
    available in research_experiments.rnj.sensitivity_geometry.
    """

    if mode != "RX_75R_25X":
        raise ValueError(
            "core geometry supports only RX_75R_25X; use "
            "research_experiments.rnj.sensitivity_geometry for other modes"
        )
    r_value = _validated_matrix(r_matrix, "r_matrix")
    x_value = _validated_matrix(x_matrix, "x_matrix")
    if r_value.shape != x_value.shape:
        raise ValueError("r_matrix and x_matrix must have the same shape")
    d_r = impedance_distance_from_reduced_R(r_value)
    d_x = impedance_distance_from_reduced_R(x_value)
    return _shared_path_geometry(
        r_value, x_value, 0.75 / _positive_mean(d_r), 0.25 / _positive_mean(d_x)
    )
