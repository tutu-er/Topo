"""Multi-mode R/X geometry and legacy distance diagnostics for research.

For a physically consistent reduced sensitivity matrix, an off-diagonal entry
is already the root-to-LCA shared-path length. RNJ therefore consumes a
weighted R/X shared-path score directly; terminal distances are retained only
as diagnostics and for methods that genuinely require an additive distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rnj_wzzt.models.lin_distflow import impedance_distance_from_reduced_R
from rnj_wzzt.graph.sensitivity_geometry import (
    _positive_mean,
    _shared_path_geometry,
    _validated_matrix,
)


_NORMALIZED_RX_WEIGHTS = {
    "RX_equal_normalized": (1.0, 1.0),
    "RX_75R_25X": (0.75, 0.25),
    "RX_25R_75X": (0.25, 0.75),
}


@dataclass(frozen=True)
class SensitivityGeometry:
    """Distance diagnostics and the direct shared-path score used by RNJ."""

    d_r: np.ndarray
    d_x: np.ndarray
    distance: np.ndarray
    root_depths: np.ndarray
    shared_paths: np.ndarray
    r_coefficient: float
    x_coefficient: float


def _mode_coefficients(mode: str, d_r: np.ndarray, d_x: np.ndarray) -> tuple[float, float]:
    if mode == "R":
        return 1.0, 0.0
    if mode == "X":
        return 0.0, 1.0
    if mode not in _NORMALIZED_RX_WEIGHTS:
        raise ValueError(f"unknown distance mode {mode!r}; choose R, X, or a normalized RX mode")
    weight_r, weight_x = _NORMALIZED_RX_WEIGHTS[mode]
    return weight_r / _positive_mean(d_r), weight_x / _positive_mean(d_x)


def distance_candidates(r_matrix: np.ndarray, x_matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Return legacy distance candidates without RNJ input validation or repair.

    Preserve raw R/X distances, including asymmetric and nonfinite inputs.
    Normalize each distance before applying its weight, as the historical API
    did, while sharing the scales and mode definitions with RNJ geometry.
    """

    d_r = impedance_distance_from_reduced_R(r_matrix)
    d_x = impedance_distance_from_reduced_R(x_matrix)
    n_r, n_x = d_r / _positive_mean(d_r), d_x / _positive_mean(d_x)
    return {
        "R": d_r,
        "X": d_x,
        **{
            mode: weight_r * n_r + weight_x * n_x
            for mode, (weight_r, weight_x) in _NORMALIZED_RX_WEIGHTS.items()
        },
    }


def sensitivity_geometry(
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    mode: str,
) -> SensitivityGeometry:
    """Build RNJ inputs without the redundant distance-to-score round trip.

    With coefficients ``c_R`` and ``c_X``, RNJ ranks
    ``S_ij = c_R R_ij + c_X X_ij``. The matching root depth is ``S_ii``.
    This is algebraically identical to constructing a distance and cancelling
    it again, but makes the actual ordering statistic explicit.
    """

    r_value = _validated_matrix(r_matrix, "r_matrix")
    x_value = _validated_matrix(x_matrix, "x_matrix")
    if r_value.shape != x_value.shape:
        raise ValueError("r_matrix and x_matrix must have the same shape")

    d_r = impedance_distance_from_reduced_R(r_value)
    d_x = impedance_distance_from_reduced_R(x_value)
    coefficient_r, coefficient_x = _mode_coefficients(mode, d_r, d_x)

    distance = coefficient_r * d_r + coefficient_x * d_x
    rooted = _shared_path_geometry(r_value, x_value, coefficient_r, coefficient_x)

    return SensitivityGeometry(
        d_r=d_r,
        d_x=d_x,
        distance=distance,
        root_depths=rooted.root_depths,
        shared_paths=rooted.shared_paths,
        r_coefficient=float(coefficient_r),
        x_coefficient=float(coefficient_x),
    )
