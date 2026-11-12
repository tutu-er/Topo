"""Distance utilities for reduced sensitivity matrices."""

from __future__ import annotations

import numpy as np


def additive_distance(matrix: np.ndarray) -> np.ndarray:
    """Return ``d_ij = M_ii + M_jj - 2 M_ij``."""

    diag = np.diag(matrix)
    return diag[:, None] + diag[None, :] - 2.0 * matrix

