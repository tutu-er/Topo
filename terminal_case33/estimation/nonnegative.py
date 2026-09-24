"""Deterministic nonnegative ridge solver."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def solve_nonnegative_ridge(
    design: np.ndarray,
    target: np.ndarray,
    ridge: float = 1e-10,
    max_iterations: int = 5000,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, int]:
    """Solve a convex nonnegative ridge problem by augmented NNLS.

    Column normalization improves conditioning. Appending ``sqrt(ridge) * I``
    preserves the objective used by the former projected-gradient solver while
    Lawson-Hanson active-set NNLS avoids thousands of Python iterations.

    The second return value is one because SciPy exposes the converged solution
    but not its internal active-set iteration count. ``tolerance`` remains in
    the stable API; convergence tolerance is managed by the NNLS implementation.
    """

    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    if design.ndim != 2 or target.ndim != 1 or len(target) != len(design):
        raise ValueError("design must be 2-D and target must be a matching vector")

    column_scale = np.maximum(np.linalg.norm(design, axis=0), 1e-12)
    normalized = design / column_scale
    if ridge > 0.0:
        variable_count = normalized.shape[1]
        augmented_design = np.vstack(
            [normalized, np.sqrt(ridge) * np.eye(variable_count)]
        )
        augmented_target = np.concatenate([target, np.zeros(variable_count)])
    else:
        augmented_design = normalized
        augmented_target = target
    normalized_solution, _residual_norm = nnls(
        augmented_design,
        augmented_target,
        maxiter=max_iterations,
    )
    return normalized_solution / column_scale, 1