"""Multi-scenario R/X least squares with explicit matrix constraints."""

from __future__ import annotations

import numpy as np

from rnj_wzzt.estimation.constrained_least_squares import solve_symmetric_least_squares


def align_scenarios(scenarios: list[dict]) -> list[dict]:
    """Align raw P, Q, and voltage-drop rows for positional resampling."""

    aligned = []
    for scenario in scenarios:
        index = scenario["P_terminal"].index
        if len(index) == 0 or not index.is_unique:
            raise ValueError("scenario time indices must be nonempty and unique")
        frames = {}
        for key in ("P_terminal", "Q_terminal", "drop_target"):
            frame = scenario[key]
            if not frame.index.is_unique or set(frame.index) != set(index):
                raise ValueError(f"{key} must contain the same unique time indices")
            frames[key] = frame.loc[index].copy()
        aligned.append({"name": scenario["name"], **frames})
    return aligned


def _aligned_arrays(scenarios: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Align by labels before converting to arrays; reject ambiguous observations."""

    if not scenarios:
        raise ValueError("at least one scenario is required")
    columns = scenarios[0]["P_terminal"].columns
    if len(columns) == 0 or not columns.is_unique:
        raise ValueError("terminal columns must be nonempty and unique")
    designs, targets = [], []
    for scenario in scenarios:
        index = scenario["P_terminal"].index
        if len(index) == 0 or not index.is_unique:
            raise ValueError("scenario time indices must be nonempty and unique")
        arrays = []
        for key in ("P_terminal", "Q_terminal", "drop_target"):
            frame = scenario[key]
            if not frame.columns.is_unique or set(frame.columns) != set(columns):
                raise ValueError(f"{key} must contain the same unique terminal columns")
            if not frame.index.is_unique or set(frame.index) != set(index):
                raise ValueError(f"{key} must contain the same unique time indices")
            array = frame.loc[index, columns].to_numpy(dtype=float)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{key} must contain only finite values")
            arrays.append(array)
        designs.append(np.hstack(arrays[:2]))
        targets.append(arrays[2])
    return designs, targets


def _symmetric_blocks(coefficients: np.ndarray) -> np.ndarray:
    n = coefficients.shape[1]
    return np.stack([
        0.5 * (block + block.T)
        for block in (coefficients[:n], coefficients[n:])
    ])


def _fixed_margins(blocks: np.ndarray, ratio: float) -> np.ndarray:
    """Choose margins once from nonnegative, symmetric unconstrained estimates."""

    margins = []
    for block in np.maximum(blocks, 0.0):
        positive = np.diag(block)[np.diag(block) > 0.0]
        scale = float(np.median(positive)) if positive.size else float(np.max(block))
        margins.append(ratio * scale)
    return np.asarray(margins)


def fit_projected_sensitivity(
    scenarios: list[dict],
    alpha: float = 0.0,
    constraint_mode: str = "ordered",
    diagonal_margin_ratio: float = 1e-6,
    constraint_refine_iterations: int = 500,
    *,
    diagnostics: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit Y = P R + Q X through the origin over all observed-root scenarios.

    The sole supported mode, ordered, solves a fixed-domain convex QP with
    SciPy. Y is the measured squared-voltage drop V_root^2 - V_terminal^2;
    P, Q, and Y are stacked directly, without centering or fitted intercepts.
    Matrix symmetry is encoded by upper-triangular variables; bounds enforce
    nonnegativity and linear inequalities enforce the diagonal order.
    Margins are fixed once from the initial unconstrained least-squares fit.

    constraint_refine_iterations is a positive solver iteration budget.
    The ordered constraints do not certify a shared tree or positive
    semidefiniteness.

    Returns R, X, the pooled centered R-squared, and the condition number of
    A = [P Q]. For constant targets, the finite R-squared convention is 1
    for a numerically zero residual and 0 otherwise; explained variance is
    undefined in that case. Optional diagnostics record solver status and loss.
    """

    if constraint_mode != "ordered":
        raise ValueError("only the 'ordered' sensitivity constraint mode is supported")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    if not np.isfinite(diagonal_margin_ratio) or diagonal_margin_ratio < 0.0:
        raise ValueError("diagonal_margin_ratio must be finite and nonnegative")
    if (isinstance(constraint_refine_iterations, (bool, np.bool_))
            or not isinstance(constraint_refine_iterations, (int, np.integer))
            or constraint_refine_iterations <= 0):
        raise ValueError("constraint_refine_iterations must be a positive integer")
    designs, targets = _aligned_arrays(scenarios)
    design = np.vstack(designs)
    target = np.vstack(targets)
    n = target.shape[1]
    if alpha > 0.0:
        fitted_design = np.vstack([design, np.sqrt(alpha) * np.eye(2 * n)])
        fitted_target = np.vstack([target, np.zeros((2 * n, n))])
    else:
        fitted_design, fitted_target = design, target
    coefficients = np.linalg.lstsq(fitted_design, fitted_target, rcond=None)[0]
    blocks = _symmetric_blocks(coefficients)
    margins = _fixed_margins(blocks, diagonal_margin_ratio)
    blocks, solver = solve_symmetric_least_squares(
        design, target, blocks, margins, alpha, constraint_refine_iterations,
    )
    r_matrix, x_matrix = blocks
    coefficients = np.vstack(blocks)
    residual_sum = float(np.sum((design @ coefficients - target)**2))
    total_sum = float(np.sum((target - target.mean(axis=0))**2))
    if np.any(target != target[0]):
        r2_score = 1.0 - residual_sum / total_sum
    else:
        zero_tolerance = np.finfo(float).eps * float(np.sum(target**2))
        r2_score = float(residual_sum <= zero_tolerance)
    condition = float(np.linalg.cond(design))
    if diagnostics is not None:
        diagnostics.update({
            **solver,
            "constraint_mode": constraint_mode,
            "alpha": float(alpha),
            "r_margin": float(margins[0]),
            "x_margin": float(margins[1]),
            "squared_residual_sum": residual_sum,
            "objective": 0.5 * (residual_sum + alpha * float(np.sum(coefficients**2))),
        })
    return r_matrix, x_matrix, float(r2_score), condition
