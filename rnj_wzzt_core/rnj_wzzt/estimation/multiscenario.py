"""Multi-scenario R/X least squares with explicit matrix constraints."""

from __future__ import annotations

import numpy as np

from rnj_wzzt.estimation.constrained_least_squares import (
    make_feasible,
    solve_symmetric_least_squares,
)
from rnj_wzzt.estimation.matrix_constraints import project_tree_covariance_matrix
from rnj_wzzt.estimation.preprocessing import apply_preprocessing_recipe
from rnj_wzzt.graph.sensitivity_geometry import distance_candidates  # Historical API re-export.


def preprocess_scenarios(scenarios: list[dict], recipe: dict) -> list[dict]:
    """Align observations, then apply one temporal recipe to P, Q, and voltage drop."""

    fitted = []
    for scenario in scenarios:
        index = scenario["P_terminal"].index
        if len(index) == 0 or not index.is_unique:
            raise ValueError("scenario time indices must be nonempty and unique")
        transformed = []
        for key in ("P_terminal", "Q_terminal", "drop_target"):
            frame = scenario[key]
            if not frame.index.is_unique or set(frame.index) != set(index):
                raise ValueError(f"{key} must contain the same unique time indices")
            transformed.append(apply_preprocessing_recipe(frame.loc[index], recipe, len(index)))
        p, q, drop = transformed
        index = p.index.intersection(q.index).intersection(drop.index)
        fitted.append({
            "name": scenario["name"], "P_terminal": p.loc[index],
            "Q_terminal": q.loc[index], "drop_target": drop.loc[index],
        })
    return fitted


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


def _fit_tree_covariance_compatibility(
    design: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    alpha: float,
    margin_ratio: float,
    iterations: int,
) -> np.ndarray:
    """Retain optional PSD feasibility refinement, without claiming optimality."""

    def repair(coefficients: np.ndarray) -> np.ndarray:
        return np.vstack([
            project_tree_covariance_matrix(block, margin_ratio=margin_ratio)
            for block in _symmetric_blocks(coefficients)
        ])

    def loss(coefficients: np.ndarray) -> float:
        residual = design @ coefficients - target
        return 0.5 * float(np.sum(residual**2) + alpha * np.sum(coefficients**2))

    coefficients = repair(np.vstack(initial))
    step = 1.0 / max(float(np.linalg.norm(design, ord=2))**2 + alpha, 1e-15)
    value = loss(coefficients)
    for _ in range(iterations):
        gradient = design.T @ (design @ coefficients - target) + alpha * coefficients
        proposal = repair(coefficients - step * gradient)
        candidate_value = loss(proposal)
        if candidate_value > value:
            step *= 0.5
            continue
        change = np.linalg.norm(proposal - coefficients)
        reference = max(np.linalg.norm(coefficients), 1e-15)
        coefficients, value = proposal, candidate_value
        if change <= 1e-8 * reference:
            break
    return coefficients.reshape(2, target.shape[1], target.shape[1])


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

    The RNJ default, ordered, and basic solve a fixed-domain convex QP with
    SciPy. Y is the measured squared-voltage drop V_root^2 - V_terminal^2;
    P, Q, and Y are stacked directly, without centering or fitted intercepts.
    Matrix symmetry is encoded by upper-triangular variables; bounds enforce
    nonnegativity and linear inequalities enforce the optional diagonal order.
    Margins are fixed once from the initial unconstrained least-squares fit.

    constraint_refine_iterations is the solver iteration budget; zero requests
    only a feasible initializer. tree_covariance retains a PSD feasibility
    heuristic, not an optimal SDP solver. No mode certifies a shared tree.

    Returns R, X, the pooled centered R-squared, and the condition number of
    A = [P Q]. For constant targets, the finite R-squared convention is 1
    for a numerically zero residual and 0 otherwise; explained variance is
    undefined in that case. Optional diagnostics record solver status and loss.
    """

    if constraint_mode not in {"basic", "ordered", "tree_covariance"}:
        raise ValueError("unknown sensitivity constraint mode")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    if not np.isfinite(diagonal_margin_ratio) or diagonal_margin_ratio < 0.0:
        raise ValueError("diagonal_margin_ratio must be finite and nonnegative")
    if (isinstance(constraint_refine_iterations, (bool, np.bool_))
            or not isinstance(constraint_refine_iterations, (int, np.integer))
            or constraint_refine_iterations < 0):
        raise ValueError("constraint_refine_iterations must be a nonnegative integer")
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
    margins = _fixed_margins(blocks, diagonal_margin_ratio) if constraint_mode == "ordered" else None
    solver = {"method": "feasible_initialization_only", "success": False, "iterations": 0}
    if constraint_mode == "tree_covariance":
        blocks = _fit_tree_covariance_compatibility(
            design, target, blocks, alpha, diagonal_margin_ratio, constraint_refine_iterations,
        )
        solver = {"method": "PSD_feasibility_heuristic", "success": False}
    elif constraint_refine_iterations:
        blocks, solver = solve_symmetric_least_squares(
            design, target, blocks, margins, alpha, constraint_refine_iterations,
        )
    else:
        blocks = make_feasible(blocks, margins)
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
            "r_margin": float(margins[0]) if constraint_mode == "ordered" else None,
            "x_margin": float(margins[1]) if constraint_mode == "ordered" else None,
            "squared_residual_sum": residual_sum,
            "objective": 0.5 * (residual_sum + alpha * float(np.sum(coefficients**2))),
        })
    return r_matrix, x_matrix, float(r2_score), condition
