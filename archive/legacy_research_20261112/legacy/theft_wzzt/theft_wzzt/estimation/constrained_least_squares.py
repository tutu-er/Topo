"""Numerically scaled convex QP for symmetric, nonnegative R/X least squares."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import Bounds, LinearConstraint, minimize


def make_feasible(blocks: np.ndarray, margins: np.ndarray | None) -> np.ndarray:
    """Build a feasible initializer, or clean roundoff after a feasibility check."""

    result = np.maximum(blocks, 0.0)
    n = result.shape[1]
    if margins is not None and n > 1:
        for block, margin in zip(result, margins):
            off = block.copy()
            np.fill_diagonal(off, -np.inf)
            np.fill_diagonal(block, np.maximum(np.diag(block), off.max(axis=1) + margin))
    return result


def solve_symmetric_least_squares(
    design: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    margins: np.ndarray | None,
    alpha: float,
    max_iterations: int,
    *,
    output_transform: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Solve a fixed-domain convex QP; raise on solver or feasibility failure.

    Cholesky whitening changes coordinates, not the loss or regularization.
    A near-singular Hessian uses spectral preconditioning: an eigenvalue
    floor bounds the coordinate transformation, while the objective is still
    evaluated from the original residuals. No extra ridge is introduced.

    An optional right transform T changes the data term to
    ``0.5 * ||(design @ vstack(R, X) - target) @ T||_F**2``.
    Ridge still penalizes the physical R/X matrices and their constraints
    remain unchanged. This includes analytically eliminated common modes.
    """

    n = target.shape[1]
    if output_transform is not None:
        output_transform = np.asarray(output_transform, dtype=float)
        if output_transform.shape != (n, n):
            raise ValueError(f"output_transform must have shape ({n}, {n})")
        if not np.all(np.isfinite(output_transform)):
            raise ValueError("output_transform must contain only finite values")
    ii, jj = np.triu_indices(n)
    k = len(ii)
    # A nearly unidentified OLS solution can have enormous opposite-signed
    # slopes. Use data units, not those unstable slopes, to scale variables.
    target_norm = float(np.linalg.norm(target))
    block_norms = np.asarray([np.linalg.norm(design[:, block*n:(block+1)*n]) for block in range(2)])
    reference_norm = float(np.linalg.norm(design))
    scales = np.asarray([
        target_norm / (value or reference_norm)
        if target_norm > 0.0 and (value > 0.0 or reference_norm > 0.0) else 1.0
        for value in block_norms
    ])
    if margins is not None:
        scales = np.maximum(scales, margins)
    # Each upper-triangular variable contributes to one or two output columns.
    features = np.zeros((target.size, 2 * k))
    shaped = features.reshape(len(target), n, 2 * k)
    for block in range(2):
        for position, (i, j) in enumerate(zip(ii, jj)):
            shaped[:, i, block * k + position] = scales[block] * design[:, block * n + j]
            if i != j:
                shaped[:, j, block * k + position] = scales[block] * design[:, block * n + i]
    if output_transform is None:
        y = target.ravel()
    else:
        # Transform output columns, not the physical matrix parameters. Keep
        # the untransformed path above unchanged for the default estimator.
        features = np.einsum("tjk,ji->tik", shaped, output_transform).reshape(target.size, 2 * k)
        y = (target @ output_transform).ravel()
    denominator = float(y @ y) or 1.0
    multiplicity = np.where(ii == jj, 1.0, 2.0)
    ridge_weights = np.concatenate([scale**2 * multiplicity for scale in scales])
    hessian = features.T @ features / denominator
    hessian.flat[:: len(hessian) + 1] += alpha * ridge_weights / denominator
    hessian = 0.5 * (hessian + hessian.T)
    linear = -(features.T @ y) / denominator

    position = np.empty((n, n), dtype=int)
    position[ii, jj] = position[jj, ii] = np.arange(k)
    rows, rhs = [], []
    if margins is not None:
        for block, margin in enumerate(margins):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    row = np.zeros(2 * k)
                    row[block * k + position[i, i]] = 1.0
                    row[block * k + position[i, j]] = -1.0
                    rows.append(row)
                    rhs.append(margin / scales[block])
    order = np.asarray(rows).reshape(-1, 2 * k)
    rhs = np.asarray(rhs)
    initial_values = (initial[:, ii, jj] / scales[:, None]).ravel()

    def original_objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        # Evaluate residuals directly, avoiding subtraction of large constants.
        residual = features @ values - y
        value = 0.5 * (residual @ residual + alpha * (ridge_weights @ values**2)) / denominator
        return float(value), hessian @ values + linear

    # Clipping an ill-conditioned unconstrained fit can create a very poor
    # initial objective. A diagonal boundary point is also feasible; use the
    # better of these initializers without changing the optimization domain.
    boundary_values = np.zeros((2, k))
    if margins is not None:
        boundary_values[:, ii == jj] = (margins / scales)[:, None]
    initial_loss = original_objective(initial_values)[0]
    boundary_loss = original_objective(boundary_values.ravel())[0]
    if boundary_loss < initial_loss - 1e-12 * max(abs(initial_loss), abs(boundary_loss), 1e-15):
        initial_values = boundary_values.ravel()

    # Cholesky may succeed on a numerically rank-deficient Hessian. Refuse
    # whitening when its inverse would amplify roundoff; do not add a ridge.
    eigenvalues = np.linalg.eigvalsh(hessian)
    well_conditioned = eigenvalues[-1] > 0.0 and eigenvalues[0] > 1e-12 * eigenvalues[-1]
    upper = None
    if well_conditioned:
        try:
            upper = np.linalg.cholesky(hessian).T
        except np.linalg.LinAlgError:
            pass
    if upper is not None:
        inverse_upper = solve_triangular(upper, np.eye(2 * k), lower=False)
        shift = inverse_upper.T @ linear
        constraints = np.vstack([np.eye(2 * k), order]) @ inverse_upper
        lower = np.concatenate([np.zeros(2 * k), rhs]) + constraints @ shift
        row_scale = np.linalg.norm(constraints, axis=1)
        constraints /= row_scale[:, None]
        lower /= row_scale
        result = minimize(
            lambda values: (0.5 * float(values @ values), values),
            upper @ initial_values + shift,
            jac=True,
            method="SLSQP",
            constraints=LinearConstraint(constraints, lower, np.inf),
            options={"maxiter": max_iterations, "ftol": 1e-12},
        )
        values = inverse_upper @ (result.x - shift)
        coordinate_system = "Cholesky_whitened"
    elif eigenvalues[-1] > 0.0:
        # Limit the inverse scale in null/near-null directions, but keep those
        # directions and the original loss. Replacing H by a floored Hessian
        # here would change the model; evaluating transformed residuals does not.
        eigenvalues, vectors = np.linalg.eigh(hessian)
        roots = np.sqrt(np.maximum(eigenvalues, 1e-6 * eigenvalues[-1]))
        transform = vectors / roots
        transformed_features = features @ transform
        transformed_ridge = np.sqrt(ridge_weights)[:, None] * transform
        constraints = np.vstack([np.eye(2 * k), order]) @ transform
        row_scale = np.linalg.norm(constraints, axis=1)
        constraints /= row_scale[:, None]
        lower = np.concatenate([np.zeros(2 * k), rhs]) / row_scale

        def preconditioned_objective(values: np.ndarray) -> tuple[float, np.ndarray]:
            residual = transformed_features @ values - y
            ridge = transformed_ridge @ values
            objective = 0.5 * (residual @ residual + alpha * (ridge @ ridge)) / denominator
            gradient = (transformed_features.T @ residual + alpha * transformed_ridge.T @ ridge) / denominator
            return float(objective), gradient

        result = minimize(
            preconditioned_objective,
            roots * (vectors.T @ initial_values),
            jac=True,
            method="SLSQP",
            constraints=LinearConstraint(constraints, lower, np.inf),
            options={"maxiter": max_iterations, "ftol": 1e-12},
        )
        values = transform @ result.x
        coordinate_system = "spectral_preconditioned"
    else:
        result = minimize(
            original_objective,
            initial_values,
            jac=True,
            method="SLSQP",
            bounds=Bounds(0.0, np.inf),
            constraints=([LinearConstraint(order, rhs, np.inf)] if len(rhs) else []),
            options={"maxiter": max_iterations, "ftol": 1e-12},
        )
        values = result.x
        coordinate_system = "original_rank_deficient"

    if not result.success or not np.all(np.isfinite(values)):
        raise RuntimeError(f"constrained R/X least squares failed: {result.message}")
    violation = max(0.0, -float(np.min(values)))
    if len(rhs):
        violation = max(violation, float(np.max(rhs - order @ values)))
    if violation > 1e-8:
        raise RuntimeError(f"constrained R/X least squares is infeasible: {violation:.6g}")
    blocks = np.zeros((2, n, n))
    blocks[:, ii, jj] = values.reshape(2, k) * scales[:, None]
    blocks[:, jj, ii] = blocks[:, ii, jj]
    blocks = make_feasible(blocks, margins)
    return blocks, {
        "method": "SLSQP_convex_QP",
        "coordinates": coordinate_system,
        "success": bool(result.success),
        "iterations": int(result.nit),
        "message": str(result.message),
        "maximum_scaled_constraint_violation_before_cleanup": violation,
    }

