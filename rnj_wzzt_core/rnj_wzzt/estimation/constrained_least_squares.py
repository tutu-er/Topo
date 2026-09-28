"""Symmetric, nonnegative R/X regression as linearly constrained least squares.

For A = [P, Q], B = vstack(R, X), and T = output_transform (default I):

    min  0.5 ||(A B - Y) T||_F^2 + 0.5 alpha (||R||_F^2 + ||X||_F^2)
    s.t. M = M.T, M_ij >= 0                         for M in {R, X},
         M_ii - M_ij >= margin_M, i != j            if margins are given.

The last constraint is entrywise diagonal order, not diagonal dominance or PSD.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import LinearConstraint, minimize


def make_feasible(blocks: np.ndarray, margins: np.ndarray | None) -> np.ndarray:
    """Clip already symmetric blocks and raise diagonals; not a projection."""

    result = np.maximum(blocks, 0.0)
    if margins is not None and result.shape[1] > 1:
        for block, margin in zip(result, margins):
            off = block.copy()
            np.fill_diagonal(off, -np.inf)
            np.fill_diagonal(block, np.maximum(np.diag(block), off.max(axis=1) + margin))
    return result


def _solve_qp(features, target, constraints, lower, initial, max_iterations):
    """Solve min ||F z - y||^2 / 2, C z >= d, using equivalent coordinates."""

    hessian = features.T @ features
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    upper = None
    if eigenvalues[-1] > 0.0 and eigenvalues[0] > 1e-12 * eigenvalues[-1]:
        try:
            upper = np.linalg.cholesky(hessian).T
        except np.linalg.LinAlgError:
            pass

    # In every branch z = transform @ (u - shift).
    shift = np.zeros_like(initial)
    if upper is not None:
        transform = solve_triangular(upper, np.eye(len(initial)), lower=False)
        shift = -transform.T @ (features.T @ target)
        start = upper @ initial + shift

        def objective(u):
            return 0.5 * float(u @ u), u

        coordinates = "Cholesky_whitened"
    else:
        if eigenvalues[-1] > 0.0:
            # Floor only the coordinate scale, never the objective's Hessian.
            roots = np.sqrt(np.maximum(eigenvalues, 1e-6 * eigenvalues[-1]))
            transform = eigenvectors / roots
            start = roots * (eigenvectors.T @ initial)
            coordinates = "spectral_preconditioned"
        else:
            transform = np.eye(len(initial))
            start = initial
            coordinates = "original_rank_deficient"
        transformed_features = features @ transform

        def objective(u):
            residual = transformed_features @ u - target
            return 0.5 * float(residual @ residual), transformed_features.T @ residual

    transformed_constraints = constraints @ transform
    transformed_lower = lower + transformed_constraints @ shift
    row_norms = np.linalg.norm(transformed_constraints, axis=1)
    result = minimize(
        objective, start, jac=True, method="SLSQP",
        constraints=LinearConstraint(
            transformed_constraints / row_norms[:, None],
            transformed_lower / row_norms, np.inf,
        ),
        options={"maxiter": max_iterations, "ftol": 1e-12},
    )
    values = transform @ (result.x - shift)
    if not result.success or not np.all(np.isfinite(values)):
        raise RuntimeError(f"constrained R/X least squares failed: {result.message}")
    violation = max(0.0, float(np.max(lower - constraints @ values)))
    if violation > 1e-8:
        raise RuntimeError(f"constrained R/X least squares is infeasible: {violation:.6g}")
    return values, {
        "method": "SLSQP_convex_QP",
        "coordinates": coordinates,
        "success": bool(result.success),
        "iterations": int(result.nit),
        "message": str(result.message),
        "maximum_scaled_constraint_violation_before_cleanup": violation,
    }


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
    """Encode the matrix problem above, solve it, and reconstruct R and X.

    design, target, initial have shapes (samples, 2*n), (samples, n),
    (2, n, n). Initial blocks are symmetric; margins is None or a fixed
    nonnegative pair. Temporal preprocessing belongs to the caller.

    Upper-triangular variables encode symmetry exactly. Ridge is appended as
    extra least-squares rows, with off-diagonal entries counted twice.
    T affects only residuals. Scaling and preconditioning leave the physical
    objective and constraints unchanged; rank-deficient fits may be nonunique.
    """

    n = target.shape[1]
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    if margins is not None:
        margins = np.asarray(margins, dtype=float)
        if margins.shape != (2,) or not np.all(np.isfinite(margins)) or np.any(margins < 0.0):
            raise ValueError("margins must be a finite, nonnegative pair")
    if output_transform is not None:
        output_transform = np.asarray(output_transform, dtype=float)
        if output_transform.shape != (n, n):
            raise ValueError(f"output_transform must have shape ({n}, {n})")
        if not np.all(np.isfinite(output_transform)):
            raise ValueError("output_transform must contain only finite values")

    # M_ij = scale_M * z_M,ij. Use data scales, not unstable OLS slopes.
    design_blocks = np.split(design, 2, axis=1)
    target_norm = float(np.linalg.norm(target))
    design_norm = float(np.linalg.norm(design))
    scales = np.ones(2)
    if target_norm > 0.0 and design_norm > 0.0:
        scales = np.array([target_norm / (np.linalg.norm(a) or design_norm) for a in design_blocks])
    if margins is not None:
        scales = np.maximum(scales, margins)

    # E_ij has ones at (i,j) and (j,i); each column is vec(P E_ij) or vec(Q E_ij).
    ii, jj = np.triu_indices(n)
    k = len(ii)
    features = np.zeros((len(target), n, 2 * k))
    for block, (a, scale) in enumerate(zip(design_blocks, scales)):
        columns = block * k + np.arange(k)
        features[:, jj, columns] = scale * a[:, ii]
        features[:, ii, columns] = scale * a[:, jj]
    y = target
    if output_transform is not None:
        features = np.einsum("tjk,ji->tik", features, output_transform)
        y = target @ output_transform
    features, y = features.reshape(target.size, 2 * k), y.ravel()

    # ||M||_F^2 = sum_i M_ii^2 + 2 sum_{i<j} M_ij^2.
    normalizer = float(np.linalg.norm(y)) or 1.0
    if alpha > 0.0:
        multiplicity = np.where(ii == jj, 1.0, 2.0)
        ridge = (scales[:, None] * np.sqrt(alpha * multiplicity)).ravel()
        features = np.vstack([features, np.diag(ridge)])
        y = np.concatenate([y, np.zeros(2 * k)])
    features, y = features / normalizer, y / normalizer

    # C z >= d combines nonnegativity and optional diagonal-order inequalities.
    rows, lower = list(np.eye(2 * k)), [0.0] * (2 * k)
    position = np.empty((n, n), dtype=int)
    position[ii, jj] = position[jj, ii] = np.arange(k)
    if margins is not None:
        for block, margin in enumerate(margins / scales):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        row = np.zeros(2 * k)
                        row[block * k + position[i, i]] = 1.0
                        row[block * k + position[i, j]] = -1.0
                        rows.append(row)
                        lower.append(margin)

    # A diagonal boundary point can be much better than a clipped, unstable fit.
    candidates = [make_feasible(initial, margins), make_feasible(np.zeros_like(initial), margins)]
    starts = [(blocks[:, ii, jj] / scales[:, None]).ravel() for blocks in candidates]
    start = min(starts, key=lambda z: np.linalg.norm(features @ z - y))
    values, diagnostics = _solve_qp(
        features, y, np.asarray(rows), np.asarray(lower), start, max_iterations,
    )

    blocks = np.zeros((2, n, n))
    blocks[:, ii, jj] = values.reshape(2, k) * scales[:, None]
    blocks[:, jj, ii] = blocks[:, ii, jj]
    # The solver has checked feasibility; repair only remaining roundoff.
    return make_feasible(blocks, margins), diagnostics

