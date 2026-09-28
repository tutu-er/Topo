"""Physics-informed constraints for reduced voltage-sensitivity matrices."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SensitivityMatrixDiagnostics:
    """Numerical violations of tree-covariance matrix properties."""

    symmetry_error: float
    minimum_entry: float
    minimum_eigenvalue: float
    minimum_diagonal_gap: float
    minimum_distance: float
    precision_positive_offdiag_fraction: float
    precision_max_positive_offdiag: float
    precision_minimum_row_sum: float

    def to_dict(self) -> dict[str, float]:
        """Return JSON-serializable diagnostics."""

        return asdict(self)


def _matrix_scale(matrix: np.ndarray) -> float:
    diagonal = np.diag(matrix)
    positive = diagonal[diagonal > 0.0]
    if positive.size:
        return float(np.median(positive))
    return float(np.max(np.abs(matrix))) if matrix.size else 0.0


def _project_diagonal_order(matrix: np.ndarray, margin: float, sweeps: int = 4) -> np.ndarray:
    """Project cyclically onto M_ii - M_ij >= margin half-spaces.

    For a violation v, the nearest symmetric matrix minimizes a**2 + 2*b**2
    subject to a - b = v. Thus the diagonal increases by 2*v/3 and both
    symmetric off-diagonal entries decrease by v/3. Each individual step is
    the exact Frobenius projection onto one half-space. Their cyclic
    composition, without Dykstra corrections, need not be the nearest point
    in the intersection. The final ``_make_ordered_feasible`` call in the
    public repair functions guarantees the bounds.
    """

    projected = matrix.copy()
    n = len(projected)
    for _ in range(sweeps):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                violation = margin - (projected[i, i] - projected[i, j])
                if violation <= 0.0:
                    continue
                projected[i, i] += 2.0 * violation / 3.0
                projected[i, j] -= violation / 3.0
                projected[j, i] = projected[i, j]
    return projected


def _make_ordered_feasible(matrix: np.ndarray, margin: float) -> np.ndarray:
    """Return a symmetric nonnegative matrix satisfying all diagonal bounds."""

    feasible = np.maximum(0.0, 0.5 * (matrix + matrix.T))
    if len(feasible) <= 1:
        return feasible
    off_diagonal = feasible.copy()
    np.fill_diagonal(off_diagonal, -np.inf)
    required_diagonal = np.max(off_diagonal, axis=1) + margin
    diagonal = np.maximum(np.diag(feasible), required_diagonal)
    np.fill_diagonal(feasible, diagonal)
    return feasible


def project_ordered_sensitivity_matrix(
    matrix: np.ndarray,
    margin_ratio: float = 1e-6,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Repair symmetry, nonnegativity, and diagonal shared-path bounds.

    This is a feasibility repair, not a nearest-point projection onto the
    intersection. The margin is fixed within this call using the input scale;
    calling the function again may select a different margin.
    """

    current = np.asarray(matrix, dtype=float)
    if current.ndim != 2 or current.shape[0] != current.shape[1]:
        raise ValueError("sensitivity matrix must be square")
    if not np.all(np.isfinite(current)):
        raise ValueError("sensitivity matrix must contain only finite values")
    if not np.isfinite(margin_ratio) or margin_ratio < 0.0:
        raise ValueError("margin_ratio must be finite and nonnegative")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    current = 0.5 * (current + current.T)
    # A tiny positive diagonal must not hide substantial off-diagonal entries.
    if not current.size or float(np.max(np.abs(current))) <= 1e-14:
        return np.zeros_like(current)
    scale = _matrix_scale(current)
    margin = margin_ratio * scale
    for _ in range(max_iterations):
        previous = current.copy()
        current = np.maximum(0.0, 0.5 * (current + current.T))
        current = _project_diagonal_order(current, margin, sweeps=4)
        change = np.linalg.norm(current - previous, ord="fro")
        reference = max(np.linalg.norm(previous, ord="fro"), 1e-15)
        if change / reference <= tolerance:
            break
    return _make_ordered_feasible(current, margin)



def project_tree_covariance_matrix(
    matrix: np.ndarray,
    margin_ratio: float = 1e-6,
    eigenvalue_floor_ratio: float = 1e-10,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Repair feasibility in a convex outer approximation of tree sensitivities.

    The intersection enforces symmetry, elementwise nonnegativity, positive
    semidefiniteness, and diagonal entries larger than all off-diagonal entries
    in their rows and columns. Cyclic projections and the final feasibility
    repair do not compute the nearest point in Frobenius norm. These matrix
    conditions alone do not establish membership in a tree model.
    """

    current = np.asarray(matrix, dtype=float)
    if current.ndim != 2 or current.shape[0] != current.shape[1]:
        raise ValueError("sensitivity matrix must be square")
    if not np.all(np.isfinite(current)):
        raise ValueError("sensitivity matrix must contain only finite values")
    if (
        not np.isfinite(margin_ratio)
        or not np.isfinite(eigenvalue_floor_ratio)
        or margin_ratio < 0.0
        or eigenvalue_floor_ratio < 0.0
    ):
        raise ValueError("constraint ratios must be finite and nonnegative")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    current = 0.5 * (current + current.T)
    if not current.size or float(np.max(np.abs(current))) <= 1e-14:
        return np.zeros_like(current)
    scale = _matrix_scale(current)
    margin = margin_ratio * scale
    eigenvalue_floor = eigenvalue_floor_ratio * scale

    for _ in range(max_iterations):
        previous = current.copy()
        current = np.maximum(0.0, 0.5 * (current + current.T))
        current = _project_diagonal_order(current, margin)
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (current + current.T))
        eigenvalues = np.maximum(eigenvalues, eigenvalue_floor)
        current = (eigenvectors * eigenvalues) @ eigenvectors.T
        current = 0.5 * (current + current.T)
        change = np.linalg.norm(current - previous, ord="fro")
        reference = max(np.linalg.norm(previous, ord="fro"), 1e-15)
        if change / reference <= tolerance:
            gap = np.diag(current)[:, None] - current
            np.fill_diagonal(gap, np.inf)
            slack = 10.0 * tolerance * scale
            if (current.min() >= -slack and gap.min() >= margin - slack
                    and np.linalg.eigvalsh(current).min() >= -slack):
                break

    current = np.maximum(0.0, 0.5 * (current + current.T))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(current)))
    if minimum_eigenvalue < eigenvalue_floor:
        current += (eigenvalue_floor - minimum_eigenvalue) * np.eye(len(current))
    return _make_ordered_feasible(current, margin)


def sensitivity_matrix_diagnostics(matrix: np.ndarray) -> SensitivityMatrixDiagnostics:
    """Measure covariance-, distance-, and inverse-Laplacian violations."""

    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("sensitivity matrix must be square")
    symmetric = 0.5 * (value + value.T)
    n = len(value)
    mask = ~np.eye(n, dtype=bool)
    diagonal_gap = np.diag(symmetric)[:, None] - symmetric
    distances = (
        np.diag(symmetric)[:, None]
        + np.diag(symmetric)[None, :]
        - 2.0 * symmetric
    )
    precision = np.linalg.pinv(symmetric, hermitian=True)
    precision_offdiag = precision[mask]
    positive_offdiag = precision_offdiag > 1e-10
    return SensitivityMatrixDiagnostics(
        symmetry_error=float(np.max(np.abs(value - value.T))) if value.size else 0.0,
        minimum_entry=float(np.min(symmetric)) if value.size else 0.0,
        minimum_eigenvalue=float(np.min(np.linalg.eigvalsh(symmetric))) if value.size else 0.0,
        minimum_diagonal_gap=float(np.min(diagonal_gap[mask])) if n > 1 else float("inf"),
        minimum_distance=float(np.min(distances[mask])) if n > 1 else float("inf"),
        precision_positive_offdiag_fraction=(
            float(np.mean(positive_offdiag)) if precision_offdiag.size else 0.0
        ),
        precision_max_positive_offdiag=(
            float(np.max(np.maximum(precision_offdiag, 0.0)))
            if precision_offdiag.size
            else 0.0
        ),
        precision_minimum_row_sum=float(np.min(precision.sum(axis=1))) if n else 0.0,
    )


def four_point_violation_summary(distance: np.ndarray) -> dict[str, float]:
    """Summarize additive-tree four-point violations over all node quadruples."""

    value = np.asarray(distance, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("distance matrix must be square")
    violations = []
    n = len(value)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for ell in range(k + 1, n):
                    sums = sorted(
                        [
                            value[i, j] + value[k, ell],
                            value[i, k] + value[j, ell],
                            value[i, ell] + value[j, k],
                        ]
                    )
                    violations.append(max(0.0, float(sums[2] - sums[1])))
    array = np.asarray(violations, dtype=float)
    return {
        "quadruple_count": int(len(array)),
        "mean_four_point_violation": float(array.mean()) if array.size else 0.0,
        "max_four_point_violation": float(array.max()) if array.size else 0.0,
        "p95_four_point_violation": float(np.quantile(array, 0.95)) if array.size else 0.0,
    }
