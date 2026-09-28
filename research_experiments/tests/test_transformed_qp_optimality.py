"""Physical-coordinate optimality checks, independent of solver encoding."""

import numpy as np
import pytest
from scipy.optimize import nnls

from rnj_wzzt.estimation.constrained_least_squares import solve_symmetric_least_squares


@pytest.mark.parametrize("collinear", [False, True])
@pytest.mark.parametrize("singular_transform", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 0.3])
@pytest.mark.parametrize("ordered", [False, True])
def test_solution_satisfies_physical_kkt_conditions(collinear, singular_transform, alpha, ordered):
    rng = np.random.default_rng(9126)
    p, q = rng.normal(size=(2, 23, 2))
    if collinear:
        q = 0.4 * p
    design = np.hstack([p, q])
    raw = np.array([[[0.1, 0.8], [0.8, 0.2]], [[0.9, -0.2], [-0.2, 0.7]]])
    target = design @ np.vstack(raw) + 0.1 * rng.normal(size=(23, 2))
    transform = (np.eye(2) - 0.5 if singular_transform
                 else np.array([[1.0, 0.35], [-0.2, 0.65]]))
    margins = np.array([0.07, 0.04]) if ordered else None
    # The public solver must construct a feasible start from this poor guess.
    blocks, diagnostics = solve_symmetric_least_squares(
        design, target, -np.ones((2, 2, 2)), margins, alpha, 500,
        output_transform=transform,
    )
    np.testing.assert_allclose(blocks, blocks.transpose(0, 2, 1), atol=1e-12)
    assert diagnostics["success"]

    # Differentiate the matrix objective directly, without production features.
    beta = np.vstack(blocks)
    residual = design @ beta - target
    gradient = design.T @ (residual @ transform @ transform.T) + alpha * beta
    r, x = blocks
    gr, gx = gradient[:2], gradient[2:]
    values = np.array([r[0, 0], r[0, 1], r[1, 1], x[0, 0], x[0, 1], x[1, 1]])
    derivative = np.array([
        gr[0, 0], gr[0, 1] + gr[1, 0], gr[1, 1],
        gx[0, 0], gx[0, 1] + gx[1, 0], gx[1, 1],
    ])
    normals, lower = np.eye(6), np.zeros(6)
    if ordered:
        normals = np.vstack([normals, [1, -1, 0, 0, 0, 0], [0, -1, 1, 0, 0, 0],
                             [0, 0, 0, 1, -1, 0], [0, 0, 0, 0, -1, 1]])
        lower = np.r_[lower, np.repeat(margins, 2)]
    slack = normals @ values - lower
    assert slack.min() >= -1e-10
    # For this convex problem, grad f = C_active.T @ lambda, lambda >= 0,
    # certifies optimality in exact arithmetic. Check it at numerical tolerance.
    active = slack < 1e-7
    if active.any():
        _, stationarity_error = nnls(normals[active].T, derivative)
    else:
        stationarity_error = np.linalg.norm(derivative)
    assert stationarity_error <= 2e-5 * max(1.0, np.linalg.norm(derivative))

