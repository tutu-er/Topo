"""Independent checks for output weighting and eliminated scalar common modes."""

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, minimize

from rnj_wzzt.estimation.constrained_least_squares import solve_symmetric_least_squares


def _problem():
    rng = np.random.default_rng(7314)
    design = rng.normal(size=(18, 4))
    design -= design.mean(axis=0)
    # R violates ordered inequalities, forcing an active constrained optimum.
    raw = np.array([[[0.1, 0.8], [0.8, 0.2]], [[0.9, 0.1], [0.1, 0.7]]])
    target = design @ np.vstack(raw) + 0.7 * np.sin(np.arange(len(design)))[:, None]
    target += 0.04 * rng.normal(size=target.shape)
    target -= target.mean(axis=0)
    return design, target, np.stack([np.eye(2), np.eye(2)]), np.array([0.07, 0.04])


def _solve(design, target, initial, margins, alpha=0.0, **kwargs):
    return solve_symmetric_least_squares(design, target, initial, margins, alpha, 500, **kwargs)[0]


def _explicit_common_mode_reference(design, target, margins, alpha, gamma):
    """Optimize physical entries and c_t directly, without production features."""
    samples = len(target)

    def matrices(values):
        a, b, c, d, e, f = values[:6]
        return np.array([[[a, b], [b, c]], [[d, e], [e, f]]])

    def objective(values):
        beta = np.vstack(matrices(values))
        common = values[6:]
        residual = design @ beta + common[:, None] - target
        value = 0.5 * (np.sum(residual**2) + alpha * np.sum(beta**2) + gamma * np.sum(common**2))
        gradient_beta = design.T @ residual + alpha * beta
        gradient = np.empty_like(values)
        for block in range(2):
            g = gradient_beta[2 * block:2 * block + 2]
            gradient[3 * block:3 * block + 3] = [g[0, 0], g[0, 1] + g[1, 0], g[1, 1]]
        gradient[6:] = residual.sum(axis=1) + gamma * common
        return float(value), gradient

    order = np.zeros((4, 6 + samples))
    order[0, :3], order[1, :3] = [1, -1, 0], [0, -1, 1]
    order[2, 3:6], order[3, 3:6] = [1, -1, 0], [0, -1, 1]
    initial = np.r_[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, np.zeros(samples)]
    result = minimize(
        objective, initial, jac=True, method="SLSQP",
        bounds=Bounds(np.r_[np.zeros(6), np.full(samples, -np.inf)], np.inf),
        constraints=LinearConstraint(order, np.repeat(margins, 2), np.inf),
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    assert result.success, result.message
    return matrices(result.x), result.x[6:], result.fun


@pytest.mark.parametrize("alpha", [0.0, 0.3])
def test_none_and_identity_keep_default_fit(alpha):
    design, target, initial, margins = _problem()
    identity = np.eye(2)
    expected = _solve(design, target, initial, margins, alpha)
    explicit_none = _solve(design, target, initial, margins, alpha, output_transform=None)
    transformed = _solve(design, target, initial, margins, alpha, output_transform=identity)
    np.testing.assert_array_equal(explicit_none, expected)
    np.testing.assert_allclose(transformed, expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_array_equal(identity, np.eye(2))


@pytest.mark.parametrize("alpha,gamma", [(0.0, 0.0), (0.3, 0.0), (0.0, 0.7), (0.3, 0.7), (2.0, 5.0)])
def test_elimination_matches_explicit_common_mode_qp(alpha, gamma):
    design, target, initial, margins = _problem()
    n = target.shape[1]
    projector = np.ones((n, n)) / n
    transform = np.eye(n) - projector + np.sqrt(gamma / (n + gamma)) * projector
    blocks = _solve(design, target, initial, margins, alpha, output_transform=transform)
    raw_residual = design @ np.vstack(blocks) - target
    common = -raw_residual.sum(axis=1) / (n + gamma)
    residual = raw_residual + common[:, None]
    objective = 0.5 * (np.sum(residual**2) + alpha * np.sum(blocks**2) + gamma * np.sum(common**2))
    reduced_objective = 0.5 * (np.sum((raw_residual @ transform)**2) + alpha * np.sum(blocks**2))
    reference, reference_common, reference_objective = _explicit_common_mode_reference(design, target, margins, alpha, gamma)
    assert objective == pytest.approx(reduced_objective, abs=1e-11)
    assert objective == pytest.approx(reference_objective, abs=2e-8)
    if alpha > 0 or gamma > 0:
        np.testing.assert_allclose(blocks, reference, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(common, reference_common, atol=3e-5, rtol=3e-5)
    else:
        # Free c leaves common matrix components unidentified: compare only
        # identifiable projected predictions, rather than arbitrary matrices.
        np.testing.assert_allclose(design @ np.vstack(blocks) @ transform,
                                   design @ np.vstack(reference) @ transform,
                                   atol=3e-5, rtol=3e-5)
    assert np.min(blocks) >= 0.0
    for block, margin in zip(blocks, margins):
        assert block[0, 0] - block[0, 1] >= margin - 1e-10
        assert block[1, 1] - block[1, 0] >= margin - 1e-10


def test_nonsymmetric_right_transform_respects_orthogonal_invariance():
    design, target, initial, margins = _problem()
    transform = np.array([[1.0, 0.4], [-0.2, 0.7]])
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    original = _solve(design, target, initial, margins, 0.2, output_transform=transform)
    rotated = _solve(design, target, initial, margins, 0.2, output_transform=transform @ rotation)
    # ||E T Q||_F = ||E T||_F; transposing T fails for this nonsymmetric input.
    np.testing.assert_allclose(rotated, original, atol=1e-8, rtol=1e-8)


def test_zero_transform_with_ridge_keeps_only_required_diagonal_margins():
    design, target, initial, margins = _problem()
    blocks = _solve(design, target, initial, margins, 0.3, output_transform=np.zeros((2, 2)))
    np.testing.assert_allclose(blocks, margins[:, None, None] * np.eye(2), atol=1e-10)


@pytest.mark.parametrize("transform", [
    np.eye(3), np.zeros((2, 1)), np.ones(2),
    np.array([[1.0, np.nan], [0.0, 1.0]]),
    np.array([[1.0, 0.0], [np.inf, 1.0]]),
])
def test_invalid_output_transform_is_rejected(transform):
    design, target, initial, margins = _problem()
    with pytest.raises(ValueError, match="output_transform"):
        _solve(design, target, initial, margins, output_transform=transform)
