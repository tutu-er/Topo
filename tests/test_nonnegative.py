"""Tests for the convex nonnegative ridge kernel."""

from __future__ import annotations

import numpy as np

from terminal_case33.estimation.nonnegative import solve_nonnegative_ridge


def test_nonnegative_ridge_satisfies_kkt_conditions() -> None:
    """The returned active-set solution should satisfy first-order optimality."""

    rng = np.random.default_rng(18)
    design = rng.normal(size=(120, 8))
    truth = np.array([1.2, 0.0, 0.4, 0.0, 0.7, 0.0, 0.2, 0.0])
    target = design @ truth + rng.normal(0.0, 0.01, size=len(design))
    ridge = 1e-4

    solution, passes = solve_nonnegative_ridge(design, target, ridge=ridge)

    scale = np.maximum(np.linalg.norm(design, axis=0), 1e-12)
    gradient = design.T @ (design @ solution - target) + ridge * scale**2 * solution
    active = solution > 1e-8
    assert passes == 1
    assert np.all(solution >= 0.0)
    assert np.max(np.abs(gradient[active])) < 1e-7
    assert np.min(gradient[~active]) > -1e-7


def test_nonnegative_ridge_rejects_negative_penalty() -> None:
    """A negative ridge coefficient is not a convex regularizer."""

    design = np.eye(2)
    target = np.ones(2)
    try:
        solve_nonnegative_ridge(design, target, ridge=-1.0)
    except ValueError as exc:
        assert "nonnegative" in str(exc)
    else:
        raise AssertionError("negative ridge should raise ValueError")
