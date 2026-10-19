"""Closed-form checks for two-topology gradient-driven EV planning.

No policy/fitter implementation is imported: these are independent mathematical
counterexamples and identities.  The linear oracles are audited on tiny vertex
sets solely for proof, never used as an EV action library.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest


def test_two_directed_source_profile_distances_need_not_be_equal():
    # h1=(u,s), h2=(2u+s,s), with reference sources zero.
    for u in (-1.0, -0.3, 0.0, 0.4, 1.0):
        s2 = -u / 2.0
        r12 = np.array([u + s2, s2])
        s1 = 0.0
        r21 = np.array([-u, s1])
        assert np.array([1.0, 1.0]) @ r12 == pytest.approx(0.0)
        assert np.array([0.0, 1.0]) @ r21 == pytest.approx(0.0)
        d12 = 0.5 * float(r12 @ r12)
        d21 = 0.5 * float(r21 @ r21)
        assert d12 == pytest.approx(u * u / 4.0)
        assert d21 == pytest.approx(u * u / 2.0)
        if u:
            assert d12 != pytest.approx(d21)


def test_two_topology_equal_weight_history_offsets_cancel_pointwise():
    for h1, h2 in itertools.product((0.0, 1.0, 20.0), repeat=2):
        for d12, d21 in ((0.0, 7.0), (1.0, 2.0), (9.0, 0.0)):
            future_mean = 0.5 * (d12 + d21)
            history_future_mean = 0.5 * (
                h2 - h1 + d12 + h1 - h2 + d21
            )
            assert history_future_mean == pytest.approx(future_mean)


def test_history_offsets_can_change_worst_of_two_directions():
    # Unlike the equal mean, min(S12,S21) retains the historical asymmetry.
    d = np.array([1.0, 4.0])
    assert np.argmin(d) == 0
    offset = np.array([2.5, -2.5])
    assert np.argmin(d + offset) == 1
    assert np.min(d + offset) != np.min(d)


@pytest.mark.parametrize("u", [-2.0, -1.0, -0.3, 0.0, 0.6, 1.0, 2.0])
def test_profile_gradient_and_global_affine_lower_bound(u):
    # q(u)=min_|z|<=1 .5*((u-z)^2+(.5u)^2), optimizer clip(u).
    def q(v):
        return 0.5 * ((v - np.clip(v, -1.0, 1.0)) ** 2 + 0.25 * v * v)

    z = float(np.clip(u, -1.0, 1.0))
    gradient = (u - z) + 0.25 * u
    for v in np.linspace(-2.5, 2.5, 41):
        assert q(v) >= q(u) + gradient * (v - u) - 1e-14
    eps = 1e-6
    numerical = (q(u + eps) - q(u - eps)) / (2.0 * eps)
    assert numerical == pytest.approx(gradient, abs=3e-7)


def test_linear_oracle_increases_frozen_profile_value():
    # Tiny vertex audit of the exact gradient oracle on a scalar interval.
    q = lambda u: 0.5 * ((u - np.clip(u, -1.0, 1.0)) ** 2 + 0.25 * u * u)
    for u in (-1.7, -0.4, 0.4, 1.7):
        g = u - np.clip(u, -1.0, 1.0) + 0.25 * u
        v = 2.0 if g > 0.0 else -2.0
        assert g * (v - u) >= 0.0
        assert q(v) >= q(u) + g * (v - u) - 1e-14
        assert q(v) > q(u)


def test_zero_gradient_at_origin_does_not_certify_maximum():
    u = 0.0
    gradient = u
    assert gradient == 0.0
    linear_values = [gradient * v for v in (-1.0, 0.0, 1.0)]
    assert max(linear_values) == min(linear_values)
    assert 0.5 * (-1.0) ** 2 > 0.5 * u ** 2


@pytest.mark.parametrize("epsilon", [0.01, 0.2, 0.9])
def test_exact_gradient_oracle_can_stop_at_bad_asymmetric_box_vertex(epsilon):
    vertices = np.array(list(itertools.product((-1.0, 2.0), (-1.0, 1.0))))
    u = np.array([-1.0, 1.0])
    gradient = np.array([u[0], epsilon * u[1]])
    v = vertices[np.argmax(vertices @ gradient)]
    np.testing.assert_array_equal(v, u)
    q = lambda w: 0.5 * (w[0] ** 2 + epsilon * w[1] ** 2)
    assert max(map(q, vertices)) == pytest.approx(0.5 * (4.0 + epsilon))
    assert q(u) == pytest.approx(0.5 * (1.0 + epsilon))
    assert max(map(q, vertices)) > q(u)
    # Every sufficiently short feasible direction from this vertex decreases q.
    for dx in (0.01, 0.02):
        for dy in (-0.01, -0.02):
            assert q(u + np.array([dx, dy])) < q(u)


@pytest.mark.parametrize("epsilon", [0.01, 0.2, 0.9])
def test_diamond_counterexample_has_arbitrarily_bad_value_ratio(epsilon):
    vertices = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    u = np.array([0.0, 1.0])
    gradient = np.array([0.0, epsilon])
    np.testing.assert_array_equal(vertices[np.argmax(vertices @ gradient)], u)
    q = lambda w: 0.5 * (w[0] ** 2 + epsilon * w[1] ** 2)
    assert q(u) / max(map(q, vertices)) == pytest.approx(epsilon)


def test_symmetric_unshifted_diagonal_box_is_not_bad_vertex_counterexample():
    vertices = np.array(list(itertools.product((-1.0, 1.0), repeat=2)))
    values = 0.5 * (vertices[:, 0] ** 2 + 0.2 * vertices[:, 1] ** 2)
    np.testing.assert_array_equal(values, np.full(4, 0.6))


@pytest.mark.parametrize("c", [-3.0, -0.2, 0.0, 0.7, 3.0])
def test_rank_one_squared_affine_global_maximum_from_two_linear_extrema(c):
    # A convex square over an arbitrary compact polytope attains its maximum
    # at either extreme scalar response. This tiny vertex audit is independent.
    vertices = np.array([[0.0, 0.0], [1.5, 0.0], [1.0, 1.0], [0.0, 0.8]])
    b = np.array([0.7, -1.2])
    response = c + vertices @ b
    lo, hi = float(min(response)), float(max(response))
    two_sign_value = 0.5 * max(lo * lo, hi * hi)
    exhaustive_value = max(0.5 * response * response)
    assert two_sign_value == pytest.approx(exhaustive_value)
    # Independently inspect nonvertex convex combinations: all map to [lo,hi].
    for alpha in np.linspace(0.0, 1.0, 21):
        point = alpha * vertices[1] + (1.0 - alpha) * vertices[3]
        assert 0.5 * (c + b @ point) ** 2 <= two_sign_value + 1e-14
