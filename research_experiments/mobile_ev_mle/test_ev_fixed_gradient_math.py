"""Independent closed-form audits of fixed-line gradient/offset mathematics.

These fixtures do not import the policy, fitters, AC routines, or MILP backend.
Source minimizers are explicit projections, not numerical optimization calls.
Passing them verifies mathematical identities, not production integration.
"""
from __future__ import annotations

import numpy as np
import pytest


def _directed_scores(distances, history_half_sse):
    """Definition D_ab + H_b - H_a, excluding the diagonal at aggregation."""
    h = np.asarray(history_half_sse, float)
    return np.asarray(distances, float) + h[None, :] - h[:, None]


def _nearest(scores):
    scores = np.asarray(scores, float)
    return float(np.mean([min(scores[a, b] for b in range(len(scores)) if b != a)
                          for a in range(len(scores))]))


@pytest.mark.parametrize("qp_sse,ha,hb", [
    (6., 1., 4.), (6., 4., 1.), (1., 5., 0.), (0., 2., 2.),
])
def test_full_qp_sse_offset_maps_to_half_history_difference(qp_sse, ha, hb):
    offset = -2. * (hb-ha)
    score_from_backend_convention = .5 * (qp_sse-offset)
    expected_joint_half_sse_difference = hb-ha+.5*qp_sse
    assert score_from_backend_convention == expected_joint_half_sse_difference
    if hb-ha+.5*qp_sse < 0:
        # A historical deficit is signed; replacing it with zero changes it.
        assert score_from_backend_convention < 0
        assert score_from_backend_convention != max(0., score_from_backend_convention)


def test_equal_weight_all_directed_history_offsets_cancel_for_every_action():
    h = np.array([3., 0., 8., 1.])
    off_diagonal = ~np.eye(len(h), dtype=bool)
    delta = h[None, :]-h[:, None]
    assert np.sum(delta[off_diagonal]) == 0.
    # Distinct action responses, including ordered (asymmetric) distances.
    actions = [
        np.arange(16, dtype=float).reshape(4, 4)/3.,
        np.arange(16, dtype=float)[::-1].reshape(4, 4)/7.,
        np.eye(4),
    ]
    for d in actions:
        score = _directed_scores(d, h)
        assert np.mean(score[off_diagonal]) == pytest.approx(
            np.mean(d[off_diagonal]), abs=2e-15)


def test_nearest_historical_offset_can_change_competitor_and_action():
    h = np.array([0., 3., 0.])
    first = np.array([[0., 1., 10.], [10., 0., 10.], [10., 10., 0.]])
    second = np.array([[0., 2., 3.], [10., 0., 10.], [10., 10., 0.]])
    # At the second action, reference A changes from B to C after history.
    assert np.argmin(second[0, 1:])+1 == 1
    assert np.argmin(_directed_scores(second, h)[0, 1:])+1 == 2
    # History also changes the better action, despite zero all-pair offset mean.
    assert _nearest(second) > _nearest(first)
    assert _nearest(_directed_scores(first, h)) > _nearest(_directed_scores(second, h))


@pytest.mark.parametrize("u", [-2., -1., -.25, 0., .25, 1., 2.])
def test_source_box_active_or_free_envelope_and_finite_difference(u):
    # min_|z|<=1 .5*((u-z)^2+(.5*u)^2), known optimizer z=clip(u).
    beta = .5
    B = np.array([1., beta])
    G = np.array([-1., 0.])
    z = float(np.clip(u, -1., 1.))
    residual = B*u+G*z
    source_gradient = float(G@residual)
    if z == -1.:
        assert source_gradient >= -1e-15
    elif z == 1.:
        assert source_gradient <= 1e-15
    else:
        assert source_gradient == pytest.approx(0., abs=1e-15)

    def q(v):
        return .5*((v-np.clip(v, -1., 1.))**2+beta**2*v*v)

    envelope = float(B@residual)
    analytic = u-z+beta**2*u
    step = 1e-6
    centered_difference = (q(u+step)-q(u-step))/(2*step)
    assert envelope == pytest.approx(analytic, abs=2e-15)
    # Even at the source active-set switches +/-1 the value is C1.
    assert centered_difference == pytest.approx(envelope, abs=3e-7)


@pytest.mark.parametrize("u", [-.8, -.3, .3, .8])
def test_rank_deficient_nonunique_sources_have_unique_residual_and_gradient(u):
    # G has rank one and two source coordinates; both z choices are optimal.
    beta = .7
    G = np.array([[1., 1.], [0., 0.]])
    B = np.array([1., beta])
    z_first = np.array([-u/2., -u/2.])
    z_second = np.array([-u, 0.])
    assert np.linalg.matrix_rank(G) == 1
    assert not np.array_equal(z_first, z_second)
    assert np.max(np.abs(z_first)) < 1.
    assert np.max(np.abs(z_second)) < 1.
    residuals = [B*u+G@z for z in (z_first, z_second)]
    projected = (np.eye(2)-G@np.linalg.pinv(G))@(B*u)
    for residual in residuals:
        assert residual == pytest.approx(projected, abs=1e-15)
        assert G.T@residual == pytest.approx([0., 0.], abs=1e-15)
        assert float(B@residual) == pytest.approx(beta**2*u, abs=1e-15)
        assert .5*float(residual@residual) == pytest.approx(.5*beta**2*u*u, abs=1e-15)


@pytest.mark.parametrize("switch", [-2., 2.])
def test_rank_deficient_bounded_source_projection_is_C1_at_switch(switch):
    # G[-1,1]^2=[-2,2]x{0}: image projection is unique, source may be nonunique.
    beta = .7
    B = np.array([1., beta])
    G = np.array([[1., 1.], [0., 0.]])

    def value_gradient(u):
        source_sum = float(np.clip(-u, -2., 2.))
        z = np.full(2, source_sum/2.)
        r = B*u+G@z
        g = G.T@r
        if abs(source_sum) == 2.:
            assert np.all(g >= -1e-14) if source_sum < 0 else np.all(g <= 1e-14)
        return .5*float(r@r), float(B@r)

    epsilon = 1e-7
    q0, g0 = value_gradient(switch)
    qleft, gleft = value_gradient(switch-epsilon)
    qright, gright = value_gradient(switch+epsilon)
    assert g0 == pytest.approx(beta**2*switch, abs=1e-15)
    assert gleft == pytest.approx(g0, abs=2e-7)
    assert gright == pytest.approx(g0, abs=2e-7)
    assert (qright-qleft)/(2*epsilon) == pytest.approx(g0, abs=5e-8)
    assert q0 == pytest.approx(.5*beta**2*switch**2, abs=1e-15)


def test_adding_training_history_changes_validation_future_statistic_target():
    # H is half training SSE; current heldout J excludes it.
    history_half_sse = np.array([10., 0.])
    heldout_full_sse = np.array([0., 1.])
    joint_full_sse = 2*history_half_sse+heldout_full_sse
    assert int(np.argmin(heldout_full_sse)) == 0
    assert int(np.argmin(joint_full_sse)) == 1
    assert .5*(joint_full_sse[1]-joint_full_sse[0]) == pytest.approx(
        history_half_sse[1]-history_half_sse[0]
        +.5*(heldout_full_sse[1]-heldout_full_sse[0]))


def test_expected_profile_half_loss_difference_requires_equal_effective_rank():
    # y=mu_A+epsilon, Cov(epsilon)=I; source tangent projectors P_A,P_B.
    # E||P epsilon||^2=tr(P), which is an exact moment identity.
    P_a = np.diag([1., 0., 0.])
    P_b = np.diag([1., 1., 0.])
    displacement = np.array([2., 0., 0.])
    distance = .5*float(displacement@P_b@displacement)
    expected_half_sse_difference = distance+.5*(np.trace(P_b)-np.trace(P_a))
    assert distance == 2.
    assert expected_half_sse_difference == 2.5
    assert expected_half_sse_difference != distance
