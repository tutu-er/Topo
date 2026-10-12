"""Independent implementation checks; no actual pilot future data are read."""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
PROJECT = Path(r"C:\Users\23761\.codex\worktrees\36a1\Topo")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import numpy as np
import pytest
from research_experiments.mobile_ev_mle.likelihood import ac_predict
from research_experiments.mobile_ev_mle.candidates import RadialCandidate
from research_experiments.mobile_ev_mle.continuous_ac_inverse import inverse_ac, forward_ac, background_logpdf, source_logpdf


FAMILIES = ((), ((0, 1),), ((0, 2),), ((1, 2),))


def candidate_and_edges(family, case="strong"):
    candidate = RadialCandidate.from_supports("independent_test", 3, family)
    r = np.array([.005 if len(s) == 3 else .028+.003*s[0] if len(s) == 1
                  else .00003 if case == "weak" else .012 for s in candidate.supports])
    x = np.array([.0035 if len(s) == 3 else .011+.002*s[0] if len(s) == 1
                  else .00002 if case == "weak" else .007 for s in candidate.supports])
    return candidate, r, x


def prior_states(rng, count=17):
    c = rng.uniform(-.012, .012, (count, 1))
    p = .12+c+rng.uniform(-.004, .004, (count, 3))
    q = .04+.25*c+rng.uniform(-.004, .004, (count, 3))
    p[np.arange(count), rng.integers(0, 3, count)] += rng.uniform(0, .3, count)
    sp, sq = rng.uniform(.065, .135, count), rng.uniform(.02, .05, count)
    return p, q, sp, sq


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("case", ("strong", "weak"))
def test_inverse_and_vector_forward_against_independent_project_ac(family, case):
    rng = np.random.default_rng(130111+len(family))
    candidate, r0, x0 = candidate_and_edges(family, case)
    p, q, sp, sq = prior_states(rng)
    r = r0[None, :]*rng.uniform(.8, 1.2, (7, candidate.m))
    x = x0[None, :]*rng.uniform(.8, 1.2, (7, candidate.m))
    generated = forward_ac(candidate, r[:, None, :], x[:, None, :], p[None], q[None], sp[None], sq[None])
    assert generated["valid"].all() and generated["high_branch"].all()
    for b in range(len(r)):
        independent = ac_predict(candidate, r[b], x[b], p, q, sp, sq)
        np.testing.assert_allclose(generated["v"][b], independent["v"], atol=2e-12, rtol=0)
        np.testing.assert_allclose(generated["master_p"][b], independent["master_p"], atol=2e-12, rtol=0)
        np.testing.assert_allclose(generated["master_q"][b], independent["master_q"], atol=2e-12, rtol=0)
        inverted = inverse_ac(candidate, r[b], x[b], p, q, independent["master_p"], independent["master_q"])
        assert inverted["valid"].all() and inverted["high_branch"].all() and inverted["certified_domain"].all()
        np.testing.assert_allclose(inverted["source_p"], sp, atol=2e-12, rtol=0)
        np.testing.assert_allclose(inverted["source_q"], sq, atol=2e-12, rtol=0)
        np.testing.assert_allclose(inverted["v"], independent["v"], atol=2e-12, rtol=0)


@pytest.mark.parametrize("family", FAMILIES)
def test_master_jacobian_real_central_difference_and_forward_reciprocal(family):
    rng = np.random.default_rng(130222)
    candidate, r, x = candidate_and_edges(family)
    r, x = 1.2*r, 1.2*x
    p, q, sp, sq = prior_states(rng, 11)
    independent = ac_predict(candidate, r, x, p, q, sp, sq)
    P, Q = independent["master_p"], independent["master_q"]
    exact = inverse_ac(candidate, r, x, p, q, P, Q)
    eps = 2e-6
    jac = np.empty((len(P), 2, 2))
    forward_jac = np.empty_like(jac)
    for j in range(2):
        plus = inverse_ac(candidate, r, x, p, q, P+eps*(j == 0), Q+eps*(j == 1))
        minus = inverse_ac(candidate, r, x, p, q, P-eps*(j == 0), Q-eps*(j == 1))
        jac[:, 0, j] = (plus["source_p"]-minus["source_p"])/(2*eps)
        jac[:, 1, j] = (plus["source_q"]-minus["source_q"])/(2*eps)
        plus_forward = ac_predict(candidate, r, x, p, q, sp+eps*(j == 0), sq+eps*(j == 1))
        minus_forward = ac_predict(candidate, r, x, p, q, sp-eps*(j == 0), sq-eps*(j == 1))
        forward_jac[:, 0, j] = (plus_forward["master_p"]-minus_forward["master_p"])/(2*eps)
        forward_jac[:, 1, j] = (plus_forward["master_q"]-minus_forward["master_q"])/(2*eps)
    np.testing.assert_allclose(exact["source_jacobian"], jac, atol=2e-9, rtol=0)
    np.testing.assert_allclose(np.einsum("tij,tjk->tik", exact["source_jacobian"], forward_jac),
                               np.broadcast_to(np.eye(2), jac.shape), atol=2e-9, rtol=0)
    np.testing.assert_allclose(exact["jacobian_abs"], np.abs(np.linalg.det(jac)), atol=3e-9, rtol=0)


@pytest.mark.parametrize("family", FAMILIES)
def test_uniform_rectangle_extrema_are_high_branch_and_jacobian_bound(family):
    candidate, r, x = candidate_and_edges(family)
    for e, support in enumerate(candidate.supports):
        r[e] = .006 if len(support) == 3 else .0216 if len(support) == 2 else .0408
        x[e] = .0042 if len(support) == 3 else .0108 if len(support) == 2 else .018
    P, Q = np.meshgrid(np.linspace(0, 1, 19), np.linspace(0, .5, 17), indexing="ij")
    for ev_site in range(3):
        p, q = np.full(P.shape+(3,), .136), np.full(P.shape+(3,), .047)
        p[..., ev_site] += .3
        result = inverse_ac(candidate, r, x, p, q, P, Q)
        assert result["valid"].all() and result["high_branch"].all() and result["certified_domain"].all()
        assert result["v"].min() > .949
        assert result["determinant"].min() > 0
        perturbation = result["source_jacobian"]-np.eye(2)
        assert np.max(np.abs(perturbation).sum(axis=-1)) < .019
        assert np.max(np.abs(result["shared_residual"])) <= 2e-13


def test_arbitrary_particle_time_noise_broadcast_and_scalar_equivalence():
    candidate, r, x = candidate_and_edges(((0, 2),))
    rng = np.random.default_rng(130333)
    p, q, sp, sq = prior_states(rng, 15)
    p, q = p.reshape(3, 5, 3), q.reshape(3, 5, 3)
    sp, sq = sp.reshape(3, 5), sq.reshape(3, 5)
    r = r[None, None, None, :]*np.linspace(.8, 1.2, 4)[:, None, None, None]
    x = x[None, None, None, :]*np.linspace(1.2, .8, 4)[:, None, None, None]
    f = forward_ac(candidate, r, x, p[None], q[None], sp[None], sq[None])
    v = inverse_ac(candidate, r, x, p[None], q[None], f["master_p"], f["master_q"])
    assert v["v"].shape == (4, 3, 5, 3)
    assert v["source_jacobian"].shape == (4, 3, 5, 2, 2)
    scalar = inverse_ac(candidate, r[2, 0, 0], x[2, 0, 0], p[1, 3], q[1, 3],
                        f["master_p"][2, 1, 3], f["master_q"][2, 1, 3])
    assert scalar["valid"] and scalar["high_branch"]
    np.testing.assert_allclose(scalar["v"], v["v"][2, 1, 3], atol=1e-14, rtol=0)
    np.testing.assert_allclose(v["source_p"], np.broadcast_to(sp[None], (4, 3, 5)), atol=2e-12, rtol=0)


def test_background_marginal_against_direct_c_quadrature_and_support():
    rng = np.random.default_rng(130444)
    c = rng.uniform(-.012, .012, (19, 1))
    p = .12+c+rng.uniform(-.004, .004, (19, 3))
    q = .04+.25*c+rng.uniform(-.004, .004, (19, 3))
    exact = np.exp(background_logpdf(p, q))
    # Independent integrate the six conditional uniform densities over c.
    cgrid = np.linspace(-.012, .012, 200001)
    step = cgrid[1]-cgrid[0]
    for i in range(len(p)):
        supported = (np.abs(p[i, None, :]-.12-cgrid[:, None]) <= .004).all(axis=-1)
        supported &= (np.abs(q[i, None, :]-.04-.25*cgrid[:, None]) <= .004).all(axis=-1)
        numeric = supported.sum()*step/(.024*.008**6)
        # An indicator integral has at most two boundary-grid errors.
        assert abs(numeric-exact[i]) <= 2.01*step/(.024*.008**6)
    assert np.isneginf(background_logpdf(np.array([.2, .12, .12]), np.full(3, .04)))
    assert np.isneginf(background_logpdf(np.full(3, .12), np.array([.02, .04, .04])))
    density_at_center = .008/(.024*.008**6)
    assert np.exp(background_logpdf(np.full(3, .12), np.full(3, .04))) == pytest.approx(density_at_center)


def test_source_uniform_normalization_and_boundaries():
    assert np.exp(source_logpdf(.1, .035))*.07*.03 == pytest.approx(1.)
    assert np.isfinite(source_logpdf(np.array([.065, .135]), np.array([.02, .05]))).all()
    assert np.isneginf(source_logpdf(np.array([.06499, .13501, np.nan]), .035)).all()


def test_zero_impedance_zero_load_and_invalid_branch_masks():
    candidate, r, x = candidate_and_edges(((0, 1),))
    result = inverse_ac(candidate, np.zeros_like(r), np.zeros_like(x), np.zeros(3), np.zeros(3), .1, .035)
    assert result["valid"] and result["high_branch"]
    np.testing.assert_allclose(result["v"], np.ones(3), atol=0, rtol=0)
    np.testing.assert_allclose(result["source_jacobian"], np.eye(2), atol=0, rtol=0)
    assert result["source_p"] == .1 and result["source_q"] == .035
    invalid = inverse_ac(candidate, r, x, np.full(3, 200.), np.full(3, 200.), .1, .035)
    assert not invalid["valid"] and not invalid["high_branch"] and not invalid["certified_domain"]
    assert np.isnan(invalid["v"]).all()
