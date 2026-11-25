"""Phase 2 checks for the revised WZZT (rnj_wzzt_dr/identification/revised.py).

- transformed columns are M applied to the package's raw columns;
- Kronecker Gram statistics equal the explicit cross-products;
- GramFitter equals the explicit fit, which equals the package fit when M = I;
- the SVD-row MIQPs (E selection, W unknown support) match brute force on small instances;
- the transformed screening gain equals the explicit one-support gain;
- the saturated noise estimate recovers the noise variance.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from itertools import combinations

import numpy as np
import pytest

from rnj_wzzt_dr.identification import revised as rv
from rnj_wzzt_dr.identification.optimization import QPFitter, feature_matrices
from rnj_wzzt_dr.identification.trees import basis, canonical, children, compatible, is_laminar
from rnj_wzzt_dr.physical.backend import Observation


def _toy(n=6, T=40, seed=0, family=None, noise=1e-3):
    rng = np.random.default_rng(seed)
    p = rng.uniform(.2, 1., (T, n))
    q = .3*p+.05*rng.normal(size=(T, n))
    family = canonical([*basis(n), *(family or [(0, 1), (0, 1, 2), (3, 4)])])
    r = rng.uniform(.01, .05, len(family)); x = rng.uniform(.005, .02, len(family))
    Z = rv.indicators(family, n)
    y = (p@Z.T*r)@Z+(q@Z.T*x)@Z
    root = 1.04+.01*np.sin(np.arange(T)/5)
    y = y+noise*rng.normal(size=y.shape)+.002*rng.normal(size=(T, 1))     # common root-meter error
    return rv.Series(p, q, y, root**2), family


def _explicit_M(T, n, c, offsets):
    S = np.eye(n)-c*np.ones((n, n))/n
    D = np.eye(T)-(np.ones((T, T))/T if offsets else 0.)
    return np.kron(D, S)


@pytest.mark.parametrize("c,offsets", [(0., False), (.7, True), (1., True), (1., False)])
def test_columns_and_gram_are_the_transformed_package_columns(c, offsets):
    s, family = _toy()
    T, n = s.p.shape
    t = rv.Transform(T, n, c, offsets)
    d = rv.Design(s, t, scale=.37)
    M = _explicit_M(T, n, c, offsets)
    Fr, Fx = feature_matrices(Observation(s.p, s.q, s.y, tuple(range(n))), family)
    F = M@np.column_stack((Fr, Fx))/.37
    assert np.allclose(d.columns(family), F, atol=1e-12)
    assert np.allclose(d.y, M@s.y.ravel()/.37, atol=1e-12)
    G, b = d.gram(family)
    assert np.allclose(G, F.T@F, rtol=1e-10, atol=1e-10)
    assert np.allclose(b, F.T@d.y, rtol=1e-10, atol=1e-10)
    assert t.rank == np.linalg.matrix_rank(M)


def test_gram_fitter_matches_explicit_fit_and_package_fit():
    s, family = _toy(n=7, seed=3)
    n = 7
    rng = np.random.default_rng(1)
    universe = canonical([*family, (5, 6), (4, 5, 6), (0, 1, 2, 3), (2, 3), *basis(n)])
    for c, offsets in ((0., False), (.75, True), (1., True)):
        d = rv.Design(s, rv.Transform(*s.p.shape, c, offsets), scale=.01)
        gf = rv.GramFitter(d, universe, basis(n), 2.)
        proper = [C for C in universe if 1 < len(C) < n]
        for _ in range(25):
            S = [C for C in proper if rng.random() < .5]
            S = [C for k, C in enumerate(S) if all(compatible(C, D) for D in S[:k])]
            fam = canonical([*basis(n), *S])
            assert abs(gf.fit(fam)[0]-rv.exact_fit(d, fam, 2.)["objective"]) <= 1e-9*(1+gf.fit(fam)[0])
    # M = I and scale = sqrt(reference): the package's bounded fit
    d = rv.Design(s, rv.Transform(*s.p.shape, 0., False), scale=np.sqrt(2e-4))
    qp = QPFitter(Observation(s.p, s.q, s.y, tuple(range(n))), reference_mse=2e-4)
    for fam in (family, canonical([*basis(n), (5, 6)])):
        assert abs(rv.exact_fit(d, fam, 2.)["objective"]-qp.fit(fam)["objective"]) <= 1e-8*(1+qp.fit(fam)["objective"])


def _laminar_subsets(proper):
    out = [()]
    for k in range(1, len(proper)+1):
        for S in combinations(proper, k):
            if is_laminar(S):
                out.append(S)
    return out


@pytest.mark.parametrize("c,offsets", [(.75, True), (1., True)])
def test_select_miqp_matches_brute_force(c, offsets):
    s, family = _toy(n=6, seed=5)
    n = 6
    d = rv.Design(s, rv.Transform(*s.p.shape, c, offsets), scale=1e-3)
    pool = canonical([(0, 1), (0, 1, 2), (3, 4), (1, 2), (2, 3), (3, 4, 5), (0, 1, 2, 3)])
    alpha = .01
    best = min(((rv.exact_fit(d, canonical([*basis(n), *S]), 2.)["objective"]+alpha*len(S)), S)
               for S in _laminar_subsets(pool))
    rec = rv.select_miqp(d, pool, basis(n), alpha, 2., 60., warm=None)
    assert rec["status"] == 2
    got = rv.exact_fit(d, rec["family"], 2.)["objective"]+alpha*rv.internal_count(rec["family"], n)
    assert abs(got-best[0]) <= 1e-6
    # tightened bounds and a cutoff at a worse attained value keep the optimum; at the optimum it is rediscovered
    others = sorted(v for v, S in ((rv.exact_fit(d, canonical([*basis(n), *S]), 2.)["objective"]+alpha*len(S), S)
                                   for S in _laminar_subsets(pool)))
    for cutoff in (others[min(3, len(others)-1)], best[0]):
        rec = rv.select_miqp(d, pool, basis(n), alpha, 2., 60., cutoff=cutoff, solver_cutoff=True, seed=7)
        assert rec["status"] == 2 and rec["tightening"] is not None
        got = rv.exact_fit(d, rec["family"], 2.)["objective"]+alpha*rv.internal_count(rec["family"], n)
        assert abs(got-best[0]) <= 1e-6


def test_response_bounds_contain_every_good_family():
    s, family = _toy(n=6, seed=6)
    n = 6
    d = rv.Design(s, rv.Transform(*s.p.shape, .75, True), scale=1e-3)
    pool = canonical([(0, 1), (0, 1, 2), (3, 4), (1, 2), (2, 3), (3, 4, 5)])
    fam = canonical([*pool, *basis(n)])
    B, uy, const, rank = rv._svd_rows(d.columns(fam), d.y)
    alpha = .01
    values = {S: rv.exact_fit(d, canonical([*basis(n), *S]), 2.) for S in _laminar_subsets(pool)}
    scored = sorted(values, key=lambda S: values[S]["objective"]+alpha*len(S))
    cutoff = values[scored[5]]["objective"]+alpha*len(scored[5])
    Rmax, Xmax, info = rv.response_bounds(B, uy, const, fam, n, d.N, cutoff, 2.)
    assert info["tightened"] > 0
    for S in scored[:6]:
        f = canonical([*basis(n), *S]); th = values[S]["theta"]; K = len(f)
        Z = rv.indicators(f, n)
        assert np.all(Z.T@th[:K] <= Rmax+1e-9) and np.all(Z.T@th[K:] <= Xmax+1e-9)
    # a coefficient bound below the per-meter sums: R_ii may exceed the bound (it sums several supports)
    small = .02
    values = {S: rv.exact_fit(d, canonical([*basis(n), *S]), small) for S in _laminar_subsets(pool)}
    scored = sorted(values, key=lambda S: values[S]["objective"]+alpha*len(S))
    cutoff = values[scored[5]]["objective"]+alpha*len(scored[5])
    Rmax, Xmax, info = rv.response_bounds(B, uy, const, fam, n, d.N, cutoff, small)
    exceeded = False
    for S in scored[:6]:
        f = canonical([*basis(n), *S]); th = values[S]["theta"]; K = len(f)
        Z = rv.indicators(f, n)
        exceeded |= bool(np.any(Z.T@th[:K] > small) or np.any(Z.T@th[K:] > small))
        assert np.all(Z.T@th[:K] <= Rmax+1e-9) and np.all(Z.T@th[K:] <= Xmax+1e-9)
    assert exceeded


@pytest.mark.parametrize("c,offsets", [(.75, True), (1., True)])
def test_extend_miqp_matches_brute_force(c, offsets):
    s, family = _toy(n=6, seed=7, family=[(0, 1, 2), (3, 4)])
    n = 6
    d = rv.Design(s, rv.Transform(*s.p.shape, c, offsets), scale=1e-3)
    current = canonical([*basis(n), (3, 4)])
    alpha = .001
    k = rv.internal_count(current, n)
    base = rv.exact_fit(d, current, 2.)["objective"]+alpha*k
    options = [C for size in range(2, n) for C in combinations(range(n), size)
               if C not in current and all(compatible(C, D) for D in current)]
    scores = {C: rv.exact_fit(d, canonical([*current, C]), 2.)["objective"]+alpha*(k+1) for C in options}
    best_C = min(scores, key=scores.get)
    rec = rv.extend_miqp(d, current, alpha, 2., 60.)
    assert rec["status"] == 2
    if scores[best_C] < base-1e-9:
        assert "support" in rec and abs(scores[rec["support"]]-scores[best_C]) <= 1e-6
    else:
        assert "support" not in rec


def test_screen_gain_is_the_one_support_gain():
    s, family = _toy(n=7, seed=9, family=[(0, 1, 2), (3, 4), (0, 1)])
    n = 7
    for c, offsets in ((.8, True), (1., True)):
        d = rv.Design(s, rv.Transform(*s.p.shape, c, offsets), scale=1e-3)
        current = canonical(basis(n))
        fit = rv.exact_fit(d, current, 2.)
        E = d.residual(current, fit["theta"])
        top = rv.gen_screen(current, n, d, E, top=5)
        assert top
        # explicit gain of the best screened support with old coefficients fixed: max over (r, x) >= 0
        gains = {}
        for C in (C for size in range(2, n) for C in combinations(range(n), size)):
            f = d.columns((C,))
            G = f.T@f; g = f.T@E.ravel()
            th = rv.bounded_ls(np.linalg.cholesky(G+1e-15*np.eye(2)).T,
                               np.linalg.solve(np.linalg.cholesky(G+1e-15*np.eye(2)), g), 1e9)
            gains[C] = 2*th@g-th@G@th
        best = max(gains, key=gains.get)
        assert gains[best] <= max(gains[C] for C in top)+1e-9*(1+gains[best])


def test_saturated_sigma2_recovers_the_noise():
    rng = np.random.default_rng(4)
    n, T, sd = 6, 400, 2e-3
    p = rng.uniform(.2, 1., (T, n)); q = .3*p+.05*rng.normal(size=(T, n))
    family = canonical([*basis(n), (0, 1), (0, 1, 2), (3, 4)])
    Z = rv.indicators(family, n)
    r = rng.uniform(.01, .05, len(family))
    y = (p@Z.T*r)@Z+sd*rng.normal(size=(T, n))
    s = rv.Series(p, q, y)
    d = rv.Design(s, rv.Transform(T, n, 0., True), 1.)
    sigma2, info = rv.saturated_sigma2(d)
    assert abs(sigma2/sd**2-1.) < .08
    d1 = rv.Design(s, rv.Transform(T, n, 1., True), 1.)
    sigma2, info = rv.saturated_sigma2(d1)       # projected noise keeps its per-coordinate variance
    assert abs(sigma2/sd**2-1.) < .08


def test_identifiable_count():
    rng = np.random.default_rng(2)
    p = rng.uniform(.2, 1., (100, 5))
    fixed = rv.Series(p, .33*p*(1+.005*rng.normal(size=p.shape)), p)
    free = rv.Series(p, rng.uniform(.05, .3, p.shape), p)
    assert rv.identifiable_per_support(fixed, .005)[0] == 1
    assert rv.identifiable_per_support(free, .005)[0] == 2
    assert rv.identifiable_per_support(fixed, None)[0] == 2


def test_fit_recovers_a_clean_toy_tree():
    s, family = _toy(n=8, T=200, seed=11, family=[(0, 1), (0, 1, 2), (3, 4), (5, 6, 7), (5, 6)], noise=2e-4)
    cfg = rv.RevisedConfig(total_seconds=120., w_mip_seconds=20., e_search_seconds=20., e_cert_seconds=20.)
    out = rv.fit(s, None, cfg)
    truth = {C for C in family if 1 < len(C) < 8}
    got = {C for C in out["selected"] if 1 < len(C) < 8}
    assert got == truth
    assert out["e"]["certified"]


@pytest.mark.parametrize("source", ["gls", "free", "none"])
def test_heuristic_e_runs_without_certificate(source):
    # regression: e="heuristic" (experiment E variants) used to raise UnboundLocalError (verdict)
    s, family = _toy(n=7, T=120, seed=5, noise=3e-4)
    cfg = rv.RevisedConfig(source=source, e="heuristic", total_seconds=60., w_mip_seconds=10., e_search_seconds=10.)
    out = rv.fit(s, None, cfg)
    assert out["e"]["certificates"] == [] and "confirmed" not in out["e"]
    assert {C for C in out["selected"] if 1 < len(C) < 7} == {C for C in family if 1 < len(C) < 7}


def test_free_source_fit_ignores_the_root_reading():
    # with a free source term every per-slot common term is projected out, so the root reading cannot matter:
    # rebuilding y = V0'^2 - V^2 with another root reading V0' gives the same fit
    s, family = _toy(n=7, T=120, seed=6, noise=3e-4)
    rng = np.random.default_rng(1)
    other = s.v0sq*(1.+.01*rng.normal(size=s.v0sq.shape))
    s2 = rv.Series(s.p, s.q, s.y+(other-s.v0sq)[:, None], other)
    cfg = rv.RevisedConfig(source="free", e="heuristic", total_seconds=60., w_mip_seconds=10., e_search_seconds=10.)
    a, b = rv.fit(s, None, cfg), rv.fit(s2, None, cfg)
    assert a["selected"] == b["selected"]
    assert abs(a["score"]["selected_objective"]-b["score"]["selected_objective"]) <= 1e-9
    assert a["score"]["c"] == b["score"]["c"] == 1.


def test_estimated_gamma_follows_the_common_error():
    # the estimated GLS strength recovers rho = common/within and grows with the shared per-slot error
    rng = np.random.default_rng(3)
    T, n = 400, 8
    p = rng.uniform(.2, 1., (T, n)); q = .3*p+.05*rng.normal(size=(T, n))
    family = canonical([*basis(n), (0, 1), (0, 1, 2), (3, 4)])
    Z = rv.indicators(family, n)
    r = rng.uniform(.01, .05, len(family)); x = rng.uniform(.005, .02, len(family))
    clean = (p@Z.T*r)@Z+(q@Z.T*x)@Z
    gammas = []
    for shared in (0., 1e-3, 4e-3):
        y = clean+1e-3*rng.normal(size=(T, n))+shared*rng.normal(size=(T, 1))
        g, info = rv.estimated_gamma(rv.Series(p, q, y, None), offsets=True)
        gammas.append(g)
        assert abs(info["rho"]-(shared/1e-3)**2) <= .25*max((shared/1e-3)**2, 1.)
    assert gammas[0] < .3 < gammas[1] < gammas[2] < 1.
