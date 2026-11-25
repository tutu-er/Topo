"""The readable edition reproduces the production code, piece by piece.

    pytest rnj_wzzt_dr/readable/test_readable.py
"""
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pytest

from rnj_wzzt_dr.identification import combo, revised as rv, trees
from rnj_wzzt_dr.readable import blocks as rb, data as rd, selection as rs, sources as rsrc
from rnj_wzzt_dr.readable.settings import Settings
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124.test_tree_builders import _metric, _random_tree
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124.test_combo import _laminar_subsets, _series


def _real_input():
    from rnj_wzzt_dr.readable.run_example import input_arrays
    return input_arrays("eulv", "eulv_td008742_feeder79_s0")


@pytest.fixture(scope="module")
def real():
    p, q, y, v0sq, truth = _real_input()
    problem = rd.Problem(p, q, y, v0sq, Settings())
    S = combo.setup(rv.Series(p, q, y, v0sq), None)
    return problem, S


@pytest.mark.parametrize("seed", range(8))
def test_distance_builders_match_production(seed):
    shared, depth, _ = _metric(*_random_tree(seed))
    noisy = shared+np.random.default_rng(seed).normal(0, .02, shared.shape)
    noisy = .5*(noisy+noisy.T)
    for tol in (0., .08):
        for readable, production in ((rsrc.rnj, trees.rnj), (rsrc.nj, trees.nj), (rsrc.rg, trees.rg)):
            for metric in (shared, noisy):
                assert rd.canonical(readable(metric, depth, tol)) == rd.canonical(production(metric, depth, tol))


def test_problem_matches_production_setup(real):
    problem, S = real
    assert abs(problem.c-S["c"]) <= 1e-12
    assert abs(problem.sigma2-S["sigma2"]) <= 1e-9*S["sigma2"]
    assert abs(problem.alpha-S["alpha"]) <= 1e-15 and problem.p_s == S["p_s"]


def test_fit_matches_production(real):
    problem, S = real
    n = problem.n
    for tree in ([], [(0, 1)], [(0, 1), (0, 1, 2)], [tuple(range(n//2))]):
        fam = rd.canonical(list(tree)+rd.basic_lines(n))
        _, objective = problem.fit(fam)
        reference = rv.exact_fit(S["design"], trees.canonical(fam), 2.)["objective"]
        assert abs(objective-reference) <= 1e-7*reference


def test_fitted_matrix_matches_production(real):
    problem, S = real
    R = rsrc.fitted_matrix(problem)
    assert np.allclose(R, S["R"], rtol=1e-9, atol=1e-12*np.abs(S["R"]).max())


def test_every_source_matches_production(real):
    problem, S = real
    R = rsrc.fitted_matrix(problem)
    for name in combo.SOURCES:
        mine = set(rsrc.source_lines(problem, name, R))
        theirs = {C for C in combo.source_family(S, name) if 1 < len(C) < problem.n}
        assert mine == theirs, name


def test_best_tree_is_the_brute_force_optimum():
    series, n = _series(0)
    problem = rd.Problem(series.p, series.q, series.y, series.v0sq, Settings())
    R = rsrc.fitted_matrix(problem)
    universe = sorted({C for name in ("rnj", "volt_m") for C in rsrc.source_lines(problem, name, R)})
    tree, info = rs.best_tree(problem, universe, gap=1e-9)
    assert info["status"] == 2
    brute = min(problem.score(list(f)) for f in _laminar_subsets(universe))
    assert abs(problem.score(tree)-brute) <= 1e-7*abs(brute)


def test_evidence_matches_production(real):
    problem, S = real
    n = problem.n
    R = rsrc.fitted_matrix(problem)
    cand = sorted({C for name in combo.SOURCES for C in rsrc.source_lines(problem, name, R)})
    tree, _ = rs.best_tree(problem, cand)
    kept = rd.internal(tree, n)
    mine = rb.evidence(problem, kept, cand)
    base = trees.canonical(trees.basis(n))
    prob = rv.Problem(rv.GramFitter(S["design"], [*cand, *base], base, 2.), cand, base, S["alpha"])
    theirs = combo.line_evidence(prob, kept, cand, S["N"])
    for C in kept:
        assert abs(mine[C]["margin"]-theirs[C]["margin"]) <= 1e-4*max(1., abs(theirs[C]["margin"]))
        assert mine[C]["alternatives"] == theirs[C]["alternatives"]
