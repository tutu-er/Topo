"""WZZT-C prototype (identification/combo.py): blocks, root unions, and the exact root search against brute force."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from rnj_wzzt_dr.identification import combo, revised as rv, trees
from rnj_wzzt_dr.identification.trees import basis, canonical, compatible


def test_blocks_and_unions():
    n = 8
    fam = canonical([*basis(n), (0, 1), (0, 1, 2), (3, 4), (5, 6, 7), (3, 4, 5, 6, 7)])
    cand, blocks = combo.terminal_blocks(fam, n, limit=.5)       # at most 4 meters
    assert (5, 6, 7) in cand and (3, 4, 5, 6, 7) not in cand
    assert set(blocks) == {(0, 1, 2), (3, 4), (5, 6, 7)}
    unions = combo.root_unions(blocks, n)
    assert len(unions) == 2**3-3-2
    assert all(1 < len(U) < n for U in unions)


def test_margin_blocks_open_weak_lines():
    n = 8
    selected = [(0, 1), (0, 1, 2), (3, 4), (5, 6, 7)]
    margins = {(0, 1): 9., (0, 1, 2): 1., (3, 4): 7., (5, 6, 7): 6.5}
    # (0,1,2) is too weak at m = 6: it is opened and its sub-line (0,1) becomes the block, meter 2 is alone
    assert set(combo.margin_blocks(selected, margins, 6., n, .5)) == {(0, 1), (2,), (3, 4), (5, 6, 7)}
    assert set(combo.margin_blocks(selected, margins, 0., n, .5)) == {(0, 1, 2), (3, 4), (5, 6, 7)}


def _series(seed, n=7, T=96):
    rng = np.random.default_rng(seed)
    p = rng.uniform(.2, 1., (T, n)); q = .3*p+.05*rng.normal(size=(T, n))
    fam = canonical([*basis(n), (0, 1), (0, 1, 2), (3, 4), (3, 4, 5, 6)])
    r = rng.uniform(.01, .05, len(fam)); x = rng.uniform(.005, .02, len(fam))
    Z = rv.indicators(fam, n)
    v0 = 1.02+.003*rng.normal(size=T)
    y = (p@Z.T*r)@Z+(q@Z.T*x)@Z+5e-4*rng.normal(size=(T, n))
    return rv.Series(p, q, y, v0**2), n


def _laminar_subsets(universe):
    out = []
    def grow(i, chosen):
        if i == len(universe):
            out.append(tuple(chosen)); return
        grow(i+1, chosen)
        C = universe[i]
        if all(compatible(C, D) for D in chosen):
            grow(i+1, chosen+[C])
    grow(0, [])
    return out


def test_every_source_gives_a_laminar_hierarchy():
    series, n = _series(4)
    S = combo.setup(series, None)
    for name in combo.SOURCES:
        fam = combo.source_family(S, name)
        assert trees.is_laminar(fam), name
        assert set(basis(n)) <= set(fam), name


def test_margins_are_nonnegative_at_the_selected_optimum():
    series, n = _series(5)
    S = combo.setup(series, None)
    cand = sorted({C for name in ("rnj", "volt_m", "upgma") for C in combo.source_family(S, name) if 1 < len(C) <= n/2})
    base = canonical(basis(n))
    rec = rv.select_miqp(S["design"], cand, base, S["alpha"], 2., 60., gap=1e-9, abs_gap=1e-12)
    kept = [C for C in rec["family"] if 1 < len(C) < n]
    prob = rv.Problem(rv.GramFitter(S["design"], [*cand, *base], base, 2.), cand, base, S["alpha"])
    margins = combo.line_margins(prob, kept, cand, S["N"])
    evidence = combo.line_evidence(prob, kept, cand, S["N"])
    for C, m in margins.items():
        assert m >= -1e-6, (C, m)
        assert abs(evidence[C]["margin"]-m) <= 1e-9*max(1., abs(m))
        # the posterior against all alternatives is at most the one against the strongest alternative
        assert 0. < evidence[C]["posterior"] <= 1./(1.+np.exp(-m))+1e-12


def test_margin_rule_runs_end_to_end():
    series, n = _series(6)
    out = combo.fit(series, None, sources=("rnj", "volt_m", "resid"), blocks_from="margin", m=4.6, repairs=2,
                    node_seconds=20.)
    rec = out["record"]
    assert rec["k"] == len(rec["blocks"]) and "evidence" in rec
    # every block is a line of the joint selection whose margin clears m, or a single meter
    kept = {tuple(l) for l in rec["joint_selection"]["kept"]}
    margin = {tuple(e["line"]): e["margin"] for e in rec["evidence"]}
    for B in map(tuple, rec["blocks"]):
        assert len(B) == 1 or (B in kept and margin[B] >= 4.6)
    # the output is never worse than the joint selection (the root search is warm-started there)
    base = canonical(basis(n))
    S = combo.setup(series, None)
    crit = lambda f: rv.exact_fit(S["design"], f, 2.)["objective"]+S["alpha"]*rv.internal_count(f, n)
    assert rec["criterion"] <= crit(canonical([*base, *kept]))+1e-9
    assert trees.is_laminar(out["selected"])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_root_search_is_the_best_family_in_its_universe(seed):
    series, n = _series(seed)
    out = combo.fit(series, None, repairs=0, node_seconds=20.)
    rec = out["record"]
    S = combo.setup(series, None)
    shared, depth, _ = trees.geometry(S["R"], S["R"], 1.)
    cand, blocks = combo.terminal_blocks(trees.rnj(shared, depth, 0.), n)
    universe = list(canonical([*cand, *combo.root_unions(blocks, n)]))
    assert rec["universe"] == len(universe)
    crit = lambda f: rv.exact_fit(S["design"], canonical([*basis(n), *f]), 2.)["objective"]+S["alpha"]*len(f)
    brute = min(crit(f) for f in _laminar_subsets(universe))
    first = rec["root_miqp"]
    assert first["status"] == 2
    # with no repair the output is the root search's family; it must be the universe's optimum
    assert out["record"]["certificates"][0]["added"] is None or rec["criterion"] <= brute+1e-9
    if out["record"]["certificates"][0]["added"] is None:
        assert abs(rec["criterion"]-brute) <= 1e-7*max(1., abs(brute))
