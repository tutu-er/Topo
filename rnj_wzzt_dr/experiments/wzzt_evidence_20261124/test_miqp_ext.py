"""Experiment M components (miqp_ext): the generalised unknown-support MIQP against brute force."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from itertools import combinations

import numpy as np
import pytest

from rnj_wzzt_dr.identification import miqp_ext as mx
from rnj_wzzt_dr.identification import revised as rv
from rnj_wzzt_dr.identification.trees import basis, canonical, children, compatible


def _design(n=6, T=60, seed=0, family=((0, 1), (0, 1, 2), (3, 4)), noise=5e-4, c=.7, offsets=True):
    rng = np.random.default_rng(seed)
    p = rng.uniform(.2, 1., (T, n))
    q = .3*p+.05*rng.normal(size=(T, n))
    fam = canonical([*basis(n), *family])
    r = rng.uniform(.01, .05, len(fam)); x = rng.uniform(.005, .02, len(fam))
    Z = rv.indicators(fam, n)
    y = (p@Z.T*r)@Z+(q@Z.T*x)@Z+noise*rng.normal(size=(T, n))
    d = rv.Design(rv.Series(p, q, y, None), rv.Transform(T, n, c, offsets), 1.)
    s2, _ = rv.saturated_sigma2(d)
    d.rescale(np.sqrt(s2))
    return d


def _supports(n, family, within=None):
    pool = range(n) if within is None else within
    fam = canonical([*basis(n), *family])
    out = []
    for size in range(2, len(list(pool))):
        for C in combinations(sorted(pool), size):
            if len(C) < n and C not in fam and all(compatible(C, D) for D in fam):
                out.append(C)
    return out


def _brute(design, family, alpha, k, within=None, force=False):
    n = design.n
    fam = canonical([*basis(n), *family])
    proper = sum(1 for C in fam if 1 < len(C) < n)
    cands = _supports(n, family, within)
    best = (np.inf, ())
    options = [()] if not force else []
    options += [(C,) for C in cands] if (k == 1 or not force) else []
    if k == 2:
        options += [(a, b) for a, b in combinations(cands, 2) if compatible(a, b)]
    for add in options:
        f = canonical([*fam, *add])
        obj = rv.exact_fit(design, f, 2.)["objective"]+alpha*(proper+len(add))
        if obj < best[0]:
            best = (obj, add)
    return best


@pytest.mark.parametrize("c", [0., .7, 1.])
def test_k1_matches_revised_extend_and_brute_force(c):
    d = _design(seed=1, c=c, family=((0, 1, 2), (3, 4)))
    fam = canonical([*basis(6), (0, 1, 2)])
    alpha = .2
    mine = mx.extend_k_miqp(d, fam, alpha, 2., 60., k=1)
    ref = rv.extend_miqp(d, fam, alpha, 2., 60.)
    brute, add = _brute(d, [(0, 1, 2)], alpha, 1)
    assert abs(mine["objective"]-brute) <= 1e-5*(1+abs(brute))
    assert abs(ref["objective"]-brute) <= 1e-5*(1+abs(brute))
    assert tuple(mine["supports"]) == add


@pytest.mark.parametrize("c", [0., 1.])
def test_k2_matches_brute_force(c):
    d = _design(n=7, seed=2, c=c, family=((0, 1), (0, 1, 2), (3, 4), (5, 6)))
    for alpha in (.05, 2.):
        mine = mx.extend_k_miqp(d, canonical([*basis(7)]), alpha, 2., 120., k=2)
        brute, add = _brute(d, [], alpha, 2)
        assert abs(mine["objective"]-brute) <= 1e-5*(1+abs(brute))
        assert sorted(mine["supports"]) == sorted(add)


def test_within_and_force_give_the_best_split_inside_a_node():
    d = _design(n=7, seed=3, c=.7, family=((0, 1), (0, 1, 2), (3, 4), (5, 6), (3, 4, 5, 6)))
    fam = [(3, 4, 5, 6)]
    node = (3, 4, 5, 6)
    out = mx.extend_k_miqp(d, canonical([*basis(7), *fam]), 1e3, 2., 60., k=1, within=node, force=True)
    brute, add = _brute(d, fam, 1e3, 1, within=node, force=True)
    assert abs(out["objective"]-brute) <= 1e-5*(1+abs(brute))
    assert tuple(out["supports"]) == add and set(add[0]) < set(node)
    assert out["bound"] <= out["objective"]+1e-9


def test_node_splits_lists_nodes_with_three_or_more_blocks():
    fam = canonical([*basis(7), (0, 1), (2, 3, 4, 5)])
    nodes = dict(mx.node_splits(fam, 7))
    assert nodes == {tuple(range(7)): 3, (2, 3, 4, 5): 4}


def test_within_respects_inner_blocks():
    # node (3,4,5,6) with an inner support (3,4): its splits are unions of the blocks (3,4), 5, 6
    d = _design(n=7, seed=4, c=1., family=((0, 1), (3, 4), (3, 4, 5), (3, 4, 5, 6)))
    fam = [(3, 4, 5, 6), (3, 4)]
    node = (3, 4, 5, 6)
    out = mx.extend_k_miqp(d, canonical([*basis(7), *fam]), 1e3, 2., 60., k=1, within=node, force=True)
    brute, add = _brute(d, fam, 1e3, 1, within=node, force=True)
    assert add[0] in {(3, 4, 5), (3, 4, 6), (5, 6)}
    assert abs(out["objective"]-brute) <= 1e-5*(1+abs(brute)) and tuple(out["supports"]) == add


@pytest.mark.parametrize("c,offsets", [(0., False), (.7, True), (1., True)])
def test_timefree_form_matches_the_svd_form_and_brute_force(c, offsets):
    d = _design(n=7, seed=7, c=c, offsets=offsets, family=((0, 1), (0, 1, 2), (3, 4), (3, 4, 5)))
    fam = canonical([*basis(7), (0, 1, 2)])
    for alpha in (.05, .5):
        tf = mx.extend_tf_miqp(d, fam, alpha, 2., 60., k=1)
        sv = mx.extend_k_miqp(d, fam, alpha, 2., 60., k=1)
        brute, add = _brute(d, [(0, 1, 2)], alpha, 1)
        assert abs(tf["objective"]-brute) <= 1e-5*(1+abs(brute)), (tf["objective"], brute)
        assert abs(sv["objective"]-brute) <= 1e-5*(1+abs(brute))
        assert tuple(tf["supports"]) == add
        assert abs(tf["family_objective"]-sv["family_objective"]) <= 1e-9


def test_timefree_pairs_and_node_certificate_match_brute_force():
    d = _design(n=7, seed=8, c=.8, family=((0, 1), (0, 1, 2), (3, 4), (5, 6), (3, 4, 5, 6)))
    two = mx.extend_tf_miqp(d, canonical(basis(7)), .05, 2., 120., k=2)
    brute, add = _brute(d, [], .05, 2)
    assert abs(two["objective"]-brute) <= 1e-5*(1+abs(brute)) and sorted(two["supports"]) == sorted(add)
    fam = [(3, 4, 5, 6), (3, 4)]
    node = (3, 4, 5, 6)
    out = mx.extend_tf_miqp(d, canonical([*basis(7), *fam]), 1e3, 2., 60., k=1, within=node, force=True)
    brute, add = _brute(d, fam, 1e3, 1, within=node, force=True)
    assert abs(out["objective"]-brute) <= 1e-5*(1+abs(brute)) and tuple(out["supports"]) == add


@pytest.mark.parametrize("mc", ["scaled", "g"])
def test_timefree_linearisations_agree_with_brute_force(mc):
    d = _design(n=7, seed=9, c=.75, family=((0, 1), (0, 1, 2), (3, 4), (3, 4, 5)))
    for fam_extra, alpha in (([], .05), ([(0, 1, 2)], .3)):
        fam = canonical([*basis(7), *fam_extra])
        out = mx.extend_tf_miqp(d, fam, alpha, 2., 60., k=1, mc=mc)
        brute, add = _brute(d, fam_extra, alpha, 1)
        assert abs(out["objective"]-brute) <= 1e-5*(1+abs(brute)) and tuple(out["supports"]) == add
    two = mx.extend_tf_miqp(d, canonical(basis(7)), .05, 2., 120., k=2, mc=mc)
    brute, add = _brute(d, [], .05, 2)
    assert abs(two["objective"]-brute) <= 1e-5*(1+abs(brute)) and sorted(two["supports"]) == sorted(add)


@pytest.mark.parametrize("node,fam_extra", [((3, 4, 5, 6), [(3, 4, 5, 6), (3, 4)]),
                                           (tuple(range(8)), [(0, 1), (2, 3, 4)])])
def test_block_aggregation_gives_the_same_certificate(node, fam_extra):
    d = _design(n=8, seed=10, c=.8, family=((0, 1), (2, 3, 4), (3, 4), (5, 6), (3, 4, 5, 6)))
    fam = canonical([*basis(8), *fam_extra])
    a = mx.extend_tf_miqp(d, fam, 1e3, 2., 60., k=1, within=node, force=True, blocks=True)
    b = mx.extend_tf_miqp(d, fam, 1e3, 2., 60., k=1, within=node, force=True)
    brute, add = _brute(d, fam_extra, 1e3, 1, within=node, force=True)
    assert abs(a["objective"]-brute) <= 1e-5*(1+abs(brute)) and tuple(a["supports"]) == add
    assert abs(b["objective"]-brute) <= 1e-5*(1+abs(brute))
    assert a["blocks"] == len(children(fam, 8)[tuple(sorted(node))])
    free = mx.extend_tf_miqp(d, fam, .05, 2., 60., k=1, within=node, blocks=True)
    brute, add = _brute(d, fam_extra, .05, 1, within=node)
    assert abs(free["objective"]-brute) <= 1e-5*(1+abs(brute))


@pytest.mark.parametrize("compress", [True, False])
def test_compressed_and_auxiliary_objectives_agree(compress):
    d = _design(n=8, seed=11, c=.7, family=((0, 1), (2, 3, 4), (3, 4), (5, 6), (3, 4, 5, 6)))
    fam_extra = [(0, 1), (2, 3, 4)]
    fam = canonical([*basis(8), *fam_extra])
    for node in (tuple(range(8)), (2, 3, 4)):
        a = mx.extend_tf_miqp(d, fam, 1e3, 2., 60., k=1, within=node, force=True, blocks=True, compress=compress)
        brute, add = _brute(d, fam_extra, 1e3, 1, within=node, force=True)
        assert abs(a["objective"]-brute) <= 1e-5*(1+abs(brute)) and tuple(a["supports"]) == add
    g = mx.extend_tf_miqp(d, fam, .05, 2., 60., k=1, compress=compress)
    brute, add = _brute(d, fam_extra, .05, 1)
    assert abs(g["objective"]-brute) <= 1e-5*(1+abs(brute)) and tuple(g["supports"]) == add


@pytest.mark.parametrize("compress", [True, False])
def test_complete_start_is_used_and_does_not_change_the_optimum(compress):
    d = _design(n=7, seed=12, c=.7, family=((0, 1), (0, 1, 2), (3, 4), (3, 4, 5)))
    fam = canonical(basis(7))
    brute, add = _brute(d, [], .05, 1)
    for start in (add[0], (5, 6)):
        out = mx.extend_tf_miqp(d, fam, .05, 2., 60., k=1, starts=[start], compress=compress)
        assert abs(out["objective"]-brute) <= 1e-5*(1+abs(brute)) and tuple(out["supports"]) == add
    lim = mx.extend_tf_miqp(d, fam, .05, 2., 60., k=1, starts=[add[0]], compress=compress, params={"NodeLimit": 0})
    assert lim["objective"] is not None and lim["objective"] <= brute+1e-6*(1+abs(brute))    # the start was accepted
