"""Labels of meter groupings (draft v17, Definition 1, Lemma 1, eqs 6-8).

With the load as a leaf, one meter grouping M can appear as the line M or as M + load: the same grouping of meters,
with the load inside it or not. Labels and anchors are about groupings: flipping M means a tree with neither of the two.
(Where the load sits is reported separately, by its places and evidence.) Without the load, a grouping is a line.

Units: crit (see model.py); the draft's H = crit N / 2, so the margin m (half-BIC units) is m / (N / 2) in crit.
"""
from __future__ import annotations

from itertools import combinations
import math
import time

import gurobipy as gp

from .miqp import tree_miqp
from .model import Fitter
from .sets import basis, canonical, compatible, lines_of

DECIDED, OPEN, UNSETTLED = "decided", "open", "unsettled"


def grouping(C, n):
    """The meter grouping that a line expresses (its meters), or None for a line with fewer than 2 or all n meters."""
    M = tuple(i for i in C if i < n)
    return M if 1 < len(M) < n else None


def local_flips(fitter, tree, pool, n):
    """Eq. (8) for groupings, in crit units.

    inside[M] = (m_M, rival): the best of removing every line of grouping M, and of swapping in a pool line whose
    grouping crosses M (with the tree's lines that cross it removed).
    outside[M] = (ell_M, rival): the best swap of a pool line whose grouping is not in the tree."""
    T0 = frozenset(tree)
    s0 = fitter.crit(T0)
    groups = {}
    for C in T0:
        if grouping(C, n):
            groups.setdefault(grouping(C, n), []).append(C)
    swaps = {}
    for D in canonical(pool):
        if D not in T0 and grouping(D, n):
            T = frozenset(E for E in T0 if compatible(E, D)) | {D}
            swaps[D] = (fitter.crit(T), T)
    inside = {}
    for M, lines in groups.items():
        T = T0-set(lines)
        best = (fitter.crit(T), T)
        for D, (value, T2) in swaps.items():
            if value < best[0] and not compatible(grouping(D, n), M):
                best = (value, T2)
        inside[M] = (best[0]-s0, best[1])
    outside = {}
    for D, (value, T) in swaps.items():
        M = grouping(D, n)
        if M not in groups and (M not in outside or s0-value > outside[M][0]):
            outside[M] = (s0-value, T)
    return inside, outside, s0


def flip_test(model, pool, M, threshold, seconds, seed=7, tighten=True, inside=True, known=None):
    """Definition 1 for grouping M (eq. 6): the best tree that decides M the other way, by the tree MIQP with the
    threshold as its cutoff (and as the level of the per-meter bounds). For M in the tree (inside), no line of grouping
    M is allowed; for M outside the tree, a line of grouping M is required. Decided needs threshold exclusion;
    open needs a verified rival at or below the threshold. Other comparisons remain unsettled.
    low is a lower bound on the original flip problem in crit units; threshold_excluded records the
    cutoff proof independently. A feasible local flip above the cutoff can coexist with a cutoff proof."""
    n = model.n
    if inside:
        universe, require = [C for C in canonical(pool) if grouping(C, n) != M], ()
    else:
        universe = canonical(pool)
        require = ([C for C in universe if grouping(C, n) == M],)
        if not require[0]:
            raise ValueError(f"no line of grouping {M} in the pool")
    r = tree_miqp(model, universe, seconds, require=require, cutoff=threshold,
                  tighten_at=threshold if tighten else None, known=known, seed=seed)
    out = dict(status=r["status"], seconds=r["seconds"], low=r["bound"], high=None, rival=None,
               threshold_excluded=r["threshold_excluded"])
    if r["family"] is not None:
        family, base = set(r["family"]), set(basis(model.leaves))
        rival = canonical(family-base)
        has_group = any(grouping(C, n) == M for C in rival)
        if (not base <= family or not set(rival).issubset(universe) or has_group == inside
                or any(not compatible(C, D) for C, D in combinations(rival, 2))):
            return dict(out, label=UNSETTLED, status=gp.GRB.NUMERIC, low=None, threshold_excluded=False)
        value = Fitter(model, rival).crit(rival)
        if not math.isfinite(value):
            return dict(out, label=UNSETTLED, status=gp.GRB.NUMERIC, low=None, threshold_excluded=False)
        out.update(high=value, rival=rival)
        within = value <= threshold+1e-9*(1.+abs(threshold))
        bound_conflict = out["low"] is not None and out["low"] > value+1e-5*(1.+abs(value))
        if (within and out["threshold_excluded"]) or bound_conflict:
            out.update(status=gp.GRB.NUMERIC, low=None, threshold_excluded=False)
        elif out["low"] is not None:
            out["low"] = min(out["low"], value)
        if within:
            return dict(out, label=OPEN)
    if out["threshold_excluded"]:
        return dict(out, label=DECIDED)
    return dict(out, label=UNSETTLED)


def label_groupings(model, tree, pool, m, seconds, seed=7, log=None, tighten=True):
    """Labels relative to the tree. Groupings of the tree: open when a local flip is within m (Lemma 1), otherwise the
    flip test. Groupings outside: open when swapping one in costs at most m; the others are not labelled.
    Returns (labels, info); labels[M] = dict(inside, label, gap=(low, high) in half-BIC units, rival, status, seconds)."""
    half = model.N/2
    tree = lines_of(tree, model.leaves)
    pool = canonical([*pool, *tree])
    started = time.perf_counter()
    inside, outside, s0 = local_flips(Fitter(model, pool), tree, pool, model.n)
    threshold = s0+m/half
    labels = {}
    for M, (mm, rival) in inside.items():
        if mm*half <= m:
            labels[M] = dict(inside=True, label=OPEN, gap=(None, mm*half), rival=rival, status=None, seconds=0.,
                             threshold_excluded=False)
    tests = sorted((M for M in inside if M not in labels), key=lambda M: inside[M][0])
    for k, M in enumerate(tests):
        r = flip_test(model, pool, M, threshold, seconds, seed, tighten, known=s0+inside[M][0])
        low = None if r["low"] is None else (r["low"]-s0)*half
        high_gap, rival = inside[M]
        if r["high"] is not None and r["high"]-s0 < high_gap:
            high_gap, rival = r["high"]-s0, r["rival"]
        labels[M] = dict(inside=True, label=r["label"], gap=(low, high_gap*half), rival=rival, status=r["status"],
                         seconds=r["seconds"], threshold_excluded=r["threshold_excluded"])
        if log:
            log(f"    flip {k+1}/{len(tests)} |M|={len(M)} local {inside[M][0]*half:.1f}: {r['label']} "
                f"(status {r['status']}, {r['seconds']:.1f} s)")
    for M, (ell, rival) in outside.items():
        if -ell*half <= m:
            labels[M] = dict(inside=False, label=OPEN, gap=(None, -ell*half), rival=rival, status=None, seconds=0.,
                             threshold_excluded=False)
    info = dict(crit=s0, tested=len(tests), outside=len(outside), seconds=time.perf_counter()-started)
    return labels, info
