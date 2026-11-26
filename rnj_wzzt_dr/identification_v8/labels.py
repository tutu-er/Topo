"""Labels of meter groupings (draft v11: Definition 1, Lemma 1, eqs 6-8).

With the load as a leaf, one meter grouping M can appear as the line M or as M + load: the same grouping of meters,
with the load inside it or not. Labels and anchors are about groupings: flipping M means a tree with neither of the two.
(Where the load sits is reported separately, by its places and evidence.) Without the load, a grouping is a line.

Units: crit (see model.py); the draft's H = crit N / 2, so the margin m (half-BIC units) is m / (N / 2) in crit.
"""
from __future__ import annotations

import time

from .miqp import tree_miqp
from .model import Fitter
from .sets import canonical, compatible, lines_of

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


def flip_test(model, pool, M, threshold, seconds, seed=7, tighten=True):
    """Definition 1 for grouping M of the tree: the tree MIQP without any line of grouping M, with the threshold as its
    cutoff (and as the level of the per-meter bounds). Decided when nothing is found below it, open when a tree is
    found (the rival), unsettled when the time runs out first."""
    n = model.n
    universe = [C for C in canonical(pool) if grouping(C, n) != M]
    r = tree_miqp(model, universe, seconds, cutoff=threshold, tighten_at=threshold if tighten else None, seed=seed)
    out = dict(status=r["status"], seconds=r["seconds"], low=r["bound"], high=None, rival=None)
    if r["family"] is not None:
        value = Fitter(model, r["family"]).crit(r["family"])
        if value <= threshold+1e-9*(1.+abs(threshold)):
            return dict(out, label=OPEN, high=value, rival=lines_of(r["family"], model.leaves))
    if r["status"] in (3, 6) or (r["bound"] is not None and r["bound"] > threshold):   # 3 infeasible, 6 cut off
        return dict(out, label=DECIDED, low=max(threshold, r["bound"] or threshold))
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
            labels[M] = dict(inside=True, label=OPEN, gap=(None, mm*half), rival=rival, status=None, seconds=0.)
    tests = sorted((M for M in inside if M not in labels), key=lambda M: inside[M][0])
    for k, M in enumerate(tests):
        r = flip_test(model, pool, M, threshold, seconds, seed, tighten)
        low = None if r["low"] is None else (r["low"]-s0)*half
        high = inside[M][0]*half if r["high"] is None else (r["high"]-s0)*half
        labels[M] = dict(inside=True, label=r["label"], gap=(low, high), rival=r["rival"], status=r["status"],
                         seconds=r["seconds"])
        if log:
            log(f"    flip {k+1}/{len(tests)} |M|={len(M)} local {inside[M][0]*half:.1f}: {r['label']} "
                f"(status {r['status']}, {r['seconds']:.1f} s)")
    for M, (ell, rival) in outside.items():
        if -ell*half <= m:
            labels[M] = dict(inside=False, label=OPEN, gap=(None, -ell*half), rival=rival, status=None, seconds=0.)
    info = dict(crit=s0, tested=len(tests), outside=len(outside), seconds=time.perf_counter()-started)
    return labels, info
