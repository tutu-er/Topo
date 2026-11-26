"""Likely lines: candidate lines left out of a returned tree whose evidence misses the line price by less than m units.

A reporting output computed after the tree is chosen; it never changes the tree. For every candidate line C of the
method's own candidate families (final.SOURCES) that is not in the tree T, the tree is changed to U = (lines of T compatible
with C) + C, and the evidence of C is half N times (score(T) - score(U)), in half-BIC units, under the same prices as the
tree (the structural prior included). It is at most 0 when T is optimal against such swaps. A line is listed when its
evidence is above -m, with m = 4.6 (Bayes factor 100), the method's own threshold.
"""
from __future__ import annotations

from . import combo, revised as rv
from .final import SOURCES
from .trees import basis, canonical, compatible


def likely_lines(S, tree, n, m=4.6, bound=2.):
    """[(line, evidence)] sorted by evidence (largest first) for the excluded candidate lines with evidence above -m."""
    inner = canonical([C for C in tree if 1 < len(C) < n])
    lines = canonical([C for s in SOURCES for C in combo.source_family(S, s) if 1 < len(C) < n] + list(inner))
    base = canonical(basis(n))
    prob = rv.Problem(rv.GramFitter(S["design"], [*lines, *base], base, bound), lines, base, S["alpha"])
    T = frozenset(inner)
    s0, half = prob.score(T), .5*S["N"]
    out = []
    for C in prob.proper:
        if C in T:
            continue
        ev = half*(s0-prob.score(frozenset(E for E in T if compatible(E, C)) | {C}))
        if ev > -m:
            out.append((C, float(ev)))
    return sorted(out, key=lambda t: -t[1])
