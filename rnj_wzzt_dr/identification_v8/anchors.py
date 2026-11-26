"""Anchors, their units and the anchored universe (draft v11, Definition 2 and Proposition 4).

Here everything is over the meters: anchors are meter groupings (sets of 2 to n-1 meters) that are pairwise compatible.
With the load, identify.with_load turns the meter lines into lines with and without the load leaf.
"""
from __future__ import annotations

from itertools import combinations

from .sets import canonical, compatible, is_line, unions


def units(anchors, n):
    """{A: units of A} for every anchor A and for the root (all meters): the largest anchors strictly inside A, and the
    meters of A outside them (as 1-tuples)."""
    anchors = canonical(anchors)
    if not all(is_line(A, n) for A in anchors) or not all(compatible(a, b) for a, b in combinations(anchors, 2)):
        raise ValueError("anchors must be compatible lines")
    out = {}
    for A in [*anchors, tuple(range(n))]:
        inner = [B for B in anchors if set(B) < set(A)]
        top = [B for B in inner if not any(set(B) < set(C) for C in inner)]
        covered = set().union(*map(set, top)) if top else set()
        out[A] = canonical([*top, *((i,) for i in A if i not in covered)])
    return out


def enclosing(C, anchors, n):
    """The smallest anchor strictly containing C, or the root."""
    above = [A for A in anchors if set(C) < set(A)]
    return min(above, key=len) if above else tuple(range(n))


def anchored_universe(anchors, n, pool=(), K=9):
    """Eq. (11): the anchors; every union of 2 to k-1 units of each anchor or root with k <= K units; and the lines of
    `pool` compatible with every anchor whose smallest enclosing anchor (or root) has more than K units.
    Returns (lines, info)."""
    anchors = canonical(anchors)
    out, big, sizes = set(anchors), set(), []
    for A, blocks in units(anchors, n).items():
        sizes.append(len(blocks))
        if len(blocks) <= K:
            out.update(unions(blocks))
        else:
            big.add(A)
    for C in canonical(pool):
        if big and is_line(C, n) and all(compatible(C, A) for A in anchors) and enclosing(C, anchors, n) in big:
            out.add(C)
    out = canonical(C for C in out if is_line(C, n))
    info = dict(anchors=len(anchors), k=sorted(sizes, reverse=True), big_nodes=len(big), lines=len(out),
                unions=sum(2**k-k-2 for k in sizes if k <= K))
    return out, info
