"""Supports and families (draft v12, Section I).

A support is a sorted tuple of leaf numbers. Leaves 0..n-1 are the meters; with the PCC power, leaf n is the unmetered
load. A family is a collection of supports; a tree is a laminar family that contains the basis (every leaf alone and the
set of all leaves). Its lines are the supports with 2 to leaves-1 leaves.
"""
from __future__ import annotations

from itertools import combinations


def canonical(family):
    """Sorted tuple of sorted supports, without duplicates."""
    return tuple(sorted({tuple(sorted(int(i) for i in C)) for C in family}, key=lambda C: (len(C), C)))


def basis(leaves):
    return canonical([*((i,) for i in range(leaves)), tuple(range(leaves))])


def compatible(C, D):
    """Nested or disjoint (two lines of one tree are always compatible; crossing lines never are)."""
    a, b = set(C), set(D)
    return not a & b or a <= b or b <= a


def is_line(C, leaves):
    return 1 < len(C) < leaves


def lines_of(family, leaves):
    return canonical(C for C in family if is_line(C, leaves))


def children(family, leaves):
    """{node: its children} for every support of the tree (basis added): the largest supports strictly inside it."""
    nodes = canonical([*family, *basis(leaves)])
    out = {}
    for v in nodes:
        inside = [C for C in nodes if set(C) < set(v)]
        out[v] = canonical(C for C in inside if not any(set(C) < set(D) for D in inside))
    return out


def unions(blocks):
    """Every union of 2 to k-1 of the k blocks (the new lines a node can take, Lemma 2)."""
    k = len(blocks)
    return [tuple(sorted(i for b in c for i in b)) for size in range(2, k) for c in combinations(blocks, size)]


def crossing_map(lines):
    """{line: the lines that cross it}, for local search."""
    lines = canonical(lines)
    return {C: frozenset(D for D in lines if not compatible(C, D)) for C in lines}
