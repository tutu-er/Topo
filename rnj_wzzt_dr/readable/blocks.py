"""Step 4 of WZZT-C (readable edition): which lines are fixed as blocks. One parameter: m.

After step 3 we have T*, the best tree among all candidates. For each line C of T* we ask how strongly the data
prefer it to its alternatives:
  (a) the same tree without C;
  (b) the same tree with a crossing candidate D in place of C (the lines of T* that D crosses are removed too).

    evidence(C) = smallest score increase over all alternatives, in half-BIC units (a log Bayes factor).

A line becomes a block when evidence(C) >= m. With equal prior odds, m = 4.6 means the data favour C over its
strongest competitor by a Bayes factor of 100 (99%). From the largest lines down: a line that fails is opened and
the lines of T* inside it are tested instead. Meters in no block are single-meter blocks.

The exact root search (step 5) then treats every block as one meter and tries every union of blocks.
"""
import math
from itertools import combinations

import numpy as np

from .data import canonical, compatible


def evidence(problem, tree_lines, candidates):
    """{line: dict(margin, posterior, alternatives)} for every line of tree_lines (the internal lines of T*)."""
    base_score = problem.score(tree_lines)
    out = {}
    for C in tree_lines:
        increases = []
        without = [line for line in tree_lines if line != C]
        increases.append(problem.score(without)-base_score)                  # (a) remove C
        for D in candidates:
            if not compatible(C, D):                                          # (b) put a crossing D in its place
                swapped = [line for line in tree_lines if compatible(line, D)]
                swapped.append(D)
                increases.append(problem.score(swapped)-base_score)
        half_bic = [problem.half_bic(value) for value in increases]
        smallest = min(half_bic)
        total = 1.
        for value in half_bic:
            total += math.exp(-min(max(value, -700.), 700.))
        out[C] = dict(margin=smallest, posterior=1./total, alternatives=len(half_bic))
    return out


def blocks_of(lines, n):
    """The largest of the given lines, plus single meters for the meters outside them."""
    lines = canonical(lines)
    largest = []
    for line in lines:
        inside_another = False
        for other in lines:
            if set(line) < set(other):
                inside_another = True
                break
        if not inside_another:
            largest.append(line)
    covered = set()
    for line in largest:
        covered |= set(line)
    singles = [(i,) for i in range(n) if i not in covered]
    return canonical(largest+singles)


def choose_blocks(tree_lines, evidence_of, m, n, size_limit=1.):
    """Top-down: a line with evidence >= m is a block; a line that fails is opened (its sub-lines are tested)."""
    allowed = [line for line in tree_lines if len(line) <= size_limit*n]
    chosen = []

    def visit(line):
        if evidence_of[line]["margin"] >= m:
            chosen.append(line)
            return
        inside = [other for other in allowed if set(other) < set(line)]
        for sub in blocks_of(inside, n):
            if len(sub) > 1:
                visit(sub)

    top = blocks_of(allowed, n)
    for line in top:
        if len(line) > 1:
            visit(line)
    return blocks_of(chosen, n)


def root_unions(blocks, n):
    """Every union of two or more blocks, except the union of all of them (that is the whole feeder)."""
    out = []
    k = len(blocks)
    for size in range(2, k):
        for picked in combinations(range(k), size):
            members = []
            for b in picked:
                members += list(blocks[b])
            out.append(tuple(sorted(members)))
    return canonical(out)
