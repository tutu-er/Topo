"""Local search and node pricing (draft v8: the incumbent of eq. (11), and eq. (12) by enumeration)."""
from __future__ import annotations

import time

from .model import Fitter
from .sets import basis, canonical, children, lines_of, unions


def local_search(fitter, crosses, start, deadline, require=()):
    """Best-improvement search over trees made of the lines in `crosses` (a crossing_map). A move removes one line, or
    adds one and removes the lines it crosses; it must keep a line of every `require` group. Returns (lines, crit)."""
    ok = lambda S: all(any(C in S for C in group) for group in require)
    S = frozenset(C for C in start if C in crosses)
    best = fitter.crit(S)
    while time.perf_counter() < deadline:
        move, move_value = None, best-1e-12
        for C in crosses:
            T = S-{C} if C in S else (S-crosses[C]) | {C}
            if ok(T):
                value = fitter.crit(T)
                if value < move_value:
                    move, move_value = T, value
            if time.perf_counter() > deadline:
                break
        if move is None:
            break
        S, best = move, move_value
    return S, best


def best_addition(model, tree, max_children=12, chunk=200):
    """Lemma 4 and eq. (12): every line compatible with the tree is a union of 2 to c-1 children of one node; all of
    them are fitted (nodes with more than `max_children` children are skipped). Returns (best line or None, info)."""
    leaves = model.leaves
    tree = lines_of(tree, leaves)
    family = canonical([*basis(leaves), *tree])
    current = Fitter(model, family).crit(tree)
    best, gain, skipped, evaluated = None, 0., 0, 0
    for node, kids in children(family, leaves).items():
        if len(kids) < 3:
            continue
        if len(kids) > max_children:
            skipped += 1
            continue
        new = [U for U in unions(kids) if U not in set(family)]
        for k in range(0, len(new), chunk):
            part = new[k:k+chunk]
            fit = Fitter(model, [*family, *part])
            for U in part:
                evaluated += 1
                g = current-fit.crit([*tree, U])
                if g > gain+1e-12:
                    best, gain = U, g
    return best, dict(gain=gain, skipped=skipped, evaluated=evaluated, certified=best is None and skipped == 0)
