"""Certificate and completion of the located source tree (S2 of identification/source.py).

A family over n + 1 leaves (leaf n: the source) is *certified* when no single further line pays: for every node, every
union of two or more of its child blocks (the source leaf counts as a block; not all blocks, which is the node itself) is
fitted exactly and priced. A node with more than `max_blocks` child blocks is not enumerated and the tree is then not
certified.

`complete` adds the best paying line to the tree and to the candidate pool and lets the exact pool MIQP, started from that
tree, decide again, until the tree is certified or no line pays. The MIQP re-decides every line in each round, so no line
is fixed by an earlier round; its tree replaces the one with the added line only when its exact score is lower.
"""
from __future__ import annotations

import itertools

from . import revised as rv
from .trees import basis, canonical, children


def certify(design, family, alpha, bound, leaves, max_blocks=12):
    """dict(best, gain, skipped, certified, evaluated) for a family over `leaves` leaves."""
    base = canonical(basis(leaves))
    start = frozenset(C for C in canonical(family) if C not in set(base))
    current = rv.GramFitter(design, [*base, *start], base, bound).fit([*base, *start])[0]+alpha*len(start)
    best, best_gain, skipped, evaluated = None, 0., 0, 0
    for node, blocks in children(canonical([*base, *start]), leaves).items():
        blocks = [tuple(b) for b in blocks]
        if len(blocks) < 3:
            continue
        if len(blocks) > max_blocks:
            skipped += 1
            continue
        unions = [tuple(sorted(i for b in c for i in b)) for k in range(2, len(blocks))
                  for c in itertools.combinations(blocks, k)]
        unions = [U for U in unions if U not in start]
        for k in range(0, len(unions), 200):
            part = unions[k:k+200]
            fitter = rv.GramFitter(design, [*base, *start, *part], base, bound)
            for U in part:
                evaluated += 1
                gain = current-(fitter.fit([*base, *start, U])[0]+alpha*(len(start)+1))
                if gain > best_gain+1e-12:
                    best, best_gain = U, gain
    return dict(best=best, gain=float(best_gain), skipped=skipped, evaluated=evaluated,
                certified=best is None and skipped == 0)


def complete(design, family, lines, alpha, bound, n, *, rounds=10, seconds=120., max_blocks=12, seed=7):
    """Repeat: certify; if a line pays, add it to the tree and to the pool and solve the pool MIQP again, started from
    the tree with the line. The better of that tree and the MIQP's tree (exact score) is kept, so the score falls in
    every round even when the MIQP stops at its time limit. At most `rounds` additions; the last tree is certified too.

    Returns (family over n + 1 leaves, record)."""
    keep = canonical(basis(n+1))
    pool = canonical([*lines, *(C for C in family if 1 < len(C) < n+1)])
    history, certified, family = [], False, canonical(family)

    def score(f):
        return rv.exact_fit(design, f, bound)["objective"]+alpha*rv.internal_count(f, n+1)
    for k in range(rounds+1):
        c = certify(design, family, alpha, bound, n+1, max_blocks)
        history.append(dict(best=None if c["best"] is None else list(c["best"]), gain=c["gain"], skipped=c["skipped"]))
        if c["best"] is None:
            certified = c["certified"]
            break
        if k == rounds:
            break
        pool = canonical([*pool, c["best"]])
        plus = canonical([*family, c["best"]])
        rec = rv.select_miqp(design, [C for C in pool if 1 < len(C) < n+1], keep, alpha, bound, seconds, warm=plus,
                             seed=seed, gap=1e-5)
        found = canonical([*keep, *rec["family"]]) if rec.get("family") else None
        history[-1]["miqp"] = dict(status=rec.get("status"), taken=found is not None and score(found) < score(plus))
        family = found if history[-1]["miqp"]["taken"] else plus
    return family, dict(certified=certified, rounds=history)
