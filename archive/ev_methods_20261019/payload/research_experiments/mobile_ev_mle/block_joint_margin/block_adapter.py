"""Observation-free structural summary of a replaceable candidate bank.

This adapter uses topology supports only.  Its block summary adds no
likelihood, calibrated confidence, source estimate, or electrical gradient.
All candidate comparisons and all declared accessible sites are retained.
"""
from __future__ import annotations

from itertools import permutations
from numbers import Integral


def _indices(values, n, label):
    values = tuple(values)
    if any(isinstance(j, bool) or not isinstance(j, Integral) for j in values):
        raise ValueError(f"{label} must contain integer terminal indices")
    result = tuple(sorted(set(int(j) for j in values)))
    if any(j < 0 or j >= n for j in result):
        raise ValueError(f"{label} contains an out-of-range terminal")
    return result


def _family(candidate, n):
    supports = []
    for raw in candidate.supports:
        support = _indices(raw, n, "support")
        if not support:
            raise ValueError("supports must be nonempty")
        supports.append(frozenset(support))
    for left in supports:
        for right in supports:
            if left & right and not (left <= right or right <= left):
                raise ValueError("crossing supports do not define a radial tree")
    # RadialCandidate includes the root/full-set edge and terminal singletons.
    # They are implicit in reduced support families and cannot be ambiguous.
    return frozenset(tuple(sorted(s)) for s in supports if 1 < len(s) < n)


def _serial_family(family):
    return [list(s) for s in sorted(family, key=lambda s: (len(s), s))]


def build_block_adapter(candidates, accessible_nodes=None):
    """Summarize common/disputed clades without pruning candidates or actions.

    ``candidates`` supplies RadialCandidate-compatible ``name``, ``n`` and
    ``supports`` attributes.  ``accessible_nodes`` contains zero-based EV
    terminal indices; None means all terminals and an empty list means none.
    Pair-specific disputed nodes are the union of the symmetric-difference
    supports.  They locate a structural disagreement, not an exact electrical
    boundary: full AC scoring must still include the root/source and outside
    channels.  Distinct candidate names with identical structure are retained
    and explicitly marked as such.
    """
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("candidate bank must be nonempty")
    n = candidates[0].n
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 2:
        raise ValueError("candidate terminal count must be an integer >= 2")
    n = int(n)
    if any(isinstance(c.n, bool) or not isinstance(c.n, Integral) or c.n != n
           for c in candidates):
        raise ValueError("all candidates must have the same terminal count")
    names = [c.name for c in candidates]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("candidate names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")

    families = tuple(_family(c, n) for c in candidates)
    common = frozenset.intersection(*families)
    disputed = frozenset.union(*families) - common
    disputed_nodes = sorted({j for support in disputed for j in support})
    accessible = (tuple(range(n)) if accessible_nodes is None else
                  _indices(accessible_nodes, n, "accessible_nodes"))
    pairs = []
    for a, b in permutations(range(len(candidates)), 2):
        difference = families[a] ^ families[b]
        pairs.append(dict(reference=names[a], competitor=names[b],
                          reference_index=a, competitor_index=b,
                          disputed_supports=_serial_family(difference),
                          disputed_nodes=sorted({j for s in difference for j in s}),
                          structurally_identical=not difference))

    n3_complete = (n == 3 and set(families) == {
        frozenset(), frozenset({(0, 1)}), frozenset({(0, 2)}),
        frozenset({(1, 2)})})
    return dict(schema="candidate_block_adapter_v1", n_terminal=n,
                candidates=[dict(name=name, supports=_serial_family(family))
                            for name, family in zip(names, families)],
                common_supports=_serial_family(common),
                disputed_supports=_serial_family(disputed),
                disputed_nodes=disputed_nodes,
                accessible_nodes=list(accessible),
                accessible_disputed_nodes=sorted(set(accessible) & set(disputed_nodes)),
                directed_pairs=pairs,
                n3_complete_reduced_bank=n3_complete,
                extra_likelihood=False,
                adds_information_beyond_candidate_bank=False,
                uses_measurements=False,
                estimates_source=False,
                restrict_actions=False,
                candidate_pruning=False,
                block_scope="candidate support disagreement, not an AC isolation boundary")


__all__ = ["build_block_adapter"]
