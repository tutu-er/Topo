"""Finite historical candidate additions; no fitting, scoring, or truth input.

``proposals(current, reference, n)`` accepts two support families or point
dicts with a ``supports`` field. It returns only new legal families. Both
original endpoints remain the caller's responsibility and are never changed.
Each added family records all generating origins after exact deduplication.

The symmetric difference's overlap-connected components define disjoint
regions. In each region V, replace every current C subseteq V (including V)
by every reference C subseteq V. Also include the union if it is laminar.
This finite completion is not an exhaustive local search or a WZZT discovery.
"""
from __future__ import annotations

from numbers import Integral


def canonical(family):
    """Sort positional supports, ignoring repeated supports and order."""
    return tuple(sorted({tuple(sorted(set(s))) for s in family},
                        key=lambda s: (len(s), s)))


def compatible(left, right):
    """The original nested-or-disjoint condition."""
    a, b = set(left), set(right)
    return not a & b or a <= b or b <= a


def laminar(family):
    return all(compatible(a, b) for i, a in enumerate(family) for b in family[:i])


def _family(value, n):
    rows = value["supports"] if isinstance(value, dict) else value
    rows = [tuple(s) for s in rows]
    if any(not s or any(not isinstance(i, Integral) or isinstance(i, bool)
                       or not 0 <= i < n for i in s) for s in rows):
        raise ValueError("Nonempty supports with integer positions in [0,n) required")
    family = canonical(rows)
    if not laminar(family):
        raise ValueError("Each historical endpoint must be laminar")
    return family


def proposals(current, reference, n):
    """Return [{family, origin, regions, origins}] without invoking a solver.

    ``origins`` retains alternative constructions of an identical new family.
    Family and region entries are positional index tuples, not terminal labels.
    Endpoints, invalid hybrids and duplicate families are not added again.
    """
    if not isinstance(n, Integral) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    current, reference = _family(current, n), _family(reference, n)
    pending = set(current) ^ set(reference)
    regions = []
    while pending:
        seed = min(pending, key=lambda s: (len(s), s))
        pending.remove(seed)
        region = set(seed)
        while True:
            touching = {s for s in pending if region & set(s)}
            if not touching:
                break
            pending -= touching
            region.update(i for s in touching for i in s)
        regions.append(tuple(sorted(region)))
    regions.sort(key=lambda s: (len(s), s))
    added = {}

    def add(family, kind, changed_regions):
        family = canonical(family)
        if family in (current, reference) or not laminar(family):
            return
        origin = dict(kind=kind, regions=list(changed_regions))
        if family not in added:
            added[family] = dict(family=family, origin=kind,
                                 regions=list(changed_regions), origins=[])
        if origin not in added[family]["origins"]:
            added[family]["origins"].append(origin)

    for region in regions:
        V = set(region)
        family = [s for s in current if not set(s) <= V]
        family += [s for s in reference if set(s) <= V]
        add(family, "single_region_replacement", [region])
    add([*current, *reference], "legal_union", regions)
    return list(added.values())
