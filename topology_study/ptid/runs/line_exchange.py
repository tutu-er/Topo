"""One-edge exchanges, following a frozen anchored estimate.

For every non-anchor line, contract it, then regroup 2..d-1 children of its
former parent. Reinstating the old line is always feasible. Each position
is evaluated against the same input tree; at most one exchange is applied.
This experiment uses meter-only trees. All public costs are H = N*crit/2.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import combinations, islice
from math import isfinite
from time import perf_counter

import gurobipy as gp
import numpy as np

from ptid.method.model import Fitter
from ptid.method.search import PRICING_TOL, node_miqp
from ptid.method.sets import canonical, children, compatible, lines_of


def _union(kids, index_set):
    return tuple(sorted(x for i in index_set for x in kids[i]))


class _Scores:
    """Refit all line coefficients at each candidate; reuse Gram matrices in batches."""

    def __init__(self, model, kept, removed, initial):
        self.model = model
        self.kept = kept
        self.values = {removed: initial}

    def evaluate(self, groups):
        stream = iter(groups)
        while part := list(islice(stream, 128)):
            new = [g for g in dict.fromkeys(part) if g not in self.values]
            if new:
                fit = Fitter(self.model, [*self.kept, *new])
                for g in new:
                    self.values[g] = fit.crit([*self.kept, g])


def _greedy(kids, removed, scores):
    indices = tuple(range(len(kids)))
    pairs = list(combinations(indices, 2))
    scores.evaluate(_union(kids, pair) for pair in pairs)
    best_pair = min(pairs, key=lambda pair: scores.values[_union(kids, pair)])
    old_indices = frozenset(i for i, child in enumerate(kids) if set(child) <= set(removed))
    # Two deterministic starts. Cache all fitted unions across the paths.
    for selected in dict.fromkeys((old_indices, frozenset(best_pair))):
        while True:
            neighbours = []
            for i in indices:
                trial = selected ^ {i}
                if 2 <= len(trial) < len(kids):
                    neighbours.append((trial, _union(kids, trial)))
            scores.evaluate(g for _, g in neighbours)
            if not neighbours:
                break
            next_set, line = min(neighbours, key=lambda p: (scores.values[p[1]], p[1]))
            if scores.values[line] >= scores.values[_union(kids, selected)] - 1e-12:
                break
            selected = next_set


def _fingerprint(model, tree):
    h = sha256(repr((tree, model.gamma, model.alpha, model.sigma2, model.yy)).encode())
    for a in (model.Pt, model.Qt, model.Yt):
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _position(model, tree, removed, parent, kids, method, seconds, K, seed,
              initial, fingerprint, previous=None):
    start = perf_counter()
    kept = tuple(x for x in tree if x != removed)
    factor = model.N / 2
    scores = _Scores(model, kept, removed, initial)
    if previous is not None:
        scores.evaluate([tuple(previous["added"])])
    width = len(kids)
    candidate_count = 2**width-width-2
    solver_status = None
    solver_value = None
    solver_bound = None
    raw_candidate = None
    if method == "miqp" and width > K:
        backend = "node_miqp"
        raw_candidate, solver_value, solver_status, solver_bound = node_miqp(
            model, kept, kids, seconds, seed=seed, return_bound=True)
        if raw_candidate is not None:
            scores.evaluate([raw_candidate])
        status = {gp.GRB.OPTIMAL: "optimal", gp.GRB.TIME_LIMIT: "time_limit",
                  gp.GRB.NUMERIC: "numeric"}.get(solver_status, str(solver_status))
    elif method == "greedy":
        backend = "greedy"
        _greedy(kids, removed, scores)
        status = "greedy_finished"
    else:
        backend = "enumeration"
        scores.evaluate(_union(kids, subset) for k in range(2, width)
                        for subset in combinations(range(width), k))
        if len(scores.values) != candidate_count:
            raise RuntimeError("Incomplete fixed-position enumeration")
        status = "exhausted"
    # Equal-scoring candidates retain the incumbent. The same acceptance
    # threshold is used for both search methods when choosing the final tree.
    added = min(scores.values, key=lambda g: (scores.values[g], g != removed, g))
    value = scores.values[added]
    if backend == "enumeration":
        lower = value
    else:
        lower = solver_bound
        if lower is not None and (not isfinite(lower) or lower > value+PRICING_TOL):
            status, lower = "invalid_bound", None
        if previous is not None and previous["bound_H"] is not None:
            prior = previous["bound_H"] / factor
            if prior <= value + PRICING_TOL:
                lower = max(prior, lower) if lower is not None else prior
        if lower is not None:
            lower = min(value, lower)
    certified = lower is not None and value-lower <= max(1e-7, 1e-4*abs(value))
    if solver_status == gp.GRB.OPTIMAL and status == "optimal" and not certified:
        status = "optimal_with_refit_gap"
    elapsed = perf_counter()-start
    prior_seconds = previous.get("cumulative_seconds", previous["seconds"]) if previous else 0.
    return dict(removed=removed, parent=parent, children=kids, width=width, method=method,
                backend=backend, added=added, H=value*factor, initial_H=initial*factor,
                gain_H=(initial-value)*factor, bound_H=None if lower is None else lower*factor,
                solver_status=solver_status, solver_objective_H=None if solver_value is None else solver_value*factor,
                solver_bound_H=None if solver_bound is None else solver_bound*factor, solver_candidate=raw_candidate,
                status=status, certified=certified, candidate_count=candidate_count,
                evaluated=len(scores.values), seconds=elapsed, cumulative_seconds=prior_seconds+elapsed,
                node_seconds=seconds,
                fingerprint=fingerprint, K=K, seed=seed)


def _inputs(model, tree, anchors, method, node_seconds, K):
    if model.n != model.leaves:
        raise ValueError("Line exchange currently requires a meter-only model")
    if method not in ("miqp", "greedy", "enumerate") or K < 2 or node_seconds <= 0:
        raise ValueError("Invalid method, K or solver budget")
    tree = lines_of(tree, model.leaves)
    if any(not compatible(a, b) for a, b in combinations(tree, 2)):
        raise ValueError("Input tree contains crossing groups")
    if not set(canonical(anchors)).issubset(tree):
        raise ValueError("All required anchors must belong to the input tree")
    return tree


def _positions(tree, leaves, anchors):
    at = children(tree, leaves)
    parents = {child: parent for parent, group in at.items() for child in group}
    for removed in tree:
        if removed not in anchors:
            parent = parents[removed]
            kids = canonical([*at[removed], *(g for g in at[parent] if g != removed)])
            yield removed, parent, kids


def evaluate_position(model, tree, removed, method, *, node_seconds=60, K=9, seed=7):
    """Solve one contraction, specified by its original line, with no truth input."""
    tree = _inputs(model, tree, (), method, node_seconds, K)
    removed = tuple(sorted(removed))
    for edge, parent, kids in _positions(tree, model.leaves, set()):
        if edge == removed:
            initial = Fitter(model, tree).crit(tree)
            return _position(model, tree, edge, parent, kids, method, node_seconds, K,
                             seed, initial, _fingerprint(model, tree))
    raise ValueError("The requested line does not belong to the input tree")


def enumerate_position(model, tree, removed):
    """Fit every legal union at one contraction, including the original line."""
    return evaluate_position(model, tree, removed, "enumerate")


def exchange(model, tree, anchors, method, *, node_seconds=60, K=9, seed=7,
             cached=None, on_result=None):
    """Choose the best single change across all non-anchor contractions.

    cached maps removed-line tuples to rows from an earlier call. Rows from
    a different model/tree, method or seed are rejected. A larger budget on
    an open MIQP re-solves the position and retains the older feasible tree.
    on_result receives every newly computed row for external checkpointing.
    """
    start = perf_counter()
    anchors = set(canonical(anchors))
    tree = _inputs(model, tree, anchors, method, node_seconds, K)
    factor = model.N / 2
    initial = Fitter(model, tree).crit(tree)
    fingerprint = _fingerprint(model, tree)
    results = []
    reused = 0
    for removed, parent, kids in _positions(tree, model.leaves, anchors):
        row = (cached or {}).get(removed)
        if row is not None:
            if (row["fingerprint"] != fingerprint or row["method"] != method
                    or row["K"] != K or row["seed"] != seed
                    or tuple(row["removed"]) != removed or canonical(row["children"]) != kids):
                raise ValueError("Incompatible saved contraction")
            if row["backend"] == "node_miqp" and row["node_seconds"] > node_seconds:
                raise ValueError("A longer saved solve cannot represent a shorter time budget")
            if row["backend"] != "node_miqp" or row["node_seconds"] == node_seconds or row["certified"]:
                results.append(row)
                reused += 1
                continue
        row = _position(model, tree, removed, parent, kids, method, node_seconds, K,
                        seed, initial, fingerprint, previous=row)
        results.append(row)
        if on_result is not None:
            on_result(row)
    winner = min(results, key=lambda r: r["H"], default=None)
    output, value, out, new = tree, initial*factor, None, None
    if winner is not None and value-winner["H"] > PRICING_TOL*factor:
        out, new = tuple(winner["removed"]), tuple(winner["added"])
        output = canonical([*(g for g in tree if g != out), new])
        value = winner["H"]
    lowers = [r["bound_H"] for r in results]
    lower = min([initial*factor, *lowers]) if all(b is not None for b in lowers) else None
    return dict(tree=output, initial_H=initial*factor, H=value, gain_H=initial*factor-value,
                removed=out, added=new, positions=results, seconds=perf_counter()-start,
                position_seconds=sum(r["seconds"] for r in results), reused=reused,
                bound_H=lower, certified=lower is not None and value-lower <= max(1e-7*factor,1e-4*abs(value)))
