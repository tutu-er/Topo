"""RNJ guidance, original WZZT, bounded-frontier T and one revocable E.

Only training observations enter this module. Every proposed family is closed
by the same bounded QP, numerical zeros are checked by refitting, and the same
regularized objective decides every structural update. There is no small-seed
selection, relative-gain threshold, nearest-neighbour repair window, positive
impedance floor, or output contraction here.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from time import perf_counter

import numpy as np

from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.joint_exit.model import build as build_selection
from research_experiments.local_rebuild.model import canonical, compatible, unions
from research_experiments.miqp_loss_benchmark_20261111 import squared_backend as backend
from research_experiments.rebuild_benchmark.engine import (
    any_extension, hierarchy_children, reconstruction_start,
)
from research_experiments.wzzt_retune.engine import extension_start


ZERO_TOLERANCE = 1e-9
LOSS_TOLERANCE = 1e-10


def basis(n):
    return canonical([*((i,) for i in range(n)), tuple(range(n))])


def _family(value):
    return canonical(value["supports"] if isinstance(value, dict) else value)


def score(point, n, penalty):
    return float(point["objective"]) + float(penalty)*sum(1 < len(s) < n for s in point["supports"])


def clean(bank, point, *, hard=()):
    """Remove numerical-zero provisional supports only after a nonworsening QP."""
    protected = set(basis(bank.data.n)) | set(canonical(hard))
    if not protected <= set(_family(point)):
        raise ValueError("The current family is missing a basis or hard support")
    current, events = point, []
    while True:
        keep = canonical(s for s, r, x in zip(current["supports"], current["r"], current["x"], strict=True)
                         if tuple(s) in protected or max(r, x) > ZERO_TOLERANCE)
        removed = canonical(set(_family(current))-set(keep))
        if not removed:
            return current, events
        proposed = bank.fit(keep, preserve=True)
        accepted = proposed["objective"] <= current["objective"] + LOSS_TOLERANCE
        events.append(dict(removed=removed, accepted=bool(accepted), before=current["objective"],
                           after=proposed["objective"], tolerance=ZERO_TOLERANCE,
                           loss_tolerance=LOSS_TOLERANCE))
        if not accepted:
            return current, events
        current = proposed


def maximum_direct_blocks(family, n):
    return max((len(v) for v in hierarchy_children(family, n).values()), default=1)


def tail_domain(family, n, *, block_limit=5, hard=()):
    """Union of all bounded complete-subtree frontiers, without distance windows.

    Each frontier partitions its parent into current complete subtrees. All
    such frontiers with at most block_limit blocks are included, including
    refinements opening old large-terminal-count supports. An old support is
    releasable iff it occurs as a proper union in one of these frontiers.
    The combined pool can be larger than any individual local neighbourhood.
    """
    if not isinstance(block_limit, int) or block_limit < 3:
        raise ValueError("block_limit must be an integer at least three")
    family = canonical([*family, *basis(n), *hard])
    if not core.is_laminar_family(family):
        raise ValueError("The current hierarchy must be laminar")
    children = hierarchy_children(family, n)

    @lru_cache(None)
    def cuts(node):
        result = {(node,)}
        if children[node]:
            partial = {()}
            for child in children[node]:
                partial = {canonical([*left, *right]) for left in partial for right in cuts(child)
                           if len(left)+len(right) <= block_limit}
                if not partial:
                    break
            result.update(partial)
        return tuple(sorted(result, key=lambda cut: (len(cut), cut)))

    frontiers, generated = [], set()
    for parent in family:
        for blocks in cuts(parent):
            if 3 <= len(blocks) <= block_limit:
                local = canonical(s for _, s in unions(blocks))
                generated.update(local)
                frontiers.append(dict(parent=parent, blocks=blocks, candidate_count=len(local)))
    protected = set(basis(n)) | set(canonical(hard))
    released = canonical((set(family) & generated)-protected)
    keep = canonical(set(family)-set(released))
    pool = canonical([*family, *generated])
    if any(not compatible(s, fixed) for s in pool for fixed in keep if fixed not in protected):
        raise RuntimeError("A frontier union crosses an unopened skeleton support")
    return dict(pool=pool, keep=keep, released=released, candidates=canonical(set(pool)-set(keep)),
                frontiers=frontiers, max_direct_blocks=maximum_direct_blocks(family, n),
                block_limit=block_limit, old_family_recoverable=set(family) <= set(pool))


def _record(stage, current, point, *, bank, penalty, started, solver, zeros, **extra):
    before = score(current, bank.data.n, penalty)
    after = score(point, bank.data.n, penalty)
    accepted = before-after > LOSS_TOLERANCE
    event = dict(stage=stage, accepted=bool(accepted), before_score=before, after_score=after,
                 gain=before-after, penalty=penalty, numerical_threshold=LOSS_TOLERANCE,
                 point=point, solver=solver, zero_events=zeros, seconds=perf_counter()-started,
                 **extra)
    return (point if accepted else current), event


def reconstruct(bank, current, *, pool, keep, reference, deadline, alpha=.003,
                stage="T", warm_families=(), hard=()):
    """Shared T/E selection MIQP; QP-close before the common J acceptance test."""
    started = perf_counter()
    if not np.isfinite(reference) or reference <= 0 or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("A positive reference and nonnegative alpha are required")
    n, penalty = bank.data.n, float(alpha)*float(reference)
    keep = canonical([*keep, *basis(n), *hard])
    pool = canonical([*pool, *keep, *current["supports"]])
    if not set(keep) <= set(_family(current)):
        raise ValueError("A fixed skeleton support is absent from the current family")
    if not core.is_laminar_family(keep):
        raise ValueError("The fixed skeleton must be laminar")
    best, best_zeros = current, []
    # These fits are solver guidance, never separate whole-tree final fallbacks.
    warm_checked = []
    for value in warm_families:
        family = canonical([*_family(value), *basis(n), *hard])
        if not set(keep) <= set(family) or not set(family) <= set(pool):
            continue
        if not core.is_laminar_family(family):
            continue
        if deadline-perf_counter() <= .05:
            break
        q, zero_rows = clean(bank, bank.fit(family, preserve=True), hard=keep)
        warm_checked.append(dict(family=family, score=score(q, n, penalty)))
        if score(q, n, penalty) < score(best, n, penalty):
            best, best_zeros = q, zero_rows
    model, meta = build_selection(bank.scaled, pool, penalty=penalty,
                                  hard=canonical([*keep, *hard]))
    raw_point = None
    seconds = deadline-perf_counter()
    if seconds > .05:
        start = reconstruction_start(bank, model, meta, best)
        values, solver = backend.solve(bank, model, start, seconds=seconds)
        solver.update(solve_started=True, requested_seconds=seconds)
        if values is not None:
            indices = [*range(len(meta["keep"])),
                       *[len(meta["keep"])+k for k, a in enumerate(values[meta["selection"]]) if a > .5]]
            rr, xx = values[model.r], values[model.x]
            violation = max(float(np.max(-rr, initial=0.)), float(np.max(rr-2., initial=0.)),
                            float(np.max(-xx, initial=0.)), float(np.max(xx-2., initial=0.)))
            if violation > 1e-7:
                raise RuntimeError("A structural incumbent exceeds coefficient tolerances")
            raw_point = dict(supports=[meta["all_supports"][i] for i in indices],
                             r=np.clip(rr[indices], 0., 2.).tolist(), x=np.clip(xx[indices], 0., 2.).tolist())
            raw_point["objective"] = backend.loss(bank.data, raw_point)
            closed = bank.fit(raw_point["supports"], preserve=True)
            if closed["objective"] > raw_point["objective"] + 1e-8:
                raise RuntimeError("QP closure worsened the feasible incumbent")
            closed, zero_rows = clean(bank, closed, hard=keep)
            if score(closed, n, penalty) < score(best, n, penalty):
                best, best_zeros = closed, zero_rows
    else:
        solver = dict(solve_started=False, status=None, dual_bound=None, relative_gap=None,
                      reason="structural_deadline", requested_seconds=0.)
    closed, zeros = clean(bank, best, hard=keep)
    # Numerical-zero cleanup protects the T skeleton; E passes only basis/hard.
    best = closed
    return _record(stage, current, best, bank=bank, penalty=penalty, started=started,
                   solver=solver, zeros=[*best_zeros, *zeros], pool=pool, keep=keep,
                   candidates=canonical(set(pool)-set(keep)), raw_point=raw_point,
                   warm_checked=warm_checked, closure_before_acceptance=True,
                   conflicts=meta["conflicts"], cardinality_cuts=meta["cardinality_cuts"],
                   clique_cuts=meta["clique_cuts"])


def forward_step(bank, current, *, rnj_pool, reference, deadline, per_mip=3600., alpha=.003, hard=()):
    """Original unrestricted one-support WZZT domain, with RNJ-only warm guidance."""
    started = perf_counter()
    family, n = _family(current), bank.data.n
    penalty = float(alpha)*float(reference)
    new = any_extension(family, n, rnj_pool)
    if new is None:
        return current, dict(stage="W", accepted=False, structurally_complete=True, seconds=perf_counter()-started,
                             before_score=score(current, n, penalty), after_score=score(current, n, penalty),
                             solver=dict(solve_started=False, status=None), zero_events=[])
    best, best_zeros, warm_checked = current, [], []
    for candidate in canonical(rnj_pool):
        if candidate in family or not all(compatible(candidate, s) for s in family):
            continue
        if deadline-perf_counter() <= .05:
            break
        q, zero_rows = clean(bank, bank.fit([*family, candidate], preserve=True), hard=hard)
        warm_checked.append(dict(support=candidate, score=score(q, n, penalty)))
        if score(q, n, penalty) < score(best, n, penalty):
            best, best_zeros, new = q, zero_rows, candidate
    # The lifted extension start and native model retain the complete current
    # family, even if its QP warm point has just released zero old supports.
    model = core.build_extension_model(bank.scaled, family, r_upper_bound=2., x_upper_bound=2.)
    activation = model.variables.add("activate_new", 1, upper=1., integral=True,
                                     objective=penalty*backend.SCALE)
    for weight in (model.new_r, model.new_x):
        model.rows.add({weight.start: 1., activation.start: -2.}, upper=0.)
    seconds = min(float(per_mip), deadline-perf_counter())
    raw_point = None
    if seconds > .05:
        start = extension_start(model, family, new, best, bank.lp.p, bank.lp.q, bank.lp.y)
        start[activation] = float(new in _family(best))
        values, solver = backend.solve(bank, model, start, seconds=seconds)
        solver.update(solve_started=True, requested_seconds=seconds)
        if values is not None and values[activation.start] > .5:
            candidate = core._round_and_validate_support(values[model.z], family, n)
            closed, zero_rows = clean(bank, bank.fit([*family, candidate], preserve=True), hard=hard)
            raw_point = dict(support=candidate, native_objective=solver.get("objective"))
            if score(closed, n, penalty) < score(best, n, penalty):
                best, best_zeros = closed, zero_rows
    else:
        solver = dict(solve_started=False, status=None, dual_bound=None, relative_gap=None,
                      reason="forward_deadline", requested_seconds=0.)
    best, zeros = clean(bank, best, hard=hard)
    return _record("W", current, best, bank=bank, penalty=penalty, started=started,
                   solver=solver, zeros=[*best_zeros, *zeros], raw_point=raw_point, warm_checked=warm_checked,
                   unrestricted_extension_domain=True, closure_before_acceptance=True)


def run(bank, *, rnj_supports, rnj_warm_families, reference, deadline, per_mip=3600.,
        alpha=.003, block_limit=5, hard=(), heartbeat=None):
    """One W path, at most one T, then one E; never reads validation or truth."""
    started = perf_counter()
    n = bank.data.n
    if not np.isfinite(reference) or reference <= 0 or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("A positive frozen reference and nonnegative alpha are required")
    if per_mip <= 0 or deadline-started <= .05:
        raise ValueError("Positive remaining computation budget is required")
    if not isinstance(block_limit, int) or block_limit < 3:
        raise ValueError("block_limit must be an integer at least three")
    hard = canonical(hard)
    current = bank.fit(canonical([*basis(n), *hard]), preserve=True)
    if not core.is_laminar_family(_family(current)):
        raise ValueError("Incompatible physical hard priors")
    current, initial_zero = clean(bank, current, hard=hard)
    warm_families = tuple(_family(f) for f in rnj_warm_families)
    rnj_pool = canonical([*rnj_supports, *(s for f in warm_families for s in f), *basis(n), *hard])
    path, events, zero_events = [current], [], [dict(stage="initial", **e) for e in initial_zero]
    visited = {_family(current)}
    forward_deadline = perf_counter()+max(0., deadline-perf_counter())/2
    def ping(stage, **extra):
        if heartbeat is not None:
            heartbeat(dict(stage=stage, elapsed_seconds=perf_counter()-started,
                           point=current, path=path, **extra))
    ping("initialized")
    reason = "forward_deadline"
    while forward_deadline-perf_counter() > .05:
        if maximum_direct_blocks(current["supports"], n) <= block_limit:
            reason = "tail_eligible"
            break
        ping("W_started", iteration=len(events))
        proposed, event = forward_step(bank, current, rnj_pool=rnj_pool, reference=reference,
                                       deadline=forward_deadline, per_mip=per_mip, alpha=alpha, hard=hard)
        events.append(event)
        zero_events.extend(dict(stage="W", **e) for e in event["zero_events"])
        if not event["accepted"]:
            reason = "structurally_complete" if event.get("structurally_complete") else "no_gain_or_unresolved"
            ping("W_stopped", event=event)
            break
        if _family(proposed) in visited:
            event.update(accepted=False, numerical_gain_passed=True, rejected_by_visited=True)
            reason = "numerical_repeated_family"
            break
        current = proposed
        visited.add(_family(current)); path.append(current)
        ping("W_complete", event=event)
    tail = dict(pool=(), keep=_family(current), frontiers=[], released=(), candidates=())
    if maximum_direct_blocks(current["supports"], n) <= block_limit and forward_deadline-perf_counter() > .05:
        ping("T_started")
        tail = tail_domain(current["supports"], n, block_limit=block_limit, hard=hard)
        current, event = reconstruct(bank, current, pool=tail["pool"], keep=tail["keep"], reference=reference,
                                     deadline=forward_deadline, alpha=alpha, stage="T", hard=hard)
        event.update(frontiers=tail["frontiers"], released=tail["released"])
        events.append(event)
        zero_events.extend(dict(stage="T", **e) for e in event["zero_events"])
        if event["accepted"]:
            path.append(current)
        ping("T_complete", event=event)
    e_pool = canonical([*rnj_pool, *(s for p in path for s in p["supports"]),
                        *tail["pool"], *combinations(range(n), 2), *basis(n), *hard])
    if deadline-perf_counter() > .05:
        ping("E_started", candidate_pool_count=len(e_pool))
        current, event = reconstruct(bank, current, pool=e_pool, keep=basis(n), reference=reference,
                                     deadline=deadline, alpha=alpha, stage="E", hard=hard,
                                     warm_families=warm_families)
        events.append(event)
        zero_events.extend(dict(stage="E", **e) for e in event["zero_events"])
        if event["accepted"]:
            path.append(current)
        ping("E_complete", event=event)
    else:
        events.append(dict(stage="E", accepted=False, skipped="structural_deadline", zero_events=[],
                           solver=dict(solve_started=False, status=None)))
    return dict(point=current, path=path, events=events, zero_events=zero_events,
                stop_reason=reason, seconds=perf_counter()-started, alpha=float(alpha),
                reference=float(reference), penalty=float(alpha)*float(reference), block_limit=block_limit,
                rnj_pool=rnj_pool, e_pool=e_pool, tail_domain=tail, initial_seeds=[],
                forward_deadline=forward_deadline, hard=hard, tree_global_optimal=False,
                complete_tree_output_fallback=False, output_contraction=None, relative_gain_gate=None,
                truth_used=False, validation_used=False)
