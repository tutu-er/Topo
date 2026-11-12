"""Independent structural switches on the original WZZT mainline.

R enables the temporary small RNJ seeds. T replaces the last one-at-a-time
extensions by the existing <=5-direct-block joint model. E, handled by
``finish``, performs exactly one revocable final-exit solve. All paths retain
the same line-count penalty and optional immediate numerical-zero cleanup.

Output selection belongs to the caller. This module never compares an output
against a complete RNJ tree and never reads validation, test data or truth.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from research_experiments.iterative_exit.zero import release_zero_atoms
from research_experiments.joint_five.model import open_small, remainder
from research_experiments.local_rebuild.model import canonical
from research_experiments.rnj_count.engine import joint_fit, original_step
from research_experiments.rnj_small_seed.engine import exit_repair, seed_family


def search(bank, common, *, reference, seed_on, tail_on, deadline,
           prefix_deadline=None, per_mip=3600., zero_release=True,
           heartbeat=None):
    """Fit one factorial forward path using an absolute wall-clock deadline.

    The caller fixes ``reference`` once per input, so switching R does not
    silently change the penalty. When R is off, initialization fits only the
    singleton/full basis; it does not call the complete-RNJ seed-scoring LP.
    T off means original WZZT continues through the small regions as well.

    RNJ influence in the LP start pool/calibration is set by the caller, not
    inferred from ``seed_on``. The returned flags document that distinction.
    """
    then = perf_counter()
    if not np.isfinite(reference) or reference < 0:
        raise ValueError("A nonnegative finite shared reference loss is required.")
    stop_at = min(float(deadline), float(prefix_deadline) if prefix_deadline is not None else float(deadline))
    if stop_at - then <= .1:
        raise TimeoutError("No forward-stage budget remains before initialization.")
    if per_mip <= 0:
        raise ValueError("The per-MIP time allowance must be positive.")
    n = bank.data.n
    penalty = .003 * reference

    def ping(stage, **values):
        if heartbeat is not None:
            heartbeat(dict(stage=stage, elapsed_seconds=perf_counter()-then,
                           seed_on=bool(seed_on), tail_on=bool(tail_on), **values))

    ping("seed_started")
    if seed_on:
        current, seeding = seed_family(bank, common["initial"], mode="small3",
                                       reference=reference, deadline=stop_at)
    else:
        basis = canonical([*((i,) for i in range(n)), tuple(range(n))])
        current = bank.fit(basis, preserve=True)
        seeding = dict(mode="none", seeds=[], rankings=[], ranking_complete=True,
                       reference_MAE=reference, complete_RNJ_seed_fit_run=False,
                       truth_used=False)
    path, events, zero_events = [current], [], []
    reason = "wall_limit"
    initial_max = remainder(current["supports"], n)

    def clean(stage):
        nonlocal current
        current, rows = release_zero_atoms(bank, current, enabled=zero_release, stage=stage)
        zero_events.extend(rows)
        if any(row["accepted"] for row in rows):
            path.append(current)
        return any(row["accepted"] for row in rows)

    clean("initial")
    ping("seed_complete", seeds=seeding["seeds"], maximum_remainder=remainder(current["supports"], n))
    for iteration in range(3*n):
        if stop_at-perf_counter() <= .1:
            break
        if tail_on and remainder(current["supports"], n) <= 5:
            ping("joint_tail_started", iteration=iteration)
            keep, released = open_small(current["supports"], n)
            current, event = joint_fit(bank, current, keep, penalty=penalty,
                                       seconds=stop_at-perf_counter(), stage="joint_tail")
            event.update(released=released, iteration=iteration)
            events.append(event)
            if event["accepted"]:
                path.append(current)
            cleaned = clean("tail_joint")
            ping("joint_tail_complete", iteration=iteration, accepted=event["accepted"],
                 status=event.get("status"), zero_cleanup_changed=cleaned)
            if cleaned:
                continue
            reason = "joint_tail_gap_met" if event.get("status") == 2 else "joint_tail_unresolved"
            break
        ping("wzzt_extension_started", iteration=iteration)
        current, event = original_step(bank, current, penalty=penalty,
                                       deadline=stop_at, per_mip=per_mip)
        event["iteration"] = iteration
        events.append(event)
        if event["accepted"]:
            path.append(current)
            clean("extension")
            ping("wzzt_extension_complete", iteration=iteration, accepted=True, status=event.get("status"))
        else:
            reason = ("structurally_complete" if event.get("structurally_complete") else
                      "wall_limit" if event.get("time_exhausted") else "no_regularized_gain_or_unresolved")
            ping("wzzt_extension_complete", iteration=iteration, accepted=False, status=event.get("status"))
            break
    else:
        reason = "step_limit"
    return dict(point=current, path=path, attempts=events, seeding=seeding,
                seconds=perf_counter()-then, stop_reason=reason,
                initial_maximum_remainder=initial_max,
                maximum_remainder=remainder(current["supports"], n),
                reference_MAE=reference, coefficient=.003, penalty=penalty,
                tree_global_optimal=False, strategy="original", zero_release=bool(zero_release),
                zero_events=zero_events, hard=(), seed_on=bool(seed_on), tail_on=bool(tail_on),
                complete_tree_output_fallback=False)


def finish(bank, common, trajectory, *, reference, deadline, exit_on,
           rnj_guidance=True, zero_release=True, heartbeat=None):
    """Optionally perform one existing final-exit solve, then release zeros.

    With ``rnj_guidance=False`` the exit uses the basis, supports generated on
    this own path and the local joint domain. Complete-RNJ warm starts, RNJ
    initial supports and RNJ-neighborhood repair windows are excluded. This
    permits a caller to switch the whole RNJ-guidance module, rather than only
    its initial seeds. LP calibration and the shared reference remain explicit
    caller-controlled inputs.

    Returns ``(trajectory, exit_record, exit_event)``. These are training-stage
    records; the caller still applies the common contraction/refit/selection.
    """
    then = perf_counter()
    if not exit_on:
        return trajectory, dict(accepted=False, skipped="module_disabled", seconds=0.), {}
    remaining = float(deadline)-then
    if remaining <= .1:
        return trajectory, dict(accepted=False, skipped="deadline", seconds=0.), {}
    if heartbeat is not None:
        heartbeat(dict(stage="final_exit_started", rnj_guidance=bool(rnj_guidance)))
    n = bank.data.n
    basis = canonical([*((i,) for i in range(n)), tuple(range(n))])
    if rnj_guidance:
        initial = common["initial"]
        kernel = np.asarray(common["kernel"])
        warm_candidates = common["bank"]
    else:
        initial = basis
        kernel = np.zeros((n, n), dtype=float)  # compact pool does not use this geometry
        warm_candidates = ()
    sentinel = object()
    previous_pool = getattr(bank, "joint_exit_seed_pool", sentinel)
    if not rnj_guidance:
        bank.joint_exit_seed_pool = canonical(C for point in trajectory["path"] for C in point["supports"])
    try:
        record, event = exit_repair(bank, initial, trajectory["point"], kernel,
                                    reference=reference, prefix_seconds=0.,
                                    warm_candidates=warm_candidates, total_seconds=remaining,
                                    # Basis-only repair has no RNJ windows but
                                    # preserves E's generic all-pairs domain.
                                    extra_repair=True,
                                    forbidden=getattr(bank, "forbidden", ()))
    finally:
        if not rnj_guidance:
            if previous_pool is sentinel:
                del bank.joint_exit_seed_pool
            else:
                bank.joint_exit_seed_pool = previous_pool
    point, zeros = release_zero_atoms(bank, record["point"], enabled=zero_release, stage="final_exit")
    path = list(trajectory["path"])
    if record["accepted"]:
        path.append(record["point"])
    if any(row["accepted"] for row in zeros):
        path.append(point)
    elapsed = perf_counter()-then
    record.update(factorial_exit_on=True, rnj_guidance=bool(rnj_guidance), wall_seconds=elapsed)
    event.update(rnj_guidance=bool(rnj_guidance), complete_tree_output_fallback=False)
    result = {**trajectory, "point": point, "path": path,
              "zero_events": [*trajectory.get("zero_events", []), *zeros],
              "final_exit_seconds": elapsed}
    if heartbeat is not None:
        heartbeat(dict(stage="final_exit_complete", rnj_guidance=bool(rnj_guidance),
                       accepted=record["accepted"], status=event.get("status"),
                       elapsed_seconds=elapsed))
    return result, record, event
