"""One train-only search engine for RNJ/NJ and all eight R/T/E ablations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
from time import perf_counter

import numpy as np

if __package__:
    from . import trees
    from . import output_floor
    from .competition import competition
    from .optimization import QPFitter, StructuralSolver, criterion, errors, internal_count, response_matrices
else:
    import trees
    import output_floor
    from competition import competition
    from optimization import QPFitter, StructuralSolver, criterion, errors, internal_count, response_matrices


@dataclass(frozen=True)
class SearchConfig:
    # Statistical/structural choices. Remaining fields are numerical budgets.
    alpha: float = .003
    internal_floor: float = .08
    block_limit: int = 5
    response_bound: float = 2.
    total_seconds: float = 14400.
    per_mip_seconds: float = 3600.
    qp_seconds: float = 300.
    output_reserve_seconds: float = 600.
    zero_tol: float = 1e-9
    gain_tol: float = 1e-10
    mip_gap: float = 1e-4
    mip_abs_gap: float = 1e-7
    validation_tie: float = 1e-12
    forward_fraction: float = .5
    relative_gain: float = 0.
    ambiguity_kappa: float = .003

    def validate(self):
        if self.alpha < 0 or not np.isfinite(self.alpha):
            raise ValueError("alpha must be finite and nonnegative")
        if not np.isfinite(self.internal_floor) or self.internal_floor < 0.:
            raise ValueError("internal_floor must be finite and nonnegative")
        if not np.isfinite(self.relative_gain) or not 0 <= self.relative_gain < 1:
            raise ValueError("relative_gain must be finite and in [0, 1)")
        if not np.isfinite(self.ambiguity_kappa) or not 0 <= self.ambiguity_kappa < 1:
            raise ValueError("ambiguity_kappa must be finite and in [0, 1)")
        if not isinstance(self.block_limit, int) or self.block_limit < 3:
            raise ValueError("block_limit must be an integer at least three")
        positive = (self.response_bound, self.total_seconds, self.per_mip_seconds, self.qp_seconds)
        if any(not np.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Bounds and solver time allowances must be positive")
        if not 0 < self.forward_fraction < 1:
            raise ValueError("forward_fraction must be between zero and one")
        if not 0 <= self.output_reserve_seconds < self.total_seconds:
            raise ValueError("Output reserve must fit within the total method budget")
        if any(not np.isfinite(v) or v < 0 for v in (self.zero_tol, self.gain_tol, self.mip_gap,
                                                    self.mip_abs_gap, self.validation_tie)):
            raise ValueError("Numerical tolerances must be finite and nonnegative")


ALIASES = {"pure_wzzt": "000", "rnj_wzzt": "100", "rte": "111"}
METHODS = ("rnj", "nj", "000", "100", "010", "001", "110", "101", "011", "111")


def _training_fingerprint(train):
    digest = hashlib.sha256(repr(tuple(train.labels)).encode())
    for array in (train.p, train.q, train.y):
        values = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(repr(values.shape).encode()); digest.update(values.tobytes())
    return digest.hexdigest()


def prepare_common(train, config=None):
    """Training-only geometry and a common reference for every ablation cell.

    Pure WZZT shares this scalar calibration for a matched penalty, while its
    support search receives no RNJ pool or warm starts. Raw generated RNJ
    supports are retained; old LP-pruned caches are deliberately not required.
    """
    config = config or SearchConfig()
    config.validate(); started = perf_counter()
    guidance = trees.rnj_guidance(train)
    fitter = QPFitter(train, bound=config.response_bound, qp_seconds=config.qp_seconds)
    reference_point = fitter.fit(guidance["reference_family"])
    reference = reference_point["mse"]
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("The training reference MSE is zero or invalid; normalized regularization is undefined")
    return dict(guidance=guidance, reference_mse=reference, reference_point=reference_point,
                seconds=perf_counter()-started, QP=fitter.records, n=train.n, labels=list(train.labels),
                training_sha256=_training_fingerprint(train),
                truth_used=False, validation_used=False)


def _fit_before_deadline(fitter, family, deadline):
    remaining = deadline-perf_counter()
    if remaining <= 0.:
        raise TimeoutError("No W pretrial budget remains")
    original_limit = fitter.qp_seconds
    fitter.qp_seconds = min(original_limit, remaining)
    try:
        return fitter.fit(family)
    finally:
        fitter.qp_seconds = original_limit


def _clean(fitter, point, config, *, protected=(), deadline=None):
    protected = set(trees.canonical([*trees.basis(fitter.data.n), *protected]))
    if not protected <= set(map(tuple, point["supports"])):
        raise ValueError("A mandatory support is absent before zero cleanup")
    current, events = point, []
    while True:
        keep = trees.canonical(C for C, r, x in zip(current["supports"], current["r"], current["x"], strict=True)
                               if tuple(C) in protected or max(r, x) > config.zero_tol)
        removed = trees.canonical(set(map(tuple, current["supports"]))-set(keep))
        if not removed:
            return current, events
        q = fitter.fit(keep) if deadline is None else _fit_before_deadline(fitter, keep, deadline)
        accepted = q["objective"] <= current["objective"]+config.gain_tol
        events.append(dict(removed=removed, accepted=bool(accepted),
                           before=current["objective"], after=q["objective"]))
        if not accepted:
            return current, events
        current = q


def _accept(current, proposal, fitter, config, *, stage, solver, zeros=(), **extra):
    before = criterion(current, fitter.data.n, config.alpha)
    after = criterion(proposal, fitter.data.n, config.alpha)
    old = {tuple(C): max(r, x) for C, r, x in
           zip(current["supports"], current["r"], current["x"], strict=True)}
    new = set(map(tuple, proposal["supports"]))
    removed = set(old)-new
    # An already-zero temporary support is a representation cleanup. A
    # relative improvement gate must not make that restriction permanent.
    zero_only = bool(removed) and new <= set(old) and all(old[C] <= config.zero_tol for C in removed)
    active_old = {C for C, value in old.items() if 1 < len(C) < fitter.data.n and value > config.zero_tol}
    active_new = {tuple(C) for C, r, x in zip(proposal["supports"], proposal["r"], proposal["x"], strict=True)
                  if 1 < len(C) < fitter.data.n and max(r, x) > config.zero_tol}
    zero_only = zero_only and active_old == active_new
    zero_only = zero_only and proposal["objective"] <= current["objective"]+config.gain_tol
    required = config.gain_tol if zero_only else max(config.gain_tol, config.relative_gain*before)
    accepted = before-after > required
    return (proposal if accepted else current), dict(
        stage=stage, accepted=bool(accepted), before_score=before, after_score=after, gain=before-after,
        numerical_threshold=config.gain_tol, required_gain=required, relative_gain=config.relative_gain,
        acceptance_rule="numerical_zero_cleanup" if zero_only else "relative_regularized_training_gain",
        point=proposal, solver=solver,
        zero_events=list(zeros), closure_before_acceptance=True, **extra)


def _close_native(raw, fitter, config, native, *, protected=()):
    """One fixed-family fit, bound consistency check, and zero cleanup.

    Check the original decoded family before cleanup changes its count. A
    feasible QP point below the reported native lower bound contradicts that
    bound, even when the solver reported an optimal status or zero gap.
    """
    closed = fitter.fit(raw["supports"])
    score = criterion(closed, fitter.data.n, config.alpha)
    native["fixed_family_qp_objective"] = score
    lower = native.get("bound")
    if lower is not None:
        tolerance = 1e-7*(1+abs(score))
        native["bound_consistency_tolerance"] = tolerance
        native["bound_consistent_with_fixed_family_qp"] = bool(lower <= score+tolerance)
        if lower > score+tolerance:
            native["bound_warning"] = "reported_lower_bound_exceeds_a_feasible_fixed_family_QP_point"
    return _clean(fitter, closed, config, protected=protected)


def _any_extension(family, n, rnj_pool, reference_family=()):
    family = set(trees.canonical(family))
    # RNJ supplies a single legal binary pattern, never a fitted tree contest.
    for C in (*trees.canonical(reference_family), *rnj_pool):
        if 1 < len(C) < n and C not in family and all(trees.compatible(C, D) for D in family):
            return C
    for _, blocks in trees.children(family, n).items():
        for left, right in combinations(blocks, 2):
            C = tuple(sorted([*left, *right]))
            if C not in family and all(trees.compatible(C, D) for D in family):
                return C
    return None


def _initialization_receipt(current, raw, closed, fitter, config, source, *, initial=None,
                            pretrial_QP_count=0, **extra):
    initial = current if initial is None else initial
    current_score = criterion(current, fitter.data.n, config.alpha)
    initial_score = criterion(initial, fitter.data.n, config.alpha)
    native_score = (errors(fitter.data, raw)["mse"]/fitter.reference_mse
                    + config.alpha*internal_count(raw, fitter.data.n)) if raw is not None else None
    return dict(current_score=current_score, initial_score=initial_score, initial_source=source,
                current_to_initial_gain=current_score-initial_score,
                initial_point=initial, closed_point=closed,
                native_returned_score=native_score,
                closed_score=criterion(closed, fitter.data.n, config.alpha) if closed is not None else None,
                native_improved_initial=bool(native_score is not None and native_score < initial_score-config.gain_tol),
                has_native_solution=raw is not None, pretrial_QP_count=pretrial_QP_count, **extra)


def _forward(current, fitter, solver, config, pool, deadline, *, reference_family=()):
    n, family = fitter.data.n, trees.canonical(current["supports"])
    candidate = _any_extension(family, n, pool, reference_family)
    if candidate is None:
        return current, dict(stage="W", accepted=False, structurally_complete=True,
                             solver=dict(solve_started=False), zero_events=[],
                             returned_point_source="current",
                             **_initialization_receipt(current, None, None, fitter, config, "current_no_extension",
                                 pretrial_seconds=0., pretrial_candidates_attempted=0, pretrial_candidates=[],
                                 pretrial_stop_reason="structurally_complete", pretrial_best=current,
                                 pretrial_best_score=criterion(current, n, config.alpha)))
    best, best_zeros = current, []
    pretrial_started, record_start = perf_counter(), len(fitter.records)
    trials, stop_reason = [], "completed_pool"
    proposal_source = "current"
    for C in trees.canonical(pool):
        if C in family or not 1 < len(C) < n or not all(trees.compatible(C, D) for D in family):
            continue
        if deadline-perf_counter() <= .01:
            stop_reason = "deadline"
            break
        trial_started, record_before = perf_counter(), len(fitter.records)
        trial_family = trees.canonical([*family, C])
        trial = dict(connection=list(C), cache_hit=trial_family in fitter.cache)
        try:
            q, zeros = _clean(fitter, _fit_before_deadline(fitter, trial_family, deadline),
                              config, deadline=deadline)
        except TimeoutError as exc:
            trial.update(status="timeout", error=str(exc), seconds=perf_counter()-trial_started,
                         new_QP_records=len(fitter.records)-record_before)
            trials.append(trial)
            stop_reason = "qp_soft_timeout"
            break
        trial.update(status="complete", score=criterion(q, n, config.alpha),
                     seconds=perf_counter()-trial_started, new_QP_records=len(fitter.records)-record_before,
                     zero_events=zeros)
        trials.append(trial)
        if trial["score"] < criterion(best, n, config.alpha):
            best, best_zeros, candidate = q, zeros, C
            proposal_source = "pool_pretrial_QP"
    initial, initial_zeros = best, best_zeros
    pretrial_seconds, pretrial_count = perf_counter()-pretrial_started, len(fitter.records)-record_start
    pretrial_deadline_exceeded = perf_counter() >= deadline
    closed = None
    seed_origin = ("RNJ_pool_single_support_QP" if initial is not current else
                   "reference_RNJ_connection" if candidate in trees.canonical(reference_family) else
                   "RNJ_pool_connection" if candidate in pool else "structural_connection")
    initial_source = "RNJ_pool_single_support_QP" if initial is not current else "current_with_zero_weight_connection"
    lookup = {tuple(C): (r, x) for C, r, x in zip(initial["supports"], initial["r"], initial["x"], strict=True)}
    initial_r, initial_x = lookup.get(candidate, (0., 0.))
    if deadline-perf_counter() > .01:
        raw, native = solver.extend(current, candidate, initial,
                                    deadline=min(deadline, perf_counter()+config.per_mip_seconds))
        if raw is not None:
            q, zeros = _close_native(raw, fitter, config, native)
            closed = q
            if criterion(q, n, config.alpha) < criterion(best, n, config.alpha):
                best, best_zeros = q, zeros
                proposal_source = "native_closed_point"
    else:
        raw, native = None, dict(solve_started=False, stop_reason="forward_budget", status=None)
    result, event = _accept(current, best, fitter, config, stage="W", solver=native, zeros=best_zeros,
                   raw_point=raw, unrestricted_support_domain=True, proposal_source=proposal_source,
                   **_initialization_receipt(current, raw, closed, fitter, config,
                       initial_source, initial=initial, initial_connection=list(candidate),
                       initial_connection_source=seed_origin, initial_new_r=initial_r, initial_new_x=initial_x,
                       initial_new_active=int(candidate in lookup), pretrial_QP_count=pretrial_count,
                       pretrial_seconds=pretrial_seconds, pretrial_candidates_attempted=len(trials),
                       pretrial_stop_reason=stop_reason, pretrial_candidates=trials,
                       pretrial_deadline_exceeded=pretrial_deadline_exceeded,
                       pretrial_best=initial, pretrial_best_score=criterion(initial, n, config.alpha),
                       pretrial_best_zero_events=initial_zeros))
    event["returned_point_source"] = proposal_source if event["accepted"] else "current"
    return result, event


def _single_reference_initial(current, reference_point, fitter, config, pool, keep):
    """Validate and rescore the one pre-fitted .75/.16 RNJ reference, without QP."""
    info = dict(reference_considered=reference_point is not None, reference_eligible=False,
                reference_selected=False, reference_score=None, reference_refit_count=0)
    if reference_point is None:
        return current, "current", dict(info, reference_reason="unavailable")
    try:
        family, r, x = output_floor._point(reference_point, fitter.data.n)
        if not set(keep) <= set(family):
            raise ValueError("reference_missing_mandatory_supports")
        if not set(family) <= set(pool):
            raise ValueError("reference_supports_outside_existing_pool")
        if np.any(r > config.response_bound) or np.any(x > config.response_bound):
            raise ValueError("reference_exceeds_coefficient_bound")
        loss = errors(fitter.data, reference_point)["mse"]
        if not np.isfinite(loss):
            raise ValueError("reference_has_nonfinite_training_loss")
        point = dict(reference_point, mse=loss, objective=loss/fitter.reference_mse,
                     reference_mse=fitter.reference_mse, fit_loss="normalized_MSE")
        score = criterion(point, fitter.data.n, config.alpha)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return current, "current", dict(info, reference_reason=str(exc))
    info.update(reference_eligible=True, reference_score=score)
    if score < criterion(current, fitter.data.n, config.alpha):
        info.update(reference_selected=True, reference_reason="single_fixed_reference_has_lower_training_score")
        return point, "single_RNJ_reference", info
    return current, "current", dict(info, reference_reason="reference_not_better_than_current")


def _reconstruct(current, fitter, solver, config, *, pool, keep, deadline, stage, reference_point=None):
    """The same selection/closure/acceptance operation implements both T and E."""
    keep = trees.canonical([*keep, *trees.basis(fitter.data.n)])
    pool = trees.canonical([*pool, *keep, *current["supports"]])
    best, best_zeros = current, []
    initial, source, seed_info = _single_reference_initial(
        current, reference_point if stage == "E" else None, fitter, config, pool, keep)
    closed = None
    if deadline-perf_counter() > .01:
        raw, native = solver.select(pool, keep, initial, deadline=deadline)
        if raw is not None:
            q, zeros = _close_native(raw, fitter, config, native, protected=keep)
            closed = q
            if criterion(q, fitter.data.n, config.alpha) < criterion(best, fitter.data.n, config.alpha):
                best, best_zeros = q, zeros
    else:
        raw, native = None, dict(solve_started=False, status=None, stop_reason="structural_budget")
    return _accept(current, best, fitter, config, stage=stage, solver=native, zeros=best_zeros,
                   pool=pool, keep=keep, raw_point=raw,
                   **_initialization_receipt(current, raw, closed, fitter, config, source,
                       initial=initial, **seed_info))


def _output(path, train, validation, reference, config):
    rows, candidates = [], []
    for index, point in enumerate(path):
        loss = errors(validation, point)
        R, X = response_matrices(point, train.n)
        rows.append(dict(point=point, index=index, internal=internal_count(point, train.n),
                         validation_mae=loss["mae"], validation_mse=loss["mse"],
                         validation_scaled_mse=loss["mse"]/reference))
        candidates.append(dict(rows[-1], R=R.tolist(), X=X.tolist(), labels=list(train.labels), frozen=True))

    def choose(key):
        best = min(row[key] for row in rows)
        return min((row for row in rows if row[key] <= best+config.validation_tie),
                   key=lambda row: (row["internal"], row[key], row["index"]))
    # The common positive training reference changes units, not MSE ordering.
    # Its dimensionless tie tolerance corresponds to reference*validation_tie
    # in physical MSE units. MAE is a diagnostic view only.
    return choose("validation_scaled_mse"), choose("validation_mae"), candidates


def fit(train, validation, method="111", config=None, common=None, heartbeat=None):
    """Fit a baseline or R/T/E cell, with no truth/test inputs in the API."""
    config = config or SearchConfig(); config.validate()
    method = ALIASES.get(method, method)
    if method not in METHODS:
        raise ValueError("Unknown method: "+method)
    if train.n != validation.n or tuple(train.labels) != tuple(validation.labels):
        raise ValueError("Train and validation terminal order must match")
    common = common if common is not None else prepare_common(train, config)
    if common["n"] != train.n or tuple(common["labels"]) != tuple(train.labels):
        raise ValueError("Common training calibration has different terminal labels")
    if common["training_sha256"] != _training_fingerprint(train):
        raise ValueError("Common calibration belongs to different training observations")
    if common["seconds"] >= config.total_seconds:
        raise TimeoutError("Shared training preprocessing exhausted the method budget")
    fitter = QPFitter(train, reference_mse=common["reference_mse"], bound=config.response_bound,
                      qp_seconds=config.qp_seconds)
    # A self-prepared common already consumes fresh wall time; do not charge it
    # twice in the active deadline. Standalone reported costs always count it once.
    fresh_started = perf_counter()
    overall_deadline = fresh_started+config.total_seconds-common["seconds"]
    guidance, events, reason = common["guidance"], [], "baseline"
    path = []

    def ping(stage, **extra):
        if heartbeat is not None:
            heartbeat(dict(stage=stage, elapsed_seconds=perf_counter()-fresh_started,
                           path=path, point=path[-1] if path else None, **extra))

    ping("initialized")
    if method in {"rnj", "nj"}:
        family = guidance["reference_family"] if method == "rnj" else trees.nj(
            guidance["shared"], guidance["depth"], collapse_factor=.16)
        closed, _ = _clean(fitter, fitter.fit(family), config)
        path = [closed]
        native_records = []
    else:
        R_on, T_on, E_on = (bool(int(bit)) for bit in method)
        pool = trees.canonical(guidance["pool"]) if R_on else trees.basis(train.n)
        reference_family = guidance["reference_family"] if R_on else ()
        current = fitter.fit(trees.basis(train.n)); path = [current]
        structural_deadline = overall_deadline-config.output_reserve_seconds
        # E-off cells can spend all structural time on W/T. E-on cells reserve
        # the same declared fraction for the one final joint solve.
        prefix_deadline = (perf_counter()+max(0., structural_deadline-perf_counter())*config.forward_fraction
                           if E_on else structural_deadline)
        visited = {trees.canonical(current["supports"])}
        solver = StructuralSolver(fitter, config)
        tail = dict(pool=(), keep=trees.canonical(current["supports"]), frontiers=[], released=())
        try:
            reason = "forward_budget"
            while prefix_deadline-perf_counter() > .01:
                if T_on and trees.max_direct_blocks(current["supports"], train.n) <= config.block_limit:
                    reason = "tail_eligible"; break
                ping("W_started", iteration=len(events))
                proposed, event = _forward(current, fitter, solver, config, pool, prefix_deadline,
                                            reference_family=reference_family)
                events.append(event)
                if not event["accepted"]:
                    reason = "structurally_complete" if event.get("structurally_complete") else "no_gain_or_unresolved"
                    ping("W_stopped", event=event)
                    break
                signature = trees.canonical(proposed["supports"])
                if signature in visited:
                    event.update(accepted=False, numerical_gain_passed=True, repeated_family=True)
                    reason = "numerical_repeated_family"; break
                current = proposed; path.append(current); visited.add(signature)
                ping("W_complete", event=event)
            if T_on and trees.max_direct_blocks(current["supports"], train.n) <= config.block_limit and prefix_deadline-perf_counter() > .01:
                ping("T_started")
                tail = trees.frontier_candidates(current["supports"], train.n, block_limit=config.block_limit)
                current, event = _reconstruct(current, fitter, solver, config, pool=tail["pool"], keep=tail["keep"],
                                               deadline=prefix_deadline, stage="T")
                event.update(frontiers=tail["frontiers"], released=tail["released"]); events.append(event)
                if event["accepted"]:
                    path.append(current)
                ping("T_complete", event=event)
            if E_on and structural_deadline-perf_counter() > .01:
                ping("E_started")
                e_pool = trees.canonical([*pool, *(C for p in path for C in p["supports"]),
                                          *tail["pool"], *combinations(range(train.n), 2)])
                current, event = _reconstruct(current, fitter, solver, config, pool=e_pool, keep=trees.basis(train.n),
                                               deadline=structural_deadline, stage="E",
                                               reference_point=common.get("reference_point") if R_on else None)
                events.append(event)
                if event["accepted"]:
                    path.append(current)
                ping("E_complete", event=event)
            elif E_on:
                events.append(dict(stage="E", accepted=False, solver=dict(solve_started=False), skipped="budget",
                    **_initialization_receipt(current, None, None, fitter, config, "current")))
        finally:
            solver.close()
        native_records = solver.records
    # Singleton-only protection baseline; all other candidates are this method's
    # own accepted path. No independent complete-RNJ final-tree fallback exists.
    output_path = path
    if method not in {"rnj", "nj"}:
        star = fitter.fit([(i,) for i in range(train.n)])
        output_path = [*path, star]
    ping("output_started")
    uncontracted_output_path = output_path
    contraction = None
    if config.internal_floor > 0.:
        floor_started = perf_counter()
        try:
            if floor_started >= overall_deadline:
                raise TimeoutError("No original method budget remains for output contraction")
            calibration = output_floor.calibrate(guidance, train)
            calibration_seconds = perf_counter()-floor_started
            output_path, contraction = output_floor.contract_bank(
                uncontracted_output_path, fitter, config.internal_floor, calibration,
                deadline=overall_deadline)
            contraction.update(bank_seconds=contraction["seconds"],
                               calibration_seconds=calibration_seconds,
                               seconds=perf_counter()-floor_started)
        except output_floor.OutputFloorError:
            raise
        except Exception as exc:
            receipt = dict(status="failed", threshold=config.internal_floor,
                metric="rx75", scope="non_singleton", stage="training_calibration",
                candidate_count=len(uncontracted_output_path), candidates=[],
                seconds=perf_counter()-floor_started, original_deadline=overall_deadline,
                deadline_exceeded=perf_counter() >= overall_deadline,
                partial_bank_accepted=False, validation_used=False, truth_used=False,
                error=type(exc).__name__+": "+str(exc))
            raise output_floor.OutputFloorError(str(exc), receipt=receipt,
                qp_records=fitter.records,
                last_point=uncontracted_output_path[-1] if uncontracted_output_path else None) from exc
    selected, mae_view, candidates = _output(output_path, train, validation, common["reference_mse"], config)
    result = dict(method=method, selected=selected, selected_mse_view=selected,
                selected_mae_view=mae_view, selection_metric="validation_mse", path=path, events=events,
                frozen_candidates=candidates, stop_reason=reason, config=asdict(config),
                reference_mse=common["reference_mse"], common_seconds=common["seconds"],
                QP=fitter.records, MIQP=native_records, output_contraction=contraction,
                native_bound_conflicts=sum(r.get("bound_consistent_with_fixed_family_qp") is False for r in native_records),
                complete_RNJ_output_fallback=False, truth_used=False, test_used=False,
                guarantee_scope="native bounds require consistency with feasible fixed-family fits; no whole-tree optimality claim")
    if contraction is not None:
        result["uncontracted_output_path"] = uncontracted_output_path
    result["competition"] = competition(result, validation)
    fresh_seconds = perf_counter()-fresh_started
    result.update(fresh_seconds=fresh_seconds, total_seconds=fresh_seconds+common["seconds"],
                  total_cap_exceeded=fresh_seconds+common["seconds"] > config.total_seconds)
    return result
