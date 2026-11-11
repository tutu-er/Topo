"""One train-only search engine for RNJ/NJ and all eight R/T/E ablations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
from time import perf_counter

import numpy as np

if __package__:
    from . import trees
    from .optimization import QPFitter, StructuralSolver, criterion, errors, internal_count, response_matrices
else:
    import trees
    from optimization import QPFitter, StructuralSolver, criterion, errors, internal_count, response_matrices


@dataclass(frozen=True)
class SearchConfig:
    # Statistical/structural choices. Remaining fields are numerical budgets.
    alpha: float = .003
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

    def validate(self):
        if self.alpha < 0 or not np.isfinite(self.alpha):
            raise ValueError("alpha must be finite and nonnegative")
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


def _clean(fitter, point, config, *, protected=()):
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
        q = fitter.fit(keep)
        accepted = q["objective"] <= current["objective"]+config.gain_tol
        events.append(dict(removed=removed, accepted=bool(accepted),
                           before=current["objective"], after=q["objective"]))
        if not accepted:
            return current, events
        current = q


def _accept(current, proposal, fitter, config, *, stage, solver, zeros=(), **extra):
    before = criterion(current, fitter.data.n, config.alpha)
    after = criterion(proposal, fitter.data.n, config.alpha)
    accepted = before-after > config.gain_tol
    return (proposal if accepted else current), dict(
        stage=stage, accepted=bool(accepted), before_score=before, after_score=after, gain=before-after,
        numerical_threshold=config.gain_tol, point=proposal, solver=solver,
        zero_events=list(zeros), closure_before_acceptance=True, **extra)


def _any_extension(family, n, rnj_pool):
    family = set(trees.canonical(family))
    for C in rnj_pool:
        if 1 < len(C) < n and C not in family and all(trees.compatible(C, D) for D in family):
            return C
    for _, blocks in trees.children(family, n).items():
        for left, right in combinations(blocks, 2):
            C = tuple(sorted([*left, *right]))
            if C not in family and all(trees.compatible(C, D) for D in family):
                return C
    return None


def _forward(current, fitter, solver, config, pool, deadline):
    n, family = fitter.data.n, trees.canonical(current["supports"])
    candidate = _any_extension(family, n, pool)
    if candidate is None:
        return current, dict(stage="W", accepted=False, structurally_complete=True,
                             solver=dict(solve_started=False), zero_events=[])
    best, best_zeros = current, []
    for C in pool:
        if perf_counter() >= deadline:
            break
        if C in family or not 1 < len(C) < n or not all(trees.compatible(C, D) for D in family):
            continue
        q, zeros = _clean(fitter, fitter.fit([*family, C]), config)
        if criterion(q, n, config.alpha) < criterion(best, n, config.alpha):
            best, best_zeros, candidate = q, zeros, C
    if deadline-perf_counter() > .01:
        raw, native = solver.extend(current, candidate, best,
                                    deadline=min(deadline, perf_counter()+config.per_mip_seconds))
        if raw is not None:
            q, zeros = _clean(fitter, fitter.fit(raw["supports"]), config)
            if criterion(q, n, config.alpha) < criterion(best, n, config.alpha):
                best, best_zeros = q, zeros
    else:
        raw, native = None, dict(solve_started=False, stop_reason="forward_budget", status=None)
    return _accept(current, best, fitter, config, stage="W", solver=native, zeros=best_zeros,
                   raw_point=raw, unrestricted_support_domain=True)


def _reconstruct(current, fitter, solver, config, *, pool, keep, deadline, stage, warm_families=()):
    """The same selection/closure/acceptance operation implements both T and E."""
    keep = trees.canonical([*keep, *trees.basis(fitter.data.n)])
    pool = trees.canonical([*pool, *keep, *current["supports"]])
    best, best_zeros = current, []
    for family in warm_families:
        if perf_counter() >= deadline:
            break
        family = trees.canonical([*family, *trees.basis(fitter.data.n)])
        if not set(keep) <= set(family) or not set(family) <= set(pool) or not trees.is_laminar(family):
            continue
        q, zeros = _clean(fitter, fitter.fit(family), config, protected=keep)
        if criterion(q, fitter.data.n, config.alpha) < criterion(best, fitter.data.n, config.alpha):
            best, best_zeros = q, zeros
    if deadline-perf_counter() > .01:
        raw, native = solver.select(pool, keep, best, deadline=deadline)
        if raw is not None:
            q, zeros = _clean(fitter, fitter.fit(raw["supports"]), config, protected=keep)
            if criterion(q, fitter.data.n, config.alpha) < criterion(best, fitter.data.n, config.alpha):
                best, best_zeros = q, zeros
    else:
        raw, native = None, dict(solve_started=False, status=None, stop_reason="structural_budget")
    return _accept(current, best, fitter, config, stage=stage, solver=native, zeros=best_zeros,
                   pool=pool, keep=keep, raw_point=raw)


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
    return choose("validation_mae"), choose("validation_scaled_mse"), candidates


def fit(train, validation, method="111", config=None, common=None):
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
    if method in {"rnj", "nj"}:
        family = guidance["reference_family"] if method == "rnj" else trees.nj(
            guidance["shared"], guidance["depth"], collapse_factor=.16)
        closed, _ = _clean(fitter, fitter.fit(family), config)
        path = [closed]
        native_records = []
    else:
        R_on, T_on, E_on = (bool(int(bit)) for bit in method)
        pool = trees.canonical(guidance["pool"]) if R_on else trees.basis(train.n)
        warm = guidance["warm_families"] if R_on else ()
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
                proposed, event = _forward(current, fitter, solver, config, pool, prefix_deadline)
                events.append(event)
                if not event["accepted"]:
                    reason = "structurally_complete" if event.get("structurally_complete") else "no_gain_or_unresolved"
                    break
                signature = trees.canonical(proposed["supports"])
                if signature in visited:
                    event.update(accepted=False, numerical_gain_passed=True, repeated_family=True)
                    reason = "numerical_repeated_family"; break
                current = proposed; path.append(current); visited.add(signature)
            if T_on and trees.max_direct_blocks(current["supports"], train.n) <= config.block_limit and prefix_deadline-perf_counter() > .01:
                tail = trees.frontier_candidates(current["supports"], train.n, block_limit=config.block_limit)
                current, event = _reconstruct(current, fitter, solver, config, pool=tail["pool"], keep=tail["keep"],
                                               deadline=prefix_deadline, stage="T")
                event.update(frontiers=tail["frontiers"], released=tail["released"]); events.append(event)
                if event["accepted"]:
                    path.append(current)
            if E_on and structural_deadline-perf_counter() > .01:
                e_pool = trees.canonical([*pool, *(C for p in path for C in p["supports"]),
                                          *tail["pool"], *combinations(range(train.n), 2)])
                current, event = _reconstruct(current, fitter, solver, config, pool=e_pool, keep=trees.basis(train.n),
                                               deadline=structural_deadline, stage="E", warm_families=warm)
                events.append(event)
                if event["accepted"]:
                    path.append(current)
            elif E_on:
                events.append(dict(stage="E", accepted=False, solver=dict(solve_started=False), skipped="budget"))
        finally:
            solver.close()
        native_records = solver.records
    # Singleton-only protection baseline; all other candidates are this method's
    # own accepted path. No independent complete-RNJ final-tree fallback exists.
    output_path = path
    if method not in {"rnj", "nj"}:
        star = fitter.fit([(i,) for i in range(train.n)])
        output_path = [*path, star]
    selected, secondary, candidates = _output(output_path, train, validation, common["reference_mse"], config)
    fresh_seconds = perf_counter()-fresh_started
    return dict(method=method, selected=selected, selected_mse_view=secondary, path=path, events=events,
                frozen_candidates=candidates, stop_reason=reason, config=asdict(config),
                reference_mse=common["reference_mse"], common_seconds=common["seconds"],
                fresh_seconds=fresh_seconds, total_seconds=fresh_seconds+common["seconds"],
                total_cap_exceeded=fresh_seconds+common["seconds"] > config.total_seconds,
                QP=fitter.records, MIQP=native_records, output_contraction=None,
                complete_RNJ_output_fallback=False, truth_used=False, test_used=False,
                guarantee_scope="each declared conditional MIQP domain; whole-tree global optimality not claimed")
