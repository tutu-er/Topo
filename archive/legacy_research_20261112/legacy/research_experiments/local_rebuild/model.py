"""Enumerated block-union MILP with a joint, original-measurement L1 fit.

Only known P/Q and observed squared-voltage drops enter this module. R/X
coefficients absorb the factor two, as in the core WZZT implementation.
The model permits multifurcations and zero-length edges. It has no source
attachment variables, intercept, regularizer, or physical topology oracle.
"""
from __future__ import annotations

from itertools import combinations
from time import perf_counter

import numpy as np

from rnj_wzzt.estimation import laminar_l1_milp as core


def canonical(family):
    return tuple(sorted({tuple(sorted(s)) for s in family}, key=lambda s: (len(s), s)))


def compatible(left, right):
    a, b = set(left), set(right)
    return not a & b or a <= b or b <= a


def unions(blocks):
    """Return every proper union of at least two disjoint frontier blocks."""
    return tuple((indices, tuple(sorted(i for a in indices for i in blocks[a])))
                 for k in range(2, len(blocks))
                 for indices in combinations(range(len(blocks)), k))


def build_model(prepared, family, region, blocks, *, hard=(), bound=2., cuts="subset"):
    """Build A1--A6, validating that the previous family is representable.

    Fixed physical priors that span blocks consume the same local cardinality
    budget as binary candidates. Candidates crossing any retained support are
    disabled. A malformed frontier is rejected instead of silently shrinking
    the old feasible domain.
    """
    family = core.normalize_supports(canonical(family), prepared.n)
    hard = core.normalize_supports(canonical(hard), prepared.n)
    if not set(hard) <= set(family):
        raise ValueError("hard priors must be present in the initial family")
    bound = core._require_positive_finite("bound", bound)
    if cuts not in ("none", "global", "subset"):
        raise ValueError("cuts must be none, global, or subset")
    region = tuple(sorted(region))
    blocks = tuple(tuple(sorted(b)) for b in blocks)
    if not 2 <= len(blocks) <= 8 or any(not b for b in blocks):
        raise ValueError("the prototype requires two to eight nonempty blocks")
    flat = [i for b in blocks for i in b]
    if len(flat) != len(set(flat)) or set(flat) != set(region):
        raise ValueError("blocks must partition the region")
    if not region or min(region) < 0 or max(region) >= prepared.n:
        raise ValueError("invalid region")
    hierarchy = {*family, *((i,) for i in range(prepared.n)), tuple(range(prepared.n))}
    if any(b not in hierarchy for b in blocks):
        raise ValueError("each block must be a current complete subtree or singleton")
    if any(not compatible(s, region) for s in family):
        raise ValueError("region must preserve the surrounding hierarchy")
    candidates_with_indices = unions(blocks)
    union_lookup = {s: indices for indices, s in candidates_with_indices}
    keep, released = [], []
    for s in family:
        inside = set(s) < set(region) and not any(set(s) <= set(b) for b in blocks)
        if inside and s not in union_lookup:
            raise ValueError("a released old support is not a union of whole blocks")
        (released if inside and s not in hard else keep).append(s)
    candidates_with_indices = tuple((idx, s) for idx, s in candidates_with_indices if s not in keep)
    candidates = tuple(s for _, s in candidates_with_indices)
    fixed_local = tuple(union_lookup[s] for s in keep if s in union_lookup)
    all_supports = (*keep, *candidates)
    # build_fixed_model only builds affine features; it intentionally does not
    # validate the union of candidates, which can contain crossing supports.
    model = core.build_fixed_model(prepared, all_supports,
                                  r_upper_bound=bound, x_upper_bound=bound)
    selection = model.variables.add("select", len(candidates), upper=1., integral=True)
    for k, s in enumerate(candidates):
        col = selection.start + k
        for weights in (model.r, model.x):
            model.rows.add({weights.start + len(keep) + k: 1., col: -bound}, upper=0.)
        if any(not compatible(s, t) for t in keep):
            model.variables.upper[col] = 0.
    conflicts = 0
    for i, j in combinations(range(len(candidates)), 2):
        if not compatible(candidates[i], candidates[j]):
            model.rows.add({selection.start + i: 1., selection.start + j: 1.}, upper=1.)
            conflicts += 1
    cut_count = 0
    if cuts != "none":
        sizes = (len(blocks),) if cuts == "global" else range(3, len(blocks) + 1)
        for size in sizes:
            for w in combinations(range(len(blocks)), size):
                w = set(w)
                entries = {selection.start + k: 1. for k, (idx, _) in
                           enumerate(candidates_with_indices) if set(idx) < w}
                rhs = size - 2 - sum(set(idx) < w for idx in fixed_local)
                if rhs < 0:
                    raise ValueError("fixed local priors violate hierarchy cardinality")
                if entries:
                    model.rows.add(entries, upper=float(rhs))
                    cut_count += 1
    return model, dict(keep=tuple(keep), released=tuple(released), candidates=candidates,
                       all_supports=all_supports, selection=selection, region=region,
                       blocks=blocks, conflicts=conflicts, cardinality_cuts=cut_count)


def prediction_loss(prepared, supports, r, x):
    R, X = core.build_matrices_from_atoms(prepared.n, supports, r, x)
    return sum(float(abs(y - p @ R.T - q @ X.T).sum()) for p, q, y in
               zip(prepared.p, prepared.q, prepared.target)) / prepared.observation_count


def _violation(values, lower, upper):
    return float(max(np.max(np.maximum(np.asarray(lower) - values, 0.), initial=0.),
                     np.max(np.maximum(values - np.asarray(upper), 0.), initial=0.)))


def solve_rebuild(scenarios, family, region, blocks, *, hard=(), bound=2.,
                  cuts="subset", seconds=3., relax=False, presolve=True, incumbent=None):
    """Keep the raw solver certificate and independently audit every incumbent.

    A time-limited feasible solution may be useful but has no local-optimum
    claim. The scan accepts it only after an audited strict improvement.
    """
    prepared = core._prepare_scenarios(scenarios)
    seconds = core._require_positive_finite("seconds", seconds)
    started = perf_counter()
    model, meta = build_model(prepared, family, region, blocks, hard=hard, bound=bound, cuts=cuts)
    witness = None
    if incumbent is not None:
        # Verify the supplied upper bound in this exact MILP, including every
        # binary and cardinality row. SciPy does not use it as a MIP start.
        lookup = {tuple(s): (float(r), float(x)) for s, r, x in
                  zip(incumbent["supports"], incumbent["r"], incumbent["x"], strict=True)}
        if not set(lookup) <= set(meta["all_supports"]):
            raise ValueError("incumbent is outside this reconstruction domain")
        values = np.zeros(model.variables.size)
        for k, s in enumerate(meta["all_supports"]):
            values[model.r.start + k], values[model.x.start + k] = lookup.get(s, (0., 0.))
        for k, s in enumerate(meta["candidates"]):
            values[meta["selection"].start + k] = float(s in lookup)
        R, X = core.build_matrices_from_atoms(prepared.n, incumbent["supports"],
                                             incumbent["r"], incumbent["x"])
        values[model.absolute] = np.concatenate([abs(y - p @ R.T - q @ X.T).ravel()
            for p, q, y in zip(prepared.p, prepared.q, prepared.target)])
        constraints = model.rows.build(model.variables.size)
        error = max(_violation(constraints.A @ values, constraints.lb, constraints.ub),
                    _violation(values, model.variables.lower, model.variables.upper))
        if not np.isfinite(values).all() or error > 1e-7:
            raise ValueError("supplied incumbent is not feasible in this MILP")
        witness = dict(objective=float(np.asarray(model.variables.objective) @ values),
                       maximum_violation=error, solver_MIP_start=False)
    original_integrality = np.asarray(model.variables.integrality, dtype=bool)
    if relax:
        model.variables.integrality = [0] * model.variables.size
    result, diagnostics = core._run_milp(model.variables, model.rows, time_limit=seconds,
                                        mip_rel_gap=0., presolve=presolve, disp=False)
    record = {k: v for k, v in meta.items() if k not in ("selection", "all_supports")}
    record.update(diagnostics=diagnostics.to_dict(),
                  certified=core.solver_diagnostics_prove_optimality(diagnostics),
                  wall_seconds=perf_counter() - started, relaxation=relax,
                  audit={"valid": False, "has_incumbent": result.x is not None}, point=None)
    record.update(presolve=presolve, known_feasible_witness=witness)
    if witness is not None and diagnostics.dual_bound is not None and diagnostics.dual_bound > witness["objective"] + 1e-8:
        record["certified"] = False
        record["certificate_error"] = "dual_bound_exceeds_verified_feasible_objective"
    if result.x is None or result.fun is None:
        return record
    values = np.asarray(result.x)
    if values.shape != (model.variables.size,) or not np.isfinite(values).all():
        record["audit"]["error"] = "nonfinite or malformed incumbent"
        return record
    rows = model.rows.build(model.variables.size)
    row_error = _violation(rows.A @ values, rows.lb, rows.ub)
    bound_error = _violation(values, model.variables.lower, model.variables.upper)
    integer_error = float(np.max(abs(values[original_integrality] -
                                     np.rint(values[original_integrality])), initial=0.))
    r, x = values[model.r], values[model.x]
    physical = prediction_loss(prepared, meta["all_supports"], np.maximum(r, 0.), np.maximum(x, 0.))
    objective_error = abs(physical - result.fun)
    audit = dict(has_incumbent=True, constraint_violation=row_error, bound_violation=bound_error,
                 integrality_violation=integer_error, objective_difference=objective_error,
                 physical_objective=physical, valid=bool(max(row_error, bound_error, objective_error,
                 0. if relax else integer_error) <= 1e-7))
    record["audit"] = audit
    record["selection_values"] = values[meta["selection"]].tolist()
    if audit["valid"] and not relax:
        selected = values[meta["selection"]] > .5
        active = [i for i, s in enumerate(meta["keep"]) if len(s) == 1 or s in canonical(hard)
                  or max(r[i], x[i]) > 1e-10] + [len(meta["keep"]) + k for k, a in enumerate(selected)
                 if a and max(r[len(meta["keep"]) + k], x[len(meta["keep"]) + k]) > 1e-10]
        supports = tuple(meta["all_supports"][i] for i in active)
        rr, xx = np.maximum(r[active], 0.), np.maximum(x[active], 0.)
        legal = core.is_laminar_family(supports) and set(canonical(hard)) <= set(supports)
        final_loss = prediction_loss(prepared, supports, rr, xx)
        audit["pruning_objective_difference"] = abs(final_loss - physical)
        audit["valid"] = legal and audit["pruning_objective_difference"] <= 1e-8
        if audit["valid"]:
            record["point"] = dict(supports=supports, r=rr.tolist(), x=xx.tolist(), objective=final_loss)
    return record


def fit_fixed(scenarios, family, *, bound=2., seconds=10.):
    prepared = core._prepare_scenarios(scenarios)
    sol = core.solve_fixed_support_l1(prepared, canonical(family), r_upper_bound=bound,
                                     x_upper_bound=bound, time_limit=seconds)
    value = prediction_loss(prepared, sol.supports, sol.r_values, sol.x_values)
    if abs(value - sol.objective) > 1e-8:
        raise RuntimeError("fixed LP objective reproduction failed")
    return dict(supports=sol.supports, r=sol.r_values.tolist(), x=sol.x_values.tolist(),
                objective=value, diagnostics=sol.diagnostics.to_dict())


def drop_zero(prepared, point, *, hard=()):
    """Match the existing pool search's removal of zero auxiliary supports."""
    active = [i for i, s in enumerate(point["supports"]) if len(s) == 1 or s in canonical(hard)
              or max(point["r"][i], point["x"][i]) > 1e-9]
    trimmed = {**point, **{k: [point[k][i] for i in active] for k in ("supports", "r", "x")}}
    trimmed["objective"] = prediction_loss(prepared, trimmed["supports"], trimmed["r"], trimmed["x"])
    if abs(trimmed["objective"] - point["objective"]) > 1e-8:
        raise RuntimeError("zero-atom removal changed the physical fit")
    return trimmed


def frontiers(family, n, *, max_blocks=5):
    """Enumerate small frontiers by opening complete current child subtrees.

    Selection is deterministic, using structure only. Regions with more than
    max_blocks immediate children are outside this first scan's coverage.
    """
    hierarchy = canonical([*family, *((i,) for i in range(n)), tuple(range(n))])
    children = {}
    for v in hierarchy:
        proper = [s for s in hierarchy if set(s) < set(v)]
        children[v] = canonical(s for s in proper if not any(set(s) < set(t) for t in proper))
    seen = set()
    for v in hierarchy:
        todo = [children[v]] if len(v) > 2 else []
        while todo:
            blocks = todo.pop(0)
            if not blocks or len(blocks) > max_blocks or (v, blocks) in seen:
                continue
            seen.add((v, blocks))
            if len(blocks) >= 3:
                yield v, blocks
            for b in blocks:
                if children[b]:
                    todo.append(canonical([s for s in blocks if s != b] + list(children[b])))


def scan(scenarios, initial, *, hard=(), bound=2., max_blocks=5, seconds=20.,
         per_solve_seconds=3., max_solves=12, cuts="subset"):
    """Accept training-L1 improvements; never use validation, test, or truth."""
    if not isinstance(max_blocks, int) or not 3 <= max_blocks <= 8:
        raise ValueError("max_blocks must be an integer from three to eight")
    if not isinstance(max_solves, int) or max_solves < 1:
        raise ValueError("max_solves must be positive")
    core._require_positive_finite("seconds", seconds)
    core._require_positive_finite("per_solve_seconds", per_solve_seconds)
    prepared = core._prepare_scenarios(scenarios)
    started = perf_counter()
    current = fit_fixed(prepared, initial, bound=bound, seconds=seconds)
    # Numerical zero auxiliary atoms have no continuing structural authority.
    old = current
    current = drop_zero(prepared, current, hard=hard)
    removed_zero = [s for s in old["supports"] if s not in current["supports"]]
    path, attempts = [current], []
    tried = set()
    reason = "scan_complete"
    while len(attempts) < max_solves:
        improved = False
        for v, blocks in frontiers(current["supports"], prepared.n, max_blocks=max_blocks):
            key = (canonical(current["supports"]), v, blocks)
            if key in tried:
                continue
            remaining = seconds - (perf_counter() - started)
            if remaining <= 0. or len(attempts) >= max_solves:
                reason = "wall_limit" if remaining <= 0. else "solve_limit"
                break
            tried.add(key)
            attempt = solve_rebuild(prepared, current["supports"], v, blocks, hard=hard,
                                    bound=bound, cuts=cuts, seconds=min(per_solve_seconds, remaining))
            point = attempt["point"]
            attempt["before_objective"] = current["objective"]
            attempt["accepted"] = bool(point and point["objective"] < current["objective"] -
                                       (1e-10 + 1e-8 * current["objective"]))
            attempts.append(attempt)
            if attempt["accepted"]:
                current = point
                path.append(current)
                improved = True
                break
        if reason != "scan_complete" or not improved:
            break
    if len(attempts) >= max_solves:
        reason = "solve_limit"
    return dict(point=current, path=path, attempts=attempts, stop_reason=reason,
                seconds=perf_counter() - started, removed_initial_zero=removed_zero,
                truth_used=False, scope="bounded scan of enumerated current-subtree frontiers")


def repair_control(scenarios, initial, pool, *, bound=2., seconds=20.):
    """Existing one-candidate repair rule, refitted with the same HiGHS LP.

    This reproduces the structural rule of wzzt_retune.engine.search rather
    than its Gurobi caching, validation selection, or contraction policy.
    """
    prepared = core._prepare_scenarios(scenarios)
    started = perf_counter()
    current = drop_zero(prepared, fit_fixed(prepared, initial, bound=bound, seconds=seconds))
    cache, attempts, path = {}, [], [current]
    reason = "pool_local_optimum"
    for _ in range(3 * prepared.n):
        best = None
        for c in canonical(pool):
            if len(c) < 2 or c in canonical(current["supports"]):
                continue
            remaining = seconds - (perf_counter() - started)
            if remaining <= 0:
                reason = "wall_limit"
                break
            retained = [s for s in current["supports"] if compatible(s, c)]
            family = canonical([*retained, c])
            if family not in cache:
                try:
                    cache[family] = drop_zero(prepared, fit_fixed(prepared, family, bound=bound, seconds=remaining))
                except RuntimeError as exc:
                    attempts.append(dict(candidate=c, error=str(exc), accepted=False))
                    reason = "uncertified_LP"
                    break
            point = cache[family]
            attempts.append(dict(candidate=c, objective=point["objective"], diagnostics=point["diagnostics"]))
            if best is None or point["objective"] < best["objective"] - 1e-12:
                best = point
        gain = 0. if best is None else current["objective"] - best["objective"]
        if gain > 1e-10 + .001 * current["objective"]:
            current = best
            path.append(current)
        else:
            break
        if reason != "pool_local_optimum":
            break
    return dict(point=current, path=path, attempts=attempts, stop_reason=reason,
                seconds=perf_counter() - started, truth_used=False)
