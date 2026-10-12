"""Observation-only passive proposals; RNJ supports remain replaceable.

Connects the current constrained sensitivity estimator, RNJ, and source-aware
WZZT to a finite AC candidate bank. Scores here are only proposal rankings.
No family or atom is certified by this module; ``ac_evidence`` stays None.
The production pipeline and the frozen v8 experiments are not modified.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import combinations
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from rnj_wzzt.estimation.multiscenario import fit_projected_sensitivity
from rnj_wzzt.estimation.unmetered_inputs import PreparedSourceInputs
from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
from rnj_wzzt.graph.rooted_neighbor_joining import rooted_neighbor_joining
from rnj_wzzt.graph.sensitivity_geometry import sensitivity_geometry
from .wzzt_adapter import fit_source_aware_wzzt


def normalize_family(supports, n):
    """Canonical nontrivial laminar supports; singleton/root edges implicit."""
    values = set()
    for support in supports:
        values_in_support = tuple(support)
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
               for i in values_in_support):
            raise ValueError("support indices must be integers")
        s = tuple(sorted(set(int(i) for i in values_in_support)))
        if not s or min(s) < 0 or max(s) >= n:
            raise ValueError("supports must contain terminal indices in 0..n-1")
        if 1 < len(s) < n:
            values.add(s)
    result = tuple(sorted(values, key=lambda s: (-len(s), s)))
    sets = list(map(set, result))
    if any(a & b and not (a <= b or b <= a) for a, b in combinations(sets, 2)):
        raise ValueError("support family must be laminar")
    return result


def family_neighbors(supports, n):
    """One clade deletion, rooted local interchange, or terminal-label swap.

    A rooted interchange moves a direct child A of C to C's parent and a
    sibling D of C into C: C becomes (C\\A) union D. All other clades stay.
    Terminal swaps are global label permutations, not certified blocks.
    The returned bank is finite and is not a complete topology enumeration.
    """
    family = normalize_family(supports, n)
    result = {family: "unchanged"}
    for dropped in family:
        result[tuple(s for s in family if s != dropped)] = "delete_clade"
    full = frozenset(range(n))
    nodes = {full, *(frozenset(s) for s in family),
             *(frozenset((i,)) for i in range(n))}
    parents = {s: min((p for p in nodes if s < p), key=len)
               for s in nodes if s != full}
    for old in family:
        c = frozenset(old)
        children = sorted((a for a in nodes if parents.get(a) == c), key=lambda a: tuple(sorted(a)))
        siblings = sorted((d for d in nodes if d != c and parents.get(d) == parents[c]),
                          key=lambda d: tuple(sorted(d)))
        if len(children) < 2:
            continue
        for a in children:
            for d in siblings:
                new = tuple(sorted((c - a) | d))
                changed = normalize_family([s for s in family if s != old] + [new], n)
                result[changed] = "rooted_interchange"
    for i, j in combinations(range(n), 2):
        changed = normalize_family(
            [tuple(j if k == i else i if k == j else k for k in s) for s in family], n)
        result.setdefault(changed, "terminal_swap")
    return result


def _as_observations(p, q, y):
    arrays = [np.asarray(v, dtype=float) for v in (p, q, y)]
    if arrays[0].ndim != 2 or min(arrays[0].shape) < 2:
        raise ValueError("P, Q, Y require at least two times and two terminals")
    if any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("P, Q, Y must be finite arrays of the same shape (T,n)")
    return tuple(a.copy() for a in arrays)


def _profile(value, t, name):
    a = np.broadcast_to(np.asarray(value, dtype=float), (t,)).copy()
    if not np.isfinite(a).all():
        raise ValueError(name + " must be finite and scalar or (T,)")
    return a


def _scenario(p, q, y, name):
    return {"name": name, "P_terminal": pd.DataFrame(p),
            "Q_terminal": pd.DataFrame(q), "drop_target": pd.DataFrame(y)}


def contrast_rank(p_effective, y, supports, coefficient_bound):
    """Bounded linear fit after subtracting each row's terminal mean.

    This profiles out an arbitrary common response at each time, solely to
    rank proposals. AC losses need not be common and remain unmodelled here.
    The rank cannot exclude a topology and is not an AC feasibility check.
    """
    n = p_effective.shape[1]
    family = (*normalize_family(supports, n), *((i,) for i in range(n)))
    target = y - y.mean(axis=1, keepdims=True)
    columns = []
    for s in family:
        z = np.zeros(n)
        z[list(s)] = 1.
        atom = p_effective[:, s].sum(axis=1)[:, None] * z
        columns.append((atom - atom.mean(axis=1, keepdims=True)).ravel())
    design = np.column_stack(columns)
    fitted = lsq_linear(design, target.ravel(), bounds=(0., coefficient_bound),
                        method="bvls", tol=1e-10, max_iter=200)
    residual = design @ fitted.x - target.ravel()
    return {"linear_contrast_mae": float(np.mean(np.abs(residual))),
            "linear_contrast_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "rank_fit_success": bool(fitted.success),
            "rank_fit_status": int(fitted.status),
            "rank_fit_optimality": float(fitted.optimality)}


def build_passive_candidates(p, q, y, *, source_p_nominal=None,
        source_q_nominal=None, source_p_error=0., source_q_error=0., kappa=.5,
        tolerance_factors=(0., .08, .16), max_candidates=32,
        wzzt_time_limit=5., wzzt_max_atoms=None, coefficient_bound=.2,
        neighbor_seed_count=4, run_wzzt=True, prior_proposals=()):
    """Build replaceable passive candidates without access to evaluation truth.

    Arrays use pu, with y = observed Vroot^2 - observed Vterminal^2.
    Known kappa=X/R is used in the RNJ proposal and contrast ranking.
    The existing WZZT proposal permits separate R/X and unknown source path;
    its larger proposal domain is deliberately NOT the AC evidence domain.
    WZZT has a per-solve budget, not a whole-call wall-time guarantee.
    prior_proposals is an iterable of old support families; these are scored
    and retained as proposals, never passed as WZZT mandatory/certified atoms.
    """
    started = perf_counter()
    p, q, y = _as_observations(p, q, y)
    t, n = p.shape
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    if not np.isfinite(coefficient_bound) or coefficient_bound <= 0:
        raise ValueError("coefficient_bound must be finite and positive")
    if isinstance(max_candidates, (bool, np.bool_)) or not isinstance(max_candidates, (int, np.integer)) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    if isinstance(neighbor_seed_count, (bool, np.bool_)) or not isinstance(neighbor_seed_count, (int, np.integer)) or neighbor_seed_count < 0:
        raise ValueError("neighbor_seed_count must be a nonnegative integer")
    if not np.isfinite(wzzt_time_limit) or wzzt_time_limit <= 0:
        raise ValueError("wzzt_time_limit must be finite and positive")
    tolerances = tuple(float(v) for v in tolerance_factors)
    if not tolerances or any(not np.isfinite(v) or v < 0 for v in tolerances):
        raise ValueError("tolerance_factors must be nonempty and nonnegative")
    if (source_p_nominal is None) != (source_q_nominal is None):
        raise ValueError("provide both nominal source P and Q, or neither")
    bp, bq = (_profile(v, t, name) for v, name in
              ((source_p_error, "source_p_error"), (source_q_error, "source_q_error")))
    if np.any(bp < 0) or np.any(bq < 0):
        raise ValueError("source error bounds must be nonnegative")
    if source_p_nominal is None and (np.any(bp) or np.any(bq)):
        raise ValueError("source error requires explicit nominal source inputs")
    source_inputs = None
    if source_p_nominal is not None:
        sp = _profile(source_p_nominal, t, "source_p_nominal")
        sq = _profile(source_q_nominal, t, "source_q_nominal")
        if np.any(sp < 0):
            raise ValueError("source_p_nominal must be nonnegative for current WZZT")
        source_inputs = PreparedSourceInputs((sp,), (sq,), (pd.RangeIndex(t),),
                                            ("declared_observation_side_nominal",))
    candidates = {}
    base_keys = set()
    diagnostics = {"rnj": [], "wzzt": {"called": False}}

    def add(supports, origin, base=False):
        family = normalize_family(supports, n)
        if family not in candidates:
            candidates[family] = {"proposal_origins": [], **contrast_rank(
                p + kappa * q, y, family, coefficient_bound)}
        if origin not in candidates[family]["proposal_origins"]:
            candidates[family]["proposal_origins"].append(origin)
        if base:
            base_keys.add(family)

    for index, proposal in enumerate(prior_proposals):
        add(proposal, f"prior_uncertified:{index}", base=True)

    # Existing estimator is called in its documented through-origin mode.
    # Temporal centering is a separate proposal preprocessing, not an AC law.
    for mode in ("raw_effective_pq", "time_centered_effective_pq"):
        effective = p + kappa * q
        target = y.copy()
        if mode.startswith("time_centered"):
            effective = effective - effective.mean(axis=0, keepdims=True)
            target -= target.mean(axis=0, keepdims=True)
        solver = {}
        try:
            r, _, r2, condition = fit_projected_sensitivity(
                [_scenario(effective, np.zeros_like(effective), target, mode)],
                diagnostics=solver)
            # With known X/R the effective R suffices. Supplying a zero X to
            # the unmodified geometry preserves its current normalization.
            geometry = sensitivity_geometry(r, np.zeros_like(r))
            scale = max(float(np.median(geometry.root_depths)), 1e-12)
            for tolerance in tolerances:
                tree = rooted_neighbor_joining(geometry.shared_paths, geometry.root_depths,
                    list(range(n)), -1, group_tolerance=tolerance * scale)
                add(rooted_clades(tree.edges, -1, list(range(n))),
                    f"rnj:{mode}:tolerance={tolerance:g}", base=True)
            effective_condition = float(np.linalg.cond(effective))
            diagnostics["rnj"].append({"mode": mode, "success": True,
                "r2": float(r2), "effective_pq_condition": effective_condition if np.isfinite(effective_condition) else None,
                "effective_pq_rank": int(np.linalg.matrix_rank(effective)),
                "augmented_pq_condition": float(condition) if np.isfinite(condition) else None,
                "augmented_zero_q_channel": True, "solver": solver})
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            diagnostics["rnj"].append({"mode": mode, "success": False,
                                       "error": f"{type(exc).__name__}: {exc}"})

    if run_wzzt:
        diagnostics["wzzt"]["called"] = True
        try:
            result = fit_source_aware_wzzt(
                [_scenario(p, q, y, "passive_observations")],
                source_inputs=source_inputs, source_p_error=bp, source_q_error=bq,
                certified_supports=(), max_atoms=2 * n - 1 if wzzt_max_atoms is None else wzzt_max_atoms,
                r_upper_bound=coefficient_bound, x_upper_bound=max(kappa, .1) * coefficient_bound,
                time_limit=wzzt_time_limit, mip_rel_gap=0.)
            diagnostics["wzzt"].update(result.summary())
            diagnostics["wzzt"]["mandatory_supports"] = [list(s) for s in result.mandatory_supports]
            diagnostics["wzzt"]["global_topology_optimality"] = False
            for index, fit in enumerate(result.path):
                add(fit.support_indices, f"wzzt:accepted_path:{index}", base=True)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            diagnostics["wzzt"]["error"] = f"{type(exc).__name__}: {exc}"
    if not candidates:
        add((), "fallback:star", base=True)

    # Choose neighborhood seeds only from observation-based rankings. Keep
    # base hypotheses available when capacity permits, even if their rank is bad.
    ordering = lambda family: (candidates[family]["linear_contrast_mae"], len(family), family)
    seeds = sorted(base_keys, key=ordering)[:neighbor_seed_count]
    for index, family in enumerate(seeds):
        for neighbor, move in family_neighbors(family, n).items():
            add(neighbor, f"local:seed={index}:{move}")
    base_order = sorted(base_keys, key=ordering)
    selected = base_order[:max_candidates]
    for family in sorted(candidates, key=ordering):
        if len(selected) == max_candidates:
            break
        if family not in selected:
            selected.append(family)
    rows = []
    for rank, family in enumerate(sorted(selected, key=ordering), 1):
        signature = repr((n, family)).encode("ascii")
        rows.append({"id": "passive_" + sha256(signature).hexdigest()[:12],
                     "supports": [list(s) for s in family], "rank": rank,
                     **candidates[family], "ac_evidence": None,
                     "certified_supports": [], "replaceable": True})
    return {"schema": 9, "n_terminal": n, "n_observation": t,
            "candidates": rows, "diagnostics": diagnostics,
            "generated_count": len(candidates), "retained_count": len(rows),
            "base_count": len(base_keys), "discarded_base_count": max(0, len(base_keys) - max_candidates),
            "coverage": "finite observation-generated proposal bank; not all radial topologies",
            "selection_uses_truth": False, "ranking_is_ac_evidence": False,
            "certified_supports": [], "rnj_supports_fixed_in_wzzt": False,
            "config": {"kappa": float(kappa), "tolerance_factors": tolerances,
                       "max_candidates": int(max_candidates), "wzzt_time_limit_per_solve": float(wzzt_time_limit),
                       "wzzt_max_atoms": 2 * n - 1 if wzzt_max_atoms is None else int(wzzt_max_atoms),
                       "coefficient_bound": float(coefficient_bound),
                       "neighbor_seed_count": int(neighbor_seed_count), "run_wzzt": bool(run_wzzt)},
            "seconds": perf_counter() - started}


__all__ = ["build_passive_candidates", "normalize_family", "family_neighbors", "contrast_rank"]
