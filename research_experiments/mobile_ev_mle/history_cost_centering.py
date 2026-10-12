"""Explicit history-constant ablations of canonical frozen norm proxies.

Pairwise projection removes each pair's unrestricted affine history-profile
floor, and changes cross-pair design priorities. Common-floor centering removes
one common constant and preserves the original criterion/action ordering.
Neither floor is a calibrated likelihood threshold or a nonlinear AC MLE.
All direction/secant bounds continue to concern the resulting frozen norm.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from itertools import accumulate
import json
import numpy as np

# Always import the actual frozen package: solve_pair has a relative QR hook.
# Loading the core as a top-level staging module breaks that hook and can test
# a different class/module identity from the intended research implementation.
from research_experiments.mobile_ev_mle.ev_information_milp import FrozenPair


def _metadata_hash(metadata):
    def encode(value):
        if isinstance(value, (np.ndarray, np.generic)):
            return value.tolist()
        raise TypeError("centering metadata must contain serializable audit values")
    return sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                             allow_nan=False, default=encode).encode()).hexdigest()


@dataclass(frozen=True)
class HistoryProjection:
    projected: np.ndarray
    perpendicular: np.ndarray
    floor: float
    free_indices: np.ndarray
    fixed_indices: np.ndarray
    row_count: int
    rank: int
    singular_values: np.ndarray
    cutoff: float
    defect_lower: float
    defect_upper: float
    roundoff_guard: float
    discarded_positive_count: int


def _history_widths(pair):
    if not isinstance(pair.metadata, dict):
        raise ValueError("history metadata must be a dictionary")
    widths = pair.metadata.get("history_row_widths")
    if not isinstance(widths, (list, tuple, np.ndarray)) or not len(widths):
        raise ValueError("history_row_widths metadata must identify the historical prefix")
    if np.asarray(widths).ndim != 1:
        raise ValueError("history_row_widths must be one-dimensional")
    if any(isinstance(w, (bool, np.bool_)) or not isinstance(w,(int,np.integer)) or w < 1 for w in widths):
        raise ValueError("history_row_widths must be positive integers")
    total = int(sum(widths))
    future = pair.metadata.get("future_row_start", total)
    if (isinstance(future, (bool, np.bool_)) or not isinstance(future, (int, np.integer))
            or total > len(pair.c) or future != total):
        raise ValueError("historical row prefix conflicts with the frozen residual layout")
    if np.any(pair.B[:total] != 0):
        raise ValueError("history-control matrix must be exactly zero for constant centering")
    return tuple(map(int,widths))


def _defect_bounds(pair, rows, new_history_intercept, removed_total):
    """Bound the actual written norm difference on the ORIGINAL free box.

    If e and t are the original/new effective historical intercepts, the
    algebraic defect is e'e-t't-removed_total+2(e-t)'M z. This form does not
    assume an exact SVD orthogonality relation. Floating guard is explicit,
    conservative engineering arithmetic, not directed-rounding intervals.
    """
    free = np.flatnonzero(pair.lower < pair.upper)
    fixed = np.flatnonzero(pair.lower == pair.upper)
    shift = pair.A[:rows, fixed] @ pair.lower[fixed]
    effective = pair.c[:rows] + shift
    target = np.asarray(new_history_intercept) + shift
    matrix = pair.A[:rows, free]
    constant = float(effective @ effective - target @ target - removed_total)
    coefficient = 2 * (effective - target) @ matrix
    low_terms = np.minimum(coefficient * pair.lower[free], coefficient * pair.upper[free])
    high_terms = np.maximum(coefficient * pair.lower[free], coefficient * pair.upper[free])
    radius = np.maximum(abs(pair.lower[free]), abs(pair.upper[free]))
    fixed_absolute = abs(pair.A[:rows, fixed]) @ abs(pair.lower[fixed])
    variable_absolute = abs(matrix) @ radius
    original_absolute = abs(pair.c[:rows]) + fixed_absolute + variable_absolute
    new_absolute = abs(new_history_intercept) + fixed_absolute + variable_absolute
    magnitude = (1.0 + abs(removed_total) + abs(constant)
        + float(original_absolute @ original_absolute + new_absolute @ new_absolute)
        + float(abs(low_terms).sum() + abs(high_terms).sum()))
    guard = 256 * np.finfo(float).eps * magnitude
    lower, upper = constant + float(low_terms.sum()) - guard, constant + float(high_terms.sum()) + guard
    if not all(np.isfinite(value) for value in (lower, upper, guard)):
        raise ValueError("nonfinite history decomposition bound")
    return float(lower), float(upper), float(guard)


def project_history(pair, *, rcond=None):
    """Project the entire H prefix after absorbing fixed box coordinates.

    Theoretical P is the orthogonal projector onto all *free* H columns.
    A numerical SVD cutoff is recorded rather than hidden. The signed affine
    decomposition defect is bounded on the actual free-coordinate box, so a
    truncated tiny column is not silently declared exactly perpendicular.
    """
    rows = sum(_history_widths(pair))
    free = np.flatnonzero(pair.lower < pair.upper)
    fixed = np.flatnonzero(pair.lower == pair.upper)
    fixed_shift = pair.A[:rows,fixed]@pair.lower[fixed]
    effective = pair.c[:rows]+fixed_shift
    matrix = pair.A[:rows,free]
    if rcond is not None and (isinstance(rcond, (bool, np.bool_)) or not np.isfinite(rcond) or rcond < 0):
        raise ValueError("rcond must be finite nonnegative")
    if matrix.shape[1]:
        vectors,singular,_ = np.linalg.svd(matrix,full_matrices=False)
        cutoff = (np.finfo(float).eps*max(matrix.shape) if rcond is None else float(rcond))*singular[0] if len(singular) else 0.
        rank = int(np.count_nonzero(singular > cutoff))
        basis = vectors[:,:rank]
        projected = basis@(basis.T@effective)
    else:
        singular,cutoff,rank = np.zeros(0),0.,0
        projected = np.zeros_like(effective)
    perpendicular = effective-projected
    floor = float(perpendicular@perpendicular)
    discarded_positive = int(np.count_nonzero(singular[rank:] > 0.0))
    lower, upper, guard = _defect_bounds(pair, rows, projected-fixed_shift, floor)
    return HistoryProjection(projected,perpendicular,floor,free,fixed,rows,rank,singular,
                             float(cutoff),lower,upper,float(guard),discarded_positive)


def _audit_record(pair,projection):
    widths = _history_widths(pair)
    blocks = []
    for start,end in zip((0,*tuple(accumulate(widths))[:-1]),accumulate(widths)):
        blocks.append(dict(row_start=start,row_end=end,
            projected_norm_squared=float(projection.projected[start:end]@projection.projected[start:end]),
            perpendicular_norm_squared=float(projection.perpendicular[start:end]@projection.perpendicular[start:end])))
    return dict(pair=pair.name,history_row_count=projection.row_count,history_blocks=blocks,
        projection_scope="one joint historical prefix, after fixed coordinates are absorbed",
        free_history_rank=projection.rank,free_coordinate_count=len(projection.free_indices),
        fixed_coordinate_count=len(projection.fixed_indices),singular_values=projection.singular_values.tolist(),
        svd_cutoff=projection.cutoff,projected_history_floor=projection.floor,
        unrestricted_affine_history_floor=projection.floor if not projection.discarded_positive_count else None,
        discarded_positive_singular_values=projection.discarded_positive_count,
        unrestricted_floor_identity_condition="full-column-space orthogonal projection in real arithmetic",
        original_history_center_cost=float(np.sum((pair.c[:projection.row_count]
            +pair.A[:projection.row_count,projection.fixed_indices]@pair.lower[projection.fixed_indices])**2)),
        decomposition_defect_lower=projection.defect_lower,decomposition_defect_upper=projection.defect_upper,
        roundoff_guard=projection.roundoff_guard,strict_interval_arithmetic=False,
        nonlinear_history_best_loss_computed=False,statistical_confidence_region=False)


def center_history_costs(pairs, *, mode="pairwise_projection", common_floor=None, rcond=None):
    """Return new FrozenPairs and explicit semantics, leaving originals intact.

    mode='pairwise_projection': q_new,k = q_old,k - floor_k (exact projection).
    mode='common_floor': full-column-space projection and sqrt(floor_k-C)
    give q_new,k=q_old,k-C in real arithmetic. Positive SVD truncation is
    rejected in this exact-equivalence mode; actual floating defects remain.
    The latter preserves max_u min_k q and q(u)-q(none) action comparisons.
    A common C can be selected across fixed/refit families in one concatenated
    call. Explicit C must satisfy 0<=C<=min(floor_k); no negative radicand is
    clipped. Existing zero-width parameter coordinates are handled correctly.
    """
    pairs = list(pairs)
    if not pairs or any(not isinstance(pair, FrozenPair) for pair in pairs):
        raise ValueError("a nonempty collection of frozen research package FrozenPairs is required")
    if len({p.name for p in pairs}) != len(pairs):
        raise ValueError("a uniquely named pair collection is required")
    if mode not in ("pairwise_projection","common_floor"):
        raise ValueError("unknown centering mode")
    if mode == "pairwise_projection" and common_floor is not None:
        raise ValueError("common_floor is only meaningful for common_floor mode")
    if any(not isinstance(pair.metadata, dict) for pair in pairs):
        raise ValueError("history metadata must be a dictionary")
    if any("history_cost_centering" in pair.metadata for pair in pairs):
        raise ValueError("already centered pairs must not be centered again")
    effective_rcond = 0.0 if mode == "common_floor" and rcond is None else rcond
    projections = [project_history(pair,rcond=effective_rcond) for pair in pairs]
    if mode == "common_floor" and any(p.discarded_positive_count for p in projections):
        raise ValueError("common_floor requires full-column-space projection; positive SVD truncation changes the criterion")
    minimum = min(p.floor for p in projections)
    if mode == "common_floor":
        if isinstance(common_floor, (bool, np.bool_)):
            raise ValueError("common floor must be a real scalar, not a boolean")
        constant = minimum if common_floor is None else float(common_floor)
        if not np.isfinite(constant) or constant < 0 or constant > minimum:
            raise ValueError("common floor must lie between zero and every pair's affine floor")
    else:
        constant = None
    transformed,records = [],[]
    for pair,projection in zip(pairs,projections):
        c = pair.c.copy()
        # Retain fixed columns, and cancel their known offset in c. This gives
        # projected effective H residual for every feasible original z.
        shift = pair.A[:projection.row_count,projection.fixed_indices]@pair.lower[projection.fixed_indices]
        c[:projection.row_count] = projection.projected-shift
        b,a = pair.B.copy(),pair.A.copy()
        removed = projection.floor if constant is None else constant
        residual_constant = 0. if constant is None else projection.floor-constant
        if mode == "common_floor":
            c = np.r_[c,np.sqrt(residual_constant)]
            b = np.vstack((b,np.zeros((1,pair.control_dimension))))
            a = np.vstack((a,np.zeros((1,len(pair.lower)))))
        metadata = deepcopy(pair.metadata)
        record = _audit_record(pair,projection)
        retained_actual = float(c[-1] * c[-1]) if mode == "common_floor" else 0.0
        dl, du, guard = _defect_bounds(pair, projection.row_count,
                                      c[:projection.row_count], removed + retained_actual)
        record.update(decomposition_defect_lower=dl, decomposition_defect_upper=du,
                      roundoff_guard=guard, retained_constant_row_square=retained_actual)
        metadata["original_ac_rollback_compatible"] = False
        metadata["requires_original_pair_for_ac_rollback"] = True
        metadata["history_cost_centering"] = {**record, "mode":mode,
            "removed_constant":removed,"retained_perpendicular_constant":residual_constant,
            "original_pair_hash":pair.fingerprint(),
            "original_metadata_hash":_metadata_hash(pair.metadata),
            "original_residual_row_count":len(pair.c),
            "added_constant_row":len(pair.c) if mode=="common_floor" else None,
            "pair_specific_offsets_change_design_criterion":mode=="pairwise_projection",
            "core_QR_layout_preserved":True,
            "ac_rollback_scope":"use original pair and same physical z; transformed residual is not original AC likelihood"}
        new = FrozenPair(pair.name,c,b,a,pair.F,pair.lower,pair.upper,pair.scale,metadata)
        transformed.append(new)
        records.append({**metadata["history_cost_centering"],"transformed_pair_hash":new.fingerprint()})
    summary = dict(mode=mode,common_floor=constant,minimum_pair_floor=minimum,
        records=records,criterion_action_order_preserved_in_exact_arithmetic=mode=="common_floor",
        no_ev_gain_preserved_in_exact_arithmetic=mode=="common_floor",
        exact_identity_condition="full-column-space orthogonal projection; floating defects bounded separately",
        numerical_action_order_certified=False,
        canonical_norm_preserved=True,proxy_bound_scope="transformed frozen affine norm only",
        bound_is_nonlinear_AC_certificate=False,loss_is_exact_nonlinear_profile_loss=False,
        relative_historical_bad_fit_evidence_removed=mode=="pairwise_projection",
        decomposition_defect_global_lower=min(r["decomposition_defect_lower"] for r in records),
        decomposition_defect_global_upper=max(r["decomposition_defect_upper"] for r in records))
    return transformed,summary


def original_bounds_from_common_centering(lower,upper,summary):
    """Restore numerical original-proxy bounds, including recorded SVD defect.

    The same formula applies pointwise and to max_u min_k q because all pairs
    share the same removed C. Pairwise offsets do not permit this simple
    action-independent restoration.
    """
    if summary["mode"] != "common_floor":
        raise ValueError("only common centering permits a common restored objective bound")
    lower,upper = float(lower),float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("finite ordered centered objective bounds required")
    constant = summary["common_floor"]
    return dict(lower=lower+constant+summary["decomposition_defect_global_lower"],
                upper=upper+constant+summary["decomposition_defect_global_upper"],
                includes_numerical_decomposition_guard=True,
                strict_interval_arithmetic=False)


def evaluate_original_ac_witness(context, original_pair, centered_pair, u, z):
    """Back-substitute physical parameters only through the verified original.

    Core nonlinear rollback must never receive centered c/extra constant rows.
    It reports one feasible nonlinear witness, not a nonlinear infimum.
    """
    record = centered_pair.metadata.get("history_cost_centering")
    if not isinstance(record, dict) or record.get("original_pair_hash") != original_pair.fingerprint():
        raise ValueError("centered pair does not match the supplied original pair")
    if record.get("original_metadata_hash") != _metadata_hash(original_pair.metadata):
        raise ValueError("original physical metadata changed after centering")
    if "history_cost_centering" in original_pair.metadata:
        raise ValueError("AC rollback requires an uncentered original pair")
    from research_experiments.mobile_ev_mle.adaptive_history import evaluate_frozen_pair_actual
    raw = {key: getattr(original_pair, key) for key in
           ("name", "c", "B", "A", "F", "lower", "upper", "scale", "metadata")}
    return evaluate_frozen_pair_actual(context, raw, u, z)
