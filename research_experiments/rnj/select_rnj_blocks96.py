"""Isolated two-terminal RNJ block screen for the authorized AC96 rerun.

This small selector is adapted explicitly from the frozen 72af runtime's
``rnj_wzzt.pipeline._select_boundary_blocks`` (lines 185--333 in the archived
package). It calls that runtime's fit, RX75 tree and block-bootstrap helpers;
it never modifies them. The only selection-policy changes are a minimum block
size of two and maximum internal-subclade frequency .15. The full-data finer
tree veto, Source rank gate and synchronized resampling remain in force.

Ordinary clade frequencies are recorded from the SAME bootstrap trees without
additional fits. They describe general downstream sets. Atomic peer frequencies
describe complete direct-terminal sibling sets and alone enter the block screen.
Neither is an impedance-weight confidence interval or topology probability.

Selected labels may subsequently define a shared binary z group in a separate
MILP caller; this module does not build, solve or change any MILP. A nonphysical
ordinary support may carry zero weights, so failed complete-peer semantics do
not by themselves prove exclusion of every true effective topology.

No runtime package is imported at module import time. Callers must pass their
already activated frozen pipeline and, for Source input, Source module.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


DEFAULT_PARAMETERS = {
    "bootstrap_replicates": 100, "block_length": 4, "confidence_threshold": .75,
    "maximum_candidate_count": 2, "tolerance_factor": .16, "seed": 20260903,
    "maximum_subclade_frequency": .15, "fine_tolerance_ratio": .5,
    "minimum_block_size": 2,
}


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite(value, name, minimum, maximum=None):
    try:
        finite = np.isscalar(value) and bool(np.isfinite(value))
    except TypeError:
        finite = False
    if isinstance(value, (bool, np.bool_)) or not finite:
        raise ValueError(f"{name} must be finite")
    if value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} outside its allowed range")


def _records(values):
    return [{"clade": sorted(c), "frequency": float(f)}
            for c, f in sorted(values.items(), key=lambda item: tuple(sorted(item[0])))]


def _clades(values):
    return [sorted(c) for c in sorted(values, key=lambda c: tuple(sorted(c)))]


def _tree_record(details, root, terminals):
    tree = details.get("tree")
    if tree is None:
        return None
    return {"root": int(root), "terminals": list(terminals),
            "edges": [[int(a), int(b), float(w)] for a, b, w in tree.edges]}


def _prefit_source_rank(scenarios, aligned_source):
    """Same conservative rank calculation as frozen multiscenario.py, no fit.

    This is needed only when the caller supplies an existing full-data R/X fit;
    reuse must not bypass the rank gate or require another QP fit.
    """
    columns = scenarios[0]["P_terminal"].columns
    design = np.vstack([np.hstack([s[name].loc[s["P_terminal"].index, columns].to_numpy(dtype=float)
                                   for name in ("P_terminal", "Q_terminal")]) for s in scenarios])
    source = np.column_stack((np.concatenate(aligned_source.p), np.concatenate(aligned_source.q)))
    u, singular, _ = np.linalg.svd(source, full_matrices=False)
    cutoff = np.finfo(float).eps*max(source.shape)*singular[0]
    source_rank = int(np.count_nonzero(singular > cutoff))
    basis = u[:, :source_rank]
    projected = design-basis@(basis.T@design)
    rank_tolerance = np.finfo(float).eps*max(projected.shape)*float(np.linalg.norm(design, ord=2))
    rank = int(np.count_nonzero(np.linalg.svd(projected, compute_uv=False) > rank_tolerance))
    return {"source_present": True, "source_design_rank": source_rank,
            "projected_design_rank": rank, "source_adjusted_identifiable": rank == design.shape[1],
            "rank_check_origin": "same frozen multiscenario SVD criterion, without repeating the supplied QP fit"}


def select_rnj_blocks96(
    scenarios, terminals, root, *, runtime_pipeline, runtime_source=None,
    source_inputs=None, sensitivity_fit=None, sensitivity_diagnostics=None,
    diagnostics=None, bootstrap_replicates=100, block_length=4,
    confidence_threshold=.75, maximum_candidate_count=2, tolerance_factor=.16,
    seed=20260903, maximum_subclade_frequency=.15, fine_tolerance_ratio=.5,
    minimum_block_size=2,
):
    """Return ``(full_clades, atomic_peer_frequency, selected, r2, condition)``.

    Inputs contain observations and known Source amplitudes only. Source input
    may be this runtime's PreparedSourceInputs or its public SourceInputs list.
    ``sensitivity_fit`` is the existing ``(R,X,r2,condition)`` on these same data;
    when supplied, its full-data fit is reused but every bootstrap is refitted.
    Source diagnostics may be retained with ``sensitivity_diagnostics``; their
    full-data rank gate is independently reproduced for a supplied fit.

    Empty eligible/selected groups are returned unchanged. Ordinary clade
    frequency does not replace the atomic-peer screen or select any supports.
    """
    _integer(bootstrap_replicates, "bootstrap_replicates", 1)
    _integer(block_length, "block_length", 1)
    _integer(maximum_candidate_count, "maximum_candidate_count", 0)
    _integer(minimum_block_size, "minimum_block_size", 2)
    _integer(seed, "seed", 0)
    _finite(confidence_threshold, "confidence_threshold", 0., 1.)
    _finite(maximum_subclade_frequency, "maximum_subclade_frequency", 0., 1.)
    _finite(tolerance_factor, "tolerance_factor", 0.)
    _finite(fine_tolerance_ratio, "fine_tolerance_ratio", np.finfo(float).tiny, 1.)
    terminals = [int(t) for t in terminals]
    if len(terminals) < 2 or len(set(terminals)) != len(terminals) or int(root) in terminals:
        raise ValueError("distinct terminal labels, excluding root, required")
    scenarios = runtime_pipeline.align_scenarios(list(scenarios))
    if not scenarios or any(list(map(int, s["P_terminal"].columns)) != terminals for s in scenarios):
        raise ValueError("scenario columns must retain supplied terminal order")
    aligned_source = None
    if source_inputs is not None:
        if runtime_source is None:
            raise ValueError("runtime_source required for validated Source alignment")
        prepared = source_inputs
        if not isinstance(prepared, runtime_pipeline.PreparedSourceInputs):
            prepared = runtime_pipeline.prepare_source_inputs(scenarios, source_inputs)
        amplitudes = runtime_source._copy_source_amplitudes(
            prepared, expected_lengths=tuple(len(s["P_terminal"]) for s in scenarios),
            time_indices=tuple(s["P_terminal"].index for s in scenarios),
        )
        if amplitudes is not None:
            aligned_source = runtime_pipeline.PreparedSourceInputs(
                p=amplitudes[0], q=amplitudes[1],
                time_indices=tuple(s["P_terminal"].index for s in scenarios), provenance=prepared.provenance,
            )
    fit_info = dict(sensitivity_diagnostics or {})
    supplied_fit = sensitivity_fit is not None
    if sensitivity_fit is None:
        sensitivity_fit = runtime_pipeline.fit_projected_sensitivity(
            scenarios, diagnostics=fit_info,
            **({"source_inputs": aligned_source} if aligned_source is not None else {}),
        )
    elif aligned_source is not None:
        fit_info.update(_prefit_source_rank(scenarios, aligned_source))
    n = len(terminals)
    if (len(sensitivity_fit) != 4 or any(np.shape(m) != (n, n) or not np.isfinite(m).all()
                                       for m in sensitivity_fit[:2])):
        raise ValueError("sensitivity_fit must contain matching finite R/X matrices, r2 and condition")
    details = {}
    full_clades, r2, condition = runtime_pipeline._rx75_rnj_clades(
        scenarios, terminals, root, tolerance_factor=tolerance_factor,
        sensitivity_fit=sensitivity_fit, tree_details=details,
    )
    full_peers = details["peers"]
    fine_details = {}
    fine_clades, _, _ = runtime_pipeline._rx75_rnj_clades(
        scenarios, terminals, root, tolerance_factor=tolerance_factor*fine_tolerance_ratio,
        sensitivity_fit=sensitivity_fit, tree_details=fine_details,
    )
    peer_counter, ordinary_counter, sub_counter = Counter(), Counter(), Counter()
    bootstrap_records = []
    unidentifiable = 0
    rng = np.random.default_rng(seed)
    for replicate in range(bootstrap_replicates):
        if aligned_source is None:
            sampled, indices = runtime_pipeline._moving_block_bootstrap_copy(scenarios, rng, block_length, return_indices=True)
            sampled_source = None
        else:
            sampled, indices = runtime_pipeline._moving_block_bootstrap_copy(scenarios, rng, block_length, return_indices=True)
            sampled_source = runtime_pipeline.PreparedSourceInputs(
                p=tuple(p[index] for p, index in zip(aligned_source.p, indices, strict=True)),
                q=tuple(q[index] for q, index in zip(aligned_source.q, indices, strict=True)),
                time_indices=tuple(s["P_terminal"].index for s in sampled), provenance=aligned_source.provenance,
            )
        sampled_info = {}
        fitted = runtime_pipeline.fit_projected_sensitivity(
            sampled, diagnostics=sampled_info,
            **({"source_inputs": sampled_source} if sampled_source is not None else {}),
        )
        sample_identifiable = bool(sampled_info.get("source_adjusted_identifiable", True))
        bootstrap_record = {"replicate": replicate, "identifiable": sample_identifiable,
            "valid_for_frequency_counts": sample_identifiable,
            "resample_indices": [np.asarray(index, dtype=int).tolist() for index in indices],
            "source_design_rank": sampled_info.get("source_design_rank"),
            "projected_design_rank": sampled_info.get("projected_design_rank"),
            "peers": None, "coarse_clades": None, "fine_clades": None}
        bootstrap_records.append(bootstrap_record)
        if not sample_identifiable:
            unidentifiable += 1
            continue
        sampled_details = {}
        clades, _, _ = runtime_pipeline._rx75_rnj_clades(
            sampled, terminals, root, tolerance_factor=tolerance_factor,
            sensitivity_fit=fitted, tree_details=sampled_details,
        )
        fine, _, _ = runtime_pipeline._rx75_rnj_clades(
            sampled, terminals, root, tolerance_factor=tolerance_factor*fine_tolerance_ratio,
            sensitivity_fit=fitted,
        )
        bootstrap_record.update(peers=_clades(sampled_details["peers"]),
                                coarse_clades=_clades(clades), fine_clades=_clades(fine))
        peer_counter.update(sampled_details["peers"])
        ordinary_counter.update(clades)
        sub_counter.update(c for c in full_peers if any(1 < len(child) < len(c) and child < c for child in clades | fine))
    valid = bootstrap_replicates-unidentifiable
    confidence = {c: peer_counter[c]/bootstrap_replicates for c in set(peer_counter) | full_peers}
    ordinary = {c: ordinary_counter[c]/bootstrap_replicates for c in set(ordinary_counter) | full_clades}
    sub = {c: sub_counter[c]/valid if valid else 1. for c in full_peers}
    full_fine = {c for c in full_peers if any(1 < len(child) < len(c) and child < c for child in fine_clades)}
    identifiable = fit_info.get("source_adjusted_identifiable", True)
    eligible, candidate_records = set(), []
    for clade in sorted(full_clades | full_peers, key=lambda c: tuple(sorted(c))):
        reasons = []
        if clade not in full_peers: reasons.append("not_complete_direct_terminal_peer")
        if len(clade) < minimum_block_size: reasons.append("below_minimum_block_size")
        if confidence.get(clade, 0.) < confidence_threshold: reasons.append("peer_frequency_below_threshold")
        if clade in full_peers and sub[clade] > maximum_subclade_frequency: reasons.append("internal_subclade_frequency_above_threshold")
        if clade in full_fine: reasons.append("full_data_fine_subclade_veto")
        if not identifiable: reasons.append("source_adjusted_design_unidentifiable")
        if not valid: reasons.append("no_identifiable_bootstrap_replicates")
        if not reasons: eligible.add(clade)
        candidate_records.append({"clade": sorted(clade), "size": len(clade), "is_full_peer": clade in full_peers,
            "ordinary_clade_frequency": ordinary.get(clade, 0.), "atomic_peer_frequency": confidence.get(clade, 0.),
            "subclade_frequency": sub.get(clade), "full_fine_veto": clade in full_fine,
            "eligible": not reasons, "rejection_reasons": reasons})
    selected = runtime_pipeline._select_disjoint(eligible, confidence, maximum_candidate_count)
    for record in candidate_records:
        clade = frozenset(record["clade"])
        record["selected"] = clade in selected
        record["eligible_but_not_selected_by_disjoint_or_capacity"] = clade in eligible and clade not in selected
    if diagnostics is not None:
        diagnostics.update(full_peers=full_peers, full_fine_subclades=full_fine, eligible=eligible,
            full_tree=_tree_record(details, root, terminals),
            full_fine_tree=_tree_record(fine_details, root, terminals),
            full_clades=_clades(full_clades), full_fine_clades=_clades(fine_clades),
            bootstrap_records=bootstrap_records,
            subclade_frequency=sub, subclade_frequency_records=_records(sub), source_fit=fit_info,
            source_adjusted_identifiable=bool(identifiable), unidentifiable_bootstrap_replicates=unidentifiable,
            valid_bootstrap_replicates=valid, minimum_block_size=minimum_block_size,
            maximum_subclade_frequency=maximum_subclade_frequency, fine_tolerance_ratio=fine_tolerance_ratio,
            ordinary_clade_frequency_records=_records(ordinary), atomic_peer_frequency_records=_records(confidence),
            frequency_denominators={"ordinary_clade": bootstrap_replicates, "atomic_peer": bootstrap_replicates, "internal_subclade": valid},
            candidate_records=candidate_records, supplied_sensitivity_fit_reused=supplied_fit,
            full_fine_veto_kept=True, truth_used_for_selection=False, empty_selection_retained=not bool(selected),
            parameters={"bootstrap_replicates": bootstrap_replicates, "block_length": block_length,
                "confidence_threshold": confidence_threshold, "maximum_candidate_count": maximum_candidate_count,
                "tolerance_factor": tolerance_factor, "seed": seed, "maximum_subclade_frequency": maximum_subclade_frequency,
                "fine_tolerance_ratio": fine_tolerance_ratio, "minimum_block_size": minimum_block_size},
            selection_policy="complete direct-terminal peers only; size>=2; peer>=.75; sub<=.15; full-fine veto retained; overrides explicitly recorded",
            ordinary_frequency_use="diagnostic only; never selects atomic groups", frequency_interpretation="bootstrap algorithmic stability, not edge-weight uncertainty or calibrated probability")
    return full_clades, confidence, selected, float(r2), float(condition)
