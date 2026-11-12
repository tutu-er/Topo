"""Observed-root NJ/RNJ controls for the paired 96-point source experiment.

The source P/Q time series are allowed inputs; its attachment is never read.
For nonzero source series, Frisch--Waugh--Lovell (FWL) projection eliminates
two unconstrained terminal-response vectors before the existing ordered R/X
regression.  This is an experiment-only extension of the NJ/RNJ controls:
the source response is not constrained to a physical root-to-source path.
All scenarios share the same response vectors.  No intercept is added.
"""

from __future__ import annotations

from collections import Counter
from math import ceil
from time import perf_counter

import numpy as np
import pandas as pd

from rnj_wzzt.estimation.multiscenario import _aligned_arrays, fit_projected_sensitivity
from rnj_wzzt.estimation.unmetered_inputs import (
    PreparedSourceInputs,
    SourceInputs,
    prepare_source_inputs,
)
from rnj_wzzt.graph.bootstrap import _boundary_cherries, _select_disjoint
from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
from rnj_wzzt.graph.rooted_neighbor_joining import rooted_neighbor_joining
from rnj_wzzt.graph.sensitivity_geometry import sensitivity_geometry
from research_experiments.rnj.rooted_study_baselines import infer_classical_nj


def _aligned_inputs(scenarios, source_inputs):
    """Return only permitted aligned measurement frames and source amplitudes."""
    scenarios = list(scenarios)
    designs, targets = _aligned_arrays(scenarios)
    columns = scenarios[0]["P_terminal"].columns.copy()
    aligned = []
    for scenario, design, target in zip(scenarios, designs, targets, strict=True):
        index = scenario["P_terminal"].index.copy()
        n = len(columns)
        aligned.append({
            "name": str(scenario.get("name", "scenario")),
            "P_terminal": pd.DataFrame(design[:, :n], index=index, columns=columns),
            "Q_terminal": pd.DataFrame(design[:, n:], index=index, columns=columns),
            "drop_target": pd.DataFrame(target, index=index, columns=columns),
        })
    if isinstance(source_inputs, PreparedSourceInputs):
        fields = (source_inputs.p, source_inputs.q, source_inputs.time_indices,
                  source_inputs.provenance)
        if len(fields[0]) != len(aligned) or any(len(v) != len(aligned) for v in fields):
            raise ValueError("source inputs must have one block per scenario")
        source_inputs = [
            SourceInputs(pd.Series(p, index=index), pd.Series(q, index=index), provenance)
            for p, q, index, provenance in zip(*fields, strict=True)
        ]
    prepared = prepare_source_inputs(aligned, source_inputs)
    return aligned, prepared


def _source_design(prepared, scenarios):
    if prepared is None:
        return np.zeros((sum(len(s["P_terminal"]) for s in scenarios), 2))
    return np.vstack([np.column_stack((p, q))
                      for p, q in zip(prepared.p, prepared.q, strict=True)])


def _source_svd(source):
    left, singular, right = np.linalg.svd(source, full_matrices=False)
    tolerance = (max(source.shape) * np.finfo(float).eps * singular[0]
                 if singular.size else 0.0)
    rank = int(np.count_nonzero(singular > tolerance))
    return left[:, :rank], singular[:rank], right[:rank], tolerance


def _residualize_source(scenarios, prepared):
    """Globally project P, Q, and Y against the same source column space."""
    designs, targets = _aligned_arrays(scenarios)
    design, target = np.vstack(designs), np.vstack(targets)
    source = _source_design(prepared, scenarios)
    basis, singular, right, tolerance = _source_svd(source)
    residual_design = design - basis @ (basis.T @ design)
    residual_target = target - basis @ (basis.T @ target)
    projected, start = [], 0
    for scenario in scenarios:
        count, n = scenario["P_terminal"].shape
        rows = slice(start, start + count)
        index, columns = scenario["P_terminal"].index, scenario["P_terminal"].columns
        projected.append({
            "name": scenario["name"],
            "P_terminal": pd.DataFrame(residual_design[rows, :n], index=index, columns=columns),
            "Q_terminal": pd.DataFrame(residual_design[rows, n:], index=index, columns=columns),
            "drop_target": pd.DataFrame(residual_target[rows], index=index, columns=columns),
        })
        start += count
    return projected, design, target, source, basis, singular, right, tolerance


def fit_tree(scenarios, source_inputs, method, *, root_bus=1):
    """Fit ``NJ`` or ``RNJ`` using observations and known source amplitudes.

    ``root_bus`` is the observed physical root label, default 1 for this bank;
    no network or simulation truth is required.  R/X are the ordered regression
    matrices used to infer the tree, rather than a subsequent tree refit.
    Source response ``g`` has rows for source P/Q and terminal columns.  For
    rank-one source series its reported value is the minimum-norm solution;
    only the combined response is identified by this nuisance regression.
    """
    started = perf_counter()
    method = str(method).upper()
    if method not in {"NJ", "RNJ"}:
        raise ValueError("method must be 'NJ' or 'RNJ'")
    aligned, prepared = _aligned_inputs(scenarios, source_inputs)
    terminals = [int(v) for v in aligned[0]["P_terminal"].columns]
    root_bus = int(root_bus)
    if root_bus in terminals:
        raise ValueError("observed root must be distinct from terminal labels")
    projection_started = perf_counter()
    (projected, design, target, source, basis, singular, right,
     svd_tolerance) = _residualize_source(aligned, prepared)
    projection_seconds = perf_counter() - projection_started
    diagnostics = {}
    fit_started = perf_counter()
    r, x, projected_r2, condition = fit_projected_sensitivity(
        projected if singular.size else aligned, diagnostics=diagnostics,
    )
    sensitivity_seconds = perf_counter() - fit_started
    recovery_started = perf_counter()
    residual = target - design @ np.vstack((r, x))
    g = (right.T @ ((basis.T @ residual) / singular[:, None])
         if singular.size else np.zeros((2, len(terminals))))
    prediction = design @ np.vstack((r, x)) + source @ g
    squared_residual = float(np.sum((prediction - target) ** 2))
    centered_total = float(np.sum((target - target.mean(axis=0)) ** 2))
    if centered_total:
        r2 = 1.0 - squared_residual / centered_total
    else:
        r2 = float(squared_residual <= np.finfo(float).eps * float(np.sum(target ** 2)))
    source_recovery_seconds = perf_counter() - recovery_started
    tree_started = perf_counter()
    if method == "NJ":
        tree = infer_classical_nj(r, x, terminals, root_bus, collapse_factor=.16)
    else:
        geometry = sensitivity_geometry(r, x)
        scale = max(float(np.median(geometry.root_depths)), 1e-12)
        tree = rooted_neighbor_joining(
            geometry.shared_paths, geometry.root_depths, terminals, root_bus,
            group_tolerance=.16 * scale,
        )
    clades = rooted_clades(tree.edges, root_bus, terminals)
    tree_seconds = perf_counter() - tree_started
    residual_design = design - basis @ (basis.T @ design)
    diagnostics.update({
        "source_model": "unconstrained_terminal_response_FWL" if singular.size else "none",
        "source_location_used": False,
        "source_path_constraints": False,
        "intercept_added": False,
        "source_rank": int(singular.size),
        "source_singular_values": singular.tolist(),
        "source_svd_tolerance": float(svd_tolerance),
        "source_condition_identified_subspace": (float(singular[0] / singular[-1])
                                                  if singular.size else None),
        "raw_design_rank": int(np.linalg.matrix_rank(design)),
        "projected_design_rank": int(np.linalg.matrix_rank(residual_design)),
        "raw_design_condition": float(np.linalg.cond(design)),
        "projected_design_condition": float(condition),
        "source_augmented_design_rank": int(np.linalg.matrix_rank(np.column_stack((design, source)))),
        "source_adjusted_squared_residual_sum": squared_residual,
        "source_response_identified_separately": bool(singular.size == 2),
        "collapse_or_tolerance_factor": .16,
        "matrix_metric_basis": "ordered_regression_before_tree_inference",
    })
    elapsed = perf_counter() - started
    return {
        "method": method, "r_matrix": r, "x_matrix": x, "R": r, "X": x,
        "tree": tree, "clades": clades, "terminals": tuple(terminals), "root_bus": root_bus,
        "r2_score": float(r2), "projected_r2_score": float(projected_r2),
        "condition": float(condition), "source_rank": int(singular.size),
        "source_response_g": g, "diagnostics": diagnostics,
        "timing": {"projection_seconds": projection_seconds,
                   "sensitivity_seconds": sensitivity_seconds,
                   "source_recovery_seconds": source_recovery_seconds,
                   "tree_seconds": tree_seconds, "elapsed_seconds": elapsed},
        "elapsed_seconds": elapsed,
    }


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _moving_block_source_copy(scenarios, prepared, rng, block_length):
    """Resample measurement and source rows together; retain fresh unique labels."""
    sampled, sources = [], []
    for block, scenario in enumerate(scenarios):
        count = len(scenario["P_terminal"])
        if count < 2:
            raise ValueError("each scenario needs at least two rows")
        length = min(block_length, count)
        starts = rng.integers(0, count, size=ceil(count / length))
        positions = np.concatenate([(s + np.arange(length)) % count for s in starts])[:count]
        replicate = {"name": scenario["name"]}
        for key in ("P_terminal", "Q_terminal", "drop_target"):
            replicate[key] = scenario[key].iloc[positions].reset_index(drop=True)
        sampled.append(replicate)
        if prepared is not None:
            sources.append(SourceInputs(
                pd.Series(prepared.p[block][positions]), pd.Series(prepared.q[block][positions]),
                prepared.provenance[block],
            ))
    return sampled, sources if prepared is not None else None


def select_rnj_initial_supports(
    scenarios, source_inputs, *, replicates=100, block_length=4,
    threshold=.75, max_blocks=2, seed=20261018, root_bus=1, full_fit=None,
    wall_seconds=None,
):
    """Select disjoint, stable full-data RNJ boundary blocks without contraction.

    Frequencies use all requested replicates as denominator, including failed
    fits.  Failures are saved individually and never replaced.  This function
    neither reads true topology nor drops a selected block based on its truth.
    ``full_fit`` may reuse the same full-data RNJ result from ``fit_tree``.
    """
    started = perf_counter()
    replicates = _positive_integer(replicates, "replicates")
    block_length = _positive_integer(block_length, "block_length")
    max_blocks = _positive_integer(max_blocks, "max_blocks", 0)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    if wall_seconds is not None and (not np.isfinite(wall_seconds) or wall_seconds <= 0):
        raise ValueError("wall_seconds must be finite and positive")
    deadline = float("inf") if wall_seconds is None else started + wall_seconds
    aligned, prepared = _aligned_inputs(scenarios, source_inputs)
    if full_fit is None:
        full_fit = fit_tree(aligned, prepared, "RNJ", root_bus=root_bus)
    elif full_fit["method"] != "RNJ" or full_fit["root_bus"] != int(root_bus):
        raise ValueError("full_fit must be the RNJ result with the same observed root")
    full_clades = set(full_fit["clades"])
    full_cherries = _boundary_cherries(full_clades)
    counts, failures, attempted, successes = Counter(), [], 0, 0
    rng = np.random.default_rng(seed)
    bootstrap_started = perf_counter()
    for replicate in range(replicates):
        if perf_counter() >= deadline:
            break
        attempted += 1
        try:
            sampled, sources = _moving_block_source_copy(aligned, prepared, rng, block_length)
            result = fit_tree(sampled, sources, "RNJ", root_bus=root_bus)
            counts.update(_boundary_cherries(result["clades"]))
            successes += 1
        except Exception as exc:
            failures.append({"replicate": replicate, "error_type": type(exc).__name__,
                             "error": str(exc)})
    bootstrap_seconds = perf_counter() - bootstrap_started
    confidence = {c: counts[c] / replicates for c in set(counts) | full_cherries}
    stable = {c for c in full_cherries if confidence[c] >= threshold}
    selected = _select_disjoint(stable, confidence, max_blocks)
    terminals = tuple(full_fit["terminals"])
    lookup = {v: i for i, v in enumerate(terminals)}
    elapsed = perf_counter() - started
    return {
        "selected": selected,
        "selected_support_labels": [tuple(sorted(c)) for c in selected],
        "selected_support_indices": [tuple(sorted(lookup[v] for v in c)) for c in selected],
        "full_clades": full_clades, "full_cherries": full_cherries,
        "confidence": confidence, "bootstrap_failures": failures,
        "bootstrap_successes": successes, "bootstrap_attempted": attempted,
        "bootstrap_unattempted": replicates - attempted,
        "status": "complete" if attempted == replicates else "bootstrap_time_limit",
        "replicates": replicates,
        "block_length": block_length, "threshold": float(threshold), "max_blocks": max_blocks,
        "seed": int(seed), "truth_used_for_selection": False,
        "wall_seconds": wall_seconds,
        "fit": full_fit, "timing": {"bootstrap_seconds": bootstrap_seconds,
                                    "full_fit_seconds": full_fit["elapsed_seconds"],
                                    "elapsed_seconds": elapsed},
        "elapsed_seconds": elapsed,
    }


__all__ = ["fit_tree", "select_rnj_initial_supports"]
