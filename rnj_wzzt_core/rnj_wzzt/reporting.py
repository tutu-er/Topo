"""Evaluation and serialized result records; truth is used only for scoring."""

from __future__ import annotations

from rnj_wzzt.estimation.laminar_l1_milp import solver_diagnostics_prove_optimality
from rnj_wzzt.graph.rooted_hierarchy import rooted_clades


def _truth_nontrivial_clades(net, terminals: list[int]) -> set[frozenset[int]]:
    edges = [
        (int(row.from_bus), int(row.to_bus))
        for row in net.branches.itertuples(index=False)
        if bool(row.is_true_closed)
    ]
    return {
        clade
        for clade in rooted_clades(edges, int(net.root_bus), terminals)
        if 1 < len(clade) < len(terminals)
    }


def _family_score(
    predicted: set[frozenset[int]],
    truth: set[frozenset[int]],
) -> dict:
    matched = predicted & truth
    precision = len(matched) / len(predicted) if predicted else float(not truth)
    recall = len(matched) / len(truth) if truth else float(not predicted)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "nontrivial_support_precision": precision,
        "nontrivial_support_recall": recall,
        "nontrivial_support_f1": f1,
    }


def _serialize(clade: frozenset[int]) -> str:
    return ",".join(map(str, sorted(clade)))


def _serialize_family(family: set[frozenset[int]]) -> str:
    return ";".join(
        _serialize(clade)
        for clade in sorted(family, key=lambda item: (len(item), tuple(sorted(item))))
    )


def _rnj_summary_row(
    *,
    case_name: str,
    scenario_count: int,
    samples_per_scenario: int,
    r2_score: float,
    condition: float,
    selection_seconds: float,
    full_clades: set[frozenset[int]],
    full_cherries: set[frozenset[int]],
    selected: list[frozenset[int]],
    confidence: dict[frozenset[int], float],
    truth_clades: set[frozenset[int]],
    truth_cherries: set[frozenset[int]],
) -> dict:
    """Build the one-row RNJ audit record for a case."""

    full_score = _family_score(full_clades, truth_clades)
    cherry_score = _family_score(full_cherries, truth_cherries)
    selected_score = _family_score(set(selected), truth_cherries)
    return {
        "case": case_name,
        "scenario_count": scenario_count,
        "samples_per_scenario": samples_per_scenario,
        "total_training_rows": scenario_count * samples_per_scenario,
        "fit_r2": r2_score,
        "condition_number": condition,
        "selection_seconds": selection_seconds,
        "full_rnj_clade_precision": full_score["nontrivial_support_precision"],
        "full_rnj_clade_recall": full_score["nontrivial_support_recall"],
        "full_rnj_clade_f1": full_score["nontrivial_support_f1"],
        "full_rnj_cherry_precision": cherry_score["nontrivial_support_precision"],
        "full_rnj_cherry_recall": cherry_score["nontrivial_support_recall"],
        "full_rnj_cherry_f1": cherry_score["nontrivial_support_f1"],
        "selected_count": len(selected),
        "selected_supports": ";".join(_serialize(clade) for clade in selected),
        "selected_confidences": ";".join(
            f"{confidence[clade]:.6g}" for clade in selected
        ),
        "selected_precision_against_truth": selected_score[
            "nontrivial_support_precision"
        ],
        "selected_recall_against_truth_cherries": selected_score[
            "nontrivial_support_recall"
        ],
        "true_cherries": _serialize_family(truth_cherries),
    }


def _rnj_candidate_rows(
    *,
    case_name: str,
    confidence: dict[frozenset[int], float],
    full_cherries: set[frozenset[int]],
    selected: list[frozenset[int]],
    truth_cherries: set[frozenset[int]],
    confidence_threshold: float,
) -> list[dict]:
    """Build traceable bootstrap-candidate records for one case."""

    rows = []
    candidates = sorted(
        set(confidence) | full_cherries,
        key=lambda item: (len(item), tuple(sorted(item))),
    )
    for clade in candidates:
        bootstrap_confidence = confidence.get(clade, 0.0)
        rows.append(
            {
                "case": case_name,
                "support_labels": _serialize(clade),
                "support_size": len(clade),
                "present_in_full_tree": clade in full_cherries,
                "bootstrap_confidence": bootstrap_confidence,
                "passes_threshold": (
                    clade in full_cherries
                    and bootstrap_confidence >= confidence_threshold
                ),
                "selected_top_k": clade in selected,
                "is_true_boundary_cherry": clade in truth_cherries,
            }
        )
    return rows


def _format_optional_number(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def _milp_result_row(
    *, case_name, initializer, result, selected, supports,
    predicted, truth_clades, elapsed,
) -> dict:
    """Keep topology scores and solver evidence in one output schema."""
    score = _family_score(predicted, truth_clades)
    missing = truth_clades - predicted
    extra = predicted - truth_clades
    diagnostics = [attempt.diagnostics for attempt in result.attempted_extensions]
    return {
        "case": case_name,
        "initializer": initializer,
        "milp_solver": result.path[result.selected_path_index].solver.solver,
        "rnj_initial_supports": ";".join(
            _serialize(clade) for clade in selected
        ),
        "frozen_support_count": len(supports),
        "frozen_leaf_singleton_count": len(result.terminal_labels),
        "frozen_rnj_block_count": len(selected),
        "support_search_mode": "unrestricted",
        "milp_terminal_count": len(result.terminal_labels),
        "elapsed_seconds": elapsed,
        "support_count": len(result.support_labels),
        "nontrivial_support_count": len(predicted),
        "clade_precision": score["nontrivial_support_precision"],
        "clade_recall": score["nontrivial_support_recall"],
        "clade_f1": score["nontrivial_support_f1"],
        "train_mae": result.train_mae,
        "validation_mae": result.validation_mae,
        "stop_reason": result.stop_reason,
        "attempt_count": len(result.attempted_extensions),
        "selected_path_index": result.selected_path_index,
        "path_length": len(result.path),
        "path_support_counts": ";".join(
            str(len(point.support_labels)) for point in result.path
        ),
        "path_validation_mae": ";".join(
            _format_optional_number(point.validation_mae)
            for point in result.path
        ),
        "attempt_statuses": ";".join(str(item.status) for item in diagnostics),
        "attempt_raw_statuses": ";".join(
            "" if item.raw_status is None else str(item.raw_status) for item in diagnostics
        ),
        "attempts_proven_optimal": sum(
            solver_diagnostics_prove_optimality(item) for item in diagnostics
        ),
        "attempt_runtimes": ";".join(
            f"{item.runtime_seconds:.6g}" for item in diagnostics
        ),
        "attempt_objectives": ";".join(
            _format_optional_number(item.objective) for item in diagnostics
        ),
        "attempt_dual_bounds": ";".join(
            _format_optional_number(item.dual_bound) for item in diagnostics
        ),
        "attempt_mip_gaps": ";".join(
            _format_optional_number(item.mip_gap) for item in diagnostics
        ),
        "attempt_messages": " | ".join(
            item.message.replace("\n", " ") for item in diagnostics
        ),
        "predicted_clades": _serialize_family(predicted),
        "true_clades": _serialize_family(truth_clades),
        "missing_clades": _serialize_family(missing),
        "extra_clades": _serialize_family(extra),
    }
