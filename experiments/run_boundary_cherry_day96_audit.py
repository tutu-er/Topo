"""Audit a few stable boundary-cherry candidates from one noisy 96-point day."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from math import ceil
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from experiments.run_laminar_l1_case33_integration import _family_score, _terminal_labels
from experiments.run_laminar_l1_noisy_multicase import (
    PAPER_STYLE_CASES,
    _truth_nontrivial_clades,
    build_benchmark_cases,
)
from experiments.run_tree_candidate_method_audit import (
    METHODS,
    _matrix_relative_error,
    _serialize_family,
    _training_block,
    _tree_clades,
)
from terminal_case33.estimation.multiscenario import fit_projected_sensitivity
from terminal_case33.utils.io import ensure_dir


DEFAULT_OUTPUT = Path("outputs/boundary_cherry_day96_audit")


def _boundary_cherries(clades: set[frozenset[int]]) -> set[frozenset[int]]:
    """Return the inclusion-minimal nontrivial rooted clades.

    In a reduced rooted latent tree these are precisely the terminal sets below
    peripheral hidden nodes whose downstream children are all terminals.
    """

    return {
        clade
        for clade in clades
        if not any(other < clade for other in clades)
    }


def _moving_block_bootstrap_copy(
    scenarios: Sequence[dict],
    rng: np.random.Generator,
    block_length: int,
) -> list[dict]:
    """Circular moving-block bootstrap preserving short-range time dependence."""

    sampled: list[dict] = []
    for scenario in scenarios:
        count = len(scenario["P_terminal"])
        if count < 2:
            raise ValueError("each scenario needs at least two rows")
        length = min(int(block_length), count)
        block_count = ceil(count / length)
        starts = rng.integers(0, count, size=block_count)
        indices = np.concatenate(
            [(start + np.arange(length)) % count for start in starts]
        )[:count]
        replicate = {"name": str(scenario.get("name", "scenario"))}
        for key in ("P_terminal", "Q_terminal", "drop_target"):
            replicate[key] = scenario[key].iloc[indices].reset_index(drop=True)
        sampled.append(replicate)
    return sampled


def _support_text(clade: frozenset[int]) -> str:
    return ",".join(map(str, sorted(clade)))


def _select_disjoint(
    candidates: set[frozenset[int]],
    confidence: dict[frozenset[int], float],
    maximum_count: int,
) -> list[frozenset[int]]:
    selected: list[frozenset[int]] = []
    occupied: set[int] = set()
    for clade in sorted(
        candidates,
        key=lambda item: (-confidence[item], -len(item), tuple(sorted(item))),
    ):
        if len(selected) >= maximum_count:
            break
        if occupied.isdisjoint(clade):
            selected.append(clade)
            occupied.update(clade)
    return selected


def run(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    cases: tuple[str, ...] = PAPER_STYLE_CASES,
    seed: int = 20260903,
    sample_count: int = 96,
    train_count: int = 96,
    modes: tuple[str, ...] = ("R", "RX_75R_25X"),
    bootstrap_replicates: int = 100,
    block_length: int = 4,
    confidence_threshold: float = 0.75,
    maximum_candidate_count: int = 2,
    rnj_tolerance_factor: float = 0.16,
    rg_tolerance: float = 0.03,
    ridge_alpha: float = 1e-6,
    response_laplace_fraction: float = 0.05,
    outlier_fraction: float = 0.05,
    gross_outlier_fraction: float = 1.5,
) -> dict:
    if sample_count < 96 or train_count < 96:
        raise ValueError("this audit requires at least one complete 96-point training day")
    if train_count > sample_count:
        raise ValueError("train_count cannot exceed sample_count")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if block_length < 1:
        raise ValueError("block_length must be positive")

    output = ensure_dir(output_dir)
    design_rows: list[dict] = []
    full_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    selected_rows: list[dict] = []

    for case_name, net in build_benchmark_cases(cases).items():
        terminals = _terminal_labels(net)
        root = int(net.root_bus)
        truth_clades = _truth_nontrivial_clades(net, terminals)
        truth_cherries = _boundary_cherries(truth_clades)
        training, r_true, x_true = _training_block(
            case_name,
            net,
            seed=seed,
            regime="response_outliers",
            sample_count=sample_count,
            train_count=train_count,
            response_laplace_fraction=response_laplace_fraction,
            outlier_fraction=outlier_fraction,
            gross_outlier_fraction=gross_outlier_fraction,
        )
        r_fit, x_fit, fit_r2, condition = fit_projected_sensitivity(
            training,
            alpha=ridge_alpha,
            constraint_mode="ordered",
        )
        design = np.hstack(
            [
                training[0]["P_terminal"].to_numpy(dtype=float),
                training[0]["Q_terminal"].to_numpy(dtype=float),
                np.ones((train_count, 1), dtype=float),
            ]
        )
        rank = int(np.linalg.matrix_rank(design))
        design_rows.append(
            {
                "case": case_name,
                "terminal_count": len(terminals),
                "sample_count": sample_count,
                "training_rows": train_count,
                "dense_regressor_count": int(design.shape[1]),
                "design_rank": rank,
                "residual_degrees_of_freedom": int(train_count - rank),
                "condition_number": float(condition),
                "dense_fit_r2": float(fit_r2),
                "dense_R_relative_frobenius_error": _matrix_relative_error(r_fit, r_true),
                "dense_X_relative_frobenius_error": _matrix_relative_error(x_fit, x_true),
                "true_boundary_cherries": _serialize_family(truth_cherries),
            }
        )

        full: dict[tuple[str, str], set[frozenset[int]]] = {}
        counters = {
            (method, mode): Counter()
            for method in METHODS
            for mode in modes
        }
        for method in METHODS:
            for mode in modes:
                exact_clades, exact_forced = _tree_clades(
                    r_true,
                    x_true,
                    mode=mode,
                    method=method,
                    terminals=terminals,
                    root=root,
                    rnj_tolerance_factor=rnj_tolerance_factor,
                    rg_tolerance=rg_tolerance,
                )
                noisy_clades, noisy_forced = _tree_clades(
                    r_fit,
                    x_fit,
                    mode=mode,
                    method=method,
                    terminals=terminals,
                    root=root,
                    rnj_tolerance_factor=rnj_tolerance_factor,
                    rg_tolerance=rg_tolerance,
                )
                exact_cherries = _boundary_cherries(exact_clades)
                full[(method, mode)] = _boundary_cherries(noisy_clades)
                exact_score = _family_score(exact_cherries, truth_cherries)
                noisy_score = _family_score(full[(method, mode)], truth_cherries)
                full_rows.append(
                    {
                        "case": case_name,
                        "method": method,
                        "mode": mode,
                        "true_boundary_cherry_count": len(truth_cherries),
                        "exact_boundary_cherries": _serialize_family(exact_cherries),
                        "exact_cherry_precision": exact_score["nontrivial_support_precision"],
                        "exact_cherry_recall": exact_score["nontrivial_support_recall"],
                        "exact_cherry_f1": exact_score["nontrivial_support_f1"],
                        "noisy_full_boundary_cherries": _serialize_family(full[(method, mode)]),
                        "noisy_full_cherry_precision": noisy_score["nontrivial_support_precision"],
                        "noisy_full_cherry_recall": noisy_score["nontrivial_support_recall"],
                        "noisy_full_cherry_f1": noisy_score["nontrivial_support_f1"],
                        "exact_forced_merges": exact_forced,
                        "noisy_full_forced_merges": noisy_forced,
                    }
                )

        rng = np.random.default_rng(seed + 20_000)
        for _ in range(bootstrap_replicates):
            sampled = _moving_block_bootstrap_copy(training, rng, block_length)
            sampled_r, sampled_x, _, _ = fit_projected_sensitivity(
                sampled,
                alpha=ridge_alpha,
                constraint_mode="ordered",
            )
            for method in METHODS:
                for mode in modes:
                    clades, _ = _tree_clades(
                        sampled_r,
                        sampled_x,
                        mode=mode,
                        method=method,
                        terminals=terminals,
                        root=root,
                        rnj_tolerance_factor=rnj_tolerance_factor,
                        rg_tolerance=rg_tolerance,
                    )
                    counters[(method, mode)].update(_boundary_cherries(clades))

        for method in METHODS:
            mode_candidates: dict[str, set[frozenset[int]]] = {}
            mode_confidences: dict[str, dict[frozenset[int], float]] = {}
            for mode in modes:
                confidence = {
                    clade: counters[(method, mode)][clade] / bootstrap_replicates
                    for clade in set(counters[(method, mode)]) | full[(method, mode)]
                }
                stable = {
                    clade
                    for clade in full[(method, mode)]
                    if confidence[clade] >= confidence_threshold
                }
                mode_candidates[mode] = stable
                mode_confidences[mode] = confidence
                selected = _select_disjoint(
                    stable,
                    confidence,
                    maximum_candidate_count,
                )
                for clade in sorted(
                    set(confidence), key=lambda item: (len(item), tuple(sorted(item)))
                ):
                    bootstrap_rows.append(
                        {
                            "case": case_name,
                            "method": method,
                            "mode": mode,
                            "support_labels": _support_text(clade),
                            "support_size": len(clade),
                            "present_in_full_data_tree": clade in full[(method, mode)],
                            "is_true_boundary_cherry": clade in truth_cherries,
                            "bootstrap_count": int(counters[(method, mode)][clade]),
                            "bootstrap_confidence": confidence[clade],
                            "passes_threshold": clade in stable,
                            "selected_top_k": clade in selected,
                        }
                    )
                for rank_index, clade in enumerate(selected, start=1):
                    selected_rows.append(
                        {
                            "case": case_name,
                            "method": method,
                            "mode": mode,
                            "selection_rank": rank_index,
                            "support_labels": _support_text(clade),
                            "support_size": len(clade),
                            "bootstrap_confidence": confidence[clade],
                            "is_true_boundary_cherry": clade in truth_cherries,
                        }
                    )

            if set(modes) >= {"R", "RX_75R_25X"}:
                joint_full = full[(method, "R")] & full[(method, "RX_75R_25X")]
                joint_confidence = {
                    clade: min(
                        mode_confidences["R"].get(clade, 0.0),
                        mode_confidences["RX_75R_25X"].get(clade, 0.0),
                    )
                    for clade in joint_full
                }
                joint_stable = {
                    clade
                    for clade in joint_full
                    if joint_confidence[clade] >= confidence_threshold
                }
                joint_selected = _select_disjoint(
                    joint_stable,
                    joint_confidence,
                    maximum_candidate_count,
                )
                for rank_index, clade in enumerate(joint_selected, start=1):
                    selected_rows.append(
                        {
                            "case": case_name,
                            "method": method,
                            "mode": "R_AND_RX75",
                            "selection_rank": rank_index,
                            "support_labels": _support_text(clade),
                            "support_size": len(clade),
                            "bootstrap_confidence": joint_confidence[clade],
                            "is_true_boundary_cherry": clade in truth_cherries,
                        }
                    )

    design_frame = pd.DataFrame(design_rows)
    full_frame = pd.DataFrame(full_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    selected_columns = [
        "case",
        "method",
        "mode",
        "selection_rank",
        "support_labels",
        "support_size",
        "bootstrap_confidence",
        "is_true_boundary_cherry",
    ]
    selected_frame = pd.DataFrame(selected_rows, columns=selected_columns)
    design_frame.to_csv(output / "design_diagnostics.csv", index=False)
    full_frame.to_csv(output / "full_boundary_cherry_metrics.csv", index=False)
    bootstrap_frame.to_csv(output / "bootstrap_boundary_cherries.csv", index=False)
    selected_frame.to_csv(output / "selected_boundary_cherries.csv", index=False)
    payload = {
        "config": {
            "cases": list(cases),
            "seed": seed,
            "noise_regime": "response_outliers",
            "sample_count": sample_count,
            "train_count": train_count,
            "modes": list(modes),
            "methods": list(METHODS),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_kind": "circular_moving_block",
            "block_length": block_length,
            "confidence_threshold": confidence_threshold,
            "maximum_candidate_count": maximum_candidate_count,
            "rnj_tolerance_factor": rnj_tolerance_factor,
            "rg_tolerance": rg_tolerance,
            "ridge_alpha": ridge_alpha,
            "response_laplace_fraction": response_laplace_fraction,
            "outlier_fraction": outlier_fraction,
            "gross_outlier_fraction": gross_outlier_fraction,
            "truth_used_for_candidate_generation": False,
            "candidate_definition": "inclusion-minimal nontrivial rooted clade",
        },
        "design": design_rows,
        "full_tree": full_rows,
        "selected_boundary_cherries": selected_rows,
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", nargs="+", choices=PAPER_STYLE_CASES, default=list(PAPER_STYLE_CASES))
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--sample-count", type=int, default=96)
    parser.add_argument("--train-count", type=int, default=96)
    parser.add_argument("--modes", nargs="+", default=["R", "RX_75R_25X"])
    parser.add_argument("--bootstrap-replicates", type=int, default=100)
    parser.add_argument("--block-length", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--maximum-candidate-count", type=int, default=2)
    args = parser.parse_args()
    run(
        args.output,
        cases=tuple(args.cases),
        seed=args.seed,
        sample_count=args.sample_count,
        train_count=args.train_count,
        modes=tuple(args.modes),
        bootstrap_replicates=args.bootstrap_replicates,
        block_length=args.block_length,
        confidence_threshold=args.confidence_threshold,
        maximum_candidate_count=args.maximum_candidate_count,
    )


if __name__ == "__main__":
    main()
