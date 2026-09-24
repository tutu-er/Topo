"""Compare RNJ, NJ, and RG candidate stability on the same noisy training block.

This audit intentionally separates two questions:

1. Does the full-data reconstructed tree match the true rooted terminal clades?
2. Which small clades survive bootstrap screening without consulting truth?

Truth is used only after candidate generation for evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_laminar_l1_case33_integration import (
    _continuous_pq,
    _family_score,
    _terminal_labels,
)
from experiments.run_laminar_l1_noisy_multicase import (
    PAPER_STYLE_CASES,
    REGIMES,
    _scenario,
    _truth_nontrivial_clades,
    build_benchmark_cases,
    contaminate_measurements,
)
from terminal_case33.estimation.multiscenario import fit_projected_sensitivity
from terminal_case33.estimation.rnj_warm_start import _bootstrap_copy
from terminal_case33.graph.latent_tree import neighbor_joining, recursive_grouping
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.graph.rooted_neighbor_joining import rooted_neighbor_joining
from terminal_case33.graph.sensitivity_geometry import sensitivity_geometry
from terminal_case33.models.lin_distflow import build_reduced_sensitivity_matrices
from terminal_case33.pipeline.peripheral_edge_proposals import _root_nj_edges
from terminal_case33.utils.io import ensure_dir


DEFAULT_OUTPUT = Path("outputs/tree_candidate_method_audit")
METHODS = ("RNJ", "NJ", "RG")


def _tree_clades(
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    *,
    mode: str,
    method: str,
    terminals: list[int],
    root: int,
    rnj_tolerance_factor: float,
    rg_tolerance: float,
) -> tuple[set[frozenset[int]], int]:
    geometry = sensitivity_geometry(r_matrix, x_matrix, mode)
    if method == "RNJ":
        scale = max(float(np.median(geometry.root_depths)), 1e-12)
        tree = rooted_neighbor_joining(
            geometry.shared_paths,
            geometry.root_depths,
            terminals,
            root,
            group_tolerance=rnj_tolerance_factor * scale,
        )
        return rooted_clades(tree.edges, root, terminals), 0
    if method == "NJ":
        tree = neighbor_joining(geometry.distance, terminals)
        edges, placed_root = _root_nj_edges(
            tree.edges,
            terminals,
            geometry.root_depths,
            root,
        )
        return rooted_clades(edges, placed_root, terminals), 0
    if method == "RG":
        count = len(terminals)
        augmented = np.zeros((count + 1, count + 1), dtype=float)
        augmented[:-1, :-1] = geometry.distance
        augmented[:-1, -1] = geometry.root_depths
        augmented[-1, :-1] = geometry.root_depths
        tree = recursive_grouping(
            augmented,
            [*terminals, root],
            tolerance=rg_tolerance,
        )
        return rooted_clades(tree.edges, root, terminals), int(tree.forced_merges)
    raise ValueError(f"unknown method {method!r}")


def _serialize_clade(clade: frozenset[int]) -> str:
    return ",".join(map(str, sorted(clade)))


def _serialize_family(family: set[frozenset[int]]) -> str:
    return ";".join(
        _serialize_clade(clade)
        for clade in sorted(family, key=lambda item: (len(item), tuple(sorted(item))))
    )


def _matrix_relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.linalg.norm(estimate - truth, ord="fro")
        / max(np.linalg.norm(truth, ord="fro"), 1e-15)
    )


def _training_block(
    case_name: str,
    net,
    *,
    seed: int,
    regime: str,
    sample_count: int,
    train_count: int,
    response_laplace_fraction: float,
    outlier_fraction: float,
    gross_outlier_fraction: float,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    terminals = _terminal_labels(net)
    r_true, x_true = build_reduced_sensitivity_matrices(
        net,
        terminals,
        voltage_model="squared-voltage",
    )
    p_true, q_true = _continuous_pq(net, sample_count, seed)
    target_true = pd.DataFrame(
        p_true.to_numpy(dtype=float) @ r_true.T
        + q_true.to_numpy(dtype=float) @ x_true.T,
        index=p_true.index,
        columns=terminals,
    )
    p_observed, q_observed, target_observed, _, _ = contaminate_measurements(
        p_true,
        q_true,
        target_true,
        seed=seed + 10_000,
        predictor_noise_fraction=REGIMES[regime],
        response_laplace_fraction=response_laplace_fraction,
        outlier_fraction=outlier_fraction,
        gross_outlier_fraction=gross_outlier_fraction,
    )
    return (
        _scenario(
            p_observed,
            q_observed,
            target_observed,
            slice(0, train_count),
            f"{case_name}_{regime}",
        ),
        r_true,
        x_true,
    )


def run(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    cases: tuple[str, ...] = ("paper15", "flynn16"),
    seed: int = 20260903,
    regime: str = "response_outliers",
    sample_count: int = 48,
    train_count: int = 32,
    modes: tuple[str, ...] = ("R", "RX_75R_25X"),
    bootstrap_replicates: int = 100,
    comparison_prefix: int = 12,
    confidence_threshold: float = 0.75,
    maximum_support_count: int = 2,
    rnj_tolerance_factor: float = 0.16,
    rg_tolerance: float = 0.03,
    ridge_alpha: float = 1e-6,
    response_laplace_fraction: float = 0.05,
    outlier_fraction: float = 0.05,
    gross_outlier_fraction: float = 1.5,
) -> dict:
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    if not 1 <= comparison_prefix <= bootstrap_replicates:
        raise ValueError("comparison_prefix must lie in [1, bootstrap_replicates]")
    output = ensure_dir(output_dir)
    full_rows: list[dict] = []
    candidate_rows: list[dict] = []
    design_rows: list[dict] = []

    for case_name, net in build_benchmark_cases(cases).items():
        terminals = _terminal_labels(net)
        root = int(net.root_bus)
        truth = _truth_nontrivial_clades(net, terminals)
        maximum_support_size = max(2, min(6, ceil(len(terminals) / 3)))
        training, r_true, x_true = _training_block(
            case_name,
            net,
            seed=seed,
            regime=regime,
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
        design_rank = int(np.linalg.matrix_rank(design))
        design_rows.append(
            {
                "case": case_name,
                "terminal_count": len(terminals),
                "training_rows": train_count,
                "dense_regressor_count": int(design.shape[1]),
                "design_rank": design_rank,
                "rank_deficiency": int(design.shape[1] - design_rank),
                "residual_degrees_of_freedom": int(train_count - design_rank),
                "condition_number": float(condition),
                "dense_fit_r2": float(fit_r2),
                "dense_R_relative_frobenius_error": _matrix_relative_error(r_fit, r_true),
                "dense_X_relative_frobenius_error": _matrix_relative_error(x_fit, x_true),
                "true_clades": _serialize_family(truth),
            }
        )

        full: dict[tuple[str, str], set[frozenset[int]]] = {}
        exact: dict[tuple[str, str], set[frozenset[int]]] = {}
        full_forced: dict[tuple[str, str], int] = {}
        counters = {
            (method, mode): Counter()
            for method in METHODS
            for mode in modes
        }
        prefix_counters = {
            (method, mode): Counter()
            for method in METHODS
            for mode in modes
        }

        for method in METHODS:
            for mode in modes:
                exact[(method, mode)], exact_forced = _tree_clades(
                    r_true,
                    x_true,
                    mode=mode,
                    method=method,
                    terminals=terminals,
                    root=root,
                    rnj_tolerance_factor=rnj_tolerance_factor,
                    rg_tolerance=rg_tolerance,
                )
                full[(method, mode)], full_forced[(method, mode)] = _tree_clades(
                    r_fit,
                    x_fit,
                    mode=mode,
                    method=method,
                    terminals=terminals,
                    root=root,
                    rnj_tolerance_factor=rnj_tolerance_factor,
                    rg_tolerance=rg_tolerance,
                )
                full_score = _family_score(full[(method, mode)], truth)
                exact_score = _family_score(exact[(method, mode)], truth)
                full_rows.append(
                    {
                        "case": case_name,
                        "method": method,
                        "mode": mode,
                        "exact_precision": exact_score["nontrivial_support_precision"],
                        "exact_recall": exact_score["nontrivial_support_recall"],
                        "exact_f1": exact_score["nontrivial_support_f1"],
                        "full_noisy_precision": full_score["nontrivial_support_precision"],
                        "full_noisy_recall": full_score["nontrivial_support_recall"],
                        "full_noisy_f1": full_score["nontrivial_support_f1"],
                        "full_noisy_matched_clades": _serialize_family(full[(method, mode)] & truth),
                        "full_noisy_missing_clades": _serialize_family(truth - full[(method, mode)]),
                        "full_noisy_extra_clades": _serialize_family(full[(method, mode)] - truth),
                        "exact_forced_merges": exact_forced,
                        "full_noisy_forced_merges": full_forced[(method, mode)],
                    }
                )

        rng = np.random.default_rng(seed + 20_000)
        for replicate_index in range(bootstrap_replicates):
            sampled = _bootstrap_copy(training, rng)
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
                    counters[(method, mode)].update(clades)
                    if replicate_index < comparison_prefix:
                        prefix_counters[(method, mode)].update(clades)

        for method in METHODS:
            for mode in modes:
                all_seen = set(counters[(method, mode)]) | full[(method, mode)]
                stable_prefix = {
                    clade
                    for clade in full[(method, mode)]
                    if 2 <= len(clade) <= maximum_support_size
                    and prefix_counters[(method, mode)][clade] / comparison_prefix
                    >= confidence_threshold
                }
                stable_all = {
                    clade
                    for clade in full[(method, mode)]
                    if 2 <= len(clade) <= maximum_support_size
                    and counters[(method, mode)][clade] / bootstrap_replicates
                    >= confidence_threshold
                }
                ranked_all = sorted(
                    stable_all,
                    key=lambda clade: (
                        -counters[(method, mode)][clade] / bootstrap_replicates,
                        len(clade),
                        tuple(sorted(clade)),
                    ),
                )
                selected_all = set(ranked_all[:maximum_support_count])
                for clade in sorted(all_seen, key=lambda item: (len(item), tuple(sorted(item)))):
                    candidate_rows.append(
                        {
                            "case": case_name,
                            "method": method,
                            "mode": mode,
                            "support_labels": _serialize_clade(clade),
                            "support_size": len(clade),
                            "present_in_full_data_tree": clade in full[(method, mode)],
                            "is_true_clade": clade in truth,
                            f"bootstrap_confidence_{comparison_prefix}": (
                                prefix_counters[(method, mode)][clade] / comparison_prefix
                            ),
                            f"stable_at_{comparison_prefix}": clade in stable_prefix,
                            f"bootstrap_confidence_{bootstrap_replicates}": (
                                counters[(method, mode)][clade] / bootstrap_replicates
                            ),
                            f"stable_at_{bootstrap_replicates}": clade in stable_all,
                            f"selected_top_{maximum_support_count}_at_{bootstrap_replicates}": (
                                clade in selected_all
                            ),
                        }
                    )

    full_frame = pd.DataFrame(full_rows)
    candidate_frame = pd.DataFrame(candidate_rows)
    design_frame = pd.DataFrame(design_rows)
    full_frame.to_csv(output / "full_tree_metrics.csv", index=False)
    candidate_frame.to_csv(output / "bootstrap_candidates.csv", index=False)
    design_frame.to_csv(output / "design_diagnostics.csv", index=False)
    payload = {
        "config": {
            "cases": list(cases),
            "seed": seed,
            "regime": regime,
            "sample_count": sample_count,
            "train_count": train_count,
            "modes": list(modes),
            "methods": list(METHODS),
            "bootstrap_replicates": bootstrap_replicates,
            "comparison_prefix": comparison_prefix,
            "confidence_threshold": confidence_threshold,
            "maximum_support_count": maximum_support_count,
            "rnj_tolerance_factor": rnj_tolerance_factor,
            "rg_tolerance": rg_tolerance,
            "ridge_alpha": ridge_alpha,
            "truth_used_for_candidate_generation": False,
        },
        "design": design_rows,
        "full_tree": full_rows,
        "stable_candidates": candidate_frame.loc[
            candidate_frame[f"stable_at_{bootstrap_replicates}"]
        ].to_dict(orient="records"),
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", nargs="+", choices=PAPER_STYLE_CASES, default=["paper15", "flynn16"])
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--regime", choices=tuple(REGIMES), default="response_outliers")
    parser.add_argument("--sample-count", type=int, default=48)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--modes", nargs="+", default=["R", "RX_75R_25X"])
    parser.add_argument("--bootstrap-replicates", type=int, default=100)
    parser.add_argument("--comparison-prefix", type=int, default=12)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--maximum-support-count", type=int, default=2)
    parser.add_argument("--rnj-tolerance-factor", type=float, default=0.16)
    parser.add_argument("--rg-tolerance", type=float, default=0.03)
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    args = parser.parse_args()
    run(
        args.output,
        cases=tuple(args.cases),
        seed=args.seed,
        regime=args.regime,
        sample_count=args.sample_count,
        train_count=args.train_count,
        modes=tuple(args.modes),
        bootstrap_replicates=args.bootstrap_replicates,
        comparison_prefix=args.comparison_prefix,
        confidence_threshold=args.confidence_threshold,
        maximum_support_count=args.maximum_support_count,
        rnj_tolerance_factor=args.rnj_tolerance_factor,
        rg_tolerance=args.rg_tolerance,
        ridge_alpha=args.ridge_alpha,
    )


if __name__ == "__main__":
    main()
