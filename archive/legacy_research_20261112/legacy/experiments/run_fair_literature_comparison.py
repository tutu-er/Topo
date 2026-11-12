"""Fair noisy-AC comparison of literature-inspired hidden-tree methods."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_latent_tree_diagnostics import _distance_and_depth, _set_score
from experiments.run_matrix_constraint_large_sweep import _simulate_pool, _terminal_buses
from experiments.run_rooted_hierarchical_aggregation import RECIPE
from terminal_case33.data.paper_style_case_bank import CASE_BUILDERS
from terminal_case33.estimation.multiscenario import fit_projected_sensitivity, preprocess_scenarios
from terminal_case33.graph.diagnostics import (
    RootPlacement,
    locate_root_from_terminal_depths,
    root_partition_is_correct,
    terminal_sibling_pairs,
)
from terminal_case33.graph.latent_tree import neighbor_joining, recursive_grouping, terminal_splits
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.graph.rooted_neighbor_joining import rooted_neighbor_joining, shared_paths_from_distances
from terminal_case33.utils.io import ensure_dir, write_json


@dataclass(frozen=True)
class MethodSpec:
    """One estimator-distance-reconstruction combination."""

    name: str
    family: str
    constraint_mode: str
    distance_mode: str
    reconstruction: str
    paper_fidelity: str


METHODS = [
    MethodSpec(
        "soumalas2017_nj_proxy",
        "Soumalas 2017",
        "basic",
        "R",
        "nj",
        "terminal additive-distance NJ proxy; not the integer Pruefer implementation",
    ),
    MethodSpec(
        "park_deka_recursive_grouping",
        "Park/Deka minimal observability",
        "basic",
        "R",
        "rg",
        "recursive grouping on an estimated terminal additive distance",
    ),
    MethodSpec(
        "pengwah2024_ordered_rg",
        "Pengwah 2024",
        "ordered",
        "R",
        "rg",
        "ordered sensitivity constraints plus recursive grouping",
    ),
    MethodSpec(
        "flynn2023_root_corrected_proxy",
        "Flynn 2023",
        "basic",
        "RX_75R_25X",
        "rnj",
        "measured transformer/root mode with rooted graph-learning proxy",
    ),
    MethodSpec(
        "proposed_ordered_rnj",
        "Current method",
        "ordered",
        "RX_75R_25X",
        "rnj",
        "ordered multi-scenario sensitivity plus rooted neighbor joining",
    ),
    MethodSpec(
        "proposed_tree_covariance_rnj",
        "Current method PSD",
        "tree_covariance",
        "RX_75R_25X",
        "rnj",
        "ordered PSD sensitivity plus rooted neighbor joining",
    ),
    MethodSpec(
        "controlled_ordered_nj",
        "Controlled reconstruction",
        "ordered",
        "RX_75R_25X",
        "nj",
        "same ordered matrix and distance as current method; NJ reconstruction only",
    ),
    MethodSpec(
        "controlled_ordered_rg",
        "Controlled reconstruction",
        "ordered",
        "RX_75R_25X",
        "rg",
        "same ordered matrix and distance as current method; RG reconstruction only",
    ),
]


def _reconstruct_and_score(
    spec: MethodSpec,
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    net,
    terminals: list[int],
    true_splits: set,
    true_siblings: set,
    true_clades: set,
    rnj_tolerance_factor: float,
    rg_tolerance: float,
) -> dict:
    """Reconstruct and score one method using common topology metrics."""

    distance, depth, _ = _distance_and_depth(r_matrix, x_matrix, spec.distance_mode)
    if spec.reconstruction == "nj":
        tree = neighbor_joining(distance, terminals)
        placement = locate_root_from_terminal_depths(tree.edges, terminals, depth)
        predicted_edges = tree.edges
        rooted_f1 = np.nan
        forced_merges = 0
    elif spec.reconstruction == "rg":
        augmented_distance = np.zeros((len(terminals) + 1, len(terminals) + 1), dtype=float)
        augmented_distance[:-1, :-1] = distance
        augmented_distance[:-1, -1] = depth
        augmented_distance[-1, :-1] = depth
        tree = recursive_grouping(
            augmented_distance,
            [*terminals, net.root_bus],
            tolerance=rg_tolerance,
        )
        placement = RootPlacement("node", net.root_bus, None, 0.0, 0.0, 0.0)
        predicted_edges = tree.edges
        rooted_f1 = np.nan
        forced_merges = tree.forced_merges
    elif spec.reconstruction == "rnj":
        scale = max(float(np.median(depth)), 1e-12)
        tree = rooted_neighbor_joining(
            shared_paths_from_distances(distance, depth),
            depth,
            terminals,
            net.root_bus,
            rnj_tolerance_factor * scale,
        )
        placement = RootPlacement("node", net.root_bus, None, 0.0, 0.0, 0.0)
        predicted_edges = tree.edges
        predicted_clades = rooted_clades(predicted_edges, net.root_bus, terminals)
        rooted_f1 = _set_score(predicted_clades, true_clades)[2]
        forced_merges = 0
    else:
        raise ValueError(f"unknown reconstruction {spec.reconstruction!r}")

    predicted_splits = terminal_splits(predicted_edges, terminals)
    predicted_siblings = terminal_sibling_pairs(predicted_edges, terminals)
    split_precision, split_recall, split_f1 = _set_score(predicted_splits, true_splits)
    sibling_precision, sibling_recall, sibling_f1 = _set_score(predicted_siblings, true_siblings)
    return {
        "split_precision": split_precision,
        "split_recall": split_recall,
        "split_f1": split_f1,
        "exact_unrooted_recovery": predicted_splits == true_splits,
        "sibling_precision": sibling_precision,
        "sibling_recall": sibling_recall,
        "sibling_f1": sibling_f1,
        "rooted_clade_f1": rooted_f1,
        "root_partition_correct": root_partition_is_correct(
            placement,
            predicted_edges,
            net.closed_edges(),
            net.root_bus,
            terminals,
        ),
        "forced_merges": forced_merges,
    }


def _summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize normal and stress regimes without outcome-based filtering."""

    rows = []
    regimes = {
        "normal_scenarios_ge_3": results["scenario_count"] >= 3,
        "stress_single_scenario": results["scenario_count"] == 1,
        "all": np.ones(len(results), dtype=bool),
    }
    for regime, mask in regimes.items():
        selected = results.loc[mask]
        for method, group in selected.groupby("method"):
            rows.append(
                {
                    "regime": regime,
                    "method": method,
                    "family": group["family"].iloc[0],
                    "paper_fidelity": group["paper_fidelity"].iloc[0],
                    "runs": len(group),
                    "mean_split_f1": group["split_f1"].mean(),
                    "median_split_f1": group["split_f1"].median(),
                    "minimum_split_f1": group["split_f1"].min(),
                    "exact_unrooted_recovery_rate": group["exact_unrooted_recovery"].mean(),
                    "mean_sibling_f1": group["sibling_f1"].mean(),
                    "root_partition_success_rate": group["root_partition_correct"].mean(),
                    "mean_rooted_clade_f1": group["rooted_clade_f1"].mean(),
                }
            )
    return pd.DataFrame(rows)


def run(
    output_dir: str | Path = "outputs/fair_literature_comparison",
    cases: list[str] | None = None,
    scenario_counts: list[int] | None = None,
    t_counts: list[int] | None = None,
    replicates: int = 2,
    pq_noise_rel: float = 0.005,
    v_noise_rel: float = 0.0002,
    rnj_tolerance_factor: float = 0.16,
    rg_tolerance: float = 0.03,
) -> dict:
    """Compare methods on identical noisy AC measurements."""

    out_dir = ensure_dir(output_dir)
    selected_cases = cases or list(CASE_BUILDERS)
    selected_scenario_counts = scenario_counts or [1, 3, 5, 10]
    selected_t_counts = t_counts or [48, 96, 288]
    maximum_scenarios = max(selected_scenario_counts)
    total_conditions = len(selected_cases) * len(selected_t_counts) * replicates * len(selected_scenario_counts)
    completed = 0
    rows = []
    for case_key in selected_cases:
        for t_count in selected_t_counts:
            for replicate in range(replicates):
                net, scenario_pool = _simulate_pool(
                    case_key,
                    t_count,
                    replicate,
                    maximum_scenarios,
                    pq_noise_rel,
                    v_noise_rel,
                )
                terminals = _terminal_buses(net)
                true_edges = net.closed_edges()
                true_splits = terminal_splits(true_edges, terminals)
                true_siblings = terminal_sibling_pairs(true_edges, terminals)
                true_clades = rooted_clades(true_edges, net.root_bus, terminals)
                for scenario_count in selected_scenario_counts:
                    fitted = preprocess_scenarios(scenario_pool[:scenario_count], RECIPE)
                    matrices = {}
                    fit_metrics = {}
                    for constraint_mode in sorted({method.constraint_mode for method in METHODS}):
                        r_matrix, x_matrix, r2_score, condition_number = fit_projected_sensitivity(
                            fitted,
                            constraint_mode=constraint_mode,
                        )
                        matrices[constraint_mode] = (r_matrix, x_matrix)
                        fit_metrics[constraint_mode] = (r2_score, condition_number)
                    for spec in METHODS:
                        r_matrix, x_matrix = matrices[spec.constraint_mode]
                        score = _reconstruct_and_score(
                            spec,
                            r_matrix,
                            x_matrix,
                            net,
                            terminals,
                            true_splits,
                            true_siblings,
                            true_clades,
                            rnj_tolerance_factor,
                            rg_tolerance,
                        )
                        r2_score, condition_number = fit_metrics[spec.constraint_mode]
                        rows.append(
                            {
                                "case": case_key,
                                "replicate": replicate,
                                "scenario_count": scenario_count,
                                "t_count": t_count,
                                "samples": scenario_count * t_count,
                                "method": spec.name,
                                "family": spec.family,
                                "constraint_mode": spec.constraint_mode,
                                "distance_mode": spec.distance_mode,
                                "reconstruction": spec.reconstruction,
                                "paper_fidelity": spec.paper_fidelity,
                                "r2_score": r2_score,
                                "condition_number": condition_number,
                                **score,
                            }
                        )
                    completed += 1
                pd.DataFrame(rows).to_csv(out_dir / "comparison_results.csv", index=False)
                print(
                    f"[literature-fair] {completed}/{total_conditions} "
                    f"case={case_key} T={t_count} replicate={replicate}",
                    flush=True,
                )

    results = pd.DataFrame(rows)
    summary = _summary(results)
    summary.to_csv(out_dir / "comparison_summary.csv", index=False)
    normal = summary.loc[summary["regime"].eq("normal_scenarios_ge_3")].sort_values(
        ["mean_split_f1", "exact_unrooted_recovery_rate"],
        ascending=False,
    )
    by_case = (
        results.loc[results["scenario_count"] >= 3]
        .groupby(["case", "method"], as_index=False)
        .agg(
            runs=("split_f1", "size"),
            mean_split_f1=("split_f1", "mean"),
            exact_recovery_rate=("exact_unrooted_recovery", "mean"),
            mean_sibling_f1=("sibling_f1", "mean"),
            root_partition_success_rate=("root_partition_correct", "mean"),
        )
    )
    by_case.to_csv(out_dir / "normal_by_case.csv", index=False)
    metrics = {
        "method": "fair_literature_inspired_noisy_ac_comparison",
        "normal_regime": "scenario_count >= 3, defined before scoring",
        "stress_regime": "scenario_count == 1, reported separately",
        "cases": selected_cases,
        "scenario_counts": selected_scenario_counts,
        "t_counts": selected_t_counts,
        "replicates": replicates,
        "data_condition_count": total_conditions,
        "method_run_count": int(len(results)),
        "pq_noise_relative_std": pq_noise_rel,
        "voltage_noise_relative_std": v_noise_rel,
        "shared_preprocess": RECIPE["name"],
        "primary_metric": "unrooted terminal split F1",
        "normal_summary": normal.to_dict(orient="records"),
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# Fair literature-inspired comparison\n\n"
        "All methods use identical noisy AC-generated P/Q/V data, terminal sets, root measurements, "
        "daily demeaning, and topology scores. The normal regime is defined from data availability "
        "before scoring: at least three coupled scenarios. Single-scenario cases are retained as a "
        "separate stress test. Proxy methods are explicitly labeled and are not claimed as exact "
        "reproductions of unavailable Pruefer or proprietary graph-learning details.\n\n"
        "## Normal regime\n\n"
        + normal.to_string(index=False)
        + "\n\n## All regimes\n\n"
        + summary.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/fair_literature_comparison")
    parser.add_argument("--cases", nargs="*", choices=list(CASE_BUILDERS), default=None)
    parser.add_argument("--scenario-counts", nargs="*", type=int, default=None)
    parser.add_argument("--T-counts", nargs="*", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--pq-noise-rel", type=float, default=0.005)
    parser.add_argument("--v-noise-rel", type=float, default=0.0002)
    parser.add_argument("--rnj-tolerance-factor", type=float, default=0.16)
    parser.add_argument("--rg-tolerance", type=float, default=0.03)
    args = parser.parse_args()
    run(
        output_dir=args.output,
        cases=args.cases,
        scenario_counts=args.scenario_counts,
        t_counts=args.T_counts,
        replicates=args.replicates,
        pq_noise_rel=args.pq_noise_rel,
        v_noise_rel=args.v_noise_rel,
        rnj_tolerance_factor=args.rnj_tolerance_factor,
        rg_tolerance=args.rg_tolerance,
    )


if __name__ == "__main__":
    main()
