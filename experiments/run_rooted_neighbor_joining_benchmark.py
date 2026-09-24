"""Large RNJ benchmark using root-depth and shared-path sensitivity estimates."""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from experiments.run_latent_tree_diagnostics import (
    _detailed_scenarios,
    _recipes,
    _set_score,
    _true_limb_lengths,
)
from terminal_case33.estimation.multiscenario import fit_projected_sensitivity, preprocess_scenarios
from terminal_case33.graph.diagnostics import terminal_limb_lengths, terminal_partition_at_node, terminal_sibling_pairs
from terminal_case33.graph.latent_tree import terminal_splits
from terminal_case33.graph.rooted_neighbor_joining import rooted_neighbor_joining
from terminal_case33.graph.sensitivity_geometry import sensitivity_geometry
from terminal_case33.models.lin_distflow import build_reduced_sensitivity_matrices
from terminal_case33.utils.io import ensure_dir, write_json


def _rooted_clades(edges: tuple[tuple[int, int, float], ...], root: int, terminals: list[int]) -> set[frozenset[int]]:
    graph = nx.Graph()
    graph.add_edges_from((int(left), int(right)) for left, right, _ in edges)
    terminal_set = set(terminals)
    parent = {int(child): int(upstream) for child, upstream in nx.bfs_predecessors(graph, root)}
    children: dict[int, list[int]] = {int(node): [] for node in graph.nodes()}
    for child, upstream in parent.items():
        children[upstream].append(child)
    clades = set()

    def visit(node: int) -> set[int]:
        found = {node} if node in terminal_set else set()
        for child in children[node]:
            found.update(visit(child))
        if 1 < len(found) < len(terminals):
            clades.add(frozenset(found))
        return found

    visit(root)
    return clades


def run(
    output_dir: str | Path = "outputs/rooted_neighbor_joining_benchmark",
    cases: list[str] | None = None,
    scenario_counts: list[int] | None = None,
    t_counts: list[int] | None = None,
    tolerance_factors: list[float] | None = None,
    distance_mode: str = "RX_75R_25X",
    pq_noise_rel: float = 0.005,
) -> dict:
    """Run RNJ over the same noisy AC matrix used for NJ/RG diagnostics."""

    out_dir = ensure_dir(output_dir)
    selected_cases = cases or ["paper15", "soumalas11", "flynn16", "pengwah18"]
    selected_scenarios = scenario_counts or [1, 3, 5]
    selected_t = t_counts or [48, 96, 288]
    selected_factors = tolerance_factors or [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
    rows = []
    total = len(selected_cases) * len(selected_scenarios) * len(selected_t)
    task = 0
    for case_key in selected_cases:
        for scenario_count in selected_scenarios:
            for t_count in selected_t:
                task += 1
                net, scenarios = _detailed_scenarios(case_key, scenario_count, t_count, pq_noise_rel)
                terminals = net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"].astype(int).tolist()
                true_edges = net.closed_edges()
                true_splits = terminal_splits(true_edges, terminals)
                true_siblings = terminal_sibling_pairs(true_edges, terminals)
                true_root_partition = terminal_partition_at_node(true_edges, net.root_bus, terminals)
                true_r, true_x = build_reduced_sensitivity_matrices(net, terminals, voltage_model="squared-voltage")
                true_geometry = sensitivity_geometry(true_r, true_x, distance_mode)
                true_depth = true_geometry.root_depths
                true_coefficients = (
                    true_geometry.r_coefficient,
                    true_geometry.x_coefficient,
                )
                true_shared = true_geometry.shared_paths
                true_limbs = _true_limb_lengths(net, terminals, true_coefficients)
                oracle = rooted_neighbor_joining(true_shared, true_depth, terminals, net.root_bus, 1e-10)
                oracle_split_f1 = _set_score(terminal_splits(oracle.edges, terminals), true_splits)[2]
                oracle_sibling_f1 = _set_score(terminal_sibling_pairs(oracle.edges, terminals), true_siblings)[2]
                oracle_root_correct = terminal_partition_at_node(oracle.edges, net.root_bus, terminals) == true_root_partition
                true_clades = _rooted_clades(oracle.edges, net.root_bus, terminals)

                for recipe in _recipes():
                    fitted = preprocess_scenarios(scenarios, recipe)
                    r_hat, x_hat, r2_score, condition_number = fit_projected_sensitivity(fitted)
                    geometry = sensitivity_geometry(r_hat, x_hat, distance_mode)
                    depth = geometry.root_depths
                    shared = geometry.shared_paths
                    depth_scale = max(float(np.median(depth)), 1e-12)
                    for factor in selected_factors:
                        tolerance = factor * depth_scale
                        result = rooted_neighbor_joining(shared, depth, terminals, net.root_bus, tolerance)
                        split_precision, split_recall, split_f1 = _set_score(
                            terminal_splits(result.edges, terminals), true_splits
                        )
                        sibling_precision, sibling_recall, sibling_f1 = _set_score(
                            terminal_sibling_pairs(result.edges, terminals), true_siblings
                        )
                        clade_precision, clade_recall, clade_f1 = _set_score(
                            _rooted_clades(result.edges, net.root_bus, terminals), true_clades
                        )
                        predicted_limbs = terminal_limb_lengths(result.edges, terminals)
                        limb_errors = [
                            abs(predicted_limbs[terminal] - true_limbs[terminal]) / max(true_limbs[terminal], 1e-12)
                            for terminal in terminals
                            if terminal in predicted_limbs
                        ]
                        rows.append(
                            {
                                "case": case_key,
                                "scenario_count": scenario_count,
                                "t_count": t_count,
                                "recipe": recipe["name"],
                                "tolerance_factor": factor,
                                "absolute_tolerance": tolerance,
                                "oracle_split_f1": oracle_split_f1,
                                "oracle_sibling_f1": oracle_sibling_f1,
                                "oracle_root_correct": oracle_root_correct,
                                "split_precision": split_precision,
                                "split_recall": split_recall,
                                "split_f1": split_f1,
                                "sibling_precision": sibling_precision,
                                "sibling_recall": sibling_recall,
                                "sibling_f1": sibling_f1,
                                "rooted_clade_precision": clade_precision,
                                "rooted_clade_recall": clade_recall,
                                "rooted_clade_f1": clade_f1,
                                "root_partition_correct": (
                                    terminal_partition_at_node(result.edges, net.root_bus, terminals) == true_root_partition
                                ),
                                "terminal_limb_relative_mae": float(np.mean(limb_errors)),
                                "r2_score": r2_score,
                                "condition_number": condition_number,
                            }
                        )
                print(f"[rnj] {task}/{total} case={case_key} scenarios={scenario_count} T={t_count}", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "rnj_all_results.csv", index=False)
    by_factor = (
        result.groupby("tolerance_factor", as_index=False)
        .agg(
            runs=("split_f1", "size"),
            mean_split_f1=("split_f1", "mean"),
            mean_sibling_f1=("sibling_f1", "mean"),
            mean_rooted_clade_f1=("rooted_clade_f1", "mean"),
            root_success_rate=("root_partition_correct", "mean"),
            mean_limb_relative_mae=("terminal_limb_relative_mae", "mean"),
        )
        .sort_values("mean_rooted_clade_f1", ascending=False)
    )
    by_factor.to_csv(out_dir / "summary_by_tolerance.csv", index=False)
    best_factor = float(by_factor.iloc[0]["tolerance_factor"])
    fixed = result[result["tolerance_factor"].eq(best_factor)].copy()
    fixed.to_csv(out_dir / "best_fixed_tolerance_results.csv", index=False)
    by_case = (
        fixed.groupby("case", as_index=False)
        .agg(
            runs=("split_f1", "size"),
            mean_split_f1=("split_f1", "mean"),
            mean_sibling_f1=("sibling_f1", "mean"),
            mean_rooted_clade_f1=("rooted_clade_f1", "mean"),
            root_success_rate=("root_partition_correct", "mean"),
        )
    )
    by_case.to_csv(out_dir / "summary_by_case.csv", index=False)
    metrics = {
        "run_count": int(len(result)),
        "best_fixed_tolerance_factor_retrospective": best_factor,
        "oracle_split_f1": float(result["oracle_split_f1"].mean()),
        "oracle_sibling_f1": float(result["oracle_sibling_f1"].mean()),
        "oracle_root_success": float(result["oracle_root_correct"].mean()),
        "mean_split_f1_at_best_fixed": float(fixed["split_f1"].mean()),
        "mean_sibling_f1_at_best_fixed": float(fixed["sibling_f1"].mean()),
        "mean_rooted_clade_f1_at_best_fixed": float(fixed["rooted_clade_f1"].mean()),
        "root_success_at_best_fixed": float(fixed["root_partition_correct"].mean()),
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# Rooted Neighbor-Joining benchmark\n\n"
        "The tolerance factor multiplies the median estimated root depth. The selected best fixed factor is "
        "retrospective and must not be treated as label-free tuning.\n\n"
        "## Tolerance sweep\n\n"
        + by_factor.to_string(index=False)
        + "\n\n## Best fixed factor by case\n\n"
        + by_case.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/rooted_neighbor_joining_benchmark")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--scenario-counts", nargs="*", type=int, default=None)
    parser.add_argument("--T-counts", nargs="*", type=int, default=None)
    parser.add_argument("--tolerance-factors", nargs="*", type=float, default=None)
    parser.add_argument("--distance-mode", default="RX_75R_25X")
    parser.add_argument("--pq-noise-rel", type=float, default=0.005)
    args = parser.parse_args()
    run(
        output_dir=args.output,
        cases=args.cases,
        scenario_counts=args.scenario_counts,
        t_counts=args.T_counts,
        tolerance_factors=args.tolerance_factors,
        distance_mode=args.distance_mode,
        pq_noise_rel=args.pq_noise_rel,
    )


if __name__ == "__main__":
    main()
