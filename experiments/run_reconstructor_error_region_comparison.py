"""Compare where NJ, RG, and RNJ make hidden-tree reconstruction errors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_current_method_line_error_analysis import _physical_clade_lines
from experiments.run_latent_tree_diagnostics import _distance_and_depth, _set_score
from experiments.run_matrix_constraint_large_sweep import _simulate_pool, _terminal_buses
from experiments.run_rooted_hierarchical_aggregation import RECIPE
from terminal_case33.data.paper_style_case_bank import CASE_BUILDERS
from terminal_case33.estimation.multiscenario import fit_projected_sensitivity, preprocess_scenarios
from terminal_case33.graph.diagnostics import terminal_sibling_pairs
from terminal_case33.graph.latent_tree import neighbor_joining, recursive_grouping, terminal_splits
from terminal_case33.graph.rooted_neighbor_joining import rooted_neighbor_joining, shared_paths_from_distances
from terminal_case33.utils.io import ensure_dir, write_json


METHODS = ("NJ", "RG", "RNJ")


def _canonical_split(side: frozenset[int], terminals: list[int]) -> frozenset[int]:
    other = frozenset(set(terminals) - set(side))
    if len(side) < len(other):
        return side
    if len(other) < len(side):
        return other
    return side if tuple(sorted(side)) <= tuple(sorted(other)) else other


def _physical_split_records(net, terminals: list[int]) -> dict[frozenset[int], dict]:
    """Map identifiable unrooted terminal splits to physical edge regions."""

    by_clade = _physical_clade_lines(net, terminals)
    grouped: dict[frozenset[int], list[dict]] = {}
    for clade, record in by_clade.items():
        split = _canonical_split(clade, terminals)
        grouped.setdefault(split, []).append(record)
    result = {}
    for split, records in grouped.items():
        regions = {record["region"] for record in records}
        if "root_adjacent_backbone" in regions:
            region = "central_root_backbone"
        elif "intermediate_hidden_backbone" in regions:
            region = "central_intermediate_backbone"
        else:
            region = "peripheral_cluster_feeder"
        result[split] = {
            "region": region,
            "physical_edges": ";".join(
                f"{record['parent_bus']}->{record['child_bus']}" for record in records
            ),
            "split_size": len(split),
            "split_members": " ".join(str(node) for node in sorted(split)),
        }
    return result


def _reconstruct(
    method: str,
    distance: np.ndarray,
    depth: np.ndarray,
    terminals: list[int],
    root: int,
    rnj_tolerance_factor: float,
    rg_tolerance: float,
) -> tuple[tuple, int]:
    """Reconstruct with common ordered matrix and RX75 distance inputs."""

    if method == "NJ":
        tree = neighbor_joining(distance, terminals)
        return tree.edges, 0
    if method == "RG":
        augmented = np.zeros((len(terminals) + 1, len(terminals) + 1), dtype=float)
        augmented[:-1, :-1] = distance
        augmented[:-1, -1] = depth
        augmented[-1, :-1] = depth
        tree = recursive_grouping(augmented, [*terminals, root], tolerance=rg_tolerance)
        return tree.edges, tree.forced_merges
    if method == "RNJ":
        scale = max(float(np.median(depth)), 1e-12)
        tree = rooted_neighbor_joining(
            shared_paths_from_distances(distance, depth),
            depth,
            terminals,
            root,
            rnj_tolerance_factor * scale,
        )
        return tree.edges, 0
    raise ValueError(f"unknown method {method!r}")


def run(
    output_dir: str | Path = "outputs/reconstructor_error_region_comparison",
    cases: list[str] | None = None,
    scenario_counts: list[int] | None = None,
    t_counts: list[int] | None = None,
    replicates: int = 2,
    pq_noise_rel: float = 0.005,
    v_noise_rel: float = 0.0002,
    rnj_tolerance_factor: float = 0.16,
    rg_tolerance: float = 0.03,
) -> dict:
    """Run a controlled reconstruction-only error-region comparison."""

    out_dir = ensure_dir(output_dir)
    selected_cases = cases or list(CASE_BUILDERS)
    selected_scenario_counts = scenario_counts or [3, 5, 10]
    selected_t_counts = t_counts or [48, 96, 288]
    if min(selected_scenario_counts) < 3:
        raise ValueError("normal-regime comparison requires at least three scenarios")
    maximum_scenarios = max(selected_scenario_counts)
    condition_rows: list[dict] = []
    split_rows: list[dict] = []
    extra_rows: list[dict] = []
    total = len(selected_cases) * len(selected_t_counts) * replicates * len(selected_scenario_counts)
    completed = 0

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
                physical_records = _physical_split_records(net, terminals)
                for scenario_count in selected_scenario_counts:
                    fitted = preprocess_scenarios(scenario_pool[:scenario_count], RECIPE)
                    r_matrix, x_matrix, _, _ = fit_projected_sensitivity(
                        fitted,
                        constraint_mode="ordered",
                    )
                    distance, depth, _ = _distance_and_depth(r_matrix, x_matrix, "RX_75R_25X")
                    condition_id = f"{case_key}_r{replicate}_s{scenario_count}_t{t_count}"
                    for method in METHODS:
                        predicted_edges, forced_merges = _reconstruct(
                            method,
                            distance,
                            depth,
                            terminals,
                            net.root_bus,
                            rnj_tolerance_factor,
                            rg_tolerance,
                        )
                        predicted_splits = terminal_splits(predicted_edges, terminals)
                        predicted_siblings = terminal_sibling_pairs(predicted_edges, terminals)
                        precision, recall, f1_score = _set_score(predicted_splits, true_splits)
                        sibling_f1 = _set_score(predicted_siblings, true_siblings)[2]
                        missing = true_splits - predicted_splits
                        extra = predicted_splits - true_splits
                        condition_rows.append(
                            {
                                "condition_id": condition_id,
                                "case": case_key,
                                "replicate": replicate,
                                "scenario_count": scenario_count,
                                "t_count": t_count,
                                "method": method,
                                "split_precision": precision,
                                "split_recall": recall,
                                "split_f1": f1_score,
                                "sibling_f1": sibling_f1,
                                "exact_recovery": predicted_splits == true_splits,
                                "missing_true_splits": len(missing),
                                "extra_predicted_splits": len(extra),
                                "forced_merges": forced_merges,
                            }
                        )
                        for split in true_splits:
                            split_rows.append(
                                {
                                    "condition_id": condition_id,
                                    "case": case_key,
                                    "replicate": replicate,
                                    "scenario_count": scenario_count,
                                    "t_count": t_count,
                                    "method": method,
                                    "recovered": split in predicted_splits,
                                    **physical_records[split],
                                }
                            )
                        for split in extra:
                            nearest = max(
                                true_splits,
                                key=lambda truth: len(split & truth) / max(len(split | truth), 1),
                            )
                            if split < nearest:
                                relation = "spurious_refinement"
                            elif split > nearest:
                                relation = "spurious_superset"
                            else:
                                relation = "cross_branch_substitution"
                            extra_rows.append(
                                {
                                    "condition_id": condition_id,
                                    "case": case_key,
                                    "replicate": replicate,
                                    "scenario_count": scenario_count,
                                    "t_count": t_count,
                                    "method": method,
                                    "extra_split_size": len(split),
                                    "extra_split_members": " ".join(str(node) for node in sorted(split)),
                                    "nearest_true_region": physical_records[nearest]["region"],
                                    "nearest_true_members": physical_records[nearest]["split_members"],
                                    "relation": relation,
                                }
                            )
                    completed += 1
                print(
                    f"[region-compare] {completed}/{total} case={case_key} T={t_count} replicate={replicate}",
                    flush=True,
                )

    conditions = pd.DataFrame(condition_rows)
    split_recovery = pd.DataFrame(split_rows)
    extras = pd.DataFrame(extra_rows)
    conditions.to_csv(out_dir / "condition_comparison.csv", index=False)
    split_recovery.to_csv(out_dir / "true_split_recovery.csv", index=False)
    extras.to_csv(out_dir / "extra_split_details.csv", index=False)

    method_summary = (
        conditions.groupby("method", as_index=False)
        .agg(
            runs=("split_f1", "size"),
            mean_split_f1=("split_f1", "mean"),
            exact_recovery_rate=("exact_recovery", "mean"),
            mean_sibling_f1=("sibling_f1", "mean"),
            missing_true_splits=("missing_true_splits", "sum"),
            extra_predicted_splits=("extra_predicted_splits", "sum"),
        )
        .sort_values("mean_split_f1", ascending=False)
    )
    method_summary.to_csv(out_dir / "method_summary.csv", index=False)
    region_summary = (
        split_recovery.groupby(["method", "region"], as_index=False)
        .agg(opportunities=("recovered", "size"), recovered=("recovered", "sum"))
        .assign(
            missed=lambda frame: frame["opportunities"] - frame["recovered"],
            miss_rate=lambda frame: frame["missed"] / frame["opportunities"],
        )
        .sort_values(["method", "miss_rate"], ascending=[True, False])
    )
    region_summary.to_csv(out_dir / "region_miss_rates.csv", index=False)
    extra_summary = (
        extras.groupby(["method", "relation", "nearest_true_region"], as_index=False)
        .agg(extra_splits=("condition_id", "size"))
        .sort_values(["method", "extra_splits"], ascending=[True, False])
    )
    extra_summary.to_csv(out_dir / "extra_split_summary.csv", index=False)
    metrics = {
        "comparison": "same ordered matrix, RX75 distance, and normal noisy AC data",
        "condition_count": int(len(conditions) / len(METHODS)),
        "method_summary": method_summary.to_dict(orient="records"),
        "region_miss_rates": region_summary.to_dict(orient="records"),
        "extra_split_summary": extra_summary.to_dict(orient="records"),
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# NJ/RG/RNJ error-region comparison\n\n"
        "All methods receive the same ordered R/X estimate and RX75 distance. RG and RNJ "
        "also receive the known root depth; NJ remains the classical unrooted algorithm. "
        "Physical edges are compared by canonical terminal splits.\n\n"
        "## Method summary\n\n"
        + method_summary.to_string(index=False)
        + "\n\n## Region miss rates\n\n"
        + region_summary.to_string(index=False)
        + "\n\n## Extra split types\n\n"
        + extra_summary.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/reconstructor_error_region_comparison")
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
