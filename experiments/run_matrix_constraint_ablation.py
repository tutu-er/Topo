"""Compare basic and tree-covariance sensitivity matrix constraints."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.run_latent_tree_diagnostics import (
    _detailed_scenarios,
    _distance_and_depth,
    _set_score,
)
from experiments.run_rooted_hierarchical_aggregation import RECIPE
from terminal_case33.estimation.matrix_constraints import (
    four_point_violation_summary,
    sensitivity_matrix_diagnostics,
)
from terminal_case33.estimation.multiscenario import (
    fit_projected_sensitivity,
    preprocess_scenarios,
)
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.graph.rooted_neighbor_joining import (
    rooted_neighbor_joining,
    shared_paths_from_distances,
)
from terminal_case33.utils.io import ensure_dir, write_json


def run(
    output_dir: str | Path = "outputs/matrix_constraint_ablation",
    cases: list[str] | None = None,
    scenario_counts: list[int] | None = None,
    t_counts: list[int] | None = None,
    diagonal_margin_ratio: float = 1e-6,
) -> dict:
    """Evaluate matrix feasibility and rooted-clade F1 under each constraint."""

    out_dir = ensure_dir(output_dir)
    selected_cases = cases or ["paper15", "soumalas11", "flynn16", "pengwah18"]
    selected_scenarios = scenario_counts or [1, 3]
    selected_t = t_counts or [96, 288]
    rows = []
    for case_key in selected_cases:
        for scenario_count in selected_scenarios:
            for t_count in selected_t:
                net, scenarios = _detailed_scenarios(
                    case_key,
                    scenario_count,
                    t_count,
                    pq_noise_rel=0.005,
                )
                terminals = net.load_buses()
                true_clades = rooted_clades(net.closed_edges(), net.root_bus, terminals)
                fitted = preprocess_scenarios(scenarios, RECIPE)
                for mode in ["basic", "ordered", "tree_covariance"]:
                    r_matrix, x_matrix, r2_score, condition_number = fit_projected_sensitivity(
                        fitted,
                        constraint_mode=mode,
                        diagonal_margin_ratio=diagonal_margin_ratio,
                    )
                    distance, depth, _ = _distance_and_depth(
                        r_matrix,
                        x_matrix,
                        "RX_75R_25X",
                    )
                    tolerance = 0.16 * max(float(pd.Series(depth).median()), 1e-12)
                    tree = rooted_neighbor_joining(
                        shared_paths_from_distances(distance, depth),
                        depth,
                        terminals,
                        net.root_bus,
                        tolerance,
                    )
                    predicted = rooted_clades(tree.edges, net.root_bus, terminals)
                    precision, recall, f1_score = _set_score(predicted, true_clades)
                    r_diagnostics = sensitivity_matrix_diagnostics(r_matrix)
                    x_diagnostics = sensitivity_matrix_diagnostics(x_matrix)
                    four_point = four_point_violation_summary(distance)
                    rows.append(
                        {
                            "case": case_key,
                            "scenario_count": scenario_count,
                            "t_count": t_count,
                            "constraint_mode": mode,
                            "r2_score": r2_score,
                            "condition_number": condition_number,
                            "clade_precision": precision,
                            "clade_recall": recall,
                            "clade_f1": f1_score,
                            "predicted_clades": len(predicted),
                            "r_min_diagonal_gap": r_diagnostics.minimum_diagonal_gap,
                            "x_min_diagonal_gap": x_diagnostics.minimum_diagonal_gap,
                            "r_min_eigenvalue": r_diagnostics.minimum_eigenvalue,
                            "x_min_eigenvalue": x_diagnostics.minimum_eigenvalue,
                            "r_min_distance": r_diagnostics.minimum_distance,
                            "x_min_distance": x_diagnostics.minimum_distance,
                            **four_point,
                        }
                    )
                print(
                    f"[matrix-constraints] case={case_key} scenarios={scenario_count} T={t_count}",
                    flush=True,
                )

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "constraint_results.csv", index=False)
    summary = (
        results.groupby("constraint_mode", as_index=False)
        .agg(
            runs=("clade_f1", "size"),
            mean_clade_f1=("clade_f1", "mean"),
            minimum_clade_f1=("clade_f1", "min"),
            mean_r2=("r2_score", "mean"),
            minimum_r_diagonal_gap=("r_min_diagonal_gap", "min"),
            minimum_x_diagonal_gap=("x_min_diagonal_gap", "min"),
            minimum_r_eigenvalue=("r_min_eigenvalue", "min"),
            minimum_x_eigenvalue=("x_min_eigenvalue", "min"),
            mean_four_point_violation=("mean_four_point_violation", "mean"),
        )
    )
    summary.to_csv(out_dir / "constraint_summary.csv", index=False)
    metrics = {
        "diagonal_margin_ratio": diagonal_margin_ratio,
        "results": summary.to_dict(orient="records"),
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# Sensitivity matrix constraint ablation\n\n"
        + summary.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/matrix_constraint_ablation")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--scenario-counts", nargs="*", type=int, default=None)
    parser.add_argument("--T-counts", nargs="*", type=int, default=None)
    parser.add_argument("--diagonal-margin-ratio", type=float, default=1e-6)
    args = parser.parse_args()
    run(
        output_dir=args.output,
        cases=args.cases,
        scenario_counts=args.scenario_counts,
        t_counts=args.T_counts,
        diagonal_margin_ratio=args.diagonal_margin_ratio,
    )


if __name__ == "__main__":
    main()
