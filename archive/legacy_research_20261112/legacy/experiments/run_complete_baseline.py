"""Run the complete constrained multi-scenario RNJ baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.run_latent_tree_diagnostics import _set_score
from experiments.run_matrix_constraint_large_sweep import _simulate_pool, _terminal_buses
from experiments.run_probabilistic_root_decoupling import _root_branch_groups
from terminal_case33.data.paper_style_case_bank import CASE_BUILDERS
from terminal_case33.estimation.baseline import CompleteBaselineResult, fit_complete_rnj_baseline
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.utils.io import ensure_dir, write_json


def _export_condition(result: CompleteBaselineResult, output_dir: Path) -> None:
    """Write all fitted matrices, residuals, and tree edges for one condition."""

    condition_dir = ensure_dir(output_dir)
    result.r_matrix.to_csv(condition_dir / "R_hat.csv")
    result.x_matrix.to_csv(condition_dir / "X_hat.csv")
    result.d_r.to_csv(condition_dir / "dR.csv")
    result.d_x.to_csv(condition_dir / "dX.csv")
    result.distance.to_csv(condition_dir / "distance.csv")
    result.root_depths.to_csv(condition_dir / "root_depths.csv")
    pd.DataFrame(
        result.tree.edges,
        columns=["u", "v", "length"],
    ).to_csv(condition_dir / "rnj_tree_edges.csv", index=False)
    pd.DataFrame(
        {
            "clade": [",".join(map(str, sorted(clade))) for clade in result.rooted_clades],
            "size": [len(clade) for clade in result.rooted_clades],
        }
    ).sort_values(["size", "clade"]).to_csv(
        condition_dir / "rooted_clades.csv",
        index=False,
    )
    residual = pd.concat(
        result.delta_v_residuals,
        names=["scenario", "sample"],
    )
    residual.to_csv(condition_dir / "delta_v_residuals.csv")
    write_json(
        condition_dir / "fit_metrics.json",
        {
            "residual_rmse": result.residual_rmse,
            "r2_score": result.r2_score,
            "condition_number": result.condition_number,
            "objective_value": result.objective_value,
            "preprocessing": result.preprocessing,
            "distance_mode": result.distance_mode,
            "constraint_mode": result.constraint_mode,
            "alpha": result.alpha,
            "tolerance_factor": result.tolerance_factor,
        },
    )


def run(
    output_dir: str | Path = "outputs/complete_baseline",
    cases: list[str] | None = None,
    scenario_counts: list[int] | None = None,
    t_counts: list[int] | None = None,
    data_replicates: int = 2,
    pq_noise_rel: float = 0.005,
    v_noise_rel: float = 0.0002,
    preprocessing: str = "raw",
    distance_mode: str = "RX_75R_25X",
    constraint_mode: str = "ordered",
    alpha: float = 0.0,
    tolerance_factor: float = 0.16,
    export_condition_artifacts: bool = True,
) -> dict:
    """Benchmark the complete baseline using only noisy measured AC P/Q/V."""

    out_dir = ensure_dir(output_dir)
    selected_cases = cases or list(CASE_BUILDERS)
    selected_scenarios = scenario_counts or [3, 5]
    selected_t = t_counts or [24, 48, 96]
    maximum_scenarios = max(selected_scenarios)
    condition_count = (
        len(selected_cases)
        * len(selected_scenarios)
        * len(selected_t)
        * data_replicates
    )
    rows = []
    completed = 0
    for case_key in selected_cases:
        for t_count in selected_t:
            for replicate in range(data_replicates):
                net, scenario_pool = _simulate_pool(
                    case_key,
                    t_count,
                    replicate,
                    maximum_scenarios,
                    pq_noise_rel,
                    v_noise_rel,
                )
                terminals = _terminal_buses(net)
                true_clades = rooted_clades(net.closed_edges(), net.root_bus, terminals)
                true_edges = tuple(
                    (int(left), int(right), 1.0)
                    for left, right in net.closed_edges()
                )
                true_root_groups = _root_branch_groups(
                    true_edges,
                    net.root_bus,
                    terminals,
                )
                for scenario_count in selected_scenarios:
                    result = fit_complete_rnj_baseline(
                        scenario_pool[:scenario_count],
                        net.root_bus,
                        preprocessing=preprocessing,
                        distance_mode=distance_mode,
                        constraint_mode=constraint_mode,
                        alpha=alpha,
                        tolerance_factor=tolerance_factor,
                    )
                    predicted = set(result.rooted_clades)
                    precision, recall, f1_score = _set_score(predicted, true_clades)
                    predicted_root_groups = _root_branch_groups(
                        result.tree.edges,
                        net.root_bus,
                        terminals,
                    )
                    row = {
                        "case": case_key,
                        "scenario_count": scenario_count,
                        "t_count": t_count,
                        "data_replicate": replicate,
                        "sample_count": scenario_count * t_count,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1_score,
                        "exact": predicted == true_clades,
                        "root_partition_exact": predicted_root_groups == true_root_groups,
                        "predicted_clade_count": len(predicted),
                        "true_clade_count": len(true_clades),
                        "residual_rmse": result.residual_rmse,
                        "r2_score": result.r2_score,
                        "condition_number": result.condition_number,
                        "objective_value": result.objective_value,
                    }
                    rows.append(row)
                    if export_condition_artifacts:
                        condition_dir = (
                            out_dir
                            / "conditions"
                            / case_key
                            / f"scenarios_{scenario_count}"
                            / f"T_{t_count}"
                            / f"replicate_{replicate}"
                        )
                        _export_condition(result, condition_dir)
                    completed += 1
                    print(
                        f"[complete-baseline] {completed}/{condition_count} case={case_key} "
                        f"scenarios={scenario_count} T={t_count} replicate={replicate}",
                        flush=True,
                    )

    results = pd.DataFrame(rows)
    results["regime"] = results["scenario_count"].map(
        lambda count: "normal_multiscenario" if count >= 3 else "single_scenario_stress"
    )
    results.to_csv(out_dir / "condition_results.csv", index=False)
    summary = (
        results.groupby("regime", as_index=False)
        .agg(
            runs=("f1", "size"),
            mean_f1=("f1", "mean"),
            exact_rate=("exact", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            root_partition_exact_rate=("root_partition_exact", "mean"),
            mean_residual_rmse=("residual_rmse", "mean"),
            mean_r2=("r2_score", "mean"),
            median_condition_number=("condition_number", "median"),
        )
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    by_case = (
        results.groupby(["regime", "case"], as_index=False)
        .agg(
            runs=("f1", "size"),
            mean_f1=("f1", "mean"),
            exact_rate=("exact", "mean"),
            root_partition_exact_rate=("root_partition_exact", "mean"),
        )
    )
    by_case.to_csv(out_dir / "summary_by_case.csv", index=False)
    metrics = {
        "method": "complete_ordered_multiscenario_RNJ_baseline",
        "condition_count": condition_count,
        "measurement_noise": {
            "pq_relative": pq_noise_rel,
            "voltage_relative": v_noise_rel,
        },
        "preprocessing": preprocessing,
        "distance_mode": distance_mode,
        "constraint_mode": constraint_mode,
        "alpha": alpha,
        "tolerance_factor": tolerance_factor,
        "explicit_outputs": [
            "R_hat",
            "X_hat",
            "dR",
            "dX",
            "distance",
            "root_depths",
            "delta_v_residuals",
            "rnj_tree_edges",
            "rooted_clades",
        ],
        "summary": summary.to_dict(orient="records"),
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# Complete constrained RNJ baseline\n\n"
        "The baseline uses only noisy measured terminal P/Q/V and observed root voltage. "
        f"It uses {preprocessing} preprocessing and one shared ordered symmetric "
        "nonnegative R/X fit without terminal offsets, "
        "RX75 additive-distance construction, and known-root RNJ. No pseudo aggregation, "
        "latent common modes, oracle values, bootstrap selection, or truth labels are used "
        "during fitting. Delta V is exported as the fitted voltage/model residual.\n\n"
        "## Summary\n\n"
        + summary.to_string(index=False)
        + "\n\n## By case\n\n"
        + by_case.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/complete_baseline")
    parser.add_argument("--cases", nargs="*", choices=list(CASE_BUILDERS), default=None)
    parser.add_argument("--scenario-counts", nargs="*", type=int, default=None)
    parser.add_argument("--T-counts", nargs="*", type=int, default=None)
    parser.add_argument("--data-replicates", type=int, default=2)
    parser.add_argument("--pq-noise-rel", type=float, default=0.005)
    parser.add_argument("--v-noise-rel", type=float, default=0.0002)
    parser.add_argument("--preprocessing", choices=["raw", "daily_demean", "difference"], default="raw")
    parser.add_argument(
        "--distance-mode",
        choices=["R", "X", "RX_equal_normalized", "RX_75R_25X", "RX_25R_75X"],
        default="RX_75R_25X",
    )
    parser.add_argument(
        "--constraint-mode",
        choices=["ordered"],
        default="ordered",
    )
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--tolerance-factor", type=float, default=0.16)
    parser.add_argument(
        "--no-condition-artifacts",
        dest="export_condition_artifacts",
        action="store_false",
    )
    parser.set_defaults(export_condition_artifacts=True)
    args = parser.parse_args()
    run(
        output_dir=args.output,
        cases=args.cases,
        scenario_counts=args.scenario_counts,
        t_counts=args.T_counts,
        data_replicates=args.data_replicates,
        pq_noise_rel=args.pq_noise_rel,
        v_noise_rel=args.v_noise_rel,
        preprocessing=args.preprocessing,
        distance_mode=args.distance_mode,
        constraint_mode=args.constraint_mode,
        alpha=args.alpha,
        tolerance_factor=args.tolerance_factor,
        export_condition_artifacts=args.export_condition_artifacts,
    )


if __name__ == "__main__":
    main()
