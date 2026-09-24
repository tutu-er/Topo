"""Large noisy AC sweep for reduced-sensitivity matrix constraints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import resource_assignment, selected_scenario_defs, simulate_case_scenario
from experiments.run_latent_tree_diagnostics import _distance_and_depth, _set_score
from experiments.run_rooted_hierarchical_aggregation import RECIPE
from terminal_case33.data.paper_style_case_bank import CASE_BUILDERS
from terminal_case33.estimation.matrix_constraints import (
    four_point_violation_summary,
    sensitivity_matrix_diagnostics,
)
from terminal_case33.estimation.multiscenario import fit_projected_sensitivity, preprocess_scenarios
from terminal_case33.graph.diagnostics import terminal_sibling_pairs
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.graph.rooted_neighbor_joining import rooted_neighbor_joining, shared_paths_from_distances
from terminal_case33.models.lin_distflow import build_reduced_sensitivity_matrices
from terminal_case33.utils.io import ensure_dir, write_json


CONSTRAINT_MODES = ["basic", "ordered", "tree_covariance"]
DISTANCE_MODES = ["R", "X", "RX_75R_25X"]


def _terminal_buses(net) -> list[int]:
    return net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"].astype(int).tolist()


def _simulate_pool(
    case_key: str,
    t_count: int,
    replicate: int,
    maximum_scenarios: int,
    pq_noise_rel: float,
    v_noise_rel: float,
) -> tuple[object, list[dict]]:
    """Simulate one nested scenario pool using noisy AC power flow."""

    net = CASE_BUILDERS[case_key]()
    assignment = resource_assignment(case_key, net)
    scenarios = []
    definitions = selected_scenario_defs(max_scenarios=maximum_scenarios)
    if len(definitions) < maximum_scenarios:
        raise ValueError(f"only {len(definitions)} scenario definitions are available")
    for definition in definitions:
        shifted_seed = int(definition["seed"]) + 100_003 * replicate
        simulation = simulate_case_scenario(
            net=net,
            assignment=assignment,
            T=t_count,
            seed=shifted_seed,
            pq_noise_rel=pq_noise_rel,
            v_noise_rel=v_noise_rel,
            root_voltage_mean=1.02,
            root_voltage_sigma=float(definition["root_voltage_sigma"]),
            profile_scenario=str(definition["profile_scenario"]),
        )
        if not simulation["ac_converged"]:
            raise RuntimeError(f"AC power flow did not converge for {case_key}")
        scenarios.append(
            {
                "name": f"{definition['name']}_rep{replicate}",
                "P_terminal": simulation["P_terminal"],
                "Q_terminal": simulation["Q_terminal"],
                "V_terminal": simulation["V_terminal"],
                "root_voltage": simulation["root_voltage"],
                "drop_target": simulation["drop_target"],
            }
        )
    return net, scenarios


def _summaries(fits: pd.DataFrame, topology: pd.DataFrame) -> dict[str, pd.DataFrame]:
    default = topology.loc[
        topology["distance_mode"].eq("RX_75R_25X")
        & np.isclose(topology["tolerance_factor"], 0.16)
    ].copy()
    merged = fits.merge(
        default[
            [
                "case",
                "replicate",
                "scenario_count",
                "t_count",
                "constraint_mode",
                "clade_f1",
                "sibling_f1",
            ]
        ],
        on=["case", "replicate", "scenario_count", "t_count", "constraint_mode"],
        how="left",
    )
    def aggregate_constraints(data: pd.DataFrame) -> pd.DataFrame:
        return (
            data.groupby("constraint_mode", as_index=False)
            .agg(
                runs=("clade_f1", "size"),
                mean_clade_f1=("clade_f1", "mean"),
                median_clade_f1=("clade_f1", "median"),
                minimum_clade_f1=("clade_f1", "min"),
                perfect_clade_rate=("clade_f1", lambda value: float(np.mean(np.isclose(value, 1.0)))),
                mean_sibling_f1=("sibling_f1", "mean"),
                mean_relative_error_r=("relative_error_r", "mean"),
                mean_relative_error_x=("relative_error_x", "mean"),
                mean_four_point_violation=("mean_four_point_violation", "mean"),
                minimum_r_diagonal_gap=("r_min_diagonal_gap", "min"),
                minimum_x_diagonal_gap=("x_min_diagonal_gap", "min"),
                minimum_r_eigenvalue=("r_min_eigenvalue", "min"),
                minimum_x_eigenvalue=("x_min_eigenvalue", "min"),
            )
            .sort_values("mean_clade_f1", ascending=False)
        )

    constraint_all = aggregate_constraints(merged)
    constraint_normal = aggregate_constraints(merged.loc[merged["scenario_count"] >= 3])
    constraint_stress = aggregate_constraints(merged.loc[merged["scenario_count"] == 1])
    by_case = default.groupby(["case", "constraint_mode"], as_index=False).agg(
        runs=("clade_f1", "size"),
        mean_clade_f1=("clade_f1", "mean"),
        minimum_clade_f1=("clade_f1", "min"),
        perfect_rate=("clade_f1", lambda value: float(np.mean(np.isclose(value, 1.0)))),
    )
    by_scenarios = default.groupby(["scenario_count", "constraint_mode"], as_index=False).agg(
        runs=("clade_f1", "size"),
        mean_clade_f1=("clade_f1", "mean"),
        minimum_clade_f1=("clade_f1", "min"),
    )
    by_t = default.groupby(["t_count", "constraint_mode"], as_index=False).agg(
        runs=("clade_f1", "size"),
        mean_clade_f1=("clade_f1", "mean"),
        minimum_clade_f1=("clade_f1", "min"),
    )
    grid = topology.groupby(["constraint_mode", "distance_mode", "tolerance_factor"], as_index=False).agg(
        runs=("clade_f1", "size"),
        mean_clade_f1=("clade_f1", "mean"),
        minimum_clade_f1=("clade_f1", "min"),
        perfect_rate=("clade_f1", lambda value: float(np.mean(np.isclose(value, 1.0)))),
    )
    return {
        "constraint_summary": constraint_normal,
        "constraint_summary_all": constraint_all,
        "constraint_summary_stress": constraint_stress,
        "by_case": by_case,
        "by_scenario_count": by_scenarios,
        "by_t_count": by_t,
        "topology_grid_summary": grid,
    }


def run(
    output_dir: str | Path = "outputs/matrix_constraint_large_sweep",
    cases: list[str] | None = None,
    scenario_counts: list[int] | None = None,
    t_counts: list[int] | None = None,
    replicates: int = 2,
    tolerance_factors: list[float] | None = None,
    pq_noise_rel: float = 0.005,
    v_noise_rel: float = 0.0002,
    diagonal_margin_ratio: float = 1e-6,
) -> dict:
    """Run a cached, reproducible, noisy AC matrix-constraint sweep."""

    out_dir = ensure_dir(output_dir)
    selected_cases = cases or list(CASE_BUILDERS)
    selected_scenario_counts = scenario_counts or [1, 3, 5, 10]
    selected_t_counts = t_counts or [48, 96, 288]
    selected_tolerances = tolerance_factors or [0.12, 0.16, 0.20]
    if replicates < 1:
        raise ValueError("replicates must be positive")
    maximum_scenarios = max(selected_scenario_counts)
    total_fits = (
        len(selected_cases)
        * len(selected_t_counts)
        * replicates
        * len(selected_scenario_counts)
        * len(CONSTRAINT_MODES)
    )
    completed_fits = 0
    fit_rows: list[dict] = []
    topology_rows: list[dict] = []

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
                true_clades = rooted_clades(net.closed_edges(), net.root_bus, terminals)
                true_siblings = terminal_sibling_pairs(net.closed_edges(), terminals)
                r_true, x_true = build_reduced_sensitivity_matrices(
                    net,
                    terminals,
                    voltage_model="squared-voltage",
                )
                for scenario_count in selected_scenario_counts:
                    fitted = preprocess_scenarios(scenario_pool[:scenario_count], RECIPE)
                    for constraint_mode in CONSTRAINT_MODES:
                        r_hat, x_hat, r2_score, condition_number = fit_projected_sensitivity(
                            fitted,
                            constraint_mode=constraint_mode,
                            diagonal_margin_ratio=diagonal_margin_ratio,
                        )
                        r_diag = sensitivity_matrix_diagnostics(r_hat)
                        x_diag = sensitivity_matrix_diagnostics(x_hat)
                        default_distance, _, _ = _distance_and_depth(r_hat, x_hat, "RX_75R_25X")
                        four_point = four_point_violation_summary(default_distance)
                        fit_rows.append(
                            {
                                "case": case_key,
                                "replicate": replicate,
                                "scenario_count": scenario_count,
                                "t_count": t_count,
                                "samples": int(sum(len(item["P_terminal"]) for item in fitted)),
                                "terminal_count": len(terminals),
                                "hidden_count": len(net.hidden_buses()),
                                "constraint_mode": constraint_mode,
                                "r2_score": r2_score,
                                "condition_number": condition_number,
                                "relative_error_r": float(np.linalg.norm(r_hat - r_true) / np.linalg.norm(r_true)),
                                "relative_error_x": float(np.linalg.norm(x_hat - x_true) / np.linalg.norm(x_true)),
                                "r_min_diagonal_gap": r_diag.minimum_diagonal_gap,
                                "x_min_diagonal_gap": x_diag.minimum_diagonal_gap,
                                "r_min_eigenvalue": r_diag.minimum_eigenvalue,
                                "x_min_eigenvalue": x_diag.minimum_eigenvalue,
                                "r_precision_positive_offdiag_fraction": r_diag.precision_positive_offdiag_fraction,
                                "x_precision_positive_offdiag_fraction": x_diag.precision_positive_offdiag_fraction,
                                **four_point,
                            }
                        )
                        for distance_mode in DISTANCE_MODES:
                            distance, depth, _ = _distance_and_depth(r_hat, x_hat, distance_mode)
                            shared_paths = shared_paths_from_distances(distance, depth)
                            depth_scale = max(float(np.median(depth)), 1e-12)
                            for tolerance_factor in selected_tolerances:
                                tree = rooted_neighbor_joining(
                                    shared_paths,
                                    depth,
                                    terminals,
                                    net.root_bus,
                                    tolerance_factor * depth_scale,
                                )
                                predicted_clades = rooted_clades(tree.edges, net.root_bus, terminals)
                                predicted_siblings = terminal_sibling_pairs(tree.edges, terminals)
                                clade_precision, clade_recall, clade_f1 = _set_score(predicted_clades, true_clades)
                                sibling_precision, sibling_recall, sibling_f1 = _set_score(
                                    predicted_siblings,
                                    true_siblings,
                                )
                                topology_rows.append(
                                    {
                                        "case": case_key,
                                        "replicate": replicate,
                                        "scenario_count": scenario_count,
                                        "t_count": t_count,
                                        "constraint_mode": constraint_mode,
                                        "distance_mode": distance_mode,
                                        "tolerance_factor": tolerance_factor,
                                        "clade_precision": clade_precision,
                                        "clade_recall": clade_recall,
                                        "clade_f1": clade_f1,
                                        "sibling_precision": sibling_precision,
                                        "sibling_recall": sibling_recall,
                                        "sibling_f1": sibling_f1,
                                        "group_tolerance": tree.group_tolerance,
                                    }
                                )
                        completed_fits += 1
                pd.DataFrame(fit_rows).to_csv(out_dir / "fit_results.csv", index=False)
                pd.DataFrame(topology_rows).to_csv(out_dir / "topology_results.csv", index=False)
                print(
                    f"[constraint-large] {completed_fits}/{total_fits} "
                    f"case={case_key} T={t_count} replicate={replicate}",
                    flush=True,
                )

    fits = pd.DataFrame(fit_rows)
    topology = pd.DataFrame(topology_rows)
    summaries = _summaries(fits, topology)
    for name, frame in summaries.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    best_grid = summaries["topology_grid_summary"].sort_values(
        ["mean_clade_f1", "minimum_clade_f1", "perfect_rate"],
        ascending=False,
    )
    metrics = {
        "method": "matrix_constraint_large_noisy_ac_sweep",
        "cases": selected_cases,
        "scenario_counts": selected_scenario_counts,
        "t_counts": selected_t_counts,
        "replicates": replicates,
        "fit_count": int(len(fits)),
        "topology_run_count": int(len(topology)),
        "preprocess": RECIPE["name"],
        "pq_noise_relative_std": pq_noise_rel,
        "voltage_noise_relative_std": v_noise_rel,
        "voltage_noise_reference": "instantaneous voltage magnitude at each terminal",
        "diagonal_margin_ratio": diagonal_margin_ratio,
        "normal_regime": "scenario_count >= 3, defined before topology scoring",
        "stress_regime": "scenario_count == 1, reported separately",
        "normal_constraint_summary": summaries["constraint_summary"].to_dict(orient="records"),
        "all_condition_summary": summaries["constraint_summary_all"].to_dict(orient="records"),
        "stress_constraint_summary": summaries["constraint_summary_stress"].to_dict(orient="records"),
        "best_fixed_topology_settings": best_grid.head(10).to_dict(orient="records"),
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# Large noisy AC matrix-constraint sweep\n\n"
        f"- Fits: {len(fits)}\n"
        f"- Topology reconstructions: {len(topology)}\n"
        f"- PQ noise: {100 * pq_noise_rel:.3f}% relative standard deviation\n"
        f"- Voltage noise: {100 * v_noise_rel:.3f}% relative standard deviation\n"
        f"- Default evaluation: RX_75R_25X, tolerance factor 0.16\n\n"
        "## Constraint summary\n\n"
        + "```text\n"
        + summaries["constraint_summary"].to_string(index=False)
        + "\n```"
        + "\n\n## Best fixed topology settings\n\n"
        + "```text\n"
        + best_grid.head(10).to_string(index=False)
        + "\n```\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/matrix_constraint_large_sweep")
    parser.add_argument("--cases", nargs="*", choices=list(CASE_BUILDERS), default=None)
    parser.add_argument("--scenario-counts", nargs="*", type=int, default=None)
    parser.add_argument("--T-counts", nargs="*", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--tolerance-factors", nargs="*", type=float, default=None)
    parser.add_argument("--pq-noise-rel", type=float, default=0.005)
    parser.add_argument("--v-noise-rel", type=float, default=0.0002)
    parser.add_argument("--diagonal-margin-ratio", type=float, default=1e-6)
    args = parser.parse_args()
    run(
        output_dir=args.output,
        cases=args.cases,
        scenario_counts=args.scenario_counts,
        t_counts=args.T_counts,
        replicates=args.replicates,
        tolerance_factors=args.tolerance_factors,
        pq_noise_rel=args.pq_noise_rel,
        v_noise_rel=args.v_noise_rel,
        diagonal_margin_ratio=args.diagonal_margin_ratio,
    )


if __name__ == "__main__":
    main()
