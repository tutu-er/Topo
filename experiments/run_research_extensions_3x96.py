"""Benchmark uncertainty RNJ, joint pseudo voltage, and honest cross-fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from experiments.run_latent_tree_diagnostics import _detailed_scenarios, _set_score
from terminal_case33.estimation.baseline import fit_complete_rnj_baseline
from terminal_case33.graph.rooted_hierarchy import (
    aggregate_rooted_scenarios,
    rooted_clades,
)
from terminal_case33.pipeline.blocked_three_stage_crossfit import (
    identify_topology_blocked_three_stage_crossfit,
)
from terminal_case33.pipeline.peripheral_edge_proposals import (
    propose_peripheral_clusters,
)
from terminal_case33.pipeline.pseudo_parent_voltage import (
    aggregate_rooted_scenarios_joint,
)
from terminal_case33.pipeline.three_stage_crossfit import (
    identify_topology_three_stage_crossfit,
)
from terminal_case33.pipeline.two_level_aggregation import (
    run_ordered_two_level_aggregation,
)
from terminal_case33.pipeline.uncertainty_aware_rnj import (
    estimate_uncertainty_aware_rnj,
)
from terminal_case33.utils.io import ensure_dir, write_json


def _score(prediction: set[frozenset[int]], truth: set[frozenset[int]]) -> dict:
    precision, recall, f1 = _set_score(prediction, truth)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "exact": bool(prediction == truth),
        "predicted_clades": int(len(prediction)),
        "matched_clades": int(len(prediction & truth)),
    }


def _lca(net, members: frozenset[int]) -> int:
    graph = net.to_networkx_graph()
    paths = [nx.shortest_path(graph, net.root_bus, int(member)) for member in members]
    common = set(paths[0]).intersection(*(set(path) for path in paths[1:]))
    return max(common, key=lambda node: nx.shortest_path_length(graph, net.root_bus, node))


def _pseudo_voltage_rmse(
    net,
    scenarios: list[dict],
    pseudo_scenarios: list[dict],
    pseudo_members: dict[int, frozenset[int]],
    truth: set[frozenset[int]],
) -> float:
    residuals = []
    for pseudo_id, members in pseudo_members.items():
        if len(members) < 2 or members not in truth:
            continue
        boundary = _lca(net, members)
        for source, pseudo in zip(scenarios, pseudo_scenarios, strict=True):
            true_vsq = source["V_bus_true"][boundary].to_numpy(dtype=float) ** 2
            estimated_vsq = pseudo["V_terminal"][pseudo_id].to_numpy(dtype=float) ** 2
            residuals.append(estimated_vsq - true_vsq)
    if not residuals:
        return float("nan")
    return float(np.sqrt(np.mean(np.concatenate(residuals) ** 2)))


def _choose_clusters(proposals):
    consensus = list(proposals.clusters_by_mode.get("nj_rg_consensus", ()))
    return "nj_rg_consensus", consensus


def run_case(
    case_key: str,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[list[dict], dict]:
    """Run every extension on one standard noisy AC case."""

    net, scenarios = _detailed_scenarios(
        case_key,
        scenario_count=3,
        t_count=96,
        pq_noise_rel=0.005,
    )
    terminals = [int(column) for column in scenarios[0]["P_terminal"].columns]
    truth = set(rooted_clades(net.closed_edges(), net.root_bus, terminals))
    base = fit_complete_rnj_baseline(
        scenarios,
        net.root_bus,
        preprocessing="daily_demean",
        distance_mode="RX_75R_25X",
        constraint_mode="ordered",
        tolerance_factor=0.16,
    )
    uncertainty = estimate_uncertainty_aware_rnj(
        scenarios,
        net.root_bus,
        tolerance_factor=0.16,
        bootstrap_replicates=bootstrap_replicates,
        block_length=12,
        confidence_level=0.90,
        seed=seed,
    )
    proposals = propose_peripheral_clusters(
        scenarios,
        net.root_bus,
        preprocessing_recipe={"name": "daily_demean", "kind": "demean"},
        distance_mode="RX_75R_25X",
        bootstrap_replicates=bootstrap_replicates,
        maximum_cluster_size=5,
        minimum_support=0.70,
        minimum_boundary_margin=0.0,
        minimum_edge_length_ratio=0.0,
        block_fraction=0.50,
        pq_extra_noise_rel=0.0025,
        voltage_extra_noise_rel=0.0001,
        seed=seed + 31,
    )
    detector_mode, clusters = _choose_clusters(proposals)
    method_predictions: dict[str, set[frozenset[int]]] = {
        "ordered_rnj": set(base.rooted_clades),
        "uncertainty_rnj": set(uncertainty.uncertain_clades),
    }
    deembedded_rmse = float("nan")
    joint_rmse = float("nan")
    condition_deembedded = float("nan")
    condition_joint = float("nan")
    if clusters:
        deembedded = run_ordered_two_level_aggregation(
            scenarios,
            net.root_bus,
            clusters,
            pseudo_voltage_mode="deembedded_vsq",
        )
        joint = run_ordered_two_level_aggregation(
            scenarios,
            net.root_bus,
            clusters,
            pseudo_voltage_mode="joint_latent_vsq",
        )
        method_predictions["aggregation_deembedded"] = set(
            deembedded.candidate_clades_by_mode["ordered_aggregation_hybrid_local"]
        )
        method_predictions["aggregation_joint_parent"] = set(
            joint.candidate_clades_by_mode["ordered_aggregation_hybrid_local"]
        )
        condition_deembedded = float(deembedded.condition_improvement)
        condition_joint = float(joint.condition_improvement)
        pseudo_deembedded, pseudo_members = aggregate_rooted_scenarios(
            scenarios,
            terminals,
            base.r_matrix.to_numpy(dtype=float),
            base.x_matrix.to_numpy(dtype=float),
            clusters,
            "deembedded_vsq",
            deembedding_weight=0.15,
        )
        pseudo_joint, joint_members, _fits = aggregate_rooted_scenarios_joint(
            scenarios,
            terminals,
            base.r_matrix.to_numpy(dtype=float),
            base.x_matrix.to_numpy(dtype=float),
            clusters,
        )
        deembedded_rmse = _pseudo_voltage_rmse(
            net, scenarios, pseudo_deembedded, pseudo_members, truth
        )
        joint_rmse = _pseudo_voltage_rmse(
            net, scenarios, pseudo_joint, joint_members, truth
        )
    else:
        method_predictions["aggregation_deembedded"] = set(base.rooted_clades)
        method_predictions["aggregation_joint_parent"] = set(base.rooted_clades)

    crossfit = identify_topology_three_stage_crossfit(
        scenarios,
        net,
        pq_noise_relative_std=0.005,
        voltage_noise_relative_std=0.0002,
        discovery_bootstrap_replicates=max(2, bootstrap_replicates // 2),
        uncertainty_bootstrap_replicates=max(2, bootstrap_replicates // 2),
        seed=seed + 73,
    )
    method_predictions["three_stage_crossfit"] = set(crossfit.selected_clades)
    blocked_crossfit = identify_topology_blocked_three_stage_crossfit(
        scenarios,
        net,
        pq_noise_relative_std=0.005,
        voltage_noise_relative_std=0.0002,
        temporal_block_length=8,
        discovery_bootstrap_replicates=max(2, bootstrap_replicates // 2),
        uncertainty_bootstrap_replicates=max(2, bootstrap_replicates // 2),
        seed=seed + 97,
    )
    method_predictions["blocked_three_stage_crossfit"] = set(
        blocked_crossfit.selected_clades
    )
    rows = []
    for method, prediction in method_predictions.items():
        rows.append(
            {
                "case": case_key,
                "method": method,
                "scenario_count": 3,
                "samples_per_scenario": 96,
                "total_samples": 288,
                "pq_noise_relative_std": 0.005,
                "voltage_noise_relative_std": 0.0002,
                "root_voltage_sigma_pu": 0.0008,
                "terminal_count": len(terminals),
                "true_clades": len(truth),
                **_score(prediction, truth),
            }
        )
    selected_cluster_count = len(clusters)
    true_cluster_rate = (
        float(np.mean([cluster.members in truth for cluster in clusters]))
        if clusters
        else float("nan")
    )
    diagnostics = {
        "case": case_key,
        "all_ac_voltage_results_finite": bool(all(scenario["V_bus_true"].notna().all().all() for scenario in scenarios)),
        "detector_mode": detector_mode,
        "selected_cluster_count": selected_cluster_count,
        "selected_cluster_true_rate": true_cluster_rate,
        "deembedded_pseudo_vsq_rmse": deembedded_rmse,
        "joint_pseudo_vsq_rmse": joint_rmse,
        "deembedded_condition_improvement": condition_deembedded,
        "joint_condition_improvement": condition_joint,
        "uncertainty_mean_shared_path_se": float(
            np.mean(uncertainty.shared_path_standard_error)
        ),
        "uncertainty_selected_clade_mean_support": float(
            np.mean(
                [
                    uncertainty.bootstrap_clade_support.get(clade, 0.0)
                    for clade in uncertainty.uncertain_clades
                ]
            )
            if uncertainty.uncertain_clades
            else 1.0
        ),
        "crossfit_topology_selection_support": crossfit.topology_selection_support,
        "crossfit_selected_sources": [fold.selected_source for fold in crossfit.folds],
        "crossfit_candidate_counts": [fold.candidate_count for fold in crossfit.folds],
        "blocked_crossfit_topology_selection_support": blocked_crossfit.topology_selection_support,
        "blocked_crossfit_selected_sources": [
            fold.selected_source for fold in blocked_crossfit.folds
        ],
        "blocked_crossfit_candidate_counts": [
            fold.candidate_count for fold in blocked_crossfit.folds
        ],
        "blocked_crossfit_stage_sample_counts": [
            list(fold.stage_sample_counts) for fold in blocked_crossfit.folds
        ],
    }
    return rows, diagnostics


def run(
    output_dir: str | Path = "outputs/research_extensions_3x96",
    cases: list[str] | None = None,
    bootstrap_replicates: int = 6,
) -> dict:
    """Run the 3-day, 96-point, noisy AC benchmark."""

    selected_cases = cases or ["paper15", "soumalas11", "flynn16", "pengwah18"]
    output = ensure_dir(output_dir)
    rows = []
    diagnostics = []
    for index, case_key in enumerate(selected_cases, start=1):
        print(
            f"[research-3x96] {index}/{len(selected_cases)} case={case_key}",
            flush=True,
        )
        case_rows, case_diagnostics = run_case(
            case_key,
            bootstrap_replicates,
            seed=20260716 + 1000 * index,
        )
        rows.extend(case_rows)
        diagnostics.append(case_diagnostics)
        print(
            "  "
            + " ".join(
                f"{row['method']}={row['f1']:.3f}" for row in case_rows
            ),
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "method_results.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "case_diagnostics.csv", index=False)
    summary_frame = (
        frame.groupby("method", as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            minimum_f1=("f1", "min"),
            exact_rate=("exact", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
        )
        .sort_values("mean_f1", ascending=False)
    )
    summary_frame.to_csv(output / "method_summary.csv", index=False)
    summary = {
        "conditions": {
            "scenario_count": 3,
            "samples_per_scenario": 96,
            "sample_interval_minutes": 15,
            "total_samples": 288,
            "power_flow": "radial AC solved independently at every sample",
            "pq_noise_relative_std": 0.005,
            "voltage_noise_relative_std": 0.0002,
            "voltage_noise_reference": "instantaneous local voltage magnitude",
            "root_voltage_sigma_pu": 0.0008,
        },
        "cases": selected_cases,
        "bootstrap_replicates": bootstrap_replicates,
        "method_summary": summary_frame.to_dict(orient="records"),
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="outputs/research_extensions_3x96",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["paper15", "soumalas11", "flynn16", "pengwah18"],
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=6)
    args = parser.parse_args()
    run(args.output_dir, args.cases, args.bootstrap_replicates)


if __name__ == "__main__":
    main()
