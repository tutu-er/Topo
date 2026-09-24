"""Tune regularized joint pseudo-parent aggregation on noisy AC scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_latent_tree_diagnostics import _detailed_scenarios, _set_score
from experiments.run_research_extensions_3x96 import _pseudo_voltage_rmse
from terminal_case33.estimation.baseline import fit_complete_rnj_baseline
from terminal_case33.graph.rooted_hierarchy import (
    aggregate_rooted_scenarios,
    rooted_clades,
)
from terminal_case33.pipeline.peripheral_edge_proposals import (
    propose_peripheral_clusters,
)
from terminal_case33.pipeline.pseudo_parent_voltage import (
    aggregate_rooted_scenarios_joint,
)
from terminal_case33.pipeline.two_level_aggregation import (
    _predictive_nrmse,
    run_ordered_two_level_aggregation,
)
from terminal_case33.utils.io import ensure_dir, write_json


RECIPE = {"name": "daily_demean", "kind": "demean"}


CONFIGS = (
    {"name": "deembedded_w015", "mode": "deembedded_vsq", "weight": 0.15},
    {"name": "deembedded_w050", "mode": "deembedded_vsq", "weight": 0.50},
    {
        "name": "joint_old_iterative_q20",
        "mode": "joint_latent_vsq",
        "gauge": 0.20,
        "optimization_gauge": 0.20,
    },
    {"name": "joint_one_shot_q20", "mode": "joint_latent_vsq", "gauge": 0.20},
    {"name": "joint_q00", "mode": "joint_latent_vsq", "gauge": 0.00},
    {"name": "joint_q05", "mode": "joint_latent_vsq", "gauge": 0.05},
    {"name": "joint_q10", "mode": "joint_latent_vsq", "gauge": 0.10},
    {
        "name": "joint_q05_anchor10",
        "mode": "joint_latent_vsq",
        "gauge": 0.05,
        "anchor": 0.10,
    },
    {
        "name": "joint_q05_smooth1",
        "mode": "joint_latent_vsq",
        "gauge": 0.05,
        "smooth": 1.0,
    },
    {
        "name": "joint_q05_prior1_smooth1",
        "mode": "joint_latent_vsq",
        "gauge": 0.05,
        "prior": 1.0,
        "smooth": 1.0,
    },
    {
        "name": "joint_q05_huber_regularized",
        "mode": "joint_latent_vsq",
        "gauge": 0.05,
        "anchor": 0.10,
        "prior": 1.0,
        "smooth": 1.0,
        "common": "huber",
    },
    {
        "name": "joint_q05_regularized_blend50",
        "mode": "joint_latent_vsq",
        "gauge": 0.05,
        "anchor": 0.10,
        "prior": 1.0,
        "smooth": 1.0,
        "blend": 0.50,
    },
    {
        "name": "joint_q10_regularized_blend75",
        "mode": "joint_latent_vsq",
        "gauge": 0.10,
        "anchor": 0.10,
        "prior": 0.50,
        "smooth": 0.50,
        "blend": 0.75,
    },
)


def _jaccard(left: set[frozenset[int]], right: set[frozenset[int]]) -> float:
    """Return Jaccard similarity for two rooted-clade sets."""

    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


def _pseudo_validation(
    pseudo_scenarios: list[dict],
    root_bus: int,
    full_clades: set[frozenset[int]],
) -> tuple[float, float]:
    """Return label-free leave-one-day prediction error and clade stability."""

    errors = []
    agreements = []
    if len(pseudo_scenarios) < 2:
        return float("nan"), 1.0
    for heldout in range(len(pseudo_scenarios)):
        training = [
            scenario
            for index, scenario in enumerate(pseudo_scenarios)
            if index != heldout
        ]
        validation = [pseudo_scenarios[heldout]]
        errors.append(_predictive_nrmse(training, validation, RECIPE))
        fold = fit_complete_rnj_baseline(
            training,
            root_bus,
            preprocessing="daily_demean",
            distance_mode="RX_75R_25X",
            constraint_mode="ordered",
            tolerance_factor=0.16,
        )
        agreements.append(_jaccard(set(fold.rooted_clades), full_clades))
    return float(np.mean(errors)), float(np.mean(agreements))


def _fit_options(config: dict) -> dict:
    """Translate compact experiment settings to solver options."""

    return {
        "gauge_quantile": float(config.get("gauge", 0.20)),
        "anchor_strength": float(config.get("anchor", 0.0)),
        "prior_weight": float(config.get("prior", 0.0)),
        "smoothness_weight": float(config.get("smooth", 0.0)),
        "common_mode_method": str(config.get("common", "mean")),
    }


def _build_pseudo(
    scenarios: list[dict],
    terminals: list[int],
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    clusters: list,
    config: dict,
) -> tuple[list[dict], dict, dict]:
    """Build pseudo measurements for one voltage estimator configuration."""

    weight = float(config.get("weight", 0.15))
    if config["mode"] == "deembedded_vsq":
        pseudo, members = aggregate_rooted_scenarios(
            scenarios,
            terminals,
            r_matrix,
            x_matrix,
            clusters,
            "deembedded_vsq",
            deembedding_weight=weight,
        )
        return pseudo, members, {}
    pseudo, members, fits = aggregate_rooted_scenarios_joint(
        scenarios,
        terminals,
        r_matrix,
        x_matrix,
        clusters,
        fit_options=_fit_options(config),
        blend_with_deembedded=float(config.get("blend", 1.0)),
        deembedding_weight=weight,
    )
    return pseudo, members, fits


def run_case(case_key: str, bootstrap_replicates: int, seed: int) -> list[dict]:
    """Evaluate one parameter grid on one standard 3x96 noisy AC case."""

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
    proposals = propose_peripheral_clusters(
        scenarios,
        net.root_bus,
        preprocessing_recipe=RECIPE,
        distance_mode="RX_75R_25X",
        bootstrap_replicates=bootstrap_replicates,
        maximum_cluster_size=5,
        minimum_support=0.70,
        minimum_boundary_margin=0.0,
        minimum_edge_length_ratio=0.0,
        block_fraction=0.50,
        pq_extra_noise_rel=0.0025,
        voltage_extra_noise_rel=0.0001,
        seed=seed,
    )
    clusters = list(proposals.clusters_by_mode.get("nj_rg_consensus", ()))
    if not clusters:
        precision, recall, f1 = _set_score(set(base.rooted_clades), truth)
        return [
            {
                "case": case_key,
                "method": "no_consensus_cluster",
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "exact": set(base.rooted_clades) == truth,
                "cluster_count": 0,
            }
        ]

    rows = []
    r_values = base.r_matrix.to_numpy(dtype=float)
    x_values = base.x_matrix.to_numpy(dtype=float)
    for config in CONFIGS:
        result = run_ordered_two_level_aggregation(
            scenarios,
            net.root_bus,
            clusters,
            pseudo_voltage_mode=config["mode"],
            deembedding_weight=float(config.get("weight", 0.15)),
            joint_fit_options=(
                _fit_options(config)
                if config["mode"] == "joint_latent_vsq"
                else None
            ),
            joint_blend_with_deembedded=float(config.get("blend", 1.0)),
        )
        prediction = set(
            result.candidate_clades_by_mode["ordered_aggregation_hybrid_local"]
        )
        precision, recall, f1 = _set_score(prediction, truth)
        pseudo, pseudo_members, fits = _build_pseudo(
            scenarios,
            terminals,
            r_values,
            x_values,
            clusters,
            config,
        )
        predictive_nrmse, clade_stability = _pseudo_validation(
            pseudo,
            net.root_bus,
            set(result.pseudo.rooted_clades),
        )
        joint_residual = (
            float(np.mean([fit.residual_nrmse for fit in fits.values()]))
            if fits
            else float("nan")
        )
        complexity = (
            0.01 * float(config.get("blend", 1.0))
            if config["mode"] == "joint_latent_vsq"
            else 0.0
        )
        local_nrmse = (
            result.mean_local_nrmse
            if np.isfinite(result.mean_local_nrmse)
            else 1.0
        )
        selection_score = (
            predictive_nrmse
            + 0.15 * (1.0 - clade_stability)
            + 0.05 * local_nrmse
            + complexity
        )
        rows.append(
            {
                "case": case_key,
                "method": config["name"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "exact": prediction == truth,
                "cluster_count": len(clusters),
                "all_clusters_true": all(
                    cluster.members in truth for cluster in clusters
                ),
                "pseudo_vsq_rmse": _pseudo_voltage_rmse(
                    net,
                    scenarios,
                    pseudo,
                    pseudo_members,
                    truth,
                ),
                "pseudo_predictive_nrmse": predictive_nrmse,
                "pseudo_clade_stability": clade_stability,
                "mean_local_nrmse": result.mean_local_nrmse,
                "mean_local_stability": result.mean_local_stability,
                "joint_residual_nrmse": joint_residual,
                "condition_improvement": result.condition_improvement,
                "label_free_selection_score": selection_score,
            }
        )
    return rows


def run(
    output_dir: str | Path,
    cases: list[str],
    bootstrap_replicates: int,
    detector_replicates: int = 1,
) -> dict:
    """Run the optimization grid and write auditable selections."""

    if detector_replicates <= 0:
        raise ValueError("detector_replicates must be positive")
    output = ensure_dir(output_dir)
    rows = []
    total = len(cases) * detector_replicates
    completed = 0
    for index, case_key in enumerate(cases, start=1):
        for detector_replicate in range(detector_replicates):
            completed += 1
            detector_seed = (
                20260716 + 1000 * index + 31 * detector_replicate
            )
            print(
                f"[joint-pseudo] {completed}/{total} case={case_key} "
                f"detector_replicate={detector_replicate}",
                flush=True,
            )
            case_rows = run_case(
                case_key,
                bootstrap_replicates,
                seed=detector_seed,
            )
            for row in case_rows:
                row["detector_replicate"] = detector_replicate
                row["detector_seed"] = detector_seed
            rows.extend(case_rows)
            print(
                "  "
                + " ".join(
                    f"{row['method']}={row['f1']:.3f}" for row in case_rows
                ),
                flush=True,
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "parameter_results.csv", index=False)
    selectable = frame[frame["method"] != "no_consensus_cluster"].copy()
    adaptive = (
        selectable.sort_values(
            ["case", "detector_replicate", "label_free_selection_score"]
        )
        .groupby(["case", "detector_replicate"], as_index=False)
        .first()
    )
    adaptive.to_csv(output / "adaptive_selection.csv", index=False)
    summary = (
        selectable.groupby("method", as_index=False)
        .agg(
            evaluated_cases=("case", "size"),
            mean_f1=("f1", "mean"),
            exact_rate=("exact", "mean"),
            mean_pseudo_vsq_rmse=("pseudo_vsq_rmse", "mean"),
            mean_selection_score=("label_free_selection_score", "mean"),
        )
        .sort_values(
            ["mean_f1", "mean_pseudo_vsq_rmse"],
            ascending=[False, True],
        )
    )
    summary.to_csv(output / "method_summary.csv", index=False)
    payload = {
        "conditions": {
            "scenario_count": 3,
            "samples_per_scenario": 96,
            "sample_interval_minutes": 15,
            "power_flow": "radial AC at every sample",
            "pq_noise_relative_std": 0.005,
            "voltage_noise_relative_std": 0.0002,
        },
        "cases": cases,
        "detector_replicates": detector_replicates,
        "adaptive_selection": adaptive.to_dict(orient="records"),
        "method_summary": summary.to_dict(orient="records"),
    }
    write_json(output / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="outputs/joint_pseudo_parent_optimization",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["paper15", "soumalas11", "flynn16", "pengwah18"],
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=6)
    parser.add_argument("--detector-replicates", type=int, default=1)
    args = parser.parse_args()
    run(
        args.output_dir,
        args.cases,
        args.bootstrap_replicates,
        args.detector_replicates,
    )


if __name__ == "__main__":
    main()