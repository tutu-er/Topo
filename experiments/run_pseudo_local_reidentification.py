"""Evaluate one-layer pseudo contraction followed by local RNJ reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.common import selected_scenario_defs
from experiments.run_complete_rooted_topology_f1 import _measured_scenario
from experiments.run_latent_tree_diagnostics import _detailed_scenarios, _distance_and_depth, _set_score
from experiments.run_rooted_hierarchical_aggregation import _bootstrap_confidence, _fit_rnj
from terminal_case33.graph.diagnostics import terminal_sibling_pairs
from terminal_case33.graph.rooted_hierarchy import (
    aggregate_rooted_scenarios,
    build_local_cluster_scenarios,
    expand_pseudo_clades,
    expand_pseudo_clades_with_local_reidentification,
    reidentify_local_clades_from_shared_paths,
    rooted_clades,
    select_peripheral_clusters,
)
from terminal_case33.graph.rooted_neighbor_joining import shared_paths_from_distances
from terminal_case33.utils.io import ensure_dir, write_json


def _evaluate(
    net,
    scenarios: list[dict],
    replicates: int,
    stability_threshold: float,
    tolerance_factor: float,
    local_tolerance_factor: float,
    local_method: str,
    max_cluster_size: int,
    seed: int,
) -> dict:
    terminals = net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"].astype(int).tolist()
    truth = rooted_clades(net.closed_edges(), net.root_bus, terminals)
    r_matrix, x_matrix, base_tree = _fit_rnj(scenarios, terminals, net.root_bus, tolerance_factor)
    base_clades = rooted_clades(base_tree.edges, net.root_bus, terminals)
    base_siblings = terminal_sibling_pairs(base_tree.edges, terminals)
    clade_confidence, sibling_confidence = _bootstrap_confidence(
        scenarios,
        terminals,
        net.root_bus,
        base_clades,
        base_siblings,
        tolerance_factor,
        replicates,
        seed,
    )
    clusters = select_peripheral_clusters(
        terminals,
        clade_confidence,
        sibling_confidence,
        stability_threshold,
        max_cluster_size,
    )
    if not clusters:
        frozen_prediction = base_clades
        local_prediction = base_clades
        local_clade_count = 0
    else:
        pseudo_scenarios, pseudo_members = aggregate_rooted_scenarios(
            scenarios,
            terminals,
            r_matrix,
            x_matrix,
            clusters,
            "deembedded_vsq",
            deembedding_weight=0.25,
        )
        _, _, pseudo_tree = _fit_rnj(
            pseudo_scenarios,
            list(pseudo_members),
            net.root_bus,
            tolerance_factor,
        )
        frozen_prediction = expand_pseudo_clades(
            pseudo_tree.edges,
            net.root_bus,
            pseudo_members,
            clusters,
        )
        if local_method == "voltage_refit":
            local_clades = {}
            for cluster in clusters:
                local_scenarios = build_local_cluster_scenarios(scenarios, pseudo_scenarios, cluster)
                _, _, local_tree = _fit_rnj(
                    local_scenarios,
                    sorted(cluster.members),
                    cluster.pseudo_id,
                    local_tolerance_factor,
                )
                local_clades[cluster.pseudo_id] = rooted_clades(
                    local_tree.edges,
                    cluster.pseudo_id,
                    sorted(cluster.members),
                )
        elif local_method == "distance_reroot":
            distance, depths, _ = _distance_and_depth(r_matrix, x_matrix, "RX_75R_25X")
            shared_paths = shared_paths_from_distances(distance, depths)
            local_clades = reidentify_local_clades_from_shared_paths(
                shared_paths,
                depths,
                terminals,
                clusters,
                local_tolerance_factor,
            )
        else:
            raise ValueError(f"unknown local_method {local_method!r}")
        local_prediction = expand_pseudo_clades_with_local_reidentification(
            pseudo_tree.edges,
            net.root_bus,
            pseudo_members,
            clusters,
            local_clades,
        )
        local_clade_count = sum(len(value) for value in local_clades.values())

    base_score = _set_score(base_clades, truth)
    frozen_score = _set_score(frozen_prediction, truth)
    local_score = _set_score(local_prediction, truth)
    return {
        "local_method": local_method,
        "terminal_count": len(terminals),
        "true_clade_count": len(truth),
        "cluster_count": len(clusters),
        "local_clade_count": local_clade_count,
        "base_f1": base_score[2],
        "frozen_f1": frozen_score[2],
        "local_reidentified_f1": local_score[2],
        "delta_vs_frozen": local_score[2] - frozen_score[2],
        "frozen_exact": frozen_prediction == truth,
        "local_reidentified_exact": local_prediction == truth,
        "frozen_predicted_clade_count": len(frozen_prediction),
        "local_predicted_clade_count": len(local_prediction),
        "local_matched_clade_count": len(local_prediction & truth),
        "local_extra_clade_count": len(local_prediction - truth),
        "local_missing_clade_count": len(truth - local_prediction),
    }


def run(
    output_dir: str | Path = "outputs/pseudo_local_reidentification",
    cases: list[str] | None = None,
    scenario_counts: list[int] | None = None,
    t_counts: list[int] | None = None,
    single_day_count: int = 15,
    single_day_t: int = 288,
    replicates: int = 10,
    stability_threshold: float = 0.9,
    tolerance_factor: float = 0.16,
    local_tolerance_factor: float = 0.56,
    local_method: str = "distance_reroot",
    max_cluster_size: int = 5,
) -> dict:
    """Compare frozen local topology with one-layer pseudo plus local RNJ."""

    out = ensure_dir(output_dir)
    selected_cases = cases or ["paper15", "soumalas11", "flynn16", "pengwah18"]
    selected_counts = scenario_counts or [1, 3, 5]
    selected_t = t_counts or [48, 96, 288, 960]
    volume_rows = []
    for case_index, case in enumerate(selected_cases):
        for scenario_count in selected_counts:
            for t_count in selected_t:
                net, scenarios = _detailed_scenarios(case, scenario_count, t_count, pq_noise_rel=0.005)
                score = _evaluate(
                    net,
                    scenarios,
                    replicates,
                    stability_threshold,
                    tolerance_factor,
                    local_tolerance_factor,
                    local_method,
                    max_cluster_size,
                    20260713 + 10000 * case_index + 100 * scenario_count + t_count,
                )
                volume_rows.append(
                    {"case": case, "scenario_count": scenario_count, "t_count": t_count, **score}
                )
                print(
                    f"[pseudo-local-volume] case={case} scenarios={scenario_count} T={t_count} "
                    f"frozen={score['frozen_f1']:.3f} local={score['local_reidentified_f1']:.3f}",
                    flush=True,
                )

    day_rows = []
    definitions = selected_scenario_defs()[:single_day_count]
    for case_index, case in enumerate(selected_cases):
        for day_index, definition in enumerate(definitions):
            net, scenarios = _measured_scenario(case, definition, single_day_t)
            score = _evaluate(
                net,
                scenarios,
                replicates,
                stability_threshold,
                tolerance_factor,
                local_tolerance_factor,
                local_method,
                max_cluster_size,
                20260714 + 10000 * case_index + 100 * day_index,
            )
            day_name = str(definition["name"]).replace("oneday_960pts_", f"oneday_{single_day_t}pts_")
            day_rows.append(
                {
                    "case": case,
                    "day": day_name,
                    "profile_scenario": definition["profile_scenario"],
                    **score,
                }
            )
            print(
                f"[pseudo-local-day] case={case} day={day_name} "
                f"frozen={score['frozen_f1']:.3f} local={score['local_reidentified_f1']:.3f}",
                flush=True,
            )

    volume = pd.DataFrame(volume_rows)
    days = pd.DataFrame(day_rows)
    volume.to_csv(out / "volume_results.csv", index=False)
    days.to_csv(out / "independent_day_results.csv", index=False)
    volume_summary = (
        volume.groupby(["scenario_count", "t_count"], as_index=False)
        .agg(
            frozen_mean_f1=("frozen_f1", "mean"),
            local_mean_f1=("local_reidentified_f1", "mean"),
            mean_delta=("delta_vs_frozen", "mean"),
            frozen_exact_rate=("frozen_exact", "mean"),
            local_exact_rate=("local_reidentified_exact", "mean"),
        )
    )
    volume_summary.to_csv(out / "volume_summary.csv", index=False)
    day_summary = (
        days.groupby("case", as_index=False)
        .agg(
            frozen_mean_f1=("frozen_f1", "mean"),
            local_mean_f1=("local_reidentified_f1", "mean"),
            mean_delta=("delta_vs_frozen", "mean"),
            frozen_exact_rate=("frozen_exact", "mean"),
            local_exact_rate=("local_reidentified_exact", "mean"),
            local_min_f1=("local_reidentified_f1", "min"),
        )
    )
    day_summary.to_csv(out / "independent_day_summary.csv", index=False)
    metrics = {
        "local_method": local_method,
        "local_tolerance_factor": local_tolerance_factor,
        "volume_runs": int(len(volume)),
        "independent_day_runs": int(len(days)),
        "volume_frozen_mean_f1": float(volume["frozen_f1"].mean()),
        "volume_local_mean_f1": float(volume["local_reidentified_f1"].mean()),
        "day_frozen_mean_f1": float(days["frozen_f1"].mean()),
        "day_local_mean_f1": float(days["local_reidentified_f1"].mean()),
        "day_frozen_exact_rate": float(days["frozen_exact"].mean()),
        "day_local_exact_rate": float(days["local_reidentified_exact"].mean()),
    }
    write_json(out / "metrics.json", metrics)
    (out / "report.md").write_text(
        "# One-layer pseudo plus local re-identification\n\n"
        "Stable clades are contracted once. The pseudo backbone is reconstructed, then each cluster "
        f"is independently reconstructed with method={local_method} and local tolerance "
        f"factor={local_tolerance_factor}.\n\n"
        "## Volume sweep\n\n"
        + volume_summary.to_string(index=False)
        + "\n\n## Independent days\n\n"
        + day_summary.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/pseudo_local_reidentification")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--scenario-counts", nargs="*", type=int, default=None)
    parser.add_argument("--T-counts", nargs="*", type=int, default=None)
    parser.add_argument("--single-day-count", type=int, default=15)
    parser.add_argument("--single-day-T", type=int, default=288)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--stability-threshold", type=float, default=0.9)
    parser.add_argument("--tolerance-factor", type=float, default=0.16)
    parser.add_argument("--local-tolerance-factor", type=float, default=0.56)
    parser.add_argument("--local-method", choices=["distance_reroot", "voltage_refit"], default="distance_reroot")
    parser.add_argument("--max-cluster-size", type=int, default=5)
    args = parser.parse_args()
    run(
        output_dir=args.output,
        cases=args.cases,
        scenario_counts=args.scenario_counts,
        t_counts=args.T_counts,
        single_day_count=args.single_day_count,
        single_day_t=args.single_day_T,
        replicates=args.replicates,
        stability_threshold=args.stability_threshold,
        tolerance_factor=args.tolerance_factor,
        local_tolerance_factor=args.local_tolerance_factor,
        local_method=args.local_method,
        max_cluster_size=args.max_cluster_size,
    )


if __name__ == "__main__":
    main()

