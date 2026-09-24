"""Complete rooted-clade F1 evaluation for the hierarchical RNJ pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.common import (
    resource_assignment,
    selected_scenario_defs,
    simulate_case_scenario,
)
from experiments.run_latent_tree_diagnostics import _detailed_scenarios, _set_score
from experiments.run_rooted_hierarchical_aggregation import _bootstrap_confidence, _fit_rnj
from terminal_case33.data.paper_style_case_bank import CASE_BUILDERS
from terminal_case33.graph.rooted_hierarchy import (
    aggregate_rooted_scenarios,
    expand_pseudo_clades,
    rooted_clades,
    select_peripheral_clusters,
)
from terminal_case33.graph.diagnostics import terminal_sibling_pairs
from terminal_case33.utils.io import ensure_dir, write_json


def _measured_scenario(case_key: str, definition: dict, t_count: int) -> tuple[object, list[dict]]:
    net = CASE_BUILDERS[case_key]()
    simulation = simulate_case_scenario(
        net=net,
        assignment=resource_assignment(case_key, net),
        T=t_count,
        seed=int(definition["seed"]),
        pq_noise_rel=0.005,
        v_noise_rel=float(definition["v_noise_rel"]),
        root_voltage_mean=1.02,
        root_voltage_sigma=float(definition["root_voltage_sigma"]),
        profile_scenario=str(definition["profile_scenario"]),
    )
    return net, [
        {
            "name": str(definition["name"]),
            "P_terminal": simulation["P_terminal"],
            "Q_terminal": simulation["Q_terminal"],
            "V_terminal": simulation["V_terminal"],
            "root_voltage": simulation["root_voltage"],
            "drop_target": simulation["drop_target"],
        }
    ]


def _evaluate(
    net,
    scenarios: list[dict],
    replicates: int,
    stability_threshold: float,
    tolerance_factor: float,
    max_cluster_size: int,
    seed: int,
) -> dict:
    terminals = net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"].astype(int).tolist()
    true_clades = rooted_clades(net.closed_edges(), net.root_bus, terminals)
    terminal_sibling_pairs(net.closed_edges(), terminals)
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
    if clusters:
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
        predicted_clades = expand_pseudo_clades(
            pseudo_tree.edges,
            net.root_bus,
            pseudo_members,
            clusters,
        )
    else:
        predicted_clades = base_clades

    precision, recall, f1_score = _set_score(predicted_clades, true_clades)
    matched = len(predicted_clades & true_clades)
    frozen_clades = {clade for cluster in clusters for clade in cluster.frozen_clades}
    return {
        "terminal_count": len(terminals),
        "true_clade_count": len(true_clades),
        "predicted_clade_count": len(predicted_clades),
        "matched_clade_count": matched,
        "clade_precision": precision,
        "clade_recall": recall,
        "topology_f1": f1_score,
        "exact_topology_recovery": predicted_clades == true_clades,
        "cluster_count": len(clusters),
        "clustered_terminal_count": len(set().union(*(cluster.members for cluster in clusters))) if clusters else 0,
        "selected_cluster_true_rate": (
            sum(cluster.members in true_clades for cluster in clusters) / len(clusters) if clusters else 1.0
        ),
        "frozen_clade_count": len(frozen_clades),
        "frozen_clades_preserved": frozen_clades <= predicted_clades,
    }


def run(
    output_dir: str | Path = "outputs/complete_rooted_topology_f1",
    cases: list[str] | None = None,
    scenario_counts: list[int] | None = None,
    t_counts: list[int] | None = None,
    single_day_count: int = 15,
    single_day_t: int = 288,
    replicates: int = 10,
    stability_threshold: float = 0.9,
    tolerance_factor: float = 0.16,
    max_cluster_size: int = 5,
) -> dict:
    """Run data-volume and independent-single-day rooted topology F1 tests."""

    out_dir = ensure_dir(output_dir)
    selected_cases = cases or ["paper15", "soumalas11", "flynn16", "pengwah18"]
    selected_scenarios = scenario_counts or [1, 3, 5]
    selected_t = t_counts or [48, 96, 288, 960]
    volume_rows = []
    task = 0
    for case_index, case_key in enumerate(selected_cases):
        for scenario_count in selected_scenarios:
            for t_count in selected_t:
                task += 1
                net, scenarios = _detailed_scenarios(case_key, scenario_count, t_count, pq_noise_rel=0.005)
                score = _evaluate(
                    net,
                    scenarios,
                    replicates,
                    stability_threshold,
                    tolerance_factor,
                    max_cluster_size,
                    seed=20260711 + 10000 * case_index + 100 * scenario_count + t_count,
                )
                volume_rows.append(
                    {
                        "case": case_key,
                        "scenario_count": scenario_count,
                        "t_count": t_count,
                        "sample_interval_minutes": 1440.0 / t_count,
                        **score,
                    }
                )
                print(
                    f"[complete-volume] {task} case={case_key} scenarios={scenario_count} "
                    f"T={t_count} F1={score['topology_f1']:.3f}",
                    flush=True,
                )

    definitions = selected_scenario_defs()[:single_day_count]
    day_rows = []
    task = 0
    for case_index, case_key in enumerate(selected_cases):
        for day_index, definition in enumerate(definitions):
            task += 1
            source_day = str(definition["name"])
            day_label = source_day.replace("oneday_960pts_", f"oneday_{single_day_t}pts_")
            net, scenarios = _measured_scenario(case_key, definition, single_day_t)
            score = _evaluate(
                net,
                scenarios,
                replicates,
                stability_threshold,
                tolerance_factor,
                max_cluster_size,
                seed=20260712 + 10000 * case_index + 100 * day_index,
            )
            day_rows.append(
                {
                    "case": case_key,
                    "day": day_label,
                    "source_profile_definition": source_day,
                    "profile_scenario": str(definition["profile_scenario"]),
                    "seed": int(definition["seed"]),
                    "t_count": single_day_t,
                    "sample_interval_minutes": 1440.0 / single_day_t,
                    **score,
                }
            )
            print(
                f"[complete-day] {task} case={case_key} day={day_label} "
                f"F1={score['topology_f1']:.3f}",
                flush=True,
            )

    volume = pd.DataFrame(volume_rows)
    days = pd.DataFrame(day_rows)
    volume.to_csv(out_dir / "volume_results.csv", index=False)
    days.to_csv(out_dir / "independent_single_day_results.csv", index=False)
    case_topology = (
        volume.groupby("case", as_index=False)
        .agg(
            terminal_count=("terminal_count", "first"),
            true_clade_count=("true_clade_count", "first"),
        )
    )
    case_topology.to_csv(out_dir / "case_topology_summary.csv", index=False)
    volume_summary = (
        volume.groupby(["scenario_count", "t_count"], as_index=False)
        .agg(
            cases=("topology_f1", "size"),
            mean_f1=("topology_f1", "mean"),
            min_f1=("topology_f1", "min"),
            exact_recovery_rate=("exact_topology_recovery", "mean"),
            mean_precision=("clade_precision", "mean"),
            mean_recall=("clade_recall", "mean"),
        )
    )
    volume_summary.to_csv(out_dir / "volume_summary.csv", index=False)
    day_summary = (
        days.groupby("case", as_index=False)
        .agg(
            days=("topology_f1", "size"),
            mean_f1=("topology_f1", "mean"),
            min_f1=("topology_f1", "min"),
            max_f1=("topology_f1", "max"),
            exact_recovery_rate=("exact_topology_recovery", "mean"),
        )
    )
    day_summary.to_csv(out_dir / "independent_single_day_summary.csv", index=False)
    by_profile = (
        days.groupby("profile_scenario", as_index=False)
        .agg(
            tests=("topology_f1", "size"),
            mean_f1=("topology_f1", "mean"),
            min_f1=("topology_f1", "min"),
            exact_recovery_rate=("exact_topology_recovery", "mean"),
        )
    )
    by_profile.to_csv(out_dir / "single_day_summary_by_profile.csv", index=False)
    metrics = {
        "volume_condition_count": int(len(volume)),
        "independent_single_day_condition_count": int(len(days)),
        "single_day_t_count": single_day_t,
        "replicates": replicates,
        "stability_threshold": stability_threshold,
        "tolerance_factor": tolerance_factor,
        "volume_mean_f1": float(volume["topology_f1"].mean()),
        "volume_exact_recovery_rate": float(volume["exact_topology_recovery"].mean()),
        "independent_single_day_mean_f1": float(days["topology_f1"].mean()),
        "independent_single_day_min_f1": float(days["topology_f1"].min()),
        "independent_single_day_exact_recovery_rate": float(days["exact_topology_recovery"].mean()),
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# Complete rooted topology F1\n\n"
        "Topology F1 is precision/recall F1 on all nontrivial true rooted terminal clades. Singleton "
        "service edges and branch impedances are not part of this score.\n\n"
        "## Data-volume sweep\n\n"
        + volume_summary.to_string(index=False)
        + "\n\n## Ground-truth topology size\n\n"
        + case_topology.to_string(index=False)
        + f"\n\n## Independent {single_day_t}-point days by case\n\n"
        + day_summary.to_string(index=False)
        + "\n\n## Independent days by profile\n\n"
        + by_profile.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/complete_rooted_topology_f1")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--scenario-counts", nargs="*", type=int, default=None)
    parser.add_argument("--T-counts", nargs="*", type=int, default=None)
    parser.add_argument("--single-day-count", type=int, default=15)
    parser.add_argument("--single-day-T", type=int, default=288)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--stability-threshold", type=float, default=0.9)
    parser.add_argument("--tolerance-factor", type=float, default=0.16)
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
        max_cluster_size=args.max_cluster_size,
    )


if __name__ == "__main__":
    main()
