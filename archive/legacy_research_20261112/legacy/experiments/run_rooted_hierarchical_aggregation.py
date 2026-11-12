"""Preserve reliable peripheral clades and re-identify only the rooted backbone."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_latent_tree_diagnostics import _detailed_scenarios, _set_score
from terminal_case33.estimation.multiscenario import fit_projected_sensitivity, preprocess_scenarios
from terminal_case33.estimation.preprocessing import squared_voltage_drop_from_observed_root
from terminal_case33.graph.diagnostics import terminal_sibling_pairs
from terminal_case33.graph.rooted_hierarchy import (
    aggregate_rooted_scenarios,
    expand_pseudo_clades,
    expand_pseudo_sibling_pairs,
    rooted_clades,
    select_peripheral_clusters,
)
from terminal_case33.graph.rooted_neighbor_joining import rooted_neighbor_joining
from terminal_case33.graph.sensitivity_geometry import sensitivity_geometry
from terminal_case33.utils.io import ensure_dir, write_json


RECIPE = {"name": "daily_demean", "kind": "demean"}


def _relative_noise(frame: pd.DataFrame, relative_std: float, rng: np.random.Generator) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    positive = np.abs(values[np.abs(values) > 0.0])
    floor = 0.01 * float(np.median(positive)) if positive.size else 1e-12
    scale = relative_std * np.maximum(np.abs(values), floor)
    return pd.DataFrame(rng.normal(0.0, scale, values.shape), index=frame.index, columns=frame.columns)


def _perturb(scenarios: list[dict], rng: np.random.Generator, pq_noise: float, voltage_noise: float) -> list[dict]:
    sampled = []
    for scenario in scenarios:
        p = scenario["P_terminal"] + _relative_noise(scenario["P_terminal"], pq_noise, rng)
        q = scenario["Q_terminal"] + _relative_noise(scenario["Q_terminal"], pq_noise, rng)
        voltage = scenario["V_terminal"] + _relative_noise(scenario["V_terminal"], voltage_noise, rng)
        sampled.append(
            {
                "name": scenario["name"],
                "P_terminal": p,
                "Q_terminal": q,
                "V_terminal": voltage,
                "root_voltage": scenario["root_voltage"],
                "drop_target": squared_voltage_drop_from_observed_root(voltage, scenario["root_voltage"]),
            }
        )
    return sampled


def _fit_rnj(
    scenarios: list[dict],
    terminals: list[int],
    root: int,
    tolerance_factor: float,
) -> tuple[np.ndarray, np.ndarray, object]:
    fitted = preprocess_scenarios(scenarios, RECIPE)
    r_matrix, x_matrix, _, _ = fit_projected_sensitivity(fitted)
    geometry = sensitivity_geometry(r_matrix, x_matrix, "RX_75R_25X")
    tolerance = tolerance_factor * max(float(np.median(geometry.root_depths)), 1e-12)
    return r_matrix, x_matrix, rooted_neighbor_joining(
        geometry.shared_paths,
        geometry.root_depths,
        terminals,
        root,
        tolerance,
    )


def _bootstrap_confidence(
    scenarios: list[dict],
    terminals: list[int],
    root: int,
    base_clades: set[frozenset[int]],
    base_siblings: set[tuple[int, int]],
    tolerance_factor: float,
    replicates: int,
    seed: int,
) -> tuple[dict[frozenset[int], float], dict[tuple[int, int], float]]:
    clade_counts = Counter({clade: 0 for clade in base_clades})
    sibling_counts = Counter({pair: 0 for pair in base_siblings})
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        sampled = _perturb(scenarios, rng, pq_noise=0.0025, voltage_noise=0.0001)
        _, _, tree = _fit_rnj(sampled, terminals, root, tolerance_factor)
        sampled_clades = rooted_clades(tree.edges, root, terminals)
        sampled_siblings = terminal_sibling_pairs(tree.edges, terminals)
        for clade in clade_counts:
            clade_counts[clade] += int(clade in sampled_clades)
        for pair in sibling_counts:
            sibling_counts[pair] += int(pair in sampled_siblings)
    denominator = max(replicates, 1)
    return (
        {clade: count / denominator for clade, count in clade_counts.items()},
        {pair: count / denominator for pair, count in sibling_counts.items()},
    )


def run(
    output_dir: str | Path = "outputs/rooted_hierarchical_aggregation",
    cases: list[str] | None = None,
    scenario_counts: list[int] | None = None,
    t_counts: list[int] | None = None,
    replicates: int = 20,
    stability_threshold: float = 0.9,
    tolerance_factor: float = 0.16,
    max_cluster_size: int = 5,
) -> dict:
    """Compare mean-voltage and sensitivity-deembedded rooted aggregation."""

    out_dir = ensure_dir(output_dir)
    selected_cases = cases or ["paper15", "soumalas11", "flynn16", "pengwah18"]
    selected_scenarios = scenario_counts or [1, 3, 5]
    selected_t = t_counts or [96, 288]
    rows = []
    cluster_rows = []
    total = len(selected_cases) * len(selected_scenarios) * len(selected_t)
    task = 0
    for case_index, case_key in enumerate(selected_cases):
        for scenario_count in selected_scenarios:
            for t_count in selected_t:
                task += 1
                net, scenarios = _detailed_scenarios(case_key, scenario_count, t_count, pq_noise_rel=0.005)
                terminals = net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"].astype(int).tolist()
                true_clades = rooted_clades(net.closed_edges(), net.root_bus, terminals)
                true_siblings = terminal_sibling_pairs(net.closed_edges(), terminals)
                r_matrix, x_matrix, base_tree = _fit_rnj(
                    scenarios, terminals, net.root_bus, tolerance_factor
                )
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
                    seed=20260711 + 1000 * case_index + 10 * scenario_count + t_count,
                )
                clusters = select_peripheral_clusters(
                    terminals,
                    clade_confidence,
                    sibling_confidence,
                    stability_threshold,
                    max_cluster_size,
                )
                frozen_clades = {clade for cluster in clusters for clade in cluster.frozen_clades}
                frozen_pairs = {pair for cluster in clusters for pair in cluster.frozen_sibling_pairs}
                base_clade_score = _set_score(base_clades, true_clades)
                base_sibling_score = _set_score(base_siblings, true_siblings)
                for cluster in clusters:
                    cluster_rows.append(
                        {
                            "case": case_key,
                            "scenario_count": scenario_count,
                            "t_count": t_count,
                            "pseudo_id": cluster.pseudo_id,
                            "members": " ".join(map(str, sorted(cluster.members))),
                            "size": len(cluster.members),
                            "confidence": cluster.confidence,
                            "frozen_clade_count": len(cluster.frozen_clades),
                            "frozen_sibling_count": len(cluster.frozen_sibling_pairs),
                            "is_true_clade": cluster.members in true_clades,
                        }
                    )

                voltage_configs = [
                    ("mean_vsq", "mean_vsq", 0.0),
                    ("shrink_025", "deembedded_vsq", 0.25),
                    ("shrink_050", "deembedded_vsq", 0.50),
                    ("shrink_075", "deembedded_vsq", 0.75),
                    ("deembedded_vsq", "deembedded_vsq", 1.0),
                ]
                for voltage_label, voltage_mode, deembedding_weight in voltage_configs:
                    if not clusters:
                        expanded_clades = base_clades
                        expanded_siblings = base_siblings
                    else:
                        pseudo_scenarios, pseudo_members = aggregate_rooted_scenarios(
                            scenarios,
                            terminals,
                            r_matrix,
                            x_matrix,
                            clusters,
                            voltage_mode,
                            deembedding_weight,
                        )
                        pseudo_nodes = list(pseudo_members)
                        _, _, pseudo_tree = _fit_rnj(
                            pseudo_scenarios,
                            pseudo_nodes,
                            net.root_bus,
                            tolerance_factor,
                        )
                        expanded_clades = expand_pseudo_clades(
                            pseudo_tree.edges, net.root_bus, pseudo_members, clusters
                        )
                        expanded_siblings = expand_pseudo_sibling_pairs(
                            pseudo_tree.edges, net.root_bus, pseudo_members, clusters
                        )

                    clade_score = _set_score(expanded_clades, true_clades)
                    sibling_score = _set_score(expanded_siblings, true_siblings)
                    true_internal = {
                        clade
                        for clade in true_clades
                        if not any(clade <= cluster.members for cluster in clusters)
                    }
                    predicted_internal = expanded_clades - frozen_clades - {cluster.members for cluster in clusters}
                    internal_score = _set_score(predicted_internal, true_internal)
                    rows.append(
                        {
                            "case": case_key,
                            "scenario_count": scenario_count,
                            "t_count": t_count,
                            "voltage_mode": voltage_label,
                            "deembedding_weight": deembedding_weight,
                            "cluster_count": len(clusters),
                            "clustered_terminal_count": len(set().union(*(cluster.members for cluster in clusters)))
                            if clusters
                            else 0,
                            "base_clade_f1": base_clade_score[2],
                            "base_sibling_f1": base_sibling_score[2],
                            "expanded_clade_f1": clade_score[2],
                            "expanded_sibling_f1": sibling_score[2],
                            "internal_clade_f1": internal_score[2] if true_internal else np.nan,
                            "delta_clade_f1": clade_score[2] - base_clade_score[2],
                            "frozen_clade_count": len(frozen_clades),
                            "frozen_sibling_count": len(frozen_pairs),
                            "frozen_clades_preserved": frozen_clades <= expanded_clades,
                            "frozen_siblings_preserved": frozen_pairs <= expanded_siblings,
                        }
                    )
                print(
                    f"[hierarchy] {task}/{total} case={case_key} scenarios={scenario_count} "
                    f"T={t_count} clusters={len(clusters)}",
                    flush=True,
                )

    result = pd.DataFrame(rows)
    clusters = pd.DataFrame(cluster_rows)
    result.to_csv(out_dir / "hierarchical_results.csv", index=False)
    clusters.to_csv(out_dir / "selected_clusters.csv", index=False)
    summary = (
        result.groupby("voltage_mode", as_index=False)
        .agg(
            runs=("expanded_clade_f1", "size"),
            mean_base_clade_f1=("base_clade_f1", "mean"),
            mean_expanded_clade_f1=("expanded_clade_f1", "mean"),
            mean_internal_clade_f1=("internal_clade_f1", "mean"),
            mean_expanded_sibling_f1=("expanded_sibling_f1", "mean"),
            mean_delta_clade_f1=("delta_clade_f1", "mean"),
            frozen_clade_preservation_rate=("frozen_clades_preserved", "mean"),
            frozen_sibling_preservation_rate=("frozen_siblings_preserved", "mean"),
        )
    )
    summary.to_csv(out_dir / "summary_by_voltage_mode.csv", index=False)
    by_case = (
        result.groupby(["case", "voltage_mode"], as_index=False)
        .agg(
            mean_base_clade_f1=("base_clade_f1", "mean"),
            mean_expanded_clade_f1=("expanded_clade_f1", "mean"),
            mean_internal_clade_f1=("internal_clade_f1", "mean"),
            mean_delta_clade_f1=("delta_clade_f1", "mean"),
        )
    )
    by_case.to_csv(out_dir / "summary_by_case.csv", index=False)
    metrics = {
        "run_count": int(len(result)),
        "selected_cluster_count": int(len(clusters)),
        "true_cluster_rate": float(clusters["is_true_clade"].mean()) if len(clusters) else 1.0,
        "stability_threshold": stability_threshold,
        "tolerance_factor": tolerance_factor,
        "summary": summary.to_dict(orient="records"),
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# Rooted hierarchical aggregation\n\n"
        "Reliable peripheral clades and sibling pairs are frozen before pseudo-level RNJ. Only the "
        "topology above contracted clades is re-identified.\n\n"
        "## Voltage proxy comparison\n\n"
        + summary.to_string(index=False)
        + "\n\n## By case\n\n"
        + by_case.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/rooted_hierarchical_aggregation")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--scenario-counts", nargs="*", type=int, default=None)
    parser.add_argument("--T-counts", nargs="*", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--stability-threshold", type=float, default=0.9)
    parser.add_argument("--tolerance-factor", type=float, default=0.16)
    parser.add_argument("--max-cluster-size", type=int, default=5)
    args = parser.parse_args()
    run(
        output_dir=args.output,
        cases=args.cases,
        scenario_counts=args.scenario_counts,
        t_counts=args.T_counts,
        replicates=args.replicates,
        stability_threshold=args.stability_threshold,
        tolerance_factor=args.tolerance_factor,
        max_cluster_size=args.max_cluster_size,
    )


if __name__ == "__main__":
    main()
