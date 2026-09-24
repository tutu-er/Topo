"""Compare NJ/RG and aggregate pseudo nodes from stable latent-tree splits."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import simulate_case, terminal_buses_from_scenarios
from terminal_case33.estimation.multiscenario import distance_candidates, fit_projected_sensitivity, preprocess_scenarios
from terminal_case33.estimation.preprocessing import squared_voltage_drop_from_observed_root
from terminal_case33.graph.latent_tree import LatentTreeResult, neighbor_joining, recursive_grouping, terminal_splits
from terminal_case33.models.lin_distflow import build_reduced_sensitivity_matrices, impedance_distance_from_reduced_R
from terminal_case33.utils.io import ensure_dir, write_json


def _recipe(name: str) -> dict:
    recipes = {
        "raw_drop": {"name": "raw_drop", "kind": "raw"},
        "daily_demean": {"name": "daily_demean", "kind": "demean"},
        "first_difference": {"name": "first_difference", "kind": "difference"},
        "rolling_highpass_w288": {"name": "rolling_highpass_w288", "kind": "rolling_highpass", "window": 288},
    }
    if name not in recipes:
        raise ValueError(f"unknown recipe {name!r}; choose one of {sorted(recipes)}")
    return recipes[name]


def _reconstruct(method: str, distance: np.ndarray, nodes: list[int], rg_tolerance: float) -> LatentTreeResult:
    if method == "nj":
        return neighbor_joining(distance, nodes)
    if method == "rg":
        return recursive_grouping(distance, nodes, tolerance=rg_tolerance)
    raise ValueError("method must be 'nj' or 'rg'")


def _fit_distance(scenarios: list[dict], recipe: dict, distance_mode: str, alpha: float) -> tuple[np.ndarray, float, float]:
    fitted = preprocess_scenarios(scenarios, recipe)
    r_matrix, x_matrix, r2_score, condition_number = fit_projected_sensitivity(fitted, alpha=alpha)
    candidates = distance_candidates(r_matrix, x_matrix)
    if distance_mode not in candidates:
        raise ValueError(f"unknown distance mode {distance_mode!r}")
    return candidates[distance_mode], r2_score, condition_number


def _relative_noise(frame: pd.DataFrame, relative_std: float, rng: np.random.Generator) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    positive = np.abs(values[np.abs(values) > 0.0])
    floor = 0.01 * float(np.median(positive)) if positive.size else 1e-12
    scale = relative_std * np.maximum(np.abs(values), floor)
    return pd.DataFrame(rng.normal(0.0, scale, values.shape), index=frame.index, columns=frame.columns)


def _perturb(
    scenarios: list[dict],
    pq_noise: float,
    voltage_noise: float,
    rng: np.random.Generator,
) -> list[dict]:
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


def _score_splits(predicted: set[frozenset[int]], truth: set[frozenset[int]]) -> dict[str, float | int]:
    matched = len(predicted & truth)
    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(truth) if truth else 1.0
    f1_score = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "split_precision": precision,
        "split_recall": recall,
        "split_f1": f1_score,
        "matched_splits": matched,
        "predicted_splits": len(predicted),
        "true_splits": len(truth),
    }


def _split_text(split: frozenset[int]) -> str:
    return " ".join(str(node) for node in sorted(split))


def _bootstrap_split_confidence(
    scenarios: list[dict],
    nodes: list[int],
    method: str,
    base_splits: set[frozenset[int]],
    recipe: dict,
    distance_mode: str,
    replicates: int,
    pq_extra_noise: float,
    voltage_extra_noise: float,
    alpha: float,
    rg_tolerance: float,
    seed: int,
) -> pd.DataFrame:
    counts = Counter({split: 0 for split in base_splits})
    if not counts:
        return pd.DataFrame(columns=["split", "size", "confidence", "members"])
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        sampled = _perturb(scenarios, pq_extra_noise, voltage_extra_noise, rng)
        distance, _, _ = _fit_distance(sampled, recipe, distance_mode, alpha)
        sampled_tree = _reconstruct(method, distance, nodes, rg_tolerance)
        sampled_splits = terminal_splits(sampled_tree.edges, nodes)
        for split in base_splits:
            counts[split] += int(split in sampled_splits)
    records = [
        {
            "split": _split_text(split),
            "size": len(split),
            "confidence": count / max(replicates, 1),
            "members": split,
        }
        for split, count in sorted(counts.items(), key=lambda item: (-item[1], len(item[0]), tuple(sorted(item[0]))))
    ]
    return pd.DataFrame(records, columns=["split", "size", "confidence", "members"])


def _select_disjoint_clusters(confidence: pd.DataFrame, threshold: float, max_size: int) -> list[frozenset[int]]:
    candidates = [
        row.members
        for row in confidence.itertuples(index=False)
        if float(row.confidence) >= threshold and 2 <= int(row.size) <= max_size
    ]
    selected: list[frozenset[int]] = []
    occupied: set[int] = set()
    for cluster in sorted(candidates, key=lambda item: (len(item), tuple(sorted(item)))):
        if occupied.isdisjoint(cluster):
            selected.append(cluster)
            occupied.update(cluster)
    return selected


def _aggregate(scenarios: list[dict], nodes: list[int], clusters: list[frozenset[int]]) -> tuple[list[dict], dict[int, frozenset[int]]]:
    grouped = set().union(*clusters) if clusters else set()
    members = list(clusters) + [frozenset([node]) for node in nodes if node not in grouped]
    pseudo_members = {900000 + index: cluster for index, cluster in enumerate(members)}
    aggregated = []
    for scenario in scenarios:
        p = pd.DataFrame(index=scenario["P_terminal"].index)
        q = pd.DataFrame(index=scenario["Q_terminal"].index)
        voltage_sq = pd.DataFrame(index=scenario["V_terminal"].index)
        for pseudo, cluster in pseudo_members.items():
            children = sorted(cluster)
            p[pseudo] = scenario["P_terminal"][children].sum(axis=1)
            q[pseudo] = scenario["Q_terminal"][children].sum(axis=1)
            voltage_sq[pseudo] = scenario["V_terminal"][children].pow(2).mean(axis=1)
        voltage = np.sqrt(np.maximum(voltage_sq, 1e-12))
        aggregated.append(
            {
                "name": scenario["name"],
                "P_terminal": p,
                "Q_terminal": q,
                "V_terminal": voltage,
                "root_voltage": scenario["root_voltage"],
                "drop_target": squared_voltage_drop_from_observed_root(voltage, scenario["root_voltage"]),
            }
        )
    return aggregated, pseudo_members


def _canonical_split(side: set[int], terminals: set[int]) -> frozenset[int] | None:
    other = terminals - side
    if min(len(side), len(other)) <= 1:
        return None
    if len(side) > len(other) or (len(side) == len(other) and tuple(sorted(side)) > tuple(sorted(other))):
        side = other
    return frozenset(side)


def _expanded_splits(
    base_splits: set[frozenset[int]],
    clusters: list[frozenset[int]],
    pseudo_tree: LatentTreeResult,
    pseudo_members: dict[int, frozenset[int]],
    terminals: list[int],
) -> set[frozenset[int]]:
    full_set = set(terminals)
    expanded = set(clusters)
    expanded.update(split for split in base_splits if any(split < cluster for cluster in clusters))
    for pseudo_split in terminal_splits(pseudo_tree.edges, list(pseudo_members)):
        side = set().union(*(pseudo_members[node] for node in pseudo_split))
        split = _canonical_split(side, full_set)
        if split is not None:
            expanded.add(split)
    return expanded


def run(
    output_dir: str | Path = "outputs/latent_tree_stability_aggregation",
    cases: list[str] | None = None,
    methods: list[str] | None = None,
    scenario_count: int = 1,
    t_count: int = 96,
    recipe_name: str = "raw_drop",
    distance_mode: str = "RX_75R_25X",
    replicates: int = 40,
    thresholds: list[float] | None = None,
    max_cluster_size: int = 5,
    pq_noise_rel: float = 0.005,
    pq_extra_noise_rel: float = 0.0025,
    voltage_extra_noise_rel: float = 0.0001,
    root_voltage_mean: float = 1.02,
    alpha: float = 0.0,
    rg_tolerance: float = 0.03,
    seed: int = 20260710,
) -> pd.DataFrame:
    """Run NJ/RG reconstruction and consensus-stable split aggregation."""

    out_dir = ensure_dir(output_dir)
    selected_cases = cases or ["paper15", "soumalas11", "flynn16", "pengwah18"]
    selected_methods = methods or ["nj", "rg"]
    selected_thresholds = thresholds or [0.6, 0.75, 0.9]
    recipe = _recipe(recipe_name)
    rows = []
    for case_index, case_key in enumerate(selected_cases):
        net, scenarios = simulate_case(case_key, scenario_count, t_count, pq_noise_rel, root_voltage_mean)
        nodes = terminal_buses_from_scenarios(scenarios)
        true_splits = terminal_splits(net.closed_edges(), nodes)
        r_true, _ = build_reduced_sensitivity_matrices(net, nodes, voltage_model="squared-voltage")
        true_distance = impedance_distance_from_reduced_R(r_true)
        measured_distance, r2_score, condition_number = _fit_distance(scenarios, recipe, distance_mode, alpha)
        case_dir = ensure_dir(out_dir / case_key)
        for method_index, method in enumerate(selected_methods):
            oracle_tree = _reconstruct(method, true_distance, nodes, 1e-8)
            oracle_score = _score_splits(terminal_splits(oracle_tree.edges, nodes), true_splits)
            base_tree = _reconstruct(method, measured_distance, nodes, rg_tolerance)
            base_splits = terminal_splits(base_tree.edges, nodes)
            base_score = _score_splits(base_splits, true_splits)
            other_method = "rg" if method == "nj" else "nj"
            other_tree = _reconstruct(other_method, measured_distance, nodes, rg_tolerance)
            shared_splits = base_splits & terminal_splits(other_tree.edges, nodes)
            confidence = _bootstrap_split_confidence(
                scenarios,
                nodes,
                method,
                shared_splits,
                recipe,
                distance_mode,
                replicates,
                pq_extra_noise_rel,
                voltage_extra_noise_rel,
                alpha,
                rg_tolerance,
                seed + 1000 * case_index + 100 * method_index,
            )
            other_confidence = _bootstrap_split_confidence(
                scenarios,
                nodes,
                other_method,
                shared_splits,
                recipe,
                distance_mode,
                replicates,
                pq_extra_noise_rel,
                voltage_extra_noise_rel,
                alpha,
                rg_tolerance,
                seed + 1000 * case_index + 100 * method_index + 50,
            )
            other_by_split = dict(zip(other_confidence["members"], other_confidence["confidence"]))
            confidence["method_confidence"] = confidence["confidence"]
            confidence["other_method_confidence"] = confidence["members"].map(other_by_split)
            confidence["confidence"] = confidence[["method_confidence", "other_method_confidence"]].min(axis=1)
            confidence["is_true_split"] = confidence["members"].map(lambda split: split in true_splits)
            export_confidence = confidence.drop(columns="members")
            export_confidence.to_csv(case_dir / f"{method}_consensus_split_confidence.csv", index=False)
            pd.DataFrame(base_tree.edges, columns=["u", "v", "length"]).to_csv(case_dir / f"{method}_latent_edges.csv", index=False)
            for threshold in selected_thresholds:
                clusters = _select_disjoint_clusters(confidence, threshold, max_cluster_size)
                if clusters:
                    pseudo_scenarios, pseudo_members = _aggregate(scenarios, nodes, clusters)
                    pseudo_nodes = list(pseudo_members)
                    pseudo_distance, pseudo_r2, pseudo_condition = _fit_distance(pseudo_scenarios, recipe, distance_mode, alpha)
                    pseudo_tree = _reconstruct(method, pseudo_distance, pseudo_nodes, rg_tolerance)
                    combined_splits = _expanded_splits(base_splits, clusters, pseudo_tree, pseudo_members, nodes)
                else:
                    pseudo_r2, pseudo_condition = r2_score, condition_number
                    combined_splits = base_splits
                aggregate_score = _score_splits(combined_splits, true_splits)
                stable_clusters_true = sum(cluster in true_splits for cluster in clusters)
                rows.append(
                    {
                        "case": case_key,
                        "method": method,
                        "scenario_count": scenario_count,
                        "t_count": t_count,
                        "threshold": threshold,
                        "oracle_split_f1": oracle_score["split_f1"],
                        "baseline_split_f1": base_score["split_f1"],
                        "aggregated_split_f1": aggregate_score["split_f1"],
                        "delta_split_f1": aggregate_score["split_f1"] - base_score["split_f1"],
                        "baseline_precision": base_score["split_precision"],
                        "baseline_recall": base_score["split_recall"],
                        "aggregated_precision": aggregate_score["split_precision"],
                        "aggregated_recall": aggregate_score["split_recall"],
                        "cluster_count": len(clusters),
                        "clustered_terminal_count": len(set().union(*clusters)) if clusters else 0,
                        "stable_cluster_precision": stable_clusters_true / len(clusters) if clusters else np.nan,
                        "base_forced_merges": base_tree.forced_merges,
                        "r2_score": r2_score,
                        "condition_number": condition_number,
                        "pseudo_r2_score": pseudo_r2,
                        "pseudo_condition_number": pseudo_condition,
                    }
                )
            print(
                f"[latent] case={case_key} method={method} oracle={oracle_score['split_f1']:.3f} "
                f"baseline={base_score['split_f1']:.3f}",
                flush=True,
            )

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "summary.csv", index=False)
    retrospective_best = (
        result.sort_values(["aggregated_split_f1", "threshold"], ascending=[False, False])
        .groupby(["case", "method"])
        .head(1)
    )
    retrospective_best.to_csv(out_dir / "retrospective_best_by_case_method.csv", index=False)
    fixed_threshold = max(selected_thresholds)
    fixed = result[result["threshold"].eq(fixed_threshold)].copy()
    fixed.to_csv(out_dir / "fixed_threshold_summary.csv", index=False)
    write_json(
        out_dir / "metrics.json",
        {
            "cases": selected_cases,
            "methods": selected_methods,
            "replicates": replicates,
            "fixed_threshold": fixed_threshold,
            "measurement_noise": {"P_Q_relative": pq_noise_rel, "V_relative": 0.0002},
            "extra_stability_noise": {"P_Q_relative": pq_extra_noise_rel, "V_relative": voltage_extra_noise_rel},
            "mean_baseline_split_f1_fixed": float(fixed["baseline_split_f1"].mean()),
            "mean_aggregated_split_f1_fixed": float(fixed["aggregated_split_f1"].mean()),
            "mean_aggregated_split_f1_retrospective_best": float(retrospective_best["aggregated_split_f1"].mean()),
        },
    )
    (out_dir / "report.md").write_text(
        "# NJ/RG Latent-Tree Stability Aggregation\n\n"
        "Topology is evaluated by nontrivial terminal splits. Stable pseudo clusters are small, disjoint "
        "split sides shared by NJ and RG and persistent under extra perturbation. Pseudo voltage is the "
        "mean squared terminal voltage, so aggregation remains an approximation.\n\n"
        "The fixed-threshold table is deployable without topology labels. The retrospective-best table "
        "uses true split F1 to choose a threshold and is diagnostic only.\n\n"
        "## Fixed threshold\n\n"
        + fixed.to_string(index=False)
        + "\n\n## Retrospective best\n\n"
        + retrospective_best.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/latent_tree_stability_aggregation")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--scenario-count", type=int, default=1)
    parser.add_argument("--T", type=int, default=96)
    parser.add_argument("--recipe", default="raw_drop")
    parser.add_argument("--distance-mode", default="RX_75R_25X")
    parser.add_argument("--replicates", type=int, default=40)
    parser.add_argument("--thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--max-cluster-size", type=int, default=5)
    parser.add_argument("--pq-noise-rel", type=float, default=0.005)
    parser.add_argument("--pq-extra-noise-rel", type=float, default=0.0025)
    parser.add_argument("--voltage-extra-noise-rel", type=float, default=0.0001)
    parser.add_argument("--root-voltage-mean", type=float, default=1.02)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--rg-tolerance", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()
    run(
        output_dir=args.output,
        cases=args.cases,
        methods=args.methods,
        scenario_count=args.scenario_count,
        t_count=args.T,
        recipe_name=args.recipe,
        distance_mode=args.distance_mode,
        replicates=args.replicates,
        thresholds=args.thresholds,
        max_cluster_size=args.max_cluster_size,
        pq_noise_rel=args.pq_noise_rel,
        pq_extra_noise_rel=args.pq_extra_noise_rel,
        voltage_extra_noise_rel=args.voltage_extra_noise_rel,
        root_voltage_mean=args.root_voltage_mean,
        alpha=args.alpha,
        rg_tolerance=args.rg_tolerance,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
