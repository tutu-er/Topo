"""Bootstrap stable hidden edges and test one-level root decoupling."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from experiments.run_edge_detector_then_rnj import (
    _ordered_fit,
    _small_rooted_clades_from_unrooted,
)
from experiments.run_latent_tree_diagnostics import _set_score
from experiments.run_matrix_constraint_large_sweep import _simulate_pool, _terminal_buses
from experiments.run_rooted_hierarchical_aggregation import _perturb
from terminal_case33.data.paper_style_case_bank import CASE_BUILDERS
from terminal_case33.graph.latent_tree import neighbor_joining, recursive_grouping
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.graph.rooted_neighbor_joining import (
    rooted_neighbor_joining,
    shared_paths_from_distances,
)
from terminal_case33.utils.io import ensure_dir, write_json


METHODS = ("nj", "rg", "rnj")


def _reconstruct_clades(
    method: str,
    distance: np.ndarray,
    depth: np.ndarray,
    terminals: list[int],
    root: int,
    rnj_tolerance_factor: float,
    rg_tolerance: float,
) -> tuple[set[frozenset[int]], tuple[tuple[int, int, float], ...]]:
    """Reconstruct rooted terminal clades with NJ, RG, or RNJ."""

    if len(terminals) < 2:
        return set(), tuple()
    if method == "nj":
        tree = neighbor_joining(distance, terminals)
        clades = _small_rooted_clades_from_unrooted(
            tree.edges,
            terminals,
            depth,
            max_cluster_size=max(len(terminals) - 1, 1),
        )
        return clades, tree.edges
    if method == "rg":
        augmented = np.zeros((len(terminals) + 1, len(terminals) + 1), dtype=float)
        augmented[:-1, :-1] = distance
        augmented[:-1, -1] = depth
        augmented[-1, :-1] = depth
        tree = recursive_grouping(augmented, [*terminals, root], tolerance=rg_tolerance)
        return rooted_clades(tree.edges, root, terminals), tree.edges
    if method == "rnj":
        tolerance = rnj_tolerance_factor * max(float(np.median(depth)), 1e-12)
        tree = rooted_neighbor_joining(
            shared_paths_from_distances(distance, depth),
            depth,
            terminals,
            root,
            tolerance,
        )
        return rooted_clades(tree.edges, root, terminals), tree.edges
    raise ValueError(f"unknown reconstruction method {method!r}")


def _root_branch_groups(
    edges: tuple[tuple[int, int, float], ...],
    root: int,
    terminals: list[int],
) -> set[frozenset[int]]:
    """Return terminal groups below each edge incident to the known root."""

    graph = nx.Graph()
    graph.add_edges_from((int(left), int(right)) for left, right, _ in edges)
    terminal_set = set(terminals)
    if root not in graph:
        return set()
    groups = set()
    for neighbor in list(graph.neighbors(root)):
        graph.remove_edge(root, neighbor)
        group = frozenset(terminal_set & nx.node_connected_component(graph, neighbor))
        graph.add_edge(root, neighbor)
        if group:
            groups.add(group)
    return groups


def _decouple_global_root_mode(
    distance: np.ndarray,
    depth: np.ndarray,
    terminals: list[int],
    root: int,
    rnj_tolerance_factor: float,
    quantile: float,
) -> tuple[float, np.ndarray, set[frozenset[int]], tuple[tuple[int, int, float], ...]]:
    """Subtract the common root-path mode and rerun RNJ at the physical root."""

    shared = shared_paths_from_distances(distance, depth)
    off_diagonal = shared[np.triu_indices(len(terminals), 1)]
    root_shift = float(np.quantile(off_diagonal, quantile)) if off_diagonal.size else 0.0
    root_shift = max(root_shift, 0.0)
    adjusted_depth = np.maximum(depth - root_shift, 0.0)
    adjusted_shared = np.maximum(shared - root_shift, 0.0)
    np.fill_diagonal(adjusted_shared, adjusted_depth)
    # Preserve the absolute impedance-length resolution after removing a root layer.
    tolerance = rnj_tolerance_factor * max(float(np.median(depth)), 1e-12)
    tree = rooted_neighbor_joining(
        adjusted_shared,
        adjusted_depth,
        terminals,
        root,
        tolerance,
    )
    return (
        root_shift,
        adjusted_depth,
        _root_branch_groups(tree.edges, root, terminals),
        tree.edges,
    )


def _root_decoupled_clades(
    local_method: str,
    distance: np.ndarray,
    depth: np.ndarray,
    terminals: list[int],
    root_groups: set[frozenset[int]],
    rnj_tolerance_factor: float,
    rg_tolerance: float,
) -> set[frozenset[int]]:
    """Remove one root layer and reconstruct each downstream branch separately."""

    position = {terminal: index for index, terminal in enumerate(terminals)}
    shared = shared_paths_from_distances(distance, depth)
    predicted: set[frozenset[int]] = set()
    for group_index, group in enumerate(sorted(root_groups, key=lambda item: tuple(sorted(item)))):
        members = sorted(group)
        if 1 < len(group) < len(terminals):
            predicted.add(group)
        if len(members) < 3:
            continue
        indices = [position[terminal] for terminal in members]
        local_distance = distance[np.ix_(indices, indices)]
        local_shared = shared[np.ix_(indices, indices)]
        off_diagonal = local_shared[np.triu_indices(len(indices), 1)]
        boundary_depth = float(np.quantile(off_diagonal, 0.1)) if off_diagonal.size else 0.0
        local_depth = np.maximum(depth[indices] - boundary_depth, 0.0)
        global_tolerance = rnj_tolerance_factor * max(float(np.median(depth)), 1e-12)
        local_tolerance_factor = global_tolerance / max(float(np.median(local_depth)), 1e-12)
        local_root = -2000000 - group_index
        local_clades, _ = _reconstruct_clades(
            local_method,
            local_distance,
            local_depth,
            members,
            local_root,
            local_tolerance_factor,
            rg_tolerance,
        )
        predicted.update(local_clades)
    return predicted


def _clade_text(clade: frozenset[int]) -> str:
    return ",".join(map(str, sorted(clade)))


def run(
    output_dir: str | Path = "outputs/probabilistic_root_decoupling",
    cases: list[str] | None = None,
    scenario_counts: list[int] | None = None,
    t_counts: list[int] | None = None,
    data_replicates: int = 1,
    bootstrap_replicates: int = 8,
    stability_thresholds: list[float] | None = None,
    rnj_tolerance_factor: float = 0.16,
    rg_tolerance: float = 0.03,
    pq_noise_rel: float = 0.005,
    v_noise_rel: float = 0.0002,
    extra_pq_noise_rel: float = 0.0025,
    extra_v_noise_rel: float = 0.0001,
    root_mode_quantile: float = 0.10,
) -> dict:
    """Benchmark reconstruction, bootstrap stable edges, and root decoupling."""

    out_dir = ensure_dir(output_dir)
    selected_cases = cases or list(CASE_BUILDERS)
    selected_scenarios = scenario_counts or [3, 5]
    selected_t = t_counts or [48, 96]
    thresholds = stability_thresholds or [0.75, 0.875]
    maximum_scenarios = max(selected_scenarios)
    topology_rows: list[dict] = []
    probability_rows: list[dict] = []
    stable_rows: list[dict] = []
    condition_count = len(selected_cases) * len(selected_t) * data_replicates * len(selected_scenarios)
    completed = 0

    for case_index, case_key in enumerate(selected_cases):
        for t_count in selected_t:
            for data_replicate in range(data_replicates):
                net, scenario_pool = _simulate_pool(
                    case_key,
                    t_count,
                    data_replicate,
                    maximum_scenarios,
                    pq_noise_rel,
                    v_noise_rel,
                )
                terminals = _terminal_buses(net)
                true_clades = rooted_clades(net.closed_edges(), net.root_bus, terminals)
                for scenario_count in selected_scenarios:
                    scenarios = scenario_pool[:scenario_count]
                    _, _, distance, depth = _ordered_fit(scenarios)
                    base_clades: dict[str, set[frozenset[int]]] = {}
                    base_edges: dict[str, tuple[tuple[int, int, float], ...]] = {}
                    for method in METHODS:
                        base_clades[method], base_edges[method] = _reconstruct_clades(
                            method,
                            distance,
                            depth,
                            terminals,
                            net.root_bus,
                            rnj_tolerance_factor,
                            rg_tolerance,
                        )

                    base_root_groups = _root_branch_groups(
                        base_edges["rnj"],
                        net.root_bus,
                        terminals,
                    )
                    root_shift, adjusted_depth, decoupled_root_groups, decoupled_edges = (
                        _decouple_global_root_mode(
                            distance,
                            depth,
                            terminals,
                            net.root_bus,
                            rnj_tolerance_factor,
                            root_mode_quantile,
                        )
                    )
                    decoupled_global_clades = rooted_clades(
                        decoupled_edges,
                        net.root_bus,
                        terminals,
                    )
                    true_root_groups = _root_branch_groups(
                        tuple((int(left), int(right), 1.0) for left, right in net.closed_edges()),
                        net.root_bus,
                        terminals,
                    )
                    clade_counts = {
                        method: Counter({clade: 0 for clade in base_clades[method]})
                        for method in METHODS
                    }
                    root_counts = Counter({group: 0 for group in base_root_groups})
                    decoupled_root_counts = Counter(
                        {group: 0 for group in decoupled_root_groups}
                    )
                    rng = np.random.default_rng(
                        20260711
                        + 100000 * case_index
                        + 10000 * data_replicate
                        + 100 * scenario_count
                        + t_count
                    )
                    for _ in range(bootstrap_replicates):
                        perturbed = _perturb(
                            scenarios,
                            rng,
                            pq_noise=extra_pq_noise_rel,
                            voltage_noise=extra_v_noise_rel,
                        )
                        _, _, sampled_distance, sampled_depth = _ordered_fit(perturbed)
                        sampled_by_method = {}
                        sampled_rnj_edges = tuple()
                        for method in METHODS:
                            sampled_clades, sampled_edges = _reconstruct_clades(
                                method,
                                sampled_distance,
                                sampled_depth,
                                terminals,
                                net.root_bus,
                                rnj_tolerance_factor,
                                rg_tolerance,
                            )
                            sampled_by_method[method] = sampled_clades
                            if method == "rnj":
                                sampled_rnj_edges = sampled_edges
                        for method in METHODS:
                            for clade in clade_counts[method]:
                                clade_counts[method][clade] += int(clade in sampled_by_method[method])
                        sampled_root_groups = _root_branch_groups(
                            sampled_rnj_edges,
                            net.root_bus,
                            terminals,
                        )
                        for group in root_counts:
                            root_counts[group] += int(group in sampled_root_groups)
                        _, _, sampled_decoupled_groups, _ = _decouple_global_root_mode(
                            sampled_distance,
                            sampled_depth,
                            terminals,
                            net.root_bus,
                            rnj_tolerance_factor,
                            root_mode_quantile,
                        )
                        for group in decoupled_root_counts:
                            decoupled_root_counts[group] += int(
                                group in sampled_decoupled_groups
                            )

                    denominator = max(bootstrap_replicates, 1)
                    probabilities = {
                        method: {
                            clade: count / denominator
                            for clade, count in clade_counts[method].items()
                        }
                        for method in METHODS
                    }
                    for method in METHODS:
                        precision, recall, f1_score = _set_score(base_clades[method], true_clades)
                        topology_rows.append(
                            {
                                "case": case_key,
                                "scenario_count": scenario_count,
                                "t_count": t_count,
                                "data_replicate": data_replicate,
                                "method": f"baseline_{method}",
                                "precision": precision,
                                "recall": recall,
                                "f1": f1_score,
                                "exact": base_clades[method] == true_clades,
                                "predicted_clades": len(base_clades[method]),
                                "root_partition_exact": (
                                    base_root_groups == true_root_groups
                                    if method == "rnj"
                                    else np.nan
                                ),
                                "root_shift": 0.0,
                            }
                        )
                        for clade, probability in probabilities[method].items():
                            probability_rows.append(
                                {
                                    "case": case_key,
                                    "scenario_count": scenario_count,
                                    "t_count": t_count,
                                    "data_replicate": data_replicate,
                                    "method": method,
                                    "clade": _clade_text(clade),
                                    "size": len(clade),
                                    "probability": probability,
                                    "is_true": clade in true_clades,
                                    "edge_level": "all_identifiable_clades",
                                }
                            )

                    common_candidates = set.intersection(*(set(base_clades[method]) for method in METHODS))
                    consensus_probability = {
                        clade: min(probabilities[method][clade] for method in METHODS)
                        for clade in common_candidates
                    }
                    probability_sets = {**probabilities, "consensus": consensus_probability}
                    for method, method_probabilities in probability_sets.items():
                        for threshold in thresholds:
                            stable = {
                                clade
                                for clade, probability in method_probabilities.items()
                                if probability >= threshold
                            }
                            stable_precision, stable_recall, stable_f1 = _set_score(stable, true_clades)
                            stable_rows.append(
                                {
                                    "case": case_key,
                                    "scenario_count": scenario_count,
                                    "t_count": t_count,
                                    "data_replicate": data_replicate,
                                    "method": method,
                                    "threshold": threshold,
                                    "stable_edge_count": len(stable),
                                    "stable_precision": stable_precision,
                                    "stable_recall": stable_recall,
                                    "stable_f1": stable_f1,
                                }
                            )

                    precision, recall, f1_score = _set_score(
                        decoupled_global_clades,
                        true_clades,
                    )
                    topology_rows.append(
                        {
                            "case": case_key,
                            "scenario_count": scenario_count,
                            "t_count": t_count,
                            "data_replicate": data_replicate,
                            "method": "root_shift_rnj",
                            "precision": precision,
                            "recall": recall,
                            "f1": f1_score,
                            "exact": decoupled_global_clades == true_clades,
                            "predicted_clades": len(decoupled_global_clades),
                            "root_partition_exact": decoupled_root_groups == true_root_groups,
                            "root_shift": root_shift,
                        }
                    )

                    for local_method in METHODS:
                        local_prediction = _root_decoupled_clades(
                            local_method,
                            distance,
                            depth,
                            terminals,
                            decoupled_root_groups,
                            rnj_tolerance_factor,
                            rg_tolerance,
                        )
                        precision, recall, f1_score = _set_score(local_prediction, true_clades)
                        topology_rows.append(
                            {
                                "case": case_key,
                                "scenario_count": scenario_count,
                                "t_count": t_count,
                                "data_replicate": data_replicate,
                                "method": f"root_decoupled_{local_method}",
                                "precision": precision,
                                "recall": recall,
                                "f1": f1_score,
                                "exact": local_prediction == true_clades,
                                "predicted_clades": len(local_prediction),
                                "root_partition_exact": decoupled_root_groups == true_root_groups,
                                "root_shift": root_shift,
                            }
                        )

                    root_probability_sets = {
                        "rnj_root_edge_raw": {
                            group: count / denominator
                            for group, count in root_counts.items()
                        },
                        "rnj_root_edge_decoupled": {
                            group: count / denominator
                            for group, count in decoupled_root_counts.items()
                        },
                    }
                    for method, method_probabilities in root_probability_sets.items():
                        for group, probability in method_probabilities.items():
                            probability_rows.append(
                                {
                                    "case": case_key,
                                    "scenario_count": scenario_count,
                                    "t_count": t_count,
                                    "data_replicate": data_replicate,
                                    "method": method,
                                    "clade": _clade_text(group),
                                    "size": len(group),
                                    "probability": probability,
                                    "is_true": group in true_root_groups,
                                    "edge_level": "root_incident",
                                }
                            )
                        for threshold in thresholds:
                            stable_root_edges = {
                                group
                                for group, probability in method_probabilities.items()
                                if probability >= threshold
                            }
                            stable_precision, stable_recall, stable_f1 = _set_score(
                                stable_root_edges,
                                true_root_groups,
                            )
                            stable_rows.append(
                                {
                                    "case": case_key,
                                    "scenario_count": scenario_count,
                                    "t_count": t_count,
                                    "data_replicate": data_replicate,
                                    "method": method,
                                    "threshold": threshold,
                                    "stable_edge_count": len(stable_root_edges),
                                    "stable_precision": stable_precision,
                                    "stable_recall": stable_recall,
                                    "stable_f1": stable_f1,
                                }
                            )
                    completed += 1
                    print(
                        f"[prob-root] {completed}/{condition_count} case={case_key} "
                        f"scenarios={scenario_count} T={t_count}",
                        flush=True,
                    )

    topology = pd.DataFrame(topology_rows)
    probabilities_frame = pd.DataFrame(probability_rows)
    stable = pd.DataFrame(stable_rows)
    for frame in (topology, probabilities_frame, stable):
        frame["regime"] = np.where(
            frame["scenario_count"].ge(3),
            "normal_multiscenario",
            "single_scenario_stress",
        )
    topology.to_csv(out_dir / "topology_results.csv", index=False)
    probabilities_frame.to_csv(out_dir / "edge_probabilities.csv", index=False)
    stable.to_csv(out_dir / "stable_edge_results.csv", index=False)
    topology_summary = (
        topology.groupby("method", as_index=False)
        .agg(
            runs=("f1", "size"),
            mean_f1=("f1", "mean"),
            exact_rate=("exact", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            root_partition_exact_rate=("root_partition_exact", "mean"),
        )
        .sort_values(["mean_f1", "exact_rate"], ascending=False)
    )
    topology_summary.to_csv(out_dir / "topology_summary.csv", index=False)
    topology_by_regime = (
        topology.groupby(["regime", "method"], as_index=False)
        .agg(
            runs=("f1", "size"),
            mean_f1=("f1", "mean"),
            exact_rate=("exact", "mean"),
            root_partition_exact_rate=("root_partition_exact", "mean"),
        )
        .sort_values(["regime", "mean_f1"], ascending=[True, False])
    )
    topology_by_regime.to_csv(out_dir / "topology_summary_by_regime.csv", index=False)
    topology_by_case = (
        topology.groupby(["case", "regime", "method"], as_index=False)
        .agg(
            runs=("f1", "size"),
            mean_f1=("f1", "mean"),
            exact_rate=("exact", "mean"),
            root_partition_exact_rate=("root_partition_exact", "mean"),
        )
        .sort_values(["case", "regime", "mean_f1"], ascending=[True, True, False])
    )
    topology_by_case.to_csv(out_dir / "topology_summary_by_case.csv", index=False)
    stable_summary = (
        stable.groupby(["method", "threshold"], as_index=False)
        .agg(
            runs=("stable_f1", "size"),
            mean_stable_precision=("stable_precision", "mean"),
            mean_stable_recall=("stable_recall", "mean"),
            mean_stable_f1=("stable_f1", "mean"),
            mean_stable_edge_count=("stable_edge_count", "mean"),
        )
        .sort_values(["mean_stable_precision", "mean_stable_recall"], ascending=False)
    )
    stable_summary.to_csv(out_dir / "stable_edge_summary.csv", index=False)
    stable_by_regime = (
        stable.groupby(["regime", "method", "threshold"], as_index=False)
        .agg(
            runs=("stable_f1", "size"),
            mean_stable_precision=("stable_precision", "mean"),
            mean_stable_recall=("stable_recall", "mean"),
            mean_stable_f1=("stable_f1", "mean"),
        )
    )
    stable_by_regime.to_csv(out_dir / "stable_edge_summary_by_regime.csv", index=False)

    probability_quality_rows = []
    for (method, edge_level), group in probabilities_frame.groupby(
        ["method", "edge_level"]
    ):
        probability = group["probability"].to_numpy(dtype=float)
        truth = group["is_true"].to_numpy(dtype=float)
        probability_quality_rows.append(
            {
                "method": method,
                "edge_level": edge_level,
                "candidate_count": len(group),
                "true_candidate_rate": float(np.mean(truth)),
                "mean_probability_true": (
                    float(np.mean(probability[truth == 1.0]))
                    if np.any(truth == 1.0)
                    else np.nan
                ),
                "mean_probability_false": (
                    float(np.mean(probability[truth == 0.0]))
                    if np.any(truth == 0.0)
                    else np.nan
                ),
                "brier_score": float(np.mean((probability - truth) ** 2)),
            }
        )
    probability_quality = pd.DataFrame(probability_quality_rows)
    probability_quality.to_csv(out_dir / "probability_quality.csv", index=False)

    probability_quality_by_regime_rows = []
    for (regime, method, edge_level), group in probabilities_frame.groupby(
        ["regime", "method", "edge_level"]
    ):
        probability = group["probability"].to_numpy(dtype=float)
        truth = group["is_true"].to_numpy(dtype=float)
        probability_quality_by_regime_rows.append(
            {
                "regime": regime,
                "method": method,
                "edge_level": edge_level,
                "candidate_count": len(group),
                "brier_score": float(np.mean((probability - truth) ** 2)),
            }
        )
    pd.DataFrame(probability_quality_by_regime_rows).to_csv(
        out_dir / "probability_quality_by_regime.csv",
        index=False,
    )

    calibrated = probabilities_frame.copy()
    calibrated["probability_bin"] = pd.cut(
        calibrated["probability"],
        bins=[-1e-12, 0.25, 0.50, 0.75, 0.875, 1.0 + 1e-12],
        labels=["[0,.25]", "(.25,.50]", "(.50,.75]", "(.75,.875]", "(.875,1]"],
        include_lowest=True,
    )
    calibration = (
        calibrated.groupby(
            ["method", "edge_level", "probability_bin"],
            observed=True,
            as_index=False,
        )
        .agg(
            candidate_count=("is_true", "size"),
            mean_probability=("probability", "mean"),
            empirical_true_rate=("is_true", "mean"),
        )
    )
    calibration.to_csv(out_dir / "probability_calibration.csv", index=False)
    metrics = {
        "condition_count": condition_count,
        "measurement_noise": {"pq_relative": pq_noise_rel, "voltage_relative": v_noise_rel},
        "extra_stability_noise": {
            "pq_relative": extra_pq_noise_rel,
            "voltage_relative": extra_v_noise_rel,
        },
        "bootstrap_replicates": bootstrap_replicates,
        "root_mode_quantile": root_mode_quantile,
        "stability_thresholds": thresholds,
        "topology_summary": topology_summary.to_dict(orient="records"),
        "topology_summary_by_regime": topology_by_regime.to_dict(orient="records"),
        "stable_edge_summary": stable_summary.to_dict(orient="records"),
        "stable_edge_summary_by_regime": stable_by_regime.to_dict(orient="records"),
        "probability_quality": probability_quality.to_dict(orient="records"),
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# Probabilistic stable edges and one-level root decoupling\n\n"
        "Probabilities are empirical persistence frequencies relative to the topology fitted "
        "from the original noisy measurements. They are not calibrated Bayesian posteriors. "
        "Root decoupling subtracts a low-quantile common root-path mode, reruns RNJ at "
        "the physical root, and reconstructs each downstream branch independently while "
        "preserving the original absolute RNJ grouping tolerance.\n\n"
        "## Complete topology\n\n"
        + topology_summary.to_string(index=False)
        + "\n\n## Topology by data regime\n\n"
        + topology_by_regime.to_string(index=False)
        + "\n\n## Stable identifiable edges\n\n"
        + stable_summary.to_string(index=False)
        + "\n\n## Probability quality\n\n"
        + probability_quality.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/probabilistic_root_decoupling")
    parser.add_argument("--cases", nargs="*", choices=list(CASE_BUILDERS), default=None)
    parser.add_argument("--scenario-counts", nargs="*", type=int, default=None)
    parser.add_argument("--T-counts", nargs="*", type=int, default=None)
    parser.add_argument("--data-replicates", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=8)
    parser.add_argument("--stability-thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--rnj-tolerance-factor", type=float, default=0.16)
    parser.add_argument("--rg-tolerance", type=float, default=0.03)
    parser.add_argument("--pq-noise-rel", type=float, default=0.005)
    parser.add_argument("--v-noise-rel", type=float, default=0.0002)
    parser.add_argument("--extra-pq-noise-rel", type=float, default=0.0025)
    parser.add_argument("--extra-v-noise-rel", type=float, default=0.0001)
    parser.add_argument("--root-mode-quantile", type=float, default=0.10)
    args = parser.parse_args()
    run(
        output_dir=args.output,
        cases=args.cases,
        scenario_counts=args.scenario_counts,
        t_counts=args.T_counts,
        data_replicates=args.data_replicates,
        bootstrap_replicates=args.bootstrap_replicates,
        stability_thresholds=args.stability_thresholds,
        rnj_tolerance_factor=args.rnj_tolerance_factor,
        rg_tolerance=args.rg_tolerance,
        pq_noise_rel=args.pq_noise_rel,
        v_noise_rel=args.v_noise_rel,
        extra_pq_noise_rel=args.extra_pq_noise_rel,
        extra_v_noise_rel=args.extra_v_noise_rel,
        root_mode_quantile=args.root_mode_quantile,
    )


if __name__ == "__main__":
    main()
