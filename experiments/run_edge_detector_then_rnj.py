"""Detect one-layer peripheral clusters with NJ/RG, then reconstruct the backbone with RNJ."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_latent_tree_diagnostics import _distance_and_depth, _set_score
from experiments.run_matrix_constraint_large_sweep import _simulate_pool, _terminal_buses
from experiments.run_rooted_hierarchical_aggregation import RECIPE, _perturb
from terminal_case33.data.paper_style_case_bank import CASE_BUILDERS
from terminal_case33.estimation.multiscenario import fit_projected_sensitivity, preprocess_scenarios
from terminal_case33.graph.diagnostics import locate_root_from_terminal_depths
from terminal_case33.graph.latent_tree import neighbor_joining, recursive_grouping
from terminal_case33.graph.rooted_hierarchy import (
    PseudoCluster,
    aggregate_rooted_scenarios,
    expand_pseudo_clades,
    rooted_clades,
)
from terminal_case33.graph.rooted_neighbor_joining import rooted_neighbor_joining, shared_paths_from_distances
from terminal_case33.utils.io import ensure_dir, write_json


DETECTORS = ("nj_rooted_clade", "rg_rooted_clade", "nj_rg_consensus")


def _ordered_fit(scenarios: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fitted = preprocess_scenarios(scenarios, RECIPE)
    r_matrix, x_matrix, _, _ = fit_projected_sensitivity(fitted, constraint_mode="ordered")
    distance, depth, _ = _distance_and_depth(r_matrix, x_matrix, "RX_75R_25X")
    return r_matrix, x_matrix, distance, depth


def _small_rooted_clades_from_unrooted(
    edges: tuple[tuple[int, int, float], ...],
    terminals: list[int],
    depth: np.ndarray,
    max_cluster_size: int,
) -> set[frozenset[int]]:
    """Root an unrooted latent tree from measured root depths and return edge clades."""

    placement = locate_root_from_terminal_depths(edges, terminals, depth)
    if placement.kind == "node":
        if placement.node is None:
            raise ValueError("node root placement must identify one node")
        root_node = int(placement.node)
        rooted_edges = edges
    else:
        if placement.edge is None:
            raise ValueError("edge root placement must identify one edge")
        left, right = placement.edge
        minimum_node = min(node for edge in edges for node in edge[:2])
        root_node = min(minimum_node - 1, -1000000)
        rooted_edges = tuple(
            edge for edge in edges if {int(edge[0]), int(edge[1])} != {left, right}
        ) + ((root_node, left, 0.0), (root_node, right, 0.0))
    return {
        clade
        for clade in rooted_clades(rooted_edges, root_node, terminals)
        if 2 <= len(clade) <= max_cluster_size
    }


def _nj_groups(
    distance: np.ndarray,
    depth: np.ndarray,
    terminals: list[int],
    max_cluster_size: int,
) -> set[frozenset[int]]:
    """Return small rooted edge clades from an NJ latent tree."""

    tree = neighbor_joining(distance, terminals)
    return _small_rooted_clades_from_unrooted(
        tree.edges,
        terminals,
        depth,
        max_cluster_size,
    )


def _rg_groups(
    distance: np.ndarray,
    depth: np.ndarray,
    terminals: list[int],
    root: int,
    rg_tolerance: float,
    max_cluster_size: int,
) -> set[frozenset[int]]:
    """Return small rooted edge clades from root-augmented recursive grouping."""

    augmented = np.zeros((len(terminals) + 1, len(terminals) + 1), dtype=float)
    augmented[:-1, :-1] = distance
    augmented[:-1, -1] = depth
    augmented[-1, :-1] = depth
    tree = recursive_grouping(augmented, [*terminals, root], tolerance=rg_tolerance)
    return {
        clade
        for clade in rooted_clades(tree.edges, root, terminals)
        if 2 <= len(clade) <= max_cluster_size
    }


def _detector_groups(
    detector: str,
    distance: np.ndarray,
    depth: np.ndarray,
    terminals: list[int],
    root: int,
    rg_tolerance: float,
    max_cluster_size: int,
) -> set[frozenset[int]]:
    nj_groups = _nj_groups(distance, depth, terminals, max_cluster_size)
    if detector == "nj_rooted_clade":
        return nj_groups
    rg_groups = _rg_groups(
        distance,
        depth,
        terminals,
        root,
        rg_tolerance,
        max_cluster_size,
    )
    if detector == "rg_rooted_clade":
        return rg_groups
    if detector == "nj_rg_consensus":
        return nj_groups & rg_groups
    raise ValueError(f"unknown detector {detector!r}")


def _cluster_confidence(
    detector: str,
    scenarios: list[dict],
    terminals: list[int],
    root: int,
    base_groups: set[frozenset[int]],
    bootstrap_replicates: int,
    rg_tolerance: float,
    max_cluster_size: int,
    seed: int,
) -> dict[frozenset[int], float]:
    """Estimate group stability under a second measurement-noise layer."""

    counts = {group: 0 for group in base_groups}
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap_replicates):
        sampled = _perturb(scenarios, rng, pq_noise=0.0025, voltage_noise=0.0001)
        _, _, distance, depth = _ordered_fit(sampled)
        sampled_groups = _detector_groups(
            detector,
            distance,
            depth,
            terminals,
            root,
            rg_tolerance,
            max_cluster_size,
        )
        for group in counts:
            counts[group] += int(group in sampled_groups)
    return {
        group: count / max(bootstrap_replicates, 1)
        for group, count in counts.items()
    }


def _normalized_boundary_margin(
    group: frozenset[int],
    distance: np.ndarray,
    depth: np.ndarray,
    terminals: list[int],
) -> float:
    """Measure separation between a candidate clade boundary and its best competitor."""

    shared = shared_paths_from_distances(distance, depth)
    position = {terminal: index for index, terminal in enumerate(terminals)}
    members = [position[terminal] for terminal in sorted(group)]
    outside = [index for index, terminal in enumerate(terminals) if terminal not in group]
    within = shared[np.ix_(members, members)][np.triu_indices(len(members), 1)]
    if not within.size or not outside:
        return 0.0
    boundary = float(np.quantile(within, 0.2))
    competing = max(float(np.median(shared[np.ix_(members, [index])])) for index in outside)
    return (boundary - competing) / max(float(np.median(depth)), 1e-12)


def _select_stable_clusters(
    confidence: dict[frozenset[int], float],
    threshold: float,
    distance: np.ndarray,
    depth: np.ndarray,
    terminals: list[int],
    min_boundary_margin: float,
) -> list[PseudoCluster]:
    """Select stable, separated, disjoint rooted clades."""

    eligible = [
        group
        for group, value in confidence.items()
        if value >= threshold
        and _normalized_boundary_margin(group, distance, depth, terminals) >= min_boundary_margin
    ]
    selected = []
    occupied = set()
    for group in sorted(
        eligible,
        key=lambda group: (-confidence[group], -len(group), tuple(sorted(group))),
    ):
        if occupied.isdisjoint(group):
            selected.append(group)
            occupied.update(group)
    return [
        PseudoCluster(
            pseudo_id=900000 + index,
            members=group,
            confidence=confidence[group],
            frozen_clades=tuple(),
            frozen_sibling_pairs=tuple(combinations(sorted(group), 2)),
        )
        for index, group in enumerate(selected)
    ]


def _rnj_clades(
    scenarios: list[dict],
    terminals: list[int],
    root: int,
    tolerance_factor: float,
) -> tuple[np.ndarray, np.ndarray, set[frozenset[int]]]:
    r_matrix, x_matrix, distance, depth = _ordered_fit(scenarios)
    tolerance = tolerance_factor * max(float(np.median(depth)), 1e-12)
    tree = rooted_neighbor_joining(
        shared_paths_from_distances(distance, depth),
        depth,
        terminals,
        root,
        tolerance,
    )
    return r_matrix, x_matrix, rooted_clades(tree.edges, root, terminals)


def _aggregate_then_rnj(
    scenarios: list[dict],
    terminals: list[int],
    root: int,
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    clusters: list[PseudoCluster],
    tolerance_factor: float,
    deembedding_weight: float,
) -> set[frozenset[int]]:
    if not clusters:
        return _rnj_clades(scenarios, terminals, root, tolerance_factor)[2]
    pseudo_scenarios, pseudo_members = aggregate_rooted_scenarios(
        scenarios,
        terminals,
        r_matrix,
        x_matrix,
        clusters,
        "deembedded_vsq",
        deembedding_weight=deembedding_weight,
    )
    _, _, distance, depth = _ordered_fit(pseudo_scenarios)
    pseudo_nodes = list(pseudo_members)
    tolerance = tolerance_factor * max(float(np.median(depth)), 1e-12)
    pseudo_tree = rooted_neighbor_joining(
        shared_paths_from_distances(distance, depth),
        depth,
        pseudo_nodes,
        root,
        tolerance,
    )
    return expand_pseudo_clades(pseudo_tree.edges, root, pseudo_members, clusters)


def run(
    output_dir: str | Path = "outputs/edge_detector_then_rnj",
    cases: list[str] | None = None,
    scenario_counts: list[int] | None = None,
    t_counts: list[int] | None = None,
    replicates: int = 2,
    bootstrap_replicates: int = 8,
    thresholds: list[float] | None = None,
    max_cluster_size: int = 4,
    rg_tolerance: float = 0.03,
    rnj_tolerance_factor: float = 0.16,
    deembedding_weight: float = 0.25,
    min_boundary_margin: float = 0.10,
    pq_noise_rel: float = 0.005,
    v_noise_rel: float = 0.0002,
) -> dict:
    """Test NJ/RG one-layer edge detection followed by pseudo-level RNJ."""

    out_dir = ensure_dir(output_dir)
    selected_cases = cases or list(CASE_BUILDERS)
    selected_scenario_counts = scenario_counts or [3, 5]
    selected_t_counts = t_counts or [48, 96]
    selected_thresholds = thresholds or [0.75, 0.875]
    maximum_scenarios = max(selected_scenario_counts)
    total = len(selected_cases) * len(selected_t_counts) * replicates * len(selected_scenario_counts)
    completed = 0
    rows = []

    for case_index, case_key in enumerate(selected_cases):
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
                for scenario_count in selected_scenario_counts:
                    scenarios = scenario_pool[:scenario_count]
                    r_matrix, x_matrix, distance, depth = _ordered_fit(scenarios)
                    _, _, baseline_clades = _rnj_clades(
                        scenarios,
                        terminals,
                        net.root_bus,
                        rnj_tolerance_factor,
                    )
                    base_precision, base_recall, base_f1 = _set_score(baseline_clades, true_clades)
                    base_exact = baseline_clades == true_clades
                    for detector_index, detector in enumerate(DETECTORS):
                        base_groups = _detector_groups(
                            detector,
                            distance,
                            depth,
                            terminals,
                            net.root_bus,
                            rg_tolerance,
                            max_cluster_size,
                        )
                        confidence = _cluster_confidence(
                            detector,
                            scenarios,
                            terminals,
                            net.root_bus,
                            base_groups,
                            bootstrap_replicates,
                            rg_tolerance,
                            max_cluster_size,
                            seed=(
                                20260711
                                + 100000 * case_index
                                + 10000 * replicate
                                + 100 * scenario_count
                                + 10 * t_count
                                + detector_index
                            ),
                        )
                        for threshold in selected_thresholds:
                            clusters = _select_stable_clusters(
                                confidence,
                                threshold,
                                distance,
                                depth,
                                terminals,
                                min_boundary_margin,
                            )
                            predicted = _aggregate_then_rnj(
                                scenarios,
                                terminals,
                                net.root_bus,
                                r_matrix,
                                x_matrix,
                                clusters,
                                rnj_tolerance_factor,
                                deembedding_weight,
                            )
                            precision, recall, f1_score = _set_score(predicted, true_clades)
                            true_clusters = sum(cluster.members in true_clades for cluster in clusters)
                            rows.append(
                                {
                                    "case": case_key,
                                    "replicate": replicate,
                                    "scenario_count": scenario_count,
                                    "t_count": t_count,
                                    "detector": detector,
                                    "threshold": threshold,
                                    "bootstrap_replicates": bootstrap_replicates,
                                    "base_candidate_groups": len(base_groups),
                                    "selected_clusters": len(clusters),
                                    "min_boundary_margin": min_boundary_margin,
                                    "selected_cluster_members": ";".join(
                                        ",".join(map(str, sorted(cluster.members)))
                                        for cluster in clusters
                                    ),
                                    "clustered_terminals": len(
                                        set().union(*(cluster.members for cluster in clusters))
                                    ) if clusters else 0,
                                    "selected_cluster_precision": (
                                        true_clusters / len(clusters) if clusters else np.nan
                                    ),
                                    "baseline_precision": base_precision,
                                    "baseline_recall": base_recall,
                                    "baseline_f1": base_f1,
                                    "baseline_exact": base_exact,
                                    "aggregated_precision": precision,
                                    "aggregated_recall": recall,
                                    "aggregated_f1": f1_score,
                                    "aggregated_exact": predicted == true_clades,
                                    "delta_f1": f1_score - base_f1,
                                }
                            )
                    completed += 1
                print(
                    f"[edge-detector-rnj] {completed}/{total} case={case_key} "
                    f"T={t_count} replicate={replicate}",
                    flush=True,
                )

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "condition_results.csv", index=False)
    summary = (
        results.groupby(["detector", "threshold"], as_index=False)
        .agg(
            runs=("aggregated_f1", "size"),
            baseline_mean_f1=("baseline_f1", "mean"),
            aggregated_mean_f1=("aggregated_f1", "mean"),
            baseline_exact_rate=("baseline_exact", "mean"),
            aggregated_exact_rate=("aggregated_exact", "mean"),
            improved=("delta_f1", lambda value: int(np.sum(value > 1e-12))),
            equal=("delta_f1", lambda value: int(np.sum(np.abs(value) <= 1e-12))),
            degraded=("delta_f1", lambda value: int(np.sum(value < -1e-12))),
            mean_delta_f1=("delta_f1", "mean"),
            mean_selected_clusters=("selected_clusters", "mean"),
            mean_cluster_precision=("selected_cluster_precision", "mean"),
        )
        .sort_values(["aggregated_mean_f1", "aggregated_exact_rate"], ascending=False)
    )
    summary.to_csv(out_dir / "method_summary.csv", index=False)
    by_case = (
        results.groupby(["case", "detector", "threshold"], as_index=False)
        .agg(
            runs=("aggregated_f1", "size"),
            baseline_mean_f1=("baseline_f1", "mean"),
            aggregated_mean_f1=("aggregated_f1", "mean"),
            mean_delta_f1=("delta_f1", "mean"),
            aggregated_exact_rate=("aggregated_exact", "mean"),
        )
    )
    by_case.to_csv(out_dir / "summary_by_case.csv", index=False)
    metrics = {
        "method": "one_layer_edge_detector_then_pseudo_RNJ",
        "comparison_subset": "normal but error-prone: scenario counts 3/5 and T 48/96",
        "condition_count": total,
        "detectors": list(DETECTORS),
        "thresholds": selected_thresholds,
        "bootstrap_replicates": bootstrap_replicates,
        "min_boundary_margin": min_boundary_margin,
        "summary": summary.to_dict(orient="records"),
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# One-layer NJ/RG detection followed by RNJ\n\n"
        "NJ first reconstructs an unrooted latent tree, which is rooted from measured "
        "root-to-terminal depths; RG includes the observed root directly. Small edge-induced "
        "clades must pass bootstrap stability and a boundary-margin test. Selected clades "
        "are contracted once; P/Q are summed, pseudo voltage is deembedded, and RNJ "
        "reconstructs the pseudo backbone.\n\n"
        + summary.to_string(index=False)
        + "\n\n## By case\n\n"
        + by_case.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/edge_detector_then_rnj")
    parser.add_argument("--cases", nargs="*", choices=list(CASE_BUILDERS), default=None)
    parser.add_argument("--scenario-counts", nargs="*", type=int, default=None)
    parser.add_argument("--T-counts", nargs="*", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=8)
    parser.add_argument("--thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--max-cluster-size", type=int, default=4)
    parser.add_argument("--rg-tolerance", type=float, default=0.03)
    parser.add_argument("--rnj-tolerance-factor", type=float, default=0.16)
    parser.add_argument("--deembedding-weight", type=float, default=0.25)
    parser.add_argument("--min-boundary-margin", type=float, default=0.10)
    parser.add_argument("--pq-noise-rel", type=float, default=0.005)
    parser.add_argument("--v-noise-rel", type=float, default=0.0002)
    args = parser.parse_args()
    run(
        output_dir=args.output,
        cases=args.cases,
        scenario_counts=args.scenario_counts,
        t_counts=args.T_counts,
        replicates=args.replicates,
        bootstrap_replicates=args.bootstrap_replicates,
        thresholds=args.thresholds,
        max_cluster_size=args.max_cluster_size,
        rg_tolerance=args.rg_tolerance,
        rnj_tolerance_factor=args.rnj_tolerance_factor,
        deembedding_weight=args.deembedding_weight,
        min_boundary_margin=args.min_boundary_margin,
        pq_noise_rel=args.pq_noise_rel,
        v_noise_rel=args.v_noise_rel,
    )


if __name__ == "__main__":
    main()
