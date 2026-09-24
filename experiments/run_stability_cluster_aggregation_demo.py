"""Edge-stability clustering demo.

The experiment starts from already noisy measured P/Q/V data, adds a second
layer of synthetic measurement perturbations, and records which MST edges remain
stable. High-stability terminal edges are contracted into clusters. The cluster
P/Q/V signals are then re-estimated on a smaller pseudo-node graph and expanded
back to a terminal-equivalent tree.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from terminal_case33.estimation.recipes import apply_preprocessing_recipe
from experiments.pipeline_helpers import _fit_and_score, _simulate_case
from experiments.run_large_sweep_filter_distance import _distance_candidates, _fast_projected_multiscenario_fit
from experiments.run_oracle_cluster_lifting_demo import _recipe, _terminal_buses_from_scenarios
from terminal_case33.graph.terminal_tree import terminal_equivalent_minimum_distance_tree
from terminal_case33.graph.metrics import edge_precision_recall_f1
from terminal_case33.estimation.preprocessing import squared_voltage_drop_from_observed_root
from terminal_case33.models.lin_distflow import build_reduced_sensitivity_matrices, impedance_distance_from_reduced_R
from terminal_case33.utils.io import ensure_dir, write_json


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _noise_like(frame: pd.DataFrame, rel_std: float, rng: np.random.Generator) -> pd.DataFrame:
    """Return additive relative Gaussian noise with a small robust floor."""

    if rel_std <= 0.0:
        return pd.DataFrame(0.0, index=frame.index, columns=frame.columns)
    values = frame.to_numpy(dtype=float)
    positives = np.abs(values[np.abs(values) > 0.0])
    floor = 0.01 * float(np.median(positives)) if positives.size else 1e-10
    scale = rel_std * np.maximum(np.abs(values), floor)
    return pd.DataFrame(rng.normal(0.0, scale, size=values.shape), index=frame.index, columns=frame.columns)


def _perturb_scenarios(
    scenarios: list[dict],
    *,
    pq_extra_noise_rel: float,
    v_extra_noise_rel: float,
    rng: np.random.Generator,
) -> list[dict]:
    """Add a second synthetic measurement-noise layer to P/Q/V scenarios."""

    perturbed = []
    for scenario in scenarios:
        P = scenario["P_terminal"] + _noise_like(scenario["P_terminal"], pq_extra_noise_rel, rng)
        Q = scenario["Q_terminal"] + _noise_like(scenario["Q_terminal"], pq_extra_noise_rel, rng)
        V = scenario["V_terminal"] + _noise_like(scenario["V_terminal"], v_extra_noise_rel, rng)
        drop = squared_voltage_drop_from_observed_root(V, scenario["root_voltage"])
        perturbed.append(
            {
                "name": scenario["name"],
                "V_terminal": V,
                "P_terminal": P,
                "Q_terminal": Q,
                "root_voltage": scenario["root_voltage"],
                "drop_target": drop,
            }
        )
    return perturbed


def _fit_distance_and_mst(
    scenarios: list[dict],
    recipe: dict,
    distance_mode: str,
    alpha: float,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Fit reduced R/X, build a distance matrix, and return its MST."""

    fitted = []
    for scenario in scenarios:
        samples_per_day = len(scenario["P_terminal"])
        P = apply_preprocessing_recipe(scenario["P_terminal"], recipe, samples_per_day)
        Q = apply_preprocessing_recipe(scenario["Q_terminal"], recipe, samples_per_day)
        D = apply_preprocessing_recipe(scenario["drop_target"], recipe, samples_per_day)
        common = P.index.intersection(Q.index).intersection(D.index)
        fitted.append(
            {
                "name": scenario["name"],
                "V_terminal": scenario["V_terminal"].loc[common],
                "P_terminal": P.loc[common],
                "Q_terminal": Q.loc[common],
                "drop_target": D.loc[common],
            }
        )
    R_hat, X_hat, r2_score, condition_number = _fast_projected_multiscenario_fit(fitted, alpha=alpha)
    distances = _distance_candidates(R_hat, X_hat)[distance_mode]
    nodes = _terminal_buses_from_scenarios(scenarios)
    return terminal_equivalent_minimum_distance_tree(nodes, distances), distances, R_hat, X_hat, r2_score, condition_number


def _edge_stability(
    scenarios: list[dict],
    recipe: dict,
    distance_mode: str,
    *,
    replicates: int,
    pq_extra_noise_rel: float,
    v_extra_noise_rel: float,
    alpha: float,
    seed: int,
) -> tuple[pd.DataFrame, list[tuple[int, int]], np.ndarray, float, float]:
    """Estimate base-MST edge confidence by perturb-and-refit sampling.

    The unperturbed measured data define the reference MST. Confidence is the
    fraction of perturbation replicates in which that same reference edge is
    selected again. Edges that appear only after perturbation are deliberately
    not used for clustering, because this demo asks whether noisy resampling
    preserves the original identification rather than discovering new edges.
    """

    rng = np.random.default_rng(seed)
    base_mst, base_dist, _R_hat, _X_hat, base_r2, base_cond = _fit_distance_and_mst(
        scenarios, recipe, distance_mode, alpha
    )
    base_set = set(base_mst)
    counts: dict[tuple[int, int], int] = {edge: 0 for edge in base_mst}
    for _ in range(replicates):
        sampled = _perturb_scenarios(
            scenarios,
            pq_extra_noise_rel=pq_extra_noise_rel,
            v_extra_noise_rel=v_extra_noise_rel,
            rng=rng,
        )
        mst, _dist, _R, _X, _r2, _cond = _fit_distance_and_mst(sampled, recipe, distance_mode, alpha)
        for edge in mst:
            if edge in base_set:
                counts[edge] += 1
    rows = [
        {
            "u": int(edge[0]),
            "v": int(edge[1]),
            "confidence": float(count / max(replicates, 1)),
            "in_base_mst": True,
        }
        for edge, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return pd.DataFrame(rows), base_mst, base_dist, base_r2, base_cond


def _stable_forest_edges(
    terminals: list[int],
    confidence: pd.DataFrame,
    threshold: float,
) -> list[tuple[int, int]]:
    """Build a maximum-confidence acyclic forest from edges above threshold."""

    graph = nx.Graph()
    graph.add_nodes_from(terminals)
    for row in confidence.itertuples(index=False):
        if float(row.confidence) >= threshold:
            graph.add_edge(int(row.u), int(row.v), weight=float(row.confidence))
    forest: list[tuple[int, int]] = []
    for component in nx.connected_components(graph):
        sub = graph.subgraph(component)
        if sub.number_of_edges() == 0:
            continue
        tree = nx.maximum_spanning_tree(sub, weight="weight")
        forest.extend(tuple(sorted((int(u), int(v)))) for u, v in tree.edges())
    return sorted(forest)


def _clusters_from_forest(terminals: list[int], forest_edges: list[tuple[int, int]]) -> dict[int, int]:
    """Return terminal to pseudo-cluster id mapping."""

    graph = nx.Graph()
    graph.add_nodes_from(terminals)
    graph.add_edges_from(forest_edges)
    mapping: dict[int, int] = {}
    for component in nx.connected_components(graph):
        cluster_id = min(int(node) for node in component)
        for node in component:
            mapping[int(node)] = cluster_id
    return mapping


def _aggregate_by_clusters(scenarios: list[dict], terminal_cluster: dict[int, int]) -> list[dict]:
    """Aggregate terminal P/Q and use mean squared voltage inside each cluster."""

    cluster_ids = sorted(set(terminal_cluster.values()))
    aggregated = []
    for scenario in scenarios:
        P_child = scenario["P_terminal"]
        Q_child = scenario["Q_terminal"]
        V_child = scenario["V_terminal"]
        P_parent = pd.DataFrame(index=P_child.index)
        Q_parent = pd.DataFrame(index=Q_child.index)
        V_parent_sq = pd.DataFrame(index=V_child.index)
        for cluster_id in cluster_ids:
            children = [bus for bus, cid in terminal_cluster.items() if cid == cluster_id]
            P_parent[cluster_id] = P_child[children].sum(axis=1)
            Q_parent[cluster_id] = Q_child[children].sum(axis=1)
            V_parent_sq[cluster_id] = V_child[children].pow(2).mean(axis=1)
        V_parent = np.sqrt(np.maximum(V_parent_sq, 1e-10))
        drop = squared_voltage_drop_from_observed_root(V_parent, scenario["root_voltage"])
        aggregated.append(
            {
                "name": scenario["name"],
                "V_terminal": V_parent,
                "P_terminal": P_parent,
                "Q_terminal": Q_parent,
                "root_voltage": scenario["root_voltage"],
                "drop_target": drop,
            }
        )
    return aggregated


def _best_bridge(
    left: list[int],
    right: list[int],
    terminals: list[int],
    base_distance: np.ndarray,
) -> tuple[int, int]:
    """Choose the shortest base-distance terminal bridge between two clusters."""

    pos = {bus: idx for idx, bus in enumerate(terminals)}
    best_edge: tuple[int, int] | None = None
    best_value = float("inf")
    for u in left:
        for v in right:
            value = float(base_distance[pos[u], pos[v]])
            if value < best_value:
                best_value = value
                best_edge = tuple(sorted((int(u), int(v))))
    if best_edge is None:
        raise ValueError("empty cluster bridge")
    return best_edge


def _expand_cluster_tree(
    terminals: list[int],
    terminal_cluster: dict[int, int],
    stable_edges: list[tuple[int, int]],
    cluster_edges: list[tuple[int, int]],
    base_distance: np.ndarray,
    base_mst: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Expand a pseudo-cluster MST back to terminal-terminal edges."""

    cluster_members: dict[int, list[int]] = {}
    for terminal, cluster_id in terminal_cluster.items():
        cluster_members.setdefault(cluster_id, []).append(terminal)
    expanded = set(tuple(sorted(edge)) for edge in stable_edges)
    for cu, cv in cluster_edges:
        expanded.add(_best_bridge(cluster_members[int(cu)], cluster_members[int(cv)], terminals, base_distance))

    graph = nx.Graph()
    graph.add_nodes_from(terminals)
    graph.add_edges_from(expanded)
    for edge in base_mst:
        if nx.is_connected(graph):
            break
        u, v = edge
        if not nx.has_path(graph, u, v):
            graph.add_edge(u, v)
    if graph.number_of_edges() > len(terminals) - 1:
        graph = nx.minimum_spanning_tree(graph)
    return sorted(tuple(sorted((int(u), int(v)))) for u, v in graph.edges())


def _score_stability_clustering(
    net,
    scenarios: list[dict],
    terminals: list[int],
    true_mst: list[tuple[int, int]],
    recipe: dict,
    distance_mode: str,
    confidence: pd.DataFrame,
    base_mst: list[tuple[int, int]],
    base_distance: np.ndarray,
    threshold: float,
    alpha: float,
) -> dict:
    """Contract stable edges, refit pseudo graph, expand, and score."""

    stable_edges = _stable_forest_edges(terminals, confidence, threshold)
    terminal_cluster = _clusters_from_forest(terminals, stable_edges)
    cluster_count = len(set(terminal_cluster.values()))
    cluster_sizes = pd.Series(list(terminal_cluster.values())).value_counts().tolist()
    if cluster_count == len(terminals):
        expanded_edges = base_mst
        pseudo_score = {"terminal_mst_f1": np.nan, "condition_number": np.nan, "r2_score": np.nan}
    elif cluster_count == 1:
        expanded_edges = stable_edges
        pseudo_score = {"terminal_mst_f1": np.nan, "condition_number": np.nan, "r2_score": np.nan}
    else:
        aggregated = _aggregate_by_clusters(scenarios, terminal_cluster)
        pseudo_nodes = _terminal_buses_from_scenarios(aggregated)
        pseudo_score = _fit_and_score(net, pseudo_nodes, aggregated, recipe, distance_mode, alpha=alpha)
        cluster_edges = pseudo_score["pred_mst"]
        expanded_edges = _expand_cluster_tree(terminals, terminal_cluster, stable_edges, cluster_edges, base_distance, base_mst)
    score = edge_precision_recall_f1(expanded_edges, true_mst)
    true_set = {tuple(sorted(edge)) for edge in true_mst}
    stable_correct = sum(1 for edge in stable_edges if tuple(sorted(edge)) in true_set)
    return {
        "threshold": float(threshold),
        "cluster_count": int(cluster_count),
        "max_cluster_size": int(max(cluster_sizes) if cluster_sizes else 1),
        "stable_edge_count": int(len(stable_edges)),
        "stable_edge_precision": float(stable_correct / len(stable_edges)) if stable_edges else np.nan,
        "expanded_correct_edges": int(len(set(expanded_edges) & true_set)),
        "expanded_total_edges": int(len(true_mst)),
        "expanded_f1": float(score["f1"]),
        "expanded_precision": float(score["precision"]),
        "expanded_recall": float(score["recall"]),
        "pseudo_condition_number": float(pseudo_score["condition_number"]),
        "pseudo_r2_score": float(pseudo_score["r2_score"]),
        "stable_edges": stable_edges,
        "expanded_edges": expanded_edges,
    }


def run(
    output_dir: str | Path = "outputs/stability_cluster_aggregation_demo",
    cases: list[str] | None = None,
    scenario_count: int = 1,
    t_count: int = 96,
    recipe_name: str = "raw_drop",
    distance_mode: str = "RX_75R_25X",
    replicates: int = 80,
    thresholds: list[float] | None = None,
    pq_extra_noise_rel: float = 0.0025,
    v_extra_noise_rel: float = 0.0001,
    pq_noise_rel: float = 0.005,
    root_voltage_mean: float = 1.02,
    alpha: float = 0.0,
    seed: int = 20260709,
) -> dict:
    """Run edge-stability clustering on low-data case-bank scenarios."""

    out_dir = ensure_dir(output_dir)
    selected_cases = cases or ["paper15", "soumalas11", "flynn16", "pengwah18"]
    selected_thresholds = thresholds or [0.55, 0.65, 0.75, 0.85]
    recipe = _recipe(recipe_name)
    rows = []
    base_rows = []
    start = time.time()
    for case_idx, case_key in enumerate(selected_cases, start=1):
        case_dir = ensure_dir(out_dir / case_key)
        net, scenarios = _simulate_case(
            case_key=case_key,
            scenario_count=scenario_count,
            t_count=t_count,
            pq_noise_rel=pq_noise_rel,
            root_voltage_mean=root_voltage_mean,
        )
        terminals = _terminal_buses_from_scenarios(scenarios)
        R_true, _X_true = build_reduced_sensitivity_matrices(net, terminals, voltage_model="squared-voltage")
        true_mst = terminal_equivalent_minimum_distance_tree(terminals, impedance_distance_from_reduced_R(R_true))
        baseline = _fit_and_score(net, terminals, scenarios, recipe, distance_mode, alpha=alpha)
        confidence, base_mst, base_distance, base_r2, base_cond = _edge_stability(
            scenarios,
            recipe,
            distance_mode,
            replicates=replicates,
            pq_extra_noise_rel=pq_extra_noise_rel,
            v_extra_noise_rel=v_extra_noise_rel,
            alpha=alpha,
            seed=seed + 1000 * case_idx,
        )
        true_set = {tuple(sorted(edge)) for edge in true_mst}
        confidence["is_true_terminal_mst_edge"] = [
            tuple(sorted((int(row.u), int(row.v)))) in true_set for row in confidence.itertuples(index=False)
        ]
        confidence.to_csv(case_dir / "edge_confidence.csv", index=False)
        pd.DataFrame(true_mst, columns=["u", "v"]).to_csv(case_dir / "true_terminal_mst.csv", index=False)
        pd.DataFrame(base_mst, columns=["u", "v"]).to_csv(case_dir / "base_terminal_mst.csv", index=False)
        base_rows.append(
            {
                "case": case_key,
                "method": "baseline_terminal_mst",
                "scenario_count": scenario_count,
                "t_count": t_count,
                "recipe": recipe_name,
                "distance_mode": distance_mode,
                "replicates": replicates,
                "base_f1": float(baseline["terminal_mst_f1"]),
                "base_correct_edges": int(baseline["correct_edges"]),
                "base_total_edges": int(baseline["total_edges"]),
                "base_r2_score": float(base_r2),
                "base_condition_number": float(base_cond),
            }
        )
        for threshold in selected_thresholds:
            score = _score_stability_clustering(
                net,
                scenarios,
                terminals,
                true_mst,
                recipe,
                distance_mode,
                confidence,
                base_mst,
                base_distance,
                threshold,
                alpha,
            )
            pd.DataFrame(score["stable_edges"], columns=["u", "v"]).to_csv(
                case_dir / f"stable_edges_thr_{threshold:.2f}.csv", index=False
            )
            pd.DataFrame(score["expanded_edges"], columns=["u", "v"]).to_csv(
                case_dir / f"expanded_tree_thr_{threshold:.2f}.csv", index=False
            )
            rows.append(
                {
                    "case": case_key,
                    "scenario_count": scenario_count,
                    "t_count": t_count,
                    "recipe": recipe_name,
                    "distance_mode": distance_mode,
                    "replicates": replicates,
                    "pq_extra_noise_rel": pq_extra_noise_rel,
                    "v_extra_noise_rel": v_extra_noise_rel,
                    "baseline_f1": float(baseline["terminal_mst_f1"]),
                    **{key: value for key, value in score.items() if key not in {"stable_edges", "expanded_edges"}},
                }
            )
        elapsed = time.time() - start
        eta = elapsed / case_idx * (len(selected_cases) - case_idx)
        print(
            f"[progress] {case_idx}/{len(selected_cases)} case={case_key} "
            f"base_f1={baseline['terminal_mst_f1']:.3f} best_expanded={max(r['expanded_f1'] for r in rows if r['case'] == case_key):.3f} "
            f"time={_fmt_seconds(elapsed)} eta={_fmt_seconds(eta)}",
            flush=True,
        )
    result = pd.DataFrame(rows)
    baseline_df = pd.DataFrame(base_rows)
    result.to_csv(out_dir / "stability_cluster_summary.csv", index=False)
    baseline_df.to_csv(out_dir / "baseline_summary.csv", index=False)
    by_threshold = (
        result.groupby("threshold", as_index=False)
        .agg(
            mean_baseline_f1=("baseline_f1", "mean"),
            mean_expanded_f1=("expanded_f1", "mean"),
            mean_delta_f1=("expanded_f1", lambda values: float(np.mean(values - result.loc[values.index, "baseline_f1"]))),
            mean_cluster_count=("cluster_count", "mean"),
            mean_stable_edge_precision=("stable_edge_precision", "mean"),
        )
        .sort_values("mean_expanded_f1", ascending=False)
    )
    by_threshold.to_csv(out_dir / "summary_by_threshold.csv", index=False)
    best = result.sort_values(["expanded_f1", "threshold"], ascending=[False, True]).groupby("case").head(1)
    best.to_csv(out_dir / "best_by_case.csv", index=False)
    metrics = {
        "cases": selected_cases,
        "scenario_count": scenario_count,
        "t_count": t_count,
        "recipe": recipe_name,
        "distance_mode": distance_mode,
        "replicates": replicates,
        "pq_extra_noise_rel": pq_extra_noise_rel,
        "v_extra_noise_rel": v_extra_noise_rel,
        "pq_noise_rel": pq_noise_rel,
        "root_voltage_mean": root_voltage_mean,
        "mean_baseline_f1": float(result["baseline_f1"].mean()) if len(result) else float("nan"),
        "mean_best_expanded_f1_by_case": float(best["expanded_f1"].mean()) if len(best) else float("nan"),
        "best_threshold_table": by_threshold.to_dict(orient="records"),
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# Edge Stability Cluster Aggregation Demo\n\n"
        "The experiment perturbs already noisy P/Q/V measurements, refits the terminal MST many times, "
        "contracts edges whose empirical frequency exceeds a threshold, aggregates P/Q/V to pseudo clusters, "
        "and expands the pseudo MST back to a terminal-equivalent tree.\n\n"
        "## Threshold Summary\n\n"
        + by_threshold.to_string(index=False)
        + "\n\n## Best By Case\n\n"
        + best.drop(columns=["stable_edges", "expanded_edges"], errors="ignore").to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/stability_cluster_aggregation_demo")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--scenario-count", type=int, default=1)
    parser.add_argument("--T", type=int, default=96)
    parser.add_argument("--recipe", default="raw_drop")
    parser.add_argument("--distance-mode", default="RX_75R_25X")
    parser.add_argument("--replicates", type=int, default=80)
    parser.add_argument("--thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--pq-extra-noise-rel", type=float, default=0.0025)
    parser.add_argument("--v-extra-noise-rel", type=float, default=0.0001)
    parser.add_argument("--pq-noise-rel", type=float, default=0.005)
    parser.add_argument("--root-voltage-mean", type=float, default=1.02)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()
    run(
        output_dir=args.output,
        cases=args.cases,
        scenario_count=args.scenario_count,
        t_count=args.T,
        recipe_name=args.recipe,
        distance_mode=args.distance_mode,
        replicates=args.replicates,
        thresholds=args.thresholds,
        pq_extra_noise_rel=args.pq_extra_noise_rel,
        v_extra_noise_rel=args.v_extra_noise_rel,
        pq_noise_rel=args.pq_noise_rel,
        root_voltage_mean=args.root_voltage_mean,
        alpha=args.alpha,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
