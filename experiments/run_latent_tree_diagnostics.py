"""Large diagnostics for root placement, edge errors, and pseudo measurements."""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from experiments.common import (
    resource_assignment,
    selected_scenario_defs,
    simulate_case_scenario,
    terminal_buses,
)
from terminal_case33.data.paper_style_case_bank import CASE_BUILDERS
from terminal_case33.estimation.multiscenario import fit_projected_sensitivity, preprocess_scenarios
from terminal_case33.graph.diagnostics import (
    locate_root_from_terminal_depths,
    root_partition_is_correct,
    terminal_limb_lengths,
    terminal_sibling_pairs,
)
from terminal_case33.graph.latent_tree import neighbor_joining, recursive_grouping, terminal_splits
from terminal_case33.models.ac_powerflow import solve_ac_power_flow_timeseries
from terminal_case33.models.lin_distflow import (
    build_reduced_sensitivity_matrices,
    impedance_distance_from_reduced_R,
    ohm_to_pu,
)
from terminal_case33.utils.io import ensure_dir, write_json


def _recipes() -> list[dict]:
    return [
        {"name": "raw_drop", "kind": "raw"},
        {"name": "daily_demean", "kind": "demean"},
    ]


def _distance_and_depth(
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    d_r = impedance_distance_from_reduced_R(r_matrix)
    d_x = impedance_distance_from_reduced_R(x_matrix)
    scale_r = float(np.mean(d_r[d_r > 0.0])) if np.any(d_r > 0.0) else 1.0
    scale_x = float(np.mean(d_x[d_x > 0.0])) if np.any(d_x > 0.0) else 1.0
    if mode == "R":
        return d_r, np.diag(r_matrix), (1.0, 0.0)
    if mode == "X":
        return d_x, np.diag(x_matrix), (0.0, 1.0)
    weights = {
        "RX_equal_normalized": (1.0, 1.0),
        "RX_75R_25X": (0.75, 0.25),
        "RX_25R_75X": (0.25, 0.75),
    }
    if mode not in weights:
        raise ValueError(f"unknown distance mode {mode!r}")
    weight_r, weight_x = weights[mode]
    distance = weight_r * d_r / max(scale_r, 1e-12) + weight_x * d_x / max(scale_x, 1e-12)
    depth = weight_r * np.diag(r_matrix) / max(scale_r, 1e-12) + weight_x * np.diag(x_matrix) / max(scale_x, 1e-12)
    return distance, depth, (weight_r / max(scale_r, 1e-12), weight_x / max(scale_x, 1e-12))


def _reconstruct(method: str, distance: np.ndarray, terminals: list[int], rg_tolerance: float):
    if method == "nj":
        return neighbor_joining(distance, terminals)
    if method == "rg":
        return recursive_grouping(distance, terminals, tolerance=rg_tolerance)
    raise ValueError(f"unknown method {method!r}")


def _set_score(predicted: set, truth: set) -> tuple[float, float, float]:
    matched = len(predicted & truth)
    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(truth) if truth else 1.0
    f1_score = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1_score


def _true_limb_lengths(net, terminals: list[int], coefficients: tuple[float, float]) -> dict[int, float]:
    graph = net.to_networkx_graph()
    coefficient_r, coefficient_x = coefficients
    result = {}
    for terminal in terminals:
        neighbor = next(iter(graph.neighbors(terminal)))
        edge = graph.edges[terminal, neighbor]
        r_length = 2.0 * ohm_to_pu(net, float(edge["r_ohm"]))
        x_length = 2.0 * ohm_to_pu(net, float(edge["x_ohm"]))
        result[terminal] = coefficient_r * r_length + coefficient_x * x_length
    return result


def _detailed_scenarios(case_key: str, scenario_count: int, t_count: int, pq_noise_rel: float) -> tuple[object, list[dict]]:
    net = CASE_BUILDERS[case_key]()
    assignment = resource_assignment(case_key, net)
    scenarios = []
    for definition in selected_scenario_defs(max_scenarios=scenario_count):
        simulation = simulate_case_scenario(
            net=net,
            assignment=assignment,
            T=t_count,
            seed=int(definition["seed"]),
            pq_noise_rel=pq_noise_rel,
            v_noise_rel=float(definition["v_noise_rel"]),
            root_voltage_mean=1.02,
            root_voltage_sigma=float(definition["root_voltage_sigma"]),
            profile_scenario=str(definition["profile_scenario"]),
        )
        ac = solve_ac_power_flow_timeseries(
            net,
            simulation["P_true"],
            simulation["Q_true"],
            v_root=simulation["root_voltage"],
            max_iter=100,
            tol=1e-10,
        )
        scenarios.append(
            {
                "name": str(definition["name"]),
                "P_terminal": simulation["P_terminal"],
                "Q_terminal": simulation["Q_terminal"],
                "V_terminal": simulation["V_terminal"],
                "root_voltage": simulation["root_voltage"],
                "drop_target": simulation["drop_target"],
                "P_true": simulation["P_true"],
                "Q_true": simulation["Q_true"],
                "V_bus_true": ac["V_bus_mag"],
                "branch_p_from_true": ac["branch_p_from_pu"],
                "branch_q_from_true": ac["branch_q_from_pu"],
            }
        )
    return net, scenarios


def _downstream_terminal_sets(net, terminals: list[int]) -> tuple[dict[int, int], dict[int, set[int]]]:
    oriented = net.orient_from_root()
    parent = {int(row.child): int(row.parent) for row in oriented.itertuples(index=False)}
    children: dict[int, list[int]] = {int(bus): [] for bus in net.buses["bus_id"]}
    for child, upstream in parent.items():
        children[upstream].append(child)
    terminal_set = set(terminals)
    descendants: dict[int, set[int]] = {}

    def visit(node: int) -> set[int]:
        found = {node} if node in terminal_set else set()
        for child in children[node]:
            found.update(visit(child))
        descendants[node] = found
        return found

    visit(int(net.root_bus))
    return parent, descendants


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _proxy_diagnostics(net, scenarios: list[dict], terminals: list[int], max_cluster_size: int) -> pd.DataFrame:
    graph = net.to_networkx_graph()
    parent, descendants = _downstream_terminal_sets(net, terminals)
    rows = []
    for child, upstream in parent.items():
        cluster = sorted(descendants[child])
        if not 2 <= len(cluster) <= max_cluster_size:
            continue
        edge_label = f"{upstream}->{child}"
        mean_errors, max_errors, corrected_errors = [], [], []
        mean_dynamic, corrected_dynamic = [], []
        p_true_errors, p_measured_errors, q_true_errors, q_measured_errors = [], [], [], []
        true_parent_values, mean_values, corrected_values = [], [], []
        for scenario in scenarios:
            true_parent = scenario["V_bus_true"][child].pow(2)
            mean_proxy = scenario["V_terminal"][cluster].pow(2).mean(axis=1)
            max_proxy = scenario["V_terminal"][cluster].pow(2).max(axis=1)
            corrected_terminal = []
            for terminal in cluster:
                estimate = scenario["V_terminal"][terminal].pow(2).copy()
                path = nx.shortest_path(graph, child, terminal)
                for left, right in zip(path[:-1], path[1:]):
                    downstream = sorted(descendants[right])
                    edge = graph.edges[left, right]
                    p_flow = scenario["P_terminal"][downstream].sum(axis=1)
                    q_flow = scenario["Q_terminal"][downstream].sum(axis=1)
                    estimate += 2.0 * (
                        ohm_to_pu(net, float(edge["r_ohm"])) * p_flow
                        + ohm_to_pu(net, float(edge["x_ohm"])) * q_flow
                    )
                corrected_terminal.append(estimate)
            corrected_proxy = pd.concat(corrected_terminal, axis=1).mean(axis=1)

            branch_p = scenario["branch_p_from_true"][edge_label]
            branch_q = scenario["branch_q_from_true"][edge_label]
            p_true_sum = scenario["P_true"][cluster].sum(axis=1)
            q_true_sum = scenario["Q_true"][cluster].sum(axis=1)
            p_measured_sum = scenario["P_terminal"][cluster].sum(axis=1)
            q_measured_sum = scenario["Q_terminal"][cluster].sum(axis=1)

            mean_errors.extend((mean_proxy - true_parent).to_numpy())
            max_errors.extend((max_proxy - true_parent).to_numpy())
            corrected_errors.extend((corrected_proxy - true_parent).to_numpy())
            mean_dynamic.extend(((mean_proxy - mean_proxy.mean()) - (true_parent - true_parent.mean())).to_numpy())
            corrected_dynamic.extend(((corrected_proxy - corrected_proxy.mean()) - (true_parent - true_parent.mean())).to_numpy())
            p_true_errors.extend((p_true_sum - branch_p).to_numpy())
            p_measured_errors.extend((p_measured_sum - branch_p).to_numpy())
            q_true_errors.extend((q_true_sum - branch_q).to_numpy())
            q_measured_errors.extend((q_measured_sum - branch_q).to_numpy())
            true_parent_values.extend(true_parent.to_numpy())
            mean_values.extend(mean_proxy.to_numpy())
            corrected_values.extend(corrected_proxy.to_numpy())

        true_array = np.asarray(true_parent_values)
        parent_std = max(float(np.std(true_array)), 1e-12)
        branch_p_rms = max(_rmse(np.concatenate([scenario["branch_p_from_true"][edge_label].to_numpy() for scenario in scenarios])), 1e-12)
        branch_q_rms = max(_rmse(np.concatenate([scenario["branch_q_from_true"][edge_label].to_numpy() for scenario in scenarios])), 1e-12)
        rows.append(
            {
                "upstream_bus": upstream,
                "boundary_bus": child,
                "cluster": " ".join(map(str, cluster)),
                "cluster_size": len(cluster),
                "mean_vsq_rmse": _rmse(mean_errors),
                "max_vsq_rmse": _rmse(max_errors),
                "drop_corrected_vsq_rmse": _rmse(corrected_errors),
                "mean_vsq_dynamic_rmse": _rmse(mean_dynamic),
                "drop_corrected_dynamic_rmse": _rmse(corrected_dynamic),
                "mean_dynamic_error_over_parent_std": _rmse(mean_dynamic) / parent_std,
                "corrected_dynamic_error_over_parent_std": _rmse(corrected_dynamic) / parent_std,
                "mean_vsq_bias": float(np.mean(mean_errors)),
                "drop_corrected_vsq_bias": float(np.mean(corrected_errors)),
                "mean_vsq_correlation": float(np.corrcoef(mean_values, true_array)[0, 1]),
                "drop_corrected_correlation": float(np.corrcoef(corrected_values, true_array)[0, 1]),
                "true_p_sum_relative_rmse_vs_branch": _rmse(p_true_errors) / branch_p_rms,
                "measured_p_sum_relative_rmse_vs_branch": _rmse(p_measured_errors) / branch_p_rms,
                "true_q_sum_relative_rmse_vs_branch": _rmse(q_true_errors) / branch_q_rms,
                "measured_q_sum_relative_rmse_vs_branch": _rmse(q_measured_errors) / branch_q_rms,
            }
        )
    return pd.DataFrame(rows)


def run(
    output_dir: str | Path = "outputs/latent_tree_diagnostics_large",
    cases: list[str] | None = None,
    scenario_counts: list[int] | None = None,
    t_counts: list[int] | None = None,
    distance_mode: str = "RX_75R_25X",
    pq_noise_rel: float = 0.005,
    rg_tolerance: float = 0.03,
    max_cluster_size: int = 5,
) -> dict:
    """Run the large root, internal-edge, and pseudo-measurement diagnostic sweep."""

    out_dir = ensure_dir(output_dir)
    selected_cases = cases or list(CASE_BUILDERS)
    selected_scenarios = scenario_counts or [1, 3, 5]
    selected_t = t_counts or [48, 96, 288]
    topology_rows = []
    proxy_rows = []
    total = len(selected_cases) * len(selected_scenarios) * len(selected_t)
    task = 0
    for case_key in selected_cases:
        for scenario_count in selected_scenarios:
            for t_count in selected_t:
                task += 1
                net, scenarios = _detailed_scenarios(case_key, scenario_count, t_count, pq_noise_rel)
                terminals = terminal_buses(net)
                true_edges = net.closed_edges()
                true_splits = terminal_splits(true_edges, terminals)
                true_siblings = terminal_sibling_pairs(true_edges, terminals)
                r_true, x_true = build_reduced_sensitivity_matrices(net, terminals, voltage_model="squared-voltage")
                true_distance, true_depth, true_coefficients = _distance_and_depth(r_true, x_true, distance_mode)
                true_limbs = _true_limb_lengths(net, terminals, true_coefficients)

                proxy = _proxy_diagnostics(net, scenarios, terminals, max_cluster_size)
                if not proxy.empty:
                    proxy.insert(0, "t_count", t_count)
                    proxy.insert(0, "scenario_count", scenario_count)
                    proxy.insert(0, "case", case_key)
                    proxy_rows.append(proxy)

                for recipe in _recipes():
                    fitted = preprocess_scenarios(scenarios, recipe)
                    r_hat, x_hat, r2_score, condition_number = fit_projected_sensitivity(fitted)
                    distance, depth, _ = _distance_and_depth(r_hat, x_hat, distance_mode)
                    for method in ("nj", "rg"):
                        oracle = _reconstruct(method, true_distance, terminals, 1e-8)
                        oracle_root = locate_root_from_terminal_depths(oracle.edges, terminals, true_depth)
                        predicted = _reconstruct(method, distance, terminals, rg_tolerance)
                        predicted_splits = terminal_splits(predicted.edges, terminals)
                        predicted_siblings = terminal_sibling_pairs(predicted.edges, terminals)
                        split_precision, split_recall, split_f1 = _set_score(predicted_splits, true_splits)
                        sibling_precision, sibling_recall, sibling_f1 = _set_score(predicted_siblings, true_siblings)
                        placement = locate_root_from_terminal_depths(predicted.edges, terminals, depth)
                        predicted_limbs = terminal_limb_lengths(predicted.edges, terminals)
                        limb_errors = [
                            abs(predicted_limbs[terminal] - true_limbs[terminal]) / max(true_limbs[terminal], 1e-12)
                            for terminal in terminals
                            if terminal in predicted_limbs
                        ]
                        internal_error = 1.0 - split_recall
                        edge_error = 1.0 - sibling_recall
                        if internal_error > edge_error + 0.05:
                            dominant = "internal"
                        elif edge_error > internal_error + 0.05:
                            dominant = "terminal_attachment"
                        else:
                            dominant = "mixed"
                        topology_rows.append(
                            {
                                "case": case_key,
                                "scenario_count": scenario_count,
                                "t_count": t_count,
                                "recipe": recipe["name"],
                                "method": method,
                                "oracle_split_f1": _set_score(terminal_splits(oracle.edges, terminals), true_splits)[2],
                                "oracle_root_partition_correct": root_partition_is_correct(
                                    oracle_root, oracle.edges, true_edges, net.root_bus, terminals
                                ),
                                "split_precision": split_precision,
                                "split_recall": split_recall,
                                "split_f1": split_f1,
                                "sibling_precision": sibling_precision,
                                "sibling_recall": sibling_recall,
                                "sibling_f1": sibling_f1,
                                "terminal_limb_relative_mae": float(np.mean(limb_errors)),
                                "terminal_limb_relative_median_error": float(np.median(limb_errors)),
                                "root_location_kind": placement.kind,
                                "root_partition_correct": root_partition_is_correct(
                                    placement, predicted.edges, true_edges, net.root_bus, terminals
                                ),
                                "root_depth_relative_rmse": placement.rmse / max(float(np.mean(depth)), 1e-12),
                                "forced_merges": predicted.forced_merges,
                                "r2_score": r2_score,
                                "condition_number": condition_number,
                                "dominant_error_region": dominant,
                            }
                        )
                print(f"[diagnostic] {task}/{total} case={case_key} scenarios={scenario_count} T={t_count}", flush=True)

    topology = pd.DataFrame(topology_rows)
    proxy = pd.concat(proxy_rows, ignore_index=True) if proxy_rows else pd.DataFrame()
    topology.to_csv(out_dir / "topology_diagnostics.csv", index=False)
    proxy.to_csv(out_dir / "pseudo_measurement_diagnostics.csv", index=False)
    topology_summary = (
        topology.groupby(["method", "recipe"], as_index=False)
        .agg(
            runs=("split_f1", "size"),
            mean_split_f1=("split_f1", "mean"),
            mean_split_recall=("split_recall", "mean"),
            mean_sibling_f1=("sibling_f1", "mean"),
            mean_sibling_recall=("sibling_recall", "mean"),
            mean_limb_relative_mae=("terminal_limb_relative_mae", "mean"),
            root_partition_success_rate=("root_partition_correct", "mean"),
            oracle_root_success_rate=("oracle_root_partition_correct", "mean"),
        )
    )
    topology_summary.to_csv(out_dir / "topology_summary.csv", index=False)
    proxy_summary = pd.DataFrame()
    if not proxy.empty:
        proxy_summary = (
            proxy.groupby("case", as_index=False)
            .agg(
                cluster_tests=("cluster", "size"),
                mean_vsq_rmse=("mean_vsq_rmse", "mean"),
                corrected_vsq_rmse=("drop_corrected_vsq_rmse", "mean"),
                mean_dynamic_error_ratio=("mean_dynamic_error_over_parent_std", "mean"),
                corrected_dynamic_error_ratio=("corrected_dynamic_error_over_parent_std", "mean"),
                p_aggregation_relative_rmse=("measured_p_sum_relative_rmse_vs_branch", "mean"),
                q_aggregation_relative_rmse=("measured_q_sum_relative_rmse_vs_branch", "mean"),
            )
        )
        proxy_summary.to_csv(out_dir / "pseudo_measurement_summary.csv", index=False)
    metrics = {
        "topology_run_count": int(len(topology)),
        "proxy_cluster_test_count": int(len(proxy)),
        "root_partition_success_rate": float(topology["root_partition_correct"].mean()),
        "oracle_root_partition_success_rate": float(topology["oracle_root_partition_correct"].mean()),
        "mean_split_recall": float(topology["split_recall"].mean()),
        "mean_sibling_recall": float(topology["sibling_recall"].mean()),
        "internal_dominant_fraction": float(topology["dominant_error_region"].eq("internal").mean()),
        "terminal_attachment_dominant_fraction": float(
            topology["dominant_error_region"].eq("terminal_attachment").mean()
        ),
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# Latent-tree root and aggregation diagnostics\n\n"
        "NJ/RG are unrooted. Root success below uses sensitivity diagonals to place the root after reconstruction.\n\n"
        "## Topology summary\n\n"
        + topology_summary.to_string(index=False)
        + "\n\n## Pseudo measurement summary\n\n"
        + proxy_summary.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/latent_tree_diagnostics_large")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--scenario-counts", nargs="*", type=int, default=None)
    parser.add_argument("--T-counts", nargs="*", type=int, default=None)
    parser.add_argument("--distance-mode", default="RX_75R_25X")
    parser.add_argument("--pq-noise-rel", type=float, default=0.005)
    parser.add_argument("--rg-tolerance", type=float, default=0.03)
    parser.add_argument("--max-cluster-size", type=int, default=5)
    args = parser.parse_args()
    run(
        output_dir=args.output,
        cases=args.cases,
        scenario_counts=args.scenario_counts,
        t_counts=args.T_counts,
        distance_mode=args.distance_mode,
        pq_noise_rel=args.pq_noise_rel,
        rg_tolerance=args.rg_tolerance,
        max_cluster_size=args.max_cluster_size,
    )


if __name__ == "__main__":
    main()
