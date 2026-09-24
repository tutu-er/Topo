"""Oracle-cluster lifting demo for hierarchical terminal aggregation.

This experiment uses the true immediate hidden parent of each terminal only for
grouping. Service-drop coefficients are estimated from terminal-layer distances,
then terminal P/Q/V measurements are lifted to pseudo-parent measurements.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from terminal_case33.estimation.recipes import apply_preprocessing_recipe
from experiments.pipeline_helpers import (
    _aggregate_to_immediate_hidden_parents,
    _fit_and_score,
    _simulate_case,
    _terminal_parent_map,
)
from experiments.run_large_sweep_filter_distance import _fast_projected_multiscenario_fit
from experiments.run_large_sweep_filter_distance import _distance_candidates
from terminal_case33.graph.terminal_tree import terminal_equivalent_minimum_distance_tree
from terminal_case33.graph.metrics import edge_precision_recall_f1
from terminal_case33.utils.io import ensure_dir, write_json


def _recipe(name: str) -> dict:
    """Return a compact preprocessing recipe by name."""

    recipes = {
        "raw_drop": {"name": "raw_drop", "kind": "raw"},
        "daily_demean": {"name": "daily_demean", "kind": "demean"},
        "first_difference": {"name": "first_difference", "kind": "difference"},
        "rolling_highpass_w240": {"name": "rolling_highpass_w240", "kind": "rolling_highpass", "window": 240},
        "rolling_highpass_w288": {"name": "rolling_highpass_w288", "kind": "rolling_highpass", "window": 288},
    }
    if name not in recipes:
        raise ValueError(f"unknown recipe {name!r}; choose one of {sorted(recipes)}")
    return recipes[name]


def _terminal_buses_from_scenarios(scenarios: list[dict]) -> list[int]:
    """Return terminal column ids from scenario data."""

    return [int(col) for col in scenarios[0]["P_terminal"].columns]


def _preprocess_for_fit(scenarios: list[dict], recipe: dict) -> list[dict]:
    """Apply one preprocessing recipe to measured scenarios."""

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
    return fitted


def _fit_matrices_for_scenarios(
    scenarios: list[dict],
    recipe: dict,
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit projected R/X matrices for a scenario list."""

    fitted = _preprocess_for_fit(scenarios, recipe)
    return _fast_projected_multiscenario_fit(fitted, alpha=alpha)


def _distance_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return additive distance induced by a reduced sensitivity matrix."""

    diag = np.diag(matrix)
    return diag[:, None] + diag[None, :] - 2.0 * matrix


def _solve_pair_sum_nonnegative(distance_submatrix: np.ndarray) -> np.ndarray:
    """Estimate nonnegative service coefficients from d_ij ~= s_i + s_j."""

    k = distance_submatrix.shape[0]
    if k == 1:
        return np.zeros(1)
    if k == 2:
        value = max(float(distance_submatrix[0, 1]) / 2.0, 0.0)
        return np.array([value, value])

    rows = []
    target = []
    for i in range(k):
        for j in range(i + 1, k):
            row = np.zeros(k)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            target.append(float(distance_submatrix[i, j]))
    A = np.vstack(rows)
    b = np.asarray(target)
    active = np.zeros(k, dtype=bool)
    coef = np.zeros(k)
    for _ in range(k + 2):
        free = ~active
        if not np.any(free):
            return coef
        sol = np.zeros(k)
        sol[free] = np.linalg.lstsq(A[:, free], b, rcond=None)[0]
        new_active = active | (sol < 0.0)
        sol[new_active] = 0.0
        coef = sol
        if np.array_equal(new_active, active):
            return coef
        active = new_active
    return np.maximum(coef, 0.0)


def _estimate_terminal_layer_service_coefficients(
    scenarios: list[dict],
    terminal_parent: dict[int, int],
    recipe: dict,
) -> pd.DataFrame:
    """Estimate per-terminal service drop coefficients from true oracle clusters."""

    fitted = _preprocess_for_fit(scenarios, recipe)
    R_hat, X_hat, _r2, _cond = _fast_projected_multiscenario_fit(fitted, alpha=0.0)
    terminals = _terminal_buses_from_scenarios(scenarios)
    dR = _distance_from_matrix(R_hat)
    dX = _distance_from_matrix(X_hat)
    pos = {bus: idx for idx, bus in enumerate(terminals)}
    records = []
    for parent in sorted(set(terminal_parent.values())):
        cluster = sorted([bus for bus, par in terminal_parent.items() if par == parent])
        idx = [pos[bus] for bus in cluster]
        r_coef = _solve_pair_sum_nonnegative(dR[np.ix_(idx, idx)])
        x_coef = _solve_pair_sum_nonnegative(dX[np.ix_(idx, idx)])
        for bus, r_value, x_value in zip(cluster, r_coef, x_coef):
            records.append(
                {
                    "terminal_bus": int(bus),
                    "oracle_parent": int(parent),
                    "cluster_size": len(cluster),
                    "r_service_sensitivity_hat": float(r_value),
                    "x_service_sensitivity_hat": float(x_value),
                    "identifiability_note": "split_pair_evenly" if len(cluster) == 2 else "least_squares_pair_sum",
                }
            )
    return pd.DataFrame(records)


def _lift_with_estimated_service_drop(
    scenarios: list[dict],
    service_coefficients: pd.DataFrame,
) -> list[dict]:
    """Build pseudo-parent P/Q/V measurements using estimated service drops."""

    coeff = service_coefficients.set_index("terminal_bus")
    parent_ids = sorted(coeff["oracle_parent"].astype(int).unique())
    lifted = []
    for scenario in scenarios:
        P_child = scenario["P_terminal"]
        Q_child = scenario["Q_terminal"]
        V_child = scenario["V_terminal"]
        P_parent = pd.DataFrame(index=P_child.index)
        Q_parent = pd.DataFrame(index=Q_child.index)
        V_parent_sq = pd.DataFrame(index=V_child.index)
        for parent in parent_ids:
            children = coeff.index[coeff["oracle_parent"].astype(int).eq(parent)].astype(int).tolist()
            P_parent[parent] = P_child[children].sum(axis=1)
            Q_parent[parent] = Q_child[children].sum(axis=1)
            recovered = []
            for child in children:
                r_hat = float(coeff.loc[child, "r_service_sensitivity_hat"])
                x_hat = float(coeff.loc[child, "x_service_sensitivity_hat"])
                # R/X are squared-voltage drop coefficients, so no extra factor 2 is applied here.
                recovered.append(V_child[child].pow(2) + r_hat * P_child[child] + x_hat * Q_child[child])
            V_parent_sq[parent] = pd.concat(recovered, axis=1).mean(axis=1)
        V_parent = np.sqrt(np.maximum(V_parent_sq, 1e-10))
        drop = pd.DataFrame(
            scenario["root_voltage"].to_numpy()[:, None] ** 2 - V_parent.pow(2).to_numpy(),
            index=V_parent.index,
            columns=V_parent.columns,
        )
        lifted.append(
            {
                "name": scenario["name"],
                "V_terminal": V_parent,
                "P_terminal": P_parent,
                "Q_terminal": Q_parent,
                "root_voltage": scenario["root_voltage"],
                "drop_target": drop,
            }
        )
    return lifted


def _pseudo_voltage_sq_from_linear_drop(
    pseudo_scenarios: list[dict],
    recipe: dict,
    alpha: float = 0.0,
) -> tuple[list[pd.DataFrame], np.ndarray, np.ndarray, float, float]:
    """Estimate pseudo-root squared voltages from root voltage minus linear drop.

    For each pseudo node c and time t, this computes

    ``v_c_hat(t) = v_0(t) - R_hat[c, :] P_pseudo(t) - X_hat[c, :] Q_pseudo(t)``.

    The value is later used as the local root squared voltage for identifying
    the terminals inside that pseudo cluster.
    """

    R_hat, X_hat, r2_score, condition_number = _fit_matrices_for_scenarios(pseudo_scenarios, recipe, alpha=alpha)
    voltage_sq = []
    for scenario in pseudo_scenarios:
        P = scenario["P_terminal"].to_numpy(dtype=float)
        Q = scenario["Q_terminal"].to_numpy(dtype=float)
        drop = P @ R_hat.T + Q @ X_hat.T
        root_sq = scenario["root_voltage"].to_numpy(dtype=float)[:, None] ** 2
        values = np.maximum(root_sq - drop, 1e-8)
        voltage_sq.append(pd.DataFrame(values, index=scenario["P_terminal"].index, columns=scenario["P_terminal"].columns))
    return voltage_sq, R_hat, X_hat, r2_score, condition_number


def _local_reroot_internal_edges(
    raw_scenarios: list[dict],
    terminal_parent: dict[int, int],
    pseudo_voltage_sq: list[pd.DataFrame],
    recipe: dict,
    distance_mode: str,
    alpha: float = 0.0,
) -> tuple[list[tuple[int, int]], pd.DataFrame]:
    """Re-identify each terminal cluster using its pseudo parent as local root."""

    rows = []
    internal_edges: list[tuple[int, int]] = []
    for parent in sorted(set(terminal_parent.values())):
        children = sorted([terminal for terminal, par in terminal_parent.items() if par == parent])
        if len(children) <= 1:
            rows.append(
                {
                    "pseudo_parent": int(parent),
                    "cluster_size": len(children),
                    "local_edge_count": 0,
                    "local_r2_score": np.nan,
                    "local_condition_number": np.nan,
                    "note": "singleton",
                }
            )
            continue
        local_scenarios = []
        for scenario, parent_voltage_sq in zip(raw_scenarios, pseudo_voltage_sq):
            root_sq = parent_voltage_sq[int(parent)].clip(lower=1e-8)
            V_child = scenario["V_terminal"][children]
            drop = pd.DataFrame(
                root_sq.to_numpy()[:, None] - V_child.pow(2).to_numpy(),
                index=V_child.index,
                columns=children,
            )
            local_scenarios.append(
                {
                    "name": f"{scenario['name']}_local_{parent}",
                    "V_terminal": V_child,
                    "P_terminal": scenario["P_terminal"][children],
                    "Q_terminal": scenario["Q_terminal"][children],
                    "root_voltage": np.sqrt(root_sq),
                    "drop_target": drop,
                }
            )
        R_local, X_local, r2_score, condition_number = _fit_matrices_for_scenarios(
            local_scenarios,
            recipe,
            alpha=alpha,
        )
        distance = _distance_candidates(R_local, X_local)[distance_mode]
        edges = terminal_equivalent_minimum_distance_tree(children, distance)
        internal_edges.extend(edges)
        rows.append(
            {
                "pseudo_parent": int(parent),
                "cluster_size": len(children),
                "local_edge_count": len(edges),
                "local_r2_score": float(r2_score),
                "local_condition_number": float(condition_number),
                "note": "local_reroot_fit",
            }
        )
    return sorted(set(tuple(sorted(edge)) for edge in internal_edges)), pd.DataFrame(rows)


def _best_bridge(
    left: list[int],
    right: list[int],
    terminals: list[int],
    base_distance: np.ndarray,
) -> tuple[int, int]:
    """Choose the shortest estimated terminal bridge between two pseudo clusters."""

    pos = {bus: idx for idx, bus in enumerate(terminals)}
    best_edge: tuple[int, int] | None = None
    best_value = float("inf")
    for u in left:
        for v in right:
            value = float(base_distance[pos[int(u)], pos[int(v)]])
            if value < best_value:
                best_value = value
                best_edge = tuple(sorted((int(u), int(v))))
    if best_edge is None:
        raise ValueError("cannot bridge empty clusters")
    return best_edge


def _expand_with_local_edges(
    terminal_nodes: list[int],
    terminal_parent: dict[int, int],
    internal_edges: list[tuple[int, int]],
    pseudo_edges: list[tuple[int, int]],
    base_distance: np.ndarray,
    base_mst: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Combine local cluster edges and pseudo-main-tree bridges."""

    cluster_members: dict[int, list[int]] = {}
    for terminal, parent in terminal_parent.items():
        cluster_members.setdefault(int(parent), []).append(int(terminal))
    graph = nx.Graph()
    graph.add_nodes_from(terminal_nodes)
    graph.add_edges_from(internal_edges)
    for cu, cv in pseudo_edges:
        graph.add_edge(
            *_best_bridge(cluster_members[int(cu)], cluster_members[int(cv)], terminal_nodes, base_distance)
        )
    for edge in base_mst:
        if nx.is_connected(graph):
            break
        u, v = edge
        if not nx.has_path(graph, u, v):
            graph.add_edge(u, v)
    if graph.number_of_edges() > len(terminal_nodes) - 1:
        graph = nx.minimum_spanning_tree(graph)
    return sorted(tuple(sorted((int(u), int(v)))) for u, v in graph.edges())


def _full_terminal_score(
    true_mst: list[tuple[int, int]],
    pred_mst: list[tuple[int, int]],
    *,
    node_count: int,
    recipe_name: str,
    distance_mode: str,
    r2_score: float,
    condition_number: float,
) -> dict:
    """Build a score dict matching the existing summary schema."""

    score = edge_precision_recall_f1(pred_mst, true_mst)
    return {
        "node_count": int(node_count),
        "edge_count": int(len(true_mst)),
        "distance_mode": distance_mode,
        "filter": recipe_name,
        "r2_score": float(r2_score),
        "condition_number": float(condition_number),
        "correct_edges": int(len(set(pred_mst) & set(true_mst))),
        "total_edges": int(len(true_mst)),
        "terminal_mst_f1": float(score["f1"]),
        "precision": float(score["precision"]),
        "recall": float(score["recall"]),
        "true_mst": true_mst,
        "pred_mst": pred_mst,
        "relative_error_R": np.nan,
        "relative_error_X": np.nan,
    }


def _pseudo_nodes_from_parent_map(terminal_parent: dict[int, int]) -> list[int]:
    """Return sorted pseudo-parent node ids."""

    return sorted(set(int(parent) for parent in terminal_parent.values()))


def _condition_improvement(baseline: dict, lifted: dict) -> float:
    """Return baseline condition number divided by lifted condition number."""

    return float(baseline["condition_number"]) / max(float(lifted["condition_number"]), 1e-12)


def run(
    output_dir: str | Path = "outputs/oracle_cluster_lifting_demo",
    cases: list[str] | None = None,
    scenario_count: int = 5,
    t_count: int = 240,
    recipe_name: str = "raw_drop",
    distance_mode: str = "RX_75R_25X",
) -> dict:
    """Run oracle cluster lifting validation."""

    out_dir = ensure_dir(output_dir)
    selected_cases = cases or ["paper15", "flynn16", "pengwah18"]
    recipe = _recipe(recipe_name)
    rows = []
    all_coefficients = []
    for case_key in selected_cases:
        net, scenarios = _simulate_case(
            case_key=case_key,
            scenario_count=scenario_count,
            t_count=t_count,
            pq_noise_rel=0.005,
            root_voltage_mean=1.02,
        )
        terminal_parent = _terminal_parent_map(net)
        terminal_nodes = _terminal_buses_from_scenarios(scenarios)
        pseudo_nodes = _pseudo_nodes_from_parent_map(terminal_parent)

        baseline = _fit_and_score(net, terminal_nodes, scenarios, recipe, distance_mode, alpha=0.0)
        rows.append({"case": case_key, "method": "original_terminal", "voltage_proxy": "measured_terminal", **baseline})
        R_base, X_base, _base_r2, _base_cond = _fit_matrices_for_scenarios(scenarios, recipe, alpha=0.0)
        base_distance = _distance_candidates(R_base, X_base)[distance_mode]

        mean_lifted = [_aggregate_to_immediate_hidden_parents(net, scenario, "mean_terminal_voltage") for scenario in scenarios]
        mean_score = _fit_and_score(net, pseudo_nodes, mean_lifted, recipe, distance_mode, alpha=0.0)
        rows.append({"case": case_key, "method": "oracle_cluster_lift", "voltage_proxy": "mean_terminal_voltage", **mean_score})

        service_coefficients = _estimate_terminal_layer_service_coefficients(scenarios, terminal_parent, recipe)
        service_coefficients.insert(0, "case", case_key)
        all_coefficients.append(service_coefficients)
        estimated_lifted = _lift_with_estimated_service_drop(scenarios, service_coefficients)
        estimated_score = _fit_and_score(net, pseudo_nodes, estimated_lifted, recipe, distance_mode, alpha=0.0)
        rows.append(
            {
                "case": case_key,
                "method": "oracle_cluster_lift",
                "voltage_proxy": "estimated_service_drop",
                **estimated_score,
            }
        )
        pseudo_voltage_sq, _R_pseudo, _X_pseudo, pseudo_r2, pseudo_cond = _pseudo_voltage_sq_from_linear_drop(
            estimated_lifted,
            recipe,
            alpha=0.0,
        )
        local_edges, local_report = _local_reroot_internal_edges(
            scenarios,
            terminal_parent,
            pseudo_voltage_sq,
            recipe,
            distance_mode,
            alpha=0.0,
        )
        reroot_expanded = _expand_with_local_edges(
            terminal_nodes,
            terminal_parent,
            local_edges,
            estimated_score["pred_mst"],
            base_distance,
            baseline["pred_mst"],
        )
        reroot_score = _full_terminal_score(
            baseline["true_mst"],
            reroot_expanded,
            node_count=len(terminal_nodes),
            recipe_name=recipe_name,
            distance_mode=distance_mode,
            r2_score=pseudo_r2,
            condition_number=pseudo_cond,
        )
        rows.append(
            {
                "case": case_key,
                "method": "oracle_cluster_lift_then_local_reroot",
                "voltage_proxy": "linear_drop_pseudo_root",
                **reroot_score,
            }
        )

        case_dir = ensure_dir(out_dir / case_key)
        pd.DataFrame(baseline["true_mst"], columns=["u", "v"]).to_csv(case_dir / "terminal_true_mst.csv", index=False)
        pd.DataFrame(baseline["pred_mst"], columns=["u", "v"]).to_csv(case_dir / "terminal_pred_mst.csv", index=False)
        pd.DataFrame(estimated_score["true_mst"], columns=["u", "v"]).to_csv(case_dir / "pseudo_true_mst.csv", index=False)
        pd.DataFrame(estimated_score["pred_mst"], columns=["u", "v"]).to_csv(case_dir / "pseudo_pred_mst.csv", index=False)
        pd.DataFrame(local_edges, columns=["u", "v"]).to_csv(case_dir / "local_reroot_internal_edges.csv", index=False)
        pd.DataFrame(reroot_expanded, columns=["u", "v"]).to_csv(case_dir / "local_reroot_expanded_terminal_tree.csv", index=False)
        local_report.to_csv(case_dir / "local_reroot_cluster_report.csv", index=False)

    summary = pd.DataFrame(rows)
    compact = summary.drop(columns=["true_mst", "pred_mst"])
    compact.to_csv(out_dir / "oracle_cluster_lifting_summary.csv", index=False)
    if all_coefficients:
        pd.concat(all_coefficients, ignore_index=True).to_csv(out_dir / "estimated_service_coefficients.csv", index=False)

    comparison_rows = []
    for case_key, case_df in compact.groupby("case"):
        base = case_df[case_df["method"].eq("original_terminal")].iloc[0].to_dict()
        for _, lifted in case_df[case_df["method"].eq("oracle_cluster_lift")].iterrows():
            lifted_dict = lifted.to_dict()
            comparison_rows.append(
                {
                    "case": case_key,
                    "voltage_proxy": lifted_dict["voltage_proxy"],
                    "baseline_f1": float(base["terminal_mst_f1"]),
                    "lifted_f1": float(lifted_dict["terminal_mst_f1"]),
                    "f1_delta": float(lifted_dict["terminal_mst_f1"] - base["terminal_mst_f1"]),
                    "baseline_nodes": int(base["node_count"]),
                    "lifted_nodes": int(lifted_dict["node_count"]),
                    "condition_improvement_ratio": _condition_improvement(base, lifted_dict),
                    "baseline_relR": float(base["relative_error_R"]),
                    "lifted_relR": float(lifted_dict["relative_error_R"]),
                    "baseline_relX": float(base["relative_error_X"]),
                    "lifted_relX": float(lifted_dict["relative_error_X"]),
                }
            )
        for _, reroot in case_df[case_df["method"].eq("oracle_cluster_lift_then_local_reroot")].iterrows():
            reroot_dict = reroot.to_dict()
            comparison_rows.append(
                {
                    "case": case_key,
                    "voltage_proxy": reroot_dict["voltage_proxy"],
                    "baseline_f1": float(base["terminal_mst_f1"]),
                    "lifted_f1": float(reroot_dict["terminal_mst_f1"]),
                    "f1_delta": float(reroot_dict["terminal_mst_f1"] - base["terminal_mst_f1"]),
                    "baseline_nodes": int(base["node_count"]),
                    "lifted_nodes": int(reroot_dict["node_count"]),
                    "condition_improvement_ratio": _condition_improvement(base, reroot_dict),
                    "baseline_relR": float(base["relative_error_R"]),
                    "lifted_relR": np.nan,
                    "baseline_relX": float(base["relative_error_X"]),
                    "lifted_relX": np.nan,
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out_dir / "oracle_cluster_lifting_comparison.csv", index=False)

    metrics = {
        "cases": selected_cases,
        "scenario_count": scenario_count,
        "t_count": t_count,
        "recipe": recipe_name,
        "distance_mode": distance_mode,
        "comparison": comparison.to_dict(orient="records"),
        "note": "Clusters are oracle true immediate hidden parents; local_reroot uses pseudo squared voltage reconstructed as root squared voltage minus fitted pseudo-layer linear drop.",
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# Oracle Cluster Lifting Demo\n\n"
        "This demo uses true immediate hidden parents only for clustering. It compares original terminal topology "
        "identification against pseudo-parent topology after P/Q aggregation and voltage lifting. It also tests "
        "a local re-rooting expansion step: each pseudo parent becomes the local voltage reference and each "
        "terminal cluster is re-identified from the reconstructed pseudo squared voltage.\n\n"
        + comparison.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/oracle_cluster_lifting_demo")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--scenario-count", type=int, default=5)
    parser.add_argument("--T", type=int, default=240)
    parser.add_argument("--recipe", default="raw_drop")
    parser.add_argument("--distance-mode", default="RX_75R_25X")
    args = parser.parse_args()
    run(
        output_dir=args.output,
        cases=args.cases,
        scenario_count=args.scenario_count,
        t_count=args.T,
        recipe_name=args.recipe,
        distance_mode=args.distance_mode,
    )


if __name__ == "__main__":
    main()
