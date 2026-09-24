"""Plot predicted-vs-true terminal MST edge differences for noisy AC data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from experiments.run_delta_v_regularization_test import (
    _preprocess_scenarios,
    _recipe_grid,
    _simulate_noisy_scenarios,
)
from experiments.run_large_sweep_filter_distance import (
    _distance_candidates,
    _fast_delta_v_regularized_multiscenario_fit,
    _fast_projected_multiscenario_fit,
)
from terminal_case33.graph.terminal_tree import terminal_equivalent_minimum_distance_tree
from terminal_case33.graph.metrics import edge_precision_recall_f1
from terminal_case33.utils.io import ensure_dir, write_json


def _edge_set(edges: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """Return sorted undirected edge tuples."""

    return {tuple(sorted((int(u), int(v)))) for u, v in edges}


def _plot_edge_difference(
    *,
    nodes: list[int],
    true_edges: list[tuple[int, int]],
    pred_edges: list[tuple[int, int]],
    output_path: Path,
    title: str,
) -> None:
    """Draw correct, missing, and extra terminal-equivalent MST edges."""

    true_set = _edge_set(true_edges)
    pred_set = _edge_set(pred_edges)
    correct = sorted(true_set & pred_set)
    missing = sorted(true_set - pred_set)
    extra = sorted(pred_set - true_set)

    base_graph = nx.Graph()
    base_graph.add_nodes_from(nodes)
    base_graph.add_edges_from(true_edges)
    overlay = nx.Graph()
    overlay.add_nodes_from(nodes)
    overlay.add_edges_from(sorted(true_set | pred_set))

    pos = nx.spring_layout(base_graph, seed=11, weight=None)
    plt.figure(figsize=(10, 7.5))
    nx.draw_networkx_nodes(overlay, pos, node_size=260, node_color="#f6f7fb", edgecolors="#333333", linewidths=0.8)
    nx.draw_networkx_labels(overlay, pos, font_size=7)
    if correct:
        nx.draw_networkx_edges(overlay, pos, edgelist=correct, edge_color="#2ca25f", width=2.4)
    if missing:
        nx.draw_networkx_edges(
            overlay,
            pos,
            edgelist=missing,
            edge_color="#de2d26",
            width=2.2,
            style="dashed",
            alpha=0.95,
        )
    if extra:
        nx.draw_networkx_edges(
            overlay,
            pos,
            edgelist=extra,
            edge_color="#2b6cb0",
            width=2.0,
            style="dashdot",
            alpha=0.95,
        )

    legend_handles = [
        plt.Line2D([0], [0], color="#2ca25f", lw=2.4, label="correct: true and predicted"),
        plt.Line2D([0], [0], color="#de2d26", lw=2.2, linestyle="--", label="missing: true only"),
        plt.Line2D([0], [0], color="#2b6cb0", lw=2.0, linestyle="-.", label="extra: predicted only"),
    ]
    plt.legend(handles=legend_handles, loc="lower left", fontsize=8, frameon=True)
    plt.title(title)
    plt.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def _save_edges(path: Path, edges: list[tuple[int, int]]) -> None:
    """Save an edge list CSV."""

    pd.DataFrame(sorted(_edge_set(edges)), columns=["u", "v"]).to_csv(path, index=False)


def _evaluate_setting(
    *,
    case_key: str,
    scenario_count: int,
    t_count: int,
    recipe_name: str,
    method: str,
    alpha: float,
    delta_penalty: float,
    distance_mode: str,
    pq_noise_rel: float,
    v_noise_rel: float,
    root_voltage_mean: float,
    output_dir: Path,
) -> dict:
    """Run one noisy topology fit and save true/predicted edge differences."""

    net, terminals, scenarios, true_mst = _simulate_noisy_scenarios(
        case_key=case_key,
        scenario_count=scenario_count,
        t_count=t_count,
        pq_noise_rel=pq_noise_rel,
        v_noise_rel=v_noise_rel,
        root_voltage_mean=root_voltage_mean,
    )
    recipe_lookup = {recipe["name"]: recipe for recipe in _recipe_grid()}
    if recipe_name not in recipe_lookup:
        raise ValueError(f"unknown recipe {recipe_name!r}; choices={sorted(recipe_lookup)}")
    fitted = _preprocess_scenarios(scenarios, recipe_lookup[recipe_name], t_count)
    if method == "standard":
        R_hat, X_hat, r2_score, condition_number = _fast_projected_multiscenario_fit(fitted, alpha=alpha)
        delta_v_rms = 0.0
    elif method == "delta_v_regularized":
        R_hat, X_hat, r2_score, condition_number, delta_v_rms = _fast_delta_v_regularized_multiscenario_fit(
            fitted,
            alpha=alpha,
            delta_penalty=delta_penalty,
        )
    else:
        raise ValueError("method must be 'standard' or 'delta_v_regularized'")

    distances = _distance_candidates(R_hat, X_hat)
    if distance_mode not in distances:
        raise ValueError(f"unknown distance mode {distance_mode!r}; choices={sorted(distances)}")
    pred_mst = terminal_equivalent_minimum_distance_tree(terminals, distances[distance_mode])

    true_set = _edge_set(true_mst)
    pred_set = _edge_set(pred_mst)
    missing = sorted(true_set - pred_set)
    extra = sorted(pred_set - true_set)
    correct = sorted(true_set & pred_set)
    score = edge_precision_recall_f1(pred_mst, true_mst)

    ensure_dir(output_dir)
    _save_edges(output_dir / "true_terminal_mst_edges.csv", true_mst)
    _save_edges(output_dir / "predicted_terminal_mst_edges.csv", pred_mst)
    _save_edges(output_dir / "correct_edges.csv", correct)
    _save_edges(output_dir / "missing_true_edges.csv", missing)
    _save_edges(output_dir / "extra_predicted_edges.csv", extra)
    pd.DataFrame(
        {
            "terminal_bus": terminals,
            "index": list(range(len(terminals))),
            "pd_kw": net.buses.set_index("bus_id").loc[terminals, "pd_kw"].to_numpy()
            * float(net.base_mva)
            * 1000.0,
            "qd_kvar": net.buses.set_index("bus_id").loc[terminals, "qd_kvar"].to_numpy()
            * float(net.base_mva)
            * 1000.0,
        }
    ).to_csv(output_dir / "terminal_nodes.csv", index=False)
    pd.DataFrame(R_hat, index=terminals, columns=terminals).to_csv(output_dir / "R_hat.csv")
    pd.DataFrame(X_hat, index=terminals, columns=terminals).to_csv(output_dir / "X_hat.csv")
    pd.DataFrame(distances[distance_mode], index=terminals, columns=terminals).to_csv(
        output_dir / f"distance_{distance_mode}.csv"
    )

    title = (
        f"{case_key}: {method}, {recipe_name}, {distance_mode}, "
        f"sc={scenario_count}, T={t_count}, F1={float(score['f1']):.3f}"
    )
    _plot_edge_difference(
        nodes=terminals,
        true_edges=true_mst,
        pred_edges=pred_mst,
        output_path=output_dir / "terminal_mst_difference.png",
        title=title,
    )

    summary = {
        "case": case_key,
        "scenario_count": int(scenario_count),
        "t_count": int(t_count),
        "total_samples": int(scenario_count * t_count),
        "recipe": recipe_name,
        "method": method,
        "alpha": float(alpha),
        "delta_penalty": float(delta_penalty) if method == "delta_v_regularized" else None,
        "distance_mode": distance_mode,
        "pq_noise_rel": float(pq_noise_rel),
        "v_noise_rel": float(v_noise_rel),
        "v_noise_percent_of_voltage_magnitude": float(v_noise_rel * 100.0),
        "root_voltage_mean": float(root_voltage_mean),
        "f1": float(score["f1"]),
        "precision": float(score["precision"]),
        "recall": float(score["recall"]),
        "correct_edges": int(len(correct)),
        "missing_edges": int(len(missing)),
        "extra_edges": int(len(extra)),
        "total_true_edges": int(len(true_set)),
        "r2_score": float(r2_score),
        "condition_number": float(condition_number),
        "delta_v_rms": float(delta_v_rms),
        "missing_true_edges": missing,
        "extra_predicted_edges": extra,
        "correct_edge_list": correct,
    }
    write_json(output_dir / "metrics.json", summary)
    pd.DataFrame(
        [
            {
                "edge_type": "correct",
                "u": u,
                "v": v,
            }
            for u, v in correct
        ]
        + [{"edge_type": "missing_true", "u": u, "v": v} for u, v in missing]
        + [{"edge_type": "extra_predicted", "u": u, "v": v} for u, v in extra]
    ).to_csv(output_dir / "edge_difference_summary.csv", index=False)
    return summary


def run(output_dir: str | Path = "outputs/topology_edge_differences") -> dict:
    """Generate representative edge-difference plots for pengwah18."""

    out_dir = ensure_dir(output_dir)
    settings = [
        {
            "name": "pengwah18_sparse_hard",
            "case_key": "pengwah18",
            "scenario_count": 1,
            "t_count": 96,
            "recipe_name": "raw_drop",
            "method": "delta_v_regularized",
            "alpha": 1e-4,
            "delta_penalty": 100.0,
            "distance_mode": "RX_75R_25X",
        },
        {
            "name": "pengwah18_high_data_best",
            "case_key": "pengwah18",
            "scenario_count": 5,
            "t_count": 480,
            "recipe_name": "daily_demean",
            "method": "delta_v_regularized",
            "alpha": 1e-6,
            "delta_penalty": 10.0,
            "distance_mode": "R",
        },
    ]
    rows = []
    for setting in settings:
        case_dir = ensure_dir(out_dir / str(setting["name"]))
        rows.append(
            _evaluate_setting(
                case_key=str(setting["case_key"]),
                scenario_count=int(setting["scenario_count"]),
                t_count=int(setting["t_count"]),
                recipe_name=str(setting["recipe_name"]),
                method=str(setting["method"]),
                alpha=float(setting["alpha"]),
                delta_penalty=float(setting["delta_penalty"]),
                distance_mode=str(setting["distance_mode"]),
                pq_noise_rel=0.005,
                v_noise_rel=0.0002,
                root_voltage_mean=1.02,
                output_dir=case_dir,
            )
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    write_json(out_dir / "metrics.json", {"runs": rows})
    return {"output_dir": str(out_dir), "runs": rows}


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/topology_edge_differences")
    args = parser.parse_args()
    result = run(output_dir=args.output)
    print(pd.DataFrame(result["runs"])[["case", "scenario_count", "t_count", "recipe", "distance_mode", "f1"]])
    print(f"wrote {result['output_dir']}")


if __name__ == "__main__":
    main()
