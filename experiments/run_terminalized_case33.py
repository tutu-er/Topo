"""Run terminal-load-only case33 experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from terminal_case33.data.case33bw_raw import load_raw_case33bw
from terminal_case33.estimation.delay import integer_delay_correction
from terminal_case33.estimation.sensitivity import estimate_reduced_sensitivity
from terminal_case33.graph.matrix_tree import build_edge_scores_from_distances, matrix_tree_edge_marginals
from terminal_case33.graph.metrics import relative_error_R, relative_error_X, rmse_R, rmse_X
from terminal_case33.models.lin_distflow import (
    build_reduced_sensitivity_matrices,
    impedance_distance_from_reduced_R,
    impedance_distance_from_reduced_X,
)
from terminal_case33.models.network import TerminalizedNetwork
from terminal_case33.models.timeseries import simulate_terminal_load_timeseries
from terminal_case33.scenario.scenario_config import ScenarioConfig, load_config
from terminal_case33.scenario.terminalize import terminalize_case33
from terminal_case33.scenario.validate_scenario import validate_terminal_load_only_scenario
from terminal_case33.utils.io import ensure_dir, write_json
from terminal_case33.visualization.plots import plot_edge_marginal_hist, plot_heatmap, plot_network


def _plot_raw_topology(path: Path, title: str = "raw case33") -> None:
    """Plot the closed raw case33 backbone."""

    raw = load_raw_case33bw(include_tie_lines=False)
    graph = nx.Graph()
    graph.add_nodes_from(raw.buses["bus_id"].astype(int))
    graph.add_edges_from((int(r.from_bus), int(r.to_bus)) for r in raw.branches[raw.branches["status"]].itertuples())
    pos = nx.spring_layout(graph, seed=1)
    colors = ["#d62728" if node == raw.root_bus else "#8c8c8c" for node in graph.nodes]
    plt.figure(figsize=(10, 7))
    nx.draw_networkx(graph, pos=pos, node_color=colors, node_size=120, font_size=7, width=0.9)
    plt.title(title)
    plt.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _write_not_applicable_outputs(out_dir: Path) -> None:
    """Write empty sensitivity and graph files for non-estimation experiments."""

    empty = pd.DataFrame()
    for name in ["R_true", "X_true", "R_hat", "X_hat", "dR", "dX", "edge_marginals", "map_tree_edges"]:
        empty.to_csv(out_dir / f"{name}.csv", index=False)
    plot_heatmap(np.zeros((1, 1)), out_dir / "sensitivity_heatmap.png", "not applicable")
    plot_edge_marginal_hist(pd.DataFrame({"edge_marginal": []}), out_dir / "edge_marginal_hist.png")


def _raw_summary(out_dir: Path) -> dict:
    """Run Experiment A and report why raw case33 is not terminal-load-only."""

    raw = load_raw_case33bw(include_tie_lines=True)
    closed = raw.branches[raw.branches["status"]]
    graph = nx.Graph()
    graph.add_nodes_from(raw.buses["bus_id"].astype(int))
    graph.add_edges_from((int(r.from_bus), int(r.to_bus)) for r in closed.itertuples())
    leaves = {n for n, d in graph.degree() if d == 1 and n != raw.root_bus}
    nonleaf_loads = raw.buses[
        (~raw.buses["bus_id"].isin(leaves | {raw.root_bus}))
        & ((raw.buses["pd_kw"] != 0.0) | (raw.buses["qd_kvar"] != 0.0))
    ]
    summary = {
        "bus_count": int(len(raw.buses)),
        "closed_branch_count": int(len(closed)),
        "connected": nx.is_connected(graph),
        "acyclic": nx.is_tree(graph),
        "total_pd_kw": float(raw.buses["pd_kw"].sum()),
        "total_qd_kvar": float(raw.buses["qd_kvar"].sum()),
        "raw_leaf_count": int(len(leaves)),
        "nonleaf_load_count": int(len(nonleaf_loads)),
        "nonleaf_load_buses": nonleaf_loads["bus_id"].astype(int).tolist(),
        "terminal_load_only": False,
        "reason": "raw case33 places loads on many non-leaf internal buses",
    }
    write_json(out_dir / "metrics.json", summary)
    write_json(out_dir / "scenario_summary.json", summary)
    _plot_raw_topology(out_dir / "topology_original.png", "raw case33 closed backbone")
    _plot_raw_topology(out_dir / "topology_terminalized.png", "raw case33 is not terminalized")
    _write_not_applicable_outputs(out_dir)
    (out_dir / "report.md").write_text(
        "# Experiment A: raw_case33_check\n\n"
        f"Raw case33 has {summary['nonleaf_load_count']} non-leaf load buses, so it does not satisfy the terminal-load-only definition.\n",
        encoding="utf-8",
    )
    return summary


def _make_net(config: ScenarioConfig, *, mode: str | None = None, service_mode: str | None = None, leakage: float | None = None) -> TerminalizedNetwork:
    """Build a terminalized network from the current scenario config."""

    del leakage
    raw = load_raw_case33bw(include_tie_lines=False)
    return terminalize_case33(
        raw,
        mode=mode or config.mode,
        service_impedance_mode=service_mode or config.service_impedance_mode,
        service_length_m=config.service_length_m,
        service_length_random_range_m=config.service_length_random_range_m,
        seed=config.seed,
    )


def _save_common_outputs(out_dir: Path, net: TerminalizedNetwork, summary: dict) -> None:
    """Save network, scenario, and topology outputs shared by most experiments."""

    write_json(out_dir / "scenario_summary.json", summary)
    net.buses.to_csv(out_dir / "buses.csv", index=False)
    net.branches.to_csv(out_dir / "branches.csv", index=False)
    _plot_raw_topology(out_dir / "topology_original.png", "raw case33 closed backbone")
    plot_network(net, out_dir / "topology_terminalized.png", "terminalized case33")


def _run_sensitivity_pipeline(
    out_dir: Path,
    net: TerminalizedNetwork,
    config: ScenarioConfig,
    *,
    profile_mode: str | None = None,
    pf_mode: str | None = None,
    noise_config: dict | None = None,
    v0_config: dict | None = None,
    delay_config: dict | None = None,
    include_root_voltage_mode: bool = False,
    hidden_load_leakage: float = 0.0,
) -> dict:
    """Run the time-series, sensitivity-estimation, and matrix-tree pipeline."""

    terminals = net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"].astype(int).tolist()
    R_true, X_true = build_reduced_sensitivity_matrices(net, terminals)
    sim = simulate_terminal_load_timeseries(
        net,
        T=config.T,
        dt_seconds=config.dt_seconds,
        profile_mode=profile_mode or config.profile_mode,
        pf_mode=pf_mode or config.pf_mode,
        noise_config=noise_config if noise_config is not None else config.noise_config,
        v0_config=v0_config if v0_config is not None else config.v0_config,
        delay_config=delay_config if delay_config is not None else config.delay_config,
        hidden_load_leakage=hidden_load_leakage,
        seed=config.seed,
    )
    P = sim["P_terminal_meas"]
    Q = sim["Q_terminal_meas"]
    if delay_config and delay_config.get("max_delay_steps", 0) > 0:
        P = integer_delay_correction(sim["V_terminal_meas"], P, int(delay_config["max_delay_steps"]))
        Q = integer_delay_correction(sim["V_terminal_meas"], Q, int(delay_config["max_delay_steps"]))
    est = estimate_reduced_sensitivity(
        sim["V_terminal_meas"],
        P,
        Q,
        method="ridge",
        preprocess="difference",
        include_root_voltage_mode=include_root_voltage_mode,
        root_voltage=sim["V_root_true"],
        alpha=1e-10,
    )
    dR = impedance_distance_from_reduced_R(R_true)
    dX = impedance_distance_from_reduced_X(X_true)
    dR_hat = impedance_distance_from_reduced_R(est.R_hat.to_numpy())
    scores = build_edge_scores_from_distances(terminals, dR_hat, impedance_distance_from_reduced_X(est.X_hat.to_numpy()), beta=1.0)
    weights = {(int(r.u), int(r.v)): float(r.weight) for r in scores.itertuples()}
    posterior = matrix_tree_edge_marginals(terminals, [(int(r.u), int(r.v)) for r in scores.itertuples()], weights)

    pd.DataFrame(R_true, index=terminals, columns=terminals).to_csv(out_dir / "R_true.csv")
    pd.DataFrame(X_true, index=terminals, columns=terminals).to_csv(out_dir / "X_true.csv")
    est.R_hat.to_csv(out_dir / "R_hat.csv")
    est.X_hat.to_csv(out_dir / "X_hat.csv")
    pd.DataFrame(dR, index=terminals, columns=terminals).to_csv(out_dir / "dR.csv")
    pd.DataFrame(dX, index=terminals, columns=terminals).to_csv(out_dir / "dX.csv")
    posterior.edge_marginals.to_csv(out_dir / "edge_marginals.csv", index=False)
    pd.DataFrame(posterior.map_tree, columns=["u", "v"]).to_csv(out_dir / "map_tree_edges.csv", index=False)
    plot_heatmap(R_true, out_dir / "sensitivity_heatmap.png", "R true")
    plot_edge_marginal_hist(posterior.edge_marginals, out_dir / "edge_marginal_hist.png")
    metrics = {
        "rmse_R": rmse_R(est.R_hat.to_numpy(), R_true),
        "rmse_X": rmse_X(est.X_hat.to_numpy(), X_true),
        "relative_error_R": relative_error_R(est.R_hat.to_numpy(), R_true),
        "relative_error_X": relative_error_X(est.X_hat.to_numpy(), X_true),
        "condition_number": est.condition_number,
        "r2_score": est.r2_score,
        "sum_marginals": posterior.sum_marginals,
        "logZ": posterior.logZ,
        "measurement_generation": "nonlinear radial AC power flow; R_true/X_true are reduced LinDistFlow reference sensitivities",
        "terminal_equivalent_note": "MAP tree over observed terminals is not the full physical hidden-node feeder",
        "hidden_load_leakage_violation": hidden_load_leakage > 0.0,
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# Sensitivity and graph-learning report\n\n"
        f"- terminals: {len(terminals)}\n"
        f"- relative_error_R: {metrics['relative_error_R']:.6g}\n"
        f"- relative_error_X: {metrics['relative_error_X']:.6g}\n"
        f"- sum_marginals: {metrics['sum_marginals']:.6g}\n\n"
        "Terminal-only MAP trees are terminal-equivalent trees, not complete physical trees.\n",
        encoding="utf-8",
    )
    return metrics


def _terminalize_experiment(out_dir: Path, config: ScenarioConfig) -> dict:
    """Run Experiment B and save the terminalized feeder without estimation."""

    net = _make_net(config)
    summary = validate_terminal_load_only_scenario(net, strict=True, output_path=out_dir / "scenario_summary.json")
    _save_common_outputs(out_dir, net, summary)
    write_json(out_dir / "network_summary.json", {**summary, **net.metadata})
    write_json(out_dir / "metrics.json", summary)
    _write_not_applicable_outputs(out_dir)
    (out_dir / "report.md").write_text(
        "# Experiment B: terminalize_hybrid_leaf\n\nAll non-root loads are on graph leaves after terminalization.\n",
        encoding="utf-8",
    )
    return summary


def _wrong_attachment(out_dir: Path, config: ScenarioConfig) -> dict:
    """Run Experiment G and attach diagnostic metadata for a swapped meter label."""

    net = _make_net(config)
    summary = validate_terminal_load_only_scenario(net, strict=True, output_path=out_dir / "scenario_summary.json")
    _save_common_outputs(out_dir, net, summary)
    terminals = net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"].astype(int).tolist()
    metrics = _run_sensitivity_pipeline(out_dir, net, config, noise_config=config.noise_config)
    if len(terminals) >= 2:
        metrics["wrong_attachment_simulated"] = {"from": terminals[0], "to": terminals[1], "note": "first two terminal labels swapped in diagnostic metadata"}
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def run(config_path: str | Path) -> None:
    """Run configured experiments and write outputs."""

    config = load_config(config_path)
    output_root = ensure_dir(Path("outputs"))
    for name in config.experiments:
        out_dir = ensure_dir(output_root / name)
        if name == "raw_case33_check":
            _raw_summary(out_dir)
        elif name == "terminalize_hybrid_leaf":
            _terminalize_experiment(out_dir, config)
        elif name == "ideal_reduced_sensitivity":
            net = _make_net(config)
            summary = validate_terminal_load_only_scenario(net, strict=True, output_path=out_dir / "scenario_summary.json")
            _save_common_outputs(out_dir, net, summary)
            _run_sensitivity_pipeline(
                out_dir,
                net,
                config,
                profile_mode="step_probe",
                pf_mode="time_varying_pf",
                noise_config={"p_std": 0.0, "q_std": 0.0, "v_std": 0.0},
                v0_config={"sigma": 0.0},
                delay_config={"max_delay_steps": 0},
                include_root_voltage_mode=False,
            )
        elif name == "noise_v0_delay":
            net = _make_net(config)
            summary = validate_terminal_load_only_scenario(net, strict=True, output_path=out_dir / "scenario_summary.json")
            _save_common_outputs(out_dir, net, summary)
            _run_sensitivity_pipeline(
                out_dir,
                net,
                config,
                profile_mode="correlated_residential",
                noise_config={"p_std": 0.0003, "q_std": 0.0003, "v_std": 0.0007},
                v0_config={"sigma": 0.002},
                delay_config={"max_delay_steps": 2},
                include_root_voltage_mode=True,
            )
        elif name == "soumalas_style_lv":
            net = _make_net(config, mode="all_loads_to_new_terminal_leaves", service_mode="lv_cable_soumalas_style")
            summary = validate_terminal_load_only_scenario(net, strict=True, output_path=out_dir / "scenario_summary.json")
            _save_common_outputs(out_dir, net, summary)
            _run_sensitivity_pipeline(out_dir, net, config, profile_mode="independent_residential", include_root_voltage_mode=True)
        elif name == "hidden_leakage_violation":
            net = _make_net(config)
            summary = validate_terminal_load_only_scenario(net, strict=True, output_path=out_dir / "scenario_summary.json")
            _save_common_outputs(out_dir, net, summary)
            _run_sensitivity_pipeline(out_dir, net, config, hidden_load_leakage=0.03, include_root_voltage_mode=True)
        elif name == "wrong_attachment":
            _wrong_attachment(out_dir, config)
        else:
            raise ValueError(f"unknown experiment: {name}")


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/terminalized_case33_default.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

