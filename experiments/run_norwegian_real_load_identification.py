"""Topology identification with a real Norwegian grid and smart-meter loads."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from experiments.run_latent_tree_diagnostics import _set_score
from experiments.run_rooted_hierarchical_aggregation import _bootstrap_confidence, _fit_rnj
from terminal_case33.data.norwegian_industrial import (
    DEFAULT_DATA_DIR,
    load_norwegian_active_power,
    load_norwegian_industrial_radial,
)
from terminal_case33.estimation.preprocessing import squared_voltage_drop_from_observed_root
from terminal_case33.graph.rooted_hierarchy import (
    aggregate_rooted_scenarios,
    expand_pseudo_clades,
    rooted_clades,
    select_peripheral_clusters,
)
from terminal_case33.models.ac_powerflow import solve_ac_power_flow_timeseries
from terminal_case33.utils.io import ensure_dir, write_json


def _relative_noise(
    frame: pd.DataFrame,
    relative_std: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    positive = np.abs(values[np.abs(values) > 0.0])
    floor = 0.01 * float(np.median(positive)) if positive.size else 1e-12
    scale = relative_std * np.maximum(np.abs(values), floor)
    noise = rng.normal(0.0, scale, values.shape)
    return pd.DataFrame(noise, index=frame.index, columns=frame.columns)


def _identifiable_clades(net, terminals: list[int], atol: float = 1e-12) -> set[frozenset[int]]:
    """Return clades separated from their parent by positive impedance."""

    graph = net.to_networkx_graph()
    directed = nx.bfs_tree(graph, net.root_bus)
    terminal_set = set(terminals)
    clades = set()
    for child, parent in nx.bfs_predecessors(graph, net.root_bus):
        edge = graph.edges[parent, child]
        if float(edge["r_ohm"]) + float(edge["x_ohm"]) <= atol:
            continue
        descendants = set(nx.descendants(directed, child)) | {child}
        clade = frozenset(descendants & terminal_set)
        if 1 < len(clade) < len(terminals):
            clades.add(clade)
    return clades


def _daily_scenarios(
    p_measured: pd.DataFrame,
    v_measured: pd.DataFrame,
    root_voltage: pd.Series,
    days: int,
) -> list[dict]:
    """Build active-only daily scenarios from available measurements."""

    scenarios = []
    for day in range(days):
        selection = slice(24 * day, 24 * (day + 1))
        p_day = p_measured.iloc[selection]
        v_day = v_measured.iloc[selection]
        root_day = root_voltage.iloc[selection]
        q_unavailable = pd.DataFrame(0.0, index=p_day.index, columns=p_day.columns)
        scenarios.append(
            {
                "name": f"day_{day + 1:03d}",
                "P_terminal": p_day,
                "Q_terminal": q_unavailable,
                "V_terminal": v_day,
                "root_voltage": root_day,
                "drop_target": squared_voltage_drop_from_observed_root(v_day, root_day),
            }
        )
    return scenarios


def _tree_positions(graph: nx.Graph, root: int, terminals: set[int]) -> dict[int, tuple[float, float]]:
    directed = nx.bfs_tree(graph, root)
    order = []

    def visit(node: int) -> None:
        children = list(directed.successors(node))
        if not children:
            order.append(node)
        for child in children:
            visit(child)

    visit(root)
    leaf_x = {node: index for index, node in enumerate(order)}
    positions: dict[int, tuple[float, float]] = {}

    def place(node: int, depth: int) -> float:
        children = list(directed.successors(node))
        if not children:
            x_value = float(leaf_x[node])
        else:
            x_value = float(np.mean([place(child, depth + 1) for child in children]))
        positions[node] = (x_value, -float(depth))
        return x_value

    place(root, 0)
    return positions


def _plot_comparison(net, inferred_edges: tuple[tuple[int, int, float], ...], path: Path) -> None:
    terminals = set(net.load_buses())
    true_graph = net.to_networkx_graph()
    inferred = nx.Graph()
    inferred.add_edges_from((int(left), int(right)) for left, right, _ in inferred_edges)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for axis, graph, title in [
        (axes[0], true_graph, "Published physical radial"),
        (axes[1], inferred, "RNJ inferred metric tree"),
    ]:
        positions = _tree_positions(graph, net.root_bus, terminals)
        colors = [
            "#c0392b" if node == net.root_bus else "#2e8b57" if node in terminals else "#4c78a8"
            for node in graph.nodes()
        ]
        nx.draw_networkx_edges(graph, positions, ax=axis, width=1.0, alpha=0.75)
        nx.draw_networkx_nodes(graph, positions, ax=axis, node_color=colors, node_size=42)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"Norwegian industrial radial {net.metadata['radial']}")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = "outputs/norwegian_real_load_identification",
    radials: list[int] | None = None,
    day_counts: list[int] | None = None,
    replicates: int = 5,
    seed: int = 20260711,
) -> dict:
    """Run active-only identification on both published Norwegian radials."""

    out_dir = ensure_dir(output_dir)
    selected_radials = radials or [1, 2]
    selected_days = sorted(day_counts or [7, 30, 90])
    maximum_days = max(selected_days)
    rng = np.random.default_rng(seed)
    result_rows = []
    clade_rows = []
    provenance_rows = []

    for radial in selected_radials:
        net = load_norwegian_industrial_radial(data_dir, radial=radial)
        terminals = net.load_buses()
        p_true = load_norwegian_active_power(
            net,
            data_dir=data_dir,
            periods=24 * maximum_days,
        ).loc[:, terminals]
        q_true = p_true * np.tan(np.arccos(0.95))
        sample = np.arange(len(p_true))
        common_mode = 1.02 + 0.0015 * np.sin(2.0 * np.pi * sample / 24.0)
        root_voltage = pd.Series(common_mode, index=p_true.index, name="V_root_pu")
        power_flow = solve_ac_power_flow_timeseries(net, p_true, q_true, root_voltage)
        if not bool(power_flow["converged"].all()):
            raise RuntimeError(f"AC power flow failed for radial {radial}")
        v_true = power_flow["V_bus_mag"].loc[:, terminals]
        p_measured = p_true + _relative_noise(p_true, 0.005, rng)
        v_measured = v_true + _relative_noise(v_true, 0.0002, rng)

        physical_clades = rooted_clades(net.closed_edges(), net.root_bus, terminals)
        identifiable_clades = _identifiable_clades(net, terminals)
        exact_metric_ceiling = _set_score(identifiable_clades, physical_clades)
        longest_tree = None

        for days in selected_days:
            scenarios = _daily_scenarios(p_measured, v_measured, root_voltage, days)
            r_matrix, x_matrix, base_tree = _fit_rnj(
                scenarios,
                terminals,
                net.root_bus,
                tolerance_factor=0.16,
            )
            base_clades = rooted_clades(base_tree.edges, net.root_bus, terminals)
            clade_confidence, sibling_confidence = _bootstrap_confidence(
                scenarios,
                terminals,
                net.root_bus,
                base_clades,
                set(),
                tolerance_factor=0.16,
                replicates=replicates,
                seed=seed + 1000 * radial + days,
            )
            clusters = select_peripheral_clusters(
                terminals,
                clade_confidence,
                sibling_confidence,
                threshold=0.9,
                max_cluster_size=5,
            )
            inferred_tree = base_tree
            inferred_clades = base_clades
            if clusters:
                pseudo_scenarios, pseudo_members = aggregate_rooted_scenarios(
                    scenarios,
                    terminals,
                    r_matrix,
                    x_matrix,
                    clusters,
                    voltage_mode="deembedded_vsq",
                    deembedding_weight=0.25,
                )
                _, _, inferred_tree = _fit_rnj(
                    pseudo_scenarios,
                    list(pseudo_members),
                    net.root_bus,
                    tolerance_factor=0.16,
                )
                inferred_clades = expand_pseudo_clades(
                    inferred_tree.edges,
                    net.root_bus,
                    pseudo_members,
                    clusters,
                )

            base_score = _set_score(base_clades, identifiable_clades)
            aggregate_score = _set_score(inferred_clades, identifiable_clades)
            result_rows.append(
                {
                    "radial": radial,
                    "days": days,
                    "samples": 24 * days,
                    "terminal_count": len(terminals),
                    "physical_clade_count": len(physical_clades),
                    "identifiable_clade_count": len(identifiable_clades),
                    "base_predicted_clades": len(base_clades),
                    "base_precision": base_score[0],
                    "base_recall": base_score[1],
                    "base_f1": base_score[2],
                    "aggregate_predicted_clades": len(inferred_clades),
                    "aggregate_precision": aggregate_score[0],
                    "aggregate_recall": aggregate_score[1],
                    "aggregate_f1": aggregate_score[2],
                    "cluster_count": len(clusters),
                    "exact_identifiable_recovery": inferred_clades == identifiable_clades,
                    "minimum_voltage_pu": float(v_true.iloc[: 24 * days].min().min()),
                    "maximum_voltage_drop_pu": float(
                        (root_voltage.iloc[: 24 * days] - v_true.iloc[: 24 * days].min(axis=1)).max()
                    ),
                    "physical_f1_ceiling_from_exact_metric": exact_metric_ceiling[2],
                }
            )
            for kind, collection in [
                ("physical", physical_clades),
                ("identifiable", identifiable_clades),
                ("base", base_clades),
                ("aggregated", inferred_clades),
            ]:
                for clade in collection:
                    clade_rows.append(
                        {
                            "radial": radial,
                            "days": days,
                            "kind": kind,
                            "size": len(clade),
                            "members": " ".join(map(str, sorted(clade))),
                        }
                    )
            if days == maximum_days:
                longest_tree = base_tree

        p_kw = p_true * 1000.0 * net.base_mva
        provenance_rows.append(
            {
                "radial": radial,
                "published_bus_count_after_pruning": len(net.buses),
                "published_branch_count_after_pruning": len(net.branches),
                "terminal_meter_count": len(terminals),
                "data_start": str(p_true.index.min()),
                "data_end": str(p_true.index.max()),
                "mean_total_load_kw": float(p_kw.sum(axis=1).mean()),
                "peak_total_load_kw": float(p_kw.sum(axis=1).max()),
                "pruned_unmetered_dead_ends": " ".join(net.metadata["pruned_unmetered_dead_ends"]),
            }
        )
        if longest_tree is not None:
            _plot_comparison(
                net,
                longest_tree.edges,
                out_dir / f"radial_{radial}_topology_comparison.png",
            )

    results = pd.DataFrame(result_rows)
    results.to_csv(out_dir / "identification_results.csv", index=False)
    pd.DataFrame(clade_rows).to_csv(out_dir / "clade_details.csv", index=False)
    pd.DataFrame(provenance_rows).to_csv(out_dir / "data_summary.csv", index=False)
    metrics = {
        "dataset": "Norwegian industrial MV/LV distribution system",
        "doi": "10.5281/zenodo.10361330",
        "license": "CC BY 4.0",
        "validation_class": "semi-empirical: real topology and active load; synthetic Q/root V; AC-generated terminal V",
        "active_power_measurement_noise_relative_std": 0.005,
        "voltage_measurement_noise_relative_to_local_voltage_magnitude_std": 0.0002,
        "assumed_power_factor_for_ac_flow": 0.95,
        "estimator_measurements": "real active power plus AC-generated noisy voltage; Q omitted",
        "results": results.to_dict(orient="records"),
        "bootstrap_replicates": int(replicates),
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# Norwegian real-load topology identification\n\n"
        "This is a semi-empirical validation. Published topology, branch R/X, and hourly active "
        "smart-meter load are real/anonymized source data. Q is completed with pf=0.95, root voltage "
        "is synthetic, and terminal voltage is generated by the radial AC solver. The estimator uses "
        "only active power and voltage because measured Q/V are not supplied.\n\n"
        "Zero-impedance ideal transformer/connection edges do not induce distinguishable impedance "
        "splits. Scores therefore use positive-impedance identifiable rooted clades; physical-clade "
        "counts and the corresponding exact-metric F1 ceiling are reported separately.\n\n"
        + results.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output", default="outputs/norwegian_real_load_identification")
    parser.add_argument("--radials", nargs="*", type=int, default=None)
    parser.add_argument("--days", nargs="*", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()
    run(
        data_dir=args.data_dir,
        output_dir=args.output,
        radials=args.radials,
        day_counts=args.days,
        replicates=args.replicates,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
