"""Audit direct R, X, and normalized RX75 scores used by RNJ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from experiments.common import simulate_case, terminal_buses
from terminal_case33.data.norwegian_industrial import load_norwegian_industrial_radial
from terminal_case33.data.paper_style_case_bank import CASE_BUILDERS
from terminal_case33.estimation.multiscenario import fit_projected_sensitivity, preprocess_scenarios
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.graph.rooted_neighbor_joining import rooted_neighbor_joining, shared_paths_from_distances
from terminal_case33.graph.sensitivity_geometry import sensitivity_geometry
from terminal_case33.models.lin_distflow import build_reduced_sensitivity_matrices


CASES = ("paper15", "soumalas11", "flynn16", "pengwah18")
MODES = ("R", "X", "RX_75R_25X")
MODE_LABELS = {"R": "R", "X": "X", "RX_75R_25X": "RX75"}
RECIPE = {"name": "daily_demean", "kind": "demean"}
FIT_CONFIG = {
    "scenario_count": 3,
    "t_count": 96,
    "pq_noise_rel": 0.005,
    "voltage_noise_rel": 0.0002,
    "root_voltage_mean": 1.02,
    "constraint_mode": "ordered",
    "tolerance_factor": 0.16,
}
OBSOLETE_ARTIFACTS = (
    "true_rx_shared_path_heatmaps.png",
    "true_topology_edge_rx.png",
    "true_topology_edge_rx.csv",
    "true_terminal_pair_rx.csv",
    "norwegian_true_rx_shared_path_heatmaps.png",
    "norwegian_true_rx_summary.csv",
    "norwegian_true_edge_rx_bars.png",
    "noisy_ac_direct_rx_performance.png",
)


def _f1(predicted: set, truth: set) -> float:
    if not predicted and not truth:
        return 1.0
    precision = len(predicted & truth) / len(predicted) if predicted else 0.0
    recall = len(predicted & truth) / len(truth) if truth else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _terminal_order(net, terminals: list[int]) -> list[int]:
    terminal_set = set(terminals)
    return [
        int(node)
        for node in nx.dfs_preorder_nodes(net.to_networkx_graph(), net.root_bus)
        if node in terminal_set
    ]


def _reorder(matrix: np.ndarray, terminals: list[int], order: list[int]) -> np.ndarray:
    indices = [terminals.index(node) for node in order]
    return matrix[np.ix_(indices, indices)]


def _pair_rows(case_key: str, source: str, matrices: dict[str, np.ndarray], order: list[int]) -> list[dict]:
    rows: list[dict] = []
    for mode, matrix in matrices.items():
        pairs = [
            (float(matrix[i, j]), int(order[i]), int(order[j]))
            for i in range(len(order))
            for j in range(i + 1, len(order))
        ]
        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        for rank, (value, left, right) in enumerate(pairs, start=1):
            rows.append(
                {
                    "case": case_key,
                    "source": source,
                    "mode": mode,
                    "rank_descending": rank,
                    "terminal_i": left,
                    "terminal_j": right,
                    "shared_path_score": value,
                }
            )
    return rows


def _true_case_data(case_key: str) -> tuple[dict, dict]:
    net = CASE_BUILDERS[case_key]()
    terminals = terminal_buses(net)
    r_matrix, x_matrix = build_reduced_sensitivity_matrices(
        net, terminals, voltage_model="squared-voltage"
    )
    order = _terminal_order(net, terminals)
    r_ordered = _reorder(r_matrix, terminals, order)
    x_ordered = _reorder(x_matrix, terminals, order)
    true_clades = rooted_clades(net.closed_edges(), net.root_bus, terminals)
    mode_f1: dict[str, float] = {}
    equivalence_error: dict[str, float] = {}
    for mode in MODES:
        geometry = sensitivity_geometry(r_matrix, x_matrix, mode)
        legacy_score = shared_paths_from_distances(geometry.distance, geometry.root_depths)
        equivalence_error[mode] = float(np.max(np.abs(geometry.shared_paths - legacy_score)))
        result = rooted_neighbor_joining(
            geometry.shared_paths,
            geometry.root_depths,
            terminals,
            net.root_bus,
            group_tolerance=1e-10,
        )
        mode_f1[mode] = _f1(rooted_clades(result.edges, net.root_bus, terminals), true_clades)
    summary = {
        "case": case_key,
        "terminal_count": len(terminals),
        "true_clade_count": len(true_clades),
        "direct_vs_distance_max_abs_error": max(equivalence_error.values()),
        "R_exact_clade_f1": mode_f1["R"],
        "X_exact_clade_f1": mode_f1["X"],
        "RX75_exact_clade_f1": mode_f1["RX_75R_25X"],
    }
    data = {
        "net": net,
        "terminals": terminals,
        "order": order,
        "true_R": r_ordered,
        "true_X": x_ordered,
        "true_clades": true_clades,
    }
    return summary, data


def _fit_case(case_key: str, true_data: dict) -> tuple[dict, dict[str, np.ndarray]]:
    net, scenarios = simulate_case(
        case_key,
        scenario_count=FIT_CONFIG["scenario_count"],
        t_count=FIT_CONFIG["t_count"],
        pq_noise_rel=FIT_CONFIG["pq_noise_rel"],
        root_voltage_mean=FIT_CONFIG["root_voltage_mean"],
    )
    fitted = preprocess_scenarios(scenarios, RECIPE)
    r_hat, x_hat, r2_score, condition_number = fit_projected_sensitivity(
        fitted, constraint_mode=FIT_CONFIG["constraint_mode"]
    )
    terminals = terminal_buses(net)
    order = true_data["order"]
    r_ordered = _reorder(r_hat, terminals, order)
    x_ordered = _reorder(x_hat, terminals, order)
    # Keep the fitted matrices themselves in the exported figures and tables.
    # Only RX75 is constructed as a normalized linear combination.
    score_matrices = {
        "R": r_ordered,
        "X": x_ordered,
        "RX_75R_25X": sensitivity_geometry(
            r_ordered, x_ordered, "RX_75R_25X"
        ).shared_paths,
    }
    mode_f1 = {}
    for mode in MODES:
        geometry = sensitivity_geometry(r_hat, x_hat, mode)
        depth_scale = max(float(np.median(geometry.root_depths)), 1e-12)
        result = rooted_neighbor_joining(
            geometry.shared_paths,
            geometry.root_depths,
            terminals,
            net.root_bus,
            group_tolerance=FIT_CONFIG["tolerance_factor"] * depth_scale,
        )
        predicted = rooted_clades(result.edges, net.root_bus, terminals)
        mode_f1[mode] = _f1(predicted, true_data["true_clades"])
    summary = {
        "case": case_key,
        "scenario_count": FIT_CONFIG["scenario_count"],
        "samples_per_scenario": FIT_CONFIG["t_count"],
        "preprocessing": RECIPE["name"],
        "constraint_mode": FIT_CONFIG["constraint_mode"],
        "pq_noise_rel": FIT_CONFIG["pq_noise_rel"],
        "voltage_noise_rel": FIT_CONFIG["voltage_noise_rel"],
        "r2_score": r2_score,
        "condition_number": condition_number,
        "relative_error_R": float(
            np.linalg.norm(r_ordered - true_data["true_R"])
            / max(np.linalg.norm(true_data["true_R"]), 1e-15)
        ),
        "relative_error_X": float(
            np.linalg.norm(x_ordered - true_data["true_X"])
            / max(np.linalg.norm(true_data["true_X"]), 1e-15)
        ),
        "R_fitted_clade_f1": mode_f1["R"],
        "X_fitted_clade_f1": mode_f1["X"],
        "RX75_fitted_clade_f1": mode_f1["RX_75R_25X"],
    }
    return summary, score_matrices


def _heatmap_axis(axis, matrix: np.ndarray, order: list[int], title: str, vmax: float, cmap: str):
    image = axis.imshow(matrix, cmap=cmap, vmin=0.0, vmax=max(vmax, 1e-12), interpolation="nearest")
    axis.set_title(title)
    tick_step = max(1, len(order) // 8)
    ticks = np.arange(0, len(order), tick_step)
    labels = [str(order[index]) for index in ticks]
    axis.set_xticks(ticks, labels, rotation=90, fontsize=7)
    axis.set_yticks(ticks, labels, fontsize=7)
    axis.set_xlabel("terminal j")
    axis.set_ylabel("terminal i")
    return image


def _plot_true_heatmaps(case_data: list[tuple[str, dict]], output: Path) -> None:
    figure, axes = plt.subplots(len(case_data), 2, figsize=(9.4, 14.8), constrained_layout=True)
    for row, (case_key, data) in enumerate(case_data):
        vmax = max(float(np.max(data["true_R"])), float(np.max(data["true_X"])), 1e-12)
        panels = (("true_R", "true R [p.u.]", "Blues"), ("true_X", "true X [p.u.]", "Oranges"))
        for column, (key, label, color) in enumerate(panels):
            image = _heatmap_axis(
                axes[row, column], data[key], data["order"], f"{case_key}: {label}", vmax, color
            )
            figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.03)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _plot_fitted_heatmaps(case_data: list[tuple[str, dict, dict]], output: Path) -> None:
    figure, axes = plt.subplots(len(case_data), 3, figsize=(13.8, 14.8), constrained_layout=True)
    for row, (case_key, true_data, fitted) in enumerate(case_data):
        raw_vmax = max(float(np.max(fitted["R"])), float(np.max(fitted["X"])), 1e-12)
        panels = (
            ("R", "fitted R [p.u.]", raw_vmax, "Blues"),
            ("X", "fitted X [p.u.]", raw_vmax, "Oranges"),
            ("RX_75R_25X", "fitted RX75 score", float(np.max(fitted["RX_75R_25X"])), "Greens"),
        )
        for column, (mode, label, vmax, color) in enumerate(panels):
            image = _heatmap_axis(
                axes[row, column], fitted[mode], true_data["order"], f"{case_key}: {label}", vmax, color
            )
            figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.03)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _plot_sorted_pairs(pair_frame: pd.DataFrame, source: str, modes: tuple[str, ...], output: Path) -> None:
    figure, axes = plt.subplots(
        len(CASES), len(modes), figsize=(4.5 * len(modes), 12.8), constrained_layout=True, squeeze=False
    )
    colors = {"R": "tab:blue", "X": "tab:orange", "RX_75R_25X": "tab:green"}
    for row, case_key in enumerate(CASES):
        for column, mode in enumerate(modes):
            axis = axes[row, column]
            selected = pair_frame.loc[
                pair_frame["case"].eq(case_key)
                & pair_frame["source"].eq(source)
                & pair_frame["mode"].eq(mode)
            ].sort_values("rank_descending")
            axis.bar(
                selected["rank_descending"], selected["shared_path_score"],
                width=0.9, color=colors[mode], linewidth=0.0
            )
            axis.set_title(f"{case_key}: {source} {MODE_LABELS[mode]}")
            axis.set_xlabel("terminal-pair rank (descending)")
            axis.set_ylabel("shared-path score")
            axis.grid(axis="y", alpha=0.2)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _assign_true_value_clusters(pair_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group terminal pairs whose true R and X shared-path values are equal."""

    true_values = (
        pair_frame.loc[
            pair_frame["source"].eq("true") & pair_frame["mode"].isin(("R", "X"))
        ]
        .pivot(
            index=["case", "terminal_i", "terminal_j"],
            columns="mode",
            values="shared_path_score",
        )
        .reset_index()
        .rename(columns={"R": "true_R", "X": "true_X"})
    )
    # Rounding only suppresses floating-point summation noise; it does not merge
    # physically different shared-path values at the plotted precision.
    true_values["_r_key"] = true_values["true_R"].round(12)
    true_values["_x_key"] = true_values["true_X"].round(12)
    membership_parts = []
    for case_key in CASES:
        case_values = true_values.loc[true_values["case"].eq(case_key)].copy()
        groups = (
            case_values[["_r_key", "_x_key"]]
            .drop_duplicates()
            .sort_values(["_r_key", "_x_key"], ascending=False)
            .reset_index(drop=True)
        )
        groups["cluster_index"] = np.arange(len(groups), dtype=int)
        groups["cluster_id"] = [f"C{index + 1}" for index in groups["cluster_index"]]
        membership_parts.append(case_values.merge(groups, on=["_r_key", "_x_key"], how="left"))
    membership = pd.concat(membership_parts, ignore_index=True).drop(columns=["_r_key", "_x_key"])
    enriched = pair_frame.merge(
        membership[
            [
                "case",
                "terminal_i",
                "terminal_j",
                "true_R",
                "true_X",
                "cluster_index",
                "cluster_id",
            ]
        ],
        on=["case", "terminal_i", "terminal_j"],
        how="left",
        validate="many_to_one",
    )
    return enriched, membership


def _cluster_styles(membership: pd.DataFrame, case_key: str) -> dict[str, tuple]:
    groups = (
        membership.loc[membership["case"].eq(case_key)]
        .sort_values("cluster_index")
        .drop_duplicates("cluster_id")
    )
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "*")
    styles = {}
    for row in groups.itertuples():
        if np.isclose(row.true_R, 0.0, atol=1e-12) and np.isclose(
            row.true_X, 0.0, atol=1e-12
        ):
            styles[row.cluster_id] = ("0.25", "X")
        else:
            styles[row.cluster_id] = (
                colors[int(row.cluster_index) % len(colors)],
                markers[int(row.cluster_index) % len(markers)],
            )
    return styles


def _cluster_legend_handles(
    membership: pd.DataFrame, case_key: str, styles: dict[str, tuple]
) -> list[Line2D]:
    groups = (
        membership.loc[membership["case"].eq(case_key)]
        .sort_values("cluster_index")
        .drop_duplicates("cluster_id")
    )
    return [
        Line2D(
            [0],
            [0],
            marker=styles[row.cluster_id][1],
            color="none",
            markerfacecolor=styles[row.cluster_id][0],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=6,
            label=f"{row.cluster_id}: true (R,X)=({row.true_R:.4g},{row.true_X:.4g})",
        )
        for row in groups.itertuples()
    ]


def _draw_clustered_pair_bars(
    axis, selected: pd.DataFrame, case_key: str, mode: str, styles: dict[str, tuple]
) -> None:
    """Color and mark fitted bars by equality groups in the true R/X matrices."""

    positions = np.arange(len(selected))
    bar_colors = [styles[cluster][0] for cluster in selected["cluster_id"]]
    axis.bar(
        positions,
        selected["shared_path_score"],
        width=0.9,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.2,
    )
    for cluster_id, cluster_rows in selected.groupby("cluster_id", sort=False):
        indices = cluster_rows.index.to_numpy()
        locations = selected.index.get_indexer(indices)
        color, marker = styles[cluster_id]
        axis.scatter(
            locations,
            cluster_rows["shared_path_score"],
            marker=marker,
            s=13,
            facecolor=color,
            edgecolor="black",
            linewidth=0.25,
            zorder=3,
        )
    axis.set_xlim(-0.8, len(selected) - 0.2)
    axis.set_title(f"{case_key}: fitted {MODE_LABELS[mode]} (descending)")
    axis.set_xlabel("terminal-pair rank")
    axis.set_ylabel("shared-path score")
    axis.grid(axis="y", alpha=0.2)


def _plot_fitted_sorted_pairs_clustered(
    pair_frame: pd.DataFrame, membership: pd.DataFrame, output: Path
) -> None:
    """Plot fitted rankings with true equal-value terminal-pair clusters encoded visually."""

    fitted = pair_frame.loc[pair_frame["source"].eq("fitted")]
    figure, axes = plt.subplots(
        len(CASES), len(MODES), figsize=(18.0, 13.0), constrained_layout=True, squeeze=False
    )
    for row, case_key in enumerate(CASES):
        styles = _cluster_styles(membership, case_key)
        for column, mode in enumerate(MODES):
            selected = fitted.loc[
                fitted["case"].eq(case_key) & fitted["mode"].eq(mode)
            ].sort_values("rank_descending").reset_index(drop=True)
            _draw_clustered_pair_bars(axes[row, column], selected, case_key, mode, styles)
        axes[row, -1].legend(
            handles=_cluster_legend_handles(membership, case_key, styles),
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
            frameon=False,
            title="true equal-value cluster",
            title_fontsize=8,
        )
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _plot_fitted_sorted_pairs_by_case(
    pair_frame: pd.DataFrame, membership: pd.DataFrame, output_dir: Path
) -> None:
    """Export case-specific cluster-colored rankings for detailed inspection."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fitted = pair_frame.loc[pair_frame["source"].eq("fitted")]
    for case_key in CASES:
        case_frame = fitted.loc[fitted["case"].eq(case_key)]
        styles = _cluster_styles(membership, case_key)
        figure, axes = plt.subplots(
            len(MODES), 1, figsize=(14.5, 9.5), constrained_layout=True, squeeze=False
        )
        for row, mode in enumerate(MODES):
            selected = case_frame.loc[case_frame["mode"].eq(mode)].sort_values(
                "rank_descending"
            ).reset_index(drop=True)
            _draw_clustered_pair_bars(axes[row, 0], selected, case_key, mode, styles)
        axes[0, 0].legend(
            handles=_cluster_legend_handles(membership, case_key, styles),
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=8,
            frameon=False,
            title="true equal-value cluster",
            title_fontsize=9,
        )
        figure.savefig(output_dir / f"{case_key}_fitted_R_X_RX75_clustered_bars.png", dpi=220)
        plt.close(figure)


def _build_r_comparison(pair_frame: pd.DataFrame) -> pd.DataFrame:
    """Align true and fitted R scores for the same terminal pairs."""

    comparison = (
        pair_frame.loc[pair_frame["mode"].eq("R")]
        .pivot(
            index=[
                "case",
                "terminal_i",
                "terminal_j",
                "cluster_index",
                "cluster_id",
                "true_R",
                "true_X",
            ],
            columns="source",
            values="shared_path_score",
        )
        .reset_index()
        .rename(columns={"true": "R_true", "fitted": "R_fitted"})
    )
    comparison["R_error"] = comparison["R_fitted"] - comparison["R_true"]
    comparison["R_absolute_error"] = comparison["R_error"].abs()
    comparison["should_be_zero"] = np.isclose(comparison["R_true"], 0.0, atol=1e-12)
    ranked_parts = []
    for case_key in CASES:
        selected = comparison.loc[comparison["case"].eq(case_key)].sort_values(
            ["R_true", "R_fitted", "terminal_i", "terminal_j"],
            ascending=[False, False, True, True],
        ).copy()
        selected["true_R_rank"] = np.arange(1, len(selected) + 1)
        ranked_parts.append(selected)
    return pd.concat(ranked_parts, ignore_index=True)


def _plot_true_vs_fitted_r(
    comparison: pd.DataFrame, membership: pd.DataFrame, output: Path
) -> None:
    """Compare aligned true/fitted R bars and their signed errors."""

    figure, axes = plt.subplots(
        len(CASES), 3, figsize=(22.5, 13.5), constrained_layout=True, squeeze=False
    )
    for row, case_key in enumerate(CASES):
        selected = comparison.loc[comparison["case"].eq(case_key)].sort_values(
            "true_R_rank"
        ).reset_index(drop=True)
        styles = _cluster_styles(membership, case_key)
        positions = np.arange(len(selected))
        colors = [styles[cluster][0] for cluster in selected["cluster_id"]]
        zero_mask = selected["should_be_zero"].to_numpy(dtype=bool)

        comparison_axis = axes[row, 0]
        comparison_axis.bar(
            positions,
            selected["R_true"],
            width=0.92,
            color=colors,
            alpha=0.32,
            edgecolor="black",
            linewidth=0.2,
            label="true R (wide/light)",
        )
        comparison_axis.bar(
            positions,
            selected["R_fitted"],
            width=0.48,
            color=colors,
            alpha=0.95,
            edgecolor="black",
            linewidth=0.25,
            label="fitted R (narrow/dark)",
        )
        for cluster_id, cluster_rows in selected.groupby("cluster_id", sort=False):
            color, marker = styles[cluster_id]
            comparison_axis.scatter(
                cluster_rows.index,
                cluster_rows["R_fitted"],
                marker=marker,
                s=14,
                facecolor=color,
                edgecolor="black",
                linewidth=0.25,
                zorder=3,
            )
        if np.any(zero_mask):
            first_zero = int(np.flatnonzero(zero_mask)[0])
            comparison_axis.axvspan(
                first_zero - 0.5,
                len(selected) - 0.5,
                color="0.75",
                alpha=0.18,
                zorder=0,
            )
        comparison_axis.set_title(f"{case_key}: true vs fitted R, aligned by terminal pair")
        comparison_axis.set_xlabel("terminal-pair rank by true R")
        comparison_axis.set_ylabel("R shared-path score [p.u.]")
        comparison_axis.set_xlim(-0.8, len(selected) - 0.2)
        comparison_axis.grid(axis="y", alpha=0.2)
        comparison_axis.legend(
            handles=[
                Patch(facecolor="0.65", edgecolor="black", alpha=0.32, label="true R: wide/light"),
                Patch(facecolor="0.35", edgecolor="black", label="fitted R: narrow/dark"),
                Patch(facecolor="0.25", edgecolor="black", label="true R = 0 group: gray/black"),
            ],
            frameon=False,
            fontsize=7,
            loc="upper right",
        )

        error_axis = axes[row, 1]
        error_axis.bar(
            positions,
            selected["R_error"],
            width=0.82,
            color=colors,
            edgecolor="black",
            linewidth=0.2,
        )
        error_axis.axhline(0.0, color="black", linewidth=0.8)
        if np.any(zero_mask):
            first_zero = int(np.flatnonzero(zero_mask)[0])
            error_axis.axvspan(
                first_zero - 0.5,
                len(selected) - 0.5,
                color="0.75",
                alpha=0.18,
                zorder=0,
            )
        error_axis.set_title(f"{case_key}: fitted R - true R")
        error_axis.set_xlabel("same terminal-pair order")
        error_axis.set_ylabel("signed R error [p.u.]")
        error_axis.set_xlim(-0.8, len(selected) - 0.2)
        error_axis.grid(axis="y", alpha=0.2)
        error_axis.legend(
            handles=_cluster_legend_handles(membership, case_key, styles),
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=7,
            frameon=False,
            title="true equal-value cluster",
            title_fontsize=8,
        )

        fitted_axis = axes[row, 2]
        fitted_sorted = selected.sort_values(
            ["R_fitted", "R_true", "terminal_i", "terminal_j"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        fitted_positions = np.arange(len(fitted_sorted))
        fitted_colors = [styles[cluster][0] for cluster in fitted_sorted["cluster_id"]]
        fitted_axis.bar(
            fitted_positions,
            fitted_sorted["R_fitted"],
            width=0.82,
            color=fitted_colors,
            edgecolor="black",
            linewidth=0.2,
        )
        for cluster_id, cluster_rows in fitted_sorted.groupby("cluster_id", sort=False):
            color, marker = styles[cluster_id]
            fitted_axis.scatter(
                cluster_rows.index,
                cluster_rows["R_fitted"],
                marker=marker,
                s=14,
                facecolor=color,
                edgecolor="black",
                linewidth=0.25,
                zorder=3,
            )
        fitted_axis.set_title(f"{case_key}: fitted R sorted by fitted magnitude")
        fitted_axis.set_xlabel("terminal-pair rank by fitted R")
        fitted_axis.set_ylabel("fitted R shared-path score [p.u.]")
        fitted_axis.set_xlim(-0.8, len(fitted_sorted) - 0.2)
        fitted_axis.grid(axis="y", alpha=0.2)
        fitted_axis.legend(
            handles=_cluster_legend_handles(membership, case_key, styles),
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=7,
            frameon=False,
            title="true equal-value cluster",
            title_fontsize=8,
        )
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _load_noisy_benchmark(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    selected = frame.loc[
        frame["constraint_mode"].eq("ordered")
        & np.isclose(frame["tolerance_factor"], 0.16)
        & frame["scenario_count"].ge(3)
        & frame["distance_mode"].isin(MODES)
    ].copy()
    overall = selected.groupby("distance_mode", as_index=False).agg(
        runs=("clade_f1", "size"),
        mean_clade_f1=("clade_f1", "mean"),
        mean_sibling_f1=("sibling_f1", "mean"),
        exact_recovery_rate=("clade_f1", lambda values: float(np.mean(np.isclose(values, 1.0)))),
    )
    by_case = selected.groupby(["case", "distance_mode"], as_index=False).agg(
        runs=("clade_f1", "size"), mean_clade_f1=("clade_f1", "mean")
    )
    return overall, by_case


def _plot_noisy_benchmark(overall: pd.DataFrame, by_case: pd.DataFrame, output: Path) -> None:
    labels = ["R", "X", "RX75"]
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    ordered = overall.set_index("distance_mode").loc[list(MODES)]
    axes[0].bar(labels, ordered["mean_clade_f1"], color=["tab:blue", "tab:orange", "tab:green"])
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("mean rooted-clade F1")
    axes[0].set_title("72 noisy-AC runs per score")
    for index, value in enumerate(ordered["mean_clade_f1"]):
        axes[0].text(index, value + 0.025, f"{value:.3f}", ha="center")
    x_axis = np.arange(len(CASES))
    width = 0.24
    for offset, (mode, label, color) in enumerate(
        zip(MODES, labels, ["tab:blue", "tab:orange", "tab:green"])
    ):
        values = (
            by_case.loc[by_case["distance_mode"].eq(mode)]
            .set_index("case")
            .loc[list(CASES), "mean_clade_f1"]
        )
        axes[1].bar(x_axis + (offset - 1) * width, values, width=width, label=label, color=color)
    axes[1].set_xticks(x_axis, CASES, rotation=20)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_ylabel("mean rooted-clade F1")
    axes[1].set_title("same estimator and tolerance, by feeder")
    axes[1].legend(frameon=False)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _norwegian_true_data(data_dir: Path) -> tuple[pd.DataFrame, list[tuple[str, dict]]]:
    rows, case_data = [], []
    for radial in (1, 2):
        net = load_norwegian_industrial_radial(data_dir, radial=radial)
        terminals = net.load_buses()
        r_matrix, x_matrix = build_reduced_sensitivity_matrices(
            net, terminals, voltage_model="squared-voltage"
        )
        order = _terminal_order(net, terminals)
        r_ordered = _reorder(r_matrix, terminals, order)
        x_ordered = _reorder(x_matrix, terminals, order)
        geometry = sensitivity_geometry(r_matrix, x_matrix, "RX_75R_25X")
        legacy = shared_paths_from_distances(geometry.distance, geometry.root_depths)
        rows.append(
            {
                "radial": radial,
                "terminal_count": len(terminals),
                "branch_count": len(net.branches),
                "zero_impedance_branch_count": int(
                    np.sum(np.hypot(net.branches["r_ohm"], net.branches["x_ohm"]) <= 1e-12)
                ),
                "direct_vs_distance_max_abs_error": float(
                    np.max(np.abs(geometry.shared_paths - legacy))
                ),
            }
        )
        case_data.append(
            (f"norwegian-r{radial}", {"order": order, "true_R": r_ordered, "true_X": x_ordered})
        )
    return pd.DataFrame(rows), case_data


def _plot_norwegian_heatmaps(case_data: list[tuple[str, dict]], output: Path) -> None:
    figure, axes = plt.subplots(len(case_data), 2, figsize=(9.8, 8.0), constrained_layout=True)
    for row, (case_key, data) in enumerate(case_data):
        vmax = max(float(np.max(data["true_R"])), float(np.max(data["true_X"])), 1e-12)
        panels = (("true_R", "true R [p.u.]", "Blues"), ("true_X", "true X [p.u.]", "Oranges"))
        for column, (key, label, color) in enumerate(panels):
            image = _heatmap_axis(
                axes[row, column], data[key], data["order"], f"{case_key}: {label}", vmax, color
            )
            figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.03)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _plot_norwegian_edge_bars(data_dir: Path, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    for radial, axis in enumerate(axes, start=1):
        net = load_norwegian_industrial_radial(data_dir, radial=radial)
        z_base = net.base_kv**2 / net.base_mva
        frame = net.branches.copy()
        frame["r_pu"] = frame["r_ohm"] / z_base
        frame["x_pu"] = frame["x_ohm"] / z_base
        frame["magnitude"] = np.hypot(frame["r_pu"], frame["x_pu"])
        zero_count = int(np.sum(frame["magnitude"] <= 1e-12))
        frame = (
            frame.loc[frame["magnitude"].gt(1e-12)]
            .sort_values("magnitude", ascending=False)
            .reset_index(drop=True)
        )
        x_axis = np.arange(len(frame))
        axis.bar(x_axis - 0.2, frame["r_pu"], width=0.4, label="R edge [p.u.]", color="tab:blue")
        axis.bar(x_axis + 0.2, frame["x_pu"], width=0.4, label="X edge [p.u.]", color="tab:orange")
        axis.set_yscale("symlog", linthresh=1e-7)
        axis.set_xlabel(f"positive-impedance branches; {zero_count} ideal zero-Z branches omitted")
        axis.set_ylabel("published branch impedance [p.u.]")
        axis.set_title(f"norwegian-r{radial}: true edge R and X")
        axis.legend(frameon=False)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _plot_norwegian_performance(results_csv: Path, output: Path) -> pd.DataFrame:
    frame = pd.read_csv(results_csv)
    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for radial, color in ((1, "tab:blue"), (2, "tab:orange")):
        selected = frame.loc[frame["radial"].eq(radial)].sort_values("days")
        axis.plot(
            selected["days"], selected["base_f1"], marker="o", linewidth=2, color=color,
            label=f"radial {radial} ({int(selected['terminal_count'].iloc[0])} terminals)",
        )
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("real-load days")
    axis.set_ylabel("identifiable rooted-clade F1")
    axis.set_title("Norwegian published topology + real active load")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return frame


def run(output_dir: Path, benchmark_csv: Path, norwegian_data_dir: Path, norwegian_results_csv: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OBSOLETE_ARTIFACTS:
        candidate = output_dir / name
        if candidate.exists():
            candidate.unlink()
    old_labeled_dir = output_dir / "fitted_pair_bars_labeled"
    for case_key in CASES:
        old_labeled = old_labeled_dir / f"{case_key}_fitted_R_X_RX75_labeled_bars.png"
        if old_labeled.exists():
            old_labeled.unlink()
    if old_labeled_dir.exists() and not any(old_labeled_dir.iterdir()):
        old_labeled_dir.rmdir()

    true_summaries, fit_summaries, pair_rows = [], [], []
    true_cases: list[tuple[str, dict]] = []
    fitted_cases: list[tuple[str, dict, dict]] = []
    for case_key in CASES:
        true_summary, true_data = _true_case_data(case_key)
        fit_summary, fitted = _fit_case(case_key, true_data)
        true_summaries.append(true_summary)
        fit_summaries.append(fit_summary)
        true_cases.append((case_key, true_data))
        fitted_cases.append((case_key, true_data, fitted))
        pair_rows.extend(
            _pair_rows(case_key, "true", {"R": true_data["true_R"], "X": true_data["true_X"]}, true_data["order"])
        )
        pair_rows.extend(_pair_rows(case_key, "fitted", fitted, true_data["order"]))

    true_frame = pd.DataFrame(true_summaries)
    fit_frame = pd.DataFrame(fit_summaries)
    pair_frame, cluster_membership = _assign_true_value_clusters(pd.DataFrame(pair_rows))
    r_comparison = _build_r_comparison(pair_frame)
    overall, by_case = _load_noisy_benchmark(benchmark_csv)
    true_frame.to_csv(output_dir / "true_case_summary.csv", index=False)
    fit_frame.to_csv(output_dir / "representative_fit_summary.csv", index=False)
    pair_frame.to_csv(output_dir / "sorted_terminal_pair_scores.csv", index=False)
    cluster_membership.to_csv(output_dir / "true_value_cluster_membership.csv", index=False)
    r_comparison.to_csv(output_dir / "true_vs_fitted_R_pair_comparison.csv", index=False)
    overall.to_csv(output_dir / "noisy_ac_summary_by_mode.csv", index=False)
    by_case.to_csv(output_dir / "noisy_ac_summary_by_case.csv", index=False)
    _plot_true_heatmaps(true_cases, output_dir / "true_R_X_heatmaps.png")
    _plot_fitted_heatmaps(fitted_cases, output_dir / "fitted_R_X_RX75_heatmaps.png")
    _plot_sorted_pairs(pair_frame, "true", ("R", "X"), output_dir / "true_R_X_sorted_pair_bars.png")
    _plot_fitted_sorted_pairs_clustered(
        pair_frame,
        cluster_membership,
        output_dir / "fitted_R_X_RX75_sorted_pair_bars.png",
    )
    _plot_fitted_sorted_pairs_by_case(
        pair_frame, cluster_membership, output_dir / "fitted_pair_bars_clustered"
    )
    _plot_true_vs_fitted_r(
        r_comparison,
        cluster_membership,
        output_dir / "true_vs_fitted_R_sorted_bars.png",
    )
    _plot_noisy_benchmark(overall, by_case, output_dir / "noisy_ac_R_X_RX75_performance.png")

    norwegian_summary, norwegian_cases = _norwegian_true_data(norwegian_data_dir)
    norwegian_summary.to_csv(output_dir / "norwegian_true_R_X_summary.csv", index=False)
    _plot_norwegian_heatmaps(norwegian_cases, output_dir / "norwegian_true_R_X_heatmaps.png")
    _plot_norwegian_edge_bars(norwegian_data_dir, output_dir / "norwegian_true_edge_R_X_bars.png")
    norwegian_performance = _plot_norwegian_performance(
        norwegian_results_csv, output_dir / "norwegian_real_load_rnj_performance.png"
    )

    metrics = {
        "cases": list(CASES),
        "fit_config": {**FIT_CONFIG, "preprocessing": RECIPE["name"]},
        "maximum_direct_vs_distance_abs_error": float(true_frame["direct_vs_distance_max_abs_error"].max()),
        "exact_true_matrix_results": true_frame.to_dict(orient="records"),
        "representative_fit_results": fit_frame.to_dict(orient="records"),
        "noisy_ac": overall.to_dict(orient="records"),
        "norwegian_true_R_X": norwegian_summary.to_dict(orient="records"),
        "norwegian_real_load": norwegian_performance[
            ["radial", "days", "terminal_count", "base_f1", "exact_identifiable_recovery"]
        ].to_dict(orient="records"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = (
        "# Direct R, X, and RX75 RNJ analysis\n\n"
        "RNJ ranks the direct shared-path score `S_ij = c_R R_ij + c_X X_ij`. "
        "The distance-to-shared-path round trip is retained only as an equivalence check.\n\n"
        "## Exact true matrices\n\n```text\n"
        + true_frame.to_string(index=False)
        + "\n```\n\n## Representative fitted matrices\n\n"
        + "Configuration: ordered fit, daily demeaning, 3 scenarios, 96 samples per scenario, "
        + "0.5% P/Q noise, 0.02% voltage noise, tolerance factor 0.16.\n\n```text\n"
        + fit_frame.to_string(index=False)
        + "\n```\n\n## Existing noisy AC benchmark\n\n```text\n"
        + overall.to_string(index=False)
        + "\n```\n"
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/direct_rx_rnj_analysis"))
    parser.add_argument(
        "--benchmark-csv", type=Path,
        default=Path("outputs/matrix_constraint_large_sweep/topology_results.csv"),
    )
    parser.add_argument(
        "--norwegian-data-dir", type=Path,
        default=Path("data_external/norwegian_industrial_zenodo_7123537"),
    )
    parser.add_argument(
        "--norwegian-results-csv", type=Path,
        default=Path("outputs/norwegian_real_load_identification/identification_results.csv"),
    )
    args = parser.parse_args()
    run(args.output, args.benchmark_csv, args.norwegian_data_dir, args.norwegian_results_csv)


if __name__ == "__main__":
    main()
