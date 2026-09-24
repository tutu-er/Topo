"""Same-condition noisy-AC benchmark of implemented literature baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_latent_tree_diagnostics import _distance_and_depth, _set_score
from experiments.run_matrix_constraint_large_sweep import _simulate_pool, _terminal_buses
from experiments.run_rooted_hierarchical_aggregation import RECIPE
from terminal_case33.data.paper_style_case_bank import CASE_BUILDERS
from terminal_case33.estimation.literature_sensitivity import (
    fit_current_sensitivity,
    fit_partial_meter_impedance,
)
from terminal_case33.estimation.multiscenario import fit_projected_sensitivity, preprocess_scenarios
from terminal_case33.graph.candidate_selection import rank_candidates_by_smart_meter_objectives
from terminal_case33.graph.cl_grouping import cl_grouping
from terminal_case33.graph.enhanced_recursive_grouping import enhanced_recursive_grouping
from terminal_case33.graph.latent_tree import recursive_grouping, terminal_splits
from terminal_case33.graph.literature_registry import literature_method_records
from terminal_case33.graph.partial_meter import insert_interval_meters
from terminal_case33.graph.prufer import soumalas_prufer_reconstruction
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.graph.rooted_neighbor_joining import rooted_neighbor_joining, shared_paths_from_distances
from terminal_case33.models.lin_distflow import impedance_distance_from_reduced_R
from terminal_case33.utils.io import ensure_dir, write_json


def _augment_root(distance: np.ndarray, depth: np.ndarray) -> np.ndarray:
    augmented = np.zeros((len(depth) + 1, len(depth) + 1), dtype=float)
    augmented[:-1, :-1] = distance
    augmented[:-1, -1] = depth
    augmented[-1, :-1] = depth
    return augmented


def _concatenate(scenarios: list[dict], field: str, columns: list[int] | None = None) -> pd.DataFrame:
    frames = []
    for scenario_index, scenario in enumerate(scenarios):
        frame = scenario[field]
        if isinstance(frame, pd.Series):
            frame = frame.to_frame("root")
        if columns is not None:
            frame = frame.loc[:, columns]
        copied = frame.copy()
        copied.index = pd.MultiIndex.from_product([[scenario_index], range(len(copied))])
        frames.append(copied)
    return pd.concat(frames)


def _average_current_fit(scenarios: list[dict], mode: str):
    estimates = [
        fit_current_sensitivity(
            scenario["V_terminal"],
            scenario["P_terminal"],
            scenario["Q_terminal"],
            transformer_mode=mode,
        )
        for scenario in scenarios
    ]
    return (
        np.mean([item.R for item in estimates], axis=0),
        np.mean([item.X for item in estimates], axis=0),
        float(np.mean([item.r2_score for item in estimates])),
        float(np.median([item.condition_number for item in estimates])),
    )


def _score(edges, terminals: list[int], true_splits: set, true_clades: set, root: int) -> dict:
    predicted_splits = terminal_splits(edges, terminals)
    split_precision, split_recall, split_f1 = _set_score(predicted_splits, true_splits)
    try:
        predicted_clades = rooted_clades(edges, root, terminals)
        _, _, rooted_f1 = _set_score(predicted_clades, true_clades)
        exact_rooted_recovery = predicted_clades == true_clades
    except (KeyError, ValueError):
        rooted_f1 = np.nan
        exact_rooted_recovery = False
    return {
        "split_precision": split_precision,
        "split_recall": split_recall,
        "split_f1": split_f1,
        "exact_unrooted_recovery": predicted_splits == true_splits,
        "exact_rooted_recovery": exact_rooted_recovery,
        "rooted_clade_f1": rooted_f1,
    }


def _minimum_span_bank(distance: np.ndarray) -> list[float]:
    positive = distance[distance > 1e-12]
    base = float(np.quantile(positive, 0.1)) if positive.size else 1.0
    return [base / divisor for divisor in (4.0, 3.0, 2.0, 1.0)]


def _run_condition(
    net, terminals: list[int], scenarios: list[dict], max_backtracking_candidates: int | None
) -> list[dict]:
    true_splits = terminal_splits(net.closed_edges(), terminals)
    true_clades = rooted_clades(net.closed_edges(), net.root_bus, terminals)
    fitted = preprocess_scenarios(scenarios, RECIPE)
    r_basic, x_basic, basic_r2, basic_condition = fit_projected_sensitivity(fitted, constraint_mode="basic")
    d_r, depth_r, _ = _distance_and_depth(r_basic, x_basic, "R")
    d_x = impedance_distance_from_reduced_R(x_basic)
    augmented_r = _augment_root(d_r, depth_r)
    _augment_root(d_x, np.diag(x_basic))
    augmented_nodes = [*terminals, net.root_bus]
    rows: list[dict] = []

    soumalas = soumalas_prufer_reconstruction(
        augmented_r,
        augmented_nodes,
        _minimum_span_bank(augmented_r),
    )
    rows.append(
        {
            "method": "soumalas2017_prufer",
            "fit_r2": basic_r2,
            "condition_number": basic_condition,
            "candidate_count": len(soumalas.candidates),
            **_score(soumalas.edges, terminals, true_splits, true_clades, net.root_bus),
        }
    )

    park = recursive_grouping(augmented_r, augmented_nodes, tolerance=0.03)
    rows.append(
        {
            "method": "park2020_recursive_grouping",
            "fit_r2": basic_r2,
            "condition_number": basic_condition,
            "candidate_count": 1,
            **_score(park.edges, terminals, true_splits, true_clades, net.root_bus),
        }
    )

    cl_tree = cl_grouping(augmented_r, augmented_nodes, tolerance=0.03)
    rows.append(
        {
            "method": "choi2011_clgrouping",
            "fit_r2": basic_r2,
            "condition_number": basic_condition,
            "candidate_count": 1,
            **_score(cl_tree.edges, terminals, true_splits, true_clades, net.root_bus),
        }
    )

    all_voltage = _concatenate(scenarios, "V_terminal", terminals)
    all_active = _concatenate(scenarios, "P_terminal", terminals)
    all_reactive = _concatenate(scenarios, "Q_terminal", terminals)
    all_root = _concatenate(scenarios, "root_voltage").iloc[:, 0]
    for method, transformer_mode, enforce in (
        ("pengwah2022_backtracking_rg", "constant", False),
        ("flynn2023_improved_rg", "free_regularized", True),
    ):
        r_current, x_current, current_r2, current_condition = _average_current_fit(scenarios, transformer_mode)
        current_dr = impedance_distance_from_reduced_R(r_current)
        current_dx = impedance_distance_from_reduced_R(x_current)
        current_augmented_r = _augment_root(current_dr, np.diag(r_current))
        current_augmented_x = _augment_root(current_dx, np.diag(x_current))
        scale = max(float(np.median(np.diag(r_current))), 1e-12)
        grouping = enhanced_recursive_grouping(
            current_augmented_r,
            augmented_nodes,
            observed_leaves=set(terminals),
            root=net.root_bus,
            epsilon=1e-5 * scale,
            kappa=10.0,
            max_candidates=max_backtracking_candidates,
            enforce_physical_constraints=enforce,
        )
        selected, objectives = rank_candidates_by_smart_meter_objectives(
            grouping.candidates,
            net.root_bus,
            terminals,
            current_augmented_r,
            current_augmented_x,
            all_voltage,
            all_active,
            all_reactive,
            all_root,
        )
        candidate = grouping.candidates[selected]
        rows.append(
            {
                "method": method,
                "fit_r2": current_r2,
                "condition_number": current_condition,
                "candidate_count": len(grouping.candidates),
                "candidate_objective": objectives[selected].total_score,
                "candidate_generation_truncated": grouping.truncated,
                **_score(candidate.edges, terminals, true_splits, true_clades, net.root_bus),
            }
        )

    # Pengwah 2024 deliberately withholds V/Q at 30% of meters.
    rng = np.random.default_rng(2401)
    shuffled = np.asarray(terminals)[rng.permutation(len(terminals))]
    smart_count = max(2, int(np.ceil(0.7 * len(terminals))))
    smart = sorted(int(node) for node in shuffled[:smart_count])
    interval = sorted(set(terminals) - set(smart))
    if interval:
        partial = fit_partial_meter_impedance(
            all_root,
            all_voltage.loc[:, smart],
            all_active.loc[:, smart],
            all_reactive.loc[:, smart],
            all_active.loc[:, interval],
        )
        d_smart = impedance_distance_from_reduced_R(partial.R_SS)
        augmented_smart = _augment_root(d_smart, np.diag(partial.R_SS))
        smart_nodes = [*smart, net.root_bus]
        smart_tree = recursive_grouping(augmented_smart, smart_nodes, tolerance=0.03)
        full_partial = insert_interval_meters(
            smart_tree.edges,
            net.root_bus,
            smart,
            interval,
            partial.R_SI,
            equality_tolerance=1e-5 * max(float(np.median(np.diag(partial.R_SS))), 1e-12),
        )
        rows.append(
            {
                "method": "pengwah2024_partial_meter",
                "fit_r2": partial.r2_score,
                "condition_number": partial.condition_number,
                "candidate_count": 1,
                "smart_meter_fraction": len(smart) / len(terminals),
                **_score(full_partial.edges, terminals, true_splits, true_clades, net.root_bus),
            }
        )

    mixed_distance, mixed_depth, _ = _distance_and_depth(r_basic, x_basic, "RX_75R_25X")
    rnj = rooted_neighbor_joining(
        shared_paths_from_distances(mixed_distance, mixed_depth),
        mixed_depth,
        terminals,
        net.root_bus,
        0.16 * max(float(np.median(mixed_depth)), 1e-12),
    )
    rows.append(
        {
            "method": "ni2011_rooted_neighbor_joining",
            "fit_r2": basic_r2,
            "condition_number": basic_condition,
            "candidate_count": 1,
            **_score(rnj.edges, terminals, true_splits, true_clades, net.root_bus),
        }
    )
    return rows


def run(
    output_dir: str | Path = "outputs/complete_literature_baselines",
    cases: list[str] | None = None,
    scenario_counts: list[int] | None = None,
    t_counts: list[int] | None = None,
    replicates: int = 1,
    pq_noise_rel: float = 0.005,
    v_noise_rel: float = 0.0002,
    max_backtracking_candidates: int | None = 128,
) -> dict:
    """Run all implemented baselines on identical noisy AC measurements."""

    out_dir = ensure_dir(output_dir)
    selected_cases = cases or list(CASE_BUILDERS)
    selected_scenarios = scenario_counts or [1, 3, 5]
    selected_t = t_counts or [48, 96]
    rows = []
    total = len(selected_cases) * len(selected_scenarios) * len(selected_t) * replicates
    completed = 0
    for case_key in selected_cases:
        for t_count in selected_t:
            for replicate in range(replicates):
                net, pool = _simulate_pool(
                    case_key,
                    t_count,
                    replicate,
                    max(selected_scenarios),
                    pq_noise_rel,
                    v_noise_rel,
                )
                terminals = _terminal_buses(net)
                for scenario_count in selected_scenarios:
                    for result in _run_condition(
                        net, terminals, pool[:scenario_count], max_backtracking_candidates
                    ):
                        rows.append(
                            {
                                "case": case_key,
                                "t_count": t_count,
                                "scenario_count": scenario_count,
                                "replicate": replicate,
                                "samples": t_count * scenario_count,
                                **result,
                            }
                        )
                    completed += 1
                    print(f"[literature] {completed}/{total} {case_key} T={t_count} S={scenario_count}", flush=True)
                    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)
    results = pd.DataFrame(rows)
    summary = (
        results.groupby("method", as_index=False)
        .agg(
            runs=("split_f1", "size"),
            mean_split_f1=("split_f1", "mean"),
            median_split_f1=("split_f1", "median"),
            exact_recovery_rate=("exact_unrooted_recovery", "mean"),
            exact_rooted_recovery_rate=("exact_rooted_recovery", "mean"),
            mean_rooted_clade_f1=("rooted_clade_f1", "mean"),
            mean_fit_r2=("fit_r2", "mean"),
        )
        .sort_values("mean_split_f1", ascending=False)
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    metadata = literature_method_records()
    write_json(out_dir / "method_metadata.json", metadata)
    metrics = {
        "measurement_generation": "radial AC power flow at every time step",
        "pq_noise_relative_std": pq_noise_rel,
        "voltage_noise_relative_std": v_noise_rel,
        "cases": selected_cases,
        "scenario_counts": selected_scenarios,
        "t_counts": selected_t,
        "replicates": replicates,
        "max_backtracking_candidates": max_backtracking_candidates,
        "condition_count": total,
        "summary": summary.to_dict(orient="records"),
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# Complete literature baseline benchmark\n\n"
        "All methods receive the same noisy measurements generated by time-series AC power flow. "
        "P/Q noise is relative to each sample magnitude; voltage noise is relative to nominal voltage magnitude. "
        "The Pengwah 2024 row deliberately exposes only 70% smart meters and treats the rest as interval meters.\n\n"
        + summary.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/complete_literature_baselines")
    parser.add_argument("--cases", nargs="*", choices=list(CASE_BUILDERS), default=None)
    parser.add_argument("--scenario-counts", nargs="*", type=int, default=None)
    parser.add_argument("--T-counts", nargs="*", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--pq-noise-rel", type=float, default=0.005)
    parser.add_argument("--v-noise-rel", type=float, default=0.0002)
    parser.add_argument(
        "--max-backtracking-candidates", type=int, default=128,
        help="0 means exhaustive Pengwah/Flynn backtracking",
    )
    args = parser.parse_args()
    candidate_limit = None if args.max_backtracking_candidates == 0 else args.max_backtracking_candidates
    run(
        args.output,
        args.cases,
        args.scenario_counts,
        args.T_counts,
        args.replicates,
        args.pq_noise_rel,
        args.v_noise_rel,
        candidate_limit,
    )


if __name__ == "__main__":
    main()
