"""Run the exact laminar L1-MILP estimator on the six-terminal LV case.

The experiment deliberately separates two questions:

* a noiseless squared-voltage LinDistFlow case checks whether the implemented
  regression can recover a matrix that is exactly in the atom model; and
* a robust case adds Laplace response noise and sparse gross outliers before a
  contiguous train/validation/test split.

Both regimes use the same continuous P/Q realization and the same physical
truth.  No truth information is supplied to the estimator.
"""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd

from terminal_case33.data.small_terminal_lv import build_small_terminal_lv_case
from terminal_case33.estimation.laminar_l1_milp import (
    evaluate_l1_matrices,
    fit_laminar_l1_sensitivity,
    solver_diagnostics_prove_optimality,
)
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.models.lin_distflow import build_reduced_sensitivity_matrices
from terminal_case33.utils.io import ensure_dir, write_json


DEFAULT_OUTPUT = Path("outputs/laminar_l1_milp_small6")


def _terminal_labels(net) -> list[int]:
    return (
        net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"]
        .astype(int)
        .tolist()
    )


def _ar1_standardized(
    rng: np.random.Generator, time_count: int, terminal_count: int, rho: float
) -> np.ndarray:
    """Generate smooth, terminal-specific unit-scale excitation."""

    innovations = rng.normal(size=(time_count, terminal_count))
    values = np.zeros_like(innovations)
    values[0] = innovations[0]
    innovation_scale = np.sqrt(1.0 - rho**2)
    for time_index in range(1, time_count):
        values[time_index] = (
            rho * values[time_index - 1]
            + innovation_scale * innovations[time_index]
        )
    values -= values.mean(axis=0, keepdims=True)
    values /= np.maximum(values.std(axis=0, keepdims=True), 1e-12)
    return values


def _continuous_pq(net, time_count: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one physically scaled continuous P/Q trajectory.

    P and Q have separate smooth innovations so that R and X are identifiable;
    a common slow mode keeps the trajectory representative of a feeder profile.
    """

    terminals = _terminal_labels(net)
    terminal_frame = net.buses.set_index("bus_id").loc[terminals]
    base = net.base_mva * 1000.0
    p_nominal = terminal_frame["pd_kw"].to_numpy(dtype=float) / base
    q_nominal = terminal_frame["qd_kvar"].to_numpy(dtype=float) / base
    rng = np.random.default_rng(seed)
    p_local = _ar1_standardized(rng, time_count, len(terminals), rho=0.78)
    q_local = _ar1_standardized(rng, time_count, len(terminals), rho=0.72)
    time = np.arange(time_count, dtype=float)
    phases = np.linspace(0.0, 2.0 * np.pi, len(terminals), endpoint=False)
    common_p = np.sin(2.0 * np.pi * time / time_count)
    common_q = np.cos(2.0 * np.pi * time / time_count + 0.35)
    terminal_wave_p = np.sin(6.0 * np.pi * time[:, None] / time_count + phases)
    terminal_wave_q = np.cos(8.0 * np.pi * time[:, None] / time_count + 0.7 * phases)
    p_multiplier = (
        1.0
        + 0.22 * common_p[:, None]
        + 0.24 * p_local
        + 0.10 * terminal_wave_p
    )
    q_multiplier = (
        1.0
        + 0.18 * common_q[:, None]
        + 0.28 * q_local
        + 0.12 * terminal_wave_q
    )
    p = np.maximum(0.15 * p_nominal[None, :], p_nominal[None, :] * p_multiplier)
    q = np.maximum(0.15 * q_nominal[None, :], q_nominal[None, :] * q_multiplier)
    index = pd.RangeIndex(time_count, name="time_index")
    return (
        pd.DataFrame(p, index=index, columns=terminals),
        pd.DataFrame(q, index=index, columns=terminals),
    )


def _response_regimes(
    exact_target: pd.DataFrame,
    *,
    seed: int,
    outlier_fraction: float,
) -> tuple[dict[str, pd.DataFrame], dict, np.ndarray]:
    """Return noiseless and Laplace-plus-outlier targets on the same timeline."""

    exact = exact_target.to_numpy(dtype=float)
    centered = exact - np.median(exact, axis=0, keepdims=True)
    signal_scale = max(float(np.median(np.abs(centered))), 1e-8)
    laplace_scale = 0.05 * signal_scale
    gross_outlier_scale = 1.5 * signal_scale
    rng = np.random.default_rng(seed)
    laplace = rng.laplace(0.0, laplace_scale, size=exact.shape)
    outlier_count = max(1, int(ceil(outlier_fraction * exact.size)))
    flat_locations = rng.choice(exact.size, size=outlier_count, replace=False)
    outlier_mask = np.zeros(exact.size, dtype=bool)
    outlier_mask[flat_locations] = True
    outlier_mask = outlier_mask.reshape(exact.shape)
    gross = np.zeros_like(exact)
    gross[outlier_mask] = (
        rng.choice(np.array([-1.0, 1.0]), size=outlier_count)
        * gross_outlier_scale
        * (1.0 + rng.exponential(scale=0.5, size=outlier_count))
    )
    contaminated = exact + laplace + gross
    regimes = {
        "noiseless": exact_target.copy(),
        "laplace_outliers": pd.DataFrame(
            contaminated,
            index=exact_target.index,
            columns=exact_target.columns,
        ),
    }
    metadata = {
        "response_noise_only": True,
        "laplace_scale_pu2": laplace_scale,
        "gross_outlier_base_scale_pu2": gross_outlier_scale,
        "outlier_fraction_requested": outlier_fraction,
        "outlier_count": outlier_count,
        "outlier_fraction_realized": outlier_count / exact.size,
        "signal_median_absolute_deviation_pu2": signal_scale,
    }
    return regimes, metadata, outlier_mask


def _scenario_slice(
    p: pd.DataFrame,
    q: pd.DataFrame,
    target: pd.DataFrame,
    selected: slice,
) -> list[dict]:
    return [
        {
            "name": "small6_continuous_scenario",
            "P_terminal": p.iloc[selected].copy(),
            "Q_terminal": q.iloc[selected].copy(),
            "drop_target": target.iloc[selected].copy(),
        }
    ]


def _score_clades(
    predicted: set[frozenset[int]], truth: set[frozenset[int]]
) -> dict:
    matched = predicted & truth
    precision = len(matched) / len(predicted) if predicted else float(not truth)
    recall = len(matched) / len(truth) if truth else float(not predicted)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    serialize = lambda family: [
        list(item) for item in sorted(family, key=lambda value: (len(value), tuple(value)))
    ]
    return {
        "predicted_nontrivial_clade_count": len(predicted),
        "true_nontrivial_clade_count": len(truth),
        "matched_nontrivial_clade_count": len(matched),
        "nontrivial_clade_precision": precision,
        "nontrivial_clade_recall": recall,
        "nontrivial_clade_f1": f1,
        "nontrivial_clade_exact": predicted == truth,
        "matched_nontrivial_clades": serialize(matched),
        "missing_nontrivial_clades": serialize(truth - predicted),
        "extra_nontrivial_clades": serialize(predicted - truth),
    }


def _matrix_errors(estimate: np.ndarray, truth: np.ndarray, prefix: str) -> dict:
    difference = np.asarray(estimate) - np.asarray(truth)
    return {
        f"{prefix}_matrix_mae": float(np.mean(np.abs(difference))),
        f"{prefix}_matrix_max_abs_error": float(np.max(np.abs(difference))),
        f"{prefix}_matrix_relative_frobenius_error": float(
            np.linalg.norm(difference, ord="fro")
            / max(np.linalg.norm(truth, ord="fro"), 1e-15)
        ),
    }


def _heldout_residual_frame(
    heldout: list[dict],
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    outlier_mask: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    scenario = heldout[0]
    p_frame = scenario["P_terminal"]
    q_frame = scenario["Q_terminal"]
    target_frame = scenario["drop_target"]
    prediction = (
        p_frame.to_numpy(dtype=float) @ r_matrix.T
        + q_frame.to_numpy(dtype=float) @ x_matrix.T
    )
    target = target_frame.to_numpy(dtype=float)
    residual = target - prediction
    records = []
    for time_offset, time_label in enumerate(target_frame.index):
        for terminal_offset, terminal in enumerate(target_frame.columns):
            records.append(
                {
                    "time_index": int(time_label),
                    "terminal": int(terminal),
                    "target_pu2": float(target[time_offset, terminal_offset]),
                    "prediction_pu2": float(prediction[time_offset, terminal_offset]),
                    "residual_pu2": float(residual[time_offset, terminal_offset]),
                    "absolute_residual_pu2": float(abs(residual[time_offset, terminal_offset])),
                    "is_injected_outlier": bool(outlier_mask[time_offset, terminal_offset]),
                }
            )
    return pd.DataFrame(records), float(np.mean(np.abs(residual)))


def _truth_atom_frame(net, terminals: list[int]) -> pd.DataFrame:
    """Aggregate physical edge coefficients by downstream terminal support."""

    graph = net.to_networkx_graph()
    oriented = nx.bfs_tree(graph, net.root_bus)
    terminal_set = set(terminals)
    position = {terminal: index for index, terminal in enumerate(terminals)}
    z_base = net.base_kv**2 / net.base_mva
    grouped: dict[tuple[int, ...], dict] = {}
    for parent, child in oriented.edges():
        descendants = {int(child), *(int(node) for node in nx.descendants(oriented, child))}
        support_labels = tuple(sorted(descendants & terminal_set))
        if not support_labels:
            continue
        edge = graph.edges[parent, child]
        record = grouped.setdefault(
            support_labels,
            {
                "support_labels": ",".join(map(str, support_labels)),
                "support_indices": ",".join(str(position[node]) for node in support_labels),
                "support_size": len(support_labels),
                "r_value": 0.0,
                "x_value": 0.0,
                "physical_edges": [],
            },
        )
        record["r_value"] += 2.0 * float(edge["r_ohm"]) / z_base
        record["x_value"] += 2.0 * float(edge["x_ohm"]) / z_base
        record["physical_edges"].append(f"{int(parent)}-{int(child)}")
    rows = []
    for record in grouped.values():
        copied = dict(record)
        copied["physical_edges"] = ";".join(copied["physical_edges"])
        rows.append(copied)
    return pd.DataFrame(rows).sort_values(
        ["support_size", "support_labels"], ignore_index=True
    )


def _path_frame(result, regime: str) -> pd.DataFrame:
    rows = []
    for path_index, point in enumerate(result.path):
        diagnostics = point.solver
        rows.append(
            {
                "regime": regime,
                "path_index": path_index,
                "algorithm_iteration": point.iteration,
                "selected_model": path_index == result.selected_path_index,
                "atom_count": len(point.support_labels),
                "supports": ";".join(
                    ",".join(map(str, support)) for support in point.support_labels
                ),
                "train_mae": point.train_mae,
                "validation_mae": point.validation_mae,
                "validation_se": point.validation_se,
                "accepted_gain": point.accepted_gain,
                "solver_status": diagnostics.status,
                "solver_success": diagnostics.success,
                "solver_message": diagnostics.message,
                "solver_objective": diagnostics.objective,
                "solver_dual_bound": diagnostics.dual_bound,
                "solver_mip_gap": diagnostics.mip_gap,
                "solver_node_count": diagnostics.node_count,
                "solver_runtime_seconds": diagnostics.runtime_seconds,
                "solver_variable_count": diagnostics.variable_count,
                "solver_binary_variable_count": diagnostics.binary_variable_count,
                "solver_constraint_count": diagnostics.constraint_count,
                "solver_certified_optimal": solver_diagnostics_prove_optimality(
                    diagnostics
                ),
            }
        )
    return pd.DataFrame(rows)


def _attempt_frame(result, regime: str) -> pd.DataFrame:
    labels = result.terminal_labels
    rows = []
    for attempt_index, attempt in enumerate(result.attempted_extensions, start=1):
        support_labels = (
            tuple(labels[index] for index in attempt.support)
            if attempt.support is not None
            else ()
        )
        diagnostics = attempt.diagnostics
        rows.append(
            {
                "regime": regime,
                "attempt_index": attempt_index,
                "candidate_support": ",".join(map(str, support_labels)),
                "candidate_support_size": len(support_labels),
                "candidate_r_value": (
                    float(attempt.r_values[-1]) if attempt.r_values.size else np.nan
                ),
                "candidate_x_value": (
                    float(attempt.x_values[-1]) if attempt.x_values.size else np.nan
                ),
                "objective": attempt.objective,
                "status": diagnostics.status,
                "success": diagnostics.success,
                "message": diagnostics.message,
                "dual_bound": diagnostics.dual_bound,
                "mip_gap": diagnostics.mip_gap,
                "node_count": diagnostics.node_count,
                "runtime_seconds": diagnostics.runtime_seconds,
                "variable_count": diagnostics.variable_count,
                "binary_variable_count": diagnostics.binary_variable_count,
                "constraint_count": diagnostics.constraint_count,
                "solver_certified_optimal": solver_diagnostics_prove_optimality(
                    diagnostics
                ),
            }
        )
    return pd.DataFrame(rows)


def _atom_frame(result, regime: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "regime": regime,
                "atom_index": atom_index,
                "support_labels": ",".join(map(str, labels)),
                "support_indices": ",".join(map(str, indices)),
                "support_size": len(indices),
                "r_value": float(r_value),
                "x_value": float(x_value),
                "is_nontrivial_clade": 1 < len(indices) < len(result.terminal_labels),
                "is_singleton": len(indices) == 1,
                "is_full_support": len(indices) == len(result.terminal_labels),
            }
            for atom_index, (indices, labels, r_value, x_value) in enumerate(
                zip(
                    result.support_indices,
                    result.support_labels,
                    result.r_values,
                    result.x_values,
                    strict=True,
                ),
                start=1,
            )
        ]
    )


def run(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    seed: int = 20260902,
    time_count: int = 72,
    train_count: int = 48,
    validation_count: int = 12,
    max_atoms: int = 11,
    outlier_fraction: float = 0.05,
    time_limit: float | None = 120.0,
) -> dict:
    """Execute both small6 regimes and persist all auditable results."""

    if time_count <= 0 or train_count <= 0 or validation_count <= 0:
        raise ValueError("time and split counts must be positive")
    test_count = time_count - train_count - validation_count
    if test_count <= 0:
        raise ValueError("the split must leave a nonempty test suffix")
    if not 0.0 < outlier_fraction < 1.0:
        raise ValueError("outlier_fraction must lie strictly between zero and one")

    out_dir = ensure_dir(output_dir)
    net = build_small_terminal_lv_case()
    terminals = _terminal_labels(net)
    r_true, x_true = build_reduced_sensitivity_matrices(
        net,
        terminals,
        voltage_model="squared-voltage",
    )
    p, q = _continuous_pq(net, time_count, seed)
    exact_target = pd.DataFrame(
        p.to_numpy(dtype=float) @ r_true.T + q.to_numpy(dtype=float) @ x_true.T,
        index=p.index,
        columns=terminals,
    )
    targets, noise_metadata, full_outlier_mask = _response_regimes(
        exact_target,
        seed=seed + 1,
        outlier_fraction=outlier_fraction,
    )

    train_slice = slice(0, train_count)
    validation_slice = slice(train_count, train_count + validation_count)
    test_slice = slice(train_count + validation_count, time_count)
    split_metadata = {
        "same_continuous_scenario": True,
        "train_time_indices": [0, train_count - 1],
        "validation_time_indices": [train_count, train_count + validation_count - 1],
        "test_time_indices": [train_count + validation_count, time_count - 1],
        "train_count": train_count,
        "validation_count": validation_count,
        "test_count": test_count,
    }
    true_clades = set(rooted_clades(net.closed_edges(), net.root_bus, terminals))

    labels = pd.Index(terminals, name="terminal")
    pd.DataFrame(r_true, index=labels, columns=labels).to_csv(out_dir / "R_true.csv")
    pd.DataFrame(x_true, index=labels, columns=labels).to_csv(out_dir / "X_true.csv")
    p.to_csv(out_dir / "P_continuous_pu.csv")
    q.to_csv(out_dir / "Q_continuous_pu.csv")
    exact_target.to_csv(out_dir / "squared_voltage_drop_exact_pu2.csv")
    _truth_atom_frame(net, terminals).to_csv(out_dir / "truth_atoms.csv", index=False)

    regime_metrics: list[dict] = []
    path_frames: list[pd.DataFrame] = []
    attempt_frames: list[pd.DataFrame] = []
    atom_frames: list[pd.DataFrame] = []
    regime_summaries: dict[str, dict] = {}

    for regime, target in targets.items():
        regime_dir = ensure_dir(out_dir / regime)
        target.to_csv(regime_dir / "squared_voltage_drop_observed_pu2.csv")
        train = _scenario_slice(p, q, target, train_slice)
        validation = _scenario_slice(p, q, target, validation_slice)
        test = _scenario_slice(p, q, target, test_slice)

        started = perf_counter()
        result = fit_laminar_l1_sensitivity(
            train,
            validation_scenarios=validation,
            max_atoms=max_atoms,
            r_upper_bound=1.0,
            x_upper_bound=1.0,
            max_bound_expansions=0,
            bound_active_fraction=0.999,
            improvement_abs_tol=1e-12,
            improvement_rel_tol=1e-9,
            zero_tolerance=1e-10,
            validation_blocks=4,
            time_limit=time_limit,
            mip_rel_gap=0.0,
            presolve=True,
            disp=False,
        )
        fit_wall_seconds = perf_counter() - started

        estimated_r = result.r_matrix
        estimated_x = result.x_matrix
        test_mask = (
            full_outlier_mask[test_slice]
            if regime == "laplace_outliers"
            else np.zeros((test_count, len(terminals)), dtype=bool)
        )
        test_residuals, test_mae = _heldout_residual_frame(
            test,
            estimated_r,
            estimated_x,
            test_mask,
        )
        test_scored_mae, test_se, test_block_mae = evaluate_l1_matrices(
            test,
            estimated_r,
            estimated_x,
            blocks_per_scenario=4,
        )
        if not np.isclose(test_mae, test_scored_mae, rtol=1e-12, atol=1e-15):
            raise RuntimeError("held-out residual export disagrees with the core zero-bias scorer")
        predicted_clades = {
            frozenset(int(node) for node in support)
            for support in result.support_labels
            if 1 < len(support) < len(terminals)
        }
        metrics = {
            "regime": regime,
            "selected_path_index": result.selected_path_index,
            "selected_atom_count": len(result.support_labels),
            "path_length": len(result.path),
            "attempted_extension_count": len(result.attempted_extensions),
            "stop_reason": result.stop_reason,
            "train_mae": result.train_mae,
            "validation_mae_zero_bias": result.validation_mae,
            "test_mae_zero_bias": test_mae,
            "test_mae_zero_bias_se": test_se,
            "fit_wall_seconds": fit_wall_seconds,
            "all_extension_attempts_certified_optimal": all(
                solver_diagnostics_prove_optimality(attempt.diagnostics)
                for attempt in result.attempted_extensions
            ),
            "maximum_extension_absolute_gap": max(
                (
                    max(
                        0.0,
                        float(attempt.diagnostics.objective)
                        - float(attempt.diagnostics.dual_bound),
                    )
                    for attempt in result.attempted_extensions
                    if attempt.diagnostics.objective is not None
                    and attempt.diagnostics.dual_bound is not None
                ),
                default=0.0,
            ),
            "maximum_extension_mip_gap": max(
                (
                    float(attempt.diagnostics.mip_gap)
                    for attempt in result.attempted_extensions
                    if attempt.diagnostics.mip_gap is not None
                ),
                default=0.0,
            ),
            "r_upper_bound_final": result.r_upper_bound,
            "x_upper_bound_final": result.x_upper_bound,
            "bound_expansions": result.bound_expansions,
            **_matrix_errors(estimated_r, r_true, "R"),
            **_matrix_errors(estimated_x, x_true, "X"),
            **_score_clades(predicted_clades, true_clades),
        }
        regime_metrics.append(metrics)

        pd.DataFrame(estimated_r, index=labels, columns=labels).to_csv(
            regime_dir / "R_estimated.csv"
        )
        pd.DataFrame(estimated_x, index=labels, columns=labels).to_csv(
            regime_dir / "X_estimated.csv"
        )
        test_residuals.to_csv(regime_dir / "test_residuals.csv", index=False)
        pd.DataFrame(
            {
                "block_index": np.arange(len(test_block_mae)),
                "test_mae_zero_bias": test_block_mae,
            }
        ).to_csv(regime_dir / "test_block_mae.csv", index=False)
        regime_path = _path_frame(result, regime)
        regime_attempts = _attempt_frame(result, regime)
        regime_atoms = _atom_frame(result, regime)
        regime_path.to_csv(regime_dir / "path.csv", index=False)
        regime_attempts.to_csv(regime_dir / "extension_attempts.csv", index=False)
        regime_atoms.to_csv(regime_dir / "selected_atoms.csv", index=False)
        write_json(regime_dir / "metrics.json", metrics)
        write_json(regime_dir / "fit_summary.json", result.summary())
        path_frames.append(regime_path)
        attempt_frames.append(regime_attempts)
        atom_frames.append(regime_atoms)
        regime_summaries[regime] = {
            "fit": result.summary(),
            "metrics": metrics,
        }

    summary_frame = pd.DataFrame(regime_metrics)
    summary_frame.to_csv(out_dir / "summary.csv", index=False)
    pd.concat(path_frames, ignore_index=True).to_csv(out_dir / "all_paths.csv", index=False)
    pd.concat(attempt_frames, ignore_index=True).to_csv(
        out_dir / "all_extension_attempts.csv", index=False
    )
    pd.concat(atom_frames, ignore_index=True).to_csv(
        out_dir / "all_selected_atoms.csv", index=False
    )
    config = {
        "observation_model": "observed_root_zero_bias; historical offset results require their archived source",
        "case": "small_terminal_lv",
        "terminal_labels": terminals,
        "terminal_count": len(terminals),
        "hidden_internal_count": len(net.hidden_buses()),
        "squared_voltage_truth": True,
        "seed": seed,
        "time_count": time_count,
        "split": split_metadata,
        "max_atoms": max_atoms,
        "coefficient_bounds": [1.0, 1.0],
        "solver": "scipy.optimize.milp/HiGHS",
        "mip_rel_gap": 0.0,
        "time_limit_seconds_per_solve": time_limit,
        "noise": noise_metadata,
    }
    output = {
        "config": config,
        "truth_nontrivial_clades": [
            list(item)
            for item in sorted(true_clades, key=lambda value: (len(value), tuple(value)))
        ],
        "regimes": regime_summaries,
    }
    write_json(out_dir / "metrics.json", output)
    report_lines = [
        "# Exact laminar L1-MILP small6 experiment",
        "",
        "One continuous 72-point P/Q scenario is split without shuffling into "
        f"train/validation/test blocks of {train_count}/{validation_count}/{test_count} points.",
        "The truth uses squared-voltage R/X matrices. The robust regime adds "
        "Laplace response noise and sparse gross outliers before splitting.",
        "Coefficient bounds are fixed at (1, 1), and every accepted extension "
        "must pass the strict primal/dual-bound or relative-gap certificate "
        "with zero requested relative MIP gap.",
        "",
        "```text",
        summary_frame.to_string(index=False),
        "```",
        "",
        "`test_mae_zero_bias` is the primary held-out metric. "
        "All predictions use PR + QX without fitting a bias on any split.",
    ]
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--time-count", type=int, default=72)
    parser.add_argument("--train-count", type=int, default=48)
    parser.add_argument("--validation-count", type=int, default=12)
    parser.add_argument("--max-atoms", type=int, default=11)
    parser.add_argument("--outlier-fraction", type=float, default=0.05)
    parser.add_argument("--time-limit", type=float, default=120.0)
    args = parser.parse_args()
    run(
        args.output,
        seed=args.seed,
        time_count=args.time_count,
        train_count=args.train_count,
        validation_count=args.validation_count,
        max_atoms=args.max_atoms,
        outlier_fraction=args.outlier_fraction,
        time_limit=args.time_limit,
    )


if __name__ == "__main__":
    main()
