"""Run auditable laminar L1-MILP integration cases derived from IEEE case33.

Two terminal-load-only variants are exercised:

* ``aggregate_leaves`` keeps the four physical leaves as terminals.  Its small
  MILPs are solved along a multi-atom forward path and serve as an end-to-end
  exact-status integration case.
* ``hybrid_full`` uses the standard hybrid terminalization with all 32 loads
  observed at leaves.  By default only one extension is attempted under a
  finite time limit, making the combinatorial scaling limit explicit.

The data-generating network is used only to synthesize the response and to
compute post-fit diagnostics.  It is never used to construct candidate
supports: every extension MILP searches all nonempty terminal subsets that are
laminar with the already selected supports.

Training, validation and test use the observed-root zero-bias model directly:
squared-voltage drop = P R^T + Q X^T. Historical intercept-model results require
their archived source and are not overwritten by this protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd
import scipy

from terminal_case33.data.case33bw_raw import load_raw_case33bw
from terminal_case33.estimation.laminar_l1_milp import (
    LaminarL1PathPoint,
    evaluate_l1_matrices,
    fit_laminar_l1_sensitivity,
    is_laminar_family,
    solver_diagnostics_prove_optimality,
)
from terminal_case33.models.lin_distflow import build_reduced_sensitivity_matrices
from terminal_case33.scenario.terminalize import terminalize_case33
from terminal_case33.utils.io import ensure_dir


DEFAULT_OUTPUT = Path("outputs/laminar_l1_milp_case33_integration")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _terminal_labels(net) -> list[int]:
    return (
        net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"]
        .astype(int)
        .tolist()
    )


def _ar1_standardized(
    rng: np.random.Generator,
    sample_count: int,
    terminal_count: int,
    rho: float,
) -> np.ndarray:
    innovations = rng.normal(size=(sample_count, terminal_count))
    values = np.zeros_like(innovations)
    values[0] = innovations[0]
    innovation_scale = np.sqrt(1.0 - rho**2)
    for sample_index in range(1, sample_count):
        values[sample_index] = (
            rho * values[sample_index - 1]
            + innovation_scale * innovations[sample_index]
        )
    values -= values.mean(axis=0, keepdims=True)
    values /= np.maximum(values.std(axis=0, keepdims=True), 1e-12)
    return values


def _continuous_pq(net, sample_count: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate identifiable smooth P/Q excitation without consulting topology."""

    terminals = _terminal_labels(net)
    terminal_frame = net.buses.set_index("bus_id").loc[terminals]
    base_power_kw = float(net.base_mva) * 1000.0
    p_nominal = terminal_frame["pd_kw"].to_numpy(dtype=float) / base_power_kw
    q_nominal = terminal_frame["qd_kvar"].to_numpy(dtype=float) / base_power_kw
    rng = np.random.default_rng(seed)
    p_local = _ar1_standardized(rng, sample_count, len(terminals), rho=0.71)
    q_local = _ar1_standardized(rng, sample_count, len(terminals), rho=0.63)
    time = np.arange(sample_count, dtype=float)
    phases = np.linspace(0.0, 2.0 * np.pi, len(terminals), endpoint=False)
    common_p = np.sin(2.0 * np.pi * time / max(sample_count, 2))
    common_q = np.cos(2.0 * np.pi * time / max(sample_count, 2) + 0.31)
    p_wave = np.sin(6.0 * np.pi * time[:, None] / max(sample_count, 2) + phases)
    q_wave = np.cos(8.0 * np.pi * time[:, None] / max(sample_count, 2) + 0.73 * phases)
    p_multiplier = 1.0 + 0.20 * common_p[:, None] + 0.27 * p_local + 0.08 * p_wave
    q_multiplier = 1.0 + 0.18 * common_q[:, None] + 0.31 * q_local + 0.10 * q_wave
    p = np.maximum(0.10 * p_nominal[None, :], p_nominal[None, :] * p_multiplier)
    q = np.maximum(0.10 * q_nominal[None, :], q_nominal[None, :] * q_multiplier)
    index = pd.RangeIndex(sample_count, name="time_index")
    return (
        pd.DataFrame(p, index=index, columns=terminals),
        pd.DataFrame(q, index=index, columns=terminals),
    )


def _scenario(
    p: pd.DataFrame,
    q: pd.DataFrame,
    target: pd.DataFrame,
    selected: slice,
    name: str,
) -> list[dict]:
    return [
        {
            "name": name,
            "P_terminal": p.iloc[selected].copy(),
            "Q_terminal": q.iloc[selected].copy(),
            "drop_target": target.iloc[selected].copy(),
        }
    ]


def _path_rows(path: tuple[LaminarL1PathPoint, ...], selected: int) -> list[dict]:
    rows: list[dict] = []
    for path_index, point in enumerate(path):
        rows.append(
            {
                "path_index": path_index,
                "algorithm_iteration": point.iteration,
                "atom_count": len(point.support_indices),
                "supports": ";".join(
                    ",".join(map(str, support)) for support in point.support_labels
                ),
                "train_mae": float(point.train_mae),
                "validation_mae_zero_bias": point.validation_mae,
                "validation_se_zero_bias": point.validation_se,
                "accepted_gain": (
                    None if point.accepted_gain is None else float(point.accepted_gain)
                ),
                "solver_status": int(point.solver.status),
                "solver_success": bool(point.solver.success),
                "solver_objective": point.solver.objective,
                "solver_dual_bound": point.solver.dual_bound,
                "solver_mip_gap": point.solver.mip_gap,
                "solver_node_count": point.solver.node_count,
                "solver_runtime_seconds": float(point.solver.runtime_seconds),
            }
        )
    for index, row in enumerate(rows):
        row["selected_by_zero_bias_one_se"] = index == selected
    return rows


def _truth_atoms(net, terminals: list[int]) -> pd.DataFrame:
    """Aggregate physical edges sharing one downstream terminal support."""

    graph = net.to_networkx_graph()
    oriented = nx.bfs_tree(graph, net.root_bus)
    terminal_set = set(terminals)
    terminal_position = {terminal: index for index, terminal in enumerate(terminals)}
    z_base = float(net.base_kv) ** 2 / float(net.base_mva)
    grouped: dict[tuple[int, ...], dict] = {}
    for parent, child in oriented.edges():
        descendants = {int(child), *(int(node) for node in nx.descendants(oriented, child))}
        support = tuple(sorted(descendants & terminal_set))
        if not support:
            continue
        edge = graph.edges[parent, child]
        record = grouped.setdefault(
            support,
            {
                "support_labels": ",".join(map(str, support)),
                "support_indices": ",".join(str(terminal_position[node]) for node in support),
                "support_size": len(support),
                "r_value": 0.0,
                "x_value": 0.0,
                "physical_edges": [],
            },
        )
        record["r_value"] += 2.0 * float(edge["r_ohm"]) / z_base
        record["x_value"] += 2.0 * float(edge["x_ohm"]) / z_base
        record["physical_edges"].append(f"{int(parent)}-{int(child)}")
    rows: list[dict] = []
    for record in grouped.values():
        copied = dict(record)
        copied["physical_edges"] = ";".join(copied["physical_edges"])
        rows.append(copied)
    return pd.DataFrame(rows).sort_values(
        ["support_size", "support_labels"], ignore_index=True
    )


def _family_score(predicted: set[frozenset[int]], truth: set[frozenset[int]]) -> dict:
    matched = predicted & truth
    precision = len(matched) / len(predicted) if predicted else float(not truth)
    recall = len(matched) / len(truth) if truth else float(not predicted)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    def serialize(family: set[frozenset[int]]) -> list[list[int]]:
        return [
            sorted(item)
            for item in sorted(family, key=lambda value: (len(value), tuple(sorted(value))))
        ]

    return {
        "predicted_nontrivial_support_count": len(predicted),
        "true_nontrivial_support_count": len(truth),
        "matched_nontrivial_support_count": len(matched),
        "nontrivial_support_precision": precision,
        "nontrivial_support_recall": recall,
        "nontrivial_support_f1": f1,
        "matched_nontrivial_supports": serialize(matched),
        "missing_nontrivial_supports": serialize(truth - predicted),
        "extra_nontrivial_supports": serialize(predicted - truth),
    }


def _matrix_error(estimate: np.ndarray, truth: np.ndarray, prefix: str) -> dict:
    difference = np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float)
    return {
        f"{prefix}_matrix_mae": float(np.mean(np.abs(difference))),
        f"{prefix}_matrix_max_abs_error": float(np.max(np.abs(difference))),
        f"{prefix}_matrix_relative_frobenius_error": float(
            np.linalg.norm(difference, ord="fro")
            / max(np.linalg.norm(truth, ord="fro"), 1e-15)
        ),
    }


def _attempt_rows(result) -> list[dict]:
    rows: list[dict] = []
    labels = result.terminal_labels
    for attempt_index, attempt in enumerate(result.attempted_extensions, start=1):
        support_labels = (
            tuple(labels[index] for index in attempt.support)
            if attempt.support is not None
            else ()
        )
        diagnostics = attempt.diagnostics
        rows.append(
            {
                "attempt_index": attempt_index,
                "incumbent_support": ",".join(map(str, support_labels)),
                "incumbent_support_size": len(support_labels),
                "incumbent_was_accepted": any(
                    tuple(support_labels) == tuple(path_support)
                    for point in result.path[1:]
                    for path_support in point.support_labels
                ),
                "objective": attempt.objective,
                "status": int(diagnostics.status),
                "success": bool(diagnostics.success),
                "message": diagnostics.message,
                "dual_bound": diagnostics.dual_bound,
                "mip_gap": diagnostics.mip_gap,
                "node_count": diagnostics.node_count,
                "runtime_seconds": float(diagnostics.runtime_seconds),
                "variable_count": int(diagnostics.variable_count),
                "binary_variable_count": int(diagnostics.binary_variable_count),
                "constraint_count": int(diagnostics.constraint_count),
                "solver_certified_optimal": solver_diagnostics_prove_optimality(
                    diagnostics
                ),
            }
        )
    return rows


def _run_one_case(
    output_dir: Path,
    *,
    case_name: str,
    terminalization_mode: str,
    seed: int,
    sample_count: int,
    train_count: int,
    validation_count: int,
    max_atoms: int,
    coefficient_bound: float,
    time_limit: float,
) -> dict:
    test_count = sample_count - train_count - validation_count
    if min(train_count, validation_count, test_count) <= 0:
        raise ValueError("train/validation/test blocks must all be nonempty")

    case_dir = ensure_dir(output_dir / case_name)
    raw = load_raw_case33bw(include_tie_lines=False)
    net = terminalize_case33(
        raw,
        mode=terminalization_mode,
        service_impedance_mode="scaled_original",
        seed=seed,
    )
    terminals = _terminal_labels(net)
    r_true, x_true = build_reduced_sensitivity_matrices(
        net,
        terminals,
        voltage_model="squared-voltage",
    )
    p, q = _continuous_pq(net, sample_count, seed)
    target = pd.DataFrame(
        p.to_numpy(dtype=float) @ r_true.T + q.to_numpy(dtype=float) @ x_true.T,
        index=p.index,
        columns=terminals,
    )

    train_slice = slice(0, train_count)
    validation_slice = slice(train_count, train_count + validation_count)
    test_slice = slice(train_count + validation_count, sample_count)
    training = _scenario(p, q, target, train_slice, f"{case_name}_train")
    validation = _scenario(p, q, target, validation_slice, f"{case_name}_validation")
    test = _scenario(p, q, target, test_slice, f"{case_name}_test")

    # No true support or true coefficient is passed below.  Binary z variables
    # search the complete nonempty subset space, with only learned laminar
    # compatibility constraints added after an atom has been accepted.
    started = perf_counter()
    result = fit_laminar_l1_sensitivity(
        training,
        validation_scenarios=validation,
        validation_blocks=min(4, validation_count),
        max_atoms=max_atoms,
        r_upper_bound=coefficient_bound,
        x_upper_bound=coefficient_bound,
        max_bound_expansions=0,
        bound_active_fraction=0.999,
        improvement_abs_tol=1e-12,
        improvement_rel_tol=1e-9,
        zero_tolerance=1e-10,
        time_limit=time_limit,
        mip_rel_gap=0.0,
        presolve=True,
        disp=False,
    )
    fit_wall_seconds = perf_counter() - started

    selected_path_index = result.selected_path_index
    path_rows = _path_rows(result.path, selected_path_index)
    selected = result.path[selected_path_index]
    test_mae, test_se, test_blocks = evaluate_l1_matrices(
        test, selected.r_matrix, selected.x_matrix,
        blocks_per_scenario=min(4, test_count),
    )
    attempt_rows = _attempt_rows(result)
    truth_atoms = _truth_atoms(net, terminals)
    true_nontrivial = {
        frozenset(map(int, str(row.support_labels).split(",")))
        for row in truth_atoms.itertuples(index=False)
        if 1 < int(row.support_size) < len(terminals)
    }
    predicted_nontrivial = {
        frozenset(map(int, support))
        for support in selected.support_labels
        if 1 < len(support) < len(terminals)
    }
    reported_gaps = [
        float(row["mip_gap"]) for row in attempt_rows if row["mip_gap"] is not None
    ]
    attempted_statuses = [int(row["status"]) for row in attempt_rows]
    selected_r = np.asarray(selected.r_values, dtype=float)
    selected_x = np.asarray(selected.x_values, dtype=float)
    coefficient_bound_active = bool(
        (selected_r.size and np.max(selected_r) >= 0.999 * coefficient_bound)
        or (selected_x.size and np.max(selected_x) >= 0.999 * coefficient_bound)
    )
    metrics = {
        "case": case_name,
        "terminalization_mode": terminalization_mode,
        "terminal_count": len(terminals),
        "terminal_labels": terminals,
        "sample_count": sample_count,
        "train_count": train_count,
        "validation_count": validation_count,
        "test_count": test_count,
        "max_atoms_requested": max_atoms,
        "accepted_path_length_excluding_zero": len(result.path) - 1,
        "selected_path_index_zero_bias_one_se": selected_path_index,
        "selected_atom_count": len(selected.support_indices),
        "selected_supports": [list(item) for item in selected.support_labels],
        "train_mae": float(selected.train_mae),
        "validation_mae_zero_bias": float(
            path_rows[selected_path_index]["validation_mae_zero_bias"]
        ),
        "validation_se_zero_bias": float(
            path_rows[selected_path_index]["validation_se_zero_bias"]
        ),
        "test_mae_zero_bias": test_mae,
        "test_se_zero_bias": test_se,
        "stop_reason": result.stop_reason,
        "attempt_count": len(attempt_rows),
        "attempt_statuses": attempted_statuses,
        "all_attempts_status_zero": bool(attempt_rows)
        and all(status == 0 for status in attempted_statuses),
        "all_attempts_certified_optimal": bool(attempt_rows)
        and all(bool(row["solver_certified_optimal"]) for row in attempt_rows),
        "maximum_reported_mip_gap": max(reported_gaps, default=None),
        "fit_wall_seconds": fit_wall_seconds,
        "maximum_extension_runtime_seconds": max(
            (float(row["runtime_seconds"]) for row in attempt_rows), default=0.0
        ),
        "coefficient_bound": coefficient_bound,
        "coefficient_bound_active_in_selected_model": coefficient_bound_active,
        "truth_used_for_candidate_generation": False,
        "candidate_generation": (
            "implicit exhaustive search over all nonempty terminal subsets; "
            "laminar compatibility only with already selected supports"
        ),
        "observation_model": "observed-root zero-bias: Y = PR + QX",
        "exactness_scope": (
            "each extension with HiGHS/SciPy status 0 is globally optimal for the "
            "bounded one-atom MILP up to solver numerical tolerances; the forward "
            "multi-atom path remains greedy"
        ),
        "global_k_atom_optimum_claimed": False,
        "selected_family_is_laminar": is_laminar_family(selected.support_indices),
        "selected_r_min_eigenvalue": float(np.min(np.linalg.eigvalsh(selected.r_matrix))),
        "selected_x_min_eigenvalue": float(np.min(np.linalg.eigvalsh(selected.x_matrix))),
        **_matrix_error(selected.r_matrix, r_true, "R"),
        **_matrix_error(selected.x_matrix, x_true, "X"),
        **_family_score(predicted_nontrivial, true_nontrivial),
    }

    labels = pd.Index(terminals, name="terminal")
    pd.DataFrame(r_true, index=labels, columns=labels).to_csv(case_dir / "R_true.csv")
    pd.DataFrame(x_true, index=labels, columns=labels).to_csv(case_dir / "X_true.csv")
    pd.DataFrame(selected.r_matrix, index=labels, columns=labels).to_csv(
        case_dir / "R_selected.csv"
    )
    pd.DataFrame(selected.x_matrix, index=labels, columns=labels).to_csv(
        case_dir / "X_selected.csv"
    )
    p.to_csv(case_dir / "P_pu.csv")
    q.to_csv(case_dir / "Q_pu.csv")
    target.to_csv(case_dir / "squared_voltage_drop_pu2.csv")
    split_frame = pd.DataFrame(
        {
            "time_index": np.arange(sample_count),
            "split": (
                ["train"] * train_count
                + ["validation"] * validation_count
                + ["test"] * test_count
            ),
        }
    )
    split_frame.to_csv(case_dir / "split.csv", index=False)
    truth_atoms.to_csv(case_dir / "truth_atoms_postfit_only.csv", index=False)
    pd.DataFrame(path_rows).to_csv(case_dir / "path.csv", index=False)
    pd.DataFrame(attempt_rows).to_csv(case_dir / "extension_attempts.csv", index=False)
    pd.DataFrame(
        {
            "block_index": np.arange(len(test_blocks)),
            "test_mae_zero_bias": test_blocks,
        }
    ).to_csv(case_dir / "test_block_mae.csv", index=False)
    pd.DataFrame(
        [
            {
                "atom_index": atom_index,
                "support_labels": ",".join(map(str, labels_value)),
                "support_indices": ",".join(map(str, indices)),
                "r_value": float(r_value),
                "x_value": float(x_value),
            }
            for atom_index, (indices, labels_value, r_value, x_value) in enumerate(
                zip(
                    selected.support_indices,
                    selected.support_labels,
                    selected.r_values,
                    selected.x_values,
                    strict=True,
                ),
                start=1,
            )
        ]
    ).to_csv(case_dir / "selected_atoms.csv", index=False)
    _write_json(case_dir / "metrics.json", metrics)
    return metrics


def run(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    seed: int = 20260903,
    aggregate_time_limit: float = 30.0,
    hybrid_time_limit: float = 20.0,
    coefficient_bound: float = 10.0,
    skip_hybrid: bool = False,
) -> dict:
    output = ensure_dir(output_dir)
    cases = [
        _run_one_case(
            output,
            case_name="aggregate_leaves",
            terminalization_mode="aggregate_to_original_leaves",
            seed=seed,
            sample_count=72,
            train_count=48,
            validation_count=12,
            max_atoms=7,
            coefficient_bound=coefficient_bound,
            time_limit=aggregate_time_limit,
        )
    ]
    if not skip_hybrid:
        cases.append(
            _run_one_case(
                output,
                case_name="hybrid_full",
                terminalization_mode="hybrid_leaf",
                seed=seed + 1,
                sample_count=12,
                train_count=8,
                validation_count=2,
                max_atoms=1,
                coefficient_bound=coefficient_bound,
                time_limit=hybrid_time_limit,
            )
        )

    summary = pd.DataFrame(
        [
            {
                "case": metrics["case"],
                "terminal_count": metrics["terminal_count"],
                "max_atoms_requested": metrics["max_atoms_requested"],
                "accepted_atom_count": metrics["accepted_path_length_excluding_zero"],
                "selected_atom_count": metrics["selected_atom_count"],
                "stop_reason": metrics["stop_reason"],
                "attempt_statuses": ",".join(map(str, metrics["attempt_statuses"])),
                "all_attempts_status_zero": metrics["all_attempts_status_zero"],
                "all_attempts_certified_optimal": metrics[
                    "all_attempts_certified_optimal"
                ],
                "maximum_reported_mip_gap": metrics["maximum_reported_mip_gap"],
                "fit_wall_seconds": metrics["fit_wall_seconds"],
                "train_mae": metrics["train_mae"],
                "validation_mae_zero_bias": metrics[
                    "validation_mae_zero_bias"
                ],
                "test_mae_zero_bias": metrics[
                    "test_mae_zero_bias"
                ],
                "R_matrix_relative_frobenius_error": metrics[
                    "R_matrix_relative_frobenius_error"
                ],
                "X_matrix_relative_frobenius_error": metrics[
                    "X_matrix_relative_frobenius_error"
                ],
                "nontrivial_support_f1": metrics["nontrivial_support_f1"],
            }
            for metrics in cases
        ]
    )
    summary.to_csv(output / "summary.csv", index=False)
    config = {
        "observation_model": "observed_root_zero_bias; historical offset results require their archived source",
        "seed": seed,
        "solver": "scipy.optimize.milp/HiGHS",
        "scipy_version": scipy.__version__,
        "requested_mip_relative_gap": 0.0,
        "aggregate_time_limit_seconds_per_solve": aggregate_time_limit,
        "hybrid_time_limit_seconds_per_solve": hybrid_time_limit,
        "coefficient_bound_for_both_R_and_X": coefficient_bound,
        "truth_used_for_candidate_generation": False,
        "zero_bias_for_training_validation_and_test": True,
        "hybrid_case_skipped": skip_hybrid,
    }
    payload = {"config": config, "cases": cases}
    _write_json(output / "metrics.json", payload)
    report = [
        "# Laminar L1-MILP IEEE case33 integration",
        "",
        "Both cases use noiseless squared-voltage LinDistFlow responses. The physical "
        "case is used to generate data and post-fit truth metrics only; the solver "
        "searches every nonempty subset implicitly and receives no true clade list.",
        "",
        "Training, validation and test MAE use Y - PR - QX without any fitted bias. "
        "Status 0 denotes HiGHS/SciPy optimality "
        "within numerical tolerances for that bounded one-atom MILP. It does not turn "
        "the greedy multi-atom path into a globally optimal K-atom solution.",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--aggregate-time-limit", type=float, default=30.0)
    parser.add_argument("--hybrid-time-limit", type=float, default=20.0)
    parser.add_argument("--coefficient-bound", type=float, default=10.0)
    parser.add_argument("--skip-hybrid", action="store_true")
    args = parser.parse_args()
    run(
        args.output,
        seed=args.seed,
        aggregate_time_limit=args.aggregate_time_limit,
        hybrid_time_limit=args.hybrid_time_limit,
        coefficient_bound=args.coefficient_bound,
        skip_hybrid=args.skip_hybrid,
    )


if __name__ == "__main__":
    main()
