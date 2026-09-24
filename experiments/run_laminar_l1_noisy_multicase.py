"""Benchmark the laminar L1-MILP estimator on multiple noisy small feeders.

The benchmark is deliberately limited to four-to-six observed terminals.  The
estimator searches every nonempty terminal subset at each forward step, so the
full 32-terminal case33 variant is a scaling stress test rather than a useful
Monte Carlo target.

Two contamination regimes are evaluated:

``response_outliers``
    Laplace response noise plus sparse gross response outliers; P and Q are
    observed exactly.

``eiv_response_outliers``
    The same response contamination plus Gaussian errors in the observed P/Q
    regressors.  This is an errors-in-variables stress test; LAD is not expected
    to remove regressor-noise bias.

No true support or true coefficient is passed to the estimator.  Truth is used
only after fitting to score matrices and downstream-terminal clades.
"""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.run_laminar_l1_case33_integration import (
    _continuous_pq,
    _family_score,
    _matrix_error,
    _terminal_labels,
    _truth_atoms,
)
from terminal_case33.data.case33bw_raw import load_raw_case33bw
# Initialize the case-bank registry before importing its clade-grid extension.
# The existing registry intentionally performs a late reciprocal import.
from terminal_case33.data import paper_style_case_bank as _paper_style_case_bank  # noqa: F401
from terminal_case33.data.clade_grid_case_bank import make_clade_grid_case
from terminal_case33.data.small_terminal_lv import build_small_terminal_lv_case
from terminal_case33.estimation.laminar_l1_milp import (
    evaluate_l1_matrices,
    fit_laminar_l1_sensitivity,
    is_laminar_family,
    solver_diagnostics_prove_optimality,
)
from terminal_case33.models.lin_distflow import build_reduced_sensitivity_matrices
from terminal_case33.scenario.terminalize import terminalize_case33
from terminal_case33.utils.io import ensure_dir


DEFAULT_OUTPUT = Path("outputs/laminar_l1_milp_noisy_multicase")
DEFAULT_SEEDS = (20260903, 20260904, 20260905)
DEFAULT_SMALL_CASES = (
    "case33_aggregate4",
    "grid4_k2",
    "small6_branched",
    "grid6_k2_deep",
    "grid6_k3_wide",
)
PAPER_STYLE_CASES = ("paper15", "soumalas11", "flynn16", "pengwah18")
AVAILABLE_CASES = (*DEFAULT_SMALL_CASES, *PAPER_STYLE_CASES)
REGIMES = {
    "response_outliers": 0.0,
    "eiv_response_outliers": 0.02,
}


def build_benchmark_cases(
    case_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, object]:
    """Build only the requested feeders, preserving their requested order."""

    selected = DEFAULT_SMALL_CASES if case_names is None else tuple(case_names)
    if not selected:
        raise ValueError("at least one case is required")
    if len(set(selected)) != len(selected):
        raise ValueError("case names must be unique")
    unknown = set(selected) - set(AVAILABLE_CASES)
    if unknown:
        raise ValueError(f"unknown cases: {sorted(unknown)}")

    cases: dict[str, object] = {}
    for case_name in selected:
        if case_name in PAPER_STYLE_CASES:
            cases[case_name] = _paper_style_case_bank.CASE_BUILDERS[case_name]()
        elif case_name == "case33_aggregate4":
            raw = load_raw_case33bw(include_tie_lines=False)
            cases[case_name] = terminalize_case33(
                raw,
                mode="aggregate_to_original_leaves",
                service_impedance_mode="scaled_original",
                seed=20260903,
            )
        elif case_name == "grid4_k2":
            cases[case_name] = make_clade_grid_case(
                4,
                2,
                impedance_scale=3.0,
                case_name="laminar_grid4_k2",
            )
        elif case_name == "small6_branched":
            cases[case_name] = build_small_terminal_lv_case()
        elif case_name == "grid6_k2_deep":
            cases[case_name] = make_clade_grid_case(
                6,
                2,
                impedance_scale=2.5,
                case_name="laminar_grid6_k2_deep",
            )
        elif case_name == "grid6_k3_wide":
            cases[case_name] = make_clade_grid_case(
                6,
                3,
                impedance_scale=3.0,
                case_name="laminar_grid6_k3_wide",
            )
    return cases


def contaminate_measurements(
    p_true: pd.DataFrame,
    q_true: pd.DataFrame,
    target_true: pd.DataFrame,
    *,
    seed: int,
    predictor_noise_fraction: float,
    response_laplace_fraction: float = 0.05,
    outlier_fraction: float = 0.05,
    gross_outlier_fraction: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, np.ndarray]:
    """Add reproducible response contamination and optional P/Q EIV noise."""

    for name, value in {
        "predictor_noise_fraction": predictor_noise_fraction,
        "response_laplace_fraction": response_laplace_fraction,
        "outlier_fraction": outlier_fraction,
        "gross_outlier_fraction": gross_outlier_fraction,
    }.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if outlier_fraction >= 1.0:
        raise ValueError("outlier_fraction must be smaller than one")
    if not (
        p_true.index.equals(q_true.index)
        and p_true.index.equals(target_true.index)
        and p_true.columns.equals(q_true.columns)
        and p_true.columns.equals(target_true.columns)
    ):
        raise ValueError("P, Q, and target frames must have identical labels")

    rng = np.random.default_rng(seed)
    exact = target_true.to_numpy(dtype=float)
    centered = exact - np.median(exact, axis=0, keepdims=True)
    signal_mad = max(float(np.median(np.abs(centered))), 1e-12)
    laplace_scale = response_laplace_fraction * signal_mad
    gross_scale = gross_outlier_fraction * signal_mad
    response_noise = rng.laplace(0.0, laplace_scale, size=exact.shape)
    outlier_count = int(ceil(outlier_fraction * exact.size)) if outlier_fraction else 0
    outlier_mask = np.zeros(exact.size, dtype=bool)
    if outlier_count:
        outlier_mask[rng.choice(exact.size, size=outlier_count, replace=False)] = True
    outlier_mask = outlier_mask.reshape(exact.shape)
    gross = np.zeros_like(exact)
    if outlier_count:
        gross[outlier_mask] = (
            rng.choice(np.array([-1.0, 1.0]), size=outlier_count)
            * gross_scale
            * (1.0 + rng.exponential(scale=0.5, size=outlier_count))
        )
    target_observed = exact + response_noise + gross

    def noisy_predictor(frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        values = frame.to_numpy(dtype=float)
        scales = predictor_noise_fraction * np.maximum(
            np.std(values, axis=0, ddof=1),
            1e-12,
        )
        perturbation = rng.normal(0.0, scales[None, :], size=values.shape)
        return (
            pd.DataFrame(values + perturbation, index=frame.index, columns=frame.columns),
            float(np.sqrt(np.mean(np.square(perturbation)))),
        )

    p_observed, p_noise_rmse = noisy_predictor(p_true)
    q_observed, q_noise_rmse = noisy_predictor(q_true)
    metadata = {
        "response_laplace_fraction_of_signal_mad": response_laplace_fraction,
        "response_laplace_scale": laplace_scale,
        "gross_outlier_fraction_of_signal_mad": gross_outlier_fraction,
        "gross_outlier_base_scale": gross_scale,
        "outlier_fraction_requested": outlier_fraction,
        "outlier_count": outlier_count,
        "outlier_fraction_realized": outlier_count / exact.size,
        "signal_mad": signal_mad,
        "predictor_noise_fraction_of_temporal_std": predictor_noise_fraction,
        "p_noise_rmse": p_noise_rmse,
        "q_noise_rmse": q_noise_rmse,
    }
    return (
        p_observed,
        q_observed,
        pd.DataFrame(target_observed, index=target_true.index, columns=target_true.columns),
        metadata,
        outlier_mask,
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


def _truth_nontrivial_clades(net, terminals: list[int]) -> set[frozenset[int]]:
    frame = _truth_atoms(net, terminals)
    return {
        frozenset(map(int, str(row.support_labels).split(",")))
        for row in frame.itertuples(index=False)
        if 1 < int(row.support_size) < len(terminals)
    }


def _attempt_metrics(result) -> dict:
    attempts = result.attempted_extensions
    certified = [solver_diagnostics_prove_optimality(item.diagnostics) for item in attempts]
    finite_gaps = [
        float(item.diagnostics.mip_gap)
        for item in attempts
        if item.diagnostics.mip_gap is not None
        and np.isfinite(item.diagnostics.mip_gap)
    ]
    return {
        "attempt_count": len(attempts),
        "all_attempts_certified_optimal": bool(attempts) and all(certified),
        "certified_attempt_count": int(sum(certified)),
        "attempt_statuses": ",".join(str(item.diagnostics.status) for item in attempts),
        "maximum_reported_mip_gap": max(finite_gaps, default=None),
        "maximum_binary_variable_count": max(
            (item.diagnostics.binary_variable_count for item in attempts),
            default=0,
        ),
        "maximum_constraint_count": max(
            (item.diagnostics.constraint_count for item in attempts),
            default=0,
        ),
    }


def run_one(
    *,
    case_name: str,
    net,
    regime: str,
    seed: int,
    sample_count: int,
    train_count: int,
    validation_count: int,
    time_limit: float,
    coefficient_bound: float,
    response_laplace_fraction: float,
    outlier_fraction: float,
    gross_outlier_fraction: float,
    initial_support_labels: tuple[tuple[int, ...], ...] = (),
) -> tuple[dict, pd.DataFrame]:
    """Fit and score one noisy topology/seed/regime combination."""

    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    test_count = sample_count - train_count - validation_count
    if min(train_count, validation_count, test_count) <= 0:
        raise ValueError("train, validation, and test blocks must all be nonempty")

    terminals = _terminal_labels(net)
    terminal_position = {int(label): index for index, label in enumerate(terminals)}
    initial_support_sets = {
        frozenset(map(int, support)) for support in initial_support_labels
    }
    try:
        initial_support_indices = tuple(
            tuple(sorted(terminal_position[int(label)] for label in support))
            for support in initial_support_labels
        )
    except KeyError as exc:
        raise ValueError("initial support contains an unknown terminal label") from exc
    r_true, x_true = build_reduced_sensitivity_matrices(
        net,
        terminals,
        voltage_model="squared-voltage",
    )
    p_true, q_true = _continuous_pq(net, sample_count, seed)
    target_true = pd.DataFrame(
        p_true.to_numpy(dtype=float) @ r_true.T
        + q_true.to_numpy(dtype=float) @ x_true.T,
        index=p_true.index,
        columns=terminals,
    )
    p_observed, q_observed, target_observed, noise, outlier_mask = (
        contaminate_measurements(
            p_true,
            q_true,
            target_true,
            seed=seed + 10_000,
            predictor_noise_fraction=REGIMES[regime],
            response_laplace_fraction=response_laplace_fraction,
            outlier_fraction=outlier_fraction,
            gross_outlier_fraction=gross_outlier_fraction,
        )
    )

    train_slice = slice(0, train_count)
    validation_slice = slice(train_count, train_count + validation_count)
    test_slice = slice(train_count + validation_count, sample_count)
    scenario_name = f"{case_name}_{regime}"
    training = _scenario(
        p_observed, q_observed, target_observed, train_slice, scenario_name
    )
    validation = _scenario(
        p_observed, q_observed, target_observed, validation_slice, scenario_name
    )
    test_noisy = _scenario(
        p_observed, q_observed, target_observed, test_slice, scenario_name
    )
    test_clean = _scenario(p_true, q_true, target_true, test_slice, scenario_name)

    started = perf_counter()
    result = fit_laminar_l1_sensitivity(
        training,
        validation_scenarios=validation,
        initial_supports=initial_support_indices,
        max_atoms=2 * len(terminals) - 1,
        r_upper_bound=coefficient_bound,
        x_upper_bound=coefficient_bound,
        max_bound_expansions=0,
        improvement_abs_tol=1e-12,
        improvement_rel_tol=1e-9,
        zero_tolerance=1e-10,
        validation_blocks=min(4, validation_count),
        time_limit=time_limit,
        mip_rel_gap=0.0,
        presolve=True,
        disp=False,
    )
    wall_seconds = perf_counter() - started
    test_mae, test_se, _ = evaluate_l1_matrices(
        test_noisy,
        result.r_matrix,
        result.x_matrix,
        fixed_intercepts=result.intercepts,
        blocks_per_scenario=min(4, test_count),
    )
    clean_test_mae, _, _ = evaluate_l1_matrices(
        test_clean,
        result.r_matrix,
        result.x_matrix,
        fixed_intercepts=result.intercepts,
        blocks_per_scenario=min(4, test_count),
    )

    truth_clades = _truth_nontrivial_clades(net, terminals)
    predicted_clades = {
        frozenset(map(int, support))
        for support in result.support_labels
        if 1 < len(support) < len(terminals)
    }
    family = _family_score(predicted_clades, truth_clades)
    r_values = np.asarray(result.r_values, dtype=float)
    x_values = np.asarray(result.x_values, dtype=float)
    bound_active = bool(
        (r_values.size and np.max(r_values) >= 0.999 * coefficient_bound)
        or (x_values.size and np.max(x_values) >= 0.999 * coefficient_bound)
    )
    split_masks = {
        "train": outlier_mask[train_slice],
        "validation": outlier_mask[validation_slice],
        "test": outlier_mask[test_slice],
    }
    metrics = {
        "case": case_name,
        "regime": regime,
        "seed": seed,
        "terminal_count": len(terminals),
        "terminal_labels": ",".join(map(str, terminals)),
        "sample_count": sample_count,
        "train_count": train_count,
        "validation_count": validation_count,
        "test_count": test_count,
        "time_limit_seconds_per_extension": time_limit,
        **noise,
        **{
            f"{split}_outlier_count": int(mask.sum())
            for split, mask in split_masks.items()
        },
        "selected_path_index": result.selected_path_index,
        "selected_atom_count": len(result.support_indices),
        "selected_supports": ";".join(
            ",".join(map(str, support)) for support in result.support_labels
        ),
        "stop_reason": result.stop_reason,
        "train_mae": result.train_mae,
        "validation_mae_fixed_training_intercept": result.validation_mae,
        "test_mae_noisy_fixed_training_intercept": test_mae,
        "test_mae_noisy_se": test_se,
        "test_mae_clean_target_fixed_training_intercept": clean_test_mae,
        "test_mae_noisy_over_signal_mad": test_mae / noise["signal_mad"],
        "fit_wall_seconds": wall_seconds,
        "coefficient_bound": coefficient_bound,
        "coefficient_bound_active": bound_active,
        "selected_family_is_laminar": is_laminar_family(result.support_indices),
        "truth_used_for_candidate_generation": False,
        "initial_support_count": len(initial_support_labels),
        "initial_supports": ";".join(
            ",".join(map(str, support)) for support in initial_support_labels
        ),
        **_attempt_metrics(result),
        **_matrix_error(result.r_matrix, r_true, "R"),
        **_matrix_error(result.x_matrix, x_true, "X"),
        **family,
        "nontrivial_support_exact": predicted_clades == truth_clades,
    }
    atom_frame = pd.DataFrame(
        [
            {
                "case": case_name,
                "regime": regime,
                "seed": seed,
                "atom_index": index,
                "support_labels": ",".join(map(str, labels)),
                "support_indices": ",".join(map(str, indices)),
                "r_value": float(r_value),
                "x_value": float(x_value),
                "is_initial_support": frozenset(map(int, labels))
                in initial_support_sets,
            }
            for index, (indices, labels, r_value, x_value) in enumerate(
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
    return metrics, atom_frame


def _aggregate(run_frame: pd.DataFrame) -> pd.DataFrame:
    grouped = run_frame.groupby(["case", "regime"], sort=True, dropna=False)
    rows: list[dict] = []
    for (case_name, regime), group in grouped:
        rows.append(
            {
                "case": case_name,
                "regime": regime,
                "run_count": len(group),
                "terminal_count": int(group["terminal_count"].iloc[0]),
                "topology_exact_rate": float(group["nontrivial_support_exact"].mean()),
                "nontrivial_support_f1_mean": float(
                    group["nontrivial_support_f1"].mean()
                ),
                "nontrivial_support_f1_min": float(
                    group["nontrivial_support_f1"].min()
                ),
                "R_relative_frobenius_error_mean": float(
                    group["R_matrix_relative_frobenius_error"].mean()
                ),
                "R_relative_frobenius_error_max": float(
                    group["R_matrix_relative_frobenius_error"].max()
                ),
                "X_relative_frobenius_error_mean": float(
                    group["X_matrix_relative_frobenius_error"].mean()
                ),
                "X_relative_frobenius_error_max": float(
                    group["X_matrix_relative_frobenius_error"].max()
                ),
                "test_mae_noisy_mean": float(
                    group["test_mae_noisy_fixed_training_intercept"].mean()
                ),
                "test_mae_noisy_over_signal_mad_mean": float(
                    group["test_mae_noisy_over_signal_mad"].mean()
                ),
                "all_runs_fully_certified": bool(
                    group["all_attempts_certified_optimal"].all()
                ),
                "timeout_or_uncertified_run_count": int(
                    (~group["all_attempts_certified_optimal"]).sum()
                ),
                "fit_wall_seconds_mean": float(group["fit_wall_seconds"].mean()),
                "fit_wall_seconds_total": float(group["fit_wall_seconds"].sum()),
            }
        )
    return pd.DataFrame(rows)


def run(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    cases: tuple[str, ...] = DEFAULT_SMALL_CASES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    regimes: tuple[str, ...] = tuple(REGIMES),
    sample_count: int = 48,
    train_count: int = 32,
    validation_count: int = 8,
    time_limit: float = 30.0,
    coefficient_bound: float = 2.0,
    response_laplace_fraction: float = 0.05,
    outlier_fraction: float = 0.05,
    gross_outlier_fraction: float = 1.5,
) -> dict:
    """Run all topology/regime/seed combinations and export audit tables."""

    if not seeds:
        raise ValueError("at least one seed is required")
    unknown = set(regimes) - set(REGIMES)
    if unknown:
        raise ValueError(f"unknown regimes: {sorted(unknown)}")
    output = ensure_dir(output_dir)
    rows: list[dict] = []
    atoms: list[pd.DataFrame] = []
    for case_name, net in build_benchmark_cases(cases).items():
        for regime in regimes:
            for seed in seeds:
                metrics, atom_frame = run_one(
                    case_name=case_name,
                    net=net,
                    regime=regime,
                    seed=seed,
                    sample_count=sample_count,
                    train_count=train_count,
                    validation_count=validation_count,
                    time_limit=time_limit,
                    coefficient_bound=coefficient_bound,
                    response_laplace_fraction=response_laplace_fraction,
                    outlier_fraction=outlier_fraction,
                    gross_outlier_fraction=gross_outlier_fraction,
                )
                rows.append(metrics)
                atoms.append(atom_frame)

    run_frame = pd.DataFrame(rows)
    summary = _aggregate(run_frame)
    run_frame.to_csv(output / "runs.csv", index=False)
    summary.to_csv(output / "summary_by_case_regime.csv", index=False)
    pd.concat(atoms, ignore_index=True).to_csv(
        output / "selected_atoms.csv", index=False
    )
    config = {
        "cases": list(cases),
        "seeds": list(seeds),
        "regimes": list(regimes),
        "regime_predictor_noise_fractions": REGIMES,
        "sample_count": sample_count,
        "train_count": train_count,
        "validation_count": validation_count,
        "test_count": sample_count - train_count - validation_count,
        "time_limit_seconds_per_extension": time_limit,
        "coefficient_bound_for_R_and_X": coefficient_bound,
        "response_laplace_fraction_of_signal_mad": response_laplace_fraction,
        "outlier_fraction": outlier_fraction,
        "gross_outlier_fraction_of_signal_mad": gross_outlier_fraction,
        "truth_used_for_candidate_generation": False,
        "validation_and_test_use_fixed_training_intercepts": True,
    }
    payload = {
        "config": config,
        "summary": summary.to_dict(orient="records"),
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    report_lines = [
        "# Laminar L1-MILP noisy multi-case benchmark",
        "",
        "Every run searches all nonempty supports implicitly. Truth is used only for post-fit scoring.",
        "Validation and test predictions keep the training L1-median intercept fixed.",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
    ]
    (output / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=AVAILABLE_CASES,
        default=list(DEFAULT_SMALL_CASES),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--regimes", nargs="+", choices=tuple(REGIMES), default=list(REGIMES))
    parser.add_argument("--sample-count", type=int, default=48)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--validation-count", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--coefficient-bound", type=float, default=2.0)
    parser.add_argument("--response-laplace-fraction", type=float, default=0.05)
    parser.add_argument("--outlier-fraction", type=float, default=0.05)
    parser.add_argument("--gross-outlier-fraction", type=float, default=1.5)
    args = parser.parse_args()
    run(
        args.output,
        cases=tuple(args.cases),
        seeds=tuple(args.seeds),
        regimes=tuple(args.regimes),
        sample_count=args.sample_count,
        train_count=args.train_count,
        validation_count=args.validation_count,
        time_limit=args.time_limit,
        coefficient_bound=args.coefficient_bound,
        response_laplace_fraction=args.response_laplace_fraction,
        outlier_fraction=args.outlier_fraction,
        gross_outlier_fraction=args.gross_outlier_fraction,
    )


if __name__ == "__main__":
    main()
