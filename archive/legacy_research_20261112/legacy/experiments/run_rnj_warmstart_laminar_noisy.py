"""Test stable partial-RNJ initialization for exact laminar L1-MILP fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.run_laminar_l1_case33_integration import (
    _continuous_pq,
    _family_score,
    _terminal_labels,
)
from experiments.run_laminar_l1_noisy_multicase import (
    PAPER_STYLE_CASES,
    REGIMES,
    _aggregate,
    _scenario,
    _truth_nontrivial_clades,
    build_benchmark_cases,
    contaminate_measurements,
    run_one,
)
from terminal_case33.estimation.rnj_warm_start import (
    StableRNJSupportSelection,
    select_stable_rnj_initial_supports,
)
from terminal_case33.models.lin_distflow import build_reduced_sensitivity_matrices
from terminal_case33.utils.io import ensure_dir


DEFAULT_OUTPUT = Path("outputs/rnj_warmstart_laminar_noisy")


def _selection_rows(
    case_name: str,
    regime: str,
    seed: int,
    selection: StableRNJSupportSelection,
) -> pd.DataFrame:
    selected = {
        tuple(map(int, labels)): confidence
        for labels, confidence in zip(
            selection.selected_support_labels,
            selection.selected_confidences,
            strict=True,
        )
    }
    full = {
        mode: {tuple(map(int, labels)) for labels in clades}
        for mode, clades in selection.full_data_clades_by_mode
    }
    rows = []
    for mode, entries in selection.bootstrap_confidences_by_mode:
        for labels, confidence in entries:
            integer_labels = tuple(map(int, labels))
            rows.append(
                {
                    "case": case_name,
                    "regime": regime,
                    "seed": seed,
                    "mode": mode,
                    "support_labels": ",".join(map(str, integer_labels)),
                    "support_size": len(integer_labels),
                    "bootstrap_confidence": float(confidence),
                    "present_in_full_data_tree": integer_labels in full[mode],
                    "selected_as_initial_support": integer_labels in selected,
                    "selected_cross_mode_confidence": selected.get(integer_labels),
                }
            )
    return pd.DataFrame(rows)


def select_case_initial_supports(
    *,
    case_name: str,
    net,
    regime: str,
    seed: int,
    sample_count: int,
    train_count: int,
    response_laplace_fraction: float,
    outlier_fraction: float,
    gross_outlier_fraction: float,
    modes: tuple[str, ...],
    bootstrap_replicates: int,
    confidence_threshold: float,
    tolerance_factor: float,
    maximum_support_size: int | None,
    maximum_support_count: int,
    ridge_alpha: float,
) -> tuple[StableRNJSupportSelection, float]:
    """Recreate the run's training block and select supports without truth."""

    terminals = _terminal_labels(net)
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
    p_observed, q_observed, target_observed, _, _ = contaminate_measurements(
        p_true,
        q_true,
        target_true,
        seed=seed + 10_000,
        predictor_noise_fraction=REGIMES[regime],
        response_laplace_fraction=response_laplace_fraction,
        outlier_fraction=outlier_fraction,
        gross_outlier_fraction=gross_outlier_fraction,
    )
    training = _scenario(
        p_observed,
        q_observed,
        target_observed,
        slice(0, train_count),
        f"{case_name}_{regime}",
    )
    started = perf_counter()
    selection = select_stable_rnj_initial_supports(
        training,
        root_bus=int(net.root_bus),
        modes=modes,
        bootstrap_replicates=bootstrap_replicates,
        confidence_threshold=confidence_threshold,
        tolerance_factor=tolerance_factor,
        maximum_support_size=maximum_support_size,
        maximum_support_count=maximum_support_count,
        ridge_alpha=ridge_alpha,
        seed=seed + 20_000,
    )
    return selection, perf_counter() - started


def _write_checkpoints(
    output: Path,
    runs: list[dict],
    atoms: list[pd.DataFrame],
    selections: list[pd.DataFrame],
) -> None:
    if selections:
        pd.concat(selections, ignore_index=True).to_csv(
            output / "rnj_bootstrap_clades.csv", index=False
        )
    if not runs:
        return
    run_frame = pd.DataFrame(runs)
    run_frame.to_csv(output / "runs.csv", index=False)
    _aggregate(run_frame).to_csv(output / "summary_by_case_regime.csv", index=False)
    pd.concat(atoms, ignore_index=True).to_csv(
        output / "selected_atoms.csv", index=False
    )


def run(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    cases: tuple[str, ...] = PAPER_STYLE_CASES,
    seeds: tuple[int, ...] = (20260903,),
    regimes: tuple[str, ...] = ("response_outliers",),
    sample_count: int = 48,
    train_count: int = 32,
    validation_count: int = 8,
    time_limit: float = 600.0,
    coefficient_bound: float = 2.0,
    response_laplace_fraction: float = 0.05,
    outlier_fraction: float = 0.05,
    gross_outlier_fraction: float = 1.5,
    modes: tuple[str, ...] = ("R", "RX_75R_25X"),
    bootstrap_replicates: int = 12,
    confidence_threshold: float = 0.75,
    tolerance_factor: float = 0.16,
    maximum_support_size: int | None = None,
    maximum_support_count: int = 3,
    ridge_alpha: float = 1e-6,
    selection_only: bool = False,
) -> dict:
    output = ensure_dir(output_dir)
    run_rows: list[dict] = []
    atom_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    selection_summaries: list[dict] = []

    for case_name, net in build_benchmark_cases(cases).items():
        for regime in regimes:
            if regime not in REGIMES:
                raise ValueError(f"unknown regime {regime!r}")
            for seed in seeds:
                selection, selection_seconds = select_case_initial_supports(
                    case_name=case_name,
                    net=net,
                    regime=regime,
                    seed=seed,
                    sample_count=sample_count,
                    train_count=train_count,
                    response_laplace_fraction=response_laplace_fraction,
                    outlier_fraction=outlier_fraction,
                    gross_outlier_fraction=gross_outlier_fraction,
                    modes=modes,
                    bootstrap_replicates=bootstrap_replicates,
                    confidence_threshold=confidence_threshold,
                    tolerance_factor=tolerance_factor,
                    maximum_support_size=maximum_support_size,
                    maximum_support_count=maximum_support_count,
                    ridge_alpha=ridge_alpha,
                )
                selection_frames.append(
                    _selection_rows(case_name, regime, seed, selection)
                )
                initial_labels = tuple(
                    tuple(map(int, support))
                    for support in selection.selected_support_labels
                )
                truth = _truth_nontrivial_clades(net, _terminal_labels(net))
                initial_score = _family_score(
                    {frozenset(support) for support in initial_labels}, truth
                )
                selection_summaries.append(
                    {
                        "case": case_name,
                        "regime": regime,
                        "seed": seed,
                        "selection_seconds": selection_seconds,
                        "initial_support_count": len(initial_labels),
                        "initial_supports": ";".join(
                            ",".join(map(str, support)) for support in initial_labels
                        ),
                        "initial_support_confidences": ";".join(
                            f"{value:.6g}" for value in selection.selected_confidences
                        ),
                        **{f"initial_{key}": value for key, value in initial_score.items()},
                    }
                )
                pd.DataFrame(selection_summaries).to_csv(
                    output / "rnj_initial_supports.csv", index=False
                )
                _write_checkpoints(output, run_rows, atom_frames, selection_frames)
                if selection_only:
                    continue

                metrics, atoms = run_one(
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
                    initial_support_labels=initial_labels,
                )
                metrics.update(selection_summaries[-1])
                metrics.update(
                    {
                        "initializer": "stable_bootstrap_rnj",
                        "rnj_modes": ",".join(modes),
                        "rnj_bootstrap_replicates": bootstrap_replicates,
                        "rnj_confidence_threshold": confidence_threshold,
                        "rnj_tolerance_factor": tolerance_factor,
                        "rnj_maximum_support_size": selection.maximum_support_size,
                        "rnj_maximum_support_count": maximum_support_count,
                        "rnj_ridge_alpha": ridge_alpha,
                    }
                )
                atoms["initializer"] = "stable_bootstrap_rnj"
                run_rows.append(metrics)
                atom_frames.append(atoms)
                _write_checkpoints(output, run_rows, atom_frames, selection_frames)

    config = {
        "cases": list(cases),
        "seeds": list(seeds),
        "regimes": list(regimes),
        "sample_count": sample_count,
        "train_count": train_count,
        "validation_count": validation_count,
        "time_limit_seconds_per_extension": time_limit,
        "coefficient_bound": coefficient_bound,
        "modes": list(modes),
        "bootstrap_replicates": bootstrap_replicates,
        "confidence_threshold": confidence_threshold,
        "tolerance_factor": tolerance_factor,
        "maximum_support_size": maximum_support_size,
        "maximum_support_count": maximum_support_count,
        "ridge_alpha": ridge_alpha,
        "selection_only": selection_only,
        "truth_used_for_candidate_generation": False,
    }
    payload = {
        "config": config,
        "selection": selection_summaries,
        "runs": run_rows,
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", nargs="+", choices=PAPER_STYLE_CASES, default=list(PAPER_STYLE_CASES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260903])
    parser.add_argument("--regimes", nargs="+", choices=tuple(REGIMES), default=["response_outliers"])
    parser.add_argument("--sample-count", type=int, default=48)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--validation-count", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--coefficient-bound", type=float, default=2.0)
    parser.add_argument("--response-laplace-fraction", type=float, default=0.05)
    parser.add_argument("--outlier-fraction", type=float, default=0.05)
    parser.add_argument("--gross-outlier-fraction", type=float, default=1.5)
    parser.add_argument("--modes", nargs="+", default=["R", "RX_75R_25X"])
    parser.add_argument("--bootstrap-replicates", type=int, default=12)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--tolerance-factor", type=float, default=0.16)
    parser.add_argument("--maximum-support-size", type=int, default=None)
    parser.add_argument("--maximum-support-count", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--selection-only", action="store_true")
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
        modes=tuple(args.modes),
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_threshold=args.confidence_threshold,
        tolerance_factor=args.tolerance_factor,
        maximum_support_size=args.maximum_support_size,
        maximum_support_count=args.maximum_support_count,
        ridge_alpha=args.ridge_alpha,
        selection_only=args.selection_only,
    )


if __name__ == "__main__":
    main()
