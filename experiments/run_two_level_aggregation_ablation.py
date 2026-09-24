"""Evaluate NJ/RG-guided Ordered two-level aggregation on noisy AC data."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from experiments.run_ac_quartet_large_sweep import (
    DATA_REGIMES,
    NOISE_REGIMES,
    _build_scenarios,
    _metrics,
)
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.pipeline.unified_topology_pipeline import (
    UnifiedTopologyConfig,
    identify_topology_unified,
)
from terminal_case33.utils.io import ensure_dir


Clade = frozenset[int]


def _filtered_f1(
    prediction: set[Clade],
    truth: set[Clade],
    *,
    maximum_size: int | None = None,
    minimum_size: int | None = None,
) -> float:
    """Compute F1 after selecting peripheral or backbone clades by size."""

    def keep(clade: Clade) -> bool:
        if maximum_size is not None and len(clade) > maximum_size:
            return False
        if minimum_size is not None and len(clade) < minimum_size:
            return False
        return True

    selected_prediction = {clade for clade in prediction if keep(clade)}
    selected_truth = {clade for clade in truth if keep(clade)}
    return float(_metrics(selected_prediction, selected_truth)["f1"])


def _method_record(
    condition_id: str,
    method: str,
    prediction: set[Clade],
    truth: set[Clade],
    peripheral_size: int,
) -> dict:
    """Build one comparable topology-accuracy record."""

    metrics = _metrics(prediction, truth)
    return {
        "condition_id": condition_id,
        "method": method,
        **metrics,
        "peripheral_f1": _filtered_f1(
            prediction,
            truth,
            maximum_size=peripheral_size,
        ),
        "backbone_f1": _filtered_f1(
            prediction,
            truth,
            minimum_size=peripheral_size + 1,
        ),
    }


def _evaluate(
    case: str,
    data_regime: str,
    noise_regime: str,
    trial: int,
    quartet_replicates: int,
    nj_bootstrap_replicates: int,
    maximum_cluster_size: int,
    paired_nj_ablation: bool,
) -> tuple[dict, list[dict]]:
    """Run one condition and expose every aggregation candidate family."""

    data = DATA_REGIMES[data_regime]
    noise = NOISE_REGIMES[noise_regime]
    condition_id = f"{case}__{data_regime}__{noise_regime}__trial{trial}"
    started = time.perf_counter()
    net, scenarios = _build_scenarios(
        case,
        data["scenario_count"],
        data["sample_count"],
        noise["pq_noise_rel"],
        noise["v_noise_rel"],
        trial,
    )
    terminals = [int(value) for value in scenarios[0]["P_terminal"].columns]
    truth = set(rooted_clades(net.closed_edges(), net.root_bus, terminals))
    validation_count = max(1, min(2, data["scenario_count"] // 4))
    topology_config = UnifiedTopologyConfig(
        validation_scenario_count=validation_count,
        quartet_replicates=quartet_replicates,
        aggregation_max_cluster_size=maximum_cluster_size,
        enable_ordered_aggregation=True,
        enable_nj_edge_aggregation=True,
        nj_edge_bootstrap_replicates=max(4, nj_bootstrap_replicates),
        nj_edge_support_threshold=0.75,
    )
    pipeline_seed = 20260715 + 1000 * trial + 37 * len(case)
    result = identify_topology_unified(
        scenarios,
        net,
        pq_noise_relative_std=noise["pq_noise_rel"],
        voltage_noise_relative_std=noise["v_noise_rel"],
        config=topology_config,
        seed=pipeline_seed,
    )
    no_nj_result = None
    if paired_nj_ablation:
        no_nj_result = identify_topology_unified(
            scenarios,
            net,
            pq_noise_relative_std=noise["pq_noise_rel"],
            voltage_noise_relative_std=noise["v_noise_rel"],
            config=replace(
                topology_config,
                enable_nj_edge_aggregation=False,
            ),
            seed=pipeline_seed,
        )
    predictions: dict[str, set[Clade]] = {
        "ordered_base": set(result.base.rooted_clades),
        "gtls_map": set(result.gtls_posterior.selected.projection.candidate.clades),
        "ac_selected_raw": set(result.ac_validated.raw_clades),
        "unified_validated": set(result.final_clades),
    }
    if no_nj_result is not None:
        predictions["unified_without_nj_edge_aggregation"] = set(no_nj_result.final_clades)
    for detector, aggregation in result.detector_aggregations.items():
        predictions[f"{detector}__ordered_base"] = set(aggregation.base.rooted_clades)
        for mode, clades in aggregation.candidate_clades_by_mode.items():
            predictions[f"{detector}__{mode}"] = set(clades)

    method_rows = [
        _method_record(
            condition_id,
            method,
            prediction,
            truth,
            maximum_cluster_size,
        )
        for method, prediction in sorted(predictions.items())
    ]
    ranked_f1 = [
        _metrics(set(score.candidate.clades), truth)["f1"]
        for score in result.ac_validated.ac_rerank.ranked
    ]
    evaluated_f1 = [_metrics(prediction, truth)["f1"] for prediction in predictions.values()]
    candidate_oracle_f1 = max(ranked_f1 + evaluated_f1, default=0.0)
    stable_clusters = set(result.stable_aggregation_clades)
    false_contractions = len(stable_clusters - truth)
    unique_aggregations = {id(value): value for value in result.detector_aggregations.values()}
    local_fits = [
        fit
        for aggregation in unique_aggregations.values()
        for fit in aggregation.local_fits.values()
    ]
    finite_local_nrmse = [
        fit.heldout_nrmse for fit in local_fits if math.isfinite(fit.heldout_nrmse)
    ]
    condition_improvements = [
        aggregation.condition_improvement for aggregation in unique_aggregations.values()
    ]
    detector_counts = {
        mode: {
            "proposed": len(result.peripheral_proposals.clusters_by_mode.get(mode, ()))
            if result.peripheral_proposals is not None
            else 0,
            "confirmed": len(result.confirmed_detector_clusters.get(mode, ())),
        }
        for mode in ("nj_stable", "nj_rg_consensus")
    }
    nj_stable_clusters = {
        cluster.members
        for clusters in result.confirmed_detector_clusters.values()
        for cluster in clusters
    }
    diagnostics = {
        "condition_id": condition_id,
        "status": "ok",
        "case": case,
        "data_regime": data_regime,
        "noise_regime": noise_regime,
        "trial": trial,
        "scenario_count": data["scenario_count"],
        "sample_count": data["sample_count"],
        "pq_noise_rel": noise["pq_noise_rel"],
        "v_noise_rel": noise["v_noise_rel"],
        "selection_source": result.selection_source,
        "candidate_count": result.ac_validated.ac_rerank.candidate_count,
        "candidate_oracle_f1": float(candidate_oracle_f1),
        "stable_cluster_count": len(stable_clusters),
        "false_contraction_count": false_contractions,
        "false_contraction_rate": (
            false_contractions / len(stable_clusters) if stable_clusters else 0.0
        ),
        "aggregation_candidate_count": sum(
            len(value.candidate_clades_by_mode)
            for value in unique_aggregations.values()
        ),
        "local_refit_admissible_rate": (
            sum(fit.admissible for fit in local_fits) / len(local_fits)
            if local_fits
            else 0.0
        ),
        "mean_local_nrmse": (
            sum(finite_local_nrmse) / len(finite_local_nrmse)
            if finite_local_nrmse
            else float("nan")
        ),
        "mean_local_stability": (
            sum(fit.clade_stability for fit in local_fits) / len(local_fits)
            if local_fits
            else float("nan")
        ),
        "mean_condition_improvement": (
            sum(condition_improvements) / len(condition_improvements)
            if condition_improvements
            else float("nan")
        ),
        "nj_proposed_cluster_count": sum(value["proposed"] for value in detector_counts.values()),
        "nj_confirmed_cluster_count": sum(value["confirmed"] for value in detector_counts.values()),
        "nj_false_cluster_count": len(nj_stable_clusters - truth),
        "nj_false_cluster_rate": (
            len(nj_stable_clusters - truth) / len(nj_stable_clusters) if nj_stable_clusters else 0.0
        ),
        "nj_detector_candidate_count": len(result.detector_aggregations),
        "without_nj_selection_source": (
            no_nj_result.selection_source if no_nj_result is not None else ""
        ),
        "without_nj_f1": (
            _metrics(set(no_nj_result.final_clades), truth)["f1"]
            if no_nj_result is not None
            else float("nan")
        ),
        "nj_edge_f1_gain": (
            _metrics(set(result.final_clades), truth)["f1"]
            - _metrics(set(no_nj_result.final_clades), truth)["f1"]
            if no_nj_result is not None
            else float("nan")
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "error": "",
    }
    return diagnostics, method_rows


def _write_summary(output: Path, diagnostics: pd.DataFrame, methods: pd.DataFrame) -> None:
    """Write aggregate accuracy, reliability, and no-harm summaries."""

    summary = (
        methods.groupby("method", as_index=False)
        .agg(
            condition_count=("f1", "size"),
            mean_f1=("f1", "mean"),
            minimum_f1=("f1", "min"),
            exact_rate=("exact", "mean"),
            peripheral_f1=("peripheral_f1", "mean"),
            backbone_f1=("backbone_f1", "mean"),
        )
        .sort_values("mean_f1", ascending=False)
    )
    summary.to_csv(output / "summary_by_method.csv", index=False)
    base = methods.loc[methods["method"].eq("ordered_base"), ["condition_id", "f1"]]
    base = base.rename(columns={"f1": "base_f1"})
    aggregation = methods.loc[
        methods["method"].str.contains("aggregation"),
        ["condition_id", "f1"],
    ]
    best_aggregation = (
        aggregation.groupby("condition_id", as_index=False)["f1"]
        .max()
        .rename(columns={"f1": "best_aggregation_f1"})
    )
    paired = base.merge(best_aggregation, on="condition_id", how="inner")
    paired["aggregation_gain"] = paired["best_aggregation_f1"] - paired["base_f1"]
    paired.to_csv(output / "paired_aggregation_gain.csv", index=False)
    full = methods.loc[methods["method"].eq("unified_validated"), ["condition_id", "f1"]].rename(
        columns={"f1": "full_f1"}
    )
    without_nj = methods.loc[
        methods["method"].eq("unified_without_nj_edge_aggregation"),
        ["condition_id", "f1"],
    ].rename(columns={"f1": "without_nj_f1"})
    paired_nj = full.merge(without_nj, on="condition_id", how="inner")
    paired_nj["nj_edge_gain"] = paired_nj["full_f1"] - paired_nj["without_nj_f1"]
    paired_nj.to_csv(output / "paired_nj_edge_gain.csv", index=False)

    lines = [
        "# Two-level aggregation ablation",
        "",
        "All conditions use noisy smart-meter measurements generated by AC power flow.",
        "Candidate oracle is diagnostic only and is never used for topology selection.",
        "",
        "| method | n | mean F1 | minimum F1 | exact | peripheral F1 | backbone F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['method']} | {int(row['condition_count'])} | {row['mean_f1']:.4f} | "
            f"{row['minimum_f1']:.4f} | {row['exact_rate']:.2%} | "
            f"{row['peripheral_f1']:.4f} | {row['backbone_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- Conditions: {len(diagnostics)}",
            f"- False contraction rate: {diagnostics['false_contraction_rate'].mean():.2%}",
            f"- Mean candidate oracle F1: {diagnostics['candidate_oracle_f1'].mean():.4f}",
            f"- Mean selected candidate count: {diagnostics['candidate_count'].mean():.2f}",
            f"- Mean best-aggregation gain over Ordered base: {paired['aggregation_gain'].mean():.4f}",
            f"- Aggregation improved conditions: {int((paired['aggregation_gain'] > 1e-12).sum())}/{len(paired)}",
            f"- Aggregation degraded conditions: {int((paired['aggregation_gain'] < -1e-12).sum())}/{len(paired)}",
        ]
    )

    if not paired_nj.empty:
        lines.extend(
            [
                f"- Mean NJ-edge aggregation gain: {paired_nj['nj_edge_gain'].mean():.4f}",
                f"- NJ-edge aggregation improved conditions: {int((paired_nj['nj_edge_gain'] > 1e-12).sum())}/{len(paired_nj)}",
                f"- NJ-edge aggregation unchanged conditions: {int((paired_nj['nj_edge_gain'].abs() <= 1e-12).sum())}/{len(paired_nj)}",
                f"- NJ-edge aggregation degraded conditions: {int((paired_nj['nj_edge_gain'] < -1e-12).sum())}/{len(paired_nj)}",
            ]
        )

    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    """Run a resumable aggregation ablation with progress and ETA."""

    output = ensure_dir(args.output)
    diagnostics_path = output / "diagnostics.csv"
    methods_path = output / "method_metrics.csv"
    diagnostics_rows = (
        pd.read_csv(diagnostics_path).to_dict("records")
        if args.resume and diagnostics_path.exists()
        else []
    )
    method_rows = (
        pd.read_csv(methods_path).to_dict("records")
        if args.resume and methods_path.exists()
        else []
    )
    completed = {
        str(row["condition_id"]) for row in diagnostics_rows if str(row.get("status", "")) == "ok"
    }
    conditions = [
        (case, data, noise, trial)
        for case in args.cases
        for data in args.data_regimes
        for noise in args.noise_regimes
        for trial in range(args.trial_offset, args.trial_offset + args.trials)
    ]
    started = time.perf_counter()
    processed = 0
    for index, (case, data, noise, trial) in enumerate(conditions, start=1):
        condition_id = f"{case}__{data}__{noise}__trial{trial}"
        if condition_id in completed:
            print(f"[{index:03d}/{len(conditions):03d}] skip {condition_id}", flush=True)
            continue
        print(f"[{index:03d}/{len(conditions):03d}] run  {condition_id}", flush=True)
        try:
            diagnostic, condition_methods = _evaluate(
                case,
                data,
                noise,
                trial,
                args.quartet_replicates,
                args.nj_bootstrap_replicates,
                args.maximum_cluster_size,
                args.paired_nj_ablation,
            )
            diagnostics_rows.append(diagnostic)
            method_rows.extend(condition_methods)
            selected = next(
                row for row in condition_methods if row["method"] == "unified_validated"
            )
            print(
                f"    selected={selected['f1']:.4f} "
                f"oracle={diagnostic['candidate_oracle_f1']:.4f} "
                f"source={diagnostic['selection_source']}",
                flush=True,
            )
        except Exception as error:
            diagnostics_rows.append(
                {
                    "condition_id": condition_id,
                    "status": "error",
                    "case": case,
                    "data_regime": data,
                    "noise_regime": noise,
                    "trial": trial,
                    "error": repr(error),
                }
            )
            print(f"    ERROR: {error!r}", flush=True)
        pd.DataFrame(diagnostics_rows).to_csv(diagnostics_path, index=False)
        pd.DataFrame(method_rows).to_csv(methods_path, index=False)
        processed += 1
        elapsed = time.perf_counter() - started
        eta = elapsed / processed * max(len(conditions) - index, 0)
        print(f"    elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

    diagnostics = pd.DataFrame(diagnostics_rows)
    diagnostics = diagnostics.loc[diagnostics["status"].eq("ok")]
    methods = pd.DataFrame(method_rows)
    if not diagnostics.empty and not methods.empty:
        _write_summary(output, diagnostics, methods)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="outputs/two_level_aggregation_ablation",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["paper15", "soumalas11", "flynn16", "pengwah18"],
    )
    parser.add_argument(
        "--data-regimes",
        nargs="+",
        choices=sorted(DATA_REGIMES),
        default=["medium"],
    )
    parser.add_argument(
        "--noise-regimes",
        nargs="+",
        choices=sorted(NOISE_REGIMES),
        default=["nominal"],
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--trial-offset", type=int, default=0)
    parser.add_argument("--quartet-replicates", type=int, default=8)
    parser.add_argument("--nj-bootstrap-replicates", type=int, default=8)
    parser.add_argument("--maximum-cluster-size", type=int, default=5)
    parser.add_argument("--paired-nj-ablation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
