"""Evaluate the coupled Ordered-RNJ, aggregation, GTLS, AC, and quartet pipeline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from experiments.run_ac_quartet_large_sweep import (
    DATA_REGIMES,
    NOISE_REGIMES,
    _build_scenarios,
    _metrics,
)
from terminal_case33.data.paper_style_case_bank import CASE_BUILDERS
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.pipeline.ac_likelihood import representative_validation_count
from terminal_case33.pipeline.unified_topology_pipeline import (
    UnifiedTopologyConfig,
    identify_topology_unified,
)
from terminal_case33.utils.io import ensure_dir, write_json


METHOD_COLUMNS = {
    "ordered_base": "ordered_base_f1",
    "gtls_map": "gtls_map_f1",
    "shared_ac_pre_quartet": "shared_ac_pre_quartet_f1",
    "unified": "unified_f1",
}


def _evaluate(
    case: str,
    data_regime: str,
    noise_regime: str,
    trial: int,
    quartet_replicates: int,
    nj_bootstrap_replicates: int,
) -> tuple[dict, list[dict]]:
    """Run one noisy AC-generated benchmark condition."""

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
    terminals = [int(column) for column in scenarios[0]["P_terminal"].columns]
    truth = set(rooted_clades(net.closed_edges(), net.root_bus, terminals))
    validation_count = representative_validation_count(data["scenario_count"])
    config = UnifiedTopologyConfig(
        validation_scenario_count=validation_count,
        quartet_replicates=quartet_replicates,
        nj_edge_bootstrap_replicates=nj_bootstrap_replicates,
    )
    result = identify_topology_unified(
        scenarios,
        net,
        pq_noise_relative_std=noise["pq_noise_rel"],
        voltage_noise_relative_std=noise["v_noise_rel"],
        config=config,
        seed=20260715 + 1000 * trial + 37 * len(case),
    )
    ac_rerank = result.ac_validated.ac_rerank
    selected_ac = ac_rerank.selected
    candidate_metrics = [
        _metrics(set(score.candidate.clades), truth) for score in ac_rerank.ranked
    ]
    candidate_oracle_f1 = max(item["f1"] for item in candidate_metrics)
    candidate_contains_exact = any(item["exact"] for item in candidate_metrics)
    ac_score_gap = (
        ac_rerank.ranked[1].selection_score - selected_ac.selection_score
        if len(ac_rerank.ranked) > 1
        else float("nan")
    )
    predictions = {
        "ordered_base": set(result.base.rooted_clades),
        "gtls_map": set(
            result.gtls_posterior.selected.projection.candidate.clades
        ),
        "shared_ac_pre_quartet": set(result.ac_validated.raw_clades),
        "unified": set(result.final_clades),
    }
    metrics = {
        method: _metrics(prediction, truth)
        for method, prediction in predictions.items()
    }
    row = {
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
        "terminal_count": len(terminals),
        "truth_clade_count": len(truth),
        "selection_source": result.selection_source,
        "base_trusted": result.base_trusted,
        "gtls_map_trusted": result.gtls_map_trusted,
        "proposed_cluster_count": len(result.proposed_aggregation_clades),
        "stable_cluster_count": len(result.stable_aggregation_clades),
        "applied_cluster_count": len(result.applied_aggregation_clades),
        "aggregation_rerun": result.aggregation_rerun,
        "base_r2_score": result.base.r2_score,
        "base_condition_number": result.base.condition_number,
        "gtls_map_probability": (
            result.gtls_posterior.selected.posterior_probability
        ),
        "gtls_top_two_probability_gap": (
            result.gtls_posterior.top_two_probability_gap
        ),
        "candidate_count": result.gtls_posterior.candidate_count,
        "ac_candidate_count": ac_rerank.candidate_count,
        "ac_candidate_oracle_f1": candidate_oracle_f1,
        "ac_candidate_contains_exact": candidate_contains_exact,
        "ac_training_scenario_count": ac_rerank.training_scenario_count,
        "ac_validation_scenario_count": ac_rerank.validation_scenario_count,
        "ac_refinement_accepted": selected_ac.edge_fit.ac_refinement_accepted,
        "ac_training_rmse_before": selected_ac.edge_fit.training_ac_rmse_before,
        "ac_training_rmse_after": selected_ac.edge_fit.training_ac_rmse_after,
        "ac_predictive_nll": selected_ac.standardized_negative_log_likelihood,
        "ac_scenario_dispersion_penalty": selected_ac.scenario_dispersion_penalty,
        "ac_score_gap": ac_score_gap,
        "elapsed_seconds": time.perf_counter() - started,
        "error": "",
    }
    for method, values in metrics.items():
        for metric, value in values.items():
            row[f"{method}_{metric}"] = value
    edge_rows = []
    for clade, evidence in result.clade_evidence.items():
        edge_rows.append(
            {
                "condition_id": condition_id,
                "case": case,
                "data_regime": data_regime,
                "noise_regime": noise_regime,
                "trial": trial,
                **evidence.to_dict(),
                "is_true_clade": clade in truth,
            }
        )
    return row, edge_rows


def _long_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert condition metrics to one row per method."""

    records = []
    for _, row in frame.iterrows():
        for method, column in METHOD_COLUMNS.items():
            records.append(
                {
                    "condition_id": row["condition_id"],
                    "case": row["case"],
                    "data_regime": row["data_regime"],
                    "noise_regime": row["noise_regime"],
                    "trial": row["trial"],
                    "method": method,
                    "f1": row[column],
                    "exact": row[column.replace("_f1", "_exact")],
                }
            )
    return pd.DataFrame(records)


def _persist(
    output: Path,
    rows: list[dict],
    evidence_rows: list[dict],
) -> None:
    """Persist resumable condition and clade tables."""

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("condition_id")
    frame.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(evidence_rows).to_csv(output / "clade_evidence.csv", index=False)


def _write_report(output: Path, frame: pd.DataFrame, long: pd.DataFrame) -> None:
    """Write a concise method and gate summary."""

    overall = (
        long.groupby("method", as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            min_f1=("f1", "min"),
            exact_rate=("exact", "mean"),
        )
        .sort_values("mean_f1", ascending=False)
    )
    source_counts = frame["selection_source"].value_counts()
    lines = [
        "# Unified topology pipeline sweep",
        "",
        "All P/Q/V measurements are noisy and generated by radial AC power flow.",
        "Ordered RNJ, GTLS marginals, AC validation, and quartet evidence are",
        "coupled by explicit gates rather than multiplied as independent probabilities.",
        "",
        "## Overall",
        "",
        "| method | mean F1 | minimum F1 | exact rate |",
        "|---|---:|---:|---:|",
    ]
    for _, row in overall.iterrows():
        lines.append(
            f"| {row['method']} | {row['mean_f1']:.4f} | "
            f"{row['min_f1']:.4f} | {row['exact_rate']:.2%} |"
        )
    lines.extend(["", "## AC-selected proposal sources", ""])
    for source, count in source_counts.items():
        lines.append(f"- {source}: {int(count)}")
    lines.extend(
        [
            "",
            "## Aggregation gates",
            "",
            f"- Proposed clusters: {int(frame['proposed_cluster_count'].sum())}",
            f"- Multi-source stable clusters: "
            f"{int(frame['stable_cluster_count'].sum())}",
            f"- Actually applied clusters: "
            f"{int(frame['applied_cluster_count'].sum())}",
            f"- Conditions rerun after cluster confirmation: "
            f"{int(frame['aggregation_rerun'].sum())}",
            f"- Trusted Ordered base conditions: "
            f"{int(frame['base_trusted'].sum())}",
            f"- Trusted GTLS MAP conditions: "
            f"{int(frame['gtls_map_trusted'].sum())}",
        ]
    )
    (output / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    """Run a resumable noisy-AC comparison sweep."""

    output = ensure_dir(args.output)
    metrics_path = output / "metrics.csv"
    evidence_path = output / "clade_evidence.csv"
    rows = (
        pd.read_csv(metrics_path).to_dict("records")
        if args.resume and metrics_path.exists()
        else []
    )
    evidence_rows = (
        pd.read_csv(evidence_path).to_dict("records")
        if args.resume and evidence_path.exists()
        else []
    )
    completed = {
        str(row["condition_id"])
        for row in rows
        if str(row.get("status", "")) == "ok"
    }
    conditions = [
        (case, data_regime, noise_regime, trial)
        for case in args.cases
        for data_regime in args.data_regimes
        for noise_regime in args.noise_regimes
        for trial in range(args.trial_offset, args.trial_offset + args.trials)
    ]
    started = time.perf_counter()
    processed = 0
    for index, condition in enumerate(conditions, start=1):
        case, data_regime, noise_regime, trial = condition
        condition_id = f"{case}__{data_regime}__{noise_regime}__trial{trial}"
        if condition_id in completed:
            print(f"[{index:03d}/{len(conditions):03d}] skip {condition_id}", flush=True)
            continue
        print(f"[{index:03d}/{len(conditions):03d}] run  {condition_id}", flush=True)
        try:
            row, edge_rows = _evaluate(
                case,
                data_regime,
                noise_regime,
                trial,
                args.quartet_replicates,
                args.nj_bootstrap_replicates,
            )
            rows.append(row)
            evidence_rows.extend(edge_rows)
            print(
                f"    unified={row['unified_f1']:.4f} "
                f"shared_ac={row['shared_ac_pre_quartet_f1']:.4f} "
                f"gtls={row['gtls_map_f1']:.4f} "
                f"source={row['selection_source']}",
                flush=True,
            )
        except Exception as error:
            rows.append(
                {
                    "condition_id": condition_id,
                    "status": "error",
                    "case": case,
                    "data_regime": data_regime,
                    "noise_regime": noise_regime,
                    "trial": trial,
                    "error": repr(error),
                }
            )
            print(f"    ERROR: {error!r}", flush=True)
        processed += 1
        _persist(output, rows, evidence_rows)
        elapsed = time.perf_counter() - started
        eta = elapsed / processed * max(len(conditions) - index, 0)
        print(f"    elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

    frame = pd.DataFrame(rows)
    frame = frame.loc[frame["status"].eq("ok")].copy()
    if frame.empty:
        raise RuntimeError("no unified-pipeline conditions completed")
    long = _long_frame(frame)
    long.to_csv(output / "method_metrics_long.csv", index=False)
    summary = (
        long.groupby("method", as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            min_f1=("f1", "min"),
            exact_rate=("exact", "mean"),
        )
        .sort_values("mean_f1", ascending=False)
    )
    summary.to_csv(output / "summary_by_method.csv", index=False)
    _write_report(output, frame, long)
    write_json(
        output / "summary.json",
        {
            "completed_conditions": len(frame),
            "requested_conditions": len(conditions),
            "cases": args.cases,
            "data_regimes": args.data_regimes,
            "noise_regimes": args.noise_regimes,
            "trials": args.trials,
            "quartet_replicates": args.quartet_replicates,
            "nj_bootstrap_replicates": args.nj_bootstrap_replicates,
            "all_measurements_noisy": True,
            "voltage_generation": "radial_ac_power_flow",
            "evidence_fusion": "explicit_label_free_gates",
        },
    )


def main() -> None:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="outputs/unified_topology_pipeline_sweep",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CASE_BUILDERS),
        default=["paper15", "soumalas11", "flynn16", "pengwah18"],
    )
    parser.add_argument(
        "--data-regimes",
        nargs="+",
        choices=sorted(DATA_REGIMES),
        default=["sparse"],
    )
    parser.add_argument(
        "--noise-regimes",
        nargs="+",
        choices=sorted(NOISE_REGIMES),
        default=["nominal"],
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--trial-offset", type=int, default=0)
    parser.add_argument("--quartet-replicates", type=int, default=12)
    parser.add_argument("--nj-bootstrap-replicates", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.quartet_replicates <= 0:
        parser.error("--quartet-replicates must be positive")
    if args.nj_bootstrap_replicates <= 0:
        parser.error("--nj-bootstrap-replicates must be positive")
    run(args)


if __name__ == "__main__":
    main()
