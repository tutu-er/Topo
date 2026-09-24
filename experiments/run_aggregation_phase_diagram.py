"""Paired aggregation on/off phase diagram over the clade-grid case family.

Sweeps {control + grid16_k2/k4/k8} x data regime x noise regime x
aggregation {on, off} x trials. The aggregation axis toggles both
``enable_ordered_aggregation`` and ``enable_nj_edge_aggregation`` via
``dataclasses.replace`` on a shared base config (same scenarios, same pipeline
seed), so on/off rows are paired per seed. Scenario construction, metrics,
persistence, and resume mirror ``run_unified_topology_pipeline_sweep.py``.

Outputs per ``--output`` directory:

- ``metrics.csv``: one row per condition (aggregation folded into
  ``condition_id``), with the unified-pipeline row schema plus ``aggregation``,
  ``clade_size_k``, ``truth_clade_count``, and backbone/peripheral F1 splits
  (``_filtered_f1`` precedent from the two-level aggregation ablation).
- ``paired_aggregation_gain.csv``: on-minus-off paired gains per
  (case, data_regime, noise_regime, trial) for unified F1, exact recovery,
  backbone/peripheral F1, and the AC candidate-oracle diagnostics.
- ``clade_evidence.csv``, ``method_metrics_long.csv``, ``report.md``,
  ``summary.json``.

``--smoke`` runs 1 case x sparse x nominal x 1 trial x on/off with
quartet/bootstrap replicates reduced to 2.
"""

from __future__ import annotations

import argparse
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
from experiments.run_two_level_aggregation_ablation import _filtered_f1
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
AGGREGATION_MODES = ("on", "off")
NOISE_ORDER = {"low": 0, "nominal": 1, "high": 2}
DEFAULT_CASES = ["flynn16", "grid16_k2", "grid16_k4", "grid16_k8"]


def _case_clade_size(net: object, truth: set[frozenset[int]]) -> int:
    """Return the designed service-clade size k for one case.

    Grid cases carry ``metadata['clade_size']``; the flynn16 control falls
    back to the smallest truth clade size so the backbone/peripheral split
    remains meaningful.
    """

    designed = getattr(net, "metadata", {}).get("clade_size")
    if designed:
        return int(designed)
    return min(len(clade) for clade in truth) if truth else 2


def _condition_id(
    case: str,
    data_regime: str,
    noise_regime: str,
    aggregation: str,
    trial: int,
) -> str:
    return f"{case}__{data_regime}__{noise_regime}__agg-{aggregation}__trial{trial}"


def _evaluate(
    case: str,
    data_regime: str,
    noise_regime: str,
    aggregation: str,
    trial: int,
    quartet_replicates: int,
    nj_bootstrap_replicates: int,
) -> tuple[dict, list[dict]]:
    """Run one paired aggregation on/off condition on noisy AC data."""

    data = DATA_REGIMES[data_regime]
    noise = NOISE_REGIMES[noise_regime]
    condition_id = _condition_id(case, data_regime, noise_regime, aggregation, trial)
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
    clade_size = _case_clade_size(net, truth)
    validation_count = representative_validation_count(data["scenario_count"])
    base_config = UnifiedTopologyConfig(
        validation_scenario_count=validation_count,
        quartet_replicates=quartet_replicates,
        nj_edge_bootstrap_replicates=nj_bootstrap_replicates,
    )
    enabled = aggregation == "on"
    config = replace(
        base_config,
        enable_ordered_aggregation=enabled,
        enable_nj_edge_aggregation=enabled,
        aggregation_max_cluster_size=max(5, clade_size),
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
    unified_prediction = predictions["unified"]
    row = {
        "condition_id": condition_id,
        "status": "ok",
        "case": case,
        "data_regime": data_regime,
        "noise_regime": noise_regime,
        "aggregation": aggregation,
        "aggregation_enabled": enabled,
        "trial": trial,
        "scenario_count": data["scenario_count"],
        "sample_count": data["sample_count"],
        "pq_noise_rel": noise["pq_noise_rel"],
        "v_noise_rel": noise["v_noise_rel"],
        "terminal_count": len(terminals),
        "clade_size_k": clade_size,
        "truth_clade_count": len(truth),
        "truth_clade_sizes": "/".join(str(len(clade)) for clade in sorted(truth, key=len)),
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
        "unified_peripheral_f1": _filtered_f1(
            unified_prediction,
            truth,
            maximum_size=clade_size,
        ),
        "unified_backbone_f1": _filtered_f1(
            unified_prediction,
            truth,
            minimum_size=clade_size + 1,
        ),
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
                "aggregation": aggregation,
                "trial": trial,
                **evidence.to_dict(),
                "is_true_clade": clade in truth,
            }
        )
    return row, edge_rows


def _long_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert condition metrics to one row per method and aggregation mode."""

    records = []
    for _, row in frame.iterrows():
        for method, column in METHOD_COLUMNS.items():
            records.append(
                {
                    "condition_id": row["condition_id"],
                    "case": row["case"],
                    "data_regime": row["data_regime"],
                    "noise_regime": row["noise_regime"],
                    "aggregation": row["aggregation"],
                    "trial": row["trial"],
                    "method": method,
                    "f1": row[column],
                    "exact": row[column.replace("_f1", "_exact")],
                }
            )
    return pd.DataFrame(records)


def _paired_gain_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build same-seed on-minus-off paired aggregation gains."""

    pair_keys = ["case", "data_regime", "noise_regime", "trial"]
    value_columns = {
        "unified_f1": "unified_f1",
        "unified_exact": "unified_exact",
        "unified_backbone_f1": "unified_backbone_f1",
        "unified_peripheral_f1": "unified_peripheral_f1",
        "ac_candidate_oracle_f1": "ac_candidate_oracle_f1",
        "ac_candidate_contains_exact": "ac_candidate_contains_exact",
        "elapsed_seconds": "elapsed_seconds",
    }
    selected = frame[pair_keys + ["aggregation", "clade_size_k"] + list(value_columns)]
    on = selected.loc[selected["aggregation"].eq("on")].drop(columns="aggregation")
    off = selected.loc[selected["aggregation"].eq("off")].drop(columns="aggregation")
    paired = on.merge(off, on=pair_keys, suffixes=("_on", "_off"))
    if paired.empty:
        return paired
    paired = paired.drop(columns=["clade_size_k_off"]).rename(
        columns={"clade_size_k_on": "clade_size_k"}
    )
    for name in value_columns:
        for side in ("_on", "_off"):
            paired[f"{name}{side}"] = paired[f"{name}{side}"].astype(float)
        paired[f"{name}_gain"] = paired[f"{name}_on"] - paired[f"{name}_off"]
    paired["noise_order"] = paired["noise_regime"].map(NOISE_ORDER)
    return paired.sort_values(
        ["case", "data_regime", "noise_order", "trial"]
    ).drop(columns="noise_order")


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


def _write_report(
    output: Path,
    frame: pd.DataFrame,
    long: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    """Write a concise aggregation on/off and gate summary."""

    overall = (
        long.groupby(["aggregation", "method"], as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            min_f1=("f1", "min"),
            exact_rate=("exact", "mean"),
        )
        .sort_values(["method", "aggregation"])
    )
    lines = [
        "# Aggregation phase diagram (paired on/off)",
        "",
        "All P/Q/V measurements are noisy and generated by radial AC power flow.",
        "Aggregation on/off rows share scenarios and pipeline seeds (paired).",
        "",
        "## Overall by aggregation mode",
        "",
        "| method | aggregation | mean F1 | minimum F1 | exact rate |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in overall.iterrows():
        lines.append(
            f"| {row['method']} | {row['aggregation']} | {row['mean_f1']:.4f} | "
            f"{row['min_f1']:.4f} | {row['exact_rate']:.2%} |"
        )
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
        ]
    )
    if not paired.empty:
        lines.extend(
            [
                "",
                "## Paired on-minus-off gains",
                "",
                f"- Paired conditions: {len(paired)}",
                f"- Mean unified F1 gain: {paired['unified_f1_gain'].mean():.4f}",
                f"- Mean backbone F1 gain: "
                f"{paired['unified_backbone_f1_gain'].mean():.4f}",
                f"- Mean peripheral F1 gain: "
                f"{paired['unified_peripheral_f1_gain'].mean():.4f}",
                f"- Exact-recovery flips off->on: "
                f"{int((paired['unified_exact_gain'] > 0).sum())}",
                f"- Exact-recovery flips on->off: "
                f"{int((paired['unified_exact_gain'] < 0).sum())}",
            ]
        )
        by_case = paired.groupby("clade_size_k", as_index=False).agg(
            mean_gain=("unified_f1_gain", "mean"),
            mean_backbone_gain=("unified_backbone_f1_gain", "mean"),
            mean_peripheral_gain=("unified_peripheral_f1_gain", "mean"),
        )
        lines.extend(
            [
                "",
                "| k | mean F1 gain | backbone gain | peripheral gain |",
                "|---:|---:|---:|---:|",
            ]
        )
        for _, row in by_case.iterrows():
            lines.append(
                f"| {int(row['clade_size_k'])} | {row['mean_gain']:.4f} | "
                f"{row['mean_backbone_gain']:.4f} | "
                f"{row['mean_peripheral_gain']:.4f} |"
            )
    (output / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    """Run a resumable paired aggregation on/off sweep."""

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
        (case, data_regime, noise_regime, aggregation, trial)
        for case in args.cases
        for data_regime in args.data_regimes
        for noise_regime in args.noise_regimes
        for aggregation in args.aggregation
        for trial in range(args.trial_offset, args.trial_offset + args.trials)
    ]
    started = time.perf_counter()
    processed = 0
    for index, condition in enumerate(conditions, start=1):
        case, data_regime, noise_regime, aggregation, trial = condition
        condition_id = _condition_id(case, data_regime, noise_regime, aggregation, trial)
        if condition_id in completed:
            print(f"[{index:03d}/{len(conditions):03d}] skip {condition_id}", flush=True)
            continue
        print(f"[{index:03d}/{len(conditions):03d}] run  {condition_id}", flush=True)
        try:
            row, edge_rows = _evaluate(
                case,
                data_regime,
                noise_regime,
                aggregation,
                trial,
                args.quartet_replicates,
                args.nj_bootstrap_replicates,
            )
            rows.append(row)
            evidence_rows.extend(edge_rows)
            print(
                f"    unified={row['unified_f1']:.4f} "
                f"backbone={row['unified_backbone_f1']:.4f} "
                f"peripheral={row['unified_peripheral_f1']:.4f} "
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
                    "aggregation": aggregation,
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
        raise RuntimeError("no aggregation phase-diagram conditions completed")
    long = _long_frame(frame)
    long.to_csv(output / "method_metrics_long.csv", index=False)
    paired = _paired_gain_frame(frame)
    paired.to_csv(output / "paired_aggregation_gain.csv", index=False)
    (
        long.groupby(["aggregation", "method"], as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            min_f1=("f1", "min"),
            exact_rate=("exact", "mean"),
        )
        .sort_values(["method", "aggregation"])
        .to_csv(output / "summary_by_method.csv", index=False)
    )
    _write_report(output, frame, long, paired)
    write_json(
        output / "summary.json",
        {
            "completed_conditions": len(frame),
            "requested_conditions": len(conditions),
            "paired_conditions": len(paired),
            "cases": args.cases,
            "data_regimes": args.data_regimes,
            "noise_regimes": args.noise_regimes,
            "aggregation_modes": args.aggregation,
            "trials": args.trials,
            "quartet_replicates": args.quartet_replicates,
            "nj_bootstrap_replicates": args.nj_bootstrap_replicates,
            "aggregation_max_cluster_size_rule": "max(5, clade_size_k)",
            "pairing": "same scenarios and pipeline seed per (case, regime, noise, trial)",
            "all_measurements_noisy": True,
            "voltage_generation": "radial_ac_power_flow",
        },
    )


def main() -> None:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="outputs/aggregation_phase_diagram",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CASE_BUILDERS),
        default=DEFAULT_CASES,
    )
    parser.add_argument(
        "--regimes",
        dest="data_regimes",
        nargs="+",
        choices=sorted(DATA_REGIMES),
        default=["sparse", "medium"],
    )
    parser.add_argument(
        "--noise",
        dest="noise_regimes",
        nargs="+",
        choices=sorted(NOISE_REGIMES),
        default=["low", "nominal", "high"],
    )
    parser.add_argument(
        "--aggregation",
        nargs="+",
        choices=list(AGGREGATION_MODES),
        default=list(AGGREGATION_MODES),
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--trial-offset", type=int, default=0)
    parser.add_argument("--quartet-replicates", type=int, default=12)
    parser.add_argument("--nj-bootstrap-replicates", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="1 case x sparse x nominal x 1 trial x on/off with replicates=2",
    )
    args = parser.parse_args()
    if args.smoke:
        args.cases = ["grid16_k4"]
        args.data_regimes = ["sparse"]
        args.noise_regimes = ["nominal"]
        args.aggregation = list(AGGREGATION_MODES)
        args.trials = 1
        args.trial_offset = 0
        args.quartet_replicates = 2
        args.nj_bootstrap_replicates = 2
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.quartet_replicates <= 0:
        parser.error("--quartet-replicates must be positive")
    if args.nj_bootstrap_replicates <= 0:
        parser.error("--nj-bootstrap-replicates must be positive")
    run(args)


if __name__ == "__main__":
    main()
