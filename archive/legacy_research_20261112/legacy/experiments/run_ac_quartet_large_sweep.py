"""Large noisy-AC sweep for RNJ candidates, AC reranking, and quartet tests."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import resource_assignment, selected_scenario_defs, simulate_case_scenario
from terminal_case33.data.paper_style_case_bank import CASE_BUILDERS
from terminal_case33.estimation.multiscenario import fit_projected_sensitivity
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.pipeline.ac_likelihood import representative_validation_count
from terminal_case33.pipeline.rnj_candidates import generate_rnj_candidates
from terminal_case33.pipeline.topology_validation import select_and_validate_topology
from terminal_case33.utils.io import ensure_dir, write_json


DATA_REGIMES = {
    "sparse": {"scenario_count": 3, "sample_count": 96},
    "medium": {"scenario_count": 5, "sample_count": 288},
    "rich": {"scenario_count": 9, "sample_count": 960},
}
NOISE_REGIMES = {
    "low": {"pq_noise_rel": 0.0025, "v_noise_rel": 0.0001},
    "nominal": {"pq_noise_rel": 0.005, "v_noise_rel": 0.0002},
    "high": {"pq_noise_rel": 0.01, "v_noise_rel": 0.0005},
}
METHOD_COLUMNS = {
    "fixed_rnj": "baseline_f1",
    "ac_bic": "ac_bic_f1",
    "quartet": "quartet_f1",
    "validated": "validated_f1",
}


def _metrics(prediction: set[frozenset[int]], truth: set[frozenset[int]]) -> dict:
    matched = len(prediction & truth)
    precision = matched / len(prediction) if prediction else float(not truth)
    recall = matched / len(truth) if truth else float(not prediction)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": prediction == truth,
        "false_positive": len(prediction - truth),
        "false_negative": len(truth - prediction),
    }


def _finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _build_scenarios(
    case: str,
    scenario_count: int,
    sample_count: int,
    pq_noise_rel: float,
    v_noise_rel: float,
    trial: int,
) -> tuple[object, list[dict]]:
    net = CASE_BUILDERS[case]()
    assignment = resource_assignment(case, net)
    definitions = selected_scenario_defs()
    if scenario_count > len(definitions):
        raise ValueError("scenario_count exceeds available operating profiles")
    offset = (3 * trial) % len(definitions)
    rotated = definitions[offset:] + definitions[:offset]
    positions = np.linspace(0, len(rotated) - 1, scenario_count, dtype=int)
    scenarios = []
    for scenario_index, position in enumerate(positions):
        definition = rotated[int(position)]
        seed = int(definition["seed"]) + 10007 * trial + 503 * scenario_index
        simulation = simulate_case_scenario(
            net=net,
            assignment=assignment,
            T=sample_count,
            seed=seed,
            pq_noise_rel=pq_noise_rel,
            v_noise_rel=v_noise_rel,
            root_voltage_mean=1.02,
            root_voltage_sigma=float(definition["root_voltage_sigma"]),
            profile_scenario=str(definition["profile_scenario"]),
        )
        if not simulation["ac_converged"]:
            raise RuntimeError("AC power flow did not converge")
        scenarios.append(
            {
                "name": f"{definition['profile_scenario']}_trial{trial}_scenario{scenario_index}",
                "profile_scenario": str(definition["profile_scenario"]),
                "T": sample_count,
                "V_terminal": simulation["V_terminal"],
                "P_terminal": simulation["P_terminal"],
                "Q_terminal": simulation["Q_terminal"],
                "root_voltage": simulation["root_voltage"],
                "drop_target": simulation["drop_target"],
            }
        )
    return net, scenarios


def _evaluate(
    condition_id: str,
    case: str,
    data_regime: str,
    noise_regime: str,
    trial: int,
    quartet_replicates: int,
) -> tuple[dict, list[dict]]:
    data = DATA_REGIMES[data_regime]
    noise = NOISE_REGIMES[noise_regime]
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
    training = scenarios[:-validation_count]
    r_matrix, x_matrix, _r2, condition_number = fit_projected_sensitivity(
        training, constraint_mode="ordered"
    )
    baseline = generate_rnj_candidates(
        r_matrix,
        x_matrix,
        terminals,
        net.root_bus,
        distance_modes=("RX_75R_25X",),
        tolerance_factors=(0.24,),
    )[0]
    result = select_and_validate_topology(
        scenarios,
        net,
        pq_noise_relative_std=noise["pq_noise_rel"],
        voltage_noise_relative_std=noise["v_noise_rel"],
        bic_complexity_weight=1.0,
        quartet_replicates=quartet_replicates,
        quartet_confidence=0.90,
        quartet_positive_support=0.80,
        quartet_minimum_effect=0.002,
        minimum_edge_strength_ratio=0.02,
        quartet_contradiction_effect=0.05,
        validation_scenario_count=validation_count,
        seed=20260714 + 1000 * trial + 37 * len(case),
    )
    clade_sets = {
        "baseline": set(baseline.clades),
        "ac_bic": set(result.raw_clades),
        "quartet": set(result.quartet_clades),
        "validated": set(result.validated_clades),
    }
    scores = {name: _metrics(value, truth) for name, value in clade_sets.items()}
    candidate_scores = [
        _metrics(set(score.candidate.clades), truth)
        for score in result.ac_rerank.ranked
    ]
    exact_ranks = [
        index + 1 for index, item in enumerate(candidate_scores) if item["exact"]
    ]
    selected = result.ac_rerank.selected
    score_gap = (
        result.ac_rerank.ranked[1].selection_score - selected.selection_score
        if len(result.ac_rerank.ranked) > 1
        else float("nan")
    )
    true_lower = [
        record.lower_confidence_bound
        for clade, record in result.quartet_evidence.items()
        if clade in truth
    ]
    false_lower = [
        record.lower_confidence_bound
        for clade, record in result.quartet_evidence.items()
        if clade not in truth
    ]
    raw, quartet, validated = (
        clade_sets["ac_bic"],
        clade_sets["quartet"],
        clade_sets["validated"],
    )
    row = {
        "condition_id": condition_id,
        "status": "ok",
        "case": case,
        "data_regime": data_regime,
        "noise_regime": noise_regime,
        "trial": trial,
        "scenario_count": data["scenario_count"],
        "sample_count": data["sample_count"],
        "training_scenario_count": len(training),
        "validation_scenario_count": validation_count,
        "pq_noise_rel": noise["pq_noise_rel"],
        "v_noise_rel": noise["v_noise_rel"],
        "terminal_count": len(terminals),
        "truth_clade_count": len(truth),
        "condition_number": condition_number,
        "candidate_count": result.ac_rerank.candidate_count,
        "candidate_oracle_f1": max(float(item["f1"]) for item in candidate_scores),
        "candidate_contains_exact": bool(exact_ranks),
        "exact_candidate_best_rank": min(exact_ranks) if exact_ranks else np.nan,
        "selected_mode": selected.candidate.distance_mode,
        "selected_tolerance": selected.candidate.tolerance_factor,
        "ac_voltage_rmse": selected.voltage_rmse,
        "ac_selection_score": selected.selection_score,
        "ac_score_gap": score_gap,
        "quartet_removed_false": len((raw - truth) - quartet),
        "quartet_removed_true": len((raw & truth) - quartet),
        "strength_removed_false": len((quartet - truth) - validated),
        "strength_removed_true": len((quartet & truth) - validated),
        "mean_true_quartet_lcb": _finite_mean(true_lower),
        "mean_false_quartet_lcb": _finite_mean(false_lower),
        "edge_strength_threshold": result.edge_strength_threshold,
        "elapsed_seconds": time.perf_counter() - started,
        "error": "",
    }
    for name, item in scores.items():
        for metric_name, value in item.items():
            row[f"{name}_{metric_name}"] = value
    edge_rows = []
    for clade, record in result.quartet_evidence.items():
        edge_rows.append(
            {
                "condition_id": condition_id,
                "case": case,
                "data_regime": data_regime,
                "noise_regime": noise_regime,
                "trial": trial,
                **record.to_dict(),
                "is_true_clade": clade in truth,
                "edge_strength": result.clade_edge_strength.get(clade, 0.0),
                "quartet_kept": clade in quartet,
                "validated_kept": clade in validated,
            }
        )
    return row, edge_rows


def _long_frame(frame: pd.DataFrame) -> pd.DataFrame:
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


def _write_report(output: Path, frame: pd.DataFrame, long: pd.DataFrame) -> None:
    overall = (
        long.groupby("method", as_index=False)
        .agg(mean_f1=("f1", "mean"), min_f1=("f1", "min"), exact_rate=("exact", "mean"))
        .sort_values("method")
    )
    by_case = (
        long.groupby(["case", "method"], as_index=False)
        .agg(mean_f1=("f1", "mean"), exact_rate=("exact", "mean"))
    )
    lines = [
        "# AC reranking and quartet large sweep",
        "",
        "All voltage samples are generated by radial AC power flow. No GTLS is used.",
        "",
        f"- Completed conditions: {len(frame)}",
        f"- Cases: {', '.join(sorted(frame['case'].unique()))}",
        "- Data: sparse=3x96, medium=5x288, rich=9x960",
        "- Noise: low=(P/Q 0.25%, V 0.01%), nominal=(0.5%, 0.02%), high=(1%, 0.05%)",
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
    lines.extend(
        [
            "",
            "## By case",
            "",
            "| case | method | mean F1 | exact rate |",
            "|---|---|---:|---:|",
        ]
    )
    for _, row in by_case.iterrows():
        lines.append(
            f"| {row['case']} | {row['method']} | "
            f"{row['mean_f1']:.4f} | {row['exact_rate']:.2%} |"
        )
    candidate_miss = int((~frame["candidate_contains_exact"].astype(bool)).sum())
    ac_misrank = int(
        (
            frame["candidate_contains_exact"].astype(bool)
            & ~frame["ac_bic_exact"].astype(bool)
        ).sum()
    )
    quartet_help = int((frame["validated_f1"] > frame["ac_bic_f1"] + 1e-12).sum())
    quartet_harm = int((frame["validated_f1"] + 1e-12 < frame["ac_bic_f1"]).sum())
    lines.extend(
        [
            "",
            "## Failure decomposition",
            "",
            f"- Exact topology absent from RNJ candidates: {candidate_miss}",
            f"- Exact candidate present but AC+BIC did not select it: {ac_misrank}",
            f"- Quartet/effect contraction improved F1: {quartet_help}",
            f"- Quartet/effect contraction reduced F1: {quartet_harm}",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(output: Path, long: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    methods = list(METHOD_COLUMNS)
    cases = sorted(long["case"].unique())
    summary = long.groupby(["case", "method"], as_index=False)["f1"].mean()
    fig, axis = plt.subplots(figsize=(10, 4.8))
    width = 0.19
    x = np.arange(len(cases))
    for method_index, method in enumerate(methods):
        values = [
            float(summary.loc[
                summary["case"].eq(case) & summary["method"].eq(method), "f1"
            ].iloc[0])
            for case in cases
        ]
        axis.bar(x + (method_index - 1.5) * width, values, width, label=method)
    axis.set_xticks(x, cases)
    axis.set_ylim(0.5, 1.01)
    axis.set_ylabel("Mean rooted-clade F1")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "mean_f1_by_case.png", dpi=180)
    plt.close(fig)


def _persist(output: Path, rows: list[dict], edge_rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("condition_id")
    frame.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(output / "edge_evidence.csv", index=False)


def run(args: argparse.Namespace) -> None:
    output = ensure_dir(args.output)
    metrics_path = output / "metrics.csv"
    edges_path = output / "edge_evidence.csv"
    rows = pd.read_csv(metrics_path).to_dict("records") if args.resume and metrics_path.exists() else []
    edge_rows = pd.read_csv(edges_path).to_dict("records") if args.resume and edges_path.exists() else []
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
        for trial in range(args.trials)
    ]
    total = len(conditions)
    started = time.perf_counter()
    processed = 0
    for index, (case, data_regime, noise_regime, trial) in enumerate(conditions, start=1):
        condition_id = f"{case}__{data_regime}__{noise_regime}__trial{trial}"
        if condition_id in completed:
            print(f"[{index:03d}/{total:03d}] skip {condition_id}", flush=True)
            continue
        print(f"[{index:03d}/{total:03d}] run  {condition_id}", flush=True)
        try:
            row, evidence = _evaluate(
                condition_id,
                case,
                data_regime,
                noise_regime,
                trial,
                args.quartet_replicates,
            )
            rows.append(row)
            edge_rows.extend(evidence)
            print(
                f"    fixed={row['baseline_f1']:.4f} ac={row['ac_bic_f1']:.4f} "
                f"validated={row['validated_f1']:.4f} "
                f"oracle={row['candidate_oracle_f1']:.4f}",
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
        _persist(output, rows, edge_rows)
        elapsed = time.perf_counter() - started
        eta = elapsed / processed * max(total - index, 0) if processed else 0.0
        print(f"    elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

    frame = pd.DataFrame(rows)
    frame = frame.loc[frame["status"].eq("ok")].copy()
    if frame.empty:
        raise RuntimeError("no conditions completed successfully")
    long = _long_frame(frame)
    long.to_csv(output / "method_metrics_long.csv", index=False)
    (
        long.groupby("method", as_index=False)
        .agg(mean_f1=("f1", "mean"), min_f1=("f1", "min"), exact_rate=("exact", "mean"))
        .to_csv(output / "summary_by_method.csv", index=False)
    )
    (
        long.groupby(["case", "method"], as_index=False)
        .agg(mean_f1=("f1", "mean"), exact_rate=("exact", "mean"))
        .to_csv(output / "summary_by_case.csv", index=False)
    )
    (
        long.groupby(["data_regime", "noise_regime", "method"], as_index=False)
        .agg(mean_f1=("f1", "mean"), min_f1=("f1", "min"), exact_rate=("exact", "mean"))
        .to_csv(output / "summary_by_regime.csv", index=False)
    )
    write_json(
        output / "summary.json",
        {
            "completed_conditions": len(frame),
            "requested_conditions": total,
            "quartet_replicates": args.quartet_replicates,
            "ac_likelihood_mode": "eiv_gaussian_ac_refined",
            "quartet_contradiction_effect": 0.05,
            "cases": args.cases,
            "data_regimes": args.data_regimes,
            "noise_regimes": args.noise_regimes,
            "trials": args.trials,
            "mean_baseline_f1": float(frame["baseline_f1"].mean()),
            "mean_ac_bic_f1": float(frame["ac_bic_f1"].mean()),
            "mean_quartet_f1": float(frame["quartet_f1"].mean()),
            "mean_validated_f1": float(frame["validated_f1"].mean()),
            "validated_exact_rate": float(frame["validated_exact"].mean()),
            "candidate_exact_coverage": float(frame["candidate_contains_exact"].mean()),
        },
    )
    _write_report(output, frame, long)
    _write_plot(output, long)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/ac_quartet_large_sweep")
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
        default=["sparse", "medium", "rich"],
    )
    parser.add_argument(
        "--noise-regimes",
        nargs="+",
        choices=sorted(NOISE_REGIMES),
        default=["low", "nominal", "high"],
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--quartet-replicates", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.quartet_replicates <= 0:
        parser.error("--quartet-replicates must be positive")
    run(args)


if __name__ == "__main__":
    main()
