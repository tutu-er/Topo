"""Systematic success-rate sweep for edge-stability clustering.

This wrapper runs the perturbation-against-unperturbed-MST confidence experiment
under multiple data-volume and perturbation settings, then reports success rates
for the baseline terminal MST and the expanded cluster tree.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from experiments.run_stability_cluster_aggregation_demo import run as run_stability_cluster
from terminal_case33.utils.io import ensure_dir, write_json


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _slug_float(value: float) -> str:
    return f"{value:.6g}".replace(".", "p").replace("-", "m")


def _summarize_success(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Summarize baseline and expanded success rates for groups."""

    if rows.empty:
        return pd.DataFrame()
    data = rows.copy()
    data["baseline_exact_success"] = data["baseline_f1"] >= 0.999999
    data["expanded_exact_success"] = data["expanded_f1"] >= 0.999999
    data["baseline_high_success"] = data["baseline_f1"] >= 0.9
    data["expanded_high_success"] = data["expanded_f1"] >= 0.9
    data["improved"] = data["expanded_f1"] > data["baseline_f1"] + 1e-12
    data["not_worse"] = data["expanded_f1"] >= data["baseline_f1"] - 1e-12
    data["delta_f1"] = data["expanded_f1"] - data["baseline_f1"]
    return (
        data.groupby(group_cols, as_index=False)
        .agg(
            n=("expanded_f1", "size"),
            mean_baseline_f1=("baseline_f1", "mean"),
            mean_expanded_f1=("expanded_f1", "mean"),
            mean_delta_f1=("delta_f1", "mean"),
            baseline_exact_success_rate=("baseline_exact_success", "mean"),
            expanded_exact_success_rate=("expanded_exact_success", "mean"),
            baseline_high_success_rate=("baseline_high_success", "mean"),
            expanded_high_success_rate=("expanded_high_success", "mean"),
            improvement_rate=("improved", "mean"),
            not_worse_rate=("not_worse", "mean"),
            mean_cluster_count=("cluster_count", "mean"),
            mean_stable_edge_precision=("stable_edge_precision", "mean"),
        )
        .sort_values(group_cols)
    )


def run(
    output_dir: str | Path = "outputs/stability_cluster_systematic_test",
    cases: list[str] | None = None,
    scenario_counts: list[int] | None = None,
    t_counts: list[int] | None = None,
    thresholds: list[float] | None = None,
    v_extra_noise_rels: list[float] | None = None,
    pq_extra_noise_rel: float = 0.0025,
    replicates: int = 60,
    recipe: str = "raw_drop",
    distance_mode: str = "RX_75R_25X",
    pq_noise_rel: float = 0.005,
    root_voltage_mean: float = 1.02,
    seed: int = 20260709,
) -> dict:
    """Run the systematic stability-cluster sweep and write CSV summaries."""

    out_dir = ensure_dir(output_dir)
    selected_cases = cases or ["paper15", "soumalas11", "flynn16", "pengwah18"]
    selected_counts = scenario_counts or [1, 2, 3]
    selected_t = t_counts or [96, 240, 480]
    selected_thresholds = thresholds or [0.55, 0.65, 0.75, 0.85]
    selected_v_noise = v_extra_noise_rels or [0.00005, 0.0001, 0.0002]
    tasks = [
        (scenario_count, t_count, v_extra)
        for scenario_count in selected_counts
        for t_count in selected_t
        for v_extra in selected_v_noise
    ]
    all_rows = []
    start = time.time()
    print(
        "stability_cluster_systematic_test start "
        f"tasks={len(tasks)} cases={selected_cases} thresholds={selected_thresholds}",
        flush=True,
    )
    for task_idx, (scenario_count, t_count, v_extra) in enumerate(tasks, start=1):
        subdir = ensure_dir(
            out_dir
            / f"sc{scenario_count}_T{t_count}_vextra{_slug_float(v_extra)}_pqextra{_slug_float(pq_extra_noise_rel)}"
        )
        run_stability_cluster(
            output_dir=subdir,
            cases=selected_cases,
            scenario_count=scenario_count,
            t_count=t_count,
            recipe_name=recipe,
            distance_mode=distance_mode,
            replicates=replicates,
            thresholds=selected_thresholds,
            pq_extra_noise_rel=pq_extra_noise_rel,
            v_extra_noise_rel=v_extra,
            pq_noise_rel=pq_noise_rel,
            root_voltage_mean=root_voltage_mean,
            seed=seed + task_idx * 10000,
        )
        result = pd.read_csv(subdir / "stability_cluster_summary.csv")
        result["condition_id"] = subdir.name
        all_rows.append(result)
        partial = pd.concat(all_rows, ignore_index=True)
        partial.to_csv(out_dir / "all_results_partial.csv", index=False)
        best_partial = partial.sort_values(["expanded_f1", "threshold"], ascending=[False, True]).groupby(
            ["case", "scenario_count", "t_count", "v_extra_noise_rel"], as_index=False
        ).head(1)
        elapsed = time.time() - start
        eta = elapsed / task_idx * (len(tasks) - task_idx)
        print(
            f"[progress] {task_idx}/{len(tasks)} sc={scenario_count} T={t_count} "
            f"v_extra={v_extra:g} mean_base={result['baseline_f1'].mean():.3f} "
            f"mean_best={best_partial['expanded_f1'].mean():.3f} "
            f"time={_fmt_seconds(elapsed)} eta={_fmt_seconds(eta)}",
            flush=True,
        )

    rows = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    rows.to_csv(out_dir / "all_results.csv", index=False)

    best_by_case_condition = (
        rows.sort_values(["expanded_f1", "threshold"], ascending=[False, True])
        .groupby(["case", "scenario_count", "t_count", "v_extra_noise_rel"], as_index=False)
        .head(1)
    )
    best_by_case_condition.to_csv(out_dir / "best_by_case_condition.csv", index=False)

    summaries = {
        "by_threshold": _summarize_success(rows, ["threshold"]),
        "by_t_count": _summarize_success(best_by_case_condition, ["t_count"]),
        "by_scenario_count": _summarize_success(best_by_case_condition, ["scenario_count"]),
        "by_v_extra_noise": _summarize_success(best_by_case_condition, ["v_extra_noise_rel"]),
        "by_case": _summarize_success(best_by_case_condition, ["case"]),
        "by_t_scenario": _summarize_success(best_by_case_condition, ["t_count", "scenario_count"]),
    }
    for name, frame in summaries.items():
        frame.to_csv(out_dir / f"success_{name}.csv", index=False)

    best_fixed_threshold = summaries["by_threshold"].sort_values("mean_expanded_f1", ascending=False).head(1)
    metrics = {
        "cases": selected_cases,
        "scenario_counts": selected_counts,
        "t_counts": selected_t,
        "thresholds": selected_thresholds,
        "v_extra_noise_rels": selected_v_noise,
        "pq_extra_noise_rel": pq_extra_noise_rel,
        "replicates": replicates,
        "recipe": recipe,
        "distance_mode": distance_mode,
        "row_count": int(len(rows)),
        "condition_count": int(len(tasks)),
        "mean_baseline_f1_all_rows": float(rows["baseline_f1"].mean()) if len(rows) else float("nan"),
        "mean_expanded_f1_all_rows": float(rows["expanded_f1"].mean()) if len(rows) else float("nan"),
        "mean_best_expanded_f1_by_case_condition": float(best_by_case_condition["expanded_f1"].mean())
        if len(best_by_case_condition)
        else float("nan"),
        "best_fixed_threshold": best_fixed_threshold.to_dict(orient="records"),
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# Stability Cluster Systematic Test\n\n"
        "Success definitions: exact success means F1 >= 1.0, high success means F1 >= 0.9, "
        "improvement means expanded F1 is larger than baseline F1. The best-by-condition tables choose "
        "the threshold with the largest expanded F1 for each case/data/noise condition.\n\n"
        "## By Threshold\n\n"
        + summaries["by_threshold"].to_string(index=False)
        + "\n\n## By T Count\n\n"
        + summaries["by_t_count"].to_string(index=False)
        + "\n\n## By Scenario Count\n\n"
        + summaries["by_scenario_count"].to_string(index=False)
        + "\n\n## By Case\n\n"
        + summaries["by_case"].to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/stability_cluster_systematic_test")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--scenario-counts", nargs="*", type=int, default=None)
    parser.add_argument("--T-counts", nargs="*", type=int, default=None)
    parser.add_argument("--thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--v-extra-noise-rels", nargs="*", type=float, default=None)
    parser.add_argument("--pq-extra-noise-rel", type=float, default=0.0025)
    parser.add_argument("--replicates", type=int, default=60)
    parser.add_argument("--recipe", default="raw_drop")
    parser.add_argument("--distance-mode", default="RX_75R_25X")
    parser.add_argument("--pq-noise-rel", type=float, default=0.005)
    parser.add_argument("--root-voltage-mean", type=float, default=1.02)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()
    run(
        output_dir=args.output,
        cases=args.cases,
        scenario_counts=args.scenario_counts,
        t_counts=args.T_counts,
        thresholds=args.thresholds,
        v_extra_noise_rels=args.v_extra_noise_rels,
        pq_extra_noise_rel=args.pq_extra_noise_rel,
        replicates=args.replicates,
        recipe=args.recipe,
        distance_mode=args.distance_mode,
        pq_noise_rel=args.pq_noise_rel,
        root_voltage_mean=args.root_voltage_mean,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
