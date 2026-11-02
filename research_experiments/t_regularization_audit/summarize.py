"""Summarize saved cardinality profiles; performs no optimization or refit.

Alpha-grid rows are diagnostic T-domain choices, not complete RNJ/WZZT/E
algorithm runs.  Their validation loss is evaluated on saved raw coefficients.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def split_arrays(path, count=None):
    with np.load(Path(path), allow_pickle=False) as archive:
        indices = np.arange(96) if count is None else np.arange(0, 96, 288 // int(count))
        labels = archive["labels"].astype(int).tolist()
        p, q, v = [np.asarray(archive[key][:, indices], float).reshape(-1, len(labels)) for key in ("P", "Q", "V")]
        root = np.asarray(archive["root"][:, indices], float).reshape(-1)
    return p, q, root[:, None] ** 2 - v ** 2


def raw_loss(point, arrays):
    p, q, y = arrays
    n = p.shape[1]
    z = np.asarray([[i in C for i in range(n)] for C in point["supports"]], float)
    R = (z.T * np.asarray(point["r"])) @ z
    X = (z.T * np.asarray(point["x"])) @ z
    return float(np.mean(np.abs(y - p @ R.T - q @ X.T)))


def topology(point, n):
    return {tuple(c) for c, r, x in zip(point["supports"], point["r"], point["x"])
            if 1 < len(c) < n and max(r, x) > 1e-9}


def metrics(predicted, truth):
    tp, fp, fn = len(predicted & truth), len(predicted - truth), len(truth - predicted)
    return dict(F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
                exact=predicted == truth, true_positive=tp, false_positive=fp,
                false_negative=fn, positive_count=len(predicted))


def no_refit_contraction(point, calibration, n, threshold=.24):
    """Only audit which supports pass the common threshold; never refit weights."""
    a, b, depth = (float(calibration[k]) for k in ("a", "b", "median_root_depth"))
    return {tuple(c) for c, r, x in zip(point["supports"], point["r"], point["x"])
            if 1 < len(c) < n and max(r, x) > 1e-9 and (a*r + b*x + 1e-10 >= threshold*depth)}


def alpha_value(value, f0):
    return None if value is None else value / f0


def markdown(rows, fields):
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field)
            cells.append("unavailable" if value is None else f"{value:.9g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarize(output, snapshot, alphas):
    output, snapshot = Path(output).resolve(), Path(snapshot).resolve()
    catalog = read(snapshot / "catalog.json")
    with (ROOT / "docs/full_ablation_20261102/lost_exact_cases.csv").open(encoding="utf-8", newline="") as stream:
        planned_jobs = [row["job"] for row in csv.DictReader(stream)]
    profiles, intervals, grid, missing, receipts = [], [], [], [], []
    for job in planned_jobs:
        file = output / "jobs" / job / "profile.json"
        if not file.exists():
            missing.append(job)
            continue
        receipt = read(file)
        receipts.append(receipt)
        meta = catalog[job]
        train, validation = split_arrays(meta["train"], meta["N"]), split_arrays(meta["validation"])
        assert {name: sha(meta[name]) for name in ("train", "validation")} == receipt["input_hashes"]
        common = read(snapshot / "common" / job / "common.json")
        truth = {tuple(c) for c in receipt["truth_reference"]["internal"]}
        f0, protected, n = receipt["reference"], receipt["protected_count"], receipt["n"]
        possible = []
        for row in receipt["profiles"]:
            entry = dict(job=job, q=row["selected_candidates"], encoded_count=row["encoded_total_count"],
                         phi_upper=row.get("phi_upper"), phi_lower=row.get("phi_lower"),
                         status=row.get("status"), relative_gap=row.get("relative_gap"), seconds=row.get("seconds"))
            if row.get("point") is not None:
                point = row["point"]
                measured = raw_loss(point, train)
                assert abs(measured - row["phi_upper"]) <= 1e-10
                entry.update(raw_train_MAE=measured, raw_validation_MAE=raw_loss(point, validation),
                             **metrics(topology(point, n), truth))
                contracted = metrics(no_refit_contraction(point, common["calibration"], n), truth)
                entry.update({"threshold_only_" + key: value for key, value in contracted.items()})
                possible.append((row, entry))
            profiles.append(entry)
        necessary = receipt.get("necessary_lambda_interval", {})
        sufficient = receipt.get("sufficient_lambda_interval", {})
        oracle = receipt["truth_reference"]
        oracle_loss = oracle.get("loss")
        interval = dict(job=job, profiles=len(receipt["profiles"]), planned_profiles=receipt["optional_capacity"] + 1,
                        complete_profiles=receipt["complete_profiles"], all_profiles_gap_met=receipt["all_profiles_gap_met"],
                        protected_count=protected, true_count=oracle["truth_count"],
                        extra_protected_count=len(oracle["extra_protected"]), true_q=oracle.get("selected_candidates"),
                        true_encoded_count=oracle.get("encoded_total_count"), true_loss=oracle_loss,
                        true_domain_feasible=oracle["feasible_topology"], alpha_current=receipt["current_lambda"] / f0,
                        cross_cardinality_necessary_alpha_lower=alpha_value(necessary.get("lower"), f0),
                        cross_cardinality_necessary_alpha_upper=alpha_value(necessary.get("upper"), f0),
                        cross_cardinality_sufficient_alpha_lower=alpha_value(sufficient.get("lower"), f0),
                        cross_cardinality_sufficient_alpha_upper=alpha_value(sufficient.get("upper"), f0),
                        same_count_improvement=-necessary["same_count_margin"] if necessary.get("same_count_margin") is not None else None,
                        same_count_counterexample=necessary.get("same_count_strictly_better"),
                        necessary_interval_empty=necessary.get("interval_empty"),
                        sufficient_interval_empty=sufficient.get("interval_empty"),
                        diagnostic_seconds=receipt["seconds"], budget_exceeded=receipt["budget_exceeded"],
                        profile_sha256=sha(file))
        complete_bounds = receipt["complete_profiles"] and all(row.get("phi_lower") is not None for row in receipt["profiles"])
        same_count_lower = sufficient.get("same_count_margin")
        interval.update(envelope_scope="cross-cardinality only; requires the separate same-q condition",
                        true_reference_ruled_out=bool(necessary.get("same_count_strictly_better") or necessary.get("interval_empty")),
                        true_reference_supported_within_tol=bool(oracle["feasible_topology"] and complete_bounds
                                                                and not necessary.get("same_count_strictly_better")
                                                                and same_count_lower is not None and same_count_lower >= -1e-10
                                                                and not sufficient.get("interval_empty")),
                        same_count_lower_margin=same_count_lower,
                        reference_support_tolerance=1e-10)
        if oracle_loss is not None:
            interval["true_independent_LP_difference"] = abs(oracle_loss - oracle["independent_LP"]["objective"])
        intervals.append(interval)
        for alpha in alphas:
            lam = alpha * f0
            if not possible:
                continue
            selected, evaluated = min(possible, key=lambda pair: (pair[0]["phi_upper"] + lam * pair[0]["selected_candidates"],
                                                                  pair[0]["selected_candidates"]))
            objective = selected["phi_upper"] + lam * (protected + selected["selected_candidates"])
            bound_rows = [row["phi_lower"] + lam * (protected + row["selected_candidates"])
                          for row in receipt["profiles"] if row.get("phi_lower") is not None]
            bound = min(bound_rows) if len(bound_rows) == len(receipt["profiles"]) and receipt["complete_profiles"] else None
            abs_gap = None if bound is None else max(0., objective - bound)
            requested_tolerance = max(1e-10, .0001 * abs(objective))
            entry = dict(job=job, alpha=alpha, penalty=lam, selected_q=selected["selected_candidates"],
                         selected_encoded_count=protected+selected["selected_candidates"], objective=objective,
                         lower_bound=bound, absolute_gap=abs_gap,
                         global_gap_met=abs_gap is not None and abs_gap <= requested_tolerance,
                         raw_train_MAE=evaluated["raw_train_MAE"], raw_validation_MAE=evaluated["raw_validation_MAE"],
                         **{key: evaluated[key] for key in ("F1", "exact", "positive_count", "true_positive", "false_positive", "false_negative")},
                         **{key: value for key, value in evaluated.items() if key.startswith("threshold_only_")})
            if oracle_loss is not None:
                reference = oracle_loss + lam * oracle["encoded_total_count"]
                entry.update(true_reference_objective=reference, selected_minus_true_objective=objective-reference)
            grid.append(entry)
    analysis = output / "analysis"
    write_csv(analysis / "phi_by_count.csv", profiles)
    write_csv(analysis / "lambda_intervals.csv", intervals)
    write_csv(analysis / "alpha_grid_raw_domain.csv", grid)
    aggregate = []
    for alpha in alphas:
        rows = [row for row in grid if row["alpha"] == alpha]
        aggregate.append(dict(alpha=alpha, planned=len(planned_jobs), completed=len(rows),
                              raw_F1=float(np.mean([r["F1"] for r in rows])) if rows else None,
                              raw_exact=sum(r["exact"] for r in rows),
                              threshold_only_F1=float(np.mean([r["threshold_only_F1"] for r in rows])) if rows else None,
                              threshold_only_exact=sum(r["threshold_only_exact"] for r in rows),
                              global_gap_met=sum(r["global_gap_met"] for r in rows)))
    write_csv(analysis / "alpha_grid_summary.csv", aggregate)
    summary = dict(planned_jobs=planned_jobs, returned=len(receipts), missing=missing,
                   profile_MIPs=sum(len(r["profiles"]) for r in receipts),
                   profile_gap_met=sum(row.get("status") == 2 for r in receipts for row in r["profiles"]),
                   profile_timeouts=sum(row.get("status") == 9 for r in receipts for row in r["profiles"]),
                   incomplete_jobs=[r["job"] for r in receipts if not r["complete_profiles"]],
                   diagnostic_seconds=sum(r["seconds"] for r in receipts), alphas=alphas,
                   same_count_counterexamples=sum(bool(r["same_count_counterexample"]) for r in intervals),
                   necessary_interval_empty=sum(bool(r["necessary_interval_empty"]) for r in intervals),
                   true_reference_ruled_out=sum(r["true_reference_ruled_out"] for r in intervals),
                   true_reference_supported_within_tol=sum(r["true_reference_supported_within_tol"] for r in intervals),
                   candidate_selection_uses_truth=False, complete_pipeline_runs=0,
                   optimizer_invoked_by_summary=False, refits_invoked_by_summary=False,
                   semantics="Raw cardinality-profile T-domain choices; threshold-only fields apply saved .24 rule without LP refit; neither is complete pipeline output",
                   driver_sha256=sha(__file__), profiles={row["job"]: row["profile_sha256"] for row in intervals})
    write(analysis / "verification.json", summary)
    lines = ["# T 正则机制诊断：保存剖面的独立后分析", "",
             "只比较固定 T 域中的训练最优结构；未运行逐步搜索、最终 E、收缩后 LP 或最终路径验证。", "",
             f"已返回 {len(receipts)}/{len(planned_jobs)} 输入，基数 MILP {summary['profile_MIPs']} 个，达到设定精度 {summary['profile_gap_met']} 个。", "",
             "alpha=lambda/f0。计数 q 是候选二元变量数，固定骨架另计常数成本；正权重线路数分别报告。", "",
             "以下数值区间只比较不同 q 的竞争结构，还必须满足同 q 条件。Flynn 虽有非空的跨基数区间，同 q 竞争解已严格优于真实参考，因此该数值区间不是让真实参考最优的可用参数区间。", "",
             markdown(intervals, ["job", "true_q", "cross_cardinality_necessary_alpha_lower", "cross_cardinality_necessary_alpha_upper", "same_count_counterexample", "true_reference_ruled_out", "true_reference_supported_within_tol"]), "",
             "下表的 raw 指原始系数；threshold_only 只对已存系数应用0.24阈值，没有重新拟合。它们不能当作完整算法成绩。", "",
             markdown(aggregate, ["alpha", "completed", "raw_F1", "raw_exact", "threshold_only_F1", "threshold_only_exact", "global_gap_met"]), "",
             "真值仅用于事后评分和参考 LP；alpha-grid 在所有保存基数解中按训练目标最小选择，没有用真值择优。", ""]
    (analysis / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default=str(ROOT / "outputs/t_regularization_audit_20261102"))
    parser.add_argument("--snapshot", default=str(ROOT / "outputs/full_ablation_20261102"))
    parser.add_argument("--alpha", type=float, action="append")
    args = parser.parse_args()
    summarize(args.output, args.snapshot, args.alpha or [.003, .006, .01, .02, .03])


if __name__ == "__main__":
    main()
