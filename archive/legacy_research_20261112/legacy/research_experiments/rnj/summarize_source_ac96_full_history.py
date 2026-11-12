"""Read-only scoring/reporting of the six full-history AC96 saved jobs.

This module invokes no estimator or solver. Saved 1SE indices determine model
selection; physical truth is used afterward for descriptive scoring of every
certified path point. Outputs are written to a separate reporting directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from research_experiments.rnj.audit_source_ac96_benchmark import (
    clade_metrics,
    physical_truth,
    relative_error,
)

ROOT = Path(__file__).resolve().parents[2]
ACTIVE_TOLERANCE = 1e-8
SUMMARY_COLUMNS = (
    "job_id", "method", "budget_multiplier", "status", "stop_reason", "has_model",
    "family_f1", "active_f1", "r_error_percent", "x_error_percent", "compute_seconds", "time_basis",
    "selected_path_index", "selected_K", "last_K", "bound_expansions", "train_mae",
    "validation_mae", "validation_se", "path_length", "attempt_count",
    "solver_time_limit_count", "uncertified_solve_count", "per_solve_seconds", "method_wall_seconds",
)
PATH_COLUMNS = (
    "job_id", "method", "budget_multiplier", "bound_expansions", "r_upper_bound", "x_upper_bound",
    "path_index", "iteration", "K", "active_K", "family_f1", "active_f1", "r_error_percent",
    "x_error_percent", "train_mae", "validation_mae", "validation_se", "accepted_gain",
    "selected", "in_final_path", "optimality_certified", "solver_seconds", "fingerprint",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def finite(value):
    return value is not None and isinstance(value, (float, int, np.number)) and np.isfinite(value)


def optimality_certified(diagnostics):
    if diagnostics.get("status") != 0 or not diagnostics.get("success"):
        return False
    if diagnostics.get("binary_variable_count") == 0:
        return True
    primal, dual = diagnostics.get("objective"), diagnostics.get("dual_bound")
    if finite(primal) and finite(dual) and abs(primal - dual) <= 1e-10:
        return True
    gap = diagnostics.get("mip_gap")
    return finite(gap) and 0 <= gap <= 1e-10


def _fingerprint(point):
    keys = ("support_indices", "r_values", "x_values", "train_mae", "validation_mae")
    body = json.dumps({key: point.get(key) for key in keys}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def score_point(point, labels, truth):
    supports = point["support_indices"]
    r, x = np.asarray(point["r_values"]), np.asarray(point["x_values"])
    family, active = [], []
    for support, rv, xv in zip(supports, r, x, strict=True):
        if 1 < len(support) < len(labels):
            clade = [labels[index] for index in support]
            family.append(clade)
            if max(rv, xv) > ACTIVE_TOLERANCE:
                active.append(clade)
    family_score = clade_metrics(family, truth["clades"], labels)
    active_score = clade_metrics(active, truth["clades"], labels)
    return {
        "K": len(supports), "active_K": int(np.count_nonzero(np.maximum(r, x) > ACTIVE_TOLERANCE)),
        "family_f1": family_score["clade_f1"], "active_f1": active_score["clade_f1"],
        "r_error_percent": 100.0 * relative_error(point["r_matrix"], truth["R"]),
        "x_error_percent": 100.0 * relative_error(point["x_matrix"], truth["X"]),
    }


def certified_path_rows(job, model, truth, progress_snapshots):
    """Keep all certified epochs; mark final and saved-selected points only."""
    labels, final = model["terminal_labels"], model.get("path", [])
    final_fingerprints = {_fingerprint(point) for point in final}
    selected_index = model.get("selected_path_index")
    chosen = (_fingerprint(final[selected_index]) if selected_index is not None and final else None)
    epoch = model.get("bound_expansions", 0)
    final_rows = []
    warnings = []
    def rows_for(state, in_final):
        rows = []
        for index, point in enumerate(state.get("path", [])):
            certified = optimality_certified(point.get("solver", {}))
            if not certified:
                warnings.append(f"{job['job_id']}: path point {index} has no strict certificate")
                continue
            fingerprint = _fingerprint(point)
            is_final = in_final or (
                state.get("bound_expansions", 0) == epoch and fingerprint in final_fingerprints
            )
            rows.append({
                "job_id": job["job_id"], "method": job["method"], "budget_multiplier": job["budget_multiplier"],
                "bound_expansions": state.get("bound_expansions", 0),
                "r_upper_bound": state.get("r_upper_bound"), "x_upper_bound": state.get("x_upper_bound"),
                "path_index": index, "iteration": point["iteration"], **score_point(point, labels, truth),
                "train_mae": point["train_mae"], "validation_mae": point.get("validation_mae"),
                "validation_se": point.get("validation_se"), "accepted_gain": point.get("accepted_gain"),
                "selected": bool(is_final and fingerprint == chosen), "in_final_path": bool(is_final),
                "optimality_certified": True, "solver_seconds": point["solver"].get("runtime_seconds"),
                "fingerprint": fingerprint,
            })
        return rows
    final_rows = rows_for(model, True)
    unique = {}
    for state in [*progress_snapshots, model]:
        for row in rows_for(state, state is model):
            unique[(row["bound_expansions"], row["fingerprint"])] = row
    history_rows = sorted(unique.values(), key=lambda row: (row["bound_expansions"], row["path_index"]))
    return final_rows, history_rows, list(dict.fromkeys(warnings))


def summarize_jobs(run_directory, dataset_directory):
    run_directory, dataset_directory = Path(run_directory), Path(dataset_directory)
    protocol = read_json(run_directory / "protocol.json")
    jobs = protocol["jobs"]
    if len(jobs) != 6:
        raise ValueError("expected a declared six-job full-history protocol")
    summaries, final_paths, all_paths, warnings = [], [], [], []
    read_files = [run_directory / "protocol.json"]
    truth_cache = {}
    for job in jobs:
        directory = run_directory / "jobs" / job["job_id"]
        result_path, model_path = directory / "result.json", directory / "model.json"
        result = read_json(result_path) if result_path.exists() else {}
        model = read_json(model_path) if model_path.exists() else {}
        read_files.extend(path for path in (result_path, model_path) if path.exists())
        row = {**dict.fromkeys(SUMMARY_COLUMNS), **job,
               "status": result.get("status", model.get("status", "not_run")),
               "stop_reason": result.get("stop_reason", model.get("stop_reason", "pending")),
               "has_model": bool(model.get("has_model", False)),
               "compute_seconds": result.get("compute_seconds", result.get("elapsed_seconds", model.get("elapsed_seconds"))),
               "time_basis": "final_method_elapsed" if result else ("latest_checkpoint_elapsed" if model else "pending"),
               "selected_path_index": model.get("selected_path_index"),
               "bound_expansions": model.get("bound_expansions"),
               "path_length": len(model.get("path", [])), "attempt_count": len(model.get("attempted_extensions", [])),
               "solver_time_limit_count": sum(e.get("status") == 1 for e in model.get("solver_events", [])),
               "uncertified_solve_count": sum(not e.get("optimality_certified", False) for e in model.get("solver_events", []))}
        if row["has_model"]:
            labels = model["terminal_labels"]
            key = tuple(labels)
            if key not in truth_cache:
                network = dataset_directory / job["case"] / "network"
                truth_cache[key] = physical_truth(network, labels)
                read_files.extend(network / name for name in ("metadata.json", "buses.csv", "branches.csv"))
            truth = truth_cache[key]
            npz_path = directory / "model.npz"
            with np.load(npz_path) as arrays:
                family = clade_metrics(model["family_clades"], truth["clades"], labels)
                active = clade_metrics(model["active_clades"], truth["clades"], labels)
                row.update(family_f1=family["clade_f1"], active_f1=active["clade_f1"],
                           r_error_percent=100.0 * relative_error(arrays["R"], truth["R"]),
                           x_error_percent=100.0 * relative_error(arrays["X"], truth["X"]))
                for suffix, target in (("train", "train_mae"), ("validation", "validation_mae")):
                    prediction = arrays[f"P_{suffix}"] @ arrays["R"].T + arrays[f"Q_{suffix}"] @ arrays["X"].T
                    row[target] = float(np.mean(abs(arrays[f"Y_{suffix}"] - prediction)))
            read_files.append(npz_path)
            row["validation_se"] = model.get("validation_se")
            if model.get("path"):
                snapshots = []
                for snapshot_path in sorted((directory / "progress").glob("*.json")):
                    snapshots.append(read_json(snapshot_path))
                    read_files.append(snapshot_path)
                current_rows, historical_rows, notes = certified_path_rows(job, model, truth, snapshots)
                final_paths.extend(current_rows)
                all_paths.extend(historical_rows)
                warnings.extend(notes)
                row["selected_K"] = len(model["support_indices"])
                row["last_K"] = len(model["path"][-1]["support_indices"])
        summaries.append(row)
    return protocol, summaries, final_paths, all_paths, warnings, list(dict.fromkeys(read_files))


def selection_effect_rows(summaries, paths):
    rows = []
    for summary in summaries:
        if "WZZT" not in summary["method"]:
            continue
        candidates = [r for r in paths if r["job_id"] == summary["job_id"]]
        selected = next((r for r in candidates if r["selected"]), None)
        last = candidates[-1] if candidates else None
        row = {"job_id": summary["job_id"], "method": summary["method"],
               "budget_multiplier": summary["budget_multiplier"], "status": summary["status"],
               "stop_reason": summary["stop_reason"], "truth_used_for_selection": False,
               "selection_rule": "saved core validation-1SE index; post-hoc truth scoring only"}
        for name, point in (("selected", selected), ("last", last)):
            for field in ("path_index", "K", "family_f1", "active_f1", "r_error_percent", "x_error_percent",
                          "train_mae", "validation_mae", "validation_se"):
                row[f"{name}_{field}"] = point[field] if point is not None else None
        for field in ("K", "family_f1", "active_f1", "r_error_percent", "x_error_percent", "train_mae", "validation_mae"):
            a, b = row[f"selected_{field}"], row[f"last_{field}"]
            row[f"selected_minus_last_{field}"] = a - b if finite(a) and finite(b) else None
        rows.append(row)
    return rows


def budget_effect_rows(summaries):
    rows = []
    for method in ("WZZT", "RNJ+WZZT"):
        one = next(r for r in summaries if r["method"] == method and r["budget_multiplier"] == 1)
        hundred = next(r for r in summaries if r["method"] == method and r["budget_multiplier"] == 100)
        returned = all(values["status"] in ("complete", "incomplete", "failed") for values in (one, hundred))
        row = {"method": method, "comparison": "same64/32 protocol; 100-times budget minus original budget",
               "both_jobs_returned": returned}
        for name, values in (("t1", one), ("t100", hundred)):
            for key in ("status", "stop_reason", "family_f1", "active_f1", "r_error_percent", "x_error_percent",
                        "compute_seconds", "selected_K", "last_K", "validation_mae", "bound_expansions"):
                row[f"{name}_{key}"] = values[key]
        for key in ("family_f1", "active_f1", "r_error_percent", "x_error_percent", "compute_seconds", "validation_mae"):
            a, b = hundred[key], one[key]
            row[f"t100_minus_t1_{key}"] = a - b if returned and finite(a) and finite(b) else None
        rows.append(row)
    return rows


def _display(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}g}"
    return str(value).replace("|", "/")


def markdown_table(rows):
    headings = ("方法", "预算", "状态", "family F1", "active F1", "R误差%", "X误差%", "秒", "选中K", "最后K", "扩界", "验证MAE", "停止原因")
    lines = ["| " + " | ".join(headings) + " |", "| " + " | ".join("---" for _ in headings) + " |"]
    for row in rows:
        values = [row["method"], "参照" if row["budget_multiplier"] is None else f"t×{row['budget_multiplier']}",
                  row["status"], row["family_f1"], row["active_f1"], row["r_error_percent"], row["x_error_percent"],
                  row["compute_seconds"], row["selected_K"], row["last_K"], row["bound_expansions"],
                  row["validation_mae"], row["stop_reason"]]
        lines.append("| " + " | ".join(_display(value) for value in values) + " |")
    return "\n".join(lines)


def plot_paths(paths, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = []
    for method in ("WZZT", "RNJ+WZZT"):
        rows = [row for row in paths if row["method"] == method]
        if not rows:
            continue
        figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
        for multiplier, color in ((1, "#2463a4"), (100, "#d16d16")):
            current = [row for row in rows if row["budget_multiplier"] == multiplier]
            if not current:
                continue
            x = [row["path_index"] for row in current]
            axes[0].plot(x, [row["train_mae"] for row in current], "--", color=color, marker=".", label=f"t{multiplier} train")
            axes[0].plot(x, [row["validation_mae"] for row in current], "-", color=color, marker=".", label=f"t{multiplier} validation")
            axes[1].plot(x, [row["family_f1"] for row in current], "-", color=color, marker=".", label=f"t{multiplier} family F1")
            axes[1].plot(x, [row["active_f1"] for row in current], ":", color=color, label=f"t{multiplier} active F1")
            for selected in (row for row in current if row["selected"]):
                axes[0].scatter([selected["path_index"]], [selected["validation_mae"]], s=80, facecolors="none", edgecolors=color, linewidths=2)
                axes[1].scatter([selected["path_index"]], [selected["family_f1"]], s=80, facecolors="none", edgecolors=color, linewidths=2)
        axes[0].set(title=f"{method}: certified final-domain paths (circles = validation 1SE selection)", ylabel="MAE, squared-voltage p.u.")
        axes[1].set(xlabel="Certified path index", ylabel="Post-hoc topology F1", ylim=(-.03, 1.03))
        for axis in axes:
            axis.grid(True, alpha=.25)
            axis.legend(fontsize=8, ncol=2)
        maximum_index = max(row["path_index"] for row in rows)
        if maximum_index <= 12:
            axes[1].set_xticks(range(maximum_index + 1))
        if maximum_index == 0:
            axes[1].set_xlim(-.25, .25)
        path = output / ("WZZT_paths.png" if method == "WZZT" else "RNJ_plus_WZZT_paths.png")
        figure.savefig(path, dpi=180)
        plt.close(figure)
        figures.append(path)
    return figures


def write_report(output, run, summaries, selection, figures, warnings, diagnosis_path):
    counts = dict(Counter(row["status"] for row in summaries))
    finished = sum(row["status"] in ("complete", "incomplete", "failed") for row in summaries)
    lines = ["# Pengwah18无源：完整主线、单日64/32与时间预算对照", "",
             f"本快照已有{finished}/6项任务返回。状态计数：{counts}。所有预定任务均保留在表中；运行中的分数和耗时来自最近认证路径快照，耗时是已保存的检查点时间。", "",
             markdown_table(summaries), "",
             "原始一天96点保持不变：前64点拟合，后32点用于验证1SE选模。NJ/RNJ仅使用训练64点；RNJ固定块也仅在训练部分按100次、块长4、阈值0.75、最多2块、种子20260903筛选。后32点参与模型选择，因此本轮没有独立测试集。", "",
             "WZZT直接调用完整核心流程：singleton初始化、联合方法追加主线RNJ固定块；R/X原子初始上界2，触及99.5%时翻倍并重启，最多4次；近零阈值1e-9，非冻结零原子经重解保持目标后才剪枝。验证误差按4个连续8点块计算SE，在最小验证MAE加其SE范围内选择原子数最少的认证模型。", "",
             "family F1沿历史pipeline口径计算全部非平凡family；active F1筛选max(r,x)>1e-8，便于与旧96点实验的活跃支撑口径比较。两者均排除singleton和全终端支撑。R/X误差为相同平方电压灵敏度矩阵的Frobenius相对误差。", "",
             "t×1为每次求解60秒、每方法300秒；t×100为6000秒和30000秒。方法时间包括RNJ筛选、建模、多次求解、核心数值核对和保存，外部独立审计另行执行。所有超时incumbent及其gap保留在原始solver记录，正式分数只使用认证路径上的验证选中模型。", "",
             "## 验证选模与路径末端", "",
             "selection_effect.csv逐项比较同一路径上的1SE选中点与最后训练点。path_points.csv给出最终系数上界域的完整认证路径；all_certified_path_points.csv另保留扩界重启前的认证路径。路径F1与R/X误差均为事后描述，选模沿用运行时已经保存的1SE索引。", ""]
    for row in selection:
        if row["selected_K"] is not None:
            lines.append(f"- {row['method']} t×{row['budget_multiplier']}：选中K={row['selected_K']}，末端K={row['last_K']}；family F1 {_display(row['selected_family_f1'])}/{_display(row['last_family_f1'])}；验证MAE {_display(row['selected_validation_mae'])}/{_display(row['last_validation_mae'])}（前者为选中点）。")
    lines += ["", "budget_comparison.csv比较同一64/32协议的两档预算。当前网络与这一天的数据是比较范围；模型恢复差异按实际分数报告，训练误差与求解证书分别表示拟合和子问题数值求解状态。", "",
              "## 与此前诊断的关系", "",
              "此前96点全训练实验与本轮同时改变了拟合点数、验证选模、bootstrap种子及完整的扩界/剪枝流程。新旧成绩用于理解配置差异；时间预算的作用由本轮t×1和t×100配对结果判断。", ""]
    if diagnosis_path is not None and diagnosis_path.exists():
        diagnostic = read_json(diagnosis_path)
        details = diagnostic["summary"]
        lines += [f"保存真网的独立算术诊断显示：根有{details['root_degree']}条子支路，真实共享clade为{details['true_nontrivial_atoms']}个；真实原子R范围{details['true_r_range'][0]:.5g}–{details['true_r_range'][1]:.5g}、X范围{details['true_x_range'][0]:.5g}–{details['true_x_range'][1]:.5g}。singleton初始模型会吸收缺失的跨终端响应，原上界2覆盖真实线路原子。该事实解释初始化系数的含义；本轮路径记录用于检查后续搜索和选模的实际变化。", "",
                  f"[此前只读算术诊断]({diagnosis_path.resolve().as_posix()})", ""]
    lines += ["## 图与证据", ""]
    for figure in figures:
        lines += [f"![{figure.stem}]({figure.resolve().as_posix()})", ""]
    lines += [f"[原始冻结运行]({run.resolve().as_posix()})；独立审计结果应与原始运行中的auditor输出一起读取。", ""]
    if warnings:
        lines += ["读取检查发现以下问题；缺少严格证书的点未计入认证路径表：", ""] + [f"- {note}" for note in warnings] + [""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def summarize_run(run_directory, dataset_directory, output_directory, *, plots=True, diagnosis_path=None):
    run, output = Path(run_directory).resolve(), Path(output_directory).resolve()
    if output == run or output in [run / "jobs", run / "inputs", run / "source_snapshot"]:
        raise ValueError("write reports into a separate directory, preserving run files")
    output.mkdir(parents=True, exist_ok=True)
    protocol, summaries, paths, historical, warnings, files = summarize_jobs(run, dataset_directory)
    pd.DataFrame(summaries, columns=SUMMARY_COLUMNS).to_csv(output / "summary.csv", index=False, float_format="%.17g")
    pd.DataFrame(paths, columns=PATH_COLUMNS).to_csv(output / "path_points.csv", index=False, float_format="%.17g")
    pd.DataFrame(historical, columns=PATH_COLUMNS).to_csv(output / "all_certified_path_points.csv", index=False, float_format="%.17g")
    selection, budget = selection_effect_rows(summaries, paths), budget_effect_rows(summaries)
    pd.DataFrame(selection).to_csv(output / "selection_effect.csv", index=False, float_format="%.17g")
    pd.DataFrame(budget).to_csv(output / "budget_comparison.csv", index=False, float_format="%.17g")
    figures = plot_paths(paths, output) if plots else []
    diagnosis = Path(diagnosis_path) if diagnosis_path is not None else None
    write_report(output, run, summaries, selection, figures, warnings, diagnosis)
    if diagnosis is not None and diagnosis.exists():
        files.append(diagnosis)
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "read_only_postprocessing": True, "estimator_or_solver_invoked": False,
                "truth_used_for_selection": False, "run_directory": str(run), "expected_jobs": 6,
                "status_counts": dict(Counter(row["status"] for row in summaries)),
                "final_certified_path_points": len(paths), "all_epoch_certified_path_points": len(historical),
                "warnings": warnings, "protocol_sha256": sha256(run / "protocol.json"),
                "postprocessor_sha256": sha256(Path(__file__)),
                "input_sha256": {str(path.resolve()): sha256(path) for path in dict.fromkeys(files)},
                "outputs": [path.name for path in sorted(output.iterdir()) if path.is_file()]}
    write_json(output / "postprocessing_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/source_ac96_full_history_pengwah18_20261018")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "outputs/source_ac96_20261018")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, default=ROOT / "outputs/source_ac96_diagnosis_20261018/pengwah18_no_source_wzzt_readonly_analysis.json")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = summarize_run(args.run_dir, args.dataset_dir, args.output_dir,
                           plots=not args.no_plots, diagnosis_path=args.diagnosis)
    print(json.dumps({key: result[key] for key in ("expected_jobs", "status_counts", "final_certified_path_points", "warnings")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
