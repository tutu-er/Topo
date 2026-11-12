"""Summarize saved 64-job AC96 results without rerunning any estimator.

Primary method metrics are copied from saved result JSON, without rescoring.
Frozen-block correctness uses the parent evaluator's saved truth clades.  A
separate rejected-incumbent appendix reconstructs saved candidate atoms and
evaluates them against physical network truth after estimation.  No candidate
score replaces a primary result.  Missing metrics remain null.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import csv
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
METRICS = ("clade_f1", "r_relative_error", "x_relative_error", "compute_seconds")
SOURCE_NAMES = {"no_source": "无源", "constant": "常值源",
                "intermittent": "间歇源", "smooth": "平滑时变源"}


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _write_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows, fields=None):
    fields = fields or (list(rows[0]) if rows else [])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def collect(output_dir, protocol):
    rows, saved_jobs = [], {}
    for case in protocol["cases"]:
        for variant in protocol["variants"]:
            for method in protocol["methods"]:
                job_id = f"{case}__{variant}__{method}"
                directory = output_dir / "jobs" / job_id
                result = read_json(directory / "result.json")
                row = {"case": case, "variant": variant, "method": method,
                       "job_id": job_id, "status": "not_run", "has_model": False,
                       "search_complete": False, "stop_reason": "result_not_yet_available",
                       **dict.fromkeys(METRICS), "solver_time_limit_count": None,
                       "uncertified_solve_count": None, "bootstrap_failed": None,
                       "bootstrap_unattempted": None, "matrix_origin": None}
                if result is not None:
                    for key in ("status", "has_model", "search_complete", "stop_reason",
                                "solver_time_limit_count", "uncertified_solve_count",
                                "bootstrap_failed", "bootstrap_unattempted", "matrix_origin"):
                        row[key] = result.get(key, row[key])
                    for metric in METRICS:
                        raw = (result.get("compute_seconds", result.get("elapsed_seconds"))
                               if metric == "compute_seconds" else result.get("metrics", {}).get(metric))
                        row[metric] = number(raw)
                rows.append(row)
                saved_jobs[job_id] = {"directory": directory, "result": result}
    return rows, saved_jobs


def aggregate(rows, *, method, variant=None):
    subset = [r for r in rows if r["method"] == method
              and (variant is None or r["variant"] == variant)]
    counts = Counter(r["status"] for r in subset)
    result = {"method": method}
    if variant is not None:
        result["variant"] = variant
    result.update({"expected": len(subset),
                   "results_available": sum(r["status"] != "not_run" for r in subset),
                   "models_available": sum(bool(r["has_model"]) for r in subset),
                   "complete": counts["complete"], "incomplete": counts["incomplete"],
                   "failed": counts["failed"], "not_run": counts["not_run"],
                   "other_status": sum(v for k, v in counts.items()
                                       if k not in {"complete", "incomplete", "failed", "not_run"})})
    for metric in METRICS:
        values = [r[metric] for r in subset if r[metric] is not None]
        result[f"{metric}_n"] = len(values)
        result[f"mean_{metric}"] = sum(values) / len(values) if values else None
    available = result["clade_f1_n"]
    exact = sum(r["clade_f1"] is not None and math.isclose(r["clade_f1"], 1., abs_tol=1e-12,
                                                         rel_tol=0.) for r in subset)
    result.update(exact_count=exact, exact_available_denominator=available,
                  exact_expected_denominator=len(subset),
                  exact_over_available=exact / available if available else None,
                  exact_over_expected=exact / len(subset) if subset else None,
                  solver_time_limit_count=sum(r["solver_time_limit_count"] or 0 for r in subset),
                  uncertified_solve_count=sum(r["uncertified_solve_count"] or 0 for r in subset))
    for channel in ("r", "x"):
        error = result[f"mean_{channel}_relative_error"]
        result[f"mean_{channel}_error_percent"] = 100 * error if error is not None else None
    return result


FROZEN_FIELDS = ["case", "variant", "method", "job_id", "status", "block_number",
                 "terminal_labels", "support_indices", "final_r", "final_x", "active",
                 "in_saved_scored_clades", "correct_against_saved_evaluation",
                 "final_model_available", "result_available"]


def frozen_blocks(rows, saved_jobs, active_tolerance):
    records = []
    for row in rows:
        if row["method"] != "RNJ+WZZT":
            continue
        saved = saved_jobs[row["job_id"]]
        model = read_json(saved["directory"] / "model.json") or {}
        screen = (read_json(saved["directory"] / "rnj_screen.json")
                  or model.get("bootstrap") or {})
        selected_labels = screen.get("selected_support_labels", screen.get("selected", []))
        selected_indices = screen.get("selected_support_indices", [])
        terminal_labels = model.get("terminal_labels", [])
        final = {}
        # A running job can already have checkpoints.  Until result.json
        # exists, its provisional weights are not final_r/final_x values.
        if saved["result"] is not None:
            for support, r, x in zip(model.get("support_indices") or [],
                                      model.get("r_values") or [], model.get("x_values") or []):
                final[tuple(sorted(int(i) for i in support))] = (number(r), number(x))
        evaluation = (saved["result"] or {}).get("evaluation")
        truth = ({frozenset(c) for c in evaluation["truth_clades"]}
                 if evaluation and "truth_clades" in evaluation else None)
        scored = ({frozenset(c) for c in evaluation["predicted_clades"]}
                  if evaluation and "predicted_clades" in evaluation else None)
        for block, labels in enumerate(selected_labels):
            labels = sorted(int(v) for v in labels)
            indices = (tuple(sorted(int(i) for i in selected_indices[block]))
                       if block < len(selected_indices) else
                       tuple(sorted(terminal_labels.index(v) for v in labels))
                       if terminal_labels and all(v in terminal_labels for v in labels) else None)
            r, x = final.get(indices, (None, None))
            records.append({
                **{key: row[key] for key in ("case", "variant", "method", "job_id", "status")},
                "block_number": block + 1,
                "terminal_labels": ";".join(map(str, labels)),
                "support_indices": ";".join(map(str, indices)) if indices is not None else None,
                "final_r": r, "final_x": x,
                "active": max(r, x) > active_tolerance if r is not None and x is not None else None,
                "in_saved_scored_clades": frozenset(labels) in scored if scored is not None else None,
                "correct_against_saved_evaluation": frozenset(labels) in truth if truth is not None else None,
                "final_model_available": bool(row["has_model"]),
                "result_available": saved["result"] is not None,
            })
    return records


INCUMBENT_FIELDS = [
    "case", "variant", "method", "job_id", "method_status", "event_number", "kind",
    "accepted", "has_incumbent", "saved_valid", "optimality_certified", "uncertified",
    "solver", "solver_status", "solver_raw_status", "solver_message", "primal_objective",
    "dual_bound", "mip_gap", "incumbent_objective", "saved_train_mae", "objective_difference",
    "constraint_violation", "bound_violation", "integrality_violation", "source_product_violation",
    "runtime_seconds", "time_limit_seconds", "candidate_reconstruction_valid", "evaluation_error",
    "clade_f1", "precision", "recall", "r_relative_error", "x_relative_error",
    "predicted_clade_count", "matched_clade_count", "true_clade_count", "truth_clade_source",
    "support_indices", "r_weights", "x_weights", "source_h", "active_clade_labels",
]


def rejected_incumbents(rows, saved_jobs, dataset_dir, active_tolerance):
    """Evaluate saved, unaccepted feasible incumbents in a separate appendix.

    The auditor's physical path construction is used only after the estimator
    has produced its log.  Candidate F1 uses the declared active-atom threshold.
    ``saved_valid`` records the original residual/constraint audit; reconstruction
    validity additionally checks the saved dimensions and finite atom weights.
    """
    import numpy as np
    from research_experiments.rnj.audit_source_ac96_benchmark import (
        clade_metrics, physical_truth, relative_error,
    )

    records, true_cache = [], {}
    for row in rows:
        if row["method"] not in {"WZZT", "RNJ+WZZT"}:
            continue
        saved = saved_jobs[row["job_id"]]
        model = read_json(saved["directory"] / "model.json") or {}
        labels = tuple(int(v) for v in model.get("terminal_labels", []))
        for index, event in enumerate(model.get("solver_events", [])):
            incumbent = event.get("incumbent_audit") or {}
            if (event.get("accepted") is not False or event.get("optimality_certified") is True
                    or incumbent.get("has_incumbent") is not True
                    or incumbent.get("valid") is not True):
                continue
            record = dict.fromkeys(INCUMBENT_FIELDS)
            record.update({
                **{key: row[key] for key in ("case", "variant", "method", "job_id")},
                "method_status": row["status"], "event_number": index + 1, "kind": event.get("kind"),
                "accepted": False, "has_incumbent": True, "saved_valid": True,
                "optimality_certified": event.get("optimality_certified"),
                "uncertified": event.get("optimality_certified") is not True,
                "solver": event.get("solver"), "solver_status": event.get("status"),
                "solver_raw_status": event.get("raw_status"), "solver_message": event.get("message"),
                "primal_objective": number(event.get("objective")),
                "dual_bound": number(event.get("dual_bound")), "mip_gap": number(event.get("mip_gap")),
                "incumbent_objective": number(incumbent.get("solver_objective")),
                "saved_train_mae": number(incumbent.get("train_mae")),
                "objective_difference": number(incumbent.get("objective_difference")),
                "runtime_seconds": number(event.get("runtime_seconds")),
                "time_limit_seconds": number(event.get("time_limit_seconds")),
                "candidate_reconstruction_valid": False,
                "support_indices": json.dumps(incumbent.get("supports"), separators=(",", ":")),
                "r_weights": json.dumps(incumbent.get("r_weights"), separators=(",", ":")),
                "x_weights": json.dumps(incumbent.get("x_weights"), separators=(",", ":")),
                "source_h": json.dumps(incumbent.get("h"), separators=(",", ":")),
            })
            for key in ("constraint_violation", "bound_violation", "integrality_violation",
                        "source_product_violation"):
                record[key] = number(incumbent.get(key))
            try:
                n = len(labels)
                supports = [tuple(int(v) for v in s) for s in incumbent["supports"]]
                r = np.asarray(incumbent["r_weights"], dtype=float)
                x = np.asarray(incumbent["x_weights"], dtype=float)
                if (n == 0 or len(set(labels)) != n or r.shape != (len(supports),)
                        or x.shape != r.shape or not np.isfinite(r).all() or not np.isfinite(x).all()
                        or any(not s or len(set(s)) != len(s) or any(v < 0 or v >= n for v in s)
                               for s in supports)):
                    raise ValueError("saved incumbent atom dimensions/weights are invalid")
                z = np.zeros((len(supports), n))
                for k, support in enumerate(supports):
                    z[k, list(support)] = 1.
                candidate_r, candidate_x = (z.T * r) @ z, (z.T * x) @ z
                predicted = {frozenset(labels[i] for i in s)
                             for s, a, b in zip(supports, r, x, strict=True)
                             if 1 < len(s) < n and max(a, b) > active_tolerance}
                key = (row["case"], labels)
                if key not in true_cache:
                    true_cache[key] = physical_truth(dataset_dir / row["case"] / "network", labels)
                physical = true_cache[key]
                evaluation = (saved["result"] or {}).get("evaluation") or {}
                truth = evaluation.get("truth_clades", physical["clades"])
                scored = clade_metrics(predicted, truth, labels)
                record.update({
                    "candidate_reconstruction_valid": True,
                    **{metric: scored[metric] for metric in ("clade_f1", "precision", "recall")},
                    "r_relative_error": relative_error(candidate_r, physical["R"]),
                    "x_relative_error": relative_error(candidate_x, physical["X"]),
                    "predicted_clade_count": scored["predicted_clades"],
                    "matched_clade_count": scored["matched_clades"],
                    "true_clade_count": scored["true_clades"],
                    "truth_clade_source": ("saved_parent_evaluation" if "truth_clades" in evaluation
                                            else "physical_network_evaluation"),
                    "active_clade_labels": json.dumps(
                        [sorted(c) for c in sorted(predicted, key=lambda c: (len(c), tuple(sorted(c))))],
                        separators=(",", ":"),
                    ),
                })
            except (OSError, KeyError, TypeError, ValueError) as exc:
                record["evaluation_error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
    return records


def fmt(value, decimals=4, factor=1.):
    return "NA" if value is None else f"{value * factor:.{decimals}f}"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _aggregate_table(aggregates, variant=False):
    table = ["| " + ("源项 | " if variant else "")
             + "方法 | 模型/计划 | complete/incomplete/failed/未出结果 | 平均F1(n) | "
             "平均R误差%(n) | 平均X误差%(n) | 平均秒(n) | 完全恢复/可评分 | 完全恢复/计划 |",
             "|" + "---|" * (10 if variant else 9)]
    for a in aggregates:
        prefix = [SOURCE_NAMES.get(a["variant"], a["variant"])] if variant else []
        values = prefix + [a["method"], f'{a["models_available"]}/{a["expected"]}',
                           f'{a["complete"]}/{a["incomplete"]}/{a["failed"]}/{a["not_run"]}',
                           f'{fmt(a["mean_clade_f1"])} ({a["clade_f1_n"]})',
                           f'{fmt(a["mean_r_error_percent"], 2)} ({a["r_relative_error_n"]})',
                           f'{fmt(a["mean_x_error_percent"], 2)} ({a["x_relative_error_n"]})',
                           f'{fmt(a["mean_compute_seconds"], 2)} ({a["compute_seconds_n"]})',
                           f'{a["exact_count"]}/{a["exact_available_denominator"]}',
                           f'{a["exact_count"]}/{a["exact_expected_denominator"]}']
        table.append("| " + " | ".join(values) + " |")
    return table


def make_report(rows, by_method, by_variant, blocks, incumbents, protocol, *, partial, figure):
    finished = sum(r["status"] != "not_run" for r in rows)
    when = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))
    report = ["# 一天96点：NJ、RNJ、WZZT与RNJ+WZZT比较", "",
              f'生成时间：{when.isoformat(timespec="seconds")}。'
              f'{"临时结果" if partial else "完整运行记录"}：{finished}/{len(rows)}项已有结果。', "",
              "四个合成低压网络，各含无源、常值源、间歇源、平滑时变源；"
              "每组使用同一背景日的96个点。源P/Q时序已知，接入位置未知。"
              "本次只有一个背景日、一个数据实现，所有方法直接拟合这96点，独立验证点数为0。", "",
              "F1为保存结果中的非平凡终端簇F1；R/X误差为估计平方电压灵敏度矩阵相对真实"
              "LinDistFlow矩阵的Frobenius范数误差。表内误差乘100，以百分数表示。"
              "计算时间采用保存的算法总耗时，包含方法实际执行的前置拟合、bootstrap和求解。", "",
              f'每次求解上限{protocol.get("per_solve_seconds")}秒，'
              f'每方法上限{protocol.get("method_wall_seconds")}秒，'
              f'进程保护余量{protocol.get("hard_timeout_grace_seconds")}秒；'
              f'求解器{protocol.get("solver")}，单求解器线程。', "",
              "## 按方法汇总", "",
              "每方法计划分母固定为16。均值保留所有有相应数值的结果，包括incomplete时保留的模型；"
              "各列括号为实际均值分母。时间均值也保留记录了耗时的失败运行。"
              "缺失值为NA，不填零；完全恢复同时列出可评分分母和计划分母。", ""]
    report.extend(_aggregate_table(by_method))
    report.extend(["", "## 按源项和方法汇总", "",
                   "每个源项×方法计划分母为4；可评分分母随缺失结果变化。", ""])
    report.extend(_aggregate_table(by_variant, variant=True))
    if figure:
        report.extend(["", "## F1图", "", f"![F1 heatmap]({figure})", "",
                       "橙框表示incomplete；灰色NA表示尚无可评分结果。"])
    report.extend(["", "## 逐案例结果", ""])
    for case_name in protocol["cases"]:
        report.extend([f"### {case_name}", "",
                       "| 源项 | 方法 | F1 | R误差% | X误差% | 秒 | status | stop_reason |",
                       "|---|---|---:|---:|---:|---:|---|---|"])
        for r in rows:
            if r["case"] == case_name:
                report.append("| " + " | ".join([
                    SOURCE_NAMES.get(r["variant"], r["variant"]), r["method"],
                    fmt(r["clade_f1"]), fmt(r["r_relative_error"], 2, 100),
                    fmt(r["x_relative_error"], 2, 100), fmt(r["compute_seconds"], 2),
                    cell(r["status"]), cell(r["stop_reason"]),
                ]) + " |")
        report.append("")
    known_blocks = [b for b in blocks if b["correct_against_saved_evaluation"] is not None]
    wrong_blocks = sum(not b["correct_against_saved_evaluation"] for b in known_blocks)
    inactive = sum(b["active"] is False for b in blocks)
    report.extend(["## 固定的RNJ边界块", "",
                   f'当前保存{len(blocks)}个固定块，{len(known_blocks)}个可与已保存评价真值比较，'
                   f'其中{wrong_blocks}个不匹配；{inactive}个最终权重均未超过有效阈值。'
                   "逐块标签、最终R/X原子权重、有效状态与已保存评分中的去留见"
                   "[frozen_blocks.csv](frozen_blocks.csv)。运行中尚无最终结果的块仅列标签，"
                   "最终权重和有效状态保持NA。这些块的正确性仅用于结果解释。", ""])
    report.extend(["## 未接受的可行候选附表", "",
                   f'保存日志中有{len(incumbents)}个未被算法接受、原数值可行性审计通过、'
                   '且未获最优性认证的候选。逐事件的状态、可行目标、对偶界、gap、'
                   "目标值、候选F1和R/X误差见"
                   "[uncertified_incumbents.csv](uncertified_incumbents.csv)。", "",
                   "这些候选由保存的原子和权重重建，仅在评价阶段读取网络参数计算误差；"
                   "附表保留各事件的optimality_certified字段。候选属于对应单步子问题的可行解，"
                   "未认证行的最优性仍未知。主表继续使用算法最后接受的模型，"
                   "候选附表不计入四种方法的正式成绩。超时回退到单点模型的F1=0，"
                   "表示严格接受规则下保留模型的结构；日志同时记录已找到的未认证可行候选。", "",
                   "## 指标口径与解释", "",
                   "1. NJ与RNJ使用相同的前置ordered R/X回归，故同一数据组的R/X误差相同。"
                   "该误差评价回归矩阵；F1评价后续成树结果。WZZT的R/X由最后接受的相容原子构造。",
                   "2. 有源NJ/RNJ通过FWL投影估计自由终端源响应，WZZT联合选择物理源路径。"
                   "跨方法差异包含估计域、平方损失/L1损失与搜索策略的共同影响。"
                   "常值和间歇源的P/Q列秩为1，平滑源为2；单独的P/Q响应分解依赖实际联合设计秩。",
                   "3. NJ的0.16用于成树后的内部边收缩；RNJ的0.16用于递归分组与根连接，"
                   "两者数值相同而作用不同。阈值已预先固定。",
                   f'4. WZZT主F1只计max(r,x)>{protocol.get("active_atom_tolerance", 1e-8):g}'
                   "的非平凡原子；NJ/RNJ计各自阈值处理后的非平凡簇。"
                   "RNJ+WZZT固定但最终零权重的块仍约束搜索，其主F1按有效原子计分。",
                   "5. 源时序秩和回归设计秩是数值诊断；源响应布尔诊断字段不作为联合响应、"
                   "源位置或拓扑可识别性的证书。bootstrap频率只表示本次重采样稳定性。", "",
                   "complete表示算法按其停止规则结束；WZZT证书对象为固定支撑或一次扩展子问题，"
                   "整条贪心路径的全局最优性未验证。incomplete行保留最后接受的模型，"
                   "超时和未认证求解保存在原始job记录；failed/缺失模型的指标保持NA。", "",
                   "主表直接汇总已保存指标。候选附表单独在估计完成后进行物理网络评价；"
                   "本后处理未重新拟合、运行求解器或调参。", ""])
    return "\n".join(report)


def plot_f1(path, rows, protocol, partial):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    lookup = {(r["case"], r["variant"], r["method"]): r for r in rows}
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.2), layout="constrained")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dedede")
    image = None
    for axis, case_name in zip(axes.flat, protocol["cases"], strict=True):
        data = np.full((len(protocol["variants"]), len(protocol["methods"])), np.nan)
        for i, variant in enumerate(protocol["variants"]):
            for j, method in enumerate(protocol["methods"]):
                row = lookup[case_name, variant, method]
                if row["clade_f1"] is not None:
                    data[i, j] = row["clade_f1"]
        image = axis.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=0, vmax=1,
                            interpolation="nearest", aspect="auto")
        axis.set_xticks(range(len(protocol["methods"])), protocol["methods"])
        axis.set_yticks(range(len(protocol["variants"])), protocol["variants"])
        axis.set_title(case_name)
        for i, variant in enumerate(protocol["variants"]):
            for j, method in enumerate(protocol["methods"]):
                row = lookup[case_name, variant, method]
                value = data[i, j]
                axis.text(j, i, "NA" if np.isnan(value) else f"{value:.2f}", ha="center", va="center",
                          color="white" if not np.isnan(value) and value < .55 else "black", fontsize=10)
                if row["status"] == "incomplete":
                    axis.add_patch(Rectangle((j - .47, i - .47), .94, .94,
                                             fill=False, edgecolor="#ff8c00", linewidth=2.6))
        axis.set_xticks(np.arange(-.5, len(protocol["methods"]), 1), minor=True)
        axis.set_yticks(np.arange(-.5, len(protocol["variants"]), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=.5)
        axis.tick_params(which="minor", length=0)
    finished = sum(r["status"] != "not_run" for r in rows)
    figure.suptitle(f'96-point AC topology F1 — {"PARTIAL" if partial else "FINAL"} '
                   f'({finished}/{len(rows)} jobs)', fontsize=14)
    figure.colorbar(image, ax=axes, shrink=.8, label="Nontrivial-clade F1")
    figure.legend(handles=[Patch(facecolor="none", edgecolor="#ff8c00", linewidth=2,
                                 label="Incomplete run")],
                  loc="outside lower center", frameon=False)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/source_ac96_benchmark_20261018",
                        help="Existing benchmark run directory (read-only inputs)")
    parser.add_argument("--report-dir", type=Path,
                        help="Derived outputs destination; defaults to --output-dir")
    parser.add_argument("--dataset-dir", type=Path,
                        default=ROOT / "outputs/source_ac96_20261018",
                        help="Physical network files used only to evaluate rejected candidate appendix")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    report_dir = (args.report_dir or args.output_dir).resolve()
    protocol = read_json(output_dir / "protocol.json")
    if protocol is None:
        raise FileNotFoundError(output_dir / "protocol.json")
    rows, saved = collect(output_dir, protocol)
    by_method = [aggregate(rows, method=m) for m in protocol["methods"]]
    by_variant = [aggregate(rows, method=m, variant=v)
                  for v in protocol["variants"] for m in protocol["methods"]]
    blocks = frozen_blocks(rows, saved, protocol.get("active_atom_tolerance", 1e-8))
    incumbents = rejected_incumbents(rows, saved, args.dataset_dir.resolve(),
                                     protocol.get("active_atom_tolerance", 1e-8))
    partial = (any(r["status"] == "not_run" for r in rows)
               or not (output_dir / "execution_complete.json").exists())
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "aggregate_by_method.csv", by_method)
    write_csv(report_dir / "aggregate_by_variant_method.csv", by_variant)
    write_csv(report_dir / "frozen_blocks.csv", blocks, FROZEN_FIELDS)
    write_csv(report_dir / "uncertified_incumbents.csv", incumbents, INCUMBENT_FIELDS)
    figure_name = None if args.no_plot else "f1_heatmap.png"
    if figure_name:
        plot_f1(report_dir / figure_name, rows, protocol, partial)
    _write_text(report_dir / "report.md", make_report(
        rows, by_method, by_variant, blocks, incumbents, protocol, partial=partial, figure=figure_name,
    ))
    print(json.dumps({"report_dir": str(report_dir), "jobs_expected": len(rows),
                      "results_available": sum(r["status"] != "not_run" for r in rows),
                      "partial": partial, "frozen_blocks": len(blocks),
                      "rejected_feasible_incumbents": len(incumbents)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
