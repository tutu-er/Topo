"""Snapshot ended AC96 jobs without opening any active model or solver file."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from research_experiments.rnj import summarize_rnj_blocks96 as saved
from research_experiments.rnj.audit_source_ac96_benchmark import Audit, clade_metrics, physical_truth, relative_error
from research_experiments.rnj.diagnose_rnj_initialization96 import verified_network


ROOT = Path(__file__).resolve().parents[2]
MAIN_METHODS = ("NJ", "RNJ", "WZZT", "RNJ+WZZT")
LABELS = {"NJ": "NJ", "RNJ": "RNJ", "WZZT": "WZZT", "RNJ+WZZT": "稳定块+WZZT",
          "RNJ+WZZT_ungrouped": "同支持不分组", "RNJ+WZZT_tree_init": "完整RNJ族+WZZT"}
VARIANT_LABELS = dict(zip(saved.VARIANTS, ("无源", "常值源", "间歇源", "平滑源")))


def blank_attempt(root, job, status):
    """Never delegate pending rows to _attempt: it hashes active model files."""
    directory = root/"jobs"/job["job_id"]
    row = {**{key: job.get(key) for key in ("job_id", "case", "variant", "method", "role")},
           "status": status, "has_model": False, "complete_model": False, "retained_fallback": False,
           "error_type": None, "stop_reason": None, "reused_baseline": False,
           "result_directory": str(directory), "attempt_root": str(root),
           "elapsed_seconds": None, "compute_seconds": None, "postprocess_seconds": None,
           **dict.fromkeys(saved.METRICS)}
    screen = directory/"rnj_screen.json"  # Written once, before fitting starts.
    evidence = {"rnj_screen.json": saved._sha(screen)} if screen.is_file() else {}
    return row, {}, directory, evidence


def completed_census(root, resume_plan, log_path):
    protocol = saved._read(root/"protocol.json")
    plan = saved._read(resume_plan)
    if saved._sha(root/"protocol.json") != plan["protocol_sha256"]:
        raise ValueError("resume plan/protocol binding changed")
    raw = log_path.read_bytes()
    # A line being appended is excluded until its newline is published.
    raw = raw[:raw.rfind(b"\n")+1]
    log = raw.decode("utf-8-sig")
    ended = set(re.findall(r"^END ([^ ]+) (?:complete|incomplete|failed) F1=[^ ]+ seconds=[\d.]+\r?$", log, re.M))
    starts = {key: int(pid) for key, pid in re.findall(r"^START ([^ ]+) pid=(\d+)\r?$", log, re.M)}
    done = set(plan["existing_final_jobs"]) | ended
    jobs = {job["job_id"]: job for job in protocol["jobs"]}
    if len(jobs) != 64 or not done <= set(jobs) or not set(starts) <= set(jobs):
        raise ValueError("invalid planned/completed job census")
    return protocol, plan, done, set(starts)-done, raw


def check_metrics(row, directory, network, audit):
    if not row["has_model"]:
        return
    model = saved._read(directory/"model.json")
    labels = model["terminal_labels"]
    truth = physical_truth(network, labels)
    key = row["job_id"]
    audit.require(key+": all96 without validation", model.get("training_rows") == 96 and model.get("validation_supplied") is False)
    for kind in ("family", "active", "resolution"):
        metrics = clade_metrics(model[kind+"_clades"], truth["clades"], labels)
        for name in ("clade_f1", "precision", "recall"):
            audit.close(key+": physical "+kind+"_"+name, row[kind+"_"+name], metrics[name], atol=1e-12, rtol=1e-10)
    with np.load(directory/"model.npz", allow_pickle=False) as arrays:
        for channel in ("R", "X"):
            audit.close(key+": physical "+channel+" relative error", row[channel.lower()+"_relative_error"], relative_error(arrays[channel], truth[channel]), atol=1e-12, rtol=1e-10)


def _metric_cell(row, metric="resolution_clade_f1", percent=False):
    value = row.get(metric)
    if value is None:
        return "运行中" if row["status"] == "running" else "—"
    number = f"{value*100:.2f}%" if percent else f"{value:.3f}"
    return number+("†" if row["status"] != "complete" else "")


def report(rows, audit, screens, created, output, recovery_rows):
    by_id = {row["job_id"]: row for row in rows}
    lines = ["# AC96阶段统计（非最终审计）", "", f"快照：{created}。保留64个新计算项和32项复用NJ/RNJ，固定96行；尚未结束的行保留空指标。",
             "", f"有效状态：{dict(Counter(row['status'] for row in rows))}；I/O独立补跑{len(recovery_rows)}项。",
             "物理指标由已结束模型和复制物理网络独立重算；全批原始求解约束审计尚未完成。本工具不拟合、不调用求解器、不读取活跃模型或solver记录。",
             "", "主F1采用统一RX75分辨率，对完整物理真簇评分。R/X误差为响应矩阵相对Frobenius误差。†表示未完成求解的保留模型，仅作fallback展示。",
             "原NJ/RNJ拟合、原指标及原时间保留，仅新增统一分辨率后处理。全部原96点拟合；电压噪声0.02%；Source P/Q已知、位置未知。", "",
             "## 各网络与源条件F1", ""]
    for case in saved.CASES:
        lines += ["### "+case, "", "| 条件 | "+" | ".join(LABELS[m] for m in saved.METHODS)+" |",
                  "|---|"+"---:|"*len(saved.METHODS)]
        for variant in saved.VARIANTS:
            selected = [by_id[f"{case}__{variant}__{method}"] for method in saved.METHODS]
            lines.append("| "+VARIANT_LABELS[variant]+" | "+" | ".join(_metric_cell(row) for row in selected)+" |")
        lines.append("")
    aggregates = saved._aggregate(rows, ("method",))
    lines += ["## 描述性汇总", "", "各方法完整结果均值的条件集合可能不同；下表同时保留完整数和未完成数。耗时均值覆盖已结束的有效尝试，包含未证明最优的运行。",
              "", "| 方法 | 完整/16 | 不完整 | 运行中 | 完整项均F1 | 完整项均R误差 | 完整项均X误差 | 已结束平均秒数 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for record in aggregates:
        value = lambda name, percent=False: "—" if record[name] is None else f"{record[name]*(100 if percent else 1):.3f}"+("%" if percent else "")
        lines.append(f"| {LABELS[record['method']]} | {record['complete']}/16 | {record['incomplete']} | {record['running']} | {value('resolution_clade_f1_complete_mean')} | {value('r_relative_error_complete_mean', True)} | {value('x_relative_error_complete_mean', True)} | {value('compute_seconds_available_mean')} |")
    shared = [(case, variant) for case in saved.CASES for variant in saved.VARIANTS
              if all(by_id[f"{case}__{variant}__{method}"]["complete_model"] for method in saved.METHODS)]
    lines += ["", f"六种配置共同完成的条件为{len(shared)}/16；同条件均值见paired_complete.csv。未完成和失败没有从96行明细删除。",
              "", "## 非完整记录与恢复", "", "| 网络/条件 | 方法 | 原始状态 | 终止原因 | 秒数 |", "|---|---|---|---|---:|"]
    for row in rows:
        if row["original_status"] not in saved.FINAL or row["original_status"] == "complete":
            continue
        lines.append(f"| {row['case']}/{VARIANT_LABELS[row['variant']]} | {LABELS[row['method']]} | {row['original_status']} | {row.get('original_stop_reason')} | {row['original_elapsed_seconds']:.3f} |")
    for recovery in recovery_rows:
        lines += ["", f"{recovery['job_id']}：原I/O失败后单独补跑，恢复资格={recovery['recovery_eligible']}；有效F1={recovery['resolution_clade_f1']:.3f}；原耗时+补跑={recovery['combined_spent_seconds']:.3f}秒。恢复选择只依据失败类型和配置/数据一致性，不依据F1。"]
    nonempty = sum(bool(row["selected_count"]) for row in screens)
    lines += ["", "## 筛选与时间口径", "", f"16条件中，{nonempty}个条件选出非空稳定末端块；详见screen_conditions.csv及screen_candidates.csv。普通clade稳定频率与完整末端同胞稳定频率按原规则分别保留。",
              "四项外部中断原始尝试已归档；其实际花费未知，明细给出checkpoint耗时下界。compute_seconds只含本次保存的有效尝试；已结束均值不包括运行中未知花费。",
              "运行非独占，旧基线未重新计时；时间用于描述本次实测消耗。完整RNJ支持固定层状兼容关系，遇到错误初始结构可能限制后续纠错。",
              "", f"物理指标核对：{audit['check_count']-audit['failed_count']}/{audit['check_count']}通过。原始/有效96行、补跑、分组及逐条哈希见CSV和snapshot_manifest.json。", ""]
    (output/"report.md").write_text("\n".join(lines), encoding="utf-8")
    return shared


def snapshot(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or output == root or output.is_relative_to(root):
        raise ValueError("use a fresh output directory outside the active run")
    log_path = ROOT/"tmp/resume_rnj_blocks96_state_20261019/native_queue.stdout.log"
    plan_path = ROOT/"outputs/rnj_blocks96_interruption_archive_20261019/resume_plan.json"
    protocol, plan, done, active, log_bytes = completed_census(root, plan_path, log_path)
    networks = {case: Path(verified_network(root, protocol, case, "evaluation_network/"+case)["directory"]) for case in saved.CASES}
    attempts, bindings = {}, {}
    for baseline, jobs in ((False, protocol["jobs"]), (True, protocol["baseline_jobs"])):
        for job in jobs:
            key = job["job_id"]
            if baseline or key in done:
                attempt = saved._attempt(root, protocol, job, baseline=baseline)
                if attempt[0]["status"] not in saved.FINAL:
                    raise ValueError("published END does not bind a final result: "+key)
            else:
                attempt = blank_attempt(root, job, "running" if key in active else "not_run")
            attempts[key] = attempt
            bindings[key] = {"directory": str(attempt[2]), "artifact_sha256": attempt[3], "model_opened": baseline or key in done}
    first = [dict(attempt[0]) for attempt in attempts.values()]
    effective = {row["job_id"]: {**row, "original_status": row["status"], "original_stop_reason": row.get("stop_reason"),
                 "original_elapsed_seconds": row["elapsed_seconds"], "original_resolution_clade_f1": row["resolution_clade_f1"],
                 "recovery_applied": False, "retry_elapsed_seconds": None, "combined_spent_seconds": row["elapsed_seconds"]} for row in first}
    retry_root = ROOT/"outputs/rnj_blocks96_io_retry_20261019"
    if (retry_root/"controller.lock").exists():
        raise ValueError("I/O recovery is active")
    retry_protocol, recoveries, recovery_bindings = saved._read(retry_root/"protocol.json"), [], []
    for job in retry_protocol["jobs"]:
        directory = retry_root/"jobs"/job["job_id"]
        if not (directory/"result.json").is_file():
            continue
        key = job["job_id"]
        if key not in done:
            raise ValueError("recovery for a nonfinal original cannot be summarized")
        retry = saved._attempt(retry_root, retry_protocol, job)
        original_job = next(j for j in protocol["jobs"] if j["job_id"] == key)
        eligible, reasons, details = saved._recovery_check(root, retry_root, protocol, retry_protocol, original_job, attempts[key], retry)
        if original_job != job:
            raise ValueError("recovery job differs")
        previous = effective[key]
        spent = previous["elapsed_seconds"]+retry[0]["elapsed_seconds"]
        record = {**retry[0], "recovery_eligible": eligible, "rejection_reasons": reasons, "combined_spent_seconds": spent}
        if eligible:
            effective[key] = {**previous, **retry[0], "recovery_applied": True,
                              "retry_elapsed_seconds": retry[0]["elapsed_seconds"], "combined_spent_seconds": spent}
        recoveries.append(record)
        recovery_bindings.append({"job_id": key, "directory": str(directory), "eligible": eligible, "reasons": reasons,
                                  "artifact_sha256": retry[3], "eligibility_evidence": details})
    interrupted = {row["job_id"]: row for row in plan["interrupted_attempts"]}
    audit = Audit()
    for row in effective.values():
        old = interrupted.get(row["job_id"])
        row.update(prior_external_interruption=bool(old), interrupted_elapsed_lower_bound_seconds=old["checkpoint_elapsed_seconds_lower_bound"] if old else None,
                   exact_total_spent_seconds=None if old else row["combined_spent_seconds"])
        if row["status"] in saved.FINAL:
            check_metrics(row, Path(row["result_directory"]), networks[row["case"]], audit)
    screens, candidates = saved._screens(attempts)
    for binding in list(bindings.values())+recovery_bindings:
        for name, checksum in binding["artifact_sha256"].items():
            if saved._sha(Path(binding["directory"])/name) != checksum:
                raise ValueError("snapshot source artifact changed")
    if not audit.result()["passed"]:
        raise ValueError("independent saved physical metric checks failed")
    output.mkdir(parents=True)
    (output/"census.stdout.snapshot.log").write_bytes(log_bytes)
    rows = list(effective.values())
    saved._write_csv(output/"first_attempt.csv", first)
    saved._write_csv(output/"effective_results.csv", rows)
    saved._write_csv(output/"recovery_attempts.csv", recoveries)
    for dimensions in (("method",), ("case", "method"), ("case", "variant", "method")):
        saved._write_csv(output/("aggregate_by_"+"_".join(dimensions)+".csv"), saved._aggregate(rows, dimensions))
    saved._write_csv(output/"screen_conditions.csv", screens)
    saved._write_csv(output/"screen_candidates.csv", candidates)
    created = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    shared = report(rows, audit.result(), screens, created, output, recoveries)
    paired = [row for row in rows if (row["case"], row["variant"]) in shared]
    saved._write_csv(output/"paired_complete.csv", saved._aggregate(paired, ("method",)))
    manifest = {"created_at_utc_plus_8": created, "scope": "partial ended-job snapshot; not final solver audit", "denominator": 96,
                "new_ended": len(done), "new_planned": 64, "reused_baselines": 32, "active_job_ids": sorted(active),
                "first_status_counts": dict(Counter(row["status"] for row in first)), "effective_status_counts": dict(Counter(row["status"] for row in rows)),
                "common_complete_conditions": shared, "original_root": str(root), "protocol_sha256": saved._sha(root/"protocol.json"),
                "census_sha256": hashlib.sha256(log_bytes).hexdigest(), "resume_plan_sha256": saved._sha(plan_path),
                "original_bindings": bindings, "recovery_bindings": recovery_bindings, "physical_metric_audit": audit.result(),
                "code_sha256": {str(Path(__file__)): saved._sha(__file__), str(Path(saved.__file__)): saved._sha(saved.__file__)},
                "full_solver_audit": False, "estimation_performed": False, "truth_used_for_selection": False}
    saved._write(output/"snapshot_manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("new_ended", "effective_status_counts", "common_complete_conditions")}, ensure_ascii=False))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"outputs/rnj_blocks96_20261019")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshot(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
