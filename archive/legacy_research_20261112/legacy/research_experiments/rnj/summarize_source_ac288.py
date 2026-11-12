"""Read saved sixteen-job AC288 evidence into a separate report directory.

Primary scores are copied from result.json. Accepted-path structural scores
are posthoc physical-truth evaluations only; neither they nor truth select a
model. No estimator, runtime package, power-flow routine or solver is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VARIANTS = ("no_source", "constant", "intermittent", "smooth")
METHODS = ("NJ", "RNJ", "WZZT", "RNJ+WZZT")
NAMES = {"no_source": "无源", "constant": "常值源", "intermittent": "间歇源", "smooth": "平滑时变源"}
ACTIVE_TOL = 1e-8


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if value is not None else None


def write_csv(path, rows, fields):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def declared_jobs(protocol):
    """Keep the full declared 4 x 4 denominator, regardless of file presence."""
    saved = {(j["variant"], j["method"]): j for j in protocol.get("jobs", [])}
    return [{**saved.get((v, m), {}), "case": "pengwah18", "variant": v, "method": m,
             "job_id": f"pengwah18__{v}__{m}"} for v in VARIANTS for m in METHODS]


SUMMARY_FIELDS = ["case", "variant", "method", "job_id", "status", "stop_reason", "result_available",
    "has_model", "checkpoint_has_model", "search_complete", "family_clade_f1", "active_clade_f1",
    "r_relative_error", "x_relative_error", "r_error_percent", "x_error_percent", "compute_seconds",
    "train_mae", "selected_path_index", "accepted_path_length", "selected_rnj_blocks", "partition_sizes",
    "partition_scope", "bound_expansions", "solver_time_limit_count", "uncertified_solve_count",
    "per_solve_seconds", "method_wall_seconds", "matrix_origin"]


def collect(directory, protocol):
    rows, evidence = [], {}
    for job in declared_jobs(protocol):
        folder = directory / "jobs" / job["job_id"]
        result = read_json(folder / "result.json")
        model = read_json(folder / "model.json") or {}
        screen = read_json(folder / "rnj_screen.json") or model.get("bootstrap")
        running = bool(model or (folder / "execution.json").exists())
        row = dict.fromkeys(SUMMARY_FIELDS)
        row.update({k: job[k] for k in ("case", "variant", "method", "job_id")})
        row.update(status="running" if running else "not_run", stop_reason="result_not_yet_available",
                   result_available=result is not None, has_model=False,
                   checkpoint_has_model=bool(model.get("has_model")), search_complete=False,
                   per_solve_seconds=job.get("per_solve_seconds"), method_wall_seconds=job.get("method_wall_seconds"))
        if result is not None:
            for key in ("status", "stop_reason", "has_model", "search_complete", "selected_path_index",
                        "bound_expansions", "solver_time_limit_count", "uncertified_solve_count"):
                row[key] = result.get(key, model.get(key, row[key]))
            row["compute_seconds"] = number(result.get("compute_seconds", result.get("elapsed_seconds")))
            metrics = result.get("metrics") or {}
            # An evaluator may not yet have written metrics. Missing stays NA.
            for key in ("family_clade_f1", "active_clade_f1", "r_relative_error", "x_relative_error"):
                row[key] = number(metrics.get(key))
            for key in ("r", "x"):
                value = row[f"{key}_relative_error"]
                row[f"{key}_error_percent"] = 100 * value if value is not None else None
            if row["has_model"]:
                row.update(train_mae=number(model.get("train_mae", model.get("train_mae_recomputed"))),
                           accepted_path_length=len(model.get("path") or []), matrix_origin=model.get("matrix_origin"))
        if screen is not None:
            row["selected_rnj_blocks"] = len(screen.get("selected_support_labels", []))
        elif job["method"] != "RNJ+WZZT":
            row["selected_rnj_blocks"] = 0
        groups = model.get("terminal_groups")
        if groups is not None and model:
            row["partition_sizes"] = compact([len(g) for g in groups])
            row["partition_scope"] = "final" if result is not None else "checkpoint"
        rows.append(row)
        evidence[job["job_id"]] = {"directory": folder, "result": result, "model": model, "screen": screen}
    return rows, evidence


def aggregate(rows):
    result = []
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        counts = Counter(r["status"] for r in selected)
        row = {"method": method, "expected": 4, "results_available": sum(r["result_available"] for r in selected),
               "models_available": sum(bool(r["has_model"]) for r in selected),
               **{status: counts[status] for status in ("complete", "incomplete", "failed", "running", "not_run")}}
        for metric in ("family_clade_f1", "active_clade_f1", "r_error_percent", "x_error_percent", "compute_seconds"):
            values = [r[metric] for r in selected if r[metric] is not None]
            row[f"{metric}_n"] = len(values)
            row[f"mean_{metric}"] = sum(values)/len(values) if values else None
        exact = sum(r["family_clade_f1"] is not None and math.isclose(r["family_clade_f1"], 1., abs_tol=1e-12, rel_tol=0.)
                    for r in selected)
        row.update(family_exact=exact, family_exact_available_denominator=row["family_clade_f1_n"],
                   family_exact_expected_denominator=4)
        result.append(row)
    return result


PATH_FIELDS = ["job_id", "variant", "method", "job_status", "result_available", "path_point_sha256",
    "first_saved_in", "bound_expansions", "path_index", "iteration", "support_count", "train_mae",
    "accepted_gain", "family_clade_f1", "active_clade_f1", "family_clade_count", "active_clade_count",
    "family_clades", "active_clades", "retained_in_current_path", "selected_final_model", "selected_checkpoint",
    "solver_status", "solver_objective", "solver_dual_bound", "solver_mip_gap", "evaluation_error"]


def point_hash(point):
    return hashlib.sha256(json.dumps(point, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def path_scores(point, labels, truth):
    from research_experiments.rnj.audit_source_ac96_benchmark import clade_metrics
    supports = point.get("support_labels")
    if supports is None:
        supports = [[labels[i] for i in s] for s in point["support_indices"]]
    r, x = point["r_values"], point["x_values"]
    if not (len(supports) == len(r) == len(x)):
        raise ValueError("saved path atom/weight lengths differ")
    family = [s for s in supports if 1 < len(s) < len(labels)]
    active = [s for s, a, b in zip(supports, r, x, strict=True)
              if 1 < len(s) < len(labels) and max(a, b) > ACTIVE_TOL]
    return {"family_clades": compact(family), "active_clades": compact(active),
            "family_clade_count": len(family), "active_clade_count": len(active),
            "family_clade_f1": clade_metrics(family, truth["clades"], labels)["clade_f1"],
            "active_clade_f1": clade_metrics(active, truth["clades"], labels)["clade_f1"]}


def accepted_paths(rows, evidence, dataset):
    """Include accepted points surviving in any saved progress snapshot.

    Bound restarts can replace the active path; those earlier accepted points
    stay in this appendix. attempted_extensions and raw incumbents are excluded.
    """
    from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
    records, cache = [], {}
    for row in rows:
        if "WZZT" not in row["method"]:
            continue
        saved = evidence[row["job_id"]]
        folder, model = saved["directory"], saved["model"]
        current_path = model.get("path") or []
        current = {point_hash(p) for p in current_path}
        selected = model.get("selected_path_index")
        selected_hash = point_hash(current_path[selected]) if isinstance(selected, int) and 0 <= selected < len(current_path) else None
        snapshots = [(str(p.relative_to(folder)), read_json(p)) for p in sorted((folder / "progress").glob("*.json"))]
        snapshots.append(("model.json", model))
        seen = set()
        for filename, snapshot in snapshots:
            for index, point in enumerate(snapshot.get("path") or []):
                digest = point_hash(point)
                phase = snapshot.get("bound_expansions")
                # Same accepted point repeated in growing path snapshots is one
                # row; identical refits in distinct bound restarts remain distinct.
                key = (phase, digest)
                if key in seen:
                    continue
                seen.add(key)
                record = dict.fromkeys(PATH_FIELDS)
                diagnostic = point.get("solver") or point.get("diagnostics") or {}
                record.update(job_id=row["job_id"], variant=row["variant"], method=row["method"],
                    job_status=row["status"], result_available=row["result_available"], path_point_sha256=digest,
                    first_saved_in=filename, bound_expansions=phase, path_index=index,
                    iteration=point.get("iteration"), support_count=len(point.get("support_indices") or []),
                    train_mae=number(point.get("train_mae")), accepted_gain=number(point.get("accepted_gain")),
                    retained_in_current_path=digest in current and phase == model.get("bound_expansions"),
                    selected_final_model=bool(row["result_available"] and row["has_model"] and digest == selected_hash and phase == model.get("bound_expansions")),
                    selected_checkpoint=bool(not row["result_available"] and digest == selected_hash and phase == model.get("bound_expansions")),
                    solver_status=diagnostic.get("status"), solver_objective=number(diagnostic.get("objective")),
                    solver_dual_bound=number(diagnostic.get("dual_bound")), solver_mip_gap=number(diagnostic.get("mip_gap")))
                try:
                    labels = tuple(snapshot.get("terminal_labels") or model.get("terminal_labels") or [])
                    if not labels:
                        raise ValueError("saved terminal labels unavailable")
                    if labels not in cache:
                        cache[labels] = physical_truth(dataset / "pengwah18/network", labels)
                    record.update(path_scores(point, labels, cache[labels]))
                except (KeyError, TypeError, ValueError, IndexError, OSError) as exc:
                    record["evaluation_error"] = f"{type(exc).__name__}: {exc}"
                records.append(record)
    return records


BOOTSTRAP_FIELDS = ["job_id", "variant", "job_status", "terminal_labels", "size", "full_coarse_peer",
    "peer_frequency", "peer_frequency_denominator", "subclade_frequency", "subclade_frequency_denominator",
    "full_fine_subclade_veto", "source_adjusted_identifiable", "eligible_saved", "selected_saved",
    "confidence_threshold", "maximum_subclade_frequency", "minimum_block_size", "rejection_reasons",
    "selected_support_indices", "valid_bootstrap_replicates", "unidentifiable_bootstrap_replicates"]


def bootstrap_candidates(rows, evidence):
    records = []
    for row in rows:
        screen = evidence[row["job_id"]]["screen"]
        if screen is None:
            continue
        diagnostics, params = screen.get("diagnostics") or {}, screen.get("parameters") or {}
        peers = {tuple(sorted(c)) for c in diagnostics.get("full_peers", [])}
        eligible = {tuple(sorted(c)) for c in diagnostics.get("eligible", [])}
        veto = {tuple(sorted(c)) for c in diagnostics.get("full_fine_subclades", [])}
        selected = {tuple(sorted(c)): i for i, c in enumerate(screen.get("selected_support_labels", []))}
        confidence = {tuple(sorted(r["clade"])): number(r.get("frequency")) for r in screen.get("confidence_records", [])}
        sub = {tuple(sorted(r["clade"])): number(r.get("frequency")) for r in diagnostics.get("subclade_frequency_records", [])}
        for clade in sorted(set(confidence) | peers | set(selected)):
            full_peer = clade in peers if "full_peers" in diagnostics else None
            fine_veto = clade in veto if "full_fine_subclades" in diagnostics else None
            is_eligible = clade in eligible if "eligible" in diagnostics else None
            minimum = diagnostics.get("minimum_block_size")
            threshold, maximum_sub = params.get("confidence_threshold"), params.get("maximum_subclade_frequency")
            freq, subfreq = confidence.get(clade), sub.get(clade)
            reasons = []
            if full_peer is False:
                reasons.append("not_full_data_complete_terminal_peer_group")
            if minimum is not None and len(clade) < minimum:
                reasons.append("below_minimum_block_size")
            if freq is not None and threshold is not None and freq < threshold:
                reasons.append("peer_frequency_below_threshold")
            if subfreq is not None and maximum_sub is not None and subfreq > maximum_sub:
                reasons.append("bootstrap_internal_subclade_veto")
            if fine_veto:
                reasons.append("full_data_fine_tolerance_subclade_veto")
            if diagnostics.get("source_adjusted_identifiable") is False:
                reasons.append("source_adjusted_design_unidentifiable")
            if diagnostics.get("valid_bootstrap_replicates") == 0:
                reasons.append("no_identifiable_bootstrap_replicates")
            if is_eligible and clade not in selected:
                reasons.append("eligible_not_selected_after_disjoint_maximum_count_rule")
            indices = screen.get("selected_support_indices", [])
            position = selected.get(clade)
            records.append(dict(zip(BOOTSTRAP_FIELDS, [row["job_id"], row["variant"], row["status"], compact(clade),
                len(clade), full_peer, freq, params.get("bootstrap_replicates"), subfreq,
                diagnostics.get("valid_bootstrap_replicates"), fine_veto, diagnostics.get("source_adjusted_identifiable"),
                is_eligible, clade in selected, threshold, maximum_sub, minimum, compact(reasons),
                compact(indices[position]) if position is not None and position < len(indices) else None,
                diagnostics.get("valid_bootstrap_replicates"), diagnostics.get("unidentifiable_bootstrap_replicates")])))
    return records


HASH_FIELDS = ["job_id", "variant", "method", "relative_input_file", "recorded_sha256", "actual_sha256",
               "file_present", "sha256_matches", "read_error"]


def raw_input_hashes(directory, rows, evidence):
    records = []
    for row in rows:
        entries = (read_json(evidence[row["job_id"]]["directory"] / "inputs_hash.json") or {}).get("file_sha256", {})
        root = (directory / "inputs" / row["variant"]).resolve()
        for relative, expected in sorted(entries.items()):
            path = (root / relative).resolve()
            record = dict.fromkeys(HASH_FIELDS)
            record.update(job_id=row["job_id"], variant=row["variant"], method=row["method"],
                          relative_input_file=relative, recorded_sha256=expected, file_present=False, sha256_matches=False)
            try:
                if not path.is_relative_to(root):
                    raise ValueError("recorded input path escapes the copied variant directory")
                record["file_present"] = path.is_file()
                record["actual_sha256"] = sha256(path) if path.is_file() else None
                record["sha256_matches"] = record["actual_sha256"] == expected
            except (OSError, ValueError) as exc:
                record["read_error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
    return records


def format_value(value, digits=4):
    return "NA" if value is None else f"{value:.{digits}f}"


def render_readme(rows, aggregates, paths, candidates, hashes, directory, protocol):
    available = sum(r["result_available"] for r in rows)
    finalized = available == 16 and (directory / "execution_complete.json").exists()
    lines = ["# Pengwah18：AC288 四工况与四方法", "",
        f"{'最终结果' if finalized else '临时快照'}：已返回 {available}/16 项；模型可用 {sum(bool(r['has_model']) for r in rows)}/16。",
        "每工况使用场景 0、1、2 各 96 点，共 288 点全部训练；无验证集、无独立测试集。",
        "有源工况的 P/Q 时序精确已知，固定共享物理位置对算法未知。",
        f"读取目录：`{directory.resolve()}`。WZZT 单次求解限时 6000 秒，方法总预算 30000 秒（包括混合方法筛选）。", "",
        "| 工况 | 方法 | family F1 | active F1 | R误差% | X误差% | 秒 | 状态 | stop_reason | 选中RNJ块 | 分区大小 |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---:|---|"]
    for row in rows:
        cells = [NAMES[row["variant"]], row["method"], format_value(row["family_clade_f1"]),
            format_value(row["active_clade_f1"]), format_value(row["r_error_percent"], 2), format_value(row["x_error_percent"], 2),
            format_value(row["compute_seconds"], 2), row["status"], row["stop_reason"],
            str(row["selected_rnj_blocks"]) if row["selected_rnj_blocks"] is not None else "NA", row["partition_sizes"] or "NA"]
        lines.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
    lines.extend(["", "| 方法 | 计划分母 | 已返回 | 模型可用 | complete | incomplete | failed | running/not_run | F1有值分母 | family精确恢复/有值 | family精确恢复/计划 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for a in aggregates:
        lines.append(f"| {a['method']} | 4 | {a['results_available']} | {a['models_available']} | {a['complete']} | {a['incomplete']} | {a['failed']} | {a['running']}/{a['not_run']} | {a['family_clade_f1_n']} | {a['family_exact']}/{a['family_exact_available_denominator']} | {a['family_exact']}/4 |")
    lines.extend(["", "表注：", "",
        "1. 主表逐项复制已保存 result.json 的评分。超时、失败和未运行保留在 16 项分母；缺失记 NA。方法均值包含有模型的 incomplete 回退，逐指标另列有效分母。运行中 checkpoint 不作为最终成绩。",
        "2. family F1 对所有非平凡 selected supports 评分；active F1 只计 max(r,x)>1e-8 的非平凡 supports。NJ/RNJ 两口径均取树 clades。R/X 相对误差为物理真值矩阵上的 Frobenius 范数比，百分数为其 100 倍。",
        "3. NJ/RNJ 共用前置零截距 ordered L2 回归的 R/X 矩阵，树恢复使用不同算法；两者 R/X 误差相同符合方法定义。有源时 FWL 消去自由源响应。WZZT 使用 L1 损失及物理 Source 路径约束；两类源响应的可行域不同。",
        "4. 无验证时 WZZT 取最后一个已认证、已接受的路径点，未用 1SE 或事后真值挑选。逐子问题求解证书限定于当前有界域，整体搜索仍为贪心。路径中的训练 MAE 下降或 F1 变化不称为拓扑收敛或泛化证据。",
        "5. RNJ 末端组要求完整直接兄弟叶组、至少 3 个叶、bootstrap peer 频率>=0.75、内部子簇频率<=0.1、全样本细容差无子簇；最多取两个互斥组。频率是算法稳定度；所选组约束后续搜索，错误组会排除真拓扑。零组时混合方法没有增加末端分区信息。",
        "6. peer 频率分母为配置的全部 bootstrap 次数；内部子簇频率分母为可识别重复次数。诊断的源秩、条件数和拟合成功状态均不作为拓扑正确性证书。",
        "7. 三个既定背景场景组成 288 点，各场景的噪声实现固定并跨工况配对；没有独立重复或独立测试。场景 0 复用原 AC96 数据。各工况共享背景与测量噪声，三种有源方案共享接入位置，源曲线在三个场景中相同。", "",
        f"[summary.csv](summary.csv) 保留 16 行；[aggregate_by_method.csv](aggregate_by_method.csv) 给出分母与均值。",
        f"[accepted_paths.csv](accepted_paths.csv) 保存 {len(paths)} 个已接受点及事后 physical family/active F1；包含 bound 重启前的历史点，排除未接受的 incumbent。",
        "路径表 selected_final_model 标记最终保存点；selected_checkpoint 标记运行中保存点。缺少物理网络或点格式不完整时，evaluation_error 保留原因而 F1 留空。",
        f"[bootstrap_candidates.csv](bootstrap_candidates.csv) 保存 {len(candidates)} 个候选及子簇否决条件。分区大小含未被选中组覆盖的 singleton；NJ/RNJ 没有 WZZT 分区。",
        f"[raw_input_sha256.csv](raw_input_sha256.csv) 保存 {len(hashes)} 条输入哈希记录，匹配 {sum(bool(r['sha256_matches']) for r in hashes)}/{len(hashes)}。没有哈希记录的 job 在 report_metadata.json 另列；未记录不计为通过。", ""])
    return "\n".join(lines)


def summarize(directory, output_dir, *, dataset_dir=ROOT / "outputs/source_ac288_20261018"):
    directory, output_dir, dataset = Path(directory).resolve(), Path(output_dir).resolve(), Path(dataset_dir).resolve()
    if output_dir == directory or output_dir == dataset or output_dir.is_relative_to(dataset):
        raise ValueError("report directory must be separate from raw run/data files")
    if any(output_dir.is_relative_to(directory / name) for name in ("jobs", "inputs", "source_snapshot")):
        raise ValueError("report directory cannot contain estimator evidence")
    protocol = read_json(directory / "protocol.json")
    if protocol is None:
        raise FileNotFoundError("protocol.json is required")
    if (protocol.get("training_samples") != 288 or protocol.get("validation_samples") != 0
            or protocol.get("validation_supplied", False) or protocol.get("case", "pengwah18") != "pengwah18"):
        raise ValueError("expected the pengwah18 288-training-row, zero-validation protocol")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("choose a fresh report directory; saved files are never overwritten")
    rows, evidence = collect(directory, protocol)
    aggregates = aggregate(rows)
    paths = accepted_paths(rows, evidence, dataset)
    candidates = bootstrap_candidates(rows, evidence)
    hashes = raw_input_hashes(directory, rows, evidence)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", rows, SUMMARY_FIELDS)
    write_csv(output_dir / "aggregate_by_method.csv", aggregates, list(aggregates[0]))
    write_csv(output_dir / "accepted_paths.csv", paths, PATH_FIELDS)
    write_csv(output_dir / "bootstrap_candidates.csv", candidates, BOOTSTRAP_FIELDS)
    write_csv(output_dir / "raw_input_sha256.csv", hashes, HASH_FIELDS)
    (output_dir / "README.md").write_text(render_readme(rows, aggregates, paths, candidates, hashes, directory, protocol), encoding="utf-8")
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "run_directory": str(directory),
        "dataset_directory": str(dataset), "protocol_sha256": sha256(directory / "protocol.json"),
        "postprocessor_sha256": sha256(__file__),
        "expected_jobs": 16, "results_available": sum(r["result_available"] for r in rows),
        "status_counts": dict(Counter(r["status"] for r in rows)), "accepted_path_rows": len(paths),
        "truth_use": "posthoc accepted-path scoring only; primary metrics copied; no model selection",
        "raw_input_hash_records": len(hashes), "raw_input_hash_mismatches": sum(not r["sha256_matches"] for r in hashes),
        "jobs_without_input_hash_records": [r["job_id"] for r in rows if not any(h["job_id"] == r["job_id"] for h in hashes)],
        "estimation_or_simulation_performed": False,
        "source_evidence_sha256": {p.relative_to(directory).as_posix(): sha256(p)
            for saved in evidence.values() for p in saved["directory"].glob("*.json") if p.is_file()}}
    (output_dir / "report_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "outputs/source_ac288_20261018")
    args = parser.parse_args()
    result = summarize(args.directory, args.output_dir, dataset_dir=args.dataset_dir)
    print(json.dumps({k: result[k] for k in ("expected_jobs", "results_available", "status_counts", "accepted_path_rows", "raw_input_hash_mismatches")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
