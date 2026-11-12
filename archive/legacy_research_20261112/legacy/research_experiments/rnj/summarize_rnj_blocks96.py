"""Summarize saved AC96 attempts, keeping recording recovery separate.

This module never imports an estimator. Production summaries require the
original execution-complete marker and all 64 final results. Partial mode is
reserved for explicitly marked, hand-written test fixtures.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CASES = ("paper15", "soumalas11", "flynn16", "pengwah18")
VARIANTS = ("no_source", "constant", "intermittent", "smooth")
NEW_METHODS = ("WZZT", "RNJ+WZZT", "RNJ+WZZT_ungrouped", "RNJ+WZZT_tree_init")
METHODS = ("NJ", "RNJ", *NEW_METHODS)
FINAL = {"complete", "incomplete", "failed"}
DRIVER = "research_experiments/rnj/run_rnj_blocks96.py"
APPROVED_RECORDING_COMMIT = "b5f93bd3583d2d83579b84e5b86137c26dc396fb"
# Canonical LF bytes of the reviewed parent and recorder-only fix. Actual byte
# hashes, including line endings, are independently retained in the manifest.
APPROVED_DRIVER_PAIR = (
    "bc8c0d2bc6756b2554712b01d0adc33ea887e7b2072e81115cd185eeb923692b",
    "1d16558f2efe6462f0d42f47b7d7c87b07fd01c49335e9d8b78e6b0966d308f0",
)
RECORDING_POLICY = {
    "permission_retry_limit": 20, "retry_delay_cap_seconds": .2,
    "solver_metadata": "annotate before first publication; no duplicate JSON replacement",
    "optimization_repeated_on_write_retry": False,
}
METRICS = tuple(f"{p}_{k}" for p in ("family", "active", "resolution")
                for k in ("clade_f1", "precision", "recall")) + (
                    "clade_f1", "r_relative_error", "x_relative_error")
BASELINE_EPHEMERAL = {"created_utc", "validation_seconds", "inventory", "baseline_provenance_sha256"}
INPUT_ARRAYS = ("P", "Q", "Y", "P_source", "Q_source")
INPUT_FILES = {f"observed/{name}.csv" for name in (
    "P_terminal", "Q_terminal", "V_terminal", "drop_target", "root_voltage", "P0_measured", "Q0_measured"
)} | {"observed/metadata.json", "source/amplitudes.csv", "source/metadata.json"}


class SummaryValidationError(ValueError):
    """Saved evidence does not satisfy the declared summary policy."""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _inside(root, relative):
    target = (root/relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise SummaryValidationError("artifact path leaves run directory: "+str(relative))
    return target


def _fixture(root):
    marker = root/"summary_fixture.json"
    return marker.exists() and _read(marker) == {"hand_written_fixture": True}


def _gate(root, allow_partial):
    if (root/"controller.lock").exists():
        raise SummaryValidationError("controller is active; no saved-model summary is permitted")
    if allow_partial and not _fixture(root):
        raise SummaryValidationError("--allow-partial requires an explicit hand-written fixture marker")
    if not allow_partial and not (root/"execution_complete.json").is_file():
        raise SummaryValidationError("original execution_complete.json is required before reading run products")
    protocol = _read(root/"protocol.json")
    jobs, baselines = protocol.get("jobs", []), protocol.get("baseline_jobs", [])
    expected_new = {(c, v, m) for c in CASES for v in VARIANTS for m in NEW_METHODS}
    expected_old = {(c, v, m) for c in CASES for v in VARIANTS for m in ("NJ", "RNJ")}
    def keys(items):
        return {(j["case"], j["variant"], j["method"]) for j in items}
    if len(jobs) != 64 or len(baselines) != 32 or keys(jobs) != expected_new or keys(baselines) != expected_old:
        raise SummaryValidationError("protocol must retain exactly 64 new and 32 reused baseline jobs")
    ids = [j["job_id"] for j in jobs+baselines]
    if len(set(ids)) != 96 or any(j["job_id"] != "__".join((j["case"], j["variant"], j["method"])) for j in jobs+baselines):
        raise SummaryValidationError("duplicate or inconsistent job identifiers")
    for job in jobs+baselines:
        directory = _inside(root, job.get("directory", "jobs/"+job["job_id"]))
        result_path = directory/"result.json"
        if not result_path.exists():
            if allow_partial:
                continue
            raise SummaryValidationError("missing final result: "+job["job_id"])
        if _read(result_path).get("status") not in FINAL and not allow_partial:
            raise SummaryValidationError("nonfinal result: "+job["job_id"])
    if not allow_partial:
        completion = _read(root/"execution_complete.json").get("completion", {})
        if completion.get("expected") != 64 or completion.get("finished") != 64:
            raise SummaryValidationError("execution marker does not certify all 64 final records")
    return protocol


def _number(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SummaryValidationError("nonfinite or nonnumeric saved metric/time")
    return value


def _attempt(root, protocol, job, *, baseline=False):
    directory = _inside(root, job.get("directory", "jobs/"+job["job_id"]))
    result = _read(directory/"result.json") if (directory/"result.json").exists() else {}
    for key in ("job_id", "case", "variant", "method"):
        if key in result and result[key] != job[key]:
            raise SummaryValidationError("result identity mismatch: "+job["job_id"])
    if result and not baseline and result.get("protocol_sha256") != _sha(root/"protocol.json"):
        raise SummaryValidationError("result protocol hash mismatch: "+job["job_id"])
    metrics = result.get("metrics", {})
    status = result.get("status", "not_run")
    has_model = bool(result.get("has_model", False))
    row = {k: job.get(k) for k in ("job_id", "case", "variant", "method", "role")}
    row.update(role="baseline_reuse" if baseline else job.get("role", "main"),
        status=status, stop_reason=result.get("stop_reason"), has_model=has_model,
        complete_model=status == "complete" and has_model,
        retained_fallback=status != "complete" and has_model,
        error_type=(result.get("error") or {}).get("type"),
        elapsed_seconds=_number(result.get("elapsed_seconds", result.get("compute_seconds"))),
        compute_seconds=_number(result.get("compute_seconds", result.get("elapsed_seconds"))),
        postprocess_seconds=_number(result.get("postprocess_seconds")),
        reused_baseline=baseline, selected_path_index=result.get("selected_path_index"),
        path_length=result.get("path_length"), selected_rnj_blocks=result.get("selected_rnj_blocks"),
        solver_time_limit_count=result.get("solver_time_limit_count"),
        uncertified_solve_count=result.get("uncertified_solve_count"),
        original_metrics=result.get("original_metrics"),
        result_directory=str(directory), attempt_root=str(root),
        **{key: _number(metrics.get(key)) for key in METRICS})
    if row["clade_f1"] != row["resolution_clade_f1"]:
        raise SummaryValidationError("primary F1 differs from saved resolution F1: "+job["job_id"])
    evidence = {name: _sha(directory/name) for name in (
        "result.json", "model.json", "model.npz", "inputs_hash.json", "payload.json", "rnj_screen.json", "reuse_provenance.json"
    ) if (directory/name).is_file()}
    return row, result, directory, evidence


def _protocol_match(original_root, retry_root, original, retry):
    reasons, details = [], {"approved_recording_commit": APPROVED_RECORDING_COMMIT}
    left, right = dict(original), dict(retry)
    old_sources, new_sources = left.pop("source_sha256", {}), right.pop("source_sha256", {})
    changed = sorted(k for k in set(old_sources)|set(new_sources) if old_sources.get(k) != new_sources.get(k))
    details["changed_source_files"] = changed
    for root, sources in ((original_root, old_sources), (retry_root, new_sources)):
        for relative, checksum in sources.items():
            path = _inside(root/"source_snapshot", relative)
            if not path.is_file() or _sha(path) != checksum:
                reasons.append("frozen_source_hash_mismatch:"+str(root)+":"+relative)
    if changed:
        if changed != [DRIVER]:
            reasons.append("unapproved_source_changes")
        else:
            paths = [_inside(root/"source_snapshot", DRIVER) for root in (original_root, retry_root)]
            pair = tuple(hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                         for path in paths) if all(path.is_file() for path in paths) else None
            details["canonical_driver_sha256"] = list(pair) if pair else None
            if pair != APPROVED_DRIVER_PAIR:
                reasons.append("source_change_is_not_the_reviewed_recording_fix")
    old_policy, new_policy = left.pop("recording_policy", None), right.pop("recording_policy", None)
    details["recording_policy"] = {"original": old_policy, "retry": new_policy}
    if old_policy != new_policy and not (old_policy is None and new_policy == RECORDING_POLICY and changed == [DRIVER]):
        reasons.append("unapproved_recording_policy_change")
    details["baseline_evidence_ephemeral_fields"] = sorted(BASELINE_EPHEMERAL)
    for protocol in (left, right):
        if "baseline_evidence" in protocol:
            protocol["baseline_evidence"] = {k: v for k, v in protocol["baseline_evidence"].items() if k not in BASELINE_EPHEMERAL}
    if left != right:
        reasons.append("nonrecording_protocol_difference")
        details["different_top_level_keys"] = sorted(k for k in set(left)|set(right) if left.get(k) != right.get(k))
    return reasons, details


def _recovery_check(original_root, retry_root, original_protocol, retry_protocol, job, first, retry):
    row, first_result, first_dir, first_sha = first
    retry_row, retry_result, retry_dir, retry_sha = retry
    reasons, details = _protocol_match(original_root, retry_root, original_protocol, retry_protocol)
    if (first_result.get("error") or {}).get("type") != "PermissionError":
        reasons.append("original_error_is_not_PermissionError")
    if row["status"] != "failed":
        reasons.append("original_attempt_is_not_failed")
    if retry_row["status"] != "complete" or not retry_row["has_model"]:
        reasons.append("retry_is_not_a_complete_model")
    for name, folder in (("original", first_dir), ("retry", retry_dir)):
        missing = [p for p in ("model.json", "model.npz", "inputs_hash.json", "payload.json") if not (folder/p).is_file()]
        if missing:
            reasons.append(name+"_missing_evidence:"+",".join(missing))
    if any("missing_evidence:" in r for r in reasons):
        return False, reasons, details
    old_model, new_model = _read(first_dir/"model.json"), _read(retry_dir/"model.json")
    if new_model.get("status") != "complete" or not new_model.get("has_model"):
        reasons.append("retry_model_is_not_complete")
    for key in ("terminal_labels", "core_parameters", "budget", "rx75_calibration", "initial_supports",
                "requested_terminal_groups", "terminal_groups"):
        if key not in old_model or key not in new_model or old_model[key] != new_model[key]:
            reasons.append("model_configuration_difference:"+key)
    if old_model.get("validation_supplied") is not False or new_model.get("validation_supplied") is not False:
        reasons.append("validation_policy_not_explicitly_false")
    if old_model.get("training_rows") != 96 or new_model.get("training_rows") != 96:
        reasons.append("training_rows_are_not_96")
    inventories = []
    for name, root, folder, protocol in (("original", original_root, first_dir, original_protocol),
                                        ("retry", retry_root, retry_dir, retry_protocol)):
        inventory = _read(folder/"inputs_hash.json").get("file_sha256", {})
        payload = _read(folder/"payload.json")
        if set(inventory) != INPUT_FILES or payload.get("input_sha256") != inventory or payload.get("job") != job:
            reasons.append(name+"_input_or_job_binding_mismatch")
        if payload.get("protocol_sha256") != _sha(root/"protocol.json") or payload.get("config") != protocol:
            reasons.append(name+"_payload_protocol_mismatch")
        input_dir = _inside(root, job.get("input_relative_directory", f"inputs/{job['case']}/{job['variant']}"))
        for relative, checksum in inventory.items():
            path = _inside(input_dir, relative)
            if not path.is_file() or _sha(path) != checksum:
                reasons.append(name+"_input_file_hash_mismatch:"+relative)
        inventories.append(inventory)
    details["input_sha256"] = {"original": inventories[0], "retry": inventories[1]}
    if inventories[0] != inventories[1]:
        reasons.append("input_hash_inventory_difference")
    with np.load(first_dir/"model.npz", allow_pickle=False) as old, np.load(retry_dir/"model.npz", allow_pickle=False) as new:
        details["input_arrays_equal"] = {}
        for key in INPUT_ARRAYS:
            equal = key in old and key in new and np.array_equal(old[key], new[key])
            details["input_arrays_equal"][key] = equal
            if not equal:
                reasons.append("saved_input_array_difference:"+key)
            if key in old and (old[key].ndim not in (1, 2) or old[key].shape[0] != 96 or not np.isfinite(old[key]).all()):
                reasons.append("invalid_saved_input_array:"+key)
        if "Y" not in old or old["Y"].ndim != 2 or old["Y"].shape[0] != 96:
            reasons.append("saved_input_is_not_all_96_rows")
        for key in ("calibration_R", "calibration_X", "calibration_source_g"):
            if key in old or key in new:
                if key not in old or key not in new or not np.array_equal(old[key], new[key]):
                    reasons.append("saved_calibration_array_difference:"+key)
    details.update(original_artifact_sha256=first_sha, retry_artifact_sha256=retry_sha,
                   metric_values_used_for_eligibility=False)
    return not reasons, reasons, details


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["job_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _canonical(value) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def _aggregate(rows, dimensions):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in dimensions)].append(row)
    records = []
    for key, subset in sorted(groups.items()):
        record = dict(zip(dimensions, key))
        record.update(planned=len(subset), **dict.fromkeys(("complete", "incomplete", "failed", "running", "not_run"), 0))
        record.update(Counter(r["status"] for r in subset))
        record["retained_fallback_models"] = sum(r["retained_fallback"] for r in subset)
        record["recovered_jobs"] = sum(bool(r.get("recovery_applied")) for r in subset)
        for metric in ("resolution_clade_f1", "family_clade_f1", "active_clade_f1", "r_relative_error", "x_relative_error", "compute_seconds", "combined_spent_seconds"):
            for scope, eligible in (("available", subset), ("complete", [r for r in subset if r["status"] == "complete"])):
                values = [r[metric] for r in eligible if r.get(metric) is not None]
                record[f"{metric}_{scope}_count"] = len(values)
                record[f"{metric}_{scope}_mean"] = sum(values)/len(values) if values else None
        records.append(record)
    return records


def _screens(attempts):
    conditions, candidates = [], []
    for case in CASES:
        for variant in VARIANTS:
            saved, errors = [], {}
            for method in NEW_METHODS:
                if method == "WZZT":
                    continue
                key = "__".join((case, variant, method))
                row, result, directory, evidence = attempts[key]
                if (directory/"rnj_screen.json").exists():
                    screen = _read(directory/"rnj_screen.json")
                    compare = {k: v for k, v in screen.items() if k not in {"elapsed_seconds", "requested_terminal_groups"}}
                    saved.append((method, screen, _canonical(compare), evidence.get("rnj_screen.json")))
                errors[method] = {"status": row["status"], "error_type": row["error_type"], "stop_reason": row["stop_reason"]}
            chosen = next((x for x in saved if x[0] == "RNJ+WZZT"), saved[0] if saved else None)
            screen = chosen[1] if chosen else {}
            records = screen.get("diagnostics", {}).get("candidate_records", [])
            reasons = Counter(reason for r in records for reason in r.get("rejection_reasons", []))
            conditions.append({"case": case, "variant": variant, "screen_available": bool(saved),
                "representative_method": chosen[0] if chosen else None,
                "available_methods": [x[0] for x in saved],
                "identical_screen_excluding_elapsed_and_requested_groups": len({x[2] for x in saved}) <= 1 if saved else None,
                "screen_sha256": {x[0]: x[3] for x in saved},
                "parameters": screen.get("parameters"),
                "selected_count": len(screen.get("selected_support_labels", [])) if saved else None,
                "selected_support_labels": screen.get("selected_support_labels"),
                "confidence_records": screen.get("confidence_records"),
                "candidate_records": records if saved else None,
                "rejection_reason_counts": dict(reasons), "method_outcomes": errors,
                "frequency_interpretation": "bootstrap algorithmic stability; no edge-weight confidence interval"})
            candidates.extend({"case": case, "variant": variant, "representative_method": chosen[0], **r} for r in records)
    return conditions, candidates


def summarize(root, *, recovery_roots=(), output_dir=None, allow_partial=False):
    """Write fixed-denominator saved-product tables; never fit or rescore truth."""
    root = Path(root).resolve()
    protocol = _gate(root, allow_partial)
    output = Path(output_dir).resolve() if output_dir else root/"saved_product_summary"
    if output == root or any(output.is_relative_to(root/p) for p in (
            "jobs", "inputs", "baseline_reuse", "source_snapshot", "evaluation_network")):
        raise SummaryValidationError("summary output must be a separate directory")
    attempts = {}
    bindings = {}
    for baseline, jobs in ((False, protocol["jobs"]), (True, protocol["baseline_jobs"])):
        for job in jobs:
            attempt = _attempt(root, protocol, job, baseline=baseline)
            attempts[job["job_id"]] = attempt
            bindings[job["job_id"]] = {"directory": str(attempt[2]), "artifact_sha256": attempt[3]}
    first_rows = [dict(a[0]) for a in attempts.values()]
    effective = {r["job_id"]: {**r, "original_status": r["status"], "original_elapsed_seconds": r["elapsed_seconds"],
        "original_retained_fallback": r["retained_fallback"], "original_resolution_clade_f1": r["resolution_clade_f1"],
        "first_attempt_metrics": {key: r[key] for key in METRICS},
        "recovery_applied": False, "retry_elapsed_seconds": None, "combined_spent_seconds": r["elapsed_seconds"]} for r in first_rows}
    recovery_rows, recovery_bindings = [], []
    for recovery_root in map(lambda p: Path(p).resolve(), recovery_roots):
        if recovery_root == root or (recovery_root/"controller.lock").exists():
            raise SummaryValidationError("recovery must be a separate inactive saved run")
        retry_protocol = _read(recovery_root/"protocol.json")
        if len({j["job_id"] for j in retry_protocol.get("jobs", [])}) != 64:
            raise SummaryValidationError("recovery protocol must retain the complete 64-job plan")
        for job in retry_protocol["jobs"]:
            directory = _inside(recovery_root, "jobs/"+job["job_id"])
            if not (directory/"result.json").exists():
                continue
            if job["job_id"] not in attempts:
                raise SummaryValidationError("unknown recovery job")
            retry = _attempt(recovery_root, retry_protocol, job)
            if retry[0]["status"] not in FINAL:
                raise SummaryValidationError("recovery attempt is still active")
            original_job = next(j for j in protocol["jobs"] if j["job_id"] == job["job_id"])
            eligible, reasons, details = _recovery_check(root, recovery_root, protocol, retry_protocol,
                original_job, attempts[job["job_id"]], retry)
            if original_job != job:
                eligible, reasons = False, reasons+["job_configuration_difference"]
            record = {**retry[0], "recovery_eligible": eligible, "rejection_reasons": reasons,
                      "original_status": attempts[job["job_id"]][0]["status"],
                      "original_elapsed_seconds": attempts[job["job_id"]][0]["elapsed_seconds"],
                      "retry_elapsed_seconds": retry[0]["elapsed_seconds"], "recovery_applied": False}
            if eligible and effective[job["job_id"]]["recovery_applied"]:
                raise SummaryValidationError("multiple eligible recoveries: predeclare one attempt rather than choosing a score")
            previous = effective[job["job_id"]]
            spent, extra = previous["combined_spent_seconds"], retry[0]["elapsed_seconds"]
            previous["combined_spent_seconds"] = spent+extra if spent is not None and extra is not None else None
            record["combined_spent_seconds"] = previous["combined_spent_seconds"]
            if eligible:
                effective[job["job_id"]] = {**previous, **retry[0],
                    "original_status": previous["original_status"], "original_elapsed_seconds": previous["original_elapsed_seconds"],
                    "original_retained_fallback": previous["original_retained_fallback"],
                    "original_resolution_clade_f1": previous["original_resolution_clade_f1"],
                    "recovery_applied": True, "retry_elapsed_seconds": retry[0]["elapsed_seconds"],
                    "combined_spent_seconds": previous["combined_spent_seconds"]}
                record["recovery_applied"] = True
            recovery_rows.append(record)
            recovery_bindings.append({"job_id": job["job_id"], "recovery_root": str(recovery_root),
                "original_protocol_sha256": _sha(root/"protocol.json"), "recovery_protocol_sha256": _sha(recovery_root/"protocol.json"),
                "original_source_sha256": protocol.get("source_sha256", {}), "recovery_source_sha256": retry_protocol.get("source_sha256", {}),
                "eligible": eligible, "rejection_reasons": reasons, **details})
    effective_rows = list(effective.values())
    screen_rows, candidate_rows = _screens(attempts)
    manifest = {"schema_version": 1, "partial_fixture_only": allow_partial, "denominator": 96,
        "new_jobs": 64, "reused_baselines": 32, "original_root": str(root),
        "original_protocol_sha256": _sha(root/"protocol.json"),
        "execution_complete_sha256": _sha(root/"execution_complete.json") if (root/"execution_complete.json").exists() else None,
        "source_sha256": protocol.get("source_sha256", {}),
        "dataset_manifest_sha256": protocol.get("dataset_manifest_sha256"),
        "baseline_evidence": protocol.get("baseline_evidence"), "original_attempts": bindings,
        "recovery_attempts": recovery_bindings, "selection_rule": "failed PermissionError + matching inputs/configuration + complete recovery; metrics never select recovery",
        "summary_code_sha256": _sha(Path(__file__)), "estimation_performed": False,
        "timing_scope": "nonexclusive; concurrency is configured capacity, not actual active count; recovery overlapped original workers and older AC288 work"}
    for binding in bindings.values():
        for name, checksum in binding["artifact_sha256"].items():
            if _sha(Path(binding["directory"])/name) != checksum:
                raise SummaryValidationError("saved original artifact changed while summarizing")
    for binding in recovery_bindings:
        directory = Path(binding["recovery_root"])/"jobs"/binding["job_id"]
        for name, checksum in binding.get("retry_artifact_sha256", {}).items():
            if _sha(directory/name) != checksum:
                raise SummaryValidationError("saved recovery artifact changed while summarizing")
    output.mkdir(parents=True, exist_ok=True)
    frozen = output/"recording_recovery_manifest.json"
    if frozen.exists() and _read(frozen) != manifest:
        raise SummaryValidationError("summary manifest is frozen; choose a new output directory")
    _write_csv(output/"first_attempt.csv", first_rows)
    _write_csv(output/"recovery_attempts.csv", recovery_rows)
    _write_csv(output/"effective_results.csv", effective_rows)
    _write_csv(output/"aggregate_by_method.csv", _aggregate(effective_rows, ("method",)))
    _write_csv(output/"aggregate_by_case.csv", _aggregate(effective_rows, ("case",)))
    _write_csv(output/"aggregate_by_case_method.csv", _aggregate(effective_rows, ("case", "method")))
    _write_csv(output/"screen_conditions.csv", screen_rows)
    _write_csv(output/"screen_candidates.csv", candidate_rows)
    _write(frozen, manifest)
    report = f"""# AC96 保存结果汇总

本表固定保留96项：64项WZZT及初始化/分组对照，32项历史NJ/RNJ复用。未完成、失败和缺失指标均保留；CSV空单元格表示无指标。`first_attempt.csv`保留首次执行，`recovery_attempts.csv`逐项列出额外尝试，`effective_results.csv`应用预声明的记录I/O恢复规则。原失败保留模型的指标用`retained_fallback`标记，不能算完整成绩。有效恢复不按F1挑选，即使恢复F1低于原失败的保留模型，也采用完整恢复。

主F1为`resolution_clade_f1`，对完整物理真簇评分；`family_clade_f1`和`active_clade_f1`另列。R/X误差为响应矩阵相对Frobenius误差。全部96点用于拟合，验证集与独立测试集均为0，因此本表不作独立泛化声明。无源与三种Source条件使用同一失败分母口径；Source提供P/Q时序而位置未知。

`aggregate_by_*`汇总effective结果，同时列出available与complete均值及各自计数。available包含保留模型，应结合失败/未完成数量阅读；complete只纳入完整结果。`screen_conditions.csv`固定16个网络/源条件，包含选块、频率、子簇频率和否决原因；bootstrap频率表示算法稳定性。跨方法screen若不一致会显式标记。

原始计算耗时不包含复用后处理时间。`original_elapsed_seconds`、`retry_elapsed_seconds`和`combined_spent_seconds`分别记录原尝试、采用恢复及全部已提供尝试的耗时。运行非独占：配置workers=4不代表恢复时实际运行4项，恢复与原任务、旧AC288任务可能重叠；耗时适合描述本次资源消耗，不作精确硬件性能排名。

分母：96；首次状态：{dict(Counter(r['status'] for r in first_rows))}；有效状态：{dict(Counter(r['status'] for r in effective_rows))}；采用恢复：{sum(r['recovery_applied'] for r in effective_rows)}。证据及恢复资格核对见`recording_recovery_manifest.json`。本工具只读取保存产物，没有拟合或调用求解器。
"""
    (output/"report.md").write_text(report, encoding="utf-8")
    return {"output_directory": str(output), "denominator": 96,
        "first_status_counts": dict(Counter(r["status"] for r in first_rows)),
        "effective_status_counts": dict(Counter(r["status"] for r in effective_rows)),
        "recovery_attempts": len(recovery_rows), "recoveries_applied": sum(r["recovery_applied"] for r in effective_rows),
        "screen_conditions": len(screen_rows), "manifest_sha256": _sha(frozen)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--recovery-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="hand-written fixtures only")
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, recovery_roots=args.recovery_root,
        output_dir=args.output_dir, allow_partial=args.allow_partial), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
