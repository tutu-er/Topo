"""Posthoc stage diagnostics for saved RNJ-tree initialization on AC96.

Production entry requires the original complete marker and the final summary's
effective table/recovery manifest. No estimator or solver is imported. Path0
is an already executed fixed-family joint L1/Source refit, not a new algorithm
run. Physical truth scores stages after selection; it never selects a point.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from research_experiments.rnj import summarize_rnj_blocks96 as summary
from research_experiments.rnj.audit_source_ac96_benchmark import (
    Audit, clade_metrics, optimality_certificate, physical_truth, relative_error,
)
from research_experiments.rnj import audit_source_ac288 as evidence
from research_experiments.rnj.audit_rx75_source_comparison import independent_tree_resolution, resolution_atoms


METHOD = "RNJ+WZZT_tree_init"
KINDS = ("family", "active", "resolution")
SCOPE = "事后阶段诊断：path0为已执行的固定候选族联合Source/RX75 L1拟合；物理F1未用于选模"


def _read(path):
    return summary._read(path)


def _sha(path):
    return summary._sha(path)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _boolean(value):
    if value in (True, "True", "true"):
        return True
    if value in (False, "False", "false"):
        return False
    raise ValueError("an explicit boolean is required in the effective table")


def _write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def _verified_artifacts(directory, binding):
    if Path(binding["directory"]).resolve() != Path(directory).resolve():
        raise ValueError("effective directory disagrees with manifest")
    for name, checksum in binding["artifact_sha256"].items():
        target = (Path(directory)/name).resolve()
        if not target.is_relative_to(Path(directory).resolve()) or not target.is_file() or _sha(target) != checksum:
            raise ValueError("summary artifact hash changed: "+str(target))


def completed_summary(root, summary_directory):
    """Complete all gates before any model or solver record is opened."""
    root, directory = Path(root).resolve(), Path(summary_directory).resolve()
    manifest_path, csv_path = directory/"recording_recovery_manifest.json", directory/"effective_results.csv"
    if (root/"controller.lock").exists():
        raise ValueError("original controller is active")
    if not (root/"execution_complete.json").is_file():
        raise ValueError("original execution_complete.json is required")
    if not manifest_path.is_file() or not csv_path.is_file():
        raise ValueError("final effective_results.csv and recording_recovery_manifest.json required")
    manifest = _read(manifest_path)
    if (manifest.get("partial_fixture_only") is not False or manifest.get("denominator") != 96
            or manifest.get("new_jobs") != 64 or manifest.get("reused_baselines") != 32
            or Path(manifest["original_root"]).resolve() != root
            or manifest.get("original_protocol_sha256") != _sha(root/"protocol.json")
            or manifest.get("execution_complete_sha256") != _sha(root/"execution_complete.json")):
        raise ValueError("summary is partial or bound to a different completed run")
    # Only after both completion and final-summary bindings exist, check all
    # final RESULT records. No model, solver event or NPZ is opened here.
    protocol = summary._gate(root, False)
    if manifest.get("dataset_manifest_sha256") != protocol.get("dataset_manifest_sha256"):
        raise ValueError("summary dataset manifest binding differs from protocol")
    rows = _csv(csv_path)
    jobs = protocol["jobs"]+protocol["baseline_jobs"]
    planned = {job["job_id"]: job for job in jobs}
    if len(rows) != 96 or len({row["job_id"] for row in rows}) != 96 or set(planned) != {row["job_id"] for row in rows}:
        raise ValueError("effective table must retain all96 unique planned jobs")
    by_id, chosen = {row["job_id"]: row for row in rows}, {}
    for key, job in planned.items():
        row = by_id[key]
        if any(row[name] != job[name] for name in ("case", "variant", "method")) or row["status"] not in summary.FINAL:
            raise ValueError("nonfinal/inconsistent effective job: "+key)
        first = manifest["original_attempts"][key]
        eligible = [record for record in manifest["recovery_attempts"] if record["job_id"] == key and record.get("eligible") is True]
        if len(eligible) > 1:
            raise ValueError("multiple eligible recoveries cannot be selected by F1")
        recovery = eligible[0] if eligible else None
        expected = (Path(recovery["recovery_root"])/"jobs"/key).resolve() if recovery else Path(first["directory"]).resolve()
        if Path(row["result_directory"]).resolve() != expected or _boolean(row["recovery_applied"]) is not bool(recovery):
            raise ValueError("effective directory does not apply the exact manifest recovery rule: "+key)
        chosen[key] = {"job": job, "effective": row, "directory": expected,
                       "first_binding": first, "recovery": recovery}
    return protocol, manifest, chosen


def verify_chosen(root, protocol, selection):
    """Validate the same final recovery; never substitute a preferred score."""
    root, job = Path(root).resolve(), selection["job"]
    recovery, directory = selection["recovery"], selection["directory"]
    original_directory = summary._inside(root, job.get("directory", "jobs/"+job["job_id"]))
    if recovery is None:
        if directory != original_directory:
            raise ValueError("original effective directory differs from protocol")
        _verified_artifacts(original_directory, selection["first_binding"])
        return
    retry_root = Path(recovery["recovery_root"]).resolve()
    if (retry_root/"controller.lock").exists():
        raise ValueError("recovery controller is active")
    # Eligibility is not a replacement for the frozen first-attempt binding:
    # an irrelevant edit could leave data/configuration eligibility unchanged.
    _verified_artifacts(original_directory, selection["first_binding"])
    retry_protocol = _read(retry_root/"protocol.json")
    if (_sha(root/"protocol.json") != recovery["original_protocol_sha256"]
            or _sha(retry_root/"protocol.json") != recovery["recovery_protocol_sha256"]):
        raise ValueError("recovery protocol hash changed")
    # Reuse the already reviewed I/O-only eligibility rule, which verifies
    # code, inputs and configuration without comparing F1 values.
    first = summary._attempt(root, protocol, job)
    retry = summary._attempt(retry_root, retry_protocol, job)
    eligible, reasons, _ = summary._recovery_check(root, retry_root, protocol, retry_protocol, job, first, retry)
    if not eligible or recovery.get("rejection_reasons"):
        raise ValueError("recovery is no longer eligible: "+";".join(reasons))
    _verified_artifacts(directory, {"directory": str(directory), "artifact_sha256": recovery["retry_artifact_sha256"]})


def verified_network(root, protocol, case, relative_directory):
    """Bind the COMPLETE copied truth-network inventory before physical scores."""
    root = Path(root).resolve()
    manifest_path = root/"dataset_manifest.json"
    checksum = _sha(manifest_path)
    if checksum != protocol.get("dataset_manifest_sha256"):
        raise ValueError("copied dataset manifest hash changed")
    manifest = _read(manifest_path)
    prefix = case+"/network/"
    expected = {name[len(prefix):]: value for name, value in manifest["file_sha256"].items() if name.startswith(prefix)}
    if not {"metadata.json", "buses.csv", "branches.csv"} <= set(expected):
        raise ValueError("physical network manifest is incomplete: "+case)
    directory = summary._inside(root, relative_directory)
    actual = {path.relative_to(directory).as_posix(): path for path in directory.rglob("*") if path.is_file()}
    if set(actual) != set(expected):
        raise ValueError("physical network file inventory differs from manifest: "+case)
    for name, path in actual.items():
        if not path.resolve().is_relative_to(directory.resolve()) or _sha(path) != expected[name]:
            raise ValueError("physical network hash changed: "+case+"/"+name)
    return {"directory": str(directory), "dataset_manifest_sha256": checksum, "file_sha256": expected}


def matching_event(point, events, *, fixed=False):
    """Bind a saved point to its unique actual certified solve event."""
    diagnostic = point.get("solver", point.get("diagnostics", {}))
    if not optimality_certificate(diagnostic):
        raise ValueError("saved point has no strict numerical certificate")
    matches = [event for event in events if all(event.get(key) == value for key, value in diagnostic.items())
               and event.get("has_incumbent") is True and event.get("optimality_certified") is True
               and optimality_certificate(event)]
    if fixed:
        matches = [event for event in matches if "z" not in event.get("variable_slices", {})
                   and event.get("is_extension") is False
                   and event.get("old_support_indices") == point["support_indices"]]
    if len(matches) != 1:
        raise ValueError("point must bind exactly one actual certified "+("fixed-family " if fixed else "")+"event")
    return matches[0]


def stage_metrics(point, arrays, labels, calibration, truth):
    """Rebuild learned Source responses from all saved h bits, without refit."""
    atom = evidence.atom_reconstruction(point, labels)
    if np.any(atom["r"] < -1e-9) or np.any(atom["x"] < -1e-9):
        raise ValueError("saved weights violate nonnegativity")
    checks = Audit()
    active_source = bool(np.any(arrays["P_source"] != 0.) or np.any(arrays["Q_source"] != 0.))
    gr, gx = evidence.audit_source_path(point, atom, active_source, checks, "stage")
    if not checks.result()["passed"]:
        raise ValueError("saved full Source path/response is inconsistent")
    prediction, loss = evidence.prediction_and_loss(arrays, atom["R"], atom["X"], gr, gx)
    if not np.isclose(loss, point["train_mae"], atol=2e-9, rtol=1e-7):
        raise ValueError("saved stage MAE differs from reconstruction")
    _, keep = resolution_atoms(point["support_indices"], point["r_values"], point["x_values"], calibration)
    clades = {"family": atom["family"], "active": atom["active"],
              "resolution": {frozenset(labels[i] for i in support) for support, retained in zip(point["support_indices"], keep, strict=True)
                             if retained and 1 < len(support) < len(labels)}}
    scores = {f"{kind}_{key}": value for kind in KINDS for key, value in clade_metrics(clades[kind], truth["clades"], labels).items()}
    scores.update(r_relative_error=relative_error(atom["R"], truth["R"]), x_relative_error=relative_error(atom["X"], truth["X"]),
                  train_mae=loss, support_count=len(point["support_indices"]))
    artifact = {"R": atom["R"], "X": atom["X"], "source_g_r": gr, "source_g_x": gx, "prediction": prediction}
    return {"metrics": scores, "clades": clades, "artifact": artifact}


def baseline_metrics(model, arrays, labels, calibration, truth):
    """Keep the historical dense ordered/FWL R/X, not a tree-constrained refit."""
    tree = independent_tree_resolution(model["tree"], labels, calibration["tau"])
    gr, gx = (np.asarray(model[name], dtype=float) for name in ("source_g_r", "source_g_x"))
    prediction, loss = evidence.prediction_and_loss(arrays, arrays["R"], arrays["X"], gr, gx)
    clades = {"family": tree["family"], "active": tree["family"], "resolution": tree["resolution"]}
    scores = {f"{kind}_{key}": value for kind in KINDS for key, value in clade_metrics(clades[kind], truth["clades"], labels).items()}
    scores.update(r_relative_error=relative_error(arrays["R"], truth["R"]), x_relative_error=relative_error(arrays["X"], truth["X"]), train_mae=loss)
    return {"metrics": scores, "clades": clades,
            "artifact": {"R": arrays["R"], "X": arrays["X"], "source_g_r": gr, "source_g_x": gx, "prediction": prediction}}


def clade_change(before, after, truth):
    added, removed = after-before, before-after
    serialize = lambda values: [sorted(v) for v in sorted(values, key=lambda v: tuple(sorted(v)))]
    return {"added_true": serialize(added & truth), "added_false": serialize(added-truth),
            "removed_true": serialize(removed & truth), "removed_false": serialize(removed-truth)}


def _gain(before, after):
    gain = before-after
    return gain, gain/abs(before) if before != 0. else (0. if gain == 0. else None)


def diagnose_condition(model, arrays, baseline_model, baseline_arrays, network_directory):
    """Pure saved-artifact stage analysis. Does not select a best-F1 point."""
    labels = list(map(int, model["terminal_labels"] if model else baseline_model["terminal_labels"]))
    if baseline_model and list(map(int, baseline_model["terminal_labels"])) != labels:
        raise ValueError("RNJ and hybrid terminal orders differ")
    truth = physical_truth(Path(network_directory), labels)
    calibration = model["rx75_calibration"] if model else baseline_model["rx75_calibration"]
    stages, errors, path_rows = {}, {}, []
    if baseline_model and baseline_model.get("has_model"):
        stages["rnj"] = baseline_metrics(baseline_model, baseline_arrays, labels, calibration, truth)
    if model and model.get("has_model") and model.get("path"):
        path = model["path"]
        if model.get("selected_path_index") != len(path)-1:
            raise ValueError("diagnostic only supports the saved last-certified point policy")
        for index, point in enumerate(path):
            try:
                event = matching_event(point, model.get("solver_events", []), fixed=index == 0)
                stage = stage_metrics(point, arrays, labels, calibration, truth)
                if not np.isclose(stage["metrics"]["train_mae"], event["objective"], atol=2e-9, rtol=1e-7):
                    raise ValueError("actual certified event objective differs from stage loss")
                elapsed = event.get("elapsed_seconds")
                if elapsed is not None and (not np.isfinite(elapsed) or elapsed < 0.):
                    raise ValueError("invalid saved elapsed time")
                stage.update(solver_sequence=event["sequence"], elapsed_seconds=elapsed)
                if index == 0:
                    stages["path0"] = stage
                if index == len(path)-1:
                    for channel in ("R", "X"):
                        if not np.allclose(arrays[channel], stage["artifact"][channel], atol=1e-10, rtol=1e-9):
                            raise ValueError("final saved matrix differs from its selected point")
                    stages["final"] = stage
                path_rows.append({"path_index": index, "iteration": point.get("iteration"), "certified": True,
                                  "selected_last_point": index == len(path)-1, "elapsed_seconds": elapsed,
                                  "solver_sequence": event["sequence"], "accepted_gain_saved": point.get("accepted_gain"), **stage["metrics"]})
            except (ValueError, KeyError, TypeError) as error:
                errors[str(index)] = f"{type(error).__name__}: {error}"
                path_rows.append({"path_index": index, "certified": False, "selected_last_point": index == len(path)-1,
                                  "diagnostic_error": errors[str(index)]})
    changes = []
    for left, right, comparison in (("rnj", "path0", "dense_RNJ_to_fixed_family_joint_refit"),
                                    ("path0", "final", "fixed_family_to_saved_greedy_final")):
        for kind in KINDS:
            available = left in stages and right in stages
            delta = clade_change(stages[left]["clades"][kind], stages[right]["clades"][kind], truth["clades"]) if available else dict.fromkeys(("added_true", "added_false", "removed_true", "removed_false"))
            changes.append({"comparison": comparison, "clade_kind": kind, "available": available,
                            **delta, **{name+"_count": len(values) if values is not None else None for name, values in delta.items()}})
    gains = {}
    for left, right, name in (("rnj", "path0", "fixed_refit"), ("path0", "final", "greedy")):
        absolute, relative = _gain(stages[left]["metrics"]["train_mae"], stages[right]["metrics"]["train_mae"]) if left in stages and right in stages else (None, None)
        gains[name+"_gain_abs"], gains[name+"_gain_rel"] = absolute, relative
    return {"stages": stages, "changes": changes, "path": path_rows, "errors": errors, "gains": gains,
            "truth_used_for_selection": False, "path0_scope": "first point of the saved final-bound path; expanded-bound restarts are not erased"}


def _load(selection):
    directory = selection["directory"]
    result = _read(directory/"result.json")
    if not result.get("has_model"):
        return None, None, result
    model = _read(directory/"model.json")
    with np.load(directory/"model.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return model, arrays, result


def diagnose(root, summary_directory, output_directory):
    root, output = Path(root).resolve(), Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a fresh diagnostic directory")
    if output == root or any(output.is_relative_to(root/name) for name in ("jobs", "inputs", "baseline_reuse", "source_snapshot", "evaluation_network")):
        raise ValueError("diagnostic output must be separate from original evidence")
    protocol, manifest, chosen = completed_summary(root, summary_directory)
    network_evidence = {}
    for case in summary.CASES:
        for variant in summary.VARIANTS:
            job = chosen[f"{case}__{variant}__{METHOD}"]["job"]
            relative = job.get("evaluation_network_directory", f"evaluation_network/{case}")
            cache_key = case+":"+relative
            if cache_key not in network_evidence:
                network_evidence[cache_key] = verified_network(root, protocol, case, relative)
    rows, changes, paths, evidence_hashes = [], [], [], {}
    artifacts = []
    for case in summary.CASES:
        for variant in summary.VARIANTS:
            key, rnj_key = f"{case}__{variant}__{METHOD}", f"{case}__{variant}__RNJ"
            selection, baseline = chosen[key], chosen[rnj_key]
            verify_chosen(root, protocol, selection)
            verify_chosen(root, protocol, baseline)
            model, arrays, result = _load(selection)
            rnj_model, rnj_arrays, rnj_result = _load(baseline)
            row = {"case": case, "variant": variant, "job_id": key, "status": result["status"],
                   "stop_reason": result.get("stop_reason"), "has_model": bool(model),
                   "result_directory": str(selection["directory"]), "recovery_applied": bool(selection["recovery"]),
                   "original_status": selection["effective"].get("original_status"),
                   "combined_spent_seconds": selection["effective"].get("combined_spent_seconds") or None,
                   "final_run_elapsed_seconds": result.get("elapsed_seconds", result.get("compute_seconds")),
                   "rnj_status": rnj_result["status"], "rnj_matrix_origin": rnj_model.get("matrix_origin") if rnj_model else None,
                   "rnj_original_compute_seconds": rnj_result.get("compute_seconds", rnj_result.get("elapsed_seconds")),
                   "bound_expansions": model.get("bound_expansions") if model else None,
                   "selected_path_index": model.get("selected_path_index") if model else None,
                   "path_length": len(model.get("path", [])) if model else None,
                   "terminal_labels": (model or rnj_model or {}).get("terminal_labels"),
                   "scope": SCOPE, "truth_used_for_selection": False}
            for stage in ("rnj", "path0", "final"):
                row.update({stage+"_"+name: None for name in ("family_clade_f1", "active_clade_f1", "resolution_clade_f1", "r_relative_error", "x_relative_error", "train_mae", "elapsed_seconds", "solver_sequence")})
            if model or rnj_model:
                relative = selection["job"].get("evaluation_network_directory", f"evaluation_network/{case}")
                details = diagnose_condition(model, arrays, rnj_model, rnj_arrays,
                                             network_evidence[case+":"+relative]["directory"])
                row.update(details["gains"], diagnostic_errors=details["errors"], path0_scope=details["path0_scope"])
                saved = {}
                saved["terminal_labels"] = np.asarray((model or rnj_model)["terminal_labels"], dtype=int)
                for stage_name, stage in details["stages"].items():
                    row.update({stage_name+"_"+name: value for name, value in stage["metrics"].items()})
                    row[stage_name+"_elapsed_seconds"] = stage.get("elapsed_seconds")
                    row[stage_name+"_solver_sequence"] = stage.get("solver_sequence")
                    saved.update({stage_name+"_"+name: value for name, value in stage["artifact"].items()})
                if saved:
                    artifacts.append((f"{case}__{variant}.npz", saved))
                changes.extend({"case": case, "variant": variant, **change} for change in details["changes"])
                paths.extend({"case": case, "variant": variant, **point} for point in details["path"])
            rows.append(row)
            for label, choice in ((key, selection), (rnj_key, baseline)):
                evidence_hashes[label] = {"directory": str(choice["directory"]), "model_json_sha256": _sha(choice["directory"]/"model.json") if (choice["directory"]/"model.json").is_file() else None,
                                          "model_npz_sha256": _sha(choice["directory"]/"model.npz") if (choice["directory"]/"model.npz").is_file() else None}
    output.mkdir(parents=True, exist_ok=True)
    (output/"reconstructed").mkdir()
    for filename, arrays in artifacts:
        np.savez_compressed(output/"reconstructed"/filename, **arrays)
    _write_csv(output/"initialization_stages.csv", rows)
    _write_csv(output/"clade_changes.csv", changes)
    _write_csv(output/"accepted_path_posthoc.csv", paths)
    metadata = {"scope": SCOPE, "condition_denominator": 16, "conditions_with_path0": sum(row["path0_resolution_clade_f1"] is not None for row in rows),
                "status_counts": dict(Counter(row["status"] for row in rows)), "original_root": str(root),
                "summary_directory": str(Path(summary_directory).resolve()), "summary_manifest_sha256": _sha(Path(summary_directory)/"recording_recovery_manifest.json"),
                "effective_results_sha256": _sha(Path(summary_directory)/"effective_results.csv"), "original_protocol_sha256": _sha(root/"protocol.json"),
                "execution_complete_sha256": _sha(root/"execution_complete.json"), "diagnostic_code_sha256": _sha(Path(__file__)),
                "evaluation_network_evidence": network_evidence,
                "evidence": evidence_hashes, "estimation_performed": False, "truth_used_for_selection": False,
                "source_response_definition": "saved learned R/X atom weights times complete h, then original-terminal incidence; no source-location truth supplied",
                "time_definition": "elapsed to the unique actual certified solve for saved path0; includes front end, bootstrap and any earlier bound stages; final run time is reported separately",
                "comparison_definition": "RNJ uses historical ordered/FWL L2 dense matrices; path0 is existing same-family joint L1/Source refit under RX75 floor; later rows are the saved greedy path, not an independently rerun method"}
    _write_json(output/"diagnostic_manifest.json", metadata)
    (output/"README.md").write_text("# RNJ初始化阶段诊断\n\n"+SCOPE+"。固定保留16个网络/源工况。\n\n"
        "`initialization_stages.csv`并列历史RNJ、保存的path0和原已选最终点；空值保留无模型、未达到path0或缺少认证事件的情况。"
        "`clade_changes.csv`分别列出RNJ到固定族拟合、固定族到后续贪婪结果的真/假簇增删。"
        "`accepted_path_posthoc.csv`按原路径顺序评分，不按F1挑选；`reconstructed/`保存重建矩阵、完整Source响应和预测。\n\n"
        "path0已属于既有算法中的固定候选族联合L1/Source子程序，并非本工具重跑的独立方法。RNJ的前置L2矩阵与它有不同的目标和结构域。"
        "Source时序已知、位置未知；物理真值仅事后评分。全部96行用于拟合，无验证或独立测试。"
        "path0为最后边界阶段路径的第一个点；到点耗时包含先前回归、筛选及可能的扩界重启。\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.run_dir, args.summary_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
