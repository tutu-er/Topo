"""Reuse the 32 frozen NJ/RNJ fits; add independent resolution scores only.

No estimator or runtime package is imported. Original files are read-only and
NPZ files are copied byte for byte. The returned inventory binds historical
producer evidence and the new postprocessing separately for a caller protocol.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import pandas as pd

from research_experiments.rnj.audit_rx75_source_comparison import (
    independent_calibration, independent_tree_resolution,
)
from research_experiments.rnj.audit_source_ac96_benchmark import (
    clade_metrics, physical_truth, relative_error,
)

ROOT = Path(__file__).resolve().parents[2]
CASES = ("paper15", "soumalas11", "flynn16", "pengwah18")
VARIANTS = ("no_source", "constant", "intermittent", "smooth")
METHODS = ("NJ", "RNJ")
TOLERANCE = 1e-10


class ReuseValidationError(ValueError):
    pass


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary, path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def _jobs():
    return [{"job_id": f"{c}__{v}__{m}", "case": c, "variant": v, "method": m}
            for c in CASES for v in VARIANTS for m in METHODS]


def _sets(clades):
    return {frozenset(map(int, c)) for c in clades}


def _serialized(clades):
    return [sorted(c) for c in sorted(_sets(clades), key=lambda c: (len(c), tuple(sorted(c))))]


def _source_hashes():
    paths = [Path(__file__), ROOT/"research_experiments/rnj/audit_rx75_source_comparison.py",
             ROOT/"research_experiments/rnj/audit_source_ac96_benchmark.py",
             ROOT/"research_experiments/rnj/audit_source_ac288.py",
             ROOT/"research_experiments/rnj/audit_rnj_initialization.py"]
    return {p.relative_to(ROOT).as_posix(): _sha(p) for p in paths}


def _literal(path, function, argument):
    if not Path(path).is_file():
        return None
    for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8-sig"))):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            args = node.args.args[-len(node.args.defaults):]
            for parameter, value in zip(args, node.args.defaults):
                if parameter.arg == argument and isinstance(value, ast.Constant):
                    return value.value
    return None


def _version_report(original, environment):
    old = original/"source_snapshot/rnj_wzzt_core/rnj_wzzt/estimation/multiscenario.py"
    current = ROOT/"outputs/source_ac288_runtime_20261018/rnj_wzzt/solver_budgets.py"
    value = None
    if current.exists():
        for node in ast.walk(ast.parse(current.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                if any(isinstance(t, ast.Name) and t.id == "DEFAULT_QP_MAX_ITERATIONS" for t in node.targets):
                    value = node.value.value
    return {"historical_git_head": environment.get("git_head"), "historical_environment": environment,
        "historical_QP_max_iterations": _literal(old, "fit_projected_sensitivity", "constraint_refine_iterations"),
        "current_72af_QP_max_iterations": value,
        "current_72af_budget_source": str(current) if current.exists() else None,
        "current_72af_budget_source_sha256": _sha(current) if current.exists() else None,
        "interpretation": "historical fits are reused, not relabelled as new-runtime fits; successful historical iteration counts are preserved",
        "timing_scope": "original compute/elapsed seconds stay historical; audit/postprocess seconds are separate; no claim of exclusive or contemporaneous timing"}


def _audit_inputs(original, dataset):
    protocol, manifest = _read(original/"protocol.json"), _read(dataset/"manifest.json")
    signature, records, extra = _sha(original/"protocol.json"), [], []
    def require(ok, label, *, supplemental=False):
        (extra if supplemental else records).append({"check": label, "passed": bool(ok)})
    require(_sha(dataset/"manifest.json") == protocol["dataset_manifest_sha256"], "dataset manifest SHA")
    require(protocol["samples_per_fit"] == 96 and protocol["validation_samples"] == 0, "96 rows and no validation")
    require(protocol["rx75_weight"] == .75 and protocol["tree_tolerance_factor"] == .16, "RX75 and original tree tolerance")
    require(manifest["settings"]["v_noise_rel"] == manifest["settings"]["root_meter_noise_rel"] == .0002, "original voltage noise .0002")
    require(tuple(protocol["cases"]) == CASES and tuple(protocol["variants"]) == VARIANTS, "all 32 predeclared baseline conditions", supplemental=True)
    snapshots = {r["path"]: r["sha256"] for r in _read(original/"source_snapshot_manifest.json")}
    for relative, checksum in protocol["source_sha256"].items():
        require(snapshots.get(relative) == checksum, "source manifest/"+relative)
        require(_sha(original/"source_snapshot"/relative) == checksum, "source snapshot/"+relative)
    for relative, checksum in snapshots.items():
        if relative not in protocol["source_sha256"]:
            require(_sha(original/"source_snapshot"/relative) == checksum, "additional archived loader/"+relative, supplemental=True)
    fits, summaries = {}, []
    for job in _jobs():
        key = job["job_id"]
        folder, condition = original/"jobs"/key, dataset/job["case"]/job["variant"]
        result, model, payload = (_read(folder/name) for name in ("result.json", "model.json", "payload.json"))
        hashes = _read(folder/"inputs_hash.json")["file_sha256"]
        copied = Path(payload["input_dir"])
        require(result["protocol_sha256"] == payload["protocol_sha256"] == signature, key+"/protocol signature")
        require(payload["config"] == protocol, key+"/producer payload config")
        require(payload["job"]["job_id"] == result["job_id"] == key, key+"/producer job identifier")
        require(result["status"] == "complete" and result["has_model"] and model["has_model"], key+"/complete saved model")
        require(model["matrix_origin"] == result["matrix_origin"] == "source_adjusted_ordered_regression", key+"/pre-tree matrix origin")
        require(model["source_h"] is None and model["support_indices"] is None, key+"/FWL response rather than physical h")
        diagnostics = model["regression_diagnostics"]
        require(diagnostics["source_location_used"] is False, key+"/unknown source location")
        originals = {p.relative_to(condition).as_posix() for sub in ("observed", "source") for p in (condition/sub).iterdir() if p.is_file()}
        copies = {p.relative_to(copied).as_posix() for p in copied.rglob("*") if p.is_file()}
        require(set(hashes) == originals == copies, key+"/exact observed-source whitelist")
        for relative, checksum in hashes.items():
            require(_sha(condition/relative) == checksum, key+"/original input/"+relative)
            require(_sha(copied/relative) == checksum, key+"/worker input copy/"+relative)
            require(manifest["file_sha256"][f"{job['case']}/{job['variant']}/{relative}"] == checksum, key+"/generation manifest/"+relative)
        frames = {name: pd.read_csv(condition/f"observed/{field}.csv", index_col="t", float_precision="round_trip")
                  for name, field in (("P", "P_terminal"), ("Q", "Q_terminal"), ("Y", "drop_target"))}
        source = pd.read_csv(condition/"source/amplitudes.csv", index_col="t", usecols=["t", "P_pu", "Q_pu"], float_precision="round_trip")
        labels, arrays = list(frames["P"].columns.astype(int)), _arrays(folder/"model.npz")
        require(labels == model["terminal_labels"], key+"/terminal order")
        for name, frame in frames.items():
            require(np.array_equal(arrays[name], frame.to_numpy()) and arrays[name].shape == (96, len(labels)), key+"/all saved "+name+" rows")
        for name, column in (("P_source", "P_pu"), ("Q_source", "Q_pu")):
            require(np.array_equal(arrays[name], source[column].to_numpy()), key+"/known source "+name)
        require(arrays["R"].shape == arrays["X"].shape == (len(labels), len(labels)), key+"/R-X dimensions")
        g = np.vstack((model["source_g_r"], model["source_g_x"]))
        U = np.column_stack((arrays["P_source"], arrays["Q_source"]))
        prediction = arrays["P"]@arrays["R"].T+arrays["Q"]@arrays["X"].T+U@g
        require(abs(float(np.mean(abs(prediction-arrays["Y"])))-model["train_mae"]) < 1e-12, key+"/saved Source prediction MAE")
        require(diagnostics["collapse_or_tolerance_factor"] == .16, key+"/saved tree tolerance")
        require(source.index.equals(frames["P"].index) and all(f.index.equals(source.index) for f in frames.values()), key+"/Source and observation time alignment", supplemental=True)
        require(g.shape == (2, len(labels)) and np.isfinite(g).all(), key+"/free source response dimensions", supplemental=True)
        rank = int(np.linalg.matrix_rank(U))
        require(rank == diagnostics["source_rank"], key+"/source rank", supplemental=True)
        require(diagnostics["success"] is True, key+"/historical QP success", supplemental=True)
        require(diagnostics["intercept_added"] is False and diagnostics["source_path_constraints"] is False, key+"/free-G regression definition", supplemental=True)
        require(np.max(abs(U.T@(arrays["Y"]-prediction)), initial=0.) <= 1e-9, key+"/free-G normal equation (audit only)", supplemental=True)
        right = np.linalg.svd(U, full_matrices=False)[2]
        require(np.max(abs(right[rank:]@g), initial=0.) <= 1e-9, key+"/minimum-norm free-G gauge (audit only)", supplemental=True)
        raw_tree = independent_tree_resolution(model["tree"], labels, 0.)
        require(_sets(model["clades"]) == raw_tree["family"], key+"/saved raw tree clades", supplemental=True)
        fits[key] = (job, folder, result, model, arrays)
        summaries.append({**job, "status": result["status"], "source_input_files": len(hashes), "training_rows": 96,
                          "historical_QP_success": diagnostics["success"], "historical_QP_iterations": diagnostics["iterations"]})
    for case in CASES:
        for path in (dataset/case/"network").iterdir():
            if path.is_file():
                require(manifest["file_sha256"].get(path.relative_to(dataset).as_posix()) == _sha(path), case+"/physical network manifest/"+path.name, supplemental=True)
        for variant in VARIANTS:
            nj, rnj = fits[f"{case}__{variant}__NJ"][4], fits[f"{case}__{variant}__RNJ"][4]
            require(np.array_equal(nj["R"], rnj["R"]) and np.array_equal(nj["X"], rnj["X"]), case+"/"+variant+"/same NJ-RNJ calibration matrices", supplemental=True)
    return protocol, manifest, fits, summaries, records, extra


def _save_audit(base, records, supplemental, summaries, *, validation_seconds):
    failures = [r["check"] for r in records+supplemental if not r["passed"]]
    report = {"baseline_denominator": 32, "checks": len(records), "passed_checks": sum(r["passed"] for r in records),
              "supplemental_checks": len(supplemental), "failures": failures, "check_records": records,
              "supplemental_records": supplemental, "jobs": summaries, "validation_seconds": validation_seconds,
              "scope": "historical 1592-check provenance/input arithmetic reproduced; no estimator, source refit or topology inference"}
    _write(base/"reuse_audit.json", report)
    fields = sorted({key for row in summaries for key in row})
    with (base/"reuse_audit.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    return report


def reuse_baselines(original_root, dataset_root, output_dir):
    """Write ``output_dir/baseline_reuse/jobs`` and return protocol bindings.

    A previously completed reuse directory is immutable: verify its source
    fingerprint and artifact inventory, then return it without rewriting time.
    """
    original, dataset, output = (Path(p).resolve() for p in (original_root, dataset_root, output_dir))
    base = output/"baseline_reuse"
    if any(base == p or base.is_relative_to(p) or p.is_relative_to(base) for p in (original, dataset)):
        raise ValueError("baseline reuse output must be separate from historical results and data")
    started = perf_counter()
    protocol, manifest, fits, summaries, records, extra = _audit_inputs(original, dataset)
    provenance_sources = _source_hashes()
    origins = {job["job_id"]: {name: _sha(folder/name) for name in ("model.json", "model.npz", "result.json", "inputs_hash.json", "payload.json")}
               for job, folder, result, model, arrays in fits.values()}
    binding = {"original_protocol_sha256": _sha(original/"protocol.json"), "dataset_manifest_sha256": _sha(dataset/"manifest.json"),
               "original_environment_sha256": _sha(original/"environment.json"),
               "original_source_snapshot_manifest_sha256": _sha(original/"source_snapshot_manifest.json"),
               "original_job_files": origins, "source_hashes": provenance_sources}
    fingerprint = _digest(binding)
    prior = base/"reuse_provenance.json"
    if prior.exists():
        saved = _read(prior)
        if saved["fingerprint"] != fingerprint or any(not r["passed"] for r in records+extra):
            raise ReuseValidationError("historical inputs, producer evidence or reuse sources changed")
        for relative, checksum in saved["inventory"].items():
            if _sha(output/relative) != checksum:
                raise ReuseValidationError("immutable reused artifact changed: "+relative)
        return {**saved, "baseline_provenance_sha256": _sha(prior), "baseline_provenance_path": prior.relative_to(output).as_posix()}
    if base.exists() and any(base.iterdir()):
        raise FileExistsError("choose a fresh baseline_reuse directory; partial evidence is preserved")
    base.mkdir(parents=True, exist_ok=True)
    report = _save_audit(base, records, extra, summaries, validation_seconds=perf_counter()-started)
    if report["failures"]:
        raise ReuseValidationError(f"baseline reuse audit failed ({len(report['failures'])}); see {base/'reuse_audit.json'}")
    version = _version_report(original, _read(original/"environment.json"))
    version["historical_QP_iteration_counts"] = sorted({row["historical_QP_iterations"] for row in summaries})
    baseline_jobs = []
    for job, folder, old_result, old_model, arrays in fits.values():
        post_started = perf_counter()
        directory = base/"jobs"/job["job_id"]
        directory.mkdir(parents=True)
        labels = old_model["terminal_labels"]
        nj = fits[f"{job['case']}__{job['variant']}__NJ"]
        calibration = {**independent_calibration(nj[4]["R"], nj[4]["X"]), "terminal_count": len(labels),
            "scale": "normalized_RX75_geometry", "calibration_source_job": nj[0]["job_id"],
            "calibration_source_npz_sha256": origins[nj[0]["job_id"]]["model.npz"],
            "calibration_contract": "fixed from historical pre-tree ordered/FWL NJ matrices; no fit or truth selection"}
        resolved = independent_tree_resolution(old_model["tree"], labels, calibration["tau"])
        truth = physical_truth(dataset/job["case"]/"network", labels)
        metrics = {}
        for prefix, clades in (("family", old_model["clades"]), ("active", old_model["clades"]), ("resolution", resolved["resolution"])):
            scores = clade_metrics(clades, truth["clades"], labels)
            metrics.update({prefix+"_"+key: value for key, value in scores.items()})
        metrics.update(r_relative_error=relative_error(arrays["R"], truth["R"]), x_relative_error=relative_error(arrays["X"], truth["X"]))
        for key in ("clade_f1", "precision", "recall"):
            if not np.isclose(metrics["family_"+key], old_result["metrics"][key], atol=1e-12, rtol=1e-12):
                raise ReuseValidationError("historical raw topology score changed: "+job["job_id"])
        for key in ("r_relative_error", "x_relative_error"):
            if not np.isclose(metrics[key], old_result["metrics"][key], atol=1e-12, rtol=1e-12):
                raise ReuseValidationError("historical physical score changed: "+job["job_id"])
        metrics.update(clade_f1=metrics["resolution_clade_f1"], precision=metrics["resolution_precision"], recall=metrics["resolution_recall"])
        shutil.copyfile(folder/"model.npz", directory/"model.npz")
        if _sha(directory/"model.npz") != origins[job["job_id"]]["model.npz"]:
            raise ReuseValidationError("NPZ byte-copy verification failed")
        post_seconds = perf_counter()-post_started
        runtime = {"compute_seconds": old_result.get("compute_seconds", old_result["elapsed_seconds"]),
                   "elapsed_seconds": old_result["elapsed_seconds"], "model_timing": old_model.get("timing"),
                   "scope": "historical producer run; postprocessing is separate"}
        model = {**old_model, "schema_version": 4, "has_model": True, "reused_baseline": True,
            "family_clades": old_model["clades"], "active_clades": old_model["clades"], "resolution_clades": _serialized(resolved["resolution"]),
            "resolution_tree": {"root": old_model["tree"]["root"], "terminals": labels, "edges": resolved["edges"],
                "clades": _serialized(resolved["resolution"]), "contracted_edges": resolved["contracted_edges"],
                "tau": calibration["tau"], "absolute_tolerance": TOLERANCE}, "rx75_calibration": calibration,
            "resolution_absolute_tolerance": TOLERANCE, "training_rows": 96, "validation_supplied": False,
            "validation_mae": None, "validation_se": None, "primary_topology_metric": "resolution_clade_f1",
            "original_metrics": old_result["metrics"], "original_runtime": runtime, "postprocess_seconds": post_seconds}
        result = {**old_result, "reused_baseline": True, "role": "baseline_reuse", "metrics": metrics,
            "original_metrics": old_result["metrics"], "original_runtime": runtime, "postprocess_seconds": post_seconds,
            "compute_seconds": runtime["compute_seconds"], "elapsed_seconds": runtime["elapsed_seconds"],
            "primary_topology_metric": "resolution_clade_f1", "producer_protocol_sha256": binding["original_protocol_sha256"],
            "truth_read_by": "reuse_parent_postprocessor_only", "estimation_performed": False}
        _write(directory/"model.json", model)
        _write(directory/"result.json", result)
        _write(directory/"reuse_provenance.json", {**job, "original_directory": str(folder), "original_file_sha256": origins[job["job_id"]],
            "original_protocol_sha256": binding["original_protocol_sha256"], "dataset_manifest_sha256": binding["dataset_manifest_sha256"],
            "input_sha256": _read(folder/"inputs_hash.json")["file_sha256"], "reused_npz_sha256": _sha(directory/"model.npz"),
            "calibration_source_job": nj[0]["job_id"], "original_runtime": runtime, "postprocess_seconds": post_seconds,
            "estimation_performed": False, "raw_tree_reinferred": False, "source_response_refitted": False})
        baseline_jobs.append({**job, "directory": directory.relative_to(output).as_posix()})
    inventory = {p.relative_to(output).as_posix(): _sha(p) for p in sorted(base.rglob("*")) if p.is_file()}
    saved = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "fingerprint": fingerprint,
        "baseline_jobs": baseline_jobs, "baseline_reuse_directory": "baseline_reuse", "inventory": inventory,
        **binding, "original_root": str(original), "dataset_root": str(dataset), "original_version": version,
        "baseline_denominator": 32, "base_audit_checks": report["checks"], "supplemental_checks": report["supplemental_checks"],
        "validation_seconds": report["validation_seconds"], "estimation_performed": False,
        "resolution_policy": "independent calibration; contract only hidden-child raw edges with strength+1e-10<tau; score against full physical tree"}
    _write(prior, saved)
    return {**saved, "baseline_provenance_sha256": _sha(prior), "baseline_provenance_path": prior.relative_to(output).as_posix()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_root", type=Path)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    result = reuse_baselines(args.original_root, args.dataset_root, args.output_dir)
    print(json.dumps({k: result[k] for k in ("baseline_denominator", "fingerprint", "baseline_provenance_sha256")}, indent=2))


if __name__ == "__main__":
    main()
