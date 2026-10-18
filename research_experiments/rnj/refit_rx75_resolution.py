"""Contract RX75-short atoms and jointly refit a finished model in isolation.

Calibration uses only the same-input NJ ordered/FWL dense R/X. Each contraction
is followed by an ordinary fixed-support L1 solve, including a fresh complete
unknown Source path. There is no positive-weight floor, new-support search or
truth-dependent selection. Optional physical scoring runs in the parent only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research_experiments.rnj.rx75_edge_floor import calibrate_rx75_floor, classify_rx75_atoms
from research_experiments.rnj.run_source_ac288 import BudgetedSolver, activate_runtime, save_model
from research_experiments.rnj.run_source_ac96_full_history import MethodBudgetExhausted, read_json, write_json, sha256

DEFAULT_RUNTIME = ROOT / "outputs/source_ac288_runtime_20261018"
DEFAULT_OUTPUT = ROOT / "outputs/source_ac288_rx75_refit_20261018"
TOLERANCE = 1e-10
INPUT_ARRAYS = ("P", "Q", "Y", "P_source", "Q_source", "row_scenario", "row_position")
FINAL_STATUSES = {"complete", "incomplete", "failed"}


def finished_job(directory):
    directory = Path(directory).resolve()
    if not (directory/"result.json").is_file():
        raise ValueError("a final result.json is required; running jobs are rejected")
    result, model = read_json(directory/"result.json"), read_json(directory/"model.json")
    if result.get("status") not in FINAL_STATUSES or model.get("status") not in FINAL_STATUSES:
        raise ValueError("job/model must have a final status; running jobs are rejected")
    if not result.get("has_model") or not model.get("has_model"):
        raise ValueError("a finished job with a saved model is required")
    with np.load(directory/"model.npz", allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in ("R", "X", *INPUT_ARRAYS)}
    return result, model, arrays


def restore_inputs(arrays, labels, source_inputs_class):
    """Recreate three ordered 96-row blocks without opening network/truth."""
    n = len(labels)
    if len(set(labels)) != n or n < 2:
        raise ValueError("distinct original terminal labels are required")
    for name in ("P", "Q", "Y"):
        if arrays[name].shape != (288, n):
            raise ValueError("expected the same 288 by n original-terminal input arrays")
    for name in ("P_source", "Q_source", "row_scenario", "row_position"):
        if arrays[name].shape != (288,):
            raise ValueError("source and row labels must have exactly 288 entries")
    if not all(np.isfinite(arrays[name]).all() for name in INPUT_ARRAYS):
        raise ValueError("nonfinite original observations or source schedules")
    if not np.array_equal(arrays["row_scenario"], np.repeat(np.arange(3), 96)) or not np.array_equal(arrays["row_position"], np.tile(np.arange(96), 3)):
        raise ValueError("row_scenario/row_position must preserve scenario_0,1,2 and local rows 0..95")
    scenarios, sources = [], []
    for i in range(3):
        rows = slice(96*i, 96*(i+1))
        index = pd.Index(arrays["row_position"][rows].astype(int), name="t")
        scenarios.append({"name": f"original_scenario_{i}",
            **{field: pd.DataFrame(arrays[name][rows], index=index, columns=labels)
               for name, field in (("P", "P_terminal"), ("Q", "Q_terminal"), ("Y", "drop_target"))}})
        sources.append(source_inputs_class(pd.Series(arrays["P_source"][rows], index=index),
            pd.Series(arrays["Q_source"][rows], index=index), "original_saved_exact_source_time_series"))
    return scenarios, sources


def contract(supports, r, x, calibration):
    classified = classify_rx75_atoms(supports, r, x, calibration, absolute_tolerance=TOLERANCE)
    return tuple(tuple(supports[i]) for i in classified["retained_indices"]), classified


def model_prediction(arrays, r, x, g_r, g_x):
    source = np.zeros_like(arrays["Y"])
    if g_r is not None:
        source += arrays["P_source"][:, None]*np.asarray(g_r)[None, :]
        source += arrays["Q_source"][:, None]*np.asarray(g_x)[None, :]
    prediction = arrays["P"] @ np.asarray(r).T+arrays["Q"] @ np.asarray(x).T+source
    return prediction, source


def _model(core, fit, labels, classification, step):
    r, x = core.build_matrices_from_atoms(len(labels), fit.supports, fit.r_values, fit.x_values)
    family = [[labels[i] for i in support] for support in fit.supports if 1 < len(support) < len(labels)]
    active = [[labels[i] for i in s] for s, a, b in zip(fit.supports, fit.r_values, fit.x_values, strict=True)
              if 1 < len(s) < len(labels) and max(a, b) > 1e-8]
    return {"R": r, "X": x, "has_model": True, "terminal_labels": labels,
        "support_indices": fit.supports, "support_labels": [[labels[i] for i in s] for s in fit.supports],
        "r_values": fit.r_values, "x_values": fit.x_values, "source_h": fit.source_h,
        "source_g_r": fit.source_response_r, "source_g_x": fit.source_response_x,
        "train_mae": fit.objective, "solver": fit.diagnostics.to_dict(),
        "family_clades": family, "active_clades": active, "clades": family,
        "resolution_classification": classification, "resolution_stable": not classification["contracted_indices"],
        "fixed_refit_step": step, "training_rows": 288, "validation_supplied": False,
        "matrix_origin": "RX75_contraction_then_joint_Source_fixed_support_L1_refit",
        "source_location_used": False, "source_h_refit_each_step": True, "positive_floor_enforced": False,
        "status": "running", "stop_reason": "post_refit_in_progress"}


def worker(payload_path):
    payload = read_json(payload_path)
    # Payload intentionally contains no network directory or Source location.
    output, config = Path(payload["output_dir"]), payload["config"]
    activate_runtime(payload["runtime_dir"])
    from rnj_wzzt.estimation import laminar_l1_milp as core
    from rnj_wzzt.estimation.unmetered_inputs import SourceInputs
    from research_experiments.rnj.prepare_source_ac288_runtime import verify_loaded_modules
    result, original, arrays_all = finished_job(output/"original")
    arrays = {name: arrays_all[name] for name in INPUT_ARRAYS}
    labels = original["terminal_labels"]
    scenarios, sources = restore_inputs(arrays, labels, SourceInputs)
    supports = tuple(tuple(s) for s in original["support_indices"])
    if not all((i,) in supports for i in range(len(labels))):
        raise ValueError("original model must retain every singleton leaf atom")
    calibration = config["calibration"]
    kept, classification = contract(supports, original["r_values"], original["x_values"], calibration)
    started = perf_counter()
    steps, last_model, last_directory = [], None, None
    status, reason, error = "failed", "not_started", None
    try:
        # Every subsequent unstable refit loses at least one internal support.
        for step in range(len(supports)-len(labels)+1):
            directory = output/"steps"/f"{step:04d}"
            directory.mkdir(parents=True)
            removed = [list(supports[i]) for i in classification["contracted_indices"]]
            write_json(directory/"input_family.json", {"previous_supports": supports,
                "retained_supports": kept, "removed_supports": removed,
                "removed_support_labels": [[labels[i] for i in s] for s in removed],
                "classification": classification, "source_h_is_not_reused": True})
            solve_started = perf_counter()
            with BudgetedSolver(core, directory, started, config["budget"]) as observer:
                fit = core.solve_fixed_support_l1(scenarios, kept,
                    r_upper_bound=config["r_upper_bound"], x_upper_bound=config["x_upper_bound"],
                    time_limit=config["budget"]["per_solve_seconds"], solver=config["solver"],
                    presolve=True, disp=False, source_inputs=sources)
            next_kept, next_classification = contract(fit.supports, fit.r_values, fit.x_values, calibration)
            model = _model(core, fit, labels, next_classification, step)
            model.update(r_upper_bound=config["r_upper_bound"], x_upper_bound=config["x_upper_bound"],
                         solver_events=observer.events, elapsed_step_seconds=perf_counter()-solve_started)
            prediction, source_prediction = model_prediction(arrays, model["R"], model["X"], model["source_g_r"], model["source_g_x"])
            np.savez_compressed(directory/"predictions.npz", prediction=prediction, source_prediction=source_prediction,
                                residual=arrays["Y"]-prediction)
            saved = save_model(directory, model, arrays, started)
            write_json(directory/"output_classification.json", next_classification)
            steps.append({"step": step, "support_count_before": len(supports), "support_count_refit": len(fit.supports),
                "contracted_before": len(removed), "contracted_after": len(next_classification["contracted_indices"]),
                "train_mae": fit.objective, "elapsed_seconds": perf_counter()-solve_started,
                "source_h": fit.source_h, "solver": fit.diagnostics.to_dict()})
            write_json(output/"steps.json", steps)
            last_model, last_directory = saved, directory
            if not next_classification["contracted_indices"]:
                status, reason = "complete", "RX75_resolution_stable_after_joint_fixed_support_refit"
                break
            if len(next_kept) >= len(fit.supports):
                raise RuntimeError("an unstable iteration must strictly contract the family")
            supports, kept, classification = fit.supports, next_kept, next_classification
        else:
            raise RuntimeError("finite support-deletion bound exceeded")
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        event_files = sorted((output/"steps").glob("*/solver_events.json"))
        events = read_json(event_files[-1]) if event_files else []
        # A new solve may exhaust the wall budget before producing any event.
        # Its predecessor can still be fully certified; that does not turn a
        # budget interruption into a mathematical or implementation failure.
        budget_interruption = isinstance(exc, (MethodBudgetExhausted, TimeoutError))
        status = "incomplete" if budget_interruption or (events and not events[-1].get("optimality_certified")) else "failed"
        reason = str(exc)
        write_json(output/"error.json", error)
    if last_model is not None:
        last_model.update(status=status, stop_reason=reason)
        write_json(output/"model.json", last_model)
        shutil.copyfile(last_directory/"model.npz", output/"model.npz")
        shutil.copyfile(last_directory/"predictions.npz", output/"predictions.npz")
    write_json(output/"runtime_import.json", {"runtime_dir": payload["runtime_dir"], "module_origins": verify_loaded_modules(payload["runtime_dir"])})
    write_json(output/"result.json", {"status": status, "stop_reason": reason, "has_model": last_model is not None,
        "resolution_stable": bool(last_model and last_model["resolution_stable"]), "steps_completed": len(steps),
        "compute_seconds": perf_counter()-started, "error": error, "truth_used_for_contraction_or_fit": False,
        "original_saved_metrics": result.get("metrics"), "new_topology_search_performed": False})
    return 0 if status == "complete" else 1


def evaluate_parent(output, network):
    """Optional truth access after worker completion; cannot alter selection."""
    from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth, clade_metrics, relative_error
    scores = {}
    for name, folder in (("original", output/"original"), ("post_refit", output)):
        if not (folder/"model.npz").exists():
            continue
        model = read_json(folder/"model.json")
        labels = model["terminal_labels"]
        truth = physical_truth(Path(network), labels)
        with np.load(folder/"model.npz") as arrays:
            score = {"r_relative_error": relative_error(arrays["R"], truth["R"]),
                     "x_relative_error": relative_error(arrays["X"], truth["X"])}
        for prefix in ("family", "active"):
            clades = model.get(f"{prefix}_clades", model.get("clades", []))
            score.update({f"{prefix}_{key}": value for key, value in clade_metrics(clades, truth["clades"], labels).items()})
        scores[name] = score
    write_json(output/"evaluation.json", {"scope": "parent-only posthoc physical scoring; no selection", "scores": scores})
    return scores


def prepare(job_dir, baseline_dir, output_dir, runtime_dir=DEFAULT_RUNTIME):
    job, baseline, output, runtime = (Path(p).resolve() for p in (job_dir, baseline_dir, output_dir, runtime_dir))
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in (job, baseline, runtime)):
        raise ValueError("output must be separate from original jobs and the frozen runtime")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose a fresh output directory; original and derived files are never overwritten")
    result, model, arrays = finished_job(job)
    front_result, front_model, front_arrays = finished_job(baseline)
    if model.get("support_indices") is None or not all(np.isfinite(model.get(name, np.nan)) and model[name] > 0
            for name in ("r_upper_bound", "x_upper_bound")):
        raise ValueError("original job must be an atom model with finite positive saved bounds")
    if front_result.get("method") != "NJ" or front_result.get("variant") != result.get("variant"):
        raise ValueError("calibration requires the same source-variant NJ baseline")
    if front_model.get("matrix_origin") != "source_adjusted_ordered_regression_before_tree_inference":
        raise ValueError("calibration must use the NJ pre-tree ordered/FWL regression matrices")
    if model["terminal_labels"] != front_model["terminal_labels"]:
        raise ValueError("job and calibration terminal orders differ")
    for name in INPUT_ARRAYS:
        if not np.array_equal(arrays[name], front_arrays[name]):
            raise ValueError(f"job and calibration baseline must use identical {name}")
    calibration = calibrate_rx75_floor(front_arrays["R"], front_arrays["X"], factor=.16)
    from research_experiments.rnj.prepare_source_ac288_runtime import verify_runtime
    verify_runtime(runtime)
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for label, directory in (("original", job), ("calibration_baseline", baseline)):
        target = output/label
        target.mkdir()
        for name in ("model.json", "model.npz", "result.json", "inputs_hash.json", "data_use.json", "runtime_import.json"):
            if (directory/name).is_file():
                shutil.copyfile(directory/name, target/name)
                hashes[f"{label}/{name}"] = {"source": str(directory/name), "sha256": sha256(directory/name)}
    config = {"calibration": calibration, "geometry_absolute_tolerance": TOLERANCE,
        "keep_semantics": "singleton always kept; other strengths + 1e-10 >= tau kept, including root stem",
        "r_upper_bound": model["r_upper_bound"], "x_upper_bound": model["x_upper_bound"],
        "budget": {"per_solve_seconds": 6000., "method_wall_seconds": 30000.}, "solver": "gurobi",
        "positive_floor_enforced": False, "new_support_search": False, "training_rows": 288, "validation_rows": 0,
        "source_h_policy": "all unknown Source path bits jointly reoptimized after every contraction",
        "truth_used_for_selection": False}
    write_json(output/"config.json", config)
    write_json(output/"input_hashes.json", hashes)
    from research_experiments.rnj import rx75_edge_floor, run_source_ac288, run_source_ac96_full_history
    write_json(output/"source_hashes.json", {str(Path(p).resolve()): sha256(p) for p in
        (__file__, rx75_edge_floor.__file__, run_source_ac288.__file__, run_source_ac96_full_history.__file__, runtime/"runtime_manifest.json")})
    write_json(output/"payload.json", {"output_dir": output, "runtime_dir": runtime, "config": config})
    return config


def run(job_dir, baseline_dir, output_dir, *, runtime_dir=DEFAULT_RUNTIME, network_dir=None):
    output = Path(output_dir).resolve()
    config = prepare(job_dir, baseline_dir, output, runtime_dir)
    started = perf_counter()
    timed_out, exitcode = False, None
    with (output/"stdout.log").open("x", encoding="utf-8") as stdout, (output/"stderr.log").open("x", encoding="utf-8") as stderr:
        try:
            process = subprocess.run([sys.executable, "-B", "-m", "research_experiments.rnj.refit_rx75_resolution", "--worker", str(output/"payload.json")],
                cwd=ROOT, env=os.environ.copy(), stdout=stdout, stderr=stderr, timeout=config["budget"]["method_wall_seconds"]+30)
            exitcode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
    if not (output/"result.json").exists():
        checkpoints = sorted((output/"steps").glob("*/model.json"))
        checkpoint = checkpoints[-1].parent if checkpoints else None
        if checkpoint is not None:
            model = read_json(checkpoint/"model.json")
            model.update(status="incomplete", stop_reason="post_refit_worker_interrupted")
            write_json(output/"model.json", model)
            for filename in ("model.npz", "predictions.npz"):
                shutil.copyfile(checkpoint/filename, output/filename)
        write_json(output/"result.json", {"status": "incomplete" if timed_out or checkpoint else "failed",
            "has_model": checkpoint is not None, "stop_reason": "process_wall_limit" if timed_out else "worker_process_failed",
            "resolution_stable": False, "process_exitcode": exitcode})
    result = read_json(output/"result.json")
    integrity = {name: sha256(entry["source"]) == entry["sha256"] and sha256(output/name) == entry["sha256"]
                 for name, entry in read_json(output/"input_hashes.json").items()}
    write_json(output/"input_integrity.json", integrity)
    result.update(parent_wall_seconds=perf_counter()-started, process_exitcode=exitcode,
                  original_and_copied_input_files_unchanged=all(integrity.values()))
    write_json(output/"result.json", result)
    if network_dir is not None:
        evaluate_parent(output, network_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--network-dir", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if not args.job_dir or not args.baseline_dir:
        parser.error("--job-dir and --baseline-dir are required")
    result = run(args.job_dir, args.baseline_dir, args.output_dir or DEFAULT_OUTPUT/args.job_dir.name,
                 runtime_dir=args.runtime_dir, network_dir=args.network_dir)
    print(json.dumps(result))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
