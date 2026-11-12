"""Run the 64 paired, training-only NJ/RNJ/WZZT Source comparisons.

Workers receive copied observations and known source schedules only.  The
parent evaluates saved models against network truth after each worker ends.
Every scheduled result, including failures and interrupted searches, survives.
"""
from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter, sleep
import traceback

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rnj_wzzt_core"))
import numpy as np
import pandas as pd

CASES = ("paper15", "soumalas11", "flynn16", "pengwah18")
VARIANTS = ("no_source", "constant", "intermittent", "smooth")
METHODS = ("NJ", "RNJ", "WZZT", "RNJ+WZZT")
ACTIVE_TOLERANCE = 1e-8


def plain(value):
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return plain(sorted(value, key=lambda item: str(item)))
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def job_catalog():
    return [{"case": case, "variant": variant, "method": method,
             "job_id": f"{case}__{variant}__{method}"}
            for case in CASES for variant in VARIANTS for method in METHODS]


def input_arrays(scenarios, source_inputs):
    scenario, source = scenarios[0], source_inputs[0]
    return {"P": scenario["P_terminal"].to_numpy(dtype=float),
            "Q": scenario["Q_terminal"].to_numpy(dtype=float),
            "Y": scenario["drop_target"].to_numpy(dtype=float),
            "P_source": source.p_pu.to_numpy(dtype=float),
            "Q_source": source.q_pu.to_numpy(dtype=float)}


def tree_model(fit):
    g = np.asarray(fit["source_response_g"])
    return {"terminal_labels": list(fit["terminals"]),
            "support_indices": None, "r_values": None, "x_values": None,
            "source_h": None, "source_g_r": g[0], "source_g_x": g[1],
            "r_upper_bound": None, "x_upper_bound": None,
            "clades": [sorted(clade) for clade in sorted(fit["clades"], key=lambda c: (len(c), sorted(c)))],
            "matrix_origin": "source_adjusted_ordered_regression",
            "solver_events": [], "regression_diagnostics": fit["diagnostics"],
            "tree": fit["tree"], "timing": fit["timing"],
            "R": fit["R"], "X": fit["X"], "status": "complete",
            "stop_reason": "tree_inference_completed"}


def wzzt_model(fit):
    labels = list(fit["labels"])
    clades = []
    if fit["r_weights"] is not None:
        for support, r, x in zip(fit["supports"], fit["r_weights"], fit["x_weights"]):
            if 1 < len(support) < len(labels) and max(r, x) > ACTIVE_TOLERANCE:
                clades.append(sorted(labels[index] for index in support))
    events = []
    for attempt in fit["attempts"]:
        event = dict(attempt.get("diagnostics", {}))
        event.update({key: attempt.get(key) for key in
                      ("kind", "accepted", "optimality_certified", "gain", "error", "time_limit_seconds")})
        event["incumbent_audit"] = attempt.get("incumbent")
        events.append(event)
    return {"terminal_labels": labels, "support_indices": fit["supports"],
            "r_values": fit["r_weights"], "x_values": fit["x_weights"],
            "source_h": fit["h"], "source_g_r": fit["source_gR"],
            "source_g_x": fit["source_gX"], "train_mae": fit["train_mae"],
            "r_upper_bound": fit["coefficient_bound"], "x_upper_bound": fit["coefficient_bound"],
            "clades": clades, "matrix_origin": "last_certified_laminar_atoms",
            "solver_events": events, "R": fit["R"], "X": fit["X"],
            "status": fit["status"], "stop_reason": fit["stop_reason"],
            "coefficient_bound_active": fit["coefficient_bound_active"],
            "initial_supports": fit["initial_supports"], "path": fit["path"],
            "certificate_scope": fit["certificate_scope"]}


def save_model(directory, model, arrays, method_started):
    model = dict(model)
    r, x = model.pop("R", None), model.pop("X", None)
    model["has_model"] = r is not None and x is not None
    model["elapsed_seconds"] = perf_counter() - method_started
    model["method_started_monotonic"] = method_started
    if model["has_model"]:
        r, x = np.asarray(r, dtype=float), np.asarray(x, dtype=float)
        g_r, g_x = np.asarray(model["source_g_r"]), np.asarray(model["source_g_x"])
        prediction = arrays["P"] @ r.T + arrays["Q"] @ x.T
        prediction += arrays["P_source"][:, None] * g_r + arrays["Q_source"][:, None] * g_x
        model["train_mae"] = float(np.mean(np.abs(prediction - arrays["Y"])))
    else:
        r = x = np.empty((0, 0))
    temporary = directory / "model.tmp.npz"
    np.savez_compressed(temporary, R=r, X=x, **arrays)
    os.replace(temporary, directory / "model.npz")
    write_json(directory / "model.json", model)
    return model


def worker(payload_path):
    payload = read_json(payload_path)
    job, config = payload["job"], payload["config"]
    directory = Path(payload["job_dir"])
    from research_experiments.rnj.prepare_source_ac96 import load_algorithm_inputs
    from research_experiments.rnj.source_ac96_trees import fit_tree, select_rnj_initial_supports
    from research_experiments.rnj.source_ac96_wzzt import fit_wzzt
    scenarios, sources = load_algorithm_inputs(payload["input_dir"])
    arrays = input_arrays(scenarios, sources)
    started = perf_counter()
    write_json(directory / "execution.json", {"pid": os.getpid(), "method_started_monotonic": started,
                                              "started_utc": datetime.now(timezone.utc).isoformat()})
    selection = None
    try:
        if job["method"] in ("NJ", "RNJ"):
            fit = fit_tree(scenarios, sources, job["method"], root_bus=config["observed_root_label"])
            model = save_model(directory, tree_model(fit), arrays, started)
        else:
            initial = None
            if job["method"] == "RNJ+WZZT":
                selection = select_rnj_initial_supports(
                    scenarios, sources, replicates=config["bootstrap_replicates"],
                    block_length=config["bootstrap_block_length"], threshold=config["bootstrap_threshold"],
                    max_blocks=config["maximum_frozen_blocks"], seed=config["bootstrap_seed"],
                    root_bus=config["observed_root_label"], wall_seconds=config["method_wall_seconds"],
                )
                selection_record = {key: value for key, value in selection.items() if key != "fit"}
                selection_record["full_fit"] = tree_model(selection["fit"])
                write_json(directory / "rnj_screen.json", selection_record)
                if selection["status"] != "complete":
                    raise TimeoutError("bootstrap_time_limit_before_WZZT")
                initial = selection["selected_support_indices"]
            remaining = config["method_wall_seconds"] - (perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("method_wall_limit_before_WZZT")
            def checkpoint(fit):
                model = wzzt_model(fit)
                if selection is not None:
                    model["bootstrap"] = {key: value for key, value in selection.items() if key != "fit"}
                save_model(directory, model, arrays, started)
            fit = fit_wzzt(scenarios, sources, initial,
                           per_solve_seconds=config["per_solve_seconds"], wall_seconds=remaining,
                           coefficient_bound=config["coefficient_bound"], solver=config["solver"],
                           checkpoint=checkpoint)
            model = wzzt_model(fit)
            if selection is not None:
                model["bootstrap"] = {key: value for key, value in selection.items() if key != "fit"}
            model = save_model(directory, model, arrays, started)
        result = {**job, "status": model["status"], "stop_reason": model["stop_reason"],
                  "search_complete": model["status"] == "complete", "has_model": model["has_model"],
                  "elapsed_seconds": perf_counter() - started}
    except Exception as error:
        result = {**job, "status": "incomplete" if isinstance(error, TimeoutError) else "failed",
                  "stop_reason": str(error), "error_type": type(error).__name__,
                  "traceback": traceback.format_exc(), "search_complete": False,
                  "has_model": False, "elapsed_seconds": perf_counter() - started}
        if (directory / "model.json").exists():
            result["has_model"] = read_json(directory / "model.json")["has_model"]
        else:
            save_model(directory, {"terminal_labels": list(scenarios[0]["P_terminal"].columns),
                       "status": result["status"], "stop_reason": result["stop_reason"],
                       "R": None, "X": None, "clades": [], "solver_events": [],
                       "source_g_r": None, "source_g_x": None,
                       "matrix_origin": None}, arrays, started)
    result["protocol_sha256"] = payload["protocol_sha256"]
    write_json(directory / "result.json", result)
    print(json.dumps({key: result[key] for key in ("job_id", "status", "stop_reason", "elapsed_seconds")}))
    return 0


def evaluate_result(directory, dataset_dir):
    """Parent-only scoring; physical network is never passed to an estimator."""
    from rnj_wzzt.models.network import TerminalizedNetwork
    from rnj_wzzt.models.lin_distflow import build_reduced_sensitivity_matrices
    from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
    result = read_json(directory / "result.json")
    metrics = dict.fromkeys(("clade_f1", "precision", "recall", "r_relative_error", "x_relative_error"))
    model = read_json(directory / "model.json") if (directory / "model.json").exists() else {}
    if model.get("has_model"):
        path = dataset_dir / result["case"] / "network"
        metadata = read_json(path / "metadata.json")
        net = TerminalizedNetwork(pd.read_csv(path / "buses.csv"), pd.read_csv(path / "branches.csv"),
                                  int(metadata["root_bus"]), metadata["base_kv"], metadata["base_mva"], {}, {}, {})
        labels = [int(label) for label in model["terminal_labels"]]
        true_r, true_x = build_reduced_sensitivity_matrices(net, labels, voltage_model="squared-voltage")
        edges = [(row.parent, row.child) for row in net.orient_from_root().itertuples()]
        truth = rooted_clades(edges, net.root_bus, labels)
        predicted = {frozenset(map(int, item)) for item in model["clades"]}
        correct = len(truth & predicted)
        precision = correct / len(predicted) if predicted else float(not truth)
        recall = correct / len(truth) if truth else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        with np.load(directory / "model.npz") as arrays:
            metrics.update(clade_f1=f1, precision=precision, recall=recall,
                           r_relative_error=float(np.linalg.norm(arrays["R"] - true_r) / np.linalg.norm(true_r)),
                           x_relative_error=float(np.linalg.norm(arrays["X"] - true_x) / np.linalg.norm(true_x)))
        result["evaluation"] = {"truth_clades": [sorted(item) for item in truth],
                                "predicted_clades": [sorted(item) for item in predicted],
                                "matched_clade_count": correct, "true_clade_count": len(truth),
                                "predicted_clade_count": len(predicted), "truth_read_by": "parent_evaluator_only"}
    result["metrics"] = metrics
    events = model.get("solver_events", [])
    result["solver_time_limit_count"] = sum(event.get("status") == 1 for event in events)
    result["uncertified_solve_count"] = sum(not event.get("optimality_certified", False) for event in events)
    result["matrix_origin"] = model.get("matrix_origin")
    result["bootstrap_failed"] = len(model.get("bootstrap", {}).get("bootstrap_failures", []))
    result["bootstrap_unattempted"] = model.get("bootstrap", {}).get("bootstrap_unattempted", 0)
    result["compute_seconds"] = result["elapsed_seconds"]
    write_json(directory / "result.json", result)
    return result


def summarize(output_dir):
    rows = []
    for job in job_catalog():
        path = output_dir / "jobs" / job["job_id"] / "result.json"
        if path.exists():
            result = read_json(path)
            row = {**job, **{key: result.get(key) for key in
                   ("status", "stop_reason", "search_complete", "has_model", "compute_seconds",
                    "solver_time_limit_count", "uncertified_solve_count", "bootstrap_failed",
                    "bootstrap_unattempted", "matrix_origin")}, **result.get("metrics", {})}
        else:
            row = {**job, "status": "not_run"}
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "summary.csv", index=False, float_format="%.17g")
    counts = dict(Counter(row["status"] for row in rows))
    write_json(output_dir / "completion.json", {"expected": 64, "status_counts": counts,
               "finished": 64 - counts.get("not_run", 0),
               "models_available": sum(bool(row.get("has_model")) for row in rows)})
    return rows


def make_protocol(args):
    source_files = [Path(__file__), ROOT / "research_experiments/rnj/source_ac96_trees.py",
                    ROOT / "research_experiments/rnj/source_ac96_wzzt.py"]
    source_files += sorted((ROOT / "rnj_wzzt_core/rnj_wzzt").rglob("*.py"))
    source_files += [ROOT / "research_experiments/rnj/rooted_study_baselines.py",
                     ROOT / "research_experiments/rnj/sensitivity_geometry.py"]
    return {"schema_version": 1, "cases": CASES, "variants": VARIANTS, "methods": METHODS,
            "expected_jobs": 64, "samples_per_fit": 96, "validation_samples": 0,
            "selection": "predeclared thresholds and last certified positive training gain",
            "source_information": "known exact P/Q time series, unknown non-root position",
            "tree_source_adapter": "FWL unconstrained terminal source responses; no location oracle",
            "tree_matrix_metric": "ordered regression before NJ/RNJ inference",
            "wzzt_source_adapter": "joint physical atom-chain h, all coefficients refitted each step",
            "observed_root_label": 1, "root_information": "observed label and noisy root voltage",
            "rx75_weight": .75, "tree_tolerance_factor": .16, "bootstrap_replicates": 100,
            "bootstrap_block_length": 4, "bootstrap_threshold": .75, "maximum_frozen_blocks": 2,
            "bootstrap_seed": 20261018, "per_solve_seconds": args.per_solve_seconds,
            "method_wall_seconds": args.method_wall_seconds, "hard_timeout_grace_seconds": 30,
            "solver": args.solver, "coefficient_bound": 2.0, "active_atom_tolerance": ACTIVE_TOLERANCE,
            "concurrency": args.workers, "solver_threads": 1, "blas_threads": 1,
            "dataset_manifest_sha256": sha256(args.dataset_dir / "manifest.json"),
            "source_sha256": {str(path.relative_to(ROOT).as_posix()): sha256(path) for path in source_files}}


def run(args):
    if not all(np.isfinite(value) and value > 0 for value in
               (args.per_solve_seconds, args.method_wall_seconds)):
        raise ValueError("Solver and method budgets must be positive and finite")
    output, dataset = args.output_dir.resolve(), args.dataset_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = plain(make_protocol(args))
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if read_json(protocol_path) != protocol:
            raise ValueError("Protocol or source hash changed; preserve this run and choose a new output directory")
    else:
        write_json(protocol_path, protocol)
    protocol_hash = sha256(protocol_path)
    if not (output / "environment.json").exists():
        versions = {}
        for name in ("numpy", "pandas", "scipy", "networkx", "gurobipy"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        write_json(output / "environment.json", {
            "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "packages": versions,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "threads": {name: os.environ[name] for name in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        })
    pending = []
    for job in job_catalog():
        directory = output / "jobs" / job["job_id"]
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / "result.json"
        if result_path.exists():
            if read_json(result_path).get("protocol_sha256") != protocol_hash:
                raise ValueError("Job protocol signature mismatch")
            evaluate_result(directory, dataset)
            continue
        input_dir = output / "inputs" / f"{job['case']}__{job['variant']}"
        file_hashes = {}
        for folder in ("observed", "source"):
            origin = dataset / job["case"] / job["variant"] / folder
            for source_path in sorted(origin.iterdir()):
                if not source_path.is_file():
                    continue
                target = input_dir / folder / source_path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(source_path, target)
                if sha256(target) != sha256(source_path):
                    raise ValueError("Copied estimator input differs from frozen source data")
                file_hashes[f"{folder}/{source_path.name}"] = sha256(target)
        write_json(directory / "inputs_hash.json", {"file_sha256": file_hashes})
        write_json(directory / "payload.json", {"job": job, "config": protocol, "job_dir": directory,
                   "input_dir": input_dir, "protocol_sha256": protocol_hash})
        pending.append((job, directory))
    if args.only_job:
        known = {job["job_id"] for job in job_catalog()}
        if not set(args.only_job) <= known:
            raise ValueError("Unknown requested job id")
        pending = [item for item in pending if item[0]["job_id"] in args.only_job]
    summarize(output)
    active = []
    while pending or active:
        while pending and len(active) < args.workers:
            job, directory = pending.pop(0)
            stdout = (directory / "stdout.log").open("a", encoding="utf-8")
            stderr = (directory / "stderr.log").open("a", encoding="utf-8")
            command = [sys.executable, "-B", "-m", "research_experiments.rnj.run_source_ac96_benchmark",
                       "--worker", str(directory / "payload.json")]
            process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=stdout, stderr=stderr)
            active.append((job, directory, process, perf_counter(), stdout, stderr))
            print(f"START {job['job_id']} pid={process.pid}", flush=True)
        for item in list(active):
            job, directory, process, launched, stdout, stderr = item
            hard_limit = protocol["method_wall_seconds"] + protocol["hard_timeout_grace_seconds"]
            killed = process.poll() is None and perf_counter() - launched > hard_limit
            if killed:
                process.kill()
                process.wait()
            if process.poll() is None:
                continue
            stdout.close()
            stderr.close()
            if not (directory / "result.json").exists():
                model = read_json(directory / "model.json") if (directory / "model.json").exists() else {}
                execution = read_json(directory / "execution.json") if (directory / "execution.json").exists() else {}
                started = execution.get("method_started_monotonic", launched)
                result = {**job, "status": "incomplete" if killed else "failed",
                          "stop_reason": "process_wall_limit" if killed else "worker_process_failed",
                          "search_complete": False, "has_model": model.get("has_model", False),
                          "elapsed_seconds": perf_counter() - started, "process_exitcode": process.returncode,
                          "protocol_sha256": protocol_hash}
                if model:
                    model.update(status=result["status"], stop_reason=result["stop_reason"])
                    write_json(directory / "model.json", model)
                write_json(directory / "result.json", result)
            result = evaluate_result(directory, dataset)
            print(f"END {job['job_id']} {result['status']} F1={result['metrics']['clade_f1']} "
                  f"seconds={result['compute_seconds']:.2f} {result['stop_reason']}", flush=True)
            active.remove(item)
            summarize(output)
        if active:
            sleep(.5)
    write_json(output / "execution_complete.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
               "protocol_sha256": protocol_hash, "completion": read_json(output / "completion.json")})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "outputs/source_ac96_20261018")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/source_ac96_benchmark_20261018")
    parser.add_argument("--solver", choices=("gurobi", "highs"), default="gurobi")
    parser.add_argument("--per-solve-seconds", type=float, default=60)
    parser.add_argument("--method-wall-seconds", type=float, default=300)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--only-job", nargs="+", help="Integration/resume subset; all 64 denominators remain recorded")
    args = parser.parse_args()
    return worker(args.worker) if args.worker else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
