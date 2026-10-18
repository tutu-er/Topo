"""Pengwah18 no-source: full mainline WZZT on one 64/32 AC day split.

Six declared jobs compare NJ/RNJ and complete core WZZT paths at original
60/300 second and 100-times 6000/30000 second solver/method budgets. Workers
see copied observations and zero source schedules; only the parent evaluates
physical truth. This driver never generates an additional observation day.
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
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter, sleep
import traceback

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rnj_wzzt_core"))
import numpy as np
import pandas as pd

CASE, VARIANT = "pengwah18", "no_source"
ACTIVE_TOLERANCE = 1e-8
CORE_PARAMETERS = {
    "max_atoms": 35, "r_upper_bound": 2.0, "x_upper_bound": 2.0,
    "bound_safety_factor": 2.0, "bound_expansion_factor": 2.0,
    "max_bound_expansions": 4, "bound_active_fraction": .995,
    "improvement_abs_tol": 1e-10, "improvement_rel_tol": 1e-8,
    "zero_tolerance": 1e-9, "validation_blocks": 4,
    "mip_rel_gap": 0.0, "presolve": True, "disp": False,
}


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
    temporary.write_text(json.dumps(plain(value), indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def job_catalog():
    jobs = []
    for method in ("NJ", "RNJ"):
        jobs.append({"case": CASE, "variant": VARIANT, "method": method,
                     "budget_multiplier": None, "per_solve_seconds": None,
                     "method_wall_seconds": 300.0,
                     "job_id": f"{CASE}__{VARIANT}__{method}"})
    for multiplier in (1, 100):
        for method in ("WZZT", "RNJ+WZZT"):
            jobs.append({"case": CASE, "variant": VARIANT, "method": method,
                         "budget_multiplier": multiplier,
                         "per_solve_seconds": 60.0 * multiplier,
                         "method_wall_seconds": 300.0 * multiplier,
                         "job_id": f"{CASE}__{VARIANT}__{method}__t{multiplier}"})
    return jobs


def split_day(scenarios, sources):
    """Slice actual input frames, preserving labels; never generate new rows."""
    from rnj_wzzt.estimation.unmetered_inputs import SourceInputs
    if len(scenarios) != 1 or len(sources) != 1:
        raise ValueError("full-history protocol requires exactly one 96-point day")
    scenario, source = scenarios[0], sources[0]
    reference = scenario["P_terminal"]
    if len(reference) != 96 or not reference.index.is_unique or not reference.columns.is_unique:
        raise ValueError("expected exactly 96 uniquely indexed observations")
    if not source.p_pu.index.equals(reference.index) or not source.q_pu.index.equals(reference.index):
        raise ValueError("source and observations must be aligned")
    if np.any(source.p_pu.to_numpy() != 0) or np.any(source.q_pu.to_numpy() != 0):
        raise ValueError("this protocol is no_source only")
    def subset(rows, name):
        return {"name": name, **{key: scenario[key].iloc[rows].copy()
                                 for key in ("P_terminal", "Q_terminal", "drop_target")}}
    train = [subset(slice(0, 64), "pengwah18_no_source_training64")]
    validation = [subset(slice(64, 96), "pengwah18_no_source_validation32")]
    training_source = [SourceInputs(source.p_pu.iloc[:64].copy(), source.q_pu.iloc[:64].copy(),
                                   source.provenance)]
    arrays = {}
    for suffix, block in (("all", scenario), ("train", train[0]), ("validation", validation[0])):
        for field, key in (("P", "P_terminal"), ("Q", "Q_terminal"), ("Y", "drop_target")):
            arrays[f"{field}_{suffix}"] = block[key].to_numpy(dtype=float)
    arrays.update(P_source_all=source.p_pu.to_numpy(dtype=float),
                  Q_source_all=source.q_pu.to_numpy(dtype=float),
                  train_positions=np.arange(64), validation_positions=np.arange(64, 96))
    split = {"total_samples": 96, "training_samples": 64, "validation_samples": 32,
             "training_positions": list(range(64)), "validation_positions": list(range(64, 96)),
             "all_time_labels": reference.index.tolist(),
             "training_time_labels": reference.index[:64].tolist(),
             "validation_time_labels": reference.index[64:].tolist(),
             "terminal_labels": reference.columns.tolist(),
             "ordering": "original observation order; first 64 train, last 32 validation",
             "validation_used_for_rnj_selection": False, "new_observations_generated": False}
    return train, validation, training_source, arrays, split


def select_snapshot_model(state, *, solver_events, progress_events, status, stop_reason):
    """Select exactly core's 1SE point, including after an external deadline."""
    from rnj_wzzt.estimation.laminar_l1_milp import _choose_path_point
    path = state.get("path", ())
    index = _choose_path_point(path) if path else None
    selected = path[index] if index is not None else None
    labels = list(state.get("terminal_labels", ()))
    family, active = [], []
    if selected is not None:
        for support, r, x in zip(selected.support_labels, selected.r_values, selected.x_values):
            if 1 < len(support) < len(labels):
                family.append(list(support))
                if max(r, x) > ACTIVE_TOLERANCE:
                    active.append(list(support))
    return {
        "schema_version": 2, "terminal_labels": labels,
        "support_indices": None if selected is None else selected.support_indices,
        "support_labels": None if selected is None else selected.support_labels,
        "r_values": None if selected is None else selected.r_values,
        "x_values": None if selected is None else selected.x_values,
        "R": None if selected is None else selected.r_matrix,
        "X": None if selected is None else selected.x_matrix,
        "train_mae": None if selected is None else selected.train_mae,
        "validation_mae": None if selected is None else selected.validation_mae,
        "validation_se": None if selected is None else selected.validation_se,
        "selected_path_index": index, "path": path,
        "attempted_extensions": state.get("attempted_extensions", ()),
        "r_upper_bound": state.get("r_upper_bound"), "x_upper_bound": state.get("x_upper_bound"),
        "bound_expansions": state.get("bound_expansions", 0),
        "initial_supports": state.get("initial_supports", ()),
        "family_clades": family, "active_clades": active, "clades": family,
        "active_atom_tolerance": ACTIVE_TOLERANCE,
        "solver_events": list(solver_events), "progress_events": list(progress_events),
        "status": status, "stop_reason": stop_reason,
        "matrix_origin": "full_core_validation_1se_selected_laminar_atoms",
        "selection": "minimum_atom_count_within_best_validation_mae_plus_its_block_se",
        "certificate_scope": "bounded one-atom subproblems; whole path is greedy",
    }


def save_model(directory, model, arrays, started):
    model = dict(model)
    r, x = model.pop("R", None), model.pop("X", None)
    model["has_model"] = r is not None and x is not None
    model["elapsed_seconds"] = perf_counter() - started
    if model["has_model"]:
        for suffix, label in (("train", "train"), ("validation", "validation")):
            prediction = arrays[f"P_{suffix}"] @ np.asarray(r).T + arrays[f"Q_{suffix}"] @ np.asarray(x).T
            model[f"{label}_mae_recomputed"] = float(np.mean(abs(arrays[f"Y_{suffix}"] - prediction)))
    temporary = directory / "model.tmp.npz"
    np.savez_compressed(temporary, R=np.empty((0, 0)) if r is None else r,
                        X=np.empty((0, 0)) if x is None else x, **arrays)
    os.replace(temporary, directory / "model.npz")
    write_json(directory / "model.json", model)
    return model


class MethodBudgetExhausted(TimeoutError):
    pass


class BudgetedSolver:
    """Observe unchanged backend results and cap each solve by remaining wall."""
    def __init__(self, core, directory, started, job):
        self.core, self.directory, self.started, self.job = core, Path(directory), started, job
        self.events = []

    def __enter__(self):
        self.original = self.core._run_milp
        self.core._run_milp = self.solve
        return self

    def __exit__(self, *exc):
        self.core._run_milp = self.original

    def solve(self, variables, rows, **kwargs):
        remaining = self.job["method_wall_seconds"] - (perf_counter() - self.started)
        if remaining <= 0:
            raise MethodBudgetExhausted("method_wall_limit_before_next_solve")
        requested = kwargs["time_limit"]
        effective = min(float(requested), remaining)
        actual = dict(kwargs, time_limit=effective)
        result, diagnostics = self.original(variables, rows, **actual)
        sequence = len(self.events) + 1
        primal = getattr(result, "x", None)
        primal_file = None
        if primal is not None:
            target = self.directory / "solver" / f"{sequence:06d}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, primal=np.asarray(primal, dtype=float))
            primal_file = str(target.relative_to(self.directory).as_posix())
        event = {**diagnostics.to_dict(), "sequence": sequence,
                 "requested_time_limit_seconds": requested, "time_limit_seconds": effective,
                 "remaining_wall_before_solve": remaining,
                 "optimality_certified": self.core.solver_diagnostics_prove_optimality(diagnostics),
                 "has_incumbent": primal is not None, "primal_file": primal_file,
                 "variable_slices": {name: [value.start, value.stop]
                                     for name, value in variables.slices.items()},
                 "elapsed_seconds": perf_counter() - self.started}
        self.events.append(event)
        write_json(self.directory / "solver" / f"{sequence:06d}.json", event)
        write_json(self.directory / "solver_events.json", self.events)
        return result, diagnostics


def _complete_stop(reason):
    return reason in {"max_atoms", "laminar_family_limit", "no_feasible_extension",
                      "no_significant_one_atom_gain", "no_significant_one_atom_gain_after_pruning",
                      "no_significant_one_atom_gain_certified_by_dual_bound",
                      "repeated_family_after_zero_pruning"}


def worker(payload_path):
    payload = read_json(payload_path)
    directory, job, config = Path(payload["job_dir"]), payload["job"], payload["config"]
    from research_experiments.rnj.prepare_source_ac96 import load_algorithm_inputs
    from research_experiments.rnj.source_ac96_trees import fit_tree
    from rnj_wzzt.estimation import laminar_l1_milp as core
    from rnj_wzzt.pipeline import _select_boundary_blocks
    scenarios, sources = load_algorithm_inputs(payload["input_dir"])
    train, validation, train_sources, arrays, split = split_day(scenarios, sources)
    write_json(directory / "split.json", split)
    started = perf_counter()
    write_json(directory / "execution.json", {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
               "method_started_monotonic": started, "job": job})
    state = {"terminal_labels": split["terminal_labels"], "path": (), "attempted_extensions": (),
             "initial_supports": (), "r_upper_bound": 2., "x_upper_bound": 2., "bound_expansions": 0}
    progress, selection, model, events = [], None, None, []
    status, reason, error = "failed", "not_started", None
    try:
        if job["method"] in ("NJ", "RNJ"):
            fit = fit_tree(train, train_sources, job["method"], root_bus=config["observed_root_label"])
            model = {"schema_version": 2, "terminal_labels": fit["terminals"], "R": fit["R"], "X": fit["X"],
                     "family_clades": fit["clades"], "active_clades": fit["clades"], "clades": fit["clades"],
                     "support_indices": None, "r_values": None, "x_values": None, "initial_supports": [],
                     "selected_path_index": None, "path": [], "attempted_extensions": [], "solver_events": [],
                     "progress_events": [], "r_upper_bound": None, "x_upper_bound": None, "bound_expansions": 0,
                     "regression_diagnostics": fit["diagnostics"], "tree": fit["tree"], "timing": fit["timing"],
                     "matrix_origin": "training64_ordered_regression_before_tree_inference",
                     "status": "complete", "stop_reason": "tree_inference_completed"}
            status, reason = "complete", "tree_inference_completed"
        else:
            labels = [int(value) for value in split["terminal_labels"]]
            initial = [(index,) for index in range(len(labels))]
            if job["method"] == "RNJ+WZZT":
                screening_started = perf_counter()
                full, confidence, selected, r2, condition = _select_boundary_blocks(
                    train, labels, config["observed_root_label"],
                    bootstrap_replicates=config["bootstrap_replicates"],
                    block_length=config["bootstrap_block_length"],
                    confidence_threshold=config["bootstrap_threshold"],
                    maximum_candidate_count=config["maximum_frozen_blocks"],
                    tolerance_factor=config["tree_tolerance_factor"], seed=config["bootstrap_seed"],
                )
                positions = {label: index for index, label in enumerate(labels)}
                selected_supports = [tuple(positions[label] for label in sorted(clade)) for clade in selected]
                initial += selected_supports
                selection = {"selector": "rnj_wzzt.pipeline._select_boundary_blocks", "full_clades": full,
                             "confidence_records": [{"clade": sorted(c), "frequency": f}
                                                    for c, f in sorted(confidence.items(), key=lambda item: tuple(sorted(item[0])))],
                             "selected_support_labels": [sorted(c) for c in selected],
                             "selected_support_indices": selected_supports, "r2_score": r2, "condition": condition,
                             "bootstrap_replicates": config["bootstrap_replicates"], "bootstrap_seed": config["bootstrap_seed"],
                             "training_sample_count": 64, "validation_used": False,
                             "elapsed_seconds": perf_counter() - screening_started, "status": "complete"}
                write_json(directory / "rnj_screen.json", selection)
            state["initial_supports"] = initial
            with BudgetedSolver(core, directory, started, job) as observer:
                events = observer.events
                def checkpoint(snapshot):
                    nonlocal state
                    state = snapshot
                    sequence = len(progress) + 1
                    write_json(directory / "progress" / f"{sequence:06d}.json", snapshot)
                    progress.append({"sequence": sequence, "event": snapshot["event"],
                                     "path_length": len(snapshot["path"]), "attempt_count": len(snapshot["attempted_extensions"]),
                                     "bound_expansions": snapshot["bound_expansions"],
                                     "r_upper_bound": snapshot["r_upper_bound"], "x_upper_bound": snapshot["x_upper_bound"],
                                     "elapsed_seconds": perf_counter() - started})
                    partial = select_snapshot_model(state, solver_events=events, progress_events=progress,
                                                    status="running", stop_reason="in_progress")
                    partial["bootstrap"] = selection
                    save_model(directory, partial, arrays, started)
                fit = core.fit_laminar_l1_sensitivity(
                    train, validation_scenarios=validation, initial_supports=initial,
                    **config["core_parameters"], time_limit=job["per_solve_seconds"],
                    solver=config["solver"], progress_callback=checkpoint,
                )
                reason = fit.stop_reason
                status = "complete" if _complete_stop(reason) else "incomplete"
                model = select_snapshot_model(state, solver_events=events, progress_events=progress,
                                              status=status, stop_reason=reason)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        timed_out = isinstance(exc, TimeoutError) or (events and events[-1]["status"] == 1)
        status, reason = ("incomplete" if timed_out else "failed"), str(exc)
        model = select_snapshot_model(state, solver_events=events, progress_events=progress,
                                      status=status, stop_reason=reason)
        write_json(directory / "error.json", error)
    model["bootstrap"] = selection
    model["core_parameters"] = config["core_parameters"] if "WZZT" in job["method"] else None
    model["budget"] = job
    saved = save_model(directory, model, arrays, started)
    write_json(directory / "result.json", {**job, "status": status, "stop_reason": reason,
               "search_complete": status == "complete", "has_model": saved["has_model"],
               "elapsed_seconds": perf_counter() - started, "protocol_sha256": payload["protocol_sha256"],
               "error": error, "truth_read_by": "parent_evaluator_only"})
    return 0


def evaluate_result(directory, dataset):
    # This function is called only in the parent after the worker exits.
    from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth, clade_metrics, relative_error
    result, model = read_json(directory / "result.json"), read_json(directory / "model.json")
    metrics = {}
    if model["has_model"]:
        truth = physical_truth(dataset / CASE / "network", model["terminal_labels"])
        family = clade_metrics(model["family_clades"], truth["clades"], model["terminal_labels"])
        active = clade_metrics(model["active_clades"], truth["clades"], model["terminal_labels"])
        for prefix, values in (("family", family), ("active", active)):
            for key in ("clade_f1", "precision", "recall"):
                metrics[f"{prefix}_{key}"] = values[key]
        with np.load(directory / "model.npz") as arrays:
            metrics.update(r_relative_error=relative_error(arrays["R"], truth["R"]),
                           x_relative_error=relative_error(arrays["X"], truth["X"]))
        metrics["clade_f1"] = metrics["family_clade_f1"]
        result["evaluation"] = {"truth_clades": [sorted(c) for c in truth["clades"]],
                                "family_counts": family, "active_counts": active,
                                "truth_read_by": "parent_evaluator_only"}
    result.update(metrics=metrics, compute_seconds=result["elapsed_seconds"],
                  matrix_origin=model.get("matrix_origin"), selected_path_index=model.get("selected_path_index"),
                  path_length=len(model.get("path", [])), bound_expansions=model.get("bound_expansions"),
                  solver_time_limit_count=sum(e.get("status") == 1 for e in model.get("solver_events", [])),
                  uncertified_solve_count=sum(not e.get("optimality_certified", False) for e in model.get("solver_events", [])))
    write_json(directory / "result.json", result)
    return result


def summarize(output):
    rows = []
    for job in job_catalog():
        path = output / "jobs" / job["job_id"] / "result.json"
        if path.exists():
            d = read_json(path)
            rows.append({**job, **{key: d.get(key) for key in ("status", "stop_reason", "search_complete", "has_model",
                         "compute_seconds", "path_length", "selected_path_index", "bound_expansions",
                         "solver_time_limit_count", "uncertified_solve_count")}, **d.get("metrics", {})})
        else:
            rows.append({**job, "status": "not_run"})
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False, float_format="%.17g")
    counts = dict(Counter(row["status"] for row in rows))
    write_json(output / "completion.json", {"expected": 6, "status_counts": counts,
               "finished": 6 - counts.get("not_run", 0), "models_available": sum(bool(r.get("has_model")) for r in rows)})


def make_protocol(args):
    paths = [Path(__file__), ROOT / "research_experiments/rnj/source_ac96_trees.py",
             ROOT / "research_experiments/rnj/prepare_source_ac96.py",
             ROOT / "research_experiments/rnj/rooted_study_baselines.py",
             ROOT / "research_experiments/rnj/sensitivity_geometry.py"]
    paths += sorted((ROOT / "rnj_wzzt_core/rnj_wzzt").rglob("*.py"))
    return plain({"schema_version": 2, "case": CASE, "variant": VARIANT, "jobs": job_catalog(),
                  "expected_jobs": 6, "total_samples": 96, "training_samples": 64, "validation_samples": 32,
                  "training_positions": list(range(64)), "validation_positions": list(range(64, 96)),
                  "observed_root_label": 1, "source_information": "explicit zero schedules; no-source only",
                  "initialization": "all terminal singletons plus selected RNJ boundary blocks for hybrid",
                  "core_entrypoint": "rnj_wzzt.estimation.laminar_l1_milp.fit_laminar_l1_sensitivity",
                  "core_parameters": CORE_PARAMETERS, "solver": args.solver,
                  "rx75_weight": .75, "tree_tolerance_factor": .16,
                  "bootstrap_replicates": 100, "bootstrap_block_length": 4, "bootstrap_threshold": .75,
                  "maximum_frozen_blocks": 2, "bootstrap_seed": 20260903,
                  "bootstrap_selector": "rnj_wzzt.pipeline._select_boundary_blocks",
                  "validation_used_for_bootstrap": False, "solver_threads": 1, "blas_threads": 1,
                  "concurrency": args.workers, "hard_timeout_grace_seconds": 30,
                  "family_score": "all nontrivial selected supports, matching pipeline",
                  "active_score": "nontrivial selected supports with max(R_atom,X_atom)>1e-8",
                  "active_atom_tolerance": ACTIVE_TOLERANCE,
                  "dataset_manifest_sha256": sha256(args.dataset_dir / "manifest.json"),
                  "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}})


def prepare(args):
    output, dataset = args.output_dir.resolve(), args.dataset_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = make_protocol(args)
    if (output / "protocol.json").exists() and read_json(output / "protocol.json") != protocol:
        raise ValueError("Frozen protocol/source changed; preserve existing run and use a new directory")
    write_json(output / "protocol.json", protocol)
    signature = sha256(output / "protocol.json")
    source_snapshot = output / "source_snapshot"
    for relative, checksum in protocol["source_sha256"].items():
        target = source_snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(ROOT / relative, target)
        if sha256(target) != checksum:
            raise ValueError("Frozen source snapshot mismatch")
    copied = output / "inputs" / f"{CASE}__{VARIANT}"
    hashes = {}
    for folder in ("observed", "source"):
        for source_path in sorted((dataset / CASE / VARIANT / folder).iterdir()):
            if not source_path.is_file():
                continue
            target = copied / folder / source_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source_path, target)
            if sha256(target) != sha256(source_path):
                raise ValueError("Copied observation/source differs from original")
            hashes[f"{folder}/{source_path.name}"] = sha256(target)
    versions = {}
    for name in ("numpy", "pandas", "scipy", "networkx", "gurobipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if not (output / "environment.json").exists():
        write_json(output / "environment.json", {"python": sys.version, "executable": sys.executable,
                   "platform": platform.platform(), "packages": versions,
                   "threads": {key: os.environ[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                   "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    pending = []
    for job in job_catalog():
        directory = output / "jobs" / job["job_id"]
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "inputs_hash.json", {"file_sha256": hashes})
        write_json(directory / "payload.json", {"job": job, "config": protocol, "job_dir": directory,
                   "input_dir": copied, "protocol_sha256": signature})
        if (directory / "result.json").exists():
            if read_json(directory / "result.json").get("protocol_sha256") != signature:
                raise ValueError("Existing job has a different protocol")
            evaluate_result(directory, dataset)
        else:
            pending.append((job, directory))
    summarize(output)
    return output, dataset, protocol, pending


def run(args):
    output, dataset, protocol, pending = prepare(args)
    if args.prepare_only:
        print(f"Prepared 6-job frozen protocol without starting workers: {output}", flush=True)
        return 0
    if args.only_job:
        known = {j["job_id"] for j in job_catalog()}
        if not set(args.only_job) <= known:
            raise ValueError("Unknown job identifier")
        pending = [(job, path) for job, path in pending if job["job_id"] in args.only_job]
    active = []
    while pending or active:
        while pending and len(active) < args.workers:
            job, directory = pending.pop(0)
            stdout, stderr = (directory / "stdout.log").open("a", encoding="utf-8"), (directory / "stderr.log").open("a", encoding="utf-8")
            process = subprocess.Popen([sys.executable, "-B", "-m", "research_experiments.rnj.run_source_ac96_full_history",
                                        "--worker", str(directory / "payload.json")], cwd=ROOT,
                                       env=os.environ.copy(), stdout=stdout, stderr=stderr)
            active.append((job, directory, process, perf_counter(), stdout, stderr))
            print(f"START {job['job_id']} pid={process.pid}", flush=True)
        for item in list(active):
            job, directory, process, launched, stdout, stderr = item
            killed = process.poll() is None and perf_counter() - launched > job["method_wall_seconds"] + protocol["hard_timeout_grace_seconds"]
            if killed:
                process.kill()
                process.wait()
            if process.poll() is None:
                continue
            stdout.close(); stderr.close()
            if not (directory / "result.json").exists():
                model = read_json(directory / "model.json") if (directory / "model.json").exists() else {"has_model": False}
                model.update(status="incomplete" if killed else "failed",
                             stop_reason="process_wall_limit" if killed else "worker_process_failed")
                write_json(directory / "model.json", model)
                execution = read_json(directory / "execution.json") if (directory / "execution.json").exists() else {}
                write_json(directory / "result.json", {**job, "status": model["status"], "stop_reason": model["stop_reason"],
                           "search_complete": False, "has_model": model["has_model"],
                           "elapsed_seconds": perf_counter() - execution.get("method_started_monotonic", launched),
                           "protocol_sha256": sha256(output / "protocol.json"), "process_exitcode": process.returncode})
            result = evaluate_result(directory, dataset)
            print(f"END {job['job_id']} {result['status']} family_F1={result['metrics'].get('family_clade_f1')} seconds={result['compute_seconds']:.2f}", flush=True)
            active.remove(item)
            summarize(output)
        if active:
            sleep(.5)
    write_json(output / "execution_complete.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
               "completion": read_json(output / "completion.json")})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "outputs/source_ac96_20261018")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/source_ac96_full_history_pengwah18_20261018")
    parser.add_argument("--solver", choices=("gurobi", "highs"), default="gurobi")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--only-job", nargs="+")
    args = parser.parse_args()
    return worker(args.worker) if args.worker else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
