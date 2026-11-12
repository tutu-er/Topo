"""Pengwah18: four source conditions, 288 fitting rows, no validation split.

Use the separately frozen terminal-group/Source runtime. Truth is read only
by the parent evaluator after each worker returns. All sixteen outcomes stay
in the denominator, including solver and outer-wall interruptions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter, sleep
import traceback

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import pandas as pd

from research_experiments.rnj.run_source_ac96_full_history import (
    plain, read_json, write_json, sha256, MethodBudgetExhausted, _complete_stop,
)

ROOT = Path(__file__).resolve().parents[2]
CASE = "pengwah18"
VARIANTS = ("no_source", "constant", "intermittent", "smooth")
METHODS = ("NJ", "RNJ", "WZZT", "RNJ+WZZT")
ACTIVE_TOLERANCE = 1e-8
CORE_PARAMETERS = {
    "max_atoms": 35, "r_upper_bound": 2., "x_upper_bound": 2.,
    "bound_safety_factor": 2., "bound_expansion_factor": 2.,
    "max_bound_expansions": 4, "bound_active_fraction": .995,
    "improvement_abs_tol": 1e-10, "improvement_rel_tol": 1e-8,
    "zero_tolerance": 1e-9, "validation_blocks": 4,
    "mip_rel_gap": 0., "presolve": True, "disp": False,
}
SCREEN_PARAMETERS = {
    "bootstrap_replicates": 100, "block_length": 4,
    "confidence_threshold": .75, "maximum_candidate_count": 2,
    "tolerance_factor": .16, "seed": 20260903,
    "maximum_subclade_frequency": .1, "fine_tolerance_ratio": .5,
}


def job_catalog():
    # Baselines first, then the group-assisted runs, then unrestricted runs.
    return [{"case": CASE, "variant": variant, "method": method,
             "per_solve_seconds": 6000. if "WZZT" in method else None,
             "method_wall_seconds": 30000. if "WZZT" in method else 300.,
             "job_id": f"{CASE}__{variant}__{method}"}
            for method in ("NJ", "RNJ", "RNJ+WZZT", "WZZT") for variant in VARIANTS]


def activate_runtime(directory):
    directory = Path(directory).resolve()
    from research_experiments.rnj.prepare_source_ac288_runtime import verify_runtime
    manifest = verify_runtime(directory)
    loaded = [name for name in sys.modules if name == "rnj_wzzt" or name.startswith("rnj_wzzt.")]
    if loaded:
        raise RuntimeError(f"runtime must be selected before imports: {loaded[:4]}")
    sys.path.insert(0, str(directory))
    import rnj_wzzt
    if not Path(rnj_wzzt.__file__).resolve().is_relative_to(directory):
        raise RuntimeError("frozen package import fell back to another checkout")
    return manifest


def input_arrays(scenarios, sources):
    if len(scenarios) != 3 or len(sources) != 3:
        raise ValueError("expected three 96-point fitting scenarios")
    columns = scenarios[0]["P_terminal"].columns
    arrays = {name: [] for name in ("P", "Q", "Y", "P_source", "Q_source")}
    metadata = []
    for i, (scenario, source) in enumerate(zip(scenarios, sources, strict=True)):
        frame = scenario["P_terminal"]
        if len(frame) != 96 or not frame.index.is_unique or not frame.columns.equals(columns):
            raise ValueError("each fitting scenario needs 96 aligned original-terminal rows")
        for key, field in (("P", "P_terminal"), ("Q", "Q_terminal"), ("Y", "drop_target")):
            other = scenario[field]
            if not other.index.equals(frame.index) or not other.columns.equals(columns):
                raise ValueError("P/Q/Y row or terminal alignment mismatch")
            arrays[key].append(other.to_numpy(dtype=float))
        if not source.p_pu.index.equals(frame.index) or not source.q_pu.index.equals(frame.index):
            raise ValueError("source schedules must retain matching scenario rows")
        arrays["P_source"].append(source.p_pu.to_numpy(dtype=float))
        arrays["Q_source"].append(source.q_pu.to_numpy(dtype=float))
        metadata.append({"name": scenario["name"], "scenario_index": i,
                         "row_range": [96*i, 96*(i+1)], "time_labels": frame.index.tolist()})
    arrays = {name: np.concatenate(blocks, axis=0) for name, blocks in arrays.items()}
    arrays.update(row_scenario=np.repeat(np.arange(3), 96), row_position=np.tile(np.arange(96), 3))
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite fitting observations")
    return arrays, {"total_samples": 288, "training_samples": 288, "validation_samples": 0,
                    "validation_supplied": False, "terminal_labels": columns.tolist(),
                    "scenarios": metadata, "ordering": "scenario_0,1,2; original row order",
                    "selection": "last_certified_path_point_without_validation"}


def select_snapshot_model(state, *, solver_events, progress_events, status, stop_reason):
    path = state.get("path", ())
    index = len(path)-1 if path else None
    point = path[index] if path else None
    labels = list(state.get("terminal_labels", ()))
    family, active = [], []
    if point is not None:
        if point.validation_mae is not None or point.validation_se is not None:
            raise ValueError("no-validation protocol received validation scores")
        for support, r, x in zip(point.support_labels, point.r_values, point.x_values, strict=True):
            if 1 < len(support) < len(labels):
                family.append(list(support))
                if max(r, x) > ACTIVE_TOLERANCE:
                    active.append(list(support))
    def value(name):
        return None if point is None else getattr(point, name)
    return {"schema_version": 3, "terminal_labels": labels,
            "support_indices": value("support_indices"), "support_labels": value("support_labels"),
            "r_values": value("r_values"), "x_values": value("x_values"),
            "R": value("r_matrix"), "X": value("x_matrix"),
            "source_h": value("source_h"), "source_g_r": value("source_response_r"),
            "source_g_x": value("source_response_x"), "train_mae": value("train_mae"),
            "validation_mae": None, "validation_se": None, "validation_supplied": False,
            "training_rows": 288, "selected_path_index": index, "last_path_index": index,
            "path": path, "attempted_extensions": state.get("attempted_extensions", ()),
            "initial_supports": state.get("initial_supports", ()),
            "terminal_groups": state.get("terminal_groups", ()),
            "requested_terminal_groups": state.get("requested_terminal_groups", ()),
            "r_upper_bound": state.get("r_upper_bound"), "x_upper_bound": state.get("x_upper_bound"),
            "bound_expansions": state.get("bound_expansions", 0),
            "family_clades": family, "active_clades": active, "clades": family,
            "active_atom_tolerance": ACTIVE_TOLERANCE, "solver_events": list(solver_events),
            "progress_events": list(progress_events), "status": status, "stop_reason": stop_reason,
            "selection": "last_certified_path_point_without_validation",
            "matrix_origin": "frozen_terminal_group_source_runtime_final_certified_atoms",
            "certificate_scope": "bounded one-atom subproblems; whole path remains greedy"}


def save_model(directory, model, arrays, started):
    model = dict(model)
    r, x = model.pop("R", None), model.pop("X", None)
    model["has_model"] = r is not None and x is not None
    model["elapsed_seconds"] = perf_counter()-started
    if model["has_model"]:
        prediction = arrays["P"] @ np.asarray(r).T + arrays["Q"] @ np.asarray(x).T
        if model.get("source_g_r") is not None:
            prediction += arrays["P_source"][:, None]*np.asarray(model["source_g_r"])[None, :]
            prediction += arrays["Q_source"][:, None]*np.asarray(model["source_g_x"])[None, :]
        model["train_mae_recomputed"] = float(np.mean(abs(arrays["Y"]-prediction)))
        model.setdefault("train_mae", model["train_mae_recomputed"])
    temporary = directory/"model.tmp.npz"
    np.savez_compressed(temporary, R=np.empty((0, 0)) if r is None else r,
                        X=np.empty((0, 0)) if x is None else x, **arrays)
    os.replace(temporary, directory/"model.npz")
    write_json(directory/"model.json", model)
    return model


class BudgetedSolver:
    """Keep raw results and independently recompute incumbent feasibility."""
    def __init__(self, core, directory, started, job):
        self.core, self.directory, self.started, self.job = core, Path(directory), started, job
        self.events = []

    def __enter__(self):
        self.original = self.core._run_milp
        self.core._run_milp = self.solve
        return self

    def __exit__(self, *args):
        self.core._run_milp = self.original

    def solve(self, variables, rows, **kwargs):
        remaining = self.job["method_wall_seconds"]-(perf_counter()-self.started)
        if remaining <= 0:
            raise MethodBudgetExhausted("method_wall_limit_before_next_solve")
        effective = min(float(kwargs["time_limit"]), remaining)
        result, diagnostics = self.original(variables, rows, **dict(kwargs, time_limit=effective))
        sequence = len(self.events)+1
        primal = getattr(result, "x", None)
        event = {**diagnostics.to_dict(), "sequence": sequence,
                 "requested_time_limit_seconds": kwargs["time_limit"], "time_limit_seconds": effective,
                 "remaining_wall_before_solve": remaining, "has_incumbent": primal is not None,
                 "optimality_certified": self.core.solver_diagnostics_prove_optimality(diagnostics),
                 "variable_slices": {name: [block.start, block.stop] for name, block in variables.slices.items()},
                 "elapsed_seconds": perf_counter()-self.started}
        if primal is not None:
            primal = np.asarray(primal, dtype=float)
            target = self.directory/"solver"/f"{sequence:06d}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, primal=primal,
                objective_coefficients=np.asarray(variables.objective), lower_bounds=np.asarray(variables.lower),
                upper_bounds=np.asarray(variables.upper), integrality=np.asarray(variables.integrality),
                constraint_row_index=np.asarray(rows.row_index, dtype=np.int64),
                constraint_col_index=np.asarray(rows.col_index, dtype=np.int64),
                constraint_data=np.asarray(rows.data), constraint_lower=np.asarray(rows.lower),
                constraint_upper=np.asarray(rows.upper))
            activity = np.bincount(np.asarray(rows.row_index, dtype=np.int64),
                                   weights=np.asarray(rows.data)*primal[np.asarray(rows.col_index, dtype=np.int64)],
                                   minlength=len(rows.lower))
            integer = np.asarray(variables.integrality) != 0
            event.update(primal_file=target.relative_to(self.directory).as_posix(),
                primal_objective_recomputed=float(np.asarray(variables.objective) @ primal),
                max_constraint_violation=float(max(0., np.max(np.asarray(rows.lower)-activity, initial=0.),
                                                    np.max(activity-np.asarray(rows.upper), initial=0.))),
                max_bound_violation=float(max(0., np.max(np.asarray(variables.lower)-primal, initial=0.),
                                               np.max(primal-np.asarray(variables.upper), initial=0.))),
                max_integrality_violation=float(np.max(abs(primal[integer]-np.rint(primal[integer])), initial=0.)),
                integer_variable_count=int(integer.sum()), variable_count=len(primal), constraint_count=len(rows.lower))
        self.events.append(event)
        write_json(self.directory/"solver"/f"{sequence:06d}.json", event)
        write_json(self.directory/"solver_events.json", self.events)
        return result, diagnostics


def worker(payload_path):
    payload = read_json(payload_path)
    directory, job, config = Path(payload["job_dir"]), payload["job"], payload["config"]
    activate_runtime(payload["runtime_dir"])
    from rnj_wzzt.estimation import laminar_l1_milp as core
    from rnj_wzzt.estimation.unmetered_inputs import prepare_source_inputs
    from rnj_wzzt.pipeline import _select_boundary_blocks
    from research_experiments.rnj.prepare_source_ac288 import load_algorithm_inputs
    from research_experiments.rnj.source_ac96_trees import fit_tree
    scenarios, sources, metadata = load_algorithm_inputs(payload["input_dir"])
    arrays, data_use = input_arrays(scenarios, sources)
    write_json(directory/"data_use.json", data_use)
    write_json(directory/"runtime_import.json", {"core_file": core.__file__, "runtime_dir": payload["runtime_dir"]})
    started = perf_counter()
    write_json(directory/"execution.json", {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
               "method_started_monotonic": started, "job": job})
    labels = [int(value) for value in data_use["terminal_labels"]]
    state = {"terminal_labels": labels, "path": (), "attempted_extensions": (),
             "initial_supports": (), "requested_terminal_groups": (),
             "terminal_groups": tuple((i,) for i in range(len(labels))),
             "r_upper_bound": 2., "x_upper_bound": 2., "bound_expansions": 0}
    progress, events, selection, model = [], [], None, None
    status, reason, error = "failed", "not_started", None
    try:
        if job["method"] in ("NJ", "RNJ"):
            fit = fit_tree(scenarios, sources, job["method"], root_bus=1)
            model = {"schema_version": 3, "terminal_labels": labels, "R": fit["R"], "X": fit["X"],
                     "family_clades": fit["clades"], "active_clades": fit["clades"], "clades": fit["clades"],
                     "source_h": None, "source_g_r": fit["source_response_g"][0],
                     "source_g_x": fit["source_response_g"][1], "support_indices": None,
                     "r_values": None, "x_values": None, "path": [], "attempted_extensions": [],
                     "solver_events": [], "progress_events": [], "initial_supports": [],
                     "terminal_groups": [], "requested_terminal_groups": [], "selected_path_index": None,
                     "last_path_index": None, "r_upper_bound": None, "x_upper_bound": None, "bound_expansions": 0,
                     "training_rows": 288, "validation_supplied": False, "validation_mae": None, "validation_se": None,
                     "regression_diagnostics": fit["diagnostics"], "tree": fit["tree"], "timing": fit["timing"],
                     "matrix_origin": "source_adjusted_ordered_regression_before_tree_inference",
                     "selection": "tree_on_all_288_rows", "status": "complete", "stop_reason": "tree_inference_completed"}
            status, reason = "complete", "tree_inference_completed"
        else:
            prepared_source = prepare_source_inputs(scenarios, sources)
            initial = [(i,) for i in range(len(labels))]
            groups = []
            if job["method"] == "RNJ+WZZT":
                diagnostics = {}
                screen_started = perf_counter()
                full, confidence, selected, r2, condition = _select_boundary_blocks(
                    scenarios, labels, 1, **config["screen_parameters"],
                    source_inputs=prepared_source, diagnostics=diagnostics)
                if "subclade_frequency" in diagnostics:
                    diagnostics["subclade_frequency_records"] = [
                        {"clade": sorted(c), "frequency": f}
                        for c, f in sorted(diagnostics.pop("subclade_frequency").items(),
                                           key=lambda pair: tuple(sorted(pair[0])))]
                positions = {label: i for i, label in enumerate(labels)}
                selected_supports = [tuple(positions[label] for label in sorted(clade)) for clade in selected]
                initial += selected_supports
                groups = [block for block in selected_supports if len(block) >= 3]
                selection = {"selector": "frozen_runtime.rnj_wzzt.pipeline._select_boundary_blocks",
                             "full_clades": full, "confidence_records": [{"clade": sorted(c), "frequency": f}
                                  for c, f in sorted(confidence.items(), key=lambda pair: tuple(sorted(pair[0])))],
                             "selected_support_labels": [sorted(c) for c in selected],
                             "selected_support_indices": selected_supports, "requested_terminal_groups": groups,
                             "diagnostics": diagnostics, "parameters": config["screen_parameters"],
                             "r2_score": r2, "condition": condition, "training_sample_count": 288,
                             "validation_used": False, "elapsed_seconds": perf_counter()-screen_started,
                             "truth_used_for_selection": False}
                write_json(directory/"rnj_screen.json", selection)
            state.update(initial_supports=initial, requested_terminal_groups=groups,
                         terminal_groups=core.normalize_terminal_groups(groups, len(labels)).groups)
            with BudgetedSolver(core, directory, started, job) as observer:
                events = observer.events
                def checkpoint(snapshot):
                    nonlocal state
                    snapshot["requested_terminal_groups"] = groups
                    state = snapshot
                    sequence = len(progress)+1
                    record = plain(snapshot)
                    record.update(sequence=sequence, elapsed_seconds=perf_counter()-started)
                    write_json(directory/"progress"/f"{sequence:06d}.json", record)
                    progress.append({"sequence": sequence, "event": snapshot["event"],
                                     "elapsed_seconds": perf_counter()-started})
                    partial = select_snapshot_model(state, solver_events=events, progress_events=progress,
                                                     status="running", stop_reason="in_progress")
                    partial["bootstrap"] = selection
                    save_model(directory, partial, arrays, started)
                fit = core.fit_laminar_l1_sensitivity(
                    scenarios, validation_scenarios=None, initial_supports=initial,
                    terminal_groups=groups if job["method"] == "RNJ+WZZT" else None,
                    source_inputs=prepared_source, **config["core_parameters"],
                    time_limit=job["per_solve_seconds"], solver=config["solver"], progress_callback=checkpoint)
                reason = fit.stop_reason
                status = "complete" if _complete_stop(reason) else "incomplete"
                model = select_snapshot_model(state, solver_events=events, progress_events=progress,
                                              status=status, stop_reason=reason)
                if model["selected_path_index"] != fit.selected_path_index:
                    raise RuntimeError("saved checkpoint differs from no-validation core selection")
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        timed_out = isinstance(exc, TimeoutError) or bool(events and events[-1]["status"] == 1)
        status, reason = ("incomplete" if timed_out else "failed"), str(exc)
        model = select_snapshot_model(state, solver_events=events, progress_events=progress,
                                      status=status, stop_reason=reason)
        write_json(directory/"error.json", error)
    model.update(bootstrap=selection, budget=job, core_parameters=config["core_parameters"] if "WZZT" in job["method"] else None)
    from research_experiments.rnj.prepare_source_ac288_runtime import verify_loaded_modules
    write_json(directory/"runtime_import.json", {"runtime_dir": payload["runtime_dir"],
               "module_origins": verify_loaded_modules(payload["runtime_dir"])})
    saved = save_model(directory, model, arrays, started)
    write_json(directory/"result.json", {**job, "status": status, "stop_reason": reason,
               "search_complete": status == "complete", "has_model": saved["has_model"],
               "elapsed_seconds": perf_counter()-started, "protocol_sha256": payload["protocol_sha256"],
               "error": error, "truth_read_by": "parent_evaluator_only"})
    return 0


def evaluate_result(directory, dataset):
    from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth, clade_metrics, relative_error
    result, model = read_json(directory/"result.json"), read_json(directory/"model.json")
    metrics = {}
    if model["has_model"]:
        truth = physical_truth(dataset/CASE/"network", model["terminal_labels"])
        for prefix, clades in (("family", model["family_clades"]), ("active", model["active_clades"])):
            values = clade_metrics(clades, truth["clades"], model["terminal_labels"])
            for key in ("clade_f1", "precision", "recall"):
                metrics[f"{prefix}_{key}"] = values[key]
        with np.load(directory/"model.npz") as data:
            metrics.update(r_relative_error=relative_error(data["R"], truth["R"]),
                           x_relative_error=relative_error(data["X"], truth["X"]))
        metrics["clade_f1"] = metrics["family_clade_f1"]
    result.update(metrics=metrics, compute_seconds=result["elapsed_seconds"],
                  selected_path_index=model.get("selected_path_index"), path_length=len(model.get("path", [])),
                  bound_expansions=model.get("bound_expansions"),
                  selected_rnj_blocks=len((model.get("bootstrap") or {}).get("selected_support_labels", [])),
                  solver_time_limit_count=sum(e.get("status") == 1 for e in model.get("solver_events", [])),
                  uncertified_solve_count=sum(not e.get("optimality_certified", False) for e in model.get("solver_events", [])))
    write_json(directory/"result.json", result)
    return result


def summarize(output):
    rows = []
    for job in job_catalog():
        path = output/"jobs"/job["job_id"]/"result.json"
        if path.exists():
            result = read_json(path)
            rows.append({**job, **{key: result.get(key) for key in ("status", "stop_reason", "has_model", "compute_seconds",
                "selected_path_index", "path_length", "bound_expansions", "selected_rnj_blocks",
                "solver_time_limit_count", "uncertified_solve_count")}, **result.get("metrics", {})})
        else:
            rows.append({**job, "status": "not_run"})
    pd.DataFrame(rows).to_csv(output/"summary.csv", index=False, float_format="%.17g")
    counts = dict(Counter(row["status"] for row in rows))
    write_json(output/"completion.json", {"expected": 16, "finished": 16-counts.get("not_run", 0),
               "status_counts": counts, "models_available": sum(bool(row.get("has_model")) for row in rows)})


def prepare(args):
    output, dataset, runtime = args.output_dir.resolve(), args.dataset_dir.resolve(), args.runtime_dir.resolve()
    from research_experiments.rnj.prepare_source_ac288_runtime import ensure_runtime
    ensure_runtime(runtime)
    output.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), ROOT/"research_experiments/rnj/run_source_ac96_full_history.py",
             ROOT/"research_experiments/rnj/source_ac96_trees.py", ROOT/"research_experiments/rnj/rooted_study_baselines.py",
             ROOT/"research_experiments/rnj/sensitivity_geometry.py",
             ROOT/"research_experiments/rnj/prepare_source_ac288.py", ROOT/"research_experiments/rnj/prepare_source_ac288_runtime.py",
             ROOT/"research_experiments/runtime_snapshots/source_ac288_72af_20261018.zip",
             ROOT/"research_experiments/runtime_snapshots/source_ac288_72af_20261018.manifest.json"]
    protocol = {"schema_version": 3, "case": CASE, "variants": VARIANTS, "methods": METHODS,
                "jobs": job_catalog(), "expected_jobs": 16, "total_samples": 288, "training_samples": 288,
                "validation_samples": 0, "validation_supplied": False, "independent_test_samples": 0,
                "selection": "last_certified_path_point_without_validation", "scenario_count": 3, "samples_per_scenario": 96,
                "observed_root_label": 1, "source_information": "exact P/Q schedules; unknown shared physical location",
                "core_parameters": CORE_PARAMETERS, "screen_parameters": SCREEN_PARAMETERS,
                "minimum_atomic_group_size": 3, "rnj_atomic_block_rule": "complete_direct_terminal_children",
                "rx75_weight": .75, "tree_tolerance_factor": .16, "solver": args.solver,
                "concurrency": args.workers, "solver_threads": 1, "blas_threads": 1, "hard_timeout_grace_seconds": 30,
                "dataset_manifest_sha256": sha256(dataset/"manifest.json"),
                "runtime_manifest_sha256": sha256(runtime/"runtime_manifest.json"),
                "runtime_directory": str(runtime), "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths},
                "truth_used_for_selection": False, "score_scope": "family and active clades; squared-voltage R/X matrices"}
    protocol = plain(protocol)
    if (output/"protocol.json").exists() and read_json(output/"protocol.json") != protocol:
        raise ValueError("frozen protocol changed; preserve this directory and choose another")
    write_json(output/"protocol.json", protocol)
    signature = sha256(output/"protocol.json")
    for relative, checksum in protocol["source_sha256"].items():
        target = output/"source_snapshot"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(ROOT/relative, target)
        if sha256(target) != checksum:
            raise ValueError("driver source snapshot mismatch")
    if not (output/"environment.json").exists():
        write_json(output/"environment.json", {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": {name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scipy", "networkx", "gurobipy")},
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "runtime": str(runtime), "threads": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}})
    hashes = {}
    for variant in VARIANTS:
        source = dataset/CASE/variant
        copied = output/"inputs"/variant
        hashes[variant] = {}
        for i in range(3):
            for folder in ("observed", "source"):
                for path in sorted((source/f"scenario_{i}"/folder).iterdir()):
                    if not path.is_file():
                        continue
                    relative = path.relative_to(source)
                    target = copied/relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        shutil.copyfile(path, target)
                    if sha256(target) != sha256(path):
                        raise ValueError("algorithm observation/source copy differs")
                    hashes[variant][relative.as_posix()] = sha256(target)
    pending = []
    for job in job_catalog():
        directory = output/"jobs"/job["job_id"]
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory/"inputs_hash.json", {"file_sha256": hashes[job["variant"]]})
        write_json(directory/"payload.json", {"job": job, "config": protocol, "job_dir": directory,
            "runtime_dir": runtime, "input_dir": output/"inputs"/job["variant"], "protocol_sha256": signature})
        if (directory/"result.json").exists():
            if read_json(directory/"result.json").get("protocol_sha256") != signature:
                raise ValueError("existing result protocol mismatch")
            evaluate_result(directory, dataset)
        else:
            pending.append((job, directory))
    summarize(output)
    return output, dataset, protocol, pending


def run(args):
    output, dataset, protocol, pending = prepare(args)
    if args.prepare_only:
        print(f"Prepared sixteen jobs; 288 fitting rows, zero validation: {output}", flush=True)
        return 0
    if args.only_job:
        if not set(args.only_job) <= {j["job_id"] for j in job_catalog()}:
            raise ValueError("unknown job identifier")
        pending = [(j, p) for j, p in pending if j["job_id"] in args.only_job]
    active = []
    while pending or active:
        while pending and len(active) < args.workers:
            job, directory = pending.pop(0)
            stdout = (directory/"stdout.log").open("a", encoding="utf-8")
            stderr = (directory/"stderr.log").open("a", encoding="utf-8")
            process = subprocess.Popen([sys.executable, "-B", "-m", "research_experiments.rnj.run_source_ac288",
                "--worker", str(directory/"payload.json")], cwd=ROOT, env=os.environ.copy(), stdout=stdout, stderr=stderr)
            active.append((job, directory, process, perf_counter(), stdout, stderr))
            print(f"START {job['job_id']} pid={process.pid}", flush=True)
        for item in list(active):
            job, directory, process, launched, stdout, stderr = item
            killed = process.poll() is None and perf_counter()-launched > job["method_wall_seconds"]+30
            if killed:
                process.kill()
                process.wait()
            if process.poll() is None:
                continue
            stdout.close()
            stderr.close()
            if not (directory/"result.json").exists():
                model = read_json(directory/"model.json") if (directory/"model.json").exists() else {"has_model": False}
                model.update(status="incomplete" if killed else "failed", stop_reason="process_wall_limit" if killed else "worker_process_failed")
                write_json(directory/"model.json", model)
                write_json(directory/"result.json", {**job, "status": model["status"], "stop_reason": model["stop_reason"],
                    "search_complete": False, "has_model": model["has_model"], "elapsed_seconds": perf_counter()-launched,
                    "protocol_sha256": sha256(output/"protocol.json"), "process_exitcode": process.returncode})
            result = evaluate_result(directory, dataset)
            print(f"END {job['job_id']} {result['status']} F1={result['metrics'].get('family_clade_f1')} seconds={result['compute_seconds']:.2f}", flush=True)
            active.remove(item)
            summarize(output)
        if active:
            sleep(.5)
    write_json(output/"execution_complete.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
               "completion": read_json(output/"completion.json")})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT/"outputs/source_ac288_20261018")
    parser.add_argument("--runtime-dir", type=Path, default=ROOT/"outputs/source_ac288_runtime_20261018")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/source_ac288_benchmark_20261018")
    parser.add_argument("--solver", choices=("gurobi", "highs"), default="gurobi")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--only-job", nargs="+")
    args = parser.parse_args()
    return worker(args.worker) if args.worker else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
