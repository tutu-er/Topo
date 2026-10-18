"""Original 96 observations: relaxed RNJ blocks and grouped WZZT controls.

NJ/RNJ results are reused after a separate evidence check. New workers receive
only copied observations and known source schedules, never the physical tree.
Every worker executes the frozen source closure and frozen 72af runtime.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import inspect
from functools import wraps
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

from research_experiments.rnj.run_source_ac288 import (
    CORE_PARAMETERS, BudgetedSolver, activate_runtime, save_model,
    select_snapshot_model as _select_snapshot_model,
)
from research_experiments.rnj.run_source_ac96_full_history import (
    plain, read_json, write_json, sha256, _complete_stop, MethodBudgetExhausted,
)
from research_experiments.rnj.run_rx75_source_comparison import (
    resolution_tree, with_resolution, controller_lock, save_checkpoint, recover_process,
)
from research_experiments.rnj.rx75_edge_floor import calibrate_rx75_floor, enforce_rx75_floor
from research_experiments.rnj.select_rnj_blocks96 import select_rnj_blocks96
from research_experiments.rnj.reuse_ac96_baselines import reuse_baselines

ROOT = Path(__file__).resolve().parents[2]
CASES = ("paper15", "soumalas11", "flynn16", "pengwah18")
VARIANTS = ("no_source", "constant", "intermittent", "smooth")
METHODS = ("WZZT", "RNJ+WZZT", "RNJ+WZZT_ungrouped", "RNJ+WZZT_tree_init")
SCREEN_PARAMETERS = {
    "bootstrap_replicates": 100, "block_length": 4, "confidence_threshold": .75,
    "maximum_candidate_count": 2, "tolerance_factor": .16, "seed": 20260903,
    "maximum_subclade_frequency": .15, "fine_tolerance_ratio": .5, "minimum_block_size": 2,
}
INPUT_NAMES = {f"observed/{key}.csv" for key in (
    "P_terminal", "Q_terminal", "V_terminal", "drop_target", "root_voltage", "P0_measured", "Q0_measured"
)} | {"observed/metadata.json", "source/amplitudes.csv", "source/metadata.json"}
DESIGN_PROTOCOL = Path("docs/rnj_blocks96_protocol_20261019.md")
METRIC_KEYS = tuple(f"{p}_{k}" for p in ("family", "active", "resolution")
                    for k in ("clade_f1", "precision", "recall")) + (
                        "clade_f1", "r_relative_error", "x_relative_error")


def job_catalog():
    # Source/no-source for the diagnostic case first; all planned cases remain.
    cases = ("soumalas11", "pengwah18", "paper15", "flynn16")
    return [{"case": c, "variant": v, "method": m,
             "role": "main" if m in ("WZZT", "RNJ+WZZT") else "configuration_control",
             "screen_parameters": SCREEN_PARAMETERS,
             "input_relative_directory": f"inputs/{c}/{v}",
             "evaluation_network_directory": f"evaluation_network/{c}",
             "per_solve_seconds": 6000., "method_wall_seconds": 30000.,
             "job_id": f"{c}__{v}__{m}"}
            for c in cases for v in VARIANTS for m in (
                "RNJ+WZZT", "RNJ+WZZT_ungrouped", "RNJ+WZZT_tree_init", "WZZT")]


def core_parameters(n):
    return {**CORE_PARAMETERS, "max_atoms": 2*int(n)-1}


class RecordedSolver(BudgetedSolver):
    """Passive builder annotations bind each raw solve to its original family."""
    def __init__(self, core, source, directory, started, job):
        super().__init__(core, directory, started, job)
        self.source, self.builders, self.metadata = source, [], {}

    def __enter__(self):
        super().__enter__()
        def wrapper(original, source_builder):
            signature = inspect.signature(original)
            @wraps(original)
            def build(*args, **kwargs):
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                supports = tuple(tuple(map(int, s)) for s in bound.arguments["supports"])
                bound.arguments["supports"] = supports
                result = original(*bound.args, **bound.kwargs)
                model = result[0] if source_builder else result
                partition = model.partition
                self.metadata[id(model.variables)] = {
                    "old_support_indices": supports,
                    "terminal_partition": None if partition is None else partition.groups,
                    "terminal_z_columns": None if model.z is None else model.terminal_z_columns,
                    "builder": original.__name__, "is_extension": model.z is not None,
                    "old_atom_count": model.r.stop-model.r.start,
                }
                return result
            return build
        for module, name, source_builder in (
            (self.core, "build_fixed_model", False), (self.core, "build_extension_model", False),
            (self.source, "build_fixed_model", False), (self.source, "build_extension_model", False),
            (self.source, "build_fixed_source_model", True), (self.source, "build_extension_source_model", True),
        ):
            original = getattr(module, name)
            self.builders.append((module, name, original))
            setattr(module, name, wrapper(original, source_builder))
        return self

    def __exit__(self, *args):
        for module, name, original in reversed(self.builders):
            setattr(module, name, original)
        super().__exit__(*args)

    def solve(self, variables, rows, **kwargs):
        metadata = self.metadata.get(id(variables))
        if metadata is None:
            raise RuntimeError("solve has no passive builder/family record")
        result = super().solve(variables, rows, **kwargs)
        self.events[-1].update(plain(metadata))
        sequence = self.events[-1]["sequence"]
        write_json(self.directory/"solver"/f"{sequence:06d}.json", self.events[-1])
        write_json(self.directory/"solver_events.json", self.events)
        return result


def select_snapshot_model(*args, **kwargs):
    return {**_select_snapshot_model(*args, **kwargs), "training_rows": 96}


def verify_inputs(directory, hashes):
    directory = Path(directory)
    actual = {p.relative_to(directory).as_posix(): sha256(p) for p in directory.rglob("*") if p.is_file()}
    if set(actual) != INPUT_NAMES or actual != hashes:
        raise ValueError("input whitelist/hash mismatch")


def load_algorithm_inputs(directory, source_class):
    """No network, simulation truth, source attachment or dataset manifest read."""
    directory = Path(directory)
    metadata = read_json(directory/"observed/metadata.json")
    scenario = {"name": f"{directory.parent.name}_{directory.name}",
                "root_observation": metadata["root_observation"],
                "scenario_settings": metadata["scenario_settings"]}
    for key in ("P_terminal", "Q_terminal", "V_terminal", "drop_target"):
        frame = pd.read_csv(directory/"observed"/f"{key}.csv", index_col="t", float_precision="round_trip")
        frame.columns = pd.Index([int(c) for c in frame.columns])
        scenario[key] = frame
    for key in ("root_voltage", "P0_measured", "Q0_measured"):
        frame = pd.read_csv(directory/"observed"/f"{key}.csv", index_col="t", float_precision="round_trip")
        if frame.shape[1] != 1:
            raise ValueError("expected a single observed series")
        scenario[key] = frame.iloc[:, 0]
    frame = pd.read_csv(directory/"source/amplitudes.csv", index_col="t",
                        usecols=["t", "P_pu", "Q_pu"], float_precision="round_trip")
    source = source_class(p_pu=frame["P_pu"], q_pu=frame["Q_pu"],
                          provenance="externally_given_exact_source_time_series")
    return [scenario], [source]


def input_arrays(scenarios, sources):
    if len(scenarios) != 1 or len(sources) != 1:
        raise ValueError("expected one complete 96-point day")
    scenario, source = scenarios[0], sources[0]
    frame = scenario["P_terminal"]
    if len(frame) != 96 or not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("expected 96 unique aligned rows and unique terminals")
    arrays = {}
    for key, field in (("P", "P_terminal"), ("Q", "Q_terminal"), ("Y", "drop_target")):
        other = scenario[field]
        if not other.index.equals(frame.index) or not other.columns.equals(frame.columns):
            raise ValueError("P/Q/Y alignment mismatch")
        arrays[key] = other.to_numpy(dtype=float)
    for key, series in (("P_source", source.p_pu), ("Q_source", source.q_pu)):
        if not series.index.equals(frame.index):
            raise ValueError("source row alignment mismatch")
        arrays[key] = series.to_numpy(dtype=float)
    arrays.update(row_scenario=np.zeros(96, dtype=int), row_position=np.arange(96))
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("nonfinite input")
    return arrays, {"total_samples": 96, "training_samples": 96, "validation_samples": 0,
        "validation_supplied": False, "terminal_labels": frame.columns.tolist(),
        "time_labels": frame.index.tolist(), "ordering": "all original 96 rows, unchanged order",
        "selection": "last_certified_path_point_without_validation"}


def initialization(labels, full_clades, selected_clades, method):
    positions = {int(label): i for i, label in enumerate(labels)}
    convert = lambda c: tuple(sorted(positions[int(v)] for v in c))
    selected = sorted({convert(c) for c in selected_clades}, key=lambda c: (len(c), c))
    if any(len(c) < 2 for c in selected):
        raise ValueError("selected groups must contain at least two terminals")
    initial = [(i,) for i in range(len(labels))]
    if method != "WZZT":
        initial += selected
    if method == "RNJ+WZZT_tree_init":
        initial += [convert(c) for c in full_clades]
    initial = sorted(set(initial), key=lambda c: (len(c), c))
    groups = selected if method in ("RNJ+WZZT", "RNJ+WZZT_tree_init") else []
    return initial, groups


def verify_experiment_modules():
    origins = {}
    for name, module in sys.modules.copy().items():
        if name.startswith("research_experiments.") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(ROOT):
                raise RuntimeError(f"helper escaped frozen closure: {name}: {path}")
            origins[name] = str(path)
    return origins


def worker(payload_path):
    payload = read_json(payload_path)
    directory, job, config = Path(payload["job_dir"]), payload["job"], payload["config"]
    if sha256(directory.parent.parent/"protocol.json") != payload["protocol_sha256"]:
        raise ValueError("worker protocol fingerprint mismatch")
    for relative, checksum in config["source_sha256"].items():
        if sha256(ROOT/relative) != checksum:
            raise ValueError("worker source closure fingerprint mismatch")
    verify_inputs(payload["input_dir"], payload["input_sha256"])
    activate_runtime(payload["runtime_dir"])
    from rnj_wzzt import pipeline
    from rnj_wzzt.estimation import laminar_l1_milp as core, laminar_l1_source as source_module
    from rnj_wzzt.estimation.unmetered_inputs import SourceInputs, prepare_source_inputs
    from research_experiments.rnj.source_ac96_trees import fit_tree
    scenarios, sources = load_algorithm_inputs(payload["input_dir"], SourceInputs)
    arrays, data_use = input_arrays(scenarios, sources)
    write_json(directory/"data_use.json", data_use)
    started = perf_counter()
    write_json(directory/"execution.json", {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "method_started_monotonic": started, "job": job, "timing_environment": config["timing_environment"]})
    labels = [int(v) for v in data_use["terminal_labels"]]
    parameters = core_parameters(len(labels))
    state = {"terminal_labels": labels, "path": (), "attempted_extensions": (), "initial_supports": (),
        "requested_terminal_groups": (), "terminal_groups": tuple((i,) for i in range(len(labels))),
        "r_upper_bound": 2., "x_upper_bound": 2., "bound_expansions": 0}
    timing, progress, events = {}, [], []
    selection, calibration, error = None, None, None
    status, reason = "failed", "not_started"
    try:
        front_started = perf_counter()
        front = fit_tree(scenarios, sources, "RNJ", root_bus=1)
        timing["front_end_fit_seconds"] = perf_counter()-front_started
        calibration = calibrate_rx75_floor(front["R"], front["X"], factor=.16)
        arrays.update(calibration_R=front["R"], calibration_X=front["X"],
                      calibration_source_g=front["source_response_g"])
        write_json(directory/"rx75_calibration.json", {**calibration,
            "matrix_origin": "pre_tree_ordered_FWL_regression", "truth_used": False,
            "front_end_method": "RNJ", "regression_diagnostics": front["diagnostics"]})
        prepared_source = prepare_source_inputs(scenarios, sources)
        full, selected = set(), []
        if job["method"] != "WZZT":
            diagnostics, screen_started = {}, perf_counter()
            full, confidence, selected, r2, condition = select_rnj_blocks96(
                scenarios, labels, 1, runtime_pipeline=pipeline, runtime_source=source_module,
                source_inputs=prepared_source, diagnostics=diagnostics,
                sensitivity_diagnostics=front["diagnostics"],
                sensitivity_fit=(front["R"], front["X"], front["projected_r2_score"], front["condition"]),
                **job["screen_parameters"])
            timing["bootstrap_seconds"] = perf_counter()-screen_started
            # Non-string clade keys are converted to explicit records for audit.
            for key in ("subclade_frequency", "clade_frequency", "peer_frequency"):
                if isinstance(diagnostics.get(key), dict):
                    diagnostics[key+"_records"] = [{"clade": sorted(c), "frequency": f}
                        for c, f in sorted(diagnostics.pop(key).items(), key=lambda p: tuple(sorted(p[0])))]
            selection = {"selector": "research_experiments.rnj.select_rnj_blocks96.select_rnj_blocks96",
                "full_clades": full, "confidence_records": [{"clade": sorted(c), "frequency": f}
                    for c, f in sorted(confidence.items(), key=lambda p: tuple(sorted(p[0])))],
                "selected_support_labels": [sorted(c) for c in selected], "diagnostics": diagnostics,
                "parameters": job["screen_parameters"], "r2_score": r2, "condition": condition,
                "training_sample_count": 96, "validation_used": False,
                "elapsed_seconds": timing["bootstrap_seconds"], "truth_used_for_selection": False,
                "empty_selection_retained": not bool(selected)}
        initial, groups = initialization(labels, full, selected, job["method"])
        state.update(initial_supports=initial, requested_terminal_groups=groups,
                     terminal_groups=core.normalize_terminal_groups(groups, len(labels)).groups)
        write_json(directory/"initialization.json", {"method": job["method"], "initial_supports": initial,
            "requested_terminal_groups": groups, "full_rnj_clades": full,
            "selected_support_labels": selected, "truth_used": False,
            "zero_old_supports_keep_laminar_constraints": True,
            "grouping_changes_search_domain_for_groups_of_size_at_least_three": True})
        if selection is not None:
            selection.update(selected_support_indices=[tuple(labels.index(v) for v in sorted(c)) for c in selected],
                             requested_terminal_groups=groups)
            write_json(directory/"rnj_screen.json", selection)
        fit_started = perf_counter()
        with enforce_rx75_floor(core, source_module, calibration), RecordedSolver(core, source_module, directory, started, job) as observer:
            events = observer.events
            def checkpoint(snapshot):
                nonlocal state
                snapshot["requested_terminal_groups"] = groups
                state = snapshot
                sequence = len(progress)+1
                record = plain(snapshot)
                record.update(sequence=sequence, elapsed_seconds=perf_counter()-started, rx75_calibration=calibration)
                write_json(directory/"progress"/f"{sequence:06d}.json", record)
                progress.append({"sequence": sequence, "event": snapshot["event"], "elapsed_seconds": perf_counter()-started})
                partial = select_snapshot_model(state, solver_events=events, progress_events=progress,
                                                status="running", stop_reason="in_progress")
                partial.update(bootstrap=selection, timing=timing, schema_version=5)
                save_checkpoint(directory, with_resolution(partial, calibration), arrays, started, sequence)
            fit = core.fit_laminar_l1_sensitivity(scenarios, validation_scenarios=None,
                initial_supports=initial, terminal_groups=groups or None, source_inputs=prepared_source,
                **parameters, time_limit=job["per_solve_seconds"], solver=config["solver"], progress_callback=checkpoint)
            reason = fit.stop_reason
            status = "complete" if _complete_stop(reason) else "incomplete"
            model = select_snapshot_model(state, solver_events=events, progress_events=progress, status=status, stop_reason=reason)
            if model["selected_path_index"] != fit.selected_path_index:
                raise RuntimeError("saved point differs from core selection")
        timing["wzzt_fit_seconds"] = perf_counter()-fit_started
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        timed_out = isinstance(exc, (TimeoutError, MethodBudgetExhausted)) or bool(events and events[-1].get("status") == 1)
        status, reason = ("incomplete" if timed_out else "failed"), str(exc)
        model = select_snapshot_model(state, solver_events=events, progress_events=progress, status=status, stop_reason=reason)
        write_json(directory/"error.json", error)
    model.update(schema_version=5, status=status, stop_reason=reason, bootstrap=selection, budget=job,
                 core_parameters=parameters, timing=timing)
    if calibration is not None:
        model = with_resolution(model, calibration)
    from research_experiments.rnj.prepare_source_ac288_runtime import verify_loaded_modules
    write_json(directory/"runtime_import.json", {"runtime_dir": payload["runtime_dir"],
        "module_origins": verify_loaded_modules(payload["runtime_dir"]), "experiment_source_root": ROOT,
        "driver_file": __file__, "experiment_module_origins": verify_experiment_modules()})
    saved = save_model(directory, model, arrays, started)
    write_json(directory/"result.json", {**job, "status": status, "stop_reason": reason,
        "search_complete": status == "complete", "has_model": saved["has_model"],
        "elapsed_seconds": perf_counter()-started, "timing": timing, "protocol_sha256": payload["protocol_sha256"],
        "error": error, "truth_read_by": "parent_evaluator_only"})
    return 0


def verify_evaluation_network(output, case):
    output = Path(output)
    protocol = read_json(output/"protocol.json")
    if sha256(output/"dataset_manifest.json") != protocol["dataset_manifest_sha256"]:
        raise ValueError("evaluation dataset manifest changed")
    prefix = f"{case}/network/"
    expected = {p[len(prefix):]: h for p, h in read_json(output/"dataset_manifest.json")["file_sha256"].items() if p.startswith(prefix)}
    directory = output/"evaluation_network"/case
    actual = {p.relative_to(directory).as_posix(): sha256(p) for p in directory.rglob("*") if p.is_file()}
    if not expected or expected != actual:
        raise ValueError("frozen evaluation network changed")
    return directory


def evaluate_result(directory):
    from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth, clade_metrics, relative_error
    directory = Path(directory)
    result, model = read_json(directory/"result.json"), read_json(directory/"model.json")
    metrics = dict.fromkeys(METRIC_KEYS)
    if model.get("has_model"):
        truth = physical_truth(verify_evaluation_network(directory.parent.parent, result["case"]), model["terminal_labels"])
        for prefix in ("family", "active", "resolution"):
            values = clade_metrics(model[prefix+"_clades"], truth["clades"], model["terminal_labels"])
            metrics.update({f"{prefix}_{key}": values[key] for key in ("clade_f1", "precision", "recall")})
        with np.load(directory/"model.npz") as arrays:
            metrics.update(r_relative_error=relative_error(arrays["R"], truth["R"]),
                           x_relative_error=relative_error(arrays["X"], truth["X"]))
        metrics["clade_f1"] = metrics["resolution_clade_f1"]
    result.update(metrics=metrics, primary_topology_metric="resolution_clade_f1", compute_seconds=result["elapsed_seconds"],
        selected_path_index=model.get("selected_path_index"), path_length=len(model.get("path", [])),
        bound_expansions=model.get("bound_expansions"),
        selected_rnj_blocks=len((model.get("bootstrap") or {}).get("selected_support_labels", [])),
        solver_time_limit_count=sum(e.get("status") == 1 for e in model.get("solver_events", [])),
        uncertified_solve_count=sum(not e.get("optimality_certified", False) for e in model.get("solver_events", [])))
    write_json(directory/"result.json", result)
    return result


def summarize(output):
    output, rows = Path(output), []
    for job in job_catalog():
        directory = output/"jobs"/job["job_id"]
        if (directory/"result.json").exists():
            result = read_json(directory/"result.json")
            rows.append({**job, **{k: result.get(k) for k in ("status", "stop_reason", "has_model", "compute_seconds",
                "selected_path_index", "path_length", "bound_expansions", "selected_rnj_blocks", "solver_time_limit_count", "uncertified_solve_count")},
                **dict.fromkeys(METRIC_KEYS), **result.get("metrics", {}),
                **{f"time_{k}": v for k, v in result.get("timing", {}).items()}})
        else:
            rows.append({**job, "status": "running" if (directory/"execution.json").exists() else "not_run", **dict.fromkeys(METRIC_KEYS)})
    pd.DataFrame(rows).to_csv(output/"summary.csv", index=False, float_format="%.17g")
    write_json(output/"completion.json", {"expected": len(rows),
        "finished": sum(r["status"] in {"complete", "incomplete", "failed"} for r in rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "models_available": sum(bool(r.get("has_model")) for r in rows)})
    # Baselines retain their historical computation time; extra scoring is a
    # separate field. Pending/failed rows stay in the complete comparison.
    comparison = [{**r, "reused_baseline": False} for r in rows]
    if (output/"protocol.json").exists():
        for job in read_json(output/"protocol.json").get("baseline_jobs", []):
            result = read_json(output/job["directory"]/"result.json")
            comparison.append({**job, **{k: result.get(k) for k in (
                "status", "stop_reason", "has_model", "compute_seconds", "postprocess_seconds")},
                **dict.fromkeys(METRIC_KEYS), **result.get("metrics", {}), "reused_baseline": True})
    pd.DataFrame(comparison).to_csv(output/"comparison.csv", index=False, float_format="%.17g")
    aggregate = []
    for method in ("NJ", "RNJ", *METHODS):
        subset = [r for r in comparison if r["method"] == method]
        if not subset:
            continue
        row = {"method": method, "planned": len(subset),
               **{state: sum(r["status"] == state for r in subset)
                  for state in ("complete", "incomplete", "failed", "running", "not_run")}}
        for key in ("resolution_clade_f1", "family_clade_f1", "active_clade_f1",
                    "r_relative_error", "x_relative_error", "compute_seconds"):
            values = [r[key] for r in subset if r.get(key) is not None]
            row[key+"_available"] = len(values)
            row[key+"_mean"] = float(np.mean(values)) if values else None
        aggregate.append(row)
    pd.DataFrame(aggregate).to_csv(output/"aggregate_by_method.csv", index=False, float_format="%.17g")
    return rows


def source_closure(root=ROOT):
    root, found = Path(root), set()
    pending = [Path(__file__).resolve().relative_to(ROOT),
        Path("research_experiments/rnj/source_ac96_trees.py"),
        Path("research_experiments/rnj/prepare_source_ac288_runtime.py"),
        Path("research_experiments/rnj/audit_source_ac96_benchmark.py")]
    while pending:
        relative = pending.pop()
        if relative in found:
            continue
        path = root/relative
        if not path.is_file():
            raise ValueError(f"missing source dependency: {relative}")
        found.add(relative)
        for parent in relative.parents:
            init = parent/"__init__.py"
            if parent != Path(".") and (root/init).is_file() and init not in found:
                pending.append(init)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            names = ([node.module, *(node.module+"."+a.name for a in node.names)]
                     if isinstance(node, ast.ImportFrom) and node.module else
                     [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                if name.startswith("research_experiments."):
                    target = Path(*name.split(".")).with_suffix(".py")
                    if (root/target).is_file():
                        pending.append(target)
    found.update(Path("research_experiments/runtime_snapshots")/p for p in (
        "source_ac288_72af_20261018.zip", "source_ac288_72af_20261018.manifest.json"))
    return {p.as_posix(): sha256(root/p) for p in sorted(found)}


def prepare(args):
    output, dataset, runtime = args.output_dir.resolve(), args.dataset_dir.resolve(), args.runtime_dir.resolve()
    from research_experiments.rnj.prepare_source_ac288_runtime import ensure_runtime
    ensure_runtime(runtime)
    manifest = read_json(dataset/"manifest.json")
    if manifest.get("completed") != 16 or manifest.get("failed") != 0:
        raise ValueError("expected all original sixteen AC conditions")
    settings = manifest["settings"]
    if any(settings[k] != v for k, v in {"t_count": 96, "v_noise_rel": .0002,
        "root_meter_noise_rel": .0002, "pq_noise_rel": .005}.items()):
        raise ValueError("original 96-point measurement precision required")
    output.mkdir(parents=True, exist_ok=True)
    baseline_reuse = reuse_baselines(args.baseline_dir.resolve(), dataset, output)
    protocol = plain({"schema_version": 5, "suite": "rnj_blocks96", "jobs": job_catalog(), "expected_jobs": 64,
        "main_jobs": 32, "configuration_control_jobs": 32, "cases": CASES, "variants": VARIANTS, "methods": METHODS,
        "reused_baseline_methods": ["NJ", "RNJ"], "baseline_reuse": "separate input/producer audit before comparison",
        "baseline_evidence": baseline_reuse,
        "baseline_jobs": baseline_reuse["baseline_jobs"],
        "total_samples": 96, "training_samples": 96, "validation_samples": 0, "validation_supplied": False,
        "independent_test_samples": 0, "selection": "last_certified_path_point_without_validation",
        "scenario_count": 1, "samples_per_scenario": 96, "observed_root_label": 1,
        "source_information": "exact P/Q schedules; unknown common physical location",
        "core_parameters_rule": "CORE_PARAMETERS with max_atoms=2*n-1", "core_parameters_base": CORE_PARAMETERS,
        "screen_parameters": SCREEN_PARAMETERS, "minimum_atomic_group_size": 2,
        "rnj_atomic_block_rule": "complete direct terminal peers; retain full-fine veto and subfrequency <= .15",
        "rx75_weight": .75, "tree_tolerance_factor": .16, "resolution_absolute_tolerance": 1e-10,
        "calibration": "pre-tree ordered/FWL matrices from same 96 observations; factor .16; no truth",
        "floor": "old non-singletons zero or a*r+b*x>=tau; new >=tau and >=2 original terminals; leaves exempt",
        "source_path": "complete original h constraints unchanged; source amplitude counted once",
        "certificate_scope": "bounded single-new-atom floor domain; greedy whole path",
        "configuration_controls": {"ungrouped": "same selected initial supports, no indivisible groups",
            "tree_init": "same selected atomic groups, add every observed RNJ clade as ordinary initial support",
            "grouping_scope": "groups with >=3 terminals restrict subdivisions as well as reduce binaries"},
        "primary_topology_metric": "resolution_clade_f1", "retained_metrics": ["family", "active", "resolution", "R", "X"],
        "solver": args.solver, "concurrency": args.workers, "solver_threads": 1, "blas_threads": 1,
        "hard_timeout_grace_seconds": 30, "truth_used_for_selection": False,
        "dataset_manifest_sha256": sha256(dataset/"manifest.json"), "runtime_manifest_sha256": sha256(runtime/"runtime_manifest.json"),
        "design_protocol_path": DESIGN_PROTOCOL.as_posix(), "design_protocol_sha256": sha256(ROOT/DESIGN_PROTOCOL),
        "runtime_directory": str(runtime), "source_sha256": source_closure(),
        "timing_environment": {"exclusive": False, "logical_cpu_count": os.cpu_count(), "new_workers": args.workers,
            "known_overlapping_experiment": "source_ac288_benchmark_20261018; up to two workers"},
        "measurement_settings": settings, "inference_scope": "diagnostic reuse of existing data; no independent test advantage claim"})
    protocol_path = output/"protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError("protocol changed; preserve this directory and choose another")
    write_json(protocol_path, protocol)
    signature = sha256(protocol_path)
    for relative, checksum in {**protocol["source_sha256"], DESIGN_PROTOCOL.as_posix(): protocol["design_protocol_sha256"]}.items():
        target = output/"source_snapshot"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(ROOT/relative, target)
        if sha256(target) != checksum:
            raise ValueError("frozen source changed")
    copied_manifest = output/"dataset_manifest.json"
    if not copied_manifest.exists():
        shutil.copyfile(dataset/"manifest.json", copied_manifest)
    if sha256(copied_manifest) != protocol["dataset_manifest_sha256"]:
        raise ValueError("frozen data manifest changed")
    for case in CASES:
        prefix = f"{case}/network/"
        for name, checksum in manifest["file_sha256"].items():
            if not name.startswith(prefix):
                continue
            target = output/"evaluation_network"/case/name[len(prefix):]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(dataset/name, target)
            if sha256(target) != checksum:
                raise ValueError("evaluation truth changed")
        verify_evaluation_network(output, case)
    if not (output/"environment.json").exists():
        write_json(output/"environment.json", {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "packages": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "networkx", "gurobipy")},
            "timing_environment": protocol["timing_environment"]})
    hashes = {}
    for case in CASES:
        for variant in VARIANTS:
            inventory = {}
            for relative in sorted(INPUT_NAMES):
                original = dataset/case/variant/relative
                checksum = sha256(original)
                if manifest["file_sha256"].get(original.relative_to(dataset).as_posix()) != checksum:
                    raise ValueError("original algorithm input changed")
                target = output/"inputs"/case/variant/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(original, target)
                inventory[relative] = checksum
            verify_inputs(output/"inputs"/case/variant, inventory)
            hashes[(case, variant)] = inventory
    pending = []
    for job in job_catalog():
        directory = output/"jobs"/job["job_id"]
        directory.mkdir(parents=True, exist_ok=True)
        inventory = hashes[(job["case"], job["variant"])]
        write_json(directory/"inputs_hash.json", {"file_sha256": inventory})
        write_json(directory/"payload.json", {"job": job, "config": protocol, "job_dir": directory,
            "runtime_dir": runtime, "input_dir": output/"inputs"/job["case"]/job["variant"],
            "input_sha256": inventory, "protocol_sha256": signature})
        if (directory/"result.json").exists():
            if read_json(directory/"result.json").get("protocol_sha256") != signature:
                raise ValueError("existing result protocol mismatch")
            evaluate_result(directory)
        else:
            pending.append((job, directory))
    summarize(output)
    return output, protocol, pending


def worker_command(output, payload):
    script = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('research_experiments.rnj.run_rnj_blocks96',run_name='__main__')"
    return [sys.executable, "-B", "-c", script, str((Path(output)/"source_snapshot").resolve()), "--worker", str(payload)]


def run(args):
    with controller_lock(args.output_dir):
        output, protocol, pending = prepare(args)
        if args.prepare_only:
            print(f"Prepared {len(job_catalog())} jobs; 96 fitting rows; original precision; no validation: {output}", flush=True)
            return 0
        if args.only_job:
            if not set(args.only_job) <= {j["job_id"] for j in job_catalog()}:
                raise ValueError("unknown job id")
            pending = [(j, d) for j, d in pending if j["job_id"] in args.only_job]
        active = []
        try:
            while pending or active:
                while pending and len(active) < args.workers:
                    job, directory = pending.pop(0)
                    stdout = (directory/"stdout.log").open("a", encoding="utf-8")
                    stderr = (directory/"stderr.log").open("a", encoding="utf-8")
                    process = subprocess.Popen(worker_command(output, directory/"payload.json"),
                        cwd=output/"source_snapshot", env=os.environ.copy(), stdout=stdout, stderr=stderr)
                    active.append((job, directory, process, perf_counter(), stdout, stderr))
                    print(f"START {job['job_id']} pid={process.pid}", flush=True)
                for item in list(active):
                    job, directory, process, launched, stdout, stderr = item
                    killed = process.poll() is None and perf_counter()-launched > job["method_wall_seconds"]+30
                    if killed:
                        process.kill(); process.wait()
                    if process.poll() is None:
                        continue
                    stdout.close(); stderr.close()
                    recover_process(directory, job, sha256(output/"protocol.json"), perf_counter()-launched,
                                    killed=killed, exitcode=process.returncode)
                    result = evaluate_result(directory)
                    print(f"END {job['job_id']} {result['status']} F1={result['metrics']['clade_f1']} seconds={result['compute_seconds']:.2f}", flush=True)
                    active.remove(item)
                    summarize(output)
                if active:
                    sleep(.5)
        finally:
            for job, directory, process, launched, stdout, stderr in active:
                if process.poll() is None:
                    process.kill(); process.wait()
                stdout.close(); stderr.close()
                recover_process(directory, job, sha256(output/"protocol.json"), perf_counter()-launched,
                                killed=True, exitcode=process.returncode)
                evaluate_result(directory)
            summarize(output)
        completion = read_json(output/"completion.json")
        if completion["finished"] == completion["expected"]:
            write_json(output/"execution_complete.json", {"completed_utc": datetime.now(timezone.utc).isoformat(), "completion": completion})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT/"outputs/source_ac96_20261018")
    parser.add_argument("--runtime-dir", type=Path, default=ROOT/"outputs/source_ac288_runtime_20261018")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT/"outputs/source_ac96_benchmark_20261018")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/rnj_blocks96_20261019")
    parser.add_argument("--solver", choices=("gurobi", "highs"), default="gurobi")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--only-job", nargs="+")
    args = parser.parse_args()
    return worker(args.worker) if args.worker else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
