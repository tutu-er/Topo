"""Paired Soumalas11 comparison with a fixed, observation-calibrated RX75 floor.

All 288 rows fit every method. Workers execute the frozen experiment closure,
read only copied observations/source schedules, and retain the last certified
path point. Only the controller evaluates physical truth after a worker exits.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
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

from research_experiments.rnj.run_source_ac288 import (
    CORE_PARAMETERS, SCREEN_PARAMETERS, BudgetedSolver, activate_runtime,
    input_arrays, select_snapshot_model, save_model,
)
from research_experiments.rnj.run_source_ac96_full_history import (
    plain, read_json, write_json, sha256, _complete_stop,
)
from research_experiments.rnj.rx75_edge_floor import (
    calibrate_rx75_floor, classify_rx75_atoms, enforce_rx75_floor,
)

ROOT = Path(__file__).resolve().parents[2]
CASE = "soumalas11"
DESIGN_PROTOCOL = Path("docs/wzzt_source_advantage_protocol_20261018.md")
VARIANTS = ("no_source", "rank2_source")
METHODS = ("NJ", "RNJ", "WZZT", "RNJ+WZZT", "RNJ+WZZT_tree_init")
REPLICATES = (0, 1, 2)
RESOLUTION_TOLERANCE = 1e-10
METRIC_KEYS = tuple(f"{prefix}_{key}" for prefix in ("family", "active", "resolution")
                    for key in ("clade_f1", "precision", "recall")) + (
                        "clade_f1", "r_relative_error", "x_relative_error")


def job_catalog():
    conditions = [(r, v) for r in REPLICATES for v in VARIANTS]
    order = [(m, p) for m in ("NJ", "RNJ") for p in conditions]
    priority = (0, "rank2_source")
    order += [(m, priority) for m in ("RNJ+WZZT_tree_init", "RNJ+WZZT", "WZZT")]
    order += [(m, p) for m in ("RNJ+WZZT", "WZZT", "RNJ+WZZT_tree_init")
              for p in conditions if p != priority]
    return [{"case": CASE, "replicate": r, "variant": v, "method": method,
             "role": "initializer_ablation" if method.endswith("_tree_init") else "main",
             "screen_parameters": {**SCREEN_PARAMETERS, "seed": SCREEN_PARAMETERS["seed"]+r},
             "per_solve_seconds": 6000. if "WZZT" in method else None,
             "method_wall_seconds": 30000. if "WZZT" in method else 300.,
             "job_id": f"{CASE}__replicate_{r}__{v}__{method}"}
            for method, (r, v) in order]


def core_parameters(terminal_count):
    return {**CORE_PARAMETERS, "max_atoms": 2*int(terminal_count)-1}


def resolution_tree(edges, root, labels, calibration):
    """Contract only edges to hidden children shorter than the fixed floor.

    Terminal leaf edges retain their original domain. A root stem can contract;
    reaching tau is retained, with the same absolute tolerance as atom scoring.
    """
    edges = [(int(a), int(b), float(w)) for a, b, w in edges]
    labels, root = set(map(int, labels)), int(root)
    adjacency = {}
    for a, b, weight in edges:
        adjacency.setdefault(a, []).append((b, weight))
        adjacency.setdefault(b, []).append((a, weight))
    parent, order = {root: None}, [root]
    for node in order:
        for child, _ in adjacency.get(node, ()):
            if child == parent[node]:
                continue
            if child in parent:
                raise ValueError("baseline edges are not a tree")
            parent[child] = node
            order.append(child)
    if set(parent) != set(adjacency) or not labels <= set(parent):
        raise ValueError("baseline tree is disconnected or misses terminals")
    representative = {root: root}
    compact, contracted = [], []
    weights = {frozenset((a, b)): w for a, b, w in edges}
    for node in order[1:]:
        p = parent[node]
        weight = weights[frozenset((p, node))]
        if node not in labels and weight + RESOLUTION_TOLERANCE < calibration["tau"]:
            representative[node] = representative[p]
            contracted.append([p, node, weight])
        else:
            representative[node] = node
            compact.append([representative[p], node, weight])
    descendants = {node: ({node} if node in labels else set()) for node in order}
    for node in reversed(order[1:]):
        descendants[parent[node]].update(descendants[node])
    clades = sorted({tuple(sorted(descendants[node])) for node in order[1:]
                     if representative[node] == node and 1 < len(descendants[node]) < len(labels)})
    return {"root": root, "terminals": sorted(labels), "edges": compact,
            "contracted_edges": contracted, "clades": clades,
            "tau": calibration["tau"], "absolute_tolerance": RESOLUTION_TOLERANCE,
            "equality": "strength + tolerance >= tau retained"}


def with_resolution(model, calibration):
    model = dict(model)
    model.update(rx75_calibration=calibration, resolution_absolute_tolerance=RESOLUTION_TOLERANCE,
                 primary_topology_metric="resolution_clade_f1")
    if model.get("support_indices") is not None:
        classified = classify_rx75_atoms(model["support_indices"], model["r_values"], model["x_values"],
                                         calibration, absolute_tolerance=RESOLUTION_TOLERANCE)
        labels = model["terminal_labels"]
        model["resolution_atoms"] = classified
        model["resolution_clades"] = [[labels[i] for i in model["support_indices"][k]]
            for k in classified["retained_indices"] if 1 < len(model["support_indices"][k]) < len(labels)]
    else:
        model.setdefault("resolution_clades", [])
    return model


def _components(runtime):
    """Import estimators only after selecting the exact frozen package."""
    activate_runtime(runtime)
    from rnj_wzzt.estimation import laminar_l1_milp as core, laminar_l1_source as source
    from rnj_wzzt.estimation.unmetered_inputs import SourceInputs, prepare_source_inputs
    from rnj_wzzt.pipeline import _select_boundary_blocks
    from research_experiments.rnj.prepare_source_advantage import load_algorithm_inputs
    from research_experiments.rnj.source_ac96_trees import fit_tree
    return core, source, SourceInputs, prepare_source_inputs, _select_boundary_blocks, load_algorithm_inputs, fit_tree


def verify_experiment_modules():
    origins = {}
    for name, module in sys.modules.copy().items():
        if name.startswith("research_experiments.") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(ROOT):
                raise RuntimeError(f"experiment helper escaped the selected source closure: {name}: {path}")
            origins[name] = str(path)
    return origins


def verify_inputs(directory, hashes):
    directory = Path(directory)
    actual = {p.relative_to(directory).as_posix(): sha256(p) for p in directory.rglob("*") if p.is_file()}
    if actual != hashes:
        raise ValueError("copied algorithm inputs changed or contain unexpected files")
    for relative in actual:
        parts = Path(relative).parts
        if len(parts) != 3 or parts[0] not in {f"scenario_{i}" for i in range(3)} or parts[1] not in {"observed", "source"}:
            raise ValueError("algorithm input whitelist violation")


def _atomic_copy(source, target):
    source, target = Path(source), Path(target)
    temporary = target.with_suffix(target.suffix+".copying")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def save_checkpoint(directory, model, arrays, started, sequence):
    """Commit a paired model before publishing the recoverable pointer.

    A process can die between the two ordinary model-file replacements. The
    immutable checkpoint pair and atomic pointer make outer recovery unambiguous.
    """
    directory = Path(directory)
    checkpoint = directory/"certified_models"/f"{sequence:06d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    saved = save_model(checkpoint, model, arrays, started)
    if not saved["has_model"]:
        return saved
    write_json(directory/"last_certified_model.json", {"sequence": sequence,
        "files": {p.relative_to(directory).as_posix(): sha256(p)
                  for p in (checkpoint/"model.json", checkpoint/"model.npz")}})
    _atomic_copy(checkpoint/"model.npz", directory/"model.npz")
    _atomic_copy(checkpoint/"model.json", directory/"model.json")
    return saved


def worker(payload_path):
    payload = read_json(payload_path)
    directory, job, config = Path(payload["job_dir"]), payload["job"], payload["config"]
    verify_inputs(payload["input_dir"], payload["input_sha256"])
    core, source_module, source_class, prepare_source, selector, loader, fit_tree = _components(payload["runtime_dir"])
    loading_started = perf_counter()
    scenarios, sources, metadata = loader(payload["input_dir"], source_inputs_class=source_class)
    arrays, data_use = input_arrays(scenarios, sources)
    if metadata["input_file_sha256"] != payload["input_sha256"]:
        raise ValueError("loader did not use exactly the frozen observation/source files")
    write_json(directory/"data_use.json", {**data_use, "loader_metadata": metadata})
    started = perf_counter()
    timing = {"input_loading_seconds": started-loading_started}
    write_json(directory/"execution.json", {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
               "method_started_monotonic": started, "job": job, "timing_environment": config["timing_environment"]})
    labels = [int(v) for v in data_use["terminal_labels"]]
    parameters = core_parameters(len(labels))
    if parameters != config["core_parameters"]:
        raise ValueError("frozen terminal count and core parameters disagree")
    state = {"terminal_labels": labels, "path": (), "attempted_extensions": (), "initial_supports": (),
             "requested_terminal_groups": (), "terminal_groups": tuple((i,) for i in range(len(labels))),
             "r_upper_bound": 2., "x_upper_bound": 2., "bound_expansions": 0}
    progress, events, selection, calibration = [], [], None, None
    status, reason, error = "failed", "not_started", None
    try:
        front_started = perf_counter()
        front_method = job["method"] if job["method"] in ("NJ", "RNJ") else (
            "RNJ" if job["method"].endswith("_tree_init") else "NJ")
        front = fit_tree(scenarios, sources, front_method, root_bus=1)
        timing["front_end_fit_seconds"] = perf_counter()-front_started
        timing["calibration_dense_seconds"] = timing["front_end_fit_seconds"]
        calibration_started = perf_counter()
        calibration = calibrate_rx75_floor(front["R"], front["X"], factor=.16)
        timing["calibration_seconds"] = perf_counter()-calibration_started
        arrays.update(calibration_R=front["R"], calibration_X=front["X"],
                      calibration_source_g=front["source_response_g"])
        write_json(directory/"rx75_calibration.json", {**calibration, "matrix_origin": "pre_tree_ordered_FWL_regression",
                   "truth_used": False, "front_end_method": front["method"], "timing": timing,
                   "regression_diagnostics": front["diagnostics"]})
        if job["method"] in ("NJ", "RNJ"):
            tree = plain(front["tree"])
            resolved = resolution_tree(tree["edges"], 1, labels, calibration)
            model = {"schema_version": 4, "terminal_labels": labels, "R": front["R"], "X": front["X"],
                     "family_clades": front["clades"], "active_clades": front["clades"], "clades": front["clades"],
                     "resolution_clades": resolved["clades"], "resolution_tree": resolved,
                     "source_h": None, "source_g_r": front["source_response_g"][0],
                     "source_g_x": front["source_response_g"][1], "support_indices": None,
                     "r_values": None, "x_values": None, "path": [], "attempted_extensions": [],
                     "solver_events": [], "progress_events": [], "initial_supports": [], "terminal_groups": [],
                     "requested_terminal_groups": [], "selected_path_index": None, "last_path_index": None,
                     "r_upper_bound": None, "x_upper_bound": None, "bound_expansions": 0,
                     "training_rows": 288, "validation_supplied": False, "validation_mae": None, "validation_se": None,
                     "regression_diagnostics": front["diagnostics"], "tree": tree, "front_end_timing": front["timing"],
                     "matrix_origin": "source_adjusted_ordered_regression_before_tree_inference",
                     "selection": "tree_on_all_288_rows"}
            status, reason = "complete", "tree_inference_completed"
        else:
            prepared_source = prepare_source(scenarios, sources)
            initial, groups = [(i,) for i in range(len(labels))], []
            if job["method"].startswith("RNJ+WZZT"):
                diagnostics, screen_started = {}, perf_counter()
                full, confidence, selected, r2, condition = selector(scenarios, labels, 1,
                    **job["screen_parameters"], source_inputs=prepared_source, diagnostics=diagnostics)
                if "subclade_frequency" in diagnostics:
                    diagnostics["subclade_frequency_records"] = [{"clade": sorted(c), "frequency": f}
                        for c, f in sorted(diagnostics.pop("subclade_frequency").items(), key=lambda pair: tuple(sorted(pair[0])))]
                positions = {label: i for i, label in enumerate(labels)}
                supports = [tuple(positions[v] for v in sorted(c)) for c in selected]
                initial += supports
                groups = [block for block in supports if len(block) >= 3]
                timing["bootstrap_seconds"] = perf_counter()-screen_started
                selection = {"selector": "frozen_runtime.rnj_wzzt.pipeline._select_boundary_blocks", "full_clades": full,
                    "confidence_records": [{"clade": sorted(c), "frequency": f} for c, f in sorted(confidence.items(), key=lambda p: tuple(sorted(p[0])))],
                    "selected_support_labels": [sorted(c) for c in selected], "selected_support_indices": supports,
                    "requested_terminal_groups": groups, "diagnostics": diagnostics, "parameters": job["screen_parameters"],
                    "r2_score": r2, "condition": condition, "training_sample_count": 288, "validation_used": False,
                    "elapsed_seconds": timing["bootstrap_seconds"], "truth_used_for_selection": False,
                    "empty_selection_retained": not bool(selected)}
                write_json(directory/"rnj_screen.json", selection)
            if job["method"].endswith("_tree_init"):
                positions = {label: i for i, label in enumerate(labels)}
                tree_supports = [tuple(sorted(positions[label] for label in clade)) for clade in full]
                initial = sorted(set(initial+tree_supports), key=lambda support: (len(support), support))
                write_json(directory/"tree_initialization.json", {"policy": "all same-call screen RNJ nontrivial clades plus singleton and strict selected supports",
                    "source": "same _select_boundary_blocks call as strict selected groups; data only", "tree_clades": full,
                    "front_fit_tree_clades": front["clades"],
                    "front_and_screen_clades_equal": {frozenset(c) for c in front["clades"]} == {frozenset(c) for c in full},
                    "tree_support_indices": tree_supports,
                    "initial_supports": initial, "requested_terminal_groups": groups, "truth_used": False,
                    "crossing_policy": "retain explicit failed result if the union is nonlaminar; no truth-based repair",
                    "limitation": "zero old supports retain unconditional laminar compatibility constraints"})
            state.update(initial_supports=initial, requested_terminal_groups=groups,
                         terminal_groups=core.normalize_terminal_groups(groups, len(labels)).groups)
            fit_started = perf_counter()
            with enforce_rx75_floor(core, source_module, calibration), BudgetedSolver(core, directory, started, job) as observer:
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
                    partial.update(bootstrap=selection, timing=timing, schema_version=4)
                    save_checkpoint(directory, with_resolution(partial, calibration), arrays, started, sequence)
                fit = core.fit_laminar_l1_sensitivity(scenarios, validation_scenarios=None,
                    initial_supports=initial, terminal_groups=groups if job["method"].startswith("RNJ+WZZT") else None,
                    source_inputs=prepared_source, **parameters, time_limit=job["per_solve_seconds"],
                    solver=config["solver"], progress_callback=checkpoint)
                reason = fit.stop_reason
                status = "complete" if _complete_stop(reason) else "incomplete"
                model = select_snapshot_model(state, solver_events=events, progress_events=progress, status=status, stop_reason=reason)
                if model["selected_path_index"] != fit.selected_path_index:
                    raise RuntimeError("saved last certified point differs from core selection")
            timing["wzzt_fit_seconds"] = perf_counter()-fit_started
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        timed_out = isinstance(exc, TimeoutError) or bool(events and events[-1].get("status") == 1)
        status, reason = ("incomplete" if timed_out else "failed"), str(exc)
        model = select_snapshot_model(state, solver_events=events, progress_events=progress, status=status, stop_reason=reason)
        write_json(directory/"error.json", error)
    model.update(schema_version=4, status=status, stop_reason=reason, bootstrap=selection, budget=job,
                 core_parameters=parameters if "WZZT" in job["method"] else None, timing=timing)
    if calibration is not None:
        model = with_resolution(model, calibration)
    from research_experiments.rnj.prepare_source_ac288_runtime import verify_loaded_modules
    write_json(directory/"runtime_import.json", {"runtime_dir": payload["runtime_dir"],
               "module_origins": verify_loaded_modules(payload["runtime_dir"]),
               "experiment_source_root": ROOT, "driver_file": __file__,
               "experiment_module_origins": verify_experiment_modules()})
    saved = save_model(directory, model, arrays, started)
    write_json(directory/"result.json", {**job, "status": status, "stop_reason": reason, "search_complete": status == "complete",
               "has_model": saved["has_model"], "elapsed_seconds": perf_counter()-started, "timing": timing,
               "protocol_sha256": payload["protocol_sha256"], "error": error, "truth_read_by": "parent_evaluator_only"})
    return 0


def evaluate_result(directory, dataset):
    from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth, clade_metrics, relative_error
    directory, dataset = Path(directory), Path(dataset)
    result, model = read_json(directory/"result.json"), read_json(directory/"model.json")
    metrics = dict.fromkeys(METRIC_KEYS)
    if model.get("has_model"):
        truth = physical_truth(verify_evaluation_network(directory.parent.parent), model["terminal_labels"])
        for prefix in ("family", "active", "resolution"):
            values = clade_metrics(model[prefix+"_clades"], truth["clades"], model["terminal_labels"])
            for key in ("clade_f1", "precision", "recall"):
                metrics[f"{prefix}_{key}"] = values[key]
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


def freeze_evaluation_network(output, dataset, manifest):
    """Truth stays in the controller's evaluation folder, outside worker inputs."""
    output, dataset = Path(output), Path(dataset)
    prefix = f"{CASE}/network/"
    expected = {name[len(prefix):]: checksum for name, checksum in manifest["file_sha256"].items() if name.startswith(prefix)}
    source = dataset/CASE/"network"
    actual = {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob("*") if p.is_file()}
    if not expected or actual != expected:
        raise ValueError("physical evaluation network differs from dataset manifest")
    for relative, checksum in expected.items():
        target = output/"evaluation_network"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source/relative, target)
        if sha256(target) != checksum:
            raise ValueError("frozen evaluation network changed")
    verify_evaluation_network(output)


def verify_evaluation_network(output):
    output = Path(output)
    protocol = read_json(output/"protocol.json")
    manifest_path = output/"dataset_manifest.json"
    if sha256(manifest_path) != protocol["dataset_manifest_sha256"]:
        raise ValueError("frozen dataset manifest changed before evaluation")
    prefix = f"{CASE}/network/"
    expected = {name[len(prefix):]: checksum for name, checksum in read_json(manifest_path)["file_sha256"].items() if name.startswith(prefix)}
    network = output/"evaluation_network"
    actual = {p.relative_to(network).as_posix(): sha256(p) for p in network.rglob("*") if p.is_file()}
    if not expected or actual != expected:
        raise ValueError("frozen evaluation network changed before evaluation")
    return network


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
    counts = dict(Counter(row["status"] for row in rows))
    completion = {"expected": len(rows), "finished": sum(row["status"] in {"complete", "incomplete", "failed"} for row in rows),
                  "status_counts": counts, "models_available": sum(bool(row.get("has_model")) for row in rows)}
    write_json(output/"completion.json", completion)
    return rows


def source_closure(root=ROOT):
    """Trace local experiment imports without importing any estimator."""
    root = Path(root)
    pending = [Path(__file__).resolve().relative_to(ROOT),
               Path("research_experiments/rnj/prepare_source_advantage.py"),
               Path("research_experiments/rnj/source_ac96_trees.py"),
               Path("research_experiments/rnj/prepare_source_ac288_runtime.py"),
               Path("research_experiments/rnj/audit_source_ac96_benchmark.py")]
    found = set()
    while pending:
        relative = pending.pop()
        if relative in found:
            continue
        path = root/relative
        if not path.is_file():
            raise ValueError(f"experiment source dependency missing: {relative}")
        found.add(relative)
        for parent in relative.parents:
            init = parent/"__init__.py"
            if parent != Path(".") and (root/init).is_file() and init not in found:
                pending.append(init)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                     [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                if name.startswith("research_experiments."):
                    target = Path(*name.split(".")).with_suffix(".py")
                    if (root/target).is_file():
                        pending.append(target)
    found.update(Path("research_experiments/runtime_snapshots")/name for name in (
        "source_ac288_72af_20261018.zip", "source_ac288_72af_20261018.manifest.json"))
    return {p.as_posix(): sha256(root/p) for p in sorted(found)}


@contextmanager
def controller_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output/"controller.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"controller lock exists: {lock}; verify its PID/host before removing a stale lock") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            import json
            json.dump({"pid": os.getpid(), "host": platform.node(), "started_utc": datetime.now(timezone.utc).isoformat()}, stream)
        yield
    finally:
        lock.unlink()


def prepare(args):
    output, dataset, runtime = args.output_dir.resolve(), args.dataset_dir.resolve(), args.runtime_dir.resolve()
    from research_experiments.rnj.prepare_source_ac288_runtime import ensure_runtime
    ensure_runtime(runtime)
    manifest = read_json(dataset/"manifest.json")
    if manifest.get("completed") != 18 or manifest.get("failed") != 0:
        raise ValueError("source data must retain all eighteen successful AC scenarios")
    output.mkdir(parents=True, exist_ok=True)
    protocol = plain({"schema_version": 4, "suite": "soumalas11", "case": CASE, "jobs": job_catalog(), "expected_jobs": 30,
        "main_jobs": 24, "initializer_ablation_jobs": 6,
        "variants": VARIANTS, "replicates": REPLICATES, "methods": METHODS, "terminal_count": 11,
        "total_samples": 288, "training_samples": 288, "validation_samples": 0, "validation_supplied": False,
        "independent_test_samples": 0, "selection": "last_certified_path_point_without_validation",
        "scenario_count": 3, "samples_per_scenario": 96, "observed_root_label": 1,
        "source_information": "exact P/Q schedules; unknown shared physical location",
        "core_parameters": core_parameters(11), "screen_parameters": SCREEN_PARAMETERS,
        "minimum_atomic_group_size": 3, "rnj_atomic_block_rule": "complete_direct_terminal_children",
        "rx75_weight": .75, "tree_tolerance_factor": .16, "resolution_absolute_tolerance": RESOLUTION_TOLERANCE,
        "baseline_raw_tree": "fit_tree output unchanged, including original NJ <= collapse and RNJ tolerance; extra resolution contraction retains equality within 1e-10 and cannot restore already removed raw edges",
        "calibration": "fixed per job from pre-tree ordered/FWL dense matrices, factor .16; no truth or final WZZT matrix",
        "floor": "old non-singleton: zero or a*r+b*x>=tau; new: >=tau and at least two original terminals; singleton domain unchanged",
        "source_path": "complete original h constraints unchanged, including zero-response paths",
        "certificate_scope": "current bounded grouped single-new-atom floor domain; whole path remains greedy",
        "tree_initialization_ablation": "all full_clades from the same strict bootstrap selector call as ordinary supports; only strict selected groups are atomic; nonlaminar union retained as failure",
        "primary_topology_metric": "resolution_clade_f1", "retained_metrics": ["family", "active", "resolution", "R", "X"],
        "solver": args.solver, "concurrency": args.workers, "solver_threads": 1, "blas_threads": 1,
        "hard_timeout_grace_seconds": 30, "truth_used_for_selection": False,
        "dataset_manifest_sha256": sha256(dataset/"manifest.json"), "runtime_manifest_sha256": sha256(runtime/"runtime_manifest.json"),
        "design_protocol_path": DESIGN_PROTOCOL.as_posix(), "design_protocol_sha256": sha256(ROOT/DESIGN_PROTOCOL),
        "runtime_directory": str(runtime), "source_sha256": source_closure(),
        "timing_environment": {"exclusive": False, "logical_cpu_count": os.cpu_count(), "new_workers": args.workers,
            "known_overlapping_experiment": "source_ac288_benchmark_20261018; two workers may remain active",
            "nominal_simultaneous_experiment_processes": args.workers+2}})
    protocol_path = output/"protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError("frozen protocol changed; preserve this directory and choose another")
    write_json(protocol_path, protocol)
    signature = sha256(protocol_path)
    for relative, checksum in protocol["source_sha256"].items():
        target = output/"source_snapshot"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(ROOT/relative, target)
        if sha256(target) != checksum:
            raise ValueError("frozen source snapshot changed")
    design_copy = output/"source_snapshot"/DESIGN_PROTOCOL
    design_copy.parent.mkdir(parents=True, exist_ok=True)
    if not design_copy.exists():
        shutil.copyfile(ROOT/DESIGN_PROTOCOL, design_copy)
    if sha256(design_copy) != protocol["design_protocol_sha256"]:
        raise ValueError("frozen design protocol changed")
    copied_manifest = output/"dataset_manifest.json"
    if not copied_manifest.exists():
        shutil.copyfile(dataset/"manifest.json", copied_manifest)
    if sha256(copied_manifest) != protocol["dataset_manifest_sha256"]:
        raise ValueError("frozen dataset manifest changed")
    freeze_evaluation_network(output, dataset, manifest)
    if not (output/"environment.json").exists():
        write_json(output/"environment.json", {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "networkx", "gurobipy")},
            "timing_environment": protocol["timing_environment"], "thread_limits": {k: os.environ[k] for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}})
    hashes = {}
    for replicate in REPLICATES:
        for variant in VARIANTS:
            relative_condition = Path(CASE)/f"replicate_{replicate}"/variant
            source, copied = dataset/relative_condition, output/"inputs"/relative_condition
            inventory = {}
            for i in range(3):
                for folder in ("observed", "source"):
                    paths = sorted((source/f"scenario_{i}"/folder).iterdir())
                    if not paths:
                        raise ValueError("empty input folder")
                    for path in paths:
                        if not path.is_file():
                            raise ValueError("input folders must contain only files")
                        relative = path.relative_to(source)
                        checksum = sha256(path)
                        if manifest["file_sha256"].get(path.relative_to(dataset).as_posix()) != checksum:
                            raise ValueError("source input differs from dataset manifest")
                        target = copied/relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if not target.exists():
                            shutil.copyfile(path, target)
                        inventory[relative.as_posix()] = checksum
            verify_inputs(copied, inventory)
            hashes[(replicate, variant)] = inventory
    pending = []
    for job in job_catalog():
        directory = output/"jobs"/job["job_id"]
        directory.mkdir(parents=True, exist_ok=True)
        inventory = hashes[(job["replicate"], job["variant"])]
        write_json(directory/"inputs_hash.json", {"file_sha256": inventory})
        write_json(directory/"payload.json", {"job": job, "config": protocol, "job_dir": directory,
            "runtime_dir": runtime, "input_dir": output/"inputs"/CASE/f"replicate_{job['replicate']}"/job["variant"],
            "input_sha256": inventory, "protocol_sha256": signature})
        if (directory/"result.json").exists():
            if read_json(directory/"result.json").get("protocol_sha256") != signature:
                raise ValueError("existing result protocol mismatch")
            evaluate_result(directory, dataset)
        else:
            pending.append((job, directory))
    summarize(output)
    return output, dataset, protocol, pending


def worker_command(output, payload):
    snapshot = str((Path(output)/"source_snapshot").resolve())
    # -c avoids executing the live driver; every local experiment import comes
    # from the checked closure. Estimator imports then use the frozen runtime.
    script = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('research_experiments.rnj.run_rx75_source_comparison',run_name='__main__')"
    return [sys.executable, "-B", "-c", script, snapshot, "--worker", str(payload)]


def recover_process(directory, job, protocol_hash, elapsed, *, killed, exitcode):
    directory = Path(directory)
    if (directory/"result.json").exists():
        return
    pointer = directory/"last_certified_model.json"
    if pointer.exists():
        record = read_json(pointer)
        paths = [directory/relative for relative in record["files"]]
        if len(paths) != 2 or {p.name for p in paths} != {"model.json", "model.npz"}:
            raise ValueError("invalid certified checkpoint pointer")
        for relative, checksum in record["files"].items():
            path = (directory/relative).resolve()
            if not path.is_relative_to((directory/"certified_models").resolve()) or sha256(path) != checksum:
                raise ValueError("certified checkpoint pair changed")
            _atomic_copy(path, directory/path.name)
    model = read_json(directory/"model.json") if (directory/"model.json").exists() else {"has_model": False}
    status, reason = ("incomplete", "process_wall_limit") if killed else ("failed", "worker_process_failed")
    model.update(status=status, stop_reason=reason)
    write_json(directory/"model.json", model)
    write_json(directory/"result.json", {**job, "status": status, "stop_reason": reason, "search_complete": False,
        "has_model": model.get("has_model", False), "elapsed_seconds": elapsed, "timing": model.get("timing", {}),
        "protocol_sha256": protocol_hash, "process_exitcode": exitcode, "retained_last_certified_checkpoint": bool(model.get("has_model")),
        "truth_read_by": "parent_evaluator_only"})


def run(args):
    with controller_lock(args.output_dir):
        output, dataset, protocol, pending = prepare(args)
        if args.prepare_only:
            print(f"Prepared 24 main + 6 initializer ablation jobs; 288 fitting rows, fixed RX75 floor, zero validation: {output}", flush=True)
            return 0
        if args.only_job:
            if not set(args.only_job) <= {j["job_id"] for j in job_catalog()}:
                raise ValueError("unknown job identifier")
            pending = [(j, d) for j, d in pending if j["job_id"] in args.only_job]
        active = []
        try:
            while pending or active:
                while pending and len(active) < args.workers:
                    job, directory = pending.pop(0)
                    stdout, stderr = (directory/"stdout.log").open("a", encoding="utf-8"), (directory/"stderr.log").open("a", encoding="utf-8")
                    process = subprocess.Popen(worker_command(output, directory/"payload.json"), cwd=output/"source_snapshot",
                                               env=os.environ.copy(), stdout=stdout, stderr=stderr)
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
                    recover_process(directory, job, sha256(output/"protocol.json"), perf_counter()-launched, killed=killed, exitcode=process.returncode)
                    result = evaluate_result(directory, dataset)
                    print(f"END {job['job_id']} {result['status']} resolutionF1={result['metrics']['clade_f1']} seconds={result['compute_seconds']:.2f}", flush=True)
                    active.remove(item)
                    summarize(output)
                if active:
                    sleep(.5)
        finally:
            # Interrupted controllers must not leave orphan solvers or release
            # the run lock while their children continue fitting.
            for job, directory, process, launched, stdout, stderr in active:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                stdout.close()
                stderr.close()
                recover_process(directory, job, sha256(output/"protocol.json"), perf_counter()-launched, killed=True, exitcode=process.returncode)
                evaluate_result(directory, dataset)
            summarize(output)
        write_json(output/"execution_complete.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
            "completion": read_json(output/"completion.json")})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT/"outputs/source_advantage_soumalas11_20261018")
    parser.add_argument("--runtime-dir", type=Path, default=ROOT/"outputs/source_ac288_runtime_20261018")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/rx75_source_comparison_20261018")
    parser.add_argument("--solver", choices=("gurobi", "highs"), default="gurobi")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--only-job", nargs="+")
    args = parser.parse_args()
    return worker(args.worker) if args.worker else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
