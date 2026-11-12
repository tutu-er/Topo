"""Independently audit the six pengwah18/no-source 64/32 history models.

This module does not import the benchmark driver, tree estimators, MILP core,
or any solver.  It reconstructs saved atoms and physical paths, recomputes
train/validation losses and four-block standard errors, checks the historical
one-standard-error choice, and verifies retained bound-restart checkpoints.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_experiments.rnj.audit_source_ac96_benchmark import (
    Audit, clade_metrics, optimality_certificate, physical_truth, relative_error,
)

CASE = "pengwah18"
VARIANT = "no_source"
JOBS = (
    ("NJ", None), ("RNJ", None), ("WZZT", "t1"), ("RNJ+WZZT", "t1"),
    ("WZZT", "t100"), ("RNJ+WZZT", "t100"),
)
ARRAY_NAMES = {"R", "X", "P_train", "Q_train", "Y_train", "P_validation", "Q_validation",
               "Y_validation", "P_all", "Q_all", "Y_all", "P_source_all", "Q_source_all",
               "train_positions", "validation_positions"}
CORE_PARAMETERS = {
    "max_atoms": 35, "improvement_abs_tol": 1e-10, "improvement_rel_tol": 1e-8,
    "zero_tolerance": 1e-9, "validation_blocks": 4, "max_bound_expansions": 4,
    "bound_expansion_factor": 2., "bound_active_fraction": .995, "mip_rel_gap": 0.,
    "r_upper_bound": 2., "x_upper_bound": 2., "bound_safety_factor": 2.,
    "presolve": True, "disp": False,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _job_id(method, budget):
    return f"{CASE}__{VARIANT}__{method}" + (f"__{budget}" if budget else "")


def validation_statistics(p, q, y, r, x, blocks=4):
    """Recompute the mainline contiguous-block loss/SE without refitting."""
    absolute = np.abs(y - p @ np.asarray(r).T - q @ np.asarray(x).T)
    means = np.asarray([float(absolute[positions].mean())
                        for positions in np.array_split(np.arange(len(y)), blocks) if len(positions)])
    return {"mae": float(absolute.mean()),
            "se": float(means.std(ddof=1) / np.sqrt(len(means))) if len(means) > 1 else 0.,
            "block_maes": means.tolist()}


def one_se_selection(path):
    """Choose smallest support count within best validation MAE + its own SE."""
    if not path:
        raise ValueError("a nonempty certified path is required")
    means = np.asarray([point["validation_mae"] for point in path], dtype=float)
    errors = np.asarray([point["validation_se"] for point in path], dtype=float)
    if not np.isfinite(means).all() or not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("finite validation MAE and nonnegative block SE are required")
    best = int(means.argmin())
    threshold = float(means[best] + errors[best])
    eligible = [int(i) for i in np.flatnonzero(means <= threshold)]
    selected = min(eligible, key=lambda i: (len(path[i]["support_indices"]), i))
    return {"best_validation_index": best, "best_validation_mae": float(means[best]),
            "best_validation_se": float(errors[best]), "threshold": threshold,
            "eligible_indices": eligible, "selected_path_index": selected,
            "selected_support_count": len(path[selected]["support_indices"])}


def atom_matrices(point, labels):
    n = len(labels)
    supports = [tuple(int(i) for i in support) for support in point["support_indices"]]
    sets = [frozenset(s) for s in supports]
    if (any(not s or len(s) != len(set(s)) or any(i < 0 or i >= n for i in s) for s in supports)
            or len(set(sets)) != len(sets)):
        raise ValueError("atom supports must be nonempty, unique and within terminal positions")
    if any(not (a <= b or b <= a or a.isdisjoint(b))
           for i, a in enumerate(sets) for b in sets[i + 1:]):
        raise ValueError("atom supports must be laminar")
    r, x = np.asarray(point["r_values"], dtype=float), np.asarray(point["x_values"], dtype=float)
    if r.shape != (len(supports),) or x.shape != r.shape or not np.isfinite(r).all() or not np.isfinite(x).all():
        raise ValueError("one finite R/X weight is required per support")
    z = np.zeros((len(supports), n))
    for k, support in enumerate(supports):
        z[k, list(support)] = 1.
    family = {frozenset(labels[i] for i in s) for s in supports if 1 < len(s) < n}
    active = {frozenset(labels[i] for i in s) for s, a, b in zip(supports, r, x, strict=True)
              if 1 < len(s) < n and max(a, b) > 1e-8}
    return (z.T * r) @ z, (z.T * x) @ z, family, active


def tree_clades(tree, labels):
    """Extract rooted terminal clades independently from saved tree edges."""
    root = int(tree["root"])
    adjacency = {root: set()}
    for edge in tree["edges"]:
        a, b = int(edge[0]), int(edge[1])
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    parent, order, queue = {root: None}, [], deque([root])
    while queue:
        node = queue.popleft()
        order.append(node)
        for other in sorted(adjacency[node]):
            if other == parent[node]:
                continue
            if other in parent:
                raise ValueError("saved tree contains a cycle")
            parent[other] = node
            queue.append(other)
    if len(parent) != len(adjacency) or not set(labels) <= set(parent) or root in labels:
        raise ValueError("saved tree must connect all observed terminals to the root")
    descendants = {node: ({node} if node in labels else set()) for node in parent}
    found = set()
    for node in reversed(order):
        if 1 < len(descendants[node]) < len(labels):
            found.add(frozenset(descendants[node]))
        if parent[node] is not None:
            descendants[parent[node]].update(descendants[node])
    return found


def audit_path(path, arrays, labels, initial, r_bound, x_bound, audit, prefix):
    """Check each saved accepted point and independently derive its 1SE choice."""
    initial_sets = {frozenset(s) for s in initial}
    previous = None
    for i, point in enumerate(path):
        tag = f"{prefix}: path[{i}]"
        r, x, _, _ = atom_matrices(point, labels)
        supports = point["support_indices"]
        audit.require(f"{tag} retains all fixed supports", initial_sets <= {frozenset(s) for s in supports})
        if i == 0:
            audit.require(f"{tag} initializes exactly fixed family", initial_sets == {frozenset(s) for s in supports})
        expected_labels = [[labels[j] for j in s] for s in supports]
        audit.require(f"{tag} positional supports match labels", point.get("support_labels") == expected_labels)
        for name, matrix, bound in (("r", r, r_bound), ("x", x, x_bound)):
            weights = np.asarray(point[f"{name}_values"], dtype=float)
            audit.require(f"{tag} {name} weights within current bounds",
                          np.all(weights >= -1e-8) and np.all(weights <= bound + 1e-8))
            audit.close(f"{tag} {name.upper()} reconstructed from atoms", point[f"{name}_matrix"], matrix,
                        atol=1e-11, rtol=1e-9)
        train = validation_statistics(arrays["P_train"], arrays["Q_train"], arrays["Y_train"], r, x)
        validation = validation_statistics(arrays["P_validation"], arrays["Q_validation"],
                                           arrays["Y_validation"], r, x)
        audit.close(f"{tag} training loss uses actual 64 rows", point["train_mae"], train["mae"], atol=1e-11)
        audit.close(f"{tag} validation loss uses actual held-out 32 rows", point["validation_mae"], validation["mae"], atol=1e-11)
        audit.close(f"{tag} validation SE uses four chronological blocks", point["validation_se"], validation["se"], atol=1e-11)
        audit.require(f"{tag} accepted solve has numerical optimality evidence", optimality_certificate(point["solver"]))
        if previous is not None:
            gain = float(previous) - float(point["train_mae"])
            threshold = 1e-10 + 1e-8 * max(abs(float(previous)), np.finfo(float).eps)
            audit.close(f"{tag} accepted gain reproduced", point["accepted_gain"], gain, atol=1e-10)
            audit.require(f"{tag} gain exceeds numerical tolerance", gain > threshold)
        previous = point["train_mae"]
    selected = one_se_selection(path)
    audit.require(f"{prefix}: 1SE uses fewest supports then first index",
                  selected["selected_path_index"] == min(selected["eligible_indices"],
                      key=lambda i: (len(path[i]["support_indices"]), i)))
    return selected


def audit_restart_trace(snapshots, model, arrays, labels, audit, prefix):
    """Check retained old-domain paths and exact restart after doubled bounds."""
    previous, pending = None, None
    restart_count = 0
    if snapshots:
        first = snapshots[0]
        audit.require(f"{prefix}: first checkpoint preserves original initialization",
                      first.get("event") == "initialized" and first.get("bound_expansions") == 0
                      and first.get("r_upper_bound") == 2. and first.get("x_upper_bound") == 2.
                      and len(first.get("path", [])) == 1 and not first.get("attempted_extensions"))
    for index, snapshot in enumerate(snapshots):
        tag = f"{prefix}: progress[{index}]"
        audit.require(f"{tag} preserves full certified path", isinstance(snapshot.get("path"), list) and bool(snapshot["path"]))
        audit.require(f"{tag} preserves attempted-extension list", isinstance(snapshot.get("attempted_extensions"), list))
        if not snapshot.get("path"):
            continue
        bounds = float(snapshot["r_upper_bound"]), float(snapshot["x_upper_bound"])
        chosen = audit_path(snapshot["path"], arrays, labels, snapshot["initial_supports"],
                            *bounds, audit, tag)
        audit.require(f"{tag} checkpoint itself uses historical 1SE choice",
                      snapshot["selected_path_index"] == chosen["selected_path_index"])
        if previous is not None:
            audit.require(f"{tag} never discards attempted extensions",
                          len(snapshot["attempted_extensions"]) >= len(previous["attempted_extensions"]))
        if snapshot["event"] == "bounds_expansion_requested":
            audit.require(f"{tag} preserves pre-expansion old path", bool(snapshot["path"]))
            audit.require(f"{tag} expansion has an attempted extension", bool(snapshot["attempted_extensions"]))
            pending = snapshot
        elif snapshot["event"] == "bounds_restarted":
            restart_count += 1
            audit.require(f"{tag} has saved pre-expansion checkpoint", pending is not None)
            if pending is not None:
                before = pending["attempted_extensions"][-1]
                audit.require(f"{tag} bound-triggering extension certified", optimality_certificate(before["diagnostics"]))
                for channel in ("r", "x"):
                    old = float(pending[f"{channel}_upper_bound"])
                    active = np.max(before[f"{channel}_values"], initial=0.) >= .995 * old
                    audit.close(f"{tag} only active {channel} bound doubles", snapshot[f"{channel}_upper_bound"],
                                old * (2. if active else 1.), atol=1e-12)
                audit.require(f"{tag} restart increments expansion count",
                              snapshot["bound_expansions"] == pending["bound_expansions"] + 1)
            audit.require(f"{tag} restart clears old-domain path", len(snapshot["path"]) == 1)
            audit.require(f"{tag} restart returns to original fixed family",
                          {frozenset(s) for s in snapshot["path"][0]["support_indices"]}
                          == {frozenset(s) for s in snapshot["initial_supports"]})
            pending = None
        previous = snapshot
    audit.require(f"{prefix}: all successful bound restarts retained", restart_count == model.get("bound_expansions", 0))
    if snapshots:
        final = snapshots[-1]
        audit.require(f"{prefix}: final model uses final checkpoint path", model["path"] == final["path"])
        for field in ("r_upper_bound", "x_upper_bound", "bound_expansions"):
            audit.close(f"{prefix}: final {field} matches checkpoint", model[field], final[field], atol=1e-12)
    return {"snapshots": len(snapshots), "successful_bound_restarts": restart_count,
            "pending_expansion_at_interruption": pending is not None}


def _positions_and_inputs(job_dir, source_dir, arrays, labels, audit, prefix):
    split = read_json(job_dir / "split.json")
    train, validation = list(range(64)), list(range(64, 96))
    for key, expected in (("total_samples", 96), ("training_samples", 64), ("validation_samples", 32),
                          ("training_positions", train), ("validation_positions", validation),
                          ("validation_used_for_rnj_selection", False), ("new_observations_generated", False)):
        audit.require(f"{prefix}: split {key} exactly predeclared", split.get(key) == expected)
    audit.close(f"{prefix}: saved training positions", arrays["train_positions"], train, atol=0, rtol=0)
    audit.close(f"{prefix}: saved validation positions", arrays["validation_positions"], validation, atol=0, rtol=0)
    for name, filename in (("P", "P_terminal"), ("Q", "Q_terminal"), ("Y", "drop_target")):
        frame = pd.read_csv(source_dir / "observed" / f"{filename}.csv", index_col="t", float_precision="round_trip")
        audit.require(f"{prefix}: {name} has 96 matching terminal columns", frame.shape == (96, len(labels))
                      and frame.index.is_unique and list(map(int, frame.columns)) == labels)
        audit.close(f"{prefix}: {name}_all exactly source observations", arrays[f"{name}_all"], frame.to_numpy(), atol=0, rtol=0)
        audit.close(f"{prefix}: {name}_train contains only first 64 source rows", arrays[f"{name}_train"], frame.iloc[train].to_numpy(), atol=0, rtol=0)
        audit.close(f"{prefix}: {name}_validation contains only last 32 source rows", arrays[f"{name}_validation"], frame.iloc[validation].to_numpy(), atol=0, rtol=0)
        audit.require(f"{prefix}: all row labels retained", split.get("all_time_labels") == frame.index.tolist())
        audit.require(f"{prefix}: training row labels retained", split.get("training_time_labels") == frame.index[train].tolist())
        audit.require(f"{prefix}: validation row labels retained", split.get("validation_time_labels") == frame.index[validation].tolist())
    source = pd.read_csv(source_dir / "source" / "amplitudes.csv", index_col="t", float_precision="round_trip")
    for key, column in (("P_source_all", "P_pu"), ("Q_source_all", "Q_pu")):
        audit.close(f"{prefix}: {key} matches external source amplitudes", arrays[key], source[column].to_numpy(), atol=0, rtol=0)
        audit.require(f"{prefix}: {key} all zero", arrays[key].shape == (96,) and not np.any(arrays[key] != 0.))


def audit_protocol(protocol, audit):
    """Check rules that must stay fixed while only the two time limits change."""
    audit.require("protocol case is only pengwah18", protocol.get("case") == CASE)
    audit.require("protocol variant is only no_source", protocol.get("variant") == VARIANT)
    core = protocol.get("core_parameters", {})
    for key, value in CORE_PARAMETERS.items():
        audit.require(f"mainline core parameter {key} fixed", core.get(key) == value)
    for key, value in {"total_samples": 96, "training_samples": 64, "validation_samples": 32,
                       "training_positions": list(range(64)), "validation_positions": list(range(64, 96)),
                       "observed_root_label": 1, "rx75_weight": .75, "tree_tolerance_factor": .16,
                       "bootstrap_replicates": 100, "bootstrap_block_length": 4, "bootstrap_threshold": .75,
                       "maximum_frozen_blocks": 2, "bootstrap_seed": 20260903,
                       "validation_used_for_bootstrap": False, "active_atom_tolerance": 1e-8,
                       "bootstrap_selector": "rnj_wzzt.pipeline._select_boundary_blocks",
                       "solver_threads": 1, "blas_threads": 1, "expected_jobs": 6}.items():
        audit.require(f"protocol fixed parameter {key}", protocol.get(key) == value)
    jobs = {job["job_id"]: job for job in protocol.get("jobs", [])}
    audit.require("protocol lists exactly six predeclared jobs", set(jobs) == {_job_id(m, b) for m, b in JOBS})
    for method, budget in JOBS:
        job = jobs.get(_job_id(method, budget), {})
        multiplier = {"t1": 1, "t100": 100}.get(budget)
        audit.require(f"{_job_id(method, budget)} time multiplier", job.get("budget_multiplier") == multiplier)
        audit.require(f"{_job_id(method, budget)} per-solve budget",
                      job.get("per_solve_seconds") == (None if multiplier is None else 60. * multiplier))
        audit.require(f"{_job_id(method, budget)} method wall budget",
                      job.get("method_wall_seconds") == (300. if multiplier is None else 300. * multiplier))
    return core


def _audit_screen(model, protocol, labels, audit, prefix):
    screen = model.get("bootstrap")
    audit.require(f"{prefix}: completed authoritative RNJ screening retained", isinstance(screen, dict))
    if not isinstance(screen, dict):
        return
    for key, value in (("selector", protocol["bootstrap_selector"]), ("training_sample_count", 64),
                       ("validation_used", False), ("bootstrap_replicates", 100), ("bootstrap_seed", 20260903)):
        audit.require(f"{prefix}: screening {key} fixed", screen.get(key) == value)
    clades = {frozenset(c) for c in screen["full_clades"]}
    boundary = {c for c in clades if not any(other < c for other in clades)}
    confidence = {frozenset(row["clade"]): row["frequency"] for row in screen["confidence_records"]}
    audit.require(f"{prefix}: confidence values use 100-replicate frequencies", all(
        np.isfinite(v) and 0 <= v <= 1 and abs(v * 100 - round(v * 100)) < 1e-9 for v in confidence.values()))
    expected, occupied = [], set()
    eligible = {c for c in boundary if confidence.get(c, 0.) >= .75}
    for clade in sorted(eligible, key=lambda c: (-confidence[c], -len(c), tuple(sorted(c)))):
        if len(expected) == 2:
            break
        if occupied.isdisjoint(clade):
            expected.append(clade)
            occupied.update(clade)
    audit.require(f"{prefix}: selected boundary blocks follow mainline stability/disjoint rule",
                  screen["selected_support_labels"] == [sorted(c) for c in expected])
    positions = {label: i for i, label in enumerate(labels)}
    extra = [[positions[label] for label in sorted(c)] for c in expected]
    audit.require(f"{prefix}: selected block labels map exactly to terminal positions",
                  screen["selected_support_indices"] == extra)
    audit.require(f"{prefix}: fixed supports are all leaves plus selected original-terminal blocks",
                  {frozenset(s) for s in model["initial_supports"]}
                  == {frozenset((i,)) for i in range(len(labels))} | {frozenset(s) for s in extra})


def _audit_solver_events(job_dir, model, job, audit, prefix):
    events = model.get("solver_events", [])
    if model.get("path"):
        audit.require(f"{prefix}: solver log covers initializer, all extensions and restarts",
                      len(events) >= 1 + len(model.get("attempted_extensions", []))
                      + model.get("bound_expansions", 0))
    for i, event in enumerate(events):
        tag = f"{prefix}: solver[{i}]"
        audit.require(f"{tag} sequence retained", event.get("sequence") == i + 1)
        audit.require(f"{tag} requested time is declared budget", event.get("requested_time_limit_seconds") == job["per_solve_seconds"])
        limit, remaining = event.get("time_limit_seconds"), event.get("remaining_wall_before_solve")
        audit.require(f"{tag} actual limit respects both budgets", limit is not None and remaining is not None
                      and 0 < limit <= job["per_solve_seconds"] and limit <= remaining)
        audit.require(f"{tag} gap/dual evidence matches certificate flag",
                      event.get("optimality_certified") is optimality_certificate(event))
        path = job_dir / "solver" / f"{i + 1:06d}.json"
        audit.require(f"{tag} individual solver record retained", path.is_file())
        if path.is_file():
            audit.require(f"{tag} individual record matches combined log", read_json(path) == event)
        if event.get("has_incumbent"):
            candidate = (job_dir / str(event.get("primal_file"))).resolve()
            safe = candidate.is_relative_to(job_dir.resolve()) and candidate.is_file()
            audit.require(f"{tag} raw feasible/uncertified incumbent retained", safe)
            if safe:
                with np.load(candidate, allow_pickle=False) as saved:
                    primal = saved["primal"]
                audit.require(f"{tag} raw incumbent finite and complete",
                              primal.shape == (event["variable_count"],) and np.isfinite(primal).all())
        if optimality_certificate(event):
            audit.require(f"{tag} certified solve retains its incumbent", event.get("has_incumbent") is True)


def audit_full_history(directory, dataset_directory, *, only_jobs=None):
    directory, dataset_directory = Path(directory), Path(dataset_directory)
    audit, rows, model_inputs = Audit(), [], {}
    protocol = read_json(directory / "protocol.json")
    audit_protocol(protocol, audit)
    signature = hashlib.sha256((directory / "protocol.json").read_bytes()).hexdigest()
    hashes = protocol.get("source_sha256", {})
    audit.require("frozen runtime source hashes are retained", bool(hashes))
    for relative, expected_hash in hashes.items():
        snapshot = (directory / "source_snapshot" / relative).resolve()
        safe = snapshot.is_relative_to((directory / "source_snapshot").resolve()) and snapshot.is_file()
        actual_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest() if safe else None
        audit.require(f"archived runtime source hash {relative}", safe and actual_hash == expected_hash)
    manifest = dataset_directory / "manifest.json"
    audit.require("source dataset manifest hash matches frozen protocol", manifest.is_file()
                  and hashlib.sha256(manifest.read_bytes()).hexdigest() == protocol.get("dataset_manifest_sha256"))
    protocol_jobs = {job["job_id"]: job for job in protocol.get("jobs", [])}
    expected = {_job_id(method, budget) for method, budget in JOBS}
    if only_jobs is not None:
        if not set(only_jobs) or not set(only_jobs) <= expected:
            raise ValueError("audit scope must be a nonempty subset of the six declared jobs")
        expected = set(only_jobs)
    actual = {p.name for p in (directory / "jobs").iterdir() if p.is_dir()} if (directory / "jobs").is_dir() else set()
    audit.require("all predeclared jobs retain an artifact directory", expected <= actual,
                  missing=sorted(expected - actual))
    input_directory = directory / "inputs" / f"{CASE}__{VARIANT}"
    original = dataset_directory / CASE / VARIANT
    for relative in ("observed/P_terminal.csv", "observed/Q_terminal.csv", "observed/drop_target.csv",
                     "source/amplitudes.csv"):
        copied, source = input_directory / relative, original / relative
        audit.require(f"copied observed input {relative} matches original",
                      copied.is_file() and source.is_file() and copied.read_bytes() == source.read_bytes())
    leaked = [str(p.relative_to(input_directory)) for p in input_directory.rglob("*")
              if p.name.lower() in {"truth", "network", "buses.csv", "branches.csv"}]
    audit.require("worker input copies exclude physical network/source location truth", not leaked, leaked=leaked)
    for method, budget in JOBS:
        key = _job_id(method, budget)
        if key not in expected:
            continue
        job_dir = directory / "jobs" / key
        result_path = job_dir / "result.json"
        row = {"job_id": key, "method": method, "budget": budget, "status": "missing"}
        rows.append(row)
        audit.require(f"{key}: final result or failure record retained", result_path.is_file())
        if not result_path.is_file():
            continue
        try:
            result = read_json(result_path)
            row["status"] = result["status"]
            audit.require(f"{key}: result bound to frozen protocol", result.get("protocol_sha256") == signature)
            audit.require(f"{key}: failed and incomplete states explicitly retained", result["status"] in {"complete", "incomplete", "failed"})
            audit.require(f"{key}: explicit model availability", isinstance(result.get("has_model"), bool))
            audit.require(f"{key}: finite elapsed time including failures", np.isfinite(result.get("elapsed_seconds", np.nan))
                          and result.get("elapsed_seconds", -1) >= 0)
            model_path, array_path = job_dir / "model.json", job_dir / "model.npz"
            if result.get("has_model") is False:
                audit.require(f"{key}: no-model metrics stay null even on process failure",
                              all(value is None for value in result.get("metrics", {}).values()))
            have_files = model_path.is_file() and array_path.is_file()
            audit.require(f"{key}: declared model has saved artifacts", not result.get("has_model") or have_files)
            if not have_files:
                audit.require(f"{key}: missing model has timeout/failure status", result["status"] != "complete")
                continue
            model = read_json(model_path)
            if "budget" in model:
                audit.require(f"{key}: actual model budget matches frozen job", model["budget"] == protocol_jobs[key])
            with np.load(array_path, allow_pickle=False) as saved:
                audit.require(f"{key}: exact array inventory", set(saved.files) == ARRAY_NAMES,
                              missing=sorted(ARRAY_NAMES - set(saved.files)), extra=sorted(set(saved.files) - ARRAY_NAMES))
                arrays = {name: saved[name].copy() for name in ARRAY_NAMES if name in saved.files}
            if set(arrays) != ARRAY_NAMES:
                continue
            audit.require(f"{key}: every saved array finite", all(np.isfinite(v).all() for v in arrays.values()))
            labels = list(map(int, model["terminal_labels"]))
            _positions_and_inputs(job_dir, original, arrays, labels, audit, key)
            model_inputs[key] = {name: arrays[name] for name in ARRAY_NAMES if name not in {"R", "X"}}
            if not result["has_model"]:
                audit.require(f"{key}: unavailable matrices are empty", arrays["R"].shape == (0, 0) and arrays["X"].shape == (0, 0))
                audit.require(f"{key}: unavailable metrics remain null", all(v is None for v in result.get("metrics", {}).values()))
                continue
            if method in {"NJ", "RNJ"}:
                family = active = tree_clades(model["tree"], labels)
                r, x = arrays["R"], arrays["X"]
                diagnostic = model.get("regression_diagnostics", model.get("diagnostics", {}))
                audit.require(f"{key}: no-source adapter rank is zero", diagnostic.get("source_rank") == 0)
                audit.require(f"{key}: regression adds no independent intercept", diagnostic.get("intercept_added") is False)
                design = np.column_stack((arrays["P_train"], arrays["Q_train"]))
                audit.require(f"{key}: fit diagnostic rank belongs to actual 64-row design",
                              diagnostic.get("raw_design_rank") == np.linalg.matrix_rank(design))
                audit.close(f"{key}: fit diagnostic condition belongs to actual 64-row design",
                            diagnostic.get("raw_design_condition"), np.linalg.cond(design), atol=1e-8)
                for name in ("source_g_r", "source_g_x"):
                    if name in model:
                        audit.require(f"{key}: {name} zero in no-source case", not np.any(np.asarray(model[name]) != 0.))
            else:
                audit.require(f"{key}: all effective core rules match frozen mainline", model.get("core_parameters") == protocol["core_parameters"])
                if method == "RNJ+WZZT":
                    _audit_screen(model, protocol, labels, audit, key)
                else:
                    audit.require(f"{key}: pure WZZT starts from every leaf singleton only",
                                  {frozenset(s) for s in model["initial_supports"]}
                                  == {frozenset((i,)) for i in range(len(labels))})
                _audit_solver_events(job_dir, model, protocol_jobs[key], audit, key)
                path = model.get("path", [])
                chosen = audit_path(path, arrays, labels, model["initial_supports"], model["r_upper_bound"],
                                    model["x_upper_bound"], audit, key)
                row["one_se_recomputed"] = chosen
                audit.require(f"{key}: selected path index follows historical 1SE", model["selected_path_index"] == chosen["selected_path_index"])
                point = path[chosen["selected_path_index"]]
                r, x, family, active = atom_matrices(point, labels)
                for name in ("support_indices", "support_labels", "r_values", "x_values"):
                    audit.require(f"{key}: selected {name} equals chosen path point", model[name] == point[name])
                for name in ("train_mae", "validation_mae", "validation_se"):
                    audit.close(f"{key}: selected {name} equals chosen path point", model[name], point[name], atol=1e-11)
                snapshots = [read_json(p) for p in sorted((job_dir / "progress").glob("*.json"))]
                audit.require(f"{key}: passive full-history checkpoints retained", bool(snapshots))
                row["restart_trace"] = audit_restart_trace(snapshots, model, arrays, labels, audit, key)
                for i, attempt in enumerate(model.get("attempted_extensions", [])):
                    certified = optimality_certificate(attempt["diagnostics"])
                    if not certified:
                        audit.require(f"{key}: unproved attempt[{i}] never exposes accepted support", attempt.get("support") is None)
                audit.require(f"{key}: bound expansion cap retained", 0 <= model["bound_expansions"] <= 4)
            audit.close(f"{key}: saved R equals selected reconstruction", arrays["R"], r, atol=1e-11)
            audit.close(f"{key}: saved X equals selected reconstruction", arrays["X"], x, atol=1e-11)
            for name, clades in (("clades", family), ("family_clades", family), ("active_clades", active)):
                audit.require(f"{key}: {name} independently reproduced", {frozenset(c) for c in model[name]} == clades)
            train = validation_statistics(arrays["P_train"], arrays["Q_train"], arrays["Y_train"], r, x)
            validation = validation_statistics(arrays["P_validation"], arrays["Q_validation"], arrays["Y_validation"], r, x)
            audit.close(f"{key}: selected train MAE recomputed", model.get("train_mae", model.get("train_mae_recomputed")), train["mae"], atol=1e-11)
            audit.close(f"{key}: selected validation MAE recomputed", model.get("validation_mae", model.get("validation_mae_recomputed")), validation["mae"], atol=1e-11)
            if model.get("validation_se") is not None:
                audit.close(f"{key}: selected validation SE recomputed", model["validation_se"], validation["se"], atol=1e-11)
            truth = physical_truth(dataset_directory / CASE / "network", labels)
            metrics = {}
            for kind, clades in (("family", family), ("active", active)):
                computed = clade_metrics(clades, truth["clades"], labels)
                metrics.update({f"{kind}_{m}": computed[m] for m in ("clade_f1", "precision", "recall")})
            metrics.update(clade_f1=metrics["family_clade_f1"],
                           r_relative_error=relative_error(r, truth["R"]), x_relative_error=relative_error(x, truth["X"]))
            for name, expected_metric in metrics.items():
                audit.close(f"{key}: independently scored {name}", result["metrics"].get(name), expected_metric, atol=1e-10)
            row["metrics_recomputed"] = metrics
        except Exception as exc:
            audit.require(f"{key}: saved result fully auditable", False, exception=f"{type(exc).__name__}: {exc}")
    if model_inputs:
        reference = next(iter(model_inputs.values()))
        for key, values in model_inputs.items():
            for name in reference:
                audit.close(f"{key}: common split/input arrays match all other methods", values[name], reference[name], atol=0, rtol=0)
    tree_r = directory / "jobs" / _job_id("NJ", None) / "model.npz"
    tree_x = directory / "jobs" / _job_id("RNJ", None) / "model.npz"
    if tree_r.exists() and tree_x.exists():
        with np.load(tree_r) as nj, np.load(tree_x) as rnj:
            for name in ("R", "X"):
                audit.close(f"NJ/RNJ share identical 64-row front-end {name}", nj[name], rnj[name], atol=1e-12)
    return {"audit": "independent saved-history, 64/32, 1SE, bound-restart and physical-path audit",
            "directory": str(directory.resolve()), "dataset_directory": str(dataset_directory.resolve()),
            "jobs_expected": len(expected), "jobs_present": len(actual & expected),
            "metrics_recomputed": rows, **audit.result()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--dataset-directory", type=Path, default=ROOT / "outputs/source_ac96_20261018")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--only-job", nargs="+")
    args = parser.parse_args()
    result = audit_full_history(args.directory, args.dataset_directory, only_jobs=args.only_job)
    output = args.output or args.directory / "independent_full_history_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("passed", "check_count", "failed_count", "jobs_expected", "jobs_present")}))
    for check in result["checks"]:
        if not check["passed"]:
            print(json.dumps(check, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
