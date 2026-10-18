"""Audit saved RNJ atomic initialization; topology truth is posthoc only.

No estimator, runtime package, optimization solver or repair is imported.
Running checkpoints and a completed screen without a model are valid stages.
Sampling metadata describes one read, never a final artifact-hash guarantee.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import time
import zipfile

for _thread_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_key] = "1"

import numpy as np


class Checks:
    def __init__(self):
        self.rows = []

    def require(self, name, condition, **details):
        self.rows.append({"check": name, "passed": bool(condition), **details})


def certified(diagnostic):
    if diagnostic.get("status") != 0 or diagnostic.get("success") is not True:
        return False
    count = diagnostic.get("binary_variable_count")
    if count == 0:
        return True
    if not isinstance(count, int) or count < 0:
        return False
    primal, dual = diagnostic.get("objective"), diagnostic.get("dual_bound")
    if primal is not None and dual is not None and np.isfinite([primal, dual]).all() and abs(primal - dual) <= 1e-10:
        return True
    gap = diagnostic.get("mip_gap")
    return gap is not None and np.isfinite(gap) and 0 <= gap <= 1e-10


def partition(groups, n):
    groups = tuple(sorted(tuple(sorted(map(int, group))) for group in groups))
    values = [value for group in groups for value in group]
    if any(not group for group in groups) or sorted(values) != list(range(n)):
        raise ValueError("partition must cover every original terminal exactly once")
    return groups


def wanted_partition(requested, n):
    selected = [tuple(sorted(map(int, group))) for group in requested]
    covered = [value for group in selected for value in group]
    if any(len(group) < 3 for group in selected) or len(set(covered)) != len(covered):
        raise ValueError("requested atomic groups must be disjoint with size >= 3")
    selected += [(i,) for i in range(n) if i not in covered]
    return partition(selected, n)


def sample_job(directory, attempts=10):
    """Retry if evidence inventory/content changes or an NPZ is still writing."""
    directory = Path(directory)
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            def paths():
                top = [directory / name for name in ("result.json", "model.json", "rnj_screen.json",
                       "data_use.json", "execution.json", "solver_events.json")]
                return sorted([p for p in top if p.is_file()] + list((directory / "progress").glob("*.json"))
                              + list((directory / "solver").glob("*.json")) + list((directory / "solver").glob("*.npz")))
            before_paths = paths()
            before = {p.relative_to(directory).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns) for p in before_paths}
            raw = {p.relative_to(directory).as_posix(): p.read_bytes() for p in before_paths}
            after_paths = paths()
            after = {p.relative_to(directory).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns) for p in after_paths}
            if before != after or any(p.read_bytes() != raw[p.relative_to(directory).as_posix()] for p in after_paths):
                raise RuntimeError("saved evidence changed during sample")
            for name, value in raw.items():
                if name.endswith(".json"):
                    json.loads(value)
                else:
                    with zipfile.ZipFile(io.BytesIO(value)) as archive:
                        archive.namelist()
            return raw, {"stable_read": True, "attempts": attempt, "files_read": sorted(raw),
                         "scope": "sampled evidence only; no final SHA or run-completion claim"}
        except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(.01)
    return {}, {"stable_read": False, "attempts": attempts, "sampling_error": last_error,
                "scope": "concurrent write or unreadable sample; model checks deferred"}


def physical_groups(network_directory, observed_labels=None):
    """Root the saved physical tree and identify COMPLETE direct-leaf groups."""
    directory = Path(network_directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    with (directory / "buses.csv").open(encoding="utf-8-sig", newline="") as handle:
        buses = list(csv.DictReader(handle))
    with (directory / "branches.csv").open(encoding="utf-8-sig", newline="") as handle:
        branches = list(csv.DictReader(handle))
    root = int(metadata["root_bus"])
    nodes = {int(row["bus_id"]) for row in buses}
    adjacency = {node: set() for node in nodes}
    truthy = lambda v: str(v).lower() in {"true", "1"}
    edge_count = 0
    for edge in branches:
        if not truthy(edge.get("is_true_closed", edge.get("status", True))):
            continue
        a, b = int(edge["from_bus"]), int(edge["to_bus"])
        if a not in nodes or b not in nodes or a == b or b in adjacency[a]:
            raise ValueError("invalid or duplicated physical edge")
        adjacency[a].add(b)
        adjacency[b].add(a)
        edge_count += 1
    parent, children, queue = {root: None}, {node: [] for node in nodes}, deque([root])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for other in sorted(adjacency[node]):
            if other == parent[node]:
                continue
            if other in parent:
                raise ValueError("physical network contains a cycle")
            parent[other] = node
            children[node].append(other)
            queue.append(other)
    if len(parent) != len(nodes) or edge_count != len(nodes) - 1:
        raise ValueError("physical network must be one connected tree")
    labels = set(map(int, observed_labels)) if observed_labels is not None else {
        int(row["bus_id"]) for row in buses if truthy(row.get("is_terminal", False))}
    if not labels:
        labels = {node for node in nodes if node != root and not children[node]}
    if not labels <= nodes or any(children[node] for node in labels):
        raise ValueError("observed terminal labels must be physical leaves")
    descendants = {node: ({node} if node in labels else set()) for node in nodes}
    clades, direct, node_records = set(), {}, []
    for node in reversed(order):
        if parent[node] is not None:
            descendants[parent[node]].update(descendants[node])
        group = frozenset(descendants[node])
        if 1 < len(group) < len(labels):
            clades.add(group)
        direct_terminals = [child for child in children[node] if child in labels]
        hidden_children = [child for child in children[node] if child not in labels]
        if len(direct_terminals) >= 2 and not hidden_children:
            direct.setdefault(frozenset(direct_terminals), []).append(node)
        if children[node]:
            node_records.append({"parent_bus": node, "direct_terminal_children": sorted(direct_terminals),
                                 "hidden_children": sorted(hidden_children), "descendant_terminals": sorted(group)})
    return {"clades": clades, "direct": direct, "nodes": node_records, "terminal_labels": sorted(labels)}


def _frequencies(records):
    return {frozenset(map(int, row["clade"])): float(row["frequency"]) for row in records or []}


def truth_classification(labels, truth):
    group = frozenset(map(int, labels))
    matching = truth["direct"].get(group, [])
    nodes = [node for node in truth["nodes"] if frozenset(node["descendant_terminals"]) == group]
    if matching:
        reason = "complete_direct_terminal_child_set"
    elif any(node["hidden_children"] for node in nodes):
        reason = "true_clade_with_hidden_child_not_atomic_terminal_group"
    elif nodes:
        reason = "not_complete_direct_terminal_child_set"
    else:
        reason = "not_a_physical_terminal_clade"
    return {"is_true_clade": group in truth["clades"],
            "is_complete_direct_terminal_child_set": bool(matching),
            "physical_parent_buses": matching, "truth_classification": reason}


def audit_initialization(job_directory, truth_network_directory):
    job = Path(job_directory).resolve()
    raw, sampling = sample_job(job)
    checks, unavailable = Checks(), []
    report = {"audit": "posthoc RNJ atomic-initialization audit", "job_directory": str(job),
              "truth_network_directory": str(Path(truth_network_directory).resolve()),
              "sampled_utc": datetime.now(timezone.utc).isoformat(), "sampling": sampling,
              "truth_usage": "posthoc classification only; no repair, parameter change or model selection",
              "final_outcome_scored": False}
    if not sampling["stable_read"]:
        return report | {"verdict": "sample_unstable_checks_deferred", "passed": None, "checks": [],
                         "unavailable_checks": ["all model checks deferred because no stable sample was obtained"]}
    decoded = {name: json.loads(value) for name, value in raw.items() if name.endswith(".json")}
    model, screen, result = (decoded.get(name) for name in ("model.json", "rnj_screen.json", "result.json"))
    progress = [(name, value) for name, value in decoded.items() if name.startswith("progress/")]
    data_use = decoded.get("data_use.json", {})
    state = result.get("status") if result else ("running" if model or screen or "execution.json" in decoded else "not_run")
    report.update(run_state=state, final_result_available=result is not None, model_available=model is not None,
                  screen_available=screen is not None, progress_snapshots=len(progress))
    if screen is None:
        return report | {"verdict": "screen_pending", "passed": None, "checks": [],
                         "unavailable_checks": ["RNJ screening has not been saved"]}
    label_sources = [(name, value.get("terminal_labels")) for name, value in
                     [("model", model or {}), ("data_use", data_use), *progress] if value.get("terminal_labels")]
    labels = list(map(int, label_sources[0][1])) if label_sources else None
    if labels is not None:
        checks.require("original algorithm terminal order is unique", len(set(labels)) == len(labels))
        for name, values in label_sources:
            checks.require(f"{name}: terminal order agrees with saved algorithm order", list(map(int, values)) == labels)
    else:
        unavailable.append("algorithm terminal order unavailable; label/index mapping and partition checks deferred")
    selected = [sorted(map(int, group)) for group in screen.get("selected_support_labels", [])]
    selected_indices = screen.get("selected_support_indices", [])
    checks.require("selected labels and positional groups have equal counts", len(selected) == len(selected_indices))
    checks.require("selected atomic groups have size at least three", all(len(c) >= 3 and len(set(c)) == len(c) for c in selected))
    checks.require("selected atomic groups are disjoint", sum(map(len, selected)) == len(set().union(*map(set, selected))) if selected else True)
    expected = None
    if labels is not None:
        position = {label: i for i, label in enumerate(labels)}
        within = all(label in position for group in selected for label in group)
        checks.require("selected groups only use original terminal labels", within)
        checks.require("selected groups are proper terminal subsets", all(len(group) < len(labels) for group in selected))
        if within:
            expected = [[position[label] for label in group] for group in selected]
            checks.require("selected labels map exactly to original-terminal indices", selected_indices == expected)
    requested = expected if expected is not None else selected_indices
    if "requested_terminal_groups" in screen:
        checks.require("screen requested groups equal selected groups", screen["requested_terminal_groups"] == requested)
    if model is None and not progress:
        unavailable.append("screen complete but no model/checkpoint yet; initial supports, actual partition and accepted path checks deferred")
    states = [("model.json", model)] if model else []
    states += progress
    checked_points, unique_points = [], set()
    actual_partition = None
    for name, snapshot in states:
        initial = snapshot.get("initial_supports")
        actual = snapshot.get("terminal_groups")
        if initial is None or actual is None or labels is None:
            unavailable.append(f"{name}: initial supports or terminal partition unavailable")
            continue
        initial_sets = {frozenset(map(int, c)) for c in initial}
        checks.require(f"{name}: initial family contains every selected RNJ group", {frozenset(c) for c in requested} <= initial_sets)
        checks.require(f"{name}: original singleton leaf atoms are retained", {frozenset((i,)) for i in range(len(labels))} <= initial_sets)
        checks.require(f"{name}: no unreported non-singleton initial supports are frozen",
                       initial_sets == {frozenset((i,)) for i in range(len(labels))} | {frozenset(c) for c in requested})
        if "requested_terminal_groups" in snapshot:
            checks.require(f"{name}: requested groups equal selected positional groups", snapshot["requested_terminal_groups"] == requested)
        try:
            actual = partition(actual, len(labels))
            expected_actual = wanted_partition(requested, len(labels))
        except ValueError as exc:
            checks.require(f"{name}: complete valid original-terminal partition", False, error=str(exc))
            unavailable.append(f"{name}: invalid partition prevents path-domain check")
            continue
        checks.require(f"{name}: actual normalized partition is selected groups plus uncovered singletons",
                       actual == expected_actual)
        actual_partition = actual
        for index, point in enumerate(snapshot.get("path") or []):
            identity = (snapshot.get("bound_expansions"), json.dumps(point, sort_keys=True, separators=(",", ":")))
            if identity in unique_points:
                continue
            unique_points.add(identity)
            diagnostic = point.get("solver", point.get("diagnostics", {}))
            tag = f"{name}: path[{index}]"
            proven = certified(diagnostic)
            checks.require(f"{tag}: saved accepted point has strict solver certificate", proven)
            supports = [frozenset(map(int, support)) for support in point["support_indices"]]
            checks.require(f"{tag}: atom supports are nonempty and within original terminal positions",
                           all(support and all(0 <= i < len(labels) for i in support) for support in supports))
            checks.require(f"{tag}: fixed initial family remains present", initial_sets <= set(supports))
            checks.require(f"{tag}: every new atom is a union of whole groups", all(
                not (support & set(group)) or set(group) <= support
                for support in supports if support not in initial_sets for group in actual))
            checked_points.append({"first_saved_in": name, "path_index": index, "iteration": point.get("iteration"),
                                   "bound_expansions": snapshot.get("bound_expansions"), "atom_count": len(supports),
                                   "strictly_certified": proven})
    solver_records = {value["sequence"]: (name, value) for name, value in decoded.items() if name.startswith("solver/")}
    for event in decoded.get("solver_events.json", []):
        if event["sequence"] in solver_records:
            checks.require(f"solve[{event['sequence']}]: individual and combined records agree",
                           solver_records[event["sequence"]][1] == event)
        else:
            solver_records[event["sequence"]] = ("solver_events.json", event)
    z_records = []
    for sequence, (name, event) in sorted(solver_records.items()):
        z_slice = event.get("variable_slices", {}).get("z")
        if z_slice is None:
            continue  # Fixed-support LP/Source-path refits have no new z.
        m = len(actual_partition) if actual_partition is not None else None
        if m is None and labels is not None:
            try:
                m = len(wanted_partition(requested, len(labels)))
            except ValueError:
                unavailable.append(f"solve[{sequence}]: valid partition unavailable")
        start, stop = map(int, z_slice)
        row = {"sequence": sequence, "record": name, "z_slice_length": stop - start,
               "partition_block_count": m, "z_binary_count": None, "strictly_certified": certified(event)}
        if m is not None:
            checks.require(f"solve[{sequence}]: z variable count equals normalized partition block count", stop - start == m)
        primal = event.get("primal_file")
        if primal in raw:
            with np.load(io.BytesIO(raw[primal]), allow_pickle=False) as saved:
                if all(key in saved.files for key in ("integrality", "lower_bounds", "upper_bounds")):
                    valid_slice = 0 <= start < stop <= len(saved["integrality"])
                    checks.require(f"solve[{sequence}]: z slice belongs to original variable arrays", valid_slice)
                    if not valid_slice:
                        z_records.append(row)
                        continue
                    integrality = saved["integrality"][start:stop]
                    lower, upper = saved["lower_bounds"][start:stop], saved["upper_bounds"][start:stop]
                    row["z_binary_count"] = int(np.count_nonzero(integrality == 1))
                    checks.require(f"solve[{sequence}]: each z variable is explicitly binary with zero/one bounds",
                                   len(integrality) == stop - start and np.all(integrality == 1)
                                   and np.array_equal(lower, np.zeros(stop - start)) and np.array_equal(upper, np.ones(stop - start)))
                    if m is not None:
                        checks.require(f"solve[{sequence}]: actual binary z count equals partition block count", row["z_binary_count"] == m)
                else:
                    unavailable.append(f"solve[{sequence}]: original z integrality/bound arrays unavailable")
        else:
            unavailable.append(f"solve[{sequence}]: no saved raw integrality arrays; z slice length checked only")
        z_records.append(row)
    truth = physical_groups(truth_network_directory, labels)
    confidence = _frequencies(screen.get("confidence_records"))
    diagnostic = screen.get("diagnostics", {})
    peers = {frozenset(map(int, group)) for group in diagnostic.get("full_peers", [])}
    sub = _frequencies(diagnostic.get("subclade_frequency_records"))
    veto = {frozenset(map(int, group)) for group in diagnostic.get("full_fine_subclades", [])}
    params = screen.get("parameters", {})
    threshold = float(params.get("confidence_threshold", .75))
    sub_limit = float(params.get("maximum_subclade_frequency", .1))
    minimum = int(diagnostic.get("minimum_block_size", 3))
    candidate_rows, selected_rows, blocked = [], [], []
    for group in sorted(set(confidence) | peers | {frozenset(c) for c in selected}, key=lambda c: tuple(sorted(c))):
        reasons = []
        if group not in peers:
            reasons.append("not_full_coarse_peer")
        if len(group) < minimum:
            reasons.append("below_minimum_size")
        if confidence.get(group, 0.) < threshold:
            reasons.append("coarse_frequency_below_threshold")
        if sub.get(group, 0.) > sub_limit:
            reasons.append("bootstrap_internal_subclade_veto")
        if group in veto:
            reasons.append("full_fine_tree_subclade_veto")
        if diagnostic.get("source_adjusted_identifiable") is False:
            reasons.append("source_adjusted_design_unidentifiable")
        if diagnostic.get("valid_bootstrap_replicates") == 0:
            reasons.append("no_identifiable_bootstrap_replicates")
        row = {"labels": sorted(group), "selected": sorted(group) in selected, "full_coarse_peer": group in peers,
               "coarse_frequency": confidence.get(group), "subclade_frequency": sub.get(group),
               "full_fine_subclade_veto": group in veto, "rejection_reasons": reasons, **truth_classification(group, truth)}
        candidate_rows.append(row)
        if row["selected"]:
            selected_rows.append(row)
        if (row["is_complete_direct_terminal_child_set"] and len(group) >= minimum and group in peers
                and confidence.get(group, 0.) >= threshold and (sub.get(group, 0.) > sub_limit or group in veto)):
            blocked.append(row)
    count = len(selected)
    true_count = sum(truth_classification(group, truth)["is_complete_direct_terminal_child_set"] for group in selected)
    empty = not selected
    report.update(verdict="checked_sample" if model or progress else "screen_only_model_pending",
                  algorithm_terminal_labels=labels, selected_groups=selected_rows,
                  selected_count=count, selected_true_complete_group_count=true_count,
                  selected_false_complete_group_count=count - true_count,
                  selected_complete_group_precision=true_count / count if count else None,
                  empty_initialization=empty, degenerated_no_atomic_rnj_prior=empty,
                  normalized_partition=None if actual_partition is None else [list(c) for c in actual_partition],
                  saved_path_points_checked=len(checked_points),
                  certified_path_points_checked=sum(point["strictly_certified"] for point in checked_points), path_points=checked_points,
                  z_extension_models_checked=len(z_records), z_extension_models=z_records,
                  true_complete_terminal_groups=[{"labels": sorted(c), "parent_buses": parents}
                                                 for c, parents in truth["direct"].items() if len(c) >= minimum],
                  high_confidence_true_groups_vetoed=blocked, high_confidence_true_groups_vetoed_count=len(blocked),
                  candidates=candidate_rows, unavailable_checks=sorted(set(unavailable)), checks=checks.rows,
                  check_count=len(checks.rows), failed_count=sum(not row["passed"] for row in checks.rows),
                  passed=all(row["passed"] for row in checks.rows))
    return report


def write_candidate_csv(path, report):
    fields = ["labels", "selected", "full_coarse_peer", "coarse_frequency", "subclade_frequency",
              "full_fine_subclade_veto", "is_true_clade", "is_complete_direct_terminal_child_set",
              "physical_parent_buses", "truth_classification", "rejection_reasons"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report.get("candidates", []):
            writer.writerow({name: json.dumps(row[name], ensure_ascii=False) if isinstance(row.get(name), list) else row.get(name)
                             for name in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_directory", type=Path)
    parser.add_argument("truth_network_directory", type=Path)
    parser.add_argument("--json", dest="json_output", type=Path)
    parser.add_argument("--csv", dest="csv_output", type=Path)
    args = parser.parse_args()
    report = audit_initialization(args.job_directory, args.truth_network_directory)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        write_candidate_csv(args.csv_output, report)
    names = ("verdict", "run_state", "final_result_available", "passed", "check_count", "failed_count",
             "selected_count", "selected_true_complete_group_count", "selected_complete_group_precision",
             "empty_initialization", "degenerated_no_atomic_rnj_prior", "certified_path_points_checked",
             "z_extension_models_checked", "high_confidence_true_groups_vetoed_count")
    print(json.dumps({name: report.get(name) for name in names}, ensure_ascii=False))
    return 1 if report.get("passed") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
