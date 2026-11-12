"""Summarize recorded own-path diagnostics without fitting or solver imports.

The zero-release comparison is the matched before/after baseline comparison.
The prefix/exit/output comparison describes saved stages on one trajectory;
it is not an isolated removal ablation of the final exit or output processing.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path

import pandas as pd


STAGE_NOTE = (
    "Recorded-stage diagnostic, not an isolated exit ablation: raw prefix and "
    "exit endpoints precede the common contraction/refit/path-validation output."
)
ZERO_NOTE = (
    "Matched own-path zero-release comparison; all baseline subproblems met "
    "the recorded gap. Baseline phase ceilings were allocated differently, "
    "and no standalone runtime is inferred."
)


def analyze_history(root: Path) -> dict:
    """Return four DataFrames and provenance metadata from existing records.

    ``zero_pairs`` contains the 120 matched own-output pairs;
    ``zero_summary`` has one row per scope (all/main/stress/four cases).
    ``stage_pairs`` contains raw prefix, raw exit and final own-output scores;
    ``stage_summary`` is long form, with one row per scope and stage.
    No file is written, and no parameter fitting or candidate selection runs.
    """
    root = Path(root).resolve()
    docs = root / "docs/iterative_exit_matched120_20261101"
    output = root / "outputs/iterative_exit_matched120_20261101"
    source_hashes: dict[str, str] = {}

    def track(path: Path) -> Path:
        path = path.resolve()
        try:
            name = path.relative_to(root).as_posix()
        except ValueError:
            name = str(path)
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read(path: Path):
        return json.loads(track(path).read_text(encoding="utf-8"))

    manifest = read(docs / "manifest.json")["files"]
    for name in ("per_model.csv", "protocol.json", "domain_diagnostics.json"):
        assert hashlib.sha256(track(docs / name).read_bytes()).hexdigest() == manifest[name]["sha256"]
    frame = pd.read_csv(track(docs / "per_model.csv"))
    protocol = read(docs / "protocol.json")
    domains = read(docs / "domain_diagnostics.json")
    assert domains["both_base_subproblems_gap_met"] == 120
    catalog = read(output / "catalog.json")

    def view(method: str, name: str) -> pd.DataFrame:
        part = frame[(frame.method == method) & (frame.view == name)].copy()
        assert len(part) == 120 and part.job.is_unique
        assert set(part.job) == set(protocol["jobs"])
        return part.set_index("job").sort_index()

    before = view("before", "own")
    own = view("after", "zero_only_own")
    raw_exit = view("after", "raw_endpoint")
    assert set(own.selected_source) <= {"baseline", "singleton"}
    headers = ["panel", "case", "bundle", "N", "voltage_sigma", "condition"]
    pd.testing.assert_frame_equal(before[headers], own[headers])
    zero = own[headers].copy()
    zero["before_F1"] = before.clade_f1
    zero["after_F1"] = own.clade_f1
    zero["delta_F1"] = own.clade_f1 - before.clade_f1
    zero["before_exact"] = before.exact
    zero["after_exact"] = own.exact
    zero["new_exact"] = ~before.exact & own.exact
    zero["lost_exact"] = before.exact & ~own.exact
    zero = zero.reset_index()

    def scopes(part: pd.DataFrame):
        yield "all", part
        for panel, group in part.groupby("panel", sort=True):
            yield panel, group
        for case, group in part.groupby("case", sort=True):
            yield case, group

    zero_summary = []
    for scope, group in scopes(zero):
        delta = group.delta_F1
        zero_summary.append({
            "scope": scope,
            "inputs": len(group),
            "F1_before": float(group.before_F1.mean()),
            "F1_after": float(group.after_F1.mean()),
            "delta_F1": float(delta.mean()),
            "exact_before": int(group.before_exact.sum()),
            "exact_after": int(group.after_exact.sum()),
            "new_exact": int(group.new_exact.sum()),
            "lost_exact": int(group.lost_exact.sum()),
            "wins": int((delta > 1e-10).sum()),
            "ties": int((abs(delta) <= 1e-10).sum()),
            "losses": int((delta < -1e-10).sum()),
        })

    truth_cache = {}

    def truth(meta: dict, labels: list) -> set[frozenset]:
        """Independently root the physical CSV tree and collect descendants."""
        labels = tuple(int(label) for label in labels)
        key = (meta["truth_network"], labels)
        if key in truth_cache:
            return truth_cache[key]
        directory = Path(meta["truth_network"])
        if not directory.is_absolute():
            directory = root / directory
        metadata = read(directory / "metadata.json")
        buses = pd.read_csv(track(directory / "buses.csv"))
        branches = pd.read_csv(track(directory / "branches.csv"))
        adjacency = {int(bus): [] for bus in buses.bus_id}
        edge_count = 0
        for edge in branches.to_dict("records"):
            if str(edge.get("is_true_closed", True)).lower() not in {"true", "1"}:
                continue
            u, v = int(edge["from_bus"]), int(edge["to_bus"])
            assert u in adjacency and v in adjacency and u != v
            adjacency[u].append(v)
            adjacency[v].append(u)
            edge_count += 1
        root_bus = int(metadata["root_bus"])
        assert edge_count == len(adjacency) - 1 and root_bus in adjacency
        parent = {root_bus: None}
        queue = deque([root_bus])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in adjacency[node]:
                if child not in parent:
                    parent[child] = node
                    queue.append(child)
        assert len(parent) == len(adjacency)
        assert len(set(labels)) == len(labels) and set(labels) <= set(adjacency)
        positions = {label: i for i, label in enumerate(labels)}
        descendants = {
            node: {positions[node]} if node in positions else set()
            for node in order
        }
        for node in reversed(order[1:]):
            descendants[parent[node]] |= descendants[node]
        result = {
            frozenset(descendants[node]) for node in order[1:]
            if 1 < len(descendants[node]) < len(labels)
        }
        truth_cache[key] = result
        return result

    def point_metrics(point: dict, target: set[frozenset], n: int):
        predicted = {
            frozenset(support)
            for support, r, x in zip(point["supports"], point["r"], point["x"])
            if 1 < len(support) < n and max(r, x) > 1e-9
        }
        f1 = 2 * len(predicted & target) / (len(predicted) + len(target)) if predicted or target else 1.0
        return float(f1), predicted == target

    stage_rows = []
    for job in protocol["jobs"]:
        meta = catalog[job]
        saved = read(output / "jobs" / job / "after/result.json")
        common = read(output / "common" / job / "common.json")
        labels = common["labels"]
        target = truth(meta, labels)
        prefix = saved["prefix"]
        baseline = saved["branches"]["baseline"]
        pre_f1, pre_exact = point_metrics(prefix["point"], target, len(labels))
        post_f1, post_exact = point_metrics(baseline["trajectory"]["point"], target, len(labels))
        assert abs(post_f1 - raw_exit.loc[job].clade_f1) < 1e-12
        assert post_exact == bool(raw_exit.loc[job].exact)
        assert saved["input_hashes"] == {
            split: protocol["input_hashes"][job][split]
            for split in ("train", "validation")
        }
        choice = saved["choices_matched"]["baseline"]
        pre_length = len(prefix["path"])
        post_length = len(baseline["trajectory"]["path"])
        index = choice["index"]
        selected_stage = (
            "singleton" if index == post_length
            else "pre_exit_path" if index < pre_length
            else "exit_path"
        )
        stage_rows.append({
            "job": job,
            **{column: own.loc[job, column] for column in headers},
            "prefix_F1": pre_f1,
            "prefix_exact": pre_exact,
            "exit_F1": post_f1,
            "exit_exact": post_exact,
            "final_F1": float(own.loc[job].clade_f1),
            "final_exact": bool(own.loc[job].exact),
            "exit_accepted": bool(baseline["exit"]["accepted"]),
            "tail_calls": sum(event["kind"] == "joint_tail" for event in prefix["attempts"]),
            "tail_accepted": sum(event["kind"] == "joint_tail" and event["accepted"] for event in prefix["attempts"]),
            "selected_stage": selected_stage,
            "warm_origin": baseline["exit"]["warm_origin"],
        })
    stages = pd.DataFrame(stage_rows)
    stage_summary = []
    for scope, group in scopes(stages):
        for stage, label in (
            ("prefix", "raw_prefix"), ("exit", "raw_exit"), ("final", "final_output")
        ):
            stage_summary.append({
                "scope": scope,
                "stage": label,
                "inputs": len(group),
                "F1": float(group[f"{stage}_F1"].mean()),
                "exact": int(group[f"{stage}_exact"].sum()),
                "recovery_rate": float(group[f"{stage}_exact"].mean()),
            })
    assert abs(stages.final_F1.mean() - .8217551430051431) < 1e-12
    assert stages.final_exact.sum() == 60
    return {
        "zero_pairs": zero,
        "zero_summary": pd.DataFrame(zero_summary),
        "stage_pairs": stages,
        "stage_summary": pd.DataFrame(stage_summary),
        "metadata": {
            "zero_note": ZERO_NOTE,
            "stage_note": STAGE_NOTE,
            "inputs": 120,
            "new_fitting_or_solver_calls": 0,
            "raw_exit_scores_verified_against_archived_CSV": 120,
            "both_base_subproblems_gap_met": domains["both_base_subproblems_gap_met"],
            "final_output_includes_public_RNJ_fallback": False,
            "exit_warm_start_origins": stages.warm_origin.value_counts().to_dict(),
            "selected_stage_counts": stages.selected_stage.value_counts().to_dict(),
            "source_hashes": source_hashes,
        },
    }
