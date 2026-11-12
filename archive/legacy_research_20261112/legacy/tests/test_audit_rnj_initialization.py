"""Posthoc initialization evidence tests; no fit, simulation or solver runs."""
import copy
import csv
import json

import numpy as np
import pytest

from research_experiments.rnj.audit_rnj_initialization import (
    audit_initialization, main, physical_groups, truth_classification, write_candidate_csv,
)

LABELS = [301, 302, 303, 304, 305, 306]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def network(directory):
    directory.mkdir()
    save_json(directory / "metadata.json", {"root_bus": 1})
    with (directory / "buses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bus_id", "is_terminal"])
        writer.writeheader()
        writer.writerows({"bus_id": i, "is_terminal": i in LABELS} for i in [1, 2, 3, 4, *LABELS])
    edges = [(1, 2), (1, 3), (2, 301), (2, 302), (2, 303), (3, 304), (3, 4), (4, 305), (4, 306)]
    with (directory / "branches.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["from_bus", "to_bus", "is_true_closed"])
        writer.writeheader()
        writer.writerows({"from_bus": a, "to_bus": b, "is_true_closed": True} for a, b in edges)
    return directory


def fixture(tmp_path, *, selected=None, stage="complete"):
    job, truth = tmp_path / "job", network(tmp_path / "network")
    selected = [[301, 302, 303]] if selected is None else selected
    position = {label: i for i, label in enumerate(LABELS)}
    indices = [[position[label] for label in group] for group in selected]
    covered = {i for group in indices for i in group}
    groups = sorted([*indices, *([i] for i in range(6) if i not in covered)])
    groups = [list(group) for group in groups]
    initial = [[i] for i in range(6)] + indices
    peers = [[301, 302, 303], [304, 305, 306]]
    screen = {"selected_support_labels": selected, "selected_support_indices": indices,
              "requested_terminal_groups": indices,
              "confidence_records": [{"clade": c, "frequency": .95} for c in peers],
              "parameters": {"confidence_threshold": .75, "maximum_subclade_frequency": .1},
              "diagnostics": {"full_peers": peers, "full_fine_subclades": [], "minimum_block_size": 3,
                  "subclade_frequency_records": [{"clade": c, "frequency": 0. if c in selected else .8} for c in peers],
                  "valid_bootstrap_replicates": 100, "source_adjusted_identifiable": True}}
    save_json(job / "rnj_screen.json", screen)
    save_json(job / "data_use.json", {"terminal_labels": LABELS})
    if stage == "screen_only":
        return job, truth
    initial_point = {"support_indices": initial, "iteration": len(initial),
                     "solver": {"status": 0, "success": True, "binary_variable_count": 0, "objective": .2}}
    extended = copy.deepcopy(initial_point)
    extended["support_indices"] = [*initial, [0, 1, 2, 3]]
    extended["iteration"] += 1
    extended["solver"] = {"status": 0, "success": True, "binary_variable_count": len(groups) + 7,
                          "objective": .1, "dual_bound": .1, "mip_gap": 0.}
    model = {"terminal_labels": LABELS, "initial_supports": initial, "requested_terminal_groups": indices,
             "terminal_groups": groups, "bound_expansions": 0, "path": [initial_point, extended], "status": stage}
    save_json(job / "model.json", model)
    first = copy.deepcopy(model)
    first["path"] = [initial_point]
    save_json(job / "progress/000001.json", first)
    save_json(job / "progress/000002.json", model)
    init_event = {"sequence": 1, "variable_slices": {}}
    event = {**extended["solver"], "sequence": 2, "variable_slices": {"z": [0, len(groups)], "source_h": [len(groups), len(groups) + 7]},
             "primal_file": "solver/000002.npz"}
    save_json(job / "solver/000001.json", init_event)
    save_json(job / "solver/000002.json", event)
    save_json(job / "solver_events.json", [init_event, event])
    np.savez_compressed(job / "solver/000002.npz", integrality=np.ones(len(groups) + 7),
                        lower_bounds=np.zeros(len(groups) + 7), upper_bounds=np.ones(len(groups) + 7))
    if stage == "complete":
        save_json(job / "result.json", {"status": "complete"})
    return job, truth


def failures(report):
    return [row["check"] for row in report["checks"] if not row["passed"]]


def test_true_group_requires_complete_direct_terminal_children_and_no_hidden_child(tmp_path):
    truth = physical_groups(network(tmp_path / "network"), LABELS)
    complete = truth_classification([301, 302, 303], truth)
    nested = truth_classification([304, 305, 306], truth)
    partial = truth_classification([301, 302], truth)
    assert complete["is_true_clade"] and complete["is_complete_direct_terminal_child_set"]
    assert nested["is_true_clade"] and not nested["is_complete_direct_terminal_child_set"]
    assert nested["truth_classification"] == "true_clade_with_hidden_child_not_atomic_terminal_group"
    assert not partial["is_complete_direct_terminal_child_set"]


@pytest.mark.parametrize("stage", ["complete", "running"])
def test_valid_initialization_and_all_saved_certified_atoms_pass(tmp_path, stage):
    job, truth = fixture(tmp_path, stage=stage)
    report = audit_initialization(job, truth)
    assert report["passed"], failures(report)
    assert report["final_result_available"] is (stage == "complete")
    assert report["run_state"] == stage
    assert report["selected_count"] == report["selected_true_complete_group_count"] == 1
    assert report["selected_complete_group_precision"] == 1.
    assert report["certified_path_points_checked"] == 2
    assert report["z_extension_models"][0]["z_binary_count"] == 4
    assert report["z_extension_models"][0]["partition_block_count"] == 4
    assert "SHA" in report["sampling"]["scope"]
    assert report["final_outcome_scored"] is False


@pytest.mark.parametrize("corruption, expected", [
    ("mapping", "selected labels map exactly"),
    ("initial", "initial family contains every selected"),
    ("request", "requested groups equal selected"),
    ("partition", "actual normalized partition"),
    ("refinement", "every new atom is a union"),
    ("z_count", "z variable count equals"),
    ("z_relaxed", "explicitly binary"),
    ("certificate", "strict solver certificate"),
])
def test_evidence_corruption_is_detected(tmp_path, corruption, expected):
    job, truth = fixture(tmp_path)
    model_path = job / "model.json"
    model = json.loads(model_path.read_text())
    if corruption == "mapping":
        path = job / "rnj_screen.json"
        screen = json.loads(path.read_text())
        screen["selected_support_indices"] = [[1, 2, 3]]
        save_json(path, screen)
    elif corruption == "initial":
        model["initial_supports"] = model["initial_supports"][:-1]
    elif corruption == "request":
        model["requested_terminal_groups"] = []
    elif corruption == "partition":
        model["terminal_groups"] = [[i] for i in range(6)]
    elif corruption == "refinement":
        model["path"][-1]["support_indices"][-1] = [0, 1]
    elif corruption == "certificate":
        model["path"][-1]["solver"].update(dual_bound=None, mip_gap=None)
    elif corruption == "z_count":
        path = job / "solver/000002.json"
        event = json.loads(path.read_text())
        event["variable_slices"]["z"] = [0, 3]
        save_json(path, event)
    elif corruption == "z_relaxed":
        integrality = np.ones(11)
        integrality[0] = 0
        np.savez_compressed(job / "solver/000002.npz", integrality=integrality,
                            lower_bounds=np.zeros(11), upper_bounds=np.ones(11))
    save_json(model_path, model)
    report = audit_initialization(job, truth)
    assert any(expected in text for text in failures(report)), failures(report)


def test_screen_only_and_empty_initialization_are_pending_not_a_final_model(tmp_path):
    job, truth = fixture(tmp_path, selected=[], stage="screen_only")
    report = audit_initialization(job, truth)
    assert report["passed"], failures(report)
    assert report["verdict"] == "screen_only_model_pending"
    assert not report["model_available"] and not report["final_result_available"]
    assert report["empty_initialization"] and report["degenerated_no_atomic_rnj_prior"]
    assert report["selected_count"] == 0
    assert report["selected_complete_group_precision"] is None
    assert report["high_confidence_true_groups_vetoed_count"] == 1
    assert report["high_confidence_true_groups_vetoed"][0]["labels"] == [301, 302, 303]
    assert report["unavailable_checks"]


def test_posthoc_wrong_atomic_group_is_reported_without_repair(tmp_path):
    job, truth = fixture(tmp_path, selected=[[304, 305, 306]], stage="screen_only")
    before = {path: path.read_bytes() for path in job.rglob("*") if path.is_file()}
    report = audit_initialization(job, truth)
    assert report["selected_count"] == 1
    assert report["selected_true_complete_group_count"] == 0
    assert report["selected_complete_group_precision"] == 0.
    assert all(path.read_bytes() == value for path, value in before.items())


def test_pending_screen_is_unavailable_not_invented_empty_initialization(tmp_path):
    truth = network(tmp_path / "network")
    job = tmp_path / "job"
    job.mkdir()
    report = audit_initialization(job, truth)
    assert report["verdict"] == "screen_pending" and report["passed"] is None
    assert "empty_initialization" not in report


def test_csv_and_default_stdout_summary(tmp_path, monkeypatch, capsys):
    job, truth = fixture(tmp_path, stage="running")
    report = audit_initialization(job, truth)
    target = tmp_path / "initialization.csv"
    write_candidate_csv(target, report)
    with target.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(report["candidates"])
    monkeypatch.setattr("sys.argv", ["audit_rnj_initialization", str(job), str(truth)])
    assert main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["run_state"] == "running" and summary["final_result_available"] is False
