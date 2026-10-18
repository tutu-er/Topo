"""Saved-product fixtures; no estimator, simulator or solver is imported."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from research_experiments.rnj import summarize_rnj_blocks96 as summary


TARGET = "soumalas11__no_source__RNJ+WZZT_tree_init"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def metrics(f1):
    return {**{key: f1 for key in summary.METRICS}, "r_relative_error": .1 if f1 is not None else None,
            "x_relative_error": .2 if f1 is not None else None}


def fixture(root, *, failed=True, complete_marker=True):
    new = [{"job_id": "__".join((case, variant, method)), "case": case, "variant": variant, "method": method,
        "role": "main" if method in ("WZZT", "RNJ+WZZT") else "configuration_control",
        "input_relative_directory": f"inputs/{case}/{variant}",
        "screen_parameters": {"confidence_threshold": .75, "maximum_subclade_frequency": .15},
        "per_solve_seconds": 6000., "method_wall_seconds": 30000.}
        for case in summary.CASES for variant in summary.VARIANTS for method in summary.NEW_METHODS]
    old = [{"job_id": "__".join((case, variant, method)), "case": case, "variant": variant, "method": method,
        "directory": "baseline_reuse/jobs/"+"__".join((case, variant, method))}
        for case in summary.CASES for variant in summary.VARIANTS for method in ("NJ", "RNJ")]
    protocol = {"suite": "hand_written_fixture", "jobs": new, "baseline_jobs": old,
        "source_sha256": {}, "training_samples": 96, "validation_supplied": False,
        "core_parameters_base": {"r_upper_bound": 2, "x_upper_bound": 2}, "solver": "fixture",
        "concurrency": 4, "dataset_manifest_sha256": "data-fixture-binding",
        "baseline_evidence": {"fingerprint": "unchanged-historical-evidence", "created_utc": "first",
            "validation_seconds": 1., "inventory": {"time_dependent_artifact": "first"}}}
    write(root/"protocol.json", protocol)
    write(root/"summary_fixture.json", {"hand_written_fixture": True})
    for job in new+old:
        directory = root/job.get("directory", "jobs/"+job["job_id"])
        is_failed = failed and job["job_id"] == TARGET
        write(directory/"result.json", {**job, "status": "failed" if is_failed else "complete",
            "has_model": True, "metrics": metrics(1. if is_failed else .7), "elapsed_seconds": 7. if is_failed else 2.,
            "compute_seconds": 7. if is_failed else 2., "stop_reason": "recording failed" if is_failed else "no_gain",
            "error": {"type": "PermissionError", "message": "fixture replace"} if is_failed else None,
            "protocol_sha256": sha(root/"protocol.json")})
    for case in summary.CASES:
        for variant in summary.VARIANTS:
            screen = {"parameters": {"confidence_threshold": .75, "maximum_subclade_frequency": .15},
                "selected_support_labels": [[101, 102]], "confidence_records": [{"clade": [101, 102], "frequency": .92}],
                "diagnostics": {"candidate_records": [
                    {"clade": [101, 102], "atomic_peer_frequency": .92, "subclade_frequency": .03,
                     "selected": True, "rejection_reasons": []},
                    {"clade": [103, 104, 105], "atomic_peer_frequency": .8, "subclade_frequency": .3,
                     "selected": False, "rejection_reasons": ["internal_subclade_frequency_above_threshold"]}]}}
            for method in summary.NEW_METHODS[1:]:
                write(root/"jobs"/"__".join((case, variant, method))/"rnj_screen.json",
                      {**screen, "elapsed_seconds": 2. if method == "RNJ+WZZT" else 3.,
                       "requested_terminal_groups": [] if method.endswith("ungrouped") else [[0, 1]]})
    if complete_marker:
        write(root/"execution_complete.json", {"completion": {"expected": 64, "finished": 64}})
    return protocol


def recovery(root, retry_root, protocol):
    retry_protocol = json.loads(json.dumps(protocol))
    retry_protocol["baseline_evidence"].update(created_utc="retry", validation_seconds=3.,
                                             inventory={"time_dependent_artifact": "retry"})
    write(retry_root/"protocol.json", retry_protocol)
    job = next(j for j in protocol["jobs"] if j["job_id"] == TARGET)
    old_dir, new_dir = (r/"jobs"/TARGET for r in (root, retry_root))
    model = {"terminal_labels": [101, 102], "core_parameters": {"max_atoms": 3, "gain_tolerance": 1e-10},
        "budget": job, "rx75_calibration": {"a": .75, "b": .25, "tau": .16},
        "initial_supports": [[0], [1], [0, 1]], "requested_terminal_groups": [[0, 1]],
        "terminal_groups": [[0, 1]], "validation_supplied": False, "training_rows": 96,
        "has_model": True, "status": "failed"}
    arrays = {key: np.full((96, 2), .1+index) for index, key in enumerate(("P", "Q", "Y"))}
    arrays.update(P_source=np.linspace(0, 1, 96), Q_source=np.linspace(1, 0, 96),
                  R=np.eye(2), X=2*np.eye(2), calibration_R=np.eye(2), calibration_X=2*np.eye(2))
    inventory = {}
    for relative in summary.INPUT_FILES:
        for r in (root, retry_root):
            path = r/"inputs/soumalas11/no_source"/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("fixture "+relative).encode())
        inventory[relative] = sha(root/"inputs/soumalas11/no_source"/relative)
    for r, directory, config in ((root, old_dir, protocol), (retry_root, new_dir, retry_protocol)):
        write(directory/"model.json", {**model, "status": "complete" if r == retry_root else "failed"})
        np.savez(directory/"model.npz", **arrays)
        write(directory/"inputs_hash.json", {"file_sha256": inventory})
        write(directory/"payload.json", {"job": job, "config": config, "input_sha256": inventory,
                                          "protocol_sha256": sha(r/"protocol.json")})
    first = read(old_dir/"result.json")
    write(new_dir/"result.json", {**first, "status": "complete", "stop_reason": "no_significant_one_atom_gain",
        "error": None, "metrics": metrics(.8), "elapsed_seconds": 4., "compute_seconds": 4.,
        "protocol_sha256": sha(retry_root/"protocol.json")})
    return retry_protocol, new_dir


def refresh_bindings(root, retry_root):
    for r in (root, retry_root):
        protocol = read(r/"protocol.json")
        for path in (r/"jobs").glob("*/result.json"):
            result = read(path)
            result["protocol_sha256"] = sha(r/"protocol.json")
            write(path, result)
        payload_path = r/"jobs"/TARGET/"payload.json"
        payload = read(payload_path)
        payload.update(config=protocol, protocol_sha256=sha(r/"protocol.json"))
        write(payload_path, payload)


def test_requires_complete_marker_before_reading_protocol(tmp_path):
    root = tmp_path/"active"
    root.mkdir()
    (root/"protocol.json").write_text("this must not be parsed", encoding="utf-8")
    with pytest.raises(summary.SummaryValidationError, match="execution_complete"):
        summary.summarize(root)
    assert not (root/"saved_product_summary").exists()


def test_complete_marker_does_not_hide_missing_or_running_results(tmp_path):
    root = tmp_path/"run"
    fixture(root)
    target = root/"jobs"/TARGET/"result.json"
    result = read(target)
    result["status"] = "running"
    write(target, result)
    with pytest.raises(summary.SummaryValidationError, match="nonfinal"):
        summary.summarize(root)
    target.unlink()
    with pytest.raises(summary.SummaryValidationError, match="missing final"):
        summary.summarize(root)


def test_fixed_96_denominator_keeps_failed_fallback_and_missing_metrics(tmp_path):
    root = tmp_path/"run"
    fixture(root)
    missing = root/"jobs/paper15__smooth__WZZT/result.json"
    value = read(missing)
    value.update(status="incomplete", has_model=False, metrics=metrics(None))
    write(missing, value)
    original = sha(root/"jobs"/TARGET/"result.json")
    result = summary.summarize(root)
    output = Path(result["output_directory"])
    first, effective = rows(output/"first_attempt.csv"), rows(output/"effective_results.csv")
    assert len(first) == len(effective) == result["denominator"] == 96
    fallback = next(r for r in first if r["job_id"] == TARGET)
    assert fallback["status"] == "failed" and fallback["retained_fallback"] == "True"
    assert float(fallback["resolution_clade_f1"]) == 1.
    absent = next(r for r in effective if r["job_id"] == "paper15__smooth__WZZT")
    assert absent["resolution_clade_f1"] == absent["r_relative_error"] == ""
    wzzt = next(r for r in rows(output/"aggregate_by_method.csv") if r["method"] == "WZZT")
    assert int(wzzt["planned"]) == 16 and int(wzzt["incomplete"]) == 1
    assert int(wzzt["resolution_clade_f1_available_count"]) == 15
    assert len(rows(output/"screen_conditions.csv")) == 16
    assert len(rows(output/"screen_candidates.csv")) == 32
    condition = rows(output/"screen_conditions.csv")[0]
    assert condition["identical_screen_excluding_elapsed_and_requested_groups"] == "True"
    assert "internal_subclade_frequency_above_threshold" in condition["rejection_reason_counts"]
    assert sha(root/"jobs"/TARGET/"result.json") == original
    assert summary.summarize(root)["manifest_sha256"] == result["manifest_sha256"]


def test_partial_mode_is_fixture_only_and_keeps_96_rows(tmp_path):
    root = tmp_path/"fixture"
    fixture(root, complete_marker=False)
    (root/"jobs"/TARGET/"result.json").unlink()
    (root/"summary_fixture.json").unlink()
    with pytest.raises(summary.SummaryValidationError, match="fixture marker"):
        summary.summarize(root, allow_partial=True)
    write(root/"summary_fixture.json", {"hand_written_fixture": True})
    result = summary.summarize(root, allow_partial=True)
    assert result["first_status_counts"]["not_run"] == 1
    assert len(rows(Path(result["output_directory"])/"effective_results.csv")) == 96
    write(root/"controller.lock", {"pid": 123})
    with pytest.raises(summary.SummaryValidationError, match="controller is active"):
        summary.summarize(root, allow_partial=True)


def test_lower_scoring_complete_recovery_is_selected_without_optimism(tmp_path):
    root, retry_root = tmp_path/"run", tmp_path/"retry"
    protocol = fixture(root)
    recovery(root, retry_root, protocol)
    first_hash = sha(root/"jobs"/TARGET/"model.npz")
    result = summary.summarize(root, recovery_roots=[retry_root])
    output = Path(result["output_directory"])
    row = next(r for r in rows(output/"effective_results.csv") if r["job_id"] == TARGET)
    assert result["recoveries_applied"] == 1
    assert row["status"] == "complete" and float(row["resolution_clade_f1"]) == .8
    assert row["original_status"] == "failed" and float(row["original_resolution_clade_f1"]) == 1.
    assert row["original_retained_fallback"] == "True"
    assert float(row["original_elapsed_seconds"]) == 7 and float(row["retry_elapsed_seconds"]) == 4
    assert float(row["compute_seconds"]) == 4 and float(row["combined_spent_seconds"]) == 11
    assert len(rows(output/"first_attempt.csv")) == len(rows(output/"effective_results.csv")) == 96
    assert len(rows(output/"recovery_attempts.csv")) == 1
    manifest = read(output/"recording_recovery_manifest.json")
    assert manifest["recovery_attempts"][0]["metric_values_used_for_eligibility"] is False
    assert manifest["recovery_attempts"][0]["input_arrays_equal"]["P_source"] is True
    assert sha(root/"jobs"/TARGET/"model.npz") == first_hash


def test_unknown_attempt_time_keeps_combined_time_unknown(tmp_path):
    root, retry_root = tmp_path/"run", tmp_path/"retry"
    protocol = fixture(root)
    recovery(root, retry_root, protocol)
    path = root/"jobs"/TARGET/"result.json"
    value = read(path)
    value.update(elapsed_seconds=None, compute_seconds=None)
    write(path, value)
    result = summary.summarize(root, recovery_roots=[retry_root])
    row = next(r for r in rows(Path(result["output_directory"])/"effective_results.csv") if r["job_id"] == TARGET)
    assert row["original_elapsed_seconds"] == row["combined_spent_seconds"] == ""
    assert float(row["retry_elapsed_seconds"]) == 4.


@pytest.mark.parametrize("change,reason", [
    ("wrong_error", "original_error_is_not_PermissionError"),
    ("source_array", "saved_input_array_difference:P_source"),
    ("source_file", "retry_input_file_hash_mismatch:source/amplitudes.csv"),
    ("core_parameter", "model_configuration_difference:core_parameters"),
    ("budget", "model_configuration_difference:budget"),
    ("protocol", "nonrecording_protocol_difference"),
    ("retry_incomplete", "retry_is_not_a_complete_model"),
])
def test_recovery_rejected_for_non_equivalent_attempts(tmp_path, change, reason):
    root, retry_root = tmp_path/"run", tmp_path/"retry"
    protocol = fixture(root)
    retry_protocol, directory = recovery(root, retry_root, protocol)
    if change == "wrong_error":
        path = root/"jobs"/TARGET/"result.json"
        data = read(path)
        data["error"]["type"] = "ValueError"
        write(path, data)
    elif change == "source_array":
        with np.load(directory/"model.npz") as loaded:
            data = dict(loaded)
        data["P_source"] = data["P_source"]+1
        np.savez(directory/"model.npz", **data)
    elif change == "source_file":
        (retry_root/"inputs/soumalas11/no_source/source/amplitudes.csv").write_bytes(b"changed")
    elif change in ("core_parameter", "budget"):
        data = read(directory/"model.json")
        data["core_parameters" if change == "core_parameter" else "budget"]["new_unapproved_value"] = 123
        write(directory/"model.json", data)
    elif change == "protocol":
        retry_protocol["core_parameters_base"]["r_upper_bound"] = 4
        write(retry_root/"protocol.json", retry_protocol)
        refresh_bindings(root, retry_root)
    elif change == "retry_incomplete":
        data = read(directory/"result.json")
        data["status"] = "incomplete"
        write(directory/"result.json", data)
    result = summary.summarize(root, recovery_roots=[retry_root])
    assert result["recoveries_applied"] == 0
    output = Path(result["output_directory"])
    retry_row = rows(output/"recovery_attempts.csv")[0]
    assert reason in json.loads(retry_row["rejection_reasons"])
    row = next(r for r in rows(output/"effective_results.csv") if r["job_id"] == TARGET)
    assert row["status"] == "failed" and float(row["resolution_clade_f1"]) == 1.
    assert float(row["combined_spent_seconds"]) == 11


def test_only_reviewed_recording_code_change_is_allowed(tmp_path, monkeypatch):
    root, retry_root = tmp_path/"run", tmp_path/"retry"
    protocol = fixture(root)
    retry_protocol, _ = recovery(root, retry_root, protocol)
    source_pair = (b"old recorder\n", b"new recorder\n")
    monkeypatch.setattr(summary, "APPROVED_DRIVER_PAIR", tuple(hashlib.sha256(v).hexdigest() for v in source_pair))
    for r, config, source in ((root, protocol, source_pair[0]), (retry_root, retry_protocol, source_pair[1])):
        target = r/"source_snapshot"/summary.DRIVER
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
        config["source_sha256"][summary.DRIVER] = sha(target)
    retry_protocol["recording_policy"] = summary.RECORDING_POLICY
    write(root/"protocol.json", protocol)
    write(retry_root/"protocol.json", retry_protocol)
    refresh_bindings(root, retry_root)
    result = summary.summarize(root, recovery_roots=[retry_root])
    assert result["recoveries_applied"] == 1
    path = retry_root/"source_snapshot"/summary.DRIVER
    path.write_bytes(b"unreviewed algorithm change\n")
    retry_protocol["source_sha256"][summary.DRIVER] = sha(path)
    write(retry_root/"protocol.json", retry_protocol)
    refresh_bindings(root, retry_root)
    result = summary.summarize(root, recovery_roots=[retry_root], output_dir=tmp_path/"unapproved")
    assert result["recoveries_applied"] == 0
    reasons = read(Path(result["output_directory"])/"recording_recovery_manifest.json")["recovery_attempts"][0]["rejection_reasons"]
    assert "source_change_is_not_the_reviewed_recording_fix" in reasons


def test_screen_disagreement_is_retained_and_manifest_is_immutable(tmp_path):
    root = tmp_path/"run"
    fixture(root)
    summary.summarize(root)
    path = root/"jobs/paper15__no_source__RNJ+WZZT_ungrouped/rnj_screen.json"
    data = read(path)
    data["selected_support_labels"] = []
    write(path, data)
    with pytest.raises(summary.SummaryValidationError, match="manifest is frozen"):
        summary.summarize(root)
    result = summary.summarize(root, output_dir=tmp_path/"second-summary")
    condition = rows(Path(result["output_directory"])/"screen_conditions.csv")[0]
    assert condition["identical_screen_excluding_elapsed_and_requested_groups"] == "False"


def test_multiple_eligible_retries_are_not_chosen_by_score(tmp_path):
    root, retry_root = tmp_path/"run", tmp_path/"retry"
    protocol = fixture(root)
    recovery(root, retry_root, protocol)
    second = tmp_path/"second"
    shutil.copytree(retry_root, second)
    data = read(second/"jobs"/TARGET/"result.json")
    data["metrics"] = metrics(.99)
    write(second/"jobs"/TARGET/"result.json", data)
    with pytest.raises(summary.SummaryValidationError, match="multiple eligible"):
        summary.summarize(root, recovery_roots=[retry_root, second])
