"""Postprocessing fixtures only: no estimation, AC simulation or solver."""
import csv
from copy import deepcopy
import json

import pytest

from research_experiments.rnj import summarize_source_ac288 as report
from research_experiments.rnj import audit_source_ac96_benchmark as scoring


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def point(supports, r, x, loss):
    return {"support_indices": supports, "support_labels": [[301+i for i in s] for s in supports],
            "r_values": r, "x_values": x, "train_mae": loss, "iteration": len(supports),
            "accepted_gain": .01, "solver": {"status": 0, "objective": loss, "dual_bound": loss, "mip_gap": 0.}}


def fixture(tmp_path):
    directory = tmp_path / "raw"
    protocol = {"jobs": [{"variant": v, "method": m, "per_solve_seconds": 6000. if "WZZT" in m else None,
                         "method_wall_seconds": 30000. if "WZZT" in m else 300.}
                        for v in report.VARIANTS for m in report.METHODS],
                "training_samples": 288, "validation_samples": 0}
    save(directory / "protocol.json", protocol)
    return directory


def install_job(directory, variant="no_source", method="WZZT", *, result=True):
    folder = directory / "jobs" / f"pengwah18__{variant}__{method}"
    p = point([[0], [1], [2], [3], [0, 1]], [.1]*4+[0.], [.1]*4+[0.], .02)
    model = {"has_model": True, "terminal_labels": [301, 302, 303, 304], "bound_expansions": 1,
             "path": [p], "selected_path_index": 0, "train_mae": .02,
             "terminal_groups": [[0], [1], [2], [3]], "matrix_origin": "fixture"}
    save(folder / "model.json", model)
    if result:
        save(folder / "result.json", {"status": "incomplete", "stop_reason": "extension_not_proven_optimal",
            "has_model": True, "search_complete": False, "compute_seconds": 6000.,
            "selected_path_index": 0, "metrics": {"family_clade_f1": .4, "active_clade_f1": .2,
                "r_relative_error": .12, "x_relative_error": .34}})
    return folder, model


def test_full_denominator_incomplete_scores_and_posthoc_paths_do_not_replace_scores(tmp_path, monkeypatch):
    directory = fixture(tmp_path)
    folder, model = install_job(directory)
    # A previous accepted path disappears after a coefficient-bound restart.
    old = deepcopy(model)
    old["bound_expansions"] = 0
    old["path"][0]["train_mae"] = .03
    save(folder / "progress/000001.json", old)
    save(folder / "progress/000002.json", old)  # repeated snapshot is not another point
    save(folder / "progress/000003.json", model)
    failed = directory / "jobs/pengwah18__constant__WZZT"
    save(failed / "result.json", {"status": "failed", "stop_reason": "worker_process_failed", "has_model": False})
    # Saved path family has perfect F1; its zero-weight active family is empty.
    monkeypatch.setattr(scoring, "physical_truth", lambda path, labels: {"clades": [[301, 302]]})
    output = tmp_path / "preview"
    metadata = report.summarize(directory, output, dataset_dir=tmp_path / "physical_only")
    with (output / "summary.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 16 and sum(r["result_available"] == "True" for r in rows) == 2
    actual = next(r for r in rows if r["job_id"] == folder.name)
    assert float(actual["family_clade_f1"]) == .4  # primary saved score remains unchanged
    assert float(actual["r_error_percent"]) == 12.
    missing = next(r for r in rows if r["variant"] == "smooth" and r["method"] == "NJ")
    assert missing["family_clade_f1"] == missing["compute_seconds"] == "" and missing["status"] == "not_run"
    with (output / "accepted_paths.csv").open(encoding="utf-8-sig") as handle:
        paths = list(csv.DictReader(handle))
    assert len(paths) == 2
    assert {p["family_clade_f1"] for p in paths} == {"1.0"}
    assert {p["active_clade_f1"] for p in paths} == {"0.0"}
    assert [p["selected_final_model"] for p in paths] == ["False", "True"]
    with (output / "aggregate_by_method.csv").open(encoding="utf-8-sig") as handle:
        aggregate = next(r for r in csv.DictReader(handle) if r["method"] == "WZZT")
    assert aggregate["expected"] == "4" and aggregate["models_available"] == "1"
    assert aggregate["incomplete"] == aggregate["failed"] == "1"
    assert aggregate["mean_family_clade_f1"] == "0.4" and aggregate["family_clade_f1_n"] == "1"
    assert metadata["expected_jobs"] == 16 and metadata["results_available"] == 2
    text = (output / "README.md").read_text(encoding="utf-8")
    assert "288 点全部训练" in text and "无验证集" in text and "临时快照" in text
    assert "没有哈希记录" in text and "拓扑收敛" in text


def test_checkpoint_is_provisional_and_truth_failure_preserves_path_row(tmp_path):
    directory = fixture(tmp_path)
    folder, _ = install_job(directory, result=False)
    output = tmp_path / "preview"
    report.summarize(directory, output, dataset_dir=tmp_path / "missing_physical")
    with (output / "summary.csv").open(encoding="utf-8-sig") as handle:
        row = next(r for r in csv.DictReader(handle) if r["job_id"] == folder.name)
    assert row["status"] == "running" and row["has_model"] == "False" and row["checkpoint_has_model"] == "True"
    assert row["family_clade_f1"] == row["train_mae"] == "" and row["partition_scope"] == "checkpoint"
    with (output / "accepted_paths.csv").open(encoding="utf-8-sig") as handle:
        paths = list(csv.DictReader(handle))
    assert len(paths) == 1 and paths[0]["selected_checkpoint"] == "True"
    assert paths[0]["family_clade_f1"] == "" and "FileNotFoundError" in paths[0]["evaluation_error"]


def test_bootstrap_denominators_and_distinct_veto_reasons(tmp_path):
    directory = fixture(tmp_path)
    folder, model = install_job(directory, method="RNJ+WZZT")
    save(folder / "rnj_screen.json", {"selected_support_labels": [[301, 302, 303]],
        "selected_support_indices": [[0, 1, 2]],
        "confidence_records": [{"clade": [301, 302, 303], "frequency": .9},
                               {"clade": [301, 302, 304], "frequency": .8},
                               {"clade": [302, 303, 304], "frequency": .74}],
        "parameters": {"bootstrap_replicates": 100, "confidence_threshold": .75, "maximum_subclade_frequency": .1},
        "diagnostics": {"full_peers": [[301, 302, 303], [301, 302, 304], [302, 303, 304]],
            "eligible": [[301, 302, 303]], "full_fine_subclades": [[301, 302, 304]],
            "minimum_block_size": 3, "source_adjusted_identifiable": True,
            "valid_bootstrap_replicates": 95, "unidentifiable_bootstrap_replicates": 5,
            "subclade_frequency_records": [{"clade": [301, 302, 303], "frequency": 0.},
                                           {"clade": [301, 302, 304], "frequency": .2}]}})
    rows, evidence = report.collect(directory, report.read_json(directory / "protocol.json"))
    candidates = report.bootstrap_candidates(rows, evidence)
    assert len(candidates) == 3
    selected = next(c for c in candidates if c["selected_saved"])
    assert selected["selected_support_indices"] == "[0,1,2]" and selected["rejection_reasons"] == "[]"
    assert selected["peer_frequency_denominator"] == 100 and selected["subclade_frequency_denominator"] == 95
    veto = next(c for c in candidates if c["full_fine_subclade_veto"])
    assert set(json.loads(veto["rejection_reasons"])) == {"bootstrap_internal_subclade_veto", "full_data_fine_tolerance_subclade_veto"}
    assert any("peer_frequency_below_threshold" in c["rejection_reasons"] for c in candidates)


def test_hash_records_verify_raw_files_and_never_follow_escape_path(tmp_path):
    directory = fixture(tmp_path)
    folder, _ = install_job(directory)
    source = directory / "inputs/no_source/scenario_0/source/amplitudes.csv"
    source.parent.mkdir(parents=True)
    source.write_text("t,P_pu,Q_pu\n0,0,0\n")
    save(folder / "inputs_hash.json", {"file_sha256": {"scenario_0/source/amplitudes.csv": report.sha256(source),
        "scenario_1/missing.csv": "missing", "../../outside.txt": "escape"}})
    rows, evidence = report.collect(directory, report.read_json(directory / "protocol.json"))
    records = report.raw_input_hashes(directory, rows, evidence)
    assert len(records) == 3 and sum(r["sha256_matches"] for r in records) == 1
    assert next(r for r in records if r["relative_input_file"].startswith(".."))["read_error"]


def test_existing_raw_and_report_files_are_never_overwritten(tmp_path):
    directory = fixture(tmp_path)
    before = (directory / "protocol.json").read_bytes()
    with pytest.raises(ValueError, match="separate"):
        report.summarize(directory, directory)
    with pytest.raises(ValueError, match="evidence"):
        report.summarize(directory, directory / "jobs/report")
    output = tmp_path / "preview"
    output.mkdir()
    sentinel = output / "summary.csv"
    sentinel.write_text("keep existing report")
    with pytest.raises(FileExistsError):
        report.summarize(directory, output)
    assert sentinel.read_text() == "keep existing report"
    assert (directory / "protocol.json").read_bytes() == before
