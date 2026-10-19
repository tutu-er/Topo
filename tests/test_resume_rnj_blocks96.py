"""File-only supervisor tests; never start an estimator or subprocess."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_experiments.rnj import resume_rnj_blocks96 as subject


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8-sig")


def test_wrong_protocol_refused_before_archive(tmp_path):
    root = tmp_path/"outputs"/subject.RUN_NAME
    write(root/"protocol.json", {"expected_jobs": 64, "concurrency": 4, "solver": "gurobi"})
    with pytest.raises(ValueError, match="pinned original"):
        subject.build_plan(workspace=tmp_path)
    assert not (tmp_path/"outputs/rnj_blocks96_interruption_archive_20261019").exists()


def test_cim_proof_bom_other_ev_allowed_but_old_pid_or_batch_rejected(tmp_path):
    root, proof = tmp_path/"outputs"/subject.RUN_NAME, tmp_path/"proof.json"
    data = {"captured_utc": datetime.now(timezone.utc).isoformat(), "complete_inventory": True,
            "target_root": str(root), "processes": [{"ProcessId": 77, "CommandLine": "python ev_other.py"}]}
    write(proof, data)
    assert subject.proof_check(proof, root, [12])["old_pids_absent"] == [12]
    data["processes"][0]["ProcessId"] = 12
    write(proof, data)
    with pytest.raises(ValueError, match="PID"):
        subject.proof_check(proof, root, [12])
    data["processes"] = [{"ProcessId": 99, "CommandLine": "python finalize_rnj_blocks96_when_complete.py"}]
    write(proof, data)
    with pytest.raises(ValueError, match="finalizer"):
        subject.proof_check(proof, root, [12])
    data["processes"] = []
    data["captured_utc"] = (datetime.now(timezone.utc)-timedelta(hours=1)).isoformat()
    write(proof, data)
    with pytest.raises(ValueError, match="fresh"):
        subject.proof_check(proof, root, [12])


def test_archive_preserves_entire_interrupted_attempt_and_refuses_changed_bytes(tmp_path):
    source, target = tmp_path/"jobs/interrupted", tmp_path/"archive/interrupted_jobs/interrupted"
    write(source/"payload.json", {"original_absolute_job_dir": str(source)})
    write(source/"progress/000001.json", {"status": "running", "elapsed_seconds": 123.})
    (source/"raw.npz").write_bytes(b"original-solver-binary\x00\xff")
    saved = subject.inventory(source)
    plan = {"workspace": str(tmp_path), "archive_dir": str(tmp_path/"archive"),
            "archive_items": [{"source": str(source), "target": str(target), "inventory": saved}]}
    subject.archive_attempts(plan)
    assert not source.exists()
    assert subject.inventory(target) == saved
    assert json.loads((target/"payload.json").read_text(encoding="utf-8-sig"))["original_absolute_job_dir"] == str(source)
    assert (tmp_path/"archive/archive_mapping.json").is_file()
    with pytest.raises(FileExistsError):
        subject.archive_attempts(plan)
    write(tmp_path/"changed/source.json", {"value": 1})
    record = subject.inventory(tmp_path/"changed")
    write(tmp_path/"changed/source.json", {"value": 2})
    bad = {"workspace": str(tmp_path), "archive_dir": str(tmp_path/"archive_changed"),
           "archive_items": [{"source": str(tmp_path/"changed"), "target": str(tmp_path/"archive_changed/raw"), "inventory": record}]}
    with pytest.raises(RuntimeError, match="source/target changed"):
        subject.archive_attempts(bad)
    assert (tmp_path/"changed/source.json").exists()


def test_absolute_move_escape_rejected(tmp_path):
    with pytest.raises(ValueError, match="strictly inside"):
        subject.inside(tmp_path.parent/"outside", tmp_path)
    with pytest.raises(ValueError, match="strictly inside"):
        subject.inside(tmp_path, tmp_path)


def test_passive_prepare_hash_guard_preserves_final_and_blocks_mutation(tmp_path):
    final = tmp_path/"jobs/final/result.json"
    write(final, {"status": "incomplete", "compute_seconds": 30000.})
    protocol = tmp_path/"protocol.json"
    write(protocol, {"existing": True})
    state = tmp_path/"state"
    state.mkdir()
    plan = {"run_root": str(tmp_path), "state_dir": str(state), "fixed_sha256": {str(protocol): subject.sha(protocol)},
            "protected_final_sha256": {str(final): subject.sha(final)}}
    called = []
    def unchanged(args):
        called.append(args)
        return "original native pending queue"
    assert subject.checked_prepare(unchanged, SimpleNamespace(), plan) == "original native pending queue"
    assert len(called) == 1
    assert (state/"native_prepare_verified.json").is_file()
    def changed(_):
        write(final, {"status": "complete"})
    with pytest.raises(RuntimeError, match="protected file changed"):
        subject.checked_prepare(changed, SimpleNamespace(), plan)


def test_execute_without_process_proof_has_no_side_effects(tmp_path):
    with pytest.raises(ValueError, match="cim-proof"):
        subject.execute({"process_proof": None})
    assert list(tmp_path.iterdir()) == []
