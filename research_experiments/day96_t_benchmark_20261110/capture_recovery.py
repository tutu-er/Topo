"""Preserve an infrastructure interruption; do not stop or launch any process."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/day96_t_benchmark_20261110"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    latest = OUTPUT / "all_state.json.tmp"
    assert latest.is_file(), "The failed atomic replacement must be preserved."
    state = json.loads(latest.read_text(encoding="utf-8"))
    recovery = OUTPUT / "recovery"
    snapshot = recovery / "manager_interruption_20261110"
    snapshot.mkdir(parents=True, exist_ok=False)
    for name in ("all_state.json", "all_state.json.tmp"):
        shutil.copy2(OUTPUT / name, snapshot / name)
    crash_time = latest.stat().st_mtime
    records, charges, before_hashes = [], {}, {}
    for file in (OUTPUT / "jobs").rglob("result.json"):
        before_hashes[str(file.relative_to(OUTPUT)).replace("\\", "/")] = sha(file)
    for active in state["active"]:
        job, method = active["job"], active["method"]
        folder = (OUTPUT / "jobs" / job / method).resolve()
        assert folder.is_relative_to((OUTPUT / "jobs").resolve())
        assert not (folder / "result.json").exists()
        payload = folder / "payload.json"
        elapsed = max(0., crash_time - payload.stat().st_mtime)
        record = dict(job=job, method=method, pid=active["pid"], interrupted_attempt_seconds=elapsed,
                      interrupted_attempt_count=1, original_total_cap=14400.,
                      payload_created_utc=datetime.fromtimestamp(payload.stat().st_mtime, timezone.utc).isoformat(),
                      interruption_utc=datetime.fromtimestamp(crash_time, timezone.utc).isoformat(),
                      charge_scope="conservative wall-clock estimate from pre-spawn payload write to failed state replacement; includes Python startup",
                      solver_failure=False, reason="manager Windows atomic-state replacement denied by an open reader")
        dest = snapshot / "jobs" / job / method
        dest.mkdir(parents=True)
        preserved = {}
        for name in ("payload.json", "checkpoint.json", "progress.json", "worker.log", "error.json"):
            file = folder / name
            if file.exists():
                shutil.copy2(file, dest / name)
                preserved[name] = sha(file)
        record["preserved_files"] = preserved
        records.append(record)
        charges[job+"/"+method] = record
    value = dict(snapshot=str(snapshot.relative_to(OUTPUT)).replace("\\", "/"),
                 manager_error="PermissionError: [WinError 5] all_state.json.tmp -> all_state.json",
                 returned_preserved=len(before_hashes), interrupted_attempts=len(records),
                 charges=charges, result_hashes_before_recovery=before_hashes,
                 scheduler_tmp_sha256=sha(snapshot / "all_state.json.tmp"),
                 total_interrupted_wall_seconds=sum(r["interrupted_attempt_seconds"] for r in records),
                 attempted_evaluations_after_retry_expected=144+len(records),
                 model_or_data_configuration_changed=False,
                 active_worker_liveness_checked_separately=True, optimizer_processes_stopped_by_this_script=False)
    (recovery / "charges.json").write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in value.items() if k not in {"charges", "result_hashes_before_recovery"}}))


if __name__ == "__main__":
    main()
