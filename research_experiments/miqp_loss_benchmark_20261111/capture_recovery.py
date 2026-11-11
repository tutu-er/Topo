"""Preserve an audited host restart; never launch or stop a process."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, x): Path(p).write_text(json.dumps(x, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")

def capture(output, boot_utc, confirmed_no_live_workers):
    if not confirmed_no_live_workers:
        raise RuntimeError("An external host-process liveness audit is required")
    output=Path(output).resolve(); protocol=read(output/"protocol.json")
    state_file=output/"all_worker_state.json"; state=read(state_file)
    boot=datetime.fromisoformat(boot_utc.replace("Z", "+00:00"))
    heartbeat=datetime.fromtimestamp(state_file.stat().st_mtime, timezone.utc)
    assert heartbeat < boot < datetime.now(timezone.utc)
    for name, digest in {**protocol["source_hashes"], **protocol["source_artifact_hashes"]}.items():
        assert sha(output/"source"/name)==digest, name
    results={f.relative_to(output).as_posix():sha(f) for f in output.glob("jobs/*/*/result.json")}
    assert len(results)==state["returned"]==78 and state["failed"]==0 and len(state["active"])==8
    snapshot=output/"recovery/host_restart_20261111"
    snapshot.mkdir(parents=True, exist_ok=False)
    for name in ("all_worker_state.json", "all_common_state.json", "postprocess.json", "postprocess.lock"):
        f=output/name
        if f.exists(): shutil.copy2(f,snapshot/name)
    charges={}
    for entry in state["active"]:
        final=Path(entry["result"]).resolve(); folder=final.parent
        assert folder.is_relative_to((output/"jobs").resolve()) and not final.exists()
        attempt=read(folder/"attempt.json"); payload=folder/"payload.json"
        assert attempt["state"]=="running" and sha(payload)==attempt["payload_sha256"]
        p=read(payload); started=datetime.fromtimestamp(payload.stat().st_mtime,timezone.utc)
        charge=(boot-started).total_seconds()
        common=read(p["common"])
        assert 0 < charge < 14400.-common["seconds"]-600.
        dest=snapshot/"jobs"/p["job"]/p["method"]; dest.mkdir(parents=True)
        saved={}
        for f in folder.iterdir():
            if f.is_file():
                shutil.copy2(f,dest/f.name); saved[f.name]=sha(f)
        charges[p["job"]+"/"+p["method"]]=dict(
            job=p["job"], method=p["method"], interrupted_attempt_seconds=charge,
            original_total_cap=14400., remaining_total_cap=14400.-charge,
            payload_created_utc=started.isoformat(), last_controller_heartbeat_utc=heartbeat.isoformat(),
            last_seen_wall_proxy_seconds=(heartbeat-started).total_seconds(),
            termination_upper_bound_utc=boot.isoformat(), charge_is_conservative_upper_bound=True,
            charge_scope="pre-spawn payload timestamp through verified subsequent system boot; downtime excluded; common charged once in successful rerun",
            preserved_files=saved, original_pid=attempt["pid"], solver_failure=False)
    receipt=dict(protocol_sha256=sha(output/"protocol.json"),catalog_sha256=sha(output/"catalog.json"),
        reason="system restarted after code-mode stdout closed; original experiment processes no longer live",
        liveness_audit="host CIM query filtered to python processes with this experiment package returned none; original recorded processes absent or reused",
        verified_system_boot_utc=boot.isoformat(), captured_utc=datetime.now(timezone.utc).isoformat(),
        returned_preserved=78, interrupted_attempts=8, never_started=10,
        logical_models_planned=96, attempts_after_recovery_expected=104,
        charges=charges,result_hashes_before_recovery=results,
        total_interrupted_wall_upper_seconds=sum(c["interrupted_attempt_seconds"] for c in charges.values()),
        model_or_data_configuration_changed=False,optimizer_processes_stopped_by_capture=False,
        frozen_sources_verified=True,capture_script_sha256=sha(__file__))
    write(snapshot/"charges.json",receipt)
    # The stale finite-waiter lock is preserved above; no live waiter remains.
    lock=output/"postprocess.lock"
    if lock.exists(): lock.unlink()
    print(json.dumps({k:v for k,v in receipt.items() if k not in {"charges","result_hashes_before_recovery"}},indent=2))
    return receipt

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("output",type=Path)
    p.add_argument("--boot-utc",required=True);p.add_argument("--confirmed-no-live-workers",action="store_true")
    a=p.parse_args();capture(a.output,a.boot_utc,a.confirmed_no_live_workers)
