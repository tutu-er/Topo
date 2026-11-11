"""Validate and resume only the missing MIQP evaluations after an audited reboot.

This standalone manager never imports or edits optimizer code, never uses old
PIDs, and never launches during `validate`. Run requires a captured ledger.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep


def read(path):
    for attempt in range(100):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 99:
                raise
            sleep(.05)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    for attempt in range(100):
        try:
            temporary.write_text(serialized, encoding="utf-8", newline="\n")
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            sleep(.05)


def preserved_results(output, audit):
    for name, digest in audit["result_hashes_before_recovery"].items():
        path = (output / name).resolve()
        if not path.is_relative_to((output / "jobs").resolve()) or sha(path) != digest:
            raise ValueError(f"A pre-recovery completed result changed: {name}")


def status(output, protocol):
    returned = failed = 0
    for job in protocol["jobs"]:
        for method in protocol["methods"]:
            final = output / "jobs" / job / method["id"] / "result.json"
            returned += final.exists()
            failed += final.with_name("error.json").exists() and not final.exists()
    return dict(planned=protocol["planned"], returned=returned, failed=failed,
                pending=protocol["planned"] - returned - failed)


def build_plan(output, audit_file):
    output, audit_file = Path(output).resolve(), Path(audit_file).resolve()
    protocol_file = output / "protocol.json"
    protocol, catalog, audit = read(protocol_file), read(output / "catalog.json"), read(audit_file)
    if sha(protocol_file) != audit["protocol_sha256"]:
        raise ValueError("The original frozen protocol changed")
    if sha(output / "catalog.json") != protocol["catalog_sha256"] or protocol["catalog_sha256"] != audit["catalog_sha256"]:
        raise ValueError("The original frozen catalog changed")
    if not audit["frozen_sources_verified"] or audit["model_or_data_configuration_changed"]:
        raise ValueError("The capture does not certify unchanged frozen sources/configuration")
    source = output / "source"
    for name, digest in {**protocol["source_hashes"], **protocol["source_artifact_hashes"]}.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or sha(path) != digest:
            raise ValueError(f"Frozen optimizer source changed: {name}")
    parent = Path(protocol["parent_output"])
    if sha(parent / "protocol.json") != protocol["parent_protocol_sha256"] or sha(parent / "catalog.json") != protocol["parent_catalog_sha256"]:
        raise ValueError("The parent frozen protocol/catalog changed")
    preserved_results(output, audit)
    cap = float(protocol["limits"]["total"])
    if cap != 14400. or int(protocol["planned"]) != 96:
        raise ValueError("Recovery is restricted to the existing 96-model, four-hour protocol")
    all_keys = {j + "/" + m["id"] for j in protocol["jobs"] for m in protocol["methods"]}
    if not set(audit["charges"]).issubset(all_keys):
        raise ValueError("The ledger names an unknown model")
    for key, entry in audit["charges"].items():
        charge = float(entry["interrupted_attempt_seconds"])
        if not math.isfinite(charge) or not 0. < charge < cap or float(entry["original_total_cap"]) != cap:
            raise ValueError(f"Invalid interruption debit: {key}")
        if abs(float(entry["remaining_total_cap"]) - (cap - charge)) > 1e-7:
            raise ValueError(f"Invalid remaining cap: {key}")
        saved_dir = audit_file.parent / "jobs" / entry["job"] / entry["method"]
        for name, digest in entry["preserved_files"].items():
            if sha(saved_dir / name) != digest:
                raise ValueError(f"A preserved interrupted-attempt file changed: {key}/{name}")
    queue = []
    for job in protocol["jobs"]:
        meta = catalog[job]
        if meta["data_error"]:
            raise ValueError(f"Data failure requires explicit review: {job}")
        for split in ("train", "validation"):
            if sha(meta[split]) != protocol["input_hashes"][job][split]:
                raise ValueError(f"A shared input changed: {job}/{split}")
        parent_common = parent / "common" / job / "common.json"
        if sha(parent_common) != protocol["parent_common_hashes"][job]:
            raise ValueError(f"Parent shared preprocessing changed: {job}")
        common = output / "common" / job / "common.json"
        common_value = read(common)
        if common_value["protocol_sha256"] != audit["protocol_sha256"]:
            raise ValueError(f"QP common references another protocol: {job}")
        for flags in protocol["methods"]:
            key = job + "/" + flags["id"]
            final = output / "jobs" / job / flags["id"] / "result.json"
            if final.exists() or final.with_name("error.json").exists():
                continue
            entry = audit["charges"].get(key)
            charge = float(entry["interrupted_attempt_seconds"]) if entry else 0.
            if cap - charge <= float(common_value["seconds"]) + float(protocol["limits"]["post_reserve"]):
                raise ValueError(f"Insufficient audited remaining time for native execution: {key}")
            original_payload = None
            if entry:
                original_payload = audit_file.parent / "jobs" / job / flags["id"] / "payload.json"
                payload = read(original_payload)
                original_attempt = read(original_payload.with_name("attempt.json"))
                if original_attempt["payload_sha256"] != sha(original_payload):
                    raise ValueError(f"Original payload hash mismatch: {key}")
                if payload["common_sha256"] != sha(common):
                    raise ValueError(f"The interrupted common preprocessing changed: {key}")
            else:
                if final.with_name("attempt.json").exists() or final.with_name("payload.json").exists():
                    raise RuntimeError(f"An uncharged previous attempt needs another audit: {key}")
                payload = dict(
                    job=job, method=flags["id"], flags=flags, train=meta["train"], validation=meta["validation"],
                    common=str(common), common_sha256=sha(common),
                    input_hashes={s: protocol["input_hashes"][job][s] for s in ("train", "validation")},
                    result=str(final), protocol_sha256=sha(protocol_file), source_hashes=protocol["source_hashes"],
                    root_label=meta["root_label"], base_kw=meta["base_kw"],
                    parent=dict(output=protocol["parent_output"], protocol_sha256=protocol["parent_protocol_sha256"],
                                common_sha256=protocol["parent_common_hashes"][job],
                                input_hashes=protocol["input_hashes"][job], source_file_count=protocol["parent_source_file_count"]),
                    frozen_source=str(source))
            if payload["job"] != job or payload["method"] != flags["id"] or payload["flags"] != flags:
                raise ValueError(f"An original payload does not match its model: {key}")
            payload["recovery_control"] = dict(
                output=str(output), charges_file=str(audit_file), charges_sha256=sha(audit_file),
                interrupted_attempt_seconds=charge,
                original_payload_file=str(original_payload) if original_payload else None)
            queue.append(payload)
    summary = dict(
        **status(output, protocol), missing_to_launch=len(queue),
        interrupted_to_resume=sum(p["recovery_control"]["interrupted_attempt_seconds"] > 0. for p in queue),
        never_started_to_launch=sum(p["recovery_control"]["interrupted_attempt_seconds"] == 0. for p in queue),
        preserved_completed_count=len(audit["result_hashes_before_recovery"]),
        interrupted_wall_upper_seconds=sum(c["interrupted_attempt_seconds"] for c in audit["charges"].values()),
        remaining_queued_method_seconds=sum(cap-p["recovery_control"]["interrupted_attempt_seconds"] for p in queue),
        original_total_cap=cap, protocol_sha256=sha(protocol_file), recovery_charges_sha256=sha(audit_file),
        frozen_sources_verified=True, model_or_data_configuration_changed=False, launched=False)
    return output, source, protocol, audit, queue, summary


def run(output, audit_file, workers=8):
    if not 1 <= workers <= 8:
        raise ValueError("Use 1..8 workers, with the original one solver thread each")
    output, source, protocol, audit, queue, summary = build_plan(output, audit_file)
    recovery = Path(audit_file).resolve().parent
    lock = recovery / "manager.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat())) + "\n")
    active = {}
    started = perf_counter()
    wrapper = Path(__file__).with_name("recovery_worker.py").resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source) + os.pathsep + str(source / "rnj_wzzt_core")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    try:
        plan = dict(summary, manager_file=str(Path(__file__).resolve()), manager_sha256=sha(__file__),
                    wrapper_file=str(wrapper), wrapper_sha256=sha(wrapper), workers=workers,
                    queue=[dict(job=p["job"], method=p["method"],
                                interrupted_attempt_seconds=p["recovery_control"]["interrupted_attempt_seconds"]) for p in queue])
        plan_file = recovery / "recovery_plan.json"
        if plan_file.exists():
            raise FileExistsError("A prior recovery plan requires explicit review before another manager starts")
        write(plan_file, plan)
        while queue or active:
            while queue and len(active) < workers:
                payload = queue.pop(0)
                target = Path(payload["result"])
                folder = recovery / "new_attempts" / payload["job"] / payload["method"]
                folder.mkdir(parents=True, exist_ok=False)
                file = folder / "payload.json"
                attempt = folder / "attempt.json"
                write(file, payload)
                log = (folder / "worker.log").open("w", encoding="utf-8")
                try:
                    proc = subprocess.Popen([sys.executable, "-B", str(wrapper), "--source", str(source), str(file)],
                                            cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                except Exception:
                    log.close()
                    raise
                # Track the live child before any further disk write: if an
                # I/O failure persists, retain the lock while it may be active.
                active[str(target)] = (proc, log, target, attempt)
                write(attempt, dict(state="running", pid=proc.pid, started_utc=datetime.now(timezone.utc).isoformat(),
                                    result=str(target), payload_sha256=sha(file), wrapper_sha256=sha(wrapper),
                                    recovery_charges_sha256=summary["recovery_charges_sha256"]))
            for key, (proc, log, target, attempt) in list(active.items()):
                if proc.poll() is None:
                    continue
                log.close(); del active[key]
                if not target.exists() and not target.with_name("error.json").exists():
                    write(target.with_name("error.json"), dict(stage="recovery_worker", exit_code=proc.returncode,
                                                               attempt_file=str(attempt), recovery_charges_sha256=summary["recovery_charges_sha256"]))
                record = read(attempt)
                write(attempt, dict(record, state="finished", exit_code=proc.returncode,
                                    finished_utc=datetime.now(timezone.utc).isoformat(),
                                    result_sha256=sha(target) if target.exists() else None))
                print("RECOVERY_FINISHED", target.parent.parent.name, target.parent.name,
                      "OK" if target.exists() else "ERROR", flush=True)
            write(recovery / "recovery_state.json", dict(
                **status(output, protocol), active=[dict(result=k, pid=v[0].pid) for k, v in active.items()],
                queued=len(queue), wall_seconds=perf_counter()-started,
                protocol_sha256=summary["protocol_sha256"], recovery_charges_sha256=summary["recovery_charges_sha256"]))
            if active:
                sleep(2.)
        preserved_results(output, audit)
        complete = dict(**status(output, protocol), preserved_completed_hashes_verified=True,
                        protocol_sha256=summary["protocol_sha256"], recovery_charges_sha256=summary["recovery_charges_sha256"],
                        manager_sha256=sha(__file__), wrapper_sha256=sha(wrapper),
                        wall_seconds=perf_counter()-started, finished_utc=datetime.now(timezone.utc).isoformat())
        write(recovery / "recovery_completion.json", complete)
        print(json.dumps(complete, indent=2))
    finally:
        # On a manager exception, leave the lock while its children may be live.
        # A new attempt always needs an explicit liveness and budget audit.
        if not active:
            lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "run"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "validate":
        print(json.dumps(build_plan(args.output, args.audit)[-1], indent=2))
    else:
        run(args.output, args.audit, args.workers)


if __name__ == "__main__":
    main()
