"""Run one immutable MIQP worker with audited infrastructure-time accounting.

Only JSON I/O and the remaining total wall-time cap are adapted. This file is
executed by absolute path as __main__, outside the frozen project import tree.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter, sleep
import traceback

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"


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


def recover(source, payload_file):
    source, payload_file = Path(source).resolve(), Path(payload_file).resolve()
    p = read(payload_file)
    control = p["recovery_control"]
    output = Path(control["output"]).resolve()
    audit_file = Path(control["charges_file"]).resolve()
    if sha(audit_file) != control["charges_sha256"]:
        raise ValueError("The authoritative recovery ledger changed")
    audit = read(audit_file)
    protocol_file = output / "protocol.json"
    if sha(protocol_file) != p["protocol_sha256"] or p["protocol_sha256"] != audit["protocol_sha256"]:
        raise ValueError("Recovery protocol mismatch")
    protocol = read(protocol_file)
    if source != (output / "source").resolve() or Path(p["frozen_source"]).resolve() != source:
        raise ValueError("Recovery must load the original frozen source")
    if p["source_hashes"] != protocol["source_hashes"]:
        raise ValueError("Worker source manifest changed")
    key = p["job"] + "/" + p["method"]
    entry = audit["charges"].get(key)
    original_cap = float(protocol["limits"]["total"])
    charge = float(entry["interrupted_attempt_seconds"]) if entry else 0.
    if original_cap != 14400. or not math.isfinite(charge) or not 0. <= charge < original_cap:
        raise ValueError("Invalid recovery time debit")
    if float(control["interrupted_attempt_seconds"]) != charge:
        raise ValueError("Worker time debit differs from the authoritative ledger")
    if entry:
        original_file = Path(control["original_payload_file"]).resolve()
        if sha(original_file) != entry["preserved_files"]["payload.json"]:
            raise ValueError("Preserved original payload changed")
        if {k: v for k, v in p.items() if k != "recovery_control"} != read(original_file):
            raise ValueError("Recovery changed an original optimizer payload")
    elif control.get("original_payload_file") is not None:
        raise ValueError("A never-started job cannot have an interrupted payload")
    final = Path(p["result"]).resolve()
    expected_final = (output / "jobs" / p["job"] / p["method"] / "result.json").resolve()
    if final != expected_final or final.exists() or final.with_name("error.json").exists():
        raise FileExistsError("A completed result or recorded failure must never be overwritten")
    pending = final.with_name("result.pending.json")
    if pending.exists():
        raise FileExistsError("An existing pending result requires another explicit recovery audit")
    remaining = original_cap - charge
    sys.path[:0] = [str(source), str(source / "rnj_wzzt_core")]
    module = importlib.import_module("research_experiments.miqp_loss_benchmark_20261111.run")
    native_file = Path(inspect.getfile(module.worker)).resolve()
    relative = native_file.relative_to(source).as_posix()
    if sha(native_file) != protocol["source_hashes"][relative] or module.LIMITS != protocol["limits"]:
        raise ValueError("The native optimizer source or original limits changed")
    # The sole numerical change is the explicitly audited remaining total time.
    module.LIMITS = {**module.LIMITS, "total": remaining}
    module.write = write
    staged = {**p, "result": str(pending)}
    staged_payload = final.with_name("native_recovery_payload.json")
    write(staged_payload, staged)
    context = dict(
        wrapper_file=str(Path(__file__).resolve()), wrapper_sha256=sha(__file__),
        native_worker_file=str(native_file), native_worker_sha256=sha(native_file),
        payload_file=str(payload_file), payload_sha256=sha(payload_file),
        original_payload_file=control.get("original_payload_file"),
        original_payload_sha256=entry["preserved_files"]["payload.json"] if entry else None,
        recovery_charges_file=str(audit_file), recovery_charges_sha256=sha(audit_file),
        protocol_sha256=p["protocol_sha256"], job=p["job"], method=p["method"],
        original_total_cap=original_cap, remaining_total_cap=remaining,
        interrupted_attempt_seconds=charge, interrupted_attempt_count=int(charge > 0.),
        charge_is_conservative_upper_bound=bool(entry), common_charged_once=True,
        new_never_started_worker=entry is None, optimizer_source_modified=False,
        adjustments=["retry atomic JSON I/O on transient Windows reader locks",
                     "subtract audited interrupted-attempt upper wall-time debit"],
        started_utc=datetime.now(timezone.utc).isoformat())
    started = perf_counter()
    write(final.with_name("recovery.json"), {**context, "stage": "running"})
    try:
        module.worker(staged_payload)
        result = read(pending)
        successful = float(result["total_seconds"])
        logical = successful + charge
        if not math.isfinite(logical) or abs(successful - float(result["fresh_seconds"]) - float(result["common_seconds"])) > 1e-6:
            raise ValueError("Native timing does not reconcile")
        result.update(
            successful_total_seconds=successful, interrupted_attempt_seconds=charge,
            interrupted_attempt_count=int(charge > 0.), logical_total_seconds=logical,
            operational_total_seconds=logical, original_total_cap=original_cap,
            logical_total_cap_exceeded=logical > original_cap + 1.,
            recovery_charges_sha256=context["recovery_charges_sha256"],
            recovery={**context, "wrapper_wall_seconds": perf_counter() - started,
                      "native_pending_sha256": sha(pending),
                      "completed_utc": datetime.now(timezone.utc).isoformat()})
        # Publish last: readers never observe a native result without its debit.
        write(final, result)
        write(final.with_name("recovery.json"), {
            **result["recovery"], "stage": "published", "result_sha256": sha(final),
            "logical_total_seconds": logical, "successful_total_seconds": successful,
            "logical_total_cap_exceeded": result["logical_total_cap_exceeded"]})
        print("RECOVERY_PUBLISHED", key, f"charge={charge:.6f}", f"logical={logical:.6f}", flush=True)
    except Exception:
        write(final.with_name("error.json"), dict(stage="recovery", recovery=context, traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("payload", type=Path)
    args = parser.parse_args()
    recover(args.source, args.payload)
