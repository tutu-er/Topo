"""I/O-only recovery wrapper around the immutable native benchmark worker.

The optimizer and frozen protocol are unchanged. An explicitly charged previous
infrastructure interruption reduces the native worker's remaining wall budget.
The result is published only after recovery accounting has been attached.
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
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    for attempt in range(50):
        try:
            temporary.write_text(serialized, encoding="utf-8", newline="\n")
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 49:
                raise
            sleep(.05)


def recover(source, payload_file):
    source, payload_file = Path(source).resolve(), Path(payload_file).resolve()
    p = read(payload_file)
    final = Path(p["result"]).resolve()
    if final.exists():
        raise FileExistsError("A completed result must never be overwritten by a recovery attempt")
    pending = final.with_name("result.pending.json")
    if pending.exists():
        raise FileExistsError("An existing pending recovery must be reviewed before another attempt")
    original_cap = float(p.get("original_total_cap", 14400.))
    charge = float(p.get("interrupted_attempt_seconds", 0.))
    if not math.isfinite(charge) or not math.isfinite(original_cap) or not 0 <= charge < original_cap:
        raise ValueError("Invalid infrastructure interruption budget charge")
    remaining = original_cap-charge
    sys.path[:0] = [str(source), str(source / "rnj_wzzt_core")]
    module = importlib.import_module("research_experiments.day96_t_benchmark_20261110.run")
    native_file = Path(inspect.getfile(module.worker)).resolve()
    assert native_file.is_relative_to(source), native_file
    assert original_cap == float(module.LIMITS["total"]) == 14400.
    # Preserve every numerical setting; only the logically remaining total changes.
    native_limits = dict(module.LIMITS)
    module.LIMITS = {**native_limits, "total": remaining}
    module.write = write
    staged = {**p, "result": str(pending)}
    staged_payload = final.with_name("native_recovery_payload.json")
    write(staged_payload, staged)
    started = perf_counter()
    context = dict(wrapper_sha256=sha(__file__), native_worker_sha256=sha(native_file),
                   original_total_cap=original_cap, remaining_total_cap=remaining,
                   interrupted_attempt_seconds=charge, recovery_charges_sha256=p.get("recovery_charges_sha256"),
                   new_never_started_worker=charge == 0., optimizer_source_modified=False,
                   adjustments=["retry atomic JSON I/O on transient Windows reader locks", "subtract explicit interrupted-attempt wall charge"],
                   started_utc=datetime.now(timezone.utc).isoformat())
    write(final.with_name("recovery.json"), {**context, "stage": "running"})
    try:
        module.worker(staged_payload)
        result = read(pending)
        successful = float(result["total_seconds"])
        logical = successful+charge
        if logical > original_cap + 1.:
            raise RuntimeError(f"Logical wall budget exceeded: {logical} > {original_cap}")
        result.update(successful_total_seconds=successful, interrupted_attempt_seconds=charge,
                      interrupted_attempt_count=int(charge > 0.),
                      logical_total_seconds=logical, original_total_cap=original_cap,
                      logical_total_cap_exceeded=logical > original_cap + 1.,
                      recovery={**context, "wrapper_wall_seconds": perf_counter()-started,
                                "native_pending_sha256": sha(pending), "completed_utc": datetime.now(timezone.utc).isoformat()})
        # The finisher looks only for result.json and cannot observe the unaccounted native result.
        write(final, result)
        write(final.with_name("recovery.json"), {**context, "stage": "published", "logical_total_seconds": logical,
                                                "successful_total_seconds": successful, "wrapper_wall_seconds": perf_counter()-started})
        print("RECOVERY_PUBLISHED", p["job"], p["method"], f"charge={charge:.6f}", f"logical={logical:.6f}", flush=True)
    except Exception:
        write(final.with_name("error.json"), dict(stage="recovery", recovery=context, traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("payload", type=Path)
    args = parser.parse_args()
    recover(args.source, args.payload)
