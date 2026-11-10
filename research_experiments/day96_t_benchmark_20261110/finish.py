"""Finite completion watcher: audit saved results, plot, then exit once.

This helper never generates data, trains a model, resumes an optimizer or
starts a recurring automation.  Start it only after the full batch is running.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter, sleep
import traceback

ROOT = Path(__file__).resolve().parents[2]


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


def counts(output, protocol):
    returned = failed = 0
    for job in protocol["jobs"]:
        for method in protocol["methods"]:
            path = output / "jobs" / job / method["id"] / "result.json"
            returned += path.exists()
            failed += path.with_name("error.json").exists() and not path.exists()
    return dict(planned=protocol["planned"], returned=returned, failed=failed,
                pending=protocol["planned"]-returned-failed)


def finish(output, *, workers=2, interval=20.):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("The diagnostic output must stay inside this repository's outputs directory")
    protocol = read(output / "protocol.json")
    assert protocol["planned"] == 144 and len(protocol["jobs"]) == 48
    lock = output / "postprocess.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    started = perf_counter()
    receipt = dict(stage="waiting", passed=False, visual_review_pending=True,
                   created_utc=datetime.now(timezone.utc).isoformat(), helper_pid=os.getpid(),
                   helper_sha256=sha(__file__), protocol_sha256=sha(output / "protocol.json"),
                   optimizer_started_by_helper=False, recurring_automation=False)
    target = output / "postprocess.json"
    try:
        while True:
            state = counts(output, protocol)
            receipt.update(state, elapsed_seconds=perf_counter()-started)
            write(target, receipt)
            if state["pending"] == 0:
                break
            sleep(min(max(interval, 1.), 60.))
        source = Path(__file__).parent
        used = [source / name for name in ("analyze.py", "candidates.py", "finish.py", "figures.py")]
        snapshot = output / "analysis_source"
        snapshot.mkdir(parents=True, exist_ok=True)
        analysis_hashes = {}
        for file in used:
            shutil.copy2(file, snapshot / file.name)
            analysis_hashes[file.name] = sha(file)
        receipt.update(stage="analyzing", analysis_source_hashes=analysis_hashes)
        write(target, receipt)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + str(ROOT / "rnj_wzzt_core")
        with (output / "postprocess.log").open("a", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-B", str(source / "analyze.py"), str(output), "--workers", str(workers)],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            receipt["analysis_exit_code"] = result.returncode
            if result.returncode:
                raise RuntimeError("Independent result analysis failed; see postprocess.log")
            verification = read(output / "analysis/verification.json")
            hashes_stable = all(sha(file) == analysis_hashes[file.name] for file in used)
            receipt.update(verification=verification, analysis_source_unchanged=hashes_stable,
                           passed=bool(verification.get("passed") and verification.get("verified") == 144 and hashes_stable))
            if receipt["passed"]:
                receipt["stage"] = "plotting"; write(target, receipt)
                result = subprocess.run([sys.executable, "-B", str(source / "figures.py"), str(output)], cwd=ROOT, env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
                receipt["figures_exit_code"] = result.returncode
                if result.returncode:
                    receipt["passed"] = False
                    raise RuntimeError("Plotting failed after numerical verification")
            else:
                receipt["figures_skipped"] = "The planned 144 evaluations are not all independently verified"
        receipt.update(stage="complete" if receipt["passed"] else "incomplete_or_failed", elapsed_seconds=perf_counter()-started,
                       completed_utc=datetime.now(timezone.utc).isoformat())
        write(target, receipt)
        print(json.dumps({k: receipt[k] for k in ("stage", "passed", "returned", "failed", "pending", "elapsed_seconds", "visual_review_pending")}), flush=True)
        return receipt
    except Exception:
        receipt.update(stage="postprocess_error", passed=False, elapsed_seconds=perf_counter()-started,
                       traceback=traceback.format_exc())
        write(target, receipt)
        raise
    finally:
        # This is the exclusive single file created above, inside the checked output directory.
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--interval", type=float, default=20.)
    args = parser.parse_args()
    finish(args.output, workers=args.workers, interval=args.interval)
