"""One finite waiter for the 96 new MIQP evaluations; never starts optimization."""
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
    for attempt in range(100):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 99: raise
            sleep(.05)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+"\n"
    for attempt in range(100):
        try:
            temporary.write_text(text, encoding="utf-8", newline="\n"); temporary.replace(path)
            return
        except PermissionError:
            if attempt == 99: raise
            sleep(.05)


def counts(output, protocol):
    returned = failed = 0
    for job in protocol["jobs"]:
        for method in protocol["methods"]:
            file = output/"jobs"/job/method["id"]/"result.json"
            returned += file.exists()
            failed += file.with_name("error.json").exists() and not file.exists()
    return dict(planned=protocol["planned"], returned=returned, failed=failed,
                pending=protocol["planned"]-returned-failed)


def finish(output, *, workers=2, interval=20.):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT/"outputs").resolve()):
        raise ValueError("Only a repository output directory is writable by this finite waiter")
    protocol = read(output/"protocol.json")
    assert protocol["planned"] == 96 and len(protocol["jobs"]) == 48
    lock = output/"postprocess.lock"
    with lock.open("x", encoding="utf-8") as stream: stream.write(str(os.getpid()))
    started = perf_counter()
    receipt = dict(stage="waiting", passed=False, visual_review_pending=True,
                   helper_pid=os.getpid(), helper_sha256=sha(__file__),
                   protocol_sha256=sha(output/"protocol.json"), created_utc=datetime.now(timezone.utc).isoformat(),
                   new_optimizer_calls=0, new_fit_calls_by_helper=0, recurring_automation=False)
    target = output/"postprocess.json"
    try:
        while True:
            receipt.update(counts(output, protocol), elapsed_seconds=perf_counter()-started)
            write(target, receipt)
            if receipt["pending"] == 0: break
            sleep(min(max(interval, 1.), 60.))
        source = Path(__file__).parent
        files = [source/name for name in ("analyze.py", "candidates.py", "figures.py", "finish.py", "deliver.py")]
        snapshot = output/"analysis_source"; snapshot.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for file in files:
            shutil.copy2(file, snapshot/file.name); hashes[file.name] = sha(file)
        write(snapshot/"manifest.json", hashes)
        receipt.update(stage="analyzing", analysis_source_hashes=hashes)
        write(target, receipt)
        env = os.environ.copy(); env["PYTHONPATH"] = str(ROOT)+os.pathsep+str(ROOT/"rnj_wzzt_core")
        with (output/"postprocess.log").open("a", encoding="utf-8") as log:
            # -m is mandatory: the independent analyzer uses relative package imports.
            completed = subprocess.run([sys.executable,"-B","-m","research_experiments.miqp_loss_benchmark_20261111.analyze",
                                        str(output),"--workers",str(workers)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            receipt["analysis_exit_code"] = completed.returncode
            if completed.returncode:
                raise RuntimeError("Independent analysis failed; see postprocess.log")
            verification = read(output/"analysis/verification.json")
            stable = all(sha(file) == hashes[file.name] for file in files)
            passed = (verification.get("passed") is True and verification.get("complete") is True
                      and verification.get("new_native_verified") == 96
                      and verification.get("reused_parent_verified") == 96 and stable)
            receipt.update(verification=verification, analysis_source_unchanged=stable, passed=passed)
            if passed:
                receipt["stage"] = "plotting"; write(target, receipt)
                completed = subprocess.run([sys.executable,"-B","-m","research_experiments.miqp_loss_benchmark_20261111.figures",
                                            str(output)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                receipt["figures_exit_code"] = completed.returncode
                if completed.returncode:
                    raise RuntimeError("Scientific figure generation failed")
            else:
                receipt["figures_skipped"] = "Not all 96 new and 96 paired parent results passed independent verification"
        receipt.update(stage="complete" if receipt["passed"] else "incomplete_or_failed",
                       elapsed_seconds=perf_counter()-started, completed_utc=datetime.now(timezone.utc).isoformat())
        write(target, receipt)
        print(json.dumps({k:receipt[k] for k in ("stage","passed","returned","failed","pending","visual_review_pending")}))
        return receipt
    except Exception:
        receipt.update(stage="postprocess_error", passed=False, elapsed_seconds=perf_counter()-started,
                       traceback=traceback.format_exc())
        write(target, receipt); raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path); parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--interval", type=float, default=20.)
    args = parser.parse_args(); finish(args.output, workers=args.workers, interval=args.interval)
