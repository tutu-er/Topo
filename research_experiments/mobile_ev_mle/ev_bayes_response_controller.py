"""Logged, single-instance frozen batch controller; scientific failures survive."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import ev_bayes_response_study as study
from . import ev_bayes_response_cases as cases
from . import gradient_price_study as io


def utc(): return datetime.now(timezone.utc).isoformat()


def process_alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        k.OpenProcess.restype = wintypes.HANDLE
        k.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        k.CloseHandle.argtypes = (wintypes.HANDLE,)
        h = k.OpenProcess(0x1000, False, int(pid))
        if not h: return False
        code = wintypes.DWORD()
        try: return bool(k.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259
        finally: k.CloseHandle(h)
    try: os.kill(int(pid), 0); return True
    except ProcessLookupError: return False


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".%d.tmp" % os.getpid())
    io.write_json(temporary, value)
    temporary.replace(path)


_SUMMARY_CACHE = {}


def summarize(root, phase):
    rows, broken = [], []
    for path in sorted((root/phase).glob("history_*.json.gz")):
        try:
            stat = path.stat(); key = (str(path), stat.st_size, stat.st_mtime_ns)
            if key not in _SUMMARY_CACHE:
                _SUMMARY_CACHE[key] = io.read_json(path)["status"]
            rows.append(dict(status=_SUMMARY_CACHE[key]))
        except Exception as exc: broken.append(dict(path=str(path), failure=repr(exc)))
    counts = {}
    for row in rows: counts[row["status"]] = counts.get(row["status"], 0)+1
    return dict(expected=len(cases.specifications(phase)), saved=len(rows),
                status_counts=counts, unreadable=broken)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--phases", nargs="+", default=["calibration", "evaluation"])
    args = parser.parse_args()
    if args.workers not in (1, 2): raise ValueError("use one or two single-thread workers")
    if not set(args.phases) <= {"development", "calibration", "evaluation"}: raise ValueError("invalid phase")
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=True)
    lock = root/"controller.lock"
    if lock.exists():
        previous = io.read_json(lock)
        if process_alive(previous["pid"]): raise RuntimeError("controller already active: "+str(previous))
        state_path = root/"controller.json"
        if state_path.exists():
            workers = io.read_json(state_path).get("workers", [])
            if any(w.get("exit_code") is None and process_alive(w["pid"]) for w in workers):
                raise RuntimeError("previous controller stopped but a worker is still alive")
        lock.rename(root/("stale_lock_%d.json" % time.time_ns()))
    frozen = study.freeze(root)
    with lock.open("x", encoding="utf-8") as f: json.dump(dict(pid=os.getpid(), started=utc()), f)
    state = dict(status="running", controller_pid=os.getpid(), started=utc(),
                 policy_id=frozen["policy_id"], source_sha256=frozen["source_sha256"], phases={})
    path = root/"controller.json"
    active = []
    try:
        for phase in args.phases:
            state["phase"] = phase
            env = os.environ.copy()
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[name] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            active = []
            for worker in range(args.workers):
                indices = [s["index"] for s in cases.specifications(phase)
                           if s["index"] % args.workers == worker and not
                           (root/phase/("history_%03d.json.gz" % s["index"])).exists()]
                if not indices: continue
                log = (root/("worker_%s_%d.log" % (phase, worker))).open("a", encoding="utf-8")
                command = [sys.executable, "-m", "research_experiments.mobile_ev_mle.ev_bayes_response_study",
                    "--output", str(root), "--phase", phase, "--worker-id", str(worker),
                    "--indices", *map(str, indices)]
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                active.append((process, log, indices))
            while active and any(p.poll() is None for p, _, _ in active):
                state["workers"] = [dict(pid=p.pid, exit_code=p.poll(), indices=indices)
                                    for p, _, indices in active]
                state["phases"][phase] = summarize(root, phase)
                state["updated"] = utc(); atomic_json(path, state)
                time.sleep(15)
            exits = [p.wait() for p, _, _ in active]
            for _, log, _ in active: log.close()
            state["phases"][phase] = summarize(root, phase)
            if any(exits) or state["phases"][phase]["saved"] != state["phases"][phase]["expected"]:
                raise RuntimeError("worker process or missing-record failure; saved files retained")
            state["stage"] = "independent_audit"; atomic_json(path, state)
            from . import ev_bayes_response_audit as audit
            audit_path = root/("audit_"+phase+".json")
            checked = io.read_json(audit_path) if audit_path.exists() else audit.run(
                root, audit_path, phases=(phase,), ghk_histories=((phase+":0",) if phase == "development" else ()))
            state["phases"][phase]["independent_audit_success"] = checked["success"]
            if not checked["success"]:
                raise RuntimeError("independent audit gate failed; review evidence before continuing")
            state["stage"] = "phase_audited"; atomic_json(path, state)
        state.update(status="statistical_complete", updated=utc(), workers=[])
        atomic_json(path, state)
    except BaseException as exc:
        # Keep live-worker identities if interrupted; the next run must inspect them.
        state.update(status="controller_failure", failure=repr(exc), updated=utc(),
            workers=[dict(pid=p.pid, exit_code=p.poll(), indices=indices) for p, _, indices in active])
        atomic_json(path, state)
        raise
    finally:
        if not any(p.poll() is None for p, _, _ in active): lock.unlink(missing_ok=True)


if __name__ == "__main__": main()
