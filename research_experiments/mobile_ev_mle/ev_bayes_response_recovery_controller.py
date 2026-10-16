"""Resume the same frozen histories after a diagnostic-thread native crash.

This is orchestration only. Scientific inputs and final records are immutable.
Live workers are adopted, every new process handles one history, and only a
native crash without a final record permits a second attempt. The runtime entry
disables the periodic traceback printer; no scientific source is patched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid


HERE = Path(__file__).resolve().parent
SCHEMA = "ev_bayes_response_runtime_recovery_v1"
NUMERIC_THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".%d.tmp" % os.getpid())
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def native_exit(code):
    """Windows NTSTATUS failures, including signed subprocess representations."""
    return code is not None and ((int(code) & 0xFFFFFFFF) >= 0x80000000)


class WindowsProcess:
    """Keep an open query handle, avoiding PID reuse during adoption."""
    def __init__(self, pid):
        import ctypes
        from ctypes import wintypes as w
        self.ctypes, self.w = ctypes, w
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = (w.DWORD, w.BOOL, w.DWORD)
        self.kernel.OpenProcess.restype = w.HANDLE
        self.kernel.GetExitCodeProcess.argtypes = (w.HANDLE, ctypes.POINTER(w.DWORD))
        self.kernel.GetExitCodeProcess.restype = w.BOOL
        self.kernel.CloseHandle.argtypes = (w.HANDLE,)
        self.kernel.GetProcessTimes.argtypes = (w.HANDLE, ctypes.POINTER(w.FILETIME),
            ctypes.POINTER(w.FILETIME), ctypes.POINTER(w.FILETIME), ctypes.POINTER(w.FILETIME))
        self.kernel.GetProcessTimes.restype = w.BOOL
        self.kernel.QueryFullProcessImageNameW.argtypes = (w.HANDLE, w.DWORD, w.LPWSTR, ctypes.POINTER(w.DWORD))
        self.kernel.QueryFullProcessImageNameW.restype = w.BOOL
        self.pid = int(pid)
        self.handle = self.kernel.OpenProcess(0x1000, False, self.pid)
        if not self.handle:
            raise ProcessLookupError(self.pid)
        create, end, kernel_time, user = [w.FILETIME() for _ in range(4)]
        if not self.kernel.GetProcessTimes(self.handle, *(ctypes.byref(x) for x in (create, end, kernel_time, user))):
            self.close()
            raise OSError("cannot verify adopted process creation time")
        self.created_filetime = (create.dwHighDateTime << 32) | create.dwLowDateTime
        buffer = ctypes.create_unicode_buffer(32768)
        length = w.DWORD(len(buffer))
        if not self.kernel.QueryFullProcessImageNameW(self.handle, 0, buffer, ctypes.byref(length)):
            self.close()
            raise OSError("cannot verify adopted process image")
        self.image = buffer.value

    def poll(self):
        code = self.w.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle, self.ctypes.byref(code)):
            raise OSError("cannot query adopted process exit")
        return None if code.value == 259 else int(code.value)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def process_alive(pid):
    if os.name == "nt":
        try:
            process = WindowsProcess(pid)
        except ProcessLookupError:
            return False
        try:
            return process.poll() is None
        finally:
            process.close()
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def verify_sources(frozen, here=HERE):
    """Read byte hashes directly; do not execute or update frozen science."""
    changed = []
    for name, expected in frozen["source_sha256"].items():
        path = (here.parents[1] / name) if name.startswith("rnj_wzzt_core/") else here / name
        actual = sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual != expected:
            changed.append(dict(name=name, expected=expected, actual=actual))
    if changed:
        raise RuntimeError("frozen scientific source mismatch: " + json.dumps(changed))
    return dict(files=len(frozen["source_sha256"]), policy_id=frozen["policy_id"], checked=utc())


def record_path(root, phase, index):
    return Path(root) / phase / ("history_%03d.json.gz" % index)


class FinalRecords:
    def __init__(self, root, frozen):
        self.root, self.frozen, self.cache = Path(root), frozen, {}

    def get(self, phase, index):
        path = record_path(self.root, phase, index)
        if not path.exists():
            return None
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.cache:
            row = read(path)
            if row.get("policy_id") != self.frozen["policy_id"]:
                raise RuntimeError("final record policy mismatch: " + str(path))
            if row.get("spec", {}).get("index") != index or row.get("spec", {}).get("phase") != phase:
                raise RuntimeError("final record identity mismatch: " + str(path))
            self.cache[key] = dict(status=row["status"], bytes=stat.st_size,
                scientific_seconds=row.get("seconds"), path=str(path))
        return self.cache[key]

    def summary(self, phase):
        specs = self.frozen["specifications"][phase]
        counts = {}
        saved = 0
        expected_names = {record_path(self.root, phase, s["index"]).name for s in specs}
        actual_names = {p.name for p in (self.root / phase).glob("history_*.json.gz")}
        if actual_names - expected_names:
            raise RuntimeError("unexpected final records: " + str(sorted(actual_names - expected_names)))
        for spec in specs:
            row = self.get(phase, spec["index"])
            if row:
                saved += 1
                counts[row["status"]] = counts.get(row["status"], 0) + 1
        return dict(expected=len(specs), saved=saved, status_counts=counts, unreadable=[])


class EventLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()] if self.path.exists() else []

    def append(self, **row):
        row = dict(time=utc(), **row)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.rows.append(row)
        return row

    def attempts(self, phase, index):
        return [r for r in self.rows if r.get("event") in ("attempt_started", "original_native_failure")
                and r.get("phase") == phase and r.get("index") == index]

    def finished_attempts(self, phase, index):
        return [r for r in self.rows if r.get("event") in ("attempt_finished", "original_native_failure")
                and r.get("phase") == phase and r.get("index") == index]


def eligible_indices(frozen, phase, records, reserved, events, max_attempts=2):
    candidates = []
    for spec in frozen["specifications"][phase]:
        index = spec["index"]
        if index in reserved or records.get(phase, index):
            continue
        attempts = events.attempts(phase, index)
        finished = events.finished_attempts(phase, index)
        if len(attempts) >= max_attempts:
            raise RuntimeError("native retry limit reached for %s:%d; no final record" % (phase, index))
        if attempts:
            if len(finished) != len(attempts):
                raise RuntimeError("unaccounted live attempt for %s:%d" % (phase, index))
            if not native_exit(finished[-1]["exit_code"]):
                raise RuntimeError("non-native missing-record failure for %s:%d" % (phase, index))
        candidates.append(index)
    return candidates


def import_original_failure(previous, records, events):
    """Only the first unsaved history was attempted by a multi-history worker."""
    phase = previous.get("phase")
    for worker in previous.get("workers", []):
        if not native_exit(worker.get("exit_code")):
            continue
        if worker.get("attempt_id"):
            duplicate = any(r.get("event") == "attempt_finished" and r.get("attempt_id") == worker["attempt_id"]
                for r in events.rows)
            if not duplicate:
                events.append(event="attempt_finished", phase=phase, index=worker["indices"][0],
                    indices=worker["indices"], pid=worker["pid"], attempt_id=worker["attempt_id"],
                    exit_code=worker["exit_code"], elapsed_seconds_unknown=True, recovered_from_controller_state=True)
            continue
        missing = [i for i in worker["indices"] if records.get(phase, i) is None]
        if not missing:
            continue
        index = missing[0]
        duplicate = any(r.get("event") == "original_native_failure" and r.get("pid") == worker["pid"]
                        for r in events.rows)
        if not duplicate:
            events.append(event="original_native_failure", phase=phase, index=index,
                pid=worker["pid"], exit_code=worker["exit_code"], original_controller_pid=previous.get("controller_pid"),
                diagnostic="native_failure_32188.json" if worker["pid"] == 32188 else None,
                original_elapsed_seconds_unknown=True, scientific_result_saved=False)


def worker_environment():
    env = os.environ.copy()
    for name in NUMERIC_THREADS:
        env[name] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def worker_command(root, phase, index, worker_id):
    return [sys.executable, "-m", "research_experiments.mobile_ev_mle.ev_bayes_response_runtime_entry",
        "--output", str(root), "--phase", phase, "--worker-id", worker_id, "--indices", str(index)]


def acquire_lock(path, previous, alive=process_alive):
    current = os.getpid()
    previous_pid = previous.get("controller_pid")
    if previous_pid and previous_pid != current and alive(previous_pid):
        raise RuntimeError("previous controller still active: %d" % previous_pid)
    path = Path(path)
    if path.exists():
        old = read(path)
        if alive(old["pid"]):
            raise RuntimeError("recovery controller already active: " + str(old["pid"]))
        path.rename(path.with_name("stale_" + path.stem + "_%d.json" % time.time_ns()))
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(pid=current, started=utc()), handle)


def acquire_runtime_locks(root, previous, alive=process_alive):
    """Claim both locks, so the legacy controller also sees the active owner."""
    paths = [Path(root) / "controller.lock", Path(root) / "recovery_controller.lock"]
    for path in paths:
        if path.exists() and alive(read(path)["pid"]):
            raise RuntimeError("controller lock still active: " + str(path))
    claimed = []
    try:
        for path in paths:
            acquire_lock(path, previous, alive=alive)
            claimed.append(path)
    except BaseException:
        for path in claimed:
            path.unlink(missing_ok=True)
        raise
    return paths


def preserve_partial(root, phase, index, runtime, attempt_id):
    partial = record_path(root, phase, index).with_name(record_path(root, phase, index).name + ".tmp")
    if not partial.exists():
        return None
    destination = Path(runtime) / ("partial_%s_%03d_%s.tmp" % (phase, index, attempt_id))
    shutil.copyfile(partial, destination)
    return dict(original=str(partial), preserved=str(destination), bytes=destination.stat().st_size,
        sha256=sha256(destination.read_bytes()).hexdigest())


def adopt_workers(previous, phase, events, *, process_factory=WindowsProcess):
    active = []
    for worker in previous.get("workers", []):
        if worker.get("exit_code") is not None:
            continue
        try:
            process = process_factory(worker["pid"])
        except ProcessLookupError:
            raise RuntimeError("worker disappeared before safe adoption; inspect final/exit records: " + str(worker["pid"]))
        if Path(process.image).resolve() != Path(sys.executable).resolve():
            process.close()
            raise RuntimeError("adopted PID is not this Python executable")
        expected_time = worker.get("created_filetime")
        if expected_time and expected_time != process.created_filetime:
            process.close()
            raise RuntimeError("adopted PID creation time changed")
        if process.poll() is not None:
            # Preserve queryable exit status and process the result in the loop.
            pass
        event = events.append(event="worker_adopted", phase=phase, pid=process.pid,
            indices=list(worker["indices"]), created_filetime=process.created_filetime,
            image=process.image, original_attempt_id=worker.get("attempt_id"), prior_runtime_seconds_unknown=True)
        active.append(dict(process=process, phase=phase, indices=list(worker["indices"]),
            adopted=True, started_monotonic=time.monotonic(), started=event["time"],
            log=None, attempt_id=worker.get("attempt_id"), created_filetime=process.created_filetime))
    return active


def worker_rows(active):
    return [dict(pid=w["process"].pid, exit_code=w["process"].poll(), phase=w["phase"],
        indices=w["indices"], adopted=w["adopted"], attempt_id=w.get("attempt_id"),
        created_filetime=w.get("created_filetime")) for w in active]


def finish_worker(worker, records, events):
    code = worker["process"].poll()
    if code is None:
        raise RuntimeError("cannot finish a live worker")
    elapsed = time.monotonic() - worker["started_monotonic"]
    finals = {str(i): records.get(worker["phase"], i) for i in worker["indices"]}
    tracked_attempt = bool(worker.get("attempt_id"))
    events.append(event="attempt_finished" if tracked_attempt else "adopted_worker_finished",
        phase=worker["phase"], index=worker["indices"][0] if tracked_attempt else None,
        indices=worker["indices"], pid=worker["process"].pid, attempt_id=worker.get("attempt_id"),
        exit_code=code, exit_code_hex="0x%08X" % (code & 0xFFFFFFFF), elapsed_seconds=elapsed,
        elapsed_scope="since adoption" if worker["adopted"] else "full worker wall time",
        final_records=finals)
    if worker["log"]:
        worker["log"].close()
    missing = [i for i in worker["indices"] if not finals[str(i)]]
    if missing and worker["adopted"] and not tracked_attempt and native_exit(code):
        # Multi-history legacy worker failed at its first missing index only.
        index = missing[0]
        events.append(event="original_native_failure", phase=worker["phase"], index=index,
            pid=worker["process"].pid, exit_code=code, original_elapsed_seconds_unknown=True,
            scientific_result_saved=False, observed_after_adoption=True)
    if missing and not native_exit(code):
        raise RuntimeError("worker exited without final record and without native failure: " + str(missing))
    if hasattr(worker["process"], "close"):
        worker["process"].close()


def audit_phase(root, phase, frozen):
    verify_sources(frozen)
    audit_path = Path(root) / ("audit_" + phase + ".json")
    if audit_path.exists():
        checked = read(audit_path)
    else:
        from . import ev_bayes_response_audit as audit
        checked = audit.run(Path(root), audit_path, phases=(phase,), ghk_histories=())
    if (not checked.get("success") or checked.get("policy_id") != frozen["policy_id"]
            or not checked.get("coverage", {}).get(phase, {}).get("exact_coverage")):
        raise RuntimeError("independent audit gate failed for " + phase)
    verify_sources(frozen)
    return checked


def development_gate(root, frozen):
    """The previous development gate is a prerequisite, never rerun here."""
    checked = read(Path(root) / "audit_development.json")
    expected = {record_path(root, "development", s["index"]).name
        for s in frozen["specifications"]["development"]}
    actual = {p.name for p in (Path(root) / "development").glob("history_*.json.gz")}
    if (not checked.get("success") or checked.get("policy_id") != frozen["policy_id"]
            or not checked.get("coverage", {}).get("development", {}).get("exact_coverage")
            or actual != expected):
        raise RuntimeError("existing development audit/coverage gate failed")
    return dict(success=True, exact_coverage=True, histories=len(expected), policy_id=frozen["policy_id"])


def run(root, *, workers=2, max_attempts=2, poll_seconds=15.):
    if workers not in (1, 2) or max_attempts != 2:
        raise ValueError("keep one/two workers and two total native attempts")
    root = Path(root).resolve()
    frozen, previous = read(root / "freeze.json"), read(root / "controller.json")
    if previous["policy_id"] != frozen["policy_id"]:
        raise RuntimeError("controller/freeze policy mismatch")
    verified = verify_sources(frozen)
    dev_gate = development_gate(root, frozen)
    runtime = root / "runtime_recovery_v1"
    runtime.mkdir(exist_ok=True)
    locks = acquire_runtime_locks(root, previous)
    records = FinalRecords(root, frozen)
    events = EventLog(runtime / "attempts.jsonl")
    active = []
    state = dict(previous, status="running", controller_pid=os.getpid(),
        updated=utc(), recovery=dict(schema=SCHEMA, source_verification=verified,
        attempts_log=str(events.path), diagnostic_timer_disabled=True,
        max_total_attempts_per_history=max_attempts, worker_limit=workers, development_gate=dev_gate))
    state.pop("failure", None)
    state.setdefault("phases", {})
    try:
        import_original_failure(previous, records, events)
        if previous.get("workers"):
            active = adopt_workers(previous, previous["phase"], events)
        if len(active) > workers:
            raise RuntimeError("adopted workers exceed allowed concurrency")
        events.append(event="recovery_started", pid=os.getpid(), original_started=previous.get("started"),
            adopted_pids=[w["process"].pid for w in active], source_verification=verified)
        for phase in ("calibration", "evaluation"):
            if any(w["phase"] != phase for w in active):
                completed = records.summary(phase)
                if completed["saved"] != completed["expected"]:
                    raise RuntimeError("active worker belongs to a different phase")
                audit_phase(root, phase, frozen)
                state["phases"][phase] = dict(completed, independent_audit_success=True)
                continue
            state["phase"] = phase
            state["stage"] = "recovery_execution"
            state["recovery"]["source_verification"] = verify_sources(frozen)
            while True:
                for worker in list(active):
                    if worker["process"].poll() is not None:
                        finish_worker(worker, records, events)
                        active.remove(worker)
                reserved = {i for w in active for i in w["indices"]}
                eligible = eligible_indices(frozen, phase, records, reserved, events, max_attempts)
                while eligible and len(active) < workers:
                    index = eligible.pop(0)
                    verify_sources(frozen)
                    if record_path(root, phase, index).exists():
                        raise RuntimeError("final record appeared before launch; rescan rather than overwrite")
                    attempt_number = len(events.attempts(phase, index)) + 1
                    attempt_id = uuid.uuid4().hex
                    partial = preserve_partial(root, phase, index, runtime, attempt_id)
                    worker_id = "recovery_%s_%03d_a%d" % (phase, index, attempt_number)
                    log_path = runtime / (worker_id + "_" + attempt_id + ".log")
                    command = worker_command(root, phase, index, worker_id)
                    log = log_path.open("x", encoding="utf-8")
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                        env=worker_environment(), cwd=HERE.parents[1],
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    created_filetime = None
                    if os.name == "nt":
                        queried = WindowsProcess(process.pid)
                        created_filetime = queried.created_filetime
                        queried.close()
                    event = events.append(event="attempt_started", phase=phase, index=index,
                        pid=process.pid, attempt=attempt_number, attempt_id=attempt_id,
                        command=command, log=str(log_path), created_filetime=created_filetime,
                        preserved_partial=partial)
                    active.append(dict(process=process, phase=phase, indices=[index], adopted=False,
                        log=log, attempt_id=attempt_id, created_filetime=created_filetime,
                        started=event["time"], started_monotonic=time.monotonic()))
                state["workers"] = worker_rows(active)
                state["phases"][phase] = records.summary(phase)
                state["updated"] = utc()
                state["recovery"]["events"] = len(events.rows)
                atomic_json(root / "controller.json", state)
                atomic_json(runtime / "checkpoint.json", state)
                if not active:
                    if state["phases"][phase]["saved"] != state["phases"][phase]["expected"]:
                        raise RuntimeError("missing frozen histories after scheduling")
                    break
                time.sleep(poll_seconds)
            state["stage"] = "independent_audit"
            atomic_json(root / "controller.json", state)
            audit_phase(root, phase, frozen)
            state["phases"][phase]["independent_audit_success"] = True
            state["stage"] = "phase_audited"
            events.append(event="phase_audited", phase=phase, summary=state["phases"][phase])
            atomic_json(root / "controller.json", state)
        state.update(status="statistical_complete", updated=utc(), workers=[])
        state["recovery"]["source_verification"] = verify_sources(frozen)
        atomic_json(root / "controller.json", state)
        atomic_json(runtime / "checkpoint.json", state)
        events.append(event="statistical_complete", pid=os.getpid())
        return state
    except BaseException as exc:
        state.update(status="controller_failure", failure=repr(exc), updated=utc(), workers=worker_rows(active))
        atomic_json(root / "controller.json", state)
        atomic_json(runtime / "checkpoint.json", state)
        events.append(event="recovery_failed", pid=os.getpid(), failure=repr(exc), workers=state["workers"])
        raise
    finally:
        if not any(w["process"].poll() is None for w in active):
            for lock in locks:
                lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=15.)
    args = parser.parse_args()
    run(args.output, workers=args.workers, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
