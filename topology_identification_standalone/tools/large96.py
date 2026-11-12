"""Frozen single-day experiments: generate, run, seal, then expose truth.

Main: 4 networks x 20 new days x 3 noise levels x 10 methods.
Pressure: 4 networks x 4 factors x 2 levels x 5 days x 4 methods.
Workers receive one 96-point NPZ, train64/selection32, and no physical case.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter, sleep
import traceback

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from tools.strict96 import split96, fingerprint

SOURCES = ("algorithm.py", "optimization.py", "trees.py", "data.py", "competition.py",
           "tools/strict96.py", "tools/large96.py")
CASES = ("paper15", "soumalas11", "flynn16", "pengwah18")
METHODS = ("rnj", "nj", "000", "100", "010", "001", "110", "101", "011", "111")
PRESSURE = {"load_scale": (.5, 1.5), "rx_ratio": (.5, 2.),
            "independent_scale": (.1, .3), "weak_edge": (.1, .3)}
TERMINAL = {"fitted", "failed", "interrupted", "budget_exceeded"}
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def utc():
    return datetime.now(timezone.utc).isoformat()


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
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    text = json.dumps(value, indent=2, allow_nan=False,
        default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))+"\n"
    for attempt in range(100):
        try:
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 99:
                raise
            sleep(.05)


def specifications(cohort):
    if cohort == "main":
        # First 24 span all networks/noise levels and two independent days.
        return [dict(case=c, replicate=r, voltage_sigma=s, profile="baseline", level=1., cohort=cohort)
                for r in range(20) for c in CASES for s in (.0002, .0005, .001)]
    if cohort == "pressure":
        return [dict(case=c, replicate=100+r, voltage_sigma=.0005, profile=p, level=v, cohort=cohort)
                for r in range(5) for c in CASES for p, levels in PRESSURE.items() for v in levels]
    raise ValueError("Unknown cohort")


def source_hashes(runtime):
    return {name: sha(Path(runtime)/name) for name in SOURCES}


def check_runtime(output, protocol, *, full_inputs=True):
    runtime = output/"runtime"
    if source_hashes(runtime) != protocol["source_sha256"]:
        raise ValueError("Frozen source changed")
    if sha(runtime/"protocol.json") != sha(output/"protocol.json"):
        raise ValueError("Runtime protocol changed")
    expected = set(SOURCES) | {"protocol.json"}
    for info in protocol["inputs"].values():
        if info.get("generation_failed"):
            continue
        expected.add(info["path"])
        if full_inputs and sha(runtime/info["path"]) != info["sha256"]:
            raise ValueError("Frozen observation changed")
    actual = {p.relative_to(runtime).as_posix() for p in runtime.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Runtime has unexpected files")


def freeze(output, cohort="main", workers=8):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Existing experiments cannot be overwritten")
    if not 1 <= workers <= 8:
        raise ValueError("Use one to eight single-thread workers")
    import algorithm
    import data
    config = algorithm.SearchConfig(); config.validate()
    spec = specifications(cohort)
    snapshot = source_hashes(BASE)
    runtime = output/"runtime"; runtime.mkdir(parents=True)
    case_snapshot = {p.relative_to(BASE/"cases").as_posix(): sha(p)
                     for p in (BASE/"cases").rglob("*") if p.is_file()}
    shutil.copytree(BASE/"cases", output/"evaluation_cases")
    for name in SOURCES:
        target = runtime/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BASE/name, target)
    inputs, evaluation, diagnostics = {}, {}, {}
    generation_started = perf_counter()
    for i, row in enumerate(spec):
        job = f"input{i:03d}"; evaluation[job] = row
        began = perf_counter()
        target = runtime/"inputs"/(job+".npz")
        try:
            observed, metadata = data.generate_day96(**{k: v for k, v in row.items() if k != "cohort"})
            target.parent.mkdir(exist_ok=True)
            np.savez_compressed(target, P=observed.p, Q=observed.q, V=observed.voltage,
                                root=observed.root, labels=observed.labels)
            train, selection = split96(target)
            inputs[job] = dict(path=f"inputs/{job}.npz", sha256=sha(target), labels=list(observed.labels),
                training_sha256=fingerprint(train), selection_sha256=fingerprint(selection))
            diagnostics[job] = dict(status="generated", seconds=perf_counter()-began, metadata=metadata)
        except Exception:
            if target.exists():
                failed_file = output/"generation_failures"/target.name
                failed_file.parent.mkdir(exist_ok=True); target.replace(failed_file)
            inputs[job] = dict(path=None, sha256=None, labels=None, training_sha256=None,
                               selection_sha256=None, generation_failed=True)
            diagnostics[job] = dict(status="failed", seconds=perf_counter()-began, error=traceback.format_exc())
    methods = METHODS if cohort == "main" else ("000", "100", "101", "111")
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "gurobipy")}
    write(output/"evaluation_map.json", evaluation); write(output/"generation.json", diagnostics)
    protocol = dict(kind="large96_single_day_v1", cohort=cohort, created_utc=utc(), inputs=inputs,
        methods=list(methods), config=asdict(config), source_sha256=snapshot, workers=workers,
        split=dict(total=96, train=[0, 64], selection=[64, 96], shuffle=False),
        project_root=str(BASE.parent), study_seed=20261112, paired_noise=True,
        planned_inputs=len(inputs), planned_methods=len(inputs)*len(methods),
        nominal_method_hours=len(inputs)*len(methods)*config.total_seconds/3600.,
        ambiguity_kappas=[0., .003, .01, .03, .1],
        evaluation_map_sha256=sha(output/"evaluation_map.json"), generation_sha256=sha(output/"generation.json"),
        evaluation_case_sha256=case_snapshot,
        generation_seconds=perf_counter()-generation_started,
        environment=dict(python=sys.version, executable=sys.executable, packages=versions),
        information_policy="one 96-point file per input, train64 fits, selection32 frozen-model ranking; no extra observations",
        common_charge="one physical common64 computation per input, charged once to each method",
        interruption_policy="terminal failures not retried; unresolved attempts require audit, no automatic budget reset",
        budget_semantics="14400s accounting cap; Gurobi native TimeLimit; BVLS 300s soft check, not OS hard timeout",
        algorithm_seed_scope="new trajectories on four known physical networks; no claim of unseen-topology generalization")
    if source_hashes(BASE) != snapshot or source_hashes(runtime) != snapshot:
        raise ValueError("Source changed during generation")
    if any(sha(BASE/"cases"/name) != h or sha(output/"evaluation_cases"/name) != h for name, h in case_snapshot.items()):
        raise ValueError("Physical case changed during generation")
    write(output/"protocol.json", protocol); write(runtime/"protocol.json", protocol)
    check_runtime(output, protocol)
    write(output/"state.json", dict(stage="frozen", **status(output)))
    return dict(output=str(output), planned=protocol["planned_methods"], inputs=len(inputs),
                generation_failed=sum(v.get("generation_failed", False) for v in inputs.values()),
                protocol_sha256=sha(output/"protocol.json"))


def result_path(output, job, method=None):
    return output/"common"/job/"common.json" if method is None else output/"runs"/job/method/"result.json"


def status(output):
    output = Path(output).resolve(); protocol = read(output/"protocol.json")
    answer = {}
    for kind in ("common", "methods"):
        tasks = [(job, None) for job in protocol["inputs"]] if kind == "common" else [
            (job, method) for job in protocol["inputs"] for method in protocol["methods"]]
        counts = dict(planned=len(tasks), pending=0, unresolved_attempts=0, **{s: 0 for s in TERMINAL})
        for job, method in tasks:
            target = result_path(output, job, method)
            index = target.with_name("terminal.json")
            if target.exists():
                value = read(index) if index.exists() else read(target)
                counts[value["status"]] += 1
            else:
                counts["pending"] += 1
                counts["unresolved_attempts"] += target.with_name("attempt.json").exists()
        answer[kind] = counts
    return answer


def terminal(target, value):
    write(target, value)
    write(target.with_name("terminal.json"), dict(status=value["status"], result_sha256=sha(target)))


def read_guard(runtime, project, allowed):
    runtime, project = Path(runtime).resolve(), Path(project).resolve()
    allowed = {Path(p).resolve() for p in allowed}
    dependencies = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    observed = []
    def audit(event, args):
        if event != "open" or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve(); mode, flags = args[1:3]
        reading = ("r" in mode or "+" in mode) if isinstance(mode, str) else (flags & (os.O_WRONLY | os.O_RDWR)) != os.O_WRONLY
        if not reading:
            return
        if path.suffix.lower() == ".npz":
            if path not in allowed:
                raise PermissionError("Another observation file was blocked")
            observed.append(str(path))
        dependency = any(path.is_relative_to(root) for root in dependencies)
        if path.is_relative_to(project) and not dependency and path not in allowed:
            raise PermissionError("A non-whitelisted project read was blocked")
    sys.addaudithook(audit)
    return observed


def worker(output, job, method):
    started = perf_counter(); output = Path(output).resolve(); runtime = output/"runtime"
    protocol = read(runtime/"protocol.json"); info = protocol["inputs"][job]
    method = None if method == "common" else method
    if job not in protocol["inputs"] or method is not None and method not in protocol["methods"]:
        raise ValueError("Unknown task")
    if Path(__file__).resolve() != runtime/"tools/large96.py":
        raise ValueError("Worker must execute its frozen runner")
    target = result_path(output, job, method); common_file = result_path(output, job)
    observed_file = runtime/info["path"]
    allowed = [runtime/name for name in SOURCES]+[runtime/"protocol.json", output/"protocol.json", observed_file, target]
    if method is not None:
        allowed.append(common_file)
    reads = read_guard(runtime, protocol["project_root"], allowed)
    record = dict(job=job, method=method, kind="method" if method else "common", started_utc=utc(), pid=os.getpid(),
                  protocol_sha256=sha(runtime/"protocol.json"), input_sha256=info["sha256"],
                  train_points=64, selection_points=32, truth_used=False, extra_observation_files_allowed=False)
    common_seconds = 0.
    try:
        shared = read(common_file) if method is not None else None
        common_seconds = float(shared["total_seconds"]) if shared is not None else 0.
        before = source_hashes(runtime)
        if before != protocol["source_sha256"] or sha(observed_file) != info["sha256"]:
            raise ValueError("Source or sole input changed")
        train_rows, selection_rows = split96(observed_file)
        if (fingerprint(train_rows), fingerprint(selection_rows)) != (info["training_sha256"], info["selection_sha256"]):
            raise ValueError("Train64/selection32 fingerprint changed")
        algorithm, data = importlib.import_module("algorithm"), importlib.import_module("data")
        loaded = [algorithm, data]+[importlib.import_module(n) for n in ("optimization", "trees", "competition")]
        if not all(Path(m.__file__).resolve().is_relative_to(runtime) for m in loaded):
            raise ValueError("A native module escaped runtime")
        config = algorithm.SearchConfig(**protocol["config"]); config.validate()
        train, selection = data.Observation(**train_rows), data.Observation(**selection_rows)
        if method is None:
            common = algorithm.prepare_common(train, config)
            native_seconds = common["seconds"]; elapsed = perf_counter()-started
            common["seconds"] = elapsed
            record.update(common=common, native_seconds=native_seconds, common_seconds=elapsed,
                          fresh_seconds=elapsed, total_seconds=elapsed,
                          status="fitted" if elapsed <= config.total_seconds+.05 else "budget_exceeded")
        else:
            common = shared["common"]
            if shared["status"] != "fitted" or shared["input_sha256"] != info["sha256"]:
                raise ValueError("Shared train64 preparation is unavailable")
            pre_fit = perf_counter()-started
            effective = replace(config, total_seconds=config.total_seconds-pre_fit)
            effective.validate()
            def heartbeat(value):
                write(target.with_name("checkpoint.json"), dict(job=job, method=method, timestamp_utc=utc(),
                    worker_elapsed_seconds=perf_counter()-started, common_seconds=common_seconds, **value))
            result = algorithm.fit(train, selection, method=method, config=effective, common=common, heartbeat=heartbeat)
            elapsed = perf_counter()-started; total = elapsed+common_seconds
            record.update(result=result, status="budget_exceeded" if result.get("total_cap_exceeded") or total > config.total_seconds+.05 else "fitted",
                          common_seconds=common_seconds, fresh_seconds=elapsed, total_seconds=total,
                          common_sha256=sha(common_file), configured_config=protocol["config"],
                          effective_config=asdict(effective), pre_fit_debit_seconds=pre_fit)
        after = source_hashes(runtime)
        if before != after or sha(observed_file) != info["sha256"]:
            raise ValueError("Frozen source/input changed during fitting")
        record.update(source_sha256_before=before, source_sha256_after=after)
    except Exception as exc:
        elapsed = perf_counter()-started
        record.update(status="budget_exceeded" if isinstance(exc, TimeoutError) else "failed",
                      error=traceback.format_exc(), timeout_reason=str(exc) if isinstance(exc, TimeoutError) else None, fresh_seconds=elapsed,
                      common_seconds=common_seconds, total_seconds=elapsed+common_seconds)
    record.update(observed_npz_opened=sorted(set(reads)), finished_utc=utc())
    terminal(target, record)


def queue(output, protocol, tasks, workers):
    active, pending = {}, list(tasks)
    environment = os.environ.copy(); environment.update({n: "1" for n in THREADS})
    last_state = 0.
    while pending or active:
        while pending and len(active) < workers:
            job, method = pending.pop(0); target = result_path(output, job, method)
            attempt = target.with_name("attempt.json")
            if target.exists() or attempt.exists():
                raise ValueError("A task has already been attempted")
            target.parent.mkdir(parents=True, exist_ok=True)
            write(attempt, dict(state="reserved", utc=utc(), job=job, method=method))
            log = target.with_name("worker.log").open("w", encoding="utf-8")
            began = perf_counter()
            try:
                process = subprocess.Popen([sys.executable, "-I", "-B", str(output/"runtime/tools/large96.py"),
                    "worker", str(output), "--job", job, "--method", method or "common"],
                    cwd=output/"runtime", env=environment, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                active[str(target)] = (process, log, job, method, began)
                write(attempt, dict(state="running", pid=process.pid, started_utc=utc(), job=job, method=method))
            except Exception:
                log.close()
                shared_seconds = read(result_path(output, job))["total_seconds"] if method else 0.
                terminal(target, dict(job=job, method=method, status="failed", method_started=False,
                    error=traceback.format_exc(), fresh_seconds=perf_counter()-began, common_seconds=shared_seconds,
                    total_seconds=perf_counter()-began+shared_seconds, protocol_sha256=sha(output/"protocol.json")))
        for name, (process, log, job, method, began) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close(); del active[name]; target = Path(name)
            elapsed = perf_counter()-began
            if code != 0 and target.exists():
                preserved = target.with_name("returned_before_nonzero_exit.json")
                shutil.copyfile(target, preserved)
                returned = read(target)
                terminal(target, dict(job=job, method=method, status="interrupted", exit_code=code,
                    reason="nonzero_exit_after_result_write", preserved_result=preserved.name,
                    preserved_result_sha256=sha(preserved), native_returned_status=returned.get("status"),
                    fresh_seconds=elapsed, common_seconds=returned.get("common_seconds", 0.),
                    total_seconds=elapsed+(returned.get("common_seconds", 0.) if method else 0.),
                    protocol_sha256=sha(output/"protocol.json")))
            if not target.exists():
                common = read(result_path(output, job))["total_seconds"] if method and result_path(output, job).exists() else 0.
                terminal(target, dict(job=job, method=method, status="interrupted", method_started=None,
                    reason="worker_exited_without_terminal", exit_code=code, fresh_seconds=elapsed,
                    common_seconds=common, total_seconds=elapsed+common, driver_observed_upper_elapsed=True,
                    protocol_sha256=sha(output/"protocol.json")))
            write(target.with_name("attempt.json"), dict(state="finished", pid=process.pid, return_code=code,
                  finished_utc=utc(), driver_observed_seconds=elapsed))
            print(job, method or "common", read(target)["status"], flush=True)
        if perf_counter()-last_state >= 5 or not active:
            write(output/"state.json", dict(stage="running", updated_utc=utc(), active={n: v[0].pid for n, v in active.items()},
                                           queued=len(pending), **status(output)))
            last_state = perf_counter()
        if active:
            sleep(.5)


def seal(output):
    output = Path(output).resolve(); protocol = read(output/"protocol.json")
    counts = status(output)
    if counts["methods"]["pending"] or counts["common"]["pending"]:
        raise ValueError("Cannot seal incomplete experiments")
    if (output/"selection_seal.json").exists():
        raise FileExistsError("Existing seal is immutable")
    check_runtime(output, protocol)
    hashes = {}
    common_hashes = {result_path(output, job).relative_to(output).as_posix(): sha(result_path(output, job))
                     for job in protocol["inputs"]}
    for job, info in protocol["inputs"].items():
        selection = split96(output/"runtime"/info["path"])[1] if not info.get("generation_failed") else None
        for method in protocol["methods"]:
            target = result_path(output, job, method); record = read(target)
            if record["status"] not in TERMINAL or record["protocol_sha256"] != sha(output/"protocol.json"):
                raise ValueError("Nonterminal or foreign result")
            hashes[target.relative_to(output).as_posix()] = sha(target)
            if record["status"] == "fitted":
                from tools.strict96 import mse
                result = record["result"]
                loss = mse(selection, result["selected"]["point"])
                if not np.isclose(loss, result["selected"]["validation_mse"], rtol=1e-8, atol=1e-15):
                    raise ValueError("Selected MSE did not independently replay")
                if record["source_sha256_after"] != protocol["source_sha256"]:
                    raise ValueError("Result source differs from frozen protocol")
                if record["common_sha256"] != common_hashes[result_path(output, job).relative_to(output).as_posix()]:
                    raise ValueError("A method used a different common64 record")
    write(output/"selection_seal.json", dict(sealed=True, all_planned_terminal=True, created_utc=utc(),
          protocol_sha256=sha(output/"protocol.json"), source_sha256=protocol["source_sha256"],
          result_sha256=hashes, common_sha256=common_hashes, counts=counts, selection_metric="MSE32_only_frozen_train64_coefficients",
          truth_not_exposed_to_workers=True))
    return counts


def run(output, workers=None, limit_inputs=None):
    output = Path(output).resolve(); protocol = read(output/"protocol.json"); check_runtime(output, protocol)
    if (output/"completion.json").exists():
        return read(output/"completion.json")
    workers = int(workers or protocol["workers"])
    if not 1 <= workers <= 8:
        raise ValueError("Use one to eight workers")
    counts = status(output)
    if counts["common"]["unresolved_attempts"] or counts["methods"]["unresolved_attempts"]:
        raise RuntimeError("Unresolved attempts require audit; budgets cannot be reset")
    lock = output/"run.lock"; descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(descriptor)
    write(lock, dict(pid=os.getpid(), started_utc=utc(), workers=workers))
    began = perf_counter(); jobs = list(protocol["inputs"])[:limit_inputs]
    invocation = output/"manager_attempts"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+f"_{os.getpid()}.json")
    write(invocation, dict(state="running", started_utc=utc(), pid=os.getpid(), inputs=len(jobs)))
    try:
        for job in jobs:
            if protocol["inputs"][job].get("generation_failed") and not result_path(output, job).exists():
                terminal(result_path(output, job), dict(job=job, status="failed", failure_stage="generation",
                    method_started=False, total_seconds=0., common_seconds=0., fresh_seconds=0.,
                    protocol_sha256=sha(output/"protocol.json")))
        queue(output, protocol, [(j, None) for j in jobs if not result_path(output, j).exists()], workers)
        tasks = []
        for round_number in range(len(protocol["methods"])):
            for i, job in enumerate(jobs):
                method = protocol["methods"][(round_number+i) % len(protocol["methods"])]
                target = result_path(output, job, method)
                if target.exists():
                    continue
                common = read(result_path(output, job))
                if common["status"] != "fitted":
                    terminal(target, dict(job=job, method=method,
                        status="budget_exceeded" if common["status"] == "budget_exceeded" else "failed",
                        failure_stage="generation" if protocol["inputs"][job].get("generation_failed") else "common",
                        common_status=common["status"],
                        method_started=False, common_seconds=common["total_seconds"], fresh_seconds=0.,
                        total_seconds=common["total_seconds"], protocol_sha256=sha(output/"protocol.json")))
                else:
                    tasks.append((job, method))
        queue(output, protocol, tasks, workers)
        final = status(output)
        elapsed = perf_counter()-began
        write(invocation, dict(state="finished", manager_elapsed_seconds=elapsed, finished_utc=utc(), **final))
        total_elapsed = sum(read(p).get("manager_elapsed_seconds", 0.) for p in (output/"manager_attempts").glob("*.json"))
        if not final["methods"]["pending"]:
            seal(output)
            write(output/"completion.json", dict(all_planned_terminal=True, manager_elapsed_seconds_all_invocations=total_elapsed,
                selection_seal_sha256=sha(output/"selection_seal.json"), **final))
        write(output/"state.json", dict(stage="finished" if not final["methods"]["pending"] else "partial",
                                        manager_elapsed_seconds_all_invocations=total_elapsed, **final))
        return final
    finally:
        lock.unlink()


def frozen_truth(directory, labels):
    """Reduced proper clades from the independently frozen physical case."""
    root = int(read(directory/"metadata.json")["root_bus"])
    with (directory/"branches.csv").open(encoding="utf-8", newline="") as stream:
        edges = [row for row in csv.DictReader(stream) if row["status"].lower() == "true"]
    adjacent = {}
    for edge in edges:
        a, b = int(edge["from_bus"]), int(edge["to_bus"])
        adjacent.setdefault(a, []).append(b); adjacent.setdefault(b, []).append(a)
    order, parent = [root], {root: None}
    for node in order:
        for child in adjacent.get(node, []):
            if child == parent[node]:
                continue
            if child in parent:
                raise ValueError("Frozen physical case is not radial")
            parent[child] = node; order.append(child)
    if len(edges) != len(order)-1 or not set(labels) <= set(order):
        raise ValueError("Frozen physical case is disconnected")
    below = {node: set() for node in order}
    for i, label in enumerate(labels):
        below[label].add(i)
    proper = set()
    for child in reversed(order[1:]):
        if 1 < len(below[child]) < len(labels):
            proper.add(tuple(sorted(below[child])))
        below[parent[child]].update(below[child])
    return [list(C) for C in sorted(proper)]


def expose_truth(output):
    output = Path(output).resolve(); protocol = read(output/"protocol.json")
    sealed = read(output/"selection_seal.json")
    if not sealed["sealed"] or not sealed["all_planned_terminal"] or sealed["protocol_sha256"] != sha(output/"protocol.json"):
        raise ValueError("Truth is unavailable before all results are sealed")
    if sha(output/"evaluation_map.json") != protocol["evaluation_map_sha256"]:
        raise ValueError("Evaluation map changed")
    for name, expected in sealed["result_sha256"].items():
        if sha(output/name) != expected:
            raise ValueError("A result changed after selection")
    for name, expected in protocol["evaluation_case_sha256"].items():
        if sha(output/"evaluation_cases"/name) != expected:
            raise ValueError("Frozen physical evaluation case changed")
    mapping = read(output/"evaluation_map.json")
    for job, info in protocol["inputs"].items():
        if info.get("generation_failed"):
            continue
        labels = info["labels"]
        target = output/"evaluation"/(job+".json")
        expected = dict(labels=labels, proper_clades=frozen_truth(output/"evaluation_cases"/mapping[job]["case"], labels))
        if target.exists() and read(target) != expected:
            raise ValueError("Existing truth evaluation changed")
        write(target, expected)
    return dict(truth_exposed_after_seal=True, inputs=len(protocol["inputs"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "status", "worker", "truth"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--cohort", choices=("main", "pressure"), default="main")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-inputs", type=int)
    parser.add_argument("--job"); parser.add_argument("--method")
    args = parser.parse_args()
    if args.command == "freeze":
        answer = freeze(args.output, args.cohort, args.workers)
    elif args.command == "run":
        answer = run(args.output, args.workers, args.limit_inputs)
    elif args.command == "status":
        answer = status(args.output)
    elif args.command == "truth":
        answer = expose_truth(args.output)
    else:
        answer = worker(args.output, args.job, args.method)
    if answer is not None:
        print(json.dumps(answer, indent=2), flush=True)
