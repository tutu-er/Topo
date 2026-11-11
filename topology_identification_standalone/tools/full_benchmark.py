"""Freeze and run 48 fresh common fits and 480 fresh MSE-selected methods.

Only this standalone folder is copied into ``output/runtime``. Each task runs
in a fresh isolated Python process, with one BLAS/Gurobi thread. No test/truth
scores are computed here. Independent post-analysis starts after all tasks end.

Commands: ``freeze OUTPUT``, ``run OUTPUT``, ``status OUTPUT``. A task that has
an attempt but no terminal result requires an explicit audit before resuming;
the runner never grants it a fresh budget automatically.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import importlib.abc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter, sleep
import traceback


BASE = Path(__file__).resolve().parents[1]
METHODS = ("rnj", "nj", "000", "100", "010", "001", "110", "101", "011", "111")
TERMINAL = {"complete", "fitted", "failed", "interrupted", "budget_exceeded"}
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
BUDGET_TOLERANCE = .05
PARENT_PACKAGES = {"research_experiments", "rnj_wzzt", "terminal_case33", "theft_wzzt"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    for attempt in range(100):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 99:
                raise
            sleep(.05)


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    text = json.dumps(value, default=_json_value, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    for attempt in range(100):
        try:
            temporary.write_text(text, encoding="utf-8", newline="\n")
            temporary.replace(path)
            if path.name in {"common.json", "result.json"} and value.get("status") in TERMINAL:
                # This tiny index serves progress polling only. Independent
                # post-analysis reads and verifies the complete raw result.
                write(path.with_name("terminal_status.json"), dict(status=value["status"],
                    result_file=path.name, fullresult_sha256=sha(path),
                    protocol_sha256=value.get("protocol_sha256")))
            return
        except PermissionError:
            if attempt == 99:
                raise
            sleep(.05)


def inside(base, relative):
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError("Expected a relative path inside the frozen runtime")
    path = (Path(base)/relative).resolve()
    if not path.is_relative_to(Path(base).resolve()):
        raise ValueError("A path escapes its frozen runtime")
    return path


class _NoParentProject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in PARENT_PACKAGES:
            raise ImportError("Parent-project imports are forbidden: "+fullname)


def native_imports(base):
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    sys.meta_path.insert(0, _NoParentProject())
    sys.path.insert(0, str(Path(base).resolve()))
    import algorithm
    import data
    return algorithm, data


def versions():
    dependencies = {}
    for name in ("numpy", "scipy", "gurobipy"):
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = "not installed"
    return dict(python=sys.version, executable=sys.executable, dependencies=dependencies)


def freeze(output, *, workers=8):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new output folder; frozen protocols are immutable")
    if not 1 <= workers <= 8:
        raise ValueError("Use one to eight single-thread workers")
    algorithm, _ = native_imports(BASE)
    config = algorithm.SearchConfig(); config.validate()
    catalog = read(BASE/"data/catalog.json")
    if len(catalog) != 48 or {m["case"] for m in catalog.values()} != {"paper15", "soumalas11", "flynn16", "pengwah18"}:
        raise ValueError("The complete four-network 48-input bundled panel is required")
    inventory = read(BASE/"data/manifest.json")["files"]
    if len(inventory) != 208:
        raise ValueError("The declared bundled-data inventory changed")
    for name, expected in inventory.items():
        if sha(inside(BASE, name)) != expected:
            raise ValueError("Bundled data changed: "+name)
    for job, meta in catalog.items():
        if meta["data_error"] or meta["N"] != 96 or meta["split_points"] != dict(train=96, validation=288, test=96):
            raise ValueError("An input is invalid or no longer has the declared day96 splits: "+job)
    native_files = {p.name: p for p in BASE.glob("*.py")}
    native_files["tools/full_benchmark.py"] = Path(__file__).resolve()
    package_tool = BASE/"tools/package.py"
    if package_tool.exists():
        native_files["tools/package.py"] = package_tool
    output.mkdir(parents=True)
    runtime = output/"runtime"; runtime.mkdir()
    for folder in ("cases", "data"):
        shutil.copytree(BASE/folder, runtime/folder)
    for name, source in native_files.items():
        target = runtime/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in ("requirements.txt", "requirements-tested.txt", "README.md", "MODEL.md", "ALGORITHMS.md", "DATA.md", "PROVENANCE.md"):
        if (BASE/name).exists():
            shutil.copyfile(BASE/name, runtime/name)
    runtime_hashes = {p.relative_to(runtime).as_posix(): sha(p) for p in sorted(runtime.rglob("*")) if p.is_file()}
    source_hashes = {name: runtime_hashes[name] for name in native_files}
    if any(sha(native_files[name]) != expected for name, expected in source_hashes.items()):
        raise ValueError("A native source changed while its snapshot was being copied")
    for name, expected in inventory.items():
        if runtime_hashes.get(name) != expected:
            raise ValueError("The copied data differs from its declared inventory: "+name)
    write(output/"catalog.json", catalog)
    input_hashes = {job: {split: sha(inside(runtime, meta[split])) for split in ("train", "validation", "test")}
                    for job, meta in catalog.items()}
    protocol = dict(created_utc=utc_now(), jobs=list(catalog), methods=list(METHODS), workers=workers,
        planned_common=48, planned_methods=480, planned_inputs=48, config=asdict(config),
        nominal_method_hours=480*config.total_seconds/3600., common_fresh=True, methods_fresh=True,
        reused_historical_fits=False, primary_selection="validation_mse", diagnostic_selection="validation_mae",
        runtime_directory="runtime", source_hashes=source_hashes, runtime_file_hashes=runtime_hashes,
        catalog_hash=sha(output/"catalog.json"), bundled_manifest_hash=sha(runtime/"data/manifest.json"),
        input_hashes=input_hashes, environment=versions(), single_native_thread=True,
        thread_environment={name: "1" for name in THREAD_VARIABLES}, budget_tolerance_seconds=BUDGET_TOLERANCE,
        common_charge="each input's complete common-worker elapsed cost charged once to each of its ten methods",
        method_queue_order="round-robin inputs, with method order rotated by input index",
        effective_budget="configured total minus method worker pre-fit setup seconds; all other parameters unchanged",
        timing_boundary="worker import/check/load/compute/checkpoint preparation counted; final atomic result write, process spawn and manager bookkeeping excluded and separately visible in attempt/calendar records",
        interruption_policy="terminal records are not retried; any attempt without a result requires explicit audit",
        truth_test_policy="native workers load train/validation only; test and physical truth reserved for independent post-analysis")
    write(output/"protocol.json", protocol)
    write(output/"state.json", dict(stage="frozen", common=status_counts(output, protocol, "common"), methods=status_counts(output, protocol, "method")))
    print(f"FROZEN: 48 fresh common tasks; 480 fresh methods; {protocol['nominal_method_hours']:g} nominal method-hours; no fitting started", flush=True)
    return protocol


def _task_folder(output, kind, job, method=None):
    return output/"common"/job if kind == "common" else output/"runs"/job/method


def status_counts(output, protocol, kind):
    tasks = [(job, None) for job in protocol["jobs"]] if kind == "common" else [(job, method) for job in protocol["jobs"] for method in protocol["methods"]]
    counts = dict(planned=len(tasks), complete=0, fitted=0, failed=0, interrupted=0, budget_exceeded=0,
                  pending=0, attempts_without_result=0)
    for job, method in tasks:
        folder = _task_folder(output, kind, job, method)
        file = folder/("common.json" if kind == "common" else "result.json")
        if file.exists():
            index = folder/"terminal_status.json"
            summary = read(index) if index.exists() else read(file)
            if index.exists() and summary.get("result_file") != file.name:
                summary = read(file)
            state = summary["status"]
            if state not in TERMINAL:
                raise ValueError("Unknown terminal task state: "+state)
            counts[state] += 1
        else:
            counts["pending"] += 1
            counts["attempts_without_result"] += (folder/"attempt.json").exists()
    return counts


def status(output):
    output = Path(output).resolve(); protocol = read(output/"protocol.json")
    return dict(common=status_counts(output, protocol, "common"), methods=status_counts(output, protocol, "method"),
                manager_lock=(output/"run.lock").exists(), completion=(output/"completion.json").exists())


def _check_frozen(output, protocol):
    runtime = output/"runtime"
    if sha(output/"catalog.json") != protocol["catalog_hash"]:
        raise ValueError("The frozen input catalog changed")
    for name, expected in protocol["runtime_file_hashes"].items():
        if sha(inside(runtime, name)) != expected:
            raise ValueError("The frozen runtime changed: "+name)


def _loaded_sources(runtime, protocol):
    records = {}
    for name in ("algorithm", "optimization", "trees", "data", "__main__"):
        module = sys.modules.get(name)
        if module is None or not getattr(module, "__file__", None):
            continue
        file = Path(module.__file__).resolve()
        if not file.is_relative_to(runtime):
            raise ValueError("A native module escaped the frozen runtime: "+name)
        relative = file.relative_to(runtime).as_posix()
        if sha(file) != protocol["source_hashes"].get(relative):
            raise ValueError("Loaded native source does not match its manifest: "+name)
        records[name] = dict(relative_path=relative, sha256=sha(file))
    if any(name.split(".")[0] in PARENT_PACKAGES for name in sys.modules):
        raise ValueError("A forbidden parent-project module was loaded")
    return records


def worker(payload_file):
    started, start_utc = perf_counter(), utc_now()
    payload = read(payload_file); output = Path(payload["output"]).resolve()
    protocol = read(output/"protocol.json"); runtime = (output/"runtime").resolve()
    target = Path(payload["result"]).resolve()
    kind, job, method = payload["kind"], payload["job"], payload.get("method")
    if kind not in {"common", "method"} or job not in protocol["jobs"] or (kind == "method" and method not in protocol["methods"]):
        raise ValueError("Unknown frozen task identity")
    expected_folder = _task_folder(output, kind, job, method)
    if target.parent != expected_folder or sha(output/"protocol.json") != payload["protocol_sha256"]:
        raise ValueError("The worker payload does not match its frozen task")
    expected_source = runtime/"tools/full_benchmark.py"
    if Path(__file__).resolve() != expected_source or sha(expected_source) != protocol["source_hashes"]["tools/full_benchmark.py"]:
        raise ValueError("Workers must execute the frozen runtime's runner")
    record = dict(job=job, method=method, kind=kind, started_utc=start_utc, pid=os.getpid(),
                  protocol_sha256=payload["protocol_sha256"], payload_sha256=sha(payload_file),
                  configured_config=protocol["config"], selection_metric="validation_mse",
                  truth_used=False, test_opened=False)
    write(target.with_name("worker_started.json"), record)
    try:
        algorithm, data = native_imports(runtime)
        before = _loaded_sources(runtime, protocol)
        for split, relative in payload["inputs"].items():
            if split not in ("train", "validation") or sha(inside(runtime, relative)) != protocol["input_hashes"][job][split]:
                raise ValueError("A worker input changed or includes a disallowed split")
        train = data.load_observations(inside(runtime, payload["inputs"]["train"]))
        config = algorithm.SearchConfig(**protocol["config"]); config.validate()
        write(target.with_name("heartbeat.json"), dict(stage="worker_initialized", elapsed_seconds=perf_counter()-started))
        if kind == "common":
            common = algorithm.prepare_common(train, config)
            native_seconds = float(common["seconds"])
            after = _loaded_sources(runtime, protocol)
            charged = perf_counter()-started
            common.update(job=job, status="complete" if charged <= config.total_seconds+BUDGET_TOLERANCE else "budget_exceeded",
                prepare_seconds=native_seconds, common_compute_seconds=native_seconds, seconds=charged,
                protocol_sha256=payload["protocol_sha256"], train_input_sha256=protocol["input_hashes"][job]["train"],
                started_utc=start_utc, finished_utc=utc_now(), pid=os.getpid(), runtime_sources_before=before,
                runtime_sources_after=after, final_result_write_included=False)
            write(target, common)
        else:
            common_file = output/"common"/job/"common.json"
            common = read(common_file)
            if common["status"] != "complete" or sha(common_file) != payload["common_sha256"]:
                raise ValueError("The shared training common record is unavailable or changed")
            validation = data.load_observations(inside(runtime, payload["inputs"]["validation"]))
            pre_fit = perf_counter()-started
            allowance = config.total_seconds-pre_fit
            if allowance <= config.output_reserve_seconds:
                raise TimeoutError("No total method allowance remains after worker setup")
            effective = replace(config, total_seconds=allowance)
            def heartbeat(value):
                write(target.with_name("checkpoint.json"), dict(job=job, method=method, timestamp_utc=utc_now(),
                    worker_elapsed_seconds=perf_counter()-started, common_seconds=common["seconds"], **value))
            result = algorithm.fit(train, validation, method=method, config=effective, common=common, heartbeat=heartbeat)
            if result.get("selection_metric") != "validation_mse" or result["selected"] != result["selected_mse_view"]:
                raise ValueError("A worker did not return the required primary MSE selection")
            after = _loaded_sources(runtime, protocol)
            fresh = perf_counter()-started; total = fresh+float(common["seconds"])
            exceeded = bool(result.get("total_cap_exceeded")) or total > config.total_seconds+BUDGET_TOLERANCE
            record.update(status="budget_exceeded" if exceeded else "fitted", result=result,
                common_sha256=payload["common_sha256"], common_seconds=float(common["seconds"]),
                common_compute_seconds=float(common["common_compute_seconds"]), fresh_seconds=fresh, total_seconds=total,
                pre_fit_debit_seconds=pre_fit, effective_config=asdict(effective), budget_exceeded=exceeded,
                train_val_input_hashes={split: protocol["input_hashes"][job][split] for split in ("train", "validation")},
                runtime_sources_before=before, runtime_sources_after=after, finished_utc=utc_now(),
                final_result_write_included=False)
            write(target, record)
    except Exception:
        elapsed = perf_counter()-started
        common_seconds = 0.
        common_file = output/"common"/job/"common.json"
        if kind == "method" and common_file.exists():
            common_seconds = float(read(common_file).get("seconds", 0.))
        record.update(status="failed", error=traceback.format_exc(), fresh_seconds=elapsed,
                      common_seconds=common_seconds, total_seconds=elapsed+common_seconds, finished_utc=utc_now())
        write(target, record)
        raise


def _payload(output, protocol, catalog, kind, job, method=None):
    folder = _task_folder(output, kind, job, method)
    result = folder/("common.json" if kind == "common" else "result.json")
    splits = ("train",) if kind == "common" else ("train", "validation")
    value = dict(output=str(output), kind=kind, job=job, method=method, result=str(result),
                 protocol_sha256=sha(output/"protocol.json"), inputs={split: catalog[job][split] for split in splits})
    if kind == "method":
        value["common_sha256"] = sha(output/"common"/job/"common.json")
    return value


def _run_queue(output, protocol, payloads, *, stage, workers):
    active, queue = {}, list(payloads)
    environment = os.environ.copy()
    environment.update({name: "1" for name in THREAD_VARIABLES})
    while queue or active:
        while queue and len(active) < workers:
            payload = queue.pop(0); target = Path(payload["result"])
            file, attempt = target.with_name("payload.json"), target.with_name("attempt.json")
            write(file, payload)
            write(attempt, dict(state="reserved", reserved_utc=utc_now(), payload_sha256=sha(file), pid=None))
            log = target.with_name("worker.log").open("w", encoding="utf-8")
            began = perf_counter()
            try:
                process = subprocess.Popen([sys.executable, "-I", "-B", str(output/"runtime/tools/full_benchmark.py"),
                    "worker", str(file)], cwd=output/"runtime", env=environment, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                write(attempt, dict(state="running", started_utc=utc_now(), payload_sha256=sha(file), pid=process.pid))
                active[str(target)] = (process, log, target, began)
            except Exception:
                log.close()
                elapsed = perf_counter()-began
                common_seconds = 0.
                if payload["kind"] == "method":
                    shared = read(output/"common"/payload["job"]/"common.json")
                    common_seconds = float(shared.get("seconds", shared.get("fresh_seconds", 0.)))
                write(target, dict(job=payload["job"], method=payload.get("method"), kind=payload["kind"],
                    status="failed", method_started=False, reason="process_launch_failed", error=traceback.format_exc(),
                    fresh_seconds=elapsed, common_seconds=common_seconds, total_seconds=elapsed+common_seconds,
                    protocol_sha256=payload["protocol_sha256"]))
                write(attempt, dict(state="launch_failed", finished_utc=utc_now(), payload_sha256=sha(file)))
        for key, (process, log, target, began) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close(); del active[key]
            if not target.exists():
                payload = read(target.with_name("payload.json"))
                observed = perf_counter()-began
                common_seconds = 0.
                if payload["kind"] == "method":
                    shared = read(output/"common"/payload["job"]/"common.json")
                    common_seconds = float(shared.get("seconds", shared.get("fresh_seconds", 0.)))
                write(target, dict(job=payload["job"], method=payload.get("method"), kind=payload["kind"],
                    status="interrupted", reason="worker_exited_without_result", exit_code=code,
                    driver_observed_seconds=observed, fresh_seconds=observed, common_seconds=common_seconds,
                    total_seconds=observed+common_seconds, seconds=observed,
                    timing_scope="driver-observed upper elapsed; native return unavailable",
                    protocol_sha256=payload["protocol_sha256"]))
            write(target.with_name("attempt.json"), dict(state="finished", finished_utc=utc_now(), pid=process.pid,
                  return_code=code, driver_observed_seconds=perf_counter()-began,
                  payload_sha256=sha(target.with_name("payload.json"))))
            result = read(target)
            print(stage, result["job"], result.get("method"), result["status"], flush=True)
        write(output/"state.json", dict(stage=stage, **status(output),
            active={key: value[0].pid for key, value in active.items()}, queued=len(queue), updated_utc=utc_now()))
        if active:
            sleep(.5)


def run(output, *, workers=None):
    output = Path(output).resolve(); protocol = read(output/"protocol.json")
    _check_frozen(output, protocol)
    workers = int(workers if workers is not None else protocol["workers"])
    if not 1 <= workers <= 8:
        raise ValueError("Use one to eight single-thread workers")
    for kind in ("common", "method"):
        if status_counts(output, protocol, kind)["attempts_without_result"]:
            raise RuntimeError("An attempt without a terminal result requires audit; no fresh budget will be granted")
    protocol_hash = sha(output/"protocol.json")
    for kind, filename in (("common", "common.json"), ("runs", "result.json")):
        for file in (output/kind).rglob(filename):
            if read(file).get("protocol_sha256") != protocol_hash:
                raise ValueError("A saved task belongs to another protocol: "+str(file))
    lock = output/"run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor); write(lock, dict(pid=os.getpid(), started_utc=utc_now(), workers=workers))
    started = perf_counter(); catalog = read(output/"catalog.json")
    invocation = output/"manager_attempts"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+f"_{os.getpid()}.json")
    write(invocation, dict(state="running", pid=os.getpid(), started_utc=utc_now(), workers=workers))
    common_queue = [_payload(output, protocol, catalog, "common", job) for job in protocol["jobs"]
                    if not (output/"common"/job/"common.json").exists()]
    _run_queue(output, protocol, common_queue, stage="common", workers=workers)
    methods_queue = []
    # Every input receives all ten methods; rotation balances method order
    # without changing an algorithm's search order or using its scores.
    for round_number in range(len(protocol["methods"])):
        for index, job in enumerate(protocol["jobs"]):
            method = protocol["methods"][(round_number+index) % len(protocol["methods"])]
            target = output/"runs"/job/method/"result.json"
            if target.exists():
                continue
            common = read(output/"common"/job/"common.json")
            if common["status"] != "complete":
                write(target, dict(job=job, method=method, kind="method", status="failed", method_started=False,
                    reason="common_preprocessing_unavailable", common_status=common["status"], fresh_seconds=0.,
                    common_seconds=float(common.get("seconds", common.get("fresh_seconds", 0.))),
                    total_seconds=float(common.get("seconds", common.get("fresh_seconds", 0.))),
                    protocol_sha256=sha(output/"protocol.json")))
            else:
                methods_queue.append(_payload(output, protocol, catalog, "method", job, method))
    _run_queue(output, protocol, methods_queue, stage="methods", workers=workers)
    final = status(output)
    if final["common"]["pending"] or final["methods"]["pending"]:
        raise RuntimeError("The manager cannot mark an unfinished queue complete")
    final.update(finished_utc=utc_now(), manager_elapsed_seconds=perf_counter()-started, workers=workers,
                 invocation_file=invocation.relative_to(output).as_posix(), manager_lock=False, completion=True)
    write(invocation, dict(state="finished", pid=os.getpid(), **final))
    elapsed = sum(float(read(file).get("manager_elapsed_seconds", 0.)) for file in (output/"manager_attempts").glob("*.json"))
    final["manager_elapsed_seconds_all_invocations"] = elapsed
    write(output/"completion.json", final)
    lock.unlink()
    write(output/"state.json", dict(stage="finished", **status(output), updated_utc=utc_now()))
    print(json.dumps(final, indent=2), flush=True)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "status", "worker"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze(args.output, workers=args.workers or 8)
    elif args.command == "run":
        run(args.output, workers=args.workers)
    elif args.command == "status":
        print(json.dumps(status(args.output), indent=2))
    else:
        worker(args.output)


if __name__ == "__main__":
    main()
