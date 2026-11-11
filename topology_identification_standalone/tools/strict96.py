"""Strict 96-observation development replay: freeze, run (64/32), then grade.

Only the original train_observed.npz is staged. Python audit events catch
accidental project-data reads; this is not an operating-system sandbox.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
from time import perf_counter

import numpy as np

BASE = Path(__file__).resolve().parents[1]
SOURCES = ("algorithm.py", "optimization.py", "trees.py", "data.py", "competition.py", "tools/strict96.py")
KAPPAS = (0., .001, .003)
CASES = ("paper15", "soumalas11", "flynn16", "pengwah18")
TERMINAL = {"fitted", "failed", "budget_exceeded"}
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    temporary.write_text(text+"\n", encoding="utf-8"); temporary.replace(path)


def inside(root, relative):
    path = (Path(root)/relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(Path(root).resolve()):
        raise ValueError("A frozen path escapes its runtime")
    return path


def fingerprint(rows):
    value = hashlib.sha256(repr(tuple(rows["labels"])).encode())
    for name in ("p", "q", "y"):
        array = np.ascontiguousarray(rows[name], dtype=np.float64)
        value.update(repr(array.shape).encode()); value.update(array.tobytes())
    return value.hexdigest()


def split96(path):
    """Copy chronological rows; neither returned split retains a full-data base."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"P", "Q", "V", "root", "labels"}:
            raise ValueError("Only P/Q/V/root/labels are allowed in the sole NPZ")
        labels = tuple(map(int, archive["labels"])); n = len(labels)
        values = [np.asarray(archive[k], float) for k in ("P", "Q", "V")]
        root = np.asarray(archive["root"], float).reshape(-1)
    if not n or len(set(labels)) != n or any(v.shape not in ((96, n), (1, 96, n)) for v in values):
        raise ValueError("The input must contain exactly one ordered day of 96 points")
    if len({v.shape for v in values}) != 1 or root.shape != (96,):
        raise ValueError("Observed arrays are not aligned")
    p, q, voltage = [v.reshape(96, n) for v in values]
    raw = dict(p=p, q=q, y=root[:, None]**2-voltage**2, voltage=voltage, root=root)
    if not all(np.isfinite(v).all() for v in raw.values()):
        raise ValueError("Nonfinite observations")
    splits = []
    for start, stop in ((0, 64), (64, 96)):
        row = dict(labels=labels)
        for name, value in raw.items():
            row[name] = value[start:stop].copy(); row[name].flags.writeable = False
        splits.append(row)
    return tuple(splits)


def python_read_guard(runtime, project_root, allowed_npz):
    """Install once in a fresh worker; native-library I/O is outside this claim."""
    runtime, project_root = Path(runtime).resolve(), Path(project_root).resolve()
    allowed = {Path(p).resolve() for p in allowed_npz}
    dependency_roots = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    observed_reads = []
    def audit(event, args):
        if event != "open" or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve(); mode, flags = args[1], args[2]
        reading = ("r" in mode or "+" in mode) if isinstance(mode, str) else (flags & os.O_ACCMODE) != os.O_WRONLY
        if not reading:
            return
        if path.suffix.lower() == ".npz" and path not in allowed:
            raise PermissionError("An unapproved NPZ read was blocked")
        if path.suffix.lower() == ".npz":
            observed_reads.append(str(path))
        dependency = any(path.is_relative_to(root) for root in dependency_roots)
        if path.is_relative_to(project_root) and not path.is_relative_to(runtime) and not dependency:
            raise PermissionError("A project-file read outside the frozen runtime was blocked")
    sys.addaudithook(audit)
    return observed_reads


def sources(runtime, protocol):
    found = {name: sha(inside(runtime, name)) for name in SOURCES}
    if found != protocol["source_sha256"]:
        raise ValueError("A frozen source changed")
    return found


def check_protocol(runtime, protocol):
    if (protocol["kind"] != "strict96_64_32_development_replay" or protocol["method"] != "101"
            or tuple(protocol["kappas"]) != KAPPAS or len(protocol["inputs"]) != 4
            or protocol["split"] != dict(total=96, train=[0, 64], selection=[64, 96], shuffle=False)
            or protocol["config"]["total_seconds"] != 14400. or protocol["config"]["per_mip_seconds"] != 3600.):
        raise ValueError("Unexpected strict96 protocol")
    expected = set(SOURCES) | {"protocol.json"}
    for identity, info in protocol["inputs"].items():
        if set(info) != {"path", "sha256", "labels", "training_sha256", "selection_sha256"} or info["path"] != f"inputs/{identity}.npz":
            raise ValueError("Worker inputs contain unexpected metadata or paths")
        expected.add(info["path"])
    actual = {p.relative_to(runtime).as_posix() for p in Path(runtime).rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("The isolated runtime contains unexpected files")
    sources(runtime, protocol)


def native(runtime):
    sys.path.insert(0, str(Path(runtime).resolve()))
    modules = {name: importlib.import_module(name) for name in ("algorithm", "optimization", "trees", "data", "competition")}
    if any(not Path(module.__file__).resolve().is_relative_to(Path(runtime).resolve()) for module in modules.values()):
        raise ValueError("A native module escaped the isolated runtime")
    return modules["algorithm"], modules["data"]


def freeze(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Output is already occupied; previous evidence is preserved")
    catalog = read(BASE/"data/catalog.json"); chosen = []
    for case in CASES:
        match = [m for m in catalog.values() if m["case"] == case and m["seed_index"] == 0 and m["voltage_sigma"] == .0005]
        if len(match) != 1:
            raise ValueError("pilot4 requires exactly one seed0/middle-noise input per case")
        chosen.append(match[0])
    if any(not (BASE/name).is_file() for name in SOURCES):
        raise FileNotFoundError("A required native source is missing")
    source_hashes = {name: sha(BASE/name) for name in SOURCES}
    sys.path.insert(0, str(BASE)); algorithm = importlib.import_module("algorithm")
    config = algorithm.SearchConfig(); config.validate()
    if config.total_seconds != 14400. or config.per_mip_seconds != 3600.:
        raise ValueError("Strict pilot uses the declared 14400/3600-second budgets")
    replace(config, relative_gain=0.).validate()
    runtime = output/"runtime"; runtime.mkdir(parents=True)
    for name in SOURCES:
        target = runtime/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(BASE/name, target)
    inputs, evaluation = {}, {}
    for number, meta in enumerate(chosen):
        identity = f"input{number:02d}"; source = inside(BASE, meta["train"])
        if source.name != "train_observed.npz":
            raise ValueError("Only the original train_observed.npz is permitted")
        target = runtime/"inputs"/(identity+".npz"); target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target); train, selection = split96(target)
        inputs[identity] = dict(path=target.relative_to(runtime).as_posix(), sha256=sha(target),
            labels=list(train["labels"]), training_sha256=fingerprint(train), selection_sha256=fingerprint(selection))
        evaluation[identity] = dict(case=meta["case"], original_job=meta["job"])
    protocol = dict(kind="strict96_64_32_development_replay", created_utc=utc(), method="101", kappas=list(KAPPAS),
        config=asdict(config), inputs=inputs, source_sha256=source_hashes,
        split=dict(total=96, train=[0, 64], selection=[64, 96], shuffle=False), workers=4,
        project_root=str(BASE.parent), historical_truth_seen=True, independent_blind_evaluation=False,
        information_policy="sole train96 only; train64 fits; final32 selects frozen models; no validation/test NPZ reads",
        common_charge="one common64 preparation per input, charged once to every kappa",
        timing_scope="worker import/load/common and per-fit work; final atomic writes/spawn/manager overhead excluded")
    versions = {}
    for name in ("numpy", "scipy", "gurobipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    protocol["environment"] = dict(python=sys.version, executable=sys.executable, packages=versions)
    write(output/"evaluation_map.json", evaluation)
    protocol["evaluation_map_sha256"] = sha(output/"evaluation_map.json")
    if {name: sha(BASE/name) for name in SOURCES} != source_hashes:
        raise ValueError("Source changed during freezing")
    sources(runtime, protocol)
    write(runtime/"protocol.json", protocol); write(output/"protocol.json", protocol)
    return dict(output=str(output), planned=12, observations_per_input=96, protocol_sha256=sha(output/"protocol.json"))


def worker(output, identity):
    started = perf_counter(); output = Path(output).resolve(); runtime = output/"runtime"
    protocol = read(runtime/"protocol.json"); info = protocol["inputs"][identity]
    if Path(__file__).resolve() != runtime/"tools/strict96.py":
        raise ValueError("Workers must execute the frozen runner")
    input_file = inside(runtime, info["path"]); folder = output/"runs"/identity
    observed_reads = python_read_guard(runtime, protocol["project_root"], [input_file])
    check_protocol(runtime, protocol)
    source_receipt = sources(runtime, protocol)
    common, common_seconds = None, None
    try:
        algorithm, data = native(runtime)
        if sha(input_file) != info["sha256"]:
            raise ValueError("The sole observed input changed")
        training, selection = split96(input_file)
        if (fingerprint(training), fingerprint(selection)) != (info["training_sha256"], info["selection_sha256"]):
            raise ValueError("The fixed chronological split changed")
        train, selected_rows = data.Observation(**training), data.Observation(**selection)
        config = algorithm.SearchConfig(**protocol["config"])
        common = algorithm.prepare_common(train, config)
        native_seconds = common["seconds"]; common_seconds = perf_counter()-started; common["seconds"] = common_seconds
        write(folder/"common.json", dict(common=common, charged_seconds=common_seconds, native_seconds=native_seconds,
            source_sha256=source_receipt, sole_input_sha256=info["sha256"], train_points=64, selection_points=32))
    except Exception as exc:
        common_seconds = perf_counter()-started
        for kappa in KAPPAS:
            write(folder/(str(kappa)+".json"), dict(status="failed", kappa=kappa, error=repr(exc),
                config=dict(protocol["config"], relative_gain=kappa), input_sha256=info["sha256"],
                split_sha256={k: info[k] for k in ("training_sha256", "selection_sha256")}, source_sha256_before=source_receipt,
                protocol_sha256=sha(runtime/"protocol.json"), train_points=64, selection_points=32,
                common_seconds=common_seconds, fresh_seconds=0., total_seconds=common_seconds, method_started=False))
        return
    for kappa in KAPPAS:
        before = perf_counter(); record = dict(kappa=kappa, method="101", train_points=64, selection_points=32,
            input_sha256=info["sha256"], split_sha256={k: info[k] for k in ("training_sha256", "selection_sha256")},
            source_sha256_before=source_receipt, protocol_sha256=sha(runtime/"protocol.json"), common_seconds=common_seconds)
        try:
            configured = replace(config, relative_gain=kappa); record["config"] = asdict(configured)
            result = algorithm.fit(train, selected_rows, method="101", config=configured, common=common)
            record.update(result=result, source_sha256_after=sources(runtime, protocol))
            if sha(input_file) != info["sha256"]:
                raise ValueError("The sole observed input changed during fitting")
            elapsed = perf_counter()-before
            exceeded = result.get("total_cap_exceeded") or elapsed+common_seconds > configured.total_seconds+.05
            record["status"] = "budget_exceeded" if exceeded else "fitted"
        except Exception as exc:
            record.update(status="failed", error=repr(exc))
        elapsed = perf_counter()-before
        record.update(fresh_seconds=elapsed, total_seconds=elapsed+common_seconds,
                      observed_npz_opened=sorted(set(observed_reads)),
                      extra_observation_files_allowed=False)
        write(folder/(str(kappa)+".json"), record)


def mse(rows, point):
    predicted = np.zeros_like(rows["y"])
    for C, r, x in zip(point["supports"], point["r"], point["x"], strict=True):
        contribution = rows["p"][:, C].sum(axis=1)*r+rows["q"][:, C].sum(axis=1)*x
        predicted[:, C] += contribution[:, None]
    return float(np.mean((rows["y"]-predicted)**2))


def seal(output, protocol):
    output = Path(output); runtime = output/"runtime"; sources(runtime, protocol)
    if (output/"selection_seal.json").exists():
        raise FileExistsError("An existing selection seal cannot be reset")
    hashes, results, means, scores, selected_by_input = {}, {}, {}, {}, {}
    for identity, info in protocol["inputs"].items():
        if sha(inside(runtime, info["path"])) != info["sha256"]:
            raise ValueError("An observed input changed before sealing")
        _, selected_rows = split96(inside(runtime, info["path"]))
        for kappa in KAPPAS:
            path = output/"runs"/identity/(str(kappa)+".json"); row = read(path)
            if row["status"] not in TERMINAL:
                raise ValueError("Every planned method must be terminal before selection is sealed")
            hashes[path.relative_to(output).as_posix()] = sha(path); results[identity, kappa] = row
            if row["status"] == "fitted":
                expected = dict(protocol["config"], relative_gain=kappa)
                if row.get("config") != expected or row.get("input_sha256") != info["sha256"] or row.get("source_sha256_after") != protocol["source_sha256"]:
                    raise ValueError("A fitted result has inconsistent source/config/input provenance")
                result = row["result"]; loss = mse(selected_rows, result["selected"]["point"])
                reference = float(result["reference_mse"])
                if reference <= 0 or not np.isfinite(reference) or not np.isclose(loss, result["selected"]["validation_mse"], rtol=1e-8, atol=1e-15):
                    raise ValueError("The frozen selection MSE does not replay")
                row["replayed_scaled_mse"] = loss/reference
        local_scores = {str(k): results[identity, k]["replayed_scaled_mse"] for k in KAPPAS if results[identity, k]["status"] == "fitted"}
        scores[identity] = local_scores
        local_best = min(local_scores.values()) if local_scores else None
        selected_by_input[identity] = min((float(k) for k, v in local_scores.items()
            if v <= local_best+protocol["config"]["validation_tie"]), default=None)
    for kappa in KAPPAS:
        rows = [results[identity, kappa] for identity in protocol["inputs"]]
        if all(r["status"] == "fitted" for r in rows):
            means[str(kappa)] = float(np.mean([r["replayed_scaled_mse"] for r in rows]))
    unique = set(selected_by_input.values())
    selected = next(iter(unique)) if len(unique) == 1 and None not in unique else None
    receipt = dict(sealed=True, created_utc=utc(), selected_kappa=selected, mean_normalized_selection_mse=means,
        selected_kappa_by_input=selected_by_input, normalized_selection_mse_by_input=scores,
        aggregate_mse_role="diagnostic_only_not_used_for_selection", selected_kappa_role="unanimous_mapping_summary_only",
        selection_rule="each input uses only its own MSE32/reference64; numerical ties choose smaller kappa",
        all_planned_terminal=True, planned=12, fitted=sum(r["status"] == "fitted" for r in results.values()),
        result_sha256=hashes, protocol_sha256=sha(output/"protocol.json"), source_sha256=protocol["source_sha256"],
        truth_opened=False, extra_validation_test_npz_opened=False, historical_truth_seen=True)
    write(output/"selection_seal.json", receipt)
    return receipt


def run(output):
    output = Path(output).resolve(); protocol = read(output/"protocol.json"); runtime = output/"runtime"
    if read(runtime/"protocol.json") != protocol:
        raise ValueError("The worker protocol differs from the parent protocol")
    check_protocol(runtime, protocol)
    if (output/"run_started.json").exists() or (output/"runs").exists() or (output/"selection_seal.json").exists():
        raise FileExistsError("An attempt already exists; no automatic retry or reset is allowed")
    write(output/"run_started.json", dict(created_utc=utc(), protocol_sha256=sha(output/"protocol.json")))
    started = perf_counter()
    def launch(identity):
        folder = output/"runs"/identity; folder.mkdir(parents=True)
        write(folder/"attempt.json", dict(state="started", created_utc=utc()))
        before = perf_counter(); error = None
        try:
            environment = dict(os.environ, **{name: "1" for name in THREADS})
            with (folder/"worker.log").open("w", encoding="utf-8") as log:
                process = subprocess.run([sys.executable, "-I", "-B", str(runtime/"tools/strict96.py"), "worker", str(output), identity],
                    cwd=runtime, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if process.returncode:
                    error = "worker exited with code "+str(process.returncode)
        except Exception as exc:
            error = repr(exc)
        for kappa in KAPPAS:
            path = folder/(str(kappa)+".json")
            if not path.exists():
                write(path, dict(status="failed", kappa=kappa, error=error or "worker returned no result",
                    config=dict(protocol["config"], relative_gain=kappa), expected_input_sha256=protocol["inputs"][identity]["sha256"],
                    protocol_sha256=sha(output/"protocol.json"), source_receipt_available=False,
                    fresh_seconds=None, common_seconds=None, total_seconds=None))
        write(folder/"attempt.json", dict(state="terminal", finished_utc=utc(), observed_worker_seconds=perf_counter()-before, error=error))
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(launch, protocol["inputs"]))
    receipt = seal(output, protocol)
    write(output/"completion.json", dict(all_planned_terminal=True, calendar_seconds=perf_counter()-started,
        selection_seal_sha256=sha(output/"selection_seal.json"), planned=12, fitted=receipt["fitted"]))
    return dict(planned=12, fitted=receipt["fitted"], selected_kappa=receipt["selected_kappa"],
                selected_kappa_by_input=receipt["selected_kappa_by_input"])


def truth(folder, labels):
    """Evaluation only, after seal verification; never called by a worker."""
    import csv
    root = int(read(Path(folder)/"metadata.json")["root_bus"]); graph = {root: []}
    with (Path(folder)/"branches.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if str(row.get("status", "True")).lower() not in {"true", "1", "1.0"}:
                continue
            a, b = int(row["from_bus"]), int(row["to_bus"])
            graph.setdefault(a, []).append(b); graph.setdefault(b, []).append(a)
    parent, order = {root: None}, [root]
    for node in order:
        for child in graph[node]:
            if child != parent[node]:
                if child in parent:
                    raise ValueError("Truth contains a cycle")
                parent[child] = node; order.append(child)
    if len(order) != len(graph) or not set(labels) <= set(graph):
        raise ValueError("Truth is disconnected or lacks terminals")
    descendants = {node: {i for i, label in enumerate(labels) if node == label} for node in graph}; family = set()
    for node in reversed(order[1:]):
        family.add(tuple(sorted(descendants[node]))); descendants[parent[node]].update(descendants[node])
    return {C for C in family if 1 < len(C) < len(labels)}


def clades(point, n):
    return {tuple(sorted(C)) for C in point["supports"] if 1 < len(C) < n}


def grade(output):
    output = Path(output).resolve(); protocol = read(output/"protocol.json"); sealed = read(output/"selection_seal.json")
    if not sealed.get("sealed") or not sealed.get("all_planned_terminal") or sealed["protocol_sha256"] != sha(output/"protocol.json"):
        raise ValueError("Truth scoring requires a valid final selection seal")
    completion = read(output/"completion.json")
    if not completion.get("all_planned_terminal") or completion["selection_seal_sha256"] != sha(output/"selection_seal.json"):
        raise ValueError("The manager did not finish the sealed run")
    if (output/"grade.json").exists():
        raise FileExistsError("An existing grade is preserved")
    planned = {f"runs/{identity}/{kappa}.json" for identity in protocol["inputs"] for kappa in KAPPAS}
    if set(sealed["result_sha256"]) != planned:
        raise ValueError("The selection seal omits planned results")
    mapping = sealed["selected_kappa_by_input"]
    if set(mapping) != set(protocol["inputs"]) or any(k is not None and k not in KAPPAS for k in mapping.values()):
        raise ValueError("The per-input sealed selection mapping is invalid")
    for relative, expected in sealed["result_sha256"].items():
        file = inside(output, relative)
        if sha(file) != expected or read(file)["status"] not in TERMINAL:
            raise ValueError("A frozen result changed after selection")
    if sha(output/"evaluation_map.json") != protocol["evaluation_map_sha256"]:
        raise ValueError("The separately held evaluation map changed")
    evaluation = read(output/"evaluation_map.json"); rows = []
    python_read_guard(BASE.parent, BASE.parent, [])
    for identity, info in protocol["inputs"].items():
        real = truth(BASE/"cases"/evaluation[identity]["case"], info["labels"]); n = len(info["labels"])
        for kappa in KAPPAS:
            record = read(output/"runs"/identity/(str(kappa)+".json"))
            row = dict(input=identity, kappa=kappa, status=record["status"], F1=None, exact=None, top2_contains_true=None,
                       total_seconds=record.get("total_seconds"))
            if record["status"] == "fitted":
                result = record["result"]; predicted = clades(result["selected"]["point"], n)
                tp, fp, fn = len(predicted & real), len(predicted-real), len(real-predicted)
                secondary = result["competition"]["secondary_index"]
                if result["competition"]["primary_index"] != result["selected"]["index"]:
                    raise ValueError("Competition changed the actual primary model")
                alternate = clades(result["frozen_candidates"][secondary]["point"], n) if secondary is not None else None
                ledger = result.get("MIQP", []); flag = "bound_consistent_with_fixed_family_qp"
                row.update(F1=2*tp/(2*tp+fp+fn) if tp+fp+fn else 1., exact=predicted == real,
                    top2_contains_true=predicted == real or alternate == real, secondary_available=secondary is not None,
                    native_bound_check_false=sum(r.get(flag) is False for r in ledger),
                    native_bound_check_missing=sum(flag not in r for r in ledger))
            rows.append(row)
    def summarize(group, **label):
        valid = [r for r in group if r["status"] == "fitted"]
        return dict(label, planned=4, fitted=len(valid), F1_mean=float(np.mean([r["F1"] for r in valid])) if valid else None,
            exact=sum(r["exact"] for r in valid), top2_contains_true=sum(r["top2_contains_true"] for r in valid),
            exact_rate_planned=sum(r["exact"] for r in valid)/4, top2_rate_planned=sum(r["top2_contains_true"] for r in valid)/4,
            secondary_available=sum(r["secondary_available"] for r in valid), native_bound_check_false=sum(r["native_bound_check_false"] for r in valid),
            native_bound_check_missing=sum(r["native_bound_check_missing"] for r in valid),
            total_seconds_recorded=sum(r["total_seconds"] for r in group if r["total_seconds"] is not None),
            timing_unknown=sum(r["total_seconds"] is None for r in group))
    summary = [summarize([r for r in rows if r["kappa"] == k], kappa=k) for k in KAPPAS]
    selected_rows = [next((r for r in rows if r["input"] == identity and r["kappa"] == kappa),
        dict(input=identity, kappa=None, status="no_eligible_kappa", total_seconds=None)) for identity, kappa in mapping.items()]
    policy = summarize(selected_rows, policy="sealed_per_input_MSE32")
    policy.update(tuning_total_seconds_recorded=sum(r["total_seconds"] for r in rows if r["total_seconds"] is not None),
                  timing_scope="total_seconds_recorded is chosen-fit cost only; tuning_total_seconds_recorded includes all three fits")
    report = dict(development_replay=True, historical_truth_seen=True, selection_seal_sha256=sha(output/"selection_seal.json"),
        selected_kappa=sealed["selected_kappa"], selected_kappa_by_input=mapping, rows=rows, summary=summary,
        selected_rows=selected_rows, selected_policy_summary=policy,
        scope="F1/top2 cover only the saved finite library; native gaps are not audited whole-tree proofs; no extra observed splits read")
    write(output/"grade.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "worker", "grade")); parser.add_argument("output", type=Path)
    parser.add_argument("identity", nargs="?"); args = parser.parse_args()
    result = worker(args.output, args.identity) if args.command == "worker" else globals()[args.command](args.output)
    if result is not None:
        print(json.dumps(result, indent=2))
