"""Independent numerical/topology audit of the fresh 48 x 10 benchmark.

Only NumPy and the standard library are imported. No algorithm, fitter, tree
builder or Gurobi module is used. Training/validation models are replayed first;
physical truth and test arrays are opened only after every planned run has a
terminal result. Four shared seed groups are the resampling units.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import combinations, permutations
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter, sleep
import traceback

import numpy as np


METHODS = ("rnj", "nj", "000", "100", "010", "001", "110", "101", "011", "111")
FACTORIAL = METHODS[2:]
TERMINAL = {"fitted", "complete", "failed", "interrupted", "budget_exceeded", "common_failed"}
MODEL_STATUSES = {"fitted", "complete"}
BOOTSTRAP_SEED = 20261111


def read(path):
    for attempt in range(100):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 99:
                raise
            sleep(.05)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["status"])
        writer.writeheader(); writer.writerows(rows)


def local(runtime, relative):
    path = (runtime/relative).resolve()
    if not path.is_relative_to(runtime):
        raise ValueError("A runtime/catalog path escapes the frozen folder")
    return path


def close(actual, saved, name, *, atol=1e-10, rtol=1e-7):
    actual, saved = float(actual), float(saved)
    if not math.isfinite(actual) or not math.isfinite(saved) or abs(actual-saved) > atol+rtol*max(abs(actual), abs(saved)):
        raise ValueError(f"{name} does not reproduce: {actual!r} != {saved!r}")


def canonical(family, n=None, *, laminar=True):
    rows = []
    for support in family:
        row = tuple(int(v) for v in support)
        if any(float(v) != i for v, i in zip(support, row)) or not row or len(set(row)) != len(row):
            raise ValueError("A support has noninteger, empty or repeated entries")
        row = tuple(sorted(row))
        if n is not None and (min(row) < 0 or max(row) >= n):
            raise ValueError("A support is outside the terminal range")
        rows.append(row)
    if len(set(rows)) != len(rows):
        raise ValueError("A family repeats a support")
    family = tuple(sorted(rows, key=lambda s: (len(s), s)))
    if laminar:
        for a, b in combinations(map(set, family), 2):
            if a & b and not (a <= b or b <= a):
                raise ValueError("A complete fitted family is not laminar")
    return family


def bases(n):
    return {*((i,) for i in range(n)), tuple(range(n))}


def internal(point, n):
    return {tuple(sorted(C)) for C in point["supports"] if 1 < len(C) < n}


def fingerprint(point):
    rows = sorted(zip(point["supports"], point["r"], point["x"], strict=True),
                  key=lambda row: (len(row[0]), tuple(row[0])))
    serial = [(list(C), float(r).hex(), float(x).hex()) for C, r, x in rows]
    return hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()


def load_observations(path):
    with np.load(path, allow_pickle=False) as archive:
        labels = tuple(map(int, archive["labels"]))
        n = len(labels)
        p, q, voltage = [np.asarray(archive[k], float).reshape(-1, n) for k in ("P", "Q", "V")]
        root = np.asarray(archive["root"], float).reshape(-1)
    y = root[:, None]**2-voltage**2
    if not labels or len(set(labels)) != n or p.shape != q.shape or p.shape != y.shape:
        raise ValueError("Invalid observed-data shape or terminal labels")
    if not all(np.isfinite(v).all() for v in (p, q, y)):
        raise ValueError("Nonfinite observed values")
    return dict(p=p, q=q, y=y, labels=labels, n=n)


def training_fingerprint(observations):
    digest = hashlib.sha256(repr(observations["labels"]).encode())
    for name in ("p", "q", "y"):
        array = np.ascontiguousarray(observations[name], dtype=np.float64)
        digest.update(repr(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def coefficients(point, n, bound):
    family = canonical(point["supports"], n)
    if len(point["r"]) != len(family) or len(point["x"]) != len(family):
        raise ValueError("Coefficient vectors do not match supports")
    # Retain the saved order: its weights correspond to that order, not to the
    # canonical sorting used for structural comparisons.
    r, x = np.asarray(point["r"], float), np.asarray(point["x"], float)
    if not np.isfinite(r).all() or not np.isfinite(x).all():
        raise ValueError("Nonfinite response coefficients")
    violation = max(float(np.max(-r, initial=0.)), float(np.max(r-bound, initial=0.)),
                    float(np.max(-x, initial=0.)), float(np.max(x-bound, initial=0.)))
    if violation > 1e-7:
        raise ValueError("Response coefficients violate their bounds")
    R, X = np.zeros((n, n)), np.zeros((n, n))
    for C, rr, xx in zip(point["supports"], r, x, strict=True):
        indices = np.asarray(C, int)
        R[np.ix_(indices, indices)] += rr
        X[np.ix_(indices, indices)] += xx
    return family, r, x, R, X, violation


def score(observations, point, bound):
    _, _, _, R, X, _ = coefficients(point, observations["n"], bound)
    residual = observations["y"]-observations["p"]@R.T-observations["q"]@X.T
    return dict(mse=float(np.mean(residual**2)), mae=float(np.mean(abs(residual))))


def qp_certificate(observations, point, reference, bound):
    """A global fixed-family lower bound from a convex supporting plane."""
    family, r, x, _, _, violation = coefficients(point, observations["n"], bound)
    columns = []
    for power in (observations["p"], observations["q"]):
        for C in point["supports"]:
            z = np.zeros(observations["n"]); z[list(C)] = 1.
            columns.append(np.multiply.outer(power@z, z).ravel())
    F = np.column_stack(columns)/np.sqrt(reference)
    y = observations["y"].ravel()/np.sqrt(reference)
    theta = np.r_[r, x]
    residual = F.astype(np.longdouble)@theta.astype(np.longdouble)-y.astype(np.longdouble)
    objective = float(np.mean(residual**2))
    gradient = np.asarray(2*F.astype(np.longdouble).T@residual/len(y), float)
    lower = max(0., objective+float(np.minimum(-gradient*theta, gradient*(bound-theta)).sum()))
    gap = max(0., objective-lower)
    if gap > 1e-6*(1+objective)+1e-10:
        raise ValueError(f"A QP-closed point lacks the fixed-family box certificate: gap={gap:g}")
    return dict(objective=objective, lower_bound=lower, certificate_gap=gap,
                bound_violation=violation, supports=len(family), variables=len(theta))


def truth_family(runtime, meta, labels):
    """Orient the bundled physical network at its root; no tree helper import."""
    folder = local(runtime, meta["truth_network"])
    metadata = read(folder/"metadata.json"); root = int(metadata["root_bus"])
    with (folder/"buses.csv").open(encoding="utf-8", newline="") as stream:
        buses = {int(row["bus_id"]) for row in csv.DictReader(stream)}
    graph = {bus: [] for bus in buses}; edges = 0
    with (folder/"branches.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if str(row.get("status", "True")).strip().lower() not in {"true", "1", "1.0"}:
                continue
            a, b = int(row["from_bus"]), int(row["to_bus"])
            graph[a].append(b); graph[b].append(a); edges += 1
    parent, order = {root: None}, [root]
    for node in order:
        for adjacent in graph[node]:
            if adjacent == parent[node]:
                continue
            if adjacent in parent:
                raise ValueError("The physical truth network contains a cycle")
            parent[adjacent] = node; order.append(adjacent)
    if len(parent) != len(buses) or edges != len(buses)-1:
        raise ValueError("The physical truth network is not a connected tree")
    position = {label: i for i, label in enumerate(labels)}
    descendants = {bus: ({position[bus]} if bus in position else set()) for bus in buses}
    family = set()
    for node in reversed(order):
        if parent[node] is not None:
            if descendants[node]:
                family.add(tuple(sorted(descendants[node])))
            descendants[parent[node]].update(descendants[node])
    if descendants[root] != set(range(len(labels))):
        raise ValueError("The physical truth does not contain the observed terminals")
    return canonical(family | bases(len(labels)), len(labels))


def topology(point, truth, n):
    predicted, real = internal(point, n), {C for C in truth if 1 < len(C) < n}
    tp, fp, fn = len(predicted & real), len(predicted-real), len(real-predicted)
    return dict(F1=2*tp/(2*tp+fp+fn) if tp+fp+fn else 1., exact=predicted == real,
                TP=tp, FP=fp, FN=fn, predicted_clades=len(predicted), true_clades=len(real))


def direct_children(family, n):
    nodes = set(family) | bases(n)
    return {v: [s for s in nodes if set(s) < set(v)
                and not any(set(s) < set(t) < set(v) for t in nodes)] for v in nodes}


def verify_record(wrapper, common, train, validation, config, method, *, budget_tolerance=.05,
                  allow_budget_exceeded=False):
    """Replay every saved closed model and its actual MSE selection event."""
    result = wrapper["result"]; n = train["n"]; bound = float(config["response_bound"])
    reference, alpha = float(common["reference_mse"]), float(config["alpha"])
    if not reference > 0. or not math.isfinite(reference):
        raise ValueError("Invalid common reference")
    if result.get("selection_metric") != "validation_mse":
        raise ValueError("The final selection metric is not validation MSE")
    if result.get("truth_used") is not False or result.get("test_used") is not False:
        raise ValueError("Training/selection information flags are incompatible")
    if result.get("output_contraction") is not None or result.get("complete_RNJ_output_fallback") is not False:
        raise ValueError("An unrequested output mapping or RNJ fallback is active")
    effective = wrapper.get("effective_config", config)
    if wrapper.get("configured_config", config) != config:
        raise ValueError("The requested configuration differs from the frozen protocol")
    for key, value in config.items():
        if key != "total_seconds" and effective.get(key) != value:
            raise ValueError("A method changed a setting beyond its setup-time debit")
    debit = float(wrapper.get("pre_fit_debit_seconds", 0.))
    close(effective["total_seconds"], config["total_seconds"]-debit, "effective total budget", atol=1e-8)
    if result["method"] != method or result["config"] != effective:
        raise ValueError("Method/effective configuration does not match the frozen protocol")
    close(result["reference_mse"], reference, "method reference", atol=1e-16)
    close(result["common_seconds"], common["seconds"], "charged common time")
    close(wrapper["total_seconds"], wrapper["fresh_seconds"]+wrapper["common_seconds"], "wrapper timing")
    close(wrapper["common_seconds"], common["seconds"], "wrapper common time")
    if not allow_budget_exceeded and (result.get("total_cap_exceeded") or wrapper["total_seconds"] > config["total_seconds"]+budget_tolerance):
        raise ValueError("An over-budget method is labelled successful")
    checked, certificates, point_cache = {}, [], {}

    def audit_point(point, closed=True, origin="point"):
        key = fingerprint(point)
        if key not in point_cache:
            measured = score(train, point, bound)
            close(measured["mse"]/reference, point.get("objective", measured["mse"]/reference),
                  "training normalized MSE", atol=1e-9)
            if "mse" in point:
                close(measured["mse"], point["mse"], "training MSE", atol=1e-13)
            if "reference_mse" in point:
                close(point["reference_mse"], reference, "point normalization reference", atol=1e-16)
            point_cache[key] = dict(measured)
        if closed and key not in checked:
            certificate = qp_certificate(train, point, reference, bound)
            checked[key] = certificate
            certificates.append(dict(model_sha256=key, origin=origin, **certificate))
        return point_cache[key]["mse"]/reference+alpha*len(internal(point, n))

    path = result["path"]
    if not path:
        raise ValueError("A fitted method has an empty path")
    path_scores = [audit_point(p, origin=f"path:{i}") for i, p in enumerate(path)]
    if method not in {"rnj", "nj"} and set(canonical(path[0]["supports"], n)) != bases(n):
        raise ValueError("W must initialize at the singleton/full basis without fixed RNJ seeds")
    if method in {"rnj", "nj"} and (len(path) != 1 or result["events"]):
        raise ValueError("An RNJ/NJ baseline has structural path search")
    for before, after in zip(path_scores, path_scores[1:]):
        if after > before+1e-8:
            raise ValueError("The accepted path increases independently replayed J")
    events, stages, solvers, ledger = [], ["baseline" if method in {"rnj", "nj"} else "initial"], [], []
    cursor, current = 0, path[0]; visited = {canonical(current["supports"], n)}
    tail_pool = set()
    R_pool = set(canonical(common["guidance"]["pool"], n, laminar=False)) if method[0:1] == "1" else bases(n)
    order = []
    for index, event in enumerate(result["events"]):
        stage = event["stage"]; order.append(stage)
        if stage not in {"W", "T", "E"}:
            raise ValueError("Unknown structural stage")
        before_J = audit_point(current)
        if "before_score" in event:
            close(before_J, event["before_score"], "event before-J", atol=1e-8)
        proposal = event.get("point")
        if proposal is not None:
            after_J = audit_point(proposal, origin=f"{stage}:{index}")
            close(after_J, event["after_score"], "event proposed-J", atol=1e-8)
            close(event["gain"], event["before_score"]-event["after_score"], "event gain", atol=1e-12)
            gain_passed = float(event["gain"]) > float(event["numerical_threshold"])
            repeated = bool(event.get("repeated_family"))
            if repeated and (canonical(proposal["supports"], n) not in visited or event["accepted"] or not event.get("numerical_gain_passed")):
                raise ValueError("The repeated-family numerical guard is inconsistent")
            if bool(event["accepted"]) != (gain_passed and not repeated):
                raise ValueError("A structural update bypasses the common J gate")
            before_point = current
            if event["accepted"]:
                cursor += 1
                if cursor >= len(path) or fingerprint(proposal) != fingerprint(path[cursor]):
                    raise ValueError("An accepted event does not reproduce its path append")
                current = path[cursor]; stages.append(stage); visited.add(canonical(current["supports"], n))
            events.append(dict(event=index, stage=stage, accepted=bool(event["accepted"]),
                before_J=before_J, proposal_J=after_J, gain=event["gain"],
                before_point=before_point, proposal_point=proposal, after_point=current,
                zero_events=event.get("zero_events", [])))
        if event.get("raw_point") is not None:
            raw = event["raw_point"]; raw_J = audit_point(raw, closed=False)
            native_objective = event.get("solver", {}).get("objective")
            if native_objective is not None:
                close(raw_J, native_objective, "raw structural objective", atol=1e-7)
        if stage == "T" and "pool" in event:
            hierarchy = direct_children(canonical(events[-1]["before_point"]["supports"], n), n)
            if max(map(len, hierarchy.values())) > config["block_limit"]:
                raise ValueError("T was triggered with an oversized direct frontier")
            old = set(canonical(events[-1]["before_point"]["supports"], n)); generated = set()
            for row in event["frontiers"]:
                blocks = [tuple(C) for C in row["blocks"]]; sets = list(map(set, blocks))
                if not 3 <= len(blocks) <= config["block_limit"] or not all(C in old for C in blocks):
                    raise ValueError("T does not use bounded complete-subtree blocks")
                if set().union(*sets) != set(row["parent"]) or any(a & b for a, b in combinations(sets, 2)):
                    raise ValueError("A T cut is not a partition")
                for size in range(2, len(blocks)):
                    generated.update(tuple(sorted(i for block in group for i in block))
                                     for group in combinations(blocks, size))
            expected_pool = old | generated
            tail_pool = set(canonical(event["pool"], n, laminar=False))
            expected_keep = old-((old & generated)-bases(n))
            if tail_pool != expected_pool or set(canonical(event["keep"], n)) != expected_keep:
                raise ValueError("T pool/mandatory skeleton differs from the recorded frontier unions")
        if stage == "E" and "pool" in event:
            expected = R_pool | {tuple(C) for p in path[:cursor+1] for C in p["supports"]} | tail_pool | set(combinations(range(n), 2)) | bases(n)
            actual = set(canonical(event["pool"], n, laminar=False))
            if actual != expected or set(canonical(event["keep"], n)) != bases(n):
                raise ValueError("The finite E pool or revocable domain differs from its contract")
        native = event.get("solver", {})
        if native.get("solve_started"):
            upper, lower, gap = native.get("objective"), native.get("bound"), native.get("gap")
            if upper is not None and lower is not None and float(lower) > float(upper)+1e-7*(1+abs(float(upper))):
                raise ValueError("A native lower bound exceeds its feasible incumbent")
            if lower is not None and lower > before_J+1e-7*(1+abs(before_J)):
                raise ValueError("A native lower bound exceeds the recoverable current model")
            for name in ("constraint_violation", "bound_violation", "integrality_violation"):
                if native.get(name) is not None and native[name] > 1e-7:
                    raise ValueError("A native incumbent violates its declared constraints")
            solvers.append(dict(event=index, stage=stage, status=native.get("status"), objective=upper,
                bound=lower, gap=gap, gap_met=(gap is not None and gap <= config["mip_gap"]+1e-9),
                absolute_gap_met=(upper is not None and lower is not None and upper-lower <= config["mip_abs_gap"]+1e-9),
                timeout=native.get("status") == 9, seconds=native.get("seconds"),
                requested_solve_seconds=native.get("requested_solve_seconds"), scope=native.get("scope")))
    if cursor != len(path)-1 or order.count("T") > 1 or order.count("E") > 1:
        raise ValueError("Path/stage counts do not match the one-T/one-E algorithm")
    if len(result.get("MIQP", [])) != len(solvers):
        raise ValueError("Native MIQP ledger count differs from the structural event records")
    for recorded, replayed in zip(result.get("MIQP", []), solvers, strict=True):
        if recorded["status"] != replayed["status"]:
            raise ValueError("Native MIQP ledger status differs from its event")
        for key in ("objective", "bound", "gap"):
            if recorded.get(key) is None or replayed.get(key) is None:
                if recorded.get(key) != replayed.get(key):
                    raise ValueError("Native MIQP ledger availability differs from its event")
            else:
                close(recorded[key], replayed[key], "native MIQP ledger "+key)
    if "T" in order and "W" in order[order.index("T")+1:] or "E" in order and any(s != "E" for s in order[order.index("E")+1:]):
        raise ValueError("The algorithm returned to an earlier search stage")
    if method not in {"rnj", "nj"}:
        if (not int(method[1]) and "T" in order) or (not int(method[2]) and "E" in order):
            raise ValueError("A disabled ablation module ran")
        if bool(int(method[2])) != (order.count("E") == 1):
            raise ValueError("An enabled E is missing its solve/skip event")
    candidates = result["frozen_candidates"]
    expected_count = len(path)+(method not in {"rnj", "nj"})
    if len(candidates) != expected_count:
        raise ValueError("The output library is not exactly own path plus its declared baseline")
    for index, row in enumerate(candidates):
        point = row["point"]; audit_point(point, origin=f"candidate:{index}")
        if row["index"] != index or tuple(row["labels"]) != train["labels"] or row.get("frozen") is not True:
            raise ValueError("Candidate order, terminal order or coefficient-freeze flag changed")
        if index < len(path):
            if fingerprint(point) != fingerprint(path[index]):
                raise ValueError("A path candidate changed its fitted coefficients")
        elif set(canonical(point["supports"], n)) != {(i,) for i in range(n)}:
            raise ValueError("The extra output candidate is not singleton-only")
        _, _, _, R, X, _ = coefficients(point, n, bound)
        if not np.allclose(row["R"], R, atol=1e-11, rtol=1e-10) or not np.allclose(row["X"], X, atol=1e-11, rtol=1e-10):
            raise ValueError("A frozen candidate response matrix is inconsistent")
        measured = score(validation, point, bound)
        close(measured["mse"], row["validation_mse"], "candidate validation MSE", atol=1e-13)
        close(measured["mse"]/reference, row["validation_scaled_mse"], "candidate normalized validation MSE")
        close(measured["mae"], row["validation_mae"], "diagnostic validation MAE", atol=1e-12)
        if len(internal(point, n)) != row["internal"]:
            raise ValueError("Candidate complexity is inconsistent")
    values = [score(validation, row["point"], bound)["mse"]/reference for row in candidates]
    best = min(values)
    eligible = [i for i, value in enumerate(values) if value <= best+config["validation_tie"]]
    selected_index = min(eligible, key=lambda i: (len(internal(candidates[i]["point"], n)), values[i], i))
    selected = result["selected"]
    if selected["index"] != selected_index or fingerprint(selected["point"]) != fingerprint(candidates[selected_index]["point"]):
        raise ValueError("The final model was not selected by normalized validation MSE")
    mse_view = result["selected_mse_view"]
    if mse_view["index"] != selected_index or fingerprint(mse_view["point"]) != fingerprint(selected["point"]):
        raise ValueError("The saved MSE alias differs from the actual final selection")
    if not result.get("QP"):
        raise ValueError("A fitted method is missing its fixed-family QP ledger")
    for index, row in enumerate(result.get("QP", [])):
        objective, lower = float(row["objective"]), float(row["lower_bound"])
        if row["status"] <= 0 or lower > objective+1e-9 or row.get("soft_cap_exceeded"):
            raise ValueError("A fixed-family QP ledger contains failure or exceeded its soft cap")
        close(row["certificate_gap"], objective-lower, "QP ledger gap", atol=1e-10)
        if objective-lower > 1e-6*(1+objective)+1e-9:
            raise ValueError("A QP ledger certificate is too loose")
        ledger.append(dict(index=index, objective=objective, lower_bound=lower,
                           certificate_gap=objective-lower, seconds=row["seconds"], status=row["status"]))
    stage = stages[selected_index] if selected_index < len(stages) else "singleton"
    return dict(result=result, events=events, solvers=solvers, certificates=certificates, QP=ledger,
                selected_stage=stage, path_J=path_scores, normalized_validation_mse=values[selected_index],
                MAE_diagnostic_differs=(result.get("selected_mae_view", selected)["index"] != selected_index))


def grouped(rows, fields):
    groups = {}
    for row in rows:
        key = tuple(row.get(name) for name in fields)
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in groups.items():
        valid = [r for r in group if r["status"] == "verified"]
        known = [r for r in group if r.get("total_seconds") is not None]
        result.append(dict(zip(fields, key), planned=len(group), verified=len(valid),
            failed=sum(r["status"] in {"failed", "common_failed"} for r in group),
            invalid=sum(r["status"] == "invalid" for r in group),
            interrupted=sum(r["status"] == "interrupted" for r in group),
            budget_exceeded=sum(r["status"] == "budget_exceeded" for r in group),
            pending=sum(r["status"] == "pending" for r in group),
            F1_verified_mean=float(np.mean([r["F1"] for r in valid])) if valid else None,
            exact=sum(bool(r["exact"]) for r in valid),
            exact_rate_planned=sum(bool(r["exact"]) for r in valid)/len(group),
            exact_rate_verified=sum(bool(r["exact"]) for r in valid)/len(valid) if valid else None,
            total_seconds_recorded=sum(r["total_seconds"] for r in known),
            timing_unknown=len(group)-len(known),
            total_seconds_verified=sum(r["total_seconds"] for r in valid),
            runtime_comparison_complete=len(valid) == len(group) and len(known) == len(group),
            optimizer_ledger_unavailable=sum(not r.get("optimizer_ledger_available", False) for r in group),
            native_timeouts=sum(r.get("native_timeouts", 0) for r in group)))
    return result


def pair_statistics(rows, methods=METHODS):
    bank = {(r["job"], r["method"]): r for r in rows}; jobs = list(dict.fromkeys(r["job"] for r in rows))
    output, pairs = [], []
    for old_method, new_method in combinations(methods, 2):
        paired = []
        for job in jobs:
            old, new = bank[job, old_method], bank[job, new_method]
            valid = old["status"] == new["status"] == "verified"
            row = dict(job=job, seed_index=old["seed_index"], old=old_method, new=new_method,
                valid=valid, delta_F1=new["F1"]-old["F1"] if valid else None,
                delta_exact=int(new["exact"])-int(old["exact"]) if valid else None,
                delta_seconds=new["total_seconds"]-old["total_seconds"] if valid else None)
            pairs.append(row)
            if valid:
                paired.append(row)
        output.append(dict(old=old_method, new=new_method, planned=len(jobs), paired=len(paired),
            wins=sum(r["delta_F1"] > 1e-12 for r in paired), ties=sum(abs(r["delta_F1"]) <= 1e-12 for r in paired),
            losses=sum(r["delta_F1"] < -1e-12 for r in paired),
            new_exact=sum(r["delta_exact"] > 0 for r in paired), lost_exact=sum(r["delta_exact"] < 0 for r in paired),
            delta_F1_mean=float(np.mean([r["delta_F1"] for r in paired])) if paired else None,
            delta_exact_sum=sum(r["delta_exact"] for r in paired),
            delta_seconds_sum=sum(r["delta_seconds"] for r in paired)))
    return output, pairs


def factorial_statistics(rows):
    bank = {(r["job"], r["method"]): r for r in rows}; jobs = list(dict.fromkeys(r["job"] for r in rows))
    effects, interactions, orders, shapley, checks = [], [], [], [], []
    for job in jobs:
        cell = {method: bank[job, method] for method in FACTORIAL}
        marginal = {}
        for bit, module in enumerate("RTE"):
            weighted = []
            for off in FACTORIAL:
                if off[bit] != "0":
                    continue
                on = off[:bit]+"1"+off[bit+1:]
                valid = cell[off]["status"] == cell[on]["status"] == "verified"
                effect = dict(job=job, seed_index=cell[off]["seed_index"], module=module, off=off, on=on, valid=valid,
                    delta_F1=cell[on]["F1"]-cell[off]["F1"] if valid else None,
                    delta_exact=int(cell[on]["exact"])-int(cell[off]["exact"]) if valid else None,
                    delta_seconds=cell[on]["total_seconds"]-cell[off]["total_seconds"] if valid else None)
                effects.append(effect); marginal[off, on] = effect
                if valid:
                    weighted.append((1/6 if off.count("1") == 1 else 1/3, effect))
            shapley.append(dict(job=job, seed_index=cell["000"]["seed_index"], module=module, valid=len(weighted) == 4,
                **{key: sum(w*r[key] for w, r in weighted) if len(weighted) == 4 else None
                   for key in ("delta_F1", "delta_exact", "delta_seconds")}))
        for order in permutations("RTE"):
            off, chain = "000", []
            for step, module in enumerate(order, 1):
                bit = "RTE".index(module); on = off[:bit]+"1"+off[bit+1:]
                effect = marginal[off, on]; orders.append(dict(order="".join(order), step=step, **effect)); chain.append(effect)
                off = on
            if all(e["valid"] for e in chain):
                for key, source in (("delta_F1", "F1"), ("delta_exact", "exact"), ("delta_seconds", "total_seconds")):
                    close(sum(e[key] for e in chain), float(cell["111"][source])-float(cell["000"][source]), "factorial telescoping", atol=1e-7)
                checks.append(dict(job=job, order="".join(order), passed=True))
        for a, b in combinations(range(3), 2):
            condition = next(i for i in range(3) if i not in (a, b))
            for value in (0, 1):
                terms = []
                for aa, bb, sign in ((0, 0, 1), (1, 0, -1), (0, 1, -1), (1, 1, 1)):
                    bits = ["0"]*3; bits[a], bits[b], bits[condition] = str(aa), str(bb), str(value)
                    terms.append(("".join(bits), sign))
                valid = all(cell[m]["status"] == "verified" for m, _ in terms)
                interactions.append(dict(job=job, seed_index=cell["000"]["seed_index"], interaction="RTE"[a]+"RTE"[b],
                    condition_module="RTE"[condition], condition_value=value, valid=valid,
                    **{key: sum(sign*float(cell[m][source]) for m, sign in terms) if valid else None
                       for key, source in (("delta_F1", "F1"), ("delta_exact", "exact"), ("delta_seconds", "total_seconds"))}))
        terms = [(m, 1 if m.count("1") % 2 else -1) for m in FACTORIAL]
        valid = all(cell[m]["status"] == "verified" for m, _ in terms)
        interactions.append(dict(job=job, seed_index=cell["000"]["seed_index"], interaction="RTE",
            condition_module=None, condition_value=None, valid=valid,
            **{key: sum(sign*float(cell[m][source]) for m, sign in terms) if valid else None
               for key, source in (("delta_F1", "F1"), ("delta_exact", "exact"), ("delta_seconds", "total_seconds"))}))
    return effects, interactions, orders, shapley, checks


def delta_summary(rows, fields):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row.get(f) for f in fields), []).append(row)
    result = []
    for key, group in groups.items():
        valid = [r for r in group if r["valid"]]
        result.append(dict(zip(fields, key), planned=len(group), paired=len(valid),
            **{metric+"_mean": float(np.mean([r[metric] for r in valid])) if valid else None
               for metric in ("delta_F1", "delta_exact", "delta_seconds")},
            F1_positive=sum(r["delta_F1"] > 1e-12 for r in valid),
            F1_negative=sum(r["delta_F1"] < -1e-12 for r in valid)))
    return result


def cluster_sensitivity(rows, draws):
    rng = np.random.default_rng(BOOTSTRAP_SEED); output = []
    shared_draws = rng.integers(0, 4, (draws, 4)) if draws else None
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        valid = len(group) == 48 and all(r["status"] == "verified" for r in group)
        row = dict(method=method, clusters=4, draws=draws, available=valid and draws > 0,
                   interpretation="four shared seed clusters; descriptive resampling sensitivity, not calibrated topology probability")
        if valid and draws > 0:
            means = np.array([[np.mean([r[metric] for r in group if r["seed_index"] == seed])
                               for metric in ("F1", "exact")] for seed in range(4)])
            sampled = means[shared_draws].mean(axis=1)
            for index, metric in enumerate(("F1", "exact_rate")):
                low, high = np.percentile(sampled[:, index], (2.5, 97.5))
                row.update({metric+"_sensitivity_low": float(low), metric+"_sensitivity_high": float(high)})
        output.append(row)
    return output


def diagnostic_summaries(wrong_clades, coverage, events):
    sizes, covered, stages = {}, {}, {}
    for row in wrong_clades:
        sizes.setdefault((row["method"], row["kind"], row["size"]), []).append(row)
    for row in coverage:
        covered.setdefault(row["method"], []).append(row)
    for row in events:
        stages.setdefault((row["method"], row["stage"]), []).append(row)
    size_rows = [dict(method=method, kind=kind, size=size, erroneous_clades=len(group),
                     inputs_with_error=len({r["job"] for r in group}))
                 for (method, kind, size), group in sorted(sizes.items())]
    coverage_rows = []
    for method, group in covered.items():
        E = [r for r in group if r["E_pool_contains_all_true_connections"] is not None]
        coverage_rows.append(dict(method=method, planned=48, verified=len(group),
            own_path_contains_true=sum(r["own_path_contains_true"] for r in group),
            output_bank_contains_true=sum(r["output_bank_contains_true"] for r in group),
            selected_exact=sum(r["selected_exact"] for r in group), ranking_gaps=sum(r["ranking_gap"] for r in group),
            E_pools_available=len(E), E_pools_cover_all_true_connections=sum(r["E_pool_contains_all_true_connections"] for r in E)))
    stage_rows = [dict(method=method, stage=stage, calls=len(group), accepted=sum(r["accepted"] for r in group),
        accepted_F1_positive=sum(r["accepted"] and r["actual_F1_change"] > 1e-12 for r in group),
        accepted_F1_negative=sum(r["accepted"] and r["actual_F1_change"] < -1e-12 for r in group),
        adopted_zero_deletions=sum(r["adopted_zero_deletions"] for r in group))
        for (method, stage), group in stages.items()]
    return size_rows, coverage_rows, stage_rows


def analyze(output, *, bootstrap=2000, allow_incomplete=False):
    started = perf_counter(); output = Path(output).resolve(); runtime = (output/"runtime").resolve()
    protocol = read(output/"protocol.json"); catalog = read(runtime/"data/catalog.json")
    jobs, methods, config = list(protocol["jobs"]), tuple(protocol["methods"]), protocol["config"]
    if len(jobs) != 48 or set(methods) != set(METHODS) or len(methods) != 10:
        raise ValueError("This frozen analysis expects exactly 48 inputs and ten methods")
    conditions = {(catalog[j]["case"], int(catalog[j]["seed_index"]), float(catalog[j]["voltage_sigma"])) for j in jobs}
    cases = {condition[0] for condition in conditions}
    expected_conditions = {(case, seed, sigma) for case in cases for seed in range(4)
                           for sigma in (.0002, .0005, .001)}
    if len(cases) != 4 or len(conditions) != 48 or conditions != expected_conditions:
        raise ValueError("The input grid is not four cases x four shared seeds x three noise levels")
    target = output/"analysis"; target.mkdir(parents=True, exist_ok=True)
    inventory = []
    for job in jobs:
        for method in METHODS:
            file = output/"runs"/job/method/"result.json"
            wrapper = read(file) if file.exists() else None
            inventory.append(dict(job=job, method=method, file=file, wrapper=wrapper,
                                  status=wrapper.get("status", "pending") if wrapper else "pending"))
    pending = [r for r in inventory if r["status"] not in TERMINAL]
    if pending:
        state = dict(complete=False, planned=480, pending=len(pending),
                     terminal=480-len(pending), truth_test_opened=False,
                     pending_runs=[dict(job=r["job"], method=r["method"], status=r["status"]) for r in pending])
        write_json(target/"incomplete.json", state)
        if not allow_incomplete:
            raise RuntimeError("Wait for every method to reach a terminal state before test/truth analysis")
        return state
    own = Path(__file__).resolve()
    own_hash = sha(own); analysis_source = output/"analysis_source"/own_hash/"analyze_full.py"
    analysis_source.parent.mkdir(parents=True, exist_ok=True)
    if analysis_source.exists() and sha(analysis_source) != own_hash:
        raise ValueError("An archived independent-analysis source changed")
    analysis_source.write_bytes(own.read_bytes())
    if sha(output/"catalog.json") != protocol["catalog_hash"] or read(output/"catalog.json") != catalog:
        raise ValueError("The frozen data catalog changed")
    for relative, expected in protocol["runtime_file_hashes"].items():
        if sha(local(runtime, relative)) != expected:
            raise ValueError("A frozen runtime/data file changed: "+relative)
    rows, audited, common_cache, observations = [], {}, {}, {}
    native_rows, qp_rows, certificate_rows, error_rows = [], [], [], []
    result_hashes, common_hashes, input_hashes = {}, {}, {}
    for job in jobs:
        meta = catalog[job]; common_file = output/"common"/job/"common.json"
        common = read(common_file); common_cache[job] = common; common_hashes[job] = sha(common_file)
        splits = {name: load_observations(local(runtime, meta[name])) for name in ("train", "validation")}
        if splits["train"]["labels"] != splits["validation"]["labels"] or tuple(meta["labels"]) != splits["train"]["labels"]:
            raise ValueError("Train/validation labels are not aligned")
        for name, obs in splits.items():
            expected_points = meta.get("split_points", {}).get(name)
            if expected_points is not None and len(obs["y"]) != expected_points:
                raise ValueError("An observed split lost declared time points")
        observations[job] = splits
        if common.get("status", "complete") == "complete":
            if common["n"] != splits["train"]["n"] or tuple(common["labels"]) != splits["train"]["labels"]:
                raise ValueError("Common terminal metadata is not aligned")
            if common.get("truth_used") is not False or common.get("validation_used") is not False:
                raise ValueError("Common preprocessing uses forbidden information")
            if common.get("training_sha256") != training_fingerprint(splits["train"]):
                raise ValueError("A common calibration belongs to other training observations")
            measured = score(splits["train"], common["reference_point"], config["response_bound"])["mse"]
            close(measured, common["reference_mse"], "common reference MSE", atol=1e-13)
            qp_certificate(splits["train"], common["reference_point"], 1., config["response_bound"])
        for split in ("train", "validation"):
            expected = protocol["input_hashes"][job][split]
            if sha(local(runtime, meta[split])) != expected:
                raise ValueError("A frozen observed input changed")
        input_hashes[job] = {s: protocol["input_hashes"][job][s] for s in ("train", "validation", "test")}
    for item in inventory:
        job, method, wrapper = item["job"], item["method"], item["wrapper"]; meta = catalog[job]
        row = dict(job=job, method=method, case=meta["case"], seed_index=int(meta["seed_index"]),
                   voltage_sigma=float(meta["voltage_sigma"]), status=item["status"],
                   returned=wrapper is not None, F1=None, exact=None, total_seconds=None,
                   fresh_seconds=None, common_seconds=None, native_timeouts=0,
                   optimizer_ledger_available=False, model_audit_error=None)
        if wrapper is not None:
            result_hashes[job+"/"+method] = sha(item["file"])
            for name in ("total_seconds", "fresh_seconds", "common_seconds"):
                if wrapper.get(name) is not None:
                    value = float(wrapper[name])
                    if not math.isfinite(value) or value < 0.:
                        raise ValueError("A runtime cost is negative or nonfinite")
                    row[name] = value
            if (wrapper.get("job") != job or wrapper.get("method") != method
                    or wrapper.get("protocol_sha256") != sha(output/"protocol.json")):
                row.update(status="invalid", error="result provenance mismatch")
                error_rows.append(dict(job=job, method=method, error="result provenance mismatch", traceback=None))
            elif item["status"] in MODEL_STATUSES or item["status"] == "budget_exceeded" and "result" in wrapper:
                try:
                    if wrapper.get("common_sha256") != common_hashes[job]:
                        raise ValueError("Method common-preprocessing provenance changed")
                    if wrapper.get("truth_used") is not False or wrapper.get("test_opened") is not False:
                        raise ValueError("The native worker opened forbidden evaluation information")
                    for phase in ("runtime_sources_before", "runtime_sources_after"):
                        if not {"algorithm", "optimization", "trees", "data"} <= set(wrapper[phase]):
                            raise ValueError("The loaded-source receipt omits an expected native module")
                        for source in wrapper[phase].values():
                            if source["sha256"] != protocol["source_hashes"].get(source["relative_path"]):
                                raise ValueError("A loaded native source differs from the frozen manifest")
                    if wrapper.get("train_val_input_hashes") != {s: input_hashes[job][s] for s in ("train", "validation")}:
                        raise ValueError("Method train/validation provenance changed")
                    checked = verify_record(wrapper, common_cache[job], observations[job]["train"],
                                            observations[job]["validation"], config, method,
                                            budget_tolerance=protocol.get("budget_tolerance_seconds", .05),
                                            allow_budget_exceeded=item["status"] == "budget_exceeded")
                    audited[job, method] = checked
                    row.update(status="budget_exceeded" if item["status"] == "budget_exceeded" else "verified",
                               selected_stage=checked["selected_stage"], optimizer_ledger_available=True,
                               MAE_diagnostic_differs=checked["MAE_diagnostic_differs"],
                               native_timeouts=sum(r["timeout"] for r in checked["solvers"]))
                    native_rows.extend(dict(job=job, method=method, **r) for r in checked["solvers"])
                    qp_rows.extend(dict(job=job, method=method, **r) for r in checked["QP"])
                    certificate_rows.extend(dict(job=job, method=method, **r) for r in checked["certificates"])
                except Exception as exc:
                    row.update(status="budget_exceeded" if item["status"] == "budget_exceeded" else "invalid",
                               error=str(exc), model_audit_error=str(exc))
                    error_rows.append(dict(job=job, method=method, error=str(exc), traceback=traceback.format_exc()))
            elif wrapper.get("reason") == "common_preprocessing_unavailable":
                row["status"] = "common_failed"
        rows.append(row)
    # Every generation/selection check above uses training and validation only.
    # Test observations and physical network files are opened from here onward.
    event_rows, coverage_rows, wrong_clades = [], [], []
    for job in jobs:
        meta, splits = catalog[job], observations[job]
        splits["test"] = load_observations(local(runtime, meta["test"]))
        if splits["test"]["labels"] != splits["train"]["labels"]:
            raise ValueError("The independent test terminal order changed")
        expected_points = meta.get("split_points", {}).get("test")
        if expected_points is not None and len(splits["test"]["y"]) != expected_points:
            raise ValueError("The independent test lost declared time points")
        if sha(local(runtime, meta["test"])) != protocol["input_hashes"][job]["test"]:
            raise ValueError("The frozen independent test changed")
        truth = truth_family(runtime, meta, splits["train"]["labels"]); n = splits["train"]["n"]
        real = {C for C in truth if 1 < len(C) < n}
        for row in (r for r in rows if r["job"] == job and r["status"] == "verified"):
            method = row["method"]; check = audited[job, method]; result = check["result"]
            selected = result["selected"]["point"]
            row.update(topology(selected, truth, n))
            for split, obs in splits.items():
                row.update({split+"_"+name: value for name, value in score(obs, selected, config["response_bound"]).items()})
            row["zero_proper_supports"] = sum(1 < len(C) < n and max(r, x) <= config["zero_tol"]
                for C, r, x in zip(selected["supports"], selected["r"], selected["x"], strict=True))
            for kind, supports in (("FP", internal(selected, n)-real), ("FN", real-internal(selected, n))):
                wrong_clades.extend(dict(job=job, method=method, kind=kind, size=len(C), indices=json.dumps(C),
                    terminal_labels=json.dumps([splits["train"]["labels"][i] for i in C])) for C in sorted(supports))
            raw_true = [i for i, p in enumerate(result["path"]) if internal(p, n) == real]
            bank_true = [r["index"] for r in result["frozen_candidates"] if internal(r["point"], n) == real]
            actual_E = next((e for e in result["events"] if e["stage"] == "E" and "pool" in e), None)
            coverage_rows.append(dict(job=job, method=method, own_path_contains_true=bool(raw_true),
                output_bank_contains_true=bool(bank_true), selected_exact=row["exact"],
                ranking_gap=bool(bank_true) and not row["exact"], true_path_indices=json.dumps(raw_true),
                true_output_indices=json.dumps(bank_true), selected_stage=row["selected_stage"],
                E_pool_contains_all_true_connections=(real <= set(map(tuple, actual_E["pool"]))) if actual_E else None,
                E_pool_missing_true_connections=(len(real-set(map(tuple, actual_E["pool"])))) if actual_E else None))
            for event in check["events"]:
                before = topology(event["before_point"], truth, n)
                proposal = topology(event["proposal_point"], truth, n)
                after = topology(event["after_point"], truth, n)
                event_rows.append(dict(job=job, method=method, event=event["event"], stage=event["stage"],
                    accepted=event["accepted"], before_J=event["before_J"], proposal_J=event["proposal_J"],
                    J_gain=event["gain"], before_F1=before["F1"], proposal_F1=proposal["F1"],
                    actual_after_F1=after["F1"], proposal_F1_change=proposal["F1"]-before["F1"],
                    actual_F1_change=after["F1"]-before["F1"],
                    zero_deletion_proposals=sum(len(z["removed"]) for z in event["zero_events"]),
                    adopted_zero_deletions=sum(len(z["removed"]) for z in event["zero_events"] if z["accepted"]) if event["accepted"] else 0))
    pair_summary, pair_rows = pair_statistics(rows)
    effects, interactions, orders, shapley, telescoping = factorial_statistics(rows)
    size_summary, coverage_summary, stage_summary = diagnostic_summaries(wrong_clades, coverage_rows, event_rows)
    summary = grouped(rows, ("method",))
    groups = {
        "by_case.csv": ("method", "case"), "by_noise.csv": ("method", "voltage_sigma"),
        "by_seed.csv": ("method", "seed_index"), "by_case_noise.csv": ("method", "case", "voltage_sigma"),
        "by_case_noise_seed.csv": ("method", "case", "voltage_sigma", "seed_index")}
    tables = dict(
        per_model=rows, summary=summary, main_methods=[r for r in summary if r["method"] in {"rnj", "nj", "000", "100", "111"}],
        paired_summary=pair_summary, paired_inputs=pair_rows, conditional_marginals=effects,
        conditional_marginals_summary=delta_summary(effects, ("module", "off", "on")),
        interactions=interactions, interactions_summary=delta_summary(interactions, ("interaction", "condition_module", "condition_value")),
        module_addition_orders=orders, shapley=shapley, shapley_summary=delta_summary(shapley, ("module",)),
        solver=native_rows, QP_ledger=qp_rows, independent_QP_certificates=certificate_rows,
        events=event_rows, coverage=coverage_rows, wrong_clades=wrong_clades, failures=error_rows,
        errors_by_size=size_summary, coverage_summary=coverage_summary, stages=stage_summary,
        four_cluster_sensitivity=cluster_sensitivity(rows, bootstrap))
    for name, table in tables.items():
        write_csv(target/(name+".csv"), table)
    for name, fields in groups.items():
        write_csv(target/name, grouped(rows, fields))
    physical_common = sum(float(c.get("seconds", c.get("fresh_seconds", 0.))) for c in common_cache.values())
    physical_prepare = sum(float(c.get("prepare_seconds", 0.)) for c in common_cache.values())
    logical_common = sum(r["common_seconds"] for r in rows if r["common_seconds"] is not None)
    fresh = sum(r["fresh_seconds"] for r in rows if r["fresh_seconds"] is not None)
    completion = read(output/"completion.json") if (output/"completion.json").exists() else {}
    timing = dict(logical_method_seconds_recorded=sum(r["total_seconds"] for r in rows if r["total_seconds"] is not None),
        logical_common_charges=logical_common, sum_method_worker_seconds=fresh,
        physical_common_worker_seconds=physical_common, physical_common_native_prepare_seconds=physical_prepare,
        physical_common_native_prepare_unknown=sum("prepare_seconds" not in c for c in common_cache.values()),
        physical_process_wall_effort_recorded=fresh+physical_common,
        native_batch_calendar_seconds=completion.get("manager_elapsed_seconds_all_invocations", completion.get("manager_elapsed_seconds")),
        method_timing_unknown=sum(r["total_seconds"] is None for r in rows),
        scope="logical methods each charge one common; actual shared common is physically performed once per input; sums are process wall effort, not CPU seconds")
    write_json(target/"timing.json", timing)
    for name in ("gurobipy", "algorithm", "optimization", "trees"):
        if name in sys.modules or any(module.startswith(name+".") for module in sys.modules):
            raise RuntimeError("The independent analysis imported a native optimizer/helper: "+name)
    if sha(own) != own_hash:
        raise RuntimeError("The independent analyzer source changed while it was running")
    artifacts = {p.name: sha(p) for p in target.glob("*.csv")}
    artifacts["timing.json"] = sha(target/"timing.json")
    passed = not error_rows and all(r["status"] != "invalid" for r in rows)
    verification = dict(passed=passed, complete=True, planned=480, returned=sum(r["returned"] for r in rows),
        native_queue_finished=True, all_models_verified=all(r["status"] == "verified" for r in rows),
        verified=sum(r["status"] == "verified" for r in rows),
        failed=sum(r["status"] in {"failed", "common_failed"} for r in rows),
        interrupted=sum(r["status"] == "interrupted" for r in rows),
        budget_exceeded=sum(r["status"] == "budget_exceeded" for r in rows),
        invalid=sum(r["status"] == "invalid" for r in rows), pending=0,
        numerically_invalid_returned_results=len(error_rows),
        all_planned_denominators_preserved=True, selection_metric="normalized validation MSE",
        MAE_used_for_final_selection=False, native_optimizer_imported=False,
        independent_QP_certificates=len(certificate_rows), native_MIQP_records=len(native_rows),
        native_optimizer_ledgers_unavailable=sum(not r["optimizer_ledger_available"] for r in rows),
        native_timeouts=sum(r["timeout"] for r in native_rows), all_native_gap_met=all(r["gap_met"] or r["absolute_gap_met"] for r in native_rows),
        main_methods=("rnj", "nj", "000", "100", "111"), clusters=4, bootstrap_draws=bootstrap,
        resampling_interpretation="four shared physical seed groups, not 48 independent repetitions; percentile ranges are sensitivity summaries",
        module_effect_scope="conditional factorial comparisons include the declared E-on/E-off budget allocation",
        tree_optimality_claimed=False, candidate_pool_coverage_is_not_saved_true_model_coverage=True,
        truth_test_opened_only_after_all_native_terminal=True, protocol_sha256=sha(output/"protocol.json"),
        analyzer_sha256=own_hash, analyzer_source_archive=str(analysis_source.relative_to(output)),
        analyzer_pre_run_frozen=protocol["runtime_file_hashes"].get("tools/analyze_full.py") == own_hash,
        runtime_file_hashes=protocol["runtime_file_hashes"],
        native_result_sha256=result_hashes, common_sha256=common_hashes, input_hashes=input_hashes,
        factorial_telescoping_checks=len(telescoping), artifacts=artifacts, errors=error_rows,
        seconds=perf_counter()-started)
    write_json(target/"verification.json", verification)
    write_json(target/"summary.json", dict(methods=summary, timing=timing, verification=verification))
    print(json.dumps(dict(verified=verification["verified"], failed=verification["failed"], invalid=verification["invalid"],
                          budget_exceeded=verification["budget_exceeded"], methods=summary), ensure_ascii=False, indent=2))
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--allow-incomplete", action="store_true", help="write status only until every native run is terminal")
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")
    outcome = analyze(args.output, bootstrap=args.bootstrap, allow_incomplete=args.allow_incomplete)
    raise SystemExit(0 if outcome.get("passed") else 2)
