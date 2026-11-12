"""Post-seal true-family QP diagnostic; never part of topology selection.

One frozen-runtime QPFitter call per available input, on its original train64.
Selection32 is scored with frozen coefficients. No MIQP or extra day is used.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np


TERMINAL = {"fitted", "failed", "interrupted", "budget_exceeded"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, relative):
    path = (root/relative).resolve()
    require(not Path(relative).is_absolute() and path.is_relative_to(root), "A diagnostic path escapes the study")
    return path


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def verify_study(output):
    """Verify every planned terminal byte before reading any true-family JSON."""
    protocol, seal, completion = [read(output/name) for name in ("protocol.json", "selection_seal.json", "completion.json")]
    require(seal.get("sealed") and seal.get("all_planned_terminal") and completion.get("all_planned_terminal"),
            "True-family diagnostics require all-terminal sealed results")
    require(seal["protocol_sha256"] == sha(output/"protocol.json") == sha(output/"runtime/protocol.json"), "Protocol SHA mismatch")
    require(completion["selection_seal_sha256"] == sha(output/"selection_seal.json"), "Completion seal SHA mismatch")
    require(protocol["split"] == dict(total=96, train=[0, 64], selection=[64, 96], shuffle=False), "Only strict96 64/32 studies are supported")
    require(protocol["config"]["response_bound"] == 2. and protocol["config"]["qp_seconds"] == 300., "Expected the frozen [0,2]/300s QP contract")
    require(protocol["planned_inputs"] == len(protocol["inputs"]) and
            protocol["planned_methods"] == len(protocol["inputs"])*len(protocol["methods"]), "Planned denominator changed")
    require(seal["source_sha256"] == protocol["source_sha256"], "Sealed source inventory changed")
    runtime = output/"runtime"; consumed = {}
    def check(path, expected):
        require(sha(path) == expected, "Frozen bytes changed: "+str(path))
        consumed[path.relative_to(output).as_posix()] = expected
    for name in ("protocol.json", "selection_seal.json", "completion.json"):
        consumed[name] = sha(output/name)
    consumed["runtime/protocol.json"] = sha(runtime/"protocol.json")
    for name, expected in protocol["source_sha256"].items():
        check(inside(runtime, name), expected)
    require({"data.py", "optimization.py", "trees.py"} <= set(protocol["source_sha256"]), "Native fixed-QP sources are missing")
    expected_methods = {f"runs/{job}/{method}/result.json" for job in protocol["inputs"] for method in protocol["methods"]}
    expected_common = {f"common/{job}/common.json" for job in protocol["inputs"]}
    require(set(seal["result_sha256"]) == expected_methods and set(seal["common_sha256"]) == expected_common, "Seal omits planned tasks")
    for inventory in (seal["result_sha256"], seal["common_sha256"]):
        for name, expected in inventory.items():
            path = inside(output, name); check(path, expected); record = read(path)
            require(record["status"] in TERMINAL and record["protocol_sha256"] == seal["protocol_sha256"], "A task is nonterminal or foreign")
    for job, info in protocol["inputs"].items():
        if not info.get("generation_failed"):
            check(inside(runtime, info["path"]), info["sha256"])
    # Truth exposure is an earlier, separate post-seal step. The diagnostic
    # records these exact JSON bytes; it does not recompute or substitute truth.
    for job, info in protocol["inputs"].items():
        if not info.get("generation_failed"):
            path = output/"evaluation"/(job+".json")
            consumed[path.relative_to(output).as_posix()] = sha(path)
    return protocol, seal, consumed


def input_rows(path, info):
    """Independent sole96 reader; return copies without a full-day array base."""
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {"P", "Q", "V", "root", "labels"}, "Unexpected observed fields")
        labels = tuple(map(int, archive["labels"]))
        p, q, voltage = [np.asarray(archive[key], float) for key in ("P", "Q", "V")]
        root = np.asarray(archive["root"], float)
    require(list(labels) == info["labels"] and p.shape == q.shape == voltage.shape == (96, len(labels)) and root.shape == (96,), "Not one aligned 96-point input")
    y = root[:, None]**2-voltage**2
    require(all(np.isfinite(a).all() for a in (p, q, y)), "Nonfinite observations")
    rows = []
    for start, stop, key in ((0, 64, "training_sha256"), (64, 96, "selection_sha256")):
        values = dict(p=p[start:stop].copy(), q=q[start:stop].copy(), y=y[start:stop].copy(), labels=labels)
        digest = hashlib.sha256(repr(labels).encode())
        for name in ("p", "q", "y"):
            a = np.ascontiguousarray(values[name], dtype=np.float64)
            digest.update(repr(a.shape).encode()); digest.update(a.tobytes())
            values[name].flags.writeable = False
        require(digest.hexdigest() == info[key], "Chronological split fingerprint changed")
        rows.append(values)
    return rows


def native_modules(runtime, source_hashes):
    sys.path.insert(0, str(runtime))
    modules = [importlib.import_module(name) for name in ("data", "optimization", "trees")]
    for module in modules:
        path = Path(module.__file__).resolve()
        require(path.is_relative_to(runtime) and sha(path) == source_hashes[path.relative_to(runtime).as_posix()],
                "Fixed-QP import escaped the frozen runtime")
    require("gurobipy" not in sys.modules, "This diagnostic must not import the MIQP backend")
    return modules


def read_guard(output, destination, own, allowed_npz):
    allowed = {p.resolve() for p in allowed_npz}
    project = Path(read(output/"protocol.json")["project_root"]).resolve()
    dependencies = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    opened = []
    def audit(event, args):
        if event != "open" or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve(); mode, flags = args[1:3]
        reading = ("r" in mode or "+" in mode) if isinstance(mode, str) else (flags & (os.O_WRONLY | os.O_RDWR)) != os.O_WRONLY
        if not reading:
            return
        if path.suffix.lower() == ".npz" and path not in allowed:
            raise PermissionError("An extra observation file was blocked in true-family diagnosis")
        if path.suffix.lower() == ".npz":
            opened.append(str(path))
        dependency = any(path.is_relative_to(base) for base in dependencies)
        if path.is_relative_to(project) and not dependency and not path.is_relative_to(output) and path != own:
            raise PermissionError("A parent-project read was blocked in true-family diagnosis")
    sys.addaudithook(audit)
    return opened


def point_summary(point, n, zero, bound):
    supports = [tuple(C) for C in point["supports"]]
    r, x = np.asarray(point["r"], float), np.asarray(point["x"], float)
    require(len(supports) == len(r) == len(x) and np.isfinite(r).all() and np.isfinite(x).all(), "Invalid point coefficients")
    require(np.all(r >= 0.) and np.all(x >= 0.) and np.all(r <= bound+1e-9) and np.all(x <= bound+1e-9), "Point violates the fixed response box")
    raw = sorted(C for C in supports if 1 < len(C) < n)
    active = sorted(C for C, a, b in zip(supports, r, x, strict=True) if 1 < len(C) < n and max(a, b) > zero)
    return dict(stored_internal_count=len(raw), active_internal_count=len(active), active_proper_clades=[list(C) for C in active])


def diagnose(output, destination=None):
    started = perf_counter(); output = Path(output).resolve(); runtime = output/"runtime"
    protocol, seal, consumed = verify_study(output)
    destination = Path(destination).resolve() if destination else output/"true_family_diagnostics"
    require(destination.is_relative_to(output) and not destination.is_relative_to(runtime), "Diagnostics must stay outside the frozen runtime")
    if destination.exists():
        raise FileExistsError("Existing oracle diagnostics are preserved; no automatic retries")
    destination.mkdir(parents=True)
    own = Path(__file__).resolve(); own_sha = sha(own)
    snapshot = destination/"source"/own_sha/own.name; snapshot.parent.mkdir(parents=True); shutil.copyfile(own, snapshot)
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=own_sha,
        source_archive=snapshot.relative_to(destination).as_posix(), consumed_sha256=consumed,
        planned_inputs=protocol["planned_inputs"], planned_methods=protocol["planned_methods"],
        oracle_role="posthoc true-family fixed-support diagnostic; forbidden for model/kappa selection",
        train_points=64, selection_points=32, total_observed_points=96,
        coefficient_bound=2., qp_soft_seconds=300., true_family_fits_per_available_input=1,
        main_budget_debit=0., no_MIqp_or_extra_observations=True)
    write(destination/"plan.json", plan)
    allowed_npz = [inside(runtime, info["path"]) for info in protocol["inputs"].values() if not info.get("generation_failed")]
    opened = read_guard(output, destination, own, allowed_npz)
    data, optimization, trees = native_modules(runtime, protocol["source_sha256"])
    alpha, zero, bound = [float(protocol["config"][name]) for name in ("alpha", "zero_tol", "response_bound")]
    summaries, comparisons = [], []
    for job, info in protocol["inputs"].items():
        began = perf_counter(); fitter = None; fitted = None; fit_calls = 0; fit_began = None; job_rows = []; read_start = len(opened)
        oracle = dict(job=job, status="failed", source_sha256=own_sha, input_sha256=info.get("sha256"),
                      selection_seal_sha256=sha(output/"selection_seal.json"), oracle_only=True)
        try:
            require(not info.get("generation_failed"), "Observed input generation failed")
            common = read(output/"common"/job/"common.json")
            require(common["status"] == "fitted", "No successful common64 reference is available")
            require(common["input_sha256"] == info["sha256"] and common["common"]["training_sha256"] == info["training_sha256"],
                    "Common64 reference belongs to different observations")
            reference = float(common["common"]["reference_mse"])
            require(np.isfinite(reference) and reference > 0, "Invalid common64 reference")
            training, selection = [data.Observation(**rows) for rows in input_rows(inside(runtime, info["path"]), info)]
            truth = read(output/"evaluation"/(job+".json"))
            require(truth["labels"] == info["labels"], "True-family terminal order changed")
            proper = trees.canonical(truth["proper_clades"])
            require(all(1 < len(C) < training.n and min(C) >= 0 and max(C) < training.n for C in proper)
                    and trees.is_laminar(proper), "Invalid true proper-clade family")
            family = trees.canonical([*trees.basis(training.n), *proper])
            fitter = optimization.QPFitter(training, reference_mse=reference, bound=bound, qp_seconds=300.)
            fit_began = perf_counter(); fit_calls = 1; fitted = fitter.fit(family); fit_seconds = perf_counter()-fit_began
            train_mse = float(np.mean((training.y-optimization.predict(training, fitted))**2))
            selection_mse = float(np.mean((selection.y-optimization.predict(selection, fitted))**2))
            counts = point_summary(fitted, training.n, zero, bound)
            active = set(map(tuple, counts["active_proper_clades"]))
            missing = sorted(set(proper)-active)
            raw_k = counts["stored_internal_count"]
            oracle.update(status="fitted", reference_mse=reference, point=fitted, proper_clades=[list(C) for C in proper],
                train_mse=train_mse, selection_mse=selection_mse, **counts,
                zero_weight_collapses_true_family=bool(missing), zero_weight_true_clades=[list(C) for C in missing],
                J_true_raw=train_mse/reference+alpha*raw_k,
                J_true_raw_lower=float(fitter.records[0]["lower_bound"])+alpha*raw_k,
                alpha=alpha, qp_seconds=fit_seconds, qp_records=fitter.records,
                validation_refit=False, extra_observations_used=False)
            for method in protocol["methods"]:
                record = read(output/"runs"/job/method/"result.json")
                row = dict(job=job, method=method, method_status=record["status"], oracle_status="fitted")
                if record["status"] != "fitted":
                    row.update(status="method_unavailable"); job_rows.append(row); continue
                require(record["input_sha256"] == info["sha256"] and
                        record["common_sha256"] == sha(output/"common"/job/"common.json"),
                        "Selected model belongs to a different input/common reference")
                point = record["result"]["selected"]["point"]
                selected_train = float(np.mean((training.y-optimization.predict(training, point))**2))
                selected_selection = float(np.mean((selection.y-optimization.predict(selection, point))**2))
                selected_counts = point_summary(point, training.n, zero, bound)
                delta_k = selected_counts["stored_internal_count"]-raw_k
                fit_term = (selected_train-train_mse)/reference; penalty_term = alpha*delta_k
                selected_J = selected_train/reference+alpha*selected_counts["stored_internal_count"]
                delta_J = selected_J-oracle["J_true_raw"]
                require(np.isclose(delta_J, fit_term+penalty_term, rtol=1e-9, atol=1e-10), "J decomposition failed")
                row.update(status="compared", reference_mse=reference,
                    selected_train_mse=selected_train, selected_selection_mse=selected_selection,
                    true_train_mse=train_mse, true_selection_mse=selection_mse,
                    selected_stored_k=selected_counts["stored_internal_count"], selected_active_k=selected_counts["active_internal_count"],
                    true_stored_k=raw_k, true_active_k=counts["active_internal_count"], delta_stored_k=delta_k,
                    selected_J=selected_J, true_J=oracle["J_true_raw"], delta_J=delta_J,
                    normalized_training_fit_difference=fit_term, count_penalty_difference=penalty_term,
                    selection_mse_difference=selected_selection-selection_mse,
                    true_family_zero_weight_collapse=bool(missing), selected_index=record["result"]["selected"]["index"],
                    selected_active_proper_clades=selected_counts["active_proper_clades"])
                job_rows.append(row)
        except Exception as exc:
            oracle.update(status="budget_exceeded" if isinstance(exc, TimeoutError) else "failed", error=traceback.format_exc())
            job_rows = [dict(job=job, method=method, method_status=read(output/"runs"/job/method/"result.json")["status"],
                             oracle_status=oracle["status"], status="oracle_unavailable") for method in protocol["methods"]]
        observed = sorted(set(opened[read_start:]))
        require(not observed or observed == [str(inside(runtime, info["path"]))], "An input's diagnostic opened another input")
        oracle.update(fixed_family_fit_calls=fit_calls, elapsed_seconds=perf_counter()-began,
                      qp_seconds=oracle.get("qp_seconds", perf_counter()-fit_began if fit_began is not None else 0.),
                      qp_records=fitter.records if fitter is not None else [], observed_npz_opened=observed)
        write(destination/"inputs"/(job+".json"), oracle)
        comparisons.extend(job_rows)
        summaries.append({key: oracle.get(key) for key in ("job", "status", "fixed_family_fit_calls", "elapsed_seconds", "qp_seconds", "zero_weight_collapses_true_family")})
    require(len(comparisons) == protocol["planned_methods"], "Diagnostic method denominator changed")
    fields = list(dict.fromkeys(key for row in comparisons for key in row))
    with (destination/"decompositions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(comparisons)
    for relative, expected in consumed.items():
        require(sha(inside(output, relative)) == expected, "Frozen source/input/result changed during posthoc diagnosis")
    require(sha(own) == own_sha, "Diagnostic source changed during execution")
    summary = dict(complete=True, created_utc=datetime.now(timezone.utc).isoformat(), planned_inputs=len(summaries),
        fitted_inputs=sum(row["status"] == "fitted" for row in summaries),
        failed_inputs=sum(row["status"] == "failed" for row in summaries),
        budget_exceeded_inputs=sum(row["status"] == "budget_exceeded" for row in summaries),
        planned_method_comparisons=len(comparisons), compared_methods=sum(row["status"] == "compared" for row in comparisons),
        unavailable_methods=sum(row["status"] != "compared" for row in comparisons),
        true_family_fit_calls=sum(row["fixed_family_fit_calls"] for row in summaries),
        posthoc_calendar_seconds=perf_counter()-started, sum_input_diagnostic_seconds=sum(row["elapsed_seconds"] for row in summaries),
        sum_true_qp_seconds=sum(row["qp_seconds"] or 0. for row in summaries),
        source_sha256=own_sha, protocol_sha256=seal["protocol_sha256"], selection_seal_sha256=sha(output/"selection_seal.json"),
        decompositions_sha256=sha(destination/"decompositions.csv"), original_bytes_unchanged=True,
        inputs=summaries, actual_topology_or_kappa_selection_changed=False, main_budget_debit=0.,
        extra_observation_files_read=0, native_MIQP_calls=0, oracle_only=True,
        interpretation="Counts use stored proper supports even if fitted weights become zero. Noise vs AC/linear bias is not separated by this decomposition.")
    write(destination/"summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--destination", type=Path)
    answer = diagnose(**vars(parser.parse_args()))
    print(json.dumps({key: answer[key] for key in ("complete", "planned_inputs", "fitted_inputs", "failed_inputs", "budget_exceeded_inputs", "compared_methods", "posthoc_calendar_seconds")}, indent=2))
