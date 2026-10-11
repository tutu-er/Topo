"""Resume an unchanged frozen AC protocol, parallelizing distinct cases only.

The original core/protocol remain untouched. Existing complete cases are
validated and hashed before being skipped. Incomplete case files are copied to
a dated archive before replacement. Aggregation occurs only after every case
has passed the completion validation; the independent AC audit runs afterwards.
"""
from __future__ import annotations

import os
# Set these before importing NumPy/SciPy, including in Windows spawned workers.
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
for _name in THREAD_ENV:
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction as F
import json
import multiprocessing
from pathlib import Path
import shutil
from time import perf_counter
import uuid

import numpy as np

import ac_shared_v8 as ac


def prefix_of(case):
    h, kind, profile, seed = case
    return f"{h}_{kind}_{profile}_{seed}"


def protocol_checks(folder):
    protocol_path = folder / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = [(h, kind, profile, seed) for h in ac.v6.CANDIDATES
                for kind, profile in [("flat", "symmetric"), ("rich", "symmetric"), ("flat", "weak")]
                for seed in (8201, 8202)]
    if protocol["mode"] != "formal" or list(map(tuple, protocol["cases"])) != expected:
        raise ValueError("resume helper accepts only the unchanged 18-case formal protocol")
    if protocol["source_sha256"] != ac.v7.sha(ac.__file__):
        raise ValueError("frozen core source hash mismatch")
    for module in (ac.v7, ac.v6):
        name = Path(module.__file__).name
        if protocol["dependency_sha256"].get(name) != ac.v7.sha(module.__file__):
            raise ValueError("frozen dependency hash mismatch: " + name)
    if protocol["actions"] != ["site1_none", *ac.ACTION_NAMES]:
        raise ValueError("frozen action library mismatch")
    if protocol["box"] != [list(v) for v in ac.BOX]:
        raise ValueError("frozen parameter box mismatch")
    return protocol, ac.v7.sha(protocol_path)


def completion_check(folder, case, protocol):
    """Structural/physical/certificate checks before any case may be skipped."""
    prefix = prefix_of(case)
    path = folder / (prefix + "_results.json")
    if not path.exists():
        return None, "no completed case result"
    try:
        ep = json.loads(path.read_text(encoding="utf-8"))
        if (ep["truth"], ep["history"], ep["profile"], ep["seed"]) != tuple(case):
            raise ValueError("case identity mismatch")
        expected = {(a, h, sharing) for a in protocol["actions"] for h in ac.v6.CANDIDATES
                    for sharing in ("path", "output", "invalid_linear")}
        actual = [(r["action"], r["candidate"], r["sharing"]) for r in ep["records"]]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError("action/candidate/ablation coverage mismatch")
        if set(ep["history_results"]) != set(ac.v6.CANDIDATES):
            raise ValueError("historical candidate coverage mismatch")
        if ep["retained"] != [h for h, r in ep["history_results"].items() if r["status"] != "excluded"]:
            raise ValueError("retained candidate mismatch")
        fits = list(ep["history_results"].values()) + [r["result"] for r in ep["records"]]
        fits += [r["result"] for r in ep["design"].get("records", [])]
        for fit in fits:
            # A completed solver failure is retained evidence, not grounds to
            # rerun a complete case until a more favorable outcome is obtained.
            if fit["status"] == "solver_failure_unresolved":
                continue
            lo, hi = F(fit["exact_lower"]), F(fit["exact_upper"])
            if not (fit["exact_primal_verified"] and fit["exact_box_dual_verified"] and
                    F(fit["lower"]) <= lo <= hi <= F(fit["upper"]) and hi <= 1):
                raise ValueError("invalid recorded bound interval")
        baseline = None
        base_truth = None
        for action in protocol["actions"]:
            obsfile = folder / f"{prefix}_{action}.npz"
            truthfile = folder / f"{prefix}_{action}_truth.json"
            with np.load(obsfile) as archive:
                obs = {k: archive[k].copy() for k in archive.files}
            truth = json.loads(truthfile.read_text(encoding="utf-8"))
            if (truth["topology"], truth["history"], truth["seed"]) != (case[0], case[1], case[3]):
                raise ValueError("input truth identity mismatch")
            if obs["P"].shape != (36, 3) or obs["Y_drop"].shape != (36, 3) or not all(truth["checks"].values()):
                raise ValueError("input shape or physical coverage failure")
            if baseline is None:
                baseline, base_truth = obs, truth
            elif not (all(np.array_equal(obs[k][:12], baseline[k][:12]) for k in baseline) and
                      truth["source_p"] == base_truth["source_p"] and truth["actual_noise"] == base_truth["actual_noise"]):
                raise ValueError("unpaired historical/source/noise inputs")
        return ep, None
    except (KeyError, ValueError, OSError, TypeError) as exc:
        return None, f"existing result not complete/valid: {type(exc).__name__}: {exc}"


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".resume-tmp")
    ac.v7.dump(temporary, value)
    os.replace(temporary, path)


def atomic_npz(path, value):
    temporary = path.with_name(path.name + ".resume-tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **value)
    os.replace(temporary, path)


def execute_case(folder_text, case, protocol):
    """Exactly the case-level calls/order in frozen ac_shared_v8.run()."""
    folder = Path(folder_text)
    checked, _ = protocol_checks(folder)
    if checked != protocol:
        raise ValueError("protocol changed while worker was starting")
    h, kind, profile, seed = case
    prefix = prefix_of(case)
    began = perf_counter()
    first, first_truth = ac.make_data(h, ac.v7.PROFILES[profile], seed, kind)
    history = ac.sliced(first, slice(0, 12))
    history_results = {j: ac.solve_outer(history, j) for j in ac.v6.CANDIDATES}
    retained = [j for j, r in history_results.items() if r["status"] != "excluded"]
    if len(retained) > 1:
        design = ac.design_ac(history, retained)
    elif retained:
        witness = ac.fit_ac_reference(history, retained[0])
        design = dict(status="conditional_singleton_skip" if witness["historical_numeric_witness"] else "compute_before_dispatch",
                      selection=None, references={retained[0]: witness})
    else:
        design = dict(status="inconsistent", selection=None)
    records = []
    for action in protocol["actions"]:
        site = int(action[4]) - 1 if action.startswith("site") else 0
        policy = action.split("_")[1] if action.startswith("site") else action
        obs, truth = ac.make_data(h, ac.v7.PROFILES[profile], seed, kind, site, policy)
        name = f"{prefix}_{action}"
        atomic_npz(folder / (name + ".npz"), obs)
        atomic_json(folder / (name + "_truth.json"), truth)
        for j in ac.v6.CANDIDATES:
            for sharing in ("path", "output"):
                result = ac.solve_outer(obs, j, sharing=sharing)
                records.append(dict(action=action, candidate=j, sharing=sharing, result=result))
            invalid_linear = ac.solve_outer(obs, j, linear_only=True)
            records.append(dict(action=action, candidate=j, sharing="invalid_linear", result=invalid_linear))
    episode = dict(truth=h, profile=profile, history=kind, seed=seed, history_results=history_results,
                   retained=retained, design=design, records=records)
    atomic_json(folder / (prefix + "_results.json"), episode)
    verified, issue = completion_check(folder, case, protocol)
    if issue:
        raise ValueError(prefix + ": " + issue)
    return dict(case=prefix, seconds=perf_counter() - began, status=design["status"],
                selected=design.get("selection"), case_result_sha256=ac.v7.sha(folder / (prefix + "_results.json")))


def resume(folder, workers=3, dry_run=False):
    folder = Path(folder).resolve()
    workspace = Path(__file__).resolve().parents[2]
    if not folder.is_relative_to(workspace):
        raise ValueError("resume output must stay within this workspace")
    if not 1 <= workers <= 3:
        raise ValueError("workers must be between 1 and 3")
    protocol, frozen_hash = protocol_checks(folder)
    completed, pending, issues, complete_hashes = [], [], {}, {}
    for case in protocol["cases"]:
        ep, issue = completion_check(folder, case, protocol)
        prefix = prefix_of(case)
        if ep is not None:
            completed.append(case)
            complete_hashes.update({path.name: ac.v7.sha(path) for path in folder.glob(prefix + "_*") if path.is_file()})
        else:
            pending.append(case)
            issues[prefix] = issue
    snapshot = dict(completed=[prefix_of(c) for c in completed], pending=[prefix_of(c) for c in pending],
                    pending_reasons=issues, protocol_sha256=frozen_hash, workers=workers)
    if dry_run:
        print(json.dumps(snapshot, indent=2), flush=True)
        return snapshot
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    metadata_path = folder / f"resume_execution_{run_id}.json"
    metadata = dict(snapshot, run_id=run_id, helper_sha256=ac.v7.sha(__file__),
                    source_sha256=ac.v7.sha(ac.__file__), thread_environment={k: os.environ[k] for k in THREAD_ENV},
                    completed_files_sha256_before=complete_hashes, archives={}, worker_results=[], errors=[],
                    interpretation="execution-only resume; unchanged frozen cases, model, source, seeds and actions")
    for case in pending:
        prefix = prefix_of(case)
        existing = [p for p in folder.glob(prefix + "_*") if p.is_file()]
        if existing:
            backup = folder / "resume_incomplete_archives" / run_id / prefix
            backup.mkdir(parents=True, exist_ok=False)
            metadata["archives"][prefix] = {}
            for path in existing:
                saved = backup / path.name
                shutil.copy2(path, saved)
                digest = ac.v7.sha(path)
                if ac.v7.sha(saved) != digest:
                    raise RuntimeError("archive copy hash mismatch")
                metadata["archives"][prefix][path.name] = digest
    atomic_json(metadata_path, metadata)
    began = perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        tasks = {pool.submit(execute_case, str(folder), case, protocol): case for case in pending}
        for task in as_completed(tasks):
            case = tasks[task]
            try:
                record = task.result()
                metadata["worker_results"].append(record)
                print(json.dumps(record), flush=True)
            except Exception as exc:
                error = dict(case=prefix_of(case), error=f"{type(exc).__name__}: {exc}")
                metadata["errors"].append(error)
                print(json.dumps(error), flush=True)
            metadata["elapsed_seconds"] = perf_counter() - began
            atomic_json(metadata_path, metadata)
    protocol_after, hash_after = protocol_checks(folder)
    if hash_after != frozen_hash or protocol_after != protocol:
        raise RuntimeError("frozen protocol changed during execution")
    for name, digest in complete_hashes.items():
        if ac.v7.sha(folder / name) != digest:
            raise RuntimeError("previously complete case file changed: " + name)
    episodes = []
    for case in protocol["cases"]:
        ep, issue = completion_check(folder, case, protocol)
        if issue:
            metadata["errors"].append(dict(case=prefix_of(case), error=issue))
        else:
            episodes.append(ep)
    metadata["completed_files_unchanged"] = True
    metadata["all_cases_complete"] = len(episodes) == len(protocol["cases"]) and not metadata["errors"]
    atomic_json(metadata_path, metadata)
    if not metadata["all_cases_complete"]:
        raise RuntimeError("resume has incomplete/error cases; preserved completed files and withheld aggregate")
    inputs = {path.name: ac.v7.sha(path) for path in folder.glob("*.npz")}
    inputs.update({path.name: ac.v7.sha(path) for path in folder.glob("*_truth.json")})
    atomic_json(folder / "results.json", dict(protocol=protocol, episodes=episodes, input_sha256=inputs,
                source_hash_unchanged=ac.v7.sha(ac.__file__) == protocol["source_sha256"]))
    import audit_ac_v8
    audit_result = audit_ac_v8.audit(folder)
    atomic_json(folder / "artifact_audit.json", audit_result)
    metadata["independent_audit_completed"] = True
    metadata["independent_audit_passed"] = audit_result["passed"]
    metadata["aggregate_sha256"] = ac.v7.sha(folder / "results.json")
    metadata["elapsed_seconds"] = perf_counter() - began
    atomic_json(metadata_path, metadata)
    if not audit_result["passed"]:
        raise RuntimeError("independent AC audit reported issues; preserved aggregate and audit")
    return metadata


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    resume(args.output, args.workers, args.dry_run)
