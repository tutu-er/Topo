"""Resume the pinned, interrupted AC96 queue; default is read-only planning.

The original native runner is unchanged. Interrupted attempts are archived and
restarted from their declared initial family, not resumed from a solver state.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

WORKSPACE = Path(__file__).resolve().parents[2]
RUN_NAME = "rnj_blocks96_20261019"
PINNED_PROTOCOL = "167eaa23d1fb9257aaefa45ae90c6805ef28c4376fae4695993e410875acea6c"
EXPECTED_FINAL = 41
INTERRUPTED = ("paper15__intermittent__RNJ+WZZT", "paper15__intermittent__RNJ+WZZT_ungrouped",
               "paper15__intermittent__WZZT", "paper15__smooth__RNJ+WZZT")
STATE_NAMES = ("finalize_rnj_blocks96_state_20261019", "finalize_rnj_blocks96_diagnostics_state_20261019")
FINALIZERS = ("finalize_rnj_blocks96_when_complete_20261019.py", "finalize_rnj_blocks96_diagnostics_when_complete_20261019.py")
FINAL = {"complete", "incomplete", "failed"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(path, workspace):
    path, workspace = Path(path).resolve(), Path(workspace).resolve()
    if path == workspace or not path.is_relative_to(workspace):
        raise ValueError("path must remain strictly inside workspace: " + str(path))
    return path


def inventory(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    result = {}
    for item in files:
        if item.is_symlink() or not item.resolve().is_relative_to(path.resolve() if path.is_dir() else path.parent.resolve()):
            raise ValueError("archive rejects symbolic links or escaped files")
        before = item.stat()
        result[item.relative_to(path).as_posix() if path.is_dir() else "."] = {"sha256": sha(item), "bytes": before.st_size}
        after = item.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("file changed while hashing: " + str(item))
    return result


def verify(mapping):
    for path, checksum in mapping.items():
        if not Path(path).is_file() or sha(path) != checksum:
            raise RuntimeError("pinned/protected file changed: " + path)


def proof_check(path, root, old_pids, *, max_age=900):
    data = read(path)
    captured = datetime.fromisoformat(data["captured_utc"].replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc)-captured).total_seconds()
    if data.get("complete_inventory") is not True or Path(data["target_root"]).resolve() != root.resolve() or not -60 <= age <= max_age:
        raise ValueError("CIM proof must be complete, fresh and bound to this run")
    processes = data["processes"]
    if not isinstance(processes, list) or not all("ProcessId" in p and "CommandLine" in p for p in processes):
        raise ValueError("CIM proof lacks complete process records")
    if set(old_pids) & {int(p["ProcessId"]) for p in processes}:
        raise ValueError("old controller/worker PID is still present in CIM proof")
    if any(any(name in str(p.get("CommandLine") or "").lower() for name in
               ("run_rnj_blocks96", "resume_rnj_blocks96", "finalize_rnj_blocks96")) for p in processes):
        raise ValueError("a batch controller/worker/finalizer is still present in CIM proof")
    return {"path": str(Path(path).resolve()), "sha256": sha(path), "captured_utc": data["captured_utc"],
            "process_count": len(processes), "old_pids_absent": sorted(old_pids)}


def build_plan(*, workspace=WORKSPACE, cim_proof=None, archive_dir=None, state_dir=None):
    workspace = Path(workspace).resolve()
    root = inside(workspace/"outputs"/RUN_NAME, workspace)
    protocol = read(root/"protocol.json")
    if sha(root/"protocol.json") != PINNED_PROTOCOL or protocol.get("expected_jobs") != 64 or protocol.get("concurrency") != 4 or protocol.get("solver") != "gurobi":
        raise ValueError("only the pinned original 64-job protocol may be resumed")
    archive = inside(archive_dir or workspace/"outputs/rnj_blocks96_interruption_archive_20261019", workspace)
    state = inside(state_dir or workspace/"tmp/resume_rnj_blocks96_state_20261019", workspace)
    if archive.exists() or state.exists() or archive.is_relative_to(root) or root.is_relative_to(archive):
        raise FileExistsError("fresh independent archive and supervisor state directories required")
    if any((root/p).exists() for p in ("execution_complete.json", "reports_final", "audits/final", "audits/supplemental")):
        raise ValueError("final products already exist; this supervisor only handles the original interruption")
    fixed = {str(root/"protocol.json"): PINNED_PROTOCOL}
    snapshot = root/"source_snapshot"
    for rel, checksum in protocol["source_sha256"].items():
        fixed[str(inside(snapshot/rel, workspace))] = checksum
    fixed[str(root/"dataset_manifest.json")] = protocol["dataset_manifest_sha256"]
    fixed[str(snapshot/protocol["design_protocol_path"])] = protocol["design_protocol_sha256"]
    runtime = inside(protocol["runtime_directory"], workspace)
    fixed[str(runtime/"runtime_manifest.json")] = protocol["runtime_manifest_sha256"]
    runtime_meta = read(runtime/"runtime_manifest.json")
    for field, base in (("runtime_sha256", runtime/"rnj_wzzt"), ("source_sha256", runtime/"source_snapshot/rnj_wzzt")):
        fixed.update({str(inside(base/p, workspace)): h for p, h in runtime_meta[field].items()})
    manifest, completed, interrupted, pending, old_pids, protected = read(root/"dataset_manifest.json"), [], [], [], [], {}
    lock = read(root/"controller.lock")
    old_pids.append(int(lock["pid"]))
    for job in protocol["jobs"]:
        folder = inside(root/"jobs"/job["job_id"], workspace)
        inputs = inside(root/job["input_relative_directory"], workspace)
        hashes = read(folder/"inputs_hash.json")["file_sha256"]
        actual = {p.relative_to(inputs).as_posix() for p in inputs.rglob("*") if p.is_file()}
        if len(hashes) != 10 or actual != set(hashes):
            raise ValueError("observed/source input whitelist changed")
        for rel, checksum in hashes.items():
            if manifest["file_sha256"].get(f"{job['case']}/{job['variant']}/{rel}") != checksum:
                raise ValueError("worker input differs from generation manifest")
            fixed[str(inside(inputs/rel, workspace))] = checksum
        if (folder/"result.json").exists():
            result = read(folder/"result.json")
            if result["status"] not in FINAL or result["protocol_sha256"] != PINNED_PROTOCOL:
                raise ValueError("existing result is not final under the pinned protocol")
            completed.append(job["job_id"])
            for name in ("result.json", "payload.json", "inputs_hash.json", "model.json", "model.npz"):
                protected[str(folder/name)] = sha(folder/name)
        elif (folder/"execution.json").exists():
            old_pids.append(int(read(folder/"execution.json")["pid"]))
            pointer = read(folder/"last_certified_model.json")
            for rel, checksum in pointer["files"].items():
                target = inside(folder/rel, workspace)
                if not target.is_relative_to((folder/"certified_models").resolve()) or sha(target) != checksum:
                    raise ValueError("interrupted certified checkpoint changed")
            progress = [read(p) for p in (folder/"progress").glob("*.json")]
            interrupted.append({"job_id": job["job_id"], "checkpoint_sequence": pointer["sequence"],
                "checkpoint_elapsed_seconds_lower_bound": max((p["elapsed_seconds"] for p in progress), default=None),
                "outcome": "external_process_interruption_no_final_result", "solver_timeout_inferred": False,
                "actual_spent_seconds": None, "exact_total_seconds": None,
                "restart": "archive interrupted attempt, then execute afresh from original initialization; no core-path resume"})
        else:
            pending.append(job["job_id"])
    if len(completed) != EXPECTED_FINAL or {p["job_id"] for p in interrupted} != set(INTERRUPTED) or len(pending) != 64-EXPECTED_FINAL-len(INTERRUPTED):
        raise ValueError("the 41 final / 4 interrupted / 19 unstarted scope changed")
    binding = protocol["baseline_evidence"]
    fixed[str(root/binding["baseline_provenance_path"])] = binding["baseline_provenance_sha256"]
    fixed.update({str(inside(root/p, workspace)): h for p, h in binding["inventory"].items()})
    helper_paths = [workspace/"tmp"/p for p in FINALIZERS]
    for name in STATE_NAMES:
        saved = read(workspace/"tmp"/name/"state.json")
        fixed.update(saved.get("helper_sha256", saved.get("code_sha256", {})))
    fixed.update({str(p): sha(p) for p in helper_paths})
    verify(fixed); verify(protected)
    proof = proof_check(cim_proof, root, old_pids) if cim_proof else None
    moves = [(root/"jobs"/p["job_id"], archive/"interrupted_jobs"/p["job_id"]) for p in interrupted]
    moves += [(root/"controller.lock", archive/"controller.lock")]
    moves += [(workspace/"tmp"/name, archive/"finalizer_states"/name) for name in STATE_NAMES]
    archive_items = [{"source": str(inside(a, workspace)), "target": str(inside(b, workspace)), "inventory": inventory(a)} for a, b in moves]
    return {"schema_version": 1, "workspace": str(workspace), "run_root": str(root), "archive_dir": str(archive),
        "state_dir": str(state), "protocol_sha256": PINNED_PROTOCOL, "existing_final_jobs": completed,
        "interrupted_attempts": interrupted, "unstarted_jobs": pending, "fixed_sha256": fixed,
        "protected_final_sha256": protected, "process_proof": proof, "archive_items": archive_items,
        "native_arguments": ["--dataset-dir", str(workspace/"outputs/source_ac96_20261018"), "--runtime-dir", str(runtime),
            "--baseline-dir", str(workspace/"outputs/source_ac96_benchmark_20261018"), "--output-dir", str(root), "--workers", "4", "--solver", "gurobi"],
        "finalizers": list(map(str, helper_paths)), "supervisor_sha256": sha(__file__),
        "attempt_accounting": {"planned_job_denominator": 64, "extra_interrupted_attempts": len(interrupted),
            "existing_io_recovery_attempts": 1, "exact_total_seconds": None,
            "summary_time_scope": "for the four interrupted jobs, native summary records only their new execution; prior interrupted cost is unknown and checkpoint elapsed is a lower bound, never zero",
            "checkpoint_score_scope": "no-final interrupted checkpoint is not a completed algorithm score"},
        "scope": "continue pending queue; archive four interrupted attempts and restart them under unchanged per-attempt budgets; keep all prior final outcomes"}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def archive_attempts(plan):
    workspace, archive = Path(plan["workspace"]), Path(plan["archive_dir"])
    archive.mkdir(parents=True, exist_ok=False)
    # Publish all original SHA bindings before the first move, including failures.
    write(archive/"resume_plan.json", plan)
    write(archive/"archive_mapping.json", plan["archive_items"])
    for item in plan["archive_items"]:
        source, target = inside(item["source"], workspace), inside(item["target"], workspace)
        if target.exists() or inventory(source) != item["inventory"]:
            raise RuntimeError("archive source/target changed before move")
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        if inventory(target) != item["inventory"]:
            raise RuntimeError("archived bytes changed")


def checked_prepare(original, args, plan):
    verify(plan["fixed_sha256"]); verify(plan["protected_final_sha256"])
    prepared = original(args)
    verify(plan["fixed_sha256"]); verify(plan["protected_final_sha256"])
    write(Path(plan["state_dir"])/"native_prepare_verified.json", {"passed": True, "protected_file_count": len(plan["protected_final_sha256"]), "protocol_sha256": sha(Path(plan["run_root"])/"protocol.json")})
    return prepared


def native(plan_path):
    plan = read(plan_path)
    snapshot = Path(plan["run_root"])/"source_snapshot"
    sys.path.insert(0, str(snapshot))
    module = importlib.import_module("research_experiments.rnj.run_rnj_blocks96")
    if Path(module.__file__).resolve() != (snapshot/"research_experiments/rnj/run_rnj_blocks96.py").resolve():
        raise RuntimeError("native runner did not import from frozen snapshot")
    original = module.prepare
    module.prepare = lambda args: checked_prepare(original, args, plan)
    sys.argv = [str(module.__file__), *plan["native_arguments"]]
    return module.main()


def execute(plan):
    if plan["process_proof"] is None:
        raise ValueError("--execute requires a fresh complete root-provided --cim-proof")
    verify(plan["fixed_sha256"]); verify(plan["protected_final_sha256"])
    verify({plan["process_proof"]["path"]: plan["process_proof"]["sha256"]})
    state_dir, archive = Path(plan["state_dir"]), Path(plan["archive_dir"])
    state_dir.mkdir(parents=True, exist_ok=False)
    state = {"status": "archiving", "started_utc": datetime.now(timezone.utc).isoformat(), "commands": []}
    # Direct file handles survive the launching shell/PTY, including all child commands.
    with (state_dir/"supervisor.stdout.log").open("x", encoding="utf-8") as out, (state_dir/"supervisor.stderr.log").open("x", encoding="utf-8") as err:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            write(state_dir/"state.json", state)
            archive_attempts(plan)
            shutil.copyfile(plan["process_proof"]["path"], archive/"process_inventory.json")
            if sha(archive/"process_inventory.json") != plan["process_proof"]["sha256"]:
                raise RuntimeError("CIM proof changed before archival")
            shutil.copyfile(__file__, archive/"resume_rnj_blocks96.py")
            write(archive/"resume_plan.json", plan)
            write(archive/"attempt_accounting.json", plan["attempt_accounting"])
            commands = [[sys.executable, "-B", str(Path(__file__).resolve()), "--native-plan", str(archive/"resume_plan.json")]]
            commands += [[sys.executable, "-B", path] for path in plan["finalizers"]]
            for label, command in zip(("native_queue", "primary_finalizer", "diagnostics_finalizer"), commands, strict=True):
                verify(plan["fixed_sha256"]); verify(plan["protected_final_sha256"])
                state.update(status=label); write(state_dir/"state.json", state)
                with (state_dir/(label+".stdout.log")).open("x", encoding="utf-8") as child_out, (state_dir/(label+".stderr.log")).open("x", encoding="utf-8") as child_err:
                    result = subprocess.run(command, cwd=plan["workspace"], stdout=child_out, stderr=child_err,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                state["commands"].append({"label": label, "returncode": result.returncode})
                write(state_dir/"state.json", state)
                if result.returncode:
                    raise RuntimeError(label+" exited "+str(result.returncode))
                if label == "native_queue" and (not (Path(plan["run_root"])/"execution_complete.json").is_file() or (Path(plan["run_root"])/"controller.lock").exists()):
                    raise RuntimeError("native queue returned without closing the full 64-job run")
            verify(plan["fixed_sha256"]); verify(plan["protected_final_sha256"])
            companion = Path(plan["run_root"])/"reports_final/recovery_attempt_accounting.json"
            if companion.exists():
                raise FileExistsError(companion)
            write(companion, plan["attempt_accounting"] | {"archive_directory": str(archive)})
            state.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
        except Exception as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            traceback.print_exc()
            raise
        finally:
            write(state_dir/"state.json", state)
            sys.stdout, sys.stderr = old_out, old_err
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cim-proof", type=Path)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--native-plan", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.native_plan:
        return native(args.native_plan)
    plan = build_plan(cim_proof=args.cim_proof, archive_dir=args.archive_dir, state_dir=args.state_dir)
    if args.execute:
        return execute(plan)
    print(json.dumps({"mode": "plan_only", "existing_final": len(plan["existing_final_jobs"]), "interrupted_to_archive_and_restart": len(plan["interrupted_attempts"]), "unstarted": len(plan["unstarted_jobs"]),
        "protected_files": len(plan["protected_final_sha256"]), "fixed_files": len(plan["fixed_sha256"]), "process_proof_supplied": plan["process_proof"] is not None,
        "archive_directory": plan["archive_dir"], "state_directory": plan["state_dir"], "protocol_sha256": plan["protocol_sha256"], "supervisor_sha256": plan["supervisor_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
