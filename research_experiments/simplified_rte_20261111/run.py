"""Four-input pilot of simplified R/T/E; freeze never starts an optimizer."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter, sleep
import traceback

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/"outputs/miqp_loss_benchmark_20261111"
PACKAGE = "research_experiments.simplified_rte_20261111.run"
METHOD = "simplified_RTE"
LIMITS = dict(total=14400., QP=300., per_mip=3600., post_reserve=600., mip_gap=.0001, threads=1)


def read(path):
    for attempt in range(100):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 99: raise
            sleep(.05)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    for attempt in range(100):
        try:
            temporary.write_text(serialized, encoding="utf-8", newline="\n")
            temporary.replace(path); return
        except PermissionError:
            if attempt == 99: raise
            sleep(.05)


def versions():
    result = {}
    for name in ("numpy", "scipy", "pandas", "gurobipy"):
        try: result[name] = metadata.version(name)
        except metadata.PackageNotFoundError: result[name] = "not-installed"
    return dict(python=sys.version, executable=sys.executable, dependencies=result)


def parameters(point):
    rows = sorted(zip(point["supports"], point["r"], point["x"]), key=lambda row: (len(row[0]), tuple(row[0])))
    return dict(supports=[list(map(int, C)) for C, _, _ in rows],
                r=[float(r) for _, r, _ in rows], x=[float(x) for _, _, x in rows])


def raw_library(points, *, train, validation, K, n):
    """Complete fitted raw predictors only: no contraction or topology map."""
    import numpy as np
    models, seen, library = [], {}, []
    for index, (point, source) in enumerate(points):
        point = parameters(point); key = digest(point)
        z = np.asarray([[i in C for i in range(n)] for C in point["supports"]], float)
        R, X = (z.T*np.asarray(point["r"]))@z, (z.T*np.asarray(point["x"]))@z
        scores = {}
        for split, arrays in (("train", train), ("validation", validation)):
            residual = arrays["y"]-arrays["p"]@R.T-arrays["q"]@X.T
            scores.update({split+"_mae": float(np.mean(abs(residual))), split+"_mse": float(np.mean(residual**2))})
        internal = sum(1 < len(C) < n for C in point["supports"])
        origin = dict(source=source, path_index=index)
        if key not in seen:
            fitted = dict(point, objective=K*scores["train_mse"], fit_loss="scaled_mse", loss_scale=K)
            row = dict(model_id=key, parameter_sha256=key, family_sha256=digest(point["supports"]), point=fitted,
                       R=R.tolist(), X=X.tolist(), proper_count=internal, origins=[origin], **scores)
            seen[key] = len(models); models.append(row)
        else:
            models[seen[key]]["origins"].append(origin)
        library.append(dict(index=index, model_id=key, source=source, point=models[seen[key]]["point"],
                            internal=internal, validation=scores["validation_mae"],
                            validation_mae=scores["validation_mae"], validation_mse=scores["validation_mse"],
                            validation_scaled_mse=K*scores["validation_mse"]))
    if not models: raise ValueError("A nonempty fitted path is required.")
    return models, library


def choose(rows, metric):
    key = "validation_mae" if metric == "mae" else "validation_scaled_mse"
    best = min(row[key] for row in rows)
    eligible = [row for row in rows if row[key] <= best+1e-12]
    result = min(eligible, key=lambda row: (row["internal"], row[key], row["index"]))
    return dict(result, rule="strict_"+metric, eligible_count=len(eligible), best_selection_score=best)


def freeze(output, *, parent=PARENT, workers=4):
    output, parent = Path(output).resolve(), Path(parent).resolve()
    if output.exists(): raise FileExistsError("Use a new immutable output directory.")
    if not 1 <= workers <= 4: raise ValueError("The pilot permits one to four single-thread workers.")
    previous, audit = read(parent/"protocol.json"), read(parent/"analysis/verification.json")
    if not (audit["passed"] and audit["complete"] and audit["new_native_verified"] == audit["reused_parent_verified"] == 96):
        raise ValueError("The complete independently verified parent panel is required.")
    if audit["protocol_sha256"] != sha(parent/"protocol.json"): raise ValueError("Parent protocol changed.")
    catalog = read(parent/"catalog.json")
    pilot = {j: m for j, m in catalog.items() if m["seed_index"] == 0 and m["voltage_sigma"] == .0005}
    if len(pilot) != 4 or len({m["case"] for m in pilot.values()}) != 4: raise ValueError("The predeclared four-network pilot changed.")
    for name, expected in {**previous["source_hashes"], **previous["source_artifact_hashes"]}.items():
        if sha(parent/"source"/name) != expected: raise ValueError("Frozen parent source changed: "+name)
    for name in ("__init__.py", "engine.py", "run.py", "analyze.py"):
        if not Path(__file__).with_name(name).exists(): raise FileNotFoundError("Complete the new package before freezing: "+name)
    common_hashes, old_hashes = {}, {}
    for job, meta in pilot.items():
        if meta["data_error"]: raise ValueError("A fixed pilot input has a generation error.")
        for split in ("train", "validation"):
            if sha(meta[split]) != previous["input_hashes"][job][split]: raise ValueError("Historical observations changed.")
        common_hashes[job] = sha(parent/"common"/job/"common.json")
        old_hashes[job] = sha(parent/"jobs"/job/"T003/result.json")
    output.mkdir(parents=True)
    shutil.copytree(parent/"source", output/"source", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    new = output/"source/research_experiments/simplified_rte_20261111"
    shutil.copytree(Path(__file__).parent, new, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, expected in previous["source_hashes"].items():
        if sha(output/"source"/name) != expected: raise ValueError("A parent Python source byte was altered.")
    source_hashes = {p.relative_to(output/"source").as_posix(): sha(p) for p in sorted((output/"source").rglob("*.py"))}
    artifacts = {**previous["source_artifact_hashes"],
                 **{p.relative_to(output/"source").as_posix(): sha(p) for p in sorted(new.rglob("*")) if p.is_file()}}
    write(output/"catalog.json", pilot)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), method=METHOD, planned=4, planned_inputs=4,
        jobs=list(pilot), workers=workers, pilot_selection=dict(seed_index=0, voltage_sigma=.0005, networks=4),
        limits=LIMITS, nominal_method_hours=16., alpha=.003, block_limit=5, hard=[],
        native_time_caps="W: <=3600 per MIQP; T/E: allocated structural-stage remaining time",
        acceptance="QP-closed J_old-J_new > 1e-10 in each W/T/E; no relative-gain parameter",
        source_hashes=source_hashes, source_artifact_hashes=artifacts,
        parent_output=str(parent), parent_protocol_sha256=sha(parent/"protocol.json"),
        parent_verification_sha256=sha(parent/"analysis/verification.json"), parent_catalog_sha256=sha(parent/"catalog.json"),
        parent_source_hashes=previous["source_hashes"], parent_common_sha256=common_hashes,
        parent_T003_result_sha256=old_hashes, catalog_sha256=sha(output/"catalog.json"),
        input_hashes={j: previous["input_hashes"][j] for j in pilot}, data_regenerated=False,
        shared_preprocessing="reuse frozen QP reference/K, RNJ geometry, initial supports and warm support families; parent common.seconds charged once per new evaluation",
        common_pool_role="Bank column container only; engine defines actual R/W/T/E candidate domains independently",
        initial_state="singleton/full basis; no retained RNJ seed blocks",
        stage_order="W to <=5 direct blocks; one frontier-union T; one E; no return to W after T cleanup",
        primary_validation="strict physical MAE on raw own_path plus singleton; internal-count/index ties within1e-12",
        secondary_validation="strict K*MSE on the identical raw fitted library; no new fitting",
        contraction=False, complete_RNJ_output_fallback=False, truth_used_in_training=False, test_opened_by_worker=False,
        comparison_scope="four-input exploratory comparison; old T003 used .24 output contraction and a different T/E policy; shared old preprocessing is reused, so runtime is not an independent fair-budget benchmark",
        environment=versions())
    write(output/"protocol.json", protocol); write(output/"state.json", dict(stage="frozen", planned=4, returned=0, failed=0, pending=4))
    print("FROZEN four-input pilot; no data generation or solver calls.", flush=True)


def check_sources(output, protocol):
    for name, expected in {**protocol["source_hashes"], **protocol["source_artifact_hashes"]}.items():
        if sha(output/"source"/name) != expected: raise ValueError("Frozen pilot source changed: "+name)
    parent = Path(protocol["parent_output"])
    if sha(parent/"protocol.json") != protocol["parent_protocol_sha256"] or sha(parent/"analysis/verification.json") != protocol["parent_verification_sha256"]:
        raise ValueError("Parent provenance changed.")
    for job in protocol["jobs"]:
        if sha(parent/"common"/job/"common.json") != protocol["parent_common_sha256"][job]: raise ValueError("Parent common changed.")
        if sha(parent/"jobs"/job/"T003/result.json") != protocol["parent_T003_result_sha256"][job]: raise ValueError("Parent baseline changed.")


def loaded_sources(source, manifest):
    rows = {}
    for name, module in sorted(sys.modules.items()):
        if not (name == "research_experiments" or name.startswith("research_experiments.") or name == "rnj_wzzt" or name.startswith("rnj_wzzt.")):
            continue
        if not getattr(module, "__file__", None): continue
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(source): raise RuntimeError("A native import escaped frozen source: "+name)
        relative = path.relative_to(source).as_posix()
        if sha(path) != manifest.get(relative): raise RuntimeError("Loaded source hash mismatch: "+name)
        rows[name] = dict(relative_path=relative, sha256=sha(path))
    return rows


def worker(payload):
    started = perf_counter(); p = read(payload)
    source = Path(p["source"]).resolve()
    if not Path(__file__).resolve().is_relative_to(source): raise RuntimeError("Run workers from the frozen pilot snapshot.")
    if sha(__file__) != p["source_hashes"].get("research_experiments/simplified_rte_20261111/run.py"):
        raise RuntimeError("The frozen worker file differs from its source manifest.")
    sys.path[:0] = [str(source), str(source/"rnj_wzzt_core")]
    import research_experiments
    research_experiments.__path__ = [str(source/"research_experiments")]
    import numpy as np
    from research_experiments.miqp_loss_benchmark_20261111 import squared_backend as backend
    from research_experiments.day96_t_benchmark_20261110.data import observed
    from research_experiments.rebuild_benchmark.engine import prepare
    target = Path(p["result"]); common = read(p["common"])
    if sha(p["common"]) != p["common_sha256"]: raise ValueError("Reused common changed.")
    if {s: sha(p[s]) for s in ("train", "validation")} != p["input_hashes"]: raise ValueError("Observed inputs changed.")
    K = float(common["normalization"]["K"]); overlay = backend.install(K)
    from . import engine
    before_sources = loaded_sources(source, p["source_hashes"])
    data, validation = prepare(observed(p["train"])), prepare(observed(p["validation"]))
    if data.p[0].shape[0] != 96 or validation.p[0].shape[0] != 288: raise ValueError("Keep every original day96 slot.")
    deadline = started+LIMITS["total"]-float(common["seconds"])
    structural_deadline = deadline-LIMITS["post_reserve"]
    class TimedBank(backend.SquaredBank):
        def fit(self, family, *, preserve=False):
            left = deadline-perf_counter()
            if left <= .05: raise TimeoutError("Total pilot evaluation allowance exhausted before a fixed-family QP.")
            self.seconds = min(LIMITS["QP"], left); self.lp.model.Params.TimeLimit = self.seconds
            return super().fit(family, preserve=preserve)
    bank = TimedBank(data, common["pool"], common["calibration"], seconds=LIMITS["QP"])
    try:
        trajectory = engine.run(bank, rnj_supports=common["initial"],
            rnj_warm_families=[point["supports"] for point in common["bank"]], reference=common["reference"],
            deadline=structural_deadline, per_mip=LIMITS["per_mip"], alpha=.003, block_limit=5, hard=(),
            heartbeat=lambda row: write(target.with_name("heartbeat.json"), row))
        structural_seconds = perf_counter()-started
        write(target.with_name("checkpoint.json"), dict(stage="RTE_complete", trajectory=trajectory))
        star = bank.fit([(i,) for i in range(data.n)])
        arrays = lambda d: dict(p=np.vstack(d.p), q=np.vstack(d.q), y=np.vstack(d.target))
        points = [(point, "own_path") for point in trajectory["path"]]+[(star, "singleton")]
        models, library = raw_library(points, train=arrays(data), validation=arrays(validation), K=K, n=data.n)
        selected, secondary = choose(library, "mae"), choose(library, "mse")
        exported = dict(schema="simplified_rte_raw_frozen_models_v1", labels=common["labels"], fit_loss="scaled_mse",
            loss_scale=K, point_objective="K*training_MSE", scoring_loss="physical_MAE", models=models,
            primary_selected_model_id=selected["model_id"], secondary_selected_model_id=secondary["model_id"],
            weights_frozen=True, future_refit_allowed=False, contraction=False,
            full_response_models=True, complete_topology_coverage_claimed=False, complete_RNJ_output_fallback=False,
            input_hashes=p["input_hashes"], source_hashes=p["source_hashes"], truth_used=False, test_opened=False)
        exported["pool_sha256"] = digest(exported); write(target.with_name("candidates.json"), exported)
        after_sources = loaded_sources(source, p["source_hashes"])
        fresh = perf_counter()-started
        result = dict(job=p["job"], method=METHOD, trajectory=trajectory, star=star, output_candidates=library,
            selected=selected, selected_mse_view=secondary, normalization=common["normalization"], reference=common["reference"],
            structural_seconds=structural_seconds, output_seconds=fresh-structural_seconds, fresh_seconds=fresh,
            common_seconds=common["seconds"], total_seconds=fresh+common["seconds"],
            shared_common_reused=True, total_cap_exceeded=fresh+common["seconds"] > LIMITS["total"]+1., limits=LIMITS,
            comparison_scope=p["comparison_scope"], common_sha256=p["common_sha256"], input_hashes=p["input_hashes"],
            protocol_sha256=p["protocol_sha256"], parent_T003_result_sha256=p["parent_T003_result_sha256"],
            QP=bank.lp.solves, QP_failures=bank.failures, runtime_overlay=overlay,
            loaded_sources_before=before_sources, loaded_sources_after=after_sources,
            candidate_export_sha256=sha(target.with_name("candidates.json")), truth_used=False, test_opened=False,
            final_RNJ_fallback=False, output_contraction=False, environment=versions())
        write(target, result)
    finally:
        bank.close()


def status(output):
    p = read(output/"protocol.json")
    returned = sum((output/"jobs"/j/"result.json").exists() for j in p["jobs"])
    failed = sum((output/"jobs"/j/"error.json").exists() and not (output/"jobs"/j/"result.json").exists() for j in p["jobs"])
    return dict(planned=4, returned=returned, failed=failed, pending=4-returned-failed)


def pilot(output, workers=4):
    protocol, catalog = read(output/"protocol.json"), read(output/"catalog.json")
    check_sources(output, protocol)
    if sha(output/"catalog.json") != protocol["catalog_sha256"]: raise ValueError("Pilot catalog changed.")
    if not 1 <= workers <= 4: raise ValueError("Use one to four workers for this pilot.")
    source = output/"source"; parent = Path(protocol["parent_output"]); queue, active = [], {}
    for job, meta in catalog.items():
        target = output/"jobs"/job/"result.json"
        if target.exists() or target.with_name("error.json").exists(): continue
        if target.with_name("attempt.json").exists(): raise RuntimeError("Audit an interrupted attempt before resuming it.")
        queue.append(dict(job=job, result=str(target), source=str(source), train=meta["train"], validation=meta["validation"],
            input_hashes={s: protocol["input_hashes"][job][s] for s in ("train", "validation")},
            common=str(parent/"common"/job/"common.json"), common_sha256=protocol["parent_common_sha256"][job],
            protocol_sha256=sha(output/"protocol.json"), source_hashes=protocol["source_hashes"],
            parent_T003_result_sha256=protocol["parent_T003_result_sha256"][job], comparison_scope=protocol["comparison_scope"]))
    env = os.environ.copy(); env["PYTHONPATH"] = str(source)+os.pathsep+str(source/"rnj_wzzt_core")
    started = perf_counter()
    while queue or active:
        while queue and len(active) < workers:
            p = queue.pop(0); target = Path(p["result"]); file = target.with_name("payload.json")
            write(file, p); log = target.with_name("worker.log").open("w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-B", "-m", PACKAGE, "worker", str(file)], cwd=source, env=env,
                stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            write(target.with_name("attempt.json"), dict(state="running", pid=proc.pid, payload_sha256=sha(file)))
            active[p["job"]] = (proc, log, target)
        for job, (proc, log, target) in list(active.items()):
            if proc.poll() is None: continue
            log.close(); del active[job]
            if not target.exists() and not target.with_name("error.json").exists(): write(target.with_name("error.json"), dict(exit_code=proc.returncode))
            write(target.with_name("attempt.json"), dict(state="finished", exit_code=proc.returncode))
            print("PILOT", job, "OK" if target.exists() else "ERROR", flush=True)
        write(output/"state.json", dict(**status(output), active={j: row[0].pid for j, row in active.items()}, queued=len(queue), wall_seconds=perf_counter()-started))
        if active: sleep(1.)
    write(output/"completion.json", dict(**status(output), wall_seconds=perf_counter()-started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "pilot", "worker", "status")); parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path, default=PARENT); parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(); output = args.output.resolve()
    try:
        if args.command == "freeze": freeze(output, parent=args.parent, workers=args.workers)
        elif args.command == "pilot": pilot(output, args.workers)
        elif args.command == "status": print(json.dumps(status(output)))
        else: worker(output)
    except Exception:
        if args.command == "worker": write(Path(read(output)["result"]).with_name("error.json"), dict(traceback=traceback.format_exc()))
        raise
