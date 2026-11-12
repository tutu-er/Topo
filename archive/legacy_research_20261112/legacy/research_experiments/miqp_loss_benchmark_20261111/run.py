"""Isolated scaled-MSE/MIQP comparison on the unchanged 48 day96 inputs.

No data generation, test scoring, truth access, or complete-RNJ output fallback.
The frozen L1 source bytes are retained and patched explicitly at runtime only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter, sleep
import traceback

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "outputs/day96_t_benchmark_20261110"
PACKAGE = "research_experiments.miqp_loss_benchmark_20261111.run"
LIMITS = dict(total=14400., prefix=7200., per_mip=3600., QP=300., LP=300.,
              post_reserve=600., mip_gap=.0001, threads=1)
METHODS = [dict(id="noT", tail=False, tail_coefficient=.003),
           dict(id="T003", tail=True, tail_coefficient=.003)]


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
    """Retry Windows reader-lock conflicts without changing the payload."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    for attempt in range(100):
        try:
            temporary.write_text(serialized, encoding="utf-8", newline="\n")
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            sleep(.05)


def versions():
    dependencies = {}
    for name in ("numpy", "scipy", "pandas", "gurobipy", "networkx"):
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = "not-installed"
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                dependencies=dependencies)


def check_sources(output, protocol):
    for name, digest in protocol["source_hashes"].items():
        if sha(output/"source"/name) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    for name, digest in protocol["source_artifact_hashes"].items():
        if sha(output/"source"/name) != digest:
            raise ValueError(f"Frozen package artifact changed: {name}")
    parent = Path(protocol["parent_output"])
    if sha(parent/"protocol.json") != protocol["parent_protocol_sha256"]:
        raise ValueError("Parent protocol changed.")
    if sha(parent/"catalog.json") != protocol["parent_catalog_sha256"]:
        raise ValueError("Parent catalog changed.")
    for job, digest in protocol["parent_common_hashes"].items():
        if sha(parent/"common"/job/"common.json") != digest:
            raise ValueError(f"Parent common preprocessing changed: {job}")


def freeze(output, *, parent=PARENT, workers=8):
    """Freeze the parent source and input references; do not generate or fit."""
    if output.exists():
        raise FileExistsError("A new immutable output directory is required.")
    if not 1 <= workers <= 8:
        raise ValueError("Use 1..8 workers, each with one native solver thread.")
    parent = Path(parent).resolve()
    old, catalog = read(parent/"protocol.json"), read(parent/"catalog.json")
    if len(catalog) != 48 or old["planned_inputs"] != 48:
        raise ValueError("The fixed 48-input day96 parent is required.")
    for name, digest in old["source_hashes"].items():
        if sha(parent/"source"/name) != digest:
            raise ValueError(f"Parent frozen source changed: {name}")
    for job, meta in catalog.items():
        if meta["data_error"]:
            continue
        for split in ("train", "validation", "test"):
            if sha(meta[split]) != old["input_hashes"][job][split]:
                raise ValueError(f"Parent observations changed: {job}/{split}")
    commons = {j: sha(parent/"common"/j/"common.json") for j in catalog
               if not catalog[j]["data_error"]}
    required = ("__init__.py", "run.py", "candidates.py", "squared_backend.py",
                "analyze.py", "verify_backend.py", "README.md")
    if any(not (Path(__file__).parent/name).is_file() for name in required):
        raise FileNotFoundError("Complete the isolated MIQP package before freezing.")
    output.mkdir(parents=True)
    source = output/"source"
    shutil.copytree(parent/"source", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    new = source/"research_experiments/miqp_loss_benchmark_20261111"
    shutil.copytree(Path(__file__).parent, new, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    marker = source/"research_experiments/__init__.py"
    if not marker.exists():
        marker.write_text('"""Frozen research package for the isolated MIQP benchmark."""\n', encoding="utf-8", newline="\n")
    # Every parent byte remains unchanged. Only a new package is added.
    for name, digest in old["source_hashes"].items():
        if sha(source/name) != digest:
            raise AssertionError(f"Copy altered a frozen parent source: {name}")
    source_hashes = {p.relative_to(source).as_posix(): sha(p) for p in sorted(source.rglob("*.py"))}
    source_artifacts = {p.relative_to(source).as_posix(): sha(p) for p in sorted(new.rglob("*")) if p.is_file()}
    write(output/"catalog.json", catalog)
    configuration = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), methods=METHODS, limits=LIMITS,
        workers=workers, planned_inputs=48, planned=96, bundles=4, fit_loss="scaled_mse",
        primary_validation_loss="mae", secondary_validation_loss="scaled_mse_same_output_library",
        loss_formula="K*mean((Y-P@R.T-Q@X.T)^2); K=f0_MAE/f0_MSE",
        normalization_reference="same complete RNJ plus singleton/full family, bounded QP [0,2]",
        base_coefficient=.003, exit_coefficient=.003, resolution=.24,
        count_penalty_scale="f0_MAE from parent; K*f0_MSE=f0_MAE",
        source_hashes=source_hashes, source_artifact_hashes=source_artifacts,
        parent_output=str(parent), parent_protocol_sha256=sha(parent/"protocol.json"),
        parent_catalog_sha256=sha(parent/"catalog.json"), parent_source_hashes=old["source_hashes"],
        parent_source_file_count=len(old["source_hashes"]), parent_common_hashes=commons,
        added_source_files=sorted(set(source_hashes)-set(old["source_hashes"])),
        added_root_package_marker="research_experiments/__init__.py",
        input_semantics=old["input_semantics"], input_hashes=old["input_hashes"],
        observations_regenerated=False, parent_source_bytes_modified=False,
        common_cost="parent common.seconds, including original MAE references/warm fits, plus new reference and warm QP costs",
        seed_policy="same RNJ geometry/initial; deletion rankings recomputed with scaled MSE",
        warm_start_policy="same parent warm families, all refitted by QP; no LP weights reused",
        warm_family_contract="same parent-provided support families; parent preprocessing frozen",
        no_complete_RNJ_output_fallback=True, zero_release=True, hard_priors=[], positive_impedance_floor=None,
        physics=old.get("physics"), truth_used_in_training=False, test_opened_by_worker=False,
        primary_independent_blocks="4 shared seed_index groups, each carrying all cases and voltage-noise levels",
        runtime_overlay="squared_backend.install(K), with per-worker alias/hash manifest",
        environment=versions(), nominal_method_hours=96*4., retuning=False)
    write(output/"configuration.json", configuration)
    protocol = dict(configuration, jobs=list(catalog), catalog_sha256=sha(output/"catalog.json"), training_started=False)
    write(output/"protocol.json", protocol)
    write(output/"state.json", dict(stage="frozen", planned=96, returned=0))
    print("FROZEN 48 unchanged parent inputs; 96 MIQP configurations; no fits started", flush=True)


def _point_scores(data, point, K, backend):
    """Append physical units without redefining native objective semantics."""
    point = deepcopy(point)
    train_mae, train_mse = backend.mae(data, point), backend.mse(data, point)
    if abs(float(point["objective"])-K*train_mse) > 1e-9+1e-7*abs(K*train_mse):
        raise ValueError("Native objective is inconsistent with K times MSE.")
    point.update(train_mae=train_mae, train_mse=train_mse,
                 train_sse=train_mse*data.observation_count,
                 native_internal_objective=float(point["objective"]),
                 loss_scale=K, fit_loss="scaled_mse", objective_kind="K*MSE", loss_metric="normalized_mse")
    return point


def _bind_frozen(payload):
    """Resolve every native research import from this new source snapshot."""
    source = Path(payload["frozen_source"]).resolve()
    if not Path(__file__).resolve().is_relative_to(source):
        raise RuntimeError("Workers must execute the new frozen driver, not the live checkout.")
    if sha(Path(__file__)) != payload["source_hashes"].get(
            "research_experiments/miqp_loss_benchmark_20261111/run.py"):
        raise RuntimeError("The running driver does not match the frozen source hash.")
    import research_experiments
    research_experiments.__path__ = [str(source/"research_experiments")]
    for name in ("rnj_wzzt", "research_experiments.rebuild_benchmark.engine",
                 "research_experiments.rnj_count.engine", "research_experiments.joint_exit.model",
                 "research_experiments.rnj_small_seed.engine", "research_experiments.revocable.engine",
                 "research_experiments.lp_rebuild.engine"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(source):
                raise RuntimeError(f"A native module was already imported from outside the frozen source: {name}")
    return source


def _checked_install(backend, K, payload):
    source = Path(payload["frozen_source"]).resolve()
    manifest = backend.install(K)
    for name, row in manifest["native_module_sources"].items():
        path = Path(row["path"]).resolve()
        if not path.is_relative_to(source):
            raise RuntimeError(f"Runtime native alias escaped the frozen source: {name}")
        relative = path.relative_to(source).as_posix()
        if row["sha256"] != payload["source_hashes"].get(relative):
            raise RuntimeError(f"Runtime native source hash does not match protocol: {name}")
    if sha(Path(backend.__file__)) != payload["source_hashes"].get(
            "research_experiments/miqp_loss_benchmark_20261111/squared_backend.py"):
        raise RuntimeError("The squared backend does not match its frozen source hash.")
    return dict(manifest, frozen_source=str(source), frozen_native_sources_verified=True)


def _save_loaded_sources(payload, stage):
    """Check all loaded project modules, including core and export helpers."""
    source, rows = Path(payload["frozen_source"]).resolve(), {}
    for name, module in sorted(sys.modules.items()):
        if not (name == "research_experiments" or name.startswith("research_experiments.") or
                name == "rnj_wzzt" or name.startswith("rnj_wzzt.")):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(source):
            raise RuntimeError(f"A loaded project module escaped frozen source: {name}: {path}")
        relative, measured = path.relative_to(source).as_posix(), sha(path)
        if payload["source_hashes"].get(relative) != measured:
            raise RuntimeError(f"A loaded project module differs from its frozen hash: {name}")
        rows[name] = dict(path=str(path), relative_path=relative, sha256=measured)
    receipt = dict(stage=stage, checked_utc=datetime.now(timezone.utc).isoformat(),
                   frozen_source=str(source), passed=True, loaded_modules=rows)
    target = Path(payload["result"]).with_name(f"loaded_sources_{stage}.json")
    write(target, receipt)
    return dict(file=str(target), sha256=sha(target), modules=len(rows), passed=True)


def common_worker(payload):
    started = perf_counter()
    p = read(payload)
    _bind_frozen(p)
    from research_experiments.day96_t_benchmark_20261110.data import observed
    from research_experiments.rebuild_benchmark.engine import prepare
    from research_experiments.local_rebuild.model import canonical
    from . import squared_backend as backend
    from .candidates import digest
    parent = read(p["parent_common"])
    if sha(p["parent_common"]) != p["parent_common_sha256"]:
        raise ValueError("Parent common preprocessing hash mismatch.")
    if {s: sha(p[s]) for s in ("train", "validation")} != p["input_hashes"]:
        raise ValueError("Observations changed.")
    data = prepare(observed(p["train"]))
    if sum(len(a) for a in data.p) != 96:
        raise ValueError("The parent training input must contain all 96 slots.")
    family = canonical([*parent["initial"], *((i,) for i in range(data.n)), tuple(range(data.n))])
    if canonical(parent["reference_point"]["supports"]) != family:
        raise ValueError("Parent reference does not use the declared full RNJ family.")
    deadline = started+LIMITS["total"]-float(parent["seconds"])-LIMITS["post_reserve"]
    if deadline-perf_counter() <= .05:
        raise TimeoutError("Parent shared preprocessing already consumed the available budget.")
    before_sources = _save_loaded_sources(p, "before")
    ref = backend.reference_fit(data, family, seconds=min(LIMITS["QP"], max(.05, deadline-perf_counter())))
    f0_mae, f0_mse = float(parent["reference"]), float(backend.mse(data, ref))
    if not (math.isfinite(f0_mae) and math.isfinite(f0_mse) and f0_mae > 0 and f0_mse > 0):
        raise ValueError("This fixed noisy benchmark requires positive finite MAE and MSE references.")
    K = f0_mae/f0_mse
    normalization = dict(f0_mae=f0_mae, f0_mse=f0_mse, K=K,
                         reference_family_sha256=digest(family), reference_bounds=[0., 2.])
    manifest = _checked_install(backend, K, p)
    write(Path(p["result"]).with_name("runtime_overlay.json"), manifest)
    bank = backend.SquaredBank(data, parent["pool"], parent["calibration"], seconds=LIMITS["QP"])
    try:
        points = []
        for old in parent["bank"]:
            left = deadline-perf_counter()
            if left <= .05:
                raise TimeoutError("Shared QP warm preprocessing exhausted the method budget.")
            bank.seconds = min(LIMITS["QP"], left)
            bank.lp.model.Params.TimeLimit = bank.seconds
            point = bank.fit(old["supports"], preserve=True)
            points.append(_point_scores(data, point, K, backend))
        raw_reference = deepcopy(ref)
        raw_reference["objective_kind"] = "physical_mse"
        reference = deepcopy(ref); reference["objective"] = K*f0_mse
        reference = _point_scores(data, reference, K, backend)
        after_sources = _save_loaded_sources(p, "after")
        fresh = perf_counter()-started
        result = dict(job=p["job"], labels=parent["labels"], pool=parent["pool"], initial=parent["initial"],
                      calibration=parent["calibration"], kernel=parent["kernel"], bank=points,
                      reference=f0_mae, reference_point=reference,
                      reference_qp_point_physical_mse=raw_reference, normalization=normalization,
                      fit_loss="scaled_mse", seed_ranking_loss="scaled_mse",
                      parent_common_seconds=float(parent["seconds"]), qp_common_seconds=fresh,
                      seconds=float(parent["seconds"])+fresh,
                      parent_common_sha256=p["parent_common_sha256"], parent_common=p["parent_common"],
                      source_input_hashes=p["input_hashes"], truth_used=False, test_opened=False,
                      QP=bank.lp.solves, QP_failures=bank.failures, LP=[], LP_failures=[],
                      runtime_overlay=manifest, environment=versions(),
                      loaded_source_receipts=dict(before=before_sources, after=after_sources),
                      warm_family_hashes=[digest(canonical(x["supports"])) for x in parent["bank"]],
                      warm_family_contract="same parent-provided support families; parent preprocessing frozen",
                      warm_weights_reused=False, protocol_sha256=p["protocol_sha256"])
        write(p["result"], result)
    finally:
        bank.close()


def worker(payload):
    started = perf_counter()
    p = read(payload)
    _bind_frozen(p)
    import numpy as np
    from . import squared_backend as backend
    from .candidates import export_candidates, verify_export
    from research_experiments.day96_t_benchmark_20261110.data import observed
    from research_experiments.rebuild_benchmark.engine import prepare
    from research_experiments.local_rebuild.model import canonical
    target = Path(p["result"]); common = read(p["common"])
    if sha(p["common"]) != p["common_sha256"]:
        raise ValueError("QP common preprocessing changed.")
    if {s: sha(p[s]) for s in ("train", "validation")} != p["input_hashes"]:
        raise ValueError("Observations changed.")
    K = float(common["normalization"]["K"])
    manifest = _checked_install(backend, K, p)
    write(target.with_name("runtime_overlay.json"), manifest)
    from research_experiments.joint_exit.selection import library, choose
    from research_experiments.full_ablation.engine import search, finish
    data, validation = prepare(observed(p["train"])), prepare(observed(p["validation"]))
    if sum(len(a) for a in data.p) != 96 or sum(len(a) for a in validation.p) != 288:
        raise ValueError("Training and validation must retain all parent continuous-day slots.")
    deadline = started+LIMITS["total"]-common["seconds"]
    structural = deadline-LIMITS["post_reserve"]
    class TimedBank(backend.SquaredBank):
        def fit(self, family, *, preserve=False):
            left = deadline-perf_counter()
            if left <= .05:
                raise TimeoutError("Total method budget exhausted before fixed-family QP.")
            self.seconds = min(LIMITS["QP"], left)
            self.lp.model.Params.TimeLimit = self.seconds
            return super().fit(family, preserve=preserve)
    bank = TimedBank(data, common["pool"], common["calibration"], seconds=LIMITS["QP"])
    try:
        before_sources = _save_loaded_sources(p, "before")
        bank.seed_start_pool = canonical(common["pool"])
        bank.joint_exit_seed_pool = canonical(common["pool"])
        heartbeat = lambda event: write(target.with_name("progress.json"), event)
        prefix_deadline = perf_counter()+min(LIMITS["prefix"], max(.1, (structural-perf_counter())/2))
        trajectory = search(bank, common, reference=common["reference"], seed_on=True, tail_on=p["flags"]["tail"],
                            tail_coefficient=p["flags"]["tail_coefficient"], deadline=structural,
                            prefix_deadline=prefix_deadline, per_mip=LIMITS["per_mip"],
                            zero_release=True, heartbeat=heartbeat)
        write(target.with_name("checkpoint.json"), dict(stage="prefix_complete", trajectory=trajectory))
        trajectory, exit_record, exit_event = finish(bank, common, trajectory, reference=common["reference"],
                                                     deadline=structural, exit_on=True, rnj_guidance=True,
                                                     zero_release=True, heartbeat=heartbeat)
        structural_seconds = perf_counter()-started
        write(target.with_name("checkpoint.json"), dict(stage="structure_complete", trajectory=trajectory,
                                                        exit_record=exit_record, exit_event=exit_event))
        trajectory["path"] = [_point_scores(data, x, K, backend) for x in trajectory["path"]]
        trajectory["point"] = _point_scores(data, trajectory["point"], K, backend)
        if isinstance(exit_record.get("point"), dict):
            exit_record["point"] = _point_scores(data, exit_record["point"], K, backend)
        star = _point_scores(data, bank.fit([(i,) for i in range(data.n)]), K, backend)
        rows = library(bank, [*trajectory["path"], star], validation, resolution=.24)
        for row in rows:
            row["point"] = _point_scores(data, row["point"], K, backend)
            val_mse = float(backend.mse(validation, row["point"]))
            row.update(source="own_path" if row["index"] < len(trajectory["path"]) else "singleton",
                       validation_mae=float(row["validation"]), validation_mse=val_mse,
                       validation_sse=val_mse*validation.observation_count,
                       validation_scaled_mse=K*val_mse)
        selected = choose(rows, "strict")
        squared_rows = [dict(row, validation=row["validation_scaled_mse"]) for row in rows]
        selected_mse_view = choose(squared_rows, "strict")
        # Preserve the old validation=MAE field on every saved row, even when
        # the secondary selector used its scaled-MSE view of that same row.
        squared_meta = dict(rule="strict_scaled_mse", eligible_count=selected_mse_view["eligible_count"],
                            best_validation_scaled_mse=selected_mse_view["best_validation"])
        selected_mse_view = dict(rows[selected_mse_view["index"]], **squared_meta)
        provenance = dict(weights_fit_split="train", truth_used_in_fitting=False, test_used_in_generation=False,
                          future_DR_used=False, input_hashes=p["input_hashes"], source_hashes=p["source_hashes"],
                          protocol_sha256=p["protocol_sha256"], method=p["method"], job=p["job"],
                          fit_loss="scaled_mse", historical_score="mae")
        points = [dict(point=point, source="own_path", stage="own_path", path_index=i)
                  for i, point in enumerate(trajectory["path"])]
        points.append(dict(point=star, source="singleton", stage="singleton", path_index=len(trajectory["path"])))
        points.extend(dict(point=row["point"], source=row["source"], stage="output_refit_024", path_index=row["index"])
                      for row in rows)
        arrays = lambda d: dict(p=np.vstack(d.p), q=np.vstack(d.q), y=np.vstack(d.target))
        exported = export_candidates(points, selected=selected, selected_mse_view=selected_mse_view,
                                     normalization=common["normalization"], labels=common["labels"],
                                     train=arrays(data), validation=arrays(validation), provenance=provenance,
                                     calibration=common["calibration"], root_label=p["root_label"], base_kw=p["base_kw"])
        verified = verify_export(exported, train=arrays(data), validation=arrays(validation))
        write(target.with_name("candidates.json"), exported)
        after_sources = _save_loaded_sources(p, "after")
        fresh = perf_counter()-started
        view = lambda row: dict(selected_index=row["index"], validation_mae=row["validation_mae"],
                                validation_mse=row["validation_mse"], validation_scaled_mse=row["validation_scaled_mse"])
        result = dict(job=p["job"], method=p["method"], flags=p["flags"], fit_loss="scaled_mse",
                      normalization=common["normalization"], selected=selected, selected_mse_view=selected_mse_view,
                      views=dict(primary_mae=view(selected), secondary_mse=view(selected_mse_view)),
                      trajectory=trajectory, exit_record=exit_record, exit_event=exit_event, output_candidates=rows,
                      structural_seconds=structural_seconds, output_seconds=fresh-structural_seconds,
                      fresh_seconds=fresh, common_seconds=common["seconds"], total_seconds=fresh+common["seconds"],
                      successful_total_seconds=fresh+common["seconds"], operational_total_seconds=fresh+common["seconds"],
                      parent_common_seconds=common["parent_common_seconds"], qp_common_seconds=common["qp_common_seconds"],
                      total_cap_exceeded=fresh+common["seconds"] > LIMITS["total"]+1., limits=LIMITS,
                      reference=common["reference"], input_hashes=p["input_hashes"], common_sha256=p["common_sha256"],
                      protocol_sha256=p["protocol_sha256"], parent=p["parent"],
                      truth_used=False, test_opened=False, QP_failures=bank.failures, LP_failures=[],
                      QP=bank.lp.solves, final_RNJ_fallback=False,
                      runtime_overlay=manifest, runtime_overlay_sha256=sha(target.with_name("runtime_overlay.json")),
                      loaded_source_receipts=dict(before=before_sources, after=after_sources),
                      environment=versions(), candidate_export_sha256=sha(target.with_name("candidates.json")),
                      candidate_export_verification=verified)
        write(target, result)
    finally:
        bank.close()


def status(output):
    protocol = read(output/"protocol.json")
    rows = [dict(result=(output/"jobs"/j/m["id"]/"result.json").exists(),
                 error=(output/"jobs"/j/m["id"]/"error.json").exists())
            for j in protocol["jobs"] for m in protocol["methods"]]
    return dict(planned=protocol["planned"], returned=sum(r["result"] for r in rows),
                failed=sum(r["error"] and not r["result"] for r in rows),
                pending=sum(not r["result"] and not r["error"] for r in rows))


def _run_queue(output, queue, *, stage, phase, workers, source, env):
    active, started = {}, perf_counter()
    for payload in queue:
        attempt = Path(payload["result"]).with_name("attempt.json")
        if attempt.exists() and read(attempt).get("state") != "finished":
            raise RuntimeError(f"Unresolved earlier worker attempt requires an explicit budget audit: {attempt}")
    while queue or active:
        while queue and len(active) < workers:
            payload = queue.pop(0); target = Path(payload["result"])
            file, attempt = target.with_name("payload.json"), target.with_name("attempt.json")
            if attempt.exists() and read(attempt).get("state") != "finished":
                raise RuntimeError(f"Unresolved earlier worker attempt requires an explicit budget audit: {attempt}")
            write(file, payload)
            log = target.with_name("worker.log").open("w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-B", "-m", PACKAGE, stage, str(file)],
                                    cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            write(attempt, dict(state="running", pid=proc.pid, started_utc=datetime.now(timezone.utc).isoformat(),
                                result=str(target), payload_sha256=sha(file), stage=stage))
            active[str(target)] = (proc, log, target, attempt)
        for key, (proc, log, target, attempt) in list(active.items()):
            if proc.poll() is None:
                continue
            log.close(); del active[key]
            if not target.exists() and not target.with_name("error.json").exists():
                write(target.with_name("error.json"), dict(stage=stage, exit_code=proc.returncode))
            record = read(attempt)
            write(attempt, dict(record, state="finished", exit_code=proc.returncode,
                                finished_utc=datetime.now(timezone.utc).isoformat()))
            print("FINISHED", stage, target.parent.name, "OK" if target.exists() else "ERROR", flush=True)
        write(output/f"{phase}_{stage}_state.json", dict(**status(output), phase=phase, stage=stage,
              active=[dict(result=k, pid=v[0].pid) for k, v in active.items()],
              queued=len(queue), wall_seconds=perf_counter()-started))
        if active:
            sleep(2.)


def dispatch(output, *, phase="pilot", workers=None):
    protocol, catalog = read(output/"protocol.json"), read(output/"catalog.json")
    check_sources(output, protocol)
    if sha(output/"catalog.json") != protocol["catalog_sha256"]:
        raise ValueError("Frozen input catalog changed.")
    jobs = [j for j in protocol["jobs"] if phase == "all" or
            (catalog[j]["seed_index"] == 0 and catalog[j]["voltage_sigma"] == .0005)]
    source, env = output/"source", os.environ.copy()
    env["PYTHONPATH"] = str(source)+os.pathsep+str(source/"rnj_wzzt_core")
    count = workers or protocol["workers"]
    if not 1 <= count <= 8:
        raise ValueError("Use 1..8 workers, each with one native solver thread.")
    common_queue = []
    for job in jobs:
        meta = catalog[job]
        if meta["data_error"]:
            for method in protocol["methods"]:
                write(output/"jobs"/job/method["id"]/"error.json", dict(stage="data", error=meta["data_error"]))
            continue
        hashes = {s: protocol["input_hashes"][job][s] for s in ("train", "validation")}
        common = output/"common"/job/"common.json"
        if not common.exists() and not common.with_name("error.json").exists():
            parent_common = Path(protocol["parent_output"])/"common"/job/"common.json"
            common_queue.append(dict(job=job, train=meta["train"], validation=meta["validation"], input_hashes=hashes,
                                     parent_common=str(parent_common), parent_common_sha256=protocol["parent_common_hashes"][job],
                                     result=str(common), protocol_sha256=sha(output/"protocol.json"),
                                     frozen_source=str(source), source_hashes=protocol["source_hashes"]))
    _run_queue(output, common_queue, stage="common", phase=phase, workers=count, source=source, env=env)
    queue = []
    for job in jobs:
        meta, common = catalog[job], output/"common"/job/"common.json"
        if not common.exists():
            for flags in protocol["methods"]:
                target = output/"jobs"/job/flags["id"]/"error.json"
                if not target.exists():
                    write(target, dict(stage="common", error="Shared QP preprocessing unavailable; denominator retained."))
            continue
        for flags in protocol["methods"]:
            target = output/"jobs"/job/flags["id"]/"result.json"
            if target.exists() or target.with_name("error.json").exists():
                continue
            parent = dict(output=protocol["parent_output"], protocol_sha256=protocol["parent_protocol_sha256"],
                          common_sha256=protocol["parent_common_hashes"][job],
                          input_hashes=protocol["input_hashes"][job], source_file_count=protocol["parent_source_file_count"])
            queue.append(dict(job=job, method=flags["id"], flags=flags, train=meta["train"], validation=meta["validation"],
                              common=str(common), common_sha256=sha(common),
                              input_hashes={s: protocol["input_hashes"][job][s] for s in ("train", "validation")},
                              result=str(target), protocol_sha256=sha(output/"protocol.json"), source_hashes=protocol["source_hashes"],
                              root_label=meta["root_label"], base_kw=meta["base_kw"], parent=parent, frozen_source=str(source)))
    _run_queue(output, queue, stage="worker", phase=phase, workers=count, source=source, env=env)
    write(output/f"{phase}_completion.json", dict(**status(output), phase=phase, phase_inputs=len(jobs),
                                                 phase_planned=len(jobs)*len(protocol["methods"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "common", "worker", "pilot", "all", "status"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(); path = args.path.resolve()
    try:
        if args.command == "freeze":
            freeze(path, parent=args.parent, workers=args.workers)
        elif args.command in ("pilot", "all"):
            dispatch(path, phase=args.command, workers=args.workers)
        elif args.command == "status":
            print(json.dumps(status(path), indent=2))
        else:
            {"common": common_worker, "worker": worker}[args.command](path)
    except Exception:
        if args.command in ("worker", "common"):
            write(Path(read(path)["result"]).with_name("error.json"), dict(stage=args.command, traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
