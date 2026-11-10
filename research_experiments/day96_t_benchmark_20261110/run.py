"""Fresh continuous-day96 paired T ablation; sources/data frozen before fits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
PARENT = ROOT / "outputs/full_ablation_20261102"
LIMITS = dict(total=14400., prefix=7200., per_mip=3600., LP=300., post_reserve=600., mip_gap=.0001, threads=1)
METHODS = [dict(id="noT", tail=False, tail_coefficient=.003),
           dict(id="T003", tail=True, tail_coefficient=.003),
           dict(id="T010", tail=True, tail_coefficient=.01)]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def freeze(output, *, workers=8, bundles=4):
    """Snapshot and generate observations only; common LP fits are a later stage."""
    if output.exists():
        raise FileExistsError("New immutable output directory required")
    parent = read(PARENT / "protocol.json")
    for name, digest in parent["source_hashes"].items():
        assert sha(PARENT / "source" / name) == digest, name
    output.mkdir(parents=True)
    source = output / "source"
    shutil.copytree(PARENT / "source", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    new_dir = source / "research_experiments/day96_t_benchmark_20261110"
    shutil.copytree(Path(__file__).parent, new_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    file = source / "research_experiments/full_ablation/engine.py"
    text, before = file.read_text(encoding="utf-8"), sha(file)
    assert text.count("heartbeat=None):") == 2
    text = text.replace("heartbeat=None):", "heartbeat=None, tail_coefficient=.003):", 1)
    old = "current, event = joint_fit(bank, current, keep, penalty=penalty,"
    assert text.count(old) == 1
    text = text.replace(old, "current, event = joint_fit(bank, current, keep, penalty=float(tail_coefficient)*reference,")
    old = "reference_MAE=reference, coefficient=.003, penalty=penalty,"
    assert text.count(old) == 1
    text = text.replace(old, "reference_MAE=reference, coefficient=.003, penalty=penalty,\n                tail_coefficient=float(tail_coefficient), tail_penalty=float(tail_coefficient)*reference,")
    file.write_text(text, encoding="utf-8", newline="\n")
    source_hashes = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob("*.py")}
    configuration = dict(created_utc=datetime.now(timezone.utc).isoformat(), methods=METHODS, limits=LIMITS,
                         workers=workers, bundles=bundles, planned_inputs=4*bundles*3, planned=4*bundles*3*3,
                         input_semantics="one continuous day with all96 quarter-hour slots; validation3 independent days288; testone independent day96",
                         base_coefficient=.003, exit_coefficient=.003, resolution=.24, count_penalty_scale="f0=complete RNJ initial+singleton/full training LP MAE",
                         no_complete_RNJ_output_fallback=True, zero_release=True, hard_priors=[], positive_impedance_floor=None,
                         source_hashes=source_hashes, parent_protocol_sha256=sha(PARENT / "protocol.json"),
                         source_overlay=dict(file=str(file.relative_to(source)), before_sha256=before, after_sha256=sha(file),
                                             change="explicit tail coefficient only; ordinary WZZT and final exit remain .003"),
                         seed_policy="train replicate10000+10*seed_index; validation+1; test+2; within eachcase these are fresh versus oldbundle0..9",
                         physics=dict(suite="reference", impedance_scale=1., load_scale=1., pq_sigma=.005,
                                      voltage_sigmas=[.0002,.0005,.001], root_physical_sigma=.003),
                         truth_used_in_training=False, test_opened_by_worker=False, retuning=False)
    write(output / "configuration.json", configuration)
    env = os.environ.copy(); env["PYTHONPATH"] = str(source) + os.pathsep + str(source / "rnj_wzzt_core")
    command = [sys.executable, "-B", "-m", "research_experiments.day96_t_benchmark_20261110.run", "generate", str(output)]
    subprocess.run(command, cwd=source, env=env, check=True)
    catalog = read(output / "catalog.json")
    hashes = {j: {k: sha(m[k]) for k in ("train", "validation", "test")} for j, m in catalog.items() if not m["data_error"]}
    protocol = dict(configuration, jobs=list(catalog), input_hashes=hashes,
                    physical_seeds_sha256=sha(output / "physical_seeds.json"), catalog_sha256=sha(output / "catalog.json"),
                    training_started=False, nominal_method_hours=configuration["planned"]*4.)
    write(output / "protocol.json", protocol)
    write(output / "state.json", dict(stage="frozen", planned=protocol["planned"], returned=0))
    print("FROZEN", len(catalog), "inputs", protocol["planned"], "fresh method evaluations", flush=True)


def generate(output):
    from .data import generate as make
    make(output, bundles=read(output / "configuration.json")["bundles"])


def common_worker(payload):
    from .data import observed
    from research_experiments.rebuild_benchmark.engine import prepare
    from research_experiments.wzzt_retune.engine import build_bank
    from research_experiments.wzzt_refine.core import regularized_pool
    from research_experiments.lp_rebuild.engine import FloorBank
    from research_experiments.local_rebuild.model import canonical
    import numpy as np
    p = read(payload); started = perf_counter()
    assert {s: sha(p[s]) for s in ("train", "validation")} == p["input_hashes"]
    obs = observed(p["train"]); assert len(obs[0]) == 96
    data = prepare(obs)
    members, pool, fronts, calibrations = build_bank(obs)
    augmented = regularized_pool(obs, pool)
    initial = canonical(next(m["family"] for m in members if m["method"] == "RNJ" and m["config"] == "raw_RX75_0.16"))
    cal = calibrations["raw"]
    R, X = np.asarray(fronts["raw"]["R"]), np.asarray(fronts["raw"]["X"])
    fits = FloorBank(data, augmented["pool"], cal, seconds=LIMITS["LP"])
    try:
        points = [fits.fit(m["family"]) for m in members if m["method"] == "RNJ" and m["preprocess"] == "raw"]
        full_initial = canonical([*initial, *((i,) for i in range(data.n)), tuple(range(data.n))])
        raw = fits.fit(full_initial, preserve=True)
        result = dict(job=p["job"], labels=list(map(int, obs[0].columns)), pool=augmented["pool"], initial=initial,
                      calibration=cal, kernel=.75*R/cal["mu_r"]+.25*X/cal["mu_x"], bank=points,
                      reference=float(raw["objective"]), reference_point=raw, seconds=perf_counter()-started,
                      source_input_hashes=p["input_hashes"], truth_used=False, test_opened=False,
                      LP=fits.lp.solves, LP_failures=fits.failures)
        result["kernel"] = result["kernel"].tolist()
        write(p["result"], result)
    finally:
        fits.close()


def worker(payload):
    import numpy as np
    from .data import observed
    from .candidates import export_candidates, verify_export
    from research_experiments.rebuild_benchmark.engine import prepare
    from research_experiments.lp_rebuild.engine import FloorBank
    from research_experiments.local_rebuild.model import canonical
    from research_experiments.joint_exit.selection import library, choose
    from research_experiments.full_ablation.engine import search, finish
    p = read(payload); target = Path(p["result"]); started = perf_counter()
    common = read(p["common"])
    assert sha(p["common"]) == p["common_sha256"]
    assert {s: sha(p[s]) for s in ("train", "validation")} == p["input_hashes"]
    data, validation = prepare(observed(p["train"])), prepare(observed(p["validation"]))
    assert sum(len(a) for a in data.p) == 96 and sum(len(a) for a in validation.p) == 288
    deadline = started+LIMITS["total"]-common["seconds"]
    structural = deadline-LIMITS["post_reserve"]
    class TimedBank(FloorBank):
        def fit(self, family, *, preserve=False):
            left = deadline-perf_counter()
            if left <= .05:
                raise TimeoutError("Total method budget exhausted before fixed family LP")
            self.seconds = min(LIMITS["LP"], left); self.lp.model.Params.TimeLimit = self.seconds
            return super().fit(family, preserve=preserve)
    bank = TimedBank(data, common["pool"], common["calibration"], seconds=LIMITS["LP"])
    try:
        bank.seed_start_pool = canonical(common["pool"]); bank.joint_exit_seed_pool = canonical(common["pool"])
        heartbeat = lambda event: write(target.with_name("progress.json"), event)
        prefix_deadline = perf_counter()+min(LIMITS["prefix"], max(.1, (structural-perf_counter())/2))
        trajectory = search(bank, common, reference=common["reference"], seed_on=True, tail_on=p["flags"]["tail"],
                            tail_coefficient=p["flags"]["tail_coefficient"], deadline=structural,
                            prefix_deadline=prefix_deadline, per_mip=LIMITS["per_mip"], zero_release=True, heartbeat=heartbeat)
        write(target.with_name("checkpoint.json"), dict(stage="prefix_complete", trajectory=trajectory))
        trajectory, exit_record, exit_event = finish(bank, common, trajectory, reference=common["reference"],
                                                     deadline=structural, exit_on=True, rnj_guidance=True,
                                                     zero_release=True, heartbeat=heartbeat)
        structural_seconds = perf_counter()-started
        write(target.with_name("checkpoint.json"), dict(stage="structure_complete", trajectory=trajectory,
                                                        exit_record=exit_record, exit_event=exit_event))
        star = bank.fit([(i,) for i in range(data.n)])
        rows = library(bank, [*trajectory["path"], star], validation, resolution=.24)
        for row in rows:
            row["source"] = "own_path" if row["index"] < len(trajectory["path"]) else "singleton"
        selected = choose(rows, "strict")
        provenance = dict(weights_fit_split="train", truth_used_in_fitting=False, test_used_in_generation=False,
                          future_DR_used=False, input_hashes=p["input_hashes"], source_hashes=p["source_hashes"],
                          protocol_sha256=p["protocol_sha256"], method=p["method"], job=p["job"])
        points = [dict(point=point, source="own_path", stage="own_path", path_index=i)
                  for i, point in enumerate(trajectory["path"])]
        points.append(dict(point=star, source="singleton", stage="singleton", path_index=len(trajectory["path"])))
        arrays = lambda d: dict(p=np.vstack(d.p), q=np.vstack(d.q), y=np.vstack(d.target))
        exported = export_candidates(points, labels=common["labels"], train=arrays(data), validation=arrays(validation),
                                     provenance=provenance, calibration=common["calibration"], root_label=p["root_label"],
                                     base_kw=p["base_kw"])
        export_verification = verify_export(exported, train=arrays(data), validation=arrays(validation))
        write(target.with_name("candidates.json"), exported)
        fresh = perf_counter()-started
        result = dict(job=p["job"], method=p["method"], flags=p["flags"], selected=selected, trajectory=trajectory,
                      exit_record=exit_record, exit_event=exit_event, output_candidates=rows,
                      structural_seconds=structural_seconds, output_seconds=fresh-structural_seconds,
                      fresh_seconds=fresh, common_seconds=common["seconds"], total_seconds=fresh+common["seconds"],
                      total_cap_exceeded=fresh+common["seconds"] > LIMITS["total"]+1., limits=LIMITS,
                      reference=common["reference"], input_hashes=p["input_hashes"], common_sha256=p["common_sha256"],
                      protocol_sha256=p["protocol_sha256"], truth_used=False, test_opened=False,
                      LP_failures=bank.failures, final_RNJ_fallback=False, candidate_export_sha256=sha(target.with_name("candidates.json")),
                      candidate_export_verification=export_verification)
        write(target, result)
    finally:
        bank.close()


def status(output):
    protocol = read(output / "protocol.json")
    rows = [dict(job=j, method=m["id"], result=(output/"jobs"/j/m["id"]/"result.json").exists(),
                 error=(output/"jobs"/j/m["id"]/"error.json").exists()) for j in protocol["jobs"] for m in protocol["methods"]]
    return dict(planned=protocol["planned"], returned=sum(r["result"] for r in rows),
                failed=sum(r["error"] and not r["result"] for r in rows), pending=sum(not r["result"] and not r["error"] for r in rows))


def dispatch(output, *, phase="pilot", workers=None):
    protocol, catalog = read(output / "protocol.json"), read(output / "catalog.json")
    for name, digest in protocol["source_hashes"].items():
        assert sha(output / "source" / name) == digest, name
    jobs = [j for j in protocol["jobs"] if phase == "all" or (catalog[j]["seed_index"] == 0 and catalog[j]["voltage_sigma"] == .0005)]
    source = output / "source"; env = os.environ.copy()
    env["PYTHONPATH"] = str(source)+os.pathsep+str(source/"rnj_wzzt_core")
    queue = []
    for job in jobs:
        meta = catalog[job]
        if meta["data_error"]:
            for method in protocol["methods"]:
                write(output/"jobs"/job/method["id"]/"error.json", dict(stage="data", error=meta["data_error"]))
            continue
        hashes = {s: protocol["input_hashes"][job][s] for s in ("train", "validation")}
        common_path = output/"common"/job/"common.json"
        if not common_path.exists():
            common_payload = common_path.with_name("payload.json")
            write(common_payload, dict(job=job, train=meta["train"], validation=meta["validation"], input_hashes=hashes, result=str(common_path)))
            subprocess.run([sys.executable,"-B","-m","research_experiments.day96_t_benchmark_20261110.run","common",str(common_payload)],
                           cwd=source, env=env, check=True)
        methods = protocol["methods"]
        for flags in methods:
            target = output/"jobs"/job/flags["id"]/"result.json"
            if target.exists() or target.with_name("error.json").exists():
                continue
            payload = dict(job=job, method=flags["id"], flags=flags, train=meta["train"], validation=meta["validation"],
                           common=str(common_path), common_sha256=sha(common_path), input_hashes=hashes, result=str(target),
                           protocol_sha256=sha(output/"protocol.json"), source_hashes=protocol["source_hashes"],
                           root_label=meta["root_label"], base_kw=meta["base_kw"])
            queue.append(payload)
    active, started = {}, perf_counter()
    while queue or active:
        while queue and len(active) < (workers or protocol["workers"]):
            payload = queue.pop(0); target = Path(payload["result"]); file = target.with_name("payload.json")
            write(file, payload); log = target.with_name("worker.log").open("w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable,"-B","-m","research_experiments.day96_t_benchmark_20261110.run","worker",str(file)],
                                    cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            active[(payload["job"],payload["method"])] = (proc, log, target)
        for key, (proc, log, target) in list(active.items()):
            if proc.poll() is None:
                continue
            log.close(); del active[key]
            if not target.exists() and not target.with_name("error.json").exists():
                write(target.with_name("error.json"), dict(exit_code=proc.returncode))
            print("FINISHED", *key, "OK" if target.exists() else "ERROR", flush=True)
        write(output / f"{phase}_state.json", dict(**status(output), phase=phase, active=[dict(job=k[0],method=k[1],pid=v[0].pid) for k,v in active.items()],
                                                 queued=len(queue), wall_seconds=perf_counter()-started))
        if active:
            sleep(2.)
    write(output / f"{phase}_completion.json", status(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "generate", "common", "worker", "pilot", "all", "status"))
    parser.add_argument("path"); parser.add_argument("--workers", type=int, default=8); parser.add_argument("--bundles", type=int, default=4)
    args = parser.parse_args(); path = Path(args.path).resolve()
    try:
        if args.command == "freeze":
            freeze(path, workers=args.workers, bundles=args.bundles)
        elif args.command in ("pilot", "all"):
            dispatch(path, phase=args.command, workers=args.workers)
        elif args.command == "status":
            print(json.dumps(status(path), indent=2))
        else:
            {"generate": generate, "common": common_worker, "worker": worker}[args.command](path)
    except Exception:
        if args.command in ("worker", "common"):
            write(Path(read(path)["result"]).with_name("error.json"), dict(traceback=traceback.format_exc()))
        raise
