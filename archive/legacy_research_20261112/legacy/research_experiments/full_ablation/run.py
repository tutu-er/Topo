"""Frozen 2x2x2 structural ablation on the existing 120 observed inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter, sleep
import traceback

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "outputs/iterative_exit_matched120_20261101"
LIMITS = dict(total=14400., prefix=7200., per_mip=3600., LP=300., post_reserve=600., mip_gap=.0001, threads=1)
METHODS = [dict(id=f"R{r}T{t}E{e}", seed=bool(r), tail=bool(t), exit=bool(e))
           for r, t, e in product((0, 1), repeat=3)]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def freeze(output):
    """Snapshot only the already-authorized model family, before any new scores."""
    import numpy as np
    import scipy
    import gurobipy as gp
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("A new immutable output directory is required")
    previous = read(PARENT / "protocol.json")
    assert read(PARENT / "analysis/verification.json")["passed"]
    source = PARENT / "sources/after"
    for name, digest in previous["source_hashes"]["after"].items():
        assert sha(source / name) == digest, name
    output.mkdir(parents=True)
    shutil.copytree(source, output / "source")
    for path in (ROOT / "research_experiments/full_ablation").glob("*.py"):
        target = output / "source/research_experiments/full_ablation" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    # Parent already has common gap=.0001; enforce and verify both builders.
    for relative in ("research_experiments/rebuild_benchmark/engine.py", "research_experiments/joint_exit/model.py"):
        path = output / "source" / relative
        text = path.read_text(encoding="utf-8")
        text = text.replace("MIPGap=.001", "MIPGap=.0001").replace("requested_relative_gap=.001", "requested_relative_gap=.0001")
        assert "MIPGap=.0001" in text
        path.write_text(text, encoding="utf-8", newline="\n")
    catalog = read(PARENT / "catalog.json")
    references = {}
    checked = {}
    for job in previous["jobs"]:
        for split in ("train", "validation", "test"):
            path = catalog[job][split]
            if path not in checked:
                checked[path] = sha(path)
            assert checked[path] == previous["input_hashes"][job][split]
        target = output / "common" / job
        target.mkdir(parents=True)
        shutil.copy2(PARENT / "common" / job / "common.json", target / "common.json")
        historical = PARENT / "common" / job / "historical.json"
        references[job] = dict(common_sha256=sha(target / "common.json"),
                               reference=float(read(historical)["trajectory"]["reference_MAE"]),
                               reference_source_sha256=sha(historical))
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), jobs=previous["jobs"], methods=METHODS,
                    planned=960, workers=8, limits=LIMITS, retuning=False,
                    design="full R x T x E structural factorial; immediate zero release fixed ON",
                    R="all RNJ structural guidance: two small seeds, geometric support pool and RNJ warm starts",
                    T="replace final one-candidate expansions by coupled <=5-block enumeration; OFF continues original WZZT",
                    E="one final revocable structural MILP; OFF directly enters common own-path output selection",
                    common_scales="shared dense-QP calibration/kernel and frozen full-RNJ LP scalar f0; base is WZZT with common calibration, not RNJ-independent preprocessing",
                    R_off="no RNJ support seed/pool/warm starts; own-path supports, generic pairs and <=5 local unions remain allowed by E",
                    objective="raw L1 MAE + .003*f0 times proper internal set count", bounds=[0, 2], resolution=.24,
                    selection="same strict validation MAE library on own path + singleton only; no complete-RNJ output fallback",
                    module_order="all six orders of adding R/T/E configurations; optimizer execution stages remain preselection/forward/tail/final exit",
                    timing="fresh worker algorithm wall plus identical charged common frontend; Python startup/import excluded; every method independently run",
                    maximum_total_method_hours=3840., positive_impedance_floor=None, hard_priors=[],
                    source_hashes={p.relative_to(output / "source").as_posix(): sha(p) for p in (output / "source").rglob("*.py")},
                    input_hashes=previous["input_hashes"], references=references,
                    parent_protocol=dict(path=str(PARENT / "protocol.json"), sha256=sha(PARENT / "protocol.json")),
                    environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__,
                                     gurobi=gp.gurobi.version(), logical_CPUs=os.cpu_count()),
                    truth_used_in_training=False, test_opened_by_worker=False)
    write(output / "protocol.json", protocol)
    write(output / "catalog.json", catalog)
    write(output / "state.json", dict(stage="frozen", planned=960, returned=0))
    delivery = ROOT / "docs" / output.name
    delivery.mkdir(parents=True, exist_ok=False)
    shutil.copy2(output / "protocol.json", delivery / "protocol.json")
    print("FROZEN: eight configurations x120 identical inputs; cap4h each,960 fresh evaluations", flush=True)


def worker(payload):
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.rebuild_benchmark.engine import prepare
    from research_experiments.lp_rebuild.engine import FloorBank
    from research_experiments.local_rebuild.model import canonical
    from research_experiments.joint_exit.selection import library, choose
    from .engine import search, finish

    p = read(payload)
    target = Path(p["result"])
    started = perf_counter()
    common = read(p["common"])
    assert sha(p["common"]) == p["common_sha256"]
    assert {s: sha(p[s]) for s in ("train", "validation")} == p["input_hashes"]
    limits, flags = p["limits"], p["flags"]
    data = prepare(observed(p["train"], p["N"]))
    validation = prepare(observed(p["validation"]))
    deadline = started + limits["total"] - common["seconds"]
    structural = deadline - limits["post_reserve"]
    basis = canonical([*((i,) for i in range(data.n)), tuple(range(data.n))])
    pool = common["pool"] if flags["seed"] else basis

    class TimedBank(FloorBank):
        def fit(self, family, *, preserve=False):
            left = deadline - perf_counter()
            if left <= .05:
                raise TimeoutError("Method total deadline before LP closure")
            self.seconds = min(limits["LP"], left)
            self.lp.model.Params.TimeLimit = self.seconds
            return super().fit(family, preserve=preserve)

    bank = TimedBank(data, pool, common["calibration"], seconds=limits["LP"])
    try:
        bank.seed_start_pool = canonical(pool)
        bank.joint_exit_seed_pool = canonical(pool)
        prefix_deadline = perf_counter() + min(limits["prefix"], max(.1, (structural - perf_counter()) / 2))
        progress = lambda event: write(target.with_name("progress.json"), event)
        write(target.with_name("checkpoint.json"), dict(stage="prefix_started", flags=flags, initial_pool_count=len(pool)))
        trajectory = search(bank, common, reference=p["reference"], seed_on=flags["seed"], tail_on=flags["tail"],
                            deadline=structural, prefix_deadline=prefix_deadline, per_mip=limits["per_mip"],
                            zero_release=True, heartbeat=progress)
        write(target.with_name("checkpoint.json"), dict(stage="prefix_complete", flags=flags, trajectory=trajectory))
        trajectory, exit_record, exit_event = finish(bank, common, trajectory, reference=p["reference"], deadline=structural,
                                                     exit_on=flags["exit"], rnj_guidance=flags["seed"], zero_release=True)
        structural_seconds = perf_counter() - started
        write(target.with_name("checkpoint.json"), dict(stage="structure_complete", flags=flags,
                                                       trajectory=trajectory, exit_record=exit_record, exit_event=exit_event))
        star = bank.fit([(i,) for i in range(data.n)])
        rows = library(bank, [*trajectory["path"], star], validation, resolution=.24)
        for row in rows:
            row["source"] = "own_path" if row["index"] < len(trajectory["path"]) else "singleton"
        selected = choose(rows, "strict")
        fresh = perf_counter() - started
        result = dict(job=p["job"], method=p["method"], flags=flags, selected=selected, trajectory=trajectory,
                      exit_record=exit_record, exit_event=exit_event, output_candidates=rows,
                      structural_seconds=structural_seconds, output_seconds=fresh - structural_seconds,
                      fresh_seconds=fresh, common_seconds=common["seconds"], total_seconds=fresh + common["seconds"],
                      total_cap_exceeded=fresh + common["seconds"] > limits["total"] + 1.,
                      limits=limits, reference=p["reference"], input_hashes=p["input_hashes"],
                      protocol_sha256=p["protocol_sha256"], common_sha256=p["common_sha256"],
                      truth_used=False, test_opened=False, LP_failures=bank.failures,
                      final_RNJ_fallback=False, initial_pool_count=len(pool))
        write(target, result)
    finally:
        bank.close()


def completion(output):
    p = read(output / "protocol.json")
    methods = []
    for method in p["methods"]:
        returned = sum((output / "jobs" / job / method["id"] / "result.json").exists() for job in p["jobs"])
        failed = [job for job in p["jobs"] if (output / "jobs" / job / method["id"] / "error.json").exists()
                  and not (output / "jobs" / job / method["id"] / "result.json").exists()]
        methods.append(dict(method=method["id"], planned=120, returned=returned, failed=failed))
    return dict(planned=960, returned=sum(m["returned"] for m in methods), failed=sum(len(m["failed"]) for m in methods), methods=methods)


def run(output):
    p, catalog = read(output / "protocol.json"), read(output / "catalog.json")
    for name, digest in p["source_hashes"].items():
        assert sha(output / "source" / name) == digest, name
    queue, active = [], {}
    cases = sorted({m["case"] for m in catalog.values()})
    case_jobs = {c: [j for j in p["jobs"] if catalog[j]["case"] == c] for c in cases}
    jobs = [case_jobs[c][i] for i in range(30) for c in cases]
    for index, job in enumerate(jobs):
        methods = p["methods"][index % 8:] + p["methods"][:index % 8]
        if index % 2:
            methods = methods[::-1]
        for flags in methods:
            target = output / "jobs" / job / flags["id"] / "result.json"
            if target.exists() or target.with_name("error.json").exists():
                continue
            meta = catalog[job]
            queue.append(dict(job=job, method=flags["id"], flags=flags, N=meta["N"], train=meta["train"], validation=meta["validation"],
                              common=str(output / "common" / job / "common.json"), **p["references"][job],
                              input_hashes={s: p["input_hashes"][job][s] for s in ("train", "validation")},
                              result=str(target), limits=p["limits"], protocol_sha256=sha(output / "protocol.json")))
    started = perf_counter()
    while queue or active:
        while queue and len(active) < p["workers"]:
            payload = queue.pop(0)
            target = Path(payload["result"])
            file = target.with_name("payload.json")
            write(file, payload)
            source = output / "source"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(source) + os.pathsep + str(source / "rnj_wzzt_core")
            log = target.with_name("worker.log").open("w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-B", "-m", "research_experiments.full_ablation.run", "worker", str(file)],
                                    cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            active[(payload["job"], payload["method"])] = (proc, log, target, perf_counter())
        for key, (proc, log, target, then) in list(active.items()):
            if proc.poll() is None:
                continue
            log.close()
            del active[key]
            if not target.exists() and not target.with_name("error.json").exists():
                write(target.with_name("error.json"), dict(exit_code=proc.returncode))
            print("finished", *key, "OK" if target.exists() else "ERROR", round(perf_counter() - then, 2), flush=True)
        state = completion(output)
        state.update(stage="running" if active or queue else "finished", queued=len(queue),
                     active=[dict(job=k[0], method=k[1], pid=v[0].pid, seconds=perf_counter() - v[3]) for k, v in active.items()],
                     batch_wall_seconds=perf_counter() - started)
        write(output / "state.json", state)
        if active:
            sleep(2.)
    write(output / "completion.json", completion(output))
    print("COMPLETE", completion(output)["returned"], "/960", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run", "worker", "status"))
    parser.add_argument("path")
    args = parser.parse_args()
    path = Path(args.path).resolve()
    try:
        if args.command == "status":
            print(completion(path))
        else:
            globals()[args.command](path)
    except Exception:
        if args.command == "worker":
            write(Path(read(path)["result"]).with_name("error.json"), dict(traceback=traceback.format_exc()))
        raise
