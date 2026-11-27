"""Evaluate verified random development feeders with per-case/method wall limits.

The four distance estimators and Proposed ID all receive P,Q,Y,V0sq on the
same eight-day hourly history. The default run evaluates available verified
cases and records missing inputs. Use --collect to rebuild the 24-feeder
summary from the private arm receipts.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import gurobipy as gp
import numpy as np
import scipy

from ptid.cases import library
from ptid.method import builders
from ptid.method.identify import identify
from ptid.runs.four_offline import digest, metrics, save, settings
from ptid.runs.passive_baselines import PARK_EPS, TOLERANCES, run_baseline, sources

METHODS = ("park", "rg", "nj", "rnj", "proposed")


def selected():
    return [n for n in library.one_twin_per_feeder() if n.startswith("dev_")]


def contract(name, method, wall):
    package = Path(library.__file__).resolve().parents[1]
    files = [*sorted((package/"method").glob("*.py")), package/"cases"/"library.py", package/"cases"/"physics.py",
             package/"runs"/"passive_baselines.py", Path(sources.__file__)]
    source = hashlib.sha256()
    for path in files:
        source.update(path.read_bytes())
    source.update(inspect.getsource(metrics).encode())
    return dict(version=1, name=name, method=method, wall_limit_s=wall, sampling="eight-days-hourly/meters-seed0/no-PCC-PQ",
                record_sha256=next(r["sha256"] for r in library.manifest() if r["name"] == name),
                settings=asdict(settings()) if method=="proposed" else dict(tolerance=TOLERANCES.get(method, PARK_EPS)),
                calculation_sha256=source.hexdigest(), python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, gurobi=list(gp.gurobi.version()))


@contextmanager
def checkpoint_solves(path):
    """Persist each completed restricted Tree MIQP before a later stage can time out."""
    calls = []
    from importlib import import_module
    module = import_module("ptid.method.identify")
    original = module.solve
    def solve(*args, **kwargs):
        tree, info = original(*args, **kwargs)
        calls.append(dict(tree=list(tree), status=info["status"], crit=info["crit"], bound=info["bound"],
                          warm_crit=info["warm_crit"], gap_closed=info["gap_closed"], pool=info["pool"],
                          domain=info["domain"]))
        save(path, calls)
        return tree, info
    module.solve = solve
    try:
        yield
    finally:
        module.solve = original


@contextmanager
def cached_resistance(cache, name):
    """Reuse the same numerical fit across methods which independently request it."""
    original = builders.r_hat
    signature = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    calls = []
    def fit(Pm, Ym, gamma):
        inputs = dict(Pm=Pm, Ym=Ym, gamma=np.asarray(gamma))
        key = hashlib.sha256(json.dumps([digest(inputs), signature, np.__version__, gp.gurobi.version()],
                                        sort_keys=True).encode()).hexdigest()
        target = cache.parent/"resistance"/f"{name}_{key}.npz"
        signal = perf_counter()
        if target.is_file():
            with np.load(target, allow_pickle=False) as saved:
                R = saved["R"]
                seconds = float(saved["seconds"])
                if str(saved["sha256"]) != digest({"R":R})["R"]:
                    raise ValueError("Shared resistance cache checksum differs")
            hit = True
        else:
            R = original(Pm, Ym, gamma)
            seconds = perf_counter()-signal
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(f".{os.getpid()}.tmp.npz")
            np.savez(tmp, R=R, seconds=seconds, sha256=digest({"R":R})["R"])
            tmp.replace(target)
            hit = False
        calls.append(dict(key=key, hit=hit, original_seconds=seconds, elapsed_seconds=perf_counter()-signal))
        return R
    builders.r_hat = fit
    try:
        yield calls
    finally:
        builders.r_hat = original


def measure(name, method, root):
    # open_case validates against the committed development SHA before unpickling.
    library.LIBRARY = root
    rec = library.open_case(name)
    h = library.readings(rec)
    common = {k: h[k] for k in ("P", "Q", "Y", "V0sq")}
    basic = dict(case=name, meters=rec["n"], source=library.load_where(rec),
                 record_sha256=next(r["sha256"] for r in library.manifest() if r["name"] == name),
                 history_sha256=digest(common), method=method)
    try:
        if method == "proposed":
            st = settings()
            t = perf_counter()
            fit = identify(**common, st=st)
            stage = fit["first"]
            item = dict(status="ok", tree=fit["tree"], seconds=perf_counter()-t, scope=fit["scope"],
                        stop=fit["stop"], settings=asdict(st), model=fit["model"], first=dict(
                            status=stage["status"], crit=stage["crit"], bound=stage["bound"],
                            gap_closed=stage["gap_closed"], warm_crit=stage["warm_crit"]))
        else:
            item = run_baseline(method, **common)
        if item["status"] == "ok":
            item["score"] = metrics(item["tree"], library.truth(rec))
            item["score_all_physical"] = metrics(item["tree"], library.exact_truth(rec))
        return {**basic, **item}
    except Exception as exc:
        return dict(**basic, status="failed", error=f"{type(exc).__name__}: {exc}", tree=None)


def summarize(base, cache, out, wall):
    names = selected()
    rows = []
    library.LIBRARY = base
    for name in names:
        checked = (base/"dev"/(name+".pkl")).is_file()
        if checked:
            rec = library.open_case(name)
            history = library.readings(rec)
            input_digests = digest({k:history[k] for k in ("P","Q","Y","V0sq")})
        for method in METHODS:
            path = cache/(name+"__"+method+".json")
            if not checked:
                rows.append(dict(case=name, method=method, status="unverified_input"))
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["run_contract"] != contract(name, method, wall):
                raise ValueError(f"Obsolete run receipt for {name}/{method}")
            if row["status"] == "ok":
                if row["history_sha256"] != input_digests or row["score"] != metrics(row["tree"],library.truth(rec)):
                    raise ValueError(f"History or score differs for {name}/{method}")
            summary = {k: row[k] for k in ("case","method","meters","source","status","history_sha256","score",
                       "score_all_physical","seconds","seconds_with_cold_preprocessing","wall_seconds","first",
                       "scope","stop","error") if k in row}
            if row["status"] != "ok":
                summary["completed_restricted_solves"] = None
                if row.get("completed_restricted_solves"):
                    summary["completed_restricted_solves"] = [
                        {k:v for k,v in step.items() if k in ("status","crit","bound","warm_crit","gap_closed","pool")}
                        for step in row["completed_restricted_solves"]]
            rows.append(summary)
    sizes = ["small", "medium", "large", "huge"]
    def table(cases):
        shared = set.intersection(*(set(r["case"] for r in rows if r["case"] in cases and r["method"]==m
                                       and r["status"]=="ok") for m in METHODS))
        answer = {}
        for m in METHODS:
            group = [r for r in rows if r["case"] in cases and r["method"]==m]
            scored = [r["score"] for r in group if r["status"]=="ok"]
            matched = [r["score"] for r in group if r["case"] in shared]
            answer[m] = dict(status_counts=dict(Counter(r["status"] for r in group)),
                             mean_f1_completed=float(np.mean([r["f1"] for r in scored])) if scored else None,
                             mean_f1_paired=float(np.mean([r["f1"] for r in matched])) if matched else None,
                             exact_paired=sum(r["exact_match"] for r in matched))
        return dict(feeder_count=len(cases), paired_feeders=sorted(shared), paired_n=len(shared), by_method=answer)
    save(out,dict(wall_limit_s=wall, settings=asdict(settings()), scheduled_jobs=len(rows), rows=rows,
                  algorithms=list(METHODS), sampling="one_twin_per_feeder, first eight days, P/Q/Y/V0sq, meter seed 0",
                  overall=table(names), by_size={s:table([n for n in names if library.kind(n)==s]) for s in sizes},
                  tuning_overlap="Blind RG, NJ and RNJ tolerances were chosen on these 24 development feeders' no-source variants (PROTOCOL_ID section 3).",
                  bounds_scope="Recorded objective values and lower bounds apply to the indicated restricted Tree MIQP. A null checkpoint means no restricted solution was saved before wall termination.",
                  score="Macro average over feeders, identifiable groupings excluding root and singleton groups."))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--jobs", nargs="+", help="one or more declared names, or omit to run the panel")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--wall", type=int, default=120)
    p.add_argument("--child", action="store_true")
    p.add_argument("--collect", action="store_true", help="Verify input hashes and scores, summarize all 24 declared feeders")
    args = p.parse_args()
    names = args.jobs or selected()
    if not set(names) <= set(selected()) or not names or args.wall < 1:
        p.error("Use the preregistered random development names and a positive wall limit.")
    base = args.cases_root.resolve()
    cache = args.cache.resolve()
    if args.collect:
        return summarize(base, cache, args.out, args.wall)
    if args.child:
        if len(names) != 1 or len(args.methods) != 1:
            p.error("Each subprocess runs exactly one case and method.")
        with cached_resistance(cache, names[0]) as calls, checkpoint_solves(args.out.with_suffix(".progress")):
            row = measure(names[0], args.methods[0], base)
        row["resistance_fits"] = calls
        row["run_contract"] = contract(names[0], args.methods[0], args.wall)
        if "seconds" in row:
            row["seconds_with_cold_preprocessing"] = row["seconds"] + sum(
                e["original_seconds"]-e["elapsed_seconds"] for e in calls if e["hit"])
        save(args.out, row)
        return 0

    def execute(job):
        name, method = job
        out = cache/(name+"__"+method+".json")
        signature = contract(name, method, args.wall)
        if out.is_file():
            r = json.loads(out.read_text(encoding="utf-8"))
            if r.get("run_contract") == signature and "wall_seconds" in r:
                return r
        progress = out.with_suffix(".progress")
        progress.unlink(missing_ok=True)
        child_out = out.with_suffix(".child")
        child_out.unlink(missing_ok=True)
        cmd = [sys.executable, "-B", "-m", "ptid.runs.random_dev_compare", "--child",
               "--jobs", name, "--methods", method, "--cases-root", str(base), "--cache", str(cache),
               "--out", str(child_out), "--wall", str(args.wall)]
        t = perf_counter()
        try:
            proc = subprocess.run(cmd, timeout=args.wall, check=False, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, errors="replace")
            if proc.returncode != 0 or not child_out.is_file():
                raise RuntimeError(f"child {proc.returncode}: {proc.stdout[-1400:]}")
            row = json.loads(child_out.read_text(encoding="utf-8"))
        except subprocess.TimeoutExpired:
            row = dict(case=name, method=method, tree=None, status="wall_limit")
        except Exception as exc:
            row = dict(case=name, method=method, tree=None, status="failed", error=f"{type(exc).__name__}: {exc}")
        if row["status"] != "ok":
            row["completed_restricted_solves"] = json.loads(progress.read_text(encoding="utf-8")) if progress.exists() else None
        row.update(wall_limit_s=args.wall, wall_seconds=perf_counter()-t, run_contract=signature)
        save(out, row)
        child_out.unlink(missing_ok=True)
        print(name, method, row["status"], f"{row['wall_seconds']:.1f}s", flush=True)
        return row

    missing = [n for n in names if not (base/"dev"/(n+".pkl")).is_file()]
    if missing and args.jobs:
        p.error(f"Missing verified files: {missing}")
    names = [n for n in names if n not in missing]
    jobs = [(n,m) for n in names for m in args.methods]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(execute, jobs))
    if args.jobs is None and set(args.methods)==set(METHODS):
        summarize(base, cache, args.out, args.wall)
    else:
        save(args.out, dict(selected=names, methods=args.methods, wall_limit_s=args.wall, missing_inputs=missing,
                            settings=asdict(settings()), rows=rows))
    return int(any(r["status"] != "ok" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
