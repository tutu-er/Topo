"""Paired passive comparisons on the four local feeders, with reusable PCC day-zero results.

From topology_study, run python -m ptid.runs.four_offline --jobs C first, then
--jobs rg,nj,rnj,park,blind,A,B. Each case/job writes a distinct artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import time
import traceback

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import gurobipy as gp
import numpy as np
import scipy

from ptid.method.builders import NAMES, builder_trees
from ptid.method.identify import Settings, identify, place_load
from ptid.method.labels import grouping
from ptid.method.model import BOUND, Fitter, Model, balance
from ptid.method.sets import basis, children, lines_of
from ptid.runs.four_feeders import CASES, data_root, history, load_feeder, meter_stream, sha256, validate_replay
from ptid.runs.identify_cases import f1
from ptid.runs.passive_baselines import run_baseline

OUT = Path(__file__).resolve().parents[2]/"experiments"/"four_feeders"
JOBS = ("C", "rg", "nj", "rnj", "park", "blind", "A", "B")


def settings():
    return Settings(rounds=1, repairs=2, short_seconds=4, long_seconds=4, flip_seconds=4,
                    node_seconds=4, start_seconds=1, evidence=False)


def cache_file(name):
    return data_root().parent/"outputs"/"topology_study_four_feeders"/"first"/f"{name}.pkl"


def digest(data):
    return {k: hashlib.sha256(a.astype("<f8").tobytes()).hexdigest() for k, a in data.items()}


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    if path.suffix == ".pkl":
        temp.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    else:
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False,
                                   default=lambda a: a.item() if isinstance(a, np.generic) else list(a))+"\n",
                        encoding="utf-8")
    temp.replace(path)


def metrics(tree, truth):
    est, real = {tuple(t) for t in tree}, set(truth)
    s = f1(est, real)
    return dict(**s, fp=len(est-real), fn=len(real-est), exact_match=est == real)


def stages(data, first, rec, st):
    n = rec["n"]
    u = balance(**data) if first["gate"]["load"] else None
    model = Model(data["P"], data["Q"], data["Y"], data["V0sq"], u, st.lam, p_s=st.p_s)
    def physical(lines):
        return sorted({g for line in lines if (g := grouping(line, n))})
    def stage(tree, crit):
        return dict(tree=physical(tree), H=model.N*crit/2, score=metrics(physical(tree), rec["truth"]))
    def solve_stage(solve):
        return dict(**stage(solve["tree"], solve["crit"]), pool=solve["pool"], status=solve["status"],
                    bound_H=None if solve["bound"] is None else model.N*solve["bound"]/2,
                    gap_closed=solve["gap_closed"], miqp_seconds=solve["miqp_seconds"])
    init = first["first"]
    domain = init["domain"]["lines"]
    fit = Fitter(model, domain)
    raw = builder_trees(data["P"], data["Q"], data["Y"], data["V0sq"], u)
    starts = [place_load(model, tree)[0] if u is not None else tree for tree in raw]
    raw_values = [fit.crit(tree) for tree in starts]
    best = int(np.argmin(raw_values))
    saved = dict(best_builder=dict(method=NAMES[best], **stage(starts[best],raw_values[best])),
                 warm=stage(init["warm_tree"], init["warm_crit"]),
                 raw_mip=None if init.get("solver_tree") is None else stage(init["solver_tree"], init["solver_crit"]),
                 first=solve_stage(init), initial_pool=physical(domain), rounds=[])
    for record in first["rounds"]:
        if "price" not in record:
            continue
        solve, price = record["solve"], record["price"]
        before = set(physical(solve["domain"]["lines"]))
        missing = set(rec["truth"])-before
        eligible_wide = set()
        for kid_list in children(solve["tree"], model.leaves).values():
            if len(kid_list) <= st.K:
                continue
            for group in missing:
                for target in ([set(group), {*group, n}] if u is not None else [set(group)]):
                    kids = [set(k) for k in kid_list if set(k) <= target]
                    if 2 <= len(kids) < len(kid_list) and set.union(*kids) == target:
                        eligible_wide.add(group)
        added = [tuple(x) for x in price["added"]]
        after = lines_of([*solve["tree"], *added], model.leaves)
        saved["rounds"].append(dict(anchored=solve_stage(solve), anchors=record["anchors"], pre_node_missing=sorted(missing),
                                    pre_node_reachable_wide=sorted(eligible_wide), added=physical(added),
                                    added_outside_pre_node=sorted(set(physical(added))-before),
                                    after_addition=stage(after,Fitter(model, after).crit(after)),
                                    enlarged=None if price["re_solve"] is None else solve_stage(price["re_solve"]),
                                    scans=price.get("scans", []), disabled=price.get("disabled", False)))
    saved["final"] = stage(first["lines"], Fitter(model, first["lines"]).crit(first["lines"]))
    if abs(saved["final"]["H"]-model.N*first["model"]["crit"]/2)>1e-6:
        raise ValueError("Final score check failed")
    return saved


def run_one(rec, data, name, job, reuse_initial=False):
    common = {k: data[k] for k in ("P", "Q", "Y", "V0sq")}
    use_pcc = job in ("A","B","C")
    observed = data if use_pcc else common
    row = dict(case=name, method=job, n=rec["n"], input_sha256=digest(observed),
               truth=rec["truth"], exact_truth=rec["exact_truth"])
    if job in ("rg", "nj", "rnj", "park"):
        result = run_baseline(job, **common)
        row.update(result)
    else:
        st = replace(settings(), node_refinement=job!="A", wide_node_miqp=job!="B")
        cached, common_time = None, 0.
        if job in ("A", "B") or (job == "C" and reuse_initial):
            path = cache_file(name)
            receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if receipt["input_sha256"] != row["input_sha256"] or receipt["cache_sha256"] != sha256(path):
                raise ValueError("Shared day-zero cache provenance mismatch")
            with path.open("rb") as fh:
                initial = pickle.load(fh)
            if initial["settings"] != asdict(settings()):
                raise ValueError("Incompatible common first-solve settings")
            if job in ("A", "B"):
                cached, common_time = initial["first"], initial["timing"]["first_tree"]
        if job == "C" and reuse_initial:
            r, seconds = initial, initial["timing"]["total"]
        else:
            start = time.perf_counter()
            r = identify(**observed, st=st, paired_first=cached)
            seconds = time.perf_counter()-start
        if job == "C" and not reuse_initial:
            path = cache_file(name)
            save(path, r)
            save(path.with_suffix(".json"), dict(case=name, input_sha256=row["input_sha256"], cache_sha256=sha256(path)))
        details = stages(data, r, rec, st)
        row.update(tree=r["tree"], lines=r["lines"], seconds=seconds, shared_first_seconds=common_time,
                   accounted_total_seconds=seconds+common_time, stages=details, settings=asdict(st),
                   status="ok", stop=r["stop"], gate=r["gate"], model=r["model"], scope=r["scope"],
                   label_statuses=dict(Counter(i["label"] for i in r["labels"])),
                   labels=[dict(group=i["grouping"],label=i["label"],inside=i["inside"],
                                gap=i["gap"],status=i["status"]) for i in r["labels"]], timing=r["timing"])
    if row["tree"] is not None:
        row["score"] = metrics(row["tree"], rec["truth"])
        row["score_exact"] = metrics(row["tree"], rec["exact_truth"])
    return row


def independent_H(data, row):
    """Explicit voltage design with scipy BVLS, separate from the solver's Gram matrix."""
    from scipy.optimize import lsq_linear
    m = row["model"]
    p, q = data["P"], data["Q"]
    n = p.shape[1]
    if row["gate"]["load"]:
        u = balance(**data)
        p, q = np.column_stack((p, u[0])), np.column_stack((q, u[1]))
        lines = row["lines"]
    else:
        lines = row["tree"]
    cols = []
    for line in (*basis(p.shape[1]), *lines):
        z = np.isin(np.arange(n), line).astype(float)
        w = z-m["gamma"]*z.mean()
        for series in (p, q):
            x = series[:, line].sum(axis=1)
            cols.append(((x-x.mean())[:, None]*w).ravel())
    y = data["Y"]-m["gamma"]*data["Y"].mean(axis=1, keepdims=True)
    fit = lsq_linear(np.column_stack(cols), (y-y.mean(0)).ravel(), bounds=(0, BOUND), method="bvls")
    if not fit.success:
        raise RuntimeError(fit.message)
    return float(fit.cost/m["sigma2"] + m["N"]*m["alpha"]*len(lines)/2)


def audit():
    summaries = {job: [] for job in JOBS}
    independent = []
    for name in CASES:
        rec = load_feeder(name)
        data = history(rec)
        hashes, rows = digest(data), {}
        for job in JOBS:
            row = json.loads((OUT/f"offline_{name}_{job}.json").read_text(encoding="utf-8"))
            if row["status"] != "ok" or any(hashes[k] != v for k, v in row["input_sha256"].items()):
                raise ValueError(f"Job failed or inputs differ: {name}/{job}")
            if metrics(row["tree"], rec["truth"]) != row["score"]:
                raise ValueError(f"Score mismatch: {name}/{job}")
            summaries[job].append(row["score"])
            rows[job] = row
            if "model" in row:
                check = independent_H(data, row)
                error = abs(check-row["stages"]["final"]["H"])
                if error > 1e-5:
                    raise ValueError(f"Explicit final score differs: {name}/{job} {error}")
                independent.append(dict(case=name, method=job, independent_H=check, absolute_error=error))
        for key in ("initial_pool", "warm", "first"):
            if any(rows[job]["stages"][key] != rows["C"]["stages"][key] for job in ("A", "B")):
                raise ValueError(f"Node arms differ before refinement: {name} {key}")
    per_method = {}
    for job, scores in summaries.items():
        per_method[job] = dict(completed=len(scores), mean_feeder_f1=float(np.mean([r["f1"] for r in scores])),
                               exact_trees=sum(r["exact_match"] for r in scores),
                               tp=sum(r["tp"] for r in scores), fp=sum(r["fp"] for r in scores), fn=sum(r["fn"] for r in scores))
    save(OUT/"offline_audit.json", dict(summaries=per_method, independent_objectives=independent,
                                        all_common_initial_states_equal=True, objective_tolerance_H=1e-5))


def endday_data(rec, data):
    """Use the delivered P0 measurements at hour 24, independently verified against the archived zero-action world."""
    name = rec["name"]
    receipt = json.loads((OUT/f"probe_{name}_2024-01-31_100_P0.json").read_text(encoding="utf-8"))
    if (receipt["start_day"] != rec["start_day"] or receipt["policy"] != "P0"
            or receipt["hours_delivered"] != 24 or receipt["orders"] or receipt["paid"] != 0):
        raise ValueError("Expected the one-day zero-procurement trajectory")
    f = cache_file(name).parents[1]/"ac"/receipt["trace_file"]
    if sha256(f) != receipt["trace_sha256"] or receipt["source_hashes"] != rec["source_hashes"]:
        raise ValueError("P0 provenance mismatch")
    with np.load(f,allow_pickle=False) as a:
        observed = {k:a["observed_"+k] for k in data}
        if a["action_pu"].shape != (24,rec["n"]) or np.any(a["action_pu"]):
            raise ValueError("P0 has missing or nonzero delivered actions")
    meters, archive = meter_stream(rec), rec["true"]
    rows = []
    for h in range(24):
        step = slice(768+4*h,768+4*h+4)
        r = meters.read(*(archive[k][step].mean(axis=0) for k in ("P","Q","V","V0","S_pcc")))
        rows.append(dict(P=r["P"], Q=r["Q"], Y=r["V0"]**2-r["V"]**2, V0sq=r["V0"]**2,
                         Ppcc=r["Ppcc"], Qpcc=r["Qpcc"]))
    errors = {}
    for k, a in observed.items():
        if not np.array_equal(data[k],a[:192]) or len(a)!=216:
            raise ValueError(f"P0 history differs for {k}")
        errors[k] = float(np.max(np.abs(a[192:]-np.asarray([r[k] for r in rows]))))
        if errors[k]>1e-8:
            raise ValueError(f"Independent zero-action readback disagrees for {k}: {errors[k]}")
    return observed, dict(trace=receipt["trace_file"], trace_sha256=receipt["trace_sha256"], delivered_hours=24,
                          orders=0, history_rows=192, evaluation_hour=24, verification_errors=errors)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--jobs", default=",".join(JOBS))
    ap.add_argument("--reuse-initial", action="store_true", help="Read an already verified C solve for scoring")
    ap.add_argument("--audit", action="store_true", help="audit all 32 saved jobs")
    ap.add_argument("--endday", action="store_true", help="20 end-of-day same-channel refits on verified P0 trajectories")
    args = ap.parse_args()
    if args.audit:
        audit()
        return 0
    env = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
               gurobi=gp.gurobi.version(), platform=platform.platform())
    success = True
    for name in args.cases.split(","):
        rec = load_feeder(name)
        data = history(rec)
        if args.endday:
            data, trajectory = endday_data(rec, data)
            for job in ("rg", "nj", "rnj", "park", "blind"):
                row = run_one(rec, data, name, job)
                row.update(trajectory=trajectory, environment=env)
                if "model" in row:
                    row["independent_H"] = independent_H(data, row)
                    if abs(row["independent_H"]-row["stages"]["final"]["H"]) > 1e-5:
                        raise ValueError(f"End-day explicit score differs on {name}")
                save(OUT/f"endday_{name}_{job}.json", row)
                print(name, "endday", job, row["status"], row.get("score",{}).get("f1"), flush=True)
                success &= row["status"]=="ok"
            continue
        save(OUT/f"data_check_{name}.json", dict(**validate_replay(rec), environment=env, history_days=8,
                                                 start_day=rec["start_day"], meter_key=rec["key"]))
        for job in args.jobs.split(","):
            if job not in JOBS:
                raise ValueError(job)
            try:
                row = run_one(rec, data, name, job, args.reuse_initial)
            except Exception:
                row = dict(case=name, method=job, status="failed", error=traceback.format_exc())
            row["environment"] = env
            save(OUT/f"offline_{name}_{job}.json", row)
            print(name, job, row["status"], row.get("score",{}).get("f1"), row.get("seconds"), row.get("error",""), flush=True)
            success &= row["status"]=="ok"
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
