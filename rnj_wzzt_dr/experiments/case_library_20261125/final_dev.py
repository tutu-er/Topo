"""The final method (identification/final.py) on the DEVELOPMENT library, before it is frozen.

    python -m rnj_wzzt_dr.experiments.case_library_20261125.final_dev run --mode blind  [--workers 6]
    python -m rnj_wzzt_dr.experiments.case_library_20261125.final_dev run --mode source [--workers 6]
    python -m rnj_wzzt_dr.experiments.case_library_20261125.final_dev table

blind:  cases without a source, no PCC; with the structural prior (lam = 1.96) and without it (lam = 0), same pipeline.
source: every twin, with the PCC: tree S1, located tree S2, path, evidence; lam = 1.96.
Output: outputs/case_library_20261125/final_<mode>.jsonl (resumable). The sealed cases are never opened here.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
import traceback
from multiprocessing import get_context

import numpy as np

from . import build, compare_dev as cd

OUT = {m: build.OUT/f"final_{m}.jsonl" for m in ("blind", "source")}


def inner(f, n):
    return {tuple(sorted(C)) for C in f if 1 < len(C) < n}


def run_one(job):
    mode, name = job
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2
    from rnj_wzzt_dr.identification import final
    try:
        rec = build.load("dev", name)
        inp = build.to_input(rec); n = inp.n
        tr, va = p2.series(inp)
        out = dict(name=name, group=rec["group"], set=rec["set"], source=rec["source"] and rec["source"]["where"],
                   share=rec["source"] and rec["source"]["share"], kind=rec["source"] and rec["source"]["kind"], n=n,
                   truth=cd.as_lists(inner(rec["truth"], n)))
        if mode == "blind":
            for label, lam in (("bonus", final.LAMBDA), ("plain", 0.)):
                r = final.fit(tr, va, st=final.Settings(lam=lam))
                out[label] = dict(lines=cd.as_lists(inner(r["tree"], n)), seconds=r["seconds"],
                                  certified=r["record"]["certified"], pool=r["record"]["pool"])
        else:
            t0 = time.perf_counter()
            r = final.fit(tr, va, inp.hist["Ppcc"], inp.hist["Qpcc"])
            out.update(s1=dict(lines=cd.as_lists(inner(r["tree"], n)), certified=r["record"]["certified"]),
                       s2=dict(lines=cd.as_lists(inner(r["located"], n)), s2=r["s2"]), seconds=time.perf_counter()-t0,
                       evidence=float(r["source_evidence"]), path=[list(C) for C in r["path"]],
                       places=[[float(c), None if D is None else list(D)] for c, D in r["places"]],
                       true_path=[list(C) for C in rec["source"]["path"]] if rec["source"] else None)
        return out
    except Exception as exc:
        return dict(name=name, error=repr(exc), trace=traceback.format_exc()[-1500:])


def run(mode, workers, tag="", min_n=0, max_n=100):
    manifest = json.loads((build.HERE/"manifest_dev.json").read_text(encoding="utf-8"))
    path = OUT[mode].with_name(OUT[mode].stem+tag+".jsonl")
    done = set()
    if path.exists():
        done = {r["name"] for r in map(json.loads, path.read_text().splitlines()) if "error" not in r}
    todo = [r["name"] for r in sorted(manifest, key=lambda r: (r["n"], r["name"])) if r["name"] not in done
            and min_n <= r["n"] <= max_n and (mode == "source" or r["source"] == "none")]
    with open(path, "a") as fh, get_context("spawn").Pool(workers, maxtasksperchild=4) as pool:
        for k, row in enumerate(pool.imap_unordered(run_one, [(mode, x) for x in todo]), 1):
            fh.write(json.dumps(row, default=float)+"\n"); fh.flush()
            print(f"[{k}/{len(todo)}] {row['name']} {row.get('error', '')}", flush=True)


def table():
    blind = {r["name"]: r for r in map(json.loads, cd.OUT.read_text().splitlines()) if "error" not in r}
    supported = {r["group"]: {tuple(C) for C, (_, d) in zip(r["truth"], r["evidence"]) if d > 0}
                 for r in blind.values() if r["source"] is None}
    groups = ("star", "street", "small", "medium", "large", "huge")

    def agg(rs, get):
        t = np.zeros(10)
        for r in rs:
            t += cd.counts(r, get(r), supported[r["group"]])
        rec, pre = t[0]/max(t[1], 1), t[0]/max(t[2], 1)
        return f"rec {100*rec:5.1f} prec {100*pre:5.1f} F1 {100*2*rec*pre/max(rec+pre, 1e-9):5.1f} root {int(t[5])}/{int(t[6])} false {t[2]-t[0]:.0f}"
    if OUT["blind"].exists():
        rows = [r for r in map(json.loads, OUT["blind"].read_text().splitlines()) if "error" not in r]
        print(f"BLIND, no source: {len(rows)} cases")
        for g in groups:
            rs = [r for r in rows if cd.group_of(r) == g]
            if not rs:
                continue
            print(f"  {g} ({len(rs)})")
            print("    with bonus   ", agg(rs, lambda r: r["bonus"]["lines"]), " sec median", round(np.median([r["bonus"]["seconds"] for r in rs])),
                  "certified", sum(r["bonus"]["certified"] for r in rs))
            print("    without bonus", agg(rs, lambda r: r["plain"]["lines"]), " sec median", round(np.median([r["plain"]["seconds"] for r in rs])),
                  "certified", sum(r["plain"]["certified"] for r in rs))
            print("    WZZT-C       ", agg([blind[r["name"]]|{"group": r["group"]} for r in rs], lambda r: r["methods"]["wzzt_c"]["lines"]))
            print("    RNJ 0.08     ", agg([blind[r["name"]] for r in rs], lambda r: r["methods"]["rnj|0.08"]["lines"]),
                  "| best tol", max((agg([blind[r["name"]] for r in rs], lambda r, t=t: r["methods"][f"rnj|{t}"]["lines"]), t) for t in cd.TOLERANCES)[1])
    if OUT["source"].exists():
        rows = [r for r in map(json.loads, OUT["source"].read_text().splitlines()) if "error" not in r]
        print(f"\nSOURCE-AWARE: {len(rows)} cases")
        for g in groups:
            for s in ("none", "source"):
                rs = [r for r in rows if cd.group_of(r) == g and (r["source"] is None) == (s == "none")]
                if not rs:
                    continue
                b = [r for r in rs if r["name"] in blind]
                print(f"  {g}, {s} ({len(rs)}): S2 {agg(rs, lambda r: r['s2']['lines'])}")
                print(f"       S1 {agg(rs, lambda r: r['s1']['lines'])}")
                if b:
                    print(f"       WZZT-C blind ({len(b)}) {agg(b, lambda r: blind[r['name']]['methods']['wzzt_c']['lines'])}")
        for s in ("none", "head", "middle", "end"):
            rs = [r for r in rows if (r["source"] or "none") == s]
            if rs:
                e = np.array([r["evidence"] for r in rs])
                print(f"  source {s}: {len(rs)}, evidence > 4.6: {int((e > 4.6).sum())}, median {np.median(e):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "table"))
    ap.add_argument("--mode", choices=("blind", "source"), default="blind")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--min_n", type=int, default=0)
    ap.add_argument("--max_n", type=int, default=100)
    args = ap.parse_args()
    run(args.mode, args.workers, args.tag, args.min_n, args.max_n) if args.what == "run" else table()


if __name__ == "__main__":
    main()
