"""Case-study experiment I5: how the method degrades with less or poorer data (DEVELOPMENT cases of at most 25 meters).

    python -m rnj_wzzt_dr.experiments.case_library_20261125.robustness run [--workers 4]
    python -m rnj_wzzt_dr.experiments.case_library_20261125.robustness table

Named feeders (star and street, six load draws each) and the random feeders of at most 25 meters, without a source and with
one at the end. Methods: the final method (identification/final.py: tree S1, located S2) and RNJ, NJ on its matrix.
Conditions (one change at a time from the baseline):
  base       hourly readings, 6 + 2 days, voltage gain and noise 0.05%, P/Q gain 0.3%, noise 0.5%
  days4      hourly, 3 + 1 days          days2   hourly, 1.5 + 0.5 days
  quarter    quarter-hourly, 144 + 48 readings (2 days)
  noise2     voltage gain and noise 0.10%      noise4  0.20%
  pq1        P/Q gain and noise 1.0% (PCC too)
Output: outputs/case_library_20261125/robustness.jsonl (resumable).
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

OUT = build.OUT/"robustness.jsonl"
CONDITIONS = {
    "base": dict(),
    "days4": dict(history=(72, 24)),
    "days2": dict(history=(36, 12)),
    "quarter": dict(per_hour=4, history=(144, 48)),
    "noise2": dict(v_gain=.001, v_noise=.001),
    "noise4": dict(v_gain=.002, v_noise=.002),
    "pq1": dict(pq_gain=.01, pcc_gain=.01, pq_noise=.01),
}
TOL = (0., .08, .24)


def run_one(job):
    cond, name = job
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2
    from rnj_wzzt_dr.identification import final, trees
    try:
        t0 = time.perf_counter()
        rec = build.load("dev", name)
        kw = dict(CONDITIONS[cond])
        inp = build.to_input(rec, kw.pop("per_hour", 1), kw.pop("history", (144, 48)), 0, **kw)
        n = inp.n
        tr, va = p2.series(inp)
        inner = lambda f: {tuple(sorted(C)) for C in f if 1 < len(C) < n}
        r = final.fit(tr, va, inp.hist["Ppcc"], inp.hist["Qpcc"])
        out = dict(cond=cond, name=name, group=rec["group"], set=rec["set"], n=n, source=rec["source"] and rec["source"]["where"],
                   truth=cd.as_lists(inner(rec["truth"])), methods={})
        out["methods"]["s1"] = cd.as_lists(inner(r["tree"]))
        out["methods"]["s2"] = cd.as_lists(inner(r["located"]))
        out["certified"] = bool(r["record"]["certified"]); out["evidence"] = float(r["source_evidence"])
        out["places"] = [[float(c), None if D is None else list(D)] for c, D in r["places"]]
        shared, depth, _ = trees.geometry(r["S"]["R"], r["S"]["R"], 1.)
        for label, make in (("rnj", trees.rnj), ("nj", trees.nj)):
            for tol in TOL:
                out["methods"][f"{label}|{tol}"] = cd.as_lists(inner(make(shared, depth, tol)))
        out["path"] = [list(C) for C in r["path"]]
        out["true_path"] = [list(C) for C in rec["source"]["path"]] if rec["source"] else None
        out["seconds"] = time.perf_counter()-t0
        return out
    except Exception as exc:
        return dict(cond=cond, name=name, error=repr(exc), trace=traceback.format_exc()[-1500:])


def run(workers):
    manifest = json.loads((build.HERE/"manifest_dev.json").read_text(encoding="utf-8"))
    done = set()
    if OUT.exists():
        done = {(r["cond"], r["name"]) for r in map(json.loads, OUT.read_text().splitlines()) if "error" not in r}
    names = [r["name"] for r in sorted(manifest, key=lambda r: (r["n"], r["name"]))
             if r["n"] <= 25 and r["source"] in ("none", "end")]
    todo = [(c, x) for x in names for c in CONDITIONS if (c, x) not in done]
    with open(OUT, "a") as fh, get_context("spawn").Pool(workers, maxtasksperchild=6) as pool:
        for k, row in enumerate(pool.imap_unordered(run_one, todo), 1):
            fh.write(json.dumps(row)+"\n"); fh.flush()
            print(f"[{k}/{len(todo)}] {row['cond']} {row['name']} {row.get('error', '')}", flush=True)


def table():
    rows = [r for r in map(json.loads, OUT.read_text().splitlines())]
    bad = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    print(f"{len(rows)} rows, {len(bad)} errors {[(r['cond'], r['name'], r['error'][:60]) for r in bad][:3]}")
    groups = {"star": "_star_", "street": "_street_", "random": None}

    def kind(r):
        return next((g for g, key in groups.items() if key and key in r["name"]), "random")

    def agg(rs, m):
        tp = fp = fn = 0
        for r in rs:
            got, t = {tuple(x) for x in r["methods"][m]}, {tuple(x) for x in r["truth"]}
            tp += len(got & t); fp += len(got-t); fn += len(t-got)
        rec, pre = tp/max(tp+fn, 1), tp/max(tp+fp, 1)
        return 100*rec, 100*pre, 200*rec*pre/max(rec+pre, 1e-9)
    for g in groups:
        for s in ("none", "end"):
            base = [r for r in rows if kind(r) == g and (r["source"] or "none") == s]
            if not base:
                continue
            print(f"\n{g}, source {s}  (F1 of S2 | S1 | RNJ best of {TOL} | NJ best; precision of S2; false alarms / placed right)")
            for c in CONDITIONS:
                rs = [r for r in base if r["cond"] == c]
                if not rs:
                    continue
                rnj = max(agg(rs, f"rnj|{t}")[2] for t in TOL)
                nj = max(agg(rs, f"nj|{t}")[2] for t in TOL)
                a = agg(rs, "s2")
                extra = ""
                if s == "none":
                    extra = f"evidence > 4.6: {sum(r['evidence'] > 4.6 for r in rs)}/{len(rs)}"
                else:
                    extra = f"evidence > 4.6: {sum(r['evidence'] > 4.6 for r in rs)}/{len(rs)}; true place within 4.6: {sum(covered(r) for r in rs)}/{len(rs)}"
                print(f"  {c:8s} n={len(rs):3d}  S2 {a[2]:5.1f}  S1 {agg(rs, 's1')[2]:5.1f}  RNJ {rnj:5.1f}  NJ {nj:5.1f}   precision {a[1]:5.1f} recall {a[0]:5.1f}   {extra}")


def covered(row):
    """The true place (the smallest returned line below which the source sits) is within 4.6 half-BIC of the best."""
    n = row["n"]
    node = min({tuple(x) for x in row["true_path"]}, key=len)
    ret = {tuple(x) for x in row["methods"]["s2"]}
    chain = [x for x in ret if set(node) <= set(x)]
    D = min(chain, key=len) if chain else None
    places = {(None if d is None else tuple(d)): c for c, d in row["places"]}
    c = places.get(D)
    return c is not None and c <= 4.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "table"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    run(args.workers) if args.what == "run" else table()


if __name__ == "__main__":
    main()
