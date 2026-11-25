"""Development comparison: WZZT-C (frozen settings) and the distance builders on the DEVELOPMENT cases only.

    python -m rnj_wzzt_dr.experiments.case_library_20261125.compare_dev run [--workers 5]
    python -m rnj_wzzt_dr.experiments.case_library_20261125.compare_dev table

All methods are source-blind here: they see the meters and the PCC voltage, not the PCC's P and Q. WZZT-R is added on
cases of at most 25 meters with `run --wzzt_r`. Output: outputs/case_library_20261125/compare_dev.jsonl (resumable).
The held-out cases are never opened by this program.
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

from . import build

OUT = build.OUT/"compare_dev.jsonl"
OUT_R = build.OUT/"compare_dev_wzzt_r.jsonl"
TOLERANCES = (0., .04, .08, .16, .24)
REAL_SETTING = dict(rnj=.08, nj=.04, rg=.08)                   # fixed on the earlier real development feeders


def as_lists(lines):
    return sorted([list(C) for C in lines], key=lambda c: (len(c), c))


def run_one(job):
    name, with_r = job
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_confirm_c import WZZT_C
    from rnj_wzzt_dr.identification import combo, revised as rv, trees
    try:
        rec = build.load("dev", name)
        inp = build.to_input(rec); n = inp.n
        tr, va = p2.series(inp)
        inner = lambda f: {tuple(sorted(C)) for C in f if 1 < len(C) < n}
        S = combo.setup(tr, va)
        score = lambda f: rv.exact_fit(S["design"], f, 2.)["objective"]+S["alpha"]*rv.internal_count(f, n)
        out = dict(name=name, group=rec["group"], set=rec["set"], source=rec["source"] and rec["source"]["where"],
                   share=rec["source"] and rec["source"]["share"], kind=rec["source"] and rec["source"]["kind"],
                   n=n, truth=as_lists(inner(rec["truth"])), methods={})
        if with_r:
            t0 = time.perf_counter()
            r = rv.fit(tr, va, rv.RevisedConfig())
            out["methods"]["wzzt_r"] = dict(lines=as_lists(inner(r["selected"])), seconds=time.perf_counter()-t0,
                                            score=score(r["selected"]))
            return out
        t0 = time.perf_counter()
        c = combo.fit(tr, va, **WZZT_C)
        out["methods"]["wzzt_c"] = dict(lines=as_lists(inner(c["selected"])), seconds=time.perf_counter()-t0,
                                        certified=c["record"]["certified"], score=score(c["selected"]))
        shared, depth, _ = trees.geometry(S["R"], S["R"], 1.)
        for label, make in (("rnj", trees.rnj), ("nj", trees.nj), ("rg", trees.rg)):
            for tol in TOLERANCES:
                out["methods"][f"{label}|{tol}"] = dict(lines=as_lists(inner(make(shared, depth, tol))))
        if rec["source"] is None:
            from .check import evidence
            out["evidence"] = evidence(rec)                     # (share of meters, change) in the order of `truth`
        return out
    except Exception as exc:
        return dict(name=name, error=repr(exc), trace=traceback.format_exc()[-1500:])


def run(workers, with_r):
    path = OUT_R if with_r else OUT
    manifest = json.loads((build.HERE/"manifest_dev.json").read_text(encoding="utf-8"))
    done = set()
    if path.exists():
        done = {r["name"] for r in map(json.loads, path.read_text().splitlines()) if "error" not in r}
    todo = [r for r in manifest if r["name"] not in done and (not with_r or (r["n"] <= 25 and r["source"] == "none"))]
    todo.sort(key=lambda r: (r["source"] != "none", r["n"]))    # cases without a source first, small ones first
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh, get_context("spawn").Pool(workers, maxtasksperchild=8) as pool:
        for k, row in enumerate(pool.imap_unordered(run_one, [(r["name"], with_r) for r in todo]), 1):
            fh.write(json.dumps(row)+"\n"); fh.flush()
            print(f"[{k}/{len(todo)}] {row['name']} {row.get('error', '')}", flush=True)


# ------------------------------------------------ tables
def group_of(row):
    if row["set"] == "named":
        return "star" if "_star_" in row["name"] else "street"
    return row["name"].split("_")[1]


def counts(row, lines, supported):
    """Hits and totals of one result: all true lines, supported true lines, root-side supported lines."""
    n = row["n"]
    truth = [tuple(C) for C in row["truth"]]
    got = {tuple(C) for C in lines}
    root = [C for C in truth if C in supported and len(C) > n/2]
    return np.array([len(got & set(truth)), len(truth), len(got), len(got & supported), len(supported),
                     len(got & set(root)), len(root), int(got == set(truth)), int(supported <= got <= set(truth)), 1])


def table():
    rows = [json.loads(x) for x in OUT.read_text().splitlines()]
    errors = [r["name"] for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    extra = {}
    if OUT_R.exists():
        extra = {r["name"]: r for r in map(json.loads, OUT_R.read_text().splitlines()) if "error" not in r}
    twin = {r["group"]: r for r in rows if r["source"] is None}
    supported = {g: {tuple(C) for C, (_, d) in zip(r["truth"], sorted_evidence(r)) if d > 0} for g, r in twin.items()}
    print(f"{len(rows)} cases ({len(errors)} errors)")
    methods = ["wzzt_c", "wzzt_r"]+[f"{b}|{t}" for b in ("rnj", "nj", "rg") for t in TOLERANCES]
    blocks = [(g, s) for s in ("none", "source") for g in ("star", "street", "small", "medium", "large", "huge")]
    for g, s in blocks:
        sel = [r for r in rows if group_of(r) == g and (r["source"] is None) == (s == "none") and r["group"] in supported]
        if not sel:
            continue
        print(f"\n## {g}, {'no unmetered load' if s == 'none' else 'with an unmetered load (methods are source-blind)'}: {len(sel)} cases")
        print("method        recall  precision  F1     supported  root-side supported  exact  nothing missed, nothing false")
        best = {}
        for m in methods:
            total = np.zeros(10)
            for r in sel:
                res = (extra.get(r["name"], {}) if m == "wzzt_r" else r).get("methods", {}).get(m)
                if res is not None:
                    total += counts(r, res["lines"], supported[r["group"]])
            if total[9] == 0:
                continue
            rec, pre = total[0]/max(total[1], 1), total[0]/max(total[2], 1)
            f1 = 2*rec*pre/max(rec+pre, 1e-12)
            line = (f"{m:12s} {100*rec:6.1f}  {100*pre:8.1f}  {100*f1:5.1f}  {int(total[3]):4d}/{int(total[4]):<4d}"
                    f"  {int(total[5]):4d}/{int(total[6]):<4d} {int(total[7]):10d}/{int(total[9])} {int(total[8]):6d}/{int(total[9])}")
            b = m.split("|")[0]
            if "|" in m:
                best.setdefault(b, []).append((f1, line, m == f"{b}|{REAL_SETTING[b]}"))
            else:
                print(line)
        for b, items in best.items():
            print(next(l for f, l, fixed in items if fixed), " (fixed)")
            print(max(items)[1], " (best tolerance in hindsight)")


def sorted_evidence(row):
    """check.evidence lists the lines by (size, members), the same order as `truth`."""
    return row["evidence"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "table"))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--wzzt_r", action="store_true")
    args = ap.parse_args()
    run(args.workers, args.wzzt_r) if args.what == "run" else table()


if __name__ == "__main__":
    main()
