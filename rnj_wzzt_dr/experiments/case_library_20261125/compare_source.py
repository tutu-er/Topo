"""Development comparison with the unmetered load known to the methods (identification/source.py), DEVELOPMENT cases only.

    python -m rnj_wzzt_dr.experiments.case_library_20261125.compare_source run [--workers 4] [--max_n 25] [--twins 4]
    python -m rnj_wzzt_dr.experiments.case_library_20261125.compare_source table

Methods: S1 (source projected out, WZZT-C unchanged), S2 (source located, pool MIQP over n + 1 leaves), and the
distance builders on the projected matrix, so every method gets the PCC balance. The source-blind results are those of
compare_dev. Output: outputs/case_library_20261125/compare_source.jsonl (resumable).
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

OUT = build.OUT/"compare_source.jsonl"


def run_one(name):
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_confirm_c import WZZT_C
    from rnj_wzzt_dr.identification import source as src, trees
    try:
        rec = build.load("dev", name)
        inp = build.to_input(rec); n = inp.n
        tr, va = p2.series(inp)
        inner = lambda f: {tuple(sorted(C)) for C in f if 1 < len(C) < n}
        t0 = time.perf_counter()
        r = src.fit(tr, va, inp.hist["Ppcc"], inp.hist["Qpcc"], settings=WZZT_C)
        out = dict(name=name, group=rec["group"], set=rec["set"], source=rec["source"] and rec["source"]["where"],
                   share=rec["source"] and rec["source"]["share"], kind=rec["source"] and rec["source"]["kind"], n=n,
                   truth=cd.as_lists(inner(rec["truth"])), seconds=time.perf_counter()-t0, methods={})
        out["methods"]["s1"] = dict(lines=cd.as_lists(inner(r["selected"])), certified=r["record"]["certified"])
        out["methods"]["s2"] = dict(lines=cd.as_lists(inner(r["located"])), status=r["status"])
        shared, depth, _ = trees.geometry(r["S"]["R"], r["S"]["R"], 1.)
        for label, make in (("rnj", trees.rnj), ("nj", trees.nj), ("rg", trees.rg)):
            for tol in cd.TOLERANCES:
                out["methods"][f"{label}|{tol}"] = dict(lines=cd.as_lists(inner(make(shared, depth, tol))))
        out["evidence"] = float(r["source_evidence"])
        out["path"] = [list(C) for C in r["path"]]
        out["true_path"] = [list(C) for C in rec["source"]["path"]] if rec["source"] else None
        out["places"] = [[float(c), None if D is None else list(D)] for c, D in r["places"]]
        up = r["u"][0]
        rows = slice((build.DAYS-1)*24-len(up), (build.DAYS-1)*24)
        from . import physics
        truth_u = physics.hourly(rec["source"]["P"])[rows] if rec["source"] else np.zeros(len(up))
        out["u_error"] = [float(np.mean(up-truth_u)), float(np.std(up-truth_u)), float(np.mean(truth_u)), float(np.std(truth_u))]
        return out
    except Exception as exc:
        return dict(name=name, error=repr(exc), trace=traceback.format_exc()[-1500:])


def run(workers, max_n):
    manifest = json.loads((build.HERE/"manifest_dev.json").read_text(encoding="utf-8"))
    done = set()
    if OUT.exists():
        done = {r["name"] for r in map(json.loads, OUT.read_text().splitlines()) if "error" not in r}
    todo = [r["name"] for r in sorted(manifest, key=lambda r: (r["n"], r["name"])) if r["n"] <= max_n and r["name"] not in done]
    with open(OUT, "a") as fh, get_context("spawn").Pool(workers, maxtasksperchild=8) as pool:
        for k, row in enumerate(pool.imap_unordered(run_one, todo), 1):
            fh.write(json.dumps(row)+"\n"); fh.flush()
            print(f"[{k}/{len(todo)}] {row['name']} {row.get('error', '')} {row.get('seconds', 0):.0f}s", flush=True)


# ------------------------------------------------ tables
def add(total, row, lines, supported):
    return total+cd.counts(row, lines, supported)


def line_of(label, t):
    rec, pre = t[0]/max(t[1], 1), t[0]/max(t[2], 1)
    return (f"{label:22s} {100*rec:6.1f}  {100*pre:8.1f}  {100*2*rec*pre/max(rec+pre, 1e-12):5.1f}  "
            f"{int(t[3]):4d}/{int(t[4]):<4d}  {int(t[5]):4d}/{int(t[6]):<4d}")


def table():
    rows = [r for r in map(json.loads, OUT.read_text().splitlines())]
    errors = [r for r in rows if "error" in r]
    rows = {r["name"]: r for r in rows if "error" not in r}
    blind = {r["name"]: r for r in map(json.loads, cd.OUT.read_text().splitlines()) if "error" not in r}
    supported = {r["group"]: {tuple(C) for C, (_, d) in zip(r["truth"], r["evidence"]) if d > 0}
                 for r in blind.values() if r["source"] is None}
    both = [r for r in rows.values() if r["group"] in supported]
    print(f"{len(rows)} cases, {len(errors)} errors {[e['name'] for e in errors][:4]}")
    for g in ("star", "street", "small", "medium", "large", "huge"):
        for s in ("none", "head", "middle", "end"):
            sel = [r for r in both if cd.group_of(r) == g and (r["source"] or "none") == s]
            if not sel:
                continue
            print(f"\n## {g}, unmetered load: {s}; {len(sel)} cases")
            print("method                  recall  precision  F1     supported  root-side supported")
            names = {"s1": "S1 (projected)", "s2": "S2 (located)", "rnj|0.08": "RNJ 0.08, projected",
                     "nj|0.04": "NJ 0.04, projected", "rg|0.08": "RG 0.08, projected"}
            for m, label in names.items():
                t = np.zeros(10)
                for r in sel:
                    t = add(t, r, r["methods"][m]["lines"], supported[r["group"]])
                print(line_of(label, t))
            for b in ("rnj", "nj", "rg"):
                best = None
                for tol in cd.TOLERANCES:
                    t = np.zeros(10)
                    for r in sel:
                        t = add(t, r, r["methods"][f"{b}|{tol}"]["lines"], supported[r["group"]])
                    f1 = 2*t[0]/max(t[1]+t[2], 1)
                    if best is None or f1 > best[0]:
                        best = (f1, line_of(f"{b.upper()} {tol} best, proj.", t))
                print(best[1])
            have = [r for r in sel if r["name"] in blind]
            for m, label in (("wzzt_c", "WZZT-C, blind"), ("rnj|0.08", "RNJ 0.08, blind"), ("nj|0.04", "NJ 0.04, blind")):
                t = np.zeros(10)
                for r in have:
                    t = add(t, r, blind[r["name"]]["methods"][m]["lines"], supported[r["group"]])
                if t[9]:
                    print(line_of(label, t)+(f"   ({int(t[9])} cases)" if t[9] != len(sel) else ""))
            exist = np.array([r["evidence"] for r in sel])
            print(f"source evidence (half-BIC): min {exist.min():.1f}, median {np.median(exist):.1f}, max {exist.max():.1f}; "
                  f"above 4.6: {int((exist > 4.6).sum())}/{len(sel)}")
            if s != "none":
                print(f"source placed correctly among the lines of the returned tree: {sum(map(path_ok, sel))}/{len(sel)}")


def path_ok(row):
    """The returned lines marked as carrying the source are exactly the returned lines that truly carry it."""
    n, true = row["n"], {tuple(C) for C in row["true_path"]}
    tree = {tuple(C) for C in row["methods"]["s2"]["lines"]}
    return {tuple(C) for C in row["path"] if len(C) < n} == {C for C in true if C in tree}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "table"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_n", type=int, default=25)
    args = ap.parse_args()
    run(args.workers, args.max_n) if args.what == "run" else table()


if __name__ == "__main__":
    main()
