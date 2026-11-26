"""The likely-lines list (identification/likely.py) on the DEVELOPMENT library, before it is registered as a secondary output.

    python -m rnj_wzzt_dr.experiments.case_library_20261125.likely_dev run --mode blind [--workers 6]
    python -m rnj_wzzt_dr.experiments.case_library_20261125.likely_dev run --mode s1    [--workers 6]
    python -m rnj_wzzt_dr.experiments.case_library_20261125.likely_dev table

blind: no-source cases; the final method without the PCC is run afresh (default Settings) and its tree is listed.
s1:    every twin with a finished row in final_source.jsonl; the stored S1 tree on the source-projected set-up is listed.
Output: outputs/case_library_20261125/likely_dev.jsonl (resumable).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import traceback
from multiprocessing import get_context

import numpy as np

from . import build, compare_dev as cd, final_dev as fd

OUT = build.OUT/"likely_dev.jsonl"
BINS = ((-4.6, -2.), (-2., 0.), (0., np.inf))


def run_one(job):
    mode, name = job
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2
    from rnj_wzzt_dr.identification import final, likely, revised as rv, source as src
    try:
        rec = build.load("dev", name)
        inp = build.to_input(rec); n = inp.n
        tr, va = p2.series(inp)
        if mode == "blind":
            r = final.fit(tr, va)
            S, tree = r["S"], r["tree"]
        else:
            row = next(x for x in map(json.loads, fd.OUT["source"].read_text().splitlines()) if x["name"] == name and "error" not in x)
            up, uq = src.balance(inp.hist["Ppcc"], inp.hist["Qpcc"], tr.join(va))
            S = final.priced(final.guarded(src.setup_projected, tr, va, up, uq, rv.RevisedConfig()), final.LAMBDA)
            tree = [tuple(C) for C in row["s1"]["lines"]]
        listed = likely.likely_lines(S, tree, n)
        return dict(mode=mode, name=name, group=rec["group"], set=rec["set"], n=n, source=rec["source"] and rec["source"]["where"],
                    truth=cd.as_lists(fd.inner(rec["truth"], n)), tree=cd.as_lists(fd.inner(tree, n)),
                    likely=[[list(C), e] for C, e in listed])
    except Exception as exc:
        return dict(mode=mode, name=name, error=repr(exc), trace=traceback.format_exc()[-1500:])


def run(mode, workers):
    manifest = json.loads((build.HERE/"manifest_dev.json").read_text(encoding="utf-8"))
    done = set()
    if OUT.exists():
        done = {(r["mode"], r["name"]) for r in map(json.loads, OUT.read_text().splitlines()) if "error" not in r}
    if mode == "blind":
        names = [r["name"] for r in sorted(manifest, key=lambda r: (-r["n"], r["name"])) if r["source"] == "none"]
    else:
        names = [r["name"] for r in map(json.loads, fd.OUT["source"].read_text().splitlines()) if "error" not in r]
    todo = [(mode, x) for x in names if (mode, x) not in done]
    with open(OUT, "a") as fh, get_context("spawn").Pool(workers, maxtasksperchild=4) as pool:
        for k, row in enumerate(pool.imap_unordered(run_one, todo), 1):
            fh.write(json.dumps(row)+"\n"); fh.flush()
            print(f"[{k}/{len(todo)}] {row['mode']} {row['name']} {row.get('error', '')}", flush=True)


def summary(rows):
    """Listed lines per case, share true, by evidence bin, and the tree's missed true lines that the list holds."""
    listed = true = missed = recovered = 0
    by_bin = np.zeros((len(BINS), 2))
    for r in rows:
        t, tree = {tuple(C) for C in r["truth"]}, {tuple(C) for C in r["tree"]}
        L = [(tuple(C), e) for C, e in r["likely"]]
        listed += len(L); true += sum(C in t for C, _ in L)
        missed += len(t-tree); recovered += sum(C in t-tree for C, _ in L)
        for C, e in L:
            k = next(i for i, (a, b) in enumerate(BINS) if a < e <= b or (b == np.inf and e > a))
            by_bin[k] += (C in t, 1)
    bins = "  ".join(f"({a}, {b}]: {int(h)}/{int(c)}" for (a, b), (h, c) in zip(BINS, by_bin) if c)
    return (f"{len(rows):3d} cases, {listed/max(len(rows), 1):.1f} listed per case, {100*true/max(listed, 1):5.1f}% true "
            f"({true}/{listed}); missed true lines on the list {recovered}/{missed}   by evidence {bins}")


def table():
    rows = [r for r in map(json.loads, OUT.read_text().splitlines())]
    print(f"{sum('error' in r for r in rows)} errors")
    rows = [r for r in rows if "error" not in r]
    for mode in ("blind", "s1"):
        rs = [r for r in rows if r["mode"] == mode]
        if not rs:
            continue
        print(f"\n{mode}: {summary(rs)}")
        for g in ("star", "street", "small", "medium", "large", "huge"):
            sub = [r for r in rs if cd.group_of(r) == g]
            if sub:
                print(f"  {g:7s} {summary(sub)}")
        if mode == "s1":
            for s in ("none", "head", "middle", "end"):
                sub = [r for r in rs if (r["source"] or "none") == s]
                if sub:
                    print(f"  source {s:7s} {summary(sub)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "table"))
    ap.add_argument("--mode", choices=("blind", "s1"), default="blind")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    run(args.mode, args.workers) if args.what == "run" else table()


if __name__ == "__main__":
    main()
