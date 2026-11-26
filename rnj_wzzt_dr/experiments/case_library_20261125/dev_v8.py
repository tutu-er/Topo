"""Development runs of Algorithm 1 (identification_v8, draft v12) on the 288 development twins, with the diagnostics
that the tests of paper/PROPOSALS_UNDER_TEST.md need. Development cases only: load_dev_case checks every file against
the development manifest, and the sealed set is never opened.

    python -m rnj_wzzt_dr.experiments.case_library_20261125.dev_v8 run --workers 8 [--blind] [--names a,b] [--max-n 25]
    python -m rnj_wzzt_dr.experiments.case_library_20261125.dev_v8 summary [--blind]

`run` is resumable: cases already in the output file are skipped. Each row is the identifier's output plus
  truth        the true meter groupings
  label_truth  per label: is its grouping true; for a grouping outside the tree, "nested" (it could be added to the tree)
               or "crossing" (it crosses a line of the tree)
  adequacy     for test S1: F of the leader and of the true tree against the saturated model (section 2 of the note)
  complete     for test S2: the last pricing found no paying line and priced every node exactly
These diagnostics are measurements for the tests; the method itself (identification_v8) does not use them.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import statistics
import time
import traceback
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy import stats

from rnj_wzzt_dr.identification_v8.identify import Settings, identify, load_dev_case, f1
from rnj_wzzt_dr.identification_v8.labels import grouping
from rnj_wzzt_dr.identification_v8.model import balance, gls_gamma, time_basis
from rnj_wzzt_dr.identification_v8.sets import basis, compatible

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2]/"outputs"/"case_library_20261125"


# ------------------------------------------------------------------ S1: adequacy against the saturated model
def adequacy(P, Q, Y, V0sq, u, trees):
    """For each tree (name -> meter groupings): F = [(RSS_tree - RSS_sat) / df1] / [RSS_sat / df2], both unbounded
    least squares under the saturated model's projection (offsets and, with u, the load's profiles removed; S on the
    meters). A tree is nested in the saturated model (R_ij = sum of r_C over the lines C holding i and j), so
    F ~ F(df1, df2) when the tree is right and the noise is Gaussian. The saturated fit is done once.
    Returns {name: dict(F, df1, df2, p)} and sigma2 = RSS_sat / df2."""
    T, n = P.shape
    gamma = gls_gamma(V0sq, Y)
    E = time_basis(T, u)
    drop = lambda A: A-E@(E.T@A)
    Pp, Qp, y = drop(P), drop(Q), drop(Y-gamma*Y.mean(1, keepdims=True)).ravel()
    after_S = lambda F: (F-gamma*F.mean(1, keepdims=True)).reshape(-1)        # F: T x n, what each meter sees

    def fit(columns):
        X = np.column_stack(columns)
        theta, _, rank, _ = np.linalg.lstsq(X, y, rcond=1e-10)
        return float(np.sum((y-X@theta)**2)), int(rank)

    ii, jj = np.triu_indices(n)
    sat_cols = []
    for F0 in (Pp, Qp):                                    # entry (i, j): meter i sees flow j, meter j sees flow i
        for i, j in zip(ii, jj):
            F = np.zeros((T, n))
            F[:, i] += F0[:, j]
            if i != j:
                F[:, j] += F0[:, i]
            sat_cols.append(after_S(F))
    rss_s, rank_s = fit(sat_cols)
    del sat_cols
    df2 = (T-E.shape[1])*(n-int(gamma == 1.))-rank_s
    out = {}
    for name, groupings in trees.items():
        tree_cols = []
        for C in [*basis(n), *groupings]:                 # line C: each meter of C sees the summed flow of C
            Z = np.zeros(n)
            Z[list(C)] = 1.
            for F0 in (Pp, Qp):
                tree_cols.append(after_S(F0[:, list(C)].sum(1)[:, None]*Z[None, :]))
        rss_t, rank_t = fit(tree_cols)
        df1 = rank_s-rank_t
        F = ((rss_t-rss_s)/df1)/(rss_s/df2)
        out[name] = dict(F=F, df1=df1, df2=df2, p=float(stats.f.sf(F, df1, df2)))
    return out, rss_s/df2


# ------------------------------------------------------------------ one case
def run_one(task):
    """task = (name, blind, settings dict or None). Returns (name, row or None, error text or None, seconds)."""
    name, blind, settings = task
    started = time.perf_counter()
    try:
        st = Settings(**settings) if settings else Settings()
        rec, inp = load_dev_case(name)
        h, n = inp.hist, inp.n
        pcc = () if blind else (h["Ppcc"], h["Qpcc"])
        r = identify(h["P"], h["Q"], h["Y"], h["V0sq"], *pcc, st=st)
        truth = {g for g in (grouping(C, n) for C in rec["truth"]) if g}
        tree = {tuple(M) for M in r["tree"]}
        label_truth = []
        for l in r["labels"]:
            M = tuple(l["grouping"])
            relation = "inside" if l["inside"] else \
                ("nested" if all(compatible(M, C) for C in tree) else "crossing")
            label_truth.append(dict(grouping=list(M), label=l["label"], relation=relation, true=M in truth))
        u = None                                           # the load term exactly as the identifier modelled it
        if r["gate"]["load"]:
            u = balance(h["P"], h["Q"], h["Y"], h["V0sq"], h["Ppcc"], h["Qpcc"])
        priced = [x["price"] for x in r["rounds"] if "price" in x]
        fits, sigma2 = adequacy(h["P"], h["Q"], h["Y"], h["V0sq"], u, dict(leader=sorted(tree), truth=sorted(truth)))
        assert abs(sigma2/r["model"]["sigma2"]-1) < 1e-6, "the saturated fit differs from the identifier's"
        row = dict(name=name, n=n, load=rec["source"] and rec["source"]["where"], blind=blind,
                   score=f1(tree, truth), truth=[list(M) for M in sorted(truth)], label_truth=label_truth,
                   adequacy=fits,
                   complete=bool(priced) and priced[-1]["still_paying"] is None and bool(priced[-1]["certified"]),
                   **{k: v for k, v in r.items() if k not in ("lines", "pool")})
        return name, row, None, time.perf_counter()-started
    except Exception:
        return name, None, traceback.format_exc(), time.perf_counter()-started


def out_path(blind):
    return OUT/("dev_v8_blind.jsonl" if blind else "dev_v8.jsonl")


def run(args):
    manifest = json.loads((HERE/"manifest_dev.json").read_text(encoding="utf-8"))
    names = [r["name"] for r in sorted(manifest, key=lambda r: -r["n"])            # largest first
             if args.max_n is None or r["n"] <= args.max_n]
    if args.names:
        names = [x for x in names if x in set(args.names.split(","))]
    path = out_path(args.blind)
    done = set()
    if path.exists():
        done = {json.loads(line)["name"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [x for x in names if x not in done]
    print(f"{len(todo)} cases to run, {len(done)} done; output {path}", flush=True)
    tasks = [(x, args.blind, None) for x in todo]
    with open(path, "a", encoding="utf-8") as fh, open(path.with_suffix(".log"), "a", encoding="utf-8") as log, \
            get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for k, (name, row, error, seconds) in enumerate(pool.imap_unordered(run_one, tasks), 1):
            if row is not None:
                fh.write(json.dumps(row, default=str)+"\n")
                fh.flush()
                line = f"[{k}/{len(todo)}] {name}: F1 {row['score']['f1']:.3f}, stop {row['stop'][:9]}, {seconds:.0f} s"
            else:
                line = f"[{k}/{len(todo)}] {name}: ERROR after {seconds:.0f} s\n{error}"
            print(line, flush=True)
            log.write(line+"\n")
            log.flush()


# ------------------------------------------------------------------ summary
def summary(args):
    rows = [json.loads(line) for line in out_path(args.blind).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"{len(rows)} cases")
    by_load = Counter(r["load"] or "none" for r in rows)
    for load in sorted(by_load):
        sel = [r for r in rows if (r["load"] or "none") == load]
        print(f"  load {load:6s}: {len(sel):3d} cases, mean F1 {statistics.mean(r['score']['f1'] for r in sel):.3f}, "
              f"gate on {sum(r['gate']['load'] for r in sel)}")
    secs = [r["timing"]["total"] for r in rows]
    print(f"time per case: median {statistics.median(secs):.0f} s, max {max(secs):.0f} s; "
          f"stop: {dict(Counter(r['stop'][:9] for r in rows))}")
    inside = Counter((x["label"], x["true"]) for r in rows for x in r["label_truth"] if x["relation"] == "inside")
    print("groupings of the tree (label, true):", dict(inside))
    outside = Counter((x["relation"], x["true"]) for r in rows for x in r["label_truth"] if x["relation"] != "inside")
    print("open groupings outside the tree (relation, true):", dict(outside))
    for who in ("truth", "leader"):
        flagged = sum(r["adequacy"][who]["p"] < .01 for r in rows)
        print(f"S1, {who}: flagged at the 1% level in {flagged} of {len(rows)} cases")
    print(f"S2: pricing complete in {sum(r['complete'] for r in rows)} of {len(rows)} cases")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("run", "summary"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--blind", action="store_true", help="without the PCC power")
    ap.add_argument("--names", default=None, help="only these development cases, comma-separated")
    ap.add_argument("--max-n", type=int, default=None, help="only cases with at most this many meters")
    args = ap.parse_args()
    run(args) if args.command == "run" else summary(args)


if __name__ == "__main__":
    main()
