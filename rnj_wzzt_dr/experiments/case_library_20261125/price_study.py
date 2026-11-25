"""Exploration: can the price of a line be set by physics and by the shape of real feeders, instead of by a heuristic?

Development cases without a source, at most 25 meters. Same candidate lines for every setting (the nine sources of
WZZT-C); the tree is the best one by local search (six starts) of

    0.5 * SSE / sigma^2 + sum over lines of price(line)

BIC:       price = (p_s / 2) ln N for every line (the method as it is; p_s = 1 or 2).
factor f:  f times the BIC price (f = 0 resolves every node; f = 1 is the method).
lambda:    BIC price minus lambda. lambda = log prior odds that a three-way node is two two-way nodes; a prior that
           favours two-way nodes is geometric in the excess and therefore linear per line (one number, from real feeders).
info:      price = 0.5 ln(1 + tau^2 I_r) + 0.5 ln(1 + tau_x^2 I_x) - lambda: the marginal likelihood under a physical prior
           on the line's r and x (tau: spread of the resistances of real line sections), with I the information of the
           line's own column after the base columns (singles and trunk) are taken out.

    python -m rnj_wzzt_dr.experiments.case_library_20261125.price_study run [--workers 6]
    python -m rnj_wzzt_dr.experiments.case_library_20261125.price_study table
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import time
import traceback
from multiprocessing import get_context

import numpy as np

from . import build, compare_dev as cd

OUT = build.OUT/"price_study.jsonl"
OUT_REAL = build.OUT/"price_study_real.jsonl"
FACTORS = (1., .75, .5, .25, .1, 0.)
LAMBDAS = (0., 2., 4., 6.)
TAU_R_OHM = .010                          # spread (ohm) of the resistance of a line section on real feeders
TAU_X_RATIO = .3                          # the same for x, relative to r
INFO_LAMBDAS = (0., 1., 2., 4.)


def run_one(name, real=False):
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_confirm_c import WZZT_C
    from rnj_wzzt_dr.identification import combo, revised as rv
    from rnj_wzzt_dr.identification.trees import basis, canonical
    try:
        if real:
            from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import capture
            inp = capture.load_input("eulv", name)
            truth = inp.truth_key
            group, network = inp.case, inp.network
        else:
            rec = build.load("dev", name)
            inp = build.to_input(rec)
            truth, group, network = rec["truth"], rec["group"], rec["network"]
        n = inp.n
        tr, va = p2.series(inp)
        S = combo.setup(tr, va)
        design, N = S["design"], S["N"]
        zbase = network["base_kv"]**2/network["base_mva"]
        tau_r = 2*TAU_R_OHM/zbase
        fams = [combo.source_family(S, s) for s in WZZT_C["sources"]]
        lines = canonical(C for F in fams for C in F if 1 < len(C) < n)
        base = canonical(basis(n))
        fitter = rv.GramFitter(design, [*lines, *base], base, 2.)
        prob = rv.Problem(fitter, lines, base, 0.)
        # information of every candidate line's r and x after the base columns
        F0 = design.columns(base)
        Q0, _ = np.linalg.qr(F0)
        info = {}
        for C in lines:
            f = design.columns([C])
            f = f-Q0@(Q0.T@f)
            Ir = float(f[:, 0]@f[:, 0])
            fx = f[:, 1]-(f[:, 1]@f[:, 0])/max(Ir, 1e-300)*f[:, 0]
            info[C] = (Ir, float(fx@fx))
        bic = .5*S["p_s"]*math.log(N)                           # half-BIC units
        prices = {}
        for f in FACTORS:
            prices[f"bic*{f}"] = {C: f*bic for C in lines}
        for lam in LAMBDAS:
            prices[f"bic-{lam:g}"] = {C: bic-lam for C in lines}
        for lam in INFO_LAMBDAS:
            prices[f"info-{lam:g}"] = {C: .5*math.log(1+tau_r**2*info[C][0])
                                       + (.5*math.log(1+(TAU_X_RATIO*tau_r)**2*info[C][1]) if S["p_s"] == 2 else 0.)
                                       - lam for C in lines}
        starts = [frozenset()]+[frozenset(C for C in F if C in prob.crosses) for F in fams[:3]]
        out = dict(name=name, group=group, n=n, p_s=S["p_s"], truth=cd.as_lists(
            {tuple(sorted(C)) for C in truth if 1 < len(C) < n}), methods={})
        if real:
            from rnj_wzzt_dr.identification import trees
            shared, depth, _ = trees.geometry(S["R"], S["R"], 1.)
            for lab, make in (("rnj", trees.rnj), ("nj", trees.nj)):
                for tol in cd.TOLERANCES:
                    out["methods"][f"{lab}|{tol}"] = dict(lines=cd.as_lists(
                        {tuple(sorted(C)) for C in make(shared, depth, tol) if 1 < len(C) < n}), seconds=0.)
        for label, p in prices.items():
            t0 = time.perf_counter()
            class P(rv.Problem):
                def score(self, T, p=p):
                    return self.f.fit([*self.keep, *T])[0]*N/2+sum(p[C] for C in T)
            q = P(fitter, lines, base, 0.)
            best = None
            for s in starts:
                x = rv.local_search(q, s, time.perf_counter()+15.)
                best = x if best is None or x[1] < best[1] else best
            out["methods"][label] = dict(lines=cd.as_lists(best[0]), seconds=time.perf_counter()-t0,
                                         value=float(best[1]))
        out["price_info"] = [float(np.median(list(prices["info-0"].values()))), float(np.median(list(prices["bic*1.0"].values())))]
        return out
    except Exception as exc:
        return dict(name=name, error=repr(exc), trace=traceback.format_exc()[-1500:])


def run_real_one(name):
    return run_one(name, real=True)


def run(workers, real=False):
    path = OUT_REAL if real else OUT
    done = set()
    if path.exists():
        done = {r["name"] for r in map(json.loads, path.read_text().splitlines()) if "error" not in r}
    if real:
        from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import capture
        from rnj_wzzt_dr.experiments.hourly_probing_20261118 import experiments as hx
        names = sorted(p.stem for p in (hx.ROOT/"eulv"/"inputs").glob("*.pkl"))
        combo_rows = [json.loads(x) for x in (capture.OUT/"combo"/"ev_pairs.jsonl").read_text().splitlines()]
        dev = {r["name"] for r in combo_rows if r["cond"] == "eulv"}
        todo = [x for x in names if x in dev and x not in done]
    else:
        manifest = json.loads((build.HERE/"manifest_dev.json").read_text(encoding="utf-8"))
        todo = [r["name"] for r in sorted(manifest, key=lambda r: (r["n"], r["name"]))
                if r["n"] <= 25 and r["source"] == "none" and r["name"] not in done]
    with open(path, "a") as fh, get_context("spawn").Pool(workers, maxtasksperchild=6) as pool:
        for k, row in enumerate(pool.imap_unordered(run_real_one if real else run_one, todo), 1):
            fh.write(json.dumps(row)+"\n"); fh.flush()
            print(f"[{k}/{len(todo)}] {row['name']} {row.get('error', '')}", flush=True)


def table(real=False):
    rows = [r for r in map(json.loads, (OUT_REAL if real else OUT).read_text().splitlines())]
    print(len(rows), "cases; errors", [r["name"] for r in rows if "error" in r][:3])
    rows = [r for r in rows if "error" not in r]
    blind = {} if real else {r["name"]: r for r in map(json.loads, cd.OUT.read_text().splitlines()) if "error" not in r}
    print("median price per line, half-BIC units: info-0", np.median([r["price_info"][0] for r in rows]).round(2), "BIC", np.median([r["price_info"][1] for r in rows]).round(2))
    for g in (("real",) if real else ("star", "street", "small", "medium")):
        sel = rows if real else [r for r in rows if cd.group_of(dict(set="named" if "_star_" in r["name"] or "_street_" in r["name"] else "random", name=r["name"])) == g]
        if not sel:
            continue
        print(f"\n## {g}: {len(sel)} cases")
        def line(label, get):
            tp = fp = fn = 0
            for r in sel:
                got, t = {tuple(x) for x in get(r)}, {tuple(x) for x in r["truth"]}
                tp += len(got & t); fp += len(got-t); fn += len(t-got)
            rec, pre = tp/max(tp+fn, 1), tp/max(tp+fp, 1)
            print(f"  {label:12s} recall {100*rec:5.1f}  precision {100*pre:5.1f}  F1 {100*2*rec*pre/max(rec+pre, 1e-9):5.1f}")
        for label in rows[0]["methods"]:
            line(label, lambda r: r["methods"][label]["lines"])
        for t in (() if real else ("rnj|0.0", "rnj|0.08", "nj|0.04")):
            line(t, lambda r: blind[r["name"]]["methods"][t]["lines"] if r["name"] in blind else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "table"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args()
    run(args.workers, args.real) if args.what == "run" else table(args.real)


if __name__ == "__main__":
    main()
