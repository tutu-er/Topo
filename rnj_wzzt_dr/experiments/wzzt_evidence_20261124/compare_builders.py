"""Amendment D of PROTOCOL_MIQP.md: root lines and terminal lines, WZZT-R against RNJ, NJ and RG.

For every input: the best-fitting R/X matrix (ordered symmetric fit to P and Q on WZZT-R's slots, per-meter
demeaned, Gurobi QP), then RNJ, NJ and RG at 3 weights x 5 tolerances; WZZT-R's selected family is read from the
stored run. Every internal line of the identifiable truth gets its position f = |C|/n, relative length
rho = r_C / median root-to-meter resistance, and whole-map evidence dJ (study judge, Phase 3 ceiling routine).

    python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.compare_builders --set phase3|eulv_16d|eulv_large
Writes outputs/wzzt_evidence_20261124/builders/<set>.jsonl (resumable).
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

from rnj_wzzt_dr.experiments.hourly_probing_20261118 import model
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import capture
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_miqp as rm
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase3 as p3
from rnj_wzzt_dr.identification import trees
from rnj_wzzt_dr.physical import backend

OUTB = capture.OUT/"builders"
MATRIX = os.environ.get("BUILDER_MATRIX", "pq")   # "pq": P and Q ordered fit (Amendment D); "ponly": WZZT-R's own
BUILDERS = dict(rnj=trees.rnj, nj=trees.nj, rg=trees.rg)
WEIGHTS = (1., .75, 0.)
TOLERANCES = (0., .04, .08, .16, .24)
DEFAULT = (.75, .16)


def internal(family, n):
    return {tuple(sorted(C)) for C in family if 1 < len(C) < n}


def jobs(which):
    if which == "phase3":
        rows = [json.loads(l) for l in (p3.OUT3/"R.jsonl").read_text().splitlines() if '"error"' not in l]
        return [(which, r["cond"], r["name"], r["selected"]) for r in rows]
    rows = [json.loads(l) for l in (rm.OUTM/"fit.jsonl").read_text().splitlines() if '"error"' not in l]
    return [(which, r["cond"], r["name"], r["selected"]) for r in rows if r["cond"] == which and r["variant"] == "R"]


def load(which, cond, name):
    return p3.load(cond, name) if which == "phase3" else rm.load(cond, name)


def line_attributes(inp):
    order, parent, incoming = backend._radial_order(inp.network)
    desc = {b: set() for b in order}
    for i, label in enumerate(inp.labels):
        desc[label].add(i)
    for c in reversed(order[1:]):
        desc[parent[c]].update(desc[c])
    resistance = {}
    for c in order[1:]:
        key = tuple(sorted(desc[c]))
        if key:
            resistance[key] = resistance.get(key, 0.)+incoming[c]["r_ohm"]
    depth = []
    for label in inp.labels:
        total, b = 0., label
        while parent.get(b) is not None:
            total += incoming[b]["r_ohm"]; b = parent[b]
        depth.append(total)
    return resistance, float(np.median(depth))


def evidence(which, cond, inp, truth_lines):
    """dJ per truth line: the Phase 3 ceiling when stored, else the same routine with the real-feeder judge."""
    if which == "phase3":
        stored = p3._CEIL_FULL.get((cond, inp.name)) if hasattr(p3, "_CEIL_FULL") else None
        if stored is None:
            path = p3.OUT3/"ceiling.jsonl"
            p3._CEIL_FULL = {}
            for line in path.read_text().splitlines():
                r = json.loads(line)
                if "error" not in r:
                    p3._CEIL_FULL[(r["cond"], r["name"])] = {tuple(s["segment"]): s["dJ"] for s in r["segments"]}
            stored = p3._CEIL_FULL.get((cond, inp.name), {})
        return {C: stored.get(C) for C in truth_lines}
    n, j = inp.n, p3.judge("eulv", inp)
    truth = model.canonical_family(inp.truth_key, n)
    fams = [truth]+[tuple(C for C in truth if C != seg) for seg in truth_lines]
    fits = p2.study_J(fams, inp, j)
    N = inp.hist["Y"].size
    extra = n if j["offsets"] else 0
    sigma2 = fits[0].sse/(N-fits[0].c.p-extra)
    J = [f.sse/(2*sigma2)+.5*(f.c.p+extra)*math.log(N) for f in fits]
    return {seg: float(J[k+1]-J[0]) for k, seg in enumerate(truth_lines)}


def run(job):
    which, cond, name, selected = job
    try:
        started = time.perf_counter()
        inp = load(which, cond, name); n = inp.n
        tr, va = p2.series(inp)
        data = tr.join(va)
        dm = lambda a: a-a.mean(0)
        t0 = time.perf_counter()
        if MATRIX == "ponly":
            # WZZT-R's own guidance matrix: P-only ordered fit under its residual operator (offsets, GLS slot term)
            from rnj_wzzt_dr.identification import revised as rv
            cfg = rv.RevisedConfig()
            design = rv.Design(data, rv.Transform(data.p.shape[0], n, rv.root_gamma(data.v0sq, data.y), cfg.offsets), 1.)
            R, qp = rv.ordered_sensitivity(design, "gurobi")
            X, info = R, dict(solver="Gurobi barrier convex_QP (P-only under M)", iterations=qp["iterations"])
        else:
            R, X, info = trees.fit_sensitivity(dict(p=dm(data.p), q=dm(data.q), y=dm(data.y)), solver="gurobi")
        fit_seconds = time.perf_counter()-t0
        truth_lines = sorted(internal(inp.truth_key, n), key=lambda s: (len(s), s))
        resistance, med = line_attributes(inp)
        dJ = evidence(which, cond, inp, truth_lines)
        lines = [dict(segment=list(C), f=len(C)/n, rho=resistance.get(C, float("nan"))/med, dJ=dJ.get(C))
                 for C in truth_lines]
        truth = set(truth_lines)
        def score(family):
            got = internal(family, n)
            return dict(found="".join("1" if C in got else "0" for C in truth_lines), out=len(got),
                        out_true=len(got & truth))
        methods = {"wzzt_r": score(selected)}
        t0 = time.perf_counter()
        for weight in WEIGHTS:
            shared, depth, _ = trees.geometry(R, X, weight)
            for tol in TOLERANCES:
                for label, build in BUILDERS.items():
                    methods[f"{label}|{weight}|{tol}"] = score(build(shared, depth, tol))
        return dict(set=which, cond=cond, name=name, n=n, T=int(data.p.shape[0]), lines=lines, methods=methods,
                    fit=dict(solver=info["solver"], iterations=info["iterations"], seconds=fit_seconds),
                    build_seconds=time.perf_counter()-t0, seconds=time.perf_counter()-started)
    except Exception as exc:
        return dict(set=which, cond=cond, name=name, error=repr(exc), trace=traceback.format_exc()[-2000:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=("phase3", "eulv_16d", "eulv_large"))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    OUTB.mkdir(parents=True, exist_ok=True)
    path = OUTB/(f"{args.set}.jsonl" if MATRIX == "pq" else f"{args.set}_{MATRIX}.jsonl")
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                done.add((r["cond"], r["name"]))
    todo = [j for j in jobs(args.set) if (j[1], j[2]) not in done]
    print(f"{args.set}: {len(todo)} inputs to run ({len(done)} done)", flush=True)
    with open(path, "a") as out, get_context("spawn").Pool(args.workers) as pool:
        for k, rec in enumerate(pool.imap_unordered(run, todo), 1):
            out.write(json.dumps(rec)+"\n"); out.flush()
            if "error" in rec:
                print(f"[{k}/{len(todo)}] {rec['name']} ERROR {rec['error']}", flush=True)
            elif k % 25 == 0 or k == len(todo):
                print(f"[{k}/{len(todo)}] {rec['name']} n={rec['n']} {rec['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
