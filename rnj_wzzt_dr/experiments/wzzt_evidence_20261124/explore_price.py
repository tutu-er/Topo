"""Exploratory (after D2, not part of any registered decision): does the lower price help stage 0 downstream?

Phase 2's ablation R_alpha (the original's fixed price 0.003 instead of the BIC price) recovers more synthetic
truths exactly but fewer supported real segments. The probing study does not use the selected family alone: it seeds
its closure with every family stage 0 outputs. This script runs the study closure on the 130 development inputs from
  - "R_alpha": the R_alpha families only, and
  - "R+R_alpha": the union of the R and R_alpha families (what a two-price output of WZZT-R would seed),
and writes phase2/bank_explore.jsonl in the format of bank.jsonl (closure as in the study, no R3).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
import traceback
from collections import defaultdict
from multiprocessing import Pool
from statistics import median

from rnj_wzzt_dr.experiments.hourly_probing_20261118 import model
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import capture
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase2 import OUT, _ladder_rows, downstream

PATH = OUT/"bank_explore.jsonl"
SOURCES = {"R_alpha": ("R_alpha",), "R+R_alpha": ("R", "R_alpha")}
_L = {}


def one(job):
    cond, name, source = job
    started = time.perf_counter()
    try:
        inp = capture.load_input(cond, name)
        n = inp.n
        fams = []
        for v in SOURCES[source]:
            if v not in _L:
                _L[v] = _ladder_rows(v)
            fams += [tuple(map(tuple, f)) for f in _L[v][(cond, name)]["families"]]
        full = [model.canonical_family([*f, *((i,) for i in range(n)), tuple(range(n))], n) for f in fams]
        out = dict(cond=cond, name=name, source=source, r3=False, seeds=len(set(full)),
                   **downstream(cond, inp, full, False))
    except Exception as exc:
        out = dict(job=list(job), error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc())
    out["job_seconds"] = time.perf_counter()-started
    return out


def report():
    rows = [json.loads(line) for line in PATH.read_text().splitlines()]
    rows = [r for r in rows if "error" not in r]
    base = [json.loads(line) for line in (OUT/"bank.jsonl").read_text().splitlines()]
    rows += [r for r in base if "error" not in r and not r.get("r3") and r["source"] in ("R", "orig")]
    B = defaultdict(dict)
    for r in rows:
        B[(r["cond"], r["name"])][r["source"]] = r
    print("| Seeds | Condition | Truth in bank | Closure finished | Leader correct | Bank size (median) | Median closure time |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for cond in ("eulv", "lock_short01", "lock_noisy"):
        for src in ("orig", "R", "R_alpha", "R+R_alpha"):
            rs = [B[k][src] for k in B if k[0] == cond and src in B[k]]
            if rs:
                print(f"| {src} | {cond} | {sum(r['truth_in_bank'] for r in rs)}/{len(rs)} "
                      f"| {sum(r['closure']['capped'] is None for r in rs)} | {sum(r['leader_correct'] for r in rs)} "
                      f"| {median(r['bank'] for r in rs):.0f} | {median(r['closure_seconds'] for r in rs):.0f} s |")
    both = [k for k in B if "R" in B[k] and "R+R_alpha" in B[k]]
    gain = [k for k in both if B[k]["R+R_alpha"]["truth_in_bank"] and not B[k]["R"]["truth_in_bank"]]
    loss = [k for k in both if B[k]["R"]["truth_in_bank"] and not B[k]["R+R_alpha"]["truth_in_bank"]]
    print(f"\nR+R_alpha against R: truth gained on {len(gain)}, lost on {len(loss)} of {len(both)} inputs")
    for k in loss:
        print("  lost:", k, "closure capped:", B[k]["R+R_alpha"]["closure"]["capped"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return
    done = set()
    if PATH.exists():
        for line in PATH.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                done.add((r["cond"], r["name"], r["source"]))
    jobs = [(c, nm, s) for s in SOURCES for c, nm in capture.select_inputs() if (c, nm, s) not in done]
    print(f"explore_price: {len(jobs)} jobs", flush=True)
    with Pool(args.workers, maxtasksperchild=4) as pool, open(PATH, "a") as fh:
        for k, out in enumerate(pool.imap_unordered(one, jobs), 1):
            fh.write(json.dumps(out)+"\n")
            fh.flush()
            print(f"[{k}/{len(jobs)}] {out.get('cond')} {out.get('name')} {out.get('source')} "
                  f"{out.get('error', '')[:80]} {out['job_seconds']:.0f}s", flush=True)
    report()


if __name__ == "__main__":
    main()
