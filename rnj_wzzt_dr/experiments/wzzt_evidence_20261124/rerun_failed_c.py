"""Comparators rerun alone on the large inputs whose confirmation job failed (PROTOCOL_WZZT_C.md section 4 and section 6).

    python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.rerun_failed_c [--workers 4]

Section 4: "An input whose job fails as a whole ... counts as a failure of WZZT-C: the comparators are rerun on it alone
and keep their results, and WZZT-C gets an empty output." Here WZZT-C is not run; it gets an empty output. WZZT-R and
the distance builders run as in `run_confirm_c.run`, each in its own try block, so that a crash empties only the method
that crashes ("errors and crashes count against the method that fails"). The builders need `combo.setup`'s P-only
matrix; if that set-up fails, they are empty too. The truth lines and their evidence are computed as in `run`.
Output: outputs/wzzt_evidence_20261124/confirm_c/large_failed_rerun.jsonl (one row per input, same format as large.jsonl).
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

from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_confirm_c as rc

OUT = rc.OUTC/"large_failed_rerun.jsonl"


def failed_names():
    rows = [json.loads(l) for l in (rc.OUTC/"large.jsonl").read_text().splitlines()]
    ok = {r["name"] for r in rows if "error" not in r}
    return sorted({r["name"] for r in rows if "error" in r} - ok)


def run(name):
    from rnj_wzzt_dr.experiments.hourly_probing_20261118 import model
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124.compare_builders import line_attributes
    from rnj_wzzt_dr.identification import combo, revised as rv, trees
    cond = "eulv_large"
    started = time.perf_counter()
    try:
        inp = rc.load(cond, name); n = inp.n
        tr, va = p2.series(inp)
        internal = lambda f: {tuple(sorted(C)) for C in f if 1 < len(C) < n}
        truth_lines = sorted(internal(inp.truth_key), key=lambda s: (len(s), s))
        truth = set(truth_lines)
        j = rc.judge(cond, inp)
        fam_t = model.canonical_family(inp.truth_key, n)
        fits = p2.study_J([fam_t]+[tuple(C for C in fam_t if C != seg) for seg in truth_lines], inp, j)
        N = inp.hist["Y"].size
        extra = n if j["offsets"] else 0
        sigma2 = fits[0].sse/(N-fits[0].c.p-extra)
        J = [f.sse/(2*sigma2)+.5*(f.c.p+extra)*math.log(N) for f in fits]
        resistance, med = line_attributes(inp)
        lines = [dict(segment=list(C), f=len(C)/n, rho=resistance.get(C, float("nan"))/med, dJ=float(J[k+1]-J[0]))
                 for k, C in enumerate(truth_lines)]
    except Exception as exc:
        return dict(set="large", cond=cond, name=name, error=repr(exc), trace=traceback.format_exc()[-2000:])

    def score(family, seconds, **extra_fields):
        got = internal(family)
        return dict(found="".join("1" if C in got else "0" for C in truth_lines), out=len(got),
                    out_true=len(got & truth), seconds=seconds, **extra_fields)
    empty = trees.canonical(trees.basis(n))
    methods, errors = {}, {}
    S = crit = None
    try:
        S = combo.setup(tr, va)
        crit = lambda f: rv.exact_fit(S["design"], f, 2.)["objective"]+S["alpha"]*rv.internal_count(f, n)
    except Exception as exc:
        errors["setup"] = repr(exc)+" | "+traceback.format_exc()[-800:]
    # WZZT-C: empty output (failed job, section 4)
    methods["wzzt_c"] = score([], None, certified=False, k=None, criterion=None, failed_job=True,
                              criterion_same_design=None if crit is None else crit(empty))
    # WZZT-R (frozen, guidance solver per deviation D4)
    t0 = time.perf_counter()
    try:
        cfg = rv.RevisedConfig()
        orr = rv.fit(tr, va, cfg)
        methods["wzzt_r"] = score(orr["selected"], time.perf_counter()-t0, certified=orr["e"].get("certified"),
                                  criterion=None if crit is None else crit(orr["selected"]), qp_solver=cfg.qp_solver)
    except Exception as exc:
        errors["wzzt_r"] = repr(exc)+" | "+traceback.format_exc()[-800:]
        methods["wzzt_r"] = score([], time.perf_counter()-t0, certified=False, criterion=None, failed=True)
    # distance builders on WZZT-R's P-only matrix, every tolerance
    t0 = time.perf_counter()
    for label, build in (("rnj", trees.rnj), ("nj", trees.nj), ("rg", trees.rg)):
        for tol in rc.TOLERANCES:
            key = f"{label}|{tol}"
            try:
                if S is None:
                    raise RuntimeError("combo.setup failed: no P-only matrix")
                shared, depth, _ = trees.geometry(S["R"], S["R"], 1.)
                methods[key] = score(build(shared, depth, tol), None)
            except Exception as exc:
                errors[key] = repr(exc)
                methods[key] = score([], None, failed=True)
    return dict(set="large", cond=cond, name=name, n=n, lines=lines, methods=methods, errors=errors, rerun=True,
                build_seconds=time.perf_counter()-t0, seconds=time.perf_counter()-started)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    done = set()
    if OUT.exists():
        done = {r["name"] for r in map(json.loads, OUT.read_text().splitlines()) if "error" not in r}
    todo = [x for x in failed_names() if x not in done]
    print(f"failed jobs to rerun without WZZT-C: {todo}", flush=True)
    with open(OUT, "a") as fh, get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for k, rec in enumerate(pool.imap_unordered(run, todo), 1):
            fh.write(json.dumps(rec)+"\n"); fh.flush()
            print(f"[{k}/{len(todo)}] {rec['name']} {rec.get('error', '')} errors={list(rec.get('errors', {}))} "
                  f"{rec.get('seconds', 0):.0f}s", flush=True)


if __name__ == "__main__":
    main()
