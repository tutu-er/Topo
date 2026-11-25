"""WZZT-C confirmation runs (PROTOCOL_WZZT_C.md): WZZT-C (frozen), WZZT-R, the original WZZT, NJ, RNJ, RG.

    python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_confirm_c --set synthetic|large [--workers 3]

Per input: the identifiable truth's internal lines with position f, relative length rho and whole-map evidence dJ
(the study judge with the input's own settings); for every method the lines found, the lines output and the true
lines output; times. Output: outputs/wzzt_evidence_20261124/confirm_c/<set>.jsonl (resumable).
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

from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import build_confirm as bc
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import capture

OUTC = bc.OUTC
# WZZT-C as frozen in PROTOCOL_WZZT_C.md (do not change after the registration)
WZZT_C = dict(sources=("rnj", "nj", "rg", "volt", "volt_m", "resid", "rrows", "upgma", "ward"),
              blocks_from="margin", m=4.6, k_max=9, repairs=10, pairs=True, pair_rounds=3, reselect=True,
              miqp_seconds=240., node_seconds=60., search_seconds=30.)
# distance builders on WZZT-R's P-only matrix, settings fixed from the development analyses (Amendment D, P-only)
BUILDER_SETTINGS = {"real": dict(rnj=.08, nj=.04, rg=.08), "synthetic": dict(rnj=.24, nj=.24, rg=.24)}
TOLERANCES = (0., .04, .08, .16, .24)
# large inputs without a development meter; the only real inputs the decision uses (PROTOCOL_WZZT_C.md, section 3)
CLEAN = {"eulv_td006145_feeder24_s0", "eulv_td006145_feeder24_s6", "eulv_td008690_feeder18_s0",
         "eulv_td008690_feeder18_s6", "eulv_td008742_feeder80_s0", "eulv_td008742_feeder80_s6",
         "eulv_td009451_feeder140_s0", "eulv_td009451_feeder140_s6", "eulv_td400079_feeder144_s0",
         "eulv_td400859_feeder31_s0", "eulv_td400859_feeder31_s6", "eulv_td401346_all_s0", "eulv_td401346_all_s6"}
JUDGE = {"conf_short01": "lock_short01", "conf_noisy": "lock_noisy", "eulv_large": "eulv"}


def jobs(which):
    if which == "synthetic":
        rows = json.loads((OUTC/"index.json").read_text())
        return [("synthetic", r["cond"], r["name"]) for r in rows if "error" not in r]
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_miqp as rm
    names = sorted(p.stem for p in (rm.OUTM/"inputs"/"eulv_large").glob("*.pkl"))
    names.sort(key=lambda nm: nm not in CLEAN)          # the 13 clean inputs first (stable: alphabetical within)
    return [("large", "eulv_large", nm) for nm in names]


def load(cond, name):
    if cond == "eulv_large":
        from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_miqp as rm
        return rm.load(cond, name)
    return bc.load(cond, name)


def judge(cond, inp):
    from rnj_wzzt_dr.experiments.hourly_probing_20261118 import experiments as hx
    s = hx.settings_for(JUDGE[cond])
    zbase = inp.network["base_kv"]**2/inp.network["base_mva"]
    return dict(settings=s, r_min=2*s.r_min_ohm/zbase, upper=s.upper_bound, offsets=s.meter_offsets,
                whiten=s.root_whitening, pq=s.pq_sigma if s.identifiable_params else None)


def run(job):
    which, cond, name = job
    from rnj_wzzt_dr.experiments.hourly_probing_20261118 import model
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124.compare_builders import line_attributes
    from rnj_wzzt_dr.identification import algorithm, combo, revised as rv, trees
    try:
        started = time.perf_counter()
        inp = load(cond, name); n = inp.n
        tr, va = p2.series(inp)
        internal = lambda f: {tuple(sorted(C)) for C in f if 1 < len(C) < n}
        truth_lines = sorted(internal(inp.truth_key), key=lambda s: (len(s), s))
        truth = set(truth_lines)
        # whole-map evidence of every truth line (Phase 3 ceiling routine, the input's own judge settings)
        j = judge(cond, inp)
        fam_t = model.canonical_family(inp.truth_key, n)
        fits = p2.study_J([fam_t]+[tuple(C for C in fam_t if C != seg) for seg in truth_lines], inp, j)
        N = inp.hist["Y"].size
        extra = n if j["offsets"] else 0
        sigma2 = fits[0].sse/(N-fits[0].c.p-extra)
        J = [f.sse/(2*sigma2)+.5*(f.c.p+extra)*math.log(N) for f in fits]
        resistance, med = line_attributes(inp)
        lines = [dict(segment=list(C), f=len(C)/n, rho=resistance.get(C, float("nan"))/med, dJ=float(J[k+1]-J[0]))
                 for k, C in enumerate(truth_lines)]
        def score(family, seconds, **extra_fields):
            got = internal(family)
            return dict(found="".join("1" if C in got else "0" for C in truth_lines), out=len(got),
                        out_true=len(got & truth), seconds=seconds, **extra_fields)
        methods = {}
        # WZZT-C
        t0 = time.perf_counter()
        oc = combo.fit(tr, va, **WZZT_C)
        methods["wzzt_c"] = score(oc["selected"], time.perf_counter()-t0, certified=oc["record"]["certified"],
                                  k=oc["record"]["k"], criterion=oc["record"]["criterion"])
        # WZZT-R (frozen, guidance solver per deviation D4)
        t0 = time.perf_counter()
        cfg = rv.RevisedConfig()
        orr = rv.fit(tr, va, cfg)
        S = combo.setup(tr, va)
        crit = lambda f: rv.exact_fit(S["design"], f, 2.)["objective"]+S["alpha"]*rv.internal_count(f, n)
        methods["wzzt_r"] = score(orr["selected"], time.perf_counter()-t0, certified=orr["e"].get("certified"),
                                  criterion=crit(orr["selected"]), qp_solver=cfg.qp_solver)
        methods["wzzt_c"]["criterion_same_design"] = crit(oc["selected"])
        # the original WZZT (demeaned), synthetic only (its cost grows steeply with the number of meters)
        if which == "synthetic":
            t0 = time.perf_counter()
            otr, ova = capture.observations(inp, True)
            common = algorithm.prepare_common(otr, capture.CONFIG)
            res = algorithm.fit(otr, ova, "111", config=capture.CONFIG, common=common)
            methods["o_dm"] = score(rv.canonical(res["selected"]["point"]["supports"]), time.perf_counter()-t0)
        # distance builders on WZZT-R's P-only matrix, every tolerance (fixed settings chosen in the analysis)
        t0 = time.perf_counter()
        shared, depth, _ = trees.geometry(S["R"], S["R"], 1.)
        for label, build in (("rnj", trees.rnj), ("nj", trees.nj), ("rg", trees.rg)):
            for tol in TOLERANCES:
                methods[f"{label}|{tol}"] = score(build(shared, depth, tol), None)
        build_seconds = time.perf_counter()-t0
        return dict(set=which, cond=cond, name=name, n=n, lines=lines, methods=methods, build_seconds=build_seconds,
                    seconds=time.perf_counter()-started)
    except Exception as exc:
        return dict(set=which, cond=cond, name=name, error=repr(exc), trace=traceback.format_exc()[-2000:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=("synthetic", "large"))
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    path = OUTC/f"{args.set}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                done.add((r["cond"], r["name"]))
    todo = [j for j in jobs(args.set) if (j[1], j[2]) not in done]
    print(f"{args.set}: {len(todo)} to run ({len(done)} done)", flush=True)
    with open(path, "a") as fh, get_context("spawn").Pool(args.workers, maxtasksperchild=4) as pool:
        for k, rec in enumerate(pool.imap_unordered(run, todo), 1):
            fh.write(json.dumps(rec)+"\n"); fh.flush()
            print(f"[{k}/{len(todo)}] {rec['name']} {rec.get('error', '')} {rec.get('seconds', 0):.0f}s", flush=True)


if __name__ == "__main__":
    main()
