"""Phase 3 runner: confirmation of the frozen WZZT-R on inputs never used in WZZT development (PROTOCOL_PHASE3.md).

Inputs: `lock_short01` and `lock_noisy` seeds 73-96 (built by the hourly-probing study), and the real feeders on
start days 1-5 and 7-11 (build_phase3.py). Stages (resumable; outputs in outputs/wzzt_evidence_20261124/phase3/):
  ceiling  data-supported truth segments (study BIC, whole history)
  orig     the original 111 on levels and demeaned data (the study's stage-0 identifier), plus RNJ1
  R        the frozen WZZT-R, plus its R0 reference tree
  bank     the study's closure (pruning, strata) from each identifier's families
  judge    the study BIC of each selected family against the truth
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
from multiprocessing import Pool

import numpy as np

from rnj_wzzt_dr.identification import algorithm
from rnj_wzzt_dr.identification import revised as rv
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, model
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import experiments as hx
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import capture
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2

OUT3 = capture.OUT/"phase3"
SYN_SEEDS = range(73, 97)
FROZEN = dict()          # WZZT-R as frozen by D2 (RevisedConfig defaults); set from PROTOCOL_PHASE3.md


def settings_cond(cond):
    return "eulv" if cond == "eulv" else cond


def input_path(cond, name):
    if cond == "eulv":
        return OUT3/"inputs"/"eulv"/f"{name}.pkl"
    return hx.ROOT/cond/"inputs"/f"{name}.pkl"


def load(cond, name):
    return bank.load(input_path(cond, name))


def inputs():
    jobs = []
    for cond in ("lock_short01", "lock_noisy"):
        rows = [r for r in json.loads((hx.ROOT/cond/"inputs.json").read_text()) if "error" not in r]
        jobs += [(cond, r["name"]) for r in rows if r["seed"] in SYN_SEEDS]
    real = sorted(p.stem for p in (OUT3/"inputs"/"eulv").glob("*.pkl"))
    jobs += [("eulv", nm) for nm in real]
    size = {j: load(*j).n for j in jobs}
    return sorted(jobs, key=lambda j: -size[j])


def internal(family, n):
    return {tuple(sorted(C)) for C in family if 1 < len(C) < n}


def metrics(selected, inp, supported):
    truth = internal(inp.truth_key, inp.n)
    got = internal(selected, inp.n)
    return dict(exact=got == truth, hit=len(got & truth), size=len(got), truth=len(truth),
                supported_hit=len(got & supported), supported=len(supported))


_CEIL = None


def ceiling_of(cond, name):
    global _CEIL
    if _CEIL is None:
        _CEIL = {}
        path = OUT3/"ceiling.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                if "error" not in r:
                    _CEIL[(r["cond"], r["name"])] = {tuple(s["segment"]) for s in r["segments"] if s["dJ"] > 0}
    return _CEIL[(cond, name)]


def judge(cond, inp):
    s = hx.settings_for(settings_cond(cond))
    zbase = inp.network["base_kv"]**2/inp.network["base_mva"]
    return dict(settings=s, r_min=2*s.r_min_ohm/zbase, upper=s.upper_bound, offsets=s.meter_offsets,
                whiten=s.root_whitening, pq=s.pq_sigma if s.identifiable_params else None)


# ------------------------------------------------------------------ stages
def exp_ceiling(job):
    cond, name = job
    inp = load(cond, name)
    n, j = inp.n, judge(cond, inp)
    truth = model.canonical_family(inp.truth_key, n)
    tint = sorted(internal(truth, n))
    fams = [truth]+[tuple(C for C in truth if C != seg) for seg in tint]
    fits = p2.study_J(fams, inp, j)
    N = inp.hist["Y"].size
    extra = n if j["offsets"] else 0
    sigma2 = fits[0].sse/(N-fits[0].c.p-extra)
    J = [f.sse/(2*sigma2)+.5*(f.c.p+extra)*math.log(N) for f in fits]
    rows = [dict(segment=list(seg), size=len(seg), dJ=float(J[k+1]-J[0])) for k, seg in enumerate(tint)]
    return dict(cond=cond, name=name, n=n, truth=len(tint), supported=sum(r["dJ"] > 0 for r in rows), segments=rows)


def exp_orig(job):
    cond, name, mode = job
    inp = load(cond, name)
    tr, va = capture.observations(inp, mode == "dm")
    started = time.perf_counter()
    common = algorithm.prepare_common(tr, capture.CONFIG)
    res = algorithm.fit(tr, va, "111", config=capture.CONFIG, common=common)
    wall = time.perf_counter()-started
    n = inp.n
    path = [rv.canonical(p["supports"]) for p in res["path"]]
    contracted = [rv.canonical(c["point"]["supports"]) for c in res["frozen_candidates"]]
    uncontracted = [rv.canonical(p["supports"]) for p in res.get("uncontracted_output_path", [])]
    selected = rv.canonical(res["selected"]["point"]["supports"])
    output = set()
    for f in contracted+uncontracted:
        output |= internal(f, n)
    rnj1 = common["guidance"]["reference_family"]
    sup = ceiling_of(cond, name)
    return dict(cond=cond, name=name, variant="O_"+mode, n=n, wall=wall, selected=sorted(internal(selected, n)),
                metrics=metrics(selected, inp, sup), output_hit=len(output & internal(inp.truth_key, n)),
                families=[sorted(internal(f, n)) for f in dict.fromkeys([selected, *contracted, *uncontracted])],
                rnj1=sorted(internal(rnj1, n)), rnj1_metrics=metrics(rnj1, inp, sup))


def exp_R(job):
    cond, name = job
    inp = load(cond, name)
    tr, va = p2.series(inp)
    cfg = rv.RevisedConfig(**FROZEN)
    started = time.perf_counter()
    out = rv.fit(tr, va, cfg)
    wall = time.perf_counter()-started
    n = inp.n
    sup = ceiling_of(cond, name)
    output = set()
    fams = [out["selected"], *out["path"], *out["alternatives"]]
    for f in fams:
        output |= internal(f, n)
    ref = out["guidance"]["reference"]
    return dict(cond=cond, name=name, variant="R", n=n, wall=wall, selected=sorted(internal(out["selected"], n)),
                metrics=metrics(out["selected"], inp, sup), output_hit=len(output & internal(inp.truth_key, n)),
                truth_in_output=any(internal(f, n) == internal(inp.truth_key, n) for f in fams),
                families=[sorted(internal(f, n)) for f in fams], e=out["e"], score=out["score"],
                timing=out["timing"], r0_reference=sorted(internal(ref, n)), r0_metrics=metrics(ref, inp, sup))


def _rows(stage):
    path = OUT3/f"{stage}.jsonl"
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                out.setdefault((r["cond"], r["name"]), {})[r.get("variant")] = r
    return out


_FAMS = {}


def exp_bank(job):
    cond, name, source = job
    if not _FAMS:
        for stage in ("orig", "R"):
            for k, v in _rows(stage).items():
                _FAMS.setdefault(k, {}).update(v)
    inp = load(cond, name)
    n = inp.n
    rows = _FAMS[(cond, name)]
    fams = ([*rows["O_lv"]["families"], *rows["O_dm"]["families"]] if source == "orig" else rows["R"]["families"])
    full = [model.canonical_family([*map(tuple, f), *((i,) for i in range(n)), tuple(range(n))], n) for f in fams]
    j = judge(cond, inp)
    started = time.perf_counter()
    D, cfits, cJ, _, crec = bank.closure(full, n, inp.hist, j["r_min"], j["upper"], offsets=j["offsets"],
                                         whiten=j["whiten"], pq_sigma=j["pq"])
    seconds = time.perf_counter()-started
    order = np.argsort(cJ-cJ.min())
    keep = [k for k in order if cJ[k]-cJ.min() <= 30.][:150]
    fams = [D[k] for k in keep]
    fits, J, _ = bank.passive_fit(fams, n, inp.hist, j["r_min"], j["upper"], offsets=j["offsets"],
                                  whiten=j["whiten"], pq_sigma=j["pq"])
    import copy
    new = copy.copy(inp)
    new.families, new.theta0 = fams, [f.theta.copy() for f in fits]
    new.truth_in_bank = any(model.nonsingleton(f) == inp.truth_key for f in fams)
    passive = bank.classify(new, j["settings"], J)
    return dict(cond=cond, name=name, source=source, closure=crec, closure_seconds=seconds, bank=len(fams),
                truth_in_bank=new.truth_in_bank, **passive)


def exp_judge(job):
    cond, name = job
    if not _FAMS:
        for stage in ("orig", "R"):
            for k, v in _rows(stage).items():
                _FAMS.setdefault(k, {}).update(v)
    inp = load(cond, name)
    n, j = inp.n, judge(cond, inp)
    rows = _FAMS[(cond, name)]
    whole = tuple(range(n))
    fams = {"truth": model.canonical_family(inp.truth_key, n)}
    for v in ("O_lv", "O_dm", "R"):
        fams[v] = model.canonical_family([*map(tuple, rows[v]["selected"]), whole], n)
    keys = list(fams)
    variants = {k: [fams[k], model.canonical_family([C for C in fams[k] if C != whole], n),
                    model.canonical_family([*fams[k], whole], n)] for k in keys}
    flat = [(k, f) for k in keys for f in dict.fromkeys(variants[k])]
    fits = p2.study_J([f for _, f in flat], inp, j)
    N = inp.hist["Y"].size
    extra = n if j["offsets"] else 0
    sigma2 = min(f.sse/(N-f.c.p-extra) for f in fits)
    J = {}
    for (k, _), f in zip(flat, fits):
        J[k] = min(J.get(k, math.inf), f.sse/(2*sigma2)+.5*(f.c.p+extra)*math.log(N))
    inner = lambda f: frozenset(C for C in f if 1 < len(C) < n)
    return dict(cond=cond, name=name, n=n, dJ={k: J[k]-J["truth"] for k in keys if k != "truth"},
                same_as_truth={k: inner(fams[k]) == inner(fams["truth"]) for k in keys})


STAGES = dict(ceiling=lambda j: exp_ceiling(j), orig=lambda j: exp_orig(j), R=lambda j: exp_R(j),
              bank=lambda j: exp_bank(j), judge=lambda j: exp_judge(j))


def _safe(item):
    stage, job = item
    started = time.perf_counter()
    try:
        out = STAGES[stage](job)
    except Exception as exc:
        out = dict(job=list(job), error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc())
    out["job_seconds"] = time.perf_counter()-started
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    base = inputs()
    if args.stage in ("ceiling", "R", "judge"):
        jobs = base
        key = lambda r: (r["cond"], r["name"])
    elif args.stage == "orig":
        jobs = [(c, nm, m) for m in ("lv", "dm") for c, nm in base]
        key = lambda r: (r["cond"], r["name"], r["variant"][2:])
    else:
        jobs = [(c, nm, s) for s in ("orig", "R") for c, nm in base]
        key = lambda r: (r["cond"], r["name"], r["source"])
    path = OUT3/f"{args.stage}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                done.add(key(r))
    jobs = [j for j in jobs if tuple(j) not in done]
    print(f"{args.stage}: {len(jobs)} jobs", flush=True)
    started = time.perf_counter()
    with Pool(args.workers, maxtasksperchild=4) as pool, open(path, "a") as fh:
        for k, out in enumerate(pool.imap_unordered(_safe, [(args.stage, j) for j in jobs])):
            fh.write(json.dumps(out, default=p2._json)+"\n"); fh.flush()
            tag = out.get("error", "")[:120] if "error" in out else ""
            print(f"[{k+1}/{len(jobs)}] {time.perf_counter()-started:7.0f}s {out.get('cond')} {out.get('name')} "
                  f"{out.get('variant', out.get('source', ''))} {out.get('job_seconds', 0):.0f}s {tag}", flush=True)


if __name__ == "__main__":
    main()
