"""Phase 2 runner: the revised WZZT (WZZT-R) on the Phase 1 inputs (PROTOCOL_PHASE2.md).

Stages (each resumable; outputs in outputs/wzzt_evidence_20261124/phase2/):
  ceiling  data-supported truth segments under the study BIC (full history), once per input
  ladder   WZZT-R and its ablations (experiment C)
  bank     the probing study's stage 0 downstream of an identifier: BIC closure (optionally modulo voltage
           equivalence, R3), pruning and strata; for the original identifier and for WZZT-R
  root     root-reading scenarios x source-term variants (experiment E)
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import hashlib
import json
import math
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from rnj_wzzt_dr.identification import revised as rv
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, model, scores
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import experiments as hx
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import capture

OUT = capture.OUT/"phase2"

VARIANTS = {
    "R": dict(),
    "R_pq": dict(guidance="pq"),
    "R_alpha": dict(score="alpha"),
    "R_plain": dict(source="none", offsets=False),
    "R_c0": dict(source="none"),
    "R_src": dict(source="free"),
    "R_wmiqp": dict(w="miqp"),
    "R_wpool": dict(w="pool"),
    "R_emiqp": dict(e="miqp"),
    "R_train": dict(history="train"),
    # experiment E (root scenarios): the search result without the certificate
    "R_h": dict(e="heuristic"),
    "R_src_h": dict(source="free", e="heuristic"),
    "R_c0_h": dict(source="none", e="heuristic"),
    # deviation D3: the GLS strength estimated from the data, so the whitening follows the root reading's quality
    "R_gest_h": dict(source="estimated", e="heuristic"),
}
SCENARIOS = ("exact", "noisy01", "noisy05", "biased", "absent")


def config(variant, **extra):
    return rv.RevisedConfig(**{**VARIANTS[variant], **extra})


def internal(family, n):
    return {tuple(sorted(C)) for C in family if 1 < len(C) < n}


def _rng(*keys):
    digest = hashlib.sha256("|".join(map(str, keys)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def series(inp, scenario="exact"):
    """Training and validation Series; root scenarios rewrite y = V0'^2 - V^2 from the stored readings."""
    h = inp.hist
    P, Q, Y, V0 = (np.asarray(h[k], float) for k in ("P", "Q", "Y", "V0sq"))
    Vsq = V0[:, None]-Y
    if scenario == "exact":
        V0n = V0
    elif scenario in ("noisy01", "noisy05"):
        sd = .001 if scenario == "noisy01" else .005
        V0n = V0*(1.+sd*_rng(inp.name, scenario).normal(size=V0.shape))**2
    elif scenario == "biased":
        V0n = V0*(1.005)**2
    elif scenario == "absent":
        V0n = np.ones_like(V0)
    else:
        raise ValueError(scenario)
    Yn = V0n[:, None]-Vsq
    t0, t1 = inp.meta["history"]
    tr, va = slice(0, t0), slice(t0, t0+t1)
    return (rv.Series(P[tr], Q[tr], Yn[tr], V0n[tr]), rv.Series(P[va], Q[va], Yn[va], V0n[va]))


def family_metrics(family, truth, n, supported=None):
    got = internal(family, n)
    out = dict(exact=got == truth, hit=len(got & truth), size=len(got), truth=len(truth))
    if supported is not None:
        out["supported_hit"] = len(got & supported)
    return out


def _json(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (set, frozenset, tuple)):
        return list(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


# ------------------------------------------------------------------ study fits (the downstream judge)
def judge(cond, inp):
    s = hx.settings_for(cond)
    zbase = inp.network["base_kv"]**2/inp.network["base_mva"]
    return dict(settings=s, r_min=2*s.r_min_ohm/zbase, upper=s.upper_bound, offsets=s.meter_offsets,
                whiten=s.root_whitening, pq=s.pq_sigma if s.identifiable_params else None)


def study_J(families, inp, j):
    fits = [bank.fit_one(f, inp.n, inp.hist, j["r_min"], j["upper"], 3, True, j["offsets"], j["whiten"], j["pq"])
            for f in families]
    return fits


def exp_ceiling(job):
    cond, name = job
    inp = capture.load_input(cond, name)
    n, j = inp.n, judge(cond, inp)
    truth = model.canonical_family(inp.truth_key, n)
    tint = sorted(internal(truth, n))
    fams = [truth]+[tuple(C for C in truth if C != seg) for seg in tint]
    fits = study_J(fams, inp, j)
    N = inp.hist["Y"].size
    extra = n if j["offsets"] else 0
    sigma2 = fits[0].sse/(N-fits[0].c.p-extra)
    J = [f.sse/(2*sigma2)+.5*(f.c.p+extra)*math.log(N) for f in fits]
    rows = [dict(segment=list(seg), size=len(seg), dJ=float(J[k+1]-J[0])) for k, seg in enumerate(tint)]
    return dict(cond=cond, name=name, n=n, truth=len(tint), supported=sum(r["dJ"] > 0 for r in rows),
                segments=rows)


# ------------------------------------------------------------------ ladder (experiment C)
def run_variant(cond, name, variant, scenario="exact", supported=None, **extra):
    inp = capture.load_input(cond, name)
    tr, va = series(inp, scenario)
    cfg = config(variant, **extra)
    if scenario == "absent" and cfg.source == "gls":
        pass   # the GLS strength then comes from the nominal root (no reading): documented weakness
    started = time.perf_counter()
    out = rv.fit(tr, va, cfg)
    wall = time.perf_counter()-started
    n = inp.n
    truth = internal(inp.truth_key, n)
    output = set()
    for fam in [*out["path"], out["selected"], *out["alternatives"]]:
        output |= internal(fam, n)
    rec = dict(cond=cond, name=name, variant=variant, scenario=scenario, n=n, wall=wall,
               selected=sorted(internal(out["selected"], n)), output=sorted(output),
               metrics=family_metrics(out["selected"], truth, n, supported),
               output_metrics=dict(hit=len(output & truth), size=len(output), truth=len(truth),
                                   supported_hit=len(output & supported) if supported is not None else None),
               truth_in_output=any(internal(f, n) == truth
                                   for f in [*out["path"], out["selected"], *out["alternatives"]]),
               families=[sorted(internal(f, n)) for f in [out["selected"], *out["path"], *out["alternatives"]]],
               score={k: v for k, v in out["score"].items()},
               w_steps=len(out["w_events"]),
               w_miqp=[e.get("miqp") for e in out["w_events"] if e.get("miqp")],
               w_sources=[e.get("generator_best", {}).get("source") for e in out["w_events"]],
               w_accepted_from=[e.get("accepted_from") for e in out["w_events"]],
               e={k: v for k, v in out["e"].items()}, timing=out["timing"])
    return rec


def load_ceiling():
    path = OUT/"ceiling.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "error" not in r:
            out[(r["cond"], r["name"])] = {tuple(s["segment"]) for s in r["segments"] if s["dJ"] > 0}
    return out


_CEILING = None


def exp_ladder(job):
    global _CEILING
    if _CEILING is None:
        _CEILING = load_ceiling()
    cond, name, variant = job
    return run_variant(cond, name, variant, supported=_CEILING.get((cond, name)))


def exp_root(job):
    global _CEILING
    if _CEILING is None:
        _CEILING = load_ceiling()
    cond, name, variant, scenario = job
    return run_variant(cond, name, variant, scenario=scenario, supported=_CEILING.get((cond, name)))


# ------------------------------------------------------------------ downstream bank (stage 0 of the probing study)
def closure_eq(bank_fams, n, hist, r_min, upper, eps, margin=6., max_seeds=12, max_rounds=4, max_size=1500,
               whiten=False, offsets=True, steps=2, pq_sigma=None):
    """bank.closure, expanding one representative per voltage-equivalence class (R3).

    Two candidates are equivalent when their fitted voltages on the history (offsets included) differ by at
    most eps everywhere. A seed equivalent to an already expanded candidate is marked expanded without
    expansion. Everything else is bank.closure."""
    fams = list(dict.fromkeys(bank_fams))
    keys = {model.nonsingleton(f): k for k, f in enumerate(fams)}
    fit = lambda f: bank.fit_one(f, n, hist, r_min, upper, steps, True, offsets, whiten, pq_sigma)
    fits = [fit(f) for f in fams]
    V0 = np.asarray(hist["V0sq"]).reshape(-1, 1)
    volt = {}

    def voltage(k):
        if k not in volt:
            f = fits[k]
            y = f.c.predict(f.theta, hist["P"], hist["Q"], hist["V0sq"])+f.offset
            volt[k] = np.sqrt(np.maximum(V0-y, .25))
        return volt[k]
    N = hist["Y"].size
    extra = n if offsets else 0
    expanded, represented, rounds, capped = [], set(), 0, None
    while True:
        J, sigma2 = scores.bic([f.sse for f in fits], [f.c.p+extra for f in fits], N)
        E = J-J.min()
        seeds = []
        for k in np.argsort(E):
            if E[k] > margin or len(seeds) >= max_seeds:
                break
            if k in represented:
                continue
            twin = next((e for e in expanded if float(np.max(np.abs(voltage(k)-voltage(e)))) <= eps), None)
            if twin is not None:
                represented.add(k); continue
            seeds.append(k)
        if not seeds:
            break
        if rounds >= max_rounds:
            capped = "rounds"; break
        if len(fams) >= max_size:
            capped = "size"; break
        rounds += 1
        for k in seeds:
            expanded.append(k); represented.add(k)
            for nb in bank.neighbours(fams[k], n):
                f = model.canonical_family(nb, n)
                key = model.nonsingleton(f)
                if key not in keys:
                    keys[key] = len(fams); fams.append(f); fits.append(fit(f))
    unexpanded = sum(1 for k in np.flatnonzero(E <= margin) if k not in represented)
    return fams, fits, J, sigma2, dict(rounds=rounds, capped=capped, size=len(fams),
                                       unexpanded_within_margin=int(unexpanded), expanded=len(expanded),
                                       represented=len(represented))


def downstream(cond, inp, families, r3, prune_E=30., max_bank=150):
    j = judge(cond, inp)
    n = inp.n
    seeds = [model.canonical_family(f, n) for f in families]
    started = time.perf_counter()
    if r3:
        D, cfits, cJ, _, crec = closure_eq(seeds, n, inp.hist, j["r_min"], j["upper"], j["settings"].equivalence_pu,
                                           offsets=j["offsets"], whiten=j["whiten"], pq_sigma=j["pq"])
    else:
        D, cfits, cJ, _, crec = bank.closure(seeds, n, inp.hist, j["r_min"], j["upper"], offsets=j["offsets"],
                                             whiten=j["whiten"], pq_sigma=j["pq"])
    seconds = time.perf_counter()-started
    order = np.argsort(cJ-cJ.min())
    keep = [k for k in order if cJ[k]-cJ.min() <= prune_E][:max_bank]
    fams = [D[k] for k in keep]
    fits, J, _ = bank.passive_fit(fams, n, inp.hist, j["r_min"], j["upper"], offsets=j["offsets"],
                                  whiten=j["whiten"], pq_sigma=j["pq"])
    new = copy.copy(inp)
    new.families, new.theta0 = fams, [f.theta.copy() for f in fits]
    new.truth_in_bank = any(model.nonsingleton(f) == inp.truth_key for f in fams)
    passive = bank.classify(new, j["settings"], J)      # includes leader_correct (BIC leader = truth)
    return dict(closure=crec, closure_seconds=seconds, bank=len(fams), truth_in_bank=new.truth_in_bank,
                truth_in_closure=any(model.nonsingleton(f) == inp.truth_key for f in D), **passive)


def _ladder_rows(variant):
    path = OUT/"ladder.jsonl"
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("variant") == variant and "error" not in r:
                rows[(r["cond"], r["name"])] = r
    return rows


_LADDER = {}


def exp_bank(job):
    cond, name, source, r3 = job
    inp = capture.load_input(cond, name)
    n = inp.n
    if source == "orig":
        # the original identifier (111, levels and demeaned runs, Phase 1 capture): every path family and the
        # selected family of both runs (the contracted copies are not stored; they are drop-neighbours)
        fams = []
        for mode in ("lv", "dm"):
            rec = capture.load_instance(capture.instance_path(cond, name, mode))
            fams += [tuple(map(tuple, f)) for f in rec["path"]]
            fams.append(tuple(map(tuple, rec["selected"])))
    else:
        if source not in _LADDER:
            _LADDER[source] = _ladder_rows(source)
        row = _LADDER[source].get((cond, name))
        if row is None:
            raise RuntimeError(f"no ladder row for {source}")
        fams = [tuple(map(tuple, f)) for f in row["families"]]
    full = [model.canonical_family([*f, *((i,) for i in range(n)), tuple(range(n))], n) for f in fams]
    out = downstream(cond, inp, full, r3)
    return dict(cond=cond, name=name, source=source, r3=r3, seeds=len(set(full)), **out)


# ------------------------------------------------------------------ jobs and main
STAGES = dict(ceiling=lambda job: exp_ceiling(job), ladder=lambda job: exp_ladder(job),
              bank=lambda job: exp_bank(job), root=lambda job: exp_root(job))


def _safe(item):
    stage, job = item
    started = time.perf_counter()
    try:
        out = STAGES[stage](job)
    except Exception as exc:
        out = dict(job=list(job), error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc())
    out["job_seconds"] = time.perf_counter()-started
    return out


def _done(path, key):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                done.add(key(r))
    return done


def inputs():
    jobs = capture.select_inputs()
    size = {}
    for cond, name in jobs:
        size[(cond, name)] = capture.load_input(cond, name).n
    return sorted(jobs, key=lambda j: -size[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ceiling", "ladder", "bank", "root"])
    ap.add_argument("--variants", default="R")
    ap.add_argument("--sources", default="orig,R")
    ap.add_argument("--r3", default="0,1")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--conds", default="eulv,lock_short01,lock_noisy")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    conds = set(args.conds.split(","))
    base = [j for j in inputs() if j[0] in conds]
    if args.limit:
        base = base[:args.limit]
    if args.stage == "ceiling":
        path, fn = OUT/"ceiling.jsonl", exp_ceiling
        jobs = base
        done = _done(path, lambda r: (r["cond"], r["name"]))
    elif args.stage == "ladder":
        path, fn = OUT/"ladder.jsonl", exp_ladder
        jobs = [(c, nm, v) for v in args.variants.split(",") for c, nm in base]
        done = _done(path, lambda r: (r["cond"], r["name"], r["variant"]))
    elif args.stage == "bank":
        path, fn = OUT/"bank.jsonl", exp_bank
        jobs = [(c, nm, s, bool(int(r3))) for s in args.sources.split(",") for r3 in args.r3.split(",")
                for c, nm in base]
        done = _done(path, lambda r: (r["cond"], r["name"], r["source"], r["r3"]))
    else:
        path, fn = OUT/"root.jsonl", exp_root
        jobs = [(c, nm, v, sc) for sc in args.scenarios.split(",") for v in args.variants.split(",")
                for c, nm in base]
        done = _done(path, lambda r: (r["cond"], r["name"], r["variant"], r["scenario"]))
    jobs = [j for j in jobs if tuple(j) not in done]
    print(f"{args.stage}: {len(jobs)} jobs", flush=True)
    started = time.perf_counter()
    with Pool(args.workers, maxtasksperchild=4) as pool, open(path, "a") as fh:
        for k, out in enumerate(pool.imap_unordered(_safe, [(args.stage, j) for j in jobs])):
            fh.write(json.dumps(out, default=_json)+"\n"); fh.flush()
            tag = out.get("error", "")[:120] if "error" in out else ""
            print(f"[{k+1}/{len(jobs)}] {time.perf_counter()-started:7.0f}s {out.get('job')} "
                  f"{out.get('cond')} {out.get('name')} {out.get('variant', out.get('source', ''))} "
                  f"{out.get('job_seconds', 0):.0f}s {tag}", flush=True)


if __name__ == "__main__":
    main()
