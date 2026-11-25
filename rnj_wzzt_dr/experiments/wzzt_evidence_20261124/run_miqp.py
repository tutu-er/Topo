"""Experiment M runner (PROTOCOL_MIQP.md): where WZZT's exact search is critical.

Stages (outputs in outputs/wzzt_evidence_20261124/miqp/<stage>.jsonl, resumable):
- fit:        WZZT-R (frozen defaults) and WZZT-R with W grown by MIQP on the new inputs (transformer networks,
              large feeders, 16-day histories); the development inputs reuse the Phase 2 ladder.
- anatomy:    W along WZZT-R's path with the unknown-support MIQP solved at every step on the same family (M1).
- pairs:      W2, the joint two-support MIQP, and the single-support MIQP from each selected family (M2), for the
              slot terms c = gamma (R), c = 0 (R_c0) and c = 1 (R_src) on the development inputs.
- resolution: node-restricted, forced single-support MIQP per node of the selected family: the certified largest
              evidence a split of the node can add and the implied history multiple (M3).
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

from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import capture
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2
from rnj_wzzt_dr.identification import miqp_ext as mx
from rnj_wzzt_dr.identification import revised as rv
from rnj_wzzt_dr.identification.trees import basis, canonical, children

OUTM = capture.OUT/"miqp"
NEW_SETS = ("eulv_large", "eulv_16d")
SOURCES = {"R": "gls", "R_c0": "none", "R_src": "free"}
LIMITS = dict(step=120., step_large=300., pair=300., node=120., anatomy_total=5400.)


# ------------------------------------------------------------------ inputs
def load(cond, name):
    if cond in NEW_SETS:
        return bank.load(OUTM/"inputs"/cond/f"{name}.pkl")
    return capture.load_input(cond, name)


def new_inputs():
    out = []
    for s in NEW_SETS:
        out += [(s, p.stem) for p in sorted((OUTM/"inputs"/s).glob("*.pkl"))]
    return out


def internal(family, n):
    return {tuple(sorted(C)) for C in family if 1 < len(C) < n}


def truths(inp):
    t = internal(inp.truth_key, inp.n)
    x = internal(inp.meta.get("exact_truth_key", inp.truth_key), inp.n)
    return t, x


def _json(v):
    return json.loads(json.dumps(v, default=p2._json))


def _rows(stage):
    path = OUTM/f"{stage}.jsonl"
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                out[(r["cond"], r["name"], r.get("variant"))] = r
    return out


_SEL = {}


def selected_family(cond, name, variant):
    """WZZT-R's selected family: Phase 2 ladder (development inputs) or this experiment's fit stage."""
    if not _SEL:
        for line in (p2.OUT/"ladder.jsonl").read_text().splitlines():
            r = json.loads(line)
            if "error" not in r and r["variant"] in SOURCES:
                _SEL[(r["cond"], r["name"], r["variant"])] = r["selected"]
        for (c, nm, v), r in _rows("fit").items():
            _SEL[(c, nm, v)] = r["selected"]
    sel = _SEL[(cond, name, variant)]
    inp_n = load(cond, name).n
    return canonical([*map(tuple, sel), *basis(inp_n)])


# ------------------------------------------------------------------ stages
def exp_fit(job):
    cond, name, variant = job
    inp = load(cond, name)
    tr, va = p2.series(inp)
    cfg = rv.RevisedConfig() if variant == "R" else rv.RevisedConfig(w="miqp")
    started = time.perf_counter()
    out = rv.fit(tr, va, cfg)
    wall = time.perf_counter()-started
    t, x = truths(inp)
    sel = internal(out["selected"], inp.n)
    fams = [out["selected"], *out["path"], *out["alternatives"]]
    ref = internal(out["guidance"]["reference"], inp.n)
    pool = internal(out["guidance"]["pool"], inp.n)
    guidance = dict(reference_hit=len(ref & t), reference_size=len(ref), pool_hit=len(pool & t), pool_size=len(pool),
                    T=int(tr.p.shape[0]+va.p.shape[0]))
    guidance["qp_solver"] = cfg.qp_solver
    return _json(dict(cond=cond, name=name, variant=variant, n=inp.n, wall=wall, selected=sorted(sel), guidance=guidance,
                      metrics=dict(exact=sel == t, hit=len(sel & t), size=len(sel), truth=len(t),
                                   exact_hit=len(sel & x), exact_truth=len(x)),
                      truth_in_output=any(internal(f, inp.n) == t for f in fams),
                      w_events=out["w_events"], e={k: v for k, v in out["e"].items() if k != "portfolio"},
                      score=out["score"], timing=out["timing"]))


def exp_anatomy(job):
    """WZZT-R's W (generator mode) with the exact single-support MIQP solved at every step on the same family."""
    cond, name = job
    inp = load(cond, name)
    tr, va = p2.series(inp)
    S = mx.setup(tr, va)
    design, guide, alpha, n = S["design"], S["guide"], S["alpha"], S["n"]
    cfg = S["config"]
    t, _ = truths(inp)
    step_limit = LIMITS["step_large"] if n > 24 else LIMITS["step"]
    crit = lambda fam, obj: obj+alpha*rv.internal_count(fam, n)
    current = canonical(basis(n))
    cur_obj = rv.exact_fit(design, current, cfg.bound)["objective"]
    steps, started = [], time.perf_counter()
    while time.perf_counter()-started < LIMITS["anatomy_total"]:
        theta = rv.exact_fit(design, current, cfg.bound)["theta"]
        E = design.residual(current, theta)
        cands = rv.gen_pool(current, n, guide["pool"])
        sources = {C: "pool" for C in cands}
        valid = rv._valid(current, n)
        for gname, gen in (("pairs", lambda: rv.gen_sibling_pairs(current, n)),
                           ("residual_rnj", lambda: rv.gen_residual_rnj(current, n, design, E)),
                           ("clusters", lambda: rv.gen_residual_clusters(current, n, E)),
                           ("screen", lambda: rv.gen_screen(current, n, design, E))):
            for C in gen():
                if valid(C):
                    sources.setdefault(C, gname); cands.add(C)
        widths = {tuple(v): len(b) for v, b in children(current, n).items()}
        rec = dict(step=len(steps), candidates=len(cands), max_width=max(widths.values()),
                   screened=all(w <= 20 for w in widths.values() if w >= 3))
        base_score = crit(current, cur_obj)
        g_best, g_score = None, base_score
        if cands:
            stepper = rv.GramFitter(design, [*current, *cands], current, cfg.bound)
            for C in cands:
                fam = canonical([*current, C])
                s = crit(fam, stepper.fit(fam)[0])
                if s < g_score-cfg.gain_tol:
                    g_best, g_score = C, s
        rec["generator"] = dict(support=g_best, source=sources.get(g_best), gain=base_score-g_score,
                                true=g_best in t if g_best else None, size=len(g_best) if g_best else None)
        # the exact step, decomposed over the nodes (block-level, time-free MIQPs), with the generators' best score
        # as cutoff: a node that cannot beat it is closed by a proof (status 6)
        cut = (g_score if g_best is not None else base_score)+1e-9
        cert = mx.certify_by_nodes(design, current, alpha, cfg.bound, step_limit, cutoff=cut)
        m = dict(status=2 if cert["proved"] else 9, gap=None, bound=None, seconds=cert["seconds"],
                 node_statuses=[r.get("status") for r in cert["nodes"]], node_widths=[r["width"] for r in cert["nodes"]])
        m_best, m_score = None, base_score
        if cert["best"]:
            C = cert["best"]["supports"][0]
            fam = canonical([*current, C])
            m_score = crit(fam, rv.exact_fit(design, fam, cfg.bound)["objective"])
            if m_score < base_score-cfg.gain_tol:
                m_best = C
        parent = min((v for v in widths if m_best and set(m_best) < set(v)), key=len, default=None)
        rec["miqp"] = dict(support=m_best, gain=base_score-m_score, true=m_best in t if m_best else None,
                           size=len(m_best) if m_best else None, status=m["status"], gap=m["gap"],
                           seconds=m["seconds"], node_width=widths.get(parent),
                           node_statuses=m["node_statuses"], node_widths=m["node_widths"], form="nodes-timefree")
        rec["critical"] = bool(m_best is not None and m_score < g_score-cfg.gain_tol)
        rec["delta_J"] = .5*S["N"]*(g_score-min(m_score, g_score))
        steps.append(rec)
        step = g_best if g_best is not None else m_best      # WZZT-R's path: generators first, MIQP if they stop
        if step is None:
            break
        nxt = canonical([*current, step])
        fitb = rv.exact_fit(design, nxt, cfg.bound)
        k = len(nxt)
        zero = [C for i, C in enumerate(nxt) if 1 < len(C) < n and max(fitb["theta"][i], fitb["theta"][k+i]) <= cfg.zero_tol]
        if zero:
            cleaned = canonical(C for C in nxt if C not in zero)
            if crit(cleaned, rv.exact_fit(design, cleaned, cfg.bound)["objective"]) <= crit(nxt, fitb["objective"])+cfg.gain_tol:
                nxt = cleaned
        if nxt == current:
            break
        current = nxt
        cur_obj = rv.exact_fit(design, current, cfg.bound)["objective"]
    final = internal(current, n)
    return _json(dict(cond=cond, name=name, n=n, c=S["c"], alpha=alpha, N=S["N"], steps=steps,
                      final=sorted(final), final_hit=len(final & t), final_size=len(final), truth=len(t),
                      seconds=time.perf_counter()-started))


def exp_pairs(job):
    cond, name, variant = job
    inp = load(cond, name)
    tr, va = p2.series(inp)
    S = mx.setup(tr, va, rv.RevisedConfig(source=SOURCES[variant]))
    design, alpha, n = S["design"], S["alpha"], S["n"]
    fam = selected_family(cond, name, variant)
    t, _ = truths(inp)
    one = mx.extend_k_miqp(design, fam, alpha, 2., LIMITS["pair"], k=1)
    two = mx.extend_k_miqp(design, fam, alpha, 2., LIMITS["pair"], k=2)
    tol = 1e-9
    base = one["family_objective"]
    rec = dict(cond=cond, name=name, variant=variant, n=n, c=S["c"], N=S["N"], alpha=alpha,
               family_size=len(internal(fam, n)), base=base)
    for key, r in (("single", one), ("pair", two)):
        rec[key] = dict(supports=r["supports"], objective=r["objective"], bound=r["bound"], status=r["status"],
                        gap=r["gap"], seconds=r["seconds"], true=[C in t for C in r["supports"]],
                        gain_J=.5*S["N"]*(base-r["objective"]) if r["objective"] is not None else None)
    rec["single_certified"] = one["status"] == 2 and not one["supports"]
    rec["pair_improves"] = bool(two["supports"] and two["objective"] is not None and two["objective"] < base-tol)
    rec["interaction_critical"] = bool(rec["single_certified"] and len(two["supports"]) == 2 and rec["pair_improves"])
    if rec["pair_improves"] and two["supports"]:
        # super-additivity check: each support alone against the pair (exact refits)
        f1 = canonical([*fam, two["supports"][0]])
        alone = []
        for C in two["supports"]:
            f = canonical([*fam, C])
            alone.append(.5*S["N"]*(base-(rv.exact_fit(design, f, 2.)["objective"]+alpha*rv.internal_count(f, n))))
        rec["pair_alone_gain_J"] = alone
    return _json(rec)


def exp_resolution(job):
    cond, name, variant = job
    inp = load(cond, name)
    tr, va = p2.series(inp)
    S = mx.setup(tr, va, rv.RevisedConfig(source=SOURCES[variant]))
    design, alpha, n, N = S["design"], S["alpha"], S["n"], S["N"]
    fam = selected_family(cond, name, variant)
    t, x = truths(inp)
    price = .5*N*alpha
    nodes = []
    for node, width in mx.node_splits(fam, n):
        r = mx.extend_tf_miqp(design, fam, alpha, 2., LIMITS["node"], k=1, within=node, force=True, blocks=True)
        gain = .5*N*(r["family_objective"]-r["objective"]+alpha) if r["objective"] is not None else None
        gmax = .5*N*(r["family_objective"]-r["bound"]+alpha) if r["bound"] is not None else None
        missed = [C for C in t if set(C) < set(node) and C not in internal(fam, n)
                  and not any(set(C) <= set(b) for b in children(fam, n)[node])]
        nodes.append(dict(node=node, width=width, best=r["supports"], best_true=[C in t for C in r["supports"]],
                          gain_J=gain, max_gain_J=gmax, status=r["status"], gap=r["gap"], seconds=r["seconds"],
                          multiple=(price/gmax if gmax and gmax > 0 else math.inf) if gmax is not None else None,
                          missed_true=missed, form="blocks-timefree", blocks=r.get("blocks")))
    return _json(dict(cond=cond, name=name, variant=variant, n=n, N=N, price_J=price, c=S["c"], nodes=nodes,
                      family=sorted(internal(fam, n))))


def exp_history(job):
    """M5: the same real feeder with a shortened history (the last T hourly slots of the 8-day input): RNJ's
    reference tree (built from the best-fitting map's entries) against WZZT-R's search on the map."""
    cond, name, T = job
    inp = load(cond, name)
    tr, va = p2.series(inp)
    data = tr.join(va)
    sl = slice(len(data.p)-int(T), len(data.p))
    short = rv.Series(data.p[sl], data.q[sl], data.y[sl], None if data.v0sq is None else data.v0sq[sl])
    started = time.perf_counter()
    out = rv.fit(short, None, rv.RevisedConfig())
    wall = time.perf_counter()-started
    t, x = truths(inp)
    n = inp.n
    sel = internal(out["selected"], n)
    ref = internal(out["guidance"]["reference"], n)
    pool = internal(out["guidance"]["pool"], n)
    m = lambda S: dict(hit=len(S & t), size=len(S), truth=len(t), exact=S == t, exact_hit=len(S & x), exact_truth=len(x))
    return _json(dict(cond=cond, name=name, variant=f"T{int(T)}", T=int(T), n=n, wall=wall, R=m(sel), reference=m(ref),
                      pool_hit=len(pool & t), pool_size=len(pool), confirmed=bool(out["e"].get("confirmed"))))


STAGES = dict(fit=exp_fit, anatomy=exp_anatomy, pairs=exp_pairs, resolution=exp_resolution, history=exp_history)


def _safe(item):
    stage, job = item
    started = time.perf_counter()
    try:
        out = STAGES[stage](job)
    except Exception as exc:
        out = dict(job=list(job), error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc())
    out["job_seconds"] = time.perf_counter()-started
    return out


ANATOMY_MAX_N = 34     # the exact single-support MIQP found no solution and no bound in 300 s at 44 meters


def jobs(stage, sets):
    dev = list(capture.select_inputs()) if "dev" in sets else []
    new = [j for j in new_inputs() if j[0] in sets]
    if stage == "fit":
        return [(c, nm, v) for c, nm in new for v in (("R", "R_wmiqp") if c == "eulv_large" else ("R",))]
    if stage == "anatomy":
        return dev+[j for j in new if j[0] == "eulv_large" and load(*j).n <= ANATOMY_MAX_N]
    if stage == "pairs":
        return [(c, nm, v) for c, nm in dev for v in SOURCES]
    if stage == "resolution":
        return [(c, nm, "R") for c, nm in dev+new]
    if stage == "history":
        return [(c, nm, T) for c, nm in dev if c == "eulv" for T in (48, 96, 192)]
    raise ValueError(stage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--sets", default="dev,eulv_large,eulv_16d")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    sets = args.sets.split(",")
    todo = jobs(args.stage, sets)
    path = OUTM/f"{args.stage}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                if args.stage == "history":
                    done.add((r["cond"], r["name"], r["T"]))
                else:
                    done.add((r["cond"], r["name"], r.get("variant")) if todo and len(todo[0]) == 3
                             else (r["cond"], r["name"]))
    todo = [j for j in todo if tuple(j) not in done]
    # largest inputs first
    size = {}
    for j in todo:
        size[j] = load(j[0], j[1]).n
    todo.sort(key=lambda j: -size[j])
    if args.limit:
        todo = todo[:args.limit]
    print(f"{args.stage}: {len(todo)} jobs", flush=True)
    OUTM.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with Pool(args.workers, maxtasksperchild=2) as pool, open(path, "a") as fh:
        for k, out in enumerate(pool.imap_unordered(_safe, [(args.stage, j) for j in todo]), 1):
            fh.write(json.dumps(out)+"\n"); fh.flush()
            print(f"[{k}/{len(todo)}] {time.perf_counter()-started:7.0f}s {out.get('cond')} {out.get('name')} "
                  f"{out.get('variant', '')} {out.get('error', '')[:100]} {out['job_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
