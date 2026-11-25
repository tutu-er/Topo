"""Phase 1 runner (PROTOCOL.md): capture, experiment A (+F), experiment B1, experiment B2.

    python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase1 capture|a|b1|b2 [--workers 14] [--limit N]

Outputs go to outputs/wzzt_evidence_20261124/ (not in Git): instances/*.pkl, a.jsonl, b1.jsonl, b2.jsonl.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
import traceback
import zlib
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from rnj_wzzt_dr.identification import algorithm
from rnj_wzzt_dr.identification.algorithm import SearchConfig
from rnj_wzzt_dr.identification.optimization import QPFitter, criterion
from rnj_wzzt_dr.identification.trees import basis, canonical

from . import capture as cap, certify, generators as gen, heuristics as heu
from .fastfit import ExtensionFitter, UniverseFitter, compatible_supports, count_compatible

OUT = cap.OUT


def _config(rec):
    return SearchConfig(**rec["config"])


def _internal(supports, n):
    return sorted({tuple(sorted(C)) for C in supports if 1 < len(C) < n})


def _truth_metrics(family, truth):
    fam, tru = set(map(tuple, family)), set(map(tuple, truth))
    return dict(hit=len(fam & tru), size=len(fam), truth=len(tru), exact=fam == tru)


def _rescore(fitter, config, keep, S):
    """The E objective (fit + alpha per internal support) and the closed criterion (after zero cleanup)."""
    fam = canonical([*keep, *S])
    point = fitter.fit(fam)
    objective = float(point["objective"])+config.alpha*sum(1 < len(C) < fitter.data.n for C in fam)
    closed, _ = algorithm._clean(fitter, point, config, protected=keep)
    return objective, float(criterion(closed, fitter.data.n, config.alpha)), _internal(closed["supports"], fitter.data.n)


# ---------------------------------------------------------------- experiment A (+F)
def exp_a(path):
    rec = cap.load_instance(path)
    out = dict(cond=rec["cond"], name=rec["name"], mode=rec["mode"], n=rec["n"], rows=len(rec["train"][0]))
    if not rec["E"]:
        return dict(out, skipped="no E stage")
    ev = rec["E"][0]["event"]
    native, raw, initial = ev["solver"], ev.get("raw_point"), ev["initial_point"]
    if not native.get("solve_started"):
        return dict(out, skipped="E MIQP not started")
    config = _config(rec)
    n, truth = rec["n"], rec["truth"]
    tr = cap.observation(rec["train"], rec["labels"])
    fitter = QPFitter(tr, reference_mse=rec["reference_mse"], bound=config.response_bound, qp_seconds=config.qp_seconds)
    pool, keep = canonical(ev["pool"]), canonical(ev["keep"])
    proper = [C for C in pool if 1 < len(C) < n]
    crossing = sum(1 for i, C in enumerate(proper) for D in proper[i+1:]
                   if set(C) & set(D) and not (set(C) <= set(D) or set(D) <= set(C)))
    out.update(pool=len(proper), crossing_pairs=crossing,
               run=dict(status=native.get("status"), objective=native.get("objective"), bound=native.get("bound"),
                        seconds=native.get("seconds"), allowance=native.get("requested_solve_seconds"),
                        family=_internal(raw["supports"], n) if raw else None))
    families = {}
    if raw:
        families["MIQP_run"] = _internal(raw["supports"], n)
    # F: the compressed MIQP with the run's own settings and start
    allowance = native.get("requested_solve_seconds") or 600.
    t = time.perf_counter()
    ptF, recF = certify.solve(fitter, config, pool, keep, initial, allowance)
    out["F"] = dict(status=recF.get("status"), objective=recF.get("objective"), bound=recF.get("bound"),
                    seconds=recF.get("seconds"), wall=time.perf_counter()-t,
                    family=_internal(ptF["supports"], n) if ptF else None)
    if ptF:
        families["MIQP_compressed"] = _internal(ptF["supports"], n)
    # certification: MIPGap 1e-6, 900 s, from the run's solution
    t = time.perf_counter()
    ptC, recC = certify.solve(fitter, config, pool, keep, raw if raw else initial, 900., mip_gap=1e-6)
    out["certify"] = dict(status=recC.get("status"), objective=recC.get("objective"), bound=recC.get("bound"),
                          gap=recC.get("gap"), seconds=time.perf_counter()-t)
    if ptC:
        families["MIQP_certified"] = _internal(ptC["supports"], n)
    # heuristics
    uf = UniverseFitter(tr, pool, rec["reference_mse"], config.alpha, bound=config.response_bound, keep=keep)
    prob = heu.Problem(uf, pool, keep)
    pool_set = set(proper)
    e_start = [C for C in _internal(initial["supports"], n) if C in pool_set]
    trees_ = [[C for C in _internal(f, n) if C in pool_set] for f in rec["guidance"]["pool_families"]]
    seed = zlib.crc32(f"{rec['cond']}/{rec['name']}/{rec['mode']}".encode())
    H = heu.run_all(prob, e_start=e_start, rnj_trees=trees_, seed=seed)
    out["heuristic_seconds"] = {k: v["seconds"] for k, v in H.items()}
    out["heuristic_search_objective"] = {k: v["objective"] for k, v in H.items()}
    for k, v in H.items():
        families[k] = [tuple(C) for C in v["family"]]
    # rescore everything with the package fit
    scored = {}
    for k, fam in families.items():
        obj, closed, closed_fam = _rescore(fitter, config, keep, fam)
        scored[k] = dict(objective=obj, closed=closed, family=fam, closed_family=closed_fam,
                         truth=_truth_metrics(fam, truth), closed_truth=_truth_metrics(closed_fam, truth))
    hk = [k for k in scored if k.startswith("H")]
    best = min(hk, key=lambda k: scored[k]["objective"])
    scored["P_portfolio"] = dict(scored[best], member=best)
    out["scored"] = scored
    out["optimum"] = min(v["objective"] for v in scored.values())
    out["certified"] = recC.get("status") == 2
    out["truth"] = len(truth)
    return out


def exp_a2(path):
    """Deviation D1 (RESULTS.md): re-certify E with the SVD-compressed MIQP, warm-started from the best known
    family of a.jsonl, and repeat F with that formulation."""
    rec = cap.load_instance(path)
    key = (rec["cond"], rec["name"], rec["mode"])
    out = dict(cond=key[0], name=key[1], mode=key[2], n=rec["n"])
    row = next((json.loads(l) for l in open(OUT/"a.jsonl")
                if tuple(json.loads(l)[k] for k in ("cond", "name", "mode")) == key), None)
    if row is None or "scored" not in row:
        return dict(out, skipped="no experiment A row")
    ev = rec["E"][0]["event"]
    config = _config(rec)
    n = rec["n"]
    tr = cap.observation(rec["train"], rec["labels"])
    fitter = QPFitter(tr, reference_mse=rec["reference_mse"], bound=config.response_bound, qp_seconds=config.qp_seconds)
    pool, keep = canonical(ev["pool"]), canonical(ev["keep"])
    best = min(row["scored"], key=lambda k: row["scored"][k]["objective"])
    warm = fitter.fit(canonical([*keep, *basis(n), *map(tuple, row["scored"][best]["family"])]))
    cls = certify.CERTIFIER
    t = time.perf_counter()
    pt, rc = certify.solve(fitter, config, pool, keep, warm, 900., mip_gap=1e-6, cls=cls)
    fam = _internal(pt["supports"], n) if pt else None
    out["certify"] = dict(status=rc.get("status"), objective=rc.get("objective"), bound=rc.get("bound"),
                          gap=rc.get("gap"), seconds=time.perf_counter()-t, family=fam, warm_from=best,
                          rescored=_rescore(fitter, config, keep, fam)[0] if fam is not None else None,
                          truth=_truth_metrics(fam, rec["truth"]) if fam is not None else None)
    native = ev["solver"]
    allowance = native.get("requested_solve_seconds") or 600.
    t = time.perf_counter()
    pF, rF = certify.solve(fitter, config, pool, keep, ev["initial_point"], allowance, cls=cls)
    famF = _internal(pF["supports"], n) if pF else None
    out["F2"] = dict(status=rF.get("status"), objective=rF.get("objective"), bound=rF.get("bound"),
                     seconds=rF.get("seconds"), wall=time.perf_counter()-t, family=famF,
                     rescored=_rescore(fitter, config, keep, famF)[0] if famF is not None else None)
    return out


def exp_a3(path):
    """Deviation D3 (PROTOCOL.md): a second certificate from the cold start, Gurobi Seed 7."""
    rec = cap.load_instance(path)
    out = dict(cond=rec["cond"], name=rec["name"], mode=rec["mode"], n=rec["n"])
    if not rec["E"]:
        return dict(out, skipped="no E stage")
    ev = rec["E"][0]["event"]
    config = _config(rec)
    n = rec["n"]
    tr = cap.observation(rec["train"], rec["labels"])
    fitter = QPFitter(tr, reference_mse=rec["reference_mse"], bound=config.response_bound, qp_seconds=config.qp_seconds)
    pool, keep = canonical(ev["pool"]), canonical(ev["keep"])
    t = time.perf_counter()
    pt, rc = certify.solve(fitter, config, pool, keep, ev["initial_point"], 900., mip_gap=1e-6,
                           cls=certify.SVDSolverNF2Seed7)
    fam = _internal(pt["supports"], n) if pt else None
    out["certify2"] = dict(status=rc.get("status"), objective=rc.get("objective"), bound=rc.get("bound"),
                           gap=rc.get("gap"), seconds=time.perf_counter()-t, family=fam,
                           rescored=_rescore(fitter, config, keep, fam)[0] if fam is not None else None)
    return out


# ---------------------------------------------------------------- experiment B1
def _generator_sets(family, n, data, point, pool, scale):
    sets, errors = {}, {}
    sets["G1_pool"] = gen.g1_pool(family, n, pool)
    sets["G2_pairs"] = gen.g2_sibling_pairs(family, n)
    for name, fn in (("G3_residual_rnj", lambda: gen.g3_residual_rnj(family, n, data, point)),
                     ("G4_clusters", lambda: gen.g4_residual_clusters(family, n, data, point)),
                     ("G5_screen", lambda: gen.g5_screen(family, n, data, point, scale))):
        try:
            sets[name] = fn()
        except Exception as exc:
            sets[name] = []; errors[name] = f"{type(exc).__name__}: {exc}"
    return sets, errors


def exp_b1(path):
    rec = cap.load_instance(path)
    config = _config(rec)
    n = rec["n"]
    tr = cap.observation(rec["train"], rec["labels"])
    fitter = QPFitter(tr, reference_mse=rec["reference_mse"], bound=config.response_bound, qp_seconds=config.qp_seconds)
    scale = np.sqrt(rec["reference_mse"])
    rows = []
    for step, w in enumerate(rec["W"]):
        ev, current = w["event"], w["current"]
        family = canonical(current["supports"])
        native = ev.get("solver") or {}
        row = dict(cond=rec["cond"], name=rec["name"], mode=rec["mode"], n=n, step=step, last=step == len(rec["W"])-1,
                   accepted=ev.get("accepted"), proposal_source=ev.get("returned_point_source"),
                   structurally_complete=bool(ev.get("structurally_complete")),
                   status=native.get("status"), miqp_seconds=native.get("seconds"),
                   compatible=count_compatible(family, n))
        if not native.get("solve_started"):
            rows.append(dict(row, skipped="no MIQP")); continue
        t = time.perf_counter()
        ext = ExtensionFitter(tr, family, rec["reference_mse"], config.alpha, bound=config.response_bound)
        none_obj = ext.objective(None)
        raw = ev.get("raw_point")
        miqp_new = None
        if raw is not None:
            new = [tuple(sorted(C)) for C in raw["supports"] if tuple(sorted(C)) not in set(family)]
            miqp_new = new[0] if new else None
        miqp_obj = ext.objective(miqp_new) if raw is not None else None
        sets, errors = _generator_sets(family, n, tr, current, w["pool"], scale)
        if row["compatible"] <= 20000:
            sets["G6_brute"], _ = compatible_supports(family, n)
        best = {}
        for k, S in sets.items():
            vals = [(ext.objective(C), C) for C in S]
            vals.append((none_obj, None))
            best[k] = min(vals, key=lambda v: v[0])
        portfolio = min((best[k] for k in best if k != "G6_brute"), key=lambda v: v[0])
        best["Gp_portfolio"] = portfolio
        # closed comparison (zero cleanup), package fit
        def closed(C):
            fam = canonical([*family, C]) if C else family
            pt, _ = algorithm._clean(fitter, fitter.fit(fam), config)
            return float(criterion(pt, n, config.alpha))
        current_closed = closed(None)
        row.update(miqp_objective=miqp_obj, gurobi_objective=native.get("objective"), bound=native.get("bound"),
                   none_objective=none_obj, miqp_support=miqp_new,
                   best={k: dict(objective=v[0], support=v[1]) for k, v in best.items()},
                   sizes={k: len(S) for k, S in sets.items()},
                   miqp_in_set={k: (miqp_new in set(S)) if miqp_new else None for k, S in sets.items()},
                   generator_errors=errors, current_closed=current_closed,
                   miqp_closed=closed(miqp_new) if raw is not None else None,
                   portfolio_closed=closed(portfolio[1]),
                   seconds=time.perf_counter()-t)
        rows.append(row)
    return rows


# ---------------------------------------------------------------- experiment B2
def _generator_hook(current, fitter):
    family = canonical(current["supports"])
    n, data = fitter.data.n, fitter.data
    scale = np.sqrt(fitter.reference_mse)
    out = set(gen.g2_sibling_pairs(family, n))
    for fn in (lambda: gen.g3_residual_rnj(family, n, data, current),
               lambda: gen.g4_residual_clusters(family, n, data, current),
               lambda: gen.g5_screen(family, n, data, current, scale)):
        try:
            out |= set(fn())
        except Exception:
            pass
    return sorted(out)


B2_VARIANTS = {"miqp_W": dict(w_mode="full"), "pool_W": dict(w_mode="pool_only"),
               "generator_W": dict(w_mode="pool_only", hook=True)}


def exp_b2(job):
    cond, name, variant = job
    spec = B2_VARIANTS[variant]
    config = replace(cap.CONFIG, w_mode=spec["w_mode"])
    inp = cap.load_input(cond, name)
    truth = sorted(cap.truth_internal(inp))
    removed = sorted(C for C in truth if len(C) >= 3)
    out = dict(cond=cond, name=name, variant=variant, n=inp.n, truth=len(truth), removed=len(removed))
    if not removed:
        return dict(out, skipped="no true support of size >= 3")
    t = time.perf_counter()
    record, result = cap.run(cond, name, "dm", config=config, common_hook=cap.truth_removed_hook,
                             pool_hook=_generator_hook if spec.get("hook") else None)
    sel, outp = set(map(tuple, record["selected"])), set(map(tuple, record["output"]))
    sources = [w["event"].get("returned_point_source") for w in record["W"] if w["event"].get("accepted")]
    out.update(seconds=time.perf_counter()-t, selected=record["selected"],
               recovered_selected=sum(C in sel for C in removed), recovered_output=sum(C in outp for C in removed),
               selected_truth=_truth_metrics(record["selected"], truth), exact=set(map(tuple, truth)) == sel,
               w_steps=len(record["W"]), w_accepted_sources=sources,
               e_status=record["E"][0]["event"]["solver"].get("status") if record["E"] else None)
    return out


# ---------------------------------------------------------------- driver
class _safe:
    """Picklable wrapper: an error becomes a record instead of stopping the pool."""
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, arg):
        try:
            return self.fn(arg)
        except Exception as exc:
            return dict(job=str(arg), error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-2000:])


def _write(path, rows):
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=lambda o: list(o) if isinstance(o, (tuple, set)) else str(o))+"\n")
        fh.flush()


def _done(path, key):
    if not path.exists():
        return set()
    out = set()
    for line in open(path):
        r = json.loads(line)
        if "error" not in r:
            out.add(key(r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("capture", "a", "a2", "a3", "b1", "b2"))
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = cap.select_inputs()[:args.limit]
    if args.stage == "capture":
        jobs = [(c, nm, m) for c, nm in inputs for m in ("lv", "dm")]
        fn, target = cap.capture_one, OUT/"capture.jsonl"
    elif args.stage in ("a", "a2", "a3", "b1"):
        paths = [cap.instance_path(c, nm, m) for c, nm in inputs for m in ("lv", "dm")]
        paths = [p for p in paths if p.exists()]
        target = OUT/f"{args.stage}.jsonl"
        if args.stage in ("a", "a2", "a3"):
            done = _done(target, lambda r: (r["cond"], r["name"], r["mode"]))
        else:
            done = _done(target, lambda r: (r["cond"], r["name"], r["mode"]) if r.get("step") == 0 else None)
        jobs = [p for p in paths if tuple(p.stem.split("__")) not in done]
        fn = _safe({"a": exp_a, "a2": exp_a2, "a3": exp_a3, "b1": exp_b1}[args.stage])
    else:
        jobs = [(c, nm, v) for c, nm in inputs for v in B2_VARIANTS]
        target = OUT/"b2.jsonl"
        done = _done(target, lambda r: (r["cond"], r["name"], r["variant"]))
        jobs = [j for j in jobs if j not in done]
        fn = _safe(exp_b2)
    print(f"{args.stage}: {len(jobs)} jobs on {args.workers} workers", flush=True)
    started = time.perf_counter()
    with Pool(args.workers, maxtasksperchild=4) as pool:
        for i, res in enumerate(pool.imap_unordered(fn, jobs), 1):
            rows = res if isinstance(res, list) else [res]
            _write(target, rows)
            tag = rows[0].get("error") if rows and "error" in rows[0] else "ok"
            print(f"[{i}/{len(jobs)}] {time.perf_counter()-started:.0f}s {tag[:120]}", flush=True)


if __name__ == "__main__":
    main()
