"""RNJ + WZZT parameter check on real and synthetic inputs (development; WZZT_REVISION.md).

Inputs: the 33 real-feeder pilot inputs with start day 0 (target: identifiable truth) and the first 33 S1 inputs
of lock_short01 (synthetic, 0.01 ohm short line; target: exact truth). Training/validation = the build's split
(real 144/48 hourly rows, synthetic 192/96 quarter-hour rows), demeaned (the identifier's second run).

Part A, RNJ guidance alone: recall of the truth's segments in the pool and in each (weight, tolerance) tree, with
the current sensitivity fit (R and X from P and Q) and with a P-only fit (the identifiable combination when the
power factor is fixed).
Part B, WZZT 111 under parameter variants: price alpha, output floor, P-only RNJ; recall and precision of the
selected family and of everything WZZT outputs, and run time.
Writes outputs/.../eulv/rnj_wzzt_sweep.jsonl."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys, time
from multiprocessing import Pool
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
import numpy as np
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, experiments as ex
from rnj_wzzt_dr.identification import algorithm, trees
from rnj_wzzt_dr.identification.algorithm import SearchConfig
from rnj_wzzt_dr.physical.backend import Observation

WEIGHTS, TOLS = (1., .75, 0.), (0., .02, .04, .08, .16, .24)
CONFIGS = {"current": dict(alpha=.003, internal_floor=.08), "floor0": dict(alpha=.003, internal_floor=0.),
           "alpha001": dict(alpha=.001, internal_floor=.08), "alpha001_floor0": dict(alpha=.001, internal_floor=0.),
           "ponly_floor0": dict(alpha=.003, internal_floor=0., ponly=True)}
_orig_guidance = trees.rnj_guidance


def ponly_guidance(train):
    """RNJ guidance from a P-only sensitivity fit (Q columns zeroed), R geometry only."""
    zero = Observation(train.p, np.zeros_like(train.q), train.y, train.labels)
    g = _orig_guidance(zero)
    R = g["R"]
    families = []
    for tol in (0., .04, .08, .16, .24):
        shared, depth, _ = trees.geometry(R, R, 1.)
        families.append(trees.rnj(shared, depth, tol))
    shared, depth, cal = trees.geometry(R, R, 1.)
    g.update(pool=trees.canonical(s for f in families for s in f), pool_families=tuple(families),
             reference_family=trees.rnj(shared, depth, .16), X=R.copy(), shared=shared, depth=depth, calibration=cal)
    return g


def internal(fam, n):
    return {tuple(C) for C in fam if 1 < len(C) < n}


def rp(found, truth):
    return dict(hit=len(found & truth), size=len(found), truth=len(truth))


def one(job):
    cond, name = job
    inp = bank.load(ex.ROOT/cond/"inputs"/f"{name}.pkl")
    n, h = inp.n, inp.hist
    t0, t1 = inp.meta["history"]
    truth = {tuple(sorted(C)) for C in inp.truth_key if 1 < len(C) < n}
    dm = lambda s: Observation(*(a[s]-a[s].mean(0) for a in (h["P"], h["Q"], h["Y"])), inp.labels)
    tr, va = dm(slice(0, t0)), dm(slice(t0, t0+t1))
    out = dict(cond=cond, input=name, case=inp.case, n=n, truth=len(truth))
    # Part A
    for label, src in (("rnj_pq", tr), ("rnj_p", Observation(tr.p, np.zeros_like(tr.q), tr.y, tr.labels))):
        R, X, _ = trees.fit_sensitivity(src)
        pool, per = set(), {}
        for w in (WEIGHTS if label == "rnj_pq" else (1.,)):
            shared, depth, _ = trees.geometry(R, X if label == "rnj_pq" else R, w)
            for tol in TOLS:
                f = internal(trees.rnj(shared, depth, tol), n)
                pool |= f; per[f"{w}/{tol}"] = rp(f, truth)
        out[label] = dict(pool=rp(pool, truth), per=per)
    # Part B
    for cname, cfg in CONFIGS.items():
        trees.rnj_guidance = ponly_guidance if cfg.get("ponly") else _orig_guidance
        algorithm.trees.rnj_guidance = trees.rnj_guidance
        sc = SearchConfig(alpha=cfg["alpha"], internal_floor=cfg["internal_floor"], total_seconds=900,
                          per_mip_seconds=300, output_reserve_seconds=120)
        t = time.perf_counter()
        try:
            res = algorithm.fit(tr, va, "111", config=sc)
        except Exception as exc:
            out[cname] = dict(error=f"{type(exc).__name__}: {exc}"); continue
        sel = internal(res["selected"]["point"]["supports"], n)
        allout = set()
        for c in res["frozen_candidates"]:
            allout |= internal(c["point"]["supports"], n)
        for p in res.get("uncontracted_output_path", []):
            allout |= internal(p["supports"], n)
        out[cname] = dict(selected=rp(sel, truth), output=rp(allout, truth), exact=sel == truth,
                          seconds=time.perf_counter()-t)
    trees.rnj_guidance = _orig_guidance; algorithm.trees.rnj_guidance = _orig_guidance
    return out


if __name__ == "__main__":
    real = [r["name"] for r in json.loads((ex.ROOT/"eulv"/"inputs.json").read_text())
            if "error" not in r and r["name"].endswith("_s0")]
    syn = [r["name"] for r in json.loads((ex.ROOT/"lock_short01"/"inputs.json").read_text())
           if "error" not in r and r.get("stratum") == "S1"][:33]
    jobs = [("eulv", nm) for nm in real]+[("lock_short01", nm) for nm in syn]
    with Pool(14) as pool, open(ex.ROOT/"eulv"/"rnj_wzzt_sweep.jsonl", "w") as fh:
        for o in pool.imap_unordered(one, jobs):
            fh.write(json.dumps(o)+"\n"); fh.flush()
            print(o["input"], "done", flush=True)
