"""Step-1 truth coverage: the current passive bank against the revised step 1.

Development check (pre-lock). For each input (case, seed, history, meter bias,
short line) it builds a single-phase passive history and compares banks:

    A    identifier output (method 111, 0.08 contraction), as used so far
    A+   A plus one-edit neighbours of A's BIC leader (the bank of bank.build)
    B    A plus the identifier path before the 0.08 contraction
    C    B plus the identifier run on per-meter demeaned data (both paths)
    D    C closed under one-edit neighbours within a BIC margin, scored with
         per-meter offsets and the loss term (the revised step 1)
    D0   the same closure scored without offsets (to separate the two parts)

It also records the leader, the truth's BIC gap, the adequacy ratio and the
bank sizes. Truth is used only for evaluation.

    python -m rnj_wzzt_dr.experiments.hourly_probing_20261118.coverage --workers 14
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import time
import traceback

import numpy as np

from rnj_wzzt_dr.physical import backend
from rnj_wzzt_dr.physical.backend import Observation

from . import bank, config, model, scores
from . import scenario as sc
from .experiments import ROOT

OUT = ROOT/"coverage"
CASES = ("soumalas11", "pengwah18", "flynn16", "paper15")
HISTORIES = {"1d": (96, 48), "3d": (288, 96)}
REDO_PREFIX = ("pengwah18",)


def jobs(seeds=range(1, 7), seeds_3d=4):
    """1-day histories on every seed; 3-day histories on the first `seeds_3d` seeds."""
    out = []
    for case in CASES:
        for seed in seeds:
            for hname in (HISTORIES if seed <= seeds_3d else ["1d"]):
                for bias in (0., .0005, .001):
                    out.append((case, seed, hname, bias, None))
                for short in (.01, .005):
                    out.append((case, seed, hname, .0005, short))
    return out


def job_name(case, seed, hname, bias, short):
    return f"{case}_s{seed}_{hname}_b{bias:g}_l{short if short else 'none'}"


def _min_edit(truth_key, keys):
    return min(len(truth_key ^ k) for k in keys) if keys else None


def one(case, seed, hname, bias, short, seconds=600., margin=6.):
    name = job_name(case, seed, hname, bias, short)
    try:
        started = time.perf_counter()
        s = config.Settings(meter_bias_sigma=bias)
        history = HISTORIES[hname]
        network = backend.load_case(case)
        network, labels, _, _ = sc.network_info(case, network)
        info = None
        if short is not None:
            network, info = bank.shorten_segment(network, labels, seed, short)
        n = len(labels)
        zbase = network["base_kv"]**2/network["base_mva"]
        r_min = 2*s.r_min_ohm/zbase
        T = sum(history)
        days = math.ceil(T/96)+1
        P, Q, root = sc.generate_load(case, seed, days, network=network)
        hs = slice(96*(days-1)-T, 96*(days-1))
        v, _ = backend._ac_voltages(network, labels, P[hs], Q[hs], root[hs])
        meters = sc.make_meters(s, n, seed)
        rng = np.random.default_rng(np.random.SeedSequence([99, int(seed)]))
        p_m, q_m, y_m, v_m, r_m = meters.read(rng, P[hs], Q[hs], v, root[hs])
        hist = dict(P=p_m, Q=q_m, Y=y_m, V0sq=r_m**2)
        tr, va = slice(0, history[0]), slice(history[0], T)
        obs = [Observation(p_m[x], q_m[x], y_m[x], labels, v_m[x], r_m[x]) for x in (tr, va)]
        truth = model.canonical_family(sc.truth_family(network, labels), n)
        tkey = model.nonsingleton(truth)
        has = lambda fams: any(model.nonsingleton(f) == tkey for f in fams)

        lv = bank.identifier_families(obs[0], obs[1], n, seconds, demean=False)
        dm = bank.identifier_families(obs[0], obs[1], n, seconds, demean=True)
        A = list(dict.fromkeys(lv["contracted"]))
        fitsA, JA, _ = bank.passive_fit(A, n, hist, r_min, s.upper_bound)
        leadA = A[int(np.argmin(JA))]
        Aplus = list(A)+[model.canonical_family(nb, n) for nb in bank.neighbours(leadA, n)]
        B = list(dict.fromkeys(A+lv["uncontracted"]))
        C = list(dict.fromkeys(B+dm["contracted"]+dm["uncontracted"]))
        rec = dict(name=name, case=case, seed=seed, history=hname, bias=bias, short=short, short_info=info, n=n,
                   neighbour_max_children=8,
                   id_seconds=[lv["seconds"], dm["seconds"]],
                   valsel_levels=model.nonsingleton(lv["selected"]) == tkey,
                   valsel_demean=model.nonsingleton(dm["selected"]) == tkey,
                   cover=dict(A=has(A), Aplus=has(Aplus), B=has(B), C=has(C),
                              lv_uncontracted=has(lv["uncontracted"]), dm_any=has(dm["contracted"]+dm["uncontracted"])),
                   size=dict(A=len(A), Aplus=len(set(map(model.nonsingleton, Aplus))), B=len(B), C=len(C)),
                   min_edit_C=_min_edit(tkey, [model.nonsingleton(f) for f in C]))
        for label, offs in (("D", True), ("D0", False)):
            t0 = time.perf_counter()
            Dk, fits, J, sigma2, crec = bank.closure(C, n, hist, r_min, s.upper_bound, margin=margin, offsets=offs)
            E = J-J.min(); L = int(np.argmin(J))
            keys = [model.nonsingleton(f) for f in Dk]
            ti = next((k for k, kk in enumerate(keys) if kk == tkey), None)
            N = hist["Y"].size
            # day-ahead voltage gap between leader and truth (single phase, no offsets)
            vgap = None
            if ti is not None and ti != L:
                Pd, Qd, V0d = P[96*(days-1):], Q[96*(days-1):], root[96*(days-1):]**2
                vL = scores.voltages(fits[L].c, fits[L].theta, Pd, Qd, V0d)
                vt = scores.voltages(fits[ti].c, fits[ti].theta, Pd, Qd, V0d)
                vgap = float(np.max(np.abs(vL-vt)))
            rec[label] = dict(cover=ti is not None, leader_correct=keys[L] == tkey,
                              truth_E=None if ti is None else float(E[ti]),
                              truth_within_30=ti is not None and E[ti] <= 30,
                              pruned_size=int(np.sum(E <= 30)), leader_truth_vgap=vgap,
                              adequacy_leader=bank.adequacy(fits[L].sse, fits[L].c.p, N, n, offs, s.voltage_sigma),
                              adequacy_truth=None if ti is None else bank.adequacy(fits[ti].sse, fits[ti].c.p, N, n, offs, s.voltage_sigma),
                              seconds=time.perf_counter()-t0, **crec)
        rec["seconds"] = time.perf_counter()-started
        return rec
    except Exception as exc:
        return dict(name=name, error=repr(exc), trace=traceback.format_exc())


def report(path=None):
    """Coverage table by condition (history, bias, short line), all cases pooled and per case."""
    path = path or OUT/"coverage.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if "error" not in r and (r["history"] == "1d" or r["seed"] <= 4)]
    rows = list({r["name"]: r for r in rows}.values())          # the latest record per input
    groups = {}
    for r in rows:
        for key in [(r["history"], r["bias"], r["short"], "all"), (r["history"], r["bias"], r["short"], r["case"])]:
            groups.setdefault(key, []).append(r)
    out = []
    for key in sorted(groups, key=lambda k: (k[0], k[2] or 0, k[1], k[3] != "all", k[3])):
        g = groups[key]
        frac = lambda f: sum(bool(f(r)) for r in g)
        tE = [r["D"]["truth_E"] for r in g if r["D"]["truth_E"] is not None]
        out.append(dict(history=key[0], bias=key[1], short=key[2], case=key[3], n=len(g),
                        A=frac(lambda r: r["cover"]["A"]), Aplus=frac(lambda r: r["cover"]["Aplus"]),
                        B=frac(lambda r: r["cover"]["B"]), C=frac(lambda r: r["cover"]["C"]),
                        D=frac(lambda r: r["D"]["cover"]), D_within30=frac(lambda r: r["D"]["truth_within_30"]),
                        D0=frac(lambda r: r["D0"]["cover"]),
                        valsel=frac(lambda r: r["valsel_levels"]),
                        lead_D=frac(lambda r: r["D"]["leader_correct"]), lead_D0=frac(lambda r: r["D0"]["leader_correct"]),
                        truth_E_median=float(np.median(tE)) if tE else None,
                        truth_E_max=float(np.max(tE)) if tE else None,
                        wrong_leader_vgap_max=max([r["D"]["leader_truth_vgap"] or 0. for r in g]),
                        pruned_median=float(np.median([r["D"]["pruned_size"] for r in g])),
                        closure_median=float(np.median([r["D"]["size"] for r in g])),
                        capped=frac(lambda r: r["D"]["capped"]),
                        adequacy_leader_D=float(np.median([r["D"]["adequacy_leader"] for r in g])),
                        adequacy_leader_D0=float(np.median([r["D0"]["adequacy_leader"] for r in g])),
                        adequacy_leader_D0_max=float(np.max([r["D0"]["adequacy_leader"] for r in g])),
                        id_seconds_median=float(np.median([sum(r["id_seconds"]) for r in g]))))
    (OUT/"coverage_summary.json").write_text(json.dumps(out, indent=1))
    hdr = "hist bias  short case        n |  A  A+  B  C  D D30 D0 | val lD lD0 | E_med E_max vgap  | pruned clos cap | adqD adqD0 max"
    print(hdr)
    for o in out:
        print(f"{o['history']:4s} {o['bias']:.4f} {str(o['short']):5s} {o['case']:10s} {o['n']:2d} | "
              f"{o['A']:2d} {o['Aplus']:2d} {o['B']:2d} {o['C']:2d} {o['D']:2d} {o['D_within30']:2d} {o['D0']:2d} | "
              f"{o['valsel']:2d} {o['lead_D']:2d} {o['lead_D0']:2d} | "
              f"{o['truth_E_median'] if o['truth_E_median'] is not None else float('nan'):5.1f} "
              f"{o['truth_E_max'] if o['truth_E_max'] is not None else float('nan'):5.1f} {1000*o['wrong_leader_vgap_max']:.2f}m | "
              f"{o['pruned_median']:5.0f} {o['closure_median']:5.0f} {o['capped']:2d} | "
              f"{o['adequacy_leader_D']:.2f} {o['adequacy_leader_D0']:.2f} {o['adequacy_leader_D0_max']:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=600.)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.report:
        report(); return
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/"coverage.jsonl"
    done = set()
    if path.exists():
        # records from before the neighbour fix (no neighbour_max_children) on feeders where the old
        # limit left merged segments unreachable are redone (Pengwah18; Flynn16 had none pre-fix)
        for l in path.read_text().splitlines():
            r = json.loads(l) if l.strip() else None
            if r and "error" not in r and ("neighbour_max_children" in r or r["case"] not in REDO_PREFIX):
                done.add(r["name"])
    todo = [j for j in jobs(range(1, a.seeds+1)) if job_name(*j) not in done][:a.limit]
    print(f"{len(todo)} jobs", flush=True)
    with ProcessPoolExecutor(a.workers) as pool, path.open("a") as fh:
        futs = [pool.submit(one, *j, seconds=a.seconds) for j in todo]
        for i, fut in enumerate(as_completed(futs)):
            rec = fut.result()
            fh.write(json.dumps(rec, default=float)+"\n"); fh.flush()
            print(i+1, rec["name"], "ERROR" if "error" in rec else
                  f"A={rec['cover']['A']} C={rec['cover']['C']} D={rec['D']['cover']} lead={rec['D']['leader_correct']} {rec['seconds']:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
