"""Is WZZT's own criterion weaker than the downstream score, and does more history help?

For the 33 real feeders (start day 0), per segment of the identifiable truth, "supported" means dropping the segment
(merging it into its parent) makes the criterion worse:
  (a) study score: BIC with GLS whitening, per-meter offsets, identifiable parameters, on the whole history
  (b) WZZT on levels: normalised training MSE + 0.003 per internal support (no offsets, no whitening)
  (c) WZZT on demeaned data (the second identifier run)
  plus (d) whether WZZT's 0.08 output contraction would remove the segment (relative response s_C < 0.08).
(a) is computed for 8 and for 18 days of history; (b)-(d) on the 8-day training part (144 h), as in the build.
Also times one fixed-family fit of each kind."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys, time
from multiprocessing import Pool
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
import numpy as np
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, model, scores, experiments as ex
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import scenario as sc
from rnj_wzzt_dr.identification import algorithm, output_floor
from rnj_wzzt_dr.identification.optimization import QPFitter
from rnj_wzzt_dr.physical.backend import Observation

S = ex.settings_for("eulv")


def history(case, seed, days_hist):
    network = sc.load_network(case)
    network, labels, _, _ = sc.network_info(case, network)
    days = days_hist+1
    P, Q, root = sc.generate_load(case, seed, days, network=network, sph=1)
    hs = slice(0, 24*days_hist)
    households = sc.households_of(network)
    res = sc.resources(S, P[hs].mean(axis=0)*network["base_kw"], network["base_kw"], households)
    physics = sc.Physics(network, labels, S, seed=seed, households=res["households"])
    v = physics.voltages(P[hs], Q[hs], root[hs])
    meters = sc.make_meters(S, len(labels), seed)
    rng = np.random.default_rng(np.random.SeedSequence([99, int(seed)]))
    p_m, q_m, y_m, v_m, r_m = meters.read(rng, P[hs], Q[hs], v, root[hs])
    return network, labels, dict(P=p_m, Q=q_m, Y=y_m, V0sq=r_m**2), v_m, r_m


def bic_support(network, labels, hist, ident):
    n = len(labels)
    zbase = network["base_kv"]**2/network["base_mva"]
    kw = dict(offsets=S.meter_offsets, whiten=S.root_whitening, pq_sigma=S.pq_sigma)
    segs = [C for C in ident if len(C) > 1]
    fams = [ident]+[tuple(D for D in ident if D != C) for C in segs]
    t = time.perf_counter()
    fits = [bank.fit_one(f, n, hist, 2*S.r_min_ohm/zbase, S.upper_bound, 3, True, **kw) for f in fams]
    sec = (time.perf_counter()-t)/len(fams)
    J, _ = scores.bic([f.sse for f in fits], [f.c.p+n for f in fits], hist["Y"].size)
    return [bool(J[k+1] > J[0]) for k in range(len(segs))], sec


def wzzt_support(obs, ident, n):
    common = algorithm.prepare_common(obs)
    fitter = QPFitter(obs, reference_mse=common["reference_mse"], bound=2.)
    segs = [C for C in ident if len(C) > 1]
    t = time.perf_counter()
    full = fitter.fit(ident)
    sec = time.perf_counter()-t
    sup = []
    for C in segs:
        drop = fitter.fit(tuple(D for D in ident if D != C))
        sup.append(bool(drop["objective"]-full["objective"] > .003))
    cal = output_floor.calibrate(common["guidance"], obs)
    look = {tuple(c): (r, x) for c, r, x in zip(full["supports"], full["r"], full["x"])}
    s_C = [(cal["a"]*look[C][0]+cal["b"]*look[C][1])/cal["m"] for C in segs]
    return sup, [bool(s >= .08) for s in s_C], sec


def one(case):
    out = dict(case=case)
    for days in (8, 18):
        network, labels, hist, v_m, r_m = history(case, 0, days)
        n = len(labels)
        ident, _ = sc.identifiable_truth_family(network, labels, S.r_min_ohm)
        ident = model.canonical_family(ident, n)
        sup, sec = bic_support(network, labels, hist, ident)
        out[f"bic_{days}"] = sup; out["fit_one_seconds"] = sec
        if days == 8:
            tr = slice(0, 144)
            lv = Observation(hist["P"][tr], hist["Q"][tr], hist["Y"][tr], labels, v_m[tr], r_m[tr])
            dm = Observation(*(a-a.mean(0) for a in (hist["P"][tr], hist["Q"][tr], hist["Y"][tr])), labels)
            out["wzzt_lv"], out["floor_lv"], out["wzzt_fit_seconds"] = wzzt_support(lv, ident, n)
            out["wzzt_dm"], out["floor_dm"], _ = wzzt_support(dm, ident, n)
            out["n"] = n
    return out


if __name__ == "__main__":
    cases = sorted(json.loads((sc.EULV_CASES/"index.json").read_text()).keys()) if isinstance(
        json.loads((sc.EULV_CASES/"index.json").read_text()), dict) else None
    if cases is None:
        cases = sorted({r["case"] for r in json.loads((ex.ROOT/"eulv"/"inputs.json").read_text())})
    with Pool(14) as pool:
        res = pool.map(one, cases)
    (ex.ROOT/"eulv"/"wzzt_criterion.json").write_text(json.dumps(res))
    tot = lambda k: sum(len(r[k]) for r in res)
    cnt = lambda k: sum(sum(r[k]) for r in res)
    print(f"{len(res)} feeders, {tot('bic_8')} identifiable-truth segments")
    for k, label in (("bic_8", "study BIC, 8 days"), ("bic_18", "study BIC, 18 days"),
                     ("wzzt_lv", "WZZT criterion, levels (6 days training)"), ("wzzt_dm", "WZZT criterion, demeaned"),
                     ("floor_lv", "survives 0.08 contraction, levels"), ("floor_dm", "survives 0.08 contraction, demeaned")):
        print(f"  {label}: supported {cnt(k)} ({100*cnt(k)/tot(k):.0f}%)")
    both = sum(a and b for r in res for a, b in zip(r["wzzt_dm"], r["floor_dm"]))
    print(f"  WZZT demeaned: supported AND survives contraction {both}")
    print(f"  seconds per fit: study fit_one median {np.median([r['fit_one_seconds'] for r in res]):.3f}, "
          f"WZZT QP median {np.median([r['wzzt_fit_seconds'] for r in res]):.3f}")
