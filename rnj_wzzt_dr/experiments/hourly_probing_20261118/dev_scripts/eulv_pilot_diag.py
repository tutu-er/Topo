"""Why the real-feeder pilot confirms wrong families: search failure or identifiability limit?

For every pilot input, refit the bank, the identifiable truth and the exact truth on the passive history with the
locked scorer, and the identifiable truth with each of its segments merged into its parent (one at a time).
  - truth_gap = J(identifiable truth) - J(bank leader): < 0 means the bank search missed a better-scoring truth
    (search failure); > 0 means the scorer prefers some bank family over the truth (identifiability limit).
  - per segment: dJ_drop = J(truth without the segment) - J(truth): < 0 means the passive history cannot support
    that segment; dv = max voltage change from merging it (history window).
Writes eulv/pilot_diag.jsonl and prints a summary."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys
from collections import Counter
from multiprocessing import Pool
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
import numpy as np
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, model, scores, experiments as ex
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import scenario as sc
from rnj_wzzt_dr.physical import backend

D = ex.ROOT/"eulv"
S = ex.settings_for("eulv")


def segments(network, labels):
    """Non-singleton descendant supports with the r_ohm of the edge above them."""
    order, parent, incoming = backend._radial_order(network)
    desc = {node: set() for node in order}
    for i, l in enumerate(labels):
        desc[l].add(i)
    out = {}
    for child in reversed(order[1:]):
        d = desc[child]
        if len(d) > 1:
            key = tuple(sorted(d))
            out[key] = out.get(key, 0.)+incoming[child]["r_ohm"]
        desc[parent[child]].update(d)
    return out


def one(name):
    inp = bank.load(D/"inputs"/f"{name}.pkl")
    n, hist = inp.n, inp.hist
    zbase = inp.network["base_kv"]**2/inp.network["base_mva"]
    r_min = 2*S.r_min_ohm/zbase
    kw = dict(offsets=S.meter_offsets, whiten=S.root_whitening, pq_sigma=S.pq_sigma if S.identifiable_params else None)
    seg_r = segments(inp.network, inp.labels)
    ident, _ = sc.identifiable_truth_family(inp.network, inp.labels, S.r_min_ohm)
    ident = model.canonical_family(ident, n)
    exact = model.canonical_family(sc.truth_family(inp.network, inp.labels), n)
    fams = list(inp.families)
    extra = [ident, exact]+[tuple(C for C in ident if C != seg) for seg in ident if len(seg) > 1]
    fits = [bank.fit_one(f, n, hist, r_min, S.upper_bound, 3, True, **kw) for f in fams+extra]
    J, sigma2 = scores.bic([f.sse for f in fits], [f.c.p+(n if S.meter_offsets else 0) for f in fits], hist["Y"].size)
    K = len(fams)
    Jb = J[:K]; L = int(np.argmin(Jb))
    Jt, Je = J[K], J[K+1]
    V0 = hist["V0sq"]
    vt = scores.voltages(fits[K].c, fits[K].theta, hist["P"], hist["Q"], V0)
    vL = scores.voltages(fits[L].c, fits[L].theta, hist["P"], hist["Q"], V0)
    segs = []
    j = K+2
    for seg in ident:
        if len(seg) > 1:
            vd = scores.voltages(fits[j].c, fits[j].theta, hist["P"], hist["Q"], V0)
            segs.append(dict(size=len(seg), r_ohm=seg_r.get(seg), dJ_drop=float(J[j]-Jt),
                             dv=float(np.max(np.abs(vd-vt)))))
            j += 1
    cl = inp.meta.get("closure") or {}
    return dict(input=name, n=n, K=K, stratum=inp.stratum, truth_in_bank=inp.truth_in_bank,
                closure=dict(rounds=cl.get("rounds"), capped=cl.get("capped"), size=cl.get("size"),
                             unexpanded=cl.get("unexpanded_within_margin")),
                truth_gap=float(Jt-Jb[L]), exact_gap=float(Je-Jb[L]), leader_type=scores.rival_type(fits[L].c.key, inp.truth_key),
                leader_truth_dv=float(np.max(np.abs(vL-vt))), sigma_y=float(np.sqrt(sigma2)),
                merged=inp.meta.get("sub_floor_segments_merged"), segments=segs)


if __name__ == "__main__":
    names = [r["name"] for r in json.loads((D/"inputs.json").read_text()) if "error" not in r]
    with Pool(14) as pool:
        out = pool.map(one, names)
    with open(D/"pilot_diag.jsonl", "w") as fh:
        for o in out:
            fh.write(json.dumps(o)+"\n")
    recs = {(r["input"], r["arm"]): r for r in map(json.loads, open(D/"pilot.jsonl"))}
    print(f"{len(out)} inputs")
    print("closure capped:", dict(Counter(str(o["closure"]["capped"]) for o in out)),
          "| unexpanded>0:", sum((o["closure"]["unexpanded"] or 0) > 0 for o in out))
    search = [o for o in out if not o["truth_in_bank"] and o["truth_gap"] < 0]
    ident_lim = [o for o in out if not o["truth_in_bank"] and o["truth_gap"] >= 0]
    print(f"S3 inputs {sum(not o['truth_in_bank'] for o in out)}: truth scores better than the bank leader "
          f"(search failure) {len(search)}, worse (identifiability limit) {len(ident_lim)}")
    print("  search-failure gaps:", sorted(round(o["truth_gap"], 1) for o in search))
    print("  identifiability gaps:", sorted(round(o["truth_gap"], 1) for o in ident_lim))
    print("in-bank inputs: truth gap", sorted(round(o["truth_gap"], 1) for o in out if o["truth_in_bank"]))
    allseg = [s for o in out for s in o["segments"]]
    weak = [s for s in allseg if s["dJ_drop"] < 0]
    print(f"identifiable-truth segments {len(allseg)}: passively unsupported (merging lowers BIC) {len(weak)}")
    for lo, hi in ((0, .01), (.01, .02), (.02, .05), (.05, 1)):
        ss = [s for s in allseg if lo <= (s["r_ohm"] or 0) < hi]
        if ss:
            print(f"  r in [{lo}, {hi}) ohm: {len(ss)} segments, unsupported {sum(s['dJ_drop'] < 0 for s in ss)}, "
                  f"median dv {np.median([s['dv'] for s in ss])*100:.3f}%")
    print("median sigma_y", np.median([o["sigma_y"] for o in out]))
    print("leader-vs-truth max voltage gap on history (S3):",
          sorted(round(100*o["leader_truth_dv"], 3) for o in out if not o["truth_in_bank"]))
    for arm in ("passive_bic", "bgreedy_P"):
        ew = [o for o in out if recs[(o["input"], arm)]["ever_confirmed_wrong"]]
        fc = [next((g["g"] for g in recs[(o["input"], arm)]["gates"] if g["confirmed"]), None) for o in ew]
        print(f"{arm}: ever wrong {len(ew)} (S1 {sum(o['stratum']=='S1' for o in ew)}, S3 {sum(o['stratum']=='S3' for o in ew)}); "
              f"first confirmed gate {dict(Counter(fc))}; trusted (closure finished) {sum(o['closure']['capped'] is None and not o['closure']['unexpanded'] for o in ew)}")
