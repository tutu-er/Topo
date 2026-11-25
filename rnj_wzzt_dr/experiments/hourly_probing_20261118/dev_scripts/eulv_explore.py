"""Exploratory (not evidence; changes a locked setting): on the 66 pilot inputs,
 (a) the strong greedy with the trust gate OFF - can probing discriminate on real feeders if it is allowed to act?
 (b) harm baseline - the identifiable truth's own voltage error vs AC truth on the evaluation day, fitted on the
     history, to compare with the wrong confirmations' error.
Writes eulv/explore_notrust.jsonl and eulv/explore_baseline.jsonl."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
import numpy as np
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, model, run, scores, experiments as ex
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import scenario as sc

D = ex.ROOT/"eulv"
S = ex.settings_for("eulv")


def job(name):
    inp = bank.load(D/"inputs"/f"{name}.pkl")
    rec = run.run(inp, "bgreedy_P", 0, S.with_(trust_gate=False))
    rec.pop("gates"); rec.pop("blocks", None)
    zbase = inp.network["base_kv"]**2/inp.network["base_mva"]
    kw = dict(offsets=S.meter_offsets, whiten=S.root_whitening, pq_sigma=S.pq_sigma)
    ident, _ = sc.identifiable_truth_family(inp.network, inp.labels, S.r_min_ohm)
    f = bank.fit_one(model.canonical_family(ident, inp.n), inp.n, inp.hist, 2*S.r_min_ohm/zbase, S.upper_bound, 3, True, **kw)
    physics = sc.Physics(inp.network, inp.labels, S, seed=inp.seed, households=inp.res["households"])
    v_bg = physics.voltages(inp.day["P"], inp.day["Q"], inp.day["root"])
    vt = scores.voltages(f.c, f.theta, inp.day["P"], inp.day["Q"], inp.day["root"]**2)
    base = dict(input=name, truth_fit_vdev=float(np.max(np.abs(vt-v_bg))))
    return rec, base


if __name__ == "__main__":
    names = [r["name"] for r in json.loads((D/"inputs.json").read_text()) if "error" not in r]
    with ProcessPoolExecutor(14) as pool, open(D/"explore_notrust.jsonl", "w") as fa, \
            open(D/"explore_baseline.jsonl", "w") as fb:
        for rec, base in pool.map(job, names):
            fa.write(json.dumps(rec, default=ex._json)+"\n"); fb.write(json.dumps(base)+"\n")
    print("done", len(names))
