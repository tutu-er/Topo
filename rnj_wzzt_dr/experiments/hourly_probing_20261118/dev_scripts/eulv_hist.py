"""Real feeders: does a longer hourly history, or counting one identifiable parameter per support (Q is a fixed
fraction of P in this data set, so x is not identifiable), put the truth back in the bank / near the leader?"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
import numpy as np
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, model
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import scenario as sc
from rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments import settings_for

JOBS = [(c, h) for c in ("eulv_td009116_feeder106", "eulv_td008690_feeder21") for h in ((24, 12), (72, 24), (144, 48))]


def one(job):
    case, hist = job
    s = settings_for("eulv")
    t = time.time()
    inp = bank.build(case, 0, s, history=hist, step1="v2")
    n = inp.n
    zb = inp.network["base_kv"]**2/inp.network["base_mva"]
    truth = model.canonical_family(sc.truth_family(inp.network, inp.labels), n)
    fams = list(inp.families)+([truth] if not inp.truth_in_bank else [])
    fits, J, _ = bank.passive_fit(fams, n, inp.hist, 2*s.r_min_ohm/zb, s.upper_bound, offsets=True, whiten=True)
    N = inp.hist["Y"].size
    sse = np.array([f.sse for f in fits])
    out = {}
    for label, per in (("2 per support (as scored)", 2), ("1 per support (identifiable)", 1)):
        p = np.array([per*len(f)+n for f in fams])          # len(f) = supports incl. singletons; + n offsets
        s2 = float(np.min(sse/(N-p)))
        Jx = sse/(2*s2)+.5*p*np.log(N)
        kt = next(k for k, f in enumerate(fams) if model.nonsingleton(f) == inp.truth_key)
        out[label] = dict(truth_minus_best=round(float(Jx[kt]-Jx.min()), 1),
                          truth_rank=int(np.sum(Jx < Jx[kt])),
                          best_supports=len(model.nonsingleton(fams[int(np.argmin(Jx))])))
    return dict(case=case, history=hist, stratum=inp.meta["passive"]["stratum"], truth_in_bank=inp.truth_in_bank,
                closure=inp.meta["closure"].get("capped"), bank=len(inp.families), N=N,
                truth_supports=len(inp.truth_key), seconds=round(time.time()-t), **out)


if __name__ == "__main__":
    with ProcessPoolExecutor(6) as pool:
        for r in pool.map(one, JOBS):
            print(json.dumps(r))
