import sys, os, json, math
sys.path.insert(0, r"C:\0-workspace\Topo\repo")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import diag
from rnj_wzzt_dr.physical import backend
from rnj_wzzt_dr.physical.backend import Observation
from rnj_wzzt_dr.identification import algorithm
from rnj_wzzt_dr.identification.algorithm import SearchConfig
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import scenario as sc, model, config

def ident(case, seed, mode, bias):
    s = config.Settings(meter_bias_sigma=bias)
    net, labels, _, _ = sc.network_info(case); n = len(labels)
    hist = (48, 24) if seed > 100 else (96, 48)
    T = sum(hist); days = math.ceil(T/96)+1
    P, Q, root = sc.generate_load(case, seed, days, network=net)
    hs = slice(96*(days-1)-T, 96*(days-1))
    v, _ = backend._ac_voltages(net, labels, P[hs], Q[hs], root[hs])
    meters = sc.make_meters(s, n, seed)
    rng = np.random.default_rng(np.random.SeedSequence([99, int(seed)]))
    p, q, y, vm, rm = meters.read(rng, P[hs], Q[hs], v, root[hs])
    tr = slice(0, hist[0]); va = slice(hist[0], T)
    def prep(x):
        a, b, c = p[x], q[x], y[x]
        if mode == "demean":
            a, b, c = a-a.mean(0), b-b.mean(0), c-c.mean(0)
        return Observation(a, b, c, labels)
    obs = [prep(tr), prep(va)]
    cfg = SearchConfig(total_seconds=900, per_mip_seconds=300, output_reserve_seconds=120)
    res = algorithm.fit(obs[0], obs[1], "111", config=cfg)
    truth = model.nonsingleton(model.canonical_family(sc.truth_family(net, labels), n))
    fams = [model.nonsingleton(model.canonical_family(c["point"]["supports"], n)) for c in res["frozen_candidates"]]
    sel = model.nonsingleton(model.canonical_family(res["selected"]["point"]["supports"], n))
    return dict(case=case, seed=seed, mode=mode, bias=bias, truth_in_bank=truth in fams, selected_correct=sel == truth)

if __name__ == "__main__":
    jobs = [(c, s, m, b) for c in ("soumalas11", "pengwah18") for s in (1, 2, 3, 4, 101, 102, 103, 104)
            for m, b in (("levels", 0.), ("demean", 0.), ("demean", .001))]
    with ProcessPoolExecutor(12) as pool:
        out = list(pool.map(ident, *zip(*jobs)))
    print(sorted(Counter((r["case"], r["mode"], r["bias"], r["truth_in_bank"], r["selected_correct"]) for r in out).items()))
    json.dump(out, open(r"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120\diag_demean.json", "w"), indent=1)
