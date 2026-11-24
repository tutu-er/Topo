import glob, math, numpy as np
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, model, config, scenario as sc, scores
s = config.Settings(meter_bias_sigma=.0005, meter_offsets=True)
for p in sorted(glob.glob(r"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120\v2_shortline\inputs\*.pkl")):
    inp = bank.load(p)
    if inp.truth_in_bank or p.endswith("_missing.pkl"):
        continue
    n = inp.n; zb = inp.network["base_kv"]**2/inp.network["base_mva"]; rmin = 2*s.r_min_ohm/zb
    truth = model.canonical_family(sc.truth_family(inp.network, inp.labels), n)
    fams = list(inp.families)+[truth]
    fits = [bank.fit_one(f, n, inp.hist, rmin, s.upper_bound, 3, True, True) for f in fams]
    J, _ = scores.bic([f.sse for f in fits], [f.c.p+n for f in fits], inp.hist["Y"].size)
    E = J-J.min()
    rank = int(np.sum(E < E[-1]))
    print(inp.name, "closure", inp.meta["closure"], "truth E", round(float(E[-1]), 2), "rank", rank, "of", len(fams), "short", inp.meta["short_line"])
