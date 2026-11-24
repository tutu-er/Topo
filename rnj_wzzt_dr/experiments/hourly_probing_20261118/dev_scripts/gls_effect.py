import json, glob, numpy as np
from concurrent.futures import ProcessPoolExecutor
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, model, config, scores

R = r"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120"

def one(path):
    inp = bank.load(path)
    s = config.Settings(meter_bias_sigma=.0005, meter_offsets=True)
    out = {"name": inp.name}
    for wh in (False, True):
        fits, J, _ = bank.passive_fit(inp.families, inp.n, inp.hist, 2*.003/.8, 2., offsets=True, whiten=wh)
        E = J-J.min(); L = int(np.argmin(J))
        keys = [model.nonsingleton(f) for f in inp.families]
        ti = next((k for k, kk in enumerate(keys) if kk == inp.truth_key), None)
        others = np.sort(E[np.arange(len(E)) != L])
        out["gls" if wh else "ols"] = dict(leader_correct=keys[L] == inp.truth_key,
                                           truth_E=None if ti is None else float(E[ti]),
                                           gap2=float(others[0]), mass=float(np.exp(-others).sum()))
    return out

if __name__ == "__main__":
    for cond in ("v2_bias_typical", "cmp_short01"):
        rows = json.load(open(rf"{R}\{cond}\inputs.json"))
        s1 = [r["name"] for r in rows if r.get("stratum") == "S1"]
        paths = [rf"{R}\{cond}\inputs\{n}.pkl" for n in s1]
        with ProcessPoolExecutor(15) as ex:
            res = list(ex.map(one, paths))
        for k in ("ols", "gls"):
            lc = sum(r[k]["leader_correct"] for r in res)
            tE = [r[k]["truth_E"] for r in res if r[k]["truth_E"] is not None]
            g2 = [r[k]["gap2"] for r in res]
            m = [r[k]["mass"] for r in res]
            conf = sum(r[k]["leader_correct"] and r[k]["mass"] <= 1/19 for r in res)
            print(f"{cond} {k}: S1 n={len(res)} leader correct {lc}, truth gap median {np.median(tE):.2f}, "
                  f"runner-up gap median {np.median(g2):.2f}, rival mass median {np.median(m):.3f}, correct & mass<=1/19: {conf}")
