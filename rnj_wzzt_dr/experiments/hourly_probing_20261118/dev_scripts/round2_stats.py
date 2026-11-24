import json, numpy as np
R = r"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120"
tot = 0; viol = 0; gs = []; vmin, vmax = 9, 0; cmax = 0; types = {}
for c in ("bias_typical", "v2_bias_typical", "v2_shortline", "v2_besteffort", "v2_shortline_besteffort"):
    recs = [json.loads(l) for l in open(rf"{R}\{c}\runs.jsonl")]
    tot += len(recs)
    for r in recs:
        viol += r["violations"]["ac_band_hours"]+r["violations"]["deadband_hours"]
        vmin = min(vmin, r["violations"]["min_v"]); vmax = max(vmax, r["violations"]["max_v"])
        gs += [g["seconds"] for g in r["gates"]]; cmax = max(cmax, r["cost_cny"])
        for t in r["ever_wrong_types"]:
            types[(c, t)] = types.get((c, t), 0)+1
    print(c, "runs", len(recs), "failures", sum(r.get("refit_failures", 0) for r in recs))
print("total days", tot, "violation hours", viol, "v range", round(vmin, 4), round(vmax, 4), "gate s mean", round(np.mean(gs), 2), "p99", round(np.percentile(gs, 99), 1), "max", round(max(gs), 1), "max cost", round(cmax, 1))
print(sorted(types.items()))
