import json, numpy as np
R = r"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120"
tot = viol = 0; gs = []; vmin, vmax = 9, 0
for c in ("cmp_main", "cmp_short01", "cmp_lowflex", "cmp_highflex"):
    for l in open(rf"{R}\{c}\runs.jsonl"):
        r = json.loads(l); tot += 1
        viol += r["violations"]["ac_band_hours"]+r["violations"]["deadband_hours"]
        vmin = min(vmin, r["violations"]["min_v"]); vmax = max(vmax, r["violations"]["max_v"])
        gs += [g["seconds"] for g in r["gates"]]
print("days", tot, "violation hours", viol, "V", round(vmin, 4), round(vmax, 4), "gate s mean", round(np.mean(gs), 2), "max", round(max(gs), 1))
