import json, collections, numpy as np
base = r"C:/0-workspace/Topo/repo/outputs/hourly_probing_dev_20261120/"
tot = collections.defaultdict(lambda: np.zeros(4))
for cond in ["conf_main", "conf_short01"]:
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(base+cond+"/runs.jsonl") as f:
        for line in f:
            r = json.loads(line)
            st = "S3" if r["stratum"] == "S3" else "all"
            up = dn = 0.
            for b in r["blocks"]:
                u = np.asarray(b["u_kw"], float); up += u[u > 0].sum(); dn += -u[u < 0].sum()
            dr = 4.*dn+1.2*up; flat = 4.*(up+dn)
            for key in [(st, r["arm"])] + ([("S1", r["arm"])] if r["stratum"] == "S1" else []):
                a = agg[key]; a["model"].append(r["cost_cny"]); a["dr"].append(dr); a["flat"].append(flat)
                a["kwh"].append(up+dn); a["probed"].append(r["probed_hours"] > 0)
            if r["arm"] != "passive_bic":
                tot[r["arm"]] += [1, up+dn, dr, flat]
    print("==", cond)
    print(f"{'set/arm':24s} {'runs':>4s} {'kWh/day':>7s} {'model':>6s} {'DR 4/1.2':>8s} {'flat 4':>7s} {'probed days':>11s} {'DR per probed day':>17s} {'p90 DR':>7s} {'max DR':>7s}")
    for key in sorted(agg):
        a = agg[key]
        if key[1] == "passive_bic": continue
        dr = np.array(a["dr"]); pr = np.array(a["probed"])
        print(f"{key[0]+'/'+key[1]:24s} {len(dr):4d} {np.mean(a['kwh']):7.2f} {np.mean(a['model']):6.2f} {dr.mean():8.2f} {np.mean(a['flat']):7.2f} {pr.mean():11.0%} {dr[pr].mean() if pr.any() else 0:17.2f} {np.percentile(dr,90):7.2f} {dr.max():7.2f}")
print("== whole confirmation campaign, both cases, all strata (probing arms)")
for arm, (n, kwh, dr, flat) in tot.items():
    print(f"{arm:18s} runs {int(n):4d}  kWh {kwh:8.0f}  DR-priced CNY {dr:8.0f}  flat-4 CNY {flat:8.0f}")
