import json, collections, numpy as np
base = r"C:/0-workspace/Topo/repo/outputs/hourly_probing_dev_20261120/"
for cond in ["conf_main", "conf_short01", "cmp_main", "cmp_short01", "dev_short01"]:
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(base+cond+"/runs.jsonl") as f:
        for line in f:
            r = json.loads(line)
            key = (r["stratum"] if r["stratum"]=="S3" else "S1S2", r["arm"])
            e = 0.; nodeh = 0; peak = 0.; up = 0.; dn = 0.
            for b in r["blocks"]:
                u = np.asarray(b["u_kw"], float)
                e += np.abs(u).sum(); nodeh += int((np.abs(u) > 1e-6).sum()); peak = max(peak, np.abs(u).max(initial=0.))
                up += u[u > 0].sum(); dn += -u[u < 0].sum()
            a = agg[key]
            a["e"].append(e); a["nodeh"].append(nodeh); a["peak"].append(peak); a["cost"].append(r["cost_cny"])
            a["hours"].append(r["probed_hours"]); a["dn"].append(dn); a["up"].append(up)
    print("==", cond)
    print(f"{'stratum/arm':28s} {'runs':>5s} {'h/day':>6s} {'kWh/day':>8s} {'p90kWh':>7s} {'maxkWh':>7s} {'kWh/h':>6s} {'peakkW':>7s} {'model CNY':>9s} {'dn share':>8s}")
    for key in sorted(agg):
        a = agg[key]; e = np.array(a["e"]); h = np.array(a["hours"])
        if h.sum() == 0: continue
        print(f"{key[0]+'/'+key[1]:28s} {len(e):5d} {h.mean():6.2f} {e.mean():8.2f} {np.percentile(e,90):7.2f} {e.max():7.2f} {e.sum()/max(h.sum(),1):6.2f} {np.max(a['peak']):7.2f} {np.mean(a['cost']):9.2f} {np.sum(a['dn'])/max(e.sum(),1e-9):8.2f}")
