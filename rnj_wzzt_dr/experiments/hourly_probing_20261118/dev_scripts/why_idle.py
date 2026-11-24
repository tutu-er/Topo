import json, collections, sys
cond = sys.argv[1]
recs = [json.loads(l) for l in open(rf"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120\{cond}\runs.jsonl")]
c = collections.Counter(); act = []; beyond = collections.Counter(); emin = []
for r in recs:
    if r["arm"] != "t_opt_P" or r["stratum"] != "S1":
        continue
    for b in r["blocks"]:
        c[b.get("reason")] += 1
        act.append(len(b.get("active", [])))
        for x in b.get("beyond", []):
            beyond[x.get("type") if isinstance(x, dict) else "?"] += 1
    emin += [g["Emin"] for g in r["gates"] if not g["confirmed"] and g["Emin"] != float("inf")]
print("reasons", c.most_common())
print("active sizes", collections.Counter(act).most_common(6))
print("beyond types", beyond.most_common())
import numpy as np
print("Emin at unconfirmed gates: median", np.median(emin), "share < ln19", np.mean(np.array(emin) < 2.944))
print("example block keys", sorted(recs[0]["blocks"][0].keys()) if recs[0]["blocks"] else None)
