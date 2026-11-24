import json, numpy as np, collections
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import coverage as cv
rows = [json.loads(l) for l in open(cv.OUT/"coverage.jsonl") if l.strip()]
rows = [r for r in rows if "error" not in r and (r["history"] == "1d" or r["seed"] <= 4)]
rows = list({r["name"]: r for r in rows}.values())
print("inputs", len(rows), "by case", collections.Counter(r["case"] for r in rows))
miss = [r for r in rows if not r["D"]["cover"]]
for r in miss:
    print("D miss", r["name"], "capped", r["D"]["capped"], "size", r["D"]["size"], "unexpanded", r["D"]["unexpanded_within_margin"], "nmc", r.get("neighbour_max_children"))
for lab in ("D", "D0"):
    s = [r[lab]["seconds"] for r in rows]; z = [r[lab]["size"] for r in rows]
    print(lab, "sec med/max", round(np.median(s), 1), round(max(s), 1), "size min/max", min(z), max(z), "capped", collections.Counter(r[lab]["capped"] for r in rows))
for case in ("soumalas11", "pengwah18", "flynn16", "paper15"):
    g = [r for r in rows if r["case"] == case]
    print(case, len(g), "A", sum(r["cover"]["A"] for r in g), "D", sum(r["D"]["cover"] for r in g), "leadD", sum(r["D"]["leader_correct"] for r in g))
ids = [sum(r["id_seconds"]) for r in rows]
print("identifier seconds (two runs) median/max", round(np.median(ids)), round(max(ids)))
