import json
from collections import Counter
D = r"C:/0-workspace/Topo/repo/outputs/hourly_probing_dev_20261120/eulv/pilot.jsonl"
R = [json.loads(l) for l in open(D)]
rs = [r for r in R if r["arm"] == "passive_bic" and r["ever_confirmed_wrong"]]
c = Counter()
for r in rs:
    g = next(g for g in r["gates"] if g["confirmed"] and not g["leader_correct"])
    c[(r["stratum"], g.get("truth_eq"), round(g.get("truth_E", -1), 1) if "truth_E" in g else None)] += 1
print("passive wrong confirmations by (stratum, truth equivalent to leader, truth E):", dict(c))
print("wrong vdev %", sorted(round(100*r["wrong_vdev"], 3) for r in rs if r["wrong_vdev"] is not None))
ok = [r for r in R if r["arm"] == "passive_bic" and r["outcome"] == "correct"]
print("correct", [(r["input"], r["stratum"], round(100*r["final_vdev"], 3)) for r in ok])
und = [r for r in R if r["arm"] == "passive_bic" and r["outcome"] == "undecided"]
print("undecided by stratum", dict(Counter(r["stratum"] for r in und)))
