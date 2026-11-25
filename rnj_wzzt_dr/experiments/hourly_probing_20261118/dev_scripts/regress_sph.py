"""Regression after making the slot length a setting: rerun stored section 18 runs and compare."""
import json, sys
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, run
from rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments import ROOT, settings_for

cond = "lock_short01"
want = {}
for l in open(ROOT/cond/"runs.jsonl"):
    r = json.loads(l)
    if r["path"] == 0 and r["arm"] in ("bgreedy_V2000_P", "passive_bic", "cert_P"):
        want.setdefault(r["input"], {})[r["arm"]] = r
names = [n for n, d in want.items() if len(d) == 3][:3]
s = settings_for(cond)
bad = 0
for nm in names:
    inp = bank.load(ROOT/cond/"inputs"/f"{nm}.pkl")
    for arm in ("passive_bic", "bgreedy_V2000_P", "cert_P"):
        new, old = run.run(inp, arm, 0, s), want[nm][arm]
        keys = ("outcome", "cost_cny", "probed_hours", "first_correct_gate", "ever_confirmed_wrong")
        same = all(new[k] == old[k] for k in keys) and [g["leader"] for g in new["gates"]] == [g["leader"] for g in old["gates"]] \
            and new["schedule"]["blocks"] == old["schedule"]["blocks"]
        bad += not same
        print(nm, arm, "identical" if same else f"DIFFERENT {[(k, new[k], old[k]) for k in keys]}")
print("all identical" if not bad else f"{bad} differ")
