"""Closure status vs truth-missing: (1) prevalence of natural misses among probe-eligible inputs,
(2) section 18 T1 split by whether the parent input's closure completed."""
import json, pathlib, sys
from collections import Counter
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank
from rnj_wzzt_dr.experiments.hourly_probing_20261118.confirm import _paired
from rnj_wzzt_dr.experiments.hourly_probing_20261118.report import _mcnemar_p, load
R = pathlib.Path(r"C:/0-workspace/Topo/repo/outputs/hourly_probing_dev_20261120")
CONDS = ("hard_short01", "hard_short005", "hard_noisy", "lock_short01", "lock_short005", "lock_noisy")


def closure_ok(c, name):
    x = bank.load(R/c/"inputs"/f"{name}.pkl")
    cl = x.meta.get("closure") or {}
    return cl.get("capped") is None and not cl.get("unexpanded_within_margin", 0)


ok = {}
print("| Condition | S1 closure ok | S1 capped | S3 closure ok | S3 capped |")
print("| --- | ---: | ---: | ---: | ---: |")
tot = Counter()
for c in CONDS:
    inp = [r for r in json.loads((R/c/"inputs.json").read_text()) if "error" not in r]
    cnt = Counter()
    for r in inp:
        if r.get("stratum") in ("S1", "S3"):
            nm = r.get("name") or r.get("input")
            ok[(c, nm)] = closure_ok(c, nm)
            cnt[(r["stratum"], ok[(c, nm)])] += 1
    tot.update(cnt)
    print(f"| {c} | {cnt[('S1', True)]} | {cnt[('S1', False)]} | {cnt[('S3', True)]} | {cnt[('S3', False)]} |")
print(f"| all | {tot[('S1', True)]} | {tot[('S1', False)]} | {tot[('S3', True)]} | {tot[('S3', False)]} |")
elig = tot[("S1", True)]+tot[("S3", True)]
print(f"natural misses among closure-complete ambiguous-or-missing inputs: {tot[('S3', True)]}/{elig} = {100*tot[('S3', True)]/elig:.1f}%")
print(f"natural misses caught by the closure flag: {tot[('S3', False)]}/{tot[('S3', True)]+tot[('S3', False)]}")
ms = Counter()
for c in ("hard_noisy", "lock_noisy"):
    for (cc, nm), v in ok.items():
        pass
print()

# section 18 T1 split by closure status of the parent input
wrong = lambda r: bool(r["ever_confirmed_wrong"])
agg = {True: [0, 0, 0], False: [0, 0, 0]}
rows = {True: Counter(), False: Counter()}
for c in ("lock_short01", "lock_short005", "lock_noisy"):
    mis = [r for r in load(c, tag="missing") if r["input"].endswith("_missing")]
    for flag in (True, False):
        sel = [r for r in mis if ok[(c, r["input"][:-len("_missing")])] == flag]
        a = [r for r in sel if r["arm"] == "bgreedy_V2000_P"]; b = [r for r in sel if r["arm"] == "passive_bic"]
        if not a:
            continue
        t = _paired(a, b, wrong)
        agg[flag][0] += t["gain"]; agg[flag][1] += t["loss"]; agg[flag][2] += t["pairs"]
        rows[flag]["runs"] += len(a); rows[flag]["bg_wrong"] += sum(map(wrong, a)); rows[flag]["pas_wrong"] += sum(map(wrong, b))
        rows[flag]["bg_probed"] += sum(r["probed_hours"] > 0 for r in a)
for flag in (True, False):
    g, l, n = agg[flag]
    x = rows[flag]
    print(f"T1, parent closure {'complete' if flag else 'capped'}: runs {x['runs']}, ever wrong strong greedy {x['bg_wrong']} vs passive "
          f"{x['pas_wrong']}, McNemar {g} / {l} of {n}, p = {_mcnemar_p(g, l):.3g}, days probed {x['bg_probed']}")
