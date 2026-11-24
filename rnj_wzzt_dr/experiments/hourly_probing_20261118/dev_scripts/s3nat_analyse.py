"""DEVELOPMENT_RUNS section 19: natural truth-missing (S3) inputs, passive vs strong greedy V = 2,000 (tag s3nat)."""
import sys
from collections import Counter
import numpy as np
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
from rnj_wzzt_dr.experiments.hourly_probing_20261118.confirm import _paired
from rnj_wzzt_dr.experiments.hourly_probing_20261118.hard import _row
from rnj_wzzt_dr.experiments.hourly_probing_20261118.report import _mcnemar_p, load

CONDS = ("hard_short01", "hard_noisy", "lock_short01", "lock_short005", "lock_noisy")
ARMS = ("passive_bic", "bgreedy_V2000_P")
wrong = lambda r: bool(r["ever_confirmed_wrong"])
wrong02 = lambda r: bool(r["ever_confirmed_wrong"]) and (r.get("wrong_vdev") or 0.) > 2e-3
final_wrong = lambda r: r["outcome"] == "wrong"

pool = {a: [] for a in ARMS}
tot = {"N1": [0, 0, 0], "wrong>0.2%": [0, 0, 0], "final wrong": [0, 0, 0]}
print("| Condition | Inputs | Arm | Runs | Outcomes | Ever wrong | Final wrong | Wrong > 0.2% | Max wrong V | Days probed | Probed h/day | CNY/day |")
print("|" + " --- |"*12)
for c in CONDS:
    rs = [dict(r, _c=c) for r in load(c, tag="s3nat") if r["stratum"] == "S3"]
    if not rs:
        print(f"| {c} | no runs |"); continue
    assert not any(r["truth_in_bank"] for r in rs), c
    by = {a: [r for r in rs if r["arm"] == a] for a in ARMS}
    for a in ARMS:
        pool[a] += by[a]
        x = _row(by[a]); n = x["n"]
        mv = "-" if x["wrong_vdev_max"] is None else f"{100*x['wrong_vdev_max']:.2f}%"
        oc = dict(Counter(r["outcome"] for r in by[a]))
        share = np.mean([r["probed_hours"] > 0 for r in by[a]])
        print(f"| {c} | {len({r['input'] for r in by[a]})} | {a} | {n} | {oc} | {x['ever_wrong']} ({100*x['ever_wrong']/n:.0f}%) | "
              f"{sum(map(final_wrong, by[a]))} | {x['wrong_over_02']} | {mv} | {100*share:.0f}% | {x['probed']:.1f} | {x['cost']:.1f} |")
    for k, f in (("N1", wrong), ("wrong>0.2%", wrong02), ("final wrong", final_wrong)):
        t = _paired(by[ARMS[1]], by[ARMS[0]], f)
        print(f"   {c} {k}: strong greedy / passive {t['gain']} / {t['loss']} of {t['pairs']}, p = {t['p']:.3g}")
        tot[k][0] += t["gain"]; tot[k][1] += t["loss"]; tot[k][2] += t["pairs"]

print("\n## pooled")
for a in ARMS:
    x = _row(pool[a]); n = x["n"]
    if not n:
        continue
    wv = [r["wrong_vdev"] for r in pool[a] if r["ever_confirmed_wrong"] and r.get("wrong_vdev") is not None]
    fv = [r["final_vdev"] for r in pool[a] if r["outcome"] == "wrong" and r.get("final_vdev") is not None]
    share = np.mean([r["probed_hours"] > 0 for r in pool[a]])
    mvp = "-" if x["wrong_vdev_max"] is None else f"{100*x['wrong_vdev_max']:.2f}%"
    print(f"{a}: runs {n}, ever wrong {x['ever_wrong']} ({100*x['ever_wrong']/n:.0f}%), final wrong {sum(map(final_wrong, pool[a]))}, "
          f"undecided {x['undecided']}, wrong>0.2% {x['wrong_over_02']}, max wrong V "
          f"{mvp}, "
          f"median wrong V {100*np.median(wv) if wv else float('nan'):.3f}%, median final-wrong V {100*np.median(fv) if fv else float('nan'):.3f}%, "
          f"days probed {100*share:.0f}%, probed h/day {x['probed']:.2f}, CNY/day {x['cost']:.1f}, CNY/probed day {x['cost_probed_day']:.1f}, "
          f"max {x['cost_max']:.0f}, violations {x['violations']}")
    print(f"   wrong types: {dict(Counter(t for r in pool[a] for t in (r.get('ever_wrong_types') or [])))}")
    print(f"   headroom excess runs: {sum((r.get('headroom_excess') or {}).get('kwh', 0.) > 1e-6 for r in pool[a])}")
for k, (g, l, n) in tot.items():
    print(f"{k} pooled: strong greedy / passive {g} / {l} of {n}, p = {_mcnemar_p(g, l):.3g}")

# runs only the strong greedy confirmed wrong: what did passive end with, and how far off were they?
A = {(r["_c"], r["input"], r["path"]): r for r in pool[ARMS[1]]}
P = {(r["_c"], r["input"], r["path"]): r for r in pool[ARMS[0]]}
only = [k for k in A if k in P and wrong(A[k]) and not wrong(P[k])]
print(f"\nruns only the strong greedy confirmed wrong: {len(only)}; passive outcome there: {dict(Counter(P[k]['outcome'] for k in only))}; "
      f"their wrong V: {sorted(round(100*(A[k]['wrong_vdev'] or 0), 3) for k in only)}")
print(f"types there: {dict(Counter(t for k in only for t in (A[k].get('ever_wrong_types') or [])))}")
