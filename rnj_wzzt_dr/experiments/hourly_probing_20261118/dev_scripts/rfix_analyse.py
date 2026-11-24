"""Section 18 L2 with the fixed random arm (tag rfix) next to the old random arm (runs.jsonl)."""
import json, sys
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
from scipy.stats import binomtest
ROOT = r"C:/0-workspace/Topo/repo/outputs/hourly_probing_dev_20261120"
CONDS = ("lock_short01", "lock_short005", "lock_noisy")
BG = "bgreedy_V2000_P"


def load(cond, fname, arm):
    out = {}
    for l in open(f"{ROOT}/{cond}/{fname}"):
        r = json.loads(l)
        if r["arm"] == arm and r["stratum"] == "S1" and not r["input"].endswith("_missing"):
            key = (cond, r["input"], r["path"])
            assert key not in out, key
            out[key] = r
    return out


def excess(r):
    h = r.get("headroom_excess") or {}
    return float(h.get("kwh", 0.)), int(h.get("hours", 0))


arms = {"strong greedy V2000": {}, "random (old cap)": {}, "random (fixed)": {}, "passive": {}, "certified": {}}
for c in CONDS:
    arms["strong greedy V2000"].update(load(c, "runs.jsonl", BG))
    arms["random (old cap)"].update(load(c, "runs.jsonl", "random_matched_P"))
    arms["random (fixed)"].update(load(c, "rfix.jsonl", "random_matched_P"))
    arms["passive"].update(load(c, "runs.jsonl", "passive_bic"))
    arms["certified"].update(load(c, "runs.jsonl", "cert_P"))
keys = sorted(arms["random (fixed)"])
for a, d in arms.items():
    assert set(d) == set(keys), (a, len(d), len(keys))

print(f"example headroom_excess record: {arms['random (old cap)'][keys[0]]['headroom_excess']}")
print("| Arm | Runs | Correct | Ever wrong | CNY/day | Runs over headroom | Excess kWh (total) |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for a, d in arms.items():
    rs = [d[k] for k in keys]
    n = len(rs)
    cor = sum(r["outcome"] == "correct" for r in rs); ew = sum(r["ever_confirmed_wrong"] for r in rs)
    days = [max(1, len(r["gates"])//24) for r in rs]   # not used for CNY/day; cost is per run
    ex = [excess(r) for r in rs]
    print(f"| {a} | {n} | {cor} ({100*cor/n:.0f}%) | {ew} ({100*ew/n:.0f}%) | "
          f"{sum(r['cost_cny'] for r in rs)/n:.1f} per run | {sum(e[0] > 1e-6 for e in ex)} | {sum(e[0] for e in ex):.1f} |")


def mc(a, b, f, label):
    x = sum(f(arms[a][k]) and not f(arms[b][k]) for k in keys)
    y = sum(f(arms[b][k]) and not f(arms[a][k]) for k in keys)
    p = binomtest(x, x+y, .5).pvalue if x+y else 1.
    print(f"{label}: {a} vs {b}: {x} / {y}, p = {p:.3g}")


cor = lambda r: r["outcome"] == "correct"
ew = lambda r: r["ever_confirmed_wrong"]
mc("strong greedy V2000", "random (fixed)", cor, "L2 correct (fixed random)")
mc("strong greedy V2000", "random (old cap)", cor, "L2 correct (old random, as registered)")
mc("random (fixed)", "random (old cap)", cor, "random fixed vs old, correct")
mc("random (fixed)", "passive", cor, "random fixed vs passive, correct")
mc("random (fixed)", "random (old cap)", ew, "random fixed vs old, ever wrong")
for c in CONDS:
    ks = [k for k in keys if k[0] == c]
    x = sum(cor(arms["strong greedy V2000"][k]) and not cor(arms["random (fixed)"][k]) for k in ks)
    y = sum(cor(arms["random (fixed)"][k]) and not cor(arms["strong greedy V2000"][k]) for k in ks)
    print(f"  {c}: strong greedy vs fixed random {x} / {y} of {len(ks)}")
