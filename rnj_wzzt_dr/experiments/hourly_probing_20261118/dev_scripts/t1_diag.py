"""Exploratory: why T1 failed in section 18 (truth-removed runs, strong greedy V=2000 vs passive)."""
import json, sys
from collections import Counter
import numpy as np
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
ROOT = r"C:/0-workspace/Topo/repo/outputs/hourly_probing_dev_20261120"
BG = "bgreedy_V2000_P"

def load(c, tag):
    return [json.loads(l) for l in open(f"{ROOT}/{c}/{tag}.jsonl")]

def first_wrong_gate(r):
    for g in r["gates"]:
        if g["confirmed"] and not g["leader_correct"]:
            return g["g"]
    return None

allp = []
for c in ("lock_short01", "lock_short005", "lock_noisy"):
    mis = [r for r in load(c, "missing") if r["input"].endswith("_missing")]
    by = {(r["input"], r["path"], r["arm"]): r for r in mis}
    keys = sorted({(r["input"], r["path"]) for r in mis})
    gain, loss, both, neither = [], [], [], []
    for k in keys:
        a, b = by.get(k+(BG,)), by.get(k+("passive_bic",))
        if not a or not b:
            continue
        wa, wb = a["ever_confirmed_wrong"], b["ever_confirmed_wrong"]
        (gain if wa and not wb else loss if wb and not wa else both if wa and wb else neither).append((a, b))
    print(f"\n== {c}: pairs {len(keys)}; probe-only wrong {len(gain)}, passive-only wrong {len(loss)}, both {len(both)}, neither {len(neither)}")
    for name, grp in (("probe-only wrong", gain), ("passive-only wrong", loss)):
        if not grp:
            continue
        w = [x[0] if name.startswith("probe") else x[1] for x in grp]
        o = [x[1] if name.startswith("probe") else x[0] for x in grp]
        vd = np.array([r["wrong_vdev"] or 0 for r in w])
        print(f"  {name}: wrong types {Counter(t for r in w for t in r['ever_wrong_types'])}; "
              f"wrong vdev median {100*np.median(vd):.3f}% max {100*vd.max():.3f}%; >0.2%: {(vd>2e-3).sum()}")
        print(f"    other arm's outcome: {Counter(r['outcome'] for r in o)}; final type of wrong arm: {Counter(r['final_type'] for r in w)}")
        print(f"    wrong arm probed hours: {Counter(r['probed_hours'] for r in w)}; first wrong gate {Counter(first_wrong_gate(r) for r in w)}")
    # both wrong: who first
    if both:
        d = [first_wrong_gate(a) - first_wrong_gate(b) for a, b in both]
        print(f"  both wrong: probe first by median {np.median(d):.1f} gates; earlier {sum(x<0 for x in d)}, same {sum(x==0 for x in d)}, later {sum(x>0 for x in d)}")
    # final state at end of day
    for arm in (BG, "passive_bic"):
        rs = [r for r in mis if r["arm"] == arm]
        print(f"  {arm}: outcomes {Counter(r['outcome'] for r in rs)}; final types {Counter(r['final_type'] for r in rs)}")
    allp += [(c, a, b) for a, b in gain+loss+both+neither]

# trust gate: adequacy of leader at gates where probing could run, natural vs truth-removed
print("\n== leader adequacy and Emin at gates (strong greedy arm)")
for c in ("lock_short01", "lock_short005", "lock_noisy"):
    nat = [r for r in load(c, "runs") if r["arm"] == BG and r["stratum"] == "S1"]
    mis = [r for r in load(c, "missing") if r["arm"] == BG and r["input"].endswith("_missing")]
    for nm, rs in (("natural", nat), ("missing", mis)):
        ad = np.array([g["adequacy"] for r in rs for g in r["gates"] if g.get("adequacy") is not None])
        em = np.array([g["Emin"] for r in rs for g in r["gates"] if g.get("Emin") is not None])
        q = lambda x: " ".join(f"{v:.3f}" for v in np.quantile(x, [.1, .5, .9, .99]))
        print(f"  {c} {nm}: gates {len(ad)} adequacy q10/50/90/99 {q(ad)}  max {ad.max():.3f}; Emin q {q(em)}")
