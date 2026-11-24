"""Exploratory: what do probing's gains (truth present) and harms (truth removed) look like? (section 18)"""
import json
from collections import Counter
import numpy as np
ROOT = r"C:/0-workspace/Topo/repo/outputs/hourly_probing_dev_20261120"
BG = "bgreedy_V2000_P"
CONDS = ("lock_short01", "lock_short005", "lock_noisy")

def load(c, tag):
    return [json.loads(l) for l in open(f"{ROOT}/{c}/{tag}.jsonl")]

def pairs(rs):
    by = {(r["input"], r["path"], r["arm"]): r for r in rs}
    for (i, p, a), r in by.items():
        if a == BG and (i, p, "passive_bic") in by:
            yield r, by[(i, p, "passive_bic")]

G_types, G_vd, H_types, H_vd = Counter(), [], Counter(), []
for c in CONDS:
    nat = [r for r in load(c, "runs") if r["stratum"] == "S1" and r["arm"] in (BG, "passive_bic")]
    mis = [r for r in load(c, "missing") if r["input"].endswith("_missing") and r["arm"] in (BG, "passive_bic")]
    gt, gv = Counter(), []
    for a, b in pairs(nat):
        if a["outcome"] == "correct" and b["outcome"] != "correct":
            gt[(b["outcome"], b["final_type"])] += 1
            if b.get("final_vdev") is not None:
                gv.append(b["final_vdev"])
    ht, hv = Counter(), []
    for a, b in pairs(mis):
        if a["ever_confirmed_wrong"] and not b["ever_confirmed_wrong"]:
            ht[a["final_type"]] += 1; hv.append(a["wrong_vdev"])
    gv, hv = np.array(gv), np.array(hv)
    print(f"\n== {c}")
    print(f"  L1 gains {sum(gt.values())}: passive ended as {dict(gt)}")
    print(f"    passive's final leader vs AC truth: median {100*np.median(gv):.3f}%, q90 {100*np.quantile(gv,.9):.3f}%, max {100*gv.max():.3f}%; "
          f"> 0.1%: {(gv>1e-3).sum()}, > 0.2%: {(gv>2e-3).sum()}")
    print(f"  T1 harms {len(hv)}: probing's wrong leader types {dict(ht)}; vdev median {100*np.median(hv):.3f}%, max {100*hv.max():.3f}%")
    G_types += Counter(t for (_, t) in gt.elements()); G_vd += list(gv); H_types += ht; H_vd += list(hv)
G_vd, H_vd = np.array(G_vd), np.array(H_vd)
print(f"\n== pooled: gains {len(G_vd)} (passive's final leader type {dict(G_types)}), passive leader vdev median {100*np.median(G_vd):.3f}% max {100*G_vd.max():.3f}%, >0.1% {(G_vd>1e-3).sum()}")
print(f"   harms {len(H_vd)} (types {dict(H_types)}), vdev median {100*np.median(H_vd):.3f}% max {100*H_vd.max():.3f}%, >0.1% {(H_vd>1e-3).sum()}")
