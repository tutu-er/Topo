"""Exploratory: can any logged gate statistic tell a truth-missing bank from a natural one? (section 18 data)"""
import json, sys
from collections import Counter
import numpy as np
ROOT = r"C:/0-workspace/Topo/repo/outputs/hourly_probing_dev_20261120"
BG = "bgreedy_V2000_P"
CONDS = ("lock_short01", "lock_short005", "lock_noisy")

def load(c, tag):
    return [json.loads(l) for l in open(f"{ROOT}/{c}/{tag}.jsonl")]

def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort()+1.
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

def run_feats(r):
    gs = r["gates"]
    lead = [g["leader"] for g in gs]
    switches = sum(a != b for a, b in zip(lead, lead[1:]))
    conf = [(g["g"], g["leader"]) for g in gs if g["confirmed"]]
    conf_leaders = []
    for _, L in conf:
        if not conf_leaders or conf_leaders[-1] != L:
            conf_leaders.append(L)
    flips = len(conf_leaders)-1 if conf_leaders else 0
    probe_gates = {b["sealed_at"] for b in r["blocks"] if any(abs(x) > 1e-9 for x in b["u_kw"])}
    pg = [g for g in gs if g["g"] in probe_gates]
    return dict(switches=switches, flips=flips, n_conf_leaders=len(set(conf_leaders)), pg=pg)

for arm in (BG, "passive_bic"):
    print(f"\n######## {arm}")
    for c in CONDS:
        nat = [r for r in load(c, "runs") if r["arm"] == arm and r["stratum"] == "S1"]
        mis = [r for r in load(c, "missing") if r["arm"] == arm and r["input"].endswith("_missing")]
        fn, fm = [run_feats(r) for r in nat], [run_feats(r) for r in mis]
        print(f"\n== {c}: natural {len(nat)}, missing {len(mis)}")
        for k in ("switches", "flips", "n_conf_leaders"):
            a, b = [f[k] for f in fm], [f[k] for f in fn]
            print(f"  {k}: missing mean {np.mean(a):.2f} (>0 in {np.mean(np.array(a)>0):.0%}), natural mean {np.mean(b):.2f} "
                  f"(>0 in {np.mean(np.array(b)>0):.0%}); AUC {auc(a, b):.2f}")
        # natural runs: flips among runs that end correct
        cor = [f["flips"] for r, f in zip(nat, fn) if r["outcome"] == "correct"]
        print(f"  natural correct runs with a flip before ending correct: {sum(x>0 for x in cor)} of {len(cor)}")
        if arm == BG:
            for k in ("adequacy", "Emin", "mass", "n_equivalent", "n_rivals"):
                a = [g[k] for f in fm for g in f["pg"] if np.isfinite(g[k])]
                b = [g[k] for f in fn for g in f["pg"] if np.isfinite(g[k])]
                print(f"  at probing gates, {k}: missing median {np.median(a):.3g}, natural median {np.median(b):.3g}, AUC {auc(a, b):.2f} "
                      f"({len(a)} / {len(b)} gates)")
        # operational accuracy of confirmations: vdev at the final gate
        vc = [r["final_vdev"] for r in nat if r["outcome"] == "correct" and r.get("final_vdev") is not None]
        vw = [r["final_vdev"] for r in mis if r["outcome"] == "wrong" and r.get("final_vdev") is not None]
        if vc and vw:
            print(f"  final voltage error: correct natural median {100*np.median(vc):.3f}% (q90 {100*np.quantile(vc,.9):.3f}%), "
                  f"wrong truth-removed median {100*np.median(vw):.3f}% (q90 {100*np.quantile(vw,.9):.3f}%, max {100*max(vw):.3f}%)")
