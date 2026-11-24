"""Exploratory: does the prequential check separate truth-missing banks from natural ones? (tag preq)"""
import json
from collections import defaultdict
import numpy as np
ROOT = r"C:/0-workspace/Topo/repo/outputs/hourly_probing_dev_20261120"
CONDS = ("lock_short01", "lock_short005", "lock_noisy")
BG = "bgreedy_V2000_P"


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean(); eq = (pos[:, None] == neg[None, :]).mean()
    return gt+.5*eq


def stats(g):
    lp = g["pre_L_probe"]/g["pre_L_probe_df"] if g["pre_L_probe_df"] else None
    lq = g["pre_L_pass"]/g["pre_L_pass_df"] if g["pre_L_pass_df"] else None
    # z-score of the leader's probed-hour sum against chi2(df), and the probe-minus-passive contrast
    z = (g["pre_L_probe"]-g["pre_L_probe_df"]*(lq or 1.))/np.sqrt(2*g["pre_L_probe_df"]) if g["pre_L_probe_df"] else None
    return dict(L_probe=lp, contrast=None if lp is None or lq is None else lp-lq, z=z,
                bank_min=g.get("pre_min_probe"), eq_min=g.get("pre_eq_min_probe"))


def first_conf(r):
    return next((g for g in r["gates"] if g["confirmed"]), None)


recs = defaultdict(list)
for c in CONDS:
    try:
        for l in open(f"{ROOT}/{c}/preq.jsonl"):
            r = json.loads(l); recs[c].append(r)
    except FileNotFoundError:
        pass

keys = ("L_probe", "contrast", "z", "bank_min", "eq_min")
print("AUC, truth-removed vs natural (strong greedy): > 0.5 means larger when the truth is missing")
for c in list(CONDS)+["pooled"]:
    rs = [r for cc in (CONDS if c == "pooled" else (c,)) for r in recs[cc] if r["arm"] == BG]
    if not rs:
        continue
    for when in ("first confirmation", "final gate", "probed gates"):
        mis, nat = defaultdict(list), defaultdict(list)
        for r in rs:
            dst = mis if r["input"].endswith("_missing") else nat
            if when == "first confirmation":
                gs = [first_conf(r)]
            elif when == "final gate":
                gs = [r["gates"][-1]]
            else:
                ph = {b["hour"]+1 for b in r["blocks"] if any(abs(x) > 1e-9 for x in b["u_kw"])}   # data of hour h arrive at gate h+1
                gs = [g for g in r["gates"] if g["g"] in ph]
            for g in gs:
                if g is None:
                    continue
                for k, v in stats(g).items():
                    if v is not None and np.isfinite(v):
                        dst[k].append(v)
        line = ", ".join(f"{k} {auc(mis[k], nat[k]):.2f} ({len(mis[k])}/{len(nat[k])})" for k in keys)
        print(f"  {c:13s} {when:18s}: {line}")

# calibration under the true model: the truth's own probe ratio in natural runs (final gate)
for c in CONDS:
    tp = [r["gates"][-1].get("pre_truth_probe") for r in recs[c] if r["arm"] == BG and not r["input"].endswith("_missing")]
    tq = [r["gates"][-1].get("pre_truth_pass") for r in recs[c] if not r["input"].endswith("_missing")]
    tp = np.array([x for x in tp if x is not None]); tq = np.array([x for x in tq if x is not None])
    if len(tp):
        print(f"{c}: truth's probed-hour ratio median {np.median(tp):.3f} q95 {np.quantile(tp,.95):.3f}; "
              f"passive-hour ratio median {np.median(tq):.3f} q95 {np.quantile(tq,.95):.3f}")

# offline veto: block a confirmation when the leader's contrast exceeds t (approximate: the path is not re-simulated)
print("\nOffline veto on the strong greedy (confirmations with contrast > t vetoed); natural correct kept / truth-removed ever-wrong left")
for t in (.05, .1, .15, .2, .3, .5):
    for c in list(CONDS)+["pooled"]:
        rs = [r for cc in (CONDS if c == "pooled" else (c,)) for r in recs[cc] if r["arm"] == BG]
        if not rs:
            continue
        ok_nat = n_nat = wr_mis = n_mis = 0
        for r in rs:
            def passes(g):
                s = stats(g)["contrast"]
                return s is None or s <= t
            if r["input"].endswith("_missing"):
                n_mis += 1
                wr_mis += any(g["confirmed"] and not g["leader_correct"] and passes(g) for g in r["gates"])
            else:
                n_nat += 1
                fg = r["gates"][-1]
                ok_nat += bool(fg["confirmed"] and fg["leader_correct"] and passes(fg))
        if c == "pooled":
            print(f"  t={t:.2f} pooled: natural correct {ok_nat}/{n_nat}, truth-removed ever wrong {wr_mis}/{n_mis}")
# reference: passive and the strong greedy without any veto
for arm in ("passive_bic", BG):
    rs = [r for c in CONDS for r in recs[c] if r["arm"] == arm]
    nat = [r for r in rs if not r["input"].endswith("_missing")]; mis = [r for r in rs if r["input"].endswith("_missing")]
    print(f"  {arm}: natural correct {sum(r['outcome'] == 'correct' for r in nat)}/{len(nat)}, "
          f"truth-removed ever wrong {sum(r['ever_confirmed_wrong'] for r in mis)}/{len(mis)}")
