"""Confirmation of the lock on fresh seeds 65-96 (DEVELOPMENT_RUNS section 18, pre-registered).

    python -m rnj_wzzt_dr.experiments.hourly_probing_20261118.lock [V]

V defaults to the primary choice of the V sweep (summary_vsweep.json, section 17).
- Natural ambiguous (S1) inputs, tag `runs`: passive, strong greedy at V, certified, random (matched to
  the certified planner's hours and spend), committee. Hypotheses L1-L4.
- Truth-removed copies of the same inputs, tag `missing`: passive and strong greedy at V. Hypotheses T1-T3.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .confirm import _paired
from .experiments import ROOT
from .hard import _row
from .report import _mcnemar_p, load, pct

CONDITIONS = ("lock_short01", "lock_short005", "lock_noisy")


def _v():
    if len(sys.argv) > 1:
        return int(sys.argv[1])
    return int(json.loads((ROOT/"summary_vsweep.json").read_text())["primary"])


def _table(title, by):
    print(f"\n### {title}")
    print("| Arm | Runs | Correct | Undecided | Ever wrong | Wrong > 0.2% | Max wrong V | Median h to correct | "
          "Days probed | Probed h/day | CNY/day | CNY per probed day | Max CNY/day | Violations |")
    print("|"+" --- |"*14)
    for a, rs in by.items():
        if not rs:
            continue
        r = _row(rs)
        mv = "-" if r["wrong_vdev_max"] is None else f"{100*r['wrong_vdev_max']:.2f}%"
        mf = "-" if r["median_first"] is None else f"{r['median_first']:.1f}"
        share = np.mean([x["probed_hours"] > 0 for x in rs])
        print(f"| {a} | {r['n']} | {pct(r['correct'], r['n'])} | {pct(r['undecided'], r['n'])} | {pct(r['ever_wrong'], r['n'])} | "
              f"{r['wrong_over_02']} | {mv} | {mf} | {100*share:.0f}% | {r['probed']:.1f} | {r['cost']:.1f} | "
              f"{r['cost_probed_day']:.1f} | {r['cost_max']:.0f} | {r['violations']} |")


def main():
    V = _v()
    bg = f"bgreedy_V{V}_P"
    vc = "bgreedy_V100_P"                 # section 18 addendum: V = 100 on the same inputs, tag vcheck
    arms = ("passive_bic", bg, "cert_P", "random_matched_P", "qbc_P") + ((vc,) if vc != bg else ())
    correct = lambda r: r["outcome"] == "correct"
    wrong = lambda r: bool(r["ever_confirmed_wrong"])
    wrong02 = lambda r: bool(r["ever_confirmed_wrong"]) and (r.get("wrong_vdev") or 0.) > 2e-3
    tests = (("L1", bg, "passive_bic", correct), ("L2", bg, "random_matched_P", correct),
             ("L3a", bg, "cert_P", correct), ("L3b", bg, "qbc_P", correct),
             ("L4_ever_wrong", bg, "passive_bic", wrong), ("Vcheck_V_vs_V100", bg, vc, correct),
             ("Vcheck_V100_vs_passive", vc, "passive_bic", correct))
    mtests = (("T1_ever_wrong", bg, "passive_bic", wrong), ("T2_wrong_over_0.2%", bg, "passive_bic", wrong02))
    out, pool, pool_m = {"V": V}, {a: [] for a in arms}, {a: [] for a in arms[:2]}
    pooled = {}
    for cond in CONDITIONS:
        nat = [r for r in load(cond)+load(cond, tag="vcheck")
               if r["stratum"] == "S1" and not r["input"].endswith("_missing")]
        mis = [r for r in load(cond, tag="missing") if r["input"].endswith("_missing")]
        if not nat and not mis:
            continue
        by = {a: [r for r in nat if r["arm"] == a] for a in arms}
        bym = {a: [r for r in mis if r["arm"] == a] for a in arms[:2]}
        for a in arms:
            pool[a] += by[a]
        for a in arms[:2]:
            pool_m[a] += bym[a]
        inputs = json.loads((ROOT/cond/"inputs.json").read_text())
        ok = [r for r in inputs if "error" not in r]
        print(f"\n## {cond}: {len(ok)} inputs, {sum(r.get('stratum') == 'S1' for r in ok)} ambiguous (S1), "
              f"truth in bank in {sum(bool(r.get('truth_in_bank')) for r in ok if r.get('stratum') == 'S1')} of them")
        _table("natural S1", by)
        _table("truth removed", bym)
        res = {}
        for k, a, b, f in tests:
            if a != b and by.get(a) and by.get(b):
                res[k] = _paired(by[a], by[b], f)
        for k, a, b, f in mtests:
            if bym[a] and bym[b]:
                res[k] = _paired(bym[a], bym[b], f)
        for k, v in res.items():
            print(f"{k}: {v['gain']} / {v['loss']} of {v['pairs']}, p = {v['p']:.3g}")
            pooled.setdefault(k, [0, 0, 0]); pooled[k][0] += v["gain"]; pooled[k][1] += v["loss"]; pooled[k][2] += v["pairs"]
        out[cond] = dict(inputs=len(ok), tests=res,
                         rows={a: _row(v) for a, v in by.items() if v}, rows_missing={a: _row(v) for a, v in bym.items() if v})
    print("\n## pooled")
    _table("natural S1", pool)
    _table("truth removed", pool_m)
    out["pooled"] = dict(rows={a: _row(v) for a, v in pool.items() if v}, rows_missing={a: _row(v) for a, v in pool_m.items() if v},
                         tests={k: dict(gain=g, loss=l, pairs=n, p=_mcnemar_p(g, l)) for k, (g, l, n) in pooled.items()})
    for k, v in out["pooled"]["tests"].items():
        print(f"{k}: {v['gain']} / {v['loss']} of {v['pairs']}, p = {v['p']:.3g}")
    # net value per run at 1,000 CNY per correct identification (the section 17 rule), on fresh seeds
    base = sum(map(correct, pool["passive_bic"]))
    nv = {a: 1000.*(sum(map(correct, rs))-base)/len(rs)-_row(rs)["cost"]
          for a, rs in pool.items() if rs and a != "passive_bic" and len(rs) == len(pool["passive_bic"])}
    for a, x in nv.items():
        print(f"net value per run, {a}: {x:.1f} CNY")
    out["pooled"]["net_value"] = nv
    (ROOT/"summary_lock.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
