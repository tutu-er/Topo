"""V sweep of the strong greedy (DEVELOPMENT_RUNS section 17, rule registered before reading results).

    python -m rnj_wzzt_dr.experiments.hourly_probing_20261118.vsweep

Inputs: the ambiguous (S1) natural inputs of the three hard conditions of section 16, 2 paths each.
Passive comes from the section 16 runs (passive buys nothing, so the 2026-11-23 budget fix does not
touch it); the strong greedy at V = 50 ... 2000 CNY comes from the sweep (tag `vsweep`, fixed code).

Rule for the best V:
- primary: the largest net value per run, NV(V) = VALUE x (correct(V) - correct(passive)) / runs - mean CNY a day,
  with VALUE = 1,000 CNY for a correct identification (about one field visit avoided);
- secondary: the cheapest V whose correct confirmations are not significantly worse than the best V's
  (exact McNemar, p >= 0.05).
"""
from __future__ import annotations

import json

import numpy as np

from .confirm import _paired
from .experiments import ROOT
from .hard import CONDITIONS, _row
from .report import _mcnemar_p, load, pct

VS = (50, 100, 200, 500, 1000, 2000)
VINF = 100000          # V -> infinity, run after the registered sweep (tag vinf): reported, never chosen
VS_ALL = VS + (VINF,)
VALUE = 1000.


def _arm(v):
    return f"bgreedy_V{v}_P"


def _pair(a, b, good):
    """Pair by condition, input and path: the three conditions reuse the same input names."""
    tag = lambda rs: [dict(r, input=f"{r['_cond']}/{r['input']}") for r in rs]
    return _paired(tag(a), tag(b), good)


def main():
    correct = lambda r: r["outcome"] == "correct"
    pas, swp, old = [], {v: [] for v in VS_ALL}, {"bgreedy_lowV_P": [], "bgreedy_P": []}
    per_cond = {}
    for cond in CONDITIONS:
        runs = [dict(r, _cond=cond) for r in load(cond) if r["stratum"] == "S1" and not r["input"].endswith("_missing")]
        sw = [dict(r, _cond=cond) for r in load(cond, tag="vsweep")+load(cond, tag="vinf")
              if r["stratum"] == "S1" and not r["input"].endswith("_missing")]
        p = [r for r in runs if r["arm"] == "passive_bic"]
        pas += p
        for a in old:
            old[a] += [r for r in runs if r["arm"] == a]
        per_cond[cond] = {"passive": (sum(map(correct, p)), len(p))}
        for v in VS_ALL:
            rr = [r for r in sw if r["arm"] == _arm(v)]
            swp[v] += rr
            per_cond[cond][v] = (sum(map(correct, rr)), len(rr))
    if not pas or not any(swp.values()):
        raise SystemExit("no sweep results yet")
    base = sum(map(correct, pas))
    rows = {"passive": _row(pas)}
    for v in VS_ALL:
        if swp[v]:
            rows[v] = _row(swp[v])
    nv = {}
    for v in VS_ALL:
        if v not in rows:
            continue
        t = _pair(swp[v], pas, correct)
        n = rows[v]["n"]
        nv[v] = dict(net_value=VALUE*(rows[v]["correct"]-base)/n-rows[v]["cost"], vs_passive=t,
                     cny_per_extra=(rows[v]["cost"]*n/(rows[v]["correct"]-base) if rows[v]["correct"] > base else None))
    reg = [v for v in nv if v in VS]                  # the registered values; V -> infinity is only reported
    best_acc = max(reg, key=lambda v: (rows[v]["correct"], -rows[v]["cost"]))
    vs_best = {v: _pair(swp[v], swp[best_acc], correct) for v in nv if v != best_acc}
    ok = [v for v in reg if v == best_acc or not (vs_best[v]["loss"] > vs_best[v]["gain"] and vs_best[v]["p"] < .05)]
    secondary = min(ok, key=lambda v: rows[v]["cost"])
    primary = max(reg, key=lambda v: nv[v]["net_value"])
    fix = {}
    for a, v in (("bgreedy_lowV_P", 200), ("bgreedy_P", 1000)):
        if old[a] and swp[v]:
            fix[f"V{v}: fixed vs section 16"] = dict(**_pair(swp[v], old[a], correct),
                                                     cost_fixed=rows[v]["cost"], cost_old=_row(old[a])["cost"])

    print("## V sweep, pooled over the three hard conditions (S1 runs)")
    print("| Arm | Runs | Correct | Undecided | Ever wrong | Median h to correct | Probed h/day | kWh/day | CNY/day | "
          "CNY per probed day | Max CNY/day | vs passive | Net value per run (CNY) | CNY per extra correct |")
    print("|"+" --- |"*14)
    r = rows["passive"]
    print(f"| passive | {r['n']} | {pct(r['correct'], r['n'])} | {pct(r['undecided'], r['n'])} | {pct(r['ever_wrong'], r['n'])} | "
          f"{r['median_first']:.1f} | 0 | 0 | 0 | 0 | 0 | | 0 | |")
    for v in nv:
        r, t = rows[v], nv[v]["vs_passive"]
        cpe = "-" if nv[v]["cny_per_extra"] is None else f"{nv[v]['cny_per_extra']:.0f}"
        print(f"| V = {v} | {r['n']} | {pct(r['correct'], r['n'])} | {pct(r['undecided'], r['n'])} | {pct(r['ever_wrong'], r['n'])} | "
              f"{r['median_first']:.1f} | {r['probed']:.1f} | {r['kwh']:.1f} | {r['cost']:.1f} | {r['cost_probed_day']:.1f} | "
              f"{r['cost_max']:.0f} | {t['gain']} / {t['loss']} (p = {t['p']:.2g}) | {nv[v]['net_value']:.1f} | {cpe} |")
    print("\n## Correct confirmations by condition")
    shown = [v for v in VS_ALL if v in rows]
    print("| Condition | Passive | " + " | ".join(f"V = {v}" for v in shown) + " |")
    print("|"+" --- |"*(len(shown)+2))
    for cond, d in per_cond.items():
        print(f"| {cond} | {d['passive'][0]}/{d['passive'][1]} | " + " | ".join(f"{d[v][0]}/{d[v][1]}" for v in shown) + " |")
    print(f"\nmost accurate: V = {best_acc}")
    for v, t in vs_best.items():
        print(f"V = {v} vs V = {best_acc}: {t['gain']} / {t['loss']} of {t['pairs']}, p = {t['p']:.3g}")
    for k, t in fix.items():
        print(f"{k}: {t['gain']} / {t['loss']} of {t['pairs']}, p = {t['p']:.3g}; CNY/day {t['cost_fixed']:.1f} vs {t['cost_old']:.1f}")
    print(f"\nprimary rule (largest net value at {VALUE:.0f} CNY per correct identification): V = {primary}")
    print(f"secondary rule (cheapest not significantly worse than the most accurate): V = {secondary}")
    out = dict(rows={str(k): v for k, v in rows.items()}, net_value={str(k): v for k, v in nv.items()},
               per_condition={c: {str(k): v for k, v in d.items()} for c, d in per_cond.items()},
               best_accuracy=best_acc, vs_best={str(k): v for k, v in vs_best.items()}, fix_check=fix,
               primary=primary, secondary=secondary)
    (ROOT/"summary_vsweep.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    main()
