"""Hard-case analysis (DEVELOPMENT_RUNS section 16, pre-registered): ambiguous inputs, linear DR prices.

    python -m rnj_wzzt_dr.experiments.hourly_probing_20261118.hard
"""
from __future__ import annotations

import json

import numpy as np

from .confirm import _paired
from .experiments import ROOT
from .report import _mcnemar_p, load, pct

CONDITIONS = ("hard_short01", "hard_short005", "hard_noisy")
ARMS = ("passive_bic", "bgreedy_P", "bgreedy_lowV_P", "qbc_P", "cert_P", "random_matched_P")
TESTS = (("HC1_bgreedy_vs_passive", "bgreedy_P", "passive_bic"),
         ("HC1b_qbc_vs_passive", "qbc_P", "passive_bic"),
         ("HC2_bgreedy_vs_random", "bgreedy_P", "random_matched_P"),
         ("bgreedy_vs_qbc", "bgreedy_P", "qbc_P"),
         ("bgreedy_vs_cert", "bgreedy_P", "cert_P"),
         ("lowV_vs_bgreedy", "bgreedy_lowV_P", "bgreedy_P"),
         ("lowV_vs_passive", "bgreedy_lowV_P", "passive_bic"),
         ("qbc_vs_random", "qbc_P", "random_matched_P"),
         ("cert_vs_passive", "cert_P", "passive_bic"))


def _kwh(r):
    return float(sum(np.abs(np.asarray(b["u_kw"], float)).sum() for b in r["blocks"]))


def _row(rs):
    first = [r["first_correct_gate"] for r in rs if r["first_correct_gate"] is not None]
    wv = [r["wrong_vdev"] for r in rs if r["ever_confirmed_wrong"] and r.get("wrong_vdev") is not None]
    cost = np.array([r["cost_cny"] for r in rs]) if rs else np.zeros(1)
    probed = np.array([r["probed_hours"] > 0 for r in rs]) if rs else np.zeros(1, bool)
    return dict(n=len(rs), correct=sum(r["outcome"] == "correct" for r in rs),
                undecided=sum(r["outcome"] not in ("correct", "wrong") for r in rs),
                ever_wrong=sum(r["ever_confirmed_wrong"] for r in rs),
                wrong_vdev_max=max(wv) if wv else None, wrong_over_02=sum(v > 2e-3 for v in wv),
                median_first=float(np.median(first)) if first else None,
                probed=float(np.mean([r["probed_hours"] for r in rs])) if rs else 0.,
                kwh=float(np.mean([_kwh(r) for r in rs])) if rs else 0.,
                cost=float(cost.mean()), cost_probed_day=float(cost[probed].mean()) if probed.any() else 0.,
                cost_max=float(cost.max()),
                violations=sum(r["violations"]["ac_band_hours"]+r["violations"]["deadband_hours"] for r in rs))


def main():
    out, pooled_rows, pooled = {}, {a: [] for a in ARMS}, {}
    correct = lambda r: r["outcome"] == "correct"
    for cond in CONDITIONS:
        try:
            rr = [r for r in load(cond) if r["stratum"] == "S1" and not r["input"].endswith("_missing")]
        except FileNotFoundError:
            continue
        if not rr:
            continue
        inputs = json.loads((ROOT/cond/"inputs.json").read_text())
        ok = [r for r in inputs if "error" not in r]
        by = {a: [r for r in rr if r["arm"] == a] for a in ARMS}
        for a in ARMS:
            pooled_rows[a] += by[a]
        rows = {a: _row(by[a]) for a in ARMS if by[a]}
        tests = {k: _paired(by[a], by[b], correct) for k, a, b in TESTS if by[a] and by[b]}
        for k, v in tests.items():
            pooled.setdefault(k, [0, 0, 0]); pooled[k][0] += v["gain"]; pooled[k][1] += v["loss"]; pooled[k][2] += v["pairs"]
        out[cond] = dict(inputs=len(ok), s1=sum(r.get("stratum") == "S1" for r in ok),
                         truth_in_bank=sum(bool(r.get("truth_in_bank")) for r in ok),
                         s1_truth_in_bank=sum(bool(r.get("truth_in_bank")) for r in ok if r.get("stratum") == "S1"),
                         rows=rows, tests=tests)
        _print(cond, out[cond])
    out["pooled"] = dict(rows={a: _row(v) for a, v in pooled_rows.items() if v},
                         tests={k: dict(gain=g, loss=l, pairs=n, p=_mcnemar_p(g, l)) for k, (g, l, n) in pooled.items()})
    _print("pooled", out["pooled"])
    (ROOT/"summary_hard.json").write_text(json.dumps(out, indent=1))
    return out


def _print(name, d):
    if "inputs" in d:
        print(f"\n## {name}: {d['inputs']} inputs, {d['s1']} ambiguous (S1); truth in bank {d['truth_in_bank']} "
              f"(S1: {d['s1_truth_in_bank']})")
    else:
        print(f"\n## {name}")
    print("| Arm | Runs | Correct | Undecided | Ever wrong | Max wrong V | Median h to correct | Probed h/day | kWh/day | "
          "CNY/day | CNY per probed day | Max CNY/day | Violations |")
    print("|"+" --- |"*13)
    for a, r in d["rows"].items():
        mf = "-" if r["median_first"] is None else f"{r['median_first']:.0f}"
        mv = "-" if r["wrong_vdev_max"] is None else f"{100*r['wrong_vdev_max']:.2f}%"
        print(f"| {a} | {r['n']} | {pct(r['correct'], r['n'])} | {pct(r['undecided'], r['n'])} | {pct(r['ever_wrong'], r['n'])} | "
              f"{mv} | {mf} | {r['probed']:.1f} | {r['kwh']:.1f} | {r['cost']:.1f} | {r['cost_probed_day']:.1f} | "
              f"{r['cost_max']:.0f} | {r['violations']} |")
    for k, v in d["tests"].items():
        print(f"{k}: {v['gain']} / {v['loss']} of {v['pairs']} pairs, p = {v['p']:.4g}")


if __name__ == "__main__":
    main()
