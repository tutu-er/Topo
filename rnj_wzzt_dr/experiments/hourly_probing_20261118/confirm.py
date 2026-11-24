"""Confirmation analysis on fresh seeds 17-32 (DEVELOPMENT_RUNS section 15, pre-registered).

    python -m rnj_wzzt_dr.experiments.hourly_probing_20261118.confirm
"""
from __future__ import annotations

import json

import numpy as np

from .experiments import ROOT
from .report import _mcnemar_p, load, pct

CASES = (("main", "conf_base_main", "conf_main"), ("short01", "conf_base_short01", "conf_short01"))
ARMS = ("passive_bic", "qbc_P", "bgreedy_P", "cert_P", "random_matched_P")


def _row(rs):
    first = [r["first_correct_gate"] for r in rs if r["first_correct_gate"] is not None]
    wv = [r["wrong_vdev"] for r in rs if r["ever_confirmed_wrong"] and r.get("wrong_vdev") is not None]
    return dict(n=len(rs), correct=sum(r["outcome"] == "correct" for r in rs),
                final_leader=sum(r["final_leader_correct"] for r in rs),
                ever_wrong=sum(r["ever_confirmed_wrong"] for r in rs),
                wrong_vdev_max=max(wv) if wv else None, wrong_over_02=sum(v > 2e-3 for v in wv),
                median_first=float(np.median(first)) if first else None,
                probed=float(np.mean([r["probed_hours"] for r in rs])) if rs else 0.,
                cost=float(np.mean([r["cost_cny"] for r in rs])) if rs else 0.,
                violations=sum(r["violations"]["ac_band_hours"]+r["violations"]["deadband_hours"] for r in rs))


def _paired(a_recs, b_recs, good):
    A = {(r["input"], r["path"]): r for r in a_recs}
    B = {(r["input"], r["path"]): r for r in b_recs}
    if len(A) != len(a_recs) or len(B) != len(b_recs):
        raise ValueError("duplicate (input, path) keys: pair within one condition")
    keys = sorted(set(A) & set(B))
    g = sum(good(A[k]) and not good(B[k]) for k in keys)
    l = sum(good(B[k]) and not good(A[k]) for k in keys)
    return dict(pairs=len(keys), gain=g, loss=l, p=_mcnemar_p(g, l))


def main():
    out = {}
    correct = lambda r: r["outcome"] == "correct"
    pooled = {}
    for case, base, ref in CASES:
        b = [r for r in load(base) if r["arm"] == "passive_bic" and not r["input"].endswith("_missing")]
        rr = [r for r in load(ref) if not r["input"].endswith("_missing")]
        if not b or not rr:
            continue
        rows = {"baseline passive": _row(b)}
        for arm in ARMS:
            rows[f"refined {arm}"] = _row([r for r in rr if r["arm"] == arm])
        inputs = [json.loads(l) for l in [(ROOT/base/"inputs.json").read_text()]][0]
        valsel = sum(r.get("validation_mse_correct", False) for r in inputs if "error" not in r)
        tests = dict(
            H1_refined_vs_baseline=_paired([r for r in rr if r["arm"] == "passive_bic"], b, correct),
            H2_qbc_vs_passive=_paired([r for r in rr if r["arm"] == "qbc_P"], [r for r in rr if r["arm"] == "passive_bic"], correct),
            bgreedy_vs_qbc=_paired([r for r in rr if r["arm"] == "bgreedy_P"], [r for r in rr if r["arm"] == "qbc_P"], correct),
            cert_vs_qbc=_paired([r for r in rr if r["arm"] == "cert_P"], [r for r in rr if r["arm"] == "qbc_P"], correct),
            qbc_vs_random=_paired([r for r in rr if r["arm"] == "qbc_P"], [r for r in rr if r["arm"] == "random_matched_P"], correct))
        for k, v in tests.items():
            pooled.setdefault(k, [0, 0]); pooled[k][0] += v["gain"]; pooled[k][1] += v["loss"]
        out[case] = dict(rows=rows, tests=tests, identifier_valsel_correct=valsel, inputs=len(inputs))
        print(f"\n## {case}: {len(inputs)} fresh inputs; repository identifier's own pick correct in {valsel}")
        print("| Pipeline / arm | Runs | Correct by end of day | Final leader correct | Ever wrong | Wrong > 0.2% V | Max wrong V dev | Median h to correct | Probed h/day | CNY/day |")
        print("|"+" --- |"*10)
        for name, r in rows.items():
            mf = "-" if r["median_first"] is None else f"{r['median_first']:.0f}"
            mv = "-" if r["wrong_vdev_max"] is None else f"{100*r['wrong_vdev_max']:.2f}%"
            print(f"| {name} | {r['n']} | {pct(r['correct'], r['n'])} | {pct(r['final_leader'], r['n'])} | {pct(r['ever_wrong'], r['n'])} | "
                  f"{r['wrong_over_02']} | {mv} | {mf} | {r['probed']:.1f} | {r['cost']:.2f} |")
        for k, v in tests.items():
            print(f"{k}: {v['gain']} / {v['loss']} of {v['pairs']} pairs, p = {v['p']:.4f}")
    out["pooled"] = {k: dict(gain=g, loss=l, p=_mcnemar_p(g, l)) for k, (g, l) in pooled.items()}
    print("\npooled:", {k: (v["gain"], v["loss"], round(v["p"], 4)) for k, v in out["pooled"].items()})
    (ROOT/"summary_confirm.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
