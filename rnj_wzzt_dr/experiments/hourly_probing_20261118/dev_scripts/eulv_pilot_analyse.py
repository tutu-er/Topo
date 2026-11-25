"""Analyse the real-feeder pilot (tag pilot)."""
import json, sys
from collections import Counter
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
import numpy as np
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank
from rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments import ROOT
from rnj_wzzt_dr.experiments.hourly_probing_20261118.report import _mcnemar_p

D = ROOT/"eulv"
rows = [r for r in json.loads((D/"inputs.json").read_text()) if "error" not in r]
meta = {}
for r in rows:
    inp = bank.load(D/"inputs"/f"{r['name']}.pkl")
    cl = inp.meta.get("closure") or {}
    meta[r["name"]] = dict(stratum=inp.meta["passive"]["stratum"], truth_in_bank=inp.truth_in_bank,
                           exact_in_bank=inp.meta.get("exact_truth_in_bank"), merged=inp.meta.get("sub_floor_segments_merged"),
                           closure_ok=cl.get("capped") is None and not cl.get("unexpanded_within_margin", 0), n=inp.n,
                           seconds=inp.meta.get("identifier_seconds"))
print(f"inputs {len(rows)}; strata {dict(Counter(m['stratum'] for m in meta.values()))}")
print(f"identifiable truth in bank {sum(m['truth_in_bank'] for m in meta.values())}, exact truth in bank "
      f"{sum(bool(m['exact_in_bank']) for m in meta.values())}, closure finished {sum(m['closure_ok'] for m in meta.values())}")
by_n = {}
for m in meta.values():
    k = "8-12" if m["n"] <= 12 else "13-18" if m["n"] <= 18 else "19-24"
    by_n.setdefault(k, []).append(m)
for k, ms in sorted(by_n.items()):
    print(f"  {k} meters: {len(ms)} inputs, truth in bank {sum(m['truth_in_bank'] for m in ms)}, "
          f"strata {dict(Counter(m['stratum'] for m in ms))}, closure finished {sum(m['closure_ok'] for m in ms)}")
recs = [json.loads(l) for l in open(D/"pilot.jsonl")] if (D/"pilot.jsonl").exists() else []
arms = sorted({r["arm"] for r in recs})
for a in arms:
    rs = [r for r in recs if r["arm"] == a]
    oc = Counter(r["outcome"] for r in rs)
    print(f"{a}: runs {len(rs)}, outcomes {dict(oc)}, ever wrong {sum(r['ever_confirmed_wrong'] for r in rs)}, "
          f"probed runs {sum(r['probed_hours'] > 0 for r in rs)}, CNY/run {np.mean([r['cost_cny'] for r in rs]):.1f}, "
          f"exact truth confirmed {sum((r.get('exact_truth') or {}).get('final_confirmed', False) for r in rs)}, "
          f"violations {sum(r['violations']['ac_band_hours'] for r in rs)}, headroom excess {sum(r['headroom_excess']['hours'] > 0 for r in rs)}")
if len(arms) == 2:
    A = {r["input"]: r for r in recs if r["arm"] == "bgreedy_P"}; P = {r["input"]: r for r in recs if r["arm"] == "passive_bic"}
    for label, f in (("correct", lambda r: r["outcome"] == "correct"), ("ever wrong", lambda r: r["ever_confirmed_wrong"])):
        for sub in ("all", "S1"):
            ks = [k for k in A if k in P and (sub == "all" or meta[k]["stratum"] == "S1")]
            g = sum(f(A[k]) and not f(P[k]) for k in ks); l = sum(f(P[k]) and not f(A[k]) for k in ks)
            print(f"  strong greedy vs passive, {label}, {sub}: {g} / {l} of {len(ks)}, p = {_mcnemar_p(g, l):.3g}")
    wrong = [(k, A[k]["wrong_vdev"], A[k]["ever_wrong_types"]) for k in A if A[k]["ever_confirmed_wrong"]]
    print("strong greedy wrong confirmations (input, voltage error, types):", wrong[:12])
