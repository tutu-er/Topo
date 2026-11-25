"""Summary of eulv_explore.py: the strong greedy without the trust gate against passive (pilot), and the wrong
confirmations' voltage error against the identifiable truth's own error."""
import json, sys
from collections import Counter
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
import numpy as np
from rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments import ROOT
from rnj_wzzt_dr.experiments.hourly_probing_20261118.report import _mcnemar_p

D = ROOT/"eulv"
P = {r["input"]: r for r in map(json.loads, open(D/"pilot.jsonl")) if r["arm"] == "passive_bic"}
G = {r["input"]: r for r in map(json.loads, open(D/"pilot.jsonl")) if r["arm"] == "bgreedy_P"}
A = {r["input"]: r for r in map(json.loads, open(D/"explore_notrust.jsonl"))}
B = {r["input"]: r for r in map(json.loads, open(D/"explore_baseline.jsonl"))}
for label, R in (("passive", P), ("strong greedy, trust gate on", G), ("strong greedy, trust gate OFF", A)):
    rs = list(R.values())
    print(f"{label}: outcomes {dict(Counter(r['outcome'] for r in rs))}, ever wrong {sum(r['ever_confirmed_wrong'] for r in rs)}, "
          f"probed runs {sum(r['probed_hours'] > 0 for r in rs)}, probed hours {sum(r['probed_hours'] for r in rs)}, "
          f"CNY/run {np.mean([r['cost_cny'] for r in rs]):.1f}, violations {sum(r['violations']['ac_band_hours'] for r in rs)}, "
          f"headroom-excess runs {sum(r['headroom_excess']['hours'] > 0 for r in rs)}")
ks = sorted(set(A) & set(P))
for label, f in (("correct", lambda r: r["outcome"] == "correct"), ("ever wrong", lambda r: r["ever_confirmed_wrong"]),
                 ("first confirmation correct", lambda r: r["first_correct_gate"] is not None)):
    g = sum(f(A[k]) and not f(P[k]) for k in ks); l = sum(f(P[k]) and not f(A[k]) for k in ks)
    print(f"  trust gate off vs passive, {label}: {g} / {l} of {len(ks)}, p = {_mcnemar_p(g, l):.3g}")
probed = [k for k in ks if A[k]["probed_hours"] > 0]
print(f"runs that probed ({len(probed)}): outcome passive -> probing:",
      dict(Counter((P[k]["outcome"], A[k]["outcome"]) for k in probed)))
wv = [(P[k]["wrong_vdev"], B[k]["truth_fit_vdev"]) for k in ks if P[k]["ever_confirmed_wrong"]]
w, t = np.array(wv).T
print(f"passive wrong confirmations {len(wv)}: their voltage error {100*np.median(w):.3f}% median, {100*w.max():.3f}% max; "
      f"the identifiable truth's own error on the same inputs {100*np.median(t):.3f}% median, {100*t.max():.3f}% max; "
      f"wrong worse than truth by > 0.05 pt in {int(np.sum(w-t > 5e-4))}")
print(f"identifiable truth's own error, all inputs: median {100*np.median([b['truth_fit_vdev'] for b in B.values()]):.3f}%, "
      f"max {100*max(b['truth_fit_vdev'] for b in B.values()):.3f}%")
