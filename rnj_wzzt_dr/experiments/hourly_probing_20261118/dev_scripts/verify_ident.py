"""Would identifiable-parameter counting change anything on the synthetic inputs? Count, over every family of
every section 18 natural S1 input, the candidates whose identifiable count differs from 2 x supports."""
import glob, json, sys
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, model
from rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments import ROOT

tot = diff = inputs = 0
for cond in ("lock_short01", "lock_short005", "lock_noisy", "hard_short01", "hard_short005", "hard_noisy"):
    s1 = {r["name"] for r in json.loads((ROOT/cond/"inputs.json").read_text()) if r.get("stratum") == "S1"}
    for nm in sorted(s1):
        inp = bank.load(ROOT/cond/"inputs"/f"{nm}.pkl")
        inputs += 1
        for f in inp.families:
            c = model.Candidate(f, inp.n)
            tot += 1
            diff += model.identifiable_p(c, inp.hist["P"], inp.hist["Q"], .005) != c.p
print(f"{inputs} synthetic S1 inputs, {tot} candidates: identifiable count differs from 2 x supports in {diff}")
