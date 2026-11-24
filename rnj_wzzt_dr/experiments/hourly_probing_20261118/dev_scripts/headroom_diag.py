"""Does the pre-fix code deliver blocks beyond battery/EV headroom? Run the strong greedy (V = 200) on
ambiguous hard_short005 inputs with the 2026-11-23 committed-block fix on ("new") or off ("old").

    python headroom_diag.py old|new N
"""
import json
import sys

import numpy as np

sys.path.insert(0, r"C:/0-workspace/Topo/repo")
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, run
from rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments import ROOT, settings_for

mode, N = sys.argv[1], int(sys.argv[2])
if mode == "old":
    run.committed = lambda s, res, g, hb, blocks, spent, soc: (float(spent), np.array(soc, float))
cond = "hard_short005"
s = settings_for(cond).with_(cert_value_cny=200.)
inputs = json.loads((ROOT/cond/"inputs.json").read_text())
names = [r["name"] for r in inputs if "error" not in r and r.get("stratum") == "S1"][:N]
out = ROOT/f"headroom_diag_{mode}.jsonl"
with open(out, "w") as fh:
    for nm in names:
        inp = bank.load(ROOT/cond/"inputs"/f"{nm}.pkl")
        rec = run.run(inp, "bgreedy_V200_P", 0, s)
        fh.write(json.dumps(dict(input=nm, outcome=rec["outcome"], cost=rec["cost_cny"],
                                 probed=rec["probed_hours"], excess=rec["headroom_excess"])) + "\n")
        fh.flush()
print("done", mode, len(names))
