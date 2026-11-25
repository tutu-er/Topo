"""WZZT (stage 0 identifier) statistics stored in the pilot inputs: time used, selected-correct, truth in its own bank."""
import json, sys
from collections import Counter
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
import numpy as np
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank
from rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments import ROOT

for cond in ("eulv", "lock_short01", "lock_noisy"):
    D = ROOT/cond
    rows = [r for r in json.loads((D/"inputs.json").read_text()) if "error" not in r]
    if cond != "eulv":
        rows = rows[:66]
    ms = [bank.load(D/"inputs"/f"{r['name']}.pkl").meta for r in rows]
    sec = np.array([m.get("identifier_seconds") or np.nan for m in ms])
    print(f"{cond}: {len(ms)} inputs; identifier seconds (both runs) median {np.nanmedian(sec):.0f}, max {np.nanmax(sec):.0f}, "
          f">=1700 s in {int(np.sum(sec >= 1700))}; selected correct {sum(bool(m.get('identifier_selected_correct')) for m in ms)}, "
          f"truth in identifier output {sum(bool(m.get('identifier_bank_has_truth')) for m in ms)}, "
          f"identifier bank size median {np.median([m.get('identifier_bank') or 0 for m in ms]):.0f}, "
          f"closure capped {dict(Counter(str((m.get('closure') or {}).get('capped')) for m in ms))}")
