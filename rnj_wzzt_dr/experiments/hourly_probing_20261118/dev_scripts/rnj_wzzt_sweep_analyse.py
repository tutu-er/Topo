"""Summary of rnj_wzzt_sweep.py: RNJ pool/tree recall and WZZT variants, real vs synthetic."""
import json, sys
from collections import defaultdict
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
import numpy as np
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank
from rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments import ROOT

R = [json.loads(l) for l in open(ROOT/"eulv"/"rnj_wzzt_sweep.jsonl")]
crit = {r["case"]: r for r in json.loads((ROOT/"eulv"/"wzzt_criterion.json").read_text())}
for cond in ("eulv", "lock_short01"):
    rs = [r for r in R if r["cond"] == cond]
    T = sum(r["truth"] for r in rs)
    print(f"\n== {cond}: {len(rs)} inputs, {T} true internal segments")
    for label in ("rnj_pq", "rnj_p"):
        hit = sum(r[label]["pool"]["hit"] for r in rs); size = sum(r[label]["pool"]["size"] for r in rs)
        print(f"  RNJ pool ({label}): recall {hit}/{T} ({100*hit/T:.0f}%), precision {hit}/{size} ({100*hit/max(size,1):.0f}%)")
        keys = rs[0][label]["per"].keys()
        best = []
        for k in keys:
            h = sum(r[label]["per"][k]["hit"] for r in rs); s = sum(r[label]["per"][k]["size"] for r in rs)
            best.append((k, h, s))
        print("    per tree (weight/tolerance: recall, precision):",
              "; ".join(f"{k}: {100*h/T:.0f}%, {100*h/max(s,1):.0f}%" for k, h, s in best))
    for c in ("current", "floor0", "alpha001", "alpha001_floor0", "ponly_floor0"):
        ok = [r for r in rs if c in r and "error" not in r[c]]
        if not ok:
            continue
        Tc = sum(r["truth"] for r in ok)
        sh = sum(r[c]["selected"]["hit"] for r in ok); ss = sum(r[c]["selected"]["size"] for r in ok)
        oh = sum(r[c]["output"]["hit"] for r in ok); os_ = sum(r[c]["output"]["size"] for r in ok)
        ex = sum(r[c]["exact"] for r in ok)
        print(f"  WZZT {c:16s}: selected recall {100*sh/Tc:.0f}% precision {100*sh/max(ss,1):.0f}%; "
              f"output recall {100*oh/Tc:.0f}% precision {100*oh/max(os_,1):.0f}%; exact {ex}/{len(ok)}; "
              f"seconds median {np.median([r[c]['seconds'] for r in ok]):.0f}, errors {len(rs)-len(ok)}")
    if cond == "eulv":
        # wzzt_criterion.json lists every non-singleton truth segment (including the full set) in canonical order;
        # count only the internal ones (1 < |C| < n), as above.
        sup = 0
        for r in rs:
            inp = bank.load(ROOT/"eulv"/"inputs"/f"{r['input']}.pkl")
            segs = [C for C in inp.truth_key if len(C) > 1]
            sup += sum(s for C, s in zip(segs, crit[r["case"]]["bic_8"]) if len(C) < inp.n)
        print(f"  (study BIC supports {sup} of these segments; resolution ceiling {100*sup/T:.0f}%)")
