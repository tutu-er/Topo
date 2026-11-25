"""Phase 2 diagnostic: score the selected families and the truth with the hourly-probing study's own BIC.

For each input, the study's fit (offsets, GLS, loss term, resistance floor, identifiable-parameter count) on the
whole history gives J for: the truth, the selected family of WZZT-R and of the original 111 runs (O_lv, O_dm).
A method whose family has J <= J(truth) found something the study's judge rates at least as well as the truth:
its miss is then an identifiability limit of the data, not a search failure. Output: phase2/judge.jsonl.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
from multiprocessing import Pool

from rnj_wzzt_dr.experiments.hourly_probing_20261118 import model
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import capture
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase2 import OUT, judge, study_J


def one(args):
    cond, name, selected = args
    inp = capture.load_input(cond, name)
    n, j = inp.n, judge(cond, inp)
    full = lambda S: model.canonical_family([*map(tuple, S), tuple(range(n))], n)
    fams = {"truth": model.canonical_family(inp.truth_key, n)}
    for mode in ("lv", "dm"):
        fams["O_"+mode] = full(capture.load_instance(capture.instance_path(cond, name, mode))["selected"])
    for v, S in selected.items():
        fams[v] = full(S)
    # the whole-feeder support is a convention (absent from the synthetic truths, always in identifier families):
    # score each family with and without it and keep the better, so that internal structures are compared
    whole = tuple(range(n))
    keys = list(fams)
    variants = {k: [fams[k], model.canonical_family([C for C in fams[k] if C != whole], n),
                    model.canonical_family([*fams[k], whole], n)] for k in keys}
    flat = [(k, f) for k in keys for f in dict.fromkeys(variants[k])]
    fits = study_J([f for _, f in flat], inp, j)
    N = inp.hist["Y"].size
    extra = n if j["offsets"] else 0
    sigma2 = min(f.sse/(N-f.c.p-extra) for f in fits)
    J = {}
    for (k, _), f in zip(flat, fits):
        J[k] = min(J.get(k, math.inf), f.sse/(2*sigma2)+.5*(f.c.p+extra)*math.log(N))
    inner = lambda f: frozenset(C for C in f if 1 < len(C) < n)
    return dict(cond=cond, name=name, n=n, dJ={k: J[k]-J["truth"] for k in keys if k != "truth"},
                same_as_truth={k: inner(fams[k]) == inner(fams["truth"]) for k in keys})


def main():
    ladder = [json.loads(line) for line in (OUT/"ladder.jsonl").read_text().splitlines()]
    sel = {}
    for r in ladder:
        if "error" not in r and r["variant"] in ("R",):
            sel.setdefault((r["cond"], r["name"]), {})[r["variant"]] = r["selected"]
    jobs = [(c, nm, s) for (c, nm), s in sel.items()]
    with Pool(4) as pool, open(OUT/"judge.jsonl", "w") as fh:
        for out in pool.imap_unordered(one, jobs):
            fh.write(json.dumps(out)+"\n")
    rows = [json.loads(line) for line in (OUT/"judge.jsonl").read_text().splitlines()]
    for cond in ("eulv", "lock_short01", "lock_noisy"):
        rs = [r for r in rows if r["cond"] == cond]
        for v in ("O_lv", "O_dm", "R"):
            at_least = sum(r["dJ"][v] <= 1e-6 for r in rs)
            exact = sum(r["same_as_truth"][v] for r in rs)
            better = sum(r["dJ"][v] < -1e-6 and not r["same_as_truth"][v] for r in rs)
            print(f"{cond:13s} {v:5s} J <= J(truth): {at_least}/{len(rs)} (exact {exact}, a different family "
                  f"rated better than the truth {better})")


if __name__ == "__main__":
    main()
