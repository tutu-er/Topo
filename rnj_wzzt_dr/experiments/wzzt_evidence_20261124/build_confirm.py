"""Fresh synthetic inputs for the WZZT-C confirmation (PROTOCOL_WZZT_C.md).

Same four feeders, settings and history as the Phase 3 synthetic conditions (`lock_short01`: one 0.01 ohm segment;
`lock_noisy`: meter stress, 0.1% bias and noise), on seeds 161-184, which no study has used (development 1-96,
probing Stage M reserve 129-160). Data only: the probing bank is not built (`use_identifier=False`).

    python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.build_confirm [--workers 4]
Writes outputs/wzzt_evidence_20261124/confirm_c/inputs/<condition>/<case>_s<seed>.pkl and index.json.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import traceback
from multiprocessing import get_context

from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import experiments as hx
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import capture

OUTC = capture.OUT/"confirm_c"
CASES = ("soumalas11", "pengwah18", "flynn16", "paper15")
SEEDS = range(161, 185)
CONDITIONS = {"conf_short01": "lock_short01", "conf_noisy": "lock_noisy"}   # new name -> settings it copies


def build_one(job):
    cond, case, seed = job
    path = OUTC/"inputs"/cond/f"{case}_s{seed}.pkl"
    if path.exists():
        return dict(cond=cond, name=path.stem, status="exists")
    try:
        src = hx.CONDITIONS[CONDITIONS[cond]]
        inp = bank.build(case, seed, hx.settings_for(CONDITIONS[cond]), history=(96, 48),
                         short_line=src.get("short_line"), step1="v1", use_identifier=False, extra_neighbours=False)
    except Exception as exc:
        return dict(cond=cond, name=f"{case}_s{seed}", error=f"{type(exc).__name__}: {exc}",
                    trace=traceback.format_exc()[-1500:])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    bank.save(inp, tmp)
    tmp.replace(path)
    return dict(cond=cond, name=inp.name, status="built", n=inp.n,
                truth=len([C for C in inp.truth_key if 1 < len(C) < inp.n]))


def load(cond, name):
    return bank.load(OUTC/"inputs"/cond/f"{name}.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    jobs = [(c, case, s) for c in CONDITIONS for case in CASES for s in SEEDS]
    OUTC.mkdir(parents=True, exist_ok=True)
    rows = []
    with get_context("spawn").Pool(args.workers) as pool:
        for rec in pool.imap_unordered(build_one, jobs):
            rows.append(rec)
            if "error" in rec:
                print(rec["name"], rec["error"], flush=True)
    rows.sort(key=lambda r: (r["cond"], r["name"]))
    (OUTC/"index.json").write_text(json.dumps(rows, indent=1))
    print(f"{sum('error' not in r for r in rows)} of {len(rows)} inputs ready")


if __name__ == "__main__":
    main()
