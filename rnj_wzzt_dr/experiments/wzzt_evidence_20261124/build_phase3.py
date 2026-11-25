"""Phase 3 inputs on the real feeders: start days never used in WZZT development (1-5 and 7-11).

Each input is the hourly-probing study's `bank.build` for condition `eulv` with no identifier (step1 "v1",
use_identifier=False): the same simulated history, day-ahead data and identifiable truth as a full build, with a
placeholder bank (the truth and its one-edit neighbours) that Phase 3 does not use. Saved to
outputs/wzzt_evidence_20261124/phase3/inputs/eulv/.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import experiments as hx
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import capture

OUT3 = capture.OUT/"phase3"
DAYS = (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)


def build_one(case, day):
    path = OUT3/"inputs"/"eulv"/f"{case}_s{day}.pkl"
    if path.exists():
        return dict(name=path.stem, status="exists")
    try:
        inp = bank.build(case, day, hx.settings_for("eulv"), history=(144, 48), step1="v1", use_identifier=False)
    except Exception as exc:
        return dict(name=f"{case}_s{day}", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    bank.save(inp, tmp)
    tmp.replace(path)
    return dict(name=inp.name, status="built", n=inp.n, truth=len([C for C in inp.truth_key if 1 < len(C) < inp.n]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    cases = sorted({r["case"] for r in json.loads((hx.ROOT/"eulv"/"inputs.json").read_text()) if "error" not in r})
    jobs = [(c, d) for c in cases for d in DAYS]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(build_one, *j) for j in jobs]
        for f in as_completed(futures):
            rows.append(f.result())
            print(json.dumps({k: rows[-1].get(k) for k in ("name", "status", "n", "error")}), flush=True)
    (OUT3/"inputs_eulv.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
