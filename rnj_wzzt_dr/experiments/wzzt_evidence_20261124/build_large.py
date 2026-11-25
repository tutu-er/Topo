"""Experiment M inputs (PROTOCOL_MIQP.md): whole transformer networks, large feeders, and 16-day histories.

    python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.build_large check     # converter regression
    python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.build_large convert   # new cases (not the 33)
    python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.build_large build --workers 4

- convert: every EU LV transformer whose whole LV network (all feeders under its LV bus, normally-open fuses
  open) has 26-80 metered connections (case eulv_<tx>_all), and every single feeder with 26-60 connections.
  Written next to the 33 study feeders in cases_eulv/ with their own index files (index_transformers.json,
  index_large_feeders.json); the study's index.json is not touched.
- build: `bank.build` for condition `eulv` with no identifier and no placeholder bank (experiment M uses only the history and the truth): the large networks on start days
  0 and 6 with 8 days of history, and the 33 study feeders on start day 0 with 16 days (14 + 2).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import filecmp
import json
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, eulv_convert
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import experiments as hx
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import scenario as sc
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import capture

RAW = Path(__file__).resolve().parents[3]/"data"/"nonsynthetic_eu_lv_20261123"/"extracted"/"Sim_files_190128_OK_V0"/"RunDss"
OUTM = capture.OUT/"miqp"
SETS = {"eulv_large": dict(index=("index_transformers.json", "index_large_feeders.json"), days=(0, 6),
                           history=(144, 48)),
        "eulv_16d": dict(index=("index.json",), days=(0,), history=(336, 48))}


def check():
    """The refactored converter reproduces the 33 study feeders file for file."""
    with tempfile.TemporaryDirectory() as tmp:
        cases, loads = Path(tmp)/"cases", Path(tmp)/"loads"
        idx = eulv_convert.convert(RAW, cases, loads)
        assert len(idx) == 33, len(idx)
        for row in idx:
            for f in ("buses.csv", "branches.csv", "metadata.json"):
                assert filecmp.cmp(cases/row["case"]/f, sc.EULV_CASES/row["case"]/f, shallow=False), (row["case"], f)
            with np.load(loads/f"{row['case']}.npz") as a, np.load(sc.EULV_LOADS/f"{row['case']}.npz") as b:
                assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files), row["case"]
        assert json.loads((cases/"index.json").read_text()) == json.loads((sc.EULV_CASES/"index.json").read_text())
    print("converter regression: 33 feeders identical")


def convert():
    before = (sc.EULV_CASES/"index.json").read_bytes()
    t = eulv_convert.convert(RAW, sc.EULV_CASES, sc.EULV_LOADS, 26, 80, whole_transformer=True,
                             index_name="index_transformers.json")
    f = eulv_convert.convert(RAW, sc.EULV_CASES, sc.EULV_LOADS, 26, 60, index_name="index_large_feeders.json")
    assert (sc.EULV_CASES/"index.json").read_bytes() == before
    print(f"{len(t)} transformer networks, {len(f)} large feeders")


def jobs():
    out = []
    for name, spec in SETS.items():
        cases = []
        for idx in spec["index"]:
            cases += [r["case"] for r in json.loads((sc.EULV_CASES/idx).read_text())]
        out += [(name, c, d, spec["history"]) for c in cases for d in spec["days"]]
    return out


def build_one(job):
    name, case, day, history = job
    path = OUTM/"inputs"/name/f"{case}_s{day}.pkl"
    if path.exists():
        return dict(name=path.stem, set=name, status="exists")
    try:
        inp = bank.build(case, day, hx.settings_for("eulv"), history=tuple(history), step1="v1",
                         use_identifier=False, extra_neighbours=False)
    except Exception as exc:
        return dict(name=f"{case}_s{day}", set=name, error=f"{type(exc).__name__}: {exc}",
                    trace=traceback.format_exc())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    bank.save(inp, tmp)
    tmp.replace(path)
    return dict(name=inp.name, set=name, status="built", n=inp.n,
                truth=len([C for C in inp.truth_key if 1 < len(C) < inp.n]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("check", "convert", "build"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None, help="build only this set")
    args = ap.parse_args()
    if args.stage == "check":
        check()
    elif args.stage == "convert":
        convert()
    else:
        todo = [j for j in jobs() if args.only in (None, j[0])]
        rows = []
        with ProcessPoolExecutor(args.workers) as pool:
            for f in as_completed([pool.submit(build_one, j) for j in todo]):
                rows.append(f.result())
                print(json.dumps({k: rows[-1].get(k) for k in ("set", "name", "status", "n", "truth", "error")}),
                      flush=True)
        OUTM.mkdir(parents=True, exist_ok=True)
        (OUTM/"inputs.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
