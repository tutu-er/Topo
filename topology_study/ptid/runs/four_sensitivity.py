"""Paired history length, noise and sampling experiments on the two preselected feeders.

Run from topology_study:
python -B -m ptid.runs.four_sensitivity --workers 2
The nominal 8-day hourly and 1x-noise rows are reused from the primary passive experiment.
The 60-minute sampling arm explicitly averages the same noisy 15-minute raw channels.
"""
from __future__ import annotations

import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import gurobipy as gp
import numpy as np
import scipy

from ptid.method.identify import identify
from ptid.runs.four_feeders import history, load_feeder
from ptid.runs.four_offline import settings
from ptid.runs.four_records import cache_root
from ptid.runs.identify_cases import f1
from ptid.runs.passive_baselines import run_baseline

FEEDERS = ("soumalas11", "pengwah18")
VARIANTS = {
    "history_2d": {"days": 2}, "history_4d": {"days": 4},
    "noise_0p5": {"noise_scale": .5}, "noise_2": {"noise_scale": 2.},
    "sample_15m": {"fine": True, "per_hour": 4}, "sample_60m": {"fine": True, "per_hour": 1},
}
CHANNELS = ("P", "Q", "Y", "V0sq")
SETTINGS = settings()


def metrics(groups, truth):
    a, b = set(map(tuple, groups)), set(map(tuple, truth))
    return dict(**f1(a, b), fp=len(a-b), fn=len(b-a), exact_tree=a == b)


def run_condition(case, variant):
    rec = load_feeder(case)
    obs = history(rec, **VARIANTS[variant])
    data = {k: obs[k] for k in CHANNELS}
    nrows, n = data["P"].shape
    # Under offsets and no extra injection the symmetric R/X design has at most n(n+1) columns.
    # Allow one spatial direction to disappear in case the common-voltage weight rounds to 1.
    dof_lower_bound = (nrows-1)*(n-1) - n*(n+1)
    rows = [run_baseline(method, **data) for method in ("rg", "nj", "rnj", "park")]
    started = perf_counter()
    try:
        if dof_lower_bound <= 0:
            raise ValueError("positive residual degrees of freedom require an explicit design-rank check")
        result = identify(**data, st=SETTINGS)
        bounds = [("first", result["first"])]
        for i, step in enumerate(result["rounds"]):
            if step.get("solve"):
                bounds.append((f"anchor_{i}", step["solve"]))
            after = (step.get("price") or {}).get("re_solve")
            if after:
                bounds.append((f"re_solve_{i}", after))
        diag = dict(scope=result["scope"], stop=result["stop"], model=result["model"],
                    tree_solves={label: {k: v.get(k) for k in ("status", "crit", "bound", "gap_closed")}
                                 for label, v in bounds})
        row = dict(method="proposed", tree=result["tree"], status="ok", diagnostics=diag)
    except Exception as exc:
        row = dict(method="proposed", tree=None, status="failed", error=f"{type(exc).__name__}: {exc}")
    row["seconds"] = perf_counter()-started
    rows.append(row)
    for item in rows:
        if item["tree"] is not None:
            item["score"] = metrics(item["tree"], rec["truth"])
            item["score_all_physical"] = metrics(item["tree"], rec["exact_truth"])
    return dict(case=case, variant=variant, options=VARIANTS[variant], end_exclusive="2024-01-31T00:00:00+01:00",
                observations=nrows, meter_count=n, history_sha256={k: hashlib.sha256(
                    np.ascontiguousarray(a).tobytes()).hexdigest() for k, a in data.items()},
                saturated_dof_lower_bound=dof_lower_bound, source_hashes=rec["source_hashes"], estimates=rows)


def save(path, results, workers):
    path.parent.mkdir(exist_ok=True, parents=True)
    ordered = sorted(results, key=lambda e: (FEEDERS.index(e["case"]), list(VARIANTS).index(e["variant"])))
    result = dict(configuration=dict(workers=workers, expected_arms=60, settings=asdict(SETTINGS),
                                     reuse_primary=["history_8d", "noise_1"], common_channels=CHANNELS,
                                     versions=dict(python=platform.python_version(), numpy=np.__version__,
                                                   scipy=scipy.__version__, gurobi=gp.gurobi.version())),
                  conditions=ordered)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, help="Summary file (default: local cache/runs/sensitivity.json)")
    args = parser.parse_args()
    args.out = args.out or cache_root()/"runs"/"sensitivity.json"
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_condition, case, variant): (case, variant)
                   for case in FEEDERS for variant in VARIANTS}
        for work in as_completed(pending):
            case, variant = pending[work]
            try:
                record = work.result()
            except Exception as exc:
                record = dict(case=case, variant=variant, error=f"{type(exc).__name__}: {exc}", estimates=[])
            results.append(record)
            save(args.out, results, args.workers)
            print(case, variant, [(r["method"], r["status"]) for r in record["estimates"]],
                  record.get("error", ""), flush=True)
    good = sum(row["status"] == "ok" for cond in results for row in cond["estimates"])
    print(f"{good}/60 arms with usable estimates", flush=True)
    return int(good != 60)


if __name__ == "__main__":
    raise SystemExit(main())
