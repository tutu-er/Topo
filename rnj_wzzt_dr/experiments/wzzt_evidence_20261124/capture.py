"""Stage 0 of Phase 1: run WZZT (111) unchanged and record every W step and the E stage.

Hooks wrap algorithm._forward and algorithm._reconstruct; they record arguments and results and never change them.
Optional hooks (used only by experiment B2) edit the shared guidance or augment W's pool.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import pickle
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from rnj_wzzt_dr.identification import algorithm, trees
from rnj_wzzt_dr.identification.algorithm import SearchConfig
from rnj_wzzt_dr.physical.backend import Observation
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, experiments as hx

OUT = Path(__file__).resolve().parents[3]/"outputs"/"wzzt_evidence_20261124"
CONFIG = SearchConfig(total_seconds=900., per_mip_seconds=300., output_reserve_seconds=120.)


def select_inputs():
    """66 real pilot inputs; per synthetic condition the 8 lowest seeds of each network."""
    jobs = [("eulv", r["name"]) for r in json.loads((hx.ROOT/"eulv"/"inputs.json").read_text()) if "error" not in r]
    for cond in ("lock_short01", "lock_noisy"):
        rows = [r for r in json.loads((hx.ROOT/cond/"inputs.json").read_text()) if "error" not in r]
        for case in sorted({r["case"] for r in rows}):
            seeds = sorted(r["seed"] for r in rows if r["case"] == case)[:8]
            jobs += [(cond, r["name"]) for r in rows if r["case"] == case and r["seed"] in seeds]
    return jobs


def load_input(cond, name):
    return bank.load(hx.ROOT/cond/"inputs"/f"{name}.pkl")


def observations(inp, demean):
    t0, t1 = inp.meta["history"]

    def obs(s):
        arrays = [np.asarray(inp.hist[k][s], float) for k in ("P", "Q", "Y")]
        if demean:
            arrays = [a-a.mean(0) for a in arrays]
        return Observation(*arrays, tuple(inp.labels))
    return obs(slice(0, t0)), obs(slice(t0, t0+t1))


def truth_internal(inp):
    return {tuple(sorted(C)) for C in inp.truth_key if 1 < len(C) < inp.n}


def internal(supports, n):
    return {tuple(sorted(C)) for C in supports if 1 < len(C) < n}


def _slim(event):
    event = dict(event)
    solver = event.get("solver")
    if isinstance(solver, dict):
        event["solver"] = {k: v for k, v in solver.items() if k != "progress"}
    return event


def run(cond, name, mode, config=CONFIG, common_hook=None, pool_hook=None):
    """One identifier run; returns (record, result). mode is 'lv' (levels) or 'dm' (demeaned)."""
    inp = load_input(cond, name)
    tr, va = observations(inp, mode == "dm")
    W, E = [], []
    orig_forward, orig_reconstruct = algorithm._forward, algorithm._reconstruct

    def forward(current, fitter, solver, config, pool, deadline, *, reference_family=()):
        added = ()
        if pool_hook is not None:
            extra = pool_hook(current, fitter)
            added = tuple(C for C in extra if C not in set(pool))
            pool = trees.canonical([*pool, *added])
        started = time.perf_counter()
        result, event = orig_forward(current, fitter, solver, config, pool, deadline,
                                     reference_family=reference_family)
        W.append(dict(current=current, pool=tuple(pool), added=added, reference_family=tuple(reference_family),
                      event=_slim(event), seconds=time.perf_counter()-started))
        return result, event

    def reconstruct(current, fitter, solver, config, *, pool, keep, deadline, stage, reference_point=None):
        started = time.perf_counter()
        result, event = orig_reconstruct(current, fitter, solver, config, pool=pool, keep=keep, deadline=deadline,
                                         stage=stage, reference_point=reference_point)
        if stage == "E":
            E.append(dict(current=current, keep=tuple(keep), reference_point=reference_point, event=_slim(event),
                          seconds=time.perf_counter()-started))
        return result, event

    algorithm._forward, algorithm._reconstruct = forward, reconstruct
    started = time.perf_counter()
    try:
        common = algorithm.prepare_common(tr, config)
        if common_hook is not None:
            common = common_hook(common, inp)
        result = algorithm.fit(tr, va, "111", config=config, common=common)
    finally:
        algorithm._forward, algorithm._reconstruct = orig_forward, orig_reconstruct
    n = inp.n
    output = set()
    for c in result["frozen_candidates"]:
        output |= internal(c["point"]["supports"], n)
    for p in result.get("uncontracted_output_path", []):
        output |= internal(p["supports"], n)
    g = common["guidance"]
    record = dict(cond=cond, name=name, mode=mode, case=inp.case, stratum=getattr(inp, "stratum", None), n=n,
                  labels=list(inp.labels), train=(tr.p, tr.q, tr.y), validation=(va.p, va.q, va.y),
                  reference_mse=common["reference_mse"], alpha=config.alpha, config=dict(vars(config)),
                  guidance=dict(pool=tuple(g["pool"]), pool_families=tuple(g["pool_families"]),
                                reference_family=tuple(g["reference_family"])),
                  W=W, E=E, path=[trees.canonical(p["supports"]) for p in result["path"]],
                  selected=sorted(internal(result["selected"]["point"]["supports"], n)), output=sorted(output),
                  truth=sorted(truth_internal(inp)), stop_reason=result["stop_reason"],
                  seconds=time.perf_counter()-started, fit_seconds=result["total_seconds"])
    return record, result


def instance_path(cond, name, mode):
    return OUT/"instances"/f"{cond}__{name}__{mode}.pkl"


def capture_one(job):
    cond, name, mode = job
    path = instance_path(cond, name, mode)
    if path.exists():
        return dict(job=job, status="exists")
    try:
        record, _ = run(cond, name, mode)
    except Exception as exc:
        return dict(job=job, status="error", error=f"{type(exc).__name__}: {exc}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(record, fh)
    tmp.replace(path)
    return dict(job=job, status="ok", seconds=record["seconds"], W=len(record["W"]), E=len(record["E"]))


def observation(arrays, labels):
    return Observation(*[np.asarray(a, float) for a in arrays], tuple(labels))


def load_instance(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def truth_removed_hook(common, inp):
    """B2: drop every true support of size >= 3 from the RNJ pool, trees and reference family."""
    removed = {C for C in truth_internal(inp) if len(C) >= 3}
    common = dict(common)
    g = dict(common["guidance"])
    keep = lambda fam: trees.canonical(C for C in fam if tuple(sorted(C)) not in removed)
    g["pool"] = keep(g["pool"])
    g["pool_families"] = tuple(keep(f) for f in g["pool_families"])
    g["reference_family"] = keep(g["reference_family"])
    common["guidance"] = g
    common["truth_removed"] = sorted(removed)
    return common
