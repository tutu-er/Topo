"""Build the case library (DESIGN.md).

    python -m rnj_wzzt_dr.experiments.case_library_20261125.build --stream dev  [--workers 6]
    python -m rnj_wzzt_dr.experiments.case_library_20261125.build --stream test [--workers 6]

One file per case: the network, the true quarter-hourly loads and AC voltages of 9 days, the unmetered load (or none)
and the truth. Meter readings are made when a case is opened (`to_input`). Files go to
outputs/case_library_20261125/<stream>/ (not in Git); their SHA-256 values go to manifest_<stream>.json next to this
file, so a held-out case cannot be changed unnoticed.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import pickle
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from rnj_wzzt_dr.experiments.hourly_probing_20261118 import model
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import scenario as sc
from rnj_wzzt_dr.experiments.hourly_probing_20261118.run import Input
from rnj_wzzt_dr.physical import backend

from . import feeders, loads, physics
from .conductors import CONDUCTORS

HERE = Path(__file__).resolve().parent
OUT = loads.REPO/"outputs"/"case_library_20261125"
DAYS = 9                                        # 8 days of history and one day for probing
R_FLOOR = .003                                  # ohm; shorter lines are merged in the identifiable truth
DROP_RANGE = (.010, .07)                        # accepted largest voltage drop below the PCC (random feeders)
SOURCES = ("none", "head", "middle", "end")     # where the unmetered load sits
SHARES = (.02, .05, .10)                        # its mean power as a share of the metered mean power
KINDS = ("lamp", "house", "block")


def key_of(*parts):
    """A stable integer per text or number, for random seeds."""
    return [p if isinstance(p, int) else zlib.crc32(str(p).encode()) for p in parts]


# ------------------------------------------------ structure of a feeder
def structure(network):
    """Per line (named by its lower bus): parent, meters below, resistance; and the root-to-bus resistance."""
    order, parent, incoming = backend._radial_order(network)
    labels = feeders.labels_of(network)
    below = {b: set() for b in order}
    for i, label in enumerate(labels):
        below[label].add(i)
    for b in reversed(order[1:]):
        below[parent[b]] |= below[b]
    depth = {order[0]: 0.}
    for b in order[1:]:
        depth[b] = depth[parent[b]]+incoming[b]["r_ohm"]
    return order, parent, incoming, below, depth


def source_bus(network, where):
    """Street node of the unmetered load: just after the trunk, half way along the street, or at its far end."""
    order, _, _, below, depth = structure(network)
    nodes = [b for b in order[1:] if len(below[b]) >= 2]
    far = max(depth[b] for b in nodes)
    target = {"head": 0., "middle": .5*far, "end": far}[where]
    return min(nodes, key=lambda b: (abs(depth[b]-target), b))


def source_path(network, bus):
    """The true lines from the PCC to the unmetered load, as sets of meters below each line."""
    _, parent, _, below, _ = structure(network)
    lines = []
    while parent[bus] is not None:
        lines.append(tuple(sorted(below[bus]))); bus = parent[bus]
    return lines[::-1]


# ------------------------------------------------ one case
def simulate(plan, stream, load_seed, source=None):
    """True states of one case. source: None or dict(where, kind, share)."""
    network = feeders.to_network(plan)
    labels = feeders.labels_of(network)
    counts = [b["customers"] for b in network["buses"] if b["is_terminal"]]
    base = plan.meta.get("old_case") or plan.name           # star and street share loads, as twins do
    key = key_of(stream, base, int(load_seed))
    start = int(np.random.default_rng(np.random.SeedSequence([20261125, 15, *key])).integers(
        0, loads.RECORDED_DAYS-DAYS+1))
    p, q = loads.metered_loads(counts, stream, key, DAYS, start)
    p, q = p/network["base_kw"], q/network["base_kw"]
    v0 = loads.root_voltage(len(p), key)
    extra, info = None, None
    if source is not None:
        bus = source_bus(network, source["where"])
        shape, tan = loads.source_profile(source["kind"], len(p), stream, start, key_of(*key, source["where"]))
        ps = source["share"]*p.sum(axis=1).mean()*shape
        extra = (bus, ps, tan*ps)
        info = dict(source, bus=bus, P=ps, Q=tan*ps, path=source_path(network, bus),
                    mean_kw=float(ps.mean()*network["base_kw"]))
    order, v, s_pcc, current = physics.power_flow(network, labels, p, q, v0, extra)
    at = {b: k for k, b in enumerate(order)}
    volt = v[:, [at[l] for l in labels]]
    i_base = network["base_kw"]/(np.sqrt(3)*network["base_kv"])
    incoming = backend._radial_order(network)[2]
    load_ratio = max((float(current[:, at[b]].max())*i_base/min(CONDUCTORS[c][2] for c in e["conductor"].split("+"))
                      for b, e in incoming.items() if e["conductor"]), default=0.)
    exact = sc.truth_family(network, labels)
    merged, n_merged = sc.identifiable_truth_family(network, labels, R_FLOOR)
    return dict(network=network, labels=labels, n=len(labels), stream=stream, load_seed=int(load_seed), start_day=start,
                key=key, true=dict(P=p, Q=q, V=volt, V0=v0, S_pcc=s_pcc), source=info,
                exact_truth=exact, truth=merged, merged_lines=n_merged,
                max_drop=float((v0[:, None]-volt).max()), min_voltage=float(volt.min()), max_loading=load_ratio,
                mean_kw=float(p.sum(axis=1).mean()*network["base_kw"]),
                peak_kw=float(p.sum(axis=1).max()*network["base_kw"]))


def plausible(rec):
    return DROP_RANGE[0] <= rec["max_drop"] <= DROP_RANGE[1] and rec["max_loading"] <= 1.


def random_case(stream, size, number):
    """The first plausible draw: (plan, how many draws were rejected)."""
    for attempt in range(200):
        plan = feeders.random_plan(1000*number+attempt, size, list(feeders.STYLES)[number % 3], stream)
        plan.name = f"{stream}_{size}_{number:03d}"
        try:
            if plausible(simulate(plan, stream, 1)):
                return plan, attempt
        except RuntimeError:                                 # the power flow has no solution: overloaded
            pass
    raise RuntimeError(f"no plausible feeder for {stream} {size} {number}")


def named_load_seed(case, seed, stream):
    """The first load draw of a seed for which the street version is plausible; star and street share it."""
    for attempt in range(200):
        rec = simulate(feeders.street_plan(case), stream, 1000*attempt+seed)
        if rec["max_drop"] <= DROP_RANGE[1] and rec["max_loading"] <= 1.:
            return 1000*attempt+seed, attempt
    raise RuntimeError(f"no plausible loads for {case}")


def source_of(index, where):
    """Balanced design: over nine feeders every location meets every share with every kind once."""
    if where == "none":
        return None
    k = SOURCES.index(where)-1
    return dict(where=where, share=SHARES[(index+k) % 3], kind=KINDS[(index//3+k) % 3])


# ------------------------------------------------ the sets
def jobs_of(stream):
    if stream == "dev":
        named = [("named", case, shape, seed) for case in feeders.OLD_CASES for shape in ("star", "street")
                 for seed in range(1, 7)]
        return named+[("random", size, k, 1) for size in feeders.SIZES for k in range(1, 7)]
    return [("random", size, k, 1) for size in feeders.SIZES for k in range(1, 16)]


def build_job(job):
    stream, index, (kind, a, b, seed) = job
    rejected = 0
    if kind == "named":
        plan = (feeders.star_plan if b == "star" else feeders.street_plan)(a)
        index = feeders.OLD_CASES.index(a)*6+seed-1            # star and street get the same unmetered loads
        name = f"{plan.name}_l{seed}"
        seed, rejected = named_load_seed(a, seed, stream)
    else:
        plan, rejected = random_case(stream, a, b)
        name = plan.name
    rows = []
    for where in SOURCES:
        rec = simulate(plan, stream, seed, source_of(index, where))
        rec.update(name=f"{name}_{where}", group=name, set=kind, rejected=rejected)
        path = OUT/stream/f"{rec['name']}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = pickle.dumps(rec)
        path.write_bytes(data)
        rows.append(dict(name=rec["name"], group=name, set=kind, n=rec["n"], style=plan.style,
                         shape=plan.meta["shape"], source=where, sha256=hashlib.sha256(data).hexdigest()))
    return rows


def load(stream, name):
    return pickle.loads((OUT/stream/f"{name}.pkl").read_bytes())


def to_input(rec, per_hour=1, history=(144, 48), meter_seed=0, **meter):
    """Meter readings of a case as an `Input` of the other studies (hist also holds the PCC's P and Q)."""
    readings = physics.read(rec["true"], physics.Meters(rec["n"], [*rec["key"], int(meter_seed)], **meter), per_hour)
    day = 24*per_hour
    end = (DAYS-1)*day
    rows = slice(end-sum(history), end)
    if rows.start < 0:
        raise ValueError("the case holds 8 days of history")
    r = {k: a[rows] for k, a in readings.items()}
    hist = dict(P=r["P"], Q=r["Q"], Y=r["V0"][:, None]**2-r["V"]**2, V0sq=r["V0"]**2, Ppcc=r["Ppcc"], Qpcc=r["Qpcc"])
    f = physics.hourly if per_hour == 1 else (lambda a: a)
    last = {k: f(rec["true"][k])[end:] for k in ("P", "Q", "V0")}
    n, network = rec["n"], rec["network"]
    settings = dict(history=list(history), per_hour=per_hour, source=rec["source"], group=rec["group"])
    key = model.nonsingleton(model.canonical_family(rec["truth"], n))
    return Input(name=rec["name"], case=rec["group"], seed=rec["load_seed"], stratum="", network=network,
                 labels=rec["labels"], n=n, base_kw=network["base_kw"], hist=hist,
                 day=dict(P=last["P"], Q=last["Q"], root=last["V0"]), families=[], theta0=[], truth_key=key,
                 truth_in_bank=False, res={}, sigma_kw=hist["P"].std(axis=0)*network["base_kw"],
                 root_profile=r["V0"][-day:], meta=settings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True, choices=("dev", "test"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    loads.customers()                                        # make the cache once
    jobs = [(args.stream, k, j) for k, j in enumerate(jobs_of(args.stream))]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for part in pool.map(build_job, jobs) for r in part]
    (HERE/f"manifest_{args.stream}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8", newline="\n")
    print(f"{args.stream}: {len(rows)} cases from {len(jobs)} feeders and load seeds")


if __name__ == "__main__":
    main()
