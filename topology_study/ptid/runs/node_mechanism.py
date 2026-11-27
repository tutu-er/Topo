"""Fixed-node exhaustive reference and paired AC missing-line experiment.

Run from topology_study with python -m ptid.runs.node_mechanism --out DIR.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import gurobipy as gp
import numpy as np
import scipy
from scipy.optimize import lsq_linear

from ptid.cases import physics
from ptid.method.identify import Settings, refine
from ptid.method.model import BOUND, Model
from ptid.method.search import PRICING_TOL, node_miqp
from ptid.method.sets import basis, canonical, children, unions


SEED = 20261128
HOURS = 192
NODE_SECONDS = 60


def observation(n, shared):
    """Run the two physical circuits with common loads and metering errors."""
    rng = np.random.default_rng([SEED, n])
    t = np.arange(4*HOURS)
    daily = np.sin(2*np.pi*t/96)
    hourly_load = rng.normal(size=(HOURS, n)).repeat(4, axis=0)
    p = .026 + .004*daily[:, None] + .006*hourly_load + .001*rng.normal(size=(len(t), n))
    p = np.maximum(p, .001)
    q = .3*p + .002*rng.normal(size=(HOURS, n)).repeat(4, axis=0)
    v0 = 1.02 + .002*daily
    labels = list(range(10, 10+n))
    group = (1, n-2)
    buses = [0, 1, *labels] + ([2] if shared else [])
    branches = []

    def connect(a, b, r, x):
        branches.append(dict(from_bus=a, to_bus=b, r_ohm=r, x_ohm=x, status=True))

    connect(0, 1, .016, .008)
    if shared:
        connect(1, 2, .064, .026)
    for i, label in enumerate(labels):
        connect(2 if shared and i in group else 1, label, .02, .008)
    network = dict(root_bus=0, buses=[dict(bus_id=b) for b in buses],
                   branches=branches, base_kv=.4, base_mva=.2)
    order, v, supply, _ = physics.power_flow(network, labels, p, q, v0)
    actual = dict(P=p, Q=q, V=v[:, [order.index(b) for b in labels]], V0=v0, S_pcc=supply)
    meter = physics.read(actual, physics.Meters(n, [SEED, n]), 1)
    data = dict(P=meter["P"], Q=meter["Q"], Y=meter["V0"][:, None]**2-meter["V"]**2,
                V0sq=meter["V0"]**2)
    digest = hashlib.sha256()
    for a in data.values():
        digest.update(np.ascontiguousarray(a, dtype="<f8").tobytes())
    return data, dict(shared=shared, true_groups=[group] if shared else [],
                      observation_sha256=digest.hexdigest(), voltage_min=float(v.min()),
                      voltage_max=float(v.max()))


class Reference:
    """Explicit time-by-meter design; bounded LS with an independent convex bound."""

    def __init__(self, data, model):
        self.data = data
        self.model = model
        y = data["Y"]-model.gamma*data["Y"].mean(axis=1, keepdims=True)
        self.target = ((y-y.mean(0))/np.sqrt(model.sigma2)).ravel()
        self.base = np.hstack([self.pair(c) for c in basis(model.n)])

    def pair(self, group):
        w = np.zeros(self.model.n)
        w[list(group)] = 1
        w = (w-self.model.gamma*w.mean())/np.sqrt(self.model.sigma2)
        cols = []
        for key in ("P", "Q"):
            flow = self.data[key][:, group].sum(axis=1)
            cols.append(((flow-flow.mean())[:, None]*w).ravel())
        return np.column_stack(cols)

    def score(self, groups):
        x = np.hstack([self.base, *(self.pair(c) for c in groups)])
        fit = lsq_linear(x, self.target, bounds=(0, BOUND), method="bvls", tol=1e-11)
        if not fit.success:
            raise ArithmeticError(fit.message)
        theta = np.clip(fit.x, 0, BOUND)
        residual = x@theta-self.target
        upper = float(residual@residual/self.model.N + self.model.alpha*len(groups))
        gradient = 2*(x.T@residual)/self.model.N
        lower = upper + float(np.minimum(-theta*gradient, (BOUND-theta)*gradient).sum())
        return lower, upper


def fixed_node(n):
    data, source = observation(n, shared=True)
    model = Model(**data)
    kids = children((), n)[tuple(range(n))]
    began = perf_counter()
    line, objective, status, bound = node_miqp(model, (), kids, NODE_SECONDS, return_bound=True)
    reference = Reference(data, model)
    candidate = reference.score([line])[1] if line is not None else None
    miqp_seconds = perf_counter()-began
    began = perf_counter()
    scorer = Reference(data, model)
    candidates = unions(kids)
    lower, upper, best = float("inf"), float("inf"), None
    for c in candidates:
        lo, hi = scorer.score([c])
        lower = min(lower, lo)
        if hi < upper:
            upper, best = hi, c
    enum_seconds = perf_counter()-began
    before = scorer.score([])[1]
    same_score = candidate is not None and abs(candidate-upper) <= PRICING_TOL
    bound_valid = bound is not None and bound <= upper+PRICING_TOL
    return dict(kind="fixed", width=n, source=source, model=dict(p_s=model.p_s, N=model.N),
                score_units="crit", before=before, tolerance=PRICING_TOL,
                enumeration=dict(count=len(candidates), seconds=enum_seconds, optimum=[lower, upper],
                                 best=best, gain=before-upper),
                miqp=dict(seconds=miqp_seconds, solver_seconds_limit=NODE_SECONDS, status=status,
                          best=line, objective=objective, refit=candidate, lower_bound=bound,
                          gain=None if candidate is None else before-candidate),
                exhaustive_score_match=same_score, bound_consistent=bound_valid,
                gap_to_independent_optimum=None if candidate is None else candidate-upper)


def controlled(shared, wide):
    data, source = observation(10, shared)
    model = Model(**data)
    # Both arms start at the identical ten-child star without candidate internal lines.
    st = Settings(rounds=1, repairs=2, short_seconds=4, long_seconds=4, start_seconds=1,
                  node_seconds=4, flip_seconds=4, evidence=False, wide_node_miqp=wide)
    began = perf_counter()
    out = refine(model, (), (), st)
    elapsed = perf_counter()-began
    tree = canonical(out["tree"])
    truth = canonical(source["true_groups"])
    ref = Reference(data, model)
    scans = [row["price"] for row in out["rounds"] if "price" in row]
    added = [line for scan in scans for line in scan["added"]]
    labels = [dict(group=g, label=l["label"], inside=l["inside"], gap_H=l["gap"])
              for g, l in out["labels"].items() if g in tree or g in truth]
    return dict(kind="controlled", arm="C" if wide else "B", source=source,
                initial_tree=[], initial_pool=[], width=10, N=model.N, p_s=model.p_s,
                settings=asdict(st), score_units="crit", before=ref.score([]),
                final_score=ref.score(tree), returned=tree, found=canonical(set(tree)&set(truth)),
                missing=canonical(set(truth)-set(tree)), extra=canonical(set(tree)-set(truth)),
                added=added, scans=scans, labels=labels, stop=out["stop"], seconds=elapsed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--part", choices=["fixed","controlled","all"], default="all")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = []
    if args.part in ("fixed", "all"):
        cases.extend((f"fixed_{n}", fixed_node, (n,)) for n in (8, 10, 11))
    if args.part in ("controlled", "all"):
        cases.extend((f"controlled_{int(shared)}_{arm}", controlled, (shared, arm=="C"))
                     for shared in (True, False) for arm in ("B","C"))
    for name, run, params in cases:
        result = run(*params)
        result.update(seed=SEED, hours=HOURS, python=platform.python_version(),
                      numpy=np.__version__, scipy=scipy.__version__, gurobi=gp.gurobi.version())
        path = args.out/f"node_{name}.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(name, path.name, flush=True)


if __name__ == "__main__":
    main()
