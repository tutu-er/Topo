"""Identifier on levels vs per-terminal demeaned data, with and without meter gain bias.

Demeaning y, P and Q per terminal is equivalent to fitting a per-meter offset,
because the model is linear in P and Q with no intercept. It removes constant
voltage-meter offsets and keeps the daily load swing.

    python -m rnj_wzzt_dr.experiments.hourly_probing_20261118.identifier_remedies
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math

import numpy as np

from rnj_wzzt_dr.identification import algorithm
from rnj_wzzt_dr.identification.algorithm import SearchConfig
from rnj_wzzt_dr.physical import backend
from rnj_wzzt_dr.physical.backend import Observation

from . import config, model
from . import scenario as sc
from .experiments import ROOT


def identify(case, seed, mode, bias):
    s = config.Settings(meter_bias_sigma=bias)
    net, labels, _, _ = sc.network_info(case)
    n = len(labels)
    hist = (48, 24) if seed > 100 else (96, 48)
    T = sum(hist); days = math.ceil(T/96)+1
    P, Q, root = sc.generate_load(case, seed, days, network=net)
    hs = slice(96*(days-1)-T, 96*(days-1))
    v, _ = backend._ac_voltages(net, labels, P[hs], Q[hs], root[hs])
    meters = sc.make_meters(s, n, seed)
    rng = np.random.default_rng(np.random.SeedSequence([99, int(seed)]))
    p, q, y, _, _ = meters.read(rng, P[hs], Q[hs], v, root[hs])

    def prep(part):
        a, b, c = p[part], q[part], y[part]
        if mode == "demean":
            a, b, c = a-a.mean(0), b-b.mean(0), c-c.mean(0)
        return Observation(a, b, c, labels)
    obs = [prep(slice(0, hist[0])), prep(slice(hist[0], T))]
    cfg = SearchConfig(total_seconds=900, per_mip_seconds=300, output_reserve_seconds=120)
    res = algorithm.fit(obs[0], obs[1], "111", config=cfg)
    truth = model.nonsingleton(model.canonical_family(sc.truth_family(net, labels), n))
    fams = [model.nonsingleton(model.canonical_family(c["point"]["supports"], n)) for c in res["frozen_candidates"]]
    sel = model.nonsingleton(model.canonical_family(res["selected"]["point"]["supports"], n))
    return dict(case=case, seed=seed, mode=mode, bias=bias, truth_in_bank=truth in fams, selected_correct=sel == truth)


def main():
    jobs = [(c, s, m, b) for c in ("soumalas11", "pengwah18") for s in (1, 2, 3, 4, 101, 102, 103, 104)
            for m, b in (("levels", 0.), ("demean", 0.), ("demean", .001))]
    with ProcessPoolExecutor(12) as pool:
        out = list(pool.map(identify, *zip(*jobs)))
    print(sorted(Counter((r["mode"], r["bias"], r["selected_correct"]) for r in out).items()))
    (ROOT/"diag_demean.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
