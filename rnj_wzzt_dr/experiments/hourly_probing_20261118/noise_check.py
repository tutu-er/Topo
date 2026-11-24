"""Proposition 6 under realistic meter noise (open item in MANUSCRIPT section IX).

For the true topology T and every one-extra-line rival W = T + C_e, simulate
passive data on the AC network, meter them, fit both with the v3 refit and
count how often E_W = J_W - J_T < ln 19. White noise gives at most 19/N. Here
the noise of y_i = V0^2 - V_i^2 shares the root meter's error across terminals,
and optionally each meter has a fixed gain bias.

    python -m rnj_wzzt_dr.experiments.hourly_probing_20261118.noise_check
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import math

import numpy as np

from rnj_wzzt_dr.physical import backend

from . import bank, config, model, scores
from . import scenario as sc
from .experiments import ROOT

SETTINGS = {
    "white_iid": dict(mode="iid"),
    "meter_physical": dict(mode="physical"),
    "meter_physical_bias01": dict(mode="physical", bias=.001),
    "meter_physical_bias01_floor0": dict(mode="physical", bias=.001, floor=0.),
    "meter_physical_floor0": dict(mode="physical", floor=0.),
}


def one(case, label, spec, seeds=300, slots=144):
    net, labels, _, _ = sc.network_info(case)
    n = len(labels)
    T = model.canonical_family(sc.truth_family(net, labels), n)
    rivals = [model.canonical_family(f, n) for f in bank.neighbours(T, n)]
    rivals = [f for f in rivals if model.nonsingleton(f) > model.nonsingleton(T)]
    floor = spec.get("floor", 2*.003/.8)
    cT = model.Candidate(T, n); cWs = [model.Candidate(f, n) for f in rivals]
    below = np.zeros(len(rivals)); lead = np.zeros(len(rivals)); anyb = 0
    s = config.Settings()
    for seed in range(seeds):
        P, Q, root = sc.generate_load(case, 1000+seed, 2)
        P, Q, root = P[-slots:], Q[-slots:], root[-slots:]
        v, _ = backend._ac_voltages(net, labels, P, Q, root)
        rng = np.random.default_rng(seed)
        if spec["mode"] == "iid":
            y_clean = root[:, None]**2-v**2
            sd = s.voltage_sigma*2*math.sqrt(2)*1.02**2
            y = y_clean+sd*rng.normal(size=y_clean.shape)      # same variance, no shared root term
            Pm = P*(1+s.pq_sigma*rng.normal(size=P.shape)); Qm = Q*(1+s.pq_sigma*rng.normal(size=Q.shape))
            v0sq = root**2
        else:
            bias = spec.get("bias", 0.)*rng.normal(size=n); rb = spec.get("bias", 0.)*rng.normal()
            meters = sc.Meters(s.voltage_sigma, s.pq_sigma, bias, rb, 0.)
            Pm, Qm, y, _, rm = meters.read(rng, P, Q, v, root)
            v0sq = rm**2
        hist = dict(P=Pm, Q=Qm, Y=y, V0sq=v0sq)
        fits, J, _ = bank.passive_fit([T]+rivals, n, hist, floor, 2.)
        E = J[1:]-J[0]
        below += E < math.log(19.)
        lead += E < -math.log(19.)
        anyb += bool(np.any(E < math.log(19.)))
    N = slots*n
    return dict(case=case, setting=label, N=N, bound=19/N, rivals=len(rivals), seeds=seeds,
                share_below_mean=float(below.mean()/seeds), share_below_max=float(below.max()/seeds),
                share_lead_mean=float(lead.mean()/seeds), any_rival_below=anyb/seeds)


def main():
    jobs = [(c, k, v) for c in ("soumalas11", "pengwah18") for k, v in SETTINGS.items()]
    with ProcessPoolExecutor(10) as pool:
        out = list(pool.map(one, *zip(*jobs)))
    (ROOT/"noise_check.json").write_text(json.dumps(out, indent=1))
    for r in out:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
