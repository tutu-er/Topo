"""(a) size of the unbalance model error; (b) identifier on time-differenced data under meter bias."""
import sys, os, json, math
sys.path.insert(0, r"C:\0-workspace\Topo\repo")
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from rnj_wzzt_dr.physical import backend
from rnj_wzzt_dr.physical.backend import Observation
from rnj_wzzt_dr.identification import algorithm
from rnj_wzzt_dr.identification.algorithm import SearchConfig
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import scenario as sc, model, config, bank

def unbalance(case, seed):
    s = config.Settings()
    net, labels, _, _ = sc.network_info(case); n = len(labels)
    P, Q, root = sc.generate_load(case, seed, 1)
    res = sc.resources(s, P.mean(0)*net["base_kw"], net["base_kw"])
    vb, _ = backend._ac_voltages(net, labels, P, Q, root)
    tp = sc.unbalanced.ThreePhase(net, labels, res["households"], seed=seed)
    vu = tp.terminal_voltages(P, Q, root)
    yb, yu = root[:, None]**2-vb**2, root[:, None]**2-vu**2
    cand = model.Candidate(sc.truth_family(net, labels), n)
    Phi = cand.design(P, Q)
    f = model.Fit(cand, np.zeros(2*cand.S), 0., 2.).refit(Phi, yu.ravel(), cand.loss_term(np.zeros(2*cand.S), P, Q, root**2).ravel()*0, use_loss=False)
    resid = yu.ravel()-Phi@f.theta
    sig = s.voltage_sigma*2*math.sqrt(2)*1.02**2
    shares = np.round(tp.load_share, 2)
    return dict(case=case, seed=seed, rms_bal_vs_unbal=float(np.sqrt(np.mean((yu-yb)**2))/sig),
                truth_fit_resid_rms=float(np.sqrt(np.mean(resid**2))/sig), max_phase_share=float(tp.load_share.max(1).mean()),
                vmin_unbal=float(vu.min()), vmin_bal=float(vb.min()))

def ident_diff(case, seed, diff):
    s = config.Settings(meter_bias_sigma=.001)
    net, labels, _, _ = sc.network_info(case); n = len(labels)
    hist = (48, 24) if seed > 100 else (96, 48)
    T = sum(hist); days = math.ceil(T/96)+1
    P, Q, root = sc.generate_load(case, seed, days, network=net)
    hs = slice(96*(days-1)-T, 96*(days-1))
    v, _ = backend._ac_voltages(net, labels, P[hs], Q[hs], root[hs])
    meters = sc.make_meters(s, n, seed)
    rng = np.random.default_rng(np.random.SeedSequence([99, int(seed)]))
    p, q, y, vm, rm = meters.read(rng, P[hs], Q[hs], v, root[hs])
    if diff:
        p, q, y, vm = np.diff(p, axis=0), np.diff(q, axis=0), np.diff(y, axis=0), None
        rm = None
        T -= 1
    tr = slice(0, hist[0]); va = slice(hist[0], T)
    obs = [Observation(p[x], q[x], y[x], labels) for x in (tr, va)]
    cfg = SearchConfig(total_seconds=900, per_mip_seconds=300, output_reserve_seconds=120)
    res = algorithm.fit(obs[0], obs[1], "111", config=cfg)
    truth = model.nonsingleton(model.canonical_family(sc.truth_family(net, labels), n))
    fams = [model.nonsingleton(model.canonical_family(c["point"]["supports"], n)) for c in res["frozen_candidates"]]
    sel = model.nonsingleton(model.canonical_family(res["selected"]["point"]["supports"], n))
    return dict(case=case, seed=seed, diff=diff, truth_in_bank=truth in fams, selected_correct=sel == truth)

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    out = {}
    out["unbalance"] = [unbalance(c, s) for c in ("soumalas11", "pengwah18") for s in (101, 102, 103, 1)]
    for r in out["unbalance"]: print(json.dumps(r))
    jobs = [(c, s, d) for c in ("soumalas11", "pengwah18") for s in (1, 2, 3, 4, 101, 102, 103, 104) for d in (False, True)]
    with ProcessPoolExecutor(8) as pool:
        out["bias_diff"] = list(pool.map(ident_diff, *zip(*jobs)))
    from collections import Counter
    print(Counter((r["case"], r["diff"], r["truth_in_bank"], r["selected_correct"]) for r in out["bias_diff"]))
    json.dump(out, open(r"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120\diag.json", "w"), indent=1)
