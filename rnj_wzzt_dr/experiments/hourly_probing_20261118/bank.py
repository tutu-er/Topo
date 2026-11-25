"""Regenerate inputs: passive history, identifier bank plus one-edit neighbours, strata.

The protocol's source banks (data/) are not on this PC. Each input here is a
fresh passive history from the case files, a bank from the repository's passive
identifier (method 111, Gurobi), extended by every one-edit neighbour of the
passive BIC leader (one extra line, one missing line, one moved leaf), so that
every rival type of PROTOCOL section 8 is present. Strata follow PROTOCOL
section 3 with the v3 scorer on the passive history.
"""
from __future__ import annotations

import copy
from itertools import combinations
import math
import pickle

import numpy as np

from rnj_wzzt_dr.identification import algorithm
from rnj_wzzt_dr.identification.algorithm import SearchConfig
from rnj_wzzt_dr.identification.trees import is_laminar
from rnj_wzzt_dr.physical import backend
from rnj_wzzt_dr.physical.backend import Observation

from . import model, scores
from . import scenario as sc
from .run import Input


# ------------------------------------------------------------------ neighbours
def _children(node, supports, n):
    inside = [C for C in supports if C < node]
    maximal = [C for C in inside if not any(C < D for D in inside)]
    covered = set().union(*maximal) if maximal else set()
    leaves = [frozenset([i]) for i in node if i not in covered]
    return maximal+leaves


def neighbours(family, n, max_children=8):
    """All one-edit laminar neighbours: add a support, drop one, move one leaf.

    Adding a support groups children of one node: every group size when the node
    has at most `max_children` children, sizes 2-3 up to 12 children, pairs beyond.
    (Until 2026-11-21 the limit was 5 children with pairs beyond, which made a
    merged-away short segment with 3+ terminals unreachable in one edit.)
    """
    fam = {frozenset(C) for C in family if len(C) > 1}
    full = frozenset(range(n))
    nodes = list(fam | {full})
    out = set()

    def emit(new):
        new = {C for C in new if len(C) > 1}
        if new != fam and is_laminar([tuple(sorted(C)) for C in new]+[(i,) for i in range(n)]):
            out.add(frozenset(new))
    for node in nodes:
        ch = _children(node, fam, n)
        c = len(ch)
        if c >= 3:
            sizes = range(2, c) if c <= max_children else range(2, min(4, c)) if c <= 12 else [2]
            for r in sizes:
                for combo in combinations(ch, r):
                    emit(fam | {frozenset().union(*combo)})
    if full not in fam:
        emit(fam | {full})
    for C in fam:
        emit(fam-{C})
    for node in nodes:
        ch = _children(node, fam, n)
        leaves = [C for C in ch if len(C) == 1]
        sub = [C for C in ch if len(C) > 1]
        parent = min((D for D in nodes if node < D), key=len, default=None)
        siblings = [] if parent is None else [C for C in _children(parent, fam, n) if len(C) > 1 and C != node]
        for leaf in leaves:
            (i,) = tuple(leaf)
            if node != full:
                emit((fam-{node}) | {node-leaf})               # up to the grandparent
                for S in siblings:
                    emit((fam-{node, S}) | {node-leaf, S | leaf})
            for S in sub:
                emit((fam-{S}) | {S | leaf})                   # down into a child support
    return [tuple(sorted(tuple(sorted(C)) for C in f)) for f in out]


# ------------------------------------------------------------------ passive fit
def passive_fit(families, n, hist, r_min, upper, steps=3, use_loss=True, offsets=False, whiten=False, pq_sigma=None):
    fits = [fit_one(fam, n, hist, r_min, upper, steps, use_loss, offsets, whiten, pq_sigma) for fam in families]
    J, sigma2 = scores.bic([f.sse for f in fits], [f.c.p+(n if offsets else 0) for f in fits], hist["Y"].size)
    return fits, J, sigma2


def fit_one(fam, n, hist, r_min, upper, steps=3, use_loss=True, offsets=False, whiten=False, pq_sigma=None):
    gamma = model.root_gamma(hist["V0sq"], hist["Y"]) if whiten else 0.
    c = model.Candidate(fam, n)
    if pq_sigma is not None:                     # count identifiable parameters only (settings.identifiable_params)
        c.p = model.identifiable_p(c, hist["P"], hist["Q"], pq_sigma)
    Phi = c.design(hist["P"], hist["Q"])
    f = model.Fit(c, np.full(2*c.S, r_min), r_min, upper)
    f.theta = np.clip(f.theta, f.lb, f.ub)
    lam = None
    for _ in range(steps):
        f.refit(Phi, hist["Y"].ravel(), lam, steps=1, use_loss=False, offsets=offsets, gamma=gamma)
        lam = c.loss_term(f.theta, hist["P"], hist["Q"], hist["V0sq"]).ravel() if use_loss else None
    if use_loss:
        f.refit(Phi, hist["Y"].ravel(), lam, steps=1, use_loss=False, offsets=offsets, gamma=gamma)
    return f


# ------------------------------------------------------------------ revised step 1
def _demeaned(obs):
    return Observation(obs.p-obs.p.mean(0), obs.q-obs.q.mean(0), obs.y-obs.y.mean(0), obs.labels)


def identifier_families(obs_train, obs_val, n, seconds, demean=False):
    """Passive identifier (method 111): contracted output, the path before the
    0.08 output contraction, and the validation-MSE pick."""
    tr, va = (_demeaned(obs_train), _demeaned(obs_val)) if demean else (obs_train, obs_val)
    cfg = SearchConfig(total_seconds=seconds, per_mip_seconds=min(300., seconds/3),
                       output_reserve_seconds=min(120., seconds/5))
    ident = algorithm.fit(tr, va, "111", config=cfg)
    canon = lambda sup: model.canonical_family(sup, n)
    return dict(contracted=[canon(c["point"]["supports"]) for c in ident["frozen_candidates"]],
                uncontracted=[canon(p["supports"]) for p in ident.get("uncontracted_output_path", [])],
                selected=canon(ident["selected"]["point"]["supports"]), seconds=ident["total_seconds"])


def closure(bank, n, hist, r_min, upper, margin=6., max_seeds=12, max_rounds=4, max_size=1500, whiten=False,
            offsets=True, steps=2, pq_sigma=None):
    """BIC neighbourhood closure: add every one-edit neighbour of each candidate
    within `margin` of the BIC leader, refit, repeat until no candidate within the
    margin is left unexpanded (or a size/round cap binds).

    The returned record says whether the closure completed or a cap bound.
    """
    bank = list(dict.fromkeys(bank))
    keys = {model.nonsingleton(f): k for k, f in enumerate(bank)}
    fits = [fit_one(f, n, hist, r_min, upper, steps, True, offsets, whiten, pq_sigma) for f in bank]
    N = hist["Y"].size
    extra = n if offsets else 0
    expanded, rounds, capped = set(), 0, None
    while True:
        J, sigma2 = scores.bic([f.sse for f in fits], [f.c.p+extra for f in fits], N)
        E = J-J.min()
        seeds = [k for k in np.argsort(E) if E[k] <= margin and k not in expanded][:max_seeds]
        if not seeds:
            break
        if rounds >= max_rounds:
            capped = "rounds"; break
        if len(bank) >= max_size:
            capped = "size"; break
        rounds += 1
        for k in seeds:
            expanded.add(k)
            for nb in neighbours(bank[k], n):
                f = model.canonical_family(nb, n)
                key = model.nonsingleton(f)
                if key not in keys:
                    keys[key] = len(bank); bank.append(f)
                    fits.append(fit_one(f, n, hist, r_min, upper, steps, True, offsets, whiten, pq_sigma))
    unexpanded = sum(1 for k in np.flatnonzero(E <= margin) if k not in expanded)
    return bank, fits, J, sigma2, dict(rounds=rounds, capped=capped, size=len(bank),
                                       unexpanded_within_margin=int(unexpanded))


def adequacy(sse, p, N, n, offsets, sigma_v, v_level=1.02):
    """Leader residual variance over the meter-noise variance of y = V0^2 - V^2.

    Each row has a terminal and the root voltage reading with relative error
    sigma_v: var(y) ~ 4 sigma_v^2 (V^4 + V0^4). A ratio well above 1 means no
    candidate explains the data to meter accuracy (truth likely missing).
    """
    dof = N-p-(n if offsets else 0)
    return float(sse/dof/(8*sigma_v**2*v_level**4))


def shorten_segment(network, labels, seed, r_target):
    """Short-line set: one internal segment with >= 2 metered terminals downstream."""
    net = copy.deepcopy(network)
    order, parent, incoming = backend._radial_order(net)
    desc = {b: set() for b in order}
    for i, l in enumerate(labels):
        desc[l].add(i)
    for c in reversed(order[1:]):
        desc[parent[c]].update(desc[c])
    choices = sorted(c for c in order[1:] if 2 <= len(desc[c]) < len(labels))
    rng = np.random.default_rng(np.random.SeedSequence([55, int(seed)]))
    child = choices[int(rng.integers(len(choices)))]
    edge = incoming[child]
    factor = r_target/edge["r_ohm"]
    for row in net["branches"]:
        if row["from_bus"] == edge["from_bus"] and row["to_bus"] == edge["to_bus"]:
            row["r_ohm"] *= factor; row["x_ohm"] *= factor
    return net, dict(segment=[edge["from_bus"], edge["to_bus"]], downstream=len(desc[child]),
                     r_ohm=r_target, factor=factor)


def build(case, seed, settings, history=(192, 96), short_line=None, identifier_seconds=900,
          extra_neighbours=True, use_identifier=True, step1="v1", prune_E=30., max_bank=150):
    """Build one Input (truth kept in the bank if the identifier/neighbours contain it).

    step1="v1": identifier output plus the one-edit neighbours of its BIC leader.
    step1="v2": the revised step 1 (identifier_families on levels and demeaned
    data, then `closure`), pruned to BIC gap <= prune_E and at most max_bank.
    """
    network = sc.load_network(case)
    network, labels, _, _ = sc.network_info(case, network)
    spd = settings.slots_per_hour*settings.hours             # rows per day (96 for 15-minute data, 24 hourly)
    meta = dict(history=list(history), short_line=None, unbalanced=settings.unbalanced,
                meter_bias_sigma=settings.meter_bias_sigma, voltage_resolution_v=settings.voltage_resolution_v)
    if short_line is not None:
        network, info = shorten_segment(network, labels, seed, short_line)
        meta["short_line"] = info
    n = len(labels)
    zbase = network["base_kv"]**2/network["base_mva"]
    r_min = 2*settings.r_min_ohm/zbase
    T_hist = sum(history)
    days = math.ceil(T_hist/spd)+1
    P, Q, root = sc.generate_load(case, seed, days, network=network, sph=settings.slots_per_hour)
    hs = slice(spd*(days-1)-T_hist, spd*(days-1))
    day = dict(P=P[spd*(days-1):], Q=Q[spd*(days-1):], root=root[spd*(days-1):])
    households = sc.households_of(network)
    res_tmp = sc.resources(settings, P[hs].mean(axis=0)*network["base_kw"], network["base_kw"], households)
    physics = sc.Physics(network, labels, settings, seed=seed, households=res_tmp["households"])
    v = physics.voltages(P[hs], Q[hs], root[hs])
    meters = sc.make_meters(settings, n, seed)
    rng = np.random.default_rng(np.random.SeedSequence([99, int(seed)]))
    p_m, q_m, y_m, v_m, r_m = meters.read(rng, P[hs], Q[hs], v, root[hs])
    hist = dict(P=p_m, Q=q_m, Y=y_m, V0sq=r_m**2)
    tr = slice(0, history[0]); va = slice(history[0], T_hist)
    obs = [Observation(p_m[s_], q_m[s_], y_m[s_], labels, v_m[s_], r_m[s_]) for s_ in (tr, va)]
    offs = settings.meter_offsets
    pq = settings.pq_sigma if settings.identifiable_params else None
    if step1 == "v2":
        # revised step 1: levels and demeaned identifier runs, both before and after
        # the 0.08 contraction, closed under one-edit neighbours within a BIC margin
        lv = identifier_families(obs[0], obs[1], n, identifier_seconds, demean=False)
        dm = identifier_families(obs[0], obs[1], n, identifier_seconds, demean=True)
        uniq = list(dict.fromkeys(lv["contracted"]))
        selected = lv["selected"]
        ident = dict(total_seconds=lv["seconds"]+dm["seconds"])
        C = list(dict.fromkeys(uniq+lv["uncontracted"]+dm["contracted"]+dm["uncontracted"]))
        D, cfits, cJ, _, crec = closure(C, n, hist, r_min, settings.upper_bound, offsets=offs,
                                        whiten=settings.root_whitening, pq_sigma=pq)
        order = np.argsort(cJ-cJ.min())
        keep = [k for k in order if cJ[k]-cJ.min() <= prune_E][:max_bank]
        bank = [D[k] for k in keep]
        meta["closure"] = dict(crec, kept=len(bank), prune_E=prune_E, max_bank=max_bank)
    else:
        if use_identifier:
            cfg = SearchConfig(total_seconds=identifier_seconds, per_mip_seconds=min(300., identifier_seconds/3),
                               output_reserve_seconds=min(120., identifier_seconds/5))
            ident = algorithm.fit(obs[0], obs[1], "111", config=cfg)
            fams = [model.canonical_family(cnd["point"]["supports"], n) for cnd in ident["frozen_candidates"]]
            selected = model.canonical_family(ident["selected"]["point"]["supports"], n)
        else:   # tests only: bank = truth and its one-edit neighbours, no Gurobi
            selected = model.canonical_family(sc.truth_family(network, labels), n)
            fams, ident = [selected], dict(total_seconds=0.)
        uniq = list(dict.fromkeys(fams))
        fits, J, _ = passive_fit(uniq, n, hist, r_min, settings.upper_bound, offsets=offs, whiten=settings.root_whitening,
                                 pq_sigma=pq)
        leader = uniq[int(np.argmin(J))]
        bank = list(uniq)
        if extra_neighbours:
            keys = {model.nonsingleton(f) for f in bank}
            for nb in neighbours(leader, n):
                f = model.canonical_family(nb, n)
                if model.nonsingleton(f) not in keys:
                    bank.append(f); keys.add(model.nonsingleton(f))
    fits, J, sigma2 = passive_fit(bank, n, hist, r_min, settings.upper_bound, offsets=offs, whiten=settings.root_whitening,
                                  pq_sigma=pq)
    truth = sc.truth_family(network, labels)
    truth_key = model.nonsingleton(model.canonical_family(truth, n))
    if settings.identifiable_truth:
        # evaluation target: the truth with its segments below the resistance floor merged (REAL_NETWORK_CASE.md);
        # the exact truth is kept for reporting
        meta["exact_truth_key"] = truth_key
        ident_truth, merged = sc.identifiable_truth_family(network, labels, settings.r_min_ohm)
        truth_key = model.nonsingleton(model.canonical_family(ident_truth, n))
        meta["sub_floor_segments_merged"] = merged
        meta["exact_truth_in_bank"] = any(model.nonsingleton(f) == meta["exact_truth_key"] for f in bank)
    p_mean_kw = p_m.mean(axis=0)*network["base_kw"]
    res = sc.resources(settings, p_mean_kw, network["base_kw"], households)
    sigma_kw = p_m.std(axis=0)*network["base_kw"]
    slot = (np.arange(T_hist)+(spd-(T_hist % spd))) % spd
    root_profile = np.array([r_m[slot == t].mean() if np.any(slot == t) else r_m.mean() for t in range(spd)])
    meta.update(identifier_bank=len(uniq), bank=len(bank), identifier_seconds=ident["total_seconds"],
                identifier_selected_correct=model.nonsingleton(selected) == truth_key,
                identifier_bank_has_truth=any(model.nonsingleton(f) == truth_key for f in uniq),
                validation_mse_correct=model.nonsingleton(selected) == truth_key)
    inp = Input(name=f"{case}_s{seed}", case=case, seed=seed, stratum="", network=network, labels=labels, n=n,
                base_kw=network["base_kw"], hist=hist, day=day, families=bank,
                theta0=[f.theta.copy() for f in fits], truth_key=truth_key,
                truth_in_bank=any(model.nonsingleton(f) == truth_key for f in bank), res=res,
                sigma_kw=sigma_kw, root_profile=root_profile, meta=meta)
    inp.meta["passive"] = classify(inp, settings, J)
    return inp


def classify(inp, settings, J=None):
    """Stratum by the v3 scorer on the passive history (PROTOCOL section 3)."""
    n = inp.n
    cands = [model.Candidate(f, n) for f in inp.families]
    if J is None:
        zbase = inp.network["base_kv"]**2/inp.network["base_mva"]
        fits, J, _ = passive_fit(inp.families, n, inp.hist, 2*settings.r_min_ohm/zbase, settings.upper_bound,
                                 offsets=settings.meter_offsets, whiten=settings.root_whitening,
                                 pq_sigma=settings.pq_sigma if settings.identifiable_params else None)
    J = np.asarray(J); L = int(np.argmin(J)); E = J-J[L]
    V0 = inp.root_profile**2
    vL = scores.voltages(cands[L], inp.theta0[L], inp.day["P"], inp.day["Q"], V0)
    eq = np.zeros(len(cands), bool); eq[L] = True
    for k in range(len(cands)):
        if k != L and E[k] < 25:
            vk = scores.voltages(cands[k], inp.theta0[k], inp.day["P"], inp.day["Q"], V0)
            eq[k] = float(np.max(np.abs(vk-vL))) <= settings.equivalence_pu
    ok, mass = scores.posterior_ok(E, eq, L, settings.alpha)
    leader_correct = cands[L].key == inp.truth_key
    rivals = sorted([k for k in range(len(cands)) if k != L and not eq[k]], key=lambda k: E[k])
    near_types = [scores.rival_type(cands[k].key, inp.truth_key) for k in rivals if E[k] < settings.ln19]
    if not inp.truth_in_bank:
        stratum = "S3"
    elif leader_correct and ok:
        stratum = "S2"
    else:
        stratum = "S1"
    inp.stratum = stratum
    return dict(stratum=stratum, leader_correct=bool(leader_correct), posterior_ok=bool(ok), mass=float(mass),
                leader_type=scores.rival_type(cands[L].key, inp.truth_key), near_types=near_types,
                n_equivalent=int(eq.sum()-1), Emin=float(E[rivals[0]]) if rivals else None)


def drop_truth(inp, settings):
    """S3 variant: the same input with the true topology removed from the bank."""
    out = copy.copy(inp)
    keep = [k for k, f in enumerate(inp.families) if model.nonsingleton(f) != inp.truth_key]
    out.families = [inp.families[k] for k in keep]
    out.theta0 = [inp.theta0[k] for k in keep]
    out.truth_in_bank = False
    out.name = inp.name+"_missing"
    out.meta = dict(inp.meta)
    out.meta["passive"] = classify(out, settings)
    return out


def save(inp, path):
    with open(path, "wb") as fh:
        pickle.dump(inp, fh)


def load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)
