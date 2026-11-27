"""Algorithm 2 of method draft v16 (Probe): one day of cost-aware probing. Development only.

Gate h, at the start of hour h, with data up to the end of hour h-1:
  1. refresh the leader and its labels: Algorithm 1, lines 4-11, on all data so far (identify.refine);
  2. relabel every target (one outside the tree that no local swap shows open gets the flip test with its grouping
     required); stop when every target is decided at this gate and at the previous one;
  3. if the leader passes the trust gate s2_L / sigma2_y <= kappa_A, buy for hour h+1 the feasible block with the
     largest J(u) = psi sum_M [kappa(delta_M, q_M(u)) - kappa(delta_M, q_M(0))] - c(u)   (eq. 15), when J > 0;
  4. hour h runs with the block sealed at the previous gate, and its readings join the data.
A last gate after hour 23 relabels on the whole day, so the last block's readings are used.

Planner choices (stated in the draft's Section V since v13):
  - a block acts at one meter with one sign; its size is 1/4, 2/4, 3/4 or all of the largest feasible size;
  - the forecast of an hour is the mean of the same hour over the data (24 rows a day, from midnight);
  - sigma2_y = 4 sigma_v^2 1.02^4, the meter class, as in the earlier probing study's trust gate;
  - the load's places are one target, open while the second-best place is within m, with that place as its rival;
  - the voltage band keeps z sigma_V of margin, sigma_V = sigma / 2 (one reading of V);
  - the separation q compares the S-weighted predictions, whose noise covariance is sigma^2 I.
Target variants (PROPOSALS.md, section 3): T0 is the draft; T1 drops groupings outside the leader that
are nested in it; T2 values those in the rival-true world, w kappa(m + e, q) with w = 1 / (1 + exp(e)).

The world (a simulated feeder, or a real one) provides:
  world.base_kw                   base power in kW (the transformer rating is ps.transformer times it)
  world.headroom(hour, pending)   (up, down) in pu per meter for that hour, after `pending` ran in the hour before
  world.run_hour(hour, u)         the hour's readings with block u (pu per meter) held for the hour:
                                  dict(P, Q, Y: n values each; V0sq, Ppcc, Qpcc: one value each)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .identify import Settings, label_rows, place_costs, refine
from .labels import DECIDED, OPEN, flip_test, grouping, local_flips
from .model import BOUND, Fitter, Model, balance
from .sets import basis, canonical, compatible


@dataclass(frozen=True)
class ProbeSettings:
    psi: float = 5.              # CNY per half-BIC unit; PLACEHOLDER until the knee of the cost-coverage curve
    variant: str = "T0"          # target rule: T0 (the draft), T1 or T2
    price_up: float = 1.2        # CNY per kWh of extra consumption
    price_down: float = 4.       # CNY per kWh of reduced consumption
    cap_hour: float = 20.        # CNY
    cap_day: float = 100.        # CNY
    band: tuple = (.96, 1.035)   # pu, for the leader and every rival
    z: float = 2.33
    transformer: float = 1.      # rating, in base powers
    kappa_A: float = 1.2         # trust gate
    sigma_v: float = 5e-4        # meter class: relative noise of a voltage reading
    sizes: int = 4
    hours: int = 24


def kappa(delta, q):
    """E min(delta, Z) for Z ~ N(q/2, q): the expected evidence (Lemma 4), counted up to delta."""
    if q <= 0.:
        return min(delta, 0.)
    s = math.sqrt(q)
    a = (q/2-delta)/s
    cdf, pdf = .5*(1+math.erf(a/math.sqrt(2))), math.exp(-a*a/2)/math.sqrt(2*math.pi)
    return q/2-(q/2-delta)*cdf-s*pdf


# ------------------------------------------------------------------ data, forecast, prediction
def setup(data, load, st):
    """The model on all data so far, and the flows of the leaves (the load's last when it is modelled)."""
    u = balance(data["P"], data["Q"], data["Y"], data["V0sq"], data["Ppcc"], data["Qpcc"]) if load else None
    model = Model(data["P"], data["Q"], data["Y"], data["V0sq"], u, st.lam)
    if u is None:
        return model, data["P"], data["Q"]
    return model, np.column_stack([data["P"], u[0]]), np.column_stack([data["Q"], u[1]])


def forecast(data, fp, fq, hour):
    """The hour's flows (centred, as in the model), V0^2 and PCC power: means of the same hour over the data."""
    rows = slice(hour % 24, None, 24)
    return dict(p=fp[rows].mean(0)-fp.mean(0), q=fq[rows].mean(0)-fq.mean(0), v0sq=data["V0sq"][rows].mean(),
                pcc=data["Ppcc"][rows].mean(), ybar=data["Y"].mean(0))


def prediction(model, family, fc):
    """A fitted tree's prediction for the forecast hour:
    s, ds: S-weighted prediction (n) and its change per pu of block at each meter (n x n), as the score sees them;
    y, dy: predicted V0^2 - V^2 per meter and its change per pu of block (for the voltage band);
    cov:   covariance of s from the fitted parameters (eq. 13; parameters at a bound held fixed)."""
    fam = canonical([*family, *basis(model.leaves)])
    fit = Fitter(model, fam)
    theta = fit.rss(fam)[1]
    k, n = len(fam), model.n
    Z = np.zeros((k, model.leaves))
    for j, C in enumerate(fam):
        Z[j, list(C)] = 1.
    Zm = Z[:, :n]
    W = Zm-model.gamma*Zm.mean(1, keepdims=True)                  # the meters' weights after S, as in Model.gram
    flow_p, flow_q = Z@fc["p"], Z@fc["q"]
    Phi = np.hstack([W.T*flow_p, W.T*flow_q])                      # n x 2k
    r, x = theta[:k], theta[k:]
    free = np.flatnonzero((theta > 1e-9) & (theta < BOUND-1e-9))
    cov = np.zeros((2*k, 2*k))
    cov[np.ix_(free, free)] = np.linalg.pinv(fit.G[np.ix_(free, free)])   # G = F'F / sigma2
    return dict(s=Phi@theta, ds=(W.T*r)@Zm, y=fc["ybar"]+Zm.T@(r*flow_p+x*flow_q), dy=(Zm.T*r)@Zm, cov=Phi@cov@Phi.T)


def separation(lead, rival, sigma2):
    """Eq. 13 for one target: q(u) = (d0 + G u)' Sinv (d0 + G u). Returns (d0, G, Sinv)."""
    Sigma = sigma2*np.eye(len(lead["s"]))+.5*lead["cov"]+.5*rival["cov"]
    return lead["s"]-rival["s"], lead["ds"]-rival["ds"], np.linalg.inv(Sigma)


# ------------------------------------------------------------------ targets and blocks
def relabel_outside(model, tree, pool, rows, keys, st):
    """Line 4 for the targets outside the tree that no local swap shows open: the flip test with their grouping required
    (eq. 6), since local swaps can show a grouping open but never decided (Lemma 1). The best local swap that brings
    the grouping in is a known flipped tree: a flip test whose bound exceeds its score is a numerical failure (labels.py).
    Returns the rows with theirs."""
    have = {tuple(x["grouping"]) for x in rows}
    s0, half = Fitter(model, tree).crit(tree), model.N/2
    swaps = local_flips(Fitter(model, canonical([*pool, *tree])), tree, canonical([*pool, *tree]), model.n)[1]
    seconds = st.flip_seconds or (st.short_seconds if model.n <= st.large_n else st.long_seconds)
    out = list(rows)
    for M in keys:
        if M != "place" and M not in have:
            known = s0-swaps[M][0] if M in swaps else None
            r = flip_test(model, pool, M, s0+st.m/half, seconds, st.seed, st.tighten, inside=False, known=known)
            gap = [None if v is None else (v-s0)*half for v in (r["low"], r["high"])]
            out.append(dict(grouping=list(M), inside=False, anchor=False, label=r["label"], gap=gap, status=r["status"],
                            seconds=r["seconds"], rival=None if r["rival"] is None else [list(C) for C in r["rival"]]))
    return out


def states(model, tree, rows, places, keys, m):
    """Each target's state at a gate: dict(label, high end of its gap, rival lines, kind), where kind is inside,
    nested (outside the tree, compatible with it), crossing, or place (the load's place)."""
    meter_tree = [g for g in (grouping(C, model.n) for C in tree) if g]
    labels = {tuple(x["grouping"]): x for x in rows}
    out = {}
    for M in keys:
        if M == "place":
            gap, _, rival = places[1] if len(places) > 1 else (math.inf, None, None)
            out[M] = dict(label=OPEN if gap <= m else DECIDED, high=gap, rival=rival, kind="place")
        else:
            x = labels[M]
            kind = "inside" if x["inside"] else "nested" if all(compatible(M, C) for C in meter_tree) else "crossing"
            out[M] = dict(label=x["label"], high=x["gap"][1], kind=kind,
                          rival=None if x["rival"] is None else [tuple(C) for C in x["rival"]])
    return out


def term(state, q, m, variant):
    """One target's share of J before psi: the evidence expected towards its decision."""
    e = state["high"]
    if variant == "T2" and state["kind"] == "nested":
        return kappa(m+e, q)/(1+math.exp(e))
    return kappa(m-e, q)


def trust(model, tree, ps):
    """The trust gate's ratio s2_L / sigma2_y."""
    fam = canonical([*tree, *basis(model.leaves)])
    s2 = Fitter(model, fam).rss(fam)[0]*model.sigma2/(model.N-2*len(fam)-model.n)
    return s2/(4*ps.sigma_v**2*1.02**4)


def band_interval(trees, i, sign, fc, sigma2, ps):
    """Sizes a >= 0 of a block (meter i, sign) that keep every tree's predicted voltages in the band, with z sigma_V
    of margin: [low, high]. Each tree's V0^2 - V^2 moves linearly in a."""
    sv = math.sqrt(sigma2)/2
    y_low, y_high = fc["v0sq"]-(ps.band[1]-ps.z*sv)**2, fc["v0sq"]-(ps.band[0]+ps.z*sv)**2
    low, high = 0., math.inf
    for t in trees:
        g = sign*t["dy"][:, i]
        moved = np.abs(g) > 1e-15
        a, b = (y_low-t["y"][moved])/g[moved], (y_high-t["y"][moved])/g[moved]
        low, high = max(low, np.minimum(a, b).max(initial=0.)), min(high, np.maximum(a, b).min(initial=math.inf))
    return low, high


def choose_block(model, lead, rivals, now, fc, up, down, budget, base_kw, ps, m):
    """Eq. 15 over the feasible one-meter blocks: (J, u in pu per meter, cost in CNY) of the best, or (0, None, 0).
    rivals: {target: its rival's prediction}; now: the targets' states."""
    seps = {k: (now[k], *separation(lead, rivals[k], model.sigma2)) for k in rivals}
    passive = sum(term(s, d0@S@d0, m, ps.variant) for s, d0, _, S in seps.values())
    best = (0., None, 0.)
    for i in range(model.n):
        for sign, room, price in ((1., up[i], ps.price_up), (-1., down[i], ps.price_down)):
            low, high = band_interval([lead, *rivals.values()], i, sign, fc, model.sigma2, ps)
            high = min(high, room, budget/(base_kw*price), ps.transformer-fc["pcc"] if sign > 0 else math.inf)
            for a in (low+(high-low)*k/ps.sizes for k in range(1, ps.sizes+1)):
                if high <= low or a <= 1e-12:
                    continue
                value = sum(term(s, (d0+G[:, i]*sign*a)@S@(d0+G[:, i]*sign*a), m, ps.variant)
                            for s, d0, G, S in seps.values())
                J = ps.psi*(value-passive)-base_kw*price*a
                if J > best[0]:
                    best = (J, sign*a*np.eye(model.n)[i], base_kw*price*a)
    return best


def day_targets(model, tree, rows, places, lead, fc, room, ps, m):
    """Line 1: the open alternatives whose rival responds differently from the leader to a block at some meter that
    has headroom during the day (Lemma 5); T1 leaves out the groupings nested in the leader."""
    keys = [tuple(x["grouping"]) for x in rows]+(["place"] if places else [])
    out = []
    for k, s in states(model, tree, rows, places, keys, m).items():
        if s["label"] != OPEN or s["rival"] is None or (ps.variant == "T1" and s["kind"] == "nested"):
            continue
        if np.abs(lead["ds"]-prediction(model, s["rival"], fc)["ds"])[:, room].max(initial=0.) > 1e-12:
            out.append(k)
    return out


# ------------------------------------------------------------------ Algorithm 2
def probe_day(data, first, world, st=Settings(), ps=ProbeSettings(), say=None):
    """One day of probing. data: the history (P, Q, Y: T x n; V0sq, Ppcc, Qpcc: T), 24 rows a day from midnight;
    first: identify()'s output on it. Returns dict(gates, tree, lines, labels, targets, cost, stop)."""
    say = say or (lambda *_: None)
    data = {k: np.asarray(v, float) for k, v in data.items()}
    n, load, m = data["P"].shape[1], first["gate"]["load"], st.m
    tree, pool, rows = first["lines"], first["pool"], first["labels"]
    room = np.max([np.add(*world.headroom(t)) for t in range(ps.hours)], axis=0) > 0
    keys, pending, spent, decided_before, gates, stop = None, np.zeros(n), 0., False, [], "end of day"
    for h in range(ps.hours+1):                          # gate h; the gate h = hours comes after the day
        started = time.perf_counter()
        model, fp, fq = setup(data, load, st)
        if h > 0:                                        # line 3: Algorithm 1, lines 4-11, on all data so far
            r = refine(model, tree, pool, st, say)
            tree, pool, rows = r["tree"], r["pool"], label_rows(r["labels"], r["anchors"])
        places = place_costs(model, [g for g in (grouping(C, n) for C in tree) if g]) if load else []
        fc = forecast(data, fp, fq, h+1)
        lead = prediction(model, tree, fc)
        if keys is None:
            keys = day_targets(model, tree, rows, places, lead, fc, room, ps, m)
        rows = relabel_outside(model, tree, pool, rows, keys, st)        # line 4
        now = states(model, tree, rows, places, keys, m)
        decided = all(s["label"] == DECIDED for s in now.values())
        gate = dict(hour=h, tree=[list(g) for g in (grouping(C, n) for C in tree) if g], trust=None, block=None,
                    targets={str(k): [s["label"], s["high"]] for k, s in now.items()})
        gates.append(gate)
        if not keys or (decided and decided_before):     # line 5
            stop = "no target" if not keys else "every target decided at two consecutive gates"
            gate["seconds"] = time.perf_counter()-started
            break
        decided_before, u = decided, np.zeros(n)
        if h+1 < ps.hours and not decided:               # line 6
            gate["trust"] = trust(model, tree, ps)
            if gate["trust"] <= ps.kappa_A:
                rivals = {k: prediction(model, s["rival"], fc) for k, s in now.items() if s["label"] == OPEN}
                up, down = world.headroom(h+1, pending)
                J, best, cost = choose_block(model, lead, rivals, now, fc, up, down,
                                             min(ps.cap_hour, ps.cap_day-spent), world.base_kw, ps, m)
                if best is not None:
                    u, spent, i = best, spent+cost, int(np.flatnonzero(best)[0])
                    gate["block"] = dict(meter=i, kw=float(best[i]*world.base_kw), cny=float(cost), J=float(J))
        gate["seconds"] = time.perf_counter()-started
        say(f"  gate {h}: {sum(s['label'] == DECIDED for s in now.values())}/{len(now)} targets decided, "
            f"trust {gate['trust']}, block {gate['block']}, {gate['seconds']:.0f} s")
        if h < ps.hours:                                 # line 7: hour h runs with the block sealed at gate h-1
            row = world.run_hour(h, pending)
            data = {k: np.concatenate([v, np.asarray(row[k], float)[None]]) for k, v in data.items()}
            pending = u
    return dict(gates=gates, tree=gates[-1]["tree"], lines=[list(C) for C in tree], labels=rows, targets=keys,
                cost=spent, stop=stop)
