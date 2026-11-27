"""Algorithm 1 of method draft v12 (Identify): load gate, tree, anchors and labels. Development only.

Lines of Algorithm 1 and where they are:
  0     the load gate: a load is modelled only when its mean power is at least 1% of the metered power   model.load_share
  1     the model: balance, M, noise variance, price                                             model.Model
  2     initial lines: the nine builder trees (builders.py); with a load, every grouping with and without it, and the
        load placed in each builder tree at its best place                                       with_load, place_load
  3     first tree: local search, then the tree MIQP                                             solve
  4-8   anchors = decided groupings (labels.py); the anchored universe (anchors.py); the tree MIQP over it with every
        anchor required; pricing and a re-solve (price); until the stopping rule holds or `rounds` searches were
        made (refine, which Algorithm 2 also calls at every gate)
  9-11  labels of the last tree
  12    with a load: its path, its places (place_costs), and its evidence with bounds (evidence)
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from collections import Counter
from dataclasses import asdict, dataclass
import time

import numpy as np

from .anchors import anchored_universe
from .builders import builder_trees
from .labels import DECIDED, grouping, label_groupings
from .miqp import tree_miqp
from .model import GATE, Fitter, Model, balance, load_share
from .search import best_addition, local_search
from .sets import canonical, crossing_map, lines_of


@dataclass(frozen=True)
class Settings:
    gate: float = GATE           # load modelled only above this share of the metered power (prior setting)
    lam: float = 1.96            # structural prior, half-BIC units (ln(1/rho), rho = 0.141 on 84 real feeders)
    m: float = 4.6               # margin of labels and anchors, half-BIC units (ln 100)
    K: int = 9                   # nodes with at most K units are searched by all their unions
    rounds: int = 2              # anchored searches at most (lines 4-8)
    warm_n: int = 35             # up to this many meters the MIQP starts from the local-search tree
    short_seconds: float = 120.  # tree MIQP limit up to warm_n meters
    long_seconds: float = 240.   # beyond warm_n meters
    flip_seconds: float = 0.     # 0: the tree MIQP's limit (the user's choice of 2026-11-26: at least that)
    tighten: bool = True         # per-meter response bounds in every MIQP (False: the ablation)
    start_seconds: float = 20.   # local search per start (doubled beyond warm_n meters)
    polish_seconds: float = 20.
    repairs: int = 10            # lines added by pricing per round
    max_children: int = 12       # wider nodes are priced by the node MIQP
    node_seconds: float = 60.    # node MIQP limit
    evidence: bool = True        # with a load: solve the no-load model too, for the evidence and its bounds
    seed: int = 7


# ------------------------------------------------------------------ the load as a leaf
def with_load(groupings, n):
    """Every meter grouping as a line without and with the load (leaf n); the load beside each meter ({i, n}); and
    all meters without the load (the load at the PCC)."""
    out = [tuple(range(n))]
    for M in groupings:
        out += [tuple(M), (*M, n)]
    out += [(i, n) for i in range(n)]
    return lines_of(out, n+1)


def load_options(meter_tree, n):
    """The admissible places of the load in a meter tree: below one of its lines, below the trunk, or at the PCC
    (beside the trunk). Returns [(place, lines over n + 1 leaves)]."""
    options = []
    for D in [*meter_tree, None]:
        options.append((D, lines_of([(*C, n) if D is not None and set(D) <= set(C) else C for C in meter_tree], n+1)))
    options.append(("pcc", lines_of([*meter_tree, tuple(range(n))], n+1)))
    return options


def place_load(model, meter_tree):
    """The best of load_options by crit: (lines, crit)."""
    options = load_options(meter_tree, model.n)
    fit = Fitter(model, [C for _, o in options for C in o])
    return min(((o, fit.crit(o)) for _, o in options), key=lambda x: x[1])


# ------------------------------------------------------------------ the searches
def solve(model, pool, starts, seconds, st, require=()):
    """Lines 3 and 6: local search from the empty tree and every start, the tree MIQP (warm-started up to warm_n
    meters, tightened at the incumbent's level), the better tree, a polish. Every tree keeps a line of each `require`
    group. Returns (lines, info); info["bound"] is the MIQP's lower bound over this pool."""
    lines = lines_of(pool, model.leaves)
    fitter, crosses = Fitter(model, lines), crossing_map(lines)
    ok = lambda S: all(any(C in S for C in group) for group in require)
    per = st.start_seconds*(1 if model.n <= st.warm_n else 2)
    tried = [local_search(fitter, crosses, s, time.perf_counter()+per, require)
             for s in [frozenset(), *starts] if ok(s)]
    incumbent, value = min(tried, key=lambda x: x[1])
    r = tree_miqp(model, lines, seconds, require=require, seed=st.seed,
                  warm=incumbent if model.n <= st.warm_n else None, tighten_at=value if st.tighten else None)
    chosen = incumbent
    if r["family"] is not None and fitter.crit(lines_of(r["family"], model.leaves)) < value:
        chosen = frozenset(lines_of(r["family"], model.leaves))
    chosen, value = local_search(fitter, crosses, chosen, time.perf_counter()+st.polish_seconds, require)
    return lines_of(chosen, model.leaves), dict(status=r["status"], miqp_seconds=r["seconds"], pool=len(lines),
                                                crit=value, bound=r["bound"])


def price(model, tree, universe, require, seconds, st):
    """Line 7: add the best paying line while one exists; after additions, solve again over the universe and the new
    lines and keep the better tree (the score never rises). Returns (tree, info)."""
    added = []
    best, info = best_addition(model, tree, st.max_children, st.node_seconds)
    while best is not None and len(added) < st.repairs:
        tree = lines_of([*tree, best], model.leaves)
        added.append(best)
        best, info = best_addition(model, tree, st.max_children, st.node_seconds)
    if added:
        again, _ = solve(model, [*universe, *tree], [frozenset(tree)], seconds, st, require)
        fit = Fitter(model, [*again, *tree])
        if fit.crit(again) < fit.crit(tree)-1e-12:
            tree = again
            best, info = best_addition(model, tree, st.max_children, st.node_seconds)
    return tree, dict(added=[list(C) for C in added], still_paying=None if best is None else list(best),
                      certified=info["certified"], enumerated=info["enumerated"], by_miqp=info["by_miqp"],
                      unproved=info["unproved"])


def evidence(model, tree, meter_pool, seconds, st):
    """Line 12: the load's evidence D = H0* - H1* (half-BIC units), with the no-load model (same data, same sigma2,
    u = 0) solved over the meter groupings. Returns dict(estimate = U0 - U1, lower = L0 - U1): H0* >= L0 (the MIQP's
    bound over the meter pool) and H1* <= U1 = H(tree), so D >= L0 - U1 over the pools searched."""
    half = model.N/2
    off = model.without_load()
    meter_tree = canonical(g for g in (grouping(C, model.n) for C in tree) if g)
    tree0, info = solve(off, meter_pool, [frozenset(meter_tree)], seconds, st)
    U1 = Fitter(model, tree).crit(tree)
    U0 = min(info["crit"], Fitter(off, meter_tree).crit(meter_tree))
    return dict(estimate=half*(U0-U1), lower=None if info["bound"] is None else half*(info["bound"]-U1),
                no_load_tree=[list(C) for C in tree0], status=info["status"])


# ------------------------------------------------------------------ Algorithm 1
def identify(P, Q, Y, V0sq, Ppcc=None, Qpcc=None, st=Settings(), log=None):
    """Algorithm 1 on all slots of one feeder (P, Q, Y = V0^2 - V^2: T x n; V0sq: T; Ppcc, Qpcc: T or None).

    Returns dict(gate, tree (meter groupings), lines and pool (over all leaves), anchors, labels, rounds, stop,
    model) and,
    with a load, dict(path, places, evidence) as well."""
    say = log or (lambda *_: None)
    started = time.perf_counter()
    P, Q, Y, V0sq = (np.asarray(a, float) for a in (P, Q, Y, V0sq))
    n = P.shape[1]
    u, gate = None, dict(share=None, threshold=st.gate, load=False)
    if Ppcc is not None:                                                     # line 0: the gate
        u = balance(P, Q, Y, V0sq, np.asarray(Ppcc, float), np.asarray(Qpcc, float))
        gate.update(share=load_share(P, u), load=load_share(P, u) >= st.gate)
        if not gate["load"]:
            u = None
    say(f"  gate: share {gate['share']}, load modelled: {gate['load']}")
    model = Model(P, Q, Y, V0sq, u, st.lam)                                   # line 1
    load = u is not None
    seconds = st.short_seconds if n <= st.warm_n else st.long_seconds
    timing = dict(model=time.perf_counter()-started)

    trees = builder_trees(P, Q, Y, V0sq, u)                                  # line 2
    groupings = canonical(C for t in trees for C in t)
    if load:
        pool = with_load(groupings, n)
        starts = [frozenset(place_load(model, t)[0]) for t in trees]
    else:
        pool, starts = groupings, [frozenset(t) for t in trees]

    t0 = time.perf_counter()
    tree, first = solve(model, pool, list(dict.fromkeys(starts)), seconds, st)  # line 3
    timing["first_tree"] = time.perf_counter()-t0
    say(f"  first tree: {len(tree)} lines over {first['pool']} candidates, MIQP status {first['status']}")

    r = refine(model, tree, pool, st, say)                                   # lines 4-11
    tree, pool, labels, anchors = r["tree"], r["pool"], r["labels"], r["anchors"]
    meter_tree = canonical(g for g in (grouping(C, n) for C in tree) if g)
    out = dict(gate=gate, tree=meter_tree, lines=tree, pool=pool, anchors=anchors, rounds=r["rounds"], stop=r["stop"],
               labels=label_rows(labels, anchors),
               model=dict(sigma2=model.sigma2, alpha=model.alpha, gamma=model.gamma, p_s=model.p_s, N=model.N,
                          crit=Fitter(model, tree).crit(tree)), settings=asdict(st))
    if load:                                                                 # line 12
        out["places"] = [(gap, str(D)) for gap, D, _ in place_costs(model, meter_tree)]
        out["path"] = [list(tuple(i for i in C if i < n)) for C in sorted(tree, key=len, reverse=True) if n in C]
        if st.evidence:
            out["evidence"] = evidence(model, tree, [g for g in (grouping(C, n) for C in pool) if g], seconds, st)
    timing["total"] = time.perf_counter()-started
    out["timing"] = timing
    return out


def refine(model, tree, pool, st, say=lambda *_: None):
    """Lines 4-11 from a tree and a pool: label, take the decided groupings as anchors, solve over the anchored universe
    with every anchor required, price; until the stopping rule (Lemma 3) holds or `rounds` searches were made.
    Algorithm 2 calls it at every gate on all data so far. Returns dict(tree, pool, labels, anchors, rounds, stop)."""
    n, load = model.n, model.leaves > model.n
    seconds = st.short_seconds if n <= st.warm_n else st.long_seconds
    rounds, previous, stop = [], None, None
    for rnd in range(st.rounds+1):
        t0 = time.perf_counter()
        labels, linfo = label_groupings(model, tree, pool, st.m, st.flip_seconds or seconds, st.seed, say, st.tighten)
        anchors = canonical(M for M, l in labels.items() if l["inside"] and l["label"] == DECIDED)
        record = dict(round=rnd, lines=len(tree), anchors=len(anchors), pool=len(pool), label_info=linfo,
                      labels=dict(Counter(l["label"] for l in labels.values() if l["inside"])))
        say(f"  round {rnd}: {len(tree)} lines, {len(anchors)} anchors, labels {record['labels']}, "
            f"{time.perf_counter()-t0:.0f} s")
        last = rounds[-1] if rounds else None
        if last is not None and anchors == previous and not last["price"]["added"] \
                and last["price"]["still_paying"] is None and last["solve"]["status"] == 2:
            stop = "converged: same anchors, pricing added nothing, last MIQP optimal (Lemma 3)"
        elif rnd == st.rounds:
            stop = "round limit"
        if stop:
            rounds.append(record)
            break
        t0 = time.perf_counter()
        meter_pool = [g for g in (grouping(C, n) for C in pool) if g]
        meter_universe, uinfo = anchored_universe(anchors, n, meter_pool, st.K)
        universe = lines_of([*(with_load(meter_universe, n) if load else meter_universe), *tree], model.leaves)
        require = [[M, (*M, n)] if load else [M] for M in anchors]
        tree, sinfo = solve(model, universe, [frozenset(tree)], seconds, st, require)
        tree, pinfo = price(model, tree, universe, require, seconds, st)
        pool = lines_of([*pool, *universe, *tree], model.leaves)
        record.update(universe=uinfo, solve=sinfo, price=pinfo, search_seconds=time.perf_counter()-t0)
        say(f"  anchored search: {uinfo['lines']} meter lines (k {uinfo['k'][:6]}, {uinfo['big_nodes']} nodes over K), "
            f"MIQP status {sinfo['status']}, priced +{len(pinfo['added'])}, {record['search_seconds']:.0f} s")
        rounds.append(record)
        previous = anchors
    say(f"  stop: {stop}")
    return dict(tree=tree, pool=pool, labels=labels, anchors=anchors, rounds=rounds, stop=stop)


def label_rows(labels, anchors):
    """The labels as plain rows (lists, floats), smallest groupings first."""
    return [dict(grouping=list(M), inside=l["inside"], anchor=M in set(anchors), label=l["label"],
                 gap=[None if g is None else float(g) for g in l["gap"]], status=l["status"],
                 seconds=l["seconds"], rival=None if l["rival"] is None else [list(C) for C in l["rival"]])
            for M, l in sorted(labels.items(), key=lambda kv: (len(kv[0]), kv[0]))]


def place_costs(model, meter_tree):
    """Line 12: every place of the load in the tree, [(gap to the best in half-BIC units, place, lines)], best first."""
    options = load_options(meter_tree, model.n)
    fit = Fitter(model, [C for _, o in options for C in o])
    costs = sorted((fit.crit(o), str(D), o) for D, o in options)
    return [(model.N/2*(c-costs[0][0]), D, o) for c, D, o in costs]
