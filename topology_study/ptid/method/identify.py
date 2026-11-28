"""Algorithm 1 of method draft v17 (Identify): load gate, tree, anchors and labels. Development only.

Lines of Algorithm 1 and where they are:
  0     the load gate: a load is modelled only when its mean power is at least 1% of the metered power   model.load_share
  1     the model: balance, M, noise variance, price                                             model.Model
  2     initial lines: the nine builder trees (builders.py); with a load, every grouping with and without it, and the
        load placed in each builder tree at its best place                                       with_load, place_load
  3     first tree: local search to convergence, the tree MIQP from it, its claims checked        solve
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
import math
import time

import gurobipy as gp
import numpy as np

from .anchors import anchored_universe
from .builders import builder_trees
from .labels import DECIDED, grouping, label_groupings
from .miqp import tree_miqp
from .model import GATE, Fitter, Model, balance, load_share
from .search import PRICING_TOL, best_addition, local_search
from .sets import canonical, crossing_map, lines_of


@dataclass(frozen=True)
class Settings:
    gate: float = GATE           # load modelled only above this share of the metered power (prior setting)
    lam: float = 1.96            # structural prior, half-BIC units (ln(1/rho), rho = 0.141 on 84 real feeders)
    m: float = 4.6               # margin for labels and anchors in score units
    K: int = 9                   # nodes with at most K units (children in pricing) are searched by all their unions;
                                 # wider tree nodes are priced by the node MIQP
    rounds: int = 2              # anchored searches at most (lines 4-8)
    large_n: int = 35            # beyond this many meters the time limits are doubled
    short_seconds: float = 120.  # tree MIQP limit up to large_n meters
    long_seconds: float = 240.   # beyond large_n meters
    flip_seconds: float = 0.     # 0: the tree MIQP's limit (the user's choice of 2026-11-26: at least that)
    tighten: bool = True         # per-meter response bounds in every MIQP (False: the ablation)
    start_seconds: float = 20.   # first pass of the local search per start (doubled beyond large_n meters); the best
                                 # start is then searched to convergence
    repairs: int = 10            # lines added by pricing per round
    node_seconds: float = 60.    # node MIQP limit
    evidence: bool = True        # with a load: solve the no-load model too, for the evidence and its bounds
    seed: int = 7
    node_refinement: bool = True  # disable node pricing for the paired tree-only comparison
    p_s: int | None = None       # None uses the measured P/Q rule; 1 or 2 fixes the penalty dimension
    wide_node_miqp: bool = True  # False keeps narrow pricing and records wide nodes as untested
    node_gamma: float = 0.       # > 0: a line found at a node with c children pays gamma ln(2^c - c - 2) more
                                 # (extended BIC; search.node_price). 0 keeps the draft's price.


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
    """Lines 3 and 6: a first pass of local search from the empty tree and every start; the best of them searched to
    convergence (no single removal or swap of a pool line lowers the score); the tree MIQP started from it and
    tightened at its level; the MIQP's tree, when better, searched to convergence too. The MIQP's claims are checked:
    a bound above the converged tree's score, or an "optimum" that the local search still improves, is a numerical
    failure (status NUMERIC, no bound). Every tree keeps a line of each `require` group. Returns (lines, info);
    info["bound"] is the MIQP's lower bound over this pool, None when there is none to trust."""
    lines = lines_of(pool, model.leaves)
    fitter, crosses = Fitter(model, lines), crossing_map(lines)
    ok = lambda S: all(any(C in S for C in group) for group in require)
    per = st.start_seconds*(1 if model.n <= st.large_n else 2)
    tried = [local_search(fitter, crosses, s, time.perf_counter()+per, require)
             for s in [frozenset(), *starts] if ok(s)]
    chosen, value = local_search(fitter, crosses, min(tried, key=lambda x: x[1])[0], math.inf, require)
    warm_tree, warm_crit = lines_of(chosen, model.leaves), value
    r = tree_miqp(model, lines, seconds, require=require, seed=st.seed, warm=chosen, known=value,
                  tighten_at=value if st.tighten else None)
    status, bound = r["status"], r["bound"]
    solver_tree, solver_crit = None, None
    if r["family"] is not None:
        found = frozenset(lines_of(r["family"], model.leaves))
        found_value = fitter.crit(found)
        solver_tree, solver_crit = lines_of(found, model.leaves), found_value
        if found_value < value:
            chosen, value = local_search(fitter, crosses, found, math.inf, require)
            if status == gp.GRB.OPTIMAL and value < found_value-1e-4*abs(found_value):
                status, bound = gp.GRB.NUMERIC, None       # the claimed optimum is improved by one move
    if bound is not None and bound > value+1e-5*(1.+abs(value)):
        status, bound = gp.GRB.NUMERIC, None
    if bound is not None:
        bound = min(bound, value)
    # Tree MIQP uses a relative gap of 1e-4 and an absolute gap of 1e-7 (crit units).
    tolerance = max(1e-7, 1e-4*abs(value))
    closed = status in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT) and bound is not None and value-bound <= tolerance
    tree = lines_of(chosen, model.leaves)
    return tree, dict(status=status, miqp_seconds=r["seconds"], pool=len(lines), tree=tree, crit=value, bound=bound,
                      warm_tree=warm_tree, warm_crit=warm_crit, solver_tree=solver_tree, solver_crit=solver_crit,
                      gap_tolerance=tolerance, gap_closed=closed,
                      domain=dict(lines=lines, require=require))


def price(model, tree, universe, require, seconds, st):
    """Line 7: add the best paying line while one exists; after additions, solve again over the universe and the new
    lines and keep the better tree (the score never rises). Returns (tree, info)."""
    if not st.node_refinement:
        return tree, dict(disabled=True, certified=False, added=[], still_paying=None, re_solve=None,
                          enumerated=0, by_miqp=0, evaluated=0, unproved=0, wide_nodes=[])
    added, re_solve = [], None
    best, info = best_addition(model, tree, st.K, st.node_seconds, wide_node_miqp=st.wide_node_miqp,
                                   node_gamma=st.node_gamma)
    scans = [info]
    while best is not None and len(added) < st.repairs:
        tree = lines_of([*tree, best], model.leaves)
        added.append(best)
        best, info = best_addition(model, tree, st.K, st.node_seconds, wide_node_miqp=st.wide_node_miqp,
                                   node_gamma=st.node_gamma)
        scans.append(info)
    if added:
        again, re_solve = solve(model, [*universe, *tree], [frozenset(tree)], seconds, st, require)
        fit = Fitter(model, [*again, *tree])
        if fit.crit(again) < fit.crit(tree)-1e-12:
            tree = again
            best, info = best_addition(model, tree, st.K, st.node_seconds, wide_node_miqp=st.wide_node_miqp,
                                   node_gamma=st.node_gamma)
            scans.append(info)
    return tree, dict(info, scans=scans, added=[list(C) for C in added], still_paying=None if best is None else list(best),
                      re_solve=re_solve)


def improving_flip(labels, domain, require, model):
    """Deduplicate already refitted flip witnesses, then select one feasible in this solve's domain.
    The recorded gains are in H units. Adoption uses the single-addition tolerance in crit units."""
    gains = {}
    for label in labels.values():
        high = label["gap"][1]
        if high is not None and high < 0 and label["rival"] is not None:
            candidate = canonical(label["rival"])
            gains[candidate] = max(gains.get(candidate, 0.), -high)
    allowed = set(domain)
    eligible = {c: g for c, g in gains.items() if g > PRICING_TOL*model.N/2 and allowed.issuperset(c)
                and all(any(line in c for line in group) for group in require)}
    best = max(eligible, key=eligible.get, default=None)
    return best, dict(count=len(gains), eligible=len(eligible), best_gain_H=max(gains.values(), default=0.),
                      eligible_gain_H=eligible.get(best, 0.))


def evidence(model, tree, pool, seconds, st):
    """Line 12, eq. (12): the load's evidence D = min H0 - min H (half-BIC units), with L0 - U1 <= D <= U0 - L1 over
    the pools searched. H0 is the no-load model (same data, same sigma2, u = 0), solved over the meter groupings of
    the pool; H is the load model, solved again over the whole pool from the tree. [L, U]: the MIQP's bound and the best
    tree found (the tree itself, when it is better)."""
    half = model.N/2
    off = model.without_load()
    meter_tree = canonical(g for g in (grouping(C, model.n) for C in tree) if g)
    meter_pool = [g for g in (grouping(C, model.n) for C in pool) if g]
    tree0, info0 = solve(off, meter_pool, [frozenset(meter_tree)], seconds, st)
    _, info1 = solve(model, pool, [frozenset(tree)], seconds, st)
    U0 = min(info0["crit"], Fitter(off, meter_tree).crit(meter_tree))
    U1 = min(info1["crit"], Fitter(model, tree).crit(tree))
    L0, L1 = info0["bound"], info1["bound"]
    return dict(estimate=half*(U0-U1), lower=None if L0 is None else half*(L0-U1),
                upper=None if L1 is None else half*(U0-L1), no_load_tree=[list(C) for C in tree0],
                status=(info0["status"], info1["status"]))


def scope(rounds):
    """The certificate's scope after the last search round: single-line pricing complete, a paying line left (repair
    limit), or incomplete (a node unproved). restricted refers to the recorded domain and gap tolerance of
    the final restricted solve, including a re-solve on the pool enlarged by pricing when one occurred."""
    last = next((x for x in reversed(rounds) if "price" in x), None)
    if last is None:
        return dict(single_line="not priced", restricted=False, restricted_round=None, restricted_solve=None)
    price_info = last["price"]
    final_adoption = rounds[-1].get("flip_witness", {}).get("action") == "adopted"
    single = "complete" if price_info["certified"] else "incomplete"
    if final_adoption or price_info.get("disabled"):
        single = "not priced"
    elif price_info["still_paying"] is not None:
        single = "paying line left"
    # Older saved rounds have no post-pricing solve record; their only available domain is the earlier solve.
    repeated = price_info.get("re_solve") is not None
    final_solve = price_info["re_solve"] if repeated else last["solve"]
    closed = final_solve.get("gap_closed")
    if closed is None or final_adoption:
        value = rounds[-1]["label_info"]["crit"] if final_adoption else final_solve.get("crit")
        bound = final_solve.get("bound")
        statuses = (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT) if final_adoption else (gp.GRB.OPTIMAL,)
        closed = (final_solve.get("status") in statuses and value is not None and bound is not None
                  and 0. <= value-bound <= max(1e-7, 1e-4*abs(value)))
    return dict(single_line=single, restricted=closed, restricted_round=last["round"],
                restricted_solve="price.re_solve" if repeated else "solve")


# ------------------------------------------------------------------ Algorithm 1
def identify(P, Q, Y, V0sq, Ppcc=None, Qpcc=None, st=Settings(), log=None, *, paired_first=None):
    """Algorithm 1 on all slots of one feeder (P, Q, Y = V0^2 - V^2: T x n; V0sq: T; Ppcc, Qpcc: T or None).

    Returns dict(gate, tree (meter groupings), lines and pool (over all leaves), anchors, labels, rounds, stop, scope,
    model) and, with a load, dict(path, places, evidence) as well."""
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
    model = Model(P, Q, Y, V0sq, u, st.lam, p_s=st.p_s)                     # line 1
    load = u is not None
    seconds = st.short_seconds if n <= st.large_n else st.long_seconds
    timing = dict(model=time.perf_counter()-started)

    trees = builder_trees(P, Q, Y, V0sq, u)                                  # line 2
    groupings = canonical(C for t in trees for C in t)
    if load:
        pool = with_load(groupings, n)
        starts = [frozenset(place_load(model, t)[0]) for t in trees]
    else:
        pool, starts = groupings, [frozenset(t) for t in trees]

    t0 = time.perf_counter()
    if paired_first is None:
        tree, first = solve(model, pool, list(dict.fromkeys(starts)), seconds, st)  # line 3
    else:
        # Experimental branch from a common first solve on the same input and settings.
        # Copy the record because later numerical checks can invalidate its bound.
        from copy import deepcopy
        first = deepcopy(paired_first)
        tree = lines_of(first["tree"], model.leaves)
        fit = Fitter(model, pool)
        if set(first["domain"]["lines"]) != set(lines_of(pool, model.leaves)) or not np.isclose(
                fit.crit(tree), first["crit"], rtol=1e-10, atol=1e-12):
            raise ValueError("Paired initial solve fails the domain or fitted-objective check")
    timing["first_tree"] = time.perf_counter()-t0
    say(f"  first tree: {len(tree)} lines over {first['pool']} candidates, MIQP status {first['status']}")

    r = refine(model, tree, pool, st, say, initial_solve=first)               # lines 4-11
    tree, pool, labels, anchors = r["tree"], r["pool"], r["labels"], r["anchors"]
    meter_tree = canonical(g for g in (grouping(C, n) for C in tree) if g)
    out = dict(gate=gate, tree=meter_tree, lines=tree, pool=pool, first=first, anchors=anchors,
               rounds=r["rounds"], stop=r["stop"],
               scope=scope(r["rounds"]), labels=label_rows(labels, anchors),
               model=dict(sigma2=model.sigma2, alpha=model.alpha, gamma=model.gamma, p_s=model.p_s, N=model.N,
                          crit=Fitter(model, tree).crit(tree)), settings=asdict(st))
    if load:                                                                 # line 12
        out["places"] = [(gap, str(D)) for gap, D, _ in place_costs(model, meter_tree)]
        out["path"] = [list(tuple(i for i in C if i < n)) for C in sorted(tree, key=len, reverse=True) if n in C]
        if st.evidence:
            out["evidence"] = evidence(model, tree, pool, seconds, st)
    timing["total"] = time.perf_counter()-started
    out["timing"] = timing
    return out


def refine(model, tree, pool, st, say=lambda *_: None, *, initial_solve=None):
    """Lines 4-11 from a tree and a pool: label, take the decided groupings as anchors, solve over the anchored universe
    with every anchor required, price; until the stopping rule (Lemma 3) holds or `rounds` searches were made.
    Algorithm 2 calls it at every gate on all data so far. Returns dict(tree, pool, labels, anchors, rounds, stop)."""
    n, load = model.n, model.leaves > model.n
    seconds = st.short_seconds if n <= st.large_n else st.long_seconds
    rounds, previous, stop = [], None, None
    last_domain, last_require, last_solve = pool, (), initial_solve

    def label_current(tree, pool):
        labels, linfo = label_groupings(model, tree, pool, st.m, st.flip_seconds or seconds, st.seed, say, st.tighten)
        if last_solve is not None and last_solve.get("bound") is not None:
            rival, info = improving_flip(labels, last_domain, last_require, model)
            if rival is not None:
                value = linfo["crit"]-info["eligible_gain_H"]/(model.N/2)
                if last_solve["bound"] > value+1e-5*(1.+abs(value)):
                    last_solve["flip_bound_conflict"] = dict(bound=last_solve["bound"], witness_crit=value)
                    last_solve.update(status=gp.GRB.NUMERIC, bound=None, gap_closed=False)
        anchors = canonical(M for M, l in labels.items() if l["inside"] and l["label"] == DECIDED)
        details = dict(lines=len(tree), anchors=len(anchors), pool=len(pool), label_info=linfo,
                      labels=dict(Counter(l["label"] for l in labels.values() if l["inside"])))
        return labels, anchors, details

    for rnd in range(st.rounds+1):
        t0 = time.perf_counter()
        labels, anchors, details = label_current(tree, pool)
        record = dict(round=rnd, **details)
        require = [[M, (*M, n)] if load else [M] for M in anchors]
        say(f"  round {rnd}: {len(tree)} lines, {len(anchors)} anchors, labels {record['labels']}, "
            f"{time.perf_counter()-t0:.0f} s")
        last = rounds[-1] if rounds else None
        if last is not None and anchors == previous and not last["price"]["added"] \
                and last["price"]["certified"] and last["solve"]["gap_closed"]:
            stop = "converged: same anchors, single-line pricing complete, restricted gap closed to tolerance"
        elif rnd == st.rounds:
            stop = "round limit"
        if stop:
            # The last search's lower bound only applies within its saved domain and requirements.
            candidate, witness = improving_flip(labels, last_domain, [*last_require, *require], model)
            record["flip_witness"] = witness
            if candidate is not None:
                record["before_flip_label_info"] = record["label_info"]
                tree = candidate
                labels, anchors, details = label_current(tree, pool)
                record.update(details)
                witness["action"] = "adopted"
                new_require = [[M, (*M, n)] if load else [M] for M in anchors]
                _, remaining = improving_flip(labels, last_domain, [*last_require, *new_require], model)
                record["after_flip_witness"] = remaining
                stop = f"final flip witness adopted and relabelled ({'round limit' if rnd == st.rounds else 'stopping check'})"
            rounds.append(record)
            break
        t0 = time.perf_counter()
        meter_pool = [g for g in (grouping(C, n) for C in pool) if g]
        meter_universe, uinfo = anchored_universe(anchors, n, meter_pool, st.K)
        universe = lines_of([*(with_load(meter_universe, n) if load else meter_universe), *tree], model.leaves)
        candidate, witness = improving_flip(labels, universe, require, model)
        record["flip_witness"] = witness
        starts = [frozenset(tree)]
        if candidate is not None:
            starts.append(frozenset(candidate))
            witness["action"] = "warm_start"
        tree, sinfo = solve(model, universe, starts, seconds, st, require)
        tree, pinfo = price(model, tree, universe, require, seconds, st)
        last_domain = lines_of([*universe, *pinfo["added"]], model.leaves)
        last_require = require
        last_solve = pinfo["re_solve"] or sinfo
        pool = lines_of([*pool, *universe, *pinfo["added"], *tree], model.leaves)
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
                 threshold_excluded=l["threshold_excluded"],
                 gap=[None if g is None else float(g) for g in l["gap"]], status=l["status"],
                 seconds=l["seconds"], rival=None if l["rival"] is None else [list(C) for C in l["rival"]])
            for M, l in sorted(labels.items(), key=lambda kv: (len(kv[0]), kv[0]))]


def place_costs(model, meter_tree):
    """Line 12: every place of the load in the tree, [(gap to the best in half-BIC units, place, lines)], best first."""
    options = load_options(meter_tree, model.n)
    fit = Fitter(model, [C for _, o in options for C in o])
    costs = sorted((fit.crit(o), str(D), o) for D, o in options)
    return [(model.N/2*(c-costs[0][0]), D, o) for c, D, o in costs]
