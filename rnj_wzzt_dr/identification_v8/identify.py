"""Algorithm 1 of method draft v8 (Identify): tree, anchors and labels. Development only.

    python -m rnj_wzzt_dr.identification_v8.identify --names dev_small_003_head [--blind] [--out FILE]

Lines of Algorithm 1 and where they are:
  1     the model: balance, M, noise variance, price                           model.Model
  2     initial lines: the nine builder trees (candidates.py), every grouping with and without the load; the load
        placed in each builder tree at its best place                          with_load, place_load
  3     first tree: local search, then the tree MIQP                            solve (search.py, miqp.py)
  4-8   anchors = decided groupings (labels.py); the anchored universe (anchors.py); the tree MIQP over it with every
        anchor required; node pricing (search.best_addition); until the anchors repeat, at most `rounds` times
  9-11  labels of the last tree
  12    with the load: its path, places and evidence
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import time

import numpy as np

from .anchors import anchored_universe
from .candidates import builder_trees
from .labels import DECIDED, grouping, label_groupings
from .miqp import tree_miqp
from .model import Fitter, Model, balance
from .search import best_addition, local_search
from .sets import canonical, crossing_map, lines_of


@dataclass(frozen=True)
class Settings:
    lam: float = 1.96            # structural prior, half-BIC units (ln(1/rho), rho = 0.141 on 84 real feeders)
    m: float = 4.6               # margin of labels and anchors, half-BIC units (ln 100)
    K: int = 9                   # nodes with at most K units are searched by all their unions
    rounds: int = 2              # anchor rounds (lines 4-8)
    warm_n: int = 35             # up to this many meters the MIQP starts from the local-search tree
    short_seconds: float = 120.  # tree MIQP limit up to warm_n meters
    long_seconds: float = 240.   # beyond warm_n meters
    flip_seconds: float = 0.     # 0: the tree MIQP's limit (the user's choice of 2026-11-26: at least that)
    start_seconds: float = 20.   # local search per start (doubled beyond warm_n meters)
    polish_seconds: float = 20.
    repairs: int = 10            # lines added by pricing per round
    max_children: int = 12       # pricing enumerates nodes with at most this many children
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
    """The meter tree with the load at each place: below one of its lines, below the trunk, or at the PCC.
    Returns [(place, lines over n + 1 leaves)]."""
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
    meters), the better tree, a polish. Every tree keeps a line of each `require` group. Returns (lines, info)."""
    lines = lines_of(pool, model.leaves)
    fitter, crosses = Fitter(model, lines), crossing_map(lines)
    ok = lambda S: all(any(C in S for C in group) for group in require)
    per = st.start_seconds*(1 if model.n <= st.warm_n else 2)
    tried = [local_search(fitter, crosses, s, time.perf_counter()+per, require)
             for s in [frozenset(), *starts] if ok(s)]
    incumbent, value = min(tried, key=lambda x: x[1])
    r = tree_miqp(model, lines, seconds, require=require, seed=st.seed,
                  warm=incumbent if model.n <= st.warm_n else None)
    chosen = incumbent
    if r["family"] is not None and fitter.crit(lines_of(r["family"], model.leaves)) < value:
        chosen = frozenset(lines_of(r["family"], model.leaves))
    chosen, value = local_search(fitter, crosses, chosen, time.perf_counter()+st.polish_seconds, require)
    return lines_of(chosen, model.leaves), dict(status=r["status"], miqp_seconds=r["seconds"], pool=len(lines),
                                                crit=value)


def price(model, tree, universe, require, seconds, st):
    """Line 7: add the best paying line (node pricing) while one exists; after additions, solve again over the
    universe and the new lines. Returns (tree, info)."""
    added = []
    best, info = best_addition(model, tree, st.max_children)
    while best is not None and len(added) < st.repairs:
        tree = lines_of([*tree, best], model.leaves)
        added.append(best)
        best, info = best_addition(model, tree, st.max_children)
    if added:
        again, _ = solve(model, [*universe, *tree], [frozenset(tree)], seconds, st, require)
        fit = Fitter(model, [*again, *tree])
        if fit.crit(again) < fit.crit(tree)-1e-12:
            tree = again
            best, info = best_addition(model, tree, st.max_children)
    return tree, dict(added=[list(C) for C in added], certified=info["certified"], skipped_nodes=info["skipped"],
                      still_paying=None if best is None else list(best))


# ------------------------------------------------------------------ Algorithm 1
def identify(P, Q, Y, V0sq, Ppcc=None, Qpcc=None, st=Settings(), log=None):
    """Algorithm 1 on all slots of one feeder (P, Q, Y = V0^2 - V^2: T x n; V0sq: T; Ppcc, Qpcc: T or None).

    Returns dict(tree (meter groupings), lines (over all leaves), anchors, labels, rounds, model) and, with the PCC
    power, dict(path, places, evidence) as well."""
    say = log or (lambda *_: None)
    started = time.perf_counter()
    P, Q, Y, V0sq = (np.asarray(a, float) for a in (P, Q, Y, V0sq))
    n = P.shape[1]
    u = None if Ppcc is None else balance(P, Q, Y, V0sq, np.asarray(Ppcc, float), np.asarray(Qpcc, float))
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
    timing["candidates"] = time.perf_counter()-started-timing["model"]

    t0 = time.perf_counter()
    tree, first = solve(model, pool, list(dict.fromkeys(starts)), seconds, st)  # line 3
    timing["first_tree"] = time.perf_counter()-t0
    say(f"  first tree: {len(tree)} lines over {first['pool']} candidates, MIQP status {first['status']}")

    rounds, previous = [], None
    for rnd in range(st.rounds+1):                                           # lines 4-11
        t0 = time.perf_counter()
        labels, linfo = label_groupings(model, tree, pool, st.m, st.flip_seconds or seconds, st.seed, say)
        anchors = canonical(M for M, l in labels.items() if l["inside"] and l["label"] == DECIDED)
        record = dict(round=rnd, lines=len(tree), anchors=len(anchors), pool=len(pool), label_info=linfo,
                      labels=dict(Counter(l["label"] for l in labels.values() if l["inside"])))
        say(f"  round {rnd}: {len(tree)} lines, {len(anchors)} anchors, labels {record['labels']}, "
            f"{time.perf_counter()-t0:.0f} s")
        if anchors == previous or rnd == st.rounds:
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

    meter_tree = canonical(g for g in (grouping(C, n) for C in tree) if g)
    out = dict(tree=meter_tree, lines=tree, anchors=anchors, rounds=rounds,
               labels=[dict(grouping=list(M), inside=l["inside"], anchor=M in set(anchors), label=l["label"],
                            gap=[None if g is None else float(g) for g in l["gap"]], status=l["status"],
                            seconds=l["seconds"], rival=None if l["rival"] is None else [list(C) for C in l["rival"]])
                       for M, l in sorted(labels.items(), key=lambda kv: (len(kv[0]), kv[0]))],
               model=dict(sigma2=model.sigma2, alpha=model.alpha, gamma=model.gamma, p_s=model.p_s, N=model.N,
                          crit=Fitter(model, tree).crit(tree)), settings=asdict(st))
    if load:                                                                 # line 12
        half = model.N/2
        options = load_options(meter_tree, n)
        fit = Fitter(model, [C for _, o in options for C in o])
        costs = sorted((fit.crit(o), D) for D, o in options)
        out["places"] = [(half*(c-costs[0][0]), D if D in (None, "pcc") else list(D)) for c, D in costs]
        out["path"] = [list(grouping(C, n) or tuple(i for i in C if i < n)) for C in sorted(tree, key=len, reverse=True)
                       if n in C]
        off = model.without_load()
        out["evidence"] = half*(Fitter(off, meter_tree).crit(meter_tree)-Fitter(model, tree).crit(tree))
    timing["total"] = time.perf_counter()-started
    out["timing"] = timing
    return out


# ------------------------------------------------------------------ development runs (never the sealed set)
def f1(est, truth):
    tp = len(est & truth)
    p = tp/len(est) if est else 0.
    r = tp/len(truth) if truth else 1.
    return dict(tp=tp, est=len(est), truth=len(truth), precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.)


def load_dev_case(name):
    """A development case's readings and truth, checked against its manifest hash. The sealed set is never opened."""
    import hashlib
    import pickle
    from rnj_wzzt_dr.experiments.case_library_20261125 import build
    manifest = {r["name"]: r["sha256"] for r in json.loads((build.HERE/"manifest_dev.json").read_text(encoding="utf-8"))}
    if name not in manifest:
        raise SystemExit(f"{name} is not a development case")
    data = (build.OUT/"dev"/f"{name}.pkl").read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest[name]:
        raise SystemExit(f"{name}: the case file does not match its manifest hash")
    rec = pickle.loads(data)
    return rec, build.to_input(rec)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--names", required=True, help="development case names, comma-separated")
    ap.add_argument("--blind", action="store_true", help="without the PCC power")
    ap.add_argument("--out", default=None, help="jsonl file to append to")
    args = ap.parse_args()
    for name in args.names.split(","):
        rec, inp = load_dev_case(name)
        h, n = inp.hist, inp.n
        print(f"{name}: n={n}, load={rec['source'] and rec['source']['where']}", flush=True)
        pcc = () if args.blind else (h["Ppcc"], h["Qpcc"])
        r = identify(h["P"], h["Q"], h["Y"], h["V0sq"], *pcc, log=lambda s: print(s, flush=True))
        inner = lambda f: {tuple(sorted(C)) for C in f if 1 < len(C) < n}
        truth = inner(rec["truth"])
        score = f1(inner(r["tree"]), truth)
        false_anchors = sum(1 for A in r["anchors"] if tuple(A) not in truth)
        print(f"  F1 {score['f1']:.3f} (P {score['precision']:.2f}, R {score['recall']:.2f}), anchors {len(r['anchors'])} "
              f"({false_anchors} not true), evidence {r.get('evidence')}, {r['timing']['total']:.0f} s", flush=True)
        if args.out:
            row = dict(name=name, n=n, load=rec["source"] and rec["source"]["where"], blind=args.blind, score=score,
                       false_anchors=false_anchors, truth=[list(C) for C in sorted(truth)],
                       **{k: v for k, v in r.items() if k != "lines"})
            with open(args.out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str)+"\n")


if __name__ == "__main__":
    main()
