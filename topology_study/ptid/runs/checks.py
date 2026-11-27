"""Two development checks that RESULTS.md reports (development cases only; the sealed set is never opened).

    python -m ptid.runs.checks extra-lines      the nested-pair cap of PROPOSALS.md, section 1, on true trees
    python -m ptid.runs.checks wide-nodes       node pricing at the wide nodes of found trees: enumeration against the
                                                node MIQP (needs a dev run file with huge cases)

extra-lines: take the TRUE tree of each no-load case with at most 20 meters and extra lines that refine it (a union of
2 .. k-1 children of one node, at most 40 per node, drawn at random). For each extra line M:
    cost(M) = H(T + M) - H(T) = beta - gain/2,   gain = (RSS(T) - RSS(T + M)) / sigma2   (half-BIC units)
and the largest change of the fitted voltages. Counts: chosen (cost < 0), open (cost <= m); the cap cost <= beta is
asserted for every line.
"""
import argparse
import itertools
import json
import math
import random
import time

import numpy as np

from ptid.cases import library
from ptid.cases.paths import RESULTS
from ptid.method.identify import place_load
from ptid.method.model import GATE, Fitter, Model, balance, load_share
from ptid.method.search import best_addition, node_miqp
from ptid.method.sets import basis, canonical, children, lines_of

M_DECIDE = math.log(100)


def refinements(truth, leaves, rng, per_node=40):
    extra = []
    for _, kids in children(truth, leaves).items():
        kids = list(kids)
        if len(kids) < 3:
            continue
        subsets = [S for r in range(2, len(kids)) for S in itertools.combinations(range(len(kids)), r)]
        rng.shuffle(subsets)
        extra += [tuple(sorted(set().union(*(set(kids[i]) for i in S)))) for S in subsets[:per_node]]
    return [C for C in canonical(extra) if C not in set(truth)]


def fitted(model, fitter, family):
    """Fitted Y (T x n) of a family, and each support's own largest drop in pu."""
    fam = canonical([*family, *fitter.base])
    _, theta = fitter.rss(family)
    k = len(fam)
    Yhat, own = np.zeros((model.T, model.n)), {}
    for j, C in enumerate(fam):
        drop = theta[j]*model.Pt[:, list(C)].sum(1)+theta[k+j]*model.Qt[:, list(C)].sum(1)
        Yhat[:, [i for i in C if i < model.n]] += drop[:, None]
        own[C] = float(np.abs(drop).max())/2
    return Yhat, own


def extra_lines(max_n=20, max_cases=24, seed=1):
    rng, rows, smallest, done = random.Random(seed), [], [], 0
    for name in sorted(r["name"] for r in library.manifest()):
        rec = library.open_case(name)
        if rec["source"] or rec["n"] > max_n:
            continue
        h = library.readings(rec)
        model = Model(h["P"], h["Q"], h["Y"], h["V0sq"])
        truth = lines_of(library.truth(rec), model.leaves)
        extra = refinements(truth, model.leaves, rng)
        if not extra:
            continue
        fitter, beta = Fitter(model, canonical([*truth, *extra])), model.alpha*model.N/2
        Y0, own = fitted(model, fitter, truth)
        smallest.append((min(own[C] for C in truth), name))
        s0, costs = fitter.rss(truth)[0], []
        for M in extra:
            Y1, _ = fitted(model, fitter, [*truth, M])
            cost = beta-(s0-fitter.rss([*truth, M])[0])/2
            assert cost <= beta+1e-9, (name, M, cost, beta)
            costs.append(cost)
            rows.append((cost, float(np.abs(Y1-Y0).max())/2))
        print(f"{name}: n={rec['n']} beta={beta:.2f} extra={len(costs)} open={sum(c <= M_DECIDE for c in costs)} "
              f"chosen={sum(c < 0 for c in costs)}", flush=True)
        done += 1
        if done >= max_cases:
            break
    rows = np.array(rows)
    print(f"\ncases {done}, extra lines {len(rows)}, chosen {int(np.sum(rows[:, 0] < 0))}, "
          f"open {int(np.sum(rows[:, 0] <= M_DECIDE))}; the cap cost <= beta held for every line")
    print("cost quantiles (0, 10, 50, 100%):", [round(float(np.quantile(rows[:, 0], q)), 2) for q in (0, .1, .5, 1)])
    eff = np.array([s for s, _ in smallest])
    print(f"smallest true line per case, own drop, pu: median {np.median(eff):.2e}, below 5e-4: {np.mean(eff < 5e-4):.0%}")


def wide_nodes(path, K=9):
    """Pricing on the final trees of huge cases: enumeration (every union) against the node MIQP beyond K children."""
    rows = [json.loads(x) for x in open(path, encoding="utf-8")]
    for r in [x for x in {x["name"]: x for x in rows}.values() if x["name"].startswith("dev_huge")]:
        rec = library.open_case(r["name"])
        h, n = library.readings(rec), rec["n"]
        u = balance(h["P"], h["Q"], h["Y"], h["V0sq"], h["Ppcc"], h["Qpcc"])
        u = u if load_share(h["P"], u) >= GATE else None
        model = Model(h["P"], h["Q"], h["Y"], h["V0sq"], u)
        meter_tree = [tuple(C) for C in r["tree"]]
        tree = list(place_load(model, meter_tree)[0]) if u is not None else lines_of(meter_tree, n)
        family = canonical([*basis(model.leaves), *tree])
        widths = sorted((len(k) for k in children(family, model.leaves).values()), reverse=True)
        if widths[0] <= K:
            print(f"{r['name']}: widest node {widths[0]}, enumerated")
            continue
        t0 = time.perf_counter(); old, oi = best_addition(model, tree, max_children=10**6); t1 = time.perf_counter()-t0
        t0 = time.perf_counter(); new, ni = best_addition(model, tree, max_children=K); t2 = time.perf_counter()-t0
        print(f"{r['name']}: widths {widths[:4]}; enumeration {old} gain {oi['gain']:.3e} ({t1:.0f} s); node MIQP {new}"
              f" gain {ni['gain']:.3e}, unproved {ni['unproved']} ({t2:.0f} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("check", choices=("extra-lines", "wide-nodes"))
    ap.add_argument("--file", default=str(RESULTS/"dev.jsonl"))
    args = ap.parse_args()
    extra_lines() if args.check == "extra-lines" else wide_nodes(args.file)


if __name__ == "__main__":
    main()
