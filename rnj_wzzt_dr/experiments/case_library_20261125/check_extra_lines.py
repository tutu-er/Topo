"""Check for PROPOSALS_UNDER_TEST.md, target rule (development cases only; the sealed set is never opened).

Take the TRUE tree of each no-load development case (at most MAX_N meters) and extra lines that refine it: a union of
2 .. k-1 children of one node (at most PER_NODE per node, drawn at random). For each extra line M, with the true tree T:

    cost(M) = H(T + M) - H(T) = beta - gain/2,   gain = (RSS(T) - RSS(T + M)) / sigma2   (half-BIC units)

and the largest change of the fitted voltages on the history, max over hours and meters of |V(T + M) - V(T)| in pu
(Y = V0^2 - V^2, so dV ~ dY / 2). Counts: chosen (cost < 0: the extra line beats the truth), open (cost <= m). Also
checks the cap cost <= beta (T + M can always copy T with r_M = x_M = 0), and reports each case's smallest true line
effect (that line's own drop at its largest flow).

Run from the repository root:  python -m rnj_wzzt_dr.experiments.case_library_20261125.check_extra_lines
"""
import glob
import itertools
import math
import os
import random

import numpy as np

from rnj_wzzt_dr.identification_v8 import identify as I
from rnj_wzzt_dr.identification_v8.model import Model, Fitter
from rnj_wzzt_dr.identification_v8.sets import lines_of, children, canonical

M_DECIDE = math.log(100)
MAX_N, PER_NODE, MAX_CASES, SEED = 20, 40, 24, 1


def refinements(truth, leaves, rng):
    """Extra lines inside the true tree: unions of 2 .. k-1 children of one node."""
    extra = []
    for node, kids in children(truth, leaves).items():
        kids = list(kids)
        k = len(kids)
        if k < 3:
            continue
        subsets = [S for r in range(2, k) for S in itertools.combinations(range(k), r)]
        rng.shuffle(subsets)
        extra += [tuple(sorted(set().union(*(set(kids[i]) for i in S)))) for S in subsets[:PER_NODE]]
    return [C for C in canonical(extra) if C not in set(truth)]


def fitted_drops(model, fitter, family):
    """Fitted Y (T x n) of a family, and each support's own largest drop in pu."""
    fam = canonical([*family, *fitter.base])
    _, theta = fitter.rss(family)
    k = len(fam)
    Yhat = np.zeros((model.T, model.n))
    own = {}
    for j, C in enumerate(fam):
        drop = theta[j]*model.Pt[:, list(C)].sum(1)+theta[k+j]*model.Qt[:, list(C)].sum(1)
        Yhat[:, [i for i in C if i < model.n]] += drop[:, None]
        own[C] = float(np.abs(drop).max())/2
    return Yhat, own


def main():
    rng = random.Random(SEED)
    names = sorted({os.path.basename(f)[:-4] for f in glob.glob("outputs/case_library_20261125/dev/*.pkl")})
    rows, smallest, done = [], [], 0
    for name in names:
        rec, inp = I.load_dev_case(name)
        if rec["source"] or inp.n > MAX_N:
            continue
        h = inp.hist
        model = Model(h["P"], h["Q"], h["Y"], h["V0sq"])
        truth = lines_of(rec["truth"], model.leaves)
        extra = refinements(truth, model.leaves, rng)
        if not extra:
            continue
        fitter = Fitter(model, canonical([*truth, *extra]))
        beta = model.alpha*model.N/2
        Y0, own = fitted_drops(model, fitter, truth)
        smallest.append((min(own[C] for C in truth), name))
        s0 = fitter.rss(truth)[0]
        costs = []
        for M in extra:
            Y1, _ = fitted_drops(model, fitter, [*truth, M])
            cost = beta-(s0-fitter.rss([*truth, M])[0])/2
            assert cost <= beta+1e-9, (name, M, cost, beta)
            costs.append(cost)
            rows.append((cost, float(np.abs(Y1-Y0).max())/2))
        print(f"{name}: n={inp.n} beta={beta:.2f} extra={len(costs)} open={sum(c <= M_DECIDE for c in costs)} "
              f"chosen={sum(c < 0 for c in costs)} min cost={min(costs):.2f}", flush=True)
        done += 1
        if done >= MAX_CASES:
            break
    rows = np.array(rows)
    opened = rows[:, 0] <= M_DECIDE
    print(f"\ncases {done}, extra lines {len(rows)}, chosen {int(np.sum(rows[:, 0] < 0))}, open {int(opened.sum())} "
          f"({opened.mean():.1%}); the cap cost <= beta held for every line")
    print("cost quantiles (0, 5, 10, 25, 50, 75, 100%):",
          [round(float(np.quantile(rows[:, 0], q)), 2) for q in (0, .05, .1, .25, .5, .75, 1)])
    for label, sel in (("all extra", np.ones(len(rows), bool)), ("open extra", opened)):
        e = rows[sel, 1]
        print(f"{label}: largest voltage change, pu: median {np.median(e):.2e}, 90% {np.quantile(e, .9):.2e}, "
              f"max {e.max():.2e}; below 5e-4: {np.mean(e < 5e-4):.0%}")
    eff = np.array([s for s, _ in smallest])
    print(f"smallest true line per case, own drop, pu: median {np.median(eff):.2e}, below 5e-4: {np.mean(eff < 5e-4):.0%}; "
          f"lowest {min(smallest)[0]:.1e} in {min(smallest)[1]}")


if __name__ == "__main__":
    main()
