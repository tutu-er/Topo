"""The tree MIQP of draft v8, eq. (11), and its flip-test use (eq. 6).

    min  RSS(r, x) / N + alpha sum_C b_C
    s.t. 0 <= r_C <= BOUND b_C, 0 <= x_C <= BOUND b_C        a line has parameters only when chosen
         b_C + b_D <= 1 for crossing C, D                     the chosen lines form a tree
         sum_{C strictly inside W} b_C <= |W| - 2              valid count cut (tightens the relaxation)
         sum_{C in group} b_C >= 1 for every required group   anchors (a meter grouping with or without the load)
The basis (each leaf alone, all leaves) is always on. RSS is written as ||A theta - c||^2 + const (model.factor).
"""
from __future__ import annotations

from itertools import combinations
import time

import gurobipy as gp
import numpy as np

from .model import BOUND, Fitter, env, factor
from .sets import basis, canonical, compatible, lines_of


def tree_miqp(model, pool, seconds, *, require=(), warm=None, cutoff=None, seed=7, gap=1e-4):
    """Best tree over the lines of `pool`. cutoff: only trees with crit below it are of interest (Gurobi Cutoff; a
    flip test's threshold). Returns dict(status, crit, bound, family, seconds, lines)."""
    started = time.perf_counter()
    leaves = model.leaves
    lines = list(lines_of(pool, leaves))
    supports = canonical([*basis(leaves), *lines])
    pos = {C: k for k, C in enumerate(supports)}
    K = len(supports)
    A, c, const = factor(*model.gram(supports), model.yy)
    with gp.Model(env=env()) as m:
        for key, value in dict(Threads=1, TimeLimit=max(.01, seconds), MIPGap=gap, MIPGapAbs=1e-7, Seed=seed,
                               NumericFocus=2, FeasibilityTol=1e-9, IntFeasTol=1e-9, OptimalityTol=1e-9).items():
            m.setParam(key, value)
        if cutoff is not None:
            m.setParam("Cutoff", cutoff+1e-9*(1.+abs(cutoff)))
        theta = m.addMVar(2*K, lb=0., ub=BOUND)                      # r of every support, then x
        on = {C: m.addVar(vtype=gp.GRB.BINARY) for C in lines}
        for C, v in on.items():
            m.addConstr(theta[pos[C]] <= BOUND*v)
            m.addConstr(theta[K+pos[C]] <= BOUND*v)
        for C, D in combinations(lines, 2):
            if not compatible(C, D):
                m.addConstr(on[C]+on[D] <= 1)
        for W in [*lines, tuple(range(leaves))]:
            inside = [on[C] for C in lines if set(C) < set(W)]
            if inside:
                m.addConstr(gp.quicksum(inside) <= len(W)-2)
        for group in require:
            m.addConstr(gp.quicksum(on[C] for C in group if C in on) >= 1)
        res = m.addMVar(len(c), lb=-gp.GRB.INFINITY)
        m.addConstr(res == A@theta-c)
        m.setObjective((res@res+const)/model.N+model.alpha*gp.quicksum(on.values()))
        if warm is not None:                                        # start from a known tree and its fit
            chosen = set(lines_of(warm, leaves)) & set(on)
            fam = canonical([*basis(leaves), *chosen])
            _, th = Fitter(model, fam).rss(fam)                    # theta in the order of fam: r's, then x's
            start = np.zeros(2*K)
            for k, C in enumerate(fam):
                start[pos[C]], start[K+pos[C]] = th[k], th[len(fam)+k]
            theta.Start = start
            for C, v in on.items():
                v.Start = float(C in chosen)
        m.optimize()
        found = m.SolCount > 0
        try:
            bound = float(m.ObjBound)
        except gp.GurobiError:
            bound = None
        return dict(status=int(m.Status), crit=float(m.ObjVal) if found else None, bound=bound, lines=len(lines),
                    family=canonical([*basis(leaves), *(C for C in lines if on[C].X > .5)]) if found else None,
                    seconds=time.perf_counter()-started)
