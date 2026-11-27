"""The tree MIQP of draft v12, eq. (9), and its flip-test use (eq. 6).

    min  RSS(r, x) / N + alpha sum_C b_C
    s.t. 0 <= r_C <= rbar_C b_C, 0 <= x_C <= xbar_C b_C         a line has parameters only when chosen
         b_C + b_D <= 1 for crossing C, D                         the chosen lines form a tree
         sum_{C strictly inside W} b_C <= |W| - 2                  count cut (valid; tightens the relaxation)
         sum_{C in group} b_C >= 1 for every required group       anchors: a meter grouping, with or without the load
         sum_{C contains i} r_C <= Rbar_i, same for x              per-meter bounds (valid below a level; optional)
The basis (each leaf alone, all leaves) is always on. RSS = ||A theta - c||^2 + const (model.factor).
"""
from __future__ import annotations

from itertools import combinations
import time

import gurobipy as gp
import numpy as np

from .model import BOUND, Fitter, env, factor
from .sets import basis, canonical, compatible, lines_of


def response_bounds(model, supports, A, c, const, level, seconds):
    """Per-meter bounds valid for every tree with crit <= level (the prices are >= 0, so its residual is at most
    N level): for each meter i the largest R_ii = sum of r over the supports containing i (and X_ii), by one convex
    QCQP per meter with the binaries dropped. 5% safety margin; meters not reached within `seconds` keep the trivial
    bound. Returns (Rbar, Xbar), or None when no coefficients fit under the level at all."""
    rhs = model.N*level-const
    if rhs <= 0:
        return None
    K, n = len(supports), model.n
    member = np.array([[i in C for i in range(n)] for C in supports], float)       # K x n
    out = np.tile(BOUND*member.sum(0), (2, 1))
    deadline = time.perf_counter()+seconds
    with gp.Model(env=env()) as m:
        for key, value in dict(Threads=1, OutputFlag=0, NumericFocus=2).items():
            m.setParam(key, value)
        theta = m.addMVar(2*K, lb=0., ub=BOUND)
        res = m.addMVar(len(c), lb=-gp.GRB.INFINITY)
        m.addConstr(res == A@theta-c)
        m.addConstr(res@res <= rhs)
        for block in (0, 1):
            for i in range(n):
                left = deadline-time.perf_counter()
                if left <= 0:
                    return out[0], out[1]
                m.setParam("TimeLimit", left)
                m.setObjective(gp.quicksum(theta[block*K+k] for k in np.flatnonzero(member[:, i])), gp.GRB.MAXIMIZE)
                m.optimize()
                if m.Status == gp.GRB.INFEASIBLE:
                    return None
                if m.Status == gp.GRB.OPTIMAL:
                    out[block, i] = min(out[block, i], 1.05*m.ObjVal+1e-9)
    return out[0], out[1]


def tree_miqp(model, pool, seconds, *, require=(), warm=None, cutoff=None, tighten_at=None, seed=7, gap=1e-4):
    """Best tree over the lines of `pool`.
    cutoff: only trees with crit below it are wanted (Gurobi Cutoff; a flip test's threshold).
    tighten_at: a crit level known to be reached (an incumbent) or the flip threshold; the per-meter bounds of
    `response_bounds` for that level are added (a quarter of the time limit at most). None: no bounds (ablation).
    Returns dict(status, crit, bound, family, seconds, lines, tightened)."""
    started = time.perf_counter()
    leaves = model.leaves
    lines = list(lines_of(pool, leaves))
    supports = canonical([*basis(leaves), *lines])
    pos = {C: k for k, C in enumerate(supports)}
    K = len(supports)
    A, c, const = factor(*model.gram(supports), model.yy)
    rbar, xbar, sums = np.full(K, BOUND), np.full(K, BOUND), None
    if tighten_at is not None:
        sums = response_bounds(model, supports, A, c, const, tighten_at, .25*seconds)
        if sums is None:                                   # nothing fits under the level: no tree is below it
            return dict(status=gp.GRB.INFEASIBLE, crit=None, bound=tighten_at, family=None, lines=len(lines),
                        seconds=time.perf_counter()-started, tightened=True)
        for k, C in enumerate(supports):
            meters = [i for i in C if i < model.n]
            if meters:
                rbar[k], xbar[k] = min(BOUND, sums[0][meters].min()), min(BOUND, sums[1][meters].min())
    with gp.Model(env=env()) as m:
        for key, value in dict(Threads=1, TimeLimit=max(.01, seconds-(time.perf_counter()-started)), MIPGap=gap,
                               MIPGapAbs=1e-7, Seed=seed, NumericFocus=2, FeasibilityTol=1e-9, IntFeasTol=1e-9,
                               OptimalityTol=1e-9).items():
            m.setParam(key, value)
        if cutoff is not None:
            m.setParam("Cutoff", cutoff+1e-9*(1.+abs(cutoff)))
        theta = m.addMVar(2*K, lb=0., ub=np.r_[rbar, xbar])           # r of every support, then x
        on = {C: m.addVar(vtype=gp.GRB.BINARY) for C in lines}
        for C, v in on.items():
            m.addConstr(theta[pos[C]] <= rbar[pos[C]]*v)
            m.addConstr(theta[K+pos[C]] <= xbar[pos[C]]*v)
        for C, D in combinations(lines, 2):
            if not compatible(C, D):
                m.addConstr(on[C]+on[D] <= 1)
        for W in [*lines, tuple(range(leaves))]:
            inside = [on[C] for C in lines if set(C) < set(W)]
            if inside:
                m.addConstr(gp.quicksum(inside) <= len(W)-2)
        for group in require:
            m.addConstr(gp.quicksum(on[C] for C in group if C in on) >= 1)
        if sums is not None:
            for i in range(model.n):
                rows = [k for k, C in enumerate(supports) if i in C]
                m.addConstr(gp.quicksum(theta[k] for k in rows) <= sums[0][i])
                m.addConstr(gp.quicksum(theta[K+k] for k in rows) <= sums[1][i])
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
                    seconds=time.perf_counter()-started, tightened=sums is not None)
