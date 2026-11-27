"""Local search, and node pricing (draft v17: the incumbent of eq. (9), and eq. (10) with Lemma 2).

Pricing asks, at every node of a tree, for the best new line: a union of 2 to c-1 of its c children. Nodes with at most
`max_children` children (K = 9 in identify.Settings) are enumerated (every union fitted); wider nodes are solved by
`node_miqp`, which uses the same feasible lines. The certificate is
for single lines only (two lines can pay together when neither pays alone: draft Remark 1), and it is complete only when
every node was enumerated or its lower bound excludes an improvement larger than the stated tolerance.
"""
from __future__ import annotations

import time

import gurobipy as gp
import numpy as np

from .model import BOUND, Fitter, env, factor_scaled
from .sets import basis, canonical, children, lines_of, unions


PRICING_TOL = 1e-7     # absolute improvement in crit; H improves by N/2 times this amount


def local_search(fitter, crosses, start, deadline, require=()):
    """Best-improvement search over trees made of the lines in `crosses` (a crossing_map). A move removes one line, or
    adds one and removes the lines it crosses; it must keep a line of every `require` group. With deadline = inf it
    runs until no move lowers the score: the value of every candidate line is checked (converged). Returns (lines,
    crit)."""
    ok = lambda S: all(any(C in S for C in group) for group in require)
    S = frozenset(C for C in start if C in crosses)
    best = fitter.crit(S)
    while time.perf_counter() < deadline:
        move, move_value = None, best-1e-12
        for C in crosses:
            T = S-{C} if C in S else (S-crosses[C]) | {C}
            if ok(T):
                value = fitter.crit(T)
                if value < move_value:
                    move, move_value = T, value
            if time.perf_counter() > deadline:
                break
        if move is None:
            break
        S, best = move, move_value
    return S, best


def columns(model, family):
    """Explicit columns (N x 2K) of the supports, under M and divided by sigma: r columns, then x columns."""
    cols = {"p": [], "q": []}
    for C in family:
        w = np.array([float(i in C) for i in range(model.n)])
        w -= model.gamma*w.mean()
        for which in ("p", "q"):
            cols[which].append(np.outer(model.flow(C, which), w).ravel())
    return np.column_stack(cols["p"]+cols["q"])/np.sqrt(model.sigma2)


def node_miqp(model, tree, kids, seconds, seed=7, *, return_bound=False):
    """Eq. (10) at one node as an MIQP: the best union of 2 to c-1 of the node's children `kids`, with every
    coefficient of the tree refitted. z_a = 1 when child a is in the new line. The new line's coefficient r enters as
    u_ab = r z_a z_b (a <= b): meter rows of child a see the flow of child b and vice versa. McCormick inequalities make
    u_ab equal to r z_a z_b at binary z, so the fit stays linear; x likewise with v_ab. The columns enter scaled
    (model.factor_scaled), as in the tree MIQP: phi = d theta, and u_ab, v_ab likewise, so each McCormick inequality
    carries its column's d.
    Returns (line or None, objective or None, status). With return_bound=True, append the
    lower bound for the best single addition at this node, also when no incumbent was found."""
    leaves, c = model.leaves, len(kids)
    family = canonical([*basis(leaves), *lines_of(tree, leaves)])
    K = len(family)
    pairs = [(a, b) for a in range(c) for b in range(a, c)]
    meters = [[i for i in kid if i < model.n] for kid in kids]
    new = {"p": [], "q": []}
    for a, b in pairs:                                     # column of u_ab (and of v_ab), under M, divided by sigma
        for which in ("p", "q"):
            X = np.zeros((model.T, model.n))
            X[:, meters[a]] += model.flow(kids[b], which)[:, None]
            if a != b:
                X[:, meters[b]] += model.flow(kids[a], which)[:, None]
            X -= model.gamma*X.mean(1, keepdims=True)
            new[which].append(X.ravel()/np.sqrt(model.sigma2))
    F = np.column_stack([columns(model, family), *new["p"], *new["q"]])
    y = model.Yt.ravel()/np.sqrt(model.sigma2)
    A, cvec, const, d = factor_scaled(F.T@F, F.T@y, model.yy)
    P = len(pairs)
    du, dv = d[2*K:2*K+P], d[2*K+P:]
    with gp.Model(env=env()) as m:
        for key, value in dict(Threads=1, TimeLimit=max(.01, seconds), MIPGap=1e-4, Seed=seed).items():  # defaults
            m.setParam(key, value)
        theta = m.addMVar(2*K, lb=0., ub=BOUND*d[:2*K])    # the tree's r and x, scaled
        rx = m.addMVar(2, lb=0., ub=BOUND)                 # the new line's r and x
        z = m.addMVar(c, vtype=gp.GRB.BINARY)
        u, v = m.addMVar(P, lb=0., ub=BOUND*du), m.addMVar(P, lb=0., ub=BOUND*dv)
        m.addConstr(z.sum() >= 2)
        m.addConstr(z.sum() <= c-1)
        for k, (a, b) in enumerate(pairs):
            for w, coef, s in ((u, rx[0], du[k]), (v, rx[1], dv[k])):     # w = s * (r z_a z_b)
                m.addConstr(w[k] <= s*coef)
                m.addConstr(w[k] <= s*BOUND*z[a])
                m.addConstr(w[k] <= s*BOUND*z[b])
                if a == b:
                    m.addConstr(w[k] >= s*coef-s*BOUND*(1-z[a]))
                else:
                    m.addConstr(w[k] >= s*coef-s*BOUND*(2-z[a]-z[b]))
        res = m.addMVar(len(cvec), lb=-gp.GRB.INFINITY)
        m.addConstr(res == A[:, :2*K]@theta+A[:, 2*K:2*K+P]@u+A[:, 2*K+P:]@v-cvec)
        lines = sum(1 for C in family if 1 < len(C) < leaves)
        m.setObjective((res@res+const)/model.N+model.alpha*(lines+1))
        m.optimize()
        status = int(m.Status)
        line, value = None, None
        if m.SolCount:
            line = tuple(sorted(i for a in range(c) if z.X[a] > .5 for i in kids[a]))
            value = float(m.ObjVal)
        if not return_bound:
            return line, value, status
        bound = None
        if status in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT):
            try:
                raw_bound = float(m.ObjBound)
                if abs(raw_bound) < gp.GRB.INFINITY:
                    bound = raw_bound
            except (gp.GurobiError, AttributeError):
                pass
        return line, value, status, bound


def best_addition(model, tree, max_children=9, node_seconds=60., chunk=200):
    """Find an improving line compatible with the tree. A wide-node time limit can leave better lines untested.
    info["certified"] means all nodes exclude any single-line improvement larger than PRICING_TOL in crit."""
    leaves = model.leaves
    tree = lines_of(tree, leaves)
    family = canonical([*basis(leaves), *tree])
    current = Fitter(model, family).crit(tree)
    best, gain = None, 0.
    info = dict(enumerated=0, by_miqp=0, unproved=0, evaluated=0, tolerance=PRICING_TOL,
                wide_nodes=[], tree_crit=current)
    for node, kids in children(family, leaves).items():
        if len(kids) < 3:
            continue
        if len(kids) > max_children:                       # too wide to enumerate: the node MIQP
            line, _, status, bound = node_miqp(model, tree, kids, node_seconds, return_bound=True)
            info["by_miqp"] += 1
            # Any child union with both new coefficients set to zero attains this upper bound.
            upper = current+model.alpha
            improves = False
            if line is not None:
                value = Fitter(model, [*family, line]).crit([*tree, line])
                upper = min(upper, value)
                improves = current-value > PRICING_TOL
                if current-value > max(gain, PRICING_TOL):
                    best, gain = line, current-value
            if bound is not None and bound > upper+PRICING_TOL:
                status, bound = gp.GRB.NUMERIC, None
            if bound is not None:
                bound = min(bound, upper)
            ruled_out = bound is not None and bound >= current-PRICING_TOL and not improves
            info["unproved"] += not (improves or ruled_out)
            info["wide_nodes"].append(dict(node=node, status=status, upper=upper, bound=bound,
                                           no_improvement=ruled_out))
            continue
        info["enumerated"] += 1
        new = [U for U in unions(kids) if U not in set(family)]
        for k in range(0, len(new), chunk):
            part = new[k:k+chunk]
            fit = Fitter(model, [*family, *part])
            for U in part:
                info["evaluated"] += 1
                g = current-fit.crit([*tree, U])
                if g > max(gain, PRICING_TOL):
                    best, gain = U, g
    info.update(gain=gain, certified=best is None and info["unproved"] == 0)
    return best, info
