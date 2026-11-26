"""Local search, and node pricing (draft v9: the incumbent of eq. (11), and eq. (12) with Lemma 4).

Pricing asks, at every node of a tree, for the best new line: a union of 2 to c-1 of its c children. Nodes with at most
`max_children` children are enumerated (every union fitted); wider nodes are solved by `node_miqp`. The certificate is
complete only when every node was enumerated or its MIQP proved optimal.
"""
from __future__ import annotations

import time

import gurobipy as gp
import numpy as np

from .model import BOUND, Fitter, env, factor
from .sets import basis, canonical, children, lines_of, unions


def local_search(fitter, crosses, start, deadline, require=()):
    """Best-improvement search over trees made of the lines in `crosses` (a crossing_map). A move removes one line, or
    adds one and removes the lines it crosses; it must keep a line of every `require` group. Returns (lines, crit)."""
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


def node_miqp(model, tree, kids, seconds, seed=7):
    """Eq. (12) at one node as an MIQP: the best union of 2 to c-1 of the node's children `kids`, with every
    coefficient of the tree refitted. z_a = 1 when child a is in the new line. The new line's coefficient r enters as
    u_ab = r z_a z_b (a <= b): meter rows of child a see the flow of child b and vice versa. McCormick inequalities make
    u_ab equal to r z_a z_b at binary z, so the fit stays linear; x likewise with v_ab.
    Returns (line or None, crit of the tree with it, status)."""
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
    A, cvec, const = factor(F.T@F, F.T@y, model.yy)
    P = len(pairs)
    with gp.Model(env=env()) as m:
        for key, value in dict(Threads=1, TimeLimit=max(.01, seconds), MIPGap=1e-4, Seed=seed, NumericFocus=2,
                               FeasibilityTol=1e-9, IntFeasTol=1e-9).items():
            m.setParam(key, value)
        theta = m.addMVar(2*K, lb=0., ub=BOUND)            # the tree's r and x
        rx = m.addMVar(2, lb=0., ub=BOUND)                 # the new line's r and x
        z = m.addMVar(c, vtype=gp.GRB.BINARY)
        u, v = m.addMVar(P, lb=0., ub=BOUND), m.addMVar(P, lb=0., ub=BOUND)
        m.addConstr(z.sum() >= 2)
        m.addConstr(z.sum() <= c-1)
        for k, (a, b) in enumerate(pairs):
            for w, coef in ((u, rx[0]), (v, rx[1])):
                m.addConstr(w[k] <= coef)
                m.addConstr(w[k] <= BOUND*z[a])
                m.addConstr(w[k] <= BOUND*z[b])
                if a == b:
                    m.addConstr(w[k] >= coef-BOUND*(1-z[a]))
                else:
                    m.addConstr(w[k] >= coef-BOUND*(2-z[a]-z[b]))
        res = m.addMVar(len(cvec), lb=-gp.GRB.INFINITY)
        m.addConstr(res == A[:, :2*K]@theta+A[:, 2*K:2*K+P]@u+A[:, 2*K+P:]@v-cvec)
        lines = sum(1 for C in family if 1 < len(C) < leaves)
        m.setObjective((res@res+const)/model.N+model.alpha*(lines+1))
        m.optimize()
        if m.SolCount == 0:
            return None, None, int(m.Status)
        line = tuple(sorted(i for a in range(c) if z.X[a] > .5 for i in kids[a]))
        return line, float(m.ObjVal), int(m.Status)


def best_addition(model, tree, max_children=12, node_seconds=60., chunk=200):
    """Lemma 4 and eq. (12): the line that lowers crit most among all lines compatible with the tree, or None.
    Returns (line, info); info["certified"]: no line pays and every node was priced exactly."""
    leaves = model.leaves
    tree = lines_of(tree, leaves)
    family = canonical([*basis(leaves), *tree])
    current = Fitter(model, family).crit(tree)
    best, gain = None, 0.
    info = dict(enumerated=0, by_miqp=0, unproved=0, evaluated=0)
    for node, kids in children(family, leaves).items():
        if len(kids) < 3:
            continue
        if len(kids) > max_children:                       # too wide to enumerate: the node MIQP
            line, value, status = node_miqp(model, tree, kids, node_seconds)
            info["by_miqp"] += 1
            info["unproved"] += status != gp.GRB.OPTIMAL
            if line is not None:
                value = Fitter(model, [*family, line]).crit([*tree, line])
                if current-value > gain+1e-12:
                    best, gain = line, current-value
            continue
        info["enumerated"] += 1
        new = [U for U in unions(kids) if U not in set(family)]
        for k in range(0, len(new), chunk):
            part = new[k:k+chunk]
            fit = Fitter(model, [*family, *part])
            for U in part:
                info["evaluated"] += 1
                g = current-fit.crit([*tree, U])
                if g > gain+1e-12:
                    best, gain = U, g
    info.update(gain=gain, certified=best is None and info["unproved"] == 0)
    return best, info
