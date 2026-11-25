"""Steps 3 and 5 of WZZT-C (readable edition): the exact MIQP that chooses the best tree from a set of lines.

Given candidate lines (from any source, of any size), find the subset that forms a valid tree and has the lowest
score, by solving one mixed-integer quadratic program:

    minimise   ||F r_x - y||^2 / N  +  alpha * (number of chosen lines)
    variables  a_C in {0, 1}    : line C is in the tree
               r_C, x_C >= 0    : its resistance and reactance, at most bound, and zero when a_C = 0
    subject to a_C + a_D <= 1   for every two lines that cross (neither contains the other, but they overlap)
               (a tree on |W| meters has at most |W| - 2 internal lines inside W)

The fitting error is measured on the whole voltage map (every meter, every slot), not on the entries of a fitted
matrix: this is the point of WZZT. Gurobi proves the optimum (or reports the gap when the time limit binds).

Same model as identification/revised.py (select_miqp), without its optional bound tightening (a speed-up that
does not change the optimum).
"""
import numpy as np
import gurobipy as gp

from .data import basic_lines, canonical, compatible, gurobi_env, internal


def compress_rows(F, y, rtol=1e-10):
    """Replace the N rows of the least-squares problem by at most 2K rows with the same fitting error (SVD)."""
    U, s, Vt = np.linalg.svd(F, full_matrices=False)
    if len(s) == 0 or s[0] <= 0.:
        return np.zeros((0, F.shape[1])), np.zeros(0), float(y@y)
    k = int((s > s[0]*rtol).sum())
    uy = U[:, :k].T@y
    rest = float(np.sum((y-U[:, :k]@uy)**2))
    return s[:k, None]*Vt[:k], uy, rest


def best_tree(problem, candidates, warm=None, seconds=240., gap=1e-4, seed=7):
    """The best tree among the candidate lines. Returns (tree lines, info about the solve)."""
    n, bound = problem.n, problem.settings.bound
    lines = canonical(list(candidates)+basic_lines(n))
    proper = internal(lines, n)                          # the lines the MIQP may switch on or off
    always = [line for line in lines if line not in proper]
    K = len(lines)
    F = problem.columns(lines)                           # N x 2K: r columns then x columns
    B, uy, rest = compress_rows(F, problem.y)
    where = {line: k for k, line in enumerate(lines)}
    with gp.Model(env=gurobi_env()) as model:
        model.Params.Threads = 1
        model.Params.TimeLimit = seconds
        model.Params.MIPGap = gap
        model.Params.MIPGapAbs = 1e-7
        model.Params.NumericFocus = 2
        model.Params.Seed = seed
        model.Params.FeasibilityTol = 1e-9
        model.Params.IntFeasTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        r = model.addMVar(K, lb=0., ub=bound)
        x = model.addMVar(K, lb=0., ub=bound)
        a = {line: model.addVar(vtype=gp.GRB.BINARY) for line in proper}
        # a line that is off has r = x = 0
        for line in proper:
            model.addConstr(r[where[line]] <= bound*a[line])
            model.addConstr(x[where[line]] <= bound*a[line])
        # crossing lines cannot both be on
        for i in range(len(proper)):
            for j in range(i+1, len(proper)):
                if not compatible(proper[i], proper[j]):
                    model.addConstr(a[proper[i]]+a[proper[j]] <= 1)
        # a tree on |W| meters has at most |W| - 2 internal lines strictly inside W (valid, makes the search faster)
        for W in proper+[tuple(range(n))]:
            inside = [a[line] for line in proper if set(line) < set(W)]
            if inside:
                model.addConstr(gp.quicksum(inside) <= len(W)-2)
        res = model.addMVar(B.shape[0], lb=-gp.GRB.INFINITY)
        model.addConstr(res == B[:, :K]@r+B[:, K:]@x-uy)
        model.setObjective((res@res+rest)/problem.N+problem.alpha*gp.quicksum(a.values()))
        if warm is not None:                             # start from a known tree (the search keeps the best)
            warm_tree = canonical(list(warm)+basic_lines(n))
            theta, _ = problem.fit(warm_tree)
            Kw = len(warm_tree)
            r0, x0 = np.zeros(K), np.zeros(K)
            for k, line in enumerate(warm_tree):
                if line in where:
                    r0[where[line]] = theta[k]
                    x0[where[line]] = theta[Kw+k]
            r.Start, x.Start = r0, x0
            for line in proper:
                a[line].Start = 1. if line in warm_tree else 0.
            res.Start = B@np.r_[r0, x0]-uy
        model.optimize()
        info = dict(status=int(model.Status), seconds=float(model.Runtime), binaries=len(proper),
                    gap=float(model.MIPGap) if model.SolCount else None)
        if model.SolCount == 0:
            return None, info
        chosen = [line for line in proper if a[line].X > .5]
    return canonical(always+chosen), info
