"""The E-stage MIQP with the data rows compressed by QR (exact reformulation), for experiments A and F.

Same variables, constraints, bounds and objective as StructuralSolver.select; only the residual block changes:
||F theta - y||^2 = ||R theta - Q'y||^2 + ||(I - QQ')y||^2 with F = QR. The constant is computed as a vector norm.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from time import perf_counter

import numpy as np

from rnj_wzzt_dr.identification.optimization import StructuralSolver, compatible, feature_matrices, predict
from rnj_wzzt_dr.identification.trees import basis, canonical


class CompressedSolver(StructuralSolver):
    def select(self, pool, keep, warm, *, deadline):
        started = perf_counter()
        n, bound = self.data.n, self.config.response_bound
        family = canonical([*pool, *keep, *basis(n)])
        fixed = set(canonical([*keep, *basis(n)]))
        if not fixed <= set(map(tuple, warm["supports"])):
            raise ValueError("A selection warm start is missing a mandatory support")
        proper = [C for C in family if 1 < len(C) < n]
        with self._model("RTE_laminar_selection_QR", deadline-started) as m:
            r = m.addMVar(len(family), lb=0., ub=bound, name="r")
            x = m.addMVar(len(family), lb=0., ub=bound, name="x")
            a = m.addMVar(len(proper), vtype="B", name="selected")
            positions = {C: i for i, C in enumerate(family)}
            selected = {C: i for i, C in enumerate(proper)}
            for C, j in selected.items():
                i = positions[C]
                m.addConstr(r[i] <= bound*a[j]); m.addConstr(x[i] <= bound*a[j])
                if C in fixed:
                    a[j].LB = 1.
            for C, D in combinations(proper, 2):
                if not compatible(C, D):
                    m.addConstr(a[selected[C]]+a[selected[D]] <= 1.)
            for W in [*proper, tuple(range(n))]:
                inside = [selected[C] for C in proper if set(C) < set(W)]
                if inside:
                    m.addConstr(a[inside].sum() <= len(W)-2)
            scale = self.fitter.residual_scale
            Fr, Fx = feature_matrices(self.data, family)
            F, y = np.column_stack((Fr, Fx))/scale, self.data.y.ravel()/scale
            Q, R = np.linalg.qr(F)
            qy = Q.T@y
            const = float(np.sum((y-Q@qy)**2))
            N = y.size
            K = len(family)
            residual = m.addMVar(len(qy), lb=-float("inf"), name="compressed_residual")
            m.addConstr(residual == R[:, :K]@r+R[:, K:]@x-qy)
            m.setObjective((residual@residual+const)/N+self.config.alpha*a.sum())
            lookup = {tuple(C): (rr, xx) for C, rr, xx in zip(warm["supports"], warm["r"], warm["x"], strict=True)}
            r0 = np.array([lookup.get(C, (0., 0.))[0] for C in family])
            x0 = np.array([lookup.get(C, (0., 0.))[1] for C in family])
            r.Start, x.Start = r0, x0
            a.Start = [float(C in lookup) for C in proper]
            residual.Start = R@np.r_[r0, x0]-qy
            start_residual = ((predict(self.data, warm)-self.data.y)/scale).ravel()
            initial_objective = float(np.mean(start_residual**2))+self.config.alpha*sum(C in lookup for C in proper)

            def decoded():
                indices = [i for i, C in enumerate(family) if C in fixed or a.X[selected[C]] > .5]
                return dict(supports=[list(family[i]) for i in indices],
                            r=np.clip(r.X[indices], 0., bound).tolist(), x=np.clip(x.X[indices], 0., bound).tolist())
            return self._finish(m, started, deadline, scope="finite laminar support pool (QR-compressed)",
                                point_factory=decoded, initial_objective=initial_objective)


class SVDSolver(StructuralSolver):
    """The E MIQP with the rows compressed by a truncated SVD (deviation D1 in RESULTS.md).

    The E pool is rank-deficient by construction (the span of z z' has dimension at most n(n+1)/2, and P and Q
    are nearly collinear on the real feeders). QR then keeps round-off rows, and Gurobi can declare a worse
    point optimal. Here F = U S V' keeps only singular values above rtol*s_max, so
    ||F theta - y||^2 = ||S_k V_k' theta - U_k'y||^2 + ||(I - U_k U_k')y||^2 up to the dropped directions
    (each at most rtol*s_max*|theta|). Variables, bounds, constraints and objective are those of
    StructuralSolver.select. numeric_focus is passed to Gurobi when set.
    """
    rtol = 1e-10
    numeric_focus = None

    def _model(self, name, seconds):
        m = super()._model(name, seconds)
        if self.numeric_focus is not None:
            m.setParam("NumericFocus", self.numeric_focus)
        return m

    def select(self, pool, keep, warm, *, deadline):
        started = perf_counter()
        n, bound = self.data.n, self.config.response_bound
        family = canonical([*pool, *keep, *basis(n)])
        fixed = set(canonical([*keep, *basis(n)]))
        if not fixed <= set(map(tuple, warm["supports"])):
            raise ValueError("A selection warm start is missing a mandatory support")
        proper = [C for C in family if 1 < len(C) < n]
        with self._model("RTE_laminar_selection_SVD", deadline-started) as m:
            r = m.addMVar(len(family), lb=0., ub=bound, name="r")
            x = m.addMVar(len(family), lb=0., ub=bound, name="x")
            a = m.addMVar(len(proper), vtype="B", name="selected")
            positions = {C: i for i, C in enumerate(family)}
            selected = {C: i for i, C in enumerate(proper)}
            for C, j in selected.items():
                i = positions[C]
                m.addConstr(r[i] <= bound*a[j]); m.addConstr(x[i] <= bound*a[j])
                if C in fixed:
                    a[j].LB = 1.
            for C, D in combinations(proper, 2):
                if not compatible(C, D):
                    m.addConstr(a[selected[C]]+a[selected[D]] <= 1.)
            for W in [*proper, tuple(range(n))]:
                inside = [selected[C] for C in proper if set(C) < set(W)]
                if inside:
                    m.addConstr(a[inside].sum() <= len(W)-2)
            scale = self.fitter.residual_scale
            Fr, Fx = feature_matrices(self.data, family)
            F, y = np.column_stack((Fr, Fx))/scale, self.data.y.ravel()/scale
            U, s, Vt = np.linalg.svd(F, full_matrices=False)
            k = int((s > s[0]*self.rtol).sum())
            B, uy = s[:k, None]*Vt[:k], U[:, :k].T@y
            const = float(np.sum((y-U[:, :k]@uy)**2))
            N, K = y.size, len(family)
            residual = m.addMVar(k, lb=-float("inf"), name="svd_residual")
            m.addConstr(residual == B[:, :K]@r+B[:, K:]@x-uy)
            m.setObjective((residual@residual+const)/N+self.config.alpha*a.sum())
            lookup = {tuple(C): (rr, xx) for C, rr, xx in zip(warm["supports"], warm["r"], warm["x"], strict=True)}
            r0 = np.array([lookup.get(C, (0., 0.))[0] for C in family])
            x0 = np.array([lookup.get(C, (0., 0.))[1] for C in family])
            r.Start, x.Start = r0, x0
            a.Start = [float(C in lookup) for C in proper]
            residual.Start = B@np.r_[r0, x0]-uy
            start_residual = ((predict(self.data, warm)-self.data.y)/scale).ravel()
            initial_objective = float(np.mean(start_residual**2))+self.config.alpha*sum(C in lookup for C in proper)
            self.rank = k

            def decoded():
                indices = [i for i, C in enumerate(family) if C in fixed or a.X[selected[C]] > .5]
                return dict(supports=[list(family[i]) for i in indices],
                            r=np.clip(r.X[indices], 0., bound).tolist(), x=np.clip(x.X[indices], 0., bound).tolist())
            return self._finish(m, started, deadline, scope="finite laminar support pool (SVD-compressed)",
                                point_factory=decoded, initial_objective=initial_objective)


class SVDSolverNF2(SVDSolver):
    """The D1 certifier: SVD rows and Gurobi NumericFocus=2."""
    numeric_focus = 2


CERTIFIER = SVDSolverNF2


def solve(fitter, config, pool, keep, warm, seconds, mip_gap=None, cls=None):
    cfg = config if mip_gap is None else replace(config, mip_gap=mip_gap)
    solver = (cls or CompressedSolver)(fitter, cfg)
    try:
        point, record = solver.select(pool, keep, warm, deadline=perf_counter()+seconds)
    finally:
        solver.close()
    return point, record


class SVDSolverNF2Seed7(SVDSolverNF2):
    """The D3 second certifier: as SVDSolverNF2, with Gurobi Seed 7 (a different search path)."""
    def _model(self, name, seconds):
        m = super()._model(name, seconds)
        m.setParam("Seed", 7)
        return m
