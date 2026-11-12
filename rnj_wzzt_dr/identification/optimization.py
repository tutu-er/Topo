"""One bounded-QP fitter and two readable convex MIQP formulations.

The fixed-family QP is bounded least squares, solved by SciPy BVLS. Gurobi is
imported only for structural MIQP searches. No imports from the parent project
or historical experiment snapshots are used.
"""
from __future__ import annotations

from itertools import combinations
from time import perf_counter

import numpy as np
from scipy.optimize import lsq_linear

if __package__:
    from .trees import basis, canonical, compatible, is_laminar
else:
    from trees import basis, canonical, compatible, is_laminar


def response_matrices(point, n):
    z = np.array([[i in C for i in range(n)] for C in point["supports"]], float)
    if not len(z):
        return np.zeros((n, n)), np.zeros((n, n))
    return (z.T*np.asarray(point["r"]))@z, (z.T*np.asarray(point["x"]))@z


def predict(data, point):
    R, X = response_matrices(point, data.n)
    return data.p@R.T + data.q@X.T


def errors(data, point):
    residual = data.y-predict(data, point)
    return dict(mae=float(np.mean(abs(residual))), mse=float(np.mean(residual**2)))


def feature_matrices(data, family):
    """Column C is z_C (z_C^T p_t), with the same ordering as Y.ravel()."""
    z = np.array([[i in C for i in range(data.n)] for C in family], float)
    if not len(z):
        empty = np.empty((data.y.size, 0))
        return empty, empty
    return tuple(np.column_stack([((values@row)[:, None]*row).ravel() for row in z])
                 for values in (data.p, data.q))


def internal_count(point, n):
    return sum(1 < len(C) < n for C in point["supports"])


def criterion(point, n, alpha):
    return float(point["objective"]) + alpha*internal_count(point, n)


class QPFitter:
    """Cache optimal fixed-family fits; retain zero coefficients until cleanup.

    SciPy's BVLS has an iteration limit, not an interruptible wall limit.
    qp_seconds is therefore checked after each fit; this soft limit is recorded
    explicitly. Native MIQP TimeLimit is a hard solver setting.
    """
    def __init__(self, data, *, reference_mse=1., bound=2., qp_seconds=300.):
        if not np.isfinite(reference_mse) or reference_mse <= 0:
            raise ValueError("reference_mse must be finite and strictly positive")
        self.data, self.reference_mse = data, float(reference_mse)
        self.bound, self.qp_seconds = float(bound), float(qp_seconds)
        if self.bound <= 0 or not np.isfinite(self.bound) or self.qp_seconds <= 0:
            raise ValueError("A positive coefficient bound and QP time allowance are required")
        self.cache, self.records = {}, []

    def fit(self, family):
        family = canonical(family)
        if family in self.cache:
            return self.cache[family]
        if not is_laminar(family) or any(not C or min(C) < 0 or max(C) >= self.data.n for C in family):
            raise ValueError("Fixed supports must be a valid laminar family")
        started = perf_counter()
        Fr, Fx = feature_matrices(self.data, family)
        scale = np.sqrt(self.reference_mse)
        F, y = np.column_stack((Fr, Fx))/scale, self.data.y.ravel()/scale
        solved = lsq_linear(F, y, bounds=(0., self.bound), method="bvls", tol=1e-12, max_iter=5000)
        if not solved.success:
            raise RuntimeError("Bounded fixed-family QP did not converge: "+solved.message)
        theta = np.clip(solved.x, 0., self.bound)
        # Normalizing by a small reference MSE amplifies roundoff in a nearly
        # exact fit. One free-variable correction improves numerical stationarity
        # without changing the objective, bounds or statistical regularization.
        F_long, y_long = F.astype(np.longdouble), y.astype(np.longdouble)
        residual_long = F_long@theta.astype(np.longdouble)-y_long
        free = (theta > 1e-12) & (theta < self.bound-1e-12)
        if np.any(free):
            correction = np.linalg.lstsq(F[:, free], -np.asarray(residual_long, float), rcond=None)[0]
            trial = theta.copy(); trial[free] += correction
            if np.all((trial >= 0.) & (trial <= self.bound)):
                trial_residual = F_long@trial.astype(np.longdouble)-y_long
                if np.mean(trial_residual**2) <= np.mean(residual_long**2)+1e-12:
                    theta, residual_long = trial, trial_residual
        residual = np.asarray(residual_long, float)
        objective = float(np.mean(residual**2))
        gradient = np.asarray(2*F_long.T@residual_long/len(y), float)
        lower = max(0., objective+float(np.minimum(-gradient*theta,
                                                  gradient*(self.bound-theta)).sum()))
        size = len(family)
        point = dict(supports=[list(C) for C in family], r=theta[:size].tolist(), x=theta[size:].tolist(),
                     objective=objective, mse=objective*self.reference_mse,
                     reference_mse=self.reference_mse, fit_loss="normalized_MSE")
        elapsed = perf_counter()-started
        record = dict(family=family, solver="SciPy BVLS: bounded convex QP", status=int(solved.status),
                      objective=objective, lower_bound=lower, certificate_gap=objective-lower,
                      seconds=elapsed, iterations=int(solved.nit), rank=int(np.linalg.matrix_rank(F)),
                      coefficient_bound=self.bound, qp_soft_seconds=self.qp_seconds,
                      soft_cap_exceeded=elapsed > self.qp_seconds)
        self.records.append(record)
        if elapsed > self.qp_seconds:
            raise TimeoutError("Fixed QP exceeded its soft wall allowance")
        if objective-lower > 1e-6*(1+objective):
            raise RuntimeError("Bounded-QP supporting-plane certificate gap is too large")
        self.cache[family] = point
        return point


def _gurobi():
    try:
        import gurobipy as gp
        return gp
    except ImportError as exc:
        raise RuntimeError("Structural searches require gurobipy and a valid Gurobi license. "
                           "RNJ/NJ and fixed-family fits work with NumPy/SciPy alone.") from exc


class StructuralSolver:
    """A single lazy Gurobi environment, shared by W/T/E within one method."""
    def __init__(self, fitter, config):
        self.fitter, self.data, self.config = fitter, fitter.data, config
        self.environment, self.records = None, []

    def close(self):
        if self.environment is not None:
            self.environment.dispose()
            self.environment = None

    def _model(self, name, seconds):
        gp = _gurobi()
        if self.environment is None:
            self.environment = gp.Env(empty=True)
            self.environment.setParam("OutputFlag", 0)
            self.environment.start()
        m = gp.Model(name, env=self.environment)
        settings = dict(Threads=1, OutputFlag=0, NonConvex=0, Seed=20261111,
                        TimeLimit=max(.01, seconds), MIPGap=self.config.mip_gap,
                        MIPGapAbs=self.config.mip_abs_gap, FeasibilityTol=1e-9,
                        IntFeasTol=1e-9, OptimalityTol=1e-9)
        for key, value in settings.items():
            m.setParam(key, value)
        return m

    def _finish(self, model, started, deadline, *, scope, point_factory):
        remaining = deadline-perf_counter()
        if remaining <= .01:
            return None, dict(solver="Gurobi convex MIQP", scope=scope, solve_started=False,
                              status=None, bound=None, gap=None, stop_reason="construction_budget")
        model.Params.TimeLimit = remaining
        model.optimize()
        feasible = bool(model.SolCount)
        bound = float(model.ObjBound)
        bound = bound if np.isfinite(bound) and abs(bound) < 1e90 else None
        record = dict(solver="Gurobi convex MIQP", scope=scope, solve_started=True,
                      status=int(model.Status), objective=float(model.ObjVal) if feasible else None,
                      bound=bound, gap=float(model.MIPGap) if feasible else None,
                      requested_gap=self.config.mip_gap, requested_absolute_gap=self.config.mip_abs_gap,
                      requested_solve_seconds=remaining, seconds=perf_counter()-started,
                      nodes=float(model.NodeCount), variables=int(model.NumVars),
                      binary_variables=int(model.NumBinVars), constraints=int(model.NumConstrs),
                      constraint_violation=float(model.ConstrVio) if feasible else None,
                      bound_violation=float(model.BoundVio) if feasible else None,
                      integrality_violation=float(model.IntVio) if feasible else None)
        point = point_factory() if feasible else None
        if point is not None:
            max_violation = max(record[k] for k in ("constraint_violation", "bound_violation", "integrality_violation"))
            if max_violation > 1e-7:
                raise RuntimeError("MIQP incumbent violates the declared constraints")
            actual = errors(self.data, point)["mse"]/self.fitter.reference_mse
            actual += self.config.alpha*internal_count(point, self.data.n)
            if abs(actual-record["objective"]) > 1e-7*(1+abs(actual)):
                raise RuntimeError("MIQP objective fails raw-observation replay")
        self.records.append(record)
        return point, record

    def select(self, pool, keep, warm, *, deadline):
        """Shared T/E: enumerate supports, select a laminar subset with binaries."""
        started = perf_counter()
        n, bound = self.data.n, self.config.response_bound
        family = canonical([*pool, *keep, *basis(n)])
        fixed = set(canonical([*keep, *basis(n)]))
        if not fixed <= set(map(tuple, warm["supports"])):
            raise ValueError("A selection warm start is missing a mandatory support")
        proper = [C for C in family if 1 < len(C) < n]
        if not fixed <= set(family):
            raise ValueError("Mandatory supports are missing from the selection pool")
        with self._model("RTE_laminar_selection", deadline-started) as m:
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
            Fr, Fx = feature_matrices(self.data, family)
            residual = m.addMVar(self.data.y.size, lb=-float("inf"), name="scaled_residual")
            m.addConstr(residual == (Fr@r+Fx@x-self.data.y.ravel())/np.sqrt(self.fitter.reference_mse))
            m.setObjective((residual@residual)/self.data.y.size+self.config.alpha*a.sum())
            lookup = {tuple(C): (rr, xx) for C, rr, xx in zip(warm["supports"], warm["r"], warm["x"], strict=True)}
            r.Start = [lookup.get(C, (0., 0.))[0] for C in family]
            x.Start = [lookup.get(C, (0., 0.))[1] for C in family]
            a.Start = [float(C in lookup) for C in proper]
            residual.Start = ((predict(self.data, warm)-self.data.y)/np.sqrt(self.fitter.reference_mse)).ravel()

            def decoded():
                indices = [i for i, C in enumerate(family) if C in fixed or a.X[selected[C]] > .5]
                return dict(supports=[list(family[i]) for i in indices],
                            r=np.clip(r.X[indices], 0., bound).tolist(), x=np.clip(x.X[indices], 0., bound).tolist())
            return self._finish(m, started, deadline, scope="finite laminar support pool", point_factory=decoded)

    def extend(self, current, candidate, warm, *, deadline):
        """Original WZZT: one unknown support, exact linearized products.

        z selects terminals; g_ij=z_i*z_j; uR_ij=r_new*g_ij and
        uX_ij=x_new*g_ij. All constraints are linear and the residual objective
        is positive semidefinite quadratic. The domain is not an RNJ shortlist.
        """
        started = perf_counter()
        family, n, bound = canonical(current["supports"]), self.data.n, self.config.response_bound
        proper = [C for C in family if 1 < len(C) < n]
        with self._model("WZZT_unknown_support", deadline-started) as m:
            r = m.addMVar(len(family), lb=0., ub=bound, name="old_r")
            x = m.addMVar(len(family), lb=0., ub=bound, name="old_x")
            z = m.addMVar(n, vtype="B", name="new_support")
            active = m.addVar(vtype="B", name="activate_new")
            nr, nx = m.addVar(lb=0., ub=bound, name="new_r"), m.addVar(lb=0., ub=bound, name="new_x")
            m.addConstr(nr <= bound*active); m.addConstr(nx <= bound*active)
            m.addConstr(z.sum() >= 2); m.addConstr(z.sum() <= n-1)
            relations = []
            for C in proper:
                inside, outside = list(C), [i for i in range(n) if i not in C]
                relation = m.addMVar(3, vtype="B", name="laminar_relation")
                relations.append((C, relation))
                m.addConstr(relation.sum() == 1)
                m.addConstr(z[inside].sum() <= len(inside)*(1-relation[0]))       # disjoint
                m.addConstr(z[outside].sum() <= len(outside)*(1-relation[1]))    # new subset
                m.addConstr(z[inside].sum() >= len(inside)*relation[2])          # old subset
                m.addConstr(len(C)-z[inside].sum()+z[outside].sum() >= 1)        # distinct
            g = m.addMVar((n, n), lb=0., ub=1., name="z_outer_z")
            ur = m.addMVar((n, n), lb=0., ub=bound, name="r_times_outer")
            ux = m.addMVar((n, n), lb=0., ub=bound, name="x_times_outer")
            m.addConstr(g <= z[:, None]); m.addConstr(g <= z[None, :])
            m.addConstr(g >= z[:, None]+z[None, :]-1)
            m.addConstr(g == g.T)
            for weights, scalar in ((ur, nr), (ux, nx)):
                m.addConstr(weights <= scalar)
                m.addConstr(weights <= bound*g)
                m.addConstr(weights >= scalar-bound*(1-g))
                m.addConstr(weights == weights.T)
            Fr, Fx = feature_matrices(self.data, family)
            old_prediction = m.addMVar(self.data.y.shape, lb=-float("inf"), name="old_prediction")
            m.addConstr(old_prediction.reshape(-1) == Fr@r+Fx@x)
            prediction = old_prediction+self.data.p@ur.T+self.data.q@ux.T
            residual = m.addMVar(self.data.y.shape, lb=-float("inf"), name="scaled_residual")
            m.addConstr(residual == (prediction-self.data.y)/np.sqrt(self.fitter.reference_mse))
            flat = residual.reshape(-1)
            m.setObjective((flat@flat)/self.data.y.size+self.config.alpha*(len(proper)+active))
            lookup = {tuple(C): (rr, xx) for C, rr, xx in zip(warm["supports"], warm["r"], warm["x"], strict=True)}
            r.Start = [lookup.get(C, (0., 0.))[0] for C in family]
            x.Start = [lookup.get(C, (0., 0.))[1] for C in family]
            zz = np.array([i in candidate for i in range(n)], float)
            rr, xx = lookup.get(tuple(candidate), (0., 0.))
            z.Start, g.Start = zz, np.outer(zz, zz)
            nr.Start, nx.Start, active.Start = rr, xx, float(tuple(candidate) in lookup)
            ur.Start, ux.Start = rr*np.outer(zz, zz), xx*np.outer(zz, zz)
            residual.Start = (predict(self.data, warm)-self.data.y)/np.sqrt(self.fitter.reference_mse)
            for C, relation in relations:
                which = 0 if not set(C) & set(candidate) else 1 if set(candidate) < set(C) else 2
                relation.Start = np.eye(3)[which]

            def decoded():
                supports = list(family)
                rr, xx = r.X.tolist(), x.X.tolist()
                if active.X > .5:
                    C = tuple(np.flatnonzero(z.X > .5).tolist())
                    if C in family or not 2 <= len(C) < n or not all(compatible(C, D) for D in family):
                        raise RuntimeError("Invalid decoded WZZT support")
                    supports.append(C); rr.append(float(nr.X)); xx.append(float(nx.X))
                return dict(supports=[list(C) for C in supports],
                            r=np.clip(rr, 0., bound).tolist(), x=np.clip(xx, 0., bound).tolist())
            return self._finish(m, started, deadline, scope="one unrestricted compatible support", point_factory=decoded)
