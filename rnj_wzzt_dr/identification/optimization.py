"""Bounded fixed-support fits and shared MSE/MAE structural formulations.

MSE fixed-family fits use SciPy BVLS; MAE fits use a sparse HiGHS LP. Gurobi is
imported only for structural MIQP/MILP searches. No imports from the parent
project or historical experiment snapshots are used.
"""
from __future__ import annotations

from itertools import combinations
from time import perf_counter

import numpy as np
from scipy.optimize import lsq_linear, linprog, OptimizeWarning
from scipy import sparse
import warnings

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


def validate_loss(loss):
    if loss not in ("mse", "mae"):
        raise ValueError("loss must be 'mse' or 'mae'")
    return loss


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) and abs(value) < 1e90 else None


def _gap(objective, bound):
    if objective is None or bound is None:
        return None
    if objective == 0.:
        return 0. if bound == 0. else None
    return _finite(abs(objective-bound)/abs(objective))


class QPFitter:
    """Cache optimal fixed-family fits; retain zero coefficients until cleanup.

    SciPy's BVLS has an iteration limit, not an interruptible wall limit.
    qp_seconds is therefore checked after each fit; this soft limit is recorded
    explicitly. HiGHS receives the remaining fixed-fit time, with construction
    and certificate checks included in the outer wall check. The legacy
    qp_seconds name is retained for both losses. Native MIP has TimeLimit.
    """
    def __init__(self, data, *, reference_mse=1., bound=2., qp_seconds=300.,
                 loss="mse", reference_loss=None):
        if not np.isfinite(reference_mse) or reference_mse <= 0:
            raise ValueError("reference_mse must be finite and strictly positive")
        self.data, self.reference_mse = data, float(reference_mse)
        self.loss = validate_loss(loss)
        self.reference_loss = float(reference_mse if reference_loss is None else reference_loss)
        if not np.isfinite(self.reference_loss) or self.reference_loss <= 0.:
            raise ValueError("reference_loss must be finite and strictly positive")
        self.bound, self.qp_seconds = float(bound), float(qp_seconds)
        if self.bound <= 0 or not np.isfinite(self.bound) or self.qp_seconds <= 0:
            raise ValueError("A positive coefficient bound and QP time allowance are required")
        self.cache, self.records = {}, []

    @property
    def residual_scale(self):
        return np.sqrt(self.reference_loss) if self.loss == "mse" else self.reference_loss

    def normalized_loss(self, point):
        return errors(self.data, point)[self.loss]/self.reference_loss

    def fit(self, family):
        family = canonical(family)
        if family in self.cache:
            return self.cache[family]
        if not is_laminar(family) or any(not C or min(C) < 0 or max(C) >= self.data.n for C in family):
            raise ValueError("Fixed supports must be a valid laminar family")
        started = perf_counter()
        if not family:
            values = dict(mse=float(np.mean(self.data.y**2)), mae=float(np.mean(abs(self.data.y))))
            objective = values[self.loss]/self.reference_loss
            point = dict(supports=[], r=[], x=[], objective=objective, **values,
                reference_mse=self.reference_mse, reference_loss=self.reference_loss,
                loss_name=self.loss, fit_loss="normalized_"+self.loss.upper())
            self.records.append(dict(family=(), solver="closed_form_empty_support", status="complete",
                objective=objective, lower_bound=objective, certificate_gap=0., seconds=0.,
                coefficient_bound=self.bound, rank=0, iterations=0, loss_name=self.loss))
            self.cache[family] = point
            return point
        if self.loss == "mae":
            return self._fit_mae(family, started)
        Fr, Fx = feature_matrices(self.data, family)
        scale = np.sqrt(self.reference_loss)
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
                     objective=objective, mse=objective*self.reference_loss,
                     mae=float(np.mean(abs(residual)))*scale,
                     reference_mse=self.reference_mse, reference_loss=self.reference_loss,
                     loss_name=self.loss, fit_loss="normalized_MSE")
        elapsed = perf_counter()-started
        record = dict(family=family, solver="SciPy BVLS: bounded convex QP", status=int(solved.status),
                      objective=objective, lower_bound=lower, certificate_gap=objective-lower,
                      seconds=elapsed, iterations=int(solved.nit), rank=int(np.linalg.matrix_rank(F)),
                      coefficient_bound=self.bound, qp_soft_seconds=self.qp_seconds,
                      soft_cap_exceeded=elapsed > self.qp_seconds, loss_name=self.loss)
        self.records.append(record)
        if elapsed > self.qp_seconds:
            raise TimeoutError("Fixed QP exceeded its soft wall allowance")
        if objective-lower > 1e-6*(1+objective):
            raise RuntimeError("Bounded-QP supporting-plane certificate gap is too large")
        self.cache[family] = point
        return point

    def _fit_mae(self, family, started):
        Fr, Fx = feature_matrices(self.data, family)
        F = np.column_stack((Fr, Fx))/self.reference_loss
        y = self.data.y.ravel()/self.reference_loss
        cells, columns = F.shape
        identity = sparse.eye(cells, format="csc")
        design = sparse.csc_matrix(F)
        A = sparse.bmat([[design, -identity], [-design, -identity]], format="csc")
        b = np.r_[y, -y]
        cost = np.r_[np.zeros(columns), np.full(cells, 1./cells)]
        bounds = [(0., self.bound)]*columns+[(0., None)]*cells
        remaining = self.qp_seconds-(perf_counter()-started)
        if remaining <= 0.:
            self.records.append(dict(family=family, solver="SciPy HiGHS dual simplex: bounded LP",
                status="construction_budget", seconds=perf_counter()-started, loss_name="mae",
                qp_soft_seconds=self.qp_seconds, soft_cap_exceeded=True))
            raise TimeoutError("Fixed MAE LP construction exhausted its wall allowance")
        options = dict(time_limit=remaining, primal_feasibility_tolerance=1e-9,
                       dual_feasibility_tolerance=1e-9, threads=1)
        lp_started = perf_counter()
        with warnings.catch_warnings(record=True) as solver_warnings:
            # SciPy forwards threads to HiGHS; preserve its forwarding warning
            # instead of silently discarding it or changing the process pool.
            warnings.simplefilter("always", OptimizeWarning)
            solved = linprog(cost, A_ub=A, b_ub=b, bounds=bounds, method="highs-ds", options=options)
        elapsed = perf_counter()-started
        record = dict(family=family, solver="SciPy HiGHS dual simplex: bounded LP", status=int(solved.status),
            solver_message=str(solved.message), seconds=elapsed, iterations=int(solved.nit),
            coefficient_bound=self.bound, qp_soft_seconds=self.qp_seconds,
            soft_cap_exceeded=elapsed > self.qp_seconds, loss_name="mae", options=options,
            native_seconds=perf_counter()-lp_started,
            option_warnings=[str(w.message) for w in solver_warnings],
            forwarded_highs_options=dict(threads=1),
            optimum_may_be_nonunique=True, tie_break_regularization=False)
        self.records.append(record)
        if not solved.success:
            if solved.status == 1:
                raise TimeoutError("Fixed MAE LP reached its limit: "+solved.message)
            raise RuntimeError("Fixed MAE LP did not converge: "+solved.message)
        if elapsed > self.qp_seconds:
            raise TimeoutError("Fixed MAE LP exceeded its wall allowance including construction")
        theta = np.clip(solved.x[:columns], 0., self.bound)
        size = len(family)
        point = dict(supports=[list(C) for C in family], r=theta[:size].tolist(), x=theta[size:].tolist())
        values = errors(self.data, point)
        objective = values["mae"]/self.reference_loss
        primal_violation = float(max(0., np.max(A@solved.x-b), -np.min(solved.x),
                                    np.max(solved.x[:columns]-self.bound)))
        dual, lo, hi = (np.asarray(solved[key].marginals) for key in ("ineqlin", "lower", "upper"))
        stationarity = cost-A.T@dual-lo-hi
        dual_violation = float(max(0., np.max(dual), -np.min(lo), np.max(hi),
                                  np.max(abs(hi[columns:]))))
        dual_objective = float(b@dual+self.bound*hi[:columns].sum())
        # Independent weak-duality witness: |s_t|<=1/cells implies
        # mean|F theta-y| >= -s'y + sum_k min(0, bound*(F's)_k).
        # Clipping only the dual witness guarantees its box, without changing
        # the LP point or introducing a coefficient tie-break.
        s = np.clip(dual[cells:]-dual[:cells], -1./cells, 1./cells)
        Fl, yl, sl = F.astype(np.longdouble), y.astype(np.longdouble), s.astype(np.longdouble)
        lower = max(0., float(-sl@yl+np.minimum(0., self.bound*(Fl.T@sl)).sum()))
        residual = F@theta-y
        signs = np.sign(residual)/cells
        changed = np.flatnonzero(s != signs)
        tolerance = 1e-7*(1.+abs(objective))
        audit = dict(primal_violation=primal_violation, dual_sign_violation=dual_violation,
            dual_stationarity_max_abs=float(np.max(abs(stationarity))),
            primal_objective=float(cost@solved.x), solver_objective=float(solved.fun),
            dual_objective=dual_objective, raw_primal_dual_gap=float(solved.fun)-dual_objective,
            direct_objective=objective, lower_bound=lower, certificate_gap=objective-lower,
            objective_tolerance=tolerance,
            dual_witness_base="sign((F/reference_loss)@theta-y/reference_loss)/residual_cells",
            dual_witness_adjustments=[[int(i),float(s[i])] for i in changed])
        record.update(objective=objective, lower_bound=lower, certificate_gap=objective-lower,
                      certificate=audit, rank=int(np.linalg.matrix_rank(F)))
        if (not np.isfinite([*values.values(), objective, lower, primal_violation, dual_violation,
                             audit['dual_stationarity_max_abs'], dual_objective]).all()
                or primal_violation > 1e-7 or dual_violation > 1e-7
                or audit['dual_stationarity_max_abs'] > 1e-7
                or abs(objective-solved.fun) > tolerance
                or abs(solved.fun-dual_objective) > tolerance
                or abs(objective-lower) > tolerance):
            record['certificate_passed'] = False
            raise RuntimeError("Fixed MAE LP failed independent primal/dual or objective replay checks")
        record['certificate_passed'] = True
        point.update(objective=objective, **values, reference_mse=self.reference_mse,
            reference_loss=self.reference_loss, loss_name="mae", fit_loss="normalized_MAE")
        record['seconds'] = perf_counter()-started
        record['soft_cap_exceeded'] = record['seconds'] > self.qp_seconds
        if record['soft_cap_exceeded']:
            raise TimeoutError("Fixed MAE LP including certificate checks exceeded its wall allowance")
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
        self.progress = None
        self.progress_interval = 10.
        if getattr(config, 'loss', 'mse') != fitter.loss:
            raise ValueError("Structural solver and fixed-family fitter use different training losses")

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

    def _finish(self, model, started, deadline, *, scope, point_factory, initial_objective=None):
        solver_name = "Gurobi convex MIQP" if self.fitter.loss == "mse" else "Gurobi MILP"
        remaining = deadline-perf_counter()
        if remaining <= .01:
            return None, dict(solver=solver_name, scope=scope, solve_started=False,
                              status=None, bound=None, gap=None, stop_reason="construction_budget",
                              initial_objective=initial_objective, loss_name=self.fitter.loss,
                              progress=[], progress_errors=[])
        model.Params.TimeLimit = remaining
        interval = float(self.progress_interval)
        if not np.isfinite(interval) or interval <= 0.:
            raise ValueError("Native progress interval must be finite and positive")
        trace, callback_errors = [], []
        last_periodic = [-float("inf")]
        callback_codes = _gurobi().GRB.Callback

        def publish(row):
            trace.append(row)
            if self.progress is not None:
                try:
                    self.progress(dict(row))
                except Exception as exc:
                    callback_errors.append(type(exc).__name__+": "+str(exc))

        def progress_callback(m, where):
            if where not in (callback_codes.MIP, callback_codes.MIPSOL):
                return
            try:
                runtime = float(m.cbGet(callback_codes.RUNTIME))
                if where == callback_codes.MIP and runtime-last_periodic[0] < interval:
                    return
                prefix = "MIP" if where == callback_codes.MIP else "MIPSOL"
                best = _finite(m.cbGet(getattr(callback_codes, prefix+"_OBJBST")))
                candidate = _finite(m.cbGet(callback_codes.MIPSOL_OBJ)) if prefix == "MIPSOL" else None
                if candidate is not None:
                    best = candidate if best is None else min(best, candidate)
                bound = _finite(m.cbGet(getattr(callback_codes, prefix+"_OBJBND")))
                count = _finite(m.cbGet(getattr(callback_codes, prefix+"_SOLCNT")))
                publish(dict(callback_event=prefix, model_name=str(m.ModelName), scope=scope,
                    seconds=_finite(perf_counter()-started), in_solver_seconds=_finite(runtime),
                    objective=best, candidate_objective=candidate, bound=bound, gap=_gap(best,bound),
                    nodes=_finite(m.cbGet(getattr(callback_codes, prefix+"_NODCNT"))),
                    incumbent_count=int(count) if count is not None else None,
                    incumbent_count_scope="native callback SOLCNT; current MIPSOL may not yet be counted"))
                last_periodic[0] = runtime
            except Exception as exc:
                callback_errors.append(type(exc).__name__+": "+str(exc))

        model.optimize(progress_callback)
        feasible = bool(model.SolCount)
        bound = _finite(model.ObjBound)
        objective = _finite(model.ObjVal) if feasible else None
        publish(dict(callback_event="FINAL", model_name=str(model.ModelName), scope=scope,
            seconds=_finite(perf_counter()-started), in_solver_seconds=_finite(model.Runtime),
            objective=objective, bound=bound, gap=_finite(model.MIPGap) if feasible else None,
            nodes=_finite(model.NodeCount), incumbent_count=int(model.SolCount), status=int(model.Status)))
        record = dict(solver=solver_name, scope=scope, solve_started=True,
                      status=int(model.Status), objective=objective,
                      bound=bound, gap=_finite(model.MIPGap) if feasible else None,
                      requested_gap=self.config.mip_gap, requested_absolute_gap=self.config.mip_abs_gap,
                      requested_solve_seconds=remaining, seconds=perf_counter()-started,
                      nodes=float(model.NodeCount), variables=int(model.NumVars),
                      binary_variables=int(model.NumBinVars), constraints=int(model.NumConstrs),
                      constraint_violation=float(model.ConstrVio) if feasible else None,
                      bound_violation=float(model.BoundVio) if feasible else None,
                      integrality_violation=float(model.IntVio) if feasible else None,
                      loss_name=self.fitter.loss, reference_loss=self.fitter.reference_loss,
                      initial_objective=initial_objective,
                      initial_to_returned_gain=(initial_objective-objective
                          if initial_objective is not None and objective is not None else None),
                      progress=trace, progress_interval_seconds=interval, progress_errors=callback_errors)
        self.records.append(record)
        point = point_factory() if feasible else None
        if point is not None:
            max_violation = max(record[k] for k in ("constraint_violation", "bound_violation", "integrality_violation"))
            if max_violation > 1e-7:
                raise RuntimeError("Native structural incumbent violates the declared constraints")
            values = errors(self.data, point)
            actual = values[self.fitter.loss]/self.fitter.reference_loss
            actual += self.config.alpha*internal_count(point, self.data.n)
            if abs(actual-record["objective"]) > 1e-7*(1+abs(actual)):
                raise RuntimeError("Native objective fails raw-observation replay")
            point.update(**values, objective=values[self.fitter.loss]/self.fitter.reference_loss,
                reference_mse=self.fitter.reference_mse, reference_loss=self.fitter.reference_loss,
                loss_name=self.fitter.loss, fit_loss="normalized_"+self.fitter.loss.upper())
            record.update(raw_training_errors=values, objective_replay=actual, objective_replay_passed=True)
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
            scale = self.fitter.residual_scale
            m.addConstr(residual == (Fr@r+Fx@x-self.data.y.ravel())/scale)
            absolute = None
            if self.fitter.loss == "mse":
                m.setObjective((residual@residual)/self.data.y.size+self.config.alpha*a.sum())
            else:
                absolute = m.addMVar(self.data.y.size, lb=0., name="absolute_residual")
                m.addConstr(absolute >= residual); m.addConstr(absolute >= -residual)
                m.setObjective(absolute.sum()/self.data.y.size+self.config.alpha*a.sum())
            lookup = {tuple(C): (rr, xx) for C, rr, xx in zip(warm["supports"], warm["r"], warm["x"], strict=True)}
            r.Start = [lookup.get(C, (0., 0.))[0] for C in family]
            x.Start = [lookup.get(C, (0., 0.))[1] for C in family]
            a.Start = [float(C in lookup) for C in proper]
            start_residual = ((predict(self.data, warm)-self.data.y)/scale).ravel()
            residual.Start = start_residual
            if absolute is not None:
                absolute.Start = abs(start_residual)
            initial_objective = float(np.mean(start_residual**2) if self.fitter.loss == "mse"
                                      else np.mean(abs(start_residual)))
            initial_objective += self.config.alpha*sum(C in lookup for C in proper)

            def decoded():
                indices = [i for i, C in enumerate(family) if C in fixed or a.X[selected[C]] > .5]
                return dict(supports=[list(family[i]) for i in indices],
                            r=np.clip(r.X[indices], 0., bound).tolist(), x=np.clip(x.X[indices], 0., bound).tolist())
            return self._finish(m, started, deadline, scope="finite laminar support pool",
                                point_factory=decoded, initial_objective=initial_objective)

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
            scale = self.fitter.residual_scale
            m.addConstr(residual == (prediction-self.data.y)/scale)
            flat = residual.reshape(-1)
            absolute = None
            if self.fitter.loss == "mse":
                m.setObjective((flat@flat)/self.data.y.size+self.config.alpha*(len(proper)+active))
            else:
                absolute = m.addMVar(self.data.y.size, lb=0., name="absolute_residual")
                m.addConstr(absolute >= flat); m.addConstr(absolute >= -flat)
                m.setObjective(absolute.sum()/self.data.y.size+self.config.alpha*(len(proper)+active))
            lookup = {tuple(C): (rr, xx) for C, rr, xx in zip(warm["supports"], warm["r"], warm["x"], strict=True)}
            r.Start = [lookup.get(C, (0., 0.))[0] for C in family]
            x.Start = [lookup.get(C, (0., 0.))[1] for C in family]
            zz = np.array([i in candidate for i in range(n)], float)
            rr, xx = lookup.get(tuple(candidate), (0., 0.))
            z.Start, g.Start = zz, np.outer(zz, zz)
            nr.Start, nx.Start, active.Start = rr, xx, float(tuple(candidate) in lookup)
            ur.Start, ux.Start = rr*np.outer(zz, zz), xx*np.outer(zz, zz)
            start_residual = (predict(self.data, warm)-self.data.y)/scale
            residual.Start = start_residual
            if absolute is not None:
                absolute.Start = abs(start_residual).ravel()
            initial_objective = float(np.mean(start_residual**2) if self.fitter.loss == "mse"
                                      else np.mean(abs(start_residual)))
            initial_objective += self.config.alpha*(len(proper)+int(tuple(candidate) in lookup))
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
            return self._finish(m, started, deadline, scope="one unrestricted compatible support",
                                point_factory=decoded, initial_objective=initial_objective)
