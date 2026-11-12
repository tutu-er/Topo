"""Isolated convex squared-loss backend for the frozen RNJ/WZZT structures.

The native residual inequalities, lifted products and hierarchy constraints are
unchanged.  Their nonnegative residual epigraph e is minimized quadratically.
For data scale S=1000 and positive c=f0_MAE/f0_MSE, the native objective is

    c/(m*S) * e'e + native linear count costs
    = S * (c*MSE + alpha*f0_MAE*internal_count).

Consequently the native ObjVal/S convention and dimensionless acceptance rules
remain valid.  Every fixed-family closure here is a fresh QP; no LP cache is
accepted.  This module never reads truth, test data or future DR observations.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

SCALE = 1000.
_LOSS_SCALE = 1.
_CONFIGURED = False


def canonical(family):
    return tuple(sorted({tuple(sorted(map(int, support))) for support in family}, key=lambda s: (len(s), s)))


def _compatible(left, right):
    a, b = set(left), set(right)
    return not a & b or a <= b or b <= a


def configure(K):
    """Configure one input's positive normalization; repeated same K is allowed."""
    global _LOSS_SCALE, _CONFIGURED
    K = float(K)
    if not math.isfinite(K) or K <= 0:
        raise ValueError("A finite positive f0_MAE/f0_MSE normalization is required")
    if _CONFIGURED and K != _LOSS_SCALE:
        raise RuntimeError("One worker process cannot change its squared-loss normalization")
    _LOSS_SCALE, _CONFIGURED = K, True
    return K


def _matrices(n, point):
    indicators = np.array([[i in support for i in range(n)] for support in point["supports"]], float)
    if indicators.size == 0:
        return np.zeros((n, n)), np.zeros((n, n))
    return ((indicators.T * np.asarray(point["r"])) @ indicators,
            (indicators.T * np.asarray(point["x"])) @ indicators)


def mse(data, point):
    R, X = _matrices(data.n, point)
    return sum(float(np.square(y-p@R.T-q@X.T).sum()) for p, q, y in
               zip(data.p, data.q, data.target)) / data.observation_count


def mae(data, point):
    R, X = _matrices(data.n, point)
    return sum(float(np.abs(y-p@R.T-q@X.T).sum()) for p, q, y in
               zip(data.p, data.q, data.target)) / data.observation_count


def loss(data, point):
    return _LOSS_SCALE*mse(data, point)


class PoolQP:
    """Reusable convex QP with inactive weights fixed to zero, and QP-only cache."""
    def __init__(self, data, pool, *, loss_scale, bound=2., seconds=300.):
        import gurobipy as gp
        self.gp = gp
        self.p, self.q, self.y = (np.asarray(value, float) for value in data)
        self.n = self.p.shape[1]
        if self.p.shape != self.q.shape or self.p.shape != self.y.shape:
            raise ValueError("P/Q/Y shape mismatch")
        self.pool, self.bound = canonical(pool), float(bound)
        self.lookup = {support: i for i, support in enumerate(self.pool)}
        self.loss_scale = float(loss_scale)
        if self.bound != 2. or not math.isfinite(self.loss_scale) or self.loss_scale <= 0:
            raise ValueError("This matched benchmark uses bounds [0,2] and a positive loss scale")
        self.cache, self.solves, self.preprocess = {}, [], "raw"
        z = np.asarray([[i in support for i in range(self.n)] for support in self.pool], float)
        self.fr = np.column_stack([((self.p@row)[:, None]*row).ravel() for row in z])
        self.fx = np.column_stack([((self.q@row)[:, None]*row).ravel() for row in z])
        self.features = np.column_stack((self.fr, self.fx))
        self.environment = gp.Env(empty=True)
        self.environment.setParam("OutputFlag", 0); self.environment.start()
        self.model = gp.Model("candidate_family_squared", env=self.environment)
        for key, value in dict(Threads=1, Method=1, FeasibilityTol=1e-9, OptimalityTol=1e-9,
                               TimeLimit=float(seconds), NonConvex=0).items():
            self.model.setParam(key, value)
        self.weights = self.model.addMVar(2*len(self.pool), lb=0., ub=0.)
        self.t = self.model.addMVar(self.y.size, lb=0., name="absolute_epigraph_for_squared_loss")
        feature = csc_matrix(self.features)
        self.model.addConstr(feature@self.weights-self.t <= self.y.ravel())
        self.model.addConstr(feature@self.weights+self.t >= self.y.ravel())
        self.quadratic_coefficient = self.loss_scale/(self.y.size*SCALE)
        self.model.setObjective(self.quadratic_coefficient*(self.t@self.t), gp.GRB.MINIMIZE)

    def fit(self, family):
        family = canonical(family)
        if family in self.cache:
            return self.cache[family]
        if any(not _compatible(a, b) for i, a in enumerate(family) for b in family[:i]):
            raise ValueError("Nonlaminar fixed family passed to QP")
        started = perf_counter(); size = len(self.pool)
        active = np.asarray([self.lookup[support] for support in family], int)
        upper = np.zeros(2*size)
        upper[active] = self.bound; upper[active+size] = self.bound
        self.weights.UB = upper
        self.model.optimize()
        if self.model.Status != self.gp.GRB.OPTIMAL:
            raise RuntimeError(f"Fixed-family QP did not reach optimality: status {self.model.Status}")
        raw = np.asarray(self.weights.X)
        values = np.clip(raw, 0., self.bound); values[upper == 0.] = 0.
        prediction = self.features@values
        physical_mse = float(np.mean(np.square((prediction-self.y.ravel())/SCALE)))
        normalized = self.loss_scale*physical_mse
        native_objective = float(self.model.ObjVal)
        if not np.isclose(native_objective/SCALE, normalized, atol=1e-8, rtol=1e-8):
            raise RuntimeError("QP physical squared-loss reproduction failed")
        r, x = values[active], values[active+size]
        kept = [i for i, support in enumerate(family) if len(support) == 1 or max(r[i], x[i]) > 1e-9]
        point = dict(supports=[list(family[i]) for i in kept], r=r[kept].tolist(), x=x[kept].tolist(),
                     objective=native_objective, physical_mse=physical_mse, normalized_squared_loss=normalized,
                     solver_objective=native_objective, loss_scale=self.loss_scale, preprocess="raw", fixed_fit="QP")
        gradient = 2*self.loss_scale*(self.features/SCALE).T@((prediction-self.y.ravel())/SCALE)/self.y.size
        violation = []
        for i in np.r_[active, active+size]:
            if values[i] <= 1e-8:
                violation.append(max(0., -gradient[i]))
            elif values[i] >= self.bound-1e-8:
                violation.append(max(0., gradient[i]))
            else:
                violation.append(abs(gradient[i]))
        audit = dict(family=family, objective=native_objective, normalized_squared_loss=normalized,
                     physical_mse=physical_mse, max_constraint_violation=float(self.model.ConstrVio),
                     max_bound_violation=float(self.model.BoundVio), projected_gradient_max=float(max(violation, default=0.)),
                     coefficient_projection_max=float(np.max(np.abs(raw-values), initial=0.)),
                     status=int(self.model.Status), seconds=perf_counter()-started, solver="gurobi_QP",
                     objective_scale=SCALE, loss_scale=self.loss_scale, quadratic_coefficient=self.quadratic_coefficient,
                     ridge=0., positive_impedance_floor=0., cache_kind="QP only")
        self.solves.append(audit); self.cache[family] = point
        return point

    def close(self):
        self.model.dispose(); self.environment.dispose()


class SquaredBank:
    """FloorBank-compatible QP bank; no positive floor or imported L1 cache."""
    def __init__(self, data, pool, calibration, *, seconds=300., floor=0., mode="depth", reference=1.):
        from rnj_wzzt.estimation import laminar_l1_milp as core
        if float(floor) != 0.:
            raise ValueError("The objective-only comparison does not add a response floor")
        self.data, self.calibration = data, calibration
        self.seconds, self.floor, self.mode, self.reference = float(seconds), 0., mode, reference
        self.loss_scale = _LOSS_SCALE
        self.scaled = core._PreparedScenarios(data.labels, tuple(p*SCALE for p in data.p),
                                              tuple(q*SCALE for q in data.q), tuple(y*SCALE for y in data.target))
        self.failures = []
        self.lp = self._make(pool)

    def _make(self, pool):
        return PoolQP((np.vstack(self.scaled.p), np.vstack(self.scaled.q), np.vstack(self.scaled.target)),
                      pool, loss_scale=self.loss_scale, seconds=self.seconds)

    def extend(self, supports):
        missing = set(canonical(supports))-set(self.lp.pool)
        if not missing:
            return
        cache, solves = self.lp.cache, self.lp.solves
        other = self._make([*self.lp.pool, *missing])
        self.lp.close(); self.lp = other
        self.lp.cache.update(cache); self.lp.solves = solves

    def fit(self, family, *, preserve=False):
        family = canonical(family); self.extend(family)
        try:
            point = self.lp.fit(family)
        except Exception as exc:
            self.failures.append(dict(family=family, status=int(self.lp.model.Status), error=str(exc), fixed_fit="QP"))
            raise
        if preserve:
            lookup = {tuple(support): (r, x) for support, r, x in zip(point["supports"], point["r"], point["x"])}
            result = dict(supports=family, r=[lookup.get(support, (0., 0.))[0] for support in family],
                          x=[lookup.get(support, (0., 0.))[1] for support in family])
        else:
            result = dict(supports=canonical(point["supports"]), r=list(point["r"]), x=list(point["x"]))
        result.update(objective=self.loss_scale*mse(self.data, result), mse=mse(self.data, result), mae=mae(self.data, result),
                      loss_metric="normalized_mse", loss_scale=self.loss_scale, fixed_fit="QP")
        if abs(result["objective"]-point["objective"]/SCALE) > 1e-8:
            raise RuntimeError("QP Bank normalized physical reproduction failed")
        return result

    def close(self):
        self.lp.close()


FloorBank = SquaredBank


def reference_fit(data, family, *, seconds=300.):
    """Independent-of-global-scale fixed-family QP used to obtain f0_MSE."""
    family = canonical(family)
    lp = PoolQP((np.vstack(data.p)*SCALE, np.vstack(data.q)*SCALE, np.vstack(data.target)*SCALE),
                family, loss_scale=1., seconds=seconds)
    try:
        raw = lp.fit(family)
        lookup = {tuple(s): (r, x) for s, r, x in zip(raw["supports"], raw["r"], raw["x"])}
        point = dict(supports=family, r=[lookup.get(s, (0., 0.))[0] for s in family],
                     x=[lookup.get(s, (0., 0.))[1] for s in family])
        point.update(objective=mse(data, point), mse=mse(data, point), mae=mae(data, point),
                     loss_metric="physical_mse", loss_scale=1., audit=lp.solves[-1],
                     reference_fit_uses_global_loss_scale=False)
        return point
    finally:
        lp.close()


def _objective(model, values):
    absolute = model.absolute
    count = absolute.stop-absolute.start
    if count <= 0:
        raise ValueError("A nonempty residual epigraph is required")
    linear = np.asarray(model.variables.objective, float).copy()
    linear[absolute] = 0.
    coefficient = _LOSS_SCALE/(count*SCALE)
    total = float(linear@values + coefficient*(values[absolute]@values[absolute]))/SCALE
    return total, linear, coefficient


def audit_vector(model, values):
    if values is None:
        return dict(valid=False, has_incumbent=False)
    values = np.asarray(values, float)
    if values.shape != (model.variables.size,) or not np.isfinite(values).all():
        return dict(valid=False, has_incumbent=False, error="malformed incumbent")
    rows = model.rows.build(model.variables.size); activity = rows.A@values
    violation = float(max(np.max(rows.lb-activity, initial=0.), np.max(activity-rows.ub, initial=0.),
                          np.max(np.asarray(model.variables.lower)-values, initial=0.),
                          np.max(values-np.asarray(model.variables.upper), initial=0.)))
    integer = np.asarray(model.variables.integrality, bool)
    integral = float(np.max(np.abs(values[integer]-np.rint(values[integer])), initial=0.))
    objective, _, coefficient = _objective(model, values)
    return dict(valid=max(violation, integral) <= 1e-7, has_incumbent=True, maximum_violation=violation,
                integrality_violation=integral, objective=objective, loss_metric="normalized_mse",
                loss_scale=_LOSS_SCALE, objective_scale=SCALE, quadratic_epigraph_coefficient=coefficient)


def _populate(m, model, *, seconds, seed):
    import gurobipy as gp
    rows = model.rows.build(model.variables.size); matrix = csr_matrix(rows.A)
    lower, upper = rows.lb, rows.ub; equal = np.isfinite(lower) & (lower == upper)
    for key, value in dict(Threads=1, OutputFlag=0, TimeLimit=max(.05, seconds), MIPGap=.0001, MIPGapAbs=1e-7,
                           FeasibilityTol=1e-9, IntFeasTol=1e-9, OptimalityTol=1e-9, DualReductions=0,
                           Seed=seed, NonConvex=0).items():
        m.setParam(key, value)
    types = np.where(np.asarray(model.variables.integrality, bool), gp.GRB.BINARY, gp.GRB.CONTINUOUS)
    v = m.addMVar(model.variables.size, lb=model.variables.lower,
                 ub=np.where(np.isposinf(model.variables.upper), gp.GRB.INFINITY, model.variables.upper), vtype=types)
    for mask, sense, rhs in ((equal, "=", lower), (~equal & np.isfinite(lower), ">", lower), (~equal & np.isfinite(upper), "<", upper)):
        if np.any(mask):
            m.addMConstr(matrix[mask], v, sense, rhs[mask])
    _, linear, coefficient = _objective(model, np.zeros(model.variables.size))
    epigraph = v[model.absolute]
    m.setObjective(linear@v + coefficient*(epigraph@epigraph), gp.GRB.MINIMIZE)
    return v


def solve(bank, model, start, *, seconds):
    import gurobipy as gp
    started = perf_counter(); warm = audit_vector(model, start)
    if not warm["valid"]:
        raise RuntimeError(f"Bad MIQP warm start: {warm}")
    with gp.Model("paired_topology_squared_MIQP", env=bank.lp.environment) as native:
        v = _populate(native, model, seconds=seconds, seed=20261030)
        v.Start = start; native.optimize()
        values = np.asarray(v.X) if native.SolCount else None
        audit = audit_vector(model, values)
        objective = float(native.ObjVal)/SCALE if native.SolCount else None
        bound = float(native.ObjBound)/SCALE if abs(native.ObjBound) < gp.GRB.INFINITY else None
        consistent = bound is None or bound <= warm["objective"]+1e-8
        if objective is not None and audit.get("valid") and abs(objective-audit["objective"]) > 1e-8:
            raise RuntimeError("MIQP incumbent objective differs from squared epigraph replay")
        record = dict(status=int(native.Status), objective=objective, dual_bound=bound,
                      relative_gap=float(native.MIPGap) if native.SolCount else None, nodes=float(native.NodeCount),
                      seconds=perf_counter()-started, variable_count=model.variables.size,
                      binary_count=int(np.count_nonzero(model.variables.integrality)), constraint_count=model.rows.size,
                      warm_start=warm, audit=audit, bound_consistent_with_known_upper=consistent,
                      solver="gurobi_convex_MIQP", scale=SCALE, requested_relative_gap=.0001,
                      loss_metric="normalized_mse", loss_scale=_LOSS_SCALE, normalized_objective_scale=SCALE)
        return values if audit.get("valid") and consistent else None, record


def solve_checkpoints(bank, model, meta, current, *, prefix_seconds, budgets=(20., 40.)):
    """The E path uses the same convex objective and native checkpoint semantics."""
    import gurobipy as gp
    from research_experiments.rebuild_benchmark.engine import reconstruction_start
    started = perf_counter(); start = reconstruction_start(bank, model, meta, current)
    warm = audit_vector(model, start)
    if not warm["valid"]:
        raise RuntimeError("Infeasible squared-loss E warm start")
    snapshots = {float(b): dict(values=start.copy(), bound=None, incumbent_at=prefix_seconds) for b in budgets}
    last, events = np.asarray(start), []

    def remember(now, bound=None, values=None):
        nonlocal last
        if values is not None:
            last = np.asarray(values)
        for budget, snapshot in snapshots.items():
            if now <= budget:
                if values is not None:
                    snapshot.update(values=last.copy(), incumbent_at=now)
                if bound is not None:
                    snapshot["bound"] = bound

    def callback(native, where):
        now = prefix_seconds+perf_counter()-started
        if where == gp.GRB.Callback.MIPSOL:
            values = native.cbGetSolution(v); remember(now, values=values)
            events.append(dict(at=now, objective=float(native.cbGet(gp.GRB.Callback.MIPSOL_OBJ))/SCALE))
        elif where == gp.GRB.Callback.MIP:
            bound = float(native.cbGet(gp.GRB.Callback.MIP_OBJBND))/SCALE
            if abs(bound) < gp.GRB.INFINITY:
                remember(now, bound=bound)
    remaining = max(budgets)-prefix_seconds-(perf_counter()-started)
    if remaining <= .05:
        return snapshots, dict(solved=False, seconds=perf_counter()-started, warm_start=warm, events=events)
    with gp.Model("revocable_exit_squared_MIQP", env=bank.lp.environment) as native:
        v = _populate(native, model, seconds=remaining, seed=20261031)
        native.Params.TimeLimit = max(.05, max(budgets)-prefix_seconds-(perf_counter()-started))
        v.Start = start; native.optimize(callback)
        now = prefix_seconds+perf_counter()-started
        if native.SolCount:
            bound = float(native.ObjBound)/SCALE
            remember(now, bound=bound if abs(bound) < gp.GRB.INFINITY else None, values=np.asarray(v.X))
        record = dict(solved=True, status=int(native.Status), seconds=perf_counter()-started, warm_start=warm, events=events,
                      final_bound=float(native.ObjBound)/SCALE if abs(native.ObjBound) < gp.GRB.INFINITY else None,
                      final_objective=float(native.ObjVal)/SCALE if native.SolCount else None,
                      binary_count=len(meta["candidates"]), variable_count=model.variables.size, constraint_count=model.rows.size,
                      solver="gurobi_convex_MIQP", requested_relative_gap=.0001,
                      loss_metric="normalized_mse", loss_scale=_LOSS_SCALE, normalized_objective_scale=SCALE)
    return snapshots, record


def install(K):
    """Explicitly replace only aliases reached by the frozen WZZT/E chain."""
    configure(K)
    names = ("research_experiments.rebuild_benchmark.engine", "research_experiments.rnj_count.engine",
             "research_experiments.joint_exit.model", "research_experiments.rnj_small_seed.engine",
             "research_experiments.revocable.engine", "research_experiments.lp_rebuild.engine")
    modules = {name: importlib.import_module(name) for name in names}
    replacements = {
        names[0]: dict(loss=loss, solve=solve, audit_vector=audit_vector),
        names[1]: dict(loss=loss, solve=solve),
        names[2]: dict(loss=loss, audit_vector=audit_vector, solve_checkpoints=solve_checkpoints),
        names[3]: dict(loss=loss, solve=solve, solve_checkpoints=solve_checkpoints),
        names[4]: dict(loss=loss, solve=solve),
        names[5]: dict(loss=loss),
    }
    aliases, sources = [], {}
    for name, module in modules.items():
        source = Path(inspect.getfile(module)).resolve()
        sources[name] = dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        for attribute, function in replacements[name].items():
            if not hasattr(module, attribute):
                raise RuntimeError(f"Frozen alias is absent: {name}.{attribute}")
            setattr(module, attribute, function)
            aliases.append(name+"."+attribute)
    return dict(loss_scale=_LOSS_SCALE, data_scale=SCALE, objective_scale=SCALE,
                internal_loss="loss_scale * physical_MSE", native_quadratic_coefficient="loss_scale/(observation_count*SCALE)",
                native_objective="SCALE * (normalized_MSE + normalized_line_count_penalty)",
                aliases=aliases, native_module_sources=sources,
                backend_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                fixed_family_solver="convex QP", structural_solver="convex MIQP", cache_kind="QP only",
                ridge=0., positive_impedance_floor=0., truth_used=False, future_DR_used=False)


install_aliases = install
