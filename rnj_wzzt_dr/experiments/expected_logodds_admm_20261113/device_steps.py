"""Resource-separable updates of the unchanged original hourly actuator."""
from copy import copy
from time import perf_counter
import numpy as np

from ...dr.device_graph import build_device_graph, _values, _attribute


def resource_slice(problem, resource):
    answer = copy(problem); answer.device = copy(problem.device)
    device = answer.device
    for name, value in vars(problem.device).items():
        if isinstance(value, np.ndarray) and value.shape[0] == problem.device.R:
            setattr(device, name, value[resource:resource+1].copy())
    device.R = 1
    mask = problem.active//problem.H == resource
    answer.active = problem.active[mask] % problem.H
    answer.lower, answer.upper = problem.lower[mask].copy(), problem.upper[mask].copy()
    answer.bounds = list(zip(answer.lower, answer.upper))
    answer.zero = np.zeros(len(answer.active)); answer.warm = answer.old = answer.zero.copy()
    answer.shape = (1, problem.H)
    answer.resource_ids = [problem.resource_ids[resource]]
    return answer


def initial_requests(problem, original, kind):
    x = np.zeros(problem.shape)
    if kind == "rotation25":
        for r in range(original.R):
            x[r] = .25*np.where((np.arange(24)//2+r % 4) % 2 == 0, 1., -1.)
            if original.kind[r] == 1:
                x[r] = np.minimum(x[r], 0.)
    elif kind != "zero":
        raise ValueError("Unknown preregistered start")
    return problem.expand(np.clip(x.ravel()[problem.active], problem.lower, problem.upper))


class OriginalDeviceStep:
    def __init__(self, problem, resource, requests, fee_scale, env):
        import gurobipy as gp
        self.gp = gp; self.problem = resource_slice(problem, resource)
        self.requests = np.asarray(requests, float).copy()
        self.delivery = self.problem.nominal(self.problem.requests_to_values(self.requests))["mean_u"][0]
        self.model = gp.Model("original_resource_"+str(resource), env=env)
        for key, value in dict(OutputFlag=0, Threads=1, MIPGap=1e-4, MIPGapAbs=1e-8,
            FeasibilityTol=1e-8, IntFeasTol=1e-8, NumericFocus=2, Seed=0).items():
            self.model.setParam(key, value)
        self.x, self.u, self.cost = build_device_graph(self.model, self.problem, self.requests)
        self.fee_scale = fee_scale
        self.model.update()
        self.quadratic = gp.quicksum(term.expr*term.expr for term in self.u[0])
        self.binaries = self.model.NumBinVars

    def solve(self, quadratic, linear, rho, deadline):
        m, gp = self.model, self.gp
        previous = self.delivery.copy()
        objective = self.cost/self.fee_scale+.5*rho*quadratic*self.quadratic
        objective -= rho*gp.quicksum(float(v)*term.expr for v, term in zip(linear, self.u[0], strict=True))
        m.setObjective(objective, gp.GRB.MINIMIZE)
        m.Params.TimeLimit = max(.001, min(.05, deadline-perf_counter()))
        m.optimize()
        report = dict(optimal=m.Status == gp.GRB.OPTIMAL, native_status=int(m.Status),
            no_incumbent=not bool(m.SolCount), binaries=self.binaries,
            gap=_attribute(m, "MIPGap") if m.SolCount else None)
        if m.SolCount:
            raw = _values(self.x)
            values = np.clip(raw.ravel()[self.problem.active], self.problem.lower, self.problem.upper)
            point = self.problem.nominal(values)
            # Replay the original controller; the retained iterate is executable.
            self.requests = point["x"]; self.delivery = point["mean_u"][0]
            report["replay_difference"] = float(np.max(abs(_values(self.u)[0]-self.delivery)))
        return self.delivery.copy(), report

    def close(self):
        self.model.dispose()


def native_direct(problem, original, evidence, fee_scale, directory, *, seconds=60.):
    """Same gain goal and full original device domain, from the zero start."""
    import gurobipy as gp
    began = perf_counter(); zero = problem.expand(problem.zero)
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0); env.start()
        with gp.Model("direct_expected_log_odds_original_domain", env=env) as model:
            for key, value in dict(OutputFlag=1, LogToConsole=0, LogFile=str(directory/"gurobi.log"),
                Threads=1, Seed=0, NonConvex=2, MIPGap=.01, NumericFocus=2,
                MIPFocus=1, FeasibilityTol=1e-8, IntFeasTol=1e-8, DualReductions=0).items():
                model.setParam(key, value)
            x, u, cost = build_device_graph(model, problem, zero)
            inactive = ~evidence.active.any(axis=0)
            for r in np.flatnonzero(inactive):
                for h in range(problem.H):
                    model.addConstr(x[r, h].expr == 0.)
            low, high = problem.device.delta_lower_kw, problem.device.delta_upper_kw
            delivery = model.addMVar(low.shape, lb=low, ub=high, name="original_delivery")
            delivery.Start = 0.
            for index in np.ndindex(low.shape):
                model.addConstr(delivery[index] == u[index].expr)
            for j, (base, effect) in enumerate(zip(evidence.base, evidence.effect, strict=True)):
                positive, negative = np.maximum(effect, 0.), np.minimum(effect, 0.)
                zl = base+low.T@positive.T+high.T@negative.T
                zh = base+high.T@positive.T+low.T@negative.T
                z = model.addMVar(base.shape, lb=zl, ub=zh, name="prediction_difference_"+str(j))
                z.Start = base
                model.addConstr(z == base+delivery.T@effect.T)
                flat = z.reshape(-1)
                # Use the original gamma for a bound on that original problem.
                # ADMM alone has a numerical headroom; it is not added here.
                model.addConstr(flat@flat >= float(evidence.base_norm2[j]+evidence.gain_required/evidence.scales[j]**2))
            model.setObjective(cost/fee_scale, gp.GRB.MINIMIZE); model.update()
            model.Params.TimeLimit = max(.01, seconds-(perf_counter()-began))
            model.write(str(directory/"model.mps.gz")); model.optimize()
            result = dict(status=int(model.Status), has_solution=bool(model.SolCount),
                qualified=False, fee=None, lower_bound=_attribute(model, "ObjBound"),
                gap=_attribute(model, "MIPGap") if model.SolCount else None,
                variables=model.NumVars, binaries=model.NumBinVars,
                total_seconds=perf_counter()-began, global_optimality_proved=False)
            if result["lower_bound"] is not None:
                result["lower_bound"] *= fee_scale
            if model.SolCount:
                raw = _values(x)
                values = np.clip(raw.ravel()[problem.active], problem.lower, problem.upper)
                point = problem.nominal(values)
                audit = original.audit(point["mean_u"])
                result.update(qualified=audit["passed"] and evidence.feasible(point["mean_u"]),
                    fee=original.fees(point["mean_u"]), gains=evidence.gains(point["mean_u"]), device=audit)
                if result["qualified"]:
                    np.savez_compressed(directory/"quote.npz", x=point["x"], mean_u=point["mean_u"])
                    result["global_gap_target_met"] = bool(model.Status == gp.GRB.OPTIMAL and result["gap"] is not None and result["gap"] <= .01)
                    result["global_optimality_proved"] = bool(result["gap"] is not None and result["gap"] <= 1e-8)
            return result
