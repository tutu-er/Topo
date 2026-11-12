"""A declared convex subset of the original actuator, not a relaxation.

Hourly raw plans must already satisfy all original power/energy/viability
bounds, so nominal clipping is inactive. Battery signs follow a fixed public
two-hour alternation. Other devices retain both allowed signs. Every accepted
plan is replayed through the original controller. No original-domain lower
bound is inferred from this restricted set.
"""
from time import perf_counter
import numpy as np

from .device_steps import resource_slice


def build_resource(model, part, resource):
    import gurobipy as gp
    device = part.device; T, H = device.T, device.H
    scale = device.scale[0]
    lower, upper = np.zeros(H), np.zeros(H)
    lower[part.active], upper[part.active] = part.lower, part.upper
    for h in range(H):
        at = slice(4*h, 4*h+4); available = scale[at] > 0
        if np.any(available):
            lower[h] = max(lower[h], np.max(device.delta_lower_kw[0, at][available]/scale[at][available]))
            upper[h] = min(upper[h], np.min(device.delta_upper_kw[0, at][available]/scale[at][available]))
    signs = np.where((np.arange(H)//2+resource % 4) % 2 == 0, 1., -1.)
    eta = float(device.eta[0])
    if eta != 1.:
        lower = np.where(signs > 0, np.maximum(lower, 0.), lower)
        upper = np.where(signs < 0, np.minimum(upper, 0.), upper)
    x = model.addMVar(H, lb=lower, ub=upper, name=f"hourly_{resource}"); x.Start = 0.
    plus = model.addMVar(H, lb=0., ub=np.maximum(upper, 0.), name=f"plus_{resource}")
    minus = model.addMVar(H, lb=0., ub=np.maximum(-lower, 0.), name=f"minus_{resource}")
    plus.Start = 0.; minus.Start = 0.; model.addConstr(x == plus-minus)
    costs = []
    for headroom in (device.delta_upper_kw[0], -device.delta_lower_kw[0]):
        coefficient = np.divide(device.dt*device.cost_full_activation[0]*scale**2,
                                headroom, out=np.zeros(T), where=headroom > 0)
        costs.append(coefficient.reshape(H, 4).sum(axis=1))
    fee = (plus*costs[0])@plus+(minus*costs[1])@minus
    energy = model.addMVar(T, lb=device.next_lo[0], ub=device.next_hi[0], name=f"energy_{resource}")
    energy.Start = 0.
    for t in range(T):
        previous = 0. if device.reset[0, t] else energy[t-1]
        factor = 1. if eta == 1. else eta if signs[t//4] > 0 else 1/eta
        model.addConstr(energy[t] == previous+device.dt*factor*scale[t]*x[t//4])
        if device.terminal[0, t]:
            model.addConstr(energy[t] == 0.)
    return dict(x=x, fee=fee, scale=scale, part=part, resource=resource)


def replay(resource, x):
    part = resource["part"]
    values = np.clip(x[part.active], part.lower, part.upper)
    point = part.nominal(values)
    raw = resource["scale"]*np.repeat(point["x"][0], 4)
    if np.max(abs(raw-point["mean_u"][0])) > 1e-7:
        raise ArithmeticError("Convex schedule triggers original nominal clipping")
    if not part.audit_delivery(point["mean_u"])["passed"]:
        raise ArithmeticError("Convex schedule violates original device domain")
    return point


class ConvexDeviceStep:
    def __init__(self, problem, resource, requests, fee_scale, env):
        import gurobipy as gp
        self.gp = gp; self.part = resource_slice(problem, resource)
        self.model = gp.Model(f"convex_original_subset_{resource}", env=env)
        for key, value in dict(OutputFlag=0, Threads=1, Method=1, FeasibilityTol=1e-9,
                               OptimalityTol=1e-9, NumericFocus=2).items():
            self.model.setParam(key, value)
        self.resource = build_resource(self.model, self.part, resource)
        self.requests = np.asarray(requests).copy()
        self.delivery = replay(self.resource, self.requests[0])["mean_u"][0]
        self.fee_scale = fee_scale
        self.quad_scale = self.resource["scale"].reshape(24, 4)**2
        self.model.update()
        if self.model.NumBinVars:
            raise ArithmeticError("Convex device model acquired integer variables")

    def solve(self, quadratic, linear, rho, deadline):
        x, model = self.resource["x"], self.model
        weights = self.quad_scale.sum(axis=1)
        coefficient = (linear*self.resource["scale"]).reshape(24, 4).sum(axis=1)
        objective = self.resource["fee"]/self.fee_scale+.5*rho*quadratic*(x*weights)@x-rho*coefficient@x
        model.setObjective(objective, self.gp.GRB.MINIMIZE)
        model.Params.TimeLimit = max(.001, min(1., deadline-perf_counter()))
        model.optimize()
        report = dict(optimal=model.Status == self.gp.GRB.OPTIMAL, native_status=int(model.Status),
                      no_incumbent=not bool(model.SolCount), binaries=0)
        if model.SolCount:
            point = replay(self.resource, x.X)
            self.requests, self.delivery = point["x"], point["mean_u"][0]
        return self.delivery.copy(), report

    def close(self):
        self.model.dispose()


def supporting_plane_start(problem, original, evidence, fee_scale, *, seconds=10.):
    """One convex QP using the global quadratic lower planes at zero.

    It can start from gain-infeasible zero. If it succeeds, the resulting
    original quadratic constraints are satisfied. Its dual bound applies only
    to this supporting-plane problem in the declared convex device subset.
    """
    import gurobipy as gp
    began = perf_counter()
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0); env.start()
        with gp.Model("zero_anchor_supporting_planes", env=env) as model:
            model.Params.OutputFlag = 0; model.Params.Threads = 1
            model.Params.Method = 1; model.Params.FeasibilityTol = 1e-9
            model.Params.OptimalityTol = 1e-9
            resources = [build_resource(model, resource_slice(problem, r), r) for r in range(original.R)]
            for j in range(len(evidence.base)):
                gradient = 2*np.einsum("tn,nr->rt", evidence.base[j], evidence.effect[j])*evidence.scales[j]**2
                terms = []
                for r, resource in enumerate(resources):
                    coefficient = (gradient[r]*resource["scale"]).reshape(24, 4).sum(axis=1)
                    terms.append(coefficient@resource["x"])
                model.addConstr(sum(terms) >= evidence.gain_required+evidence.buffer)
            model.setObjective(sum(r["fee"] for r in resources)/fee_scale, gp.GRB.MINIMIZE)
            model.Params.TimeLimit = max(.01, seconds-(perf_counter()-began)); model.optimize()
            result = dict(native_status=int(model.Status), has_solution=bool(model.SolCount), qualified=False,
                total_seconds=perf_counter()-began, fee=None, lower_bound_scope="supporting-plane convex subset only")
            if model.SolCount:
                points = [replay(resource, resource["x"].X) for resource in resources]
                x = np.vstack([p["x"] for p in points]); delivery = original.nominal(x)
                result.update(qualified=original.audit(delivery)["passed"] and evidence.feasible(delivery),
                    fee=original.fees(delivery), gains=evidence.gains(delivery), x=x, mean_u=delivery)
            return result
