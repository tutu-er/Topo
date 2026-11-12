"""Continuous hourly device models with explicit differences from the actuator."""
from time import perf_counter
import numpy as np

from ..expected_logodds_admm_20261113.device_steps import resource_slice


def build_resource(model, problem, original, resource, mode):
    import gurobipy as gp
    if mode not in ('storage_relaxation', 'linear_response'):
        raise ValueError(mode)
    part = resource_slice(problem, resource)
    dev = part.device; scale = dev.scale[0]; H, T = dev.H, dev.T
    lower, upper = np.zeros(H), np.zeros(H)
    lower[part.active], upper[part.active] = part.lower, part.upper
    for h in range(H):
        sl = slice(4*h, 4*h+4); mask = scale[sl] > 0
        if mask.any():
            lower[h] = max(lower[h], np.max(dev.delta_lower_kw[0, sl][mask]/scale[sl][mask]))
            upper[h] = min(upper[h], np.min(dev.delta_upper_kw[0, sl][mask]/scale[sl][mask]))
    x = model.addMVar(H, lb=lower, ub=upper, name=f'x{resource}')
    plus = model.addMVar(H, lb=0, ub=np.maximum(upper, 0), name=f'p{resource}')
    minus = model.addMVar(H, lb=0, ub=np.maximum(-lower, 0), name=f'm{resource}')
    model.addConstr(x == plus-minus)
    for h in range(H):
        if upper[h] > 0 and lower[h] < 0:
            model.addConstr(plus[h]/upper[h]+minus[h]/(-lower[h]) <= 1)
    costs = []
    for headroom in (dev.delta_upper_kw[0], -dev.delta_lower_kw[0]):
        c = np.divide(dev.dt*dev.cost_full_activation[0]*scale**2, headroom,
                      out=np.zeros(T), where=headroom > 0)
        costs.append(c.reshape(H, 4).sum(axis=1))
    cost_plus,cost_minus=plus,minus
    if mode == 'storage_relaxation' and float(dev.eta[0]) != 1.:
        # Contract price is charged on net delivery, as in the original model.
        # Energy-flow split is separate and may contain artificial cycling.
        cost_plus=model.addMVar(H,lb=0,ub=np.maximum(upper,0),name=f'cp{resource}')
        cost_minus=model.addMVar(H,lb=0,ub=np.maximum(-lower,0),name=f'cm{resource}')
        model.addConstr(x == cost_plus-cost_minus)
    fee = (cost_plus*costs[0])@cost_plus+(cost_minus*costs[1])@cost_minus
    energy = None
    if mode == 'storage_relaxation':
        energy = model.addMVar(T, lb=dev.next_lo[0], ub=dev.next_hi[0], name=f'e{resource}')
        eta = float(dev.eta[0])
        for t in range(T):
            previous = 0. if dev.reset[0,t] else energy[t-1]
            model.addConstr(energy[t] == previous+dev.dt*scale[t]*(eta*plus[t//4]-minus[t//4]/eta))
            if dev.terminal[0,t]: model.addConstr(energy[t] == 0.)
    # In linear_response mode there are power/availability bounds but no energy
    # coupling. It is an explicitly simplified DR plant, not a storage policy.
    return dict(x=x, plus=plus, minus=minus, fee=fee, costs=costs, scale=scale,
                lower=lower, upper=upper, energy=energy, part=part, mode=mode)


def point(resource):
    x = np.asarray(resource['x'].X)
    delivery = resource['scale']*np.repeat(x, 4)
    simultaneous = resource['scale']*np.repeat(np.minimum(resource['plus'].X, resource['minus'].X),4)
    return dict(x=x, delivery=delivery, model_fee=float(resource['fee'].getValue()),
                simultaneous_kwh=float(.25*np.sum(np.maximum(simultaneous,0))))


class ContinuousStep:
    def __init__(self, problem, original, resource, mode, fee_scale, env, initial):
        import gurobipy as gp
        self.gp=gp;self.model=gp.Model(f'{mode}_{resource}',env=env)
        for k,v in dict(OutputFlag=0,Threads=1,Method=1,FeasibilityTol=1e-9,OptimalityTol=1e-9).items():
            self.model.setParam(k,v)
        self.resource=build_resource(self.model,problem,original,resource,mode)
        self.requests=np.asarray(initial).reshape(1,24).copy()
        self.delivery=self.resource['scale']*np.repeat(self.requests[0],4)
        self.fee_scale=fee_scale;self.model.update()
        if self.model.NumIntVars:raise ArithmeticError('Continuous update acquired integers')

    def solve(self,quadratic,linear,rho,deadline):
        x=self.resource['x'];scale=self.resource['scale']
        weights=(scale**2).reshape(24,4).sum(axis=1)
        coeff=(linear*scale).reshape(24,4).sum(axis=1)
        self.model.setObjective(self.resource['fee']/self.fee_scale+.5*rho*quadratic*(x*weights)@x-rho*coeff@x)
        self.model.Params.TimeLimit=max(.001,min(.5,deadline-perf_counter()))
        self.model.optimize()
        if self.model.SolCount:
            p=point(self.resource);self.requests=p['x'][None];self.delivery=p['delivery']
        return self.delivery.copy(),dict(optimal=self.model.Status==2,no_incumbent=not bool(self.model.SolCount),
                                         native_status=int(self.model.Status))

    def close(self):self.model.dispose()


def supporting_start(data, mode, seconds=6., direction=None):
    """Convex supporting-plane QP; failure proves only this linear restriction."""
    import gurobipy as gp
    began=perf_counter();e=data['evidence'];o=data['original']
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0);env.start()
        with gp.Model('continuous_supporting_planes',env=env) as m:
            for k,v in dict(OutputFlag=0,Threads=1,Method=1,FeasibilityTol=1e-9,OptimalityTol=1e-9).items():m.setParam(k,v)
            resources=[build_resource(m,data['problem'],o,r,mode) for r in range(o.R)]
            at=np.zeros_like(o.scale) if direction is None else direction
            z=e.base+e.contributions(at).sum(axis=1)
            gains=e.gains(at)
            for j in range(len(z)):
                gradient=2*np.einsum('tn,nr->rt',z[j],e.effect[j])*e.scales[j]**2
                terms=[(gradient[r]*rsc['scale']).reshape(24,4).sum(axis=1)@rsc['x'] for r,rsc in enumerate(resources)]
                m.addConstr(sum(terms)>=e.gain_required+e.buffer-gains[j]+np.sum(gradient*at))
            m.setObjective(sum(r['fee'] for r in resources)/data['fee_scale'])
            m.Params.TimeLimit=max(.01,seconds-(perf_counter()-began));m.optimize()
            result=dict(status=int(m.Status),qualified=False,seconds=perf_counter()-began,
                        lower_bound_scope='supporting-plane QP in the declared continuous plant only')
            if m.SolCount:
                points=[point(r) for r in resources];u=np.vstack([p['delivery'] for p in points]);x=np.vstack([p['x'] for p in points])
                result.update(qualified=e.feasible(u),u=u,x=x,gains=e.gains(u),net_fee=o.fees(u),
                              model_fee=sum(p['model_fee'] for p in points),simultaneous_kwh=sum(p['simultaneous_kwh'] for p in points))
            return result


def power_box_upper(data):
    """Valid but possibly loose upper bounds: ignore all energy/time coupling."""
    o=data['original'];e=data['evidence'];lo=o.delta_lower_kw;hi=o.delta_upper_kw
    ans=[]
    for b,F,s in zip(e.base,e.effect,e.scales,strict=True):
        positive=np.maximum(F,0);negative=np.minimum(F,0)
        lower=lo.T@positive.T+hi.T@negative.T
        upper=hi.T@positive.T+lo.T@negative.T
        ans.append(float(np.maximum(2*b*lower+lower**2,2*b*upper+upper**2).sum()*s*s))
    return np.array(ans)
