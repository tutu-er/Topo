"""Original exact hourly-request graph; Gurobi loads only on solving."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class _Term:
    expr: object
    low: float
    high: float
    start: float
    variable: object = None


class _Graph:
    """Bound propagation and exact two-input min/max, with warm graph values."""
    def __init__(self, model, gp):
        self.model, self.gp, self.serial = model, gp, 0

    def variable(self, low, high, start, name="v"):
        if not np.isfinite([low, high, start]).all() or low > high + 1e-8:
            raise ValueError("Invalid finite graph bounds")
        high = max(low, high)
        if low == high: return _Term(float(low), low, high, float(low))
        self.serial += 1
        v = self.model.addVar(lb=float(low), ub=float(high), name=f"{name}_{self.serial}")
        v.Start = float(np.clip(start, low, high))
        return _Term(v, float(low), float(high), float(start), v)

    def affine(self, term, scale=1., offset=0.):
        ends = [scale*term.low+offset, scale*term.high+offset]
        return _Term(scale*term.expr+offset, min(ends), max(ends), scale*term.start+offset)

    def as_variable(self, term):
        if term.variable is None:
            self.serial += 1
            term.variable = self.model.addVar(lb=term.low, ub=term.high, name=f"link_{self.serial}")
            term.variable.Start = term.start
            self.model.addConstr(term.variable == term.expr)
        return term.variable

    def extremum(self, a, b, *, maximum):
        if a.high <= b.low: return b if maximum else a
        if b.high <= a.low: return a if maximum else b
        op = max if maximum else min
        result = self.variable(op(a.low,b.low), op(a.high,b.high), op(a.start,b.start), "clip")
        fn = self.model.addGenConstrMax if maximum else self.model.addGenConstrMin
        fn(self.as_variable(result), [self.as_variable(a), self.as_variable(b)])
        return result

    def constant(self, value):
        return _Term(float(value), float(value), float(value), float(value))

    def inverse(self, term, efficiency):
        if efficiency == 1.: return term
        if term.high <= 0: return self.affine(term, efficiency)
        if term.low >= 0: return self.affine(term, 1/efficiency)
        return self.extremum(self.affine(term, efficiency), self.affine(term, 1/efficiency), maximum=True)


def build_device_graph(model, problem, initial=None):
    """Return hourly request variables, slot delivery variables and exact fee."""
    import gurobipy as gp
    device = problem.device
    R, T, H, dt = device.R, device.T, device.H, device.dt
    warm = problem.expand(problem.warm) if initial is None else np.asarray(initial, float)
    if warm.shape == problem.zero.shape: warm = problem.expand(warm)
    problem.requests_to_values(warm)
    warm_u = device.nominal(warm)
    low, high = np.zeros(R*H), np.zeros(R*H)
    low[problem.active], high[problem.active] = problem.lower, problem.upper
    low, high = low.reshape(R,H), high.reshape(R,H)
    graph = _Graph(model, gp)
    requests = np.empty((R,H), object)
    deliveries = np.empty((R,T), object)
    objective = gp.QuadExpr()
    zero = graph.constant(0.)
    for r in range(R):
        for h in range(H):
            requests[r,h] = graph.variable(low[r,h], high[r,h], warm[r,h], "x")
        previous = zero
        state_needed = bool(np.isfinite(device.next_lo[r]).any() or np.isfinite(device.next_hi[r]).any()
                            or device.terminal[r].any())
        eta = float(device.eta[r])
        for t in range(T):
            if device.reset[r,t]: previous = zero
            cap_low, cap_high = float(device.delta_lower_kw[r,t]), float(device.delta_upper_kw[r,t])
            raw = graph.affine(requests[r,t//device.decision_slots], float(device.scale[r,t]))
            lower, upper = graph.constant(cap_low), graph.constant(cap_high)
            if state_needed:
                if np.isfinite(device.next_lo[r,t]):
                    rate = graph.affine(previous, -1/dt, float(device.next_lo[r,t])/dt)
                    lower = graph.extremum(lower, graph.inverse(rate,eta), maximum=True)
                if np.isfinite(device.next_hi[r,t]):
                    rate = graph.affine(previous, -1/dt, float(device.next_hi[r,t])/dt)
                    upper = graph.extremum(upper, graph.inverse(rate,eta), maximum=False)
                model.addConstr(lower.expr <= upper.expr)
            clipped = graph.extremum(graph.extremum(raw,lower,maximum=True),upper,maximum=False)
            power = graph.variable(max(cap_low,clipped.low), min(cap_high,clipped.high), float(warm_u[r,t]), "u")
            model.addConstr(power.expr == clipped.expr)
            deliveries[r,t] = power
            if power.low >= 0:
                plus, minus = power, zero
            elif power.high <= 0:
                plus, minus = zero, graph.affine(power,-1.)
            else:
                plus = graph.variable(0.,power.high,max(power.start,0.),"positive")
                minus = graph.variable(0.,-power.low,max(-power.start,0.),"negative")
                direction = model.addVar(vtype=gp.GRB.BINARY,name=f"direction_{r}_{t}")
                direction.Start = float(power.start>=0)
                model.addConstr(power.expr == plus.expr-minus.expr)
                model.addConstr(plus.expr <= plus.high*direction)
                model.addConstr(minus.expr <= minus.high*(1-direction))
            if cap_high > 0:
                objective += dt*float(device.cost_full_activation[r,t])/cap_high*plus.expr*plus.expr
            if cap_low < 0:
                objective += dt*float(device.cost_full_activation[r,t])/(-cap_low)*minus.expr*minus.expr
            if state_needed:
                lo = max(previous.low+dt*cap_low/eta, float(device.next_lo[r,t]))
                hi = min(previous.high+dt*eta*cap_high, float(device.next_hi[r,t]))
                energy_start = previous.start+dt*(eta*max(power.start,0.)-max(-power.start,0.)/eta)
                energy = graph.variable(lo,hi,energy_start,"energy")
                model.addConstr(energy.expr == previous.expr+dt*(eta*plus.expr-minus.expr/eta))
                if device.terminal[r,t]: model.addConstr(energy.expr == 0.)
                previous = energy
    return requests, deliveries, objective


def _values(terms):
    return np.array([float(t.expr) if t.low==t.high else float(t.variable.X)
                     for t in terms.ravel()]).reshape(terms.shape)


def _attribute(model, name):
    try:
        value = float(getattr(model,name))
        return value if np.isfinite(value) and abs(value)<1e100 else None
    except Exception: return None
