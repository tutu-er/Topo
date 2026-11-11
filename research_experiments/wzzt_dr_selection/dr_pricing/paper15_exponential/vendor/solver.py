"""Global expected-evidence QCQP with the unchanged hourly device controller.

The optimizer encodes the original finite piecewise-affine device graph exactly;
it does not replace hourly requests with free quarter-hour deliveries. Gurobi is
imported only when a positive-cost solve is necessary. Output arrays stay private.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np


def _json(path, value):
    def plain(v):
        if isinstance(v, dict): return {str(k):plain(x) for k,x in v.items()}
        if isinstance(v, (list, tuple)): return [plain(x) for x in v]
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        return v
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _digest(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()


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


def solve(problem, evidence, threshold, out, *, seconds=120., mip_gap=.02,
          threads=1, seed=0, initial=None):
    """Find minimum original fee subject to expected quadratic evidence.

    Bounds refer to this planning model; expected evidence is not a posterior
    chance guarantee. A zero-fee feasible point is certified analytically.
    """
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/"request.json").exists(): raise ValueError("Existing run preserved; use a new output directory")
    if not np.isfinite([threshold,seconds,mip_gap]).all() or seconds<=0 or not 0<=mip_gap<1:
        raise ValueError("Invalid finite solver limits")
    Q, linear = np.asarray(evidence.Q,float), np.asarray(evidence.linear,float)
    R,T = problem.device.R,problem.device.T
    dense = Q.ndim == 2
    valid_shape = ((Q.shape==(R*T,R*T) and linear.shape==(R*T,)) if dense
                   else (Q.shape==(T,R,R) and linear.shape==(T,R)))
    if not valid_shape or not np.isfinite(Q).all() or not np.isfinite(linear).all():
        raise ValueError("Evidence dimensions differ from original devices")
    arrays = dict(Q=Q,linear=linear,constant=np.array(evidence.constant))
    arrays.update({key:np.asarray(getattr(problem.device,key)) for key in (
        "delta_lower_kw","delta_upper_kw","cost_full_activation","eta","energy_low","energy_high",
        "reset","terminal","future_lo","future_hi")})
    arrays.update(active=np.asarray(problem.active),request_lower=problem.lower,request_upper=problem.upper,
                  dt_hours=np.array(problem.device.dt),decision_slots=np.array(problem.device.decision_slots))
    warm=problem.expand(problem.warm) if initial is None else np.asarray(initial,float)
    if warm.shape==problem.zero.shape: warm=problem.expand(warm)
    problem.requests_to_values(warm)
    arrays["warm_requests"]=warm
    sources={"solver.py":Path(__file__)}
    for name,obj in (("problem.py",problem),("core.py",evidence)):
        filename=inspect.getsourcefile(type(obj))
        if filename is not None: sources[name]=Path(filename)
    try: package_version=version("gurobipy")
    except PackageNotFoundError: package_version=None
    request = dict(threshold=float(threshold),seconds=seconds,mip_gap=mip_gap,threads=threads,seed=seed,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sources.items()},
        source_paths={name:str(path.resolve()) for name,path in sources.items()},
        environment=dict(python=platform.python_version(),numpy=np.__version__,gurobipy=package_version),
        dt_hours=problem.device.dt,decision_slots=problem.device.decision_slots,case_config=problem.config,
        input_hashes={key:_digest(value) for key,value in arrays.items()},
        evidence_kind=getattr(evidence,"kind","unspecified"), physical_graph="unchanged hourly causal projection",
        initial=_digest(warm))
    _json(out/"request.json",request)
    np.savez_compressed(out/"inputs.npz",**arrays)
    for name,path in sources.items(): (out/("source_"+name)).write_bytes(path.read_bytes())
    began=time.perf_counter()
    try:
        zero=problem.nominal(problem.zero)
        zero_value=float(evidence.value(zero["mean_u"]))
        if zero["fee"]==0 and zero_value>=threshold and problem.audit_delivery(zero["mean_u"])["passed"]:
            result=dict(x=zero["x"],mean_u=zero["mean_u"],unit_reward=problem.unit_reward(zero["mean_u"]),
                objective=0.,objbound=0.,mipgap=0.,solver_status="CERTIFIED_COST_ZERO",
                solver_performed=False,optimal_for_model=True,has_solution=True,qualified=True,
                global_gap_target_met=True,verified_relative_gap=0.,
                evidence=zero_value,threshold=threshold,model_evidence=zero_value,
                replay_delivery_error=0.,fee_error=0.,evidence_replay_error=0.,
                device_audit=problem.audit_delivery(zero["mean_u"]),
                certificate="Nonnegative fees and a zero-fee feasible point; no Gurobi optimization run")
        else:
            import gurobipy as gp
            with gp.Env(empty=True) as env:
                env.setParam("OutputFlag",0); env.start()
                with gp.Model("bayesian_expected_evidence",env=env) as model:
                    for key,value in dict(LogToConsole=0,OutputFlag=1,LogFile=str(out/"gurobi.log"),Threads=threads,Seed=seed,
                        TimeLimit=seconds,MIPGap=mip_gap,NonConvex=2,NumericFocus=2,
                        FeasibilityTol=1e-8,IntFeasTol=1e-8,OptimalityTol=1e-8,DualReductions=0).items(): model.setParam(key,value)
                    x,u,cost=build_device_graph(model,problem,initial)
                    quadratic=gp.QuadExpr(float(evidence.constant))
                    if dense:
                        flat=u.ravel()
                        for i in range(R*T):
                            if linear[i]!=0: quadratic+=float(linear[i])*flat[i].expr
                            for j in range(i,R*T):
                                coefficient=Q[i,i] if i==j else Q[i,j]+Q[j,i]
                                if coefficient!=0:
                                    quadratic+=float(coefficient)*flat[i].expr*flat[j].expr
                    else:
                        for t in range(T):
                            for r in range(R):
                                if linear[t,r]!=0: quadratic+=float(linear[t,r])*u[r,t].expr
                                for s in range(R):
                                    if Q[t,r,s]!=0: quadratic+=float(Q[t,r,s])*u[r,t].expr*u[s,t].expr
                    model.addQConstr(quadratic>=float(threshold),name="expected_log_evidence")
                    model.setObjective(cost,gp.GRB.MINIMIZE);model.update()
                    model.write(str(out/"model.mps.gz"))
                    model.optimize()
                    status=int(model.Status);has=model.SolCount>0
                    result=dict(solver_status=status,solver_performed=True,has_solution=has,
                        solver_objective=_attribute(model,"ObjVal") if has else None,
                        objbound=_attribute(model,"ObjBound"),mipgap=_attribute(model,"MIPGap") if has else None,
                        optimal_for_model=False,global_gap_target_met=False,threshold=float(threshold),
                        variables=int(model.NumVars),binary_variables=int(model.NumBinVars),
                        general_constraints=int(model.NumGenConstrs),nodes=_attribute(model,"NodeCount"),
                        qualified=False,objective=None,x=None,mean_u=None,unit_reward=None)
                    if has:
                        requests,delivery=_values(x),_values(u)
                        bounded=problem.expand(np.clip(requests.ravel()[problem.active],problem.lower,problem.upper))
                        repair=float(np.max(np.abs(bounded-requests)))
                        requests=bounded
                        replay=problem.nominal(problem.requests_to_values(requests))
                        value=float(evidence.value(replay["mean_u"]))
                        model_value=(float(evidence.constant+delivery.ravel()@Q@delivery.ravel()
                                           +linear@delivery.ravel()) if dense else
                                     float(evidence.constant+np.einsum("tr,trs,ts->",delivery.T,Q,delivery.T)
                                           +np.sum(linear*delivery.T)))
                        error=float(np.max(np.abs(delivery-replay["mean_u"])))
                        audit=problem.audit_delivery(replay["mean_u"])
                        result.update(x=requests,mean_u=replay["mean_u"],unit_reward=problem.unit_reward(replay["mean_u"]),
                            objective=float(replay["fee"]),evidence=value,model_evidence=model_value,
                            replay_delivery_error=error,request_bound_repair_max=repair,fee_error=abs(replay["fee"]-model.ObjVal),
                            evidence_replay_error=abs(value-model_value),device_audit=audit,
                            qualified=bool(audit["passed"] and repair<=1e-6 and error<=1e-6
                                and abs(replay["fee"]-model.ObjVal)<=1e-6*max(1.,abs(model.ObjVal))
                                and value>=threshold-1e-7*max(1.,abs(threshold))))
                        if result["objbound"] is not None:
                            result["verified_relative_gap"]=max(0.,replay["fee"]-result["objbound"])/max(1e-12,abs(replay["fee"]))
                            result["global_gap_target_met"]=bool(status==gp.GRB.OPTIMAL and result["qualified"]
                                and result["verified_relative_gap"]<=max(mip_gap,1e-7))
                        result["optimality_scope"]="Global solver bounds to requested relative gap, subject to numerical tolerances"
        result["seconds"]=time.perf_counter()-began
        if result["has_solution"]:
            np.savez_compressed(out/"quote.npz",x=result["x"],mean_u=result["mean_u"],unit_reward=result["unit_reward"])
        _json(out/"summary.json",{key:value for key,value in result.items() if key not in ("x","mean_u","unit_reward")})
        _json(out/"complete.json",dict(status="complete",qualified=result["qualified"],solver_performed=result["solver_performed"]))
        return result
    except Exception as exc:
        _json(out/"failure.json",dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-began))
        raise
