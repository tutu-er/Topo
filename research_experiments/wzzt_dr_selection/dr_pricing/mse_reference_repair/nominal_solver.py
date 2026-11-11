"""Minimum original fee in the no-projection inner domain of the device map.

``solve_inner(d, g, out, seconds=60)`` uses the existing resource adapter p=d['p']
and study.BlockGate g. A request x(r,h) delivers scale(r,t)*x(r,h(t)). Every
slot satisfies the original power and next-energy bounds, resets and terminal
conditions. Inductively this raw delivery is unchanged by the original causal
projection. This is a subset of the original request-to-delivery possibilities;
its solver lower bound is NOT a lower bound for the full controller problem.

The returned best point is independently replayed through p.nominal. No paid
fallback is returned when the original probability condition is not attained.
This module performs no fit, AC call, or optimization at import time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.linalg import cho_solve
from scipy.sparse import csr_matrix, diags


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _attribute(model, name):
    try:
        value = float(getattr(model, name))
        return value if np.isfinite(value) and abs(value) < 1e90 else None
    except (AttributeError, RuntimeError):
        return None


def solve_inner(d, g, out, seconds=60):
    """Return (replayed_original_point_or_None, result); all arrays stay in out."""
    import gurobipy as gp
    p, dev = d['p'], d['p'].device
    R, T, H, N, S = dev.R, dev.T, dev.H, p.N, dev.decision_slots
    node, ratio = d['original_device'].node, d['original_device'].qratio
    if g.forecast.shape != (T, 2*N) or T != H*S or seconds <= 0:
        raise ValueError("Inconsistent gate/device dimensions or nonpositive time limit")
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter(); best = None
    lower, upper = np.zeros(R*H), np.zeros(R*H)
    lower[p.active], upper[p.active] = p.lower, p.upper
    lower, upper = lower.reshape(R, H), upper.reshape(R, H)
    scale = dev.scale
    coef_plus = np.divide(dev.dt*dev.cost_full_activation*scale**2,
        dev.delta_upper_kw, out=np.zeros_like(scale), where=dev.delta_upper_kw > 0)
    coef_minus = np.divide(dev.dt*dev.cost_full_activation*scale**2,
        -dev.delta_lower_kw, out=np.zeros_like(scale), where=dev.delta_lower_kw < 0)
    coef_plus, coef_minus = [a.reshape(R, H, S).sum(-1) for a in (coef_plus, coef_minus)]
    _write(out/'request.json', dict(seconds=float(seconds), resources=R, active_hours=len(p.active),
        gate_lambda=float(g.lam), gate_h=float(g.h), original_domain_lower_bound=0.,
        domain='original capabilities, nominal requests that need no causal projection',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    result = dict(status='not_started', qualified=False, original_domain_lower_bound=0.,
        inner_lower_bound=None, original_global_minimum_proven=False)
    try:
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0); env.start()
            with gp.Model('nominal_no_projection_inner', env=env) as m:
                for k, val in dict(OutputFlag=1, LogToConsole=0, LogFile=str(out/'gurobi.log'),
                    TimeLimit=float(seconds), Threads=1, MIPGap=.02, NonConvex=2,
                    NumericFocus=2, MIPFocus=1, Seed=0, FeasibilityTol=1e-8,
                    IntFeasTol=1e-8, DualReductions=0).items(): m.setParam(k, val)
                x = m.addMVar((R, H), lb=lower, ub=upper, name='hourly_request')
                xp = m.addMVar((R, H), lb=0., ub=np.maximum(upper, 0.), name='positive_request')
                xm = m.addMVar((R, H), lb=0., ub=np.maximum(-lower, 0.), name='negative_request')
                m.addConstr(x == xp-xm)
                x.Start = xp.Start = xm.Start = np.zeros((R, H))
                cost = (xp.reshape(-1) @ diags(coef_plus.ravel(), format='csr') @ xp.reshape(-1)
                        + xm.reshape(-1) @ diags(coef_minus.ravel(), format='csr') @ xm.reshape(-1)).item()
                u = [[float(scale[r, t])*x[r, t//S].item() for t in range(T)] for r in range(R)]
                for r in range(R):
                    eta = float(dev.eta[r])
                    if eta < 1.:
                        for h in range(H):
                            if lower[r, h] < 0 < upper[r, h]:
                                sign = m.addVar(vtype=gp.GRB.BINARY, name=f'direction_{r}_{h}')
                                sign.Start = 1.
                                m.addConstr(xp[r, h] <= float(upper[r, h])*sign)
                                m.addConstr(xm[r, h] <= -float(lower[r, h])*(1-sign))
                    energy = gp.LinExpr()
                    for t in range(T):
                        if dev.reset[r, t]: energy = gp.LinExpr()
                        m.addConstr(u[r][t] >= float(dev.delta_lower_kw[r, t]))
                        m.addConstr(u[r][t] <= float(dev.delta_upper_kw[r, t]))
                        if eta == 1.:
                            energy += dev.dt*u[r][t]
                        else:
                            h = t//S
                            energy += float(dev.dt*scale[r, t])*(eta*xp[r, h].item()-xm[r, h].item()/eta)
                        if np.isfinite(dev.next_lo[r, t]): m.addConstr(energy >= float(dev.next_lo[r, t]))
                        if np.isfinite(dev.next_hi[r, t]): m.addConstr(energy <= float(dev.next_hi[r, t]))
                        if dev.terminal[r, t]: m.addConstr(energy == 0.)
                lo, hi = np.zeros((T, 2*N)), np.zeros((T, 2*N))
                for r, n in enumerate(node):
                    lo[:, n] += dev.delta_lower_kw[r]; hi[:, n] += dev.delta_upper_kw[r]
                    qlo, qhi = ratio[r]*dev.delta_lower_kw[r], ratio[r]*dev.delta_upper_kw[r]
                    lo[:, N+n] += np.minimum(qlo, qhi); hi[:, N+n] += np.maximum(qlo, qhi)
                z = m.addMVar((T, 2*N), lb=lo, ub=hi, name='node_delta_kw')
                for t in range(T):
                    for n in range(N):
                        ids = np.flatnonzero(node == n)
                        m.addConstr(z[t, n] == gp.quicksum(u[r][t] for r in ids))
                        m.addConstr(z[t, N+n] == gp.quicksum(float(ratio[r])*u[r][t] for r in ids))
                z.Start = np.zeros((T, 2*N))
                mapping = cho_solve((g.chol, True), g.W)/g.base
                center = cho_solve((g.chol, True), g.W@g.forecast.ravel())
                pos, neg = np.maximum(mapping, 0.), np.minimum(mapping, 0.)
                vlo = center+pos@lo.ravel()+neg@hi.ravel()
                vhi = center+pos@hi.ravel()+neg@lo.ravel()
                v = m.addMVar(len(center), lb=vlo, ub=vhi, name='Gaussian_integral_aux')
                m.addConstr(csr_matrix(g.Pi)@v == g.W@g.forecast.ravel()+csr_matrix(g.W/g.base)@z.reshape(-1))
                v.Start = center
                score = gp.QuadExpr(float(g.c))
                for t in range(T):
                    mean = g.forecast[t]
                    score += (z[t]@(g.K/g.base**2)@z[t]+(2*mean@g.K/g.base)@z[t]).item()+float(mean@g.K@mean)
                score -= 2*(v@csr_matrix(g.Pi)@v).item()
                m.addQConstr(score >= float(g.h)); m.setObjective(cost, gp.GRB.MINIMIZE)
                m.update(); m.write(str(out/'model.mps.gz')); m.optimize()
                result.update(status=int(m.Status), has_solution=bool(m.SolCount),
                    inner_lower_bound=_attribute(m, 'ObjBound'), reported_gap=_attribute(m, 'MIPGap'),
                    variables=m.NumVars, binaries=m.NumBinVars, solver_upper_bound=_attribute(m, 'ObjVal'))
                if m.SolCount:
                    m.write(str(out/'solver.sol'))
                    raw = np.asarray(x.X); values = np.clip(raw.ravel()[p.active], p.lower, p.upper)
                    requests = p.expand(values); linear_u = scale*np.repeat(requests, S, axis=1)
                    point = p.nominal(values)
                    replay = float(np.max(np.abs(linear_u-point['mean_u']), initial=0.))
                    direct_fee = float(dev.fees(linear_u)); fee_error = abs(direct_fee-point['fee'])
                    audit = p.audit_delivery(point['mean_u'])
                    aggregate = np.column_stack(d['original_device'].aggregate(point['mean_u']))
                    bound = g.bound(aggregate)
                    paid_upper = result['solver_upper_bound']
                    valid = (replay <= 1e-7 and fee_error <= 1e-7*max(1., point['fee'])
                             and audit['passed'] and bound['sufficient'] and paid_upper is not None
                             and point['fee'] <= paid_upper+1e-6*max(1., abs(paid_upper)))
                    result.update(controller_replay_max_kw=replay, fee_replay_error=fee_error,
                        request_bound_repair_max=float(np.max(np.abs(raw-requests), initial=0.)),
                        device=audit, bound=bound, replay_qualified=bool(valid))
                    if valid:
                        best = point
                        result.update(qualified=True, fee=float(point['fee']),
                            fee_epigraph_slack=float(paid_upper-point['fee']),
                            original_global_minimum_proven=bool(point['fee'] == 0.))
                        np.savez_compressed(out/'candidate.npz', x=requests, u=point['mean_u'])
    except Exception as exc:
        result.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    result['seconds'] = time.perf_counter()-began
    result['bound_scope'] = 'inner_lower_bound only applies to no-projection subdomain; original nonnegative-fee lower bound remains zero'
    _write(out/'summary.json', result)
    return best, result
