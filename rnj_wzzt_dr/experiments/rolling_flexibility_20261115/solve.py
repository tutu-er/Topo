"""Cost-minimal buffered QCQP, audited before any first-step execution.

Native bounds concern the encoded remaining-plan QCQP, not the original
chance event, physical AC reliability, or a globally optimal feedback policy.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np


BUFFER = 1e-8


def expression(u, q, a, c):
    q=np.asarray(q)
    hours=np.asarray(a).shape[0]
    return sum(u[h] @ (q if q.ndim==2 else q[h]) @ u[h] for h in range(hours)) + 2*(a*u).sum() + float(c)


def box_upper(q, a, c, lower, upper):
    """Outer-box bound with an explicit floating-point padding."""
    q, a, lo, hi = [np.asarray(x, np.longdouble) for x in (q, a, lower, upper)]
    center, radius = (lo+hi)/2, (hi-lo)/2
    value = np.einsum('hi,hij,hj->', center, q, center) + 2*np.sum(a*center) + c
    linear = np.einsum('hij,hj->hi', q, center) + a
    terms = 2*np.sum(np.abs(linear)*radius) + np.einsum('hi,hij,hj->', radius, np.abs(q), radius)
    return float(value+terms) + 1e-10*max(1., float(abs(value)+abs(terms)))


def objective(case, u, paid, prices):
    request = np.asarray(case.request_delta(u), float)
    peak = np.maximum(np.asarray(paid, float), np.maximum(np.max(np.abs(request), axis=0),
                      np.max(np.abs(case.service_delta),axis=0)))
    return float(case.fee(u) + np.dot(prices, peak-paid))


def check(case, evidence, u):
    domain, gate = case.audit(u), evidence.verify_expansion(u)
    encoded = (bool(domain['passed']) and bool(gate['feasible'])
        and np.all(np.asarray(gate.get('margins',[]),float)>=BUFFER)
        and np.all(np.asarray(gate.get('guard_gains',[]),float)>=0.)
        and np.all(np.asarray(gate.get('progress_margins',[]),float)>=-1e-9))
    return dict(passed=bool(domain['passed'] and gate['feasible']), encoded_passed=bool(encoded),
                domain=domain, evidence=gate)


def plan(case, evidence, *, env, seconds, paid_capacity, capacity_prices, warm=None):
    began = perf_counter()
    paid, prices = np.asarray(paid_capacity, float), np.asarray(capacity_prices, float)
    receipt = dict(seconds_budget=float(seconds), feasible=False, optimizer_invoked=False,
        status='not_solved', source=None, lower_bound_yuan=None, relative_gap=None,
        global_optimality_verified=False, initial=[],
        bound_scope='Encoded remaining-plan cost QCQP only; no closed-loop or AC certificate')
    zero = np.zeros(case.shape)
    valid = []
    for name, value in [('zero_extra', zero), ('previous_repaired', warm)]:
        if value is None:
            continue
        try:
            value = case.array(value)
            audit = check(case, evidence, value)
            cost = objective(case, value, paid, prices)
            receipt['initial'].append(dict(name=name, feasible=audit['passed'], cost_yuan=cost))
            if audit['passed']:
                valid.append((cost, name, value.copy(), audit))
        except Exception as exc:
            receipt['initial'].append(dict(name=name, feasible=False, error=repr(exc)))
    best = min(valid, key=lambda row: row[0]) if valid else None
    if best is not None and best[0] <= 1e-12:
        receipt.update(status='zero_cost_sufficient', feasible=True, source=best[1], cost_yuan=best[0],
            lower_bound_yuan=0., relative_gap=0., global_optimality_verified=True,
            bound_scope='Nonnegative cost and independently verified original direct domain; encoded numerical buffer separately recorded',
            audit=best[3], seconds=perf_counter()-began)
        return best[2], receipt
    # Prove only this fixed-law sufficient QCQP infeasible over an outer box.
    if evidence.mode != 'normal_fixed':
        bounds = [box_upper(q, a, c, case.lower, case.upper) for q,a,c in
                  zip(evidence.gate_q, evidence.gate_a, evidence.gate_c)]
        receipt['gate_outer_box_upper'] = bounds
        if any(v < -1e-7 for v in bounds):
            receipt.update(status='infeasible_by_probability_box_bound', seconds=perf_counter()-began)
            return None, receipt
    if seconds <= perf_counter()-began:
        receipt.update(status='budget_before_native', seconds=perf_counter()-began)
        if best:
            receipt.update(feasible=True, source=best[1], cost_yuan=best[0], audit=best[3])
        return None if best is None else best[2], receipt
    import gurobipy as gp
    with gp.Model('quarter_hour_remaining_evidence', env=env) as model:
        model.setParam('OutputFlag', 0); model.setParam('Threads', 1)
        model.setParam('Seed', 0); model.setParam('NonConvex', 2)
        model.setParam('FeasibilityTol', 1e-9); model.setParam('NumericFocus', 1)
        model.setParam('MIPGap', 1e-4)
        u, fee = case.attach(model)
        peak = model.addMVar(case.R, lb=paid, name='purchased_daily_peak')
        request = u + np.asarray(case.service_delta)
        model.addConstr(peak[None, :] >= request)
        model.addConstr(peak[None, :] >= -request)
        # Firm service must remain callable if the voluntary extra is cancelled.
        model.addConstr(peak >= np.max(np.abs(case.service_delta),axis=0))
        for k, row in enumerate(evidence.rows):
            if evidence.mode == 'normal_fixed':
                v = expression(u, evidence.variance_q[k], evidence.variance_a[k], evidence.variance_c[k])
                mu = expression(u, evidence.mu_q[k], evidence.mu_a[k], evidence.mu_c[k])
                ub = box_upper(evidence.variance_q[k], evidence.variance_a[k], evidence.variance_c[k], case.lower, case.upper)
                s = model.addVar(lb=0., ub=np.sqrt(max(0., ub))+1e-7, name=f'std_{k}')
                model.addConstr(v <= s*s)
                model.addConstr(mu >= float(row['z'])*s + BUFFER)
            else:
                model.addConstr(expression(u, evidence.gate_q[k], evidence.gate_a[k], evidence.gate_c[k]) >= BUFFER)
        for _,_,_,q,a in getattr(evidence, 'guards', []):
            model.addConstr(expression(u, q, a, 0.) >= 0.)
        for q,a,c in zip(getattr(evidence, 'progress_q', []), getattr(evidence, 'progress_a', []), getattr(evidence, 'progress_c', [])):
            model.addConstr(expression(u,q,a,c) >= -1e-9)
        model.setObjective(fee + prices @ (peak-paid), gp.GRB.MINIMIZE)
        if best is not None:
            u.Start = best[2]
        elif warm is not None:
            try:
                if case.audit(warm)['passed']:
                    u.Start = case.array(warm)
            except Exception:
                pass
        model.Params.TimeLimit = max(.001, seconds-(perf_counter()-began))
        receipt['optimizer_invoked'] = True
        model.optimize()
        receipt.update(native_status=int(model.Status), solution_count=int(model.SolCount),
                       native_runtime=float(model.Runtime))
        try:
            raw_bound = float(model.ObjBound)
            if not np.isfinite(raw_bound) or abs(raw_bound) > 1e90:
                raw_bound = None
        except (AttributeError, gp.GurobiError):
            raw_bound = None
        receipt['raw_native_bound_yuan'] = raw_bound
        if model.SolCount:
            candidate = np.asarray(u.X)
            audit = check(case, evidence, candidate)
            receipt['native_audit'] = audit
            if audit['passed']:
                cost = objective(case,candidate,paid,prices)
                if best is None or cost < best[0]-1e-9:
                    best = (cost, 'native_found' if best is None else 'native_improved_initial', candidate, audit)
        if best is None:
            receipt.update(status='native_infeasible' if model.Status==3 else 'unresolved')
        else:
            lower = None
            if raw_bound is not None and best[3]['encoded_passed']:
                if raw_bound <= best[0]+1e-7*max(1., best[0]):
                    lower = max(0., min(raw_bound, best[0]))
                else:
                    receipt['bound_withdrawn'] = 'Native bound exceeds independently checked feasible cost'
            elif raw_bound is not None:
                receipt['bound_withdrawn'] = 'Point passes original direct audit but not every encoded buffer; encoded bound not compared to its cost'
            receipt.update(status='feasible', feasible=True, source=best[1], cost_yuan=best[0],
                lower_bound_yuan=lower, relative_gap=None if lower is None else
                (best[0]-lower)/max(1e-12,abs(best[0])), audit=best[3])
    receipt['seconds'] = perf_counter()-began
    return None if best is None else best[2], receipt
