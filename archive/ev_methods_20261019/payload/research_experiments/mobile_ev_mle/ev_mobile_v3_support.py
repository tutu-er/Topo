"""Deterministic tangent MILP warm start for the frozen quadratic proxy.

For each convex pair q, ell(u)=q(p)+grad(q,p)'(u-p) <= q(u).
The mean-nearest aggregate preserves this inequality and equality at p.
This auxiliary MILP supplies a feasible optimized start, not a global solution
or upper bound for the quadratic maximization. No route library is used.
"""
from __future__ import annotations

from time import perf_counter
import traceback

import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import csr_matrix, hstack

from . import ev_mobile_v3_projection as proxy
from .ev_original_mip_backend import solve_linear_mip


def _tangents(pairs, parent_u):
    parent = np.asarray(parent_u, float).ravel()
    result = []
    for p in pairs:
        residual = p.projected_residual + p.projected_control @ (parent-p.reference_u.ravel())
        gradient = p.projected_control.T @ residual
        q0 = proxy.evaluate_pair(p, parent_u)
        result.append(dict(a=p.a_label, b=p.b_label, parent_q=float(q0),
                           gradient=gradient, intercept=float(q0-gradient@parent)))
    return result


def _evaluate(rows, labels, u):
    flat = np.asarray(u, float).ravel()
    values = [dict(a=r['a'], b=r['b'], value=float(r['intercept']+r['gradient']@flat)) for r in rows]
    nearest = [dict(a=a, value=min(r['value'] for r in values if r['a']==a)) for a in labels]
    return dict(score=float(np.mean([r['value'] for r in nearest])), nearest=nearest, pairs=values)


def solve_support_start(pairs, domain, parent_operation, *, solver_seconds=.75,
                        total_seconds=2., backend='gurobi', mip_rel_gap=1e-8,
                        feasibility_tolerance=1e-7, gradient_zero_tolerance=1e-12):
    """Return a parent-protected full operation x for an upper-PWL warm start.

    ``domain`` can be the public domain or its tightened operation dictionary.
    Parent x is complete, so no extra completion solve is needed. There is at
    most one MILP call. Its dual bounds only the tangent lower problem; exact
    quadratic scores select between its physically valid proposal and parent.
    A zero gradient, exhausted budget or solver failure retains the parent.
    """
    began = perf_counter()
    if (not np.isfinite([solver_seconds,total_seconds,mip_rel_gap,feasibility_tolerance,
                         gradient_zero_tolerance]).all() or solver_seconds<=0 or total_seconds<0
            or mip_rel_gap<0 or feasibility_tolerance<=0 or gradient_zero_tolerance<0):
        raise ValueError('Finite nonnegative budget/tolerances and positive solver time required')
    pairs, labels = proxy._pairs_and_labels(pairs)
    op = proxy._operation(domain, pairs[0].control_dimension)
    shape = pairs[0].control_shape
    parent = np.asarray(parent_operation, float)
    parent_audit = proxy._audit(parent, op, feasibility_tolerance)
    out = dict(success=False, MILP_calls=0, failures=[], all_pairs_count=len(pairs),
        support_optimization_completed=False,
        candidate_names=list(labels), parent_audit=parent_audit, backend=backend,
        solver_seconds=float(solver_seconds), total_seconds=float(total_seconds),
        random_proposals=0, complete_route_enumeration=False,
        bound_scope='Solver dual is an upper bound only for the tangent lower MILP, not for max Q.',
        proof='q(u)-ell(u)=.5*||C*(u-parent)||^2 >= 0; min and mean preserve order; equality at parent.',
        parameter_optimization_calls=0, AC_calls=0, gradient_zero_tolerance=gradient_zero_tolerance)
    if not parent_audit['feasible']:
        out.update(reason='invalid_parent_operation',seconds=perf_counter()-began)
        return out
    parent_u = np.asarray(op['U']@parent).reshape(shape)
    parent_exact = proxy.aggregate_pairs(pairs, parent_u)
    rows = _tangents(pairs, parent_u)
    parent_support = _evaluate(rows, labels, parent_u)
    if abs(parent_exact['score']-parent_support['score'])>1e-8*(1+abs(parent_exact['score'])):
        raise RuntimeError('Tangent and exact quadratic disagree at parent')
    selected = dict(x=parent.copy(), u=parent_u.copy(), exact=parent_exact,
                    support=parent_support, origin='parent_operation', audit=parent_audit)
    out.update(parent_exact_score=parent_exact['score'],parent_support_score=parent_support['score'],
        tangent_rows=rows, n_operation_variables=len(parent), n_reference_eta=len(labels),
        gradient_max_absolute=float(max(np.max(abs(r['gradient']),initial=0.) for r in rows)))
    N, K = len(parent), len(labels)
    constraints = [LinearConstraint(hstack((c.A,csr_matrix((c.A.shape[0],K))),format='csr'),c.lb,c.ub)
                   for c in op['constraints']]
    lower, upper = list(op['lower']), list(op['upper'])
    for a in labels:
        relevant=[r for r in rows if r['a']==a]
        low=[r['intercept']+np.where(r['gradient']>=0,r['gradient']*op['control_lower'],
            r['gradient']*op['control_upper']).sum() for r in relevant]
        high=[r['intercept']+np.where(r['gradient']>=0,r['gradient']*op['control_upper'],
            r['gradient']*op['control_lower']).sum() for r in relevant]
        lower.append(float(min(low)));upper.append(float(min(high)))
    for r in rows:
        row=np.zeros(N+K);row[:N]=-np.asarray(r['gradient']@op['U']).ravel()
        row[N+labels.index(r['a'])]=1.
        constraints.append(LinearConstraint(csr_matrix(row[None,:]),-np.inf,r['intercept']))
    objective=np.r_[np.zeros(N),-np.ones(K)/K]
    integer=np.r_[op['integrality'],np.zeros(K,int)]
    bounds=Bounds(lower,upper)
    start=np.r_[parent,[r['value'] for r in parent_support['nearest']]]
    master=dict(lower=np.asarray(lower),upper=np.asarray(upper),integrality=integer,
                constraints=constraints)
    out['master_start_audit']=proxy._audit(start,master,feasibility_tolerance)
    if not out['master_start_audit']['feasible']:
        raise RuntimeError('Complete parent warm start violates the tangent master')
    remaining=float(total_seconds)-(perf_counter()-began)
    if out['gradient_max_absolute']<=gradient_zero_tolerance:
        out['reason']='zero_pair_gradients_keep_parent; upper_PWL_exploration_still_needed'
    elif remaining<=0:
        out['reason']='budget_exhausted_before_support_solver'
    else:
        out['MILP_calls']=1
        solve_began=perf_counter()
        try:
            result=solve_linear_mip(objective,integer,bounds,constraints,
                time_limit=min(float(solver_seconds),remaining),mip_rel_gap=mip_rel_gap,
                start=start,backend=backend)
            dual=getattr(result,'mip_dual_bound',None)
            bound=None if dual is None or not np.isfinite(dual) else -float(dual)
            solver={key:getattr(result,key,None) for key in ('status','message','fun','mip_gap',
                'mip_node_count','backend_used','backend_requested','fallback_used','fallback_reason',
                'start_supplied','start_used','start_policy','backend_seconds')}
            solver.update(support_best_bound=bound,seconds=perf_counter()-solve_began,
                dual_scope='Tangent lower MILP maximum only; not a quadratic-design upper bound')
            solver['bound_consistent_with_parent'] = (None if bound is None else
                bool(bound >= parent_support['score']-1e-7*(1+abs(parent_support['score']))))
            if solver['bound_consistent_with_parent'] is False:
                out['failures'].append(dict(kind='support_dual_below_feasible_parent',
                    support_best_bound=bound,parent_support_score=parent_support['score']))
            out['solver']=solver
            if result.x is None:
                out['failures'].append(dict(kind='support_solver_no_incumbent',status=solver['status'],message=solver['message']))
            else:
                vector=np.asarray(result.x,float)
                solver['master_incumbent_audit']=proxy._audit(vector,master,feasibility_tolerance)
                proposed=vector[:N]
                audit=proxy._audit(proposed,op,feasibility_tolerance)
                solver['operation_incumbent_audit']=audit
                if not audit['feasible']:
                    out['failures'].append(dict(kind='support_physical_violation',audit=audit))
                else:
                    u=np.asarray(op['U']@proposed).reshape(shape)
                    exact=proxy.aggregate_pairs(pairs,u);linear=_evaluate(rows,labels,u)
                    repaired=np.r_[proposed,[r['value'] for r in linear['nearest']]]
                    solver['reconstructed_master_audit']=proxy._audit(repaired,master,feasibility_tolerance)
                    out['support_optimization_completed']=bool(solver['status']==0
                        and solver['master_incumbent_audit']['feasible'])
                    if not solver['master_incumbent_audit']['feasible']:
                        out['failures'].append(dict(kind='support_auxiliary_violation',audit=solver['master_incumbent_audit']))
                    out['proposal']=dict(power_kw=u,operation_x=proposed,exact_score=exact['score'],
                        exact_aggregation=exact,support_score=linear['score'],support_aggregation=linear,audit=audit)
                    # The exact Q comparison, rather than status or the tangent
                    # dual, protects signed/negative parent scores as well.
                    if exact['score']>selected['exact']['score']:
                        selected=dict(x=proposed.copy(),u=u.copy(),exact=exact,support=linear,
                                      origin='support_MILP',audit=audit)
            out['reason']='support_solve_finished_with_parent_protection'
        except Exception as exc:
            out['failures'].append(dict(kind='support_solver_exception',failure=type(exc).__name__+': '+str(exc),
                                        traceback=traceback.format_exc(),seconds=perf_counter()-solve_began))
            out['reason']='support_solver_exception_keep_parent'
    out.update(success=True,operation_x=selected['x'],power_kw=selected['u'],
        exact_score=selected['exact']['score'],exact_aggregation=selected['exact'],
        support_score=selected['support']['score'],support_aggregation=selected['support'],
        selected_origin=selected['origin'],selected_audit=selected['audit'],
        inheritance_protected=selected['exact']['score']>=parent_exact['score'],
        exact_gain=selected['exact']['score']-parent_exact['score'],
        seconds=perf_counter()-began)
    out['soft_budget_overrun_seconds']=max(0.,out['seconds']-total_seconds)
    return out
