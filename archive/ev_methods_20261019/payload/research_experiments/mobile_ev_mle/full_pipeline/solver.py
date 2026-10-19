"""Uniform square-secant MILP with strict original-operation replay.

The objective is min_k (a_k + b_k u + q_k u**2), q_k >= 0.
Only this frozen quadratic surrogate is bounded. Every returned operation
passes the original physical replay. An invalid solver incumbent is retained
in diagnostics and may be repaired by an LP fixing *all* operation integers.
No complete route or power-curve enumeration is used.
"""
from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, linprog, milp
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

from ..g_precision.milp import _plain, _task_arrays, _values, _box_upper

OP_TOLERANCE = 1e-7
SOLVER_TOLERANCE = 1e-9


def operation_diagnostics(domain, x, *, tolerance=OP_TOLERANCE):
    """Full original replay plus every offending row/variable, not just a flag."""
    x = np.asarray(x, float)
    n = domain.model.n_variables
    if x.shape != (n,) or not np.isfinite(x).all():
        return dict(feasible=False, issues=['shape_or_nonfinite'], max_violation=None)
    arrays = domain.model.arrays()
    con = arrays['constraints']
    lower, upper = np.asarray(domain.model.lower), np.asarray(domain.model.upper)
    integer = np.asarray(domain.model.integer, bool)
    ax = np.asarray(con.A @ x).ravel()
    vl, vu = np.maximum(lower-x, 0), np.maximum(x-upper, 0)
    rl, ru = np.maximum(np.asarray(con.lb)-ax, 0), np.maximum(ax-np.asarray(con.ub), 0)
    iv = np.where(integer, abs(x-np.rint(x)), 0)
    labels = {}
    for label in ('u', 'charge', 'discharge', 'energy', 'mode', 'arc'):
        ids = np.asarray(getattr(domain.indices, label))
        for index in np.ndindex(ids.shape):
            labels[int(ids[index])] = dict(kind=label, coordinates=list(index))
    variables = [dict(index=int(i), value=float(x[i]), lower=float(lower[i]),
        upper=float(upper[i]), lower_violation=float(vl[i]), upper_violation=float(vu[i]),
        integrality_violation=float(iv[i]), **labels.get(int(i), {}))
        for i in np.flatnonzero(np.maximum.reduce((vl, vu, iv)) > tolerance)]
    rows = [dict(index=int(i), value=float(ax[i]), lower=float(con.lb[i]),
        upper=float(con.ub[i]), lower_violation=float(rl[i]), upper_violation=float(ru[i]))
        for i in np.flatnonzero(np.maximum(rl, ru) > tolerance)]
    replay = domain.check(x, tolerance=tolerance)
    return dict(**replay, tolerance=tolerance, solver_tolerance=SOLVER_TOLERANCE,
        box_violation=float(max(vl.max(initial=0), vu.max(initial=0))),
        linear_violation=float(max(rl.max(initial=0), ru.max(initial=0))),
        integer_violation=float(iv.max(initial=0)), variable_violations=variables,
        linear_row_violations=rows)


def repair_operation(domain, x, *, time_limit=1.):
    """Nearest L1 continuous plan with the incumbent's integer path/modes fixed.

    The LP includes every original operation constraint. Rounding proposes an
    integer pattern; it does not by itself make the incumbent feasible.
    """
    began = perf_counter()
    x = np.asarray(x, float)
    before = operation_diagnostics(domain, x)
    n = domain.model.n_variables
    if x.shape != (n,) or not np.isfinite(x).all():
        return None, dict(success=False, before=before, reason='invalid_vector', seconds=perf_counter()-began)
    integer = np.asarray(domain.model.integer, bool)
    low, high = np.asarray(domain.model.lower, float).copy(), np.asarray(domain.model.upper, float).copy()
    rounded = np.rint(x[integer])
    if np.any(rounded < low[integer]) or np.any(rounded > high[integer]):
        return None, dict(success=False, before=before, reason='rounded_integer_outside_bounds', seconds=perf_counter()-began)
    low[integer] = high[integer] = rounded
    con = domain.model.arrays()['constraints']
    A = csr_matrix(con.A)
    cl, cu = np.asarray(con.lb), np.asarray(con.ub)
    eq = np.isfinite(cl) & np.isfinite(cu) & (cl == cu)
    lo, hi = np.isfinite(cl) & ~eq, np.isfinite(cu) & ~eq
    eye = csr_matrix(np.eye(n))
    zero = csr_matrix((A.shape[0], n))
    aug = hstack((A, zero), format='csr')
    Aub = vstack((aug[hi], -aug[lo], hstack((eye, -eye)), hstack((-eye, -eye))), format='csr')
    bub = np.r_[cu[hi], -cl[lo], x, -x]
    result = linprog(np.r_[np.zeros(n), np.ones(n)], A_ub=Aub, b_ub=bub,
        A_eq=aug[eq] if np.any(eq) else None, b_eq=cl[eq] if np.any(eq) else None,
        bounds=list(zip(np.r_[low, np.zeros(n)], np.r_[high, np.full(n, np.inf)])),
        method='highs', options=dict(time_limit=max(float(time_limit), .01),
            primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
    fixed = None if result.x is None else np.asarray(result.x[:n])
    after = None if fixed is None else operation_diagnostics(domain, fixed)
    success = bool(after is not None and after['feasible'])
    return fixed if success else None, dict(success=success, before=before, after=after,
        status=int(result.status), message=str(result.message),
        l1_change=None if fixed is None else float(np.sum(abs(fixed-x))),
        max_change=None if fixed is None else float(np.max(abs(fixed-x), initial=0)),
        fixed_integer_count=int(integer.sum()), fixed_integer_values=rounded.tolist(),
        all_original_constraints_included=True, seconds=perf_counter()-began)


def _backend(c, integer, low, high, constraints, *, seconds, relative_gap, backend, start=None):
    began = perf_counter()
    if backend == 'scipy':
        result = milp(c, integrality=integer, bounds=Bounds(low, high), constraints=constraints,
            options=dict(time_limit=float(seconds), mip_rel_gap=float(relative_gap), presolve=True))
        result.backend_details = dict(backend_used='scipy', strict_gurobi_tolerances=False,
            start_used=False, threads=None)
        return result
    import gurobipy as gp
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0); env.start()
        with gp.Model(env=env) as model:
            model.Params.OutputFlag=0; model.Params.Threads=1; model.Params.Seed=0
            model.Params.TimeLimit=float(seconds); model.Params.MIPGap=float(relative_gap)
            model.Params.FeasibilityTol=SOLVER_TOLERANCE
            model.Params.IntFeasTol=SOLVER_TOLERANCE
            model.Params.OptimalityTol=SOLVER_TOLERANCE
            model.Params.NumericFocus=1
            kinds=np.where(np.asarray(integer, bool), gp.GRB.INTEGER, gp.GRB.CONTINUOUS)
            kinds=np.where(np.asarray(integer, bool) & (low >= 0) & (high <= 1), gp.GRB.BINARY, kinds)
            z=model.addMVar(len(c), lb=low, ub=high, obj=c, vtype=kinds)
            A=csr_matrix(constraints.A); cl=np.asarray(constraints.lb); cu=np.asarray(constraints.ub)
            eq=np.isfinite(cl)&np.isfinite(cu)&(cl==cu)
            for mask,sense,rhs in ((eq,'=',cl),(np.isfinite(cl)&~eq,'>',cl),(np.isfinite(cu)&~eq,'<',cu)):
                if np.any(mask):model.addMConstr(A[mask],z,sense,rhs[mask])
            if start is not None:z.Start=np.where(np.isnan(start),gp.GRB.UNDEFINED,start)
            model.optimize(); status=int(model.Status)
            sol=np.asarray(z.X) if model.SolCount else None
            dual=float(model.ObjBound) if model.IsMIP else (float(model.ObjVal) if status==gp.GRB.OPTIMAL else None)
            if dual is not None and not np.isfinite(dual):dual=None
            return OptimizeResult(x=sol, fun=None if sol is None else float(c@sol),
                status=0 if status==gp.GRB.OPTIMAL else 1 if status==gp.GRB.TIME_LIMIT else 4,
                message='Gurobi status '+str(status), success=status==gp.GRB.OPTIMAL,
                mip_dual_bound=dual, mip_gap=float(model.MIPGap) if model.SolCount else None,
                mip_node_count=float(model.NodeCount), backend_details=dict(backend_used='gurobi',
                    gurobi_status=status, threads=1, seed=0, FeasibilityTol=SOLVER_TOLERANCE,
                    IntFeasTol=SOLVER_TOLERANCE, OptimalityTol=SOLVER_TOLERANCE, NumericFocus=1,
                    start_used=start is not None, gurobi_version='.'.join(map(str,gp.gurobi.version())),
                    seconds=perf_counter()-began))


def _master(domain, data, scale, segments, *, seconds, backend, parent, attained_lower):
    """Build one common square graph; all pair coefficients use these squares."""
    operation=domain.as_dict(); con=domain.model.arrays()['constraints']
    low=list(domain.model.lower); high=list(domain.model.upper); integer=list(domain.model.integer)
    base_n=len(low); uid=np.asarray(domain.indices.u).ravel()
    ulo=np.asarray(operation['control_lower'], float); uhi=np.asarray(operation['control_upper'],float)
    rows=[]; rlo=[]; rhi=[]
    def variable(lo,hi,kind=0):
        idx=len(low);low.append(float(lo));high.append(float(hi));integer.append(kind);return idx
    def row(entries,lo=-np.inf,hi=np.inf):
        rows.append(entries);rlo.append(float(lo));rhi.append(float(hi))
    for i in range(con.A.shape[0]):
        rr=csr_matrix(con.A).getrow(i);row(dict(zip(rr.indices,rr.data)),con.lb[i],con.ub[i])
    squares=[]; selector=[]; errors=[]; n_binary=0
    for k,(l,h) in enumerate(zip(ulo,uhi)):
        if l==h:
            squares.append(variable(l*l,l*l));errors.append(0.);continue
        nodes=np.linspace(l,h,int(segments)+1)
        if l<0<h:nodes=np.unique(np.r_[nodes,0.])
        sid=variable(0. if l<=0<=h else min(l*l,h*h),max(l*l,h*h));squares.append(sid)
        delta=[variable(0,1,1) for _ in range(len(nodes)-1)]
        frac=[variable(0,1) for _ in delta];n_binary+=len(delta)
        row(dict.fromkeys(delta,1.),1.,1.)
        eu={int(uid[k]):1.};es={sid:1.}
        for d,f,a,b in zip(delta,frac,nodes[:-1],nodes[1:]):
            row({f:1.,d:-1.},hi=0.)
            eu[d]=-float(a);eu[f]=-float(b-a);es[d]=-float(a*a);es[f]=-float(b*b-a*a)
        row(eu,0,0);row(es,0,0)
        selector.append((k,delta,frac,nodes))
        errors.append(float(np.max(np.diff(nodes)**2)/4))
    coarse=_box_upper(data,ulo,uhi)
    # No target lower cut: a feasible parent remains protected externally and
    # its partial start is submitted, avoiding numerically tight objective cuts.
    target=variable(-np.inf,coarse/scale)
    for task in data:
        entries={target:1.}
        for i,b,q,s in zip(uid,task['linear'].ravel(),task['quadratic'].ravel(),squares):
            if b:entries[int(i)]=entries.get(int(i),0)-float(b/scale)
            if q:entries[int(s)]=entries.get(int(s),0)-float(q/scale)
        row(entries,hi=task['constant']/scale)
    rr=[];cc=[];vv=[]
    for i,entries in enumerate(rows):
        for j,v in entries.items():
            if v:rr.append(i);cc.append(j);vv.append(v)
    matrix=coo_matrix((vv,(rr,cc)),shape=(len(rows),len(low))).tocsr()
    cost=np.zeros(len(low));cost[target]=-1.
    start=None
    if parent is not None:
        start=np.full(len(low),np.nan);start[:base_n]=parent
        u=np.asarray(parent)[uid];sq=u*u
        for k,d,f,nodes in selector:
            j=int(np.clip(np.searchsorted(nodes,u[k],side='right')-1,0,len(nodes)-2))
            t=float(np.clip((u[k]-nodes[j])/(nodes[j+1]-nodes[j]),0,1))
            start[d]=0;start[f]=0;start[d[j]]=1;start[f[j]]=t
            sq[k]=(1-t)*nodes[j]**2+t*nodes[j+1]**2
        start[squares]=sq
    result=_backend(cost,np.asarray(integer),np.asarray(low),np.asarray(high),
        LinearConstraint(matrix,rlo,rhi),seconds=seconds,relative_gap=1e-8,backend=backend,start=start)
    raw=None if result.x is None else np.asarray(result.x[:base_n]).copy()
    dual=getattr(result,'mip_dual_bound',None)
    numerical_guard=1024*np.finfo(float).eps*(1+abs(coarse))
    upper=coarse if dual is None or not np.isfinite(dual) else min(coarse,-float(dual)*scale+numerical_guard)
    return raw,dict(status=int(result.status),message=str(result.message),solver_success=bool(result.success),
        solver_mip_gap=getattr(result,'mip_gap',None),solver_nodes=getattr(result,'mip_node_count',None),
        backend=backend,backend_details=result.backend_details,master_form='shared_control_square_strict',
        n_variables=len(low),n_rows=len(rows),n_binary=int(np.sum(integer)),
        n_shared_square_binary=n_binary,n_shared_square_controls=len(selector),
        objective_upper=upper,solver_best_bound=None if dual is None else -float(dual)*scale,
        chord_incumbent=None if result.fun is None else -float(result.fun)*scale,
        square_segment_error_upper=float(max(np.asarray(t['quadratic']).ravel()@errors for t in data)),
        zero_knot_included=True,incumbent_is_original_lower=False)


def solve_tasks(domain,tasks,parents,*,iterations=2,segments=4,max_segments=16,
                solver_seconds=2.,total_seconds=12.,backend='gurobi',relative_gap=.01,absolute_gap=1e-5):
    """Same task interface as g_precision; changed solver policy is explicit."""
    began=perf_counter()
    if backend not in ('gurobi','scipy'):raise ValueError('backend must be gurobi or scipy')
    if not isinstance(parents,Mapping):raise ValueError('parents must map names to full operations')
    if any(isinstance(v,bool) or int(v)!=v or v<1 for v in (iterations,segments,max_segments)) or max_segments<segments:
        raise ValueError('positive integer iterations/segments and max_segments>=segments required')
    if not np.isfinite([solver_seconds,total_seconds,relative_gap,absolute_gap]).all() or min(solver_seconds,total_seconds)<=0 or min(relative_gap,absolute_gap)<0:
        raise ValueError('positive budgets and nonnegative gaps required')
    shape=domain.indices.u.shape;data=_task_arrays(tasks,shape)
    operation=domain.as_dict();ulo=np.asarray(operation['control_lower']);uhi=np.asarray(operation['control_upper'])
    magnitude=max(abs(t['constant'])+float(np.sum(abs(t['linear'].ravel())*np.maximum(abs(ulo),abs(uhi))+t['quadratic'].ravel()*np.maximum(abs(ulo),abs(uhi))**2)) for t in data)
    scale=max(magnitude,1e-12)
    if not np.isfinite(scale):raise ValueError('task magnitude overflows the operation domain')
    upper=_box_upper(data,ulo,uhi);coarse=upper
    candidates=[];failures=[];parent_logs={};logs=[]
    def keep(x,origin):
        audit=operation_diagnostics(domain,x)
        if not audit['feasible']:raise ValueError(str(audit['issues']))
        power=np.asarray(domain.decode(x)['u_kw'])
        values,score,lower=_values(data,power)
        row=dict(operation_x=np.asarray(x).copy(),power_kw=power,task_values=values,
            score=score,score_lower=lower,origin=origin,operation_check=audit)
        candidates.append(row);return row
    for name,x in parents.items():
        try:parent_logs[name]=dict(success=True,**keep(x,'parent:'+name))
        except Exception as exc:
            parent_logs[name]=dict(success=False,failure=str(exc))
            failures.append(dict(kind='parent_rejected',origin=name,message=str(exc)))
    best_parent=max(candidates,key=lambda r:r['score']) if candidates else None
    for it in range(int(iterations)):
        remaining=total_seconds-(perf_counter()-began)
        if remaining<=0:break
        winner=max(candidates,key=lambda r:r['score']) if candidates else None
        count=min(int(max_segments),int(segments)*2**it)
        t=perf_counter()
        try:
            raw,log=_master(domain,data,scale,count,seconds=min(float(solver_seconds),remaining),
                backend=backend,parent=None if winner is None else winner['operation_x'],
                attained_lower=None if winner is None else winner['score_lower'])
            log.update(iteration=it,control_segments=count,seconds=perf_counter()-t)
            logs.append(log)
            proposed=log['objective_upper']
            if winner is not None and proposed<winner['score_lower']:
                failures.append(dict(kind='solver_upper_below_feasible_original_lower',iteration=it,
                    rejected_upper=proposed,lower=winner['score_lower']))
            else:upper=min(upper,proposed)
            if raw is not None:
                audit=operation_diagnostics(domain,raw);log['raw_operation_validation']=audit
                if not audit['feasible']:
                    log['raw_operation_x']=raw.tolist()
                    failures.append(dict(kind='raw_master_operation_rejected',iteration=it,diagnostics=audit))
                    remaining=total_seconds-(perf_counter()-began)
                    fixed,repair=repair_operation(domain,raw,time_limit=min(1.,max(remaining,.01)))
                    log['operation_repair']=repair
                    if fixed is not None:keep(fixed,'repaired:'+str(it))
                    else:failures.append(dict(kind='operation_repair_failed',iteration=it,diagnostics=repair))
                else:keep(raw,'master:'+str(it))
            else:failures.append(dict(kind='master_without_incumbent',iteration=it,message=log['message']))
        except Exception as exc:
            failures.append(dict(kind='master_exception',iteration=it,message=type(exc).__name__+': '+str(exc)))
        winner=max(candidates,key=lambda r:r['score']) if candidates else None
        if winner is not None and upper<winner['score_lower']:
            failures.append(dict(kind='numerical_upper_below_original_lower',iteration=it,
                rejected_upper=upper,lower=winner['score_lower']))
            upper=coarse
        if winner is not None and upper-winner['score_lower']<=absolute_gap+relative_gap*max(abs(upper),abs(winner['score_lower']),1e-12):break
    winner=max(candidates,key=lambda r:r['score']) if candidates else None
    lower=None if winner is None else winner['score_lower']
    gap=None if lower is None else upper-lower
    target=None if lower is None else absolute_gap+relative_gap*max(abs(upper),abs(lower),1e-12)
    complete=bool(winner is not None and gap>=0 and gap<=target)
    return _plain(dict(success=winner is not None,
        operation_x=None if winner is None else winner['operation_x'],
        power_kw=None if winner is None else winner['power_kw'],
        task_values=None if winner is None else winner['task_values'],
        score=None if winner is None else winner['score'],
        selected_origin=None if winner is None else winner['origin'],
        best_parent_score=None if best_parent is None else best_parent['score'],
        score_gain=None if winner is None or best_parent is None else winner['score']-best_parent['score'],
        inheritance_protected=bool(winner is not None and (best_parent is None or winner['score']>=best_parent['score'])),
        objective_lower=lower,objective_upper=upper,gap=gap,target_gap=target,completed=complete,unfinished=not complete,
        coarse_box_upper=coarse,represented_surrogate_upper=upper,completion_error_bound=0.,objective_scale=scale,
        nuisance_dimension=0,task_names=[t['name'] for t in data],parents=parent_logs,iterations=logs,n_iterations=len(logs),
        status='completed' if complete else 'budget_or_segment_gap',failures=failures,backend=backend,
        seconds=perf_counter()-began,operation_check=None if winner is None else winner['operation_check'],
        solver_policy='strict_tolerances_then_fixed_integer_continuous_L1_repair',
        bound_scope='fixed affine plus diagonal quadratic surrogate; numerical solver bounds; no nonlinear AC certificate',
        uses_nuisance_refit=False,incumbent_is_original_score_lower=False,
        chord_incumbent_is_original_lower=False,returned_score_is_original_feasible_plan_score=True,
        full_route_or_curve_enumeration=False))


__all__=['solve_tasks','operation_diagnostics','repair_operation']
