"""All-candidate scalar source-profile MILP, with optional history-loss offsets.

Every topology keeps one trained line vector. Pairwise QP tables share the EV
operation weights/segments. Certificates concern frozen affine residuals;
complete AC fits decide whether each candidate action is accepted.
"""
from __future__ import annotations

from hashlib import sha256
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds,LinearConstraint
from scipy.sparse import csr_matrix,hstack

from . import ev_scalar_distance_milp as scalar
from . import ev_fixed_train_design as fixed
from . import ev_gradient_flow_design as gradient
from .ev_original_domain_milp import evaluate_original_pair,_solve_mip


def _groups(context):
    names=[c.name for c in context['bank']];models=context['models']
    if len(names)<2 or len(set(names))!=len(names):raise ValueError('at least two distinct candidates required')
    actual=[(row['a'],row['b']) for row in models]
    expected={(a,b) for a in names for b in names if a!=b}
    if len(actual)!=len(expected) or set(actual)!=expected or len(context['pairs'])!=len(expected):
        raise ValueError('all directed candidate pairs must be retained exactly once')
    if any(pair.name!=row.get('name') for pair,row in zip(context['pairs'],models)):
        raise ValueError('candidate-pair matrices and metadata order differ')
    return names,{a:[i for i,m in enumerate(models) if m['a']==a] for a in names}


def _offsets(context,history_losses=None):
    names,_=_groups(context)
    if history_losses is None:history_losses={name:0. for name in names}
    if set(history_losses)!=set(names) or not all(np.isfinite(v) for v in history_losses.values()):
        raise ValueError('one finite known history loss per candidate required')
    return np.array([float(history_losses[m['b']])-float(history_losses[m['a']])
        for m in context['models']])


def aggregate_values(context,values,history_losses=None):
    names,groups=_groups(context);values=np.asarray(values,float)
    if values.shape!=(len(context['pairs']),) or not np.isfinite(values).all():
        raise ValueError('finite value required for every directed pair')
    offsets=_offsets(context,history_losses);shifted=values+offsets
    nearest=[]
    for a in names:
        best=min(shifted[i] for i in groups[a])
        active=[context['models'][i]['b'] for i in groups[a]
            if np.isclose(shifted[i],best,atol=1e-10,rtol=1e-10)]
        nearest.append(dict(a=a,score=float(best),active_competitors=active))
    return dict(score=float(np.mean([r['score'] for r in nearest])),nearest=nearest,
        pair_values=values,pair_offsets=offsets,pair_shifted_values=shifted,
        historical_offset_scope='known attained training loss differences, not exact expected LLR')


def _time_parts(context):
    if not context.get('success'):raise ValueError('successful fixed context required')
    _groups(context);T,n=context['control_shape'];m=T*(n+2);parts=[]
    for pair in context['pairs']:
        if (pair.c.shape!=(m,) or pair.B.shape!=(m,T*n) or pair.A0.shape!=(m,2*T)
                or np.any(pair.At) or np.any(pair.shift)):
            raise ValueError('fixed affine At=shift=0 and two independent source coordinates per time required')
        blocks=[]
        for t in range(T):
            rows=np.r_[np.arange(t*n,(t+1)*n),T*n+t,T*n+T+t]
            controls=np.arange(t*n,(t+1)*n);sources=np.arange(2*t,2*t+2)
            if np.any(pair.B[np.ix_(rows,np.setdiff1d(np.arange(T*n),controls))]) or np.any(
                    pair.A0[np.ix_(rows,np.setdiff1d(np.arange(2*T),sources))]):
                raise ValueError('cross-time electrical coupling prevents scalar profiles')
            blocks.append(dict(c=pair.c[rows],B=pair.B[np.ix_(rows,controls)],G=pair.A0[np.ix_(rows,sources)],
                lower=pair.lower[sources],upper=pair.upper[sources],center=pair.base_center[sources]))
        parts.append(blocks)
    return parts


def build_systematic_tables(context,domain,*,grid_points=3,cache=None,deadline=None):
    began=perf_counter();parts=_time_parts(context);T,n=context['control_shape'];P=len(parts)
    if grid_points not in (3,5,9,17):raise ValueError('declared grids are 3/5/9/17')
    if domain.indices.u.shape!=(T,n):raise ValueError('operation shape differs')
    op=domain.as_dict();lo=op['control_lower'].reshape(T,n);hi=op['control_upper'].reshape(T,n)
    if np.any(lo>0) or np.any(hi<0):raise ValueError('each power domain must contain zero')
    cache={} if cache is None else cache;calls=hits=0;tables=[];background=[];pair_errors=[]
    grids=[[np.unique(np.r_[lo[t,j],0.,hi[t,j]] if grid_points==3 else
        np.r_[np.linspace(lo[t,j],hi[t,j],grid_points),0.]) for j in range(n)] for t in range(T)]
    def node(k,t,j,p):
        nonlocal calls,hits
        row=parts[k][t];b=row['B'][:,j];digest=sha256()
        for val in (row['c'],b if p else np.zeros_like(b),row['G'],row['lower'],row['upper'],row['center'],
                    np.array([p,lo[t,j],hi[t,j]])):
            arr=np.asarray(val,dtype='<f8');digest.update(repr(arr.shape).encode());digest.update(arr.tobytes())
        key=digest.hexdigest()
        if key in cache:hits+=1;return cache[key]
        if deadline is not None and perf_counter()>=deadline:raise TimeoutError('table source-QP budget exhausted')
        value=scalar.scalar_profile(row['c'],b,row['G'],row['lower'],row['upper'],row['center'],float(p),
            power_range=(lo[t,j],hi[t,j]));calls+=1
        raw=value['source_QP']
        value['source_QP']={key:raw[key] for key in ('success','status','message','optimality',
            'physical_bound_violation','active_lower','active_upper','fixed_indices','seconds')}
        cache[key]=value
        return value
    def zero_for_site(value,k,t,j):
        # Share the zero-injection source witness, but retain the correct
        # site-specific dual slope and guard without solving another QP.
        row=parts[k][t];b=row['B'][:,j];r=value['residual'];d=row['c']-row['G']@row['center']
        g=row['G'].T@r;slope=float(b@r);pscale=max(abs(lo[t,j]),abs(hi[t,j]))
        guard=512*np.finfo(float).eps*(1+abs(r@d)+.5*abs(r@r)+
            float(abs(g)@np.maximum(abs(row['lower']),abs(row['upper'])))+abs(slope)*pscale)
        intercept=value['dual_intercept']+value['dual_numeric_guard']-guard
        lower=max(0.,intercept)
        return dict(value,dual_slope=slope,dual_intercept=intercept,dual_numeric_guard=guard,
            dual_lower=lower,primal_dual_gap=max(0.,value['primal_upper']-lower),curvature_upper=float(b@b))
    try:
        for k in range(P):
            pair_table=[];pair_zero=[];error=0.
            for t in range(T):
                zero=node(k,t,0,0.);pair_zero.append(zero);time_table=[];errors=[zero['primal_dual_gap']]
                for j in range(n):
                    grid=grids[t][j];values=[zero_for_site(zero,k,t,j) if p==0 else node(k,t,j,float(p)) for p in grid]
                    L=float(parts[k][t]['B'][:,j]@parts[k][t]['B'][:,j]);h=float(np.max(np.diff(grid),initial=0.))
                    err=L*h*h/8+max(v['primal_dual_gap'] for v in values)
                    time_table.append(dict(grid=grid,actual_node_count=len(grid),
                        primal_upper=np.array([v['primal_upper'] for v in values]),
                        dual_lower=np.array([v['dual_lower'] for v in values]),nodes=values,
                        curvature_upper=L,spacing=h,curvature_chord_error=L*h*h/8,
                        endpoint_gap=max(v['primal_dual_gap'] for v in values),error_upper=err,
                        curve_parameters=dict(c=parts[k][t]['c'],b=parts[k][t]['B'][:,j],G=parts[k][t]['G'],
                            source_lower=parts[k][t]['lower'],source_upper=parts[k][t]['upper'],
                            source_center=parts[k][t]['center'],power_range=[lo[t,j],hi[t,j]])))
                    errors.append(err)
                pair_table.append(time_table);error+=max(errors)
            tables.append(pair_table);background.append(pair_zero);pair_errors.append(error)
    except TimeoutError as exc:
        return dict(success=False,failure=str(exc),QP_calls=calls,cache_hits=hits,seconds=perf_counter()-began,
            expected_directed_pairs=P,complete_directed_pair_tables=len(tables))
    _,groups=_groups(context)
    return dict(success=True,problem_id=scalar.problem_identity(context,domain),grid_points=grid_points,
        tables=tables,background=background,grids=grids,pair_error_upper=np.array(pair_errors),
        nearest_error_upper=float(np.mean([max(pair_errors[i] for i in ids) for ids in groups.values()])),
        QP_calls=calls,cache_hits=hits,seconds=perf_counter()-began,expected_directed_pairs=P,
        score_units='directed half-SSE before known history offsets',
        bound_scope='frozen affine profile; numerical dual guard recorded; no nonlinear AC bound')


def action_bounds(context,power_kw,history_losses=None):
    began=perf_counter();u=np.asarray(power_kw,float)
    results=[evaluate_original_pair(pair,u.ravel()) for pair in context['pairs']]
    lower=aggregate_values(context,[.5*r['tangent_lower_numeric'] for r in results],history_losses)
    upper=aggregate_values(context,[.5*r['primal_upper'] for r in results],history_losses)
    return dict(success=all(r['success'] for r in results),dual_lower=lower['score'],primal_upper=upper['score'],
        lower_aggregation=lower,upper_aggregation=upper,pairs=results,seconds=perf_counter()-began,
        scope='feasible-design lower from all source-QP duals; inner primals are upper values')


def _realize(domain,u,*,seconds=2.,backend='gurobi'):
    op=domain.as_dict();v=np.asarray(u,float).ravel()
    result=_solve_mip(np.zeros(len(op['lower'])),op['integrality'],Bounds(op['lower'],op['upper']),
        [*op['constraints'],LinearConstraint(op['u_map'],v,v)],time_limit=max(.001,float(seconds)),backend=backend)
    audit=domain.check(result.x)
    return dict(success=bool(result.x is not None and audit['feasible']),x=result.x,audit=audit,
        status=int(result.status),message=str(result.message))


def solve_systematic_milp(context,domain,*,tables,history_losses=None,initial_operation=None,
        solver_seconds=2.,backend='gurobi',mip_rel_gap=1e-8):
    began=perf_counter();names,groups=_groups(context);K=len(names);P=len(context['pairs']);T,n=context['control_shape']
    if not tables.get('success') or tables.get('problem_id')!=scalar.problem_identity(context,domain):
        raise ValueError('tables differ from frozen context/operation domain')
    offsets=_offsets(context,history_losses);op=domain.as_dict();N=len(op['lower']);points=tables['grid_points']
    indices={};size=N;add_binary=add_continuous=0
    if points!=3:
        for t in range(T):
            for j in range(n):
                count=len(tables['grids'][t][j]);lam=np.arange(size,size+count);size+=count
                seg=np.arange(size,size+max(count-1,0));size+=len(seg)
                indices[t,j]=(lam,seg);add_continuous+=count;add_binary+=len(seg)
    eta=np.arange(size,size+K);size+=K;add_continuous+=K;width=size-N
    lower=np.r_[op['lower'],np.zeros(width)];upper=np.r_[op['upper'],np.ones(width)]
    integer=np.r_[op['integrality'],np.zeros(width,int)]
    constraints=[LinearConstraint(hstack((csr_matrix(c.A),csr_matrix((c.A.shape[0],width))),format='csr'),c.lb,c.ub)
        for c in op['constraints']]
    expressions=np.zeros((P,size));constants=np.zeros(P);pair_coarse=[]
    if points!=3:
        for t in range(T):
            for j in range(n):
                grid=tables['grids'][t][j];lam,seg=indices[t,j];y=int(domain.indices.connected_arc[t,j])
                integer[seg]=1
                row=np.zeros(size);row[lam]=1.;row[y]-=1.;constraints.append(LinearConstraint(row,0.,0.))
                row=np.zeros(size);row[int(domain.indices.u[t,j])]=1.;row[lam]-=grid
                constraints.append(LinearConstraint(row,0.,0.))
                if len(seg):
                    row=np.zeros(size);row[seg]=1.;row[y]-=1.;constraints.append(LinearConstraint(row,0.,0.))
                    for k,index in enumerate(lam):
                        row=np.zeros(size);row[index]=1.
                        if k:row[seg[k-1]]-=1.
                        if k<len(seg):row[seg[k]]-=1.
                        constraints.append(LinearConstraint(row,-np.inf,0.))
    for pair_index in range(P):
        constants[pair_index]=sum(v['primal_upper'] for v in tables['background'][pair_index]);coarse=0.
        for t in range(T):
            q0=tables['background'][pair_index][t]['primal_upper'];time_max=q0
            for j in range(n):
                node=tables['tables'][pair_index][t][j];grid=node['grid'];values=node['primal_upper']
                time_max=max(time_max,float(np.max(values)))
                if points==3:
                    if grid[-1]>0:expressions[pair_index,domain.indices.charge[t,j]]+=(values[-1]-q0)/grid[-1]
                    if grid[0]<0:expressions[pair_index,domain.indices.discharge[t,j]]+=(values[0]-q0)/(-grid[0])
                else:
                    lam,_=indices[t,j];expressions[pair_index,lam]+=values
                    expressions[pair_index,int(domain.indices.connected_arc[t,j])]-=q0
            coarse+=time_max
        pair_coarse.append(coarse)
    for reference_index,a in enumerate(names):
        ids=groups[a];lower[eta[reference_index]]=min(offsets[i] for i in ids)-1e-9
        upper[eta[reference_index]]=min(pair_coarse[i]+offsets[i] for i in ids)+1e-9
        for i in ids:
            row=-expressions[i].copy();row[eta[reference_index]]+=1.
            constraints.append(LinearConstraint(row,-np.inf,constants[i]+offsets[i]))
    objective=np.zeros(size);objective[eta]=-1./K;start=None
    if initial_operation is not None:
        original=np.asarray(initial_operation,float)
        if domain.check(original)['feasible']:
            start=np.zeros(size);start[:N]=original
            if points!=3:
                power=np.asarray(op['u_map']@original).reshape(T,n)
                for t in range(T):
                    for j in range(n):
                        grid=tables['grids'][t][j];lam,seg=indices[t,j]
                        connected=original[int(domain.indices.connected_arc[t,j])]
                        if connected<.5:continue
                        if len(grid)==1:start[lam[0]]=1.;continue
                        p=float(np.clip(power[t,j],grid[0],grid[-1]))
                        at=int(np.clip(np.searchsorted(grid,p,side='right')-1,0,len(grid)-2))
                        alpha=(p-grid[at])/(grid[at+1]-grid[at]);start[lam[at]]=1.-alpha
                        start[lam[at+1]]=alpha;start[seg[at]]=1.
            values=constants+expressions@start+offsets
            for index,a in enumerate(names):start[eta[index]]=min(values[i] for i in groups[a])
    solve_began=perf_counter()
    result=_solve_mip(objective,integer,Bounds(lower,upper),constraints,time_limit=solver_seconds,
        mip_rel_gap=mip_rel_gap,backend=backend,start=start)
    solving=perf_counter()-solve_began
    x=None if result.x is None else np.asarray(result.x[:N],float);audit=domain.check(x)
    power=None if x is None else np.asarray(op['u_map']@x).reshape(T,n)
    action=None if not audit['feasible'] else action_bounds(context,power,history_losses)
    bound=getattr(result,'mip_dual_bound',None);ub=None if bound is None or not np.isfinite(bound) else -float(bound)
    lb=None if action is None else action['dual_lower']
    consistent=bool(ub is None or lb is None or ub+1e-8*(1+abs(ub))>=lb)
    return dict(success=bool(action is not None and action['success'] and consistent),power_kw=power,
        operation_x=x,operation=audit,grid_points=points,status=int(result.status),message=str(result.message),
        solver_proxy_incumbent=None if result.fun is None else -float(result.fun),
        solver_proxy_best_bound=ub,solver_mip_gap=getattr(result,'mip_gap',None),
        frozen_objective_lower=lb,frozen_objective_upper=ub,
        frozen_absolute_gap=None if lb is None or ub is None or not consistent else max(0.,ub-lb),
        bound_consistent=consistent,action_QP=action,added_binary_count=add_binary,
        added_continuous_count=add_continuous,reference_eta_count=K,
        source_pair_count=P,all_candidates_retained=True,original_binary_count=int(np.count_nonzero(op['integrality'])),
        history_offsets=offsets,table_problem_id=tables['problem_id'],
        nearest_chord_error_upper=tables['nearest_error_upper'],
        method='original_C_D_pair_chords' if points==3 else 'pair_shared_adjacent_SOS2',
        scope='global frozen affine nearest objective with known history offsets; not full AC',
        full_AC_global_gap=None,MILP_seconds=solving,seconds=perf_counter()-began)


def _AC_value(context,profile,history_losses):
    if not profile.get('success') or len(profile.get('pairs',()))!=len(context['pairs']):return None
    if not all(row.get('success') for row in profile['pairs']):return None
    return aggregate_values(context,[row['half_SSE'] for row in profile['pairs']],history_losses)


def scalar_policy(context,domain,initials,*,history_losses=None,grids=(3,9),total_seconds=15.,
        fit_options=None,seed=0,solver_seconds=2.,backend='gurobi',cache=None):
    """All-pair complete-AC acceptance with retained public/parent feasible plans."""
    began=perf_counter();deadline=began+float(total_seconds);_groups(context);_offsets(context,history_losses)
    if total_seconds<=0 or solver_seconds<=0 or not initials:raise ValueError('positive budgets and initial plans required')
    if not grids or any(k not in (3,5,9,17) for k in grids):raise ValueError('supported grids are 3/5/9/17')
    engine=gradient.ProfileEngine(context,fit_options=fit_options,seed=seed,cache=cache)
    record=dict(success=False,initial_candidates=[],grids=[],failures=[],history_losses=history_losses,
        aggregation='mean_reference_nearest_with_known_history_offsets',
        configuration=dict(grids=list(grids),solver_seconds=solver_seconds,total_seconds=total_seconds,backend=backend),
        total_seconds=float(total_seconds),solver_seconds=float(solver_seconds),all_candidates_retained=True,
        directed_pair_count=len(context['pairs']),candidate_names=[c.name for c in context['bank']],
        uses_truth=False,uses_validation=False,uses_future_noise=False,line_parameters_fixed=True,
        one_trained_parameter_vector=True,random_action_proposals=False,
        bound_scope='raw frozen affine proposals; selected AC plan may be an inherited plan',
        full_AC_global_gap=None,history_scope='known attained training half-SSE offsets, not exact expected LLR',
        budget_scope='soft deadline; node-QP/AC-fit checks; complete action-QP certificate batch is atomic')
    checked=[];seen=set();fallback=None;operation_calls=0
    for origin,power in initials.items():
        if power is None:continue
        u=np.asarray(power,float)
        if u.shape!=context['control_shape'] or not np.isfinite(u).all():
            record['initial_candidates'].append(dict(origin=origin,success=False,failure='invalid initial power'));continue
        key=fixed._control_key(u)
        if key in seen:continue
        seen.add(key)
        if perf_counter()>=deadline:
            record['initial_candidates'].append(dict(origin=origin,attempted=False,failure='initial plan budget exhausted'));continue
        realized=_realize(domain,u,seconds=min(solver_seconds,max(.001,deadline-perf_counter())),backend=backend)
        operation_calls+=1
        row=dict(origin=origin,operation=realized['audit'],success=False)
        if realized['success']:
            if fallback is None:fallback=dict(origin=origin,power=u.copy(),operation=realized['audit'])
            profile=engine.evaluate(u,'ac',deadline=deadline);value=_AC_value(context,profile,history_losses)
            row.update(profile=profile,score=None if value is None else value['score'],aggregation=value)
            if value is not None:
                row['success']=True;checked.append(dict(origin=origin,power=u.copy(),x=realized['x'],
                    score=value['score'],profile=profile,aggregation=value))
        record['initial_candidates'].append(row)
    if not checked:
        record.update(failure='no feasible complete-AC initial score',seconds=perf_counter()-began,
            counters=engine.counters(),fallback_power=None if fallback is None else fallback['power'],
            fallback_operation=None if fallback is None else fallback['operation'],selected_score=None)
        return None,record
    best=max(checked,key=lambda row:row['score']);record['initial_best_score']=best['score']
    record['initialization_seconds']=perf_counter()-began
    table_cache={}
    for points in grids:
        if perf_counter()>=deadline:
            record['failures'].append('budget before grid '+str(points));break
        grid_began=perf_counter();row=dict(grid_points=points,before_score=best['score'],before_origin=best['origin'],accepted=False)
        table=build_systematic_tables(context,domain,grid_points=points,cache=table_cache,deadline=deadline)
        row['table']=table
        if not table['success']:
            row['failure']=table['failure'];row['seconds']=perf_counter()-grid_began;record['grids'].append(row);break
        remaining=deadline-perf_counter()
        if remaining<=0:
            row['failure']='budget before MILP';row['seconds']=perf_counter()-grid_began;record['grids'].append(row);break
        try:
            solved=solve_systematic_milp(context,domain,tables=table,history_losses=history_losses,
                initial_operation=best['x'],solver_seconds=min(solver_seconds,remaining),backend=backend)
            row['raw']=solved
            if solved['success'] and perf_counter()<deadline:
                prediction=engine.evaluate(solved['power_kw'],'ac',deadline=deadline)
                value=_AC_value(context,prediction,history_losses)
                row.update(raw_AC_profile=prediction,raw_AC_score=None if value is None else value['score'],raw_aggregation=value)
                tolerance=1e-9+1e-10*abs(best['score'])
                if value is not None and value['score']>best['score']+tolerance:
                    row['accepted']=True;best=dict(origin='scalar_MILP_'+str(points),power=solved['power_kw'].copy(),
                        x=solved['operation_x'],score=value['score'],profile=prediction,aggregation=value)
                else:row['rejection_reason']='required source fits failed or complete AC score did not improve'
            elif solved['success']:row['failure']='budget before complete AC acceptance'
            else:row['failure']='no complete feasible raw MILP proposal'
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            row['failure']=type(exc).__name__+':'+str(exc)
        row.update(selected_score=best['score'],selected_origin=best['origin'],seconds=perf_counter()-grid_began)
        record['grids'].append(row)
    # An operation-feasible MILP can still lack an action-QP certificate.
    # Explicit None is a saved failure state, not a mapping with zero fields.
    table_logs=[row.get('table') or {} for row in record['grids']]
    raw_logs=[row.get('raw') or {} for row in record['grids']]
    action_logs=[row.get('action_QP') or {} for row in raw_logs]
    record.update(success=True,selected_score=best['score'],selected_origin=best['origin'],
        selected_profile=best['profile'],selected_aggregation=best['aggregation'],selected_operation=domain.check(best['x']),
        selected_operation_x=best['x'],selected_is_new_MILP_proposal=best['origin'].startswith('scalar_MILP_'),
        inheritance_protected=best['score']>=record['initial_best_score']-1e-10,
        seconds=perf_counter()-began,counters=engine.counters(),
        table_QP_calls=sum(row.get('QP_calls',0) for row in table_logs),
        table_cache_hits=sum(row.get('cache_hits',0) for row in table_logs))
    record['action_QP_calls']=sum(len(row.get('pairs') or []) for row in action_logs)
    record['total_frozen_QP_calls']=record['table_QP_calls']+record['action_QP_calls']
    record['operation_realization_calls']=operation_calls
    record['primary_MILP_calls']=sum('raw' in row for row in record['grids'])
    record['primary_MILP_feasible']=sum(row.get('success') is True for row in raw_logs)
    record['primary_MILP_seconds']=sum(row.get('MILP_seconds') or 0. for row in raw_logs)
    record['table_seconds']=sum(row.get('seconds') or 0. for row in table_logs)
    record['all_requested_grids_processed']=len(record['grids'])==len(grids) and all(
        row.get('success') is True for row in raw_logs)
    record['grid_failure_reasons']=[row['failure'] for row in record['grids'] if row.get('failure')]
    record['time_budget_exceeded']=record['seconds']>total_seconds
    return best['power'].copy(),record
