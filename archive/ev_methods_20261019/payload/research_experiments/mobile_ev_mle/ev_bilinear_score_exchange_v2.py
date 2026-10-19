"""Numerically scaled witness exchange; v1 sources/results are preserved.

Initial witnesses are QP explanations at an actual feasible control, not
midpoints of broad latent boxes. Residual ranges use valid control-LP dual
bounds. A conflicting numerical MILP upper is rejected and the previous
upper retained with an unfinished status. Bounds concern the declared
bilinear proxy, never full nonlinear AC or a future statistical benefit.
"""
from __future__ import annotations
from collections.abc import Mapping
from time import perf_counter
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, issparse, vstack
from .ev_bilinear_support_milp import BilinearPair,evaluate_qp,fixed_witness_upper
from .ev_bilinear_score_exchange import witness_score_quadratic
from .ev_profile_chord_milp import _constraints,check_control,residual_intervals,chord_value


def _lp_form(lower,upper,constraints):
    n=len(lower); groups=_constraints(constraints,n)
    if groups:
        A=vstack([coo_matrix(c.A) for c in groups],format='csr')
        lo=np.concatenate([c.lb for c in groups]);hi=np.concatenate([c.ub for c in groups])
    else:
        A=coo_matrix((0,n)).tocsr();lo=hi=np.array([])
    eq=np.isfinite(lo)&np.isfinite(hi)&(lo==hi)
    up=np.isfinite(hi)&~eq;down=np.isfinite(lo)&~eq
    return dict(A_ub=vstack([A[up],-A[down]],format='csr'),b_ub=np.r_[hi[up],-lo[down]],
        A_eq=A[eq],b_eq=hi[eq],bounds=list(zip(lower,upper)),physical=groups)


def _linear_dual_lower(c,low,high,form,*,time_limit=2.):
    """Global linear lower from arbitrary dual values plus box correction.

    Clipping inequality marginals to <=0 enforces the dual sign; minimising
    the remaining stationarity residual over the whole box makes the bound
    valid without assuming exact dual stationarity.
    """
    c=np.asarray(c,float)
    result=linprog(c,A_ub=form['A_ub'],b_ub=form['b_ub'],A_eq=form['A_eq'],b_eq=form['b_eq'],
        bounds=form['bounds'],method='highs',options=dict(time_limit=float(time_limit)))
    box=float(np.minimum(c*low,c*high).sum())
    if result.status!=0 or result.x is None:
        return box,dict(success=False,status=int(result.status),bound=box,method='box_fallback')
    yu=np.minimum(np.asarray(result.ineqlin.marginals,float),0.)
    ye=np.asarray(result.eqlin.marginals,float)
    remaining=c-form['A_ub'].T@yu-form['A_eq'].T@ye
    products=np.r_[form['b_ub']*yu,form['b_eq']*ye,np.minimum(remaining*low,remaining*high)]
    bound=float(np.sum(products.astype(np.longdouble)))
    guard=256*np.finfo(float).eps*(1+float(np.sum(abs(products))))
    bound=max(box,bound-guard)
    return bound,dict(success=True,status=int(result.status),bound=bound,primal=float(result.fun),
        dual_box_correction=float(np.minimum(remaining*low,remaining*high).sum()),numeric_guard=guard,
        method='signed_dual_with_stationarity_box_correction')


def tight_residual_intervals(q,low,high,form):
    a,b,variable=residual_intervals(q,low,high);audits=[]
    for row in np.flatnonzero(variable):
        dl,left=_linear_dual_lower(q.T[row],low,high,form)
        dr,right=_linear_dual_lower(-q.T[row],low,high,form)
        pad=128*np.finfo(float).eps*(1+abs(q.d[row])+max(abs(dl),abs(dr)))
        a[row]=max(a[row],float(q.d[row]+dl-pad))
        b[row]=min(b[row],float(q.d[row]-dr+pad))
        if a[row]>b[row]:
            # Never use an inward inconsistent enclosure.
            aa,bb,_=residual_intervals(q,low,high);a[row],b[row]=aa[row],bb[row]
        audits.append(dict(row=int(row),lower_lp=left,upper_lp=right,interval=[a[row],b[row]]))
    return a,b,variable,audits


def solve_tight_chord_master(quadratics,low,high,constraints,*,attained_lower,
        segments=4,time_limit=2.,mip_rel_gap=1e-8,feasible_incumbents=(),feasibility_tolerance=1e-7):
    """Independent v2 master with tight ranges and attained-score scaling."""
    began=perf_counter();qs=tuple(quadratics);low=np.asarray(low,float);high=np.asarray(high,float)
    form=_lp_form(low,high,constraints);intervals=[tight_residual_intervals(q,low,high,form) for q in qs]
    upper_scores=[]
    for q,(a,b,_,_) in zip(qs,intervals):
        value=np.sum(np.maximum(a*a,b*b).astype(np.longdouble))-np.longdouble(q.offset)
        upper_scores.append(float(value))
    coarse_upper=min(upper_scores)
    # An attained lower is feasible for the witness master. Using it as
    # gamma's lower bound does not discard any possible master optimum.
    score_scale=max(1e-12,abs(coarse_upper),abs(attained_lower),abs(coarse_upper-attained_lower))
    if coarse_upper<attained_lower:
        return dict(u=None,objective_lower=attained_lower,objective_upper=coarse_upper,
            status=4,message='residual enclosure upper below attained lower',bound_consistent=False,
            scale=score_scale,seconds=perf_counter()-began,interval_audits=intervals)
    guard=256*np.finfo(float).eps*(1+abs(coarse_upper)+abs(attained_lower))
    n=len(low);gamma=n
    lbs=list(low)+[(attained_lower-guard)/score_scale]
    ubs=list(high)+[(coarse_upper+guard)/score_scale];integer=[0]*(n+1)
    rows=[];los=[];his=[];knots_by_q=[];errors=[]
    def row(values,lo=-np.inf,hi=np.inf):
        rows.append({int(k):float(v) for k,v in values.items() if v!=0});los.append(float(lo));his.append(float(hi))
    def variables(count,binary=False):
        start=len(lbs);lbs.extend([0.]*count);ubs.extend([1.]*count);integer.extend([int(binary)]*count)
        return np.arange(start,start+count)
    for constraint in form['physical']:
        A=constraint.A.tocsr() if issparse(constraint.A) else constraint.A
        for j in range(A.shape[0]):
            if issparse(A):r=A.getrow(j);values=dict(zip(r.indices,r.data))
            else:values=dict(enumerate(A[j]))
            row(values,constraint.lb[j],constraint.ub[j])
    for q,(a,b,var,_) in zip(qs,intervals):
        score={gamma:1.};constant=np.longdouble(-q.offset);knots_group=[];err=0.
        for j,(left,right,changes) in enumerate(zip(a,b,var)):
            if not changes:
                constant+=np.longdouble(left)**2;knots_group.append(np.array([left]));continue
            nodes=np.linspace(left,right,int(segments)+1);knots_group.append(nodes)
            if np.any(np.diff(nodes)<=0):raise ValueError('residual range cannot support requested segments')
            choose,fraction=variables(int(segments),True),variables(int(segments))
            row(dict(zip(choose,np.ones(int(segments)))),1,1)
            residual=dict(enumerate(q.T[j]))
            for k,(x0,x1) in enumerate(zip(nodes[:-1],nodes[1:])):
                row({fraction[k]:1.,choose[k]:-1.},hi=0.)
                residual[choose[k]]=-x0;residual[fraction[k]]=-(x1-x0)
                score[choose[k]]=-x0*x0/score_scale
                score[fraction[k]]=-(x1*x1-x0*x0)/score_scale
            row(residual,-q.d[j],-q.d[j]);err+=float(np.max(np.diff(nodes))**2/4)
        row(score,hi=float((constant+guard)/score_scale))
        knots_by_q.append(knots_group);errors.append(err)
    rr=[];cc=[];vv=[]
    for i,r in enumerate(rows):
        for j,v in r.items():rr.append(i);cc.append(j);vv.append(v)
    matrix=coo_matrix((vv,(rr,cc)),shape=(len(rows),len(lbs))).tocsc()
    objective=np.zeros(len(lbs));objective[gamma]=-1.
    result=milp(objective,integrality=np.asarray(integer),bounds=Bounds(lbs,ubs),
        constraints=LinearConstraint(matrix,los,his),options=dict(time_limit=float(time_limit),
        mip_rel_gap=float(mip_rel_gap),presolve=True))
    candidates=[]
    for u,origin in [(u,'retained_feasible_incumbent') for u in feasible_incumbents]+([] if result.x is None else [(result.x[:n],'milp_incumbent')]):
        audit=check_control(u,low,high,form['physical'],tolerance=feasibility_tolerance)
        if audit['feasible']:
            candidates.append(dict(u=np.asarray(u,float).copy(),objective=min(q.value(u) for q in qs),
                physical_validation=audit,origin=origin))
    best=max(candidates,key=lambda r:r['objective']) if candidates else None
    dual=getattr(result,'mip_dual_bound',None)
    if dual is None and result.status==0 and not any(integer):dual=result.fun
    solver_upper=None if dual is None or not np.isfinite(dual) else float(-dual*score_scale+guard)
    candidate_upper=coarse_upper+guard if solver_upper is None else min(coarse_upper+guard,solver_upper)
    # Unlike v1, a negative numerical gap is never silently tolerated.
    consistent=best is None or candidate_upper>=best['objective']
    return dict(u=None if best is None else best['u'],objective_lower=None if best is None else best['objective'],
        objective_upper=float(candidate_upper),coarse_upper=float(coarse_upper+guard),solver_best_bound=solver_upper,
        bound_consistent=bool(consistent),scale=float(score_scale),numeric_guard=float(guard),
        status=int(result.status),message=str(result.message),solver_success=bool(result.success),
        solver_mip_gap=getattr(result,'mip_gap',None),solver_nodes=getattr(result,'mip_node_count',None),
        physical_validation=None if best is None else best['physical_validation'],
        selected_origin=None if best is None else best['origin'],
        chord_incumbent=None if result.fun is None else float(-result.fun*score_scale),
        n_variables=len(lbs),n_binary=int(sum(integer)),n_rows=len(rows),segments=int(segments),
        secant_error_max=float(max(errors)),secant_error_by_quadratic=errors,
        interval_audits=[dict(name=q.name,residual_lower=a,residual_upper=b,lp=audit)
                        for q,(a,b,var,audit) in zip(qs,intervals)],seconds=perf_counter()-began)


def exchange_bilinear_scores_v2(pairs,control_lower,control_upper,constraints=None,*,offsets=None,
        score_scale=.5,initial_u=None,max_iterations=8,segments=4,max_segments=32,
        solver_seconds=5.,total_seconds=60.,absolute_gap=1e-5,relative_gap=1e-3,
        mip_rel_gap=1e-8,feasibility_tolerance=1e-7):
    began=perf_counter();pairs=tuple(pairs);low=np.asarray(control_lower,float);high=np.asarray(control_upper,float)
    if (not pairs or low.ndim!=1 or not len(low) or high.shape!=low.shape or not np.isfinite([low,high]).all()
            or np.any(low>high) or any(not isinstance(p,BilinearPair) or p.control_dimension!=len(low) for p in pairs)
            or len({p.name for p in pairs})!=len(pairs)):
        raise ValueError('unique compatible pairs and finite ordered control box required')
    offsets={p.name:0. for p in pairs} if offsets is None else offsets
    if not isinstance(offsets,Mapping) or set(offsets)!={p.name for p in pairs}:
        raise ValueError('exactly one offset for every required pair required')
    off=np.asarray([offsets[p.name] for p in pairs],float)
    if not np.isfinite(off).all() or not np.isfinite(score_scale) or score_scale<=0:raise ValueError('finite offsets and positive score scale required')
    for key,value in [('max_iterations',max_iterations),('segments',segments),('max_segments',max_segments)]:
        if isinstance(value,bool) or int(value)!=value or value<1:raise ValueError(key+' must be positive integer')
    if max_segments<segments:raise ValueError('max_segments must be at least segments')
    if any(not np.isfinite(x) or x<=0 for x in (solver_seconds,total_seconds)):raise ValueError('positive budgets required')
    if any(not np.isfinite(x) or x<0 for x in (absolute_gap,relative_gap,mip_rel_gap,feasibility_tolerance)):raise ValueError('nonnegative tolerances required')
    physical=_constraints(constraints,len(low));form=_lp_form(low,high,physical)
    if initial_u is None:
        feasibility=linprog(np.zeros(len(low)),A_ub=form['A_ub'],b_ub=form['b_ub'],A_eq=form['A_eq'],b_eq=form['b_eq'],bounds=form['bounds'],method='highs')
        if not feasibility.success:
            return dict(u=None,status='initial_operations_infeasible',completed=False,unfinished=True,
                objective_lower=None,objective_upper=None,raw_gap=None,gap=None,iterations=[],evaluations=[],qp_calls=0,
                failures=[dict(kind='initial_operations_infeasible',message=str(feasibility.message))],seconds=perf_counter()-began)
        initial_u=feasibility.x
    witnesses=[[] for p in pairs];evaluations=[];iterations=[];failures=[];qp_calls=0;best=None;lower=None;upper=None
    def evaluate(u,origin):
        nonlocal qp_calls,best,lower
        audit=check_control(u,low,high,physical,tolerance=feasibility_tolerance)
        if not audit['feasible']:raise ValueError('control violates operations')
        u=np.asarray(u,float).copy();sol=[];added=[]
        for k,p in enumerate(pairs):
            qp_calls+=1;s=evaluate_qp(p,u);sol.append(s)
            novel=not any(np.array_equal(s['z'],z) for z in witnesses[k]);added.append(novel)
            if novel:witnesses[k].append(s['z'].copy())
        slo=score_scale*(np.asarray([s['q_tangent_lower'] for s in sol])-off)
        shi=score_scale*(np.asarray([s['q_primal_upper'] for s in sol])-off)
        e=dict(u=u,origin=origin,score_lower=float(min(slo)),score_upper=float(min(shi)),
            pair_score_lower=slo,pair_score_upper=shi,qp_results=sol,added_witness=added,
            physical_validation=audit,all_qp_converged=all(s['success'] for s in sol))
        evaluations.append(e)
        if lower is None or e['score_lower']>lower:best=e;lower=e['score_lower']
        if not e['all_qp_converged']:failures.append(dict(kind='inner_qp_not_converged',origin=origin))
        return e
    initial=evaluate(initial_u,'initial_feasible_control')
    upper=min(score_scale*(fixed_witness_upper(p,low,high,zs[0])**2-o) for p,zs,o in zip(pairs,witnesses,off))
    initial_upper=float(upper)
    midpoint_upper=min(score_scale*(fixed_witness_upper(p,low,high)**2-o) for p,o in zip(pairs,off))
    current_segments=int(segments)
    def target():return float(absolute_gap+relative_gap*max(1e-12,abs(upper),abs(lower)))
    def converged():return upper>=lower and upper-lower<=target()
    status='gap_reached_initial_witness_bound' if converged() else 'iteration_limit'
    if upper<lower:
        # This would invalidate even the independently obtained box bound.
        status='initial_numerical_bound_inconsistency';failures.append(dict(kind=status,lower=lower,upper=upper))
    elif not converged():
        for it in range(int(max_iterations)):
            remaining=total_seconds-(perf_counter()-began)
            if remaining<=0:status='time_budget';break
            qs=[];compressions=[]
            for p,zs,o in zip(pairs,witnesses,off):
                for j,z in enumerate(zs):
                    q,a=witness_score_quadratic(p,z,o,score_scale,low,high,name=p.name+'__w'+str(j))
                    qs.append(q);compressions.append(dict(pair=p.name,witness=j,**a))
            previous_upper=upper
            master=solve_tight_chord_master(qs,low,high,physical,attained_lower=lower,
                segments=current_segments,time_limit=min(solver_seconds,remaining),mip_rel_gap=mip_rel_gap,
                feasible_incumbents=[best['u']],feasibility_tolerance=feasibility_tolerance)
            record=dict(iteration=it,master=master,compressions=compressions,previous_upper=previous_upper,
                selected_u=master['u'],segments=current_segments,upper_update_accepted=False,
                global_upper=upper,global_lower=lower,witnesses_before=[len(zs) for zs in witnesses])
            iterations.append(record)
            if master['u'] is None:
                status='master_without_valid_incumbent';failures.append(dict(kind=status,message=master['message']));break
            repeat=any(np.allclose(master['u'],e['u'],rtol=0.,atol=1e-9) for e in evaluations)
            entry=evaluate(master['u'],'master_'+str(it))
            proposed_upper=min(previous_upper,float(master['objective_upper']))
            # Validate against both the witness master's retained control and
            # every true inner-QP lower already attained, with no gap clipping.
            trustworthy=bool(master['bound_consistent'] and np.isfinite(proposed_upper) and proposed_upper>=lower)
            if not trustworthy:
                upper=previous_upper
                record.update(global_upper=upper,global_lower=lower,rejected_upper=proposed_upper,
                    witnesses_after=[len(zs) for zs in witnesses],gap=upper-lower)
                failures.append(dict(kind='numerical_master_upper_rejected',iteration=it,
                    candidate_upper=proposed_upper,safe_previous_upper=previous_upper,attained_lower=lower))
                status='numerical_master_upper_rejected';break
            upper=proposed_upper
            record.update(upper_update_accepted=True,global_upper=upper,global_lower=lower,
                witnesses_after=[len(zs) for zs in witnesses],gap=upper-lower,target_gap=target(),
                repeated_control=repeat,refinement_next=False)
            if converged():status='gap_reached';break
            if (repeat or not any(entry['added_witness'])) and current_segments<max_segments:
                current_segments=min(2*current_segments,int(max_segments));record['refinement_next']=True
            elif repeat and not any(entry['added_witness']) and current_segments>=max_segments:
                status='repeated_control_with_gap';break
    gap=upper-lower
    completed=status in ('gap_reached','gap_reached_initial_witness_bound')
    return dict(u=best['u'],objective_lower=float(lower),objective_upper=float(upper),raw_gap=float(gap),
        gap=float(gap),target_gap=target(),status=status,completed=completed,unfinished=not completed,
        initial_fixed_witness_upper=initial_upper,midpoint_fixed_witness_upper=float(midpoint_upper),
        initialization='all required QP explanations at a feasible control; no midpoint witnesses',
        iterations=iterations,evaluations=evaluations,witnesses=witnesses,witness_counts=[len(zs) for zs in witnesses],
        qp_calls=qp_calls,inner_qp_failures=sum(not e['all_qp_converged'] for e in evaluations),failures=failures,
        best_evaluation_index=next(i for i,e in enumerate(evaluations) if e is best),
        offsets=dict(zip([p.name for p in pairs],off.tolist())),score_scale=float(score_scale),
        required_pairs=[p.name for p in pairs],final_segments=current_segments,seconds=perf_counter()-began,
        full_curve_enumeration=False,self_profile_subtracted=False,fixed_historical_offsets=True,
        original_global_optimum_certified=False,
        scope='numerical signed score bounds for declared bilinear proxy; common boxes and linear operations; no full AC certificate')
