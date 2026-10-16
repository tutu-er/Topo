"""A shared-line quadratic MILP proposal, accepted by full AC joint profiles.

The reference for every candidate is its single training line vector. This
module accepts no simulated future or true topology. All K(K-1) comparisons
are retained, including failed comparisons. Each domain has an incremental
soft budget; fixed-parent dependencies are billed separately by the driver.
"""
from time import perf_counter
import numpy as np
from . import ev_fixed_train_design as fixed
from . import ev_mobile_v2_design as full
from . import ev_mobile_v3_ac as ac
from . import ev_mobile_v3_projection as proxy
from . import ev_systematic_design as scalar
from .ev_mobile_v3_domain import tighten_operation_bounds


def projected_policy(context,domain,initials,*,seed=0,joint_options=None,
                     total_seconds=60.,solver_seconds=3.,segments=4,max_steps=1,
                     engine=None):
    began=perf_counter(); deadline=began+total_seconds
    if not initials:raise ValueError('feasible parent required')
    engine=engine or full.JointPredictiveEngine(context,fit_options=joint_options,seed=seed)
    record=dict(success=False,initials=[],steps=[],failures=[],selected_origin=None,
        configuration=dict(total_seconds=total_seconds,solver_seconds=solver_seconds,
            segments=segments,max_steps=max_steps),
        reference_lines_fixed=True,one_trained_parameter_vector=True,
        competitors_shared_line_tangent=True,AC_acceptance_shared_lines=True,
        all_candidates_retained=True,uses_truth=False,uses_validation=False,
        uses_future_noise=False,random_action_proposals=False,
        bound_scope='each frozen quadratic only; no global AC bound',
        budget_scope='incremental domain soft wall budget, parents billed separately')
    before=engine.counters(); best=None;seen=set()
    # Fixed parent is deliberately first; a deadline cannot replace it with
    # another family's unscored fallback. Supplied parents are never discarded
    # merely because a proposed new plan scores poorly.
    items=list(initials.items())
    items.sort(key=lambda kv:0 if kv[0]=='fixed_parent' else 1)
    fallback=None
    for origin,power in items:
        u=np.asarray(power,float);key=fixed._control_key(u)
        if key in seen:continue
        seen.add(key)
        realized=scalar._realize(domain,u,seconds=2.)
        row=dict(origin=origin,success=False,operation=realized['audit'])
        record['initials'].append(row)
        if not realized['success']:
            row['failure']='parent operation infeasible';continue
        if fallback is None:fallback=(u.copy(),realized['x'],origin)
        if perf_counter()>=deadline:
            row['failure']='budget before parent full AC';break
        result=engine.evaluate(u,deadline=deadline)
        row.update(profile=result,score=result.get('score'),success=bool(result['success']))
        if result['success'] and (best is None or result['score']>best['score']+1e-9):
            best=dict(power=u.copy(),x=realized['x'],origin=origin,
                      score=float(result['score']),profile=result)
    if best is None:
        record.update(failure='no complete full AC parent profile',seconds=perf_counter()-began,
            fallback_origin=None if fallback is None else fallback[2])
        if fallback is None:return None,record
        record['selected_operation_x']=fallback[1];return fallback[0],record
    record['parent_score']=best['score']; record['parent_origin']=best['origin']
    for step in range(max_steps):
        if perf_counter()>=deadline:record['failures'].append('budget before proposal');break
        row=dict(step=step,accepted=False,before_score=best['score'],before_origin=best['origin'])
        record['steps'].append(row)
        try:
            pairs,geometry=ac.projected_context(context,best['power'])
            row['geometry']=geometry
            left=deadline-perf_counter()
            if left<=0:row['failure']='budget after geometry';break
            operation=tighten_operation_bounds(domain)
            row['reachability']=operation['graph_reachability']
            solved=proxy.solve_projected_milp(pairs,operation,
                initial_operations=[best['x']],segments=segments,
                solver_seconds=min(solver_seconds,left),total_seconds=left)
            row['master']=solved
            proposed=solved.get('power_kw')
            if proposed is None:row['failure']='no MILP feasible incumbent';continue
            if np.allclose(proposed,best['power'],rtol=0,atol=1e-8):
                row['reason']='parent retained by frozen quadratic';continue
            if perf_counter()>=deadline:row['failure']='budget before proposal full AC';continue
            result=engine.evaluate(proposed,deadline=deadline);row['AC_profile']=result
            row['AC_score']=result.get('score')
            if result['success'] and result['score']>best['score']+1e-9+1e-10*abs(best['score']):
                row['accepted']=True
                best=dict(power=np.asarray(proposed).copy(),x=solved['operation_x'],
                    origin='shared_projection_MILP_'+str(step),score=float(result['score']),profile=result)
            else:row['reason']='incomplete full AC profile or no AC improvement'
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            row['failure']=type(exc).__name__+': '+str(exc)
    after=engine.counters()
    record.update(success=True,selected_origin=best['origin'],selected_score=best['score'],
        selected_profile=best['profile'],selected_operation_x=best['x'],
        operation=domain.check(best['x']),power_kw=best['power'],
        parent_preserved=best['score']>=record['parent_score']-1e-8,
        seconds=perf_counter()-began,deadline_reached=perf_counter()>=deadline,
        profile_evaluations=after['profile_evaluations']-before['profile_evaluations'],
        joint_fit_calls=after['joint_fit_calls']-before['joint_fit_calls'])
    return best['power'],record
