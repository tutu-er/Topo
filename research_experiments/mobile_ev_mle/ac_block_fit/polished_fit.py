"""Optional, budgeted stationarity polish after the unchanged normal fitter.

The original fitter is never monkeypatched. The polish uses exact full AC,
the original box/cone map, bounded QR steps, a frozen entry Jacobian scaling,
and the common original-scale projected gradient. Its loss tolerance is an
explicit FP64 numerical acceptance rule, not a certified nonlinear upper bound.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import inspect
from pathlib import Path
from time import perf_counter
import numpy as np

from .. import likelihood as lk
from ..ev_rx_fit import _RxInputProblem
from .blocks import problem_blocks, gradient, column_norms, projected_gradient, matvec
from .linear import solve_block_step
from .nonlinear import fit_rx_blocks as _normal_fit


_NORMAL_SIGNATURE = inspect.signature(_normal_fit)


def _source_hashes():
    paths = [Path(__file__), Path(__file__).with_name('blocks.py'),
             Path(__file__).with_name('linear.py'), Path(__file__).with_name('nonlinear.py')]
    return {path.name: sha256(path.read_bytes()).hexdigest() for path in paths}


def _physical_check(problem, vector):
    """Audit actual source/optimizer bounds, R/X domain, AC, and voltage."""
    base = problem.base
    r, x, source = problem.unpack(vector)
    physical = base.cache.get('physical', {})
    voltage = np.abs(np.asarray(physical.get('voltage', [np.nan])))
    finite = bool(np.isfinite(np.r_[r, x, source.ravel()]).all() and np.isfinite(voltage).all())
    box = bool(np.all(vector >= problem.lower-1e-12) and np.all(vector <= problem.upper+1e-12))
    domain = bool(problem.mapping.contains(r, x))
    vlow, vhigh = base.voltage_bounds
    voltage_ok = bool(finite and voltage.min() >= vlow and voltage.max() <= vhigh)
    equation = float(physical.get('ac_equation_residual', np.inf))
    balance = float(physical.get('power_balance_residual', np.inf))
    return dict(passed=bool(finite and box and domain and voltage_ok and equation<=2e-10 and balance<=2e-10),
        finite=finite, optimizer_and_source_box_feasible=box, RX_domain_feasible=domain,
        voltage_feasible=voltage_ok, voltage_bounds=[float(vlow),float(vhigh)],
        voltage_range=[float(voltage.min()),float(voltage.max())],
        ac_equation_residual=equation,power_balance_residual=balance)


def polish_stationarity(problem, vector, *, target=1e-5, max_steps=3,
                        remaining_nfev=3, damping=1e-4, inner_tol=1e-10,
                        inner_maxiter=2000, block_function=problem_blocks,
                        step_function=solve_block_step):
    """At most three full-AC trials, strictly inside the caller's remainder.

    Entry AC/gradient replay is an independent audit, not a trial. Each trial,
    including a failed AC solve, consumes one residual-evaluation allowance.
    The complete loss band is anchored once to entry F0; it never accumulates.
    Rejected caches are restored without an extra physical solve.
    """
    started=perf_counter()
    for value, name in ((max_steps,'max_steps'),(remaining_nfev,'remaining_nfev')):
        if isinstance(value,bool) or int(value)!=value or value<0:
            raise ValueError(name+' must be a nonnegative integer')
    if not np.isfinite([target,damping,inner_tol]).all() or min(target,damping,inner_tol)<=0:
        raise ValueError('positive finite tolerances and damping required')
    if max_steps>3:
        raise ValueError('this frozen polish rule allows at most three steps')
    current=np.asarray(vector,float).copy()
    if current.shape!=problem.lower.shape or not np.isfinite(current).all():
        raise ValueError('finite optimizer vector matching bounds required')
    initial_calls=problem.base.ac_calls
    residual=np.asarray(problem.evaluate(current),float).copy()
    line,source=block_function(problem,current)
    grad=gradient(line,source,residual)
    initial_pg=projected_gradient(current,grad,problem.lower,problem.upper,problem.scale)
    initial_loss=.5*float(residual@residual)
    loss_tolerance=256*np.finfo(float).eps*max(1.,initial_loss)
    loss_limit=initial_loss+loss_tolerance
    norms=column_norms(line,source)
    scale=1./np.where(norms>0,norms,1.)
    initial_physical=_physical_check(problem,current)
    entry_audit_ac_calls=problem.base.ac_calls-initial_calls
    current_pg=initial_pg; current_loss=initial_loss
    trial_nfev=0; accepted=0; trace=[]
    status='initial_physical_failure' if not initial_physical['passed'] else 'step_budget_exhausted'
    if initial_physical['passed'] and initial_pg<=target:
        status='already_common_stationary'
    elif initial_physical['passed']:
        for index in range(min(int(max_steps),int(remaining_nfev))):
            d=line.shape[-1]
            scaled_line=line*scale[:d][None,None,:]
            scaled_source=source*scale[d:].reshape(problem.t,1,2)
            item=dict(step=index,entry_F=initial_loss,previous_F=current_loss,
                previous_common_pg=current_pg,damping=damping,
                initial_anchored_loss_limit=loss_limit,loss_tolerance=loss_tolerance,
                accepted=False)
            try:
                step=step_function(residual.reshape(source.shape[:2]),scaled_line,scaled_source,
                    (problem.lower-current)/scale,(problem.upper-current)/scale,
                    damping=damping,kkt_tol=inner_tol,max_iter=inner_maxiter)
            except (RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
                item.update(reason='bounded_QR_failure',error=type(error).__name__+': '+str(error))
                trace.append(item);status='inner_failure';break
            item.update(inner_success=bool(step['success']),inner_kkt=float(step['kkt']),
                        inner_error_bound=step.get('error_bound'),inner_iterations=int(step['iterations']))
            if not step['success']:
                item['reason']='bounded_QR_did_not_meet_KKT';trace.append(item)
                status='inner_failure';break
            proposed=current+scale*np.asarray(step['step'])
            bound_tolerance=128*np.finfo(float).eps*np.maximum(1.,np.abs(current))
            if (not np.isfinite(proposed).all() or np.any(proposed<problem.lower-bound_tolerance)
                    or np.any(proposed>problem.upper+bound_tolerance)):
                item['reason']='bounded_QR_returned_infeasible_step';trace.append(item)
                status='inner_failure';break
            trial=np.minimum(np.maximum(proposed,problem.lower),problem.upper)
            linear_delta=matvec(line,source,trial-current).ravel()
            predicted=-float(residual@linear_delta)-.5*float(linear_delta@linear_delta)
            item['predicted_linear_model_decrease']=predicted
            if not np.isfinite(predicted) or predicted<=0:
                item['reason']='nonpositive_linear_model_decrease';trace.append(item)
                status='non_descent_step';break
            saved_cache=deepcopy(problem.base.cache)
            calls_before_trial=problem.base.ac_calls
            trial_nfev+=1
            try:
                trial_residual=np.asarray(problem.evaluate(trial),float).copy()
                if not np.isfinite(trial_residual).all():
                    raise FloatingPointError('nonfinite full-AC residual')
                trial_loss=.5*float(trial_residual@trial_residual)
                physical=_physical_check(problem,trial)
                item.update(trial_F=trial_loss,actual_F_change=trial_loss-current_loss,
                    F_change_from_entry=trial_loss-initial_loss,physical=physical,
                    stable_actual_decrease=.5*float((residual-trial_residual)@(residual+trial_residual)))
                if not physical['passed']:
                    item['reason']='trial_physical_failure'
                elif trial_loss>loss_limit:
                    item['reason']='loss_above_initial_numerical_band'
                else:
                    trial_line,trial_source=block_function(problem,trial)
                    trial_grad=gradient(trial_line,trial_source,trial_residual)
                    trial_pg=projected_gradient(trial,trial_grad,problem.lower,problem.upper,problem.scale)
                    item['trial_common_pg']=trial_pg
                    if not np.isfinite(trial_pg) or not trial_pg<current_pg:
                        item['reason']='common_projected_gradient_not_strictly_lower'
                    else:
                        item.update(accepted=True,reason='loss_band_and_common_gradient',
                                    accepted_positive_F_change=bool(trial_loss>current_loss))
                        current,residual,line,source=trial,trial_residual,trial_line,trial_source
                        current_loss,current_pg=trial_loss,trial_pg
                        accepted+=1
            except (RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
                item.update(reason='trial_AC_or_derivative_failure',error=type(error).__name__+': '+str(error))
            item['actual_AC_calls']=problem.base.ac_calls-calls_before_trial
            trace.append(item)
            if not item['accepted']:
                problem.base.cache=saved_cache
                status=item['reason'];break
            if current_pg<=target:
                status='common_stationarity_reached';break
        else:
            status='no_remaining_budget' if remaining_nfev==0 else 'step_budget_exhausted'
    final_physical=_physical_check(problem,current)
    target_pass=bool(final_physical['passed'] and current_pg<=target)
    return dict(x=current,fun=residual,success=target_pass,status=status,
        initial_F=initial_loss,final_F=current_loss,actual_F_change=current_loss-initial_loss,
        initial_common_pg=initial_pg,final_common_pg=current_pg,common_pg_target=target,
        common_stationarity_pass=target_pass,improved=bool(current_pg<initial_pg),
        accepted_steps=accepted,trial_nfev=trial_nfev,max_steps=int(max_steps),
        remaining_nfev_at_entry=int(remaining_nfev),entry_audit_ac_calls=entry_audit_ac_calls,
        actual_trial_AC_calls=sum(row.get('actual_AC_calls',0) for row in trace),
        actual_AC_calls=problem.base.ac_calls-initial_calls,
        physical=final_physical,initial_physical=initial_physical,trace=trace,
        frozen_entry_jacobian_scale=scale,original_common_scale=problem.scale.copy(),
        loss_tolerance=loss_tolerance,loss_limit=loss_limit,
        accepted_positive_F_changes=[row['actual_F_change'] for row in trace
                                     if row.get('accepted_positive_F_change')],
        acceptance_scope='FP64 numerical loss band and original-scale coordinate projected gradient; not a nonlinear upper-bound proof',
        loss_band_accumulates=False,strict_inner_KKT_required=True,
        seconds=perf_counter()-started,source_sha256=_source_hashes())


def _independent_replay(problem, vector):
    """Fresh CPU full AC; recompute residual, likelihood, state and gradient."""
    began=perf_counter();base=problem.base
    r,x,source=problem.unpack(vector)
    base.ac_calls+=1
    if base.counter is not None:
        base.counter['ac_calls']=base.counter.get('ac_calls',0)+1
    physical=lk.ac_predict(base.candidate,r,x,base.p,base.q,source[:,0],source[:,1],base.observations.root_v)
    residual=np.column_stack(((physical['v']+base.bias['v']-base.observations.v)/base.sigmas['v'],
        (physical['master_p']-base.observations.master_p)/base.sigmas['master_p'],
        (physical['master_q']-base.observations.master_q)/base.sigmas['master_q'])).ravel()
    physical_vector=problem.physical_vector(vector)
    base.cache=dict(vector=physical_vector.copy(),residual=residual.copy(),physical=physical)
    line,local=problem_blocks(problem,vector)
    grad=gradient(line,local,residual)
    pg=projected_gradient(vector,grad,problem.lower,problem.upper,problem.scale)
    audit=_physical_check(problem,vector)
    loss=float(residual@residual)
    return dict(residual=residual,physical=physical,r=r,x=x,source=source,
        loss=loss,loglik=-.5*(loss+base.normalizer),common_pg=pg,physical_audit=audit,
        seconds=perf_counter()-began,independent_full_AC_calls=1)


def fit_rx_blocks_polished(candidate,observations,model,*,polish_target=1e-5,
                           polish_max_steps=3,normal_fitter=None,**options):
    """Strict opt-in wrapper; preserve normal failure and the original default.

    ``success`` requires normal success, physical replay and the common
    stationarity target. Relative-loss success remains separately recorded.
    A max-budget failure receives no additional optimization trials.
    """
    began=perf_counter()
    bound=_NORMAL_SIGNATURE.bind(candidate,observations,model,**options);bound.apply_defaults()
    supplied=dict(bound.arguments)
    if supplied['fixed_r'] is None and supplied['fixed_x'] is None and supplied['multistart']!=1:
        raise ValueError('polished free-line wrapper requires one declared start for a shared residual budget')
    counter=supplied['call_counter']
    counter_at_entry={} if counter is None else deepcopy(counter)
    fitter=_normal_fit if normal_fitter is None else normal_fitter
    clock=perf_counter();normal=fitter(candidate,observations,model,**options)
    normal_seconds=perf_counter()-clock
    diagnostics=deepcopy(normal.diagnostics)
    normal_record=dict(success=bool(normal.success),status=int(normal.status),nfev=int(normal.nfev),
        loss=float(normal.loss),loglik=float(normal.loglik),optimality=float(normal.optimality),
        reference_scaled_projected_gradient=diagnostics.get('reference_scaled_projected_gradient'),
        stationarity_pass=diagnostics.get('stationarity_pass'),
        stationarity_tolerance=diagnostics.get('stationarity_tolerance'),
        termination_definition=diagnostics.get('termination_definition'))
    diagnostics['normal_fit']=normal_record
    diagnostics['normal_success']=bool(normal.success)
    diagnostics['normal_stationarity_pass']=diagnostics.get('stationarity_pass')
    diagnostics['normal_stationarity_tolerance']=diagnostics.get('stationarity_tolerance')
    diagnostics['normal_termination_definition']=diagnostics.get('termination_definition')
    normal_AC_calls=int(normal.diagnostics.get('ac_evaluations',
        0 if counter is None else counter.get('ac_calls',0)-counter_at_entry.get('ac_calls',0)))
    problem=None

    def account(trials=0):
        """Common cost record, including exceptional initialization/replay."""
        local_calls=0 if problem is None else int(problem.base.ac_calls)
        delta={} if counter is None else {key:counter.get(key,0)-counter_at_entry.get(key,0)
            for key in set(counter)|set(counter_at_entry)
            if isinstance(counter.get(key,0),(int,float)) and isinstance(counter_at_entry.get(key,0),(int,float))}
        total_calls=int(delta.get('ac_calls',normal_AC_calls+local_calls))
        # An external counter also observes a constructor/initialization error
        # before the new problem object can be assigned to this wrapper.
        local_calls=max(local_calls,total_calls-normal_AC_calls)
        elapsed=perf_counter()-began
        diagnostics.update(total_nfev=int(normal.nfev)+int(trials),
            ac_evaluations=total_calls,ac_time_slice_evaluations=total_calls*observations.t,
            normal_AC_calls=normal_AC_calls,polish_and_replay_actual_AC_calls=local_calls,
            full_wrapper_AC_calls=total_calls,counter_delta=delta,
            counter_snapshot={} if counter is None else deepcopy(counter),
            external_counter_supplied=counter is not None,
            normal_seconds=normal_seconds,polished_wrapper_seconds=elapsed,seconds=elapsed)
        return dict(optimization_nfev_total=int(normal.nfev)+int(trials),
            normal_AC_calls=normal_AC_calls,polish_and_replay_actual_AC_calls=local_calls,
            full_wrapper_AC_calls=total_calls,counter_delta=delta,
            normal_seconds=normal_seconds,wrapper_seconds=elapsed)

    def common_criterion(passed,verified):
        diagnostics.update(stationarity_pass=bool(passed),common_stationarity_pass=bool(passed),
            stationarity_tolerance=float(polish_target),common_stationarity_verified=bool(verified),
            termination_definition='optional wrapper success requires normal success, independent physical replay and common original-scale PG target',
            stationarity_coordinates='original problem.scale in R_alpha optimizer box; not physical R_X cone KKT at collapsed corners')

    budget=int(supplied['max_nfev']);remaining=max(0,budget-int(normal.nfev))
    if supplied['fixed_r'] is not None or supplied['fixed_x'] is not None:
        diagnostics['polish']=dict(status='fixed_lines_original_method_retained',attempted=False)
        diagnostics['polish'].update(account())
        return replace(normal,diagnostics=diagnostics)
    if not normal.success:
        reason='normal_budget_failure' if normal.status==0 or normal.nfev>=budget else 'normal_failure_retained'
        diagnostics['polish']=dict(status=reason,attempted=False,trial_nfev=0,
                                   optimization_nfev_total=int(normal.nfev),budget=budget)
        diagnostics['polish'].update(account())
        common_criterion(False,False)
        diagnostics['failure_kind']=diagnostics.get('failure_kind') or 'computation'
        return replace(normal,diagnostics=diagnostics)
    domain={key:supplied[key] for key in ('r_bounds','x_bounds','ratio_bounds',
        'source_p_bounds','source_q_bounds','voltage_bounds')}
    initial=dict(r=normal.r,x=normal.x,source_p=normal.source_p,source_q=normal.source_q)
    try:
        problem=_RxInputProblem(candidate,observations,model,initial=initial,
            call_counter=supplied['call_counter'],**domain)
        r,alpha=problem.mapping.inverse(normal.r,normal.x)
        full=np.r_[r,alpha,np.c_[normal.source_p,normal.source_q].ravel()]
        vector=full[problem.free]
        rebuilt_r,rebuilt_x,_=problem.unpack(vector)
        reconstruction=float(max(np.max(abs(rebuilt_r-normal.r),initial=0.),
                                 np.max(abs(rebuilt_x-normal.x),initial=0.)))
        if reconstruction>1e-12:
            raise ValueError('normal R/X state reconstruction differs by '+str(reconstruction))
        outcome=polish_stationarity(problem,vector,target=polish_target,max_steps=polish_max_steps,
            remaining_nfev=remaining,damping=1e-4,inner_tol=supplied['inner_tol'],
            inner_maxiter=supplied['inner_maxiter'])
    except (RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
        diagnostics['polish']=dict(status='polish_initialization_or_derivative_failure',attempted=False,
            error=type(error).__name__+': '+str(error),optimization_nfev_total=int(normal.nfev),
            source_sha256=_source_hashes())
        diagnostics['polish'].update(account())
        common_criterion(False,False)
        diagnostics['failure_kind']='computation'
        return replace(normal,success=False,status=-8,
            message='stationarity polish initialization/derivative failure: '+str(error),diagnostics=diagnostics)
    try:
        replay=_independent_replay(problem,outcome['x'])
        replay_agrees=bool(abs(replay['loss']-2*outcome['final_F'])
                           <=512*np.finfo(float).eps*max(1.,abs(replay['loss'])))
        physical_pass=bool(replay['physical_audit']['passed'] and replay_agrees)
        stationarity=bool(replay['common_pg']<=polish_target)
        success=bool(normal.success and physical_pass and stationarity)
        kept={key:value for key,value in outcome.items() if key not in ('x','fun')}
        kept.update(attempted=True,normal_seconds=normal_seconds,
            optimization_nfev_total=int(normal.nfev)+outcome['trial_nfev'],budget=budget,
            independent_replay={key:value for key,value in replay.items()
                                if key in ('loss','loglik','common_pg','physical_audit','seconds','independent_full_AC_calls')},
            independent_loss_matches=bool(replay_agrees),RX_reconstruction_max_abs=reconstruction)
        kept.update(account(outcome['trial_nfev']))
        common_criterion(stationarity,True)
        diagnostics.update(polish=kept,normal_success=bool(normal.success),
            common_stationarity_pass=stationarity,reference_scaled_projected_gradient=replay['common_pg'],
            physical_voltage_feasible=replay['physical_audit']['voltage_feasible'],
            ratio_domain_feasible=replay['physical_audit']['RX_domain_feasible'],
            total_nfev=int(normal.nfev)+outcome['trial_nfev'],
            polished_wrapper_seconds=perf_counter()-began,
            solver='normal_block_QR_then_budgeted_common_stationarity_polish',
            success_definition='normal success AND independent physical replay AND common original-scale PG target',
            global_optimality_certified=False,
            failure_kind=None if success else ('physical_domain' if not replay['physical_audit']['passed'] else 'computation'))
        physical=replay['physical'];base=problem.base
        predicted=dict(physical,p=base.p+base.bias['p'],q=base.q+base.bias['q'],
                       v=physical['v']+base.bias['v'],physical_v=physical['v'])
        message='common stationarity target passed' if success else 'retained finite state; common stationarity or physical audit did not pass'
        return replace(normal,r=replay['r'].copy(),x=replay['x'].copy(),
            source_p=replay['source'][:,0].copy(),source_q=replay['source'][:,1].copy(),
            loss=replay['loss'],loglik=replay['loglik'],success=success,
            status=1 if success else -7,message=message,
            nfev=int(normal.nfev)+outcome['trial_nfev'],optimality=replay['common_pg'],
            predicted=predicted,diagnostics=diagnostics)
    except (RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
        diagnostics['polish']=dict(status='independent_replay_failure',attempted=True,
            error=type(error).__name__+': '+str(error),trace=outcome['trace'],
            trial_nfev=outcome['trial_nfev'],optimization_nfev_total=int(normal.nfev)+outcome['trial_nfev'],
            source_sha256=_source_hashes())
        diagnostics['polish'].update(account(outcome['trial_nfev']))
        common_criterion(False,False)
        diagnostics['failure_kind']='computation'
        return replace(normal,success=False,status=-8,nfev=int(normal.nfev)+outcome['trial_nfev'],
            message='independent full-AC replay failure: '+str(error),diagnostics=diagnostics)


__all__=['polish_stationarity','fit_rx_blocks_polished']
