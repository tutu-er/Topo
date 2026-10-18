"""Bounded damped Gauss--Newton with exact AC and block-QR linear steps.

This is a development solver for the same loss and box/cone domain, not a
copy of SciPy TRF's reflected trust-region path. Line and source bounds enter
the convex local step; actual full AC decides acceptance.
"""
from time import perf_counter
import numpy as np

from .. import likelihood as lk
from ..ev_fixed_train_fit import _compact_bias
from ..ev_rx_fit import _RxInputProblem
from ..reference_benchmark.fitting import history_linear_initial
from .blocks import problem_blocks, gradient, matvec, column_norms, projected_gradient
from .linear import solve_block_step


def minimize_block_lm(problem, *, max_nfev=250, tol=1e-8, inner_tol=1e-10,
                      inner_maxiter=2000, scale_mode='reference'):
    if scale_mode not in ('reference','jac'):
        raise ValueError('scale_mode must be reference or jac')
    if max_nfev < 1 or tol <= np.finfo(float).eps or inner_tol <= 0:
        raise ValueError('invalid numerical budget or precision')
    x = problem.start.copy(); trace = []; timings = dict(ac=0.,derivatives=0.,linear=0.)
    nfev = 0; status = 0; message = 'maximum AC evaluation budget reached'
    clock = perf_counter(); residual = problem.evaluate(x)
    timings['ac'] += perf_counter()-clock; nfev += 1
    loss = .5*float(residual@residual); damping = None; njev = 0
    scale = problem.scale.copy(); norm_cache = None
    while nfev < max_nfev:
        clock = perf_counter(); line, source = problem_blocks(problem,x)
        timings['derivatives'] += perf_counter()-clock; njev += 1
        if scale_mode == 'jac':
            norms = column_norms(line,source)
            norm_cache = norms if norm_cache is None else np.maximum(norm_cache,norms)
            # Identically zero columns retain their declared characteristic scale.
            scale = np.divide(1.,norm_cache,out=problem.scale.copy(),where=norm_cache>0)
        grad = gradient(line,source,residual)
        opt = projected_gradient(x,grad,problem.lower,problem.upper,scale)
        if opt <= tol:
            status = 1; message = 'scaled projected gradient tolerance reached'; break
        d = line.shape[-1]
        scaled_line = line*scale[:d][None,None,:]
        scaled_source = source*scale[d:].reshape(problem.t,1,2)
        if damping is None:
            damping = max(1e-8,1e-3*float(np.max(column_norms(scaled_line,scaled_source)**2,initial=0.)))
        accepted = False
        for attempt in range(20):
            clock = perf_counter()
            step = solve_block_step(residual.reshape(source.shape[:2]),scaled_line,scaled_source,
                (problem.lower-x)/scale,(problem.upper-x)/scale,
                damping=damping,kkt_tol=inner_tol,max_iter=inner_maxiter)
            timings['linear'] += perf_counter()-clock
            info = dict(nfev=nfev,loss=2*loss,damping=float(damping),
                inner_success=bool(step['success']),inner_kkt=float(step['kkt']),
                inner_iterations=int(step['iterations']),accepted=False)
            if not step['success']:
                trace.append(info); status=-4; message='bounded linear subproblem did not meet KKT tolerance'; break
            eta = np.asarray(step['step']); delta=scale*eta
            predicted_residual = residual+matvec(scaled_line,scaled_source,eta).ravel()
            prediction = loss-.5*float(predicted_residual@predicted_residual)
            if prediction <= 0 or not np.isfinite(prediction):
                trace.append(info); status=-3; message='no positive model decrease at current precision'; break
            # The BVLS step already satisfies bounds; this removes ulp drift only.
            trial = np.minimum(np.maximum(x+delta,problem.lower),problem.upper)
            clock = perf_counter()
            try:
                trial_residual = problem.evaluate(trial)
                trial_loss = .5*float(trial_residual@trial_residual)
                error = None
            except (RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                trial_loss = np.inf; error=repr(exc)
            timings['ac'] += perf_counter()-clock; nfev += 1
            actual=loss-trial_loss; gain=actual/prediction
            info.update(predicted_decrease=float(prediction),actual_decrease=float(actual),
                gain=float(gain),ac_error=error)
            if actual > 0 and gain >= 1e-4:
                old_loss=loss; x=trial; residual=trial_residual; loss=trial_loss
                info['accepted']=True; accepted=True
                if gain>.75:damping=max(1e-18,damping*.3)
                elif gain<.25:damping*=2
                if actual <= tol*max(old_loss,np.finfo(float).tiny) and gain>.25:
                    status=2; message='relative full-AC objective tolerance reached'
                elif np.linalg.norm(delta) <= tol*(tol+np.linalg.norm(x)):
                    status=3; message='parameter-step tolerance reached'
            else:
                damping=min(1e30,damping*10.)
            trace.append(info)
            if accepted or status or nfev>=max_nfev:break
        if status or nfev>=max_nfev:break
        if not accepted:
            status=-5; message='damping retry budget exhausted';break
    # Restore the retained accepted state; failed trial states never leak out.
    before_calls=problem.base.ac_calls
    clock=perf_counter();residual=problem.evaluate(x)
    timings['ac']+=perf_counter()-clock
    final_replay_calls=problem.base.ac_calls-before_calls
    clock=perf_counter();line,source=problem_blocks(problem,x)
    timings['derivatives']+=perf_counter()-clock
    njev+=1
    grad=gradient(line,source,residual)
    opt=projected_gradient(x,grad,problem.lower,problem.upper,scale)
    return dict(x=x,fun=residual,success=status>0,status=status,message=message,
        nfev=nfev,njev=njev,optimality=opt,reference_scaled_projected_gradient=
        projected_gradient(x,grad,problem.lower,problem.upper,problem.scale),
        gradient_infinity=float(np.max(abs(grad),initial=0.)),
        stationarity_pass=bool(opt<=tol),stationarity_tolerance=tol,
        final_replay_ac_calls=final_replay_calls,
        budget_definition='max_nfev counts optimization residual trials; initialization and final audit separately recorded',
        trace=trace,timings=timings,scale=scale)


def fit_rx_blocks(candidate,observations,model,*,initial=None,fixed_r=None,fixed_x=None,
                  r_bounds=(1e-6,.1),x_bounds=(1e-6,.1),ratio_bounds=(.083/.642,.080/.208),
                  source_p_bounds=(-.5,.5),source_q_bounds=(-.5,.5),
                  voltage_bounds=(.85,1.15),multistart=1,max_nfev=250,tol=1e-8,
                  seed=0,call_counter=None,inner_tol=1e-10,inner_maxiter=2000,
                  scale_mode='reference'):
    if fixed_r is not None or fixed_x is not None:
        from .fixed_source import fit_fixed_sources
        return fit_fixed_sources(candidate,observations,model,initial=initial,fixed_r=fixed_r,
            fixed_x=fixed_x,r_bounds=r_bounds,x_bounds=x_bounds,ratio_bounds=ratio_bounds,
            source_p_bounds=source_p_bounds,source_q_bounds=source_q_bounds,
            voltage_bounds=voltage_bounds,multistart=multistart,max_nfev=max_nfev,tol=tol,
            seed=seed,call_counter=call_counter)
    if multistart!=1:
        raise ValueError('development block solver currently requires one declared start')
    began=perf_counter(); init_record=dict(method='caller_supplied_arrived_history_start')
    domain=dict(r_bounds=r_bounds,x_bounds=x_bounds,ratio_bounds=ratio_bounds,
                source_p_bounds=source_p_bounds,source_q_bounds=source_q_bounds)
    if initial is None:
        initial,init_record=history_linear_initial(candidate,observations,model,**domain)
    problem=_RxInputProblem(candidate,observations,model,initial=initial,
        voltage_bounds=voltage_bounds,call_counter=call_counter,**domain)
    if call_counter is not None:call_counter['fit_calls']=call_counter.get('fit_calls',0)+1
    trials=[]; requested=problem.start.copy()
    for power in range(13):
        full=problem.expand(requested)
        full[:problem.m]=problem.mapping.lower+(.5**power)*(full[:problem.m]-problem.mapping.lower)
        problem.start=np.clip(full[problem.free],problem.lower+problem.margin,problem.upper-problem.margin)
        try:
            problem.evaluate(problem.start);trials.append(dict(line_factor=.5**power,success=True));break
        except (RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            trials.append(dict(line_factor=.5**power,success=False,error=repr(exc)))
    if not trials[-1]['success']:
        raise RuntimeError('all declared initial AC feasibility trials failed: '+str(trials))
    result=minimize_block_lm(problem,max_nfev=max_nfev,tol=tol,inner_tol=inner_tol,
        inner_maxiter=inner_maxiter,scale_mode=scale_mode)
    x=result['x']; residual=result['fun'];loss=float(residual@residual)
    problem.evaluate(x);base=problem.base;phy=base.cache['physical']
    r,xx,source=problem.unpack(x); voltage=abs(phy['voltage'])
    voltage_ok=bool(np.min(voltage)>=voltage_bounds[0] and np.max(voltage)<=voltage_bounds[1])
    domain_ok=problem.mapping.contains(r,xx)
    valid=bool(result['success'] and voltage_ok and domain_ok and np.isfinite(loss))
    diagnostics=dict(objective='conditional_known_diagonal_Gaussian_output_given_measured_PQ',
        solver='bounded_damped_Gauss_Newton_BVLS_block_QR',fixed_lines=False,
        initialization=init_record,initial_ac_trials=trials,truth_used=False,future_used=False,
        physical_voltage_feasible=voltage_ok,ratio_domain_feasible=domain_ok,
        source_spatial_location_shared=True,source_components=['P','Q'],source_node=1,
        root_voltage_known=True,P_Q_inputs_fixed=True,P_Q_measurement_error_propagated=False,
        input_EV_already_in_total_PQ=True,global_optimality_certified=False,
        shared_optimizer_parameters=int(problem.width-2*problem.t),
        source_parameters=2*problem.t,residual_count=problem.t*(problem.n+2),
        output_normalizer=base.normalizer,total_nfev=result['nfev'],
        ac_evaluations=base.ac_calls,ac_time_slice_evaluations=base.ac_calls*problem.t,
        full_global_Jacobian_formed=False,scale_mode=scale_mode,
        reference_scaled_projected_gradient=result['reference_scaled_projected_gradient'],
        stationarity_pass=result['stationarity_pass'],stationarity_tolerance=tol,
        stationarity_coordinates='R_alpha_optimizer_box; not a physical R_X_cone_KKT_certificate at collapsed corners',
        termination_definition='solver success may use ftol/xtol, as in reference; strict stationarity separately recorded',
        residual_budget_definition=result['budget_definition'],
        final_replay_ac_calls=result['final_replay_ac_calls'],
        gradient_infinity=result['gradient_infinity'],inner_tol=inner_tol,
        inner_iterations=sum(item['inner_iterations'] for item in result['trace']),
        trace=result['trace'],timings=result['timings'],
        ac_equation_residual=float(phy['ac_equation_residual']),
        power_balance_residual=float(phy['power_balance_residual']),
        min_voltage=float(np.min(voltage)),max_voltage=float(np.max(voltage)),
        seconds=perf_counter()-began,
        failure_kind=None if valid else ('physical_domain' if not voltage_ok or not domain_ok else 'computation'))
    biases=[_compact_bias(base.bias[key]) for key in ('p','q','v')]
    predicted=dict(phy,p=base.p+base.bias['p'],q=base.q+base.bias['q'],
                   v=phy['v']+base.bias['v'],physical_v=phy['v'])
    return lk.FitResult(candidate.name,r.copy(),xx.copy(),base.p.copy(),base.q.copy(),
        source[:,0].copy(),source[:,1].copy(),*biases,loss,-.5*(loss+base.normalizer),
        valid,int(result['status']),str(result['message']),int(result['nfev']),
        float(result['optimality']),predicted,diagnostics)
