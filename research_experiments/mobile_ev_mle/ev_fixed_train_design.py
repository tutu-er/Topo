"""Training-only EV MILP with fixed per-topology lines and source-only rivals.

P/Q readings are calibrated inputs, not latent fitted meter channels. Each
candidate's R/X is frozen at its training fit. Validation observations are
absent from this module's API. The nonlinear AC source fit and the frozen
linear source-QP are recorded separately; proxy bounds concern the latter.
"""
from __future__ import annotations

from hashlib import sha256
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from . import likelihood as lk
from .ev_fixed_train_fit import fit_fixed_inputs
from .ev_mobile_service_domain import MobileServiceConfig, build_mobile_service_domain
from .ev_original_domain_milp import (OriginalDomainPair, evaluate_original_pair,
                                      exchange_original_domain)
from .symmetric_ac_local import mean_and_jacobian

CHANNELS = ('v', 'master_p', 'master_q')
DOMAIN_FLAGS = dict(charge_fixed=(False, False), v2g_fixed=(False, True),
                    charge_mobile=(True, False), v2g_mobile=(True, True))
INHERITS = dict(charge_fixed=(), v2g_fixed=('charge_fixed',),
    charge_mobile=('charge_fixed',),
    v2g_mobile=('charge_fixed', 'v2g_fixed', 'charge_mobile'))
METHODS = ('none', 'constant', 'quality_fixed', 'quality_mobile', 'random_mobile',
    *DOMAIN_FLAGS, 'refined_v2g_mobile', 'capped_v2g_mobile')
CONFIG = dict(version='fixed_training_lines_source_profile_v1', base_kw=20.,
    horizon=6, command_power_kw=6./1.05, task_cap=20.,
    initial_energy_kwh=24., terminal_energy_kwh=26.85, capacity_kwh=60.,
    dt_hours=.25, eta_charge=.95, eta_discharge=.95, throughput_kwh=7.5,
    drive_kwh=.2, travel_slots=1, setup_slots=1, max_moves=1,
    travel_worker_minutes=15., setup_worker_minutes=15., driver_budget_minutes=30.,
    min_arrival_throughput_kwh=.25, proxy_channel_RMS_gate=.25,
    forecast='calibrated training PQ mean and candidate training source mean',
    forecast_scenarios=1, validation_used=False, random_role='independent comparator only')
DEFAULT_EXCHANGE = dict(max_iterations=6, control_segments=4, residual_segments=4,
    max_segments=16, solver_seconds=2., total_seconds=35., absolute_gap=.01,
    relative_gap=.01, witness_policy='affine_preferred', backend='gurobi',
    master_form='shared_control_square')


def _broadcast(value, shape, label):
    try:
        out = np.broadcast_to(np.asarray(value, float), shape).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(label+' must broadcast to '+str(shape)) from exc
    if not np.isfinite(out).all():
        raise ValueError(label+' must be finite')
    return out


def _line_id(r, x):
    digest = sha256()
    for value in (r, x):
        a = np.ascontiguousarray(value, dtype='<f8')
        digest.update(repr(a.shape).encode()); digest.update(a.tobytes())
    return digest.hexdigest()


def _source_vector(p, q):
    return np.column_stack((p, q)).ravel()


def _source_parts(value):
    value = np.asarray(value, float).reshape(-1, 2)
    return value[:, 0].copy(), value[:, 1].copy()


def _geometry(candidate, line, background, root, meter, source):
    sp, sq = _source_parts(source)
    return mean_and_jacobian(candidate, line['r'], line['x'], background['p'],
        background['q'], sp, sq, root, meter, channels=CHANNELS)


def _channel_values(residual, T, n):
    residual = np.asarray(residual, float)
    if residual.shape != (T*(n+2),):
        raise ValueError('V/root residual dimension does not match horizon')
    blocks = dict(v=residual[:T*n], master_p=residual[T*n:T*(n+1)],
                  master_q=residual[T*(n+1):])
    return {name:dict(SSE=float(v@v), RMS=float(np.sqrt(np.mean(v*v))),
        max_abs=float(np.max(abs(v), initial=0.))) for name,v in blocks.items()}


def build_fixed_contexts(training, bank, train_fits, train_meter, future_meter,
                         lower_kw, upper_kw, reference_kw, *, base_kw=20.,
                         source_p_bounds=(-.5,.5), source_q_bounds=(-.5,.5)):
    """Freeze all lines; build every directed source-profiled affine pair.

    Controls flatten in time-major order. Source coordinates interleave
    (source_P[t], source_Q[t]). The pair residual is competitor minus reference,
    whitened in channel order V, root-P, root-Q. Only training enters forecasts.
    At=0 and shift=0 are declared first-order approximations, not AC identities.
    """
    began=perf_counter(); bank=tuple(bank); reference=np.asarray(reference_kw,float)
    if (reference.ndim!=2 or reference.shape[1]!=training.n or not len(reference)
            or not np.isfinite(reference).all() or not np.isfinite(base_kw) or base_kw<=0):
        raise ValueError('finite time-by-terminal reference and positive base_kw required')
    T,n=reference.shape; K=T*n; width=2*n+2
    def bound(value):
        a=np.asarray(value,float)
        return a.reshape(K).copy() if a.size==K else _broadcast(a,reference.shape,'control bound').ravel()
    low,high=bound(lower_kw),bound(upper_kw)
    if (not np.isfinite([low,high]).all() or np.any(low>high)
            or np.any(reference.ravel()<low) or np.any(reference.ravel()>high)):
        raise ValueError('ordered control box must contain reference')
    if len(bank)<2 or len({c.name for c in bank})!=len(bank) or any(c.n!=n for c in bank):
        raise ValueError('at least two unique compatible candidates required')
    diagnostic=dict(uses_validation=False,uses_truth=False,uses_future_noise=False,
        forecast_scenarios=1,forecast_uses_training_only=True,source_interface=1,
        line_parameters_fixed=True,PQ_direct_inputs=True,PQ_in_residual=False,
        EV_in_total_PQ_once=True,control_shape=[T,n],channels=list(CHANNELS),
        source_baseline='each candidate own training mean',ac_calls=0,failure=None)
    answer=dict(success=False,pairs=[],models=[],forecasts={},diagnostics=diagnostic)
    try:
        if train_meter.fit_bias or future_meter.fit_bias:
            raise ValueError('independently calibrated fixed biases required')
        if training.master_p is None or training.master_q is None:
            raise ValueError('training root P/Q anchors required')
        if any(c.name not in train_fits or not train_fits[c.name].success for c in bank):
            raise ValueError('every required training fit must succeed')
        source_low=np.tile([float(source_p_bounds[0]),float(source_q_bounds[0])],T)
        source_high=np.tile([float(source_p_bounds[1]),float(source_q_bounds[1])],T)
        if not np.isfinite([source_low,source_high]).all() or np.any(source_low>=source_high):
            raise ValueError('finite ordered source boxes required')
        pmean=np.mean(training.p-_broadcast(train_meter.bias_p,training.p.shape,'training P bias'),axis=0)
        qmean=np.mean(training.q-_broadcast(train_meter.bias_q,training.q.shape,'training Q bias'),axis=0)
        background=dict(p=np.tile(pmean,(T,1)),q=np.tile(qmean,(T,1)))
        total=dict(p=background['p']+reference/base_kw,q=background['q'].copy())
        root=np.full(T,float(np.mean(training.root_v)))
        geometries={};lines={};sources={}
        control_columns=np.array([t*width+j for t in range(T) for j in range(n)])
        source_columns=np.array([t*width+2*n+j for t in range(T) for j in range(2)])
        for candidate in bank:
            fit=train_fits[candidate.name]
            r=np.asarray(fit.r,float).copy();x=np.asarray(fit.x,float).copy()
            if r.shape!=(candidate.m,) or x.shape!=r.shape or not np.isfinite([r,x]).all():
                raise ValueError('training line dimensions invalid: '+candidate.name)
            r.setflags(write=False);x.setflags(write=False)
            lines[candidate.name]=dict(r=r,x=x,sha256=_line_id(r,x))
            source=_source_vector(np.full(T,np.mean(fit.source_p)),np.full(T,np.mean(fit.source_q)))
            if np.any(source<source_low) or np.any(source>source_high):
                raise ValueError('training mean source outside declared physical box: '+candidate.name)
            sources[candidate.name]=source
            diagnostic['ac_calls']+=1
            geometry=_geometry(candidate,lines[candidate.name],total,root,future_meter,source)
            geometries[candidate.name]=geometry
            sp,sq=_source_parts(source)
            answer['forecasts'][candidate.name]=dict(p=total['p'].copy(),q=total['q'].copy(),
                source_p=sp,source_q=sq,root_v=root.copy(),line_sha256=lines[candidate.name]['sha256'])
        models=[];pairs=[]
        for a in bank:
            for b in bank:
                if a.name==b.name:continue
                ga,gb=geometries[a.name],geometries[b.name]
                B=(gb['state_jacobian'][:,control_columns]-ga['state_jacobian'][:,control_columns])/base_kw
                G=gb['state_jacobian'][:,source_columns]
                c=gb['mean']-ga['mean']-B@reference.ravel()
                name=a.name+'__to__'+b.name
                pair=OriginalDomainPair(name,c,B,G,np.zeros((K,len(c),2*T)),
                    source_low,source_high,sources[b.name],np.zeros((2*T,K)))
                pairs.append(pair)
                models.append(dict(name=name,a=a.name,b=b.name,task_weight=1/(len(bank)*(len(bank)-1)),
                    source_center=sources[b.name].copy(),reference_source=sources[a.name].copy(),
                    reference_residual=gb['mean']-ga['mean'],reference_kw=reference.ravel().copy(),
                    competitor_line_sha256=lines[b.name]['sha256'],reference_line_sha256=lines[a.name]['sha256'],
                    historical_training_losses=dict(reference=float(train_fits[a.name].loss),
                        competitor=float(train_fits[b.name].loss)),offset=0.,
                    bound_scope='frozen first-order future V/root model, not nonlinear AC',
                    source_control_cross_terms_retained=False,line_control_cross_terms_absent=True))
        answer.update(success=True,pairs=pairs,models=models,bank=bank,train_fits=train_fits,
            future_meter=future_meter,lines=lines,background=background,root_v=root,
            source_p_bounds=tuple(map(float,source_p_bounds)),source_q_bounds=tuple(map(float,source_q_bounds)),
            source_lower=source_low,source_upper=source_high,control_lower=low,control_upper=high,
            reference_kw=reference.copy(),base_kw=float(base_kw),control_shape=(T,n))
        diagnostic.update(pairs=len(pairs),residual_rows=T*(n+2),source_coordinates=2*T,
            R_X_coordinates_in_competitor_QP=0,At_exactly_zero=True,shift_exactly_zero=True,
            training_line_sha256={k:v['sha256'] for k,v in lines.items()},
            source_p_bounds=list(source_p_bounds),source_q_bounds=list(source_q_bounds))
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
        answer['pairs']=[];answer['models']=[]
        diagnostic['failure']=type(exc).__name__+':'+str(exc)
    diagnostic['seconds']=perf_counter()-began
    return answer


def check_fixed_witness(context, power_kw, source_by_pair=None, *, gate=.25,
                        voltage_bounds=(.85,1.15)):
    """Replay all proxy-optimal source explanations through fixed-line AC."""
    if not context.get('success'):raise ValueError('successful fixed context required')
    u=np.asarray(power_kw,float).reshape(context['control_shape']);T,n=u.shape
    total=dict(p=context['background']['p']+u/context['base_kw'],q=context['background']['q'])
    bank={c.name:c for c in context['bank']};checks=[];qp_results=[];calls=0
    for model,pair in zip(context['models'],context['pairs']):
        try:
            qp=evaluate_original_pair(pair,u.ravel());qp_results.append(qp)
            xi=np.asarray(qp['physical_parameter'] if source_by_pair is None else source_by_pair[model['name']],float)
            a,b=model['a'],model['b']
            line_sha_ok=bool(_line_id(context['lines'][a]['r'],context['lines'][a]['x'])==model['reference_line_sha256']
                and _line_id(context['lines'][b]['r'],context['lines'][b]['x'])==model['competitor_line_sha256'])
            if not line_sha_ok:raise ValueError('context line differs from fixed training snapshot')
            calls+=1
            ga=_geometry(bank[a],context['lines'][a],total,context['root_v'],context['future_meter'],model['reference_source'])
            calls+=1
            gb=_geometry(bank[b],context['lines'][b],total,context['root_v'],context['future_meter'],xi)
            exact=gb['mean']-ga['mean'];approx=pair.residual(u.ravel(),xi)
            channel=_channel_values(exact-approx,T,n)
            maxrms=max(r['RMS'] for r in channel.values())
            voltage_ok=all(np.min(g['physical']['v'])>=voltage_bounds[0]
                and np.max(g['physical']['v'])<=voltage_bounds[1] for g in (ga,gb))
            box_ok=bool(np.all(xi>=pair.lower-1e-10) and np.all(xi<=pair.upper+1e-10))
            checks.append(dict(name=model['name'],success=True,qp_converged=qp['success'],
                proxy_source=xi,proxy_residual=approx,complete_AC_residual=exact,
                proxy_SSE=float(approx@approx),complete_AC_witness_SSE=float(exact@exact),
                channel_error=channel,max_channel_RMS=maxrms,voltage_feasible=bool(voltage_ok),
                source_box_feasible=box_ok,gate_passed=bool(qp['success'] and voltage_ok and box_ok and maxrms<=gate),
                line_parameters_fixed=True,line_SHA_matches=line_sha_ok))
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            checks.append(dict(name=model['name'],success=False,gate_passed=False,
                failure=type(exc).__name__+':'+str(exc)))
    return dict(success=all(r['success'] for r in checks),gate_passed=all(r['gate_passed'] for r in checks),
        pairs=checks,qp_results=qp_results,gate=float(gate),ac_calls=calls,
        scope='fixed-source witness AC replay; QP bounds remain surrogate-only')


def evaluate_fixed_action(context, power_kw, *, fit_options=None, seed=0):
    """Noiseless training-predictive AC distance with rival sources refitted.

    Training R/X and reference source forecasts stay fixed. Source estimates
    of the rival may vary by time at the same physical interface. The returned
    nonlinear distances are attained local fits, never surrogate lower bounds.
    """
    began=perf_counter()
    if not context.get('success'):raise ValueError('successful fixed context required')
    u=np.asarray(power_kw,float)
    if u.shape!=context['control_shape'] or not np.isfinite(u).all():
        raise ValueError('finite matching time-by-terminal power required')
    T,n=u.shape;meter=context['future_meter'];options=dict(multistart=2,max_nfev=250,tol=1e-10)
    options.update(fit_options or {})
    for key in ('fixed_r','fixed_x','initial','seed','call_counter'):
        if key in options:raise ValueError('design owns '+key)
    for key in ('source_p_bounds','source_q_bounds'):
        if key in options and tuple(options[key])!=tuple(context[key]):
            raise ValueError('AC and proxy source domains must match')
        options[key]=context[key]
    totalp=context['background']['p']+u/context['base_kw'];totalq=context['background']['q'].copy()
    before={name:_line_id(v['r'],v['x']) for name,v in context['lines'].items()}
    if any(before[k]!=v['sha256'] for k,v in context['lines'].items()):
        raise ValueError('context line differs from fixed training snapshot')
    physical={};failures=[];counter=dict(fit_calls=0,solver_calls=0,ac_calls=0)
    bank={c.name:c for c in context['bank']}
    for name,candidate in bank.items():
        try:
            source=context['forecasts'][name];line=context['lines'][name]
            counter['ac_calls']+=1
            predicted=lk.ac_predict(candidate,line['r'],line['x'],totalp,totalq,
                source['source_p'],source['source_q'],context['root_v'])
            vlo,vhi=options.get('voltage_bounds',(.85,1.15))
            if np.min(predicted['v'])<vlo or np.max(predicted['v'])>vhi:
                raise ValueError('reference AC voltage outside declared bounds')
            physical[name]=predicted
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            failures.append(dict(reference=name,failure=type(exc).__name__+':'+str(exc)))
    pairs=[];fit_results={}
    for k,(model,pair) in enumerate(zip(context['models'],context['pairs'])):
        a,b=model['a'],model['b'];name=model['name']
        entry=dict(name=name,a=a,b=b,weight=model['task_weight'],success=False,
            reference_line_sha256=before[a],competitor_line_sha256=before[b],
            line_parameters_fixed=True,source_interface=1,EV_in_total_PQ_once=True)
        try:
            if a not in physical:raise RuntimeError('reference AC prediction failed')
            phy=physical[a]
            obs=lk.Observations(p=totalp+_broadcast(meter.bias_p,(T,n),'future P bias'),
                q=totalq+_broadcast(meter.bias_q,(T,n),'future Q bias'),
                v=phy['v']+_broadcast(meter.bias_v,(T,n),'future V bias'),
                root_v=context['root_v'],master_p=phy['master_p'],master_q=phy['master_q'])
            source=context['forecasts'][b];line=context['lines'][b]
            initial=SimpleNamespace(r=line['r'].copy(),x=line['x'].copy(),p=totalp,q=totalq,
                source_p=source['source_p'].copy(),source_q=source['source_q'].copy())
            fitted=fit_fixed_inputs(bank[b],obs,meter,initial=initial,
                fixed_r=line['r'].copy(),fixed_x=line['x'].copy(),seed=int(seed)+k,
                call_counter=counter,**options)
            fit_results[name]=fitted
            if not np.array_equal(fitted.r,line['r']) or not np.array_equal(fitted.x,line['x']):
                raise RuntimeError('fixed training line changed during source fit')
            if not np.allclose(fitted.p,totalp,rtol=0,atol=1e-14) or not np.allclose(fitted.q,totalq,rtol=0,atol=1e-14):
                raise RuntimeError('direct PQ input changed during source fit')
            qp=evaluate_original_pair(pair,u.ravel())
            entry.update(success=bool(fitted.success),added=.5*float(fitted.loss),joint=float(fitted.loss),
                source_p=fitted.source_p,source_q=fitted.source_q,solver=fitted.summary(),
                proxy_QP=qp,proxy_to_AC_distance_difference=float(fitted.loss-qp['primal_upper']),
                fit_source_only=True,fit_fixed_PQ_unchanged=True)
            if not fitted.success:entry['failure']='source fit did not converge'
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError,TypeError) as exc:
            entry['failure']=type(exc).__name__+':'+str(exc)
        pairs.append(entry)
    unchanged=all(before[k]==v['sha256']==_line_id(v['r'],v['x']) for k,v in context['lines'].items())
    success=bool(len(pairs)==len(context['pairs']) and all(x['success'] for x in pairs) and not failures and unchanged)
    result=dict(success=success,pairs=pairs,failures=failures,cost=counter,fit_results=fit_results,
        line_parameters_unchanged=unchanged,uses_validation=False,uses_truth=False,uses_future_noise=False,
        seconds=perf_counter()-began,global_AC_certificate=False,
        scope='attained fixed-training-line AC predictive source profiles')
    if success:
        values=np.array([r['added'] for r in pairs]);weights=np.array([r['weight'] for r in pairs])
        result.update(mean=float(weights@values),cap20_mean=float(weights@np.minimum(values,20)),worst=float(values.min()))
    return result


def _task_value(prediction,cap=None):
    if not prediction.get('success'):return None
    values=np.array([r['added'] for r in prediction['pairs']],float)
    weights=np.array([r['weight'] for r in prediction['pairs']],float)
    if not np.isfinite(values).all() or abs(weights.sum()-1)>1e-10:return None
    return float(weights@(values if cap is None else np.minimum(values,cap)))


def _service(T,n,*,mobile,discharge):
    access=np.ones((T,n),bool)
    if not mobile:access[:,1:]=False
    return build_mobile_service_domain(MobileServiceConfig(n_slots=T,n_sites=n,
        dt_hours=CONFIG['dt_hours'],p_charge_max_kw=CONFIG['command_power_kw'],
        p_discharge_max_kw=CONFIG['command_power_kw'],allow_discharge=discharge,
        battery_capacity_kwh=CONFIG['capacity_kwh'],initial_soc=CONFIG['initial_energy_kwh']/CONFIG['capacity_kwh'],
        terminal_energy_kwh=CONFIG['terminal_energy_kwh'],eta_charge=CONFIG['eta_charge'],eta_discharge=CONFIG['eta_discharge'],
        access=access,mobile=mobile,travel_slots=CONFIG['travel_slots'],setup_slots=CONFIG['setup_slots'],
        travel_energy_kwh=CONFIG['drive_kwh'],max_moves=CONFIG['max_moves'],max_throughput_kwh=CONFIG['throughput_kwh'],
        travel_cost=CONFIG['travel_worker_minutes'],setup_cost=CONFIG['setup_worker_minutes'],
        driver_budget=CONFIG['driver_budget_minutes'],cost_units='worker_minutes',
        min_arrival_throughput_kwh=CONFIG['min_arrival_throughput_kwh']))


def _realize(domain,power):
    op=domain.as_dict();u=np.asarray(power,float).ravel()
    solved=milp(np.zeros(len(op['lower'])),integrality=op['integrality'],
        bounds=Bounds(op['lower'],op['upper']),constraints=[*op['constraints'],LinearConstraint(op['u_map'],u,u)],
        options=dict(time_limit=3.,mip_rel_gap=1e-8))
    audit=domain.check(solved)
    return dict(success=solved.x is not None and audit['feasible'],x=solved.x,audit=audit,
        solver_status=int(solved.status),solver_message=str(solved.message))


def _control_key(power):
    value=np.round(np.asarray(power,float),10);value[value==0.]=0.
    return value.astype('<f8').tobytes()


def _horizon(meter,n):
    v=np.asarray(meter.v_sigma)
    if v.ndim==2:
        if v.shape[1] not in (1,n):raise ValueError('future voltage model terminal dimension')
        return v.shape[0]
    for name in ('master_p_sigma','master_q_sigma'):
        value=np.asarray(getattr(meter,name))
        if value.ndim==1 and len(value)>1:return len(value)
    return CONFIG['horizon']


def select_fixed_controls(training, bank, train_fits, train_meter, future_meter, *,
                           seed, fit_options=None, exchange_options=None, modes=None):
    """Joint route/continuous-power MILP; random prices only post-hoc baseline.

    `modes` is a requested method-name sequence. Needed deterministic parent
    domains are computed automatically. Both primary siblings share the exact
    same deterministic initial plans and exchange configuration. No validation
    argument is accepted. A failed proxy AC gate retains legal checked parents.
    """
    began=perf_counter();requested=METHODS if modes is None else tuple(modes)
    if not requested or len(set(requested))!=len(requested) or not set(requested)<=set(METHODS):
        raise ValueError('modes must be unique known method names')
    options=dict(DEFAULT_EXCHANGE);options.update(exchange_options or {})
    forbidden={'operation','offsets','weights','score_caps','initial_controls','initial_operation_points',
        'score_scale','aggregation'} & set(exchange_options or {})
    if forbidden:raise ValueError('selector owns exchange fields '+repr(sorted(forbidden)))
    if options['master_form']!='shared_control_square':
        raise ValueError('fixed-line policy requires shared_control_square master')
    T,n=_horizon(future_meter,training.n),training.n
    baseline=np.zeros((T,n));baseline[:,0]=(CONFIG['terminal_energy_kwh']-CONFIG['initial_energy_kwh'])/(
        CONFIG['eta_charge']*T*CONFIG['dt_hours'])
    source_options={k:(fit_options or {}).get(k,(-.5,.5)) for k in ('source_p_bounds','source_q_bounds')}
    context=build_fixed_contexts(training,bank,train_fits,train_meter,future_meter,
        -CONFIG['command_power_kw'],CONFIG['command_power_kw'],baseline,base_kw=CONFIG['base_kw'],**source_options)
    out=dict(methods={},failures={},uses_validation=False,uses_truth=False,uses_future_noise=False,
        random_used_by_primary=False,context=context['diagnostics'],predicted_AC={},
        model_task_count=len(context['pairs']),configuration=dict(CONFIG),exchange_configuration=options)
    choices=dict(none=np.zeros_like(baseline),constant=baseline.copy())
    if not context['success']:
        out['failures']['context']=context['diagnostics']['failure']
        for name in requested:
            if name not in choices:choices[name]=None
        out['seconds']=perf_counter()-began
        return {name:choices[name] for name in requested},out
    domains={name:_service(T,n,mobile=flags[0],discharge=flags[1]) for name,flags in DOMAIN_FLAGS.items()}
    base_point=_realize(domains['charge_fixed'],baseline)
    out['constant_operation']=base_point['audit']
    if not base_point['success']:
        choices['constant']=None;out['failures']['constant']='baseline operation infeasible'
    cache={}
    def predicted(power):
        key=_control_key(power)
        if key not in cache:cache[key]=evaluate_fixed_action(context,power,fit_options=fit_options,seed=seed+500)
        return cache[key]
    def choose(checked,cap):
        valid=[r for r in checked if r['operation']['feasible'] and _task_value(r['prediction'],cap) is not None]
        return max(valid,key=lambda r:_task_value(r['prediction'],cap)) if valid else None
    def optimize(name,d,initial,cap=None):
        record=dict(initial_candidates=[],cap=cap,random_used=False,failures=[],recentered=False)
        checked=[];points=[];seen=set()
        for origin,power in initial:
            if power is None or _control_key(power) in seen:continue
            seen.add(_control_key(power));realized=_realize(d,power)
            pred=predicted(power) if realized['success'] else dict(success=False,pairs=[])
            record['initial_candidates'].append(dict(origin=origin,power_kw=power,
                operation=realized['audit'],AC_score=_task_value(pred,cap)))
            if realized['success']:
                checked.append(dict(origin=origin,power_kw=power,operation=realized['audit'],prediction=pred))
                points.append(realized['x'])
        best=choose(checked,cap)
        initial_digest=sha256()
        for entry in record['initial_candidates']:
            initial_digest.update(entry['origin'].encode());initial_digest.update(_control_key(entry['power_kw']))
        record.update(initial_best_origin=None if best is None else best['origin'],
            initial_best_score=None if best is None else _task_value(best['prediction'],cap),
            initial_actions_sha256=initial_digest.hexdigest(),configured_exchange_budget=dict(options))
        op=d.as_dict()
        try:
            result=exchange_original_domain(context['pairs'],op['control_lower'],op['control_upper'],operation=op,
                offsets=np.zeros(len(context['pairs'])),weights=np.full(len(context['pairs']),1/len(context['pairs'])),
                initial_operation_points=points,score_caps=None if cap is None else np.full(len(context['pairs']),cap),
                **options)
            record['raw']=result
            if result.get('u') is None:raise ValueError('MILP exchange returned no feasible action')
            power=np.asarray(result['u']).reshape(T,n);realized=_realize(d,power)
            witness=check_fixed_witness(context,power,gate=CONFIG['proxy_channel_RMS_gate'],
                voltage_bounds=(fit_options or {}).get('voltage_bounds',(.85,1.15)))
            record.update(proxy_AC_check=witness,raw_proxy_gate=witness['gate_passed'],raw_operation=realized['audit'])
            rawpred=predicted(power) if realized['success'] else dict(success=False,pairs=[])
            record['raw_prediction']=rawpred
            if realized['success'] and witness['gate_passed']:
                checked.append(dict(origin='MILP',power_kw=power,operation=realized['audit'],prediction=rawpred))
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError,TypeError) as exc:
            record['failures'].append(type(exc).__name__+':'+str(exc))
        winner=choose(checked,cap)
        if winner is None:
            record['guard']=dict(accepted=False,reason='no complete checked deterministic plan')
            return None,record
        score=_task_value(winner['prediction'],cap)
        is_new=bool(winner['origin']=='MILP' and not any(_control_key(winner['power_kw'])==
            _control_key(r['power_kw']) for r in record['initial_candidates']))
        record.update(selected_origin=winner['origin'],selected_score=score,operation=winner['operation'],
            selected_is_new_MILP_proposal=is_new,inherited_or_initial_retained=not is_new,
            inheritance_protected=best is None or score>=_task_value(best['prediction'],cap)-1e-10,
            guard=dict(accepted=winner['origin']=='MILP',reason='maximum checked training-only AC predictive score',
                score=score,scope='attained AC score, not a design lower certificate'))
        return np.asarray(winner['power_kw']).copy(),record
    if choices['constant'] is not None:out['predicted_AC']['constant']=predicted(baseline)
    quality_prices=1./_broadcast(future_meter.v_sigma,(T,n),'future voltage sigma')**2
    quality_prices/=max(1.,float(np.max(quality_prices)))
    selected={};need_design=bool(set(requested)&set((*DOMAIN_FLAGS,'refined_v2g_mobile','capped_v2g_mobile')))
    if need_design or set(requested)&{'quality_fixed','quality_mobile'}:
        for name,d in domains.items():
            quality=d.solve_prices(quality_prices,time_limit=3.)
            qaudit=d.check(quality);qpower=d.decode(quality)['u_kw'] if qaudit['feasible'] else None
            if name in ('charge_fixed','v2g_mobile'):
                qname='quality_fixed' if name=='charge_fixed' else 'quality_mobile'
                choices[qname]=qpower;out['methods'][qname]=dict(operation=qaudit)
            if not need_design:continue
            initial=[('constant',choices['constant']),*[(p,selected[p]) for p in INHERITS[name] if selected.get(p) is not None]]
            if qpower is not None:initial.append(('quality_'+name,qpower))
            power,record=optimize(name,d,initial)
            selected[name]=power;choices[name]=power;out['methods'][name]=record
            if power is not None:out['predicted_AC'][name]=predicted(power)
        initial=[('constant',choices['constant']),*[(k,v) for k,v in selected.items() if v is not None]]
        if choices.get('quality_mobile') is not None:initial.append(('quality_mobile',choices['quality_mobile']))
        for name,cap in (('refined_v2g_mobile',None),('capped_v2g_mobile',CONFIG['task_cap'])):
            if name not in requested:continue
            power,record=optimize(name,domains['v2g_mobile'],initial,cap)
            choices[name]=power;out['methods'][name]=record
            if power is not None:out['predicted_AC'][name]=predicted(power)
    # This comparator is generated after every requested primary plan.
    if 'random_mobile' in requested:
        solved=domains['v2g_mobile'].solve_prices(np.random.default_rng(seed+81).normal(size=(T,n)),time_limit=3.)
        audit=domains['v2g_mobile'].check(solved)
        choices['random_mobile']=domains['v2g_mobile'].decode(solved)['u_kw'] if audit['feasible'] else None
        out['methods']['random_mobile']=dict(operation=audit,solver_status=int(solved.status),
            role='independent comparator only; never inherited')
    out.update(seconds=perf_counter()-began,unique_predicted_actions=len(cache),
        predicted_fit_calls=sum(r['cost']['fit_calls'] for r in cache.values()),
        predicted_ac_calls=sum(r['cost']['ac_calls'] for r in cache.values()),
        all_prediction_records=list(cache.values()),line_parameters_unchanged=all(r['line_parameters_unchanged'] for r in cache.values()))
    return {name:choices.get(name) for name in requested},out
