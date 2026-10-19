"""One trained line vector; independent precision controls and grouped MILP.

No simulator case, validation, true topology or future noise is accepted.
Full-AC source fits score each deterministic or MILP proposal identically.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np

from . import ev_fixed_train_design as old
from .ev_original_domain_milp import exchange_original_domain
from .ev_rx_precision import solve_squared_precision


def ac_score(prediction, aggregation):
    if not prediction.get('success') or not prediction.get('pairs'):
        return None
    rows=prediction['pairs']
    if not all(r.get('success') for r in rows):return None
    values=np.array([r['added'] for r in rows],float)
    if not np.isfinite(values).all():return None
    if aggregation=='mean':return float(np.mean(values))
    if aggregation!='nearest':raise ValueError('mean or nearest required')
    groups={r['a'] for r in rows}
    return float(np.mean([min(r['added'] for r in rows if r['a']==a) for a in sorted(groups)]))


def select_controls(training,bank,train_fits,train_meter,future_meter,*,forecast,
                    seed,fit_options=None,exchange_options=None,highlow=None):
    """Four matched operation domains; 2 precision and 2 model objectives."""
    began=perf_counter();T,n=old._horizon(future_meter,training.n),training.n
    options=dict(old.DEFAULT_EXCHANGE);options.update(exchange_options or {})
    forbidden={'operation','offsets','weights','score_caps','initial_controls',
        'initial_operation_points','score_scale','aggregation','task_groups'}
    if forbidden & set(exchange_options or {}):raise ValueError('selector owns aggregation and domain fields')
    baseline=np.zeros((T,n));baseline[:,0]=(old.CONFIG['terminal_energy_kwh']-
        old.CONFIG['initial_energy_kwh'])/(old.CONFIG['eta_charge']*T*old.CONFIG['dt_hours'])
    choices={'none':np.zeros_like(baseline),'constant':baseline,'highlow':highlow}
    domains={name:old._service(T,n,mobile=flags[0],discharge=flags[1])
        for name,flags in old.DOMAIN_FLAGS.items()}
    record=dict(methods={},failures={},uses_validation=False,uses_truth=False,
        uses_future_noise=False,random_used_by_primary=False,
        one_trained_parameter_vector=True,baseline_independent_of_AC=True,
        exchange_configuration=options,configuration=dict(old.CONFIG),
        proxy_bound_units='half whitened SSE; exchange default score_scale=.5; same units as AC scores')
    prices=1/old._broadcast(future_meter.v_sigma,(T,n),'voltage sigma')**2
    prices/=max(1.,float(np.max(prices)))
    # These baselines deliberately run before any topology-dependent context.
    for name,domain in domains.items():
        key='linear_'+name
        try:
            solved,tie=old._quality_plan(domain,prices,'early_stay');audit=domain.check(solved)
            choices[key]=domain.decode(solved)['u_kw'] if audit['feasible'] else None
            record['methods'][key]=dict(operation=audit,tie_break=tie,
                uses_topology=False,uses_load_forecast=False)
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            choices[key]=None;record['failures'][key]=type(exc).__name__+':'+str(exc)
        initial=[v for v in (baseline,highlow,choices[key]) if v is not None]
        initial.extend(choices['square_'+p] for p in old.INHERITS[name]
            if choices.get('square_'+p) is not None)
        key='square_'+name
        try:
            squared=solve_squared_precision(domain,future_meter,initial_powers=initial,
                exchange_options=options)
            choices[key]=squared.get('power_kw');record['methods'][key]=squared
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            choices[key]=None;record['failures'][key]=type(exc).__name__+':'+str(exc)
    source={k:(fit_options or {}).get(k,(-.5,.5)) for k in ('source_p_bounds','source_q_bounds')}
    context=old.build_fixed_contexts(training,bank,train_fits,train_meter,future_meter,
        -old.CONFIG['command_power_kw'],old.CONFIG['command_power_kw'],baseline,
        base_kw=old.CONFIG['base_kw'],forecast=forecast,**source)
    record['context']=context['diagnostics'];record['model_task_count']=len(context['pairs'])
    if not context['success']:
        record['failures']['context']=context['diagnostics'].get('failure','AC context failed')
        for agg in ('mean','nearest'):
            choices.update({agg+'_'+name:None for name in domains})
        record['seconds']=perf_counter()-began
        return choices,record
    cache={}
    def predicted(u):
        key=old._control_key(u)
        if key not in cache:
            cache[key]=old.evaluate_fixed_action(context,u,fit_options=fit_options,seed=seed+500)
        return cache[key]
    groups={m['name']:m['a'] for m in context['models']}
    for agg in ('mean','nearest'):
        for name,domain in domains.items():
            key=agg+'_'+name;checked=[];initial=[];points=[];seen=set()
            candidates=[('constant',baseline),('highlow',highlow),
                ('linear_'+name,choices.get('linear_'+name)),
                ('square_'+name,choices.get('square_'+name))]
            candidates.extend((agg+'_'+p,choices.get(agg+'_'+p)) for p in old.INHERITS[name])
            for origin,u in candidates:
                if u is None or old._control_key(u) in seen:continue
                seen.add(old._control_key(u));realized=old._realize(domain,u)
                if not realized['success']:continue
                pred=predicted(u);score=ac_score(pred,agg)
                initial.append(dict(origin=origin,power_kw=u,AC_score=score,
                    operation=realized['audit']))
                points.append(realized['x'])
                if score is not None:checked.append((score,origin,np.asarray(u),pred))
            method=dict(initial_candidates=initial,aggregation=agg,task_groups=groups,
                configured_exchange_budget=options,failures=[])
            op=domain.as_dict()
            try:
                raw=exchange_original_domain(context['pairs'],op['control_lower'],op['control_upper'],
                    operation=op,offsets=np.zeros(len(groups)),weights=np.full(len(groups),1/len(groups)),
                    initial_operation_points=points,aggregation=agg,
                    **({'task_groups':groups} if agg=='nearest' else {}),**options)
                method['raw']=raw
                if raw.get('u') is None:raise RuntimeError('no MILP incumbent')
                u=np.asarray(raw['u']).reshape(T,n);realized=old._realize(domain,u)
                method['raw_operation']=realized['audit']
                if realized['success']:
                    pred=predicted(u);method['raw_prediction']=pred;score=ac_score(pred,agg)
                    if score is not None:checked.append((score,'MILP',u,pred))
                # Same fixed witness check as the preceding stage; diagnostic only.
                method['proxy_AC_check']=old.check_fixed_witness(context,u,
                    gate=old.CONFIG['proxy_channel_RMS_gate'])
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError,TypeError) as exc:
                method['failures'].append(type(exc).__name__+':'+str(exc))
            if checked:
                best=max(checked,key=lambda row:row[0]);choices[key]=best[2].copy()
                initial_values=[row['AC_score'] for row in initial if row['AC_score'] is not None]
                method.update(selected_score=best[0],selected_origin=best[1],
                    selected_is_new_MILP_proposal=best[1]=='MILP' and old._control_key(best[2]) not in seen,
                    initial_best_score=max(initial_values) if initial_values else None,
                    inheritance_protected=not initial_values or best[0]>=max(initial_values)-1e-10,
                    all_AC_tasks_required=True,proxy_gate_is_diagnostic=True)
            else:choices[key]=None;method['failures'].append('no complete feasible AC plan')
            record['methods'][key]=method
    # Both objectives are reported for all baseline and selected actions.
    record['predictions']={}
    for name,u in choices.items():
        if u is not None:
            pred=predicted(u)
            record['predictions'][name]=dict(mean=ac_score(pred,'mean'),nearest=ac_score(pred,'nearest'),
                success=pred['success'],pairs=pred['pairs'])
    record.update(all_prediction_records=list(cache.values()),unique_predicted_actions=len(cache),
        seconds=perf_counter()-began,line_parameters_unchanged=all(
            r.get('line_parameters_unchanged',False) for r in cache.values()))
    return choices,record
