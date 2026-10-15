"""Strict two-candidate, fixed-site end-to-end EV response-metric study.

Star and clade_01 are declared before simulation. Training-only planning fixes
one R/X vector per topology. Validation and new observations independently fit
only the original physical source. No WZZT or truth-based shortlist is used.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import traceback

import numpy as np
import scipy

from . import ev_dynamic_study as dynamic
from . import ev_fixed_train_study as old
from . import ev_fixed_train_design as fixed
from . import ev_rx_study as rx
from . import ev_gradient_flow_study as previous
from . import ev_gradient_flow_design as gradient
from . import gradient_price_study as io
from .ev_dynamic_loads import fit_forecast,FORECAST_CONFIG
from .ev_rx_precision import solve_squared_precision
from .information_context import meter_to_record
from .likelihood import RadialCandidate

HERE=Path(__file__).parent
FAMILIES=('star','clade_01')
METHODS=('none','constant','highlow','square','nearest','gradac','contrast_joint','contrast_box','contrast_load')
CONFIG=deepcopy(previous.CONFIG)
CONFIG.update(version='strict_binary_fixed_site_response_metric_v1',
    select_modes=[name for name in METHODS if name!='highlow'],
    domains=['v2g_fixed'],declared_candidates=list(FAMILIES),
    candidate_protocol='predeclared star/clade_01; same two candidates retained for final evaluation',
    wzzt_used=False,planning_seconds=35.,
    contrast=dict(solver_seconds=2.,total_seconds=35.,backend='gurobi'),
    load_pressure=dict(components=2,scale='sqrt(gap+t+1)',directions='joint PQ innovation covariance PCs',
        scenarios=5,statistical_coverage_claimed=False),
    development_seeds=list(range(994101,994105)),
    calibration_seeds=list(range(994301,994319)),
    evaluation_seeds=list(range(994201,994209)),
    planning_anchor='public analytic square curve; all methods use identical frozen context',
    scope='conditional two-candidate LV cable synthetic experiment; fixed site0 V2G only')


def declared_bank():
    return [RadialCandidate.from_supports('star',3,()),
            RadialCandidate.from_supports('clade_01',3,((0,1),))]


def candidate_records(bank=None):
    return [dict(name=c.name,supports=c.supports) for c in (declared_bank() if bank is None else bank)]


def validate_config(config):
    groups=[set(config[key]) for key in ('development_seeds','calibration_seeds','evaluation_seeds')]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)):
        raise ValueError('development/calibration/evaluation histories must be disjoint')
    if any(len(config[key])!=len(set(config[key])) for key in
           ('development_seeds','calibration_seeds','evaluation_seeds')):
        raise ValueError('duplicate independent history seeds')
    if (config['declared_candidates']!=list(FAMILIES) or config['domains']!=['v2g_fixed']
            or config.get('wzzt_used') is not False):
        raise ValueError('strict predeclared two-candidate fixed-site experiment required')
    if (config['planning_seconds']<=0 or config['horizon']!=6
            or config['public_voltage_repeat_counts']!=[5]*6):
        raise ValueError('six uniform-precision slots and positive planning budget required')
    if tuple(config['select_modes'])!=tuple(m for m in METHODS if m!='highlow'):
        raise ValueError('exact nine-method protocol required')
    if config.get('load_pressure')!=dict(components=2,scale='sqrt(gap+t+1)',
            directions='joint PQ innovation covariance PCs',scenarios=5,statistical_coverage_claimed=False):
        raise ValueError('frozen deterministic five-scenario pressure rule required')


def family_for_seed(seed, config=CONFIG):
    for key in ('development_seeds','calibration_seeds','evaluation_seeds'):
        if seed in config[key]:return config[key].index(seed)%2
    raise ValueError('seed is absent from frozen independent-history plan')


def source_hashes():
    result=previous.source_hashes()
    for name in ('ev_response_metric_design.py','ev_response_metric_study.py',
                 'ev_response_metric_protocol.md','ev_response_metric_math.md'):
        result[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    return result


def load_pressure_scenarios(training,train_meter,forecast,*,base_kw=20.,components=2):
    """Deterministic nominal +/- principal residual directions, not a CI.

    One-step errors are reconstructed using the same training P/PF AR1 rules
    as the nominal forecast.  sqrt(h) scales principal innovation directions
    across the heldout gap and horizon; it is a declared stress rule, not an
    estimated predictive coverage region. Source paths and R/X are untouched.
    """
    p=np.asarray(training.p,float)-fixed._broadcast(train_meter.bias_p,np.shape(training.p),'P bias')
    q=np.asarray(training.q,float)-fixed._broadcast(train_meter.bias_q,np.shape(training.q),'Q bias')
    if p.ndim!=2 or p.shape!=q.shape or len(p)<4 or not np.isfinite([p,q]).all():
        raise ValueError('at least four finite calibrated training PQ rows required')
    meta=forecast['metadata'];n=p.shape[1];T=len(forecast['p'])
    if meta.get('training_only') is not True or any(meta.get(key) is not False
            for key in ('uses_validation','uses_truth','uses_future_noise')):
        raise ValueError('explicit training-only forecast metadata required')
    if components!=2:raise ValueError('the frozen pressure protocol has exactly two PCs')
    ar=meta['ar1'];phi=np.asarray(ar['p']['phi']);intercept=np.asarray(ar['p']['intercept'])
    fp=np.asarray(ar['pf']['phi']);fi=np.asarray(ar['pf']['intercept'])
    positive=np.maximum(p,0.);nonnegative_q=np.maximum(q,0.)
    pf=np.divide(positive,np.hypot(positive,nonnegative_q),out=np.full_like(p,.97),where=positive>0.)
    pf=np.clip(pf,*FORECAST_CONFIG['pf_bounds'])
    plo,phi_power=np.asarray(FORECAST_CONFIG['p_bounds_kw'])/base_kw
    pp=np.clip(intercept+phi*p[:-1],plo,phi_power)
    pf_next=np.clip(fi+fp*pf[:-1],*FORECAST_CONFIG['pf_bounds'])
    pq=pp*np.tan(np.arccos(pf_next))
    residual=np.column_stack((p[1:]-pp,q[1:]-pq))
    residual-=residual.mean(axis=0)
    covariance=residual.T@residual/max(len(residual)-2,1)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    order=np.argsort(-eigenvalues,kind='stable');eigenvalues=np.maximum(eigenvalues[order],0.)
    eigenvectors=eigenvectors[:,order]
    for j in range(eigenvectors.shape[1]):
        pivot=int(np.argmax(abs(eigenvectors[:,j])))
        if eigenvectors[pivot,j]<0:eigenvectors[:,j]*=-1
    steps=np.asarray(meta['forecast_steps'],int)
    if steps.shape!=(T,) or np.any(steps<1):raise ValueError('forecast step indices required')
    scale=np.sqrt(steps.astype(float))
    nominal_p=np.asarray(forecast['p'],float);nominal_q=np.asarray(forecast['q'],float)
    pf_low,pf_high=FORECAST_CONFIG['pf_bounds']
    records=[]
    def append(name,raw_p,raw_q,pc,sign):
        pc_p=np.clip(raw_p,plo,phi_power)
        qlo=pc_p*np.tan(np.arccos(pf_high));qhi=pc_p*np.tan(np.arccos(pf_low))
        pc_q=np.clip(raw_q,qlo,qhi)
        clips=dict(p=int(np.count_nonzero(pc_p!=raw_p)),q_pf=int(np.count_nonzero(pc_q!=raw_q)),
            p_bounds_kw=list(FORECAST_CONFIG['p_bounds_kw']),pf_bounds=list(FORECAST_CONFIG['pf_bounds']),
            raw_p_min=float(raw_p.min()),raw_p_max=float(raw_p.max()),
            raw_q_min=float(raw_q.min()),raw_q_max=float(raw_q.max()))
        records.append(dict(name=name,p=pc_p,q=pc_q,metadata=dict(training_only=True,
            uses_validation=False,uses_truth=False,uses_future_noise=False,principal_component=pc,
            sign=sign,forecast_steps=steps.tolist(),scale_rule='sqrt(h) deterministic pressure',
            clips=clips,statistical_coverage_claimed=False)))
    append('nominal',nominal_p,nominal_q,None,0)
    # Preserve the nominal forecast exactly; any difference flags an inconsistent
    # public physical forecast, rather than silently changing its baseline.
    if not np.array_equal(records[0]['p'],nominal_p) or not np.allclose(records[0]['q'],nominal_q,rtol=0,atol=1e-14):
        raise ValueError('nominal forecast is outside declared P/PF physical bounds')
    records[0]['p']=nominal_p.copy();records[0]['q']=nominal_q.copy()
    for pc in range(components):
        direction=np.sqrt(eigenvalues[pc])*eigenvectors[:,pc]
        for sign in (1,-1):
            delta=sign*scale[:,None]*direction[None,:]
            append(f'pc{pc+1}_'+('plus' if sign>0 else 'minus'),
                nominal_p+delta[:,:n],nominal_q+delta[:,n:],pc+1,sign)
    report=dict(training_only=True,uses_validation=False,uses_truth=False,uses_future_noise=False,
        covariance=covariance,eigenvalues=eigenvalues,eigenvectors=eigenvectors,
        residual_count=len(residual),residual_dimension=2*n,covariance_denominator=max(len(residual)-2,1),
        residual_model='one-step P/PF AR1; Q reconstructed before joint PQ residual covariance',
        scenario_count=len(records),principal_components=components,forecast_steps=steps,
        pressure_scale=scale,statistical_coverage_claimed=False,
        units='P/Q per unit on 20kW base',parameter_uncertainty_included=False,
        source_uncertainty_added=False,clips={r['name']:r['metadata']['clips'] for r in records})
    return records,report


def freeze(directory,config):
    validate_config(config);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    sources=source_hashes()
    record=dict(config=config,source_sha256=sources,policy_id=io.identity(config,sources),
        base_commit='0b88c343',environment=dict(python=platform.python_version(),
            numpy=np.__version__,scipy=scipy.__version__))
    path=directory/'freeze.json'
    if path.exists():
        saved=io.read_json(path)
        if saved['policy_id']!=record['policy_id']:raise ValueError('freeze differs; preserve prior directory')
        return saved
    io.write_json(path,record)
    snapshot={name:base64.b64encode(((HERE.parents[1]/name) if name.startswith('rnj_wzzt_core/')
        else HERE/name).read_bytes()).decode() for name in sources}
    io.write_json(directory/'source_snapshot.json.gz',dict(source_sha256=sources,
        source_base64=snapshot),compressed=True)
    return record


def select_plans(training,bank,fitted,training_meter,future_meter,forecast,config,*,seed,counter):
    """Planning boundary: training, trained models, forecast and public device only."""
    from .ev_response_metric_design import select_contrast_controls
    if [c.name for c in bank]!=list(FAMILIES):raise ValueError('predeclared pair required')
    began=perf_counter();T,n=config['horizon'],training.n
    line_ids={c.name:fixed._line_id(fitted[c.name].r,fitted[c.name].x) for c in bank}
    curves=previous.public_curves(T,n)
    domain=fixed._service(T,n,mobile=False,discharge=True)
    plans={name:curves[name].copy() for name in ('none','constant','highlow')}
    metadata=dict(methods={},timing={},one_trained_parameter_vector=True,
        uses_truth=False,uses_validation=False,uses_future_noise=False,random_used_by_primary=False,
        declared_candidates=list(FAMILIES),WZZT_used=False,domain='v2g_fixed')
    setup_seconds=perf_counter()-began;metadata['public_setup_seconds']=setup_seconds
    for name in ('none','constant','highlow'):
        metadata['timing'][name]=dict(planning_actual_seconds=0.,planning_standalone_seconds=0.,
            dependency_seconds=0.,soft_budget_overrun_seconds=0.,actual_calls={})
    before=previous._counts(counter);began=perf_counter()
    try:
        options=deepcopy(config['exchange']);options['total_seconds']=max(.01,config['planning_seconds']-setup_seconds)
        square=solve_squared_precision(domain,future_meter,
            initial_powers=[curves['constant'],curves['highlow'],curves['analytic_square']],
            exchange_options=options)
        canonical=curves['analytic_square'].copy();qstar=float(np.sum(canonical**2))
        raw_q=square.get('Q')
        square.update(raw_power_kw=square.get('power_kw'),raw_Q=raw_q,
            selected_power_kw=canonical,selected_Q=qstar,
            raw_above_proved_optimum=bool(raw_q is not None and raw_q>qstar+1e-5),
            tie_break='public saturated order at the independently proved six-slot uniform-domain Q optimum',
            proof_reference='ev_rx_precision_proof.md')
        plans['square']=canonical;metadata['methods']['square']=square
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
        plans['square']=curves['analytic_square'].copy()
        metadata['methods']['square']=dict(success=False,fallback='public_analytic_square',failure=str(exc))
    own=perf_counter()-began
    metadata['timing']['square']=dict(planning_actual_seconds=own,planning_standalone_seconds=own+setup_seconds,
        dependency_seconds=setup_seconds,dependencies=[],actual_calls=previous._delta(counter,before),
        soft_budget_overrun_seconds=max(0.,own+setup_seconds-config['planning_seconds']))
    before=previous._counts(counter);began=perf_counter()
    context=fixed.build_fixed_contexts(training,bank,fitted,training_meter,future_meter,
        -fixed.CONFIG['command_power_kw'],fixed.CONFIG['command_power_kw'],plans['square'],
        forecast=forecast,**config['source'])
    context_seconds=perf_counter()-began;context_calls=previous._delta(counter,before)
    metadata['context']=context['diagnostics']
    scenario_began=perf_counter()
    try:pressure,pressure_record=load_pressure_scenarios(training,training_meter,forecast)
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
        pressure=None;pressure_record=dict(success=False,failure=type(exc).__name__+':'+str(exc))
    scenario_seconds=perf_counter()-scenario_began
    metadata['load_pressure']=pressure_record;metadata['load_pressure_scenarios']=pressure
    for name in ('nearest','gradac','contrast_joint','contrast_box','contrast_load'):
        before=previous._counts(counter);began=perf_counter()
        dep=setup_seconds+metadata['timing']['square']['planning_actual_seconds']+context_seconds
        if name=='contrast_load':dep+=scenario_seconds
        remaining=config['planning_seconds']-dep
        initials=dict(constant=curves['constant'],highlow=curves['highlow'],square=plans['square'])
        try:
            if not context['success']:raise RuntimeError('fixed-line context failed')
            if name=='contrast_load' and pressure is None:raise RuntimeError('training load-pressure scenarios failed')
            if remaining<=0:
                power=plans['square'].copy();row=dict(success=False,selected_origin='dependency_budget_fallback',
                    failure='no remaining standalone planning budget',executed_fallback='square')
            elif name=='nearest':
                power,row=previous.fixed_surrogate_policy(context,domain,initials,config,
                    seed=seed,seconds=remaining)
            elif name=='gradac':
                opts=deepcopy(config['gradient']);opts['total_seconds']=remaining
                power,row=gradient.select_gradient_controls(context,domain,initials,method='ac',
                    aggregation='binarymean',fit_options={**config['profile'],**config['source']},
                    options=opts,seed=seed,cache=None)
            else:
                opts=deepcopy(config['contrast']);opts['total_seconds']=remaining
                power,row=select_contrast_controls(context,domain,initials,
                    projection='joint' if name=='contrast_joint' else 'bounded_average',options=opts,seed=seed,
                    **(dict(prediction_scenarios=pressure[1:]) if name=='contrast_load' else {}))
            if power is None:
                fallback=row.get('fallback_power')
                power=np.asarray(fallback).copy() if fallback is not None else plans['square'].copy()
                row['executed_fallback']='first_feasible' if fallback is not None else 'square'
            plans[name]=power;metadata['methods'][name]=row
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            plans[name]=plans['square'].copy()
            metadata['methods'][name]=dict(success=False,fallback='square',failure=type(exc).__name__+':'+str(exc))
        own=perf_counter()-began;calls=previous._delta(counter,before)
        charged=dict(calls)
        for block in (metadata['timing']['square']['actual_calls'],context_calls):
            for key,value in block.items():charged[key]=charged.get(key,0)+value
        metadata['timing'][name]=dict(planning_actual_seconds=own,planning_standalone_seconds=own+dep,
            dependency_seconds=dep,dependencies=['square'],charged_ancestors=['square'],
            context_seconds=context_seconds,context_calls=context_calls,actual_calls=calls,
            scenario_seconds=scenario_seconds if name=='contrast_load' else 0.,
            standalone_calls=charged,soft_budget_overrun_seconds=max(0.,own+dep-config['planning_seconds']))
    metadata['line_parameters_unchanged']=all(line_ids[c.name]==fixed._line_id(fitted[c.name].r,fitted[c.name].x)
        for c in bank)
    return plans,metadata


def planning_diagnostics(training,bank,fitted,training_meter,future_meter,forecast,plans,config,*,seed,counter):
    """Score all executed actions with training models, after planning is over.

    The diagnostic has no true/future observations. Its cost never pays for a
    selector or changes any action; the final evaluator later joins actual M.
    """
    from .ev_response_metric_design import build_contrast,evaluate_contrast
    began=perf_counter();before=previous._counts(counter)
    context=fixed.build_fixed_contexts(training,bank,fitted,training_meter,future_meter,
        -fixed.CONFIG['command_power_kw'],fixed.CONFIG['command_power_kw'],plans['square'],
        forecast=forecast,**config['source'])
    record=dict(success=False,methods={},uses_truth=False,uses_validation=False,
        uses_future_noise=False,scope='training-model diagnostics after all plans are frozen; excluded from pipeline time')
    if not context['success']:
        record.update(failure=context['diagnostics']['failure'],seconds=perf_counter()-began,
            calls=previous._delta(counter,before));return record
    engine=gradient.ProfileEngine(context,fit_options={**config['profile'],**config['source']},seed=seed)
    models={};record['models']={}
    for label,projection in (('joint','joint'),('box','bounded_average')):
        try:
            model=build_contrast(context,plans['square'],projection=projection)
            models[label]=model
            record['models'][label]={key:value for key,value in model.items() if key not in ('responses','context')}
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            record['models'][label]=dict(success=False,failure=str(exc))
    pressure_models={}
    if 'box' in models:
        try:
            pressure,_=load_pressure_scenarios(training,training_meter,forecast)
            pressure_models['nominal']=models['box']
            for scenario in pressure[1:]:
                varied=dict(context);varied['background']=dict(p=scenario['p'],q=scenario['q'])
                pressure_models[scenario['name']]=build_contrast(varied,plans['square'],
                    projection='bounded_average',fixed_weights=models['box']['weights'])
            record['load_models']={name:{key:value for key,value in model.items()
                if key not in ('responses','context')} for name,model in pressure_models.items()}
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            pressure_models={};record['load_model_failure']=type(exc).__name__+':'+str(exc)
    cache={}
    for name in METHODS:
        power=plans.get(name)
        if power is None:
            record['methods'][name]=dict(success=False,failure='no executed action');continue
        key=fixed._control_key(power)
        if key not in cache:
            profile=engine.evaluate(power,'ac')
            value=gradient._aggregate(profile['pairs'],'binarymean') if profile['success'] else dict(score=None)
            row=dict(success=profile['success'],future_profile_distance=value['score'],full_AC_profile=profile)
            for label,model in models.items():
                try:row['contrast_'+label]=evaluate_contrast(model,power)
                except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    row['contrast_'+label]=dict(success=False,score=None,failure=str(exc))
            if pressure_models:
                try:
                    values={label:evaluate_contrast(model,power) for label,model in pressure_models.items()}
                    good=all(value['success'] for value in values.values())
                    row['contrast_load']=dict(success=good,score=min(value['score'] for value in values.values())
                        if good else None,scenario_evaluations=values)
                except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    row['contrast_load']=dict(success=False,score=None,failure=str(exc))
            cache[key]=row
        record['methods'][name]=dict(cache[key],curve_id=hashlib.sha256(key).hexdigest())
    record.update(success=all(row.get('success') for row in record['methods'].values()),
        seconds=perf_counter()-began,calls=previous._delta(counter,before),
        profile_counters=engine.counters(),unique_curves=len(cache))
    return record


def _run_trial(seed,family_index,config,counter):
    if family_index not in (0,1):raise ValueError('true family must be star or clade_01')
    began=perf_counter();hashes=source_hashes()
    bank=declared_bank()  # Candidate declaration precedes any simulator truth.
    case=dynamic.make_case(seed,family_index,config)
    tr,tm,T=case['train'],case['tm'],config['horizon']
    started=perf_counter();before=previous._counts(counter)
    fitted,training=rx.train_bank(bank,tr,tm,config,case['assets']['seeds']['policy'])
    train_seconds=perf_counter()-started;train_calls=previous._delta(counter,before)
    true_name=FAMILIES[family_index]
    if next(c for c in bank if c.name==true_name).supports!=case['assets']['candidate'].supports:
        raise ValueError('simulated family differs from predeclared pair')
    record=dict(schema='ev_strict_binary_response_metric_v1',seed=seed,family_index=family_index,
        config=config,source_sha256=hashes,policy_id=io.identity(config,hashes),
        training=io.observations_record(tr),training_meter=meter_to_record(tm),
        future_meter=meter_to_record(case['fm']),validation=io.observations_record(case['validation']),
        validation_meter=meter_to_record(case['vm']),training_result=training,
        candidates=candidate_records(bank),proposal=dict(wzzt_used=False,predeclared_candidates=True,
            candidate_selection_uses_truth=False),true_name=true_name,candidate_miss=False,methods=[],
        truth=dict(r=case['assets']['r'],x=case['assets']['x'],supports=case['assets']['candidate'].supports),
        background=case['background'],seeds=case['assets']['seeds'])
    if not training['all_complete']:
        record.update(status='training_failure',seconds=perf_counter()-began);return record
    started=perf_counter();before=previous._counts(counter)
    _,validated=old.fit_bank(bank,case['validation'],case['vm'],trained=fitted,
        options={**config['profile'],**config['source']},seed=seed+100)
    validation_seconds=perf_counter()-started;validation_calls=previous._delta(counter,before)
    started=perf_counter()
    forecast=fit_forecast(tr,tm,fitted,horizon=T,gap=config['forecast_gap'],base_kw=20.,kind='ar1')
    forecast_seconds=perf_counter()-started
    record.update(forecast=forecast,validation_result=validated)
    started=perf_counter()
    plans,selection=select_plans(tr,bank,fitted,tm,case['fm'],forecast,config,seed=seed+200,counter=counter)
    record.update(selection=selection,selection_seconds=perf_counter()-started)
    cache={};common=train_seconds+validation_seconds+forecast_seconds
    for name in METHODS:
        power=plans.get(name);row=dict(name=name,power_kw=power,status='selection_failure')
        row['planning_success']=selection['methods'].get(name,{}).get('success',power is not None)
        if power is None:record['methods'].append(row);continue
        started=perf_counter()
        operation=(dict(success=True,battery_service=False) if name=='none' else
            dynamic.operation_record(power,mobile=False,discharge=name not in ('constant','highlow')))
        audit_seconds=perf_counter()-started;row['operation']=operation
        if not operation['success']:
            row['status']='operation_failure';record['methods'].append(row);continue
        key=fixed._control_key(power);hit=key in cache
        row['curve_id']=hashlib.sha256(key).hexdigest()
        if not hit:
            started=perf_counter();before=previous._counts(counter)
            try:value=dynamic.actual_curve(case,bank,fitted,validated,power,config,seed=seed+500)
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                value=dict(status='physical_or_fit_failure',M=None,failure=str(exc))
            value['actual_evaluation_seconds']=perf_counter()-started
            value['actual_evaluation_calls']=previous._delta(counter,before);cache[key]=value
        row.update(cache[key]);row['actual_fit_cache_hit']=hit
        timing=selection['timing'][name];future=float(row.get('noisy',{}).get('seconds',0.))
        row['runtime']=dict(common_history_pipeline_seconds=common,operational_future_fit_seconds=future,
            **timing,final_operation_audit_seconds=audit_seconds,
            end_to_end_standalone_seconds=common+timing['planning_standalone_seconds']+future+audit_seconds,
            standalone_AC_batches=train_calls.get('AC_batches',0)+validation_calls.get('AC_batches',0)
                +timing.get('standalone_calls',timing.get('actual_calls',{})).get('AC_batches',0)
                +row.get('noisy',{}).get('cost',{}).get('ac_calls',0),
            simulation_and_clean_diagnostic_seconds=max(0.,row['actual_evaluation_seconds']-future))
        record['methods'].append(row)
    record['planning_diagnostics']=planning_diagnostics(tr,bank,fitted,tm,case['fm'],forecast,
        plans,config,seed=seed+200,counter=counter)
    record['common_runtime']=dict(training_and_candidates_seconds=train_seconds,
        validation_seconds=validation_seconds,forecast_seconds=forecast_seconds,
        training_calls=train_calls,validation_calls=validation_calls)
    fit=fitted[true_name];assets=case['assets']
    record['training_parameter_error']=dict(R=float(np.linalg.norm(fit.r-assets['r'])/np.linalg.norm(assets['r'])),
        X=float(np.linalg.norm(fit.x-assets['x'])/np.linalg.norm(assets['x'])))
    record.update(status='complete' if validated['all_complete'] and all(r['status']=='complete' for r in record['methods'])
        else 'partial_failure',seconds=perf_counter()-began,unique_actual_curves=len(cache),
        one_trained_parameter_vector=True,validation_used_by_selector=False,future_noise_used_by_selector=False)
    return record


def run_trial(seed,family_index,config=None):
    cfg=deepcopy(CONFIG if config is None else config);validate_config(cfg)
    if family_index not in (0,1):raise ValueError('strict two-candidate true family required')
    before=source_hashes();began=perf_counter()
    with dynamic.count_physics_calls() as counter:
        try:record=_run_trial(seed,family_index,cfg,counter)
        except Exception as exc:
            record=dict(schema='ev_strict_binary_response_metric_v1',seed=seed,family_index=family_index,
                config=cfg,source_sha256=before,policy_id=io.identity(cfg,before),methods=[],
                candidates=candidate_records(),true_name=FAMILIES[family_index],candidate_miss=False,
                status='driver_failure',failure=type(exc).__name__+':'+str(exc),traceback=traceback.format_exc(),
                seconds=perf_counter()-began)
    record.update(actual_solver_calls=counter,source_unchanged=before==source_hashes())
    return io.serial(record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',required=True)
    args=parser.parse_args();cfg=deepcopy(CONFIG);frozen=freeze(args.output,cfg)
    for index,seed in enumerate(args.seeds):
        family=family_for_seed(seed,cfg);path=args.output/f'trial_{seed}.json.gz'
        if path.exists():raise ValueError('existing trial retained; use a fresh output directory')
        record=run_trial(seed,family,cfg)
        io.write_json(path,record,compressed=True)
        if record['policy_id']!=frozen['policy_id'] or not record['source_unchanged']:
            raise RuntimeError('source/config changed; trial retained, inspect and refreeze')
        planned=next(cfg[key] for key in ('development_seeds','calibration_seeds','evaluation_seeds') if seed in cfg[key])
        io.write_json(args.output/'checkpoint.json',dict(completed=len(list(args.output.glob('trial_*.json.gz'))),
            planned=len(planned),batch_completed=index+1,batch_planned=len(args.seeds),last_seed=seed,
            policy_id=frozen['policy_id']))
        print(json.dumps(dict(seed=seed,status=record['status'],seconds=record['seconds'],
            M={row['name']:row.get('M') for row in record['methods']})),flush=True)
        if record['status']=='driver_failure':raise RuntimeError('driver failure saved; inspect before continuing')


if __name__=='__main__':main()
