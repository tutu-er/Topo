"""Deterministic compact-MILP policy; random prices are evaluation controls only.

The selector accepts historical meters/fits and public device data. It never
receives simulator parameters, future readings, or future noise. Large EV
domains inherit physically feasible smaller-domain policies, not random ones.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import ev_original_study as base
from . import gradient_price_study as io
from .ev_original_ac_check import (build_multisite_context,
    evaluate_multisite_predicted_action, make_ac_context, evaluate_ac_witness,
    recenter_at_absolute)
from .ev_original_domain_milp import from_moving_model, exchange_original_domain

HERE = Path(__file__).resolve().parent
DOMAIN_FLAGS = {
    'charge_fixed': (False, False), 'v2g_fixed': (False, True),
    'charge_mobile': (True, False), 'v2g_mobile': (True, True),
}
INHERITS = {
    'charge_fixed': (), 'v2g_fixed': ('charge_fixed',),
    'charge_mobile': ('charge_fixed',),
    'v2g_mobile': ('charge_fixed', 'v2g_fixed', 'charge_mobile'),
}
CONFIG = dict(
    version='compact_deterministic_v2', physical=deepcopy(base.CONFIG),
    master_form='shared_control_square', prediction='center', task_cap=20.,
    methods=['none','constant','quality_fixed','quality_mobile','random_mobile',
        'charge_fixed','v2g_fixed','charge_mobile','v2g_mobile',
        'refined_v2g_mobile','capped_v2g_mobile'],
    command_power_kw=6./1.05,
    command_derating='rated 6 kW divided by maximum declared tracking gain 1.05',
    ensemble=dict(max_explanations=3,history_gate_half_sse=2.,radius_fraction=.15,
        max_backtracks=2,ensemble_axes=[0,1],tracking_fraction=.05,
        load_floor_pu=.002,source_floor_pu=.005),
    execution=dict(independent_history_workers=4,solver_threads=1,blas_threads=1),
    exchange=dict(max_iterations=6, control_segments=4, residual_segments=4,
        max_segments=16, solver_seconds=2., total_seconds=35.,
        absolute_gap=.01, relative_gap=.01, witness_policy='affine_preferred',
        backend='gurobi'),
    proxy_channel_RMS_gate=.25, max_recenter=1,
    random_role='independent baseline only; generated after MILP selection',
    candidate_policy='all four candidates and every directed pair',
    development_seeds=list(range(989101,989105)),
    evaluation_seeds=list(range(989201,989213)),
    calibration_seeds=list(range(989301,989337)),
    formal_alpha=.1, calibrated_before_evaluation=False,
    selector_uses_truth=False, selector_uses_future_noise=False,
)


def operations_config():
    config=deepcopy(base.CONFIG)
    config['device']['power_kw']=CONFIG['command_power_kw']
    return config


def service_domain(*,mobile,discharge):
    original=base.service_domain(mobile=mobile,discharge=discharge)
    config=replace(original.config,p_charge_max_kw=CONFIG['command_power_kw'],
        p_discharge_max_kw=CONFIG['command_power_kw'])
    return base.build_mobile_service_domain(config)


def control_key(power):
    # Canonicalize negligible integer-MIP noise before sharing fit evaluations.
    value=np.round(np.asarray(power, float),10)
    value[value==0.]=0.
    return value.astype('<f8').tobytes()


def task_value(result, cap=None):
    """Predeclared task mean. A cap limits easy tasks' marginal influence."""
    if not result.get('success'):
        return None
    pairs = result['pairs']
    values = np.asarray([r['added'] for r in pairs], float)
    weights = np.asarray([r.get('weight', 1./len(pairs)) for r in pairs], float)
    if (not np.isfinite(values).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.)):
        raise ValueError('complete finite task values and normalized weights required')
    if cap is not None:
        if not np.isfinite(cap) or cap < 0:
            raise ValueError('finite nonnegative task cap required')
        values = np.minimum(values, cap)
    return float(weights @ values)


def choose_checked(checked, cap=None):
    """Stable deterministic maximum among complete, physically checked plans."""
    legal = [r for r in checked if r.get('operation', {}).get('feasible')
             and task_value(r['prediction'], cap) is not None]
    if not legal:
        return None
    return max(legal, key=lambda r: task_value(r['prediction'], cap))


def inherited_candidates(name, selected):
    """Only deterministic policies from nested physical domains are inherited."""
    return [(parent, selected[parent]) for parent in INHERITS[name]
            if selected.get(parent) is not None]


def _model_bank(history, bank, hfits, hm, fm, baseline, domain, prediction):
    bounds = {k:v for k,v in base.fit_options(domain).items()
              if k in ('r_bounds','x_bounds')}
    if prediction == 'center':
        context = build_multisite_context(history, bank, hfits, hm, fm,
            baseline, -6., 6., base_kw=base.BASE_KW, line_bounds=bounds)
        evaluator = lambda power, jm, seed: evaluate_multisite_predicted_action(
            history, bank, hfits, hm, fm, jm, power, base_kw=base.BASE_KW,
            fit_options=base.fit_options(domain), seed=seed)
        return context, evaluator
    if prediction!='ensemble':raise ValueError('unknown prediction model')
    from .ev_prediction_ensemble import (build_history_explanations,
        build_forecast_scenarios,build_ensemble_contexts,evaluate_ensemble_action)
    options=CONFIG['ensemble']
    explanations=build_history_explanations(history,bank,hfits,hm,
        fit_options=base.fit_options(domain),**{k:options[k] for k in
            ('max_explanations','history_gate_half_sse','radius_fraction','max_backtracks')})
    scenes=build_forecast_scenarios(history,hm,hfits,len(baseline),base_kw=base.BASE_KW,
        **{k:options[k] for k in ('tracking_fraction','load_floor_pu','source_floor_pu')})
    context=build_ensemble_contexts(history,bank,hfits,hm,fm,baseline,-6.,6.,
        base_kw=base.BASE_KW,line_bounds=bounds,explanations=explanations,
        scenarios=scenes,ensemble_axes=options['ensemble_axes'])
    context['diagnostics'].update(history_explanations=explanations['records'],
        explanation_diagnostics=explanations['diagnostics'],
        forecast_scenarios=scenes,context_records=context['context_records'],
        failure=None if context['success'] else 'one_or_more_ensemble_contexts_failed')
    evaluator=lambda power,jm,seed:evaluate_ensemble_action(history,bank,hfits,hm,fm,jm,power,
        explanations=explanations,scenarios=scenes,ensemble_axes=options['ensemble_axes'],
        base_kw=base.BASE_KW,fit_options=base.fit_options(domain),seed=seed,contexts=context['contexts'])
    return context,evaluator


def select_controls(history, bank, hfits, hm, fm, jm, *, seed, domain,
                    prediction=None, master_form=None):
    """Optimize routes and continuous power, with H-only AC incumbent protection."""
    began = perf_counter()
    prediction = CONFIG['prediction'] if prediction is None else prediction
    master_form = CONFIG['master_form'] if master_form is None else master_form
    T, n = base.CONFIG['horizon'], history.n
    baseline = np.zeros((T,n))
    dcfg = base.CONFIG['device']
    baseline[:,0] = (dcfg['terminal_energy_kwh']-dcfg['initial_energy_kwh']) / (
        dcfg['eta_charge']*T*base.CONFIG['dt_hours'])
    domains = {name:service_domain(mobile=flags[0],discharge=flags[1])
               for name,flags in DOMAIN_FLAGS.items()}
    choices = dict(none=np.zeros_like(baseline),constant=baseline.copy())
    out = dict(methods={},failures={},uses_truth=False,uses_future_noise=False,
        random_used_by_primary=False,master_form=master_form,prediction=prediction,
        all_candidates=len(bank),physical_directed_pairs=len(bank)*(len(bank)-1),
        predicted_AC={},context=None)
    base_point = base.realize_control(domains['charge_fixed'],baseline)
    out['constant_operation'] = base_point['audit']
    if not base_point['success']:
        choices['constant']=None
        out['failures']['constant']='operation_infeasible'
        out['seconds']=perf_counter()-began
        return choices,out
    context,evaluator = _model_bank(history,bank,hfits,hm,fm,baseline,domain,prediction)
    out['context']=context['diagnostics']
    if not context['success']:
        out['failures']['context']=context['diagnostics']['failure']
        for name in (*DOMAIN_FLAGS,'refined_v2g_mobile','capped_v2g_mobile'):
            choices[name]=baseline.copy()
        out['seconds']=perf_counter()-began
        return choices,out
    models=context['models'];pairs=[from_moving_model(m) for m in models]
    offsets=[float(hfits[m['b']].loss) for m in models]
    weights=np.asarray([m.get('task_weight',1./len(models)) for m in models])
    if not np.isclose(weights.sum(),1.):
        raise ValueError('normalized model weights required')
    cache={}
    def predicted(power):
        key=control_key(power)
        if key not in cache:
            cache[key]=evaluator(np.asarray(power,float),jm,seed+500)
        return cache[key]
    baseline_ac=predicted(baseline)
    out['predicted_AC']['constant']=baseline_ac
    prices=np.tile(np.asarray(base.CONFIG['public_voltage_repeat_counts'])[:,None],(1,n))
    selected={};selected_predictions={}
    for name,d in domains.items():
        quality=d.solve_prices(prices,time_limit=3.)
        qaudit=d.check(quality)
        quality_power=d.decode(quality)['u_kw'] if qaudit['feasible'] else None
        if name in ('charge_fixed','v2g_mobile'):
            qname='quality_fixed' if name=='charge_fixed' else 'quality_mobile'
            choices[qname]=quality_power
            out['methods'][qname]=dict(operation=qaudit)
        initial=[('constant',baseline),*inherited_candidates(name,selected)]
        if quality_power is not None:
            initial.append(('quality_'+name,quality_power))
        power,record=_optimize_domain(name,d,initial,models,pairs,offsets,weights,
            history,bank,hfits,hm,fm,predicted,master_form=master_form)
        selected[name]=power;choices[name]=power;out['methods'][name]=record
        if power is not None:
            selected_predictions[name]=predicted(power)
            out['predicted_AC'][name]=selected_predictions[name]
    # Mean and capped refinements receive exactly the same deterministic
    # available plans and optimization budgets. Neither receives the other.
    # Random is generated only after both primary objectives finish.
    d=domains['v2g_mobile']
    initial=[('constant',baseline),*[(k,v) for k,v in selected.items() if v is not None]]
    if choices.get('quality_mobile') is not None:
        initial.append(('quality_mobile',choices['quality_mobile']))
    mean_power,mean_record=_optimize_domain('refined_v2g_mobile',d,initial,models,pairs,offsets,weights,
        history,bank,hfits,hm,fm,predicted,master_form=master_form)
    choices['refined_v2g_mobile']=mean_power;out['methods']['refined_v2g_mobile']=mean_record
    if mean_power is not None:out['predicted_AC']['refined_v2g_mobile']=predicted(mean_power)
    power,record=_optimize_domain('capped_v2g_mobile',d,initial,models,pairs,offsets,weights,
        history,bank,hfits,hm,fm,predicted,master_form=master_form,cap=CONFIG['task_cap'])
    choices['capped_v2g_mobile']=power;out['methods']['capped_v2g_mobile']=record
    if power is not None:out['predicted_AC']['capped_v2g_mobile']=predicted(power)
    random=domains['v2g_mobile'].solve_prices(
        np.random.default_rng(seed+81).normal(size=(T,n)),time_limit=3.)
    audit=domains['v2g_mobile'].check(random)
    choices['random_mobile']=None if not audit['feasible'] else domains['v2g_mobile'].decode(random)['u_kw']
    out['methods']['random_mobile']=dict(operation=audit,solver_status=int(random.status),
        solver_message=str(random.message),role='independent comparator; never inherited')
    out.update(seconds=perf_counter()-began,unique_predicted_actions=len(cache),
        predicted_fit_calls=sum(x.get('fit_calls',0) for x in cache.values()),
        predicted_ac_calls=sum(x.get('ac_calls',0) for x in cache.values()),
        all_prediction_records=list(cache.values()),model_task_count=len(models))
    return choices,out


def _optimize_domain(name,d,initial,models,pairs,offsets,weights,history,bank,hfits,
                     hm,fm,predicted,*,master_form,cap=None):
    T,n=base.CONFIG['horizon'],history.n
    record=dict(initial_candidates=[],raw=None,guard=None,proxy_AC_checks=[],
        recentered=False,cap=cap,random_used=False,failures=[])
    checked=[];points=[];controls=[]
    seen=set()
    for origin,power in initial:
        key=control_key(power)
        if key in seen:continue
        seen.add(key)
        r=base.realize_control(d,power)
        entry=dict(origin=origin,power_kw=np.asarray(power,float),operation=r['audit'],
            prediction=predicted(power) if r['success'] else dict(success=False,pairs=[]))
        record['initial_candidates'].append(dict(origin=origin,operation=r['audit'],
            AC_score=task_value(entry['prediction'],cap)))
        if r['success']:
            checked.append(entry);points.append(r['x'])
    best=choose_checked(checked,cap)
    record['initial_best_score']=None if best is None else task_value(best['prediction'],cap)
    record['initial_best_origin']=None if best is None else best['origin']
    op=d.as_dict()
    def exchange(local_pairs):
        return exchange_original_domain(local_pairs,op['control_lower'],op['control_upper'],
            operation=op,offsets=offsets,weights=weights,initial_controls=controls,
            initial_operation_points=points,master_form=master_form,
            score_caps=None if cap is None else np.full(len(local_pairs),cap),
            **CONFIG['exchange'])
    def checks(result,local_models):
        selected=result['evaluations'][result['best_evaluation_index']]
        return [dict(name=m['name'],check=evaluate_ac_witness(m,result['u'],
            qp['physical_parameter'],make_ac_context(m,history,bank,hfits,hm,fm,
                voltage_limits=(.9,1.1))))
            for m,qp in zip(local_models,selected['qp_results'])]
    def passes(rows):
        return all(r['check']['complete_AC_voltage_feasible']
            and r['check']['physical_box_feasible']
            and r['check']['max_channel_RMS']<=CONFIG['proxy_channel_RMS_gate'] for r in rows)
    try:
        local_models=models;result=exchange(pairs);record['raw']=result
        if result.get('u') is None:
            raise ValueError('master returned no executable input')
        audits=checks(result,local_models);record['proxy_AC_checks'].append(audits)
        if not passes(audits) and CONFIG['max_recenter']:
            selected=result['evaluations'][result['best_evaluation_index']]
            local_models=[recenter_at_absolute(m,result['u'],qp['physical_parameter'],
                make_ac_context(m,history,bank,hfits,hm,fm))
                if not a['check']['max_channel_RMS']<=CONFIG['proxy_channel_RMS_gate'] else m
                for m,qp,a in zip(models,selected['qp_results'],audits)]
            record['before_recenter']=result;record['recentered']=True
            controls.append(np.asarray(result['u']).ravel())
            result=exchange([from_moving_model(m) for m in local_models]);record['raw']=result
            if result.get('u') is None:raise ValueError('recenter returned no executable input')
            audits=checks(result,local_models);record['proxy_AC_checks'].append(audits)
        power=np.asarray(result['u']).reshape(T,n)
        physical=base.realize_control(d,power)
        rawpred=predicted(power) if physical['success'] else dict(success=False,pairs=[])
        entry=dict(origin='MILP',power_kw=power,operation=physical['audit'],prediction=rawpred)
        record['raw_prediction']=rawpred
        gate=passes(audits)
        if gate:checked.append(entry)
        record['raw_proxy_gate']=gate
        record['raw_operation']=physical['audit']
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError,TypeError) as exc:
        record['failures'].append(type(exc).__name__+':'+str(exc))
    winner=choose_checked(checked,cap)
    if winner is None:
        record['guard']=dict(accepted=False,reason='all_H_only_AC_checks_failed')
        return None,record
    final=task_value(winner['prediction'],cap)
    record.update(selected_origin=winner['origin'],operation=winner['operation'],
        selected_score=final,inheritance_protected=best is None or final>=task_value(best['prediction'],cap)-1e-10,
        guard=dict(accepted=winner['origin']=='MILP',reason='maximum_checked_H_only_AC_score',
            score=final,scope='attained H-only prediction; future actual separation evaluated independently'))
    return np.asarray(winner['power_kw']).copy(),record


def run_trial(seed,family,domain='asset',*,prediction=None,master_form=None):
    began=perf_counter();case=base.make_case(seed,family);prepared=base.prepare(case,domain)
    choices,selection=select_controls(case['history'],prepared['candidates'],prepared['fits'],
        case['hm'],case['fm'],case['jm'],seed=case['seeds']['policy'],domain=domain,
        prediction=prediction,master_form=master_form)
    rows=base.evaluate(case,prepared,choices,domain)
    return dict(seed=seed,domain=domain,seeds=case['seeds'],history=io.observations_record(case['history']),
        historical_fits=[io.fit_record(c,prepared['fits'][c.name]) for c in prepared['candidates'] if c.name in prepared['fits']],
        historical_failures=prepared['failures'],historical_cost=prepared['cost'],proposal=prepared['proposal'],
        measurement_models={k:base.meter_to_record(case[k]) for k in ('hm','fm','jm')},
        selection=selection,rows=rows,operations_config=operations_config(),truth={k:case[k] for k in
            ('family','r','x','p','q','source_p','source_q','root_v','white','assets')},
        source_interface=1,short_edge=False,independent_unit='one historical sample',seconds=perf_counter()-began,
        global_AC_MLE_certified=False,statistical_risk_calibrated=False)


def source_hashes():
    hashes=base.source_hashes()
    for name in ('ev_compact_study.py','ev_compact_protocol.md','ev_shared_square_milp.py',
                 'ev_prediction_ensemble.py'):
        path=HERE/name
        if path.exists():hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['development','evaluation','calibration'],default='development')
    parser.add_argument('--prediction',choices=['center','ensemble'],default=CONFIG['prediction'])
    parser.add_argument('--master-form',choices=['residual_chord','shared_control_square'],default=CONFIG['master_form'])
    parser.add_argument('--domain',choices=['asset','numeric'],default='asset')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--limit',type=int)
    parser.add_argument('--pilot-gate',type=Path)
    parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    if args.workers<1 or not 0<=args.worker<args.workers:
        raise ValueError('worker index must lie in 0..workers-1')
    seeds=CONFIG[args.phase+'_seeds'];hashes=source_hashes()
    policy=dict(config=CONFIG,prediction=args.prediction,master_form=args.master_form,
                domain=args.domain,source_sha256=hashes)
    policy_id=hashlib.sha256(json.dumps(io.serial(policy),sort_keys=True).encode()).hexdigest()
    if args.phase!='development':
        if args.pilot_gate is None:raise ValueError('saved passed pilot gate required')
        gate=io.read_json(args.pilot_gate)
        if not gate.get('passed') or gate.get('policy_id')!=policy_id:
            raise ValueError('gate failed or belongs to another source/config')
        if set(gate.get('development_seeds',[]))&set(seeds):raise ValueError('history reuse')
    freeze=dict(**policy,policy_config=policy,policy_id=policy_id,phase=args.phase,seeds=seeds,
                frozen_before_trials=True,random_used_by_primary=False)
    args.output.mkdir(parents=True,exist_ok=True);fp=args.output/'freeze.json'
    if fp.exists() and io.read_json(fp)!=io.serial(freeze):raise ValueError('freeze differs; choose new directory')
    if not fp.exists():
        io.write_json(fp,freeze)
        snapshot={}
        for name,h in hashes.items():
            path=HERE/name if not name.startswith('rnj_wzzt_core/') else HERE.parents[1]/name
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('source changed before snapshot')
            snapshot[name]=dict(sha256=h,bytes_base64=base64.b64encode(raw).decode('ascii'))
        io.write_json(args.output/'source_snapshot.json.gz',snapshot,compressed=True)
    for index,seed in enumerate(seeds[:args.limit]):
        if index%args.workers!=args.worker:continue
        path=args.output/f'trial_{seed}.json.gz'
        if path.exists():continue
        if source_hashes()!=hashes:raise ValueError('frozen source changed')
        result=run_trial(seed,base.FAMILIES[index%4],args.domain,
            prediction=args.prediction,master_form=args.master_form)
        result.update(policy_id=policy_id,policy_config=policy,phase=args.phase,config=CONFIG,
            prediction=args.prediction,master_form=args.master_form,source_sha256=hashes)
        if source_hashes()!=hashes:
            io.write_json(args.output/('invalid_'+path.name),result,compressed=True)
            raise ValueError('source changed during trial; preserved invalid evidence')
        io.write_json(path,result,compressed=True)
        summary=dict(seed=seed,seconds=result['seconds'],selection_seconds=result['selection']['seconds'],
            margins={r['method']:r.get('margin') for r in result['rows']},
            selected_origins={k:v.get('selected_origin') for k,v in result['selection']['methods'].items()
                if k in (*DOMAIN_FLAGS,'refined_v2g_mobile','capped_v2g_mobile')})
        io.write_json(args.output/f'checkpoint_worker_{args.worker}.json',dict(completed=len(list(args.output.glob('trial_*.json.gz'))),
            planned=len(seeds),last=summary,research_complete=False))
        print(json.dumps(io.serial(summary)),flush=True)


if __name__=='__main__':main()
