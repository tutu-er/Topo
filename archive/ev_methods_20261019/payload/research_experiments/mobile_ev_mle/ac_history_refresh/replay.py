"""Replay saved fixed EV plans with an observed-only AC history refresh.

There is no case generator, action optimizer, future refit initializer, or
measurement-noise generator here. Each candidate is fitted once per history;
the future joint samples are shared across every saved fixed plan.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from . import io
from . import optimizer
from . import posterior
from .. import ev_ac_repair_analysis as analysis
from .. import ev_ac_repair_propagation as propagation


MODEL_NAMES=('G00','G10','G01','G11')


def _array_descriptor(value):
    if value is None:return None
    array=np.asarray(value,dtype='<f8')
    return dict(shape=list(array.shape),sha256=sha256(array.tobytes()).hexdigest(),
        reconstructible_from_saved_observed_data_MAP_and_frozen_source=True)


def _compact_MAP(record):
    return {key:(_array_descriptor(value) if key in ('J_eta','J_state','Jacobian') else value)
            for key,value in record.items()}


def _compact_local(record):
    result={key:value for key,value in record.items() if key not in
        ('untruncated','bounded','affine_jacobian','prior_mean','expansion_point','expansion_mean','affine_offset')}
    d=record.get('line_dimension',0)
    for key in ('untruncated','bounded'):
        row=record.get(key)
        if row is not None:
            result[key]=dict(mean=row['mean'],line_covariance=np.asarray(row['cov'])[:d,:d],
                covariance=_array_descriptor(row['cov']))
    for key in ('affine_jacobian','prior_mean','expansion_point','expansion_mean','affine_offset'):
        if key in record:result[key]=_array_descriptor(record[key])
    return result


def _compact_future(record):
    # Complete future eta+zF moments live once in context.candidate_updates.
    return {key:(_array_descriptor(value) if key in ('future_conditional_matrix',
        'future_innovation_covariance') else value) for key,value in record.items()
        if key not in ('untruncated','bounded')}


def fit_observed(inputs,*,solver_options=None,truncation_options=None,
                 problem_factory=None,solve=None,local_update=None,future_update=None,
                 context_builder=None,progress=None):
    """Consume only sanitized H+V/public data and fit each candidate once."""
    began=perf_counter();problem_factory=problem_factory or optimizer.ACHistoryProblem
    solve=solve or optimizer.solve_problem;local_update=local_update or posterior.final_local_posterior
    future_update=future_update or posterior.future_marginal
    context_builder=context_builder or posterior.build_forecast_context
    callback=(lambda *args:None) if progress is None else progress
    results={};future_results={};failures={};calls=0
    settings=inputs['options']['truncation'] if truncation_options is None else truncation_options
    for candidate in inputs['candidates']:
        name=candidate.name;initial=inputs['initializations'][name]
        row=dict(initialization={k:v for k,v in initial.items() if k not in ('eta','latent')},
                 status='not_run',MAP_solves=0)
        results[name]=row
        if initial['status']!='ok':
            row.update(status='initialization_failure');failures[name]=initial['status'];continue
        try:
            numeric={key:value for key,value in (solver_options or {}).items()}
            physical={key:inputs['options'][key] for key in ('source_p_bounds','source_q_bounds','voltage_bounds')}
            for key,value in physical.items():
                if key in numeric and not np.array_equal(np.asarray(numeric[key]),np.asarray(value)):
                    raise ValueError('original physical domain must remain unchanged: '+key)
            problem=problem_factory(candidate,inputs['history'],inputs['history_meter'],
                inputs['line_priors'][name],inputs['K_D'],options=physical)
            calls+=1;row['MAP_solves']=1
            fitted=solve(problem,initial['eta'],initial['latent'],options={**physical,**numeric})
            row['MAP']=_compact_MAP(fitted)
            local=local_update(problem,fitted,truncation_options=settings);row['posterior']=_compact_local(local)
            future=future_update(local,inputs['K_D'],inputs['K_FD'],inputs['K_FF'])
            future_results[name]=future;row['future']=_compact_future(future)
            row['status']='ok' if fitted.get('success') and local.get('success') and future.get('success') else 'candidate_failure'
            if row['status']!='ok':failures[name]=dict(MAP=fitted.get('status'),
                posterior=local.get('status'),future=future.get('status'))
        except Exception as exc:
            row.update(status='candidate_exception',failure=type(exc).__name__+': '+str(exc))
            failures[name]=row['failure']
        callback('candidate_'+name,dict(status=row['status'],MAP_solves=row['MAP_solves']))
        # Large historical matrices have already been summarized. No history
        # covariance/Jacobian is retained while fitting the next candidate.
        problem=fitted=local=None
    context=None;context_record=None
    if not failures and len(future_results)==len(inputs['candidates']):
        try:
            context,context_record=context_builder(inputs['candidates'],future_results,
                inputs['future_meter'],horizon=inputs['horizon'],base_kw=inputs['base_kw'],
                root_v=inputs['future_root_v'],options=inputs['options'])
        except Exception as exc:failures['context']=type(exc).__name__+': '+str(exc)
    record=dict(status='ok' if context is not None else 'history_refresh_failure',
        candidates=results,failures=failures,MAP_solves=calls,
        expected_candidates=inputs['candidate_names'],no_candidate_dropped=True,
        original_prior_used_once=True,historical_likelihood_used_once=True,
        old_topology_weights_used=False,uses_future_measurements=False,
        context=context_record,seconds=perf_counter()-began)
    return dict(record=record,context=context,context_record=context_record)


def _failure_score(names):
    return dict(available=False,retained_candidates=list(names),unresolved=1.,
        false_rejection=0.,correct_decisive=0.,wrong_decisive=0.,
        MAP_error=None,MAP_error_missing_range=[0.,1.],
        scope='operational retain-all fallback; point prediction unavailable')


def replay_record(raw,*,base_kw=20.,future_root_v=1.,draws=256,replicates=4,
                  solver_options=None,truncation_options=None,variance_multiplier=1.,
                  fit_function=None,forecast_function=None,score_function=None,progress=None):
    """New B inference on unchanged saved plans and future measurements.

    All forecasts are fixed before `evaluation_payload` reads any future data
    or true label. `variance_multiplier` must come from an independently saved
    calibration in a later frozen phase; development uses its default one.
    """
    began=perf_counter();callback=(lambda *args:None) if progress is None else progress
    spec={k:v for k,v in raw.get('spec',{}).items()}
    result=dict(schema='ac_history_refresh_fixed_replay_v1',spec=spec,
        status='input_failure',plans={},actions={},failures={},n_independent_histories=1,
        config=dict(draws=int(draws),replicates=int(replicates),variance_multiplier=float(variance_multiplier),
            base_kw=float(base_kw),future_root_v=future_root_v,models=list(MODEL_NAMES),
            no_new_actions=True,no_new_measurements=True,no_new_scientific_cases=True,
            common_posterior_samples_across_plans=True,solver_options=solver_options,
            truncation_options=truncation_options))
    try:inputs=io.prepare_history(raw,base_kw=base_kw,future_root_v=future_root_v)
    except Exception as exc:
        result['failures']['input']=type(exc).__name__+': '+str(exc)
        result['seconds']=perf_counter()-began;return result
    names=inputs['candidate_names'];result.update(candidate_names=names,
        observed_input_sha256=inputs['observed_input_sha256'],plans=inputs['plans'],
        public_options=inputs['options'],process_hyperparameters=inputs['hyperparameters'],
        initializations={name:{k:v for k,v in row.items() if k not in ('eta','latent')}
                         for name,row in inputs['initializations'].items()})
    result['failures'].update({'plan_'+k:v for k,v in inputs['plan_failures'].items()})
    fit_function=fit_function or fit_observed;forecast_function=forecast_function or propagation.forecast_moments
    score_function=score_function or analysis.scores
    try:
        fitted=fit_function(inputs,solver_options=solver_options,truncation_options=truncation_options,
                            progress=progress)
    except Exception as exc:
        fitted=dict(record=dict(status='history_refresh_exception',
            failures={'exception':type(exc).__name__+': '+str(exc)},no_candidate_dropped=True),
            context=None,context_record=None)
    result['history_refresh']=fitted['record']
    if fitted['context'] is None:result['failures']['history_refresh']=fitted['record']['failures']
    cache={};seen_power={};seed=int(np.random.SeedSequence(
        [int(spec.get('seed',0)),6200,20261017]).generate_state(1)[0])
    # Forecast pass: future observation/label containers have not been read.
    for label,plan in inputs['plans'].items():
        key=plan['action_key'];power=plan['power']
        if key in result['actions']:
            if not np.array_equal(power,seen_power[key]):raise io.InputFailure('same action key has different powers')
            result['actions'][key]['methods'].append(label);continue
        seen_power[key]=power.copy()
        action=dict(methods=[label],power=power,propagation=dict(status='context_unavailable',banks={}),
                    regimes={},failures={})
        result['actions'][key]=action
        if fitted['context'] is not None:
            try:
                action['propagation']=forecast_function(fitted['context'],fitted['context_record'],power,
                    seed=seed,draws=draws,replicates=replicates,posterior_cache=cache)
                if action['propagation'].get('status')!='ok':
                    action['failures']['propagation']=action['propagation'].get('status')
            except Exception as exc:action['failures']['propagation']=type(exc).__name__+': '+str(exc)
        callback('forecast_'+label,dict(status=action['propagation']['status']))
    result['posterior_sampling']={name:{k:v for k,v in row.items() if k!='samples'}
                                  for name,row in cache.items()}
    result['posterior_cache_candidates']=list(cache)
    # Scoring pass: the only future information read is saved output vectors.
    try:evaluation=io.evaluation_payload(raw,inputs)
    except Exception as exc:
        result['failures']['evaluation_input']=type(exc).__name__+': '+str(exc)
        evaluation=dict(true_name=None,true_index=None,candidate_miss=True,actions={})
    result.update(true_name=evaluation['true_name'],true_index=evaluation['true_index'],
                  candidate_miss=evaluation['candidate_miss'])
    for key,action in result['actions'].items():
        usable=analysis.usable_banks(action)
        for regime in ('regular','shift'):
            actual=evaluation['actions'].get(key,{}).get(regime)
            row=dict(scores={name:_failure_score(names) for name in MODEL_NAMES})
            action['regimes'][regime]=row
            if actual is None:
                row.update(status='measurement_missing');action['failures'][regime]='saved measurement missing';continue
            row.update(status='ok',values=actual['values'],clean=actual['clean'])
            if evaluation['true_index'] is None:
                row.update(status='candidate_miss');continue
            for model,bank in usable.items():
                try:
                    noisy=score_function(bank,actual['values'],evaluation['true_index'],a=variance_multiplier)
                    clean=score_function(bank,actual['clean'][None],evaluation['true_index'],a=variance_multiplier)
                    row['scores'][model]=dict(available=True,noisy=noisy,clean=clean)
                except Exception as exc:action['failures'][regime+'_'+model]=type(exc).__name__+': '+str(exc)
        if action['failures']:result['failures']['action_'+key]=action['failures']
    result['status']='complete' if not result['failures'] and not result['candidate_miss'] else 'partial_failure'
    result['seconds']=perf_counter()-began
    return result


def replay_file(source,output,**kwargs):
    source,output=Path(source),Path(output)
    if source.resolve()==output.resolve():raise ValueError('B output must differ from original A raw')
    if output.exists():raise FileExistsError('preserve existing B record: '+str(output))
    raw=io.load_record(source)
    result=replay_record(raw,**kwargs)
    result['source_A']=dict(path=str(source.resolve()),sha256=sha256(source.read_bytes()).hexdigest(),
                           original_record_unchanged=True)
    io.save_record(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--draws',type=int,default=256);parser.add_argument('--replicates',type=int,default=4)
    args=parser.parse_args();source=Path(args.input)
    # This development entry point cannot be used to inspect A evaluation.
    if source.parent.name!='development' or source.name not in ('history_000.json.gz','history_007.json.gz'):
        parser.error('before B freeze this entry point permits saved development histories 0 and 7 only')
    replay_file(source,args.output,draws=args.draws,replicates=args.replicates)


if __name__=='__main__':main()
