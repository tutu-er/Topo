"""Independent fixed-plan AC-prediction repair pilot and study runner.

This file creates only new histories.  Planning functions receive an explicit
observed-only dictionary.  Truth/future trajectories enter only the subsequent
evaluation block; no corrected prediction or refit changes the fixed plans.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from itertools import product
from hashlib import sha256
import os
from pathlib import Path
from time import perf_counter
import traceback

NUMERIC_THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
# Set before importing NumPy/SciPy in the standalone worker. Controllers must
# also pass these values before interpreter startup if NumPy was imported first.
for _thread_name in NUMERIC_THREADS:
    os.environ[_thread_name]='1'
import numpy as np

from . import ev_bayes_response_cases as cases
from . import ev_bayes_response_ac as posterior
from . import ev_bayes_response_study as old
from . import ev_bayes_response_milp as old_milp
from . import ev_bayes_small_study as small
from . import ev_bayes_small_probability as probability
from . import ev_mobile_v3_cases as public
from . import ev_mobile_v4_domain as nested
from . import ev_rx_study as train
from . import ev_systematic_refit as legacy_refit
from . import ev_ac_repair_refit as refit
from . import ev_ac_repair_propagation as propagation
from . import gradient_price_study as io
from . import likelihood as lk
from .information_context import meter_to_record


CONFIG = dict(version='fixed_plan_ac_repair_v2_quadrature18', horizon=8, evidence_count=20,
    methods=['constant', 'SQMD', 'BCMD'], regimes=['regular', 'shift'],
    draws=256, replicates=4, noise_repeats=1, clean_refit=True,
    training_multistart=1, training_max_nfev=300,
    refit_multistart=2, refit_max_nfev=400, refit_tol=1e-10,
    planning_samples_per_candidate=256, independent_samples_per_candidate=1024,
    posterior_options=deepcopy(old.CONFIG['posterior_options']),
    prior_topology='same approximate historical evidence as old joint model',
    P_Q_scope='Bayes retains input error; conditional refit holds measured P/Q fixed',
    generation='new nominal synthetic LV histories; no PV or wind; unit-PF EV',
    selector_scope='observed training/evidence, public meters and candidates only',
    posterior_samples_saved=False, corrected_model_used_for_planning=False,
    no_evaluation_tuning=True, seed_bases=dict(
        development=2026120100, calibration=2026121000, evaluation=2026122000))
# More numerical work, with every original GHK precision threshold unchanged.
CONFIG['posterior_options']['truncation']['max_power']=18


def specifications(phase):
    if phase not in CONFIG['seed_bases']:
        raise ValueError('development, calibration or evaluation required')
    if phase == 'development':
        combinations = [(bank, family, (32,96)[family % 2],
            ('ordinary','correlated')[(family//2+bi) % 2])
            for family in range(4) for bi, bank in enumerate(('n3','n4'))]
    else:
        combinations = list(product(('n3','n4'), range(4),
                                    (32,96), ('ordinary','correlated')))
    return [dict(phase=phase, index=i, stratum_id=i, seed=CONFIG['seed_bases'][phase]+i,
        bank=bank, family_index=family, H=H, excitation=excitation,
        innovation='nominal', innovation_multiplier=1., replicate=0,
        legacy_audit=False) for i,(bank,family,H,excitation) in enumerate(combinations)]


def _seed(seed, tag):
    return int(np.random.SeedSequence([int(seed), int(tag), 20261017]).generate_state(1)[0])


def _rng(seed, tag):
    return np.random.default_rng(np.random.SeedSequence([int(seed),int(tag),20261017]))


def _planning(observed, fits, posterior_options, seed, progress):
    """No case, physical asset, true label or future trajectory accepted."""
    began = perf_counter(); failures = {}; costs = {}; logs = {}; record = {}
    context = None; context_record = None; objective = None
    T = observed['horizon']; n = observed['training'].n
    template = replace(public.service_domain(T,n,mobile=True).config, n_slots=T)
    domains = nested.build_nested_domains(template)
    constant = np.zeros((T,n))
    constant[:,0] = (template.terminal_energy_kwh-template.initial_soc*
        template.battery_capacity_kwh)/(template.eta_charge*T*template.dt_hours)
    highlow = np.zeros_like(constant)
    highlow[:T//2,0] = constant[0,0]*T/(T//2)
    baseline_operations = {name:small.complete_curve(domains['fixed_charge'],power)
        for name,power in (('constant',constant),('highlow',highlow))}
    x = baseline_operations['constant']
    plans = dict(constant=dict(domain='fixed_charge',power=constant,
        action_key=nested.action_key(constant),physicalcheck=domains['fixed_charge'].check(x),
        operation_x=x,decoded=domains['fixed_charge'].decode(x)))
    try:
        context,context_record = posterior.build_context(observed['training'],
            observed['evidence'],observed['train_meter'],observed['evidence_meter'],
            observed['candidates'],fits,observed['future_meter'], horizon=T,
            base_kw=observed['base_kw'],options=posterior_options)
        K=len(observed['candidates']); dimension=T*(3*n+2)
        noise=_rng(seed,2000).standard_normal((1,K,
            CONFIG['planning_samples_per_candidate'],dimension))
        objective=old.Objective(context,noise,posterior)
    except Exception as exc:
        failures['posterior_context']=type(exc).__name__+': '+str(exc)
        if hasattr(exc,'evidence'):record['context_failure_evidence']=exc.evidence
    progress('context',dict(available=context is not None,failures=failures))
    squares = None
    try:
        squares,logs['square'],costs['square']=old.square_family(domains,
            observed['future_meter'],constant,highlow,
            iterations=old.CONFIG['square_base_iterations'])
        x=squares['mobile_bidir'];power=domains['mobile_bidir'].decode(x)['u_kw']
        plans['SQMD']=dict(domain='mobile_bidir',power=power,
            action_key=nested.action_key(power),physicalcheck=domains['mobile_bidir'].check(x),
            operation_x=x,decoded=domains['mobile_bidir'].decode(x))
    except Exception as exc:
        failures['square_planning']=type(exc).__name__+': '+str(exc)
    progress('square_planned',dict(available='SQMD' in plans))
    if objective is not None and squares is not None:
        try:
            ranked,logs['ranked'],costs['ranked']=old.risk_family(domains,
                objective,squares,baseline_operations)
            bc,logs['BC'],costs['BC']=old.risk_family(domains,objective,squares,
                baseline_operations,optimizer=old_milp.solve_bc_milp,extra_plans=ranked)
            x=bc['mobile_bidir'];power=domains['mobile_bidir'].decode(x)['u_kw']
            plans['BCMD']=dict(domain='mobile_bidir',power=power,
                action_key=nested.action_key(power),physicalcheck=domains['mobile_bidir'].check(x),
                operation_x=x,decoded=domains['mobile_bidir'].decode(x))
        except Exception as exc:
            failures['BC_planning']=type(exc).__name__+': '+str(exc)
    progress('plans_fixed',dict(methods=list(plans)))
    centers={} if context is None else {r['candidate'].name:dict(r=r['r'],x=r['x'])
                                        for r in context['rows']}
    if objective is not None:
        costs['objective']=dict(cache_calls=objective.calls,ac_states=objective.ac_states,
            seconds=objective.seconds,included_in_family_seconds=True,
            entries={key:dict(seconds=row['seconds'],
                AC_states=row['forecast'].get('AC_states')) for key,row in objective.cache.items()})
    record.update(plans=plans,planning_logs=logs,costs=costs,failures=failures,
        operation_config=asdict(template),prediction=context_record,
        posterior_centers={} if context is None else {r['candidate'].name:
            dict(r=r['r'],x=r['x'],joint_mean=r['joint_mean'],
                 joint_covariance=r['joint_covariance'],line_dimension=r['line_dimension'])
            for r in context['rows']},seconds=perf_counter()-began)
    return record,context,context_record,objective,centers


def source_hashes():
    result=old.source_hashes();here=Path(__file__).resolve().parent
    for name in ('ev_ac_repair_study.py','ev_ac_repair_refit.py',
                 'ev_ac_repair_propagation.py','ev_ac_repair_math.md'):
        path=here/name
        if path.exists():result[name]=sha256(path.read_bytes()).hexdigest()
    return result


def run(spec, *, draws=256, replicates=4, noise_repeats=1, clean_refit=True,
        progress=None):
    return _run(spec,draws=draws,replicates=replicates,
        noise_repeats=noise_repeats,clean_refit=clean_refit,progress=progress)


def _run(spec, *, draws=256, replicates=4, noise_repeats=1, clean_refit=True,
         progress=None):
    began=perf_counter();spec=deepcopy(spec)
    hash_begin=perf_counter()
    initial_sha=source_hashes()
    initial_hash_seconds=perf_counter()-hash_begin
    if int(noise_repeats)!=noise_repeats or noise_repeats<1:
        raise ValueError('noise_repeats must be a positive integer')
    callback=(lambda *args:None) if progress is None else progress
    def report(stage, details=None):
        callback(stage,dict(elapsed_seconds=perf_counter()-began,**(details or {})))
    case=cases.make_case(spec);cfg=deepcopy(case['config'])
    observed=cases.planning_inputs(case)  # Explicit boundary before any selection.
    record=dict(spec=spec,config=dict(CONFIG,draws=int(draws),replicates=int(replicates),
        noise_repeats=int(noise_repeats),clean_refit=bool(clean_refit)),
        provenance=case['provenance'],candidate_names=[c.name for c in observed['candidates']],
        candidate_structures=[dict(name=c.name,n=c.n,supports=c.supports) for c in observed['candidates']],
        observed_training=io.observations_record(observed['training']),
        observed_evidence=io.observations_record(observed['evidence']),
        train_meter=meter_to_record(observed['train_meter']),
        evidence_meter=meter_to_record(observed['evidence_meter']),
        future_meter=meter_to_record(observed['future_meter']),plans={},actions={},
        failures={},costs={},n_independent_histories=1,
        code_initial_sha=initial_sha,numeric_thread_environment={k:os.environ.get(k) for k in NUMERIC_THREADS})
    cfg['training'].update(max_nfev=CONFIG['training_max_nfev'],
                           multistart=CONFIG['training_multistart'])
    fits,training=train.train_bank(observed['candidates'],observed['training'],
        observed['train_meter'],cfg,_seed(spec['seed'],1000))
    record['training_result']=training
    if not training['all_complete']:
        record['failures']['training']=training['failures']
    report('training_complete',dict(all_complete=training['all_complete']))
    posterior_options=dict(CONFIG['posterior_options'],
        r_bounds=cfg['training']['r_bounds'],x_bounds=cfg['training']['x_bounds'],
        ratio_bounds=cfg['ratio_bounds'],**cfg['source'],seed=_seed(spec['seed'],5000))
    planning,context,context_record,objective,centers=_planning(observed,fits,
        posterior_options,spec['seed'],report)
    record.update({k:v for k,v in planning.items() if k not in ('failures','costs','seconds')})
    record['failures'].update(planning['failures']);record['costs']['planning']=planning['costs']
    record['costs']['planning_seconds']=planning['seconds']
    # Conditional re-estimation starts only after all plans have been fixed.
    D=lk.concat_observations(observed['training'],observed['evidence'])
    dm=legacy_refit.concat_models((observed['training'],observed['evidence']),
                                 (observed['train_meter'],observed['evidence_meter']))
    fit_options=dict(cfg['training'],**cfg['source'],ratio_bounds=cfg['ratio_bounds'])
    fit_options.update(multistart=CONFIG['refit_multistart'],max_nfev=CONFIG['refit_max_nfev'],
                       tol=CONFIG['refit_tol'])
    history=refit.fit_history(observed['candidates'],D,dm,fit_options,_seed(spec['seed'],6100))
    record['refit_history']=history.record();record['refit_options']=fit_options
    if not history.complete:record['failures']['conditional_history']=history.failures
    report('conditional_history_complete',dict(complete=history.complete,seconds=history.seconds))
    K=len(observed['candidates']);T=observed['horizon'];n=D.n;dimension=T*(3*n+2)
    integration_noise=_rng(spec['seed'],3000).standard_normal((1,K,
        CONFIG['independent_samples_per_candidate'],dimension))
    white=_rng(spec['seed'],4000).standard_normal((int(noise_repeats),dimension))
    record['actual_white_noise']=white
    posterior_cache={}
    for method,plan in record['plans'].items():
        key=plan['action_key']
        if key in record['actions']:
            record['actions'][key]['methods'].append(method);continue
        power=np.asarray(plan['power']);action=dict(methods=[method],power=power,
            propagation=dict(status='context_unavailable',banks={}),regimes={},failures={})
        if context is not None:
            reference=None
            try:
                reference=objective.get(power)['forecast']
                action['propagation']=dict(status='propagation_not_run',banks={'G00':reference['bank']})
                action['propagation']=propagation.forecast_moments(context,context_record,
                    power,draws=int(draws),replicates=int(replicates),
                    seed=_seed(spec['seed'],6200),posterior_cache=posterior_cache,reference=reference)
                if action['propagation']['status']!='ok':
                    action['failures']['propagation']=action['propagation']['status']
            except Exception as exc:
                action['failures']['propagation']=type(exc).__name__+': '+str(exc)
                if reference is not None:
                    action['propagation']=dict(status='propagation_exception',
                        banks={'G00':reference['bank']},failure=action['failures']['propagation'])
            action['declared_risk']={}
            action['declared_risk_seconds']={}
            for label,bank in action['propagation']['banks'].items():
                risk_started=perf_counter()
                try:
                    score=probability.evaluate_bank(bank,integration_noise,
                        abstention_cost=old.CONFIG['abstention_cost'])
                    action['declared_risk'][label]=old.score_summary(score)
                except Exception as exc:
                    action['failures']['risk_'+label]=type(exc).__name__+': '+str(exc)
                action['declared_risk_seconds'][label]=perf_counter()-risk_started
        report('propagated_'+method,dict(status=action['propagation']['status'],
            banks=list(action['propagation']['banks'])))
        # Everything below this line is generation/evaluation only.
        for ri,regime in enumerate(CONFIG['regimes']):
            replay=cases.make_case(spec,regime=regime)
            physical,truth_inputs=old._actual_physical(case,power,replay)
            clean=np.asarray(physical['mean']);values=clean[None]+white
            clean_observation=old.first_observation(physical['physical'],case['fm'],np.zeros(dimension))
            actual_observations=[old.first_observation(physical['physical'],case['fm'],w) for w in white]
            actual=dict(clean=clean,values=values,
                observations=[io.observations_record(o) for o in actual_observations],
                clean_observations=io.observations_record(clean_observation),refits=[],
                physical=dict(ac_equation_residual=float(physical['physical']['ac_equation_residual']),
                    power_balance_residual=float(physical['physical']['power_balance_residual']),
                    min_voltage=float(np.min(physical['physical']['v'])),
                    max_voltage=float(np.max(physical['physical']['v']))),
                truth_inputs_evaluation_only=truth_inputs)
            for noise_index,E in enumerate(actual_observations):
                try:
                    fitted=refit.evaluate_refits(observed['candidates'],D,dm,E,case['fm'],
                        history,fit_options,_seed(spec['seed'],7000+100*ri+noise_index),
                        bayes_centers=centers)
                    actual['refits'].append(fitted)
                    if not fitted['complete']:
                        action['failures'][regime+'_refit_'+str(noise_index)]=fitted['failures']
                except Exception as exc:
                    failure=type(exc).__name__+': '+str(exc)
                    actual['refits'].append(dict(complete=False,error=failure,keep_all_on_failure=True,
                        retained_on_failure=record['candidate_names']))
                    action['failures'][regime+'_refit_'+str(noise_index)]=failure
                report('refit_'+method+'_'+regime+'_'+str(noise_index),dict(
                    complete=actual['refits'][-1]['complete']))
            if clean_refit:
                try:
                    actual['clean_refit']=refit.evaluate_refits(observed['candidates'],D,dm,
                        clean_observation,case['fm'],history,fit_options,
                        _seed(spec['seed'],7400+ri),bayes_centers=centers)
                    if not actual['clean_refit']['complete']:
                        action['failures'][regime+'_clean_refit']=actual['clean_refit']['failures']
                except Exception as exc:
                    actual['clean_refit']=dict(complete=False,error=type(exc).__name__+': '+str(exc),
                        keep_all_on_failure=True,retained_on_failure=record['candidate_names'])
                    action['failures'][regime+'_clean_refit']=actual['clean_refit']['error']
            else:
                actual['clean_refit']=dict(status='not_run',reason='explicit budget option',
                    clean_measurement_saved=True)
            action['regimes'][regime]=actual
            report('evaluated_'+method+'_'+regime)
        record['actions'][key]=action
    record['posterior_sampling']={name:{k:v for k,v in row.items() if k!='samples'}
                                    for name,row in posterior_cache.items()}
    true_index=next(i for i,c in enumerate(case['bank'])
        if c.supports==case['truth']['candidate'].supports)
    record.update(true_name=case['bank'][true_index].name,true_index=true_index,
        true_lines_evaluation_only=dict(r=case['truth']['r'],x=case['truth']['x']))
    for key,action in record['actions'].items():
        if action['failures']:record['failures']['action_'+key]=action['failures']
    missing=sorted(set(CONFIG['methods'])-set(record['plans']))
    if missing:record['failures']['missing_plans']=missing
    if objective is not None:
        initial=record['costs']['planning'].get('objective',{})
        record['costs']['final_center_objective']=dict(
            cache_calls=objective.calls,ac_states=objective.ac_states,
            seconds=objective.seconds,
            extra_seconds_after_planning=objective.seconds-initial.get('seconds',0.),
            extra_AC_states_after_planning=objective.ac_states-initial.get('ac_states',0),
            already_in_total_elapsed=True)
    record.update(status='complete' if not record['failures'] else 'partial_failure')
    hash_end=perf_counter()
    record['code_end_sha']=source_hashes()
    end_hash_seconds=perf_counter()-hash_end
    record['source_unchanged_during_run']=record['code_end_sha']==initial_sha
    record['seconds']=perf_counter()-began
    record['costs'].update(total_seconds=record['seconds'],
        source_hash_begin_seconds=initial_hash_seconds,source_hash_end_seconds=end_hash_seconds,
        conditional_history_seconds=history.seconds,
        propagation_seconds=sum(a['propagation'].get('seconds',0.) for a in record['actions'].values()),
        propagation_AC_states=sum(a['propagation'].get('diagnostics',{}).get('AC_states',0)
                                  for a in record['actions'].values()),
        declared_risk_seconds=sum(sum(a.get('declared_risk_seconds',{}).values())
                                  for a in record['actions'].values()),
        refit_future_seconds=sum(r.get('seconds',0.)
            for a in record['actions'].values() for g in a['regimes'].values()
            for r in g['refits']+[g.get('clean_refit',{})]))
    report('history_finished',dict(status=record['status'],failures=list(record['failures'])))
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--phase',choices=tuple(CONFIG['seed_bases']),default='development')
    parser.add_argument('--index',required=True,type=int)
    parser.add_argument('--draws',default=256,type=int)
    parser.add_argument('--replicates',default=4,type=int)
    parser.add_argument('--noise-repeats',default=1,type=int)
    parser.add_argument('--skip-clean-refit',action='store_true')
    args=parser.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    specs=specifications(args.phase)
    if not 0<=args.index<len(specs):parser.error('index outside phase specifications')
    spec=specs[args.index]
    path=root/args.phase/('history_%03d.json.gz'%args.index)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('preserve existing history: '+str(path))
    progress_path=path.with_name(path.name.replace('.json.gz','.progress.json'))
    def progress(stage,details):
        io.write_json(progress_path,dict(spec=spec,stage=stage,**details))
        print('AC_REPAIR',args.phase,args.index,stage,details,flush=True)
    began=perf_counter()
    try:
        result=run(spec,draws=args.draws,replicates=args.replicates,
            noise_repeats=args.noise_repeats,clean_refit=not args.skip_clean_refit,
            progress=progress)
    except Exception as exc:
        result=dict(spec=spec,status='runner_failure',failures=dict(
            runner=type(exc).__name__+': '+str(exc)),traceback=traceback.format_exc(),
            seconds=perf_counter()-began)
    old.save_record(path,result)
    print('AC_REPAIR_SAVED',str(path),result['status'],result['seconds'],flush=True)


if __name__=='__main__':main()
