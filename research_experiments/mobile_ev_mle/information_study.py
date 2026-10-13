"""Independent-history AC evaluation of local structural information prices.

Truth/noise are evaluator inputs only. All controllers receive H, fitted H,
public measurement quality and device constraints. Old frozen modules stay
unchanged. This criterion is local projected information; final losses are
always attained complete AC joint fits over shared R/X and latent P/Q/source.
"""
from __future__ import annotations
import argparse
import base64
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
import scipy
from . import gradient_price_study as old
from .likelihood import Observations, ac_predict, concat_observations
from .simulation import generate, observe, model, BASE_KW, truth_metadata
from .candidates import propose, normalize_family
from .single_site_milp import SingleSitePlan, constant_site_plan, check_single_site_plan
from .gradient_price import GradientPriceEvaluator
from .gradient_price_search import SearchSettings, optimize_price_curves
from .information_context import (quality_models, cheap_quality_plan,
                                 meter_to_record)
from .structured_information import StructuredInformationEvaluator
from .information_targeting import history_target_weights
from .information_context_counter import build_centres
from .adaptive_statistics import binomial_exact_interval, paired_continuous_effects

HERE=Path(__file__).resolve().parent
METHODS=('none','constant_fixed','quality_global','quality_local','information_seed',
         'information_fixed','information_random','information_designed',
         'nominal_information','uniform_information','profile_designed')
QUALITY=('uniform','temporal','spatial')
PILOT_CASES=(('star',()),('moderate',((0,1),)),('strong',((0,2),)),
             ('moderate',((1,2),)),('weak',((0,1),)),('weak',((1,2),)))
DEFAULT_CONFIG=dict(version='structural_information_v2_targeted',history_points=12,future_points=4,
    energy=.15,p_charge_max=.3,p_discharge_max=0.,dt_hours=.25,gate_half_sse=6.,
    wzzt_seconds=.3,line_lognormal_sigma=.10,alpha=.1,
    information_settings=dict(max_iterations=4,max_starts=3,initial_trust_pu=.15,
        minimum_trust_pu=.003,gain_tolerance=1e-6,milp_time_limit=5.,gradient_mode='full'),
    profile_settings=dict(max_iterations=2,max_starts=3,initial_trust_pu=.15,
        minimum_trust_pu=.003,gain_tolerance=1e-6,milp_time_limit=5.,gradient_mode='full'),
    information=dict(scales_pu=[.003,.003],max_anchors=2,perturb_scale_pu=.003,
        fd_step_kw=.01,rcond=1e-10,target_weights='H_only_entropy_regularized_tasks',target_epsilon=.15,target_tau_half_sse=1.,retain_perturbed_points=True,
        anchor_aggregation='minimum',EV_increment='subtract_same_quality_no_EV'),
    design_fit_options=dict(multistart=1,max_nfev=150,tol=1e-10,linear_solver='exact'),
    evaluation_fit_options=dict(multistart=2,max_nfev=200,tol=1e-10,linear_solver='exact'),
    population=dict(quality_mass=[1/3]*3,case_mass=[.15,.15,.40,.30],
        case_names=['star','weak','moderate','strong'],cherry_families='uniform_three',
        history_innovation_amplitudes=[.004,.012],forecast='H_only_mean',
        source_interface='same common junction',root_voltage=1.,calibrated_bias=True,
        internal_r_pu=dict(weak=.00003,moderate=.003,strong=.012),
        internal_x_pu=dict(weak=.00002,moderate=.002,strong=.007)))


def configuration(config=None):
    out={**DEFAULT_CONFIG,**(config or {})}
    for key in ('information_settings','profile_settings','information','design_fit_options',
                'evaluation_fit_options','population'):
        out[key]={**DEFAULT_CONFIG[key],**((config or {}).get(key,{}) or {})}
    if out['history_points']!=12 or out['future_points']!=4:
        raise ValueError('this frozen local study requires H12/N4')
    return out


def source_hashes():
    result=old.source_hashes()
    for name in ('information_study.py','information_context.py','structured_information.py',
                 'symmetric_ac_local.py','information_targeting.py','information_context_counter.py'):
        result[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    return result


def identity(config,hashes):return old.identity(config,hashes)
serial=old.serial
write_json=old.write_json
read_json=old.read_json


def make_path(seed,config,*,case=None,truth_family=None,quality=None):
    streams=np.random.SeedSequence(int(seed)).spawn(5)
    seeds={name:int(s.generate_state(1)[0]) for name,s in zip(
        ('generator','policy','evaluation_noise','public_quality','design_integration'),streams)}
    rng=np.random.default_rng(seeds['generator']);pop=config['population']
    if case is None:case=str(rng.choice(pop['case_names'],p=pop['case_mass']))
    if case not in pop['case_names']:raise ValueError('unknown case')
    families=(((0,1),),((0,2),),((1,2),))
    if truth_family is None:truth_family=() if case=='star' else families[int(rng.integers(3))]
    path=generate(int(rng.integers(2**31)),'weak' if case=='weak' else 'strong',
        history_count=config['history_points'],future_count=config['future_points'],truth_family=truth_family)
    for i,support in enumerate(path.candidate.supports):
        if 1<len(support)<path.candidate.n:
            path.r[i]=pop['internal_r_pu'][case];path.x[i]=pop['internal_x_pu'][case]
    path.r*=rng.lognormal(0,config['line_lognormal_sigma'],len(path.r))
    path.x*=rng.lognormal(0,config['line_lognormal_sigma'],len(path.x))
    h=config['history_points'];amplitude=float(rng.choice(pop['history_innovation_amplitudes']))
    common=rng.uniform(-.012,.012,(h,1))
    path.p[:h]=.12+common+rng.uniform(-amplitude,amplitude,(h,3))
    path.q[:h]=.04+.25*common+rng.uniform(-amplitude,amplitude,(h,3))
    public_rng=np.random.default_rng(seeds['public_quality'])
    drawn=str(public_rng.choice(QUALITY,p=pop['quality_mass']))
    quality=drawn if quality is None else quality
    if quality not in QUALITY:raise ValueError('unknown quality')
    permutation=public_rng.permutation(4).tolist()
    noise_rng=np.random.default_rng(seeds['evaluation_noise'])
    white={name:noise_rng.standard_normal((4,3) if name in ('p','q','v') else (4,))
        for name in ('p','q','v','master_p','master_q')}
    return path,seeds,case,quality,permutation,white,amplitude


def future_observe(path,plan,future_model,white):
    h=path.history_count;p=path.p[h:].copy()+plan.p;q=path.q[h:].copy()+plan.q
    physical=ac_predict(path.candidate,path.r,path.x,p,q,path.source_p[h:],path.source_q[h:],root_v=1.)
    values={}
    for channel in ('p','q','v','master_p','master_q'):
        pred=np.asarray(physical[channel]);sigma=np.broadcast_to(getattr(future_model,channel+'_sigma'),pred.shape)
        bias=np.asarray(getattr(future_model,'bias_'+channel)) if channel in ('p','q','v') else 0.
        values[channel]=pred+bias+sigma*white[channel]
    return Observations(root_v=np.ones(4),**values),physical


def centres_record(centres):
    result=[]
    for c in centres:
        future=c.future
        f={name:getattr(future,name) if hasattr(future,name) else future[name]
           for name in ('p','q','source_p','source_q','root_v')}
        result.append(dict(id=c.id,anchor_id=c.anchor_id,target_support=c.target_support,
            name=c.candidate.name,supports=c.candidate.supports,r=c.r,x=c.x,
            history_p=c.history_p,history_q=c.history_q,history_source_p=c.history_source_p,
            history_source_q=c.history_source_q,history_root_v=c.history_root_v,future=f,metadata=dict(c.metadata)))
    return serial(result)


def select_controls(history,candidates,historical_fits,history_model,future_model,quality_record,*,
                    constraints,config,seed=0):
    """No truth, future physical path or evaluation noise in this interface."""
    began=perf_counter();t=config['future_points'];constant=constant_site_plan(0,t,history.n,constraints)
    zero=SingleSitePlan(0,np.zeros((t,history.n)),np.zeros((t,history.n)),np.zeros(t,int),
                        np.full(t+1,constraints.initial_soc),dict(method='no_EV'))
    choices={m:constant.get('plan') for m in METHODS};choices['none']=zero
    method_records={m:dict(status='constant_fallback') for m in METHODS}
    method_records['none']=dict(status='predeclared_no_EV')
    method_records['constant_fixed']=dict(status=constant['status'])
    ranking=sorted([c for c in candidates if c.name in historical_fits and historical_fits[c.name].success
        and np.isfinite(historical_fits[c.name].loss)],key=lambda c:(historical_fits[c.name].loss,c.name))
    ok=len(ranking)==len(candidates) and len(candidates)>=2
    gap=.5*(historical_fits[ranking[1].name].loss-historical_fits[ranking[0].name].loss) if ok else None
    activated=bool(ok and gap<=config['gate_half_sse'])
    record=dict(activated=activated,history_gap=gap,methods=method_records,searches={},
        selector_uses_truth=False,selector_uses_future_noise=False,public_quality=quality_record,
        WZZT_role='candidate_structure_only',source_is_estimated_by='anchored_AC_H_fit',
        global_design_optimality_certified=False,selector_fit_calls=0,selector_ac_calls=0,selector_ac_call_count_available=True)
    cheap={}
    for kind in ('global','local','random'):
        try:cheap[kind]=cheap_quality_plan(kind,quality_record,constraints,n_sites=history.n,
                time_limit=5.,random_seed=seed+903)
        except (ValueError,RuntimeError,FloatingPointError) as error:
            cheap[kind]=dict(success=False,error=str(error))
    record['cheap_plans']=serial(cheap)
    for kind in ('global','local'):
        method='quality_'+kind
        if cheap[kind].get('plan') is not None:
            choices[method]=cheap[kind]['plan'];method_records[method]=dict(status='optimized_public_quality',
                selected_site=int(cheap[kind]['plan'].site))
        else:method_records[method]=dict(status='selection_failure_fallback',reason='quality_MILP_failure')
    record['random_site']=None if cheap['random'].get('plan') is None else int(cheap['random']['plan'].site)
    if not activated:
        record['inactive_reason']='historical_fit_failure' if not ok else 'historical_gap_above_gate'
        record['seconds']=perf_counter()-began;return choices,record
    info=config['information']
    try:
        centres,context=build_centres(history,candidates,historical_fits,history_model,horizon=t,
            fit_options=config['evaluation_fit_options'],gate_half_sse=config['gate_half_sse'],
            max_anchors=info['max_anchors'],perturb_scale_pu=info['perturb_scale_pu'],seed=seed+400,
            retain_perturbed_points=info.get('retain_perturbed_points',False))
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
        centres=[];context=dict(activated=False,reason='context_failure',error=str(error),
            fit_calls=int(getattr(error,'fit_calls',0)),ac_calls=int(getattr(error,'ac_calls',0)),
            seconds=float(getattr(error,'seconds',0.)),call_count_unavailable=not bool(getattr(error,'call_count_available',False)),
            failed_context_record=error.record() if hasattr(error,'record') else None)
        for m in METHODS[4:10]:method_records[m]=dict(status='selection_failure_fallback',reason=str(error))
    record['context']=context;record['centres']=centres_record(centres)
    record['selector_ac_call_count_available']=not context.get('call_count_unavailable',False)
    record['selector_fit_calls']+=int(context.get('fit_calls',0));record['selector_ac_calls']+=int(context.get('ac_calls',0))
    settings=SearchSettings(**config['information_settings'])
    target_record=history_target_weights(candidates,historical_fits,epsilon=info['target_epsilon'],tau=info['target_tau_half_sse'])
    record['target_weight_record']=serial({k:v for k,v in target_record.items() if k!='weights'})
    target_weights=target_record.get('weights')
    if centres and target_record['success']:
        first_anchor=centres[0].anchor_id
        runs=(('seeds',centres,SearchSettings(**{**asdict(settings),'max_iterations':0})),
              ('robust',centres,settings),('nominal',[c for c in centres if c.anchor_id==first_anchor],settings),
              ('uniform',centres,settings))
        for tag,selected_centres,st in runs:
            evaluator=None
            initialization_counted=False
            try:
                evaluator=StructuredInformationEvaluator(history,selected_centres,history_model,future_model=future_model,
                    base_kva=BASE_KW,q_over_p=constraints.q_over_p,
                    weight=.5*np.diag(np.asarray(info['scales_pu'],float)**2),
                    fd_step_kw=info['fd_step_kw'],rcond=info['rcond'],
                    target_weights=None if tag=='uniform' else target_weights)
                record['selector_ac_calls']+=evaluator.initialization_ac_calls
                initialization_counted=True
                result=optimize_price_curves(evaluator,constraints,sites=range(history.n),settings=st)
                result['initialization_ac_calls']=evaluator.initialization_ac_calls
                result['initialization_seconds']=evaluator.initialization_seconds
                record['searches'][tag]=serial(result)
                record['searches'][tag].update(criterion='local_structural_EV_information_increment',
                    joint_competitor_refit_in_selection=False,nuisance='all_shared_lines_and_H/N_latent_states_projected_locally')
                record['selector_ac_calls']+=int(result.get('total_ac_calls',0))
                plans=result.get('site_plans',{})
                if tag=='robust':targets={'information_fixed':0,'information_random':record['random_site'],
                                        'information_designed':result.get('selected_site')}
                else:targets={('information_seed' if tag=='seeds' else 'nominal_information' if tag=='nominal' else 'uniform_information'):result.get('selected_site')}
                for method,site in targets.items():
                    plan=plans.get(site)
                    if plan is not None:
                        choices[method]=plan;method_records[method]=dict(status='optimized',selected_site=int(site),
                            predicted_score=result['site_evaluations'][site]['score'])
                    else:method_records[method]=dict(status='selection_failure_fallback',reason='no_valid_information_plan')
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                spent=(max(0,int(evaluator._ac_calls)-(int(evaluator.initialization_ac_calls) if initialization_counted else 0))
                    if evaluator is not None else int(getattr(error,'initialization_ac_calls',0)))
                record['selector_ac_calls']+=spent
                record['searches'][tag]=dict(success=False,error=type(error).__name__+':'+str(error),
                    failed_initialization_record=error.record() if hasattr(error,'record') else None,
                    partial_ac_calls_retained=spent)
                methods=('information_fixed','information_random','information_designed') if tag=='robust' else (
                    'information_seed' if tag=='seeds' else 'nominal_information' if tag=='nominal' else 'uniform_information',)
                for method in methods:method_records[method]=dict(status='selection_failure_fallback',reason=str(error))
    else:
        record['information_not_started_reason']=context.get('reason','no_history_supported_star_anchor') if not centres else 'required_target_weight_failure'
        if centres and not target_record['success']:
            for method in METHODS[4:10]:method_records[method]=dict(status='selection_failure_fallback',reason='required_target_weight_failure')
    profile_evaluator=None
    try:
        supported=[c for c in ranking if .5*(historical_fits[c.name].loss-historical_fits[ranking[0].name].loss)<=config['gate_half_sse']]
        if len(supported)>=2:
            profile_evaluator=GradientPriceEvaluator(history,supported,historical_fits,history_model,horizon=t,
                future_model=future_model,fit_options=config['design_fit_options'],seed=seed+700)
            result=optimize_price_curves(profile_evaluator,constraints,settings=SearchSettings(**config['profile_settings']))
            record['searches']['profile']=serial(result)
            record['searches']['profile']['comparison_scope']='old complete method; reference forecasts and iteration budget differ from information policy'
            record['selector_fit_calls']+=result.get('fit_calls',0)
            record['selector_ac_calls']+=result.get('total_ac_calls',0)
            if result.get('plan') is not None:
                choices['profile_designed']=result['plan'];method_records['profile_designed']=dict(status='optimized',
                    selected_site=result['selected_site'],predicted_score=result['score'])
            else:method_records['profile_designed']=dict(status='selection_failure_fallback',reason='profile_search_failed')
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
        partial_ac=0 if profile_evaluator is None else int(profile_evaluator.ac_evaluations+profile_evaluator.response_ac_calls)
        partial_fit=0 if profile_evaluator is None else int(profile_evaluator.fit_calls)
        record['selector_ac_calls']+=partial_ac;record['selector_fit_calls']+=partial_fit
        record['searches']['profile']=dict(success=False,error=str(error),partial_ac_calls_retained=partial_ac,partial_fit_calls_retained=partial_fit)
        method_records['profile_designed']=dict(status='selection_failure_fallback',reason=str(error))
    record['seconds']=perf_counter()-began
    return choices,record


def run_trial(seed,*,case=None,truth_family=None,quality=None,config=None):
    began=perf_counter();config=configuration(config);hashes=source_hashes();pid=identity(config,hashes)
    path,seeds,label,quality,permutation,white,amplitude=make_path(seed,config,case=case,truth_family=truth_family,quality=quality)
    h=config['history_points'];history,_=observe(path,slice(0,h));historical_model=model(3)
    future_model,joint_model,quality_record=quality_models(history,historical_model,kind=quality,row_permutation=permutation)
    candidates,proposal=propose(history,run_wzzt=True,wzzt_time_limit=config['wzzt_seconds'])
    fits,hfail,hcost=old.fit_all_candidates(candidates,history,historical_model,seed=seeds['policy']+10,
        fit_options=config['evaluation_fit_options'])
    constraints=old.device_constraints(config)
    choices,selection=select_controls(history,candidates,fits,historical_model,future_model,quality_record,
        constraints=constraints,config=config,seed=seeds['policy'])
    true_family=normalize_family(path.candidate.supports,3)
    true_name=next((c.name for c in candidates if normalize_family(c.supports,3)==true_family),None)
    initial={name:f for name,f in fits.items() if f.success};records=[];complete=not hfail and len(initial)==len(candidates)
    joint_cache={};executed_ac_calls=executed_fit_calls=0
    for method in METHODS:
        start=perf_counter();plan=choices[method]
        if plan is None:
            records.append(dict(method=method,status='operational_failure',fits=[],losses={},true_name=true_name,
                true_covered=true_name is not None,seconds=perf_counter()-start));complete=False;continue
        validation=dict(feasible=True,energy_pu_hours=0.) if method=='none' else check_single_site_plan(plan,constraints)
        if not validation['feasible']:
            records.append(dict(method=method,status='operational_failure',action=plan.summary(),validation=validation,
                fits=[],losses={},true_name=true_name,true_covered=true_name is not None,seconds=perf_counter()-start));complete=False;continue
        try:future,physical=future_observe(path,plan,future_model,white)
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            records.append(dict(method=method,status='physical_failure',action=plan.summary(),validation=validation,
                error=str(error),fits=[],losses={},true_name=true_name,true_covered=true_name is not None,
                seconds=perf_counter()-start));complete=False;continue
        if not (abs(physical['voltage']).min()>=.9 and abs(physical['voltage']).max()<=1.1):
            records.append(dict(method=method,status='physical_failure',action=plan.summary(),validation=validation,
                error='actual AC voltage outside operational domain',fits=[],losses={},true_name=true_name,
                true_covered=true_name is not None,voltage_min=float(abs(physical['voltage']).min()),
                voltage_max=float(abs(physical['voltage']).max()),operational_voltage_ok=False,seconds=perf_counter()-start))
            complete=False;continue
        joint=concat_observations(history,future);key=(plan.p.tobytes(),plan.q.tobytes());cached=key in joint_cache
        if not cached:
            joint_cache[key]=old.fit_all_candidates(candidates,joint,joint_model,initial=initial,
                seed=seeds['policy']+200,fit_options=config['evaluation_fit_options'])
        jf,failed,cost=joint_cache[key];actual_ac=0 if cached else cost['ac_calls'];actual_fit=0 if cached else cost['fit_calls']
        executed_ac_calls+=actual_ac;executed_fit_calls+=actual_fit
        losses={c.name:float(jf[c.name].loss) for c in candidates if c.name in jf and jf[c.name].success and np.isfinite(jf[c.name].loss)}
        good=len(losses)==len(candidates);complete &= good
        row=dict(method=method,status='complete' if good else 'joint_fit_failure',action=plan.summary(),validation=validation,
            selection=selection['methods'][method],fits=[old.fit_record(c,jf[c.name]) for c in candidates if c.name in jf],
            losses=losses,fit_failures=failed,ac_calls=actual_ac,fit_calls=actual_fit,
            equivalent_action_fit_cache_hit=cached,fit_seconds=0. if cached else cost['seconds'],
            observations=old.observations_record(future),true_covered=true_name is not None,true_name=true_name,
            operational_voltage_ok=bool(abs(physical['voltage']).min()>=.9 and abs(physical['voltage']).max()<=1.1),
            voltage_min=float(abs(physical['voltage']).min()),voltage_max=float(abs(physical['voltage']).max()))
        if good:
            best=min(losses.values());row.update(winner=min(losses,key=lambda n:(losses[n],n)),
                gaps={n:v-best for n,v in losses.items()})
            if true_name is not None:
                f=jf[true_name];wrong=[v for n,v in losses.items() if n!=true_name]
                row.update(true_loss=losses[true_name],true_gap=losses[true_name]-best,
                    signed_margin_full_SSE=min(wrong)-losses[true_name],wrong_winner=row['winner']!=true_name,
                    impedance_absolute_rmse_pu=float(np.sqrt(np.mean(np.r_[(f.r-path.r)**2,(f.x-path.x)**2]))),
                    impedance_relative_rmse=float(np.sqrt(np.mean(np.r_[((f.r-path.r)/path.r)**2,((f.x-path.x)/path.x)**2]))))
        row['seconds']=perf_counter()-start;records.append(row)
    complete=bool(complete and true_name is not None)
    result=dict(history_id=str(seed),run_id=str(seed)+'_'+quality,seed=int(seed),case=label,quality=quality,
        history_innovation_amplitude=amplitude,seeds=seeds,config=config,policy_id=pid,source_sha256=hashes,
        measurement_models=dict(history=meter_to_record(historical_model),future=meter_to_record(future_model),joint=meter_to_record(joint_model)),
        quality_record=quality_record,history_observations=old.observations_record(history),
        history_hash=old.fingerprint(history.p,history.q,history.v,history.master_p,history.master_q),
        proposal=proposal,candidate_names=[c.name for c in candidates],
        historical_fits=[old.fit_record(c,fits[c.name]) for c in candidates if c.name in fits],historical_failures=hfail,
        historical_cost=hcost,device_constraints=asdict(constraints),selection=selection,records=records,
        true_covered=true_name is not None,true_name=true_name,truth=truth_metadata(path),complete=complete,
        relative_score=max(row['true_gap'] for row in records) if complete else np.inf,
        calibration_score_is_infinite=not complete,independent_history_unit=True,selection_uses_truth=False,
        selection_uses_future_noise=False,white_evaluation_noise=white,seconds=perf_counter()-began,
        total_ac_calls=hcost['ac_calls']+selection['selector_ac_calls']+executed_ac_calls,
        total_ac_call_count_available=selection.get('selector_ac_call_count_available',True),
        joint_unique_action_count=len(joint_cache),actual_joint_fit_calls=executed_fit_calls,
        ac_call_definition='attempted algorithm calls including failed context entries; unavailable flag denotes a lower bound; equal actions cache exact joint fits',
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    return serial(result)


def summarize(trials,calibration=None):
    if calibration is not None:
        for t in trials:old.apply_trial_gate(t,calibration)
    by_method={}
    for method in METHODS:
        rows=[next(r for r in t['records'] if r['method']==method) for t in trials]
        good=[r for r in rows if r['status']=='complete'];cov=[r for r in rows if r.get('true_covered')]
        by_method[method]=dict(all_history_count=len(rows),complete_count=len(good),failed_count=len(rows)-len(good),
            only_truth_retained=sum(r.get('only_truth_retained',False) for r in rows) if calibration else None,
            unresolved=sum(r.get('unresolved',False) for r in rows) if calibration else None,
            false_rejections=sum(r.get('false_rejection',False) for r in cov) if calibration else None,
            false_rejection_denominator=len(cov) if calibration else None,
            signed_margin_mean=float(np.mean([r['signed_margin_full_SSE'] for r in good])) if good else None,
            signed_margin_median=float(np.median([r['signed_margin_full_SSE'] for r in good])) if good else None,
            mean_seconds=float(np.mean([r['seconds'] for r in rows])),actual_ac_calls=sum(r.get('ac_calls',0) for r in rows),
            selection_failures=sum(t['selection']['methods'][method]['status']=='selection_failure_fallback' for t in trials))
    effects=[]
    for method,baseline in (('information_designed','constant_fixed'),('information_designed','information_seed'),
        ('information_designed','quality_global'),('information_designed','quality_local'),
        ('information_designed','profile_designed'),('information_designed','nominal_information'),
        ('information_designed','uniform_information')):
        valid=[];units_for_pairs={};wins=losses=ties=0
        for t in trials:
            rows={r['method']:r for r in t['records']}
            if all(rows[m]['status']=='complete' and 'signed_margin_full_SSE' in rows[m] for m in (method,baseline)):
                pair=(rows[method]['signed_margin_full_SSE'],rows[baseline]['signed_margin_full_SSE'])
                valid.append((t['run_id'],*pair));units_for_pairs.setdefault(t['history_id'],[]).append(pair)
                if calibration:
                    a,b=rows[method]['only_truth_retained'],rows[baseline]['only_truth_retained']
                    wins+=bool(a and not b);losses+=bool(b and not a);ties+=a==b
        independent=[(hid,float(np.mean([v[0] for v in pairs])),float(np.mean([v[1] for v in pairs])))
            for hid,pairs in units_for_pairs.items()]
        effects.append(dict(method=method,baseline=baseline,paired_condition_count=len(valid),
            paired_count=len(independent),aggregation='average available paired conditions within each independent history',unique_correct_discordant_wins=wins,
            unique_correct_discordant_losses=losses,unique_correct_ties=ties,
            signed_margin_effect=paired_continuous_effects([x[0] for x in independent],[x[1] for x in independent],[x[2] for x in independent]) if independent else None))
    units=len(set(t['history_id'] for t in trials));policy_reject=sum(t.get('any_method_false_rejection',False) for t in trials)
    return dict(run_count=len(trials),independent_history_count=units,complete_count=sum(t['complete'] for t in trials),
        failure_count=sum(not t['complete'] for t in trials),candidate_coverage_failures=sum(not t['true_covered'] for t in trials),
        any_method_false_rejections=policy_reject if calibration else None,
        policy_false_rejection_interval=binomial_exact_interval(policy_reject,len(trials)) if calibration and trials else None,
        statistical_note='pilot quality repeats are paired conditions of the same H; formal one quality per independent H',
        method_summary=by_method,paired_effects=effects,calibration=calibration,
        total_seconds=sum(t['seconds'] for t in trials),total_ac_calls=sum(t['total_ac_calls'] for t in trials),
        histories_with_unavailable_AC_counts=sum(not t.get('total_ac_call_count_available',True) for t in trials),
        max_selection_seconds=max((t['selection']['seconds'] for t in trials),default=0.),
        case_counts={c:sum(t['case']==c for t in trials) for c in DEFAULT_CONFIG['population']['case_names']},
        quality_counts={q:sum(t['quality']==q for t in trials) for q in QUALITY})


def save_source_snapshot(output,hashes):
    files={name:base64.b64encode((HERE/name).read_bytes()).decode('ascii') for name in hashes}
    write_json(Path(output)/'science_source_snapshot.json.gz',dict(schema='gradient_price_source_bytes_v1',
        source_sha256=hashes,files=files),compressed=True)


def run_batch(phase,output,*,count=None,start_seed=None,config=None):
    config=configuration(config);output=Path(output);directory=output/phase;directory.mkdir(parents=True,exist_ok=True)
    if phase not in ('pilot','calibration','evaluation'):raise ValueError('unknown phase')
    count=int((6 if phase=='pilot' else 39 if phase=='calibration' else 80) if count is None else count)
    if count<1 or phase=='pilot' and count>len(PILOT_CASES):raise ValueError('invalid count')
    start_seed=int((930001 if phase=='pilot' else 940001 if phase=='calibration' else 950001) if start_seed is None else start_seed)
    hashes=source_hashes();pid=identity(config,hashes)
    batch=dict(phase=phase,independent_history_count=count,run_count=count*3 if phase=='pilot' else count,
        seeds=list(range(start_seed,start_seed+count)),config=config,source_sha256=hashes,policy_id=pid,
        pilot_cases=PILOT_CASES[:count] if phase=='pilot' else None,quality_repeats=list(QUALITY) if phase=='pilot' else None)
    if (directory/'config.json').exists() and read_json(directory/'config.json')!=serial(batch):
        raise ValueError('saved phase config/source changed; use a fresh output')
    frozen=dict(config=config,source_sha256=hashes,policy_id=pid)
    if phase!='pilot':
        if (output/'freeze.json').exists() and read_json(output/'freeze.json')!=serial(frozen):raise ValueError('formal freeze differs')
        if phase=='evaluation' and not (output/'freeze.json').exists():raise ValueError('calibration freeze required')
        if not (output/'freeze.json').exists():write_json(output/'freeze.json',frozen);save_source_snapshot(output,hashes)
    calibration=read_json(output/'calibration/thresholds.json') if phase=='evaluation' else None
    if calibration and set(map(str,batch['seeds']))&set(calibration['calibration_history_ids']):raise ValueError('calibration/evaluation overlap')
    write_json(directory/'config.json',batch);trials=[]
    for i,seed in enumerate(batch['seeds']):
        specs=[(*PILOT_CASES[i],q) for q in QUALITY] if phase=='pilot' else [(None,None,None)]
        for case,family,quality in specs:
            suffix='_'+quality if quality else '';p=directory/f'trial_{seed}{suffix}.json.gz'
            trial=read_json(p) if p.exists() else run_trial(seed,case=case,truth_family=family,quality=quality,config=config)
            if trial['policy_id']!=pid:raise ValueError('saved trial belongs to another source/policy')
            if calibration:old.apply_trial_gate(trial,calibration)
            if not p.exists():write_json(p,trial,compressed=True)
            trials.append(trial);write_json(directory/'progress.json',dict(completed_runs=len(trials),planned_runs=batch['run_count'],
                independent_histories_completed=len({t['history_id'] for t in trials}),last_seed=seed,policy_id=pid))
            print(json.dumps(dict(phase=phase,index=len(trials),planned=batch['run_count'],seed=seed,quality=trial['quality'],
                case=trial['case'],seconds=trial['seconds'],selection_seconds=trial['selection']['seconds'],complete=trial['complete']),ensure_ascii=False),flush=True)
    if phase=='calibration':
        calibration=old.calibration_result(trials,alpha=config['alpha']);write_json(directory/'thresholds.json',calibration)
    summary=summarize(trials,calibration if phase=='evaluation' else None);write_json(directory/'summary.json',summary)
    return trials,summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('pilot','calibration','evaluation'),default='pilot')
    p.add_argument('--output',type=Path,default=HERE/'evidence/structured_information/v2_targeted');p.add_argument('--count',type=int)
    p.add_argument('--start-seed',type=int);p.add_argument('--config',type=Path);a=p.parse_args()
    run_batch(a.phase,a.output,count=a.count,start_seed=a.start_seed,config=None if a.config is None else read_json(a.config))


if __name__=='__main__':main()