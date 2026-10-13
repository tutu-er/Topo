"""Independent-history evaluation of single-site EV gradient-price policies.

Simulation truth and future observation noise live only in run_trial. The
controller receives historical meters, their fits, and public device limits.
Frozen calibration/evaluation use the same source and policy hashes.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from . import likelihood
from .likelihood import concat_observations
from .simulation import generate, observe, model, BASE_KW, ZBASE_OHM, truth_metadata
from .candidates import propose, normalize_family
from .single_site_milp import SingleSiteConstraints, SingleSitePlan, constant_site_plan, check_single_site_plan, solve_site_step
from .gradient_price import GradientPriceEvaluator
from .gradient_price_search import SearchSettings, optimize_price_curves
from .adaptive_statistics import conformal_upper_quantile, binomial_exact_interval, paired_continuous_effects
from .run_study import fingerprint

HERE=Path(__file__).resolve().parent
METHODS=('none','constant_fixed','optimized_fixed','optimized_random','optimized_designed','pq_only_designed')
DEFAULT_CONFIG=dict(version='single_site_gradient_price_v2_supported',history_points=12,future_points=4,
    energy=.15,p_charge_max=.3,p_discharge_max=0.,dt_hours=.25,gate_half_sse=6.,
    wzzt_seconds=.3,line_lognormal_sigma=.10,alpha=.1,reference_policy='all_within_gate',
    settings=dict(max_iterations=2,max_starts=3,initial_trust_pu=.15,minimum_trust_pu=.003,
                  gain_tolerance=1e-6,milp_time_limit=5.,gradient_mode='full'),
    design_fit_options=dict(multistart=1,max_nfev=150,tol=1e-10,linear_solver='exact'),
    evaluation_fit_options=dict(multistart=2,max_nfev=200,tol=1e-10,linear_solver='exact'),
    population=dict(topology_mass=[.25]*4,cherry_strong_probability=.5,forecast='history_mean',
                    source_interface='same common junction',root_voltage=1.,calibrated_bias=True))
PILOT_CASES=(('star',()),('strong',((0,1),)),('strong',((0,2),)),
             ('strong',((1,2),)),('weak',((0,1),)),('weak',((1,2),)))


def configuration(config=None):
    out={**DEFAULT_CONFIG,**(config or {})}
    for key in ('settings','design_fit_options','evaluation_fit_options','population'):
        out[key]={**DEFAULT_CONFIG[key],**((config or {}).get(key,{}) or {})}
    return out


def serial(value):
    if isinstance(value,SingleSitePlan):return serial(value.summary())
    if isinstance(value,np.ndarray):return serial(value.tolist())
    if isinstance(value,np.generic):return serial(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items() if k!='fit_results'}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    return value


def write_json(path,value,*,compressed=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(serial(value),ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode('utf-8')
    if compressed:data=gzip.compress(data,mtime=0)
    path.write_bytes(data)


def read_json(path):
    path=Path(path);data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def source_hashes():
    names=('gradient_price_study.py','gradient_price_search.py','gradient_price.py','single_site_milp.py',
           'likelihood.py','simulation.py','candidates.py','design.py','adaptive_statistics.py','run_study.py')
    hashes={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names}
    for n in ('passive_bridge_v9.py','wzzt_adapter.py','ac_general_v9.py'):
        hashes['../mobile_ev_probing/'+n]=hashlib.sha256((HERE.parent/'mobile_ev_probing'/n).read_bytes()).hexdigest()
    return hashes


def identity(config,hashes):
    return hashlib.sha256(json.dumps(serial(dict(config=config,source_sha256=hashes)),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def observations_record(obs):
    return {key:getattr(obs,key) for key in ('p','q','v','root_v','master_p','master_q','source_p','source_q')}


def fit_record(candidate,fit):
    return dict(name=candidate.name,supports=[list(s) for s in candidate.supports],
        **fit.summary(),half_loss=.5*float(fit.loss),r_ohm=fit.r*ZBASE_OHM,x_ohm=fit.x*ZBASE_OHM,
        p=fit.p,q=fit.q,source_p=fit.source_p,source_q=fit.source_q,
        bias_p=fit.bias_p,bias_q=fit.bias_q,bias_v=fit.bias_v,
        predicted={key:fit.predicted[key] for key in ('p','q','v','master_p','master_q') if key in fit.predicted},
        global_MLE_certified=False)


def fit_all_candidates(candidates,observations,meter,*,initial=None,seed=0,fit_options=None):
    """Serial counted fits; an exception cannot remove already spent AC calls."""
    options=dict(fit_options or {})
    if 'initial' in options or 'seed' in options:raise ValueError('initial/seed supplied by driver')
    fits,failures={},{};calls=[0];original=likelihood.ac_predict;began=perf_counter()
    def counted(*a,**k):calls[0]+=1;return original(*a,**k)
    likelihood.ac_predict=counted
    try:
        for i,candidate in enumerate(candidates):
            try:
                fitted=likelihood.fit_candidate(candidate,observations,meter,
                    initial=(initial or {}).get(candidate.name),seed=int(seed)+i,**options)
                fits[candidate.name]=fitted
                if not fitted.success or not np.isfinite(fitted.loss):failures[candidate.name]='fit_nonconverged'
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                failures[candidate.name]=type(error).__name__+':'+str(error)
    finally:likelihood.ac_predict=original
    return fits,failures,dict(ac_calls=calls[0],fit_calls=len(candidates),seconds=perf_counter()-began)


def device_constraints(config,n=3):
    t=config['future_points'];access=np.ones((t,n),bool);access[:min(2,t),2]=False
    return SingleSiteConstraints(dt_hours=config['dt_hours'],energy=config['energy'],
        p_charge_max=config['p_charge_max'],p_discharge_max=config['p_discharge_max'],
        availability=access,travel_slots=np.ones((n,n),int)-np.eye(n,dtype=int),
        travel_energy=.01*(np.ones((n,n))-np.eye(n)),initial_site=0)


def _search_site_plans(result):
    if 'site_plans' in result:return result['site_plans']
    # Compatibility with the initial search revision; these are optimized
    # site plans already obtained by search, not another search or action library.
    plans={}
    for row in result.get('site_records',[]):
        if 'selected_plan' in row:
            p=row['selected_plan'];plans[int(row['site'])]=SingleSitePlan(p['site'],np.asarray(p['p']),np.asarray(p['q']),np.asarray(p['sites']),np.asarray(p['soc']),p.get('solver',{}))
    return plans


def select_controls(history,candidates,historical_fits,meter,*,constraints,horizon,
                    settings=None,fit_options=None,gate_half_sse=6.,reference_policy='all_within_gate',seed=0):
    """H-only selection API: no simulation path, truth, or future noise input."""
    started=perf_counter();settings=settings or SearchSettings(max_iterations=2,max_starts=3)
    if reference_policy not in ('all_within_gate','two_closest'):
        raise ValueError('reference_policy must be all_within_gate or two_closest')
    if not np.isfinite(gate_half_sse) or gate_half_sse < 0:
        raise ValueError('gate_half_sse must be finite and nonnegative')
    constant=constant_site_plan(0,horizon,history.n,constraints)
    baseline=constant.get('plan');zero=SingleSitePlan(0,np.zeros((horizon,history.n)),np.zeros((horizon,history.n)),
        np.zeros(horizon,int),np.full(horizon+1,constraints.initial_soc),dict(method='no_EV'))
    choices={name:baseline for name in METHODS};choices['none']=zero
    records={name:dict(status='constant_fallback') for name in METHODS}
    records['none']=dict(status='predeclared_no_EV')
    records['constant_fixed']=dict(status=constant['status'],validation=constant.get('validation'))
    ranking=sorted((c for c in candidates if c.name in historical_fits and historical_fits[c.name].success
                    and np.isfinite(historical_fits[c.name].loss)),key=lambda c:(historical_fits[c.name].loss,c.name))
    history_ok=len(ranking)==len(candidates) and len(candidates)>=2
    gap=.5*(historical_fits[ranking[1].name].loss-historical_fits[ranking[0].name].loss) if history_ok else None
    activated=bool(history_ok and gap <= gate_half_sse)
    best_loss=historical_fits[ranking[0].name].loss if ranking else None
    successful_names={c.name for c in ranking}
    historical_gaps={c.name:(.5*(historical_fits[c.name].loss-best_loss) if c.name in successful_names else None) for c in candidates}
    supported=[c for c in ranking if historical_gaps[c.name] <= gate_half_sse]
    design=ranking[:2] if reference_policy=='two_closest' else supported
    record=dict(activated=activated,gate_half_sse=gate_half_sse,history_gap=gap,
        gate_reason='ambiguous_history' if activated else 'history_fit_failure' if not history_ok else 'historical_gap_above_gate',
        design_candidates=[c.name for c in design],supported_count=len(supported),reference_policy=reference_policy,
        all_candidate_historical_gaps=historical_gaps,methods=records,
        selector_uses_truth=False,selector_uses_future_noise=False,searches={},
        source_is_estimated_by='anchored_AC_fit',WZZT_role='candidate_structure_only',
        global_design_optimality_certified=False,selector_fit_calls=0,selector_ac_calls=0)
    if not constant['success']:
        record['constant_failure']=constant['termination']
    if not activated:
        for method in METHODS[2:]:records[method]=dict(status='not_activated_constant',reason=record['gate_reason'])
        record['seconds']=perf_counter()-started
        return choices,record
    public_constants={site:constant_site_plan(site,horizon,history.n,constraints) for site in range(history.n)}
    public_probes={site:solve_site_step(site,horizon,history.n,constraints,intercepts=[0.],
        gradients=np.zeros((1,horizon)),time_limit=settings.milp_time_limit) for site in range(history.n)}
    feasible_sites=[site for site,out in public_probes.items() if out['success']]
    record['public_operation_probes']=serial(public_probes)
    random_site=int(np.random.default_rng(seed+901).choice(feasible_sites)) if feasible_sites else None
    record['random_site']=random_site;record['public_feasible_sites']=feasible_sites
    for mode in ('full','pq_only'):
        try:
            evaluator=GradientPriceEvaluator(history,design,historical_fits,meter,horizon=horizon,
                fit_options=fit_options,q_over_p=constraints.q_over_p,seed=seed+100,base_kva=BASE_KW)
            search_settings=SearchSettings(**{**asdict(settings),'gradient_mode':mode})
            result=optimize_price_curves(evaluator,constraints,sites=range(history.n),settings=search_settings)
            record['searches'][mode]=serial(result)
            record['selector_fit_calls']+=int(result.get('fit_calls',0))
            record['selector_ac_calls']+=int(result.get('total_ac_calls',0))
            plans=_search_site_plans(result)
            targets={'pq_only_designed':result.get('selected_site')} if mode=='pq_only' else {
                'optimized_fixed':0,'optimized_random':random_site,'optimized_designed':result.get('selected_site')}
            for method,site in targets.items():
                plan=plans.get(site) if site is not None else None
                if plan is not None:
                    choices[method]=plan;records[method]=dict(status='optimized',selected_site=int(site))
                    evaluations=result.get('site_evaluations',{})
                    evaluation=evaluations.get(site)
                    if evaluation is not None:records[method]['predicted_score']=evaluation.get('score')
                else:
                    fallback=public_probes.get(site,constant)
                    choices[method]=fallback.get('plan')
                    records[method]=dict(status='selection_failure_fallback',selected_site=site,reason='no_valid_site_search_plan')
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            record['searches'][mode]=dict(success=False,error=type(error).__name__+':'+str(error))
            targets=('pq_only_designed',) if mode=='pq_only' else ('optimized_fixed','optimized_random','optimized_designed')
            for method in targets:records[method]=dict(status='selection_failure_fallback',reason=str(error))
    record['seconds']=perf_counter()-started
    return choices,record


def _make_path(seed,config,*,case=None,truth_family=None):
    streams=np.random.SeedSequence(int(seed)).spawn(3)
    seeds={name:int(stream.generate_state(1)[0]) for name,stream in zip(('generator','policy','evaluation_noise'),streams)}
    rng=np.random.default_rng(seeds['generator'])
    families=((),((0,1),),((0,2),),((1,2),))
    if truth_family is None:
        if case=='star':truth_family=()
        elif case in ('strong','weak'):truth_family=families[int(rng.integers(1,4))]
        else:truth_family=families[int(rng.choice(4,p=config['population']['topology_mass']))]
    actual_case=case if case in ('strong','weak') else ('strong' if rng.random()<config['population']['cherry_strong_probability'] else 'weak')
    path=generate(int(rng.integers(2**31)),actual_case,history_count=config['history_points'],future_count=config['future_points'],truth_family=truth_family)
    path.r*=rng.lognormal(0,config['line_lognormal_sigma'],len(path.r))
    path.x*=rng.lognormal(0,config['line_lognormal_sigma'],len(path.x))
    noise_rng=np.random.default_rng(seeds['evaluation_noise']);h=path.history_count;t=config['future_points']
    for name,sigma in (('p',.0006),('q',.0004),('v',.00012),('master_p',.001),('master_q',.001)):
        shape=(t,path.candidate.n) if name in ('p','q','v') else (t,)
        path.noise[name][h:]=noise_rng.normal(0,sigma,shape)
    label='star' if not truth_family else actual_case
    return path,seeds,label


def run_trial(seed,*,case=None,truth_family=None,config=None,output=None):
    started=perf_counter();config=configuration(config);hashes=source_hashes();pid=identity(config,hashes)
    path,seeds,label=_make_path(seed,config,case=case,truth_family=truth_family)
    h=config['history_points'];history,_=observe(path,slice(0,h));meter=model(3)
    candidates,proposal=propose(history,run_wzzt=True,wzzt_time_limit=config['wzzt_seconds'])
    fits,hfail,hcost=fit_all_candidates(candidates,history,meter,seed=seeds['policy']+10,
        fit_options=config['evaluation_fit_options'])
    constraints=device_constraints(config)
    choices,selection=select_controls(history,candidates,fits,meter,constraints=constraints,
        horizon=config['future_points'],settings=SearchSettings(**config['settings']),
        fit_options=config['design_fit_options'],gate_half_sse=config['gate_half_sse'],
        reference_policy=config['reference_policy'],seed=seeds['policy'])
    # Truth is inspected only after the controller has returned its actions.
    true_family=normalize_family(path.candidate.supports,3)
    true_name=next((c.name for c in candidates if normalize_family(c.supports,3)==true_family),None)
    initial={name:f for name,f in fits.items() if f.success};records=[]
    complete=bool(not hfail and len(initial)==len(candidates))
    for method in METHODS:
        began=perf_counter();plan=choices[method]
        if plan is None:
            records.append(dict(method=method,status='operational_failure',fits=[],losses={},ac_calls=0,fit_calls=0,
                true_covered=true_name is not None,seconds=perf_counter()-began));complete=False;continue
        from .design import EVAction
        action=EVAction(method,plan.p,plan.q,plan.sites,'single_site_gradient_price')
        check=dict(feasible=True,energy_pu_hours=0.) if method=='none' else check_single_site_plan(plan,constraints)
        if not check['feasible']:
            records.append(dict(method=method,status='operational_failure',action=plan.summary(),validation=check,
                fits=[],losses={},ac_calls=0,fit_calls=0,true_covered=true_name is not None,seconds=perf_counter()-began));complete=False;continue
        try:
            future,physical=observe(path,slice(h,None),action)
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            records.append(dict(method=method,status='physical_failure',action=plan.summary(),validation=check,
                error=type(error).__name__+':'+str(error),fits=[],losses={},ac_calls=0,fit_calls=0,
                true_covered=true_name is not None,true_name=true_name,seconds=perf_counter()-began))
            complete=False;continue
        joint=concat_observations(history,future)
        jf,failed,cost=fit_all_candidates(candidates,joint,meter,initial=initial,seed=seeds['policy']+200,
            fit_options=config['evaluation_fit_options'])
        losses={c.name:float(jf[c.name].loss) for c in candidates if c.name in jf and jf[c.name].success and np.isfinite(jf[c.name].loss)}
        good=len(losses)==len(candidates);complete &= good
        row=dict(method=method,status='complete' if good else 'joint_fit_failure',action=plan.summary(),validation=check,
            selection=selection['methods'][method],fits=[fit_record(c,jf[c.name]) for c in candidates if c.name in jf],
            losses=losses,fit_failures=failed,**cost,observations=observations_record(future),
            observation_hash=fingerprint(joint.p,joint.q,joint.v,joint.master_p,joint.master_q),
            true_covered=true_name is not None,true_name=true_name,
            terminal_voltage_min=float(physical['v'].min()),terminal_voltage_max=float(physical['v'].max()),
            voltage_min=float(np.abs(physical['voltage']).min()),voltage_max=float(np.abs(physical['voltage']).max()),
            operational_voltage_ok=bool(np.abs(physical['voltage']).min()>=.9 and np.abs(physical['voltage']).max()<=1.1))
        if good:
            winner=min(losses,key=lambda name:(losses[name],name));row['winner']=winner
            row['gaps']={name:loss-min(losses.values()) for name,loss in losses.items()}
            if true_name is not None:
                f=jf[true_name];row.update(true_loss=losses[true_name],true_gap=row['gaps'][true_name],wrong_winner=winner!=true_name,
                    impedance_absolute_rmse_pu=float(np.sqrt(np.mean(np.r_[(f.r-path.r)**2,(f.x-path.x)**2]))),
                    impedance_relative_rmse=float(np.sqrt(np.mean(np.r_[((f.r-path.r)/path.r)**2,((f.x-path.x)/path.x)**2]))))
        row['seconds']=perf_counter()-began;records.append(row)
    coverage=true_name is not None;complete &= coverage
    score=max(row['true_gap'] for row in records) if complete else np.inf
    result=dict(history_id=str(seed),seed=int(seed),case=label,seeds=seeds,config=config,policy_id=pid,source_sha256=hashes,
        history_observations=observations_record(history),history_hash=fingerprint(history.p,history.q,history.v,history.master_p,history.master_q),
        proposal=proposal,candidate_names=[c.name for c in candidates],historical_fits=[fit_record(c,fits[c.name]) for c in candidates if c.name in fits],
        historical_failures=hfail,historical_cost=hcost,device_constraints=asdict(constraints),selection=selection,records=records,
        true_covered=coverage,true_name=true_name,truth=truth_metadata(path),complete=bool(complete),
        relative_score=score,calibration_score_is_infinite=not complete,independent_history_unit=True,
        selection_uses_truth=False,selection_uses_future_noise=False,seconds=perf_counter()-started,
        total_ac_calls=hcost['ac_calls']+selection['selector_ac_calls']+sum(row.get('ac_calls',0) for row in records),
        ac_call_definition='algorithm ac_predict calls; simulator truth responses reported separately',
        simulation_ac_calls=1+sum('action' in row and row['status']!='operational_failure' for row in records),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    if output is not None:write_json(Path(output)/f'trial_{seed}.json.gz',result,compressed=True)
    return result


def apply_trial_gate(trial,calibration):
    """Apply a fixed relative gate; incomplete fits retain all candidates."""
    if trial['policy_id'] != calibration['policy_id']:
        raise ValueError('policy hash differs from calibration')
    if trial['history_id'] in calibration['calibration_history_ids']:
        raise ValueError('calibration/evaluation history overlap')
    gate=calibration['relative']
    threshold=np.inf if gate['vacuous'] else float(gate['threshold'])
    names=trial.get('candidate_names',sorted({name for r in trial['records'] for name in r.get('losses',{})}))
    for row in trial['records']:
        truth=row.get('true_name',trial.get('true_name'))
        if row['status']=='complete':
            retained=[name for name in names if row['gaps'][name]<=threshold]
            status='vacuous_retained_all' if gate['vacuous'] else 'relative_gate_applied'
        else:
            retained=list(names);status='fit_failure_retained_all'
        row.update(retained_candidates=retained,gate_status=status,
            relative_gate_threshold=None if np.isinf(threshold) else threshold,
            false_rejection=bool(truth is not None and truth not in retained),
            only_truth_retained=bool(truth is not None and retained==[truth]),
            unresolved=len(retained)>1,
            wrong_candidates_removed=sum(name!=truth and name not in retained for name in names) if truth is not None else None)
    trial['any_method_false_rejection']=any(r['false_rejection'] for r in trial['records'])
    return trial

def calibration_result(trials,*,alpha=.1):
    if not trials:raise ValueError('calibration requires independent histories')
    ids=[t['history_id'] for t in trials]
    if len(set(ids))!=len(ids):raise ValueError('duplicate calibration history')
    pids={t['policy_id'] for t in trials}
    if len(pids)!=1:raise ValueError('mixed policy hashes in calibration')
    scores=[t['relative_score'] if t['complete'] else np.inf for t in trials]
    gate=conformal_upper_quantile(scores,alpha)
    return dict(policy_id=next(iter(pids)),calibration_history_ids=ids,relative=gate.summary(),
        failed_history_count=sum(not t['complete'] for t in trials),all_history_count=len(trials),
        statistic='max_method true full-SSE loss minus minimum candidate loss',alpha=alpha,
        scope='frozen policy and declared synthetic independent-history mixture')


def summarize_trials(trials,calibration=None):
    ids=[t['history_id'] for t in trials]
    if len(set(ids))!=len(ids):raise ValueError('duplicate evaluation history')
    if calibration is not None:
        if any(t['policy_id']!=calibration['policy_id'] for t in trials):raise ValueError('policy hash differs from calibration')
        if set(ids)&set(calibration['calibration_history_ids']):raise ValueError('calibration/evaluation history overlap')
        threshold=np.inf if calibration['relative']['vacuous'] else calibration['relative']['threshold']
        for trial in trials:apply_trial_gate(trial,calibration)
    else:threshold=None
    rows=[]
    for method in METHODS:
        records=[next(r for r in t['records'] if r['method']==method) for t in trials]
        covered_all=[r for r in records if r.get('true_covered')]
        covered=[r for r in covered_all if r['status']=='complete']
        rejected=eliminated=unresolved=0
        for r in covered_all:
            if threshold is None:continue
            retained=r['retained_candidates']
            rejected+=bool(r['false_rejection'])
            eliminated+=bool(r['only_truth_retained'])
            unresolved+=len(retained)>1
        rows.append(dict(method=method,all_history_count=len(trials),complete_count=sum(r['status']=='complete' for r in records),
            failed_count=sum(r['status']!='complete' for r in records),covered_complete_count=len(covered),
            false_rejections=rejected if threshold is not None else None,
            false_rejection_denominator=len(covered_all) if threshold is not None else 0,
            incomplete_retained_all_count=len(covered_all)-len(covered),
            false_rejection_interval=binomial_exact_interval(rejected,len(covered_all)) if threshold is not None and covered_all else None,
            only_truth_retained=eliminated if threshold is not None else None,unresolved=unresolved if threshold is not None else None,
            physical_failures=sum(r['status'] in ('operational_failure','physical_failure') or not r.get('operational_voltage_ok',True) for r in records),
            wrong_winners=sum(bool(r.get('wrong_winner')) for r in covered),
            impedance_absolute_rmse_pu=[r['impedance_absolute_rmse_pu'] for r in covered if 'impedance_absolute_rmse_pu' in r],
            impedance_relative_rmse=[r['impedance_relative_rmse'] for r in covered if 'impedance_relative_rmse' in r],
            mean_seconds=float(np.mean([r['seconds'] for r in records])) if records else None,
            total_ac_calls=sum(r.get('ac_calls',0) for r in records),
            absolute_loss_diagnostic=[min(r['losses'].values()) for r in records if r['status']=='complete'],
            true_gaps=[r['true_gap'] for r in covered]))
    paired=[]
    for method in METHODS[2:]:
        valid=[]
        for t in trials:
            rs={r['method']:r for r in t['records']}
            if all(rs[m]['status']=='complete' and rs[m].get('true_gap') is not None for m in (method,'constant_fixed')):
                valid.append((t['history_id'],rs[method]['true_gap'],rs['constant_fixed']['true_gap']))
        paired.append(dict(method=method,baseline='constant_fixed',paired_history_count=len(valid),
            effects=paired_continuous_effects([r[0] for r in valid],[r[1] for r in valid],[r[2] for r in valid]) if valid else None))
    return dict(all_history_count=len(trials),complete_history_count=sum(t['complete'] for t in trials),
        failed_history_count=sum(not t['complete'] for t in trials),candidate_coverage_failures=sum(not t['true_covered'] for t in trials),
        activation_count=sum(t['selection']['activated'] for t in trials),
        any_method_false_rejections=sum(bool(t.get('any_method_false_rejection')) for t in trials) if calibration is not None else None,
        policy_false_rejection_denominator=sum(bool(t['true_covered']) for t in trials) if calibration is not None else 0,
        selection_failure_counts={m:sum(t['selection']['methods'][m]['status']=='selection_failure_fallback' for t in trials) for m in METHODS},
        method_summary=rows,paired_effects=paired,calibration=calibration,
        risk_denominator_note='all histories retained; fitting failures shown separately and scored infinity in calibration',
        seconds=sum(t['seconds'] for t in trials),total_ac_calls=sum(t['total_ac_calls'] for t in trials))


def run_batch(phase,output,*,count=None,start_seed=None,config=None):
    if phase not in ('pilot','calibration','evaluation'):raise ValueError('unknown phase')
    config=configuration(config);output=Path(output);directory=output/phase;directory.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes();pid=identity(config,hashes)
    count=(6 if phase=='pilot' else 12 if phase=='calibration' else 24) if count is None else int(count)
    if count < 0 or (phase=='pilot' and count>len(PILOT_CASES)):raise ValueError('invalid predeclared count')
    start_seed=(800001 if phase=='pilot' else 810001 if phase=='calibration' else 820001) if start_seed is None else int(start_seed)
    batch=dict(phase=phase,count=count,seeds=list(range(start_seed,start_seed+count)),config=config,source_sha256=hashes,policy_id=pid,
               pilot_cases=PILOT_CASES[:count] if phase=='pilot' else None)
    if (directory/'config.json').exists() and read_json(directory/'config.json')!=serial(batch):raise ValueError('batch config/source changed; create a new output')
    if phase!='pilot':
        freeze=output/'freeze.json';frozen=dict(config=config,source_sha256=hashes,policy_id=pid)
        if freeze.exists() and read_json(freeze)!=serial(frozen):raise ValueError('formal source/policy differs from frozen version')
        if not freeze.exists():
            if phase=='evaluation':raise ValueError('calibration freeze required before evaluation')
            write_json(freeze,frozen)
    calibration=read_json(output/'calibration'/'thresholds.json') if phase=='evaluation' else None
    if calibration is not None and set(map(str,batch['seeds'])) & set(calibration['calibration_history_ids']):
        raise ValueError('calibration/evaluation history overlap before sampling')
    write_json(directory/'config.json',batch);trials=[]
    for i,seed in enumerate(batch['seeds']):
        path=directory/f'trial_{seed}.json.gz'
        if path.exists():trial=read_json(path)
        else:
            case,family=PILOT_CASES[i] if phase=='pilot' else (None,None)
            trial=run_trial(seed,case=case,truth_family=family,config=config,output=None)
            if calibration is not None:apply_trial_gate(trial,calibration)
            write_json(path,trial,compressed=True)
        if trial['policy_id']!=pid:raise ValueError('saved trial belongs to another frozen policy')
        trials.append(trial)
        write_json(directory/'progress.json',dict(completed=len(trials),planned=count,last_seed=seed,policy_id=pid))
        print(json.dumps(dict(phase=phase,index=i+1,count=count,seed=seed,case=trial['case'],seconds=trial['seconds'],
            activated=trial['selection']['activated'],complete=trial['complete']),ensure_ascii=False),flush=True)
    if phase=='calibration':
        calibration=calibration_result(trials,alpha=config['alpha']);write_json(directory/'thresholds.json',calibration)
    summary=summarize_trials(trials,calibration if phase=='evaluation' else None)
    write_json(directory/'summary.json',summary)
    return trials,summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('pilot','calibration','evaluation','summarize'),default='pilot')
    parser.add_argument('--output',type=Path,default=HERE/'evidence'/'gradient_price'/'v2_supported')
    parser.add_argument('--count',type=int);parser.add_argument('--start-seed',type=int);parser.add_argument('--config',type=Path)
    args=parser.parse_args();config=None if args.config is None else read_json(args.config)
    if args.phase=='summarize':
        for phase in ('pilot','calibration','evaluation'):
            folder=args.output/phase
            if not folder.exists():continue
            trials=[read_json(p) for p in sorted(folder.glob('trial_*.json.gz'))]
            calibration=read_json(args.output/'calibration'/'thresholds.json') if phase=='evaluation' else None
            write_json(folder/'summary.json',summarize_trials(trials,calibration))
    else:run_batch(args.phase,args.output,count=args.count,start_seed=args.start_seed,config=config)


if __name__=='__main__':main()
