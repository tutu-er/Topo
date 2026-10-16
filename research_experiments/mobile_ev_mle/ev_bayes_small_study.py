"""Small frozen AC / Bayesian EV test. Truth is restricted to evaluators."""
from __future__ import annotations

import argparse
from dataclasses import replace, asdict, is_dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import platform
import traceback
import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint

from . import ev_bayes_small_ac as ac
from . import ev_bayes_small_probability as probability
from . import ev_bayes_small_milp as optimization
from . import ev_mobile_v4_cases as cases
from . import ev_mobile_v4_domain as nested
from . import ev_mobile_v4_study as previous
from . import ev_mobile_v3_cases as public
from . import ev_rx_study as rx
from . import ev_systematic_refit as refit
from . import likelihood as lk
from . import gradient_price_study as io
from .ev_fixed_train_fit import subset_observations, subset_model
from .ev_rx_fit import fit_rx_inputs
from .ev_rx_precision import solve_squared_precision, precision_weights
from .ev_original_mip_backend import solve_linear_mip
from .information_context import meter_to_record


HERE=Path(__file__).resolve().parent
CONFIG=dict(version='bayes_small_v1',train=80,evidence=20,horizon=6,
    planning_samples_per_candidate=256,independent_samples_per_candidate=2048,
    actual_noise_repeats=32,source_interface=1,solver_seconds=5.,max_proposals=2,
    segments=4,exp_segments=8,power_radius_kw=3.,abstention_cost=.1,
    credible_mass=.9,fit_max_nfev=300,fit_multistart=1,
    decision_scope='fixed-trained-line local Gaussian model; no parameter integration',
    primary_target='expected MAP classification error',regimes=['regular','shift'],
    uncertainty='joint PQ and same-interface source; harmonic/AR empirical Gaussian process',
    old_sources_unchanged=True,holdout_used_to_refit_lines=False,
    covariance_in_master='frozen at anchor; recomputed for full AC acceptance',
    nonmonotonic_actual_outcomes_retained=True)


def specifications(phase):
    if phase=='development':
        return [dict(index=i,seed=2026102600+i,family_index=f,phase=phase)
                for i,f in enumerate((1,0))]
    if phase!='evaluation':raise ValueError('development or evaluation required')
    return [dict(index=i,seed=2026102700+i,family_index=i%4,phase=phase) for i in range(8)]


def hashes():
    values=previous.source_hashes()
    for path in [HERE/('ev_bayes_small_'+suffix+'.py') for suffix in ('ac','probability','milp','study')]:
        values[path.name]=sha256(path.read_bytes()).hexdigest()
    for name in ('ev_bayes_small_protocol.md',):
        values[name]=sha256((HERE/name).read_bytes()).hexdigest()
    return values


def freeze(root):
    import base64
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    data=dict(config=CONFIG,specifications={p:specifications(p) for p in ('development','evaluation')},
        source_sha256=hashes(),environment=dict(python=platform.python_version(),numpy=np.__version__,
            scipy=scipy.__version__,platform=platform.platform()),base_commit='d6718af9')
    data['policy_id']=io.identity(data['config'],data['source_sha256'])
    path=root/'freeze.json'
    if path.exists() and io.read_json(path)!=io.serial(data):
        raise ValueError('freeze changed; preserve old run and create separate directory')
    io.write_json(path,data)
    snapshot={name:base64.b64encode(((HERE.parents[1]/name) if name.startswith('rnj_wzzt_core/') else HERE/name).read_bytes()).decode()
              for name in data['source_sha256']}
    io.write_json(root/'source_snapshot.json.gz',dict(source_sha256=data['source_sha256'],source_base64=snapshot),compressed=True)
    return data


def complete_curve(domain,power,seconds=5.):
    arrays=domain.model.arrays();n=domain.model.n_variables
    control=domain.as_dict()['control_matrix'] if 'control_matrix' in domain.as_dict() else domain.as_dict().get('u_map')
    # The public direct net-power indices are exact and require no route reconstruction.
    indices=domain.indices.u.ravel();A=np.zeros((len(indices),n));A[np.arange(len(indices)),indices]=1
    constraints=[arrays['constraints'],LinearConstraint(A,np.asarray(power).ravel(),np.asarray(power).ravel())]
    result=solve_linear_mip(np.zeros(n),integrality=arrays['integrality'],bounds=arrays['bounds'],
        constraints=constraints,time_limit=seconds,mip_rel_gap=0.,backend='gurobi')
    if result.x is None or not domain.check(result.x)['feasible']:
        raise RuntimeError('specified baseline curve has no returned feasible operation')
    return np.asarray(result.x)


def score_summary(value):
    return {k:v for k,v in value.items() if k not in ('samples','stratum_samples','strata_samples')}


def serial_record(value):
    if is_dataclass(value):return serial_record(asdict(value))
    if isinstance(value,dict):return {k:serial_record(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial_record(v) for v in value]
    return io.serial(value)


class Objective:
    def __init__(self,context,noise):
        self.context=context;self.noise=noise;self.cache={};self.calls=0;self.ac_states=0;self.seconds=0.
        self.objective_id=io.identity(dict(log_prior=context['log_prior'],
            latent=[dict(m=r['latent_mean'],C=r['latent_cov'],line=r['line_sha256']) for r in context['rows']],
            noise_sha=sha256(np.ascontiguousarray(noise).tobytes()).hexdigest()),{})

    def get(self,power):
        key=nested.action_key(power)
        if key not in self.cache:
            started=perf_counter();forecast=ac.forecast_bank(self.context,power,with_jacobian=True)
            score=probability.evaluate_bank(forecast['bank'],self.noise,abstention_cost=CONFIG['abstention_cost'])
            self.calls+=1;self.ac_states+=forecast['AC_states'];self.seconds+=perf_counter()-started
            self.cache[key]=dict(power_kw=np.asarray(power).copy(),forecast=forecast,score=score,
                score_value=-float(score['risk']),objective_id=self.objective_id)
        return self.cache[key]


def square_family(domains,meter,constant,highlow):
    plans={};logs={};began=perf_counter()
    for name,domain in domains.items():
        parents=[nested.lift_operation(domains[p],domain,plans[p]) for p in nested.PARENTS[name]]
        initial={'constant':constant,'highlow':highlow,
                 **{'parent_'+str(i):domain.decode(x)['u_kw'] for i,x in enumerate(parents)}}
        result=solve_squared_precision(domain,meter,initial,exchange_options=dict(
            total_seconds=12.,solver_seconds=CONFIG['solver_seconds'],max_iterations=CONFIG['max_proposals'],
            control_segments=CONFIG['segments'],residual_segments=4,max_segments=8))
        candidates=list(parents)
        if result.get('success'):candidates.append(np.asarray(result['operation']['x']))
        if not candidates:raise RuntimeError('square family has no feasible '+name)
        weights=precision_weights(meter,domain.indices.u.shape)
        Q=lambda x:float(np.sum(weights*domain.decode(x)['u_kw']**2))
        winner=candidates[0]
        for x in candidates[1:]:
            if Q(x)>Q(winner)+1e-9:winner=x
        plans[name]=winner
        logs[name]=dict(result=result,Q=Q(winner),parent_Q=[Q(x) for x in parents],
            inherited_Q_not_decreased=all(Q(winner)>=Q(x)-1e-8 for x in parents))
    return plans,logs,perf_counter()-began


def bayes_family(domains,objective,squares,baseline_x):
    plans={};logs={};began=perf_counter()
    for name,domain in domains.items():
        started=perf_counter();options=[]
        for p in nested.PARENTS[name]:options.append(('parent:'+p,nested.lift_operation(domains[p],domain,plans[p])))
        for label,x in baseline_x.items():
            lifted=x if name=='fixed_charge' else nested.lift_operation(domains['fixed_charge'],domain,x)
            options.append((label,lifted))
        options.append(('square:'+name,squares[name]))
        if not options:raise RuntimeError('no initial plan')
        ranked=[(origin,x,objective.get(domain.decode(x)['u_kw'])) for origin,x in options]
        origin,winner,record=ranked[0]
        for candidate_origin,x,candidate_record in ranked[1:]:
            if candidate_record['score_value']>record['score_value']+1e-12:
                origin,winner,record=candidate_origin,x,candidate_record
        anchor_record=record;events=[]
        for iteration in range(CONFIG['max_proposals']):
            power=domain.decode(winner)['u_kw'];forecast=record['forecast']
            try:
                result=optimization.solve_bc_milp(domain,forecast['bank'],forecast['jacobian'],power,
                    segments=CONFIG['segments'],exp_segments=CONFIG['exp_segments'],
                    seconds=CONFIG['solver_seconds'],power_radius=CONFIG['power_radius_kw'])
            except Exception as exc:
                result=dict(success=False,operation_x=None,failure=type(exc).__name__+': '+str(exc),
                            exception_preserves_parent=True)
            event=dict(iteration=iteration,milp=result,accepted=False,parent_risk=-record['score_value'])
            x=result.get('operation_x')
            if x is not None:
                try:
                    proposed=objective.get(domain.decode(np.asarray(x))['u_kw'])
                    event.update(proposed_risk=-proposed['score_value'],
                        risk_gain=proposed['score_value']-record['score_value'])
                    if proposed['score_value']>record['score_value']+1e-12:
                        winner=np.asarray(x);record=proposed;origin='BC_MILP';event['accepted']=True
                except Exception as exc:event['full_AC_failure']=type(exc).__name__+': '+str(exc)
            events.append(event)
        plans[name]=np.asarray(winner).copy()
        parent_risks={p:float(-objective.get(domains[p].decode(plans[p])['u_kw'])['score_value']) for p in nested.PARENTS[name]}
        logs[name]=dict(origin=origin,initial_risk=-anchor_record['score_value'],risk=-record['score_value'],
            events=events,parent_risks=parent_risks,
            successful_milp_proposals=sum(bool(e['milp'].get('success')) for e in events),
            failed_milp_proposals=sum(not bool(e['milp'].get('success')) for e in events),
            accepted_milp_proposals=sum(bool(e['accepted']) for e in events),
            executable_plan_available=True,
            inherited_risk_not_increased=all(-record['score_value']<=r+1e-12 for r in parent_risks.values()),
            seconds=perf_counter()-started)
    return plans,logs,perf_counter()-began


def _credible(posterior,mass):
    order=np.argsort(-posterior,kind='stable');cum=np.cumsum(posterior[order]);n=int(np.searchsorted(cum,mass))+1
    return order[:min(n,len(order))]


def actual_observations(case,power,white):
    T=CONFIG['horizon'];start=case['future_start'];rows=slice(start,start+T);truth=case['truth']
    physical=lk.ac_predict(truth['candidate'],truth['r'],truth['x'],
        case['p'][rows]+power/case['base_kw'],case['q'][rows],case['source_p'][rows],case['source_q'][rows],case['root_v'][rows])
    from . import ev_original_study as generator
    return generator.observation(physical,case['fm'],white),physical


def actual_evaluate(case,power,bank,noise_seed):
    T,n=power.shape;R=CONFIG['actual_noise_repeats'];K=len(case['bank'])
    true_index=next(i for i,c in enumerate(case['bank']) if c.supports==case['truth']['candidate'].supports)
    rng=np.random.default_rng(noise_seed)
    white={k:rng.standard_normal((R,T,n) if k in ('p','q','v') else (R,T)) for k in ac.CHANNELS}
    samples=[];first=None
    # Physical trajectory fixed; independent meter noise is paired across policies.
    zero={k:np.zeros_like(v[0]) for k,v in white.items()}
    _,physical=actual_observations(case,power,zero)
    from . import ev_original_study as generator
    for r in range(R):
        observed=generator.observation(physical,case['fm'],{k:v[r] for k,v in white.items()})
        if first is None:first=observed
        y=ac.observations_vector(observed,case['fm'])
        post=probability.gaussian_posterior(y,bank['means'][0],bank['covariances'][0],log_prior=bank['log_prior'])
        probabilities=np.asarray(post['posterior']).reshape(-1)
        chosen=int(np.argmax(probabilities));credible=_credible(probabilities,CONFIG['credible_mass'])
        target=np.eye(K)[true_index]
        samples.append(dict(correct=int(chosen==true_index),confidence=float(probabilities[chosen]),
            true_probability=float(probabilities[true_index]),posterior=probabilities,
            brier=float(np.sum((probabilities-target)**2)),negative_log_true_probability=float(-np.asarray(post['log_posterior']).reshape(-1)[true_index]),
            unique_correct=int(len(credible)==1 and chosen==true_index),
            wrong_unique=int(len(credible)==1 and chosen!=true_index),unresolved=int(len(credible)>1),
            true_rejected=int(true_index not in credible),credible_size=len(credible)))
    summary={key:float(np.mean([s[key] for s in samples])) for key in samples[0] if key!='posterior'}
    summary.update(n_noise=R,n_independent_histories=1,true_index=true_index,
        ac_equation_residual=float(physical['ac_equation_residual']),power_balance_residual=float(physical['power_balance_residual']),
        actual_v_min=float(np.min(physical['v'])),actual_v_max=float(np.max(physical['v'])),
        physical_voltage_feasible=bool(np.min(physical['v'])>=.85 and np.max(physical['v'])<=1.15),
        samples=samples,noise_seed=noise_seed)
    return summary,first


def run(spec):
    began=perf_counter();seed=spec['seed'];T=CONFIG['horizon']
    generated=cases.make_case('n3',seed,spec['family_index'],'regular')
    generated['config']['horizon']=T
    generated['fm']=replace(generated['fm'],
        v_sigma=np.broadcast_to(np.asarray(generated['tm'].v_sigma)[0]/np.sqrt(30./T),(T,3)),
        **{k+'_sigma':np.asarray(getattr(generated['tm'],k+'_sigma')).reshape(-1)[0]/np.sqrt(30./T)
           for k in ('p','q','master_p','master_q')})
    D=lk.concat_observations(generated['train'],generated['validation'])
    dm=refit.concat_models((generated['train'],generated['validation']),(generated['tm'],generated['vm']))
    H=CONFIG['train'];training=subset_observations(D,slice(0,H));evidence=subset_observations(D,slice(H,None))
    tm=subset_model(dm,D,slice(0,H));em=subset_model(dm,D,slice(H,None))
    cfg=generated['config'];cfg['training']['max_nfev']=CONFIG['fit_max_nfev'];cfg['training']['multistart']=CONFIG['fit_multistart']
    fits,fitting=rx.train_bank(generated['bank'],training,tm,cfg,seed+1000)
    record=dict(spec=spec,config=CONFIG,training_result=fitting,provenance=generated['provenance'],
        observed=io.observations_record(D),observed_meter=meter_to_record(dm),future_meter=meter_to_record(generated['fm']),
        candidate_names=[c.name for c in generated['bank']],candidate_miss=False,methods={})
    record['effective_provenance']=dict(train=H,evidence=D.t-H,future=T,
        per_slot_precision_weight=30./T,total_precision_weight=30.,
        integration_seconds_per_slot=3000./T,total_integration_seconds=3000.,
        generator_provenance_above_is_original_96_4_12_template=True)
    print('TRAINED',spec['phase'],spec['index'],fitting['all_complete'],round(perf_counter()-began,2),flush=True)
    if not fitting['all_complete']:return dict(record,status='training_failure',seconds=perf_counter()-began)
    context,prediction=ac.build_context(training,evidence,tm,em,generated['bank'],fits,generated['fm'],horizon=T,base_kw=generated['base_kw'])
    record['prediction']=prediction
    dimension=(3*training.n+2)*T;K=len(fits)
    noise=np.random.default_rng(seed+2000).standard_normal((1,K,CONFIG['planning_samples_per_candidate'],dimension))
    objective=Objective(context,noise)
    template=replace(public.service_domain(T,training.n,mobile=True).config,n_slots=T)
    domains=nested.build_nested_domains(template);record['operation_config']=io.serial(asdict(template))
    constant=np.zeros((T,training.n));constant[:,0]=(template.terminal_energy_kwh-template.initial_soc*template.battery_capacity_kwh)/(template.eta_charge*T*template.dt_hours)
    highlow=np.zeros_like(constant);highlow[:T//2,0]=constant[0,0]*T/(T//2)
    baseline_x={name:complete_curve(domains['fixed_charge'],u) for name,u in (('constant',constant),('highlow',highlow))}
    squares,square_logs,square_seconds=square_family(domains,generated['fm'],constant,highlow)
    bayes,bayes_logs,bayes_seconds=bayes_family(domains,objective,squares,baseline_x)
    record.update(square_logs=square_logs,bayes_logs=bayes_logs,square_seconds=square_seconds,bayes_seconds=bayes_seconds)
    plans={**{k:('fixed_charge',v) for k,v in baseline_x.items()},
        **{'square__'+k:(k,v) for k,v in squares.items()},**{'bayes__'+k:(k,v) for k,v in bayes.items()}}
    independent=np.random.default_rng(seed+3000).standard_normal((1,K,CONFIG['independent_samples_per_candidate'],dimension))
    unique={};independent_evaluations={};powers={'none':np.zeros_like(constant)}
    for method,(name,x) in plans.items():
        powers[method]=domains[name].decode(x)['u_kw']
        record['methods'][method]=dict(domain=name,operation_x=x,operation=domains[name].check(x),
            decoded=domains[name].decode(x))
        if method.startswith('bayes__'):
            log=bayes_logs[name]
            record['methods'][method].update(planning_success=log['successful_milp_proposals']>0,
                planning_improvement=log['accepted_milp_proposals']>0,
                fallback_to_initial=log['accepted_milp_proposals']==0,
                executable_plan_available=True)
        elif method.startswith('square__'):
            record['methods'][method].update(planning_success=bool(square_logs[name]['result'].get('success')),
                executable_plan_available=True)
    record['methods']['none']=dict(domain=None,operation_task_feasible=False,role='same sampling length, no battery service')
    for method,power in powers.items():
        key=nested.action_key(power);scored=objective.get(power)
        if key not in unique:
            checked=probability.evaluate_bank(scored['forecast']['bank'],independent,abstention_cost=CONFIG['abstention_cost'])
            independent_evaluations[key]=checked
            unique[key]=dict(power_kw=power,forecast=scored['forecast'],planning_score=score_summary(scored['score']),
                independent_score=score_summary(checked),actual={})
        record['methods'][method].update(action_key=key,power_kw=power,
            planning_score=score_summary(scored['score']),independent_score=unique[key]['independent_score'])
    print('PLANNED',spec['phase'],spec['index'],len(unique),round(perf_counter()-began,2),flush=True)
    paired={}
    for name in domains:
        left=record['methods']['bayes__'+name]['action_key'];right=record['methods']['square__'+name]['action_key']
        paired['bayes_minus_square__'+name]=probability.compare_evaluations(independent_evaluations[left],independent_evaluations[right])
        for p in nested.PARENTS[name]:
            right=record['methods']['bayes__'+p]['action_key']
            paired['bayes__'+name+'_minus_'+p]=probability.compare_evaluations(independent_evaluations[left],independent_evaluations[right])
    record['independent_paired']=paired
    first_observations={};true_future_inputs={}
    for regime in CONFIG['regimes']:
        replay=cases.make_case('n3',seed,spec['family_index'],regime)
        replay['config']['horizon']=T;replay['fm']=generated['fm']
        if not np.array_equal(replay['train'].v,generated['train'].v):raise AssertionError('future regime changed training')
        rows=slice(replay['future_start'],replay['future_start']+T)
        true_future_inputs[regime]={k:np.asarray(replay[k])[rows].copy() for k in ('p','q','source_p','source_q','root_v')}
        for key,row in unique.items():
            actual,first=actual_evaluate(replay,row['power_kw'],row['forecast']['bank'],seed+4000)
            row['actual'][regime]=actual;first_observations[(regime,key)]=first
    # Original complete AC joint-fit check on three prespecified plans, one noise draw.
    old_options={**cfg['training'],**cfg['source'],'ratio_bounds':cfg['ratio_bounds']}
    Dfits={}
    for candidate in generated['bank']:
        f=fits[candidate.name]
        try:Dfits[candidate.name]=fit_rx_inputs(candidate,D,dm,initial=f,fixed_r=f.r,fixed_x=f.x,seed=seed+5000,**old_options)
        except Exception as exc:
            record.setdefault('legacy_prefit_failures',{})[candidate.name]=type(exc).__name__+': '+str(exc)
    legacy={};true_name=next(c.name for c in generated['bank'] if c.supports==generated['truth']['candidate'].supports)
    for method in ('constant','square__mobile_bidir','bayes__mobile_bidir'):
        key=record['methods'][method]['action_key']
        for regime in CONFIG['regimes']:
            ident=regime+'__'+key
            if ident not in legacy:
                result=previous.evaluate_future(generated['bank'],D,dm,first_observations[(regime,key)],generated['fm'],Dfits,old_options,seed+6000)
                if result['complete']:
                    result['M']=min(v for k,v in result['losses'].items() if k!=true_name)-result['losses'][true_name]
                    result['selected']=min(result['losses'],key=result['losses'].get)
                legacy[ident]=result
    record.update(actions=unique,legacy_joint_audit=legacy,true_name=true_name,
        true_lines=dict(r=generated['truth']['r'],x=generated['truth']['x']),
        true_future_inputs_evaluation_only=true_future_inputs,
        objective_calls=objective.calls,AC_predictive_states=objective.ac_states,objective_seconds=objective.seconds,
        objective_id=objective.objective_id,status='complete',seconds=perf_counter()-began)
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--phase',choices=['development','evaluation'],default='development')
    parser.add_argument('--indices',nargs='*',type=int);args=parser.parse_args();root=Path(args.output)
    frozen=freeze(root);specs=specifications(args.phase)
    for spec in specs:
        if args.indices is not None and spec['index'] not in args.indices:continue
        path=root/args.phase/('history_%03d.json.gz'%spec['index']);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():continue
        try:record=run(spec)
        except Exception as exc:
            record=dict(spec=spec,status='execution_failure',failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        record['policy_id']=frozen['policy_id'];io.write_json(path,serial_record(record),compressed=True)
        io.write_json(root/'checkpoint.json',dict(phase=args.phase,last_index=spec['index'],last_status=record['status'],
            completed_files=[p.name for p in sorted((root/args.phase).glob('history_*.json.gz'))],policy_id=frozen['policy_id']))
        print('SAVED',args.phase,spec['index'],record['status'],flush=True)


if __name__=='__main__':main()
