"""Independent saved-path and frozen-history verification; never solve or run AC."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from ... import io
from ..customer_flexibility_20261113.model import CustomerCase
from ..customer_flexibility_20261113.placement import CandidateEvidence
from ..flexibility_sweep_20261113.fitting import calibrated_means
from ..astra_review_20261113.solver import Recovery
from .data import load_entry
from .solver import SymmetricRecovery
from .solver_v2 import CalibratedSymmetricRecovery
from .solver_v3 import WeightedRecovery
from .catalog import resources

ROOT=Path(__file__).resolve().parents[3]

def evidence_for(mode,case,score,val,forecast,base,plan,voltage_sigma):
    """Dispatch the recorded mathematical model; never optimize."""
    if mode in ('nominal','calibrated'):
        return Recovery(case,score,val,forecast,base,mode,voltage_sigma)
    if mode in ('multiplicative_absolute','capped_absolute'):
        return CalibratedSymmetricRecovery(case,score,val,forecast,base,mode,plan['target_mass'])
    if mode in ('weighted_nominal','weighted_capped'):
        return WeightedRecovery(case,score,val,forecast,base,mode,plan['target_mass'])
    if mode in ('symmetric','symmetric_guard','symmetric_full','symmetric_full_guard'):
        return SymmetricRecovery(case,score,val,forecast,base,
            mode='symmetric_guard' if 'guard' in mode else 'symmetric',target_mass=plan['target_mass'])
    raise ValueError('Unrecognized recorded evidence model: '+mode)

def verify(run,out):
    if out.exists():raise ValueError('Preserve earlier verification')
    out.mkdir(parents=True);p=io.read(run/'protocol.json');entries=io.read(run/'inputs.json')
    if not (run/'complete.json').exists():raise ValueError('Only audit completed batch')
    io.write(out/'protocol.json',dict(source=io.ref(__file__),batch_protocol=io.ref(run/'protocol.json'),new_optimization=0,new_AC=0,
        all400_manifest=io.ref(ROOT/'outputs/astra_inventory_20261114/mainline400.json'),
        scope='All400 historical generation assumptions, all saved complete rows and returned policies. Replay saved shared seed arrays where available; otherwise cross-check named seed cost receipts.'))
    errors=[];maxerr=defaultdict(float);counts=Counter();certificates=[];counterexamples=[];resource_summary={};meta_cache={}
    def require(condition,message):
        if not condition:errors.append(message)
    def near(a,b,key,tol=1e-8):
        error=float(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0));maxerr[key]=max(maxerr[key],error)
        require(error<=tol,key+' mismatch '+str(error))
    all400=io.read(ROOT/'outputs/astra_inventory_20261114/mainline400.json')
    for e in all400:
        batch=Path(e['request']['batch'])
        if str(batch) not in meta_cache:meta_cache[str(batch)]=io.read(batch/'generation.json')
        m=meta_cache[str(batch)][e['request']['job']]['metadata']
        require(m['physical_branch_sha256']==e['physical_branch_sha256'],'Original branch hash '+e['id'])
        for field in ('case','profile','level','voltage_sigma','pq_sigma','root_sigma'):
            require(m[field]==e['physical_parameters'][field],'Original physical parameter '+e['id']+'/'+field)
        require(e['training_points']==64 and e['validation_points']==32,'Original history split '+e['id'])
        require(e['physical_parameters']['pq_sigma']==.005,'P/Q meter sigma '+e['id'])
        counts['original400_input_assumptions']+=1
        if e['case'] not in resource_summary:
            c=resources(e,p.get('resource_scale',1.))
            nominal_capacity=0.
            for r in c.resources:
                if r['kind']=='pv':nominal_capacity+=r['inverter_kva']/np.sqrt(1+r.get('q_over_p',0)**2)
                else:nominal_capacity+=r['power_kw']
            resource_summary[e['case']]=dict(resources=c.R,distinct_nodes=len(set(c.node)),labels=list(c.labels),
                kind_counts=dict(Counter(r['kind'] for r in c.resources)),nameplate_active_kw=nominal_capacity,
                maximum_simultaneous_positive_kw=float(np.maximum(c.upper,0).sum(1).max()),
                maximum_simultaneous_negative_kw=float(np.maximum(-c.lower,0).sum(1).max()),
                base_kw=m['base_kw'],base_kv=m['base_kv'],history_device_baseline=False,new_day_device_baseline=True)
            resource_summary[e['case']].update(contract_scale=p.get('resource_scale',1.),installed_scale=p.get('baseline_scale',1.))
    for entry in entries:
        score,train,val,truth=load_entry(entry)
        require(score['cells']==val.y.size,'Scoring history cells '+entry['id'])
        case=CustomerCase(io.read(run/'resource_configs'/(entry['case']+'.json')))
        H=np.asarray([v['H'] for v in score['parameters']]);history=np.array([v['history_sse'] for v in score['parameters']])
        bank=score['native_bank'];NH=np.asarray([v['H'] for v in bank]);NS=np.array([v['history_sse'] for v in bank]);target=score['map_index']
        power=np.c_[val.p,val.q];near(history,np.sum((val.y[None]-np.einsum('tc,knc->ktn',power,H))**2,axis=(1,2)),'validation_only_SSE',1e-11)
        means=calibrated_means(score,val)
        for day in range(p['days']):
            d=run/'days'/entry['id']/f'day{day:03d}'
            if not (d/'complete.json').exists():counts['incomplete_task']+=1;continue
            rows=io.read(d/'results.json');plans=io.read(d/'plans.json');seed_record=io.read(d/'shared_seeds.json');generation=io.read(d/'generation.json')
            require(generation['physical_branch_sha256']==entry['physical_branch_sha256'],'New branch hash '+entry['id'])
            require(generation['profile']==entry['profile'] and generation['level']==entry['level'],'New pressure parameters '+entry['id'])
            with np.load(d/'background.npz') as z:background={k:np.array(z[k]) for k in z.files}
            with np.load(d/'planner.npz') as z:planner={k:np.array(z[k]) for k in z.files}
            original=np.c_[background['P'],background['Q']];replicate=p['replicate_base']+entry['original_index']*100+day
            rng=np.random.default_rng(np.random.SeedSequence([p['study_seed'],replicate,991]))
            near(planner['forecast'],original*(1+.05*rng.normal(size=original.shape)),'forecast_seed_and_5percent',1e-13)
            rng=np.random.default_rng(np.random.SeedSequence([p['study_seed'],replicate,773]));near(background['response'],rng.uniform(.9,1.,case.R),'response_seed',1e-13)
            counts['saved_days']+=1
            accepted={r['name']:r['cost_yuan'] for r in seed_record.get('records',[]) if r.get('accepted')}
            seed_points={}
            if (d/'shared_points.npz').exists():
                with np.load(d/'shared_points.npz') as z:seed_points={k:np.array(z[k]) for k in z.files}
                seal=io.read(d/'seal.json')
                require(seal.get('shared_points') is not None and io.ref(d/'shared_points.npz')['sha256']==seal['shared_points']['sha256'],'Shared seed array hash')
                require(set(seed_points)==set(accepted),'Shared seed action names')
                for name,point in seed_points.items():
                    require(case.audit(point)['passed'],'Saved seed resource feasibility')
                    near(case.fee(point),accepted[name],'saved_seed_action_fee',1e-7)
                    counts['saved_seed_resource_replays']+=1
                counts['days_with_saved_seed_arrays']+=1
            else:counts['days_without_saved_seed_arrays']+=1
            for mode,plan in plans.items():
                u=planner[mode];counts['planned_policies']+=1
                counts['quote_exceptions']+=int(plan.get('status')=='exception')
                require(case.audit(u)['passed'],'Requested resource feasibility '+entry['id']+'/'+mode)
                if plan.get('feasible'):near(case.fee(u),plan['cost_yuan'],'plan_fee',1e-7)
                else:near(u,np.zeros(case.shape),'explicit_zero_fallback',1e-13)
                if mode not in ('nominal','calibrated'):
                    near(plan.get('shared_seed_seconds',0),seed_record['seconds'],'shared_seed_time_charge',1e-10)
                    for record in plan.get('initial',{}).get('candidates',[]):
                        if record.get('name') in accepted and record.get('cost_yuan') is not None:
                            near(record['cost_yuan'],accepted[record['name']],'shared_seed_cost_identity',1e-7);counts['seed_cost_records']+=1
                    initial=plan.get('initial',{})
                    if initial.get('feasible') and initial.get('name') in accepted:
                        near(initial['cost_yuan'],accepted[initial['name']],'selected_initial_cost',1e-7)
                if plan.get('feasible') and plan.get('lower_bound_yuan') is not None:
                    require(plan['lower_bound_yuan']<=plan['cost_yuan']+1e-7,'Bound exceeds feasible cost')
                if plan.get('status')=='exception':continue
                if len(H)==1:
                    near(u,np.zeros(case.shape),'single_candidate_zero',1e-13)
                    counts['single_candidate_zero_quotes']+=1
                    continue
                if mode=='old_gamma2':
                    other=[i for i in range(len(H)) if i!=target]
                    evidence=CandidateEvidence(case,H,target,other,planner['forecast'],score['sigma2'],generation['base_kw'],means)
                    if plan.get('feasible'):require(np.min(evidence.direct(u)-evidence.required(2.))>=-1e-7,'Old gain direct constraint')
                elif mode in ('nominal','calibrated'):
                    evidence=evidence_for(mode,case,score,val,planner['forecast'],generation['base_kw'],plan,entry['voltage_sigma'])
                    info=evidence.direct(u);delivered=evidence.direct(u*background['response'][None])
                    if plan.get('feasible'):require(info['feasible'],'Native recovery direct constraint')
                    truth_in=score['models'][target]['topology_id']==truth['truth_id']
                    lower=info['model_recovery_lower'];delivered_lower=delivered['model_recovery_lower']
                else:
                    evidence=evidence_for(mode,case,score,val,planner['forecast'],generation['base_kw'],plan,entry['voltage_sigma'])
                    info=evidence.direct(u);delivered=evidence.direct(u*background['response'][None])
                    if plan.get('feasible'):require(info['feasible'],'Symmetric direct constraint')
                    truth_in=truth['truth_id'] in plan['target_topology_ids']
                    lower=info['worst_recovery_lower'];delivered_lower=delivered['worst_recovery_lower']
                if seed_points:
                    initial=plan.get('initial',{})
                    for record in initial.get('candidates',[]):
                        name=record.get('name')
                        if name not in seed_points:continue
                        point=seed_points[name]
                        valid=(np.min(evidence.direct(point)-evidence.required(2.))>=-1e-7) if mode=='old_gamma2' else evidence.direct(point)['feasible']
                        require(bool(valid)==bool(record.get('qcqp_feasible')),'Seed evidence feasibility '+mode)
                        counts['seed_evidence_replays']+=1
                    name=initial.get('name')
                    if name in seed_points and initial.get('feasible'):
                        seed=seed_points[name]
                        if plan.get('source')=='verified_initial_retained':near(u,seed,'retained_seed_action',1e-10)
                        if plan.get('source')=='native_improved_initial':require(case.fee(u)<case.fee(seed)-1e-9,'Reported strict initial improvement')
                if mode!='old_gamma2' and plan.get('feasible'):
                    for physics in ('AC','linear'):
                        r=next((r for r in rows if r['strategy']==mode and r['physics']==physics),{})
                        certificates.append(dict(id=entry['id'],day=day,role=entry['role'],case=entry['case'],mode=mode,physics=physics,
                            truth_in_selected_set=truth_in,model_lower=lower,delivered_action_model_lower=delivered_lower,
                            representative_correct=bool(r.get('status')=='complete' and r.get('map_correct')),
                            native_correct=bool(r.get('status')=='complete' and r.get('native_map_correct'))))
            cache={}
            for r in rows:
                if r['status']!='complete':counts['execution_failed']+=1;continue
                strategy,physics=r['strategy'],r['physics'];u=planner[strategy];actual=u*background['response'][None]
                require(case.audit(actual)['passed'],'Delivered domain violation')
                near(case.fee(u),r['planned_fee_yuan'],'requested_fee',1e-7);near(case.fee(actual),r['settled_fee_yuan'],'settled_fee',1e-7)
                with np.load(d/(strategy+'_'+physics+'.npz')) as z:obs={k:np.array(z[k]) for k in z.files}
                actual_power=original+case.shift_pq_pu(actual,generation['base_kw'])
                near(np.c_[obs['true_P'],obs['true_Q']],actual_power,'delivered_true_power',1e-12)
                near(obs['P'],obs['true_P']*(1+generation['pq_sigma']*background['noise0']),'P_meter_point5percent',1e-13)
                near(obs['Q'],obs['true_Q']*(1+generation['pq_sigma']*background['noise1']),'Q_meter_point5percent',1e-13)
                near(obs['V'],obs['true_V']*(1+generation['voltage_sigma']*background['noise2']),'V_meter',1e-13)
                near(obs['root'],background['root']*(1+generation['voltage_sigma']*background['noise3']),'root_meter',1e-13)
                if strategy in cache:
                    for field in ('P','Q','root','true_P','true_Q'):near(obs[field],cache[strategy][field],'AC_linear_identical_inputs',1e-13)
                else:cache[strategy]=obs
                power=np.c_[obs['P'],obs['Q']];Y=obs['root'][:,None]**2-obs['V']**2
                near(Y,obs['Y'],'Y_identity',1e-13)
                SSE=history+np.sum((Y[None]-np.einsum('tc,knc->ktn',power,H))**2,axis=(1,2));near(SSE,obs['losses'],'saved_SSE',1e-10)
                logp=-SSE/(2*score['sigma2']);logp-=logsumexp(logp);near(np.exp(logp),np.exp(obs['log_probability']),'saved_probability',1e-9)
                nativeSSE=NS+np.sum((Y[None]-np.einsum('tc,knc->ktn',power,NH))**2,axis=(1,2))
                scaled=nativeSSE/(score['cells']+Y.size)/score['reference_mse'];eligible=np.flatnonzero(scaled<=scaled.min()+score['validation_tie'])
                chosen=min(eligible,key=lambda i:(bank[i]['internal'],scaled[i],i))
                require(int(chosen)==r['native_selected_index'],'Native selection index mismatch')
                require((bank[chosen]['topology_id']==truth['truth_id'])==r['native_map_correct'],'Native selection truth mismatch')
                counts['native_selection_roundtrips']+=1
                before=next((x for x in rows if x['strategy']=='no_dfs' and x['physics']==physics),{})
                if strategy!='no_dfs' and not strategy.endswith('_rotation') and r.get('p_true') is not None and before.get('p_true') is not None and r['p_true']<before['p_true']-1e-12:
                    counterexamples.append(dict(id=entry['id'],day=day,role=entry['role'],case=entry['case'],strategy=strategy,physics=physics,
                        p_true_change=r['p_true']-before['p_true'],native_before=before['native_map_correct'],native_after=r['native_map_correct']))
    groups={}
    for mode,physics in sorted({(r['mode'],r['physics']) for r in certificates}):
        own=[r for r in certificates if r['mode']==mode and r['physics']==physics and r['model_lower']>=.9-1e-6]
        conditional=[r for r in own if r['truth_in_selected_set']]
        groups[mode+'/'+physics]=dict(feasible_model_passed=len(own),truth_in_selected_set=len(conditional),
            representative_wrong_with_truth_in_set=sum(not r['representative_correct'] for r in conditional),
            native_wrong_with_truth_in_set=sum(not r['native_correct'] for r in conditional),
            delivered_action_still_model90=sum(r['delivered_action_model_lower']>=.9-1e-6 for r in conditional),
            missing_condition_excluded=len(own)-len(conditional))
    result=dict(protocol=io.ref(out/'protocol.json'),counts=dict(counts),maximum_errors=dict(maxerr),resources=resource_summary,
        certificate_calibration=groups,negative_counterexamples_count=len(counterexamples),
        negative_counterexample_summary={mode+'/'+physics:dict(count=len(group),worst=min(r['p_true_change'] for r in group)) for mode,physics in sorted({(r['strategy'],r['physics']) for r in counterexamples}) if (group:=[r for r in counterexamples if r['strategy']==mode and r['physics']==physics])},
        passed=not errors,errors=errors,seed_verification_limit='Saved arrays were replayed for resource feasibility, fee and consumer evidence constraints where present. Days without seed arrays permit only cross-log named costs and selected-initial receipts; no LPs were rerun.',
        proof_scope='Native infeasible status is a solver proof about its recorded QCQP constraints. Unresolved means no certified action in budget. This audit does not establish original chance-problem infeasibility.')
    io.write(out/'certificate_private.json',certificates);io.write(out/'negative_counterexamples_private.json',counterexamples);io.write(out/'summary.json',result)
    return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args()
    result=verify(args.run.resolve(),args.out.resolve());print('passed',result['passed'],'counts',result['counts'],'errors',len(result['errors']))
