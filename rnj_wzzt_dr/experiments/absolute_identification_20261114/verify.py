"""Independent saved-path and frozen-history verification; never solve or run AC."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
import hashlib
import json
from collections import Counter,defaultdict
from datetime import datetime,timezone
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
    if mode in ('symmetric','symmetric_guard','symmetric_full','symmetric_full_guard','pq_trust1','pq_trust2'):
        return SymmetricRecovery(case,score,val,forecast,base,
            mode='symmetric_guard' if 'guard' in mode or mode.startswith('pq_trust') else 'symmetric',target_mass=plan['target_mass'])
    raise ValueError('Unrecognized recorded evidence model: '+mode)

def verify(run,out,*,completed_only=False,limit_completed=0,history_id=None):
    if out.exists():raise ValueError('Preserve earlier verification')
    out.mkdir(parents=True);p=io.read(run/'protocol.json');entries=io.read(run/'inputs.json')
    trust_snapshot=run/'source_snapshot'/Path(__file__).with_name('trusted_direction.py').relative_to(ROOT)
    legacy_object_hash=trust_snapshot.exists() and 'io.contract_hash(dict(' in trust_snapshot.read_text(encoding='utf-8')
    if history_id is not None:
        entries=[e for e in entries if e['id']==history_id]
        if not entries:raise ValueError('Requested history absent from batch')
    if not (run/'complete.json').exists() and not completed_only:raise ValueError('Only audit completed batch unless partial preflight explicitly requested')
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    io.write(out/'protocol.json',dict(source=io.ref(__file__),source_snapshot=io.ref(out/'source_snapshot.py'),batch_protocol=io.ref(run/'protocol.json'),new_optimization=0,new_AC=0,
        all400_manifest=io.ref(ROOT/'outputs/astra_inventory_20261114/mainline400.json'),partial_preflight=completed_only,limit_completed=limit_completed,history_id=history_id,
        scope='All400 historical generation assumptions, all saved complete rows and returned policies. Replay saved shared seed arrays where available; otherwise cross-check named seed cost receipts.'))
    errors=[];maxerr=defaultdict(float);counts=Counter();certificates=[];counterexamples=[];resource_summary={};meta_cache={};trusted_records=[]
    metadata_defects=[];training_contracts={};nested_trust_records=[]
    def progress(status,histories_scanned=0,last_history=None):
        io.write(out/'progress.json',dict(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),
            histories_scanned=histories_scanned,histories_planned=len(entries),last_history=last_history,
            checked_days=counts['saved_days'],planned_days=len(entries)*p['days'],
            numerical_errors=len(errors),metadata_hash_defect=len(metadata_defects)))
    progress('running')
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
    histories_scanned=0
    for entry_index,entry in enumerate(entries):
        if limit_completed and counts['saved_days']>=limit_completed:break
        score,train,val,truth=load_entry(entry)
        contracts=dict(labels=io.contract_hash(np.asarray(train.labels,dtype=np.int64)),
            p=io.contract_hash(np.asarray(train.p,dtype=np.float64)),q=io.contract_hash(np.asarray(train.q,dtype=np.float64)))
        training_contracts[entry['id']]=dict(source_observed=entry['request'].get('observed'),
            original_training_fingerprint=entry['request'].get('training_sha256'),array_hashes=contracts,
            canonical_array_contract_sha256=hashlib.sha256(json.dumps(contracts,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            scope='Numeric arrays reconstructed by load_entry after original observed-file and split verification; stored defective object-hash receipt is not used for parameters or constraints.')
        require(score['cells']==val.y.size,'Scoring history cells '+entry['id'])
        case=CustomerCase(io.read(run/'resource_configs'/(entry['case']+'.json')))
        H=np.asarray([v['H'] for v in score['parameters']]);history=np.array([v['history_sse'] for v in score['parameters']])
        bank=score['native_bank'];NH=np.asarray([v['H'] for v in bank]);NS=np.array([v['history_sse'] for v in bank]);target=score['map_index']
        power=np.c_[val.p,val.q];near(history,np.sum((val.y[None]-np.einsum('tc,knc->ktn',power,H))**2,axis=(1,2)),'validation_only_SSE',1e-11)
        means=calibrated_means(score,val)
        for day in range(p['days']):
            if limit_completed and counts['saved_days']>=limit_completed:break
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
            delivered_trust=io.read(d/'delivered_trust.json') if (d/'delivered_trust.json').exists() else {}
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
            trusted_cases={}
            if any(mode.startswith('pq_trust') for mode in plans):
                from .trusted_direction import PQTrustedCase
                trusted_cases={scale:PQTrustedCase(case,train,base_kw=generation['base_kw'],scale=scale) for scale in (1.,2.)}
                pkw=train.p*generation['base_kw'];qkw=train.q*generation['base_kw']
                independently_fitted_kappa=np.array([np.linalg.lstsq(pkw[:,i,None],qkw[:,i],rcond=None)[0][0] for i in range(case.N)])
                hourly_training_budget=case.H*np.square(qkw-pkw*independently_fitted_kappa[None]).mean(0).sum()
                for scale,trusted in trusted_cases.items():
                    near(trusted.kappa,independently_fitted_kappa,'independent_kW_regression_kappa',1e-12)
                    near(trusted.trust_limit,scale**2*hourly_training_budget,'independent_kW_training_budget',1e-9)
                for record in seed_record.get('records',[]):
                    if record.get('source')!='radial_trust_scaling':continue
                    scale=float(record['name'][2]);trusted=trusted_cases[scale]
                    parent=seed_points[record['parent']];info=trusted.trust_audit(parent)
                    factor=min(1.,np.sqrt(info['limit']/info['energy'])) if info['energy']>0 else 1.
                    near(record['scale'],factor,'trust_seed_radial_factor',1e-12)
                    near(seed_points[record['name']],factor*parent,'trust_seed_radial_action',1e-12)
                    require(trusted.audit(seed_points[record['name']])['passed'],'Radially scaled seed planning domain')
                    counts['trust_scaled_seed_replays']+=1
            for mode,plan in plans.items():
                u=planner[mode];counts['planned_policies']+=1
                counts['quote_exceptions']+=int(plan.get('status')=='exception')
                require(case.audit(u)['passed'],'Requested resource feasibility '+entry['id']+'/'+mode)
                planning_case=case
                if mode in ('pq_trust1','pq_trust2'):
                    planning_case=trusted_cases[float(mode[-1])]
                    requested=planning_case.trust_audit(u);executed=planning_case.trust_audit(u*background['response'][None])
                    if plan.get('feasible'):require(requested['passed'],'Returned action violates planning P/Q restriction')
                    if plan.get('trust'):
                        require(trust_snapshot.exists() and plan['trust']['source']['sha256']==io.ref(trust_snapshot)['sha256'],'Trust receipt source matches sealed implementation')
                        if legacy_object_hash:
                            metadata_defects.append(dict(id=entry['id'],day=day,mode=mode,field='training_pq_sha256',
                                issue='Locked implementation passed a dict to ndarray contract_hash; object-array bytes contain process-dependent pointers.',
                                declared_hash=plan['trust']['training_pq_sha256'],replacement_array_contract=training_contracts[entry['id']]['canonical_array_contract_sha256']))
                        else:require(planning_case.trust_receipt['training_pq_sha256']==plan['trust']['training_pq_sha256'],'Stable trust training source hash')
                        near(planning_case.kappa,plan['trust']['kappa'],'train_only_PQ_kappa',1e-13)
                        near(requested['limit'],plan['trust']['limit'],'train_only_PQ_limit',1e-10)
                    if mode in delivered_trust:
                        near(executed['energy'],delivered_trust[mode]['energy'],'delivered_PQ_departure',1e-10)
                        require(executed['passed']==delivered_trust[mode]['passed'],'Delivered planning restriction receipt')
                    else:require(False,'Missing delivered planning restriction audit')
                    trusted_records.append(dict(id=entry['id'],day=day,mode=mode,quote_feasible=bool(plan.get('feasible')),
                        requested_passed=requested['passed'],delivered_passed=executed['passed'],delivered_relative_usage=executed['relative_usage']))
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
                    evidence=evidence_for(mode,planning_case,score,val,planner['forecast'],generation['base_kw'],plan,entry['voltage_sigma'])
                    info=evidence.direct(u);delivered=evidence.direct(u*background['response'][None])
                    if plan.get('feasible'):require(info['feasible'],'Native recovery direct constraint')
                    truth_in=score['models'][target]['topology_id']==truth['truth_id']
                    lower=info['model_recovery_lower'];delivered_lower=delivered['model_recovery_lower']
                else:
                    evidence=evidence_for(mode,planning_case,score,val,planner['forecast'],generation['base_kw'],plan,entry['voltage_sigma'])
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
                        domain=planning_case.audit(point)['passed']
                        require(bool(domain)==bool(record.get('resource_feasible')),'Seed planning-domain feasibility '+mode)
                        # This existing receipt field records evidence-only
                        # feasibility; planning-domain feasibility is separate.
                        valid=(np.min(evidence.direct(point)-evidence.required(2.))>=-1e-7) if mode=='old_gamma2' else evidence.direct(point)['feasible']
                        require(bool(valid)==bool(record.get('qcqp_feasible')),'Seed evidence feasibility '+mode)
                        counts['seed_evidence_replays']+=1
                    name=initial.get('name')
                    if name in seed_points and initial.get('feasible'):
                        seed=seed_points[name]
                        require(planning_case.audit(seed)['passed'],'Selected initial satisfies complete planning domain')
                        initial_evidence=(np.min(evidence.direct(seed)-evidence.required(2.))>=-1e-7) if mode=='old_gamma2' else evidence.direct(seed)['feasible']
                        require(initial_evidence,'Selected initial satisfies evidence constraints')
                        if plan.get('source')=='verified_initial_retained':near(u,seed,'retained_seed_action',1e-10)
                        if plan.get('source')=='native_improved_initial':require(case.fee(u)<case.fee(seed)-1e-9,'Reported strict initial improvement')
                if mode!='old_gamma2' and plan.get('feasible'):
                    for physics in ('AC','linear'):
                        r=next((r for r in rows if r['strategy']==mode and r['physics']==physics),{})
                        certificates.append(dict(id=entry['id'],day=day,role=entry['role'],case=entry['case'],mode=mode,physics=physics,
                            truth_in_selected_set=truth_in,model_lower=lower,delivered_action_model_lower=delivered_lower,
                            representative_correct=bool(r.get('status')=='complete' and r.get('map_correct')),
                            native_correct=bool(r.get('status')=='complete' and r.get('native_map_correct'))))
            if plans.get('pq_trust1',{}).get('feasible') and 'pq_trust2' in plans:
                small,large=plans['pq_trust1'],plans['pq_trust2'];u1=planner['pq_trust1']
                require(trusted_cases[2.].audit(u1)['passed'],'One-scale action remains in two-scale planning domain')
                near(trusted_cases[1.].fee(u1),trusted_cases[2.].fee(u1),'Same action same fee across trust domains',1e-9)
                nested_evidence=True;minimum_pair_margin=None
                if len(H)>1:
                    e2=evidence_for('pq_trust2',trusted_cases[2.],score,val,planner['forecast'],generation['base_kw'],small,entry['voltage_sigma'])
                    nested_info=e2.direct(u1);nested_evidence=nested_info['feasible'];minimum_pair_margin=nested_info['minimum_pair_margin']
                    if large.get('targets') is not None:require(small['targets']==large['targets'],'Identical target sets across trust scales')
                require(nested_evidence,'One-scale action satisfies two-scale evidence constraints')
                cost1=case.fee(u1);cost2=case.fee(planner['pq_trust2']) if large.get('feasible') else None
                nested_trust_records.append(dict(id=entry['id'],day=day,one_scale_cost=cost1,two_scale_cost=cost2,
                    two_scale_costlier=cost2 is not None and cost2>cost1+1e-6,
                    excess_returned_cost=None if cost2 is None else max(0.,cost2-cost1),
                    two_scale_no_quote=not bool(large.get('feasible')),two_scale_status=large.get('status'),
                    two_scale_infeasibility_conflict=bool(large.get('infeasibility_proved')),
                    two_scale_bound_conflict=large.get('lower_bound_yuan') is not None and large['lower_bound_yuan']>cost1+1e-6,
                    known_feasible_two_scale_upper=min(cost1,cost2) if cost2 is not None else cost1,
                    minimum_pair_margin=minimum_pair_margin))
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
        histories_scanned=entry_index+1
        progress('running',histories_scanned,entry['id'])
    groups={}
    for mode,physics in sorted({(r['mode'],r['physics']) for r in certificates}):
        own=[r for r in certificates if r['mode']==mode and r['physics']==physics and r['model_lower']>=.9-1e-6]
        conditional=[r for r in own if r['truth_in_selected_set']]
        groups[mode+'/'+physics]=dict(feasible_model_passed=len(own),truth_in_selected_set=len(conditional),
            representative_wrong_with_truth_in_set=sum(not r['representative_correct'] for r in conditional),
            native_wrong_with_truth_in_set=sum(not r['native_correct'] for r in conditional),
            delivered_action_still_model90=sum(r['delivered_action_model_lower']>=.9-1e-6 for r in conditional),
            missing_condition_excluded=len(own)-len(conditional))
    trust_summary={mode:dict(checked=len(own),feasible_quotes=sum(v['quote_feasible'] for v in own),
        requested_violations=sum(not v['requested_passed'] for v in own),delivered_planning_restriction_violations=sum(not v['delivered_passed'] for v in own),
        maximum_delivered_relative_usage=max((v['delivered_relative_usage'] for v in own if v['delivered_relative_usage'] is not None),default=None),
        scope='Additional planning restriction only; actual physical-device audits use the unchanged original case.')
        for mode in sorted({v['mode'] for v in trusted_records}) if (own:=[v for v in trusted_records if v['mode']==mode])}
    result=dict(protocol=io.ref(out/'protocol.json'),counts=dict(counts),maximum_errors=dict(maxerr),resources=resource_summary,trusted_directions=trust_summary,
        nested_trust=dict(verified_one_scale_feasible_points=len(nested_trust_records),
            two_scale_costlier=sum(v['two_scale_costlier'] for v in nested_trust_records),
            two_scale_no_quote_with_known_feasible=sum(v['two_scale_no_quote'] for v in nested_trust_records),
            two_scale_infeasibility_conflicts=sum(v['two_scale_infeasibility_conflict'] for v in nested_trust_records),
            two_scale_bound_conflicts=sum(v['two_scale_bound_conflict'] for v in nested_trust_records),
            maximum_excess_returned_cost=max((v['excess_returned_cost'] for v in nested_trust_records if v['excess_returned_cost'] is not None),default=None),
            scope='Same evidence constraints and fees, nested planning domains. A worse returned quote or unresolved larger-domain solve reflects finite-budget optimization, not an increase in optimal procurement cost. Executed policies are unchanged.'),
        metadata_hash_defect=len(metadata_defects),metadata_hash_defect_scope='Original source-file hashes and independently reconstructed numeric-array hashes establish provenance. The original defective receipt is preserved and is not claimed reproducible. Numerical checks do not depend on that field.',
        coverage=dict(planned_days=len(entries)*p['days'],checked_days=counts['saved_days'],partial=counts['saved_days']!=len(entries)*p['days']),
        certificate_calibration=groups,negative_counterexamples_count=len(counterexamples),
        negative_counterexample_summary={mode+'/'+physics:dict(count=len(group),worst=min(r['p_true_change'] for r in group)) for mode,physics in sorted({(r['strategy'],r['physics']) for r in counterexamples}) if (group:=[r for r in counterexamples if r['strategy']==mode and r['physics']==physics])},
        passed=not errors,numerical_passed=not errors,errors=errors,seed_verification_limit='Saved arrays were replayed for resource feasibility, fee and consumer evidence constraints where present. Days without seed arrays permit only cross-log named costs and selected-initial receipts; no LPs were rerun.',
        proof_scope='Native infeasible status is a solver proof about its recorded QCQP constraints. Unresolved means no certified action in budget. This audit does not establish original chance-problem infeasibility.')
    io.write(out/'certificate_private.json',certificates);io.write(out/'negative_counterexamples_private.json',counterexamples);io.write(out/'trusted_directions_private.json',trusted_records)
    io.write(out/'metadata_hash_defects_private.json',metadata_defects);io.write(out/'training_array_hash_sidecar.json',training_contracts);io.write(out/'summary.json',result)
    io.write(out/'nested_trust_private.json',nested_trust_records)
    progress('complete',histories_scanned)
    return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--completed-only',action='store_true');a.add_argument('--limit-completed',type=int,default=0);a.add_argument('--history-id');args=a.parse_args()
    result=verify(args.run.resolve(),args.out.resolve(),completed_only=args.completed_only,limit_completed=args.limit_completed,history_id=args.history_id);print('numerical_passed',result['passed'],'coverage',result['coverage'],'counts',result['counts'],'errors',len(result['errors']),'metadata_hash_defect',result['metadata_hash_defect'])
