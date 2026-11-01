"""Independent five-case planning replay; no solver, actual packet or truth.

This is a post-freeze external audit. It does not call the optimizer, fitter,
design adapter or physical producer. All model matrices and gains are rebuilt
from archived historical points, observed P/Q and physical schedule DTOs.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,os,pickle,traceback
from pathlib import Path

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import numpy as np

EXPECTED=['paper15_b2_v0005','flynn16_b4_v0005','flynn16_b5_v0002','pengwah18_b5_v0002','flynn16_b7_v0002']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_domain(problem,price):
    price=np.asarray(price,float);lam=np.asarray(problem['elasticities'],float)
    D=np.asarray(problem['response_factors'],float);q=np.abs(D).sum(axis=1)
    A=np.asarray(problem['device_matrix'],float);rhs=np.asarray(problem['device_rhs'],float)
    e=np.asarray(problem['reduction_hours'],float)
    caps=np.minimum(np.asarray(problem['price_upper'],float),np.asarray(problem['actual_amplitude_upper'],float)/((1+q)*lam))
    amplitude=lam*price
    device=A@amplitude+np.abs((A*amplitude)@D).sum(axis=1)
    violation=max(0.,float(np.max(-price,initial=0.)),float(np.max(price-caps,initial=0.)),
        float(np.max(device-rhs,initial=0.)),float((e*lam*(1+q))@(price*price)-problem['budget']),
        max((np.count_nonzero(amplitude[g]>0)-1 for g in problem['exclusive_groups']),default=0))
    return dict(maximum_violation=violation,expected_payment=float((e*lam)@(price*price)),
        hard_payment=float((e*lam*(1+q))@(price*price)),caps=caps,amplitude=amplitude)


def paired_gain(base,columns,quote,normalizer):
    return (np.abs(base+columns@quote)-np.abs(base)).sum()/normalizer


def lower_quantile(gains,probability=.9):
    gains=np.asarray(gains,float);required=int(np.ceil(probability*len(gains)-1e-12))
    return float(np.sort(gains)[len(gains)-required])


def contracted_supports(point,calibration):
    threshold=.16*float(calibration['median_root_depth'])
    return [block for block,r,x in zip(point['supports'],point['r'],point['x'],strict=True)
        if len(block)==1 or calibration['a']*r+calibration['b']*x+1e-10>=threshold]


def final_focuses(output_classes,labels,raw_added):
    added=[[labels[i] for i in b] for b in sorted(output_classes[1]-output_classes[0])]
    removed=[[labels[i] for i in b] for b in sorted(output_classes[0]-output_classes[1])]
    chosen_added=added if added or removed else raw_added
    focuses=[list(b) for b in sorted({tuple(sorted(b)) for b in chosen_added+removed},key=lambda b:(len(b),b))]
    return chosen_added,removed,focuses


def run(root):
    root=Path(root).resolve();out=root/'independent_planning_audit.json'
    if out.exists():raise FileExistsError('retain original independent planning audit')
    checks=[];opened=[];results=[]
    def check(name,condition):
        checks.append(dict(name=name,passed=bool(condition)))
        if not condition:raise AssertionError(name)
    def safe(path):
        path=Path(path).resolve();normalized=str(path).replace('\\','/').lower()
        if '/actual/' in normalized or 'truth_evaluation' in normalized or path.name.startswith('test_'):
            raise RuntimeError('planning audit forbids actual/test/truth reads: '+str(path))
        return path
    def read(path):
        path=safe(path);opened.append(str(path));return json.loads(path.read_text(encoding='utf-8'))
    def checked(item):
        path=safe(item['path']);check('hash_'+path.name,sha(path)==item['sha256']);return path
    def unpickle(path):
        path=safe(path);opened.append(str(path))
        with path.open('rb') as stream:return pickle.load(stream)
    def close(name,a,b,atol=1e-12):check(name,bool(np.allclose(a,b,rtol=0,atol=atol)))
    audit=dict(status='started',created_utc=datetime.now(timezone.utc).isoformat(),
        new_solver_calls=0,new_parameter_fits=0,new_AC=0,actual_observation_reads=0,true_structure_reads=0,
        source_scope='post-freeze external arithmetic audit; not pre-experiment source logic')
    try:
        p=read(root/'protocol.json');end=read(root/'preparation_ended.json');f=read(checked(end['price_freeze']))
        checked(f['protocol']);checked(f['source_snapshot']);checked(f['planning_summary'])
        for file,h in p['source_sha256'].items():check('source_'+Path(file).name,sha(file)==h)
        check('five_predeclared_conditions',len(p['cases'])==5 and [c['name'] for c in p['cases']]==EXPECTED)
        check('three_networks',len({c['case'] for c in p['cases']})==3)
        check('unchanged_whole_history_and_budget',p['full_history_rows']==[288,288] and p['budget']==1.6)
        check('samples64_holdout1000',p['planning_samples']==64 and p['holdout_samples']==1000)
        check('all_cases_retained_in_price_freeze',set(f['cases'])=={c['name'] for c in p['cases'] if c['status']=='qualified'})
        planning=read(root/'planning_summary.json');tables={c['condition']:c for c in planning['cases']}
        holdout=read(root/'model_holdout/summary.json');checked(holdout['price_freeze'])
        holdouts={c['condition']:c for c in holdout['cases']}
        for spec in p['cases']:
            name=spec['name'];check(name+'_qualification_recorded',spec['status'] in ('qualified','qualification_failed'))
            if spec['status']!='qualified':
                results.append(dict(condition=name,status=spec['status'],denominator_retained=True));continue
            c=read(checked(spec['case_protocol']));pool=read(checked(spec['pool']));runtime=read(checked(spec['runtime_protocol']))
            problem=read(checked(spec['problem']));qual=read(checked(spec['qualification']))
            check(name+'_same_history_bindings',pool['history_packet']==c['history_packet'] and pool['historical_fit']==c['historical_baseline_result'])
            history=unpickle(checked(c['history_packet']))['history'];old=read(checked(c['historical_baseline_result']))
            payload={k:v for k,v in pool.items() if k!='pool_sha256'}
            check(name+'_pool_fingerprint',hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==pool['pool_sha256'])
            labels=pool['labels'];n=len(labels);N=96*n;matrices=[];output_classes=[]
            check(name+'_original_root_base_labels',history['labels']==labels==c['labels'] and history['root_label']==c['root_label']==pool['root_label'] and history['base_kw']==c['base_kw']==pool['base_kw'])
            check(name+'_unchanged_calibration',pool['calibration']==old['calibrations']['raw'])
            check(name+'_frozen_order',pool['selection_order']==['coarse','refined'])
            days=history['training']+history['validation']
            check(name+'_full288_288',len(history['training'])==len(history['validation'])==3 and all(len(d['P_terminal'])==96 for d in days))
            P=np.stack([d['P_terminal'].to_numpy(float) for d in days]);Q=np.stack([d['Q_terminal'].to_numpy(float) for d in days])
            check(name+'_six_complete_days',P.shape==Q.shape==(6,96,n))
            Y=np.stack([d['V_root_observed'].to_numpy(float).reshape(-1,1)**2-d['V_terminal'].to_numpy(float)**2 for d in days])
            parameter_rows=[]
            for model in pool['models']:
                point=old['selection_candidates'][model['historical_index']]
                check(name+'_point_unchanged_'+model['candidate_id'],point==model['point'])
                rx=[]
                for key in ('r','x'):
                    weights=np.asarray(point[key],float)
                    check(name+'_zero_weak_kept_'+model['candidate_id']+'_'+key,np.all((weights>=0)&(weights<=2)))
                    check(name+'_weights_bytes_'+model['candidate_id']+'_'+key,weights.tobytes()==np.asarray(model['point'][key],float).tobytes())
                    matrix=np.zeros((n,n))
                    for support,w in zip(point['supports'],weights,strict=True):
                        indicator=np.zeros(n);indicator[support]=1.;matrix+=float(w)*np.outer(indicator,indicator)
                    check(name+'_matrix_bytes_'+model['candidate_id']+'_'+key,matrix.tobytes()==np.asarray(model[key.upper()],float).tobytes())
                    rx.append(matrix)
                prediction=P@rx[0].T+Q@rx[1].T
                close(name+'_original_training_loss_'+model['candidate_id'],np.abs(prediction[:3]-Y[:3]).mean(),point['objective'],1e-8)
                close(name+'_original_validation_loss_'+model['candidate_id'],np.abs(prediction[3:]-Y[3:]).mean(),point['validation_mae'],1e-10)
                final_supports=contracted_supports(point,pool['calibration'])
                check(name+'_original_contract_'+model['candidate_id'],final_supports==model['frozen_output_supports'])
                output_classes.append(frozenset(tuple(sorted(b)) for b in final_supports))
                matrices.append(rx);parameter_rows.append(dict(candidate_id=model['candidate_id'],historical_index=model['historical_index'],unchanged=True))
            loss={m['candidate_id']:m['historical_validation_mae'] for m in pool['models']};best=min(loss.values())
            historical_eligible=[k for k in pool['selection_order'] if loss[k]<=1.03*best+1e-12]
            check(name+'_historical_gate',historical_eligible==spec['history_eligible_ids'])
            final_distinct=len(set(output_classes))==2
            if 'final_output_distinct' in spec:check(name+'_final_dispute_flag',final_distinct==spec['final_output_distinct'])
            # Audit the repaired objective's structural scope using all old
            # raw candidates, without calling its selector or reading truth.
            raw=[(i,v) for i,v in enumerate(old['selection_candidates']) if v.get('preprocess','raw')=='raw']
            raw_best=min(v['validation_mae'] for _,v in raw);cutoff=1.03*raw_best+1e-12
            eligible_raw=[(i,v) for i,v in raw if v['validation_mae']<=cutoff]
            base_index=old['models']['multistart_original_rule']['selected_uncontracted']['selection_index']
            base_point=old['selection_candidates'][base_index]
            base_class=frozenset(tuple(sorted(b)) for b in contracted_supports(base_point,pool['calibration']))
            alternative=[]
            for i,v in eligible_raw:
                family=frozenset(tuple(sorted(b)) for b in contracted_supports(v,pool['calibration']))
                if i!=base_index and family!=base_class:alternative.append((i,v,family))
            rule=pool['competition_rule']
            if 'no_final_dispute' in rule:
                check(name+'_all_eligible_final_classes_checked',rule['eligible_distinct_final_indices']==[i for i,_,_ in alternative])
                check(name+'_no_dispute_semantics',rule['no_final_dispute']==(not alternative))
                check(name+'_actual_selected_point_kept',rule['selected_index']==base_index==pool['models'][0]['historical_index'])
                if alternative:
                    path=base_point.get('path_index')
                    nexts=[(i,v) for i,v in raw if path is not None and v.get('origin')==base_point.get('origin') and v.get('path_index')==path+1 and len(v['supports'])>len(base_point['supports'])]
                    next_index=min(nexts,key=lambda v:v[0])[0] if nexts else None
                    if next_index in [i for i,_,_ in alternative]:expected_index=next_index
                    else:expected_index=min(alternative,key=lambda v:(len(base_class^v[2]),len(v[1]['supports']),v[1]['validation_mae'],v[0]))[0]
                    check(name+'_history_only_distinct_competitor',expected_index==rule['alternative_index']==pool['models'][1]['historical_index'])
            units=qual['units_per_resource'];lam=np.asarray(problem['elasticities'],float);J=len(lam)
            check(name+'_inventory1to5',isinstance(units,int) and 1<=units<=5)
            check(name+'_inventory_lambda',J==10 and np.all(lam==.1*units))
            check(name+'_physical_inventory_declaration',qual['baseline_kw']==.2*units and qual['power_upper_kw']==.4*units and qual['actual_modulation_cap_kw']==.2*units and qual['prefix_energy_kwh']==.4*units)
            check(name+'_case_noise_preserved',runtime['voltage_sigma']==c['voltage_sigma'] and runtime['pq_sigma']==c['pq_sigma']==.005)
            D=np.asarray(problem['response_factors'],float);q=np.abs(D).sum(axis=1)
            close(name+'_response_factors',D,np.array([[.15,.05]]*5+[[.15,-.05]]*5))
            close(name+'_relative_range',q,np.full(J,.2))
            H=np.asarray(problem['profiles'],float);Hp=H[:N].reshape(96,n,J);Hq=H[N:].reshape(96,n,J)
            check(name+'_full_PQ_profiles',H.shape==(2*N,J))
            e=np.asarray(problem['reduction_hours'],float);base_kw=c['base_kw'];r0,x0=matrices[0];r1,x1=matrices[1]
            # Reconstruct the historical resource qualification independently.
            def family(model):
                point=model['point'];supports=point.get('full_fit',{}).get('supports',point['supports'])
                return {tuple(sorted(labels[i] for i in block)) for block in supports}
            common=family(pool['models'][0])&family(pool['models'][1])
            column=np.abs((r1-r0)+.35*(x1-x0)).sum(axis=0)/(n*base_kw)*1e6
            pmin=P.min(axis=(0,1))*base_kw;qmin=Q.min(axis=(0,1))*base_kw
            focus_added,focus_removed,focuses=final_focuses(output_classes,labels,pool['dispute_added_supportlabels'])
            resource_choice=None
            for inventory in range(5,0,-1):
                options=[]
                for focus_labels in focuses:
                    focus=set(focus_labels);containers=[set(block) for block in common if focus<set(block)]
                    if not containers:continue
                    region=min(containers,key=lambda x:(len(x),tuple(sorted(x))))
                    eligible=[i for i in range(n) if pmin[i]>=.2*inventory and qmin[i]>=.07*inventory]
                    inside=[i for i in eligible if labels[i] in focus];outside=[i for i in eligible if labels[i] in region-focus]
                    if not inside or not outside:continue
                    selected=[max(part,key=lambda i:(column[i],-labels[i])) for part in (inside,outside)]
                    options.append((sum(column[i] for i in selected),tuple(sorted(focus)),selected))
                if options:
                    resource_choice=(inventory,min(options,key=lambda v:(-v[0],v[1])));break
            check(name+'_historical_pair_exists',resource_choice is not None)
            chosen_units,(_,chosen_focus,chosen_indices)=resource_choice
            check(name+'_largest_feasible_inventory',chosen_units==units)
            check(name+'_historical_only_focus',list(chosen_focus)==qual['dispute_labels'])
            check(name+'_historical_only_pair',[labels[i] for i in chosen_indices]==qual['selected_labels'])
            check(name+'_final_disputes_localized',qual['all_added_disputes']==focus_added and qual['all_removed_disputes']==(focus_removed or pool['dispute_removed_supportlabels']))
            check(name+'_raw_disputes_retained',qual['raw_added_disputes']==pool['dispute_added_supportlabels'] and qual['raw_removed_disputes']==pool['dispute_removed_supportlabels'])
            base_delta=P@r1.T+Q@x1.T-(P@r0.T+Q@x0.T)
            columns=np.column_stack([((Hp[:,:,j]/base_kw)@r1.T+(Hq[:,:,j]/base_kw)@x1.T-(Hp[:,:,j]/base_kw)@r0.T-(Hq[:,:,j]/base_kw)@x0.T).ravel() for j in range(J)])
            plans={k:read(checked(v)) for k,v in f['cases'][name]['plans'].items()}
            check(name+'_all3policies',set(plans)==set(p['policies'])=={'zero','rotation','optimized'})
            for policy,plan in plans.items():
                prices=np.asarray(plan['prices'],float);domain=original_domain(problem,prices)
                check(name+'_original_domain_'+policy,domain['maximum_violation']<=1e-7)
                close(name+'_plan_amplitude_'+policy,np.asarray(plan['amplitudes']),lam*prices)
                close(name+'_expected_fee_'+policy,domain['expected_payment'],plan['expected_payment'])
                close(name+'_hard_fee_'+policy,domain['hard_payment'],plan['hard_payment_bound'])
                close(name+'_plan_factors_'+policy,plan['response_factors'],D)
                close(name+'_plan_lambda_'+policy,plan['elasticities'],lam)
                resources={r['resource_id']:r for r in plan['resources']};pp=np.zeros_like(Hp);qq=np.zeros_like(Hq)
                check(name+'_2_resources_'+policy,len(resources)==2)
                for rid,resource in resources.items():
                    idx=labels.index(resource['node_label'])
                    check(name+'_contained_component_'+policy+'_'+rid,P[:,:,idx].min()*base_kw>=resource['baseline_kw'] and Q[:,:,idx].min()*base_kw>=.35*resource['baseline_kw'])
                    check(name+'_resource_baseline_'+policy+'_'+rid,resource['baseline_kw']==.2*units and resource['power_upper_kw']==.4*units)
                for j,module in enumerate(plan['modules']):
                    rid,shape=next(iter(module['profiles'].items()));shape=np.asarray(shape,float);idx=labels.index(resources[rid]['node_label'])
                    pp[:,idx,j]=shape;qq[:,idx,j]=.35*shape
                    check(name+'_energy_neutral_'+policy+'_'+str(j),abs(.25*shape.sum())<=1e-12)
                    close(name+'_duration_'+policy+'_'+str(j),.25*np.maximum(-shape,0).sum(),e[j])
                check(name+'_physical_Hp_bytes_'+policy,pp.tobytes()==Hp.tobytes())
                check(name+'_physical_Hq_bytes_'+policy,qq.tobytes()==Hq.tobytes())
            folder=root/'cases'/name;draws=read(folder/'planning_draws.json')['draws'];stored=unpickle(folder/'planning_scenarios.pkl')
            rng=np.random.default_rng(spec['planning_seed']);B=[];C=[]
            check(name+'_64draws',len(draws)==len(stored)==64)
            for i,draw in enumerate(draws):
                day=int(rng.integers(6));z=rng.uniform(-1,1,2);b=base_delta[day].ravel();a=columns*(lam*(1+D@z))
                check(name+'_draw_day_'+str(i),day==draw['historical_day'])
                check(name+'_draw_factor_'+str(i),np.array_equal(z,np.asarray(draw['factors'])))
                close(name+'_affine_base_'+str(i),b,stored[i]['base'],1e-15)
                close(name+'_affine_columns_'+str(i),a,stored[i]['price_coeff'],1e-15)
                check(name+'_normalizer_'+str(i),stored[i]['normalizer']==N);B.append(b);C.append(a)
            B=np.stack(B);C=np.stack(C);optimization=read(checked(f['cases'][name]['optimization']))
            reused_from=f['cases'][name].get('reused_quote_from')
            if reused_from:
                old_opt_path=checked(reused_from);old_opt=read(old_opt_path);old_folder=old_opt_path.parent
                check(name+'_reused_quote_JSON_unchanged',old_opt==optimization)
                check(name+'_reused_problem_bytes',sha(old_folder/'problem.json')==sha(folder/'problem.json'))
                check(name+'_reused_scenario_bytes',sha(old_folder/'planning_scenarios.pkl')==sha(folder/'planning_scenarios.pkl'))
                check(name+'_reused_draws_bytes',sha(old_folder/'planning_draws.json')==sha(folder/'planning_draws.json'))
                old_pool=read(old_folder/'frozen_pool.json')
                check(name+'_reused_calibration',old_pool['calibration']==pool['calibration'])
                for a,b in zip(old_pool['models'],pool['models'],strict=True):
                    check(name+'_reused_point_'+b['candidate_id'],a['point']==b['point'])
                    for key in ('R','X'):check(name+'_reused_matrix_bytes_'+b['candidate_id']+key,np.asarray(a[key],float).tobytes()==np.asarray(b[key],float).tobytes())
            gamma=f['cases'][name]['gamma_native'];planning_table=[]
            if np.any(plans['optimized']['prices']):check(name+'_paid_objective_has_final_dispute',final_distinct and len(historical_eligible)==2)
            for row in tables[name]['table']:
                policy=row['policy'];price=np.asarray(plans[policy]['prices']);shift=np.einsum('sij,j->si',C,price)
                values=(np.abs(B+shift)-np.abs(B)).sum(axis=1)/N;quantile=lower_quantile(values)*1e6
                close(name+'_quantile_'+policy,quantile,row['quantile_micro'],1e-8)
                check(name+'_positive_count_'+policy,int((values>1e-12).sum())==row['positive_passes'])
                planning_table.append(dict(policy=policy,denominator=64,positive_count=int((values>1e-12).sum()),
                    target_count=int((values>=gamma-1e-12).sum()) if gamma>1e-12 else None,
                    quantile_micro=quantile,expected_payment=original_domain(problem,price)['expected_payment']))
                if policy=='optimized':
                    close(name+'_gamma_replay',lower_quantile(values),gamma,1e-14)
                    if 'scenario_gain_native' in optimization:close(name+'_saved_gains',values,optimization['scenario_gain_native'],1e-14)
            warm_best=0.;warm_fee=0.;warm_index=None;warm_count=0;groups=problem['exclusive_groups']
            domain0=original_domain(problem,np.zeros(J));caps=domain0['caps'];beta=e*lam*(1+q);warm=[np.zeros(J)]
            if optimization.get('solver_calls',0):
                for a in groups[0]:
                    for b in groups[1]:
                        for angle in np.linspace(0,np.pi/2,9):
                            for radius in (.5,1.):
                                quote=np.zeros(J);quote[a]=min(caps[a],radius*np.sqrt(problem['budget']/beta[a])*np.cos(angle))
                                quote[b]=min(caps[b],radius*np.sqrt(problem['budget']/beta[b])*np.sin(angle));quote[np.abs(quote)<1e-12]=0.;warm.append(quote)
            for i,quote in enumerate(warm):
                domain=original_domain(problem,quote);check(name+'_warm_domain_'+str(i),domain['maximum_violation']<=1e-7)
                values=(np.abs(B+np.einsum('sij,j->si',C,quote))-np.abs(B)).sum(axis=1)/N;v=lower_quantile(values)*1e6
                if v>warm_best+1e-8 or (abs(v-warm_best)<=1e-8 and domain['expected_payment']<warm_fee):
                    warm_best=v;warm_fee=domain['expected_payment'];warm_index=i
                warm_count+=1
            if optimization.get('solver_calls',0):
                check(name+'_451warm_verified',warm_count==optimization['provided_warm_start_count']==optimization['verified_warm_start_count']==451)
                close(name+'_warm_quantile',warm_best,optimization['warm_start_best_quantile_micro'],1e-8)
                check(name+'_warm_index',warm_index==optimization['warm_start_best_index'])
                check(name+'_58required',optimization['required_scenarios']==58)
                primary=optimization['primary'];check(name+'_bound_field_preserved','best_bound' in primary)
                if primary['best_bound'] is not None:
                    check(name+'_bound_not_below_incumbent',primary['best_bound']>=gamma*1e6-1e-5 or not optimization.get('feasible_incumbent'))
                check(name+'_status_semantics',primary['solver_reported_optimal']==(primary['status_code']==2))
                if optimization.get('feasible_incumbent'):
                    close(name+'_returned_price',optimization['prices'],plans['optimized']['prices'])
                    seed_equal=np.array_equal(np.asarray(optimization['prices']),np.asarray(optimization['warm_start_best_prices']))
                else:
                    check(name+'_failed_optimizer_zero_fallback',f['cases'][name]['planning_status']=='optimizer_failed_zero_fallback' and not np.any(plans['optimized']['prices']))
                    seed_equal=None
            else:
                status=f['cases'][name]['planning_status']
                legitimate=(len(historical_eligible)==1 and status=='one_historical_eligible_no_purchase') or (not final_distinct and status=='no_final_structure_dispute_no_purchase')
                check(name+'_explicit_no_purchase',legitimate and not np.any(plans['optimized']['prices']) and gamma==0);seed_equal=None
            held={k:[] for k in plans};rng=np.random.default_rng(spec['holdout_seed'])
            for i in range(p['holdout_samples']):
                day=int(rng.integers(6));z=rng.uniform(-1,1,2);b=base_delta[day].ravel();a=columns*(lam*(1+D@z))
                for policy,plan in plans.items():held[policy].append(paired_gain(b,a,np.asarray(plan['prices']),N))
            held_table=[]
            check(name+'_holdout_seed',holdouts[name]['seed']==spec['holdout_seed'])
            for row in holdouts[name]['table']:
                values=np.asarray(held[row['policy']]);positive=int((values>1e-12).sum());target=int((values>=gamma-1e-12).sum()) if gamma>1e-12 else None
                check(name+'_holdout_denominator_'+row['policy'],row['denominator']==len(values)==1000)
                check(name+'_holdout_positive_'+row['policy'],positive==row['positive_count'])
                check(name+'_holdout_target_'+row['policy'],target==row['target_count'])
                close(name+'_holdout_mean_'+row['policy'],values.mean(),row['gain_mean'],1e-14)
                close(name+'_holdout_p10_'+row['policy'],np.quantile(values,.1),row['gain_p10'],1e-14)
                held_table.append(dict(policy=row['policy'],denominator=1000,positive_count=positive,target_count=target,
                    mean_gain=float(values.mean()),p10_gain=float(np.quantile(values,.1))))
            results.append(dict(condition=name,status='complete',inventory_units=units,historical_eligible_ids=historical_eligible,
                final_output_distinct=final_distinct,eligible_distinct_final_indices=[i for i,_,_ in alternative],
                planning_status=f['cases'][name]['planning_status'],quote_reused_from=reused_from,
                parameter_replay=parameter_rows,planning_table=planning_table,holdout_table=held_table,gamma_native=gamma,
                independent_domain_probe_count=warm_count,optimizer_reported_warm_start_count=optimization.get('provided_warm_start_count',0),
                new_case_optimizer_calls=0 if reused_from else optimization.get('solver_calls',0),
                warm_best_quantile_micro=warm_best,returned_quote_equals_best_warm_start=seed_equal,
                optimizer_primary=optimization.get('primary'),optimizer_secondary=optimization.get('secondary'),
                optimized_original_domain={k:v for k,v in original_domain(problem,plans['optimized']['prices']).items() if k not in ('caps','amplitude')}))
        check('all5condition_denominator',len(results)==5)
        check('no_actual_test_truth_reads',all('/actual/' not in v.replace('\\','/').lower() and 'truth_evaluation' not in v.lower() for v in opened))
        audit.update(status='complete',check_count=len(checks),passed=sum(c['passed'] for c in checks),checks=checks,cases=results,
            condition_denominator=5,network_count=3,qualified_conditions=sum(c['status']=='complete' for c in results),
            failed_or_unresolved_conditions=sum(c['status']!='complete' for c in results),read_artifact_count=len(opened),source_sha256=sha(__file__),
            scope='all fitted parameters frozen; finite working-law checks only, no recovery or population probability certificate')
    except Exception as error:
        audit.update(status='failed',check_count=len(checks),passed=sum(c['passed'] for c in checks),checks=checks,cases=results,
            error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),source_sha256=sha(__file__))
        out.write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');raise
    out.write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=audit['status'],checks=audit['check_count'],passed=audit['passed'],cases=[dict(condition=c['condition'],status=c['status'],inventory=c.get('inventory_units'),gamma_micro=c.get('gamma_native',0.)*1e6,seed_equal=c.get('returned_quote_equals_best_warm_start')) for c in results])))
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();run(args.root)
