"""Scalar-only whitelist; raw load, voltage, AC and fitted R/X stay private."""
from pathlib import Path
import argparse,json,hashlib
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def export(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    p=read(root/'protocol.json');s=read(root/'summary.json');f=read(root/'price_freeze.json');rows=s['rows']
    rule_ids=[e['rule_id'] for r in rows if r['status']=='complete' for e in r['evaluations']]
    rule_ids=list(dict.fromkeys(rule_ids));tables=[];aggregate=[]
    def metrics(group,rid,denom):
        values=[next(e for e in r['evaluations'] if e['rule_id']==rid) for r in group if r['status']=='complete']
        errors=[next(e for e in r['evaluations'] if e['rule_id']==rid)['full_errors'] if r['status']=='complete' else None for r in group]
        return dict(denominator=denom,complete=len(values),unresolved=denom-len(values),
            errors=errors,error_sum=sum(v['full_errors'] for v in values),
            missing_sum=sum(len(v['missing']) for v in values),extra_sum=sum(len(v['extra']) for v in values),
            recoveries=sum(v['full_errors']==0 for v in values))
    for c in p['cases']:
        for rid in rule_ids:
            groups={pol:[r for r in rows if r['condition']==c['name'] and r['policy']==pol] for pol in p['policies']}
            for pol,group in groups.items():
                improvements=[];rotation_difference=[];rule_difference=[]
                for r in group:
                    z=next(x for x in groups['zero'] if x['path']==r['path']);b=next(x for x in groups['rotation'] if x['path']==r['path'])
                    if r['status']==z['status']==b['status']=='complete':
                        e=next(e for e in r['evaluations'] if e['rule_id']==rid)['full_errors']
                        ez=next(e for e in z['evaluations'] if e['rule_id']==rid)['full_errors'];eb=next(e for e in b['evaluations'] if e['rule_id']==rid)['full_errors']
                        old=next(e for e in r['evaluations'] if e['rule_id']=='original_ungated_w1_tau3pct')['full_errors']
                        improvements.append(ez-e);rotation_difference.append(eb-e);rule_difference.append(old-e)
                tables.append(dict(condition=c['name'],case=c['case'],policy=pol,rule_id=rid,**metrics(group,rid,4),
                    paired_complete=len(improvements),DR_error_reductions=improvements,rotation_error_reductions=rotation_difference,rule_error_reductions=rule_difference))
    for rid in rule_ids:
        for pol in p['policies']:
            group=[r for r in rows if r['policy']==pol]
            aggregate.append(dict(rule_id=rid,policy=pol,**metrics(group,rid,20)))
    configs=[];quotes=[];statuses=[]
    for c in p['cases']:
        inputs=read(c['case_protocol']['path']);pool=read(c['pool']['path'])
        q=read(c['qualification']['path']) if c['status']=='qualified' else {}
        configs.append(dict(name=c['name'],case=c['case'],status=c['status'],voltage_sigma=inputs['voltage_sigma'],pq_sigma=inputs['pq_sigma'],
            labels=inputs['labels'],base_kw=inputs['base_kw'],root_label=inputs['root_label'],
            inventory_N=q.get('units_per_resource'),selected_nodes=q.get('selected_labels'),focus=q.get('dispute_labels'),
            all_added_disputes=q.get('all_added_disputes'),all_removed_disputes=q.get('all_removed_disputes'),
            history_eligible_ids=c['history_eligible_ids'],planning_seed=c['planning_seed'],holdout_seed=c['holdout_seed'],response_seed=c['response_seed'],
            paths=c['paths'],profiles=c['profiles'],models=[dict(candidate_id=m['candidate_id'],raw_count=m['raw_support_count'],output_count=m['frozen_output_support_count'],
                historical_index=m['historical_index'],historical_validation_mae=m['historical_validation_mae']) for m in pool['models']]))
        if c['name'] not in f['cases']:continue
        case=f['cases'][c['name']];o=read(case['optimization']['path'])
        statuses.append(dict(condition=c['name'],status=case['planning_status'],gamma_native=case['gamma_native'],primary=o.get('primary'),secondary=o.get('secondary'),
            returned_equals_warm=o.get('prices')==o.get('warm_start_best_prices') if o.get('warm_start_best_prices') is not None else None,
            feasible_incumbent=o.get('feasible_incumbent'),elapsed_seconds=o.get('elapsed_seconds')))
        for pol,item in case['plans'].items():
            plan=read(item['path']);quotes.append(dict(condition=c['name'],policy=pol,prices=plan['prices'],amplitudes=plan['amplitudes'],
                selected_types=[m['module_id'] for m,a in zip(plan['modules'],plan['amplitudes']) if a>0],
                expected_cost=plan['expected_payment'],hard_budget_bound=plan['hard_payment_bound']))
    write(out/'configuration.json',dict(conditions=configs,primary_rule=p['primary_rule'],actual_validation_rows=384,new_day_weights=[1,2,4],
        new_day_tolerances=[.03,.02,.01],history_tolerance=.03,history_rows=[288,288],new_day_rows=96,hard_budget=1.6,
        predeclared_case_days=20,physical_networks=3,planning_samples=64,holdout_samples=1000,solver_seconds=60,replicate_base=p['replicate_base']))
    write(out/'planning.json',dict(status='complete',quotes=quotes,solver_statuses=statuses,holdout=read(root/'model_holdout/summary.json')['cases']))
    record_keys=['condition','case','policy','path','record_id','status','full_errors','missing','extra','selected_candidate','actual_payment',
        'new_distance','distance_gain','distance_target_met','paired_DR_error_reduction','rule_error_reduction','candidate_errors','correct_candidate_ids']
    public_rows=[]
    for row in rows:
        item={k:row[k] for k in record_keys if k in row}
        if row['status']=='complete':item['rule_scores']=[{k:e[k] for k in ['rule_id','selected_candidate','full_errors','missing','extra','exit_margins'] if k in e} for e in row['evaluations']]
        public_rows.append(item)
    main_tables=[]
    for item in s['table']:
        x=dict(item);x['model_charge_mean_including_failed']=x.pop('mean_payment')
        g=[r for r in rows if r['condition']==x['condition'] and r['policy']==x['policy']]
        valid_fees=[r['actual_payment'] for r in g if r['status']=='complete']
        x['valid_mean_payment']=sum(valid_fees)/len(valid_fees) if valid_fees else None
        x['failed_undelivered_model_charge']=sum(r.get('actual_payment',0.) or 0. for r in g if r['status']!='complete')
        main_tables.append(x)
    write(out/'summary.json',dict(status='complete',main_table=main_tables,records=public_rows,requested=60,complete=s['complete'],failed_or_unresolved=60-s['complete'],
        conditions=5,networks=3,predeclared_case_days=20,new_fit_calls=0,new_LP=0,new_search_calls=0,AC_attempted_points=s['AC_attempted_points'],valid_AC_points=s['valid_AC_points']))
    write(out/'rules.json',dict(status='complete',case_tables=tables,aggregate=aggregate))
    audits=[]
    for file in ['independent_planning_audit.json','independent_selection_audit.json','independent_physical_audit_v2.json']:
        a=read(root/file);audits.append(dict(file=file,sha256=sha(root/file),
            status=a.get('status'),checks=a.get('check_count',a.get('checks_count',len(a.get('checks',[])))),failures=a.get('failure_count',len(a.get('failures',[])))))
    first_audit=read(root/'independent_physical_audit.json')
    write(out/'audits.json',dict(status='complete',audits=audits,total_final_checks=sum(a['checks'] for a in audits),pure_tests=81,
        retained_initial_audit_failure=dict(status=first_audit['status'],error_type=first_audit['error_type'],reason='audit treated design.pkl as JSON; v2 only hash-checks it',sha256=sha(root/'independent_physical_audit.json')),
        raw_trajectories_and_RX_private=True))
    write(out/'physical_supplement.json',read(root/'independent_physical_supplement.json'))
    examples=[]
    for name,path,rid in [('paper15_b2_v0005',1501,'w1_tau2pct'),('flynn16_b4_v0005',1512,'w2_tau1pct')]:
        group=[]
        for pol in p['policies']:
            row=next(r for r in rows if r['condition']==name and r['path']==path and r['policy']==pol)
            e=next(e for e in row['evaluations'] if e['rule_id']==rid)
            group.append(dict(policy=pol,coarse_exit_margin_micro=e['exit_margins']['coarse']*1e6,
                full_errors=e['full_errors'],selected=e['selected_candidate'],distance_gain_micro=row['distance_gain']*1e6,
                payment=row['actual_payment']))
        examples.append(dict(condition=name,path=path,rule=rid,table=group))
    write(out/'mechanism_examples.json',dict(status='complete',examples=examples,scope='prespecified sensitivity results, not substituted for primary rule'))
    rejected=root.parent/'run_v1';oldp=read(rejected/'protocol.json');rejected_cases=[]
    for c in oldp['cases']:
        q=read(c['pool']['path'])
        rejected_cases.append(dict(condition=c['name'],raw_counts=[m['raw_support_count'] for m in q['models']],
            output_counts=[m['frozen_output_support_count'] for m in q['models']],
            same_final_output={tuple(sorted(b)) for b in q['models'][0]['frozen_output_supports']}=={tuple(sorted(b)) for b in q['models'][1]['frozen_output_supports']}))
    write(out/'rejected_preparation.json',dict(status='undeployed_historical_diagnostic',actual_records=0,holdout_scenarios=0,
        case_rows=rejected_cases,
        source_snapshot_sha256=sha(rejected/'source_snapshot.zip'),planning_summary_sha256=sha(rejected/'planning_summary.json')))
    write(out/'environment.json',read(root/'environment.json'))
    write(out/'manifest.json',dict(status='complete',private_root='wzzt_newday_weighted_v1/run_v2',
        private_hashes={n:sha(root/n) for n in ['protocol.json','price_freeze.json','source_snapshot.zip','summary.json','ended.json',
            'actual/observations_frozen.json','actual/predictions_frozen.json','actual/scored.json']},
        public_hashes={x.name:sha(x) for x in out.glob('*.json')},scope='explicit scalar/discrete whitelist, no raw P/Q/V/AC or full R/X'))
    print(json.dumps(dict(public_files=[x.name for x in out.iterdir()],complete=s['complete'],planned=60,
        independent_condition_days=20,main_rule=p['primary_rule']),ensure_ascii=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    x=a.parse_args();export(x.root,x.output)
