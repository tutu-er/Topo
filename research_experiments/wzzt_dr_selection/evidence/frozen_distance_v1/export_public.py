"""Explicit scalar-only export; never exports PQV, AC arrays or fitted R/X."""
from pathlib import Path
import argparse,hashlib,json

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def export(root,destination):
    root=Path(root);destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    s=read(root/'summary.json');p=read(root/'protocol.json');pool=read(root/'frozen_pool.json')
    f=read(root/'price_freeze.json');opt=read(root/'optimization.json');old=root.parent/'run_v1'
    equal={n:sha(root/n)==sha(old/n) for n in ['problem.json','planning_draws.json','private_planning_scenarios.pkl']}
    assert all(equal.values()),equal
    models=[]
    for a,b in zip(pool['models'],read(old/'frozen_pool.json')['models']):
        assert a['R']==b['R'] and a['X']==b['X'] and a['point']==b['point']
        models.append(dict(candidate_id=a['candidate_id'],historical_index=a['historical_index'],
            historical_validation_mae=a['historical_validation_mae'],raw_count=a['raw_support_count'],
            output_count=a['frozen_output_support_count']))
    record_keys=['record_id','policy','path','status','full_errors','missing','extra','selected_candidate',
        'true_structure_in_frozen_pool','correct_candidate_ids','new_distance','distance_gain',
        'distance_target_met','actual_payment','worst_payment','cumulative_pair_margin','day_only_pair_margin',
        'wrong_model_excluded','day_only_wrong_model_excluded','day_only_errors','final_error_improvement',
        'parameters_changed','cumulative_validation_rows']
    result=dict(status='complete',table=s['table'],records=[{k:r[k] for k in record_keys if k in r} for r in s['rows']],
        independent_actual_days=4,requested_records=12,completed=s['completed'],failed_or_unresolved=12-s['completed'],
        new_parameter_fits=0,new_search_calls=0,new_LP=0,valid_AC_points=s['completed_valid_AC_time_points'],
        attempted_AC_points=s['attempted_AC_time_points'],gamma_native=s['gamma_native'],
        frozen_models=models,source_scope='two historical disputed complete models; historical estimates are not physical true parameters',
        probability_scope=s['probability_scope'])
    write(destination/'summary.json',result)
    plans=[]
    for policy,reference in f['plans'].items():
        plan=read(reference['path']);plans.append(dict(policy=policy,prices=plan['prices'],amplitudes=plan['amplitudes'],
            expected_payment=plan['expected_payment'],hard_payment_bound=plan['hard_payment_bound'],
            selected_module_ids=[m['module_id'] for m,a in zip(plan['modules'],plan['amplitudes']) if a>0]))
    write(destination/'planning.json',dict(status='complete',table=read(root/'planning_summary.json')['table'],quotes=plans,
        primary=opt['primary'],secondary=opt['secondary'],elapsed_seconds=opt['elapsed_seconds'],
        conditional_finite_scenario_optimality=opt['conditional_finite_scenario_optimality'],
        warm_start_quotes=opt['provided_warm_start_count'],verified_warm_start_quotes=opt['verified_warm_start_count'],
        returned_quote_equals_best_warm_start=opt['prices']==opt['warm_start_best_prices'],
        warm_start_best_quantile_micro=opt['warm_start_best_quantile_micro'],
        v1_failed_solver=dict(primary=read(old/'optimization.json')['primary'],gamma_micro=0.,new_AC=0,new_holdout=0),
        v1_v2_identical_inputs=equal,model_holdout=read(root/'model_holdout/summary.json')['table']))
    write(destination/'configuration.json',{k:p[k] for k in ['id','case','labels','root_label','base_kw','voltage_sigma','pq_sigma',
        'budget','aggregate_units','original_history_rows','new_training_rows','new_validation_rows',
        'cumulative_validation_rows','planning_samples','planning_seed','model_holdout_samples','model_holdout_seed',
        'actual_paths','actual_profiles','actual_response_seed','actual_replicate_base','solver_seconds','event',
        'validation_rule','parameter_rule','candidate_scope','resource_scope','budget_intensity_note','scenario_scope']})
    audits=[]
    for name in ['external_gate_before_score.json','independent_price_audit.json','independent_score_audit.json','independent_physical_audit.json']:
        a=read(root/name)
        checks=a.get('check_count',len(a.get('checks',[])))
        failures=a.get('failures',[c for c in a.get('checks',[]) if not c.get('passed',False)])
        assert not failures,(name,failures)
        audits.append(dict(file=name,sha256=sha(root/name),checks=checks,failures=0,
            computation='no new solve, fit or AC; independent arithmetic replay'))
    physical=read(root/'independent_physical_audit.json')
    # Preserve only the scalar distance/billing table and no individual arrays.
    for k in ['table','paired_table','distance_table','paired_distance_table']:
        if k in physical:
            write(destination/'physical_distance.json',dict(source_sha256=sha(root/'independent_physical_audit.json'),**{k:physical[k]}))
            break
    write(destination/'audit_summary.json',dict(status='complete',audits=audits,total_checks=sum(a['checks'] for a in audits),
        final_pure_tests=40,branch_ampacity_checked=False,all_raw_or_aggregate_trajectories_private=True))
    env=read(root/'environment.json');env.pop('executable',None);write(destination/'environment.json',env)
    private_names=['protocol.json','frozen_pool.json','problem.json','source_snapshot.zip','price_freeze.json',
        'preparation_ended.json','optimization.json','summary.json','ended.json','model_holdout/summary.json',
        'actual/observations_frozen.json','actual/predictions_frozen.json','actual/scored.json']
    private_refs={name:sha(root/name) for name in private_names}
    write(destination/'manifest.json',dict(status='complete',private_run='wzzt_frozen_distance_v1/run_v2',
        private_artifact_sha256=private_refs,public_files={x.name:sha(x) for x in destination.glob('*.json')},
        explicit_export='configuration, scalar scores/costs/margins, seeds and hashes only; no P/Q/V/AC/fitted R/X'))
    print(json.dumps(dict(public_files=sorted(x.name for x in destination.iterdir()),audits=sum(a['checks'] for a in audits),input_equality=equal)))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    c=a.parse_args();export(c.root,c.output)
