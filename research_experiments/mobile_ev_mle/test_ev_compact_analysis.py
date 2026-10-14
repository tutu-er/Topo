"""Saved-evidence analysis tests; never run simulation, fitting or selection."""
from copy import deepcopy
import gzip
import json
import pytest
from . import ev_compact_analysis as a


def trial(seed=1):
    names=['star','c01','c02','c12'];families=[[],[[0,1]],[[0,2]],[[1,2]]]
    fits=[dict(name=n,success=True,loss=1. if n=='star' else 3.,r=[1.,2.],x=[.1,.2]) for n in names]
    pairs=[dict(a=x,b=y,added=float(i),weight=1/12,success=True) for i,(x,y) in enumerate(
        (x,y) for x in names for y in names if x!=y)]
    pred=dict(success=True,pairs=pairs)
    power=[[2.,0.,0.],[2.,0.,0.]]
    operation=dict(feasible=True,u_kw=power,travel_energy_kwh=0.,driver_cost=0.,cost_units='worker_minutes',
        moves=0,conversion_loss_kwh=.1,energy_kwh=[24.,25.,26.85])
    raw=dict(objective_lower=1.,objective_upper=2.,gap=1.,completed=False,unfinished=True,status='iteration_limit',
        evaluations=[dict(origin='initial_operation_0')],best_evaluation_index=0,iterations=[],qp_calls=12,seconds=.5)
    records={name:dict(raw=deepcopy(raw),operation=deepcopy(operation),initial_candidates=[dict(origin='constant',operation=deepcopy(operation))],
        selected_origin='constant',selected_score=5.5,initial_best_score=5.5,initial_best_origin='constant',
        inheritance_protected=True,raw_proxy_gate=True,random_used=False,failures=[],guard=dict(accepted=False)) for name in a.DESIGNS}
    for name in ('quality_fixed','quality_mobile','random_mobile'):records[name]=dict(operation=deepcopy(operation))
    mean=dict(all_candidates=True,fits=deepcopy(fits),candidate_distances=[dict(candidate=n,success=True,
        attained_joint_SSE=fits[i]['loss'],added_half_SSE=float(i),historical_SSE=.5,
        future_channel_SSE={'v':.2,'p':.1,'q':.1,'master_p':.05,'master_q':.05}) for i,n in enumerate(names)])
    return dict(seed=seed,phase='development',domain='asset',config={'physical':{'device':{'terminal_energy_kwh':26.85}}},
        source_sha256={'model.py':'abc'},policy_id='center',history={'p':[[1.,1.,1.]],'v':[[1.,1.,1.]]},
        truth=dict(family=[],r=[1.,2.],x=[.1,.2],p=[[1.,1.,1.]]*3,q=[[.1,.1,.1]]*3,
            white={'p':[[0.,0.,0.]]*3},source_p=[.2]*3,source_q=[.1]*3,root_v=[1.]*3),
        measurement_models={'hm':{'v_sigma':.001}},proposal={'bank':[dict(name=n,supports=f) for n,f in zip(names,families)]},
        selection=dict(methods=records,predicted_AC={name:deepcopy(pred) for name in a.METHODS},
            constant_operation=operation,random_used_by_primary=False,prediction='center',seconds=1.),
        rows=[dict(method=name,status='complete',power_kw=deepcopy(power),candidate_miss=False,fits=deepcopy(fits),
            grid_net_kwh=1.,grid_throughput_kwh=1.,noiseless_AC_refit=deepcopy(mean),cost={'fit_calls':4,'ac_calls':16}) for name in a.METHODS],
        seconds=3.)


def find(result,name):return next(row for row in result['method_rows'] if row['method']==name)


def test_ten_methods_one_history_not_one_hundred_twenty_tasks():
    result=a.analyze_trial(trial());row=find(result,'constant')
    assert len(result['method_rows'])==10 and len(result['candidate_rows'])==80
    assert len(result['paired_rows'])==11
    assert row['actual_M']==pytest.approx(2.) and row['actual_half_M']==pytest.approx(1.)
    assert row['true_candidate_joint_SSE']==1. and row['minimum_joint_SSE']==1.
    assert row['predicted_mean']==pytest.approx(5.5) and row['predicted_tasks']==12
    assert row['R_relative']==row['X_relative']==0.
    summary=a.aggregate_method(result['method_rows'])
    assert summary['history_denominator']==10 # ten method measurements, not the normal per-method grouping


def test_failed_or_missing_method_kept_with_four_candidates_each_stage():
    t=trial();t['rows']=[r for r in t['rows'] if r['method']!='random_mobile']
    t['rows'][0]=dict(method='none',status='fit_failure',fits=[dict(name='star',loss=1.,success=True)],failures={'c01':'bad'})
    result=a.analyze_trial(t);row=find(result,'none');missing=find(result,'random_mobile')
    assert row['actual_M'] is None and row['true_candidate_joint_SSE']==1.
    assert missing['status']=='missing_method' and missing['actual_M'] is None
    assert len(result['candidate_rows'])==80
    summary=a.aggregate_method([missing],planned_histories=4)
    assert summary['history_denominator']==1 and summary['planned_histories']==4
    assert summary['actual_failed_or_missing']==1 and summary['metrics']['actual_M']['n']==0


def test_inherited_plan_ledger_is_actual_selected_plan_not_constant_fallback():
    t=trial();record=t['selection']['methods']['v2g_mobile'];record['selected_origin']='charge_mobile'
    record['operation'].update(travel_energy_kwh=.2,driver_cost=30.,moves=1)
    result=a.analyze_trial(t);row=find(result,'v2g_mobile')
    assert row['inherited_or_initial_retained'] and not row['retained_constant']
    assert row['drive_kwh']==.2 and row['worker_minutes']==30. and row['moves']==1
    assert row['operation_ledger_matches_executed_curve']


def test_wrong_curve_does_not_borrow_unexecuted_route_resources():
    t=trial();t['selection']['methods']['v2g_mobile']['operation']['u_kw']=[[0.,2.,0.],[0.,2.,0.]]
    row=find(a.analyze_trial(t),'v2g_mobile')
    assert not row['operation_ledger_matches_executed_curve'] and row['drive_kwh'] is None


def test_milp_label_with_initial_curve_does_not_count_as_new_proposal():
    t=trial();record=t['selection']['methods']['charge_fixed'];record['selected_origin']='MILP'
    result=a.analyze_trial(t);row=find(result,'charge_fixed')
    assert not row['selected_is_new_MILP_proposal'] and row['new_MILP_proposals']==0


def test_new_milp_curve_recenter_counts_and_inheritance_violation_recorded():
    t=trial();record=t['selection']['methods']['v2g_mobile'];power=[[0.,2.,0.],[0.,2.,0.]]
    master=dict(u=[0.,2.,0.,0.,2.,0.],n_variables=300,n_rows=500,n_binary=200,
        n_operation_binary=128,n_shared_square_binary=72,seconds=.1,status=0,master_form='shared_control_square',
        objective_upper=4.,surrogate_incumbent=3.,solver_best_bound=4.,bound_consistent=True,
        incumbent_is_true_score_lower=False)
    record['raw'].update(evaluations=[dict(origin='master_0')],iterations=[dict(iteration=0,master=master)])
    record['before_recenter']=deepcopy(record['raw']);record['recentered']=True
    record['selected_origin']='MILP';record['operation']['u_kw']=power
    record['selected_score']=1.;record['initial_best_score']=2.;record['inheritance_protected']=False
    next(r for r in t['rows'] if r['method']=='v2g_mobile')['power_kw']=power
    result=a.analyze_trial(t);row=find(result,'v2g_mobile')
    assert row['main_MILP_calls']==2 and row['unique_MILP_proposals']==1 and row['new_MILP_proposals']==1
    assert row['selected_is_new_MILP_proposal'] and not row['inheritance_check']
    assert row['proxy_seconds']==1. and row['master_seconds']==pytest.approx(.2)
    assert len(result['master_rows'])==2


def test_ensemble_scores_weight_tasks_and_cap_each_task_before_average():
    pred=dict(success=True,pairs=[dict(added=0.,weight=.5,success=True),dict(added=100.,weight=.5,success=True)])
    summary=a.prediction_summary(pred)
    assert summary['mean']==50. and summary['cap20_mean']==10.
    t=trial();many=deepcopy(t['selection']['predicted_AC']['constant']['pairs'])*6
    for row in many:row['weight']=1/72
    t['selection']['predicted_AC']['v2g_mobile']=dict(success=True,pairs=many)
    row=find(a.analyze_trial(t),'v2g_mobile')
    assert row['predicted_tasks']==72 and row['predicted_mean']==pytest.approx(5.5)
    # The summary still receives one per-history row.
    assert a.aggregate_method([row])['history_denominator']==1


def test_incomplete_prediction_is_missing_score_not_partial_mean():
    pred=dict(success=False,pairs=[dict(added=1.,weight=.5,success=True),dict(added=None,weight=.5,success=False)])
    summary=a.prediction_summary(pred)
    assert summary['mean'] is None and summary['available_pairs']==1 and not summary['success']
    with pytest.raises(ValueError,match='weights'):
        a.prediction_summary(dict(success=True,pairs=[dict(added=1.,weight=2.,success=True)]))


def save_directory(path,trials):
    path.mkdir();first=trials[0]
    freeze=dict(phase=first['phase'],seeds=[1,2,3,4],source_sha256=first['source_sha256'],
        config=first['config'],policy_id=first['policy_id'],prediction=first['selection']['prediction'],master_form='shared_control_square')
    (path/'freeze.json').write_text(json.dumps(freeze),encoding='utf-8')
    for t in trials:
        with gzip.open(path/f"trial_{t['seed']}.json.gz",'wt',encoding='utf-8') as handle:json.dump(t,handle)
    return freeze


def test_directory_keeps_missing_histories_separate_from_method_failures(tmp_path):
    directory=tmp_path/'one';save_directory(directory,[trial(1),trial(2)])
    result=a.summarize_directory(directory)
    assert result['summary']['available_histories']==2 and result['summary']['planned_histories']==4
    assert result['summary']['missing_seeds']==[3,4]
    assert result['summary']['methods']['v2g_mobile']['history_denominator']==2
    assert result['summary']['methods']['v2g_mobile']['actual_complete']==2
    assert len(result['method_rows'])==20
    assert '2/4' in a.markdown(result)


@pytest.mark.parametrize('field,value',[('phase','evaluation'),('source_sha256',{'other.py':'hash'}),('policy_id','wrong')])
def test_directory_rejects_mixed_phase_or_changed_freeze(tmp_path,field,value):
    directory=tmp_path/'bad';t=trial();save_directory(directory,[t]);t[field]=value
    with gzip.open(directory/'trial_1.json.gz','wt',encoding='utf-8') as handle:json.dump(t,handle)
    with pytest.raises(ValueError):a.summarize_directory(directory)


def test_center_ensemble_pairing_requires_exact_history_truth_noise_device(tmp_path):
    one=tmp_path/'center';two=tmp_path/'ensemble';save_directory(one,[trial()])
    t=trial();t['selection']['prediction']='ensemble';t['policy_id']='ensemble';save_directory(two,[t])
    left=a.summarize_directory(one);right=a.summarize_directory(two)
    out=a.compare_directories([left,right]);assert len(out['rows'])==10 and not out['excluded']
    t['truth']['white']['p'][0][0]=1.
    # Fingerprint comparison is independent of source hash/config policy labels.
    changed=a.analyze_trial(t,label='changed')
    right['identities']=[changed['identity']];right['method_rows']=changed['method_rows']
    rejected=a.compare_directories([left,right])
    assert not rejected['rows'] and rejected['excluded'][0]['reason']=='phase_or_data_identity_mismatch'


def test_cross_directory_development_evaluation_never_pooled(tmp_path):
    one=tmp_path/'dev';two=tmp_path/'eval';save_directory(one,[trial()])
    t=trial();t['phase']='evaluation';save_directory(two,[t])
    out=a.compare_directories([a.summarize_directory(one),a.summarize_directory(two)])
    assert not out['rows'] and out['excluded'][0]['checks']['phase'] is False
    assert out['pooled_history_count'] is None


def test_selected_prediction_failure_and_none_channels_keep_candidate_denominators():
    t=trial();row=next(r for r in t['rows'] if r['method']=='capped_v2g_mobile')
    row['noiseless_AC_refit']=dict(all_candidates=False,fits=[],candidate_distances=[dict(candidate='star',future_channel_SSE=None)])
    result=a.analyze_trial(t);records=[r for r in result['candidate_rows'] if r['method']=='capped_v2g_mobile' and r['stage']=='noiseless']
    assert len(records)==4 and all(not r['success'] for r in records)
    assert all(r['future_V_SSE'] is None for r in records)


def test_nonminute_worker_cost_is_not_mislabeled_as_minutes():
    t=trial();t['selection']['methods']['charge_mobile']['operation'].update(driver_cost=100.,cost_units='yuan')
    row=find(a.analyze_trial(t),'charge_mobile')
    assert row['worker_cost']==100. and row['worker_cost_units']=='yuan' and row['worker_minutes'] is None


def test_quality_prediction_can_reuse_saved_h_only_cache_without_future_data():
    t=trial();power=[[0.,2.,0.],[0.,2.,0.]]
    next(row for row in t['rows'] if row['method']=='quality_mobile')['power_kw']=power
    t['selection']['methods']['quality_mobile']['operation']['u_kw']=power
    pred=t['selection']['predicted_AC'].pop('quality_mobile');pred['power_kw']=power
    t['selection']['all_prediction_records']=[pred]
    row=find(a.analyze_trial(t),'quality_mobile')
    assert row['predicted_mean']==pytest.approx(5.5)
    assert row['prediction_source']=='saved_H_only_cache_same_curve'


def test_fit_cache_call_accounting_follows_execution_order_not_report_order():
    t=trial()
    for row in t['rows']:
        row['noiseless_AC_refit']['cost']={'fit_calls':4,'ac_calls':16}
        row['cache_hit']=True
    # Execute the last-displayed capped plan first, all curves identical.
    t['rows']=sorted(t['rows'],key=lambda row:row['method']!='capped_v2g_mobile')
    t['rows'][0]['cache_hit']=False
    result=a.analyze_trial(t)
    assert find(result,'capped_v2g_mobile')['noiseless_fit_calls']==4
    assert find(result,'constant')['noiseless_fit_calls']==0
    assert sum(row['actual_fit_calls'] for row in result['method_rows'])==4
    assert sum(row['noiseless_fit_calls'] for row in result['method_rows'])==4


def test_optional_refined_method_present_only_when_declared_or_saved():
    t=trial();legacy=a.analyze_trial(t)
    assert len(legacy['method_rows'])==10 and len(legacy['paired_rows'])==11
    assert 'refined_v2g_mobile' not in legacy['declared_methods']
    t['config']['methods']=list(a.SUPPORTED_METHODS)
    declared=a.analyze_trial(t)
    assert len(declared['method_rows'])==11 and len(declared['paired_rows'])==14
    missing=find(declared,'refined_v2g_mobile')
    assert missing['status']=='missing_method' and missing['actual_M'] is None
    # A saved method also provides a backward-compatible declaration.
    del t['config']['methods']
    saved=deepcopy(t['rows'][-1]);saved['method']='refined_v2g_mobile';t['rows'].append(saved)
    t['selection']['methods']['refined_v2g_mobile']=deepcopy(t['selection']['methods']['v2g_mobile'])
    result=a.analyze_trial(t)
    assert find(result,'refined_v2g_mobile')['actual_M']==2.
    assert len(result['paired_rows'])==14


def test_derated_operations_config_overrides_base_device_and_enters_hash():
    t=trial();t['config']['physical']['device']['power_kw']=6.
    t['config']['ensemble']={'tracking_fraction':.05}
    legacy=a.analyze_trial(t);r=find(legacy,'constant')
    assert r['command_power_limit_kw']==6. and r['forecast_tracking_power_limit_kw']==pytest.approx(6.3)
    assert r['forecast_tracking_within_hardware'] is False
    t['config']['command_power_kw']=6/1.05
    derated=a.analyze_trial(t);r=find(derated,'constant')
    assert r['command_power_limit_kw']==pytest.approx(6/1.05)
    assert r['forecast_tracking_within_hardware'] is True
    assert legacy['identity']['device_sha256']!=derated['identity']['device_sha256']
    t['operations_config']={'device':{'power_kw':5.5}}
    saved=a.analyze_trial(t)
    assert find(saved,'constant')['command_power_limit_kw']==5.5


def test_refined_cap_same_initial_set_and_budget_are_audited_not_assumed():
    t=trial();saved=deepcopy(t['rows'][-1]);saved['method']='refined_v2g_mobile';t['rows'].append(saved)
    t['selection']['methods']['refined_v2g_mobile']=deepcopy(t['selection']['methods']['capped_v2g_mobile'])
    result=a.analyze_trial(t);pair=next(r for r in result['paired_rows'] if r['contrast']=='refined_vs_cap')
    assert pair['same_initial_action_set'] and pair['same_configured_exchange_budget']
    t['selection']['methods']['refined_v2g_mobile']['initial_candidates'][0]['operation']['u_kw']=[[0.,2.,0.],[0.,2.,0.]]
    result=a.analyze_trial(t);pair=next(r for r in result['paired_rows'] if r['contrast']=='refined_vs_cap')
    assert not pair['same_initial_action_set']
