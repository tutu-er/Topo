"""Saved-evidence analysis tests. No AC model, fit or selector is called."""
from copy import deepcopy
import pytest
from . import ev_original_analysis as a


def bundle():
    names=['star','c01','c02','c12'];families=[[],[[0,1]],[[0,2]],[[1,2]]]
    pairs=[dict(a=x,b=y,added=0.,joint=0.,self_relative=0.,weight=1/12,success=True)
        for x in names for y in names if x!=y]
    fits=[dict(name=n,loss=1. if n=='star' else 2.,success=True) for n in names]
    predicted=dict(success=True,required_fit_states=16,returned_fit_states=16,mean_added_score=0.,worst_added_score=0.,
        pairs=pairs,records=[dict(a=n,success=True,failures={},fits=deepcopy(fits)) for n in names])
    power=[[1.,0.,0.]];op=dict(feasible=True,u_kw=power,travel_energy_kwh=0.,driver_cost=0.,
        conversion_loss_kwh=.05,moves=0,energy_kwh=[24.,26.85])
    models=[dict(a=r['a'],b=r['b'],name=r['a']+'->'+r['b']) for r in pairs]
    raw=dict(objective_lower=0.,objective_upper=100.,gap=100.,completed=False,unfinished=True,status='time_budget',
        required_pairs=[m['name'] for m in models],evaluations=[dict(operation_validation={'feasible':True},
            qp_results=[{} for _ in pairs],origin='initial_operation_0')],best_evaluation_index=0,iterations=[],qp_calls=12)
    config=dict(pilot_seeds=[1,2,3,4],proxy_channel_RMS_gate=.25,statistical_rejection_rule=None)
    freeze=dict(phase='pilot',domain='asset',seeds=[1,2,3,4],config=config,source_sha256={'model.py':'abc'})
    selection=dict(all_candidates=4,all_ordered_pairs=12,context_models=models,uses_truth=False,uses_future_noise=False,
        constant_operation=op,methods={name:dict(raw=deepcopy(raw),guard={'accepted':False},operation=deepcopy(op)) for name in a.DESIGNS},
        predicted_AC={name:deepcopy(predicted) for name in a.METHODS},failures={},seconds=1.)
    for name in ('quality_fixed','quality_mobile','random_mobile'):selection['methods'][name]=dict(operation=deepcopy(op))
    rows=[dict(method=name,status='complete',power_kw=power,candidate_miss=False,fits=deepcopy(fits),
        grid_net_kwh=1.,grid_throughput_kwh=1.,failures={}) for name in a.METHODS]
    trials=[dict(seed=seed,phase='pilot',domain='asset',config=config,source_sha256=freeze['source_sha256'],
        truth={'family':[]},proposal={'bank':[dict(name=n,supports=f) for n,f in zip(names,families)]},
        historical_fits=deepcopy(fits),historical_failures={},selection=deepcopy(selection),rows=deepcopy(rows)) for seed in range(1,5)]
    hashes=[dict(name=f'trial_{s}.json.gz',sha256=f'hash{s}') for s in range(1,5)]
    review=dict(success=True,review_errors=[],audited_histories=4,available_independent_histories=4,
        missing_histories=[],phase='pilot',trial_hashes=hashes,
        source_audit=dict(success=True,current_source_matches=True,rows=[dict(name='model.py',sha256='abc',current_sha256='abc')]))
    return freeze,trials,review,hashes


def gate(freeze,trials,review,hashes):
    return a.build_pilot_gate(freeze,trials,review,trial_hashes=hashes,current_hashes={'model.py':'abc'})


def test_four_histories_not_forty_eight_tasks_and_performance_not_gate_condition():
    freeze,trials,review,hashes=bundle();out=gate(freeze,trials,review,hashes)
    assert out['ready_for_independent_evaluation']
    assert not out['performance_improvement_required']
    analyzed=a.analyze_trial(trials[0])
    assert len(analyzed['method_rows'])==9 and len(analyzed['paired_rows'])==8
    row=next(r for r in analyzed['method_rows'] if r['method']=='constant')
    assert row['actual_M']==pytest.approx(1.) and row['actual_half_M']==pytest.approx(.5)
    assert row['predicted_unresolved']==12 and row['predicted_pair_count']==12
    assert len(analyzed['predicted_pair_rows'])==9*12


def test_partial_history_stale_review_or_source_blocks_gate():
    freeze,trials,review,hashes=bundle()
    assert not gate(freeze,trials[:3],review,hashes)['ready_for_independent_evaluation']
    bad=deepcopy(review);bad['trial_hashes'][0]['sha256']='stale'
    assert not gate(freeze,trials,bad,hashes)['ready_for_independent_evaluation']
    bad=deepcopy(review);bad['source_audit']['current_source_matches']=False
    assert not gate(freeze,trials,bad,hashes)['ready_for_independent_evaluation']


def test_all_required_pairs_and_ordered_numeric_bounds_checked():
    freeze,trials,review,hashes=bundle()
    trials[0]['selection']['methods']['charge_fixed']['raw']['objective_upper']=-1.
    out=gate(freeze,trials,review,hashes)
    assert not out['ready_for_independent_evaluation']
    assert '1:charge_fixed:final_proxy_interval' in out['failed_conditions']
    freeze,trials,review,hashes=bundle();trials[0]['selection']['context_models'].pop()
    assert not gate(freeze,trials,review,hashes)['ready_for_independent_evaluation']


def test_accepted_actions_require_all_twelve_RMS_checks_and_sixteen_fits():
    freeze,trials,review,hashes=bundle();selection=trials[0]['selection'];rec=selection['methods']['charge_fixed']
    rec['guard']=dict(accepted=True,proxy_same_witness_gate=True)
    checks=[dict(a=m['a'],b=m['b'],check=dict(physical_box_feasible=True,
        complete_AC_voltage_feasible=True,max_channel_RMS=.1)) for m in selection['context_models']]
    rec['proxy_AC_checks']=[checks]
    selection['predicted_AC']['charge_fixed__raw']=deepcopy(selection['predicted_AC']['constant'])
    assert gate(freeze,trials,review,hashes)['ready_for_independent_evaluation']
    checks[-1]['check']['max_channel_RMS']=.251
    assert not gate(freeze,trials,review,hashes)['ready_for_independent_evaluation']
    checks[-1]['check']['max_channel_RMS']=.1
    selection['predicted_AC']['charge_fixed__raw']['returned_fit_states']=15
    assert not gate(freeze,trials,review,hashes)['ready_for_independent_evaluation']


def test_rejected_raw_route_does_not_charge_its_driving_cost_to_dispatched_baseline():
    _,trials,_,_=bundle();rec=trials[0]['selection']['methods']['v2g_mobile']
    rec['operation']['travel_energy_kwh']=.2;rec['operation']['driver_cost']=30.
    analyzed=a.analyze_trial(trials[0]);row=next(r for r in analyzed['method_rows'] if r['method']=='v2g_mobile')
    assert row['fallback_to_constant'] and row['drive_kwh']==0 and row['worker_minutes']==0
    assert row['proxy_gap']==100. and row['raw_unfinished']


def test_failed_method_kept_in_denominator_and_missing_score_is_not_zero():
    _,trials,_,_=bundle();trials[0]['rows'][0]=dict(method='none',status='fit_failure',candidate_miss=False,failures={'c01':'failed'})
    analysis=a.analyze_trial(trials[0]);r=next(r for r in analysis['method_rows'] if r['method']=='none')
    assert r['actual_M'] is None and r['correct_min_loss'] is None
    summary=a.aggregate_method([r]);assert summary['available_histories']==1 and summary['complete_actual_histories']==0
    assert summary['metrics']['actual_M']['mean'] is None and summary['status_counts']=={'fit_failure':1}


def test_tiny_operation_reconstruction_roundoff_matches_same_physical_curve():
    assert a.same_curve([[0.,6.,-2.]],[[1e-12,6.+1e-12,-2.-1e-12]])
    assert not a.same_curve([[0.,6.,-2.]],[[0.,6.,-1.99]])


def test_explicit_missing_future_channel_cost_preserves_candidate_failure_row():
    _,trials,_,_=bundle();row=trials[0]['rows'][1]
    row['noiseless_AC_refit']=dict(all_candidates=False,fits=[],candidate_distances=[
        dict(candidate='c01',success=False,reason='future_fit_missing',future_channel_SSE=None)])
    result=a.analyze_trial(trials[0]);detail=next(r for r in result['noiseless_candidate_rows'] if r['method']=='constant' and r['candidate']=='c01')
    assert detail['candidate']=='c01' and not detail['success']
    assert detail['future_V_SSE'] is None and detail['joint_SSE'] is None
    unavailable=[r for r in result['noiseless_candidate_rows'] if r['method']=='constant']
    assert len(unavailable)==4 and all(not r['success'] for r in unavailable)
