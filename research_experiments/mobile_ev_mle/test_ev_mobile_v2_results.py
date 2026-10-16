"""Small postprocessing fixtures for denominators and attribution."""
from copy import deepcopy
import gzip
import json
import pytest
from . import ev_mobile_v2_results as results
from . import ev_reliability_v2 as rel
from .test_ev_mobile_v2_analysis_audit import fixture


def test_paired_case_slots_are_not_reported_as_independent_histories():
    a,phase=fixture(1,'development');b=deepcopy(a);b['case_id']='train8'
    phase['records'].append(b)
    output=results.analyze(phase)
    assert output['overall']['case_history_slots']==2
    assert output['overall']['unique_base_seeds']==1
    assert output['overall']['paired_seed_clusters']=={'1':['ordinary96','train8']}
    assert output['by_case']['train8']['unique_base_seeds']==1
    assert not output['has_calibrated_rules']
    assert output['overall']['methods']['scalar_fixed']['inference']['joint_total']['M']['n']==2


def test_mobility_reports_power_location_change_separately_from_net_profile():
    record,phase=fixture(1,'development');methods={m['name']:m for m in record['methods']}
    methods['scalar_fixed']['power_kw']=[[1.,0.,0.]]
    methods['scalar_mobile']['power_kw']=[[0.,0.,1.]]
    methods['scalar_mobile']['operation'].update(driver_cost=30.,audit=dict(moves=1,travel_energy_kwh=.2))
    for value in methods['scalar_mobile']['inference'].values():value.update(losses={'star':10.,'cherry':22.},M=12.)
    output=results.analyze(phase);mobile=output['overall']['mobility']['scalar']
    assert mobile['actual_moved']==1 and mobile['same_power_matrix']==0 and mobile['same_net_profile']==1
    assert mobile['driver_minutes']['mean']==30.
    assert mobile['inference']['joint_total']['win']==1
    assert mobile['inference']['joint_total']['delta']['mean']==2.


def test_all_five_rules_keep_failure_denominators_and_parameter_error_fields(tmp_path):
    rows=[fixture(i,'calibration')[0] for i in range(19)]
    _,cal=fixture(0,'calibration');cal['records']=rows
    thresholds=rel.calibrate(cal,tmp_path)
    record,ev=fixture(100,'evaluation')
    record['methods'][0]['inference']['joint_total']['complete']=False
    record['methods'][1]['parameter_error']=dict(training_R=.2,training_X=.3,joint_R=.1,joint_X=.25)
    output=results.analyze(ev,thresholds=thresholds)
    assert set(output['overall']['methods']['constant']['rules'])==set(rel.RULES)
    for rule in rel.RULES:
        value=output['overall']['methods']['none']['rules'][rule]['joint_total']
        assert value['denominator']==1 and value['failed_keep_all']==1 and value['false_reject']==0
    errors=output['overall']['methods']['constant']['parameter_error']
    assert errors['fixed_R_error']['mean']==.2 and errors['joint_X_error']['mean']==.25
    with pytest.raises(ValueError,match='frozen calibrated thresholds'):results.analyze(ev)


def test_same_losses_different_calibrated_decision_is_identified(tmp_path):
    rows=[fixture(i,'calibration')[0] for i in range(19)];_,cal=fixture(0,'calibration');cal['records']=rows
    thresholds=rel.calibrate(cal,tmp_path)
    record,ev=fixture(100,'evaluation')
    thresholds['rules']['global_R']['scalar_mobile']['joint_total']['all'].update(threshold=100.,infinite=False)
    output=results.analyze(ev,thresholds=thresholds)
    changes=output['overall']['mobility']['scalar']['rules']['global_R']['joint_total']
    assert changes['retained_changed']==1 and changes['same_losses_different_retained']==1
    assert changes['correct_loss']==1


def test_gradient_selected_and_inherited_gradient_are_distinguished_without_mutation():
    record,phase=fixture(1,'development')
    fixed=results._gradient_metadata(dict(success=True,selected_origin='joint_gradient_MILP',
        selected_score=5.,initial_best_score=4.,starts=[dict(iterations=[dict(accepted=True)])]))
    mobile=results._gradient_metadata(dict(success=True,selected_origin='fixed_parent',
        selected_score=5.,initial_best_score=5.,starts=[dict(iterations=[dict(accepted=True)])]))
    metadata=dict(rows={results.history_key(record):{
        'jointgrad_fixed':dict(selected_origin='joint_gradient_MILP',gradient=fixed),
        'jointgrad_mobile':dict(selected_origin='fixed_parent',gradient=mobile)}})
    old=deepcopy(metadata);output=results.analyze(phase,selection_metadata=metadata)
    a=output['overall']['methods']['jointgrad_fixed']['gradient']
    b=output['overall']['methods']['jointgrad_mobile']['gradient']
    assert a['direct_gradient_selected']==1 and a['gradient_MILP_accepted']==1
    assert b['direct_gradient_selected']==0 and b['inherits_fixed_gradient']==1
    assert b['accepted_step_without_final_gradient']==1
    assert metadata==old


def test_raw_metadata_projection_uses_case_and_seed_and_retains_only_small_fields(tmp_path):
    record,phase=fixture(1,'development');record['index']=3
    raw=record['selection']['methods']['jointgrad_fixed']
    raw['raw']=dict(success=True,selected_origin='scalar_jump',selected_score=4.,initial_best_score=4.,
        selected_profile={'large_AC_states':['x']*1000},starts=[])
    path=tmp_path/'case_003_1.json.gz'
    with gzip.open(path,'wt',encoding='utf-8') as stream:json.dump(record,stream)
    metadata=results.extract_selection_metadata(phase,tmp_path)
    value=metadata['rows']['ordinary96:1']['jointgrad_fixed']
    assert value['selected_origin']=='scalar_jump' and value['gradient']['gradient_MILP_calls']==0
    assert 'large_AC_states' not in json.dumps(metadata)
    assert metadata['source_files'][0]['key']=='ordinary96:1'


def guard(success=True,complete=True,slack=0.):
    return dict(success=success,complete=complete,reference_floor={'star':1.},
        reference_values={'star':1.+slack},slacks={'star':slack},
        numerical_tolerance={'star':2e-7},tolerance_coefficient=1e-7,
        tolerance_rule='relative_to_floor_1_plus_abs',violated_references=[] if success else ['star'])


def test_guard_rejections_are_distinct_from_incomplete_fits_and_nonaccepted_attempts():
    raw=dict(success=True,reference_guard_enabled=True,reference_floor={'star':1.},
        guard_tolerance_coefficient=1e-7,selected_origin='fixed_parent',
        selected_reference_guard=guard(slack=-1e-7),
        initial_candidates=[dict(reference_guard=guard(False)),dict(reference_guard=guard()),
                            dict(failure='initial budget exhausted')],
        starts=[dict(iterations=[dict(accepted=False,reference_guard=guard(False)),
                                dict(accepted=False,reference_guard=guard(False,False)),
                                dict(accepted=False,failure='no feasible MILP'),
                                dict(accepted=True,reference_guard=guard())])])
    value=results._gradient_metadata(raw)
    assert value['reference_guard_enabled'] is True and value['reference_guard_requested'] is True
    assert value['initial_guard_checked']==2 and value['initial_guard_rejected_complete']==1
    assert value['proposal_guard_checked']==3 and value['proposal_guard_rejected_complete']==1
    assert value['proposal_guard_incomplete']==1 and value['gradient_attempts_not_accepted']==3
    assert value['selected_reference_guard']['min_raw_slack']==-1e-7
    assert value['selected_reference_guard']['min_effective_slack']==1e-7


def test_guard_summary_retains_requested_but_unavailable_fallback(tmp_path):
    record,phase=fixture(1,'development')
    raw=record['selection']['methods']['jointgrad_mobile']
    raw['raw']=dict(success=False,reference_guard_requested=True,failure='Bfixed unavailable')
    path=tmp_path/'case_000_1.json.gz'
    with gzip.open(path,'wt',encoding='utf-8') as stream:json.dump(record,stream)
    metadata=results.extract_selection_metadata(phase,tmp_path)
    output=results.analyze(phase,selection_metadata=metadata)
    value=output['overall']['methods']['jointgrad_mobile']['gradient']
    assert value['reference_guard_requested']==1 and value['reference_guard_enabled']==0
    assert value['selected_guard_checked']==0 and value['selected_guard_missing_when_requested']==1


def test_guard_projection_and_summary_preserve_floor_values_and_actual_selected_check(tmp_path):
    record,phase=fixture(1,'development')
    raw=record['selection']['methods']['jointgrad_mobile']
    raw['raw']=dict(success=True,reference_guard_enabled=True,reference_floor={'star':1.},
        reference_floor_parent='jointgrad_fixed',reference_floor_cap=5.411,
        guard_tolerance_coefficient=1e-7,selected_reference_guard=guard(),selected_origin='fixed_parent',
        initial_candidates=[dict(reference_guard=guard(False))],starts=[])
    path=tmp_path/'case_000_1.json.gz'
    with gzip.open(path,'wt',encoding='utf-8') as stream:json.dump(record,stream)
    metadata=results.extract_selection_metadata(phase,tmp_path)
    output=results.analyze(phase,selection_metadata=metadata)
    value=output['overall']['methods']['jointgrad_mobile']['gradient']
    assert value['reference_floors']['star']['mean']==1.
    assert value['selected_guard_checked']==1 and value['selected_guard_passed']==1
    assert value['guard_counts']['initial_guard_rejected_complete']==1
    selected=output['by_history'][0]['methods']['jointgrad_mobile']['selection']['gradient']
    assert selected['reference_floor_parent']=='jointgrad_fixed' and selected['reference_floor_cap']==5.411


def test_small_gradient_projection_keeps_each_selected_reference_nearest_gap():
    raw=dict(success=True,selected_score=3.,selected_profile=dict(aggregation=dict(nearest=[
        dict(a='star',half_gap=-2.),dict(a='cherry',half_gap=8.)])))
    old=deepcopy(raw);value=results._gradient_metadata(raw)
    assert value['selected_reference_nearest_gaps']=={'star':-2.,'cherry':8.}
    assert raw==old
    assert results._gradient_metadata({})['selected_reference_nearest_gaps']=={}


@pytest.mark.parametrize('true_mobile,tradeoff,true_reversal',[(3.,True,False),(7.,False,True)])
def test_posthoc_mean_tradeoff_and_true_reference_prediction_reversal_are_separate(true_mobile,tradeoff,true_reversal):
    record,phase=fixture(1,'evaluation');truth=record['true_name'];other=next(c['name'] for c in record['candidates'] if c['name']!=truth)
    methods={r['name']:r for r in record['methods']}
    methods['jointgrad_fixed']['inference']['joint_total'].update(losses={truth:10.,other:20.},M=10.)
    methods['jointgrad_mobile']['inference']['joint_total'].update(losses={truth:10.,other:9.},M=-1.)
    metadata=dict(rows={results.history_key(record):{
        'jointgrad_fixed':dict(gradient=dict(selected_score=5.,selected_reference_nearest_gaps={truth:5.,other:5.})),
        'jointgrad_mobile':dict(gradient=dict(selected_score=6.,selected_reference_nearest_gaps={truth:true_mobile,other:12.-true_mobile}))}})
    old=deepcopy(metadata);output=results.prediction_rank_diagnostic(phase,metadata)
    row=output['by_history'][0]
    assert row['delta_mean_predictive']==1. and row['delta_actual_M']==-11.
    assert row['mean_prediction_reversal'] and row['mean_sacrifices_true_reference']==tradeoff
    assert row['true_reference_prediction_reversal']==true_reversal
    assert output['overall']['all_three_comparable']==1
    assert metadata==old and not output['changes_selection']


def test_posthoc_prediction_diagnostic_keeps_missing_metadata_and_failed_inference_denominators():
    record,phase=fixture(1,'evaluation')
    methods={r['name']:r for r in record['methods']};methods['jointgrad_mobile']['inference']['joint_total']['complete']=False
    output=results.prediction_rank_diagnostic(phase,dict(rows={}))
    assert output['overall']['denominator']==1 and output['overall']['metadata_missing']==1
    assert output['overall']['inference_incomplete']==1 and output['overall']['mean_M_comparable']==0
    assert output['by_history'][0]['delta_actual_M'] is None
