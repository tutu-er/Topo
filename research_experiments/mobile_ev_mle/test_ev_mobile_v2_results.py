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
