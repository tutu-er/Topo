"""Compact delivery invariants, failure retention and raw immutability tests."""
from copy import deepcopy
import gzip
import json
import pytest
from . import ev_compact_delivery as d
from . import ev_compact_analysis as a
from .test_ev_compact_analysis import trial


def enriched():
    t=trial();record=t['selection']['methods']['v2g_mobile']
    qp=dict(success=True,status=1,message='ok',physical_parameter=[1.,2.],xi=[1.,2.],z=[.1,.2],
        residual=[.3,.4],primal_upper=.25,tangent_lower_numeric=.24,numerical_gap=.01,
        original_action_lower=[0.,0.],original_action_upper=[3.,3.],physical_bound_violation=0.,seconds=.001)
    evaluation=dict(origin='initial_operation_0',u=[2.,0.,0.,2.,0.,0.],operation_x=[0.,1.,2.],
        score_lower=1.,score_upper=2.,pair_score_lower=[1.]*12,pair_score_upper=[2.]*12,
        operation_validation={'feasible':True},qp_results=[deepcopy(qp) for _ in range(12)],all_qp_converged=True)
    master=dict(u=evaluation['u'],operation_x=evaluation['operation_x'],objective_upper=2.,surrogate_incumbent=1.8,
        solver_best_bound=2.,status=1,solver_success=False,solver_mip_gap=.1,n_variables=200,n_binary=72,
        seconds=2.,numeric_guard=1e-10,shared_square_structure=dict(valid=True,pairs=[dict(
            occupancy_sum_lp_upper=1.,LP_audit={'huge_dual':[5.]*1000})]),
        shared_square_witnesses=[dict(name='star__to__c01',upward_cost_guard=1e-10,
            witness_validation=dict(valid=True,LP_audits=[{'x':[3.]*1000}],physical_enclosure_lower=[0.]*1000))])
    record['raw'].update(evaluations=[deepcopy(evaluation),deepcopy(evaluation)],best_evaluation_index=1,
        iterations=[dict(iteration=0,master=deepcopy(master),global_lower=1.,global_upper=2.,gap=1.)],
        witnesses=[[dict(physical_intercept=[0.]*1000,physical_slope=[[0.]*6]*1000,origin='initial',
            validation=dict(valid=True,LP_audits=[{'huge':[5.]*1000}]))] for _ in range(12)])
    record['raw']['evaluations'][1]['origin']='master_0'
    record['before_recenter']=deepcopy(record['raw'])
    record['proxy_AC_checks']=[[dict(name='pair',check=dict(max_channel_RMS=.1,complete_AC_residual=[0.]*1000,
        proxy_residual=[.1]*1000,proxy_minus_AC=[.1]*1000,complete_AC_cost=0.,proxy_cost=10.))]]
    pred=deepcopy(t['selection']['predicted_AC']['constant']);pred['power_kw']=t['rows'][1]['power_kw']
    pred['records']=[dict(a='star',success=True,fits=deepcopy(t['rows'][1]['fits']),
        predicted_observations={'p':[[1.,2.,3.]]},forecast_state={'p':[[1.,2.,3.]]},failures={},cost={'fit_calls':4})]
    pred['records'][0]['fits'][0]['p']=[[2.]*3]*12
    t['selection']['predicted_AC']['constant']=pred
    t['selection']['all_prediction_records']=[pred]
    record['raw_prediction']=pred
    return t


def test_slim_keeps_all_actual_measurements_fits_truth_and_failure_denominators():
    raw=enriched();raw['rows'][-1]=dict(method='capped_v2g_mobile',status='operation_or_design_failure',candidate_miss=False)
    before=deepcopy(raw);slim=d.slim_trial(raw,raw_path='local/trial_1.json.gz',raw_sha256='rawhash',raw_bytes=100000)
    for key in ('history','historical_fits','truth','measurement_models','rows','config','source_sha256'):
        if key in raw:assert slim[key]==raw[key]
    assert raw==before
    assert slim['artifact_schema']==d.SCHEMA and slim['delivery']['complete_raw_replay_replacement'] is False
    assert slim['delivery']['source_raw_sha256']=='rawhash'
    assert len(slim['rows'])==10 and slim['rows'][-1]['status']=='operation_or_design_failure'
    assert 'capped_v2g_mobile' in slim['delivery']['declared_methods']


def test_predictions_keep_all_scores_failures_and_forecasts_but_drop_full_fits():
    raw=enriched();raw['selection']['all_prediction_records'][0]['failures']=[{'kind':'declared_failure'}]
    slim=d.slim_trial(raw,raw_path='raw.gz',raw_sha256='hash')
    prediction=slim['selection']['all_prediction_records'][0]
    assert prediction['pairs']==raw['selection']['all_prediction_records'][0]['pairs']
    assert prediction['failures']==[{'kind':'declared_failure'}]
    record=prediction['records'][0]
    assert record['forecast_state']=={'p':[[1.,2.,3.]]}
    assert 'predicted_observations' not in record
    assert record['fits'][0]['name']=='star' and record['fits'][0]['loss']==1.
    assert 'r' not in record['fits'][0] and 'p' not in record['fits'][0]


def test_all_iteration_bounds_and_best_full_qps_survive_without_lp_bulk():
    raw=enriched();slim=d.slim_trial(raw,raw_path='raw.gz',raw_sha256='hash')
    for key in ('raw','before_recenter'):
        original=raw['selection']['methods']['v2g_mobile'][key]
        exchange=slim['selection']['methods']['v2g_mobile'][key]
        assert len(exchange['evaluations'])==2
        assert exchange['evaluations'][0]['pair_score_lower']==[1.]*12
        assert 'physical_parameter' not in exchange['evaluations'][0]['qp_results'][0]
        assert exchange['evaluations'][0]['qp_results'][0]['status']==1
        assert exchange['evaluations'][1]['qp_results']==original['evaluations'][1]['qp_results']
        assert exchange['evaluations'][1]['operation_x']==[0.,1.,2.]
        master=exchange['iterations'][0]['master']
        for field in ('solver_best_bound','surrogate_incumbent','objective_upper','solver_mip_gap','n_variables','n_binary','seconds','status'):
            assert master[field]==original['iterations'][0]['master'][field]
        assert master['shared_square_structure']['pairs'][0]['occupancy_sum_lp_upper']==1.
        assert 'LP_audit' not in master['shared_square_structure']['pairs'][0]
        assert 'physical_intercept' not in exchange['witness_summary'][0][0]
        assert 'LP_audits' not in exchange['witness_summary'][0][0]['validation']
    checks=slim['selection']['methods']['v2g_mobile']['proxy_AC_checks'][0][0]['check']
    assert checks['max_channel_RMS']==.1 and 'complete_AC_residual' not in checks


def test_slim_can_regenerate_primary_analysis_without_new_ac_or_original_raw():
    raw=enriched();slim=d.slim_trial(raw,raw_path='raw.gz',raw_sha256='hash')
    original=a.analyze_trial(raw);reduced=a.analyze_trial(slim)
    fields=('status','actual_M','noiseless_M','R_relative','X_relative','predicted_mean','predicted_worst',
        'selected_origin','initial_best_score','proxy_gap','main_MILP_calls','new_MILP_proposals')
    for left,right in zip(original['method_rows'],reduced['method_rows']):
        assert left['method']==right['method']
        for field in fields:assert left[field]==right[field]


def directory(path,*,planned=None):
    path.mkdir();t=enriched()
    freeze=dict(config=t['config'],source_sha256=t['source_sha256'],phase=t['phase'],policy_id=t['policy_id'],
        seeds=[1] if planned is None else planned)
    (path/'freeze.json').write_text(json.dumps(freeze),encoding='utf8')
    (path/'source_snapshot.json.gz').write_bytes(gzip.compress(b'{"model.py":"test-source"}',mtime=0))
    raw=gzip.compress(json.dumps(t).encode('utf8'),mtime=0)
    (path/'trial_1.json.gz').write_bytes(raw)
    return t,raw


def test_export_is_byte_deterministic_and_never_changes_raw(tmp_path):
    source=tmp_path/'raw';target=tmp_path/'slim';_,raw=directory(source)
    manifest=d.export_directory(source,target)
    assert manifest['exported_seeds']==[1] and not manifest['partial']
    assert (source/'trial_1.json.gz').read_bytes()==raw
    assert manifest['trial_files'][0]['raw_sha256']==d.sha256_file(source/'trial_1.json.gz')
    slim=target/manifest['trial_files'][0]['slim_file'];assert slim.name=='trial_1.slim.json.gz'
    assert d.read_json(slim)['artifact_schema']==d.SCHEMA
    for name in ('freeze.json','source_snapshot.json.gz'):assert (target/name).read_bytes()==(source/name).read_bytes()
    assert d.export_directory(source,target)==manifest
    # The slim directory remains consumable by the saved-evidence analyzer.
    analyzed=a.summarize_directory(target)
    assert analyzed['summary']['available_histories']==1


def test_incomplete_export_requires_explicit_label_and_lists_missing_seeds(tmp_path):
    source=tmp_path/'raw';target=tmp_path/'slim';directory(source,planned=[1,2])
    with pytest.raises(ValueError,match='incomplete'):d.export_directory(source,target)
    assert not target.exists()
    manifest=d.export_directory(source,target,allow_partial=True)
    assert manifest['partial'] and manifest['missing_seeds']==[2]


def test_export_refuses_raw_overwrite_or_different_existing_artifact(tmp_path):
    source=tmp_path/'raw';target=tmp_path/'slim';directory(source)
    with pytest.raises(ValueError,match='differ'):d.export_directory(source,source)
    target.mkdir();(target/'trial_1.slim.json.gz').write_bytes(b'private-existing-data')
    with pytest.raises(ValueError,match='overwrite'):d.export_directory(source,target)
    assert (target/'trial_1.slim.json.gz').read_bytes()==b'private-existing-data'


def test_changed_policy_or_missing_source_snapshot_blocks_export(tmp_path):
    source=tmp_path/'raw';target=tmp_path/'slim';t,_=directory(source)
    t['source_sha256']={'different':'value'}
    (source/'trial_1.json.gz').write_bytes(gzip.compress(json.dumps(t).encode(),mtime=0))
    with pytest.raises(ValueError,match='frozen'):d.export_directory(source,target)
    (source/'source_snapshot.json.gz').unlink()
    with pytest.raises(ValueError,match='source evidence'):d.export_directory(source,target)


def test_invalid_best_index_and_double_slimming_are_not_silent():
    raw=enriched();raw['selection']['methods']['v2g_mobile']['raw']['best_evaluation_index']=10
    with pytest.raises(ValueError,match='best index'):d.slim_trial(raw,raw_path='raw.gz',raw_sha256='hash')
    raw=enriched();slim=d.slim_trial(raw,raw_path='raw.gz',raw_sha256='hash')
    with pytest.raises(ValueError,match='already a slim'):d.slim_trial(slim,raw_path='raw.gz',raw_sha256='hash')
