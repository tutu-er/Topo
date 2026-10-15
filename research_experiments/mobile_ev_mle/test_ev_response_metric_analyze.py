"""Saved two-family calibration; no simulation or strategy execution."""
from copy import deepcopy
from pathlib import Path

import pytest

from . import ev_response_metric_analyze as analyze
from . import ev_dynamic_collect as collector


def fixture_config():
    return dict(declared_candidates=['star','clade_01'],select_modes=[m for m in analyze.METHODS if m!='highlow'],
        diagnostics=False,mean_policy_ablation=False,development_seeds=[994101,994102],
        calibration_seeds=[994301,994302],evaluation_seeds=[994201,994202])


def fixture_trial(seed,family,*,failure=False):
    config=fixture_config();sources={'fixture':'constant'}
    identity=collector.object_hash(dict(config=config,source_sha256=sources))
    names=list(analyze.FAMILIES)
    fits=[dict(candidate=name,name=name,success=True,loss=1.,r=[.004],x=[.001]) for name in names]
    training=dict(all_complete=True,fits=deepcopy(fits),failures={},seconds=.1,cost={})
    validation=dict(all_complete=True,fits=deepcopy(fits),failures={},seconds=.05,cost={})
    result=dict(seed=seed,family_index=family,true_name=names[family],candidate_miss=False,
        config=config,source_sha256=sources,policy_id=identity,source_unchanged=True,
        candidates=[dict(name=name,supports=[]) for name in names],training_result=training,
        validation_result=validation,status='complete',seconds=1.,methods=[])
    for index,name in enumerate(analyze.METHODS):
        future=deepcopy(training)
        for fit in future['fits']:fit['loss']=1. if fit['candidate']==names[family] else 3.+index
        losses={fit['candidate']:fit['loss']+1. for fit in future['fits']}
        result['methods'].append(dict(name=name,status='complete',power_kw=[[2.,0.,0.]],
            operation=dict(success=True),noisy=future,noiseless=future,
            loss_by_candidate=losses,M=2.+index,clean_future_M=2.+index,clean_future_complete=True,
            planning_success=True,curve_id=name))
    if failure:
        result.update(status='driver_failure',methods=[],training_result={},validation_result={})
    return result


def fixture_phase(path,phase,*,failure_seed=None,missing_seed=None):
    path.mkdir();cfg=fixture_config();sources={'fixture':'constant'}
    collector.write_json(path/'freeze.json',dict(config=cfg,source_sha256=sources,
        policy_id=collector.object_hash(dict(config=cfg,source_sha256=sources))))
    for family,seed in enumerate(cfg[phase+'_seeds']):
        if seed==missing_seed:continue
        collector.write_json(path/f'trial_{seed}.json.gz',fixture_trial(seed,family,failure=seed==failure_seed))
    return path


def test_rank_correlation_handles_ties_and_undefined_constant_values():
    assert analyze.spearman([1.,2.,2.,4.],[1.,3.,3.,5.])==pytest.approx(1.)
    assert analyze.spearman([1.,2.,3.],[3.,2.,1.])==pytest.approx(-1.)
    assert analyze.spearman([0.,0.],[1.,2.]) is None
    assert analyze.spearman([1.],[2.]) is None


def test_strict_two_family_calibration_and_independent_evaluation(tmp_path):
    cal=fixture_phase(tmp_path/'cal','calibration');ev=fixture_phase(tmp_path/'ev','evaluation')
    block=analyze.read_phase(cal,'calibration')
    thresholds=analyze.calibrate(block,alpha=.5,expected_per_family=1)
    assert thresholds['families']==['star','clade_01']
    assert set(thresholds['calibration']['square'])=={'star','clade_01'}
    result=analyze.evaluate(analyze.read_phase(ev,'evaluation'),thresholds)
    for method in analyze.METHODS:
        overall=result['evaluation'][method]['overall']
        assert overall['total']==2 and overall['unique_correct']==2
        assert overall['false_reject']==0
        assert set(result['evaluation'][method]['by_true_family'])=={'star','clade_01'}


def test_failed_histories_retained_as_inf_and_keep_exactly_two_candidates(tmp_path):
    cal=fixture_phase(tmp_path/'cal','calibration',failure_seed=994301)
    ev=fixture_phase(tmp_path/'ev','evaluation',failure_seed=994201)
    block=analyze.read_phase(cal,'calibration')
    assert block['planned_method_slots']==18
    assert block['failed_method_slots']==9
    thresholds=analyze.calibrate(block,alpha=.5,expected_per_family=1)
    assert thresholds['calibration']['square']['star']['threshold']=='inf'
    result=analyze.evaluate(analyze.read_phase(ev,'evaluation'),thresholds)
    decision=next(r for r in result['decisions'] if r['seed']=='994201')['methods']['square']
    assert decision['failed_keep_all'] and decision['retained']==['star','clade_01']
    assert result['evaluation']['square']['overall']['total']==2
    assert result['evaluation']['square']['overall']['failed_keep_all']==1


def test_pending_histories_are_preserved_and_incomplete_calibration_rejected(tmp_path):
    cal=fixture_phase(tmp_path/'cal','calibration',missing_seed=994302)
    block=analyze.read_phase(cal,'calibration')
    assert block['planned_histories']==2 and block['observed_histories']==1
    assert block['pending_method_slots']==9
    assert block['methods']['square']['M']['total']==2
    with pytest.raises(ValueError,match='missing'):analyze.calibrate(block,expected_per_family=1)


def test_saved_thresholds_exist_before_any_evaluation_read(tmp_path,monkeypatch):
    cal=fixture_phase(tmp_path/'cal','calibration');ev=fixture_phase(tmp_path/'ev','evaluation')
    output=tmp_path/'analysis';original=analyze.read_phase;events=[]
    def guarded(path,phase):
        if phase=='evaluation':assert (output/'thresholds.json').exists()
        events.append(phase)
        return original(path,phase)
    monkeypatch.setattr(analyze,'read_phase',guarded)
    report=analyze.analyze(output,calibration=cal,evaluation=ev,alpha=.5,expected_per_family=1)
    assert events==['calibration','evaluation']
    assert report['thresholds_persisted_before_evaluation_read']
    assert report['analyzer']['new_AC_calls']==report['analyzer']['new_fit_calls']==0
    assert (output/'summary.json').exists() and (output/'summary.md').exists()


def test_threshold_tampering_and_unrelated_candidate_bank_rejected(tmp_path):
    cal=fixture_phase(tmp_path/'cal','calibration');ev=fixture_phase(tmp_path/'ev','evaluation')
    thresholds=analyze.calibrate(analyze.read_phase(cal,'calibration'),alpha=.5,expected_per_family=1)
    thresholds['calibration']['square']['star']['threshold']=100.
    with pytest.raises(ValueError,match='hash'):analyze.evaluate(analyze.read_phase(ev,'evaluation'),thresholds)
    path=ev/'trial_994201.json.gz';record=collector.read_json(path)
    record['candidates'].append(dict(name='clade_02',supports=[]));collector.write_json(path,record)
    with pytest.raises(ValueError,match='two-candidate'):analyze.read_phase(ev,'evaluation')
def test_action_ranking_can_expose_reverse_order_among_strong_curves():
    rows=[dict(full_AC_distance=x,joint_contrast=x,box_contrast=x,load_contrast=x,
        M=y,clean_future_M=y) for x,y in [(0.,0.),(1.,1.),(10.,12.),(11.,11.),(12.,10.)]]
    all_actions=analyze.rank_actions(rows)
    strong=analyze.rank_actions(rows[2:])
    assert all_actions['full_AC_distance_vs_M']['spearman']>0
    assert strong['full_AC_distance_vs_M']['spearman']==pytest.approx(-1.)
