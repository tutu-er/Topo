"""Predeclared-group calibration: counts, infinity, same-case comparison/order."""
from copy import deepcopy

import pytest

from . import ev_systematic_analyze as base
from . import ev_systematic_stratified as strata


def record(seed,H,phase='calibration',losses=None):
    losses={'a':1.,'b':2.} if losses is None else losses
    return dict(seed=seed,case_id='ordinary96',phase=phase,true_name='a',candidate_miss=False,status='complete',
        config=dict(training_points=H,validation_points=4,horizon=2),candidate_names=['a','b'],
        methods=[dict(name=name,power_kw=[[0.,0.],[0.,0.]],planning_failure=False,operation=dict(success=True),
            planning_seconds=.2,total_seconds=2.,parameter_error={},action_key='same',
            inference={mode:dict(losses=deepcopy(losses),complete=True,M=losses['b']-losses['a'],
                seconds=.1,total_seconds=1.) for mode in base.INFERENCE}) for name in base.METHODS])


def phase(rows):
    return dict(phase=rows[0]['phase'],records=rows,identity=dict(source_id='110',config_id='study110'),
        files=[dict(path='case'+str(r['seed']),sha256='hash'+str(r['seed'])) for r in rows])


def two_group_calibration():
    return [record(i,8,losses={'a':100.+i,'b':0.}) for i in range(10)]+[
        record(100+i,96,losses={'a':float(i),'b':0.}) for i in range(10)]


def test_predeclared_groups_use_own_sample_counts_and_do_not_expand(tmp_path):
    thresholds=strata.calibrate(phase(two_group_calibration()),tmp_path)
    values=thresholds['methods']['none']['fixed_total']
    assert tuple(thresholds['groups'])==('8','32','96')
    assert values['8']['n']==10 and values['8']['order_rank']==10 and values['8']['threshold']==109.
    assert values['96']['n']==10 and values['96']['threshold']==9.
    assert values['32']['n']==0 and values['32']['infinite'] and values['32']['missing_group']
    assert thresholds['global_reference']['none']['fixed_total']['threshold']==108.
    assert not thresholds['primary_thresholds_modified'] and not thresholds['adequacy_rule_combined']


def test_same_cases_compare_unique_and_false_rejection_with_original_denominators(tmp_path):
    cal=two_group_calibration();thresholds=strata.calibrate(phase(cal),tmp_path)
    rows=[record(1000,96,'evaluation',{'a':0.,'b':50.}),record(1001,8,'evaluation',{'a':109.,'b':0.})]
    result=strata.analyze(phase(rows),thresholds,tmp_path,calibration_records=cal)
    local=result['overall']['none']['fixed_total'];glob=result['global_reference']['none']['fixed_total']
    assert local['denominator']==glob['denominator']==2
    assert local['unique_correct']==1 and glob['unique_correct']==0
    assert local['false_reject']==0 and glob['false_reject']==1
    assert result['by_group']['96']['histories']==1 and result['by_group']['8']['histories']==1
    assert result['by_group']['32']['histories']==0
    assert result['by_group']['32']['stratified']['none']['fixed_total']['false_reject_rate'] is None
    assert result['paired_count_delta']['none']['fixed_total']['delta_unique_correct']==1
    assert local['total_seconds']==glob['total_seconds']
    assert result['analysis_seconds']>=0


def test_missing_group_keeps_all_but_insufficient_nonempty_group_is_not_missing(tmp_path):
    thresholds=strata.calibrate(phase([record(1,8)]),tmp_path)
    missing=strata.decision(record(100,32,'evaluation'),'none','fixed_total',thresholds)
    assert missing['category']=='missing_group_keep_all' and missing['retained']==['a','b']
    assert missing['missing_group'] and not missing['false_reject']
    insufficient=strata.decision(record(101,8,'evaluation'),'none','fixed_total',thresholds)
    assert insufficient['infinite_threshold'] and not insufficient['missing_group']
    assert insufficient['category']=='unresolved'
    unsupported=strata.decision(record(102,12,'evaluation'),'none','fixed_total',thresholds)
    assert unsupported['missing_group'] and unsupported['retained']==['a','b']
    assert strata.training_group(record(1,True)) is None
    with pytest.raises(ValueError,match='outside predeclared'):
        strata.calibrate(phase([record(2,12)]),tmp_path/'other')


def test_failed_score_zero_does_not_reduce_group_calibration_denominator(tmp_path):
    rows=[record(i,32,losses={'a':float(i),'b':0.}) for i in range(12)]
    for method in rows[-1]['methods']:
        for value in method['inference'].values():value.update(complete=False,M=None)
    thresholds=strata.calibrate(phase(rows),tmp_path)
    cutoff=thresholds['methods']['none']['joint_total']['32']
    assert cutoff['n']==12 and cutoff['scores'][-1]==0. and cutoff['incomplete_calibration']==1
    failed=record(100,32,'evaluation')
    for method in failed['methods']:
        for value in method['inference'].values():value.update(complete=False,M=None)
    decision=strata.decision(failed,'none','joint_total',thresholds)
    assert decision['category']=='failed_keep_all' and decision['retained']==['a','b']


def test_primary_reference_matches_and_primary_file_remains_byte_identical(tmp_path):
    cal=phase(two_group_calibration());primary=base.calibrate(cal,tmp_path)
    path=tmp_path/'thresholds.json';before=path.read_bytes()
    thresholds=strata.calibrate(cal,tmp_path,primary_thresholds=primary)
    assert thresholds['primary_thresholds_checked'] and path.read_bytes()==before
    assert strata.calibrate(cal,tmp_path,primary_thresholds=primary)==thresholds
    wrong=deepcopy(primary);wrong['methods']['none']['fixed_total']['threshold']+=1
    with pytest.raises(ValueError,match='differs from primary'):
        strata.calibrate(cal,tmp_path/'wrong',primary_thresholds=wrong)
    altered=deepcopy(cal);altered['records'][0]['config']['training_points']=96
    with pytest.raises(ValueError,match='existing thresholds differ'):
        strata.calibrate(altered,tmp_path)


def test_seeds_identity_candidate_coverage_and_durable_save_are_checked(tmp_path):
    cal=phase([record(i,96) for i in range(12)]);thresholds=strata.calibrate(cal,tmp_path)
    with pytest.raises(ValueError,match='durably saved'):
        strata.analyze(phase([record(100,96,'evaluation')]),thresholds,tmp_path/'empty')
    with pytest.raises(ValueError,match='overlap'):
        strata.analyze(phase([record(0,96,'evaluation')]),thresholds,tmp_path,calibration_records=cal['records'])
    changed=phase([record(100,96,'evaluation')]);changed['identity']['source_id']='different'
    with pytest.raises(ValueError,match='source_id differs'):strata.analyze(changed,thresholds,tmp_path)
    missing=record(100,96,'evaluation');missing['candidate_miss']=True
    with pytest.raises(ValueError,match='prohibits candidate miss'):
        strata.analyze(phase([missing]),thresholds,tmp_path)


def test_formal_cli_saves_all_group_thresholds_before_first_eval_read(monkeypatch,tmp_path):
    cal=phase([record(i,(8,32,96)[i%3]) for i in range(108)])
    evaluation=phase([record(1000+i,(8,32,96)[i%3],'evaluation') for i in range(72)])
    output=tmp_path/'strata';events=[]
    def read(directory,*,expected_phase=None):
        events.append(expected_phase)
        if expected_phase=='evaluation':assert (output/'stratified_thresholds.json').exists()
        return cal if expected_phase=='calibration' else evaluation
    monkeypatch.setattr(base,'read_phase',read)
    monkeypatch.setattr('sys.argv',['strata','--calibration','cal','--evaluation','eval','--output',str(output)])
    strata.main()
    assert events==['calibration','evaluation']
    assert (output/'stratified_summary.md').exists()
