"""Absolute-error gate, finite sample union bound ingredients and durable order."""
from copy import deepcopy
import json

import pytest

from . import ev_systematic_analyze as base
from . import ev_systematic_adequacy as gates


def record(seed,phase='calibration',losses=None,case_id='ordinary96'):
    losses={'a':1.,'b':2.} if losses is None else losses
    methods=[]
    for name in base.METHODS:
        methods.append(dict(name=name,power_kw=[[0.,0.,0.],[0.,0.,0.]],planning_seconds=0.,total_seconds=1.,
            planning_failure=False,operation=dict(success=True),action_key='zero',parameter_error={},
            inference={mode:dict(losses=deepcopy(losses),complete=True,M=losses['b']-losses['a'],seconds=.1)
                       for mode in base.INFERENCE}))
    return dict(case_id=case_id,seed=seed,phase=phase,true_name='a',candidate_miss=False,status='complete',
        candidate_names=['a','b'],config=dict(training_points=10,validation_points=2,horizon=2),methods=methods)


def phase(rows):
    return dict(records=rows,phase=rows[0]['phase'],identity=dict(source_id='science110',config_id='config110'),
        files=[dict(path='fixture'+str(r['seed']),sha256=str(r['seed'])) for r in rows],
        manifest=dict(config=dict(formal_cases=['ordinary96','train32'])))


def test_output_count_uses_real_output_count_and_keeps_total_holdout_distinct():
    row=record(1);method=row['methods'][0]
    assert gates.output_count(row,method,'fixed_total')==70
    assert gates.output_count(row,method,'joint_total')==70
    assert gates.output_count(row,method,'fixed_holdout')==20
    method['power_kw']=[[0.,0.],[0.]]
    with pytest.raises(ValueError,match='rectangular'):gates.output_count(row,method,'joint_total')


def test_shared_offset_escapes_relative_score_but_is_detected_by_absolute_gate(tmp_path):
    thresholds=gates.calibrate(phase([record(i) for i in range(19)]),tmp_path)
    low=record(100,'evaluation',{'a':1.,'b':2.})
    high=record(101,'evaluation',{'a':1001.,'b':1002.})
    first=gates.scores(low,'scalar_fixed','fixed_total');second=gates.scores(high,'scalar_fixed','fixed_total')
    assert [v['R'] for v in first['candidate_scores'].values()]==[0.,1.]
    assert [v['R'] for v in second['candidate_scores'].values()]==[0.,1.]
    cutoff=thresholds['methods']['scalar_fixed']['fixed_total']
    good=gates.decision(low,'scalar_fixed','fixed_total',cutoff)
    bad=gates.decision(high,'scalar_fixed','fixed_total',cutoff)
    assert good['retained']==['a'] and not good['false_reject']
    assert bad['retained']==[] and bad['all_models_fail_adequacy'] and bad['false_reject']
    assert bad['category']=='empty_all_models_fail_adequacy'


def test_two_quantiles_have_declared_order_and_leave_primary_file_unchanged(tmp_path):
    original=tmp_path/'thresholds.json';original.write_text('{"primary":"intact"}',encoding='utf-8')
    rows=[record(i,losses={'a':float(i+1),'b':float(i+2)}) for i in range(108)]
    thresholds=gates.calibrate(phase(rows),tmp_path,expected_count=108)
    cutoff=thresholds['methods']['none']['fixed_total']
    assert cutoff['R']['order_rank']==104 and cutoff['A']['order_rank']==104
    assert cutoff['A']['threshold']==pytest.approx(104/70)
    assert thresholds['marginal_false_rejection_bound']==.1
    assert not thresholds['primary_thresholds_modified'] and thresholds['additional_diagnostic']
    assert original.read_text(encoding='utf-8')=='{"primary":"intact"}'
    assert gates.calibrate(phase(rows),tmp_path,expected_count=108)==thresholds
    with pytest.raises(ValueError,match='existing thresholds differ'):
        gates.calibrate(phase(rows),tmp_path,alpha_R=.04,alpha_A=.06)


def test_failures_keep_all_with_both_scores_zero_and_empty_sets_count_as_false_reject(tmp_path):
    cal=[record(i) for i in range(19)]
    for method in cal[-1]['methods']:
        method['power_kw']=None
        for mode in method['inference'].values():mode.update(complete=False,M=None)
    thresholds=gates.calibrate(phase(cal),tmp_path)
    cutoff=thresholds['methods']['none']['fixed_total']
    assert cutoff['R_scores'][-1]==0. and cutoff['A_scores'][-1]==0.
    rows=[record(100,'evaluation'),record(101,'evaluation',{'a':1001.,'b':1002.})]
    failed=record(102,'evaluation')
    for method in failed['methods']:
        for value in method['inference'].values():value.update(complete=False,M=None)
    rows.append(failed)
    result=gates.analyze(phase(rows),thresholds,tmp_path,formal=True,calibration_records=cal)
    counts=result['overall']['none']['fixed_total']
    assert counts['denominator']==3 and counts['unique_correct']==1
    assert counts['empty_all_models_fail_adequacy']==1 and counts['failed_keep_all']==1
    assert counts['false_reject']==1 and counts['false_reject_rate']==pytest.approx(1/3)
    last=result['by_history'][-1]['methods']['none']['fixed_total']
    assert last['retained']==['a','b'] and not last['all_models_fail_adequacy']


def test_pressure_flags_empirical_and_missing_truth_not_hidden(tmp_path):
    thresholds=gates.calibrate(phase([record(i) for i in range(19)]),tmp_path)
    pressure=record(100,'development',{'a':1001.,'b':1002.},case_id='bias_misspec96')
    missing=record(101,'development',{'a':1001.,'b':1002.},case_id='candidate_miss96')
    missing.update(true_name=None,candidate_miss=True)
    result=gates.analyze(phase([pressure,missing]),thresholds,tmp_path)
    assert result['outside_declared_calibration_case_support']==2
    counts=result['overall']['none']['fixed_total']
    assert counts['candidate_miss']==1 and counts['all_models_fail_adequacy']==2
    assert counts['false_reject']==1  # Missing true candidate has separate coverage failure.
    with pytest.raises(ValueError,match='prohibits candidate miss'):
        gates.calibrate(phase([dict(missing,phase='calibration')]),tmp_path/'bad')


def test_auxiliary_threshold_durability_and_history_separation_are_checked(tmp_path):
    cal=[record(i) for i in range(19)]
    thresholds=gates.calibrate(phase(cal),tmp_path)
    with pytest.raises(ValueError,match='durably saved'):
        gates.analyze(phase([record(100,'evaluation')]),thresholds,tmp_path/'empty',formal=True)
    with pytest.raises(ValueError,match='overlap'):
        gates.analyze(phase([record(0,'evaluation')]),thresholds,tmp_path,formal=True,calibration_records=cal)
    changed=phase([record(100,'evaluation')]);changed['identity']['source_id']='different'
    with pytest.raises(ValueError,match='source_id differs'):gates.analyze(changed,thresholds,tmp_path,formal=True)


def test_infinite_cutoffs_remain_json_safe_and_retain_all(tmp_path):
    thresholds=gates.calibrate(phase([record(1)]),tmp_path)
    cutoff=thresholds['methods']['none']['fixed_total']
    assert cutoff['R']['infinite'] and cutoff['A']['infinite']
    assert cutoff['R']['threshold'] is None and cutoff['A']['threshold'] is None
    result=gates.decision(record(2,'evaluation',{'a':1001.,'b':1002.}),'none','fixed_total',cutoff)
    assert result['retained']==['a','b'] and result['category']=='unresolved'
    json.dumps(result,allow_nan=False)


def test_cli_saves_auxiliary_thresholds_before_first_evaluation_read(monkeypatch,tmp_path):
    cal=phase([record(i) for i in range(108)]);evaluation=phase([record(1000+i,'evaluation') for i in range(72)])
    output=tmp_path/'aux';events=[]
    def read(directory,*,expected_phase=None):
        events.append(expected_phase)
        if expected_phase=='evaluation':assert (output/'adequacy_thresholds.json').exists()
        return cal if expected_phase=='calibration' else evaluation
    monkeypatch.setattr(base,'read_phase',read)
    monkeypatch.setattr('sys.argv',['adequacy','--calibration','cal','--evaluation','eval','--output',str(output)])
    gates.main()
    assert events==['calibration','evaluation']
    assert (output/'adequacy_summary.md').exists()
