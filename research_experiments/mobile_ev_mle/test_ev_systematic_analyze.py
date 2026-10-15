"""Pure JSON fixtures: tie ranks, separate thresholds, failures and blind order."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from . import ev_systematic_analyze as analysis


def record(seed,phase='development',case_id='ordinary96',true='a',gap=2.):
    methods=[]
    for index,name in enumerate(analysis.METHODS):
        losses={true:10.,'b' if true!='b' else 'a':10.+gap+index/10.,'c':14.+index/10.}
        modes={key:dict(losses=deepcopy(losses),complete=True,
            M=min(v for k,v in losses.items() if k!=true)-losses[true],seconds=.01+index/100)
            for key in analysis.INFERENCE}
        methods.append(dict(name=name,power_kw=[[float(index)]],planning_seconds=index/10.,
            total_seconds=1.+index/10.,inference=modes,planning_failure=False,
            operation=dict(success=True),action_key=str(index),
            parameter_error=dict(training_R=.1,training_X=.2,joint_R=.05,joint_X=.08)))
    return dict(case_id=case_id,seed=seed,phase=phase,true_name=true,candidate_miss=False,
        status='complete',candidate_names=sorted(losses),methods=methods,
        checks=dict(source_id='source-v1',config_id='config-v1'))


def write_phase(directory,records):
    directory.mkdir(parents=True,exist_ok=True)
    for row in records:
        path=directory/f"case_{row['case_id']}_{row['seed']}.json.gz"
        with gzip.open(path,'wt',encoding='utf-8') as stream:json.dump(row,stream)


def phase(rows):
    return dict(records=rows,phase=rows[0]['phase'],identity=dict(source_id='source-v1',config_id='config-v1'),
        files=[dict(path='fixture_'+str(row['seed']),sha256='seed'+str(row['seed'])) for row in rows])


def test_tie_aware_spearman_is_standard_and_handles_constant_rank():
    assert analysis.spearman([1,2,2,4],[1,3,3,9])==pytest.approx(1.)
    assert analysis.spearman([1,2,3],[3,2,1])==pytest.approx(-1.)
    assert analysis.spearman([1,1],[2,3]) is None
    assert analysis.spearman([1],[1]) is None
    with pytest.raises(ValueError):analysis.spearman([1,2],[3])


def test_finite_sample_rank_and_infinite_threshold_are_json_safe():
    scores=list(range(108));value=analysis.conformal_threshold(scores,.1)
    assert value['order_rank']==99 and value['threshold']==98 and not value['infinite']
    infinite=analysis.conformal_threshold([0.],.1)
    assert infinite['threshold'] is None and infinite['infinite'] and infinite['order_rank']==2
    json.dumps(infinite,allow_nan=False)
    row=record(1)
    value=analysis.decision(row,'none','fixed_total',infinite)
    assert value['category']=='unresolved' and value['retained']==['a','b','c']


def test_read_phase_preserves_full_schema_and_denominators(tmp_path):
    rows=[record(1),record(2,true='b')]
    # An explicit incomplete inference keeps its original record and score 0.
    rows[1]['methods'][0]['inference']['joint_total'].update(complete=False,losses={'b':10.},M=None)
    directory=tmp_path/'development';write_phase(directory,rows)
    loaded=analysis.read_phase(directory,expected_phase='development')
    assert loaded['identity']==dict(source_id='source-v1',config_id='config-v1')
    output=analysis.descriptive_analysis(loaded)
    assert output['overall']['histories']==2
    assert output['overall']['truth_counts']=={'a':1,'b':1}
    joint=output['overall']['methods']['none']['inference']['joint_total']
    assert joint['complete']==1 and joint['incomplete']==1 and joint['denominator']==2
    assert joint['vs_same_domain_square']['incomplete']==1
    assert len(output['by_history'])==2 and len(output['source_files'])==2
    output_dir=tmp_path/'output'
    analysis.analyze_development(directory,output_dir)
    assert (output_dir/'development_summary.json').exists() and (output_dir/'development_summary.md').exists()


def test_calibration_failure_score_zero_and_separate_method_inference_thresholds(tmp_path):
    rows=[record(i,'calibration') for i in range(19)]
    for index,row in enumerate(rows):
        # true a is worse than b by index; this is nonconformity index.
        for method in row['methods']:
            for key in analysis.INFERENCE:
                method['inference'][key]['losses']={'a':10.+index,'b':10.,'c':50.}
                method['inference'][key]['M']=-float(index)
    rows[-1]['methods'][0]['inference']['joint_total']['complete']=False
    thresholds=analysis.calibrate(phase(rows),tmp_path,alpha=.1)
    assert thresholds['methods']['none']['fixed_total']['threshold']==17.
    assert thresholds['methods']['none']['joint_total']['threshold']==16.
    assert thresholds['methods']['none']['joint_total']['scores'][-1]==0.
    assert thresholds['methods']['none']['joint_total']['incomplete_calibration']==1
    assert (tmp_path/'thresholds.json').exists()
    assert analysis.calibrate(phase(rows),tmp_path,alpha=.1)==thresholds
    with pytest.raises(ValueError,match='existing thresholds differ'):
        analysis.calibrate(phase(rows),tmp_path,alpha=.2)


def test_evaluation_counts_original_denominators_and_keep_all_failures(tmp_path):
    cal=[record(i,'calibration') for i in range(19)]
    thresholds=analysis.calibrate(phase(cal),tmp_path)
    rows=[record(100+i,'evaluation') for i in range(4)]
    # Complete: unique correct, unique wrong/false reject, unresolved; then failure.
    for method in rows[1]['methods']:
        for key in analysis.INFERENCE:method['inference'][key].update(losses={'a':15.,'b':10.,'c':20.},M=-5.)
    for method in rows[2]['methods']:
        for key in analysis.INFERENCE:method['inference'][key].update(losses={'a':10.,'b':10.,'c':20.},M=0.)
    for method in rows[3]['methods']:
        method['planning_failure']=True
        for key in analysis.INFERENCE:method['inference'][key].update(complete=False,M=None)
    result=analysis.evaluate(phase(rows),thresholds,tmp_path,calibration_records=cal)
    value=result['overall']['none']['fixed_total']
    assert value['denominator']==4 and value['unique_correct']==1 and value['unique_wrong']==1
    assert value['unresolved']==1 and value['failed_keep_all']==1
    assert value['false_reject']==1 and value['false_reject_rate']==.25
    assert value['unresolved_including_failed']==2
    assert result['by_history'][-1]['methods']['none']['fixed_total']['retained']==['a','b','c']
    assert sum(value[k] for k in ('unique_correct','unique_wrong','unresolved','failed_keep_all','candidate_miss'))==4
    assert result['descriptive']['overall']['methods']['none']['planning_failure']==1


def test_formal_misses_overlap_identity_and_threshold_durability_rejected(tmp_path):
    cal=[record(i,'calibration') for i in range(19)]
    thresholds=analysis.calibrate(phase(cal),tmp_path)
    bad=record(100,'calibration');bad['candidate_miss']=True
    with pytest.raises(ValueError,match='prohibits candidate miss'):
        analysis.calibrate(phase([bad]),tmp_path/'different')
    rows=[record(100,'evaluation')]
    changed=phase(rows);changed['identity']['source_id']='source-v2'
    with pytest.raises(ValueError,match='source_id differs'):analysis.evaluate(changed,thresholds,tmp_path)
    with pytest.raises(ValueError,match='durably saved'):analysis.evaluate(phase(rows),thresholds,tmp_path/'missing')
    overlapping=[record(0,'evaluation')]
    with pytest.raises(ValueError,match='overlap'):analysis.evaluate(phase(overlapping),thresholds,tmp_path,calibration_records=cal)


def test_mobile_baseline_and_action_ties_do_not_become_independent_samples():
    row=record(1)
    # Duplicate realized action is an audit, not another independent history.
    for method in row['methods']:method['action_key']='same'
    result=analysis.descriptive_analysis(phase([row]))
    comparison=result['by_history'][0]['comparison']['fixed_total_vs_joint_total']
    assert comparison['method_pairs']==10 and comparison['unique_actions']==1
    assert comparison['unique_action_M_spearman'] is None
    assert result['overall']['methods']['scalar_mobile']['inference']['fixed_total']['vs_same_domain_square']['baseline']=='square_mobile'
    assert result['overall']['methods']['scalar_fixed']['inference']['fixed_total']['vs_same_domain_square']['baseline']=='square_fixed'


def test_explicit_global_config_identity_allows_phase_specific_run_metadata():
    rows=[record(1,'calibration')]
    one=analysis._stable_identity(dict(config={'phase':'calibration'},policy_id='cal-run'),rows)
    two=analysis._stable_identity(dict(config={'phase':'evaluation'},policy_id='eval-run'),rows)
    assert one==two==dict(source_id='source-v1',config_id='config-v1')
    analysis._identity_match(one,two)


def test_route_total_time_is_kept_separate_from_combined_adapter_time():
    row=record(1)
    for method in row['methods']:
        method['total_seconds']=99.
        method['inference']['fixed_total']['total_seconds']=2.
        method['inference']['joint_total']['total_seconds']=8.
    result=analysis.descriptive_analysis(phase([row]))
    value=result['overall']['methods']['none']
    assert value['total_seconds']['mean']==99.
    assert value['inference']['fixed_total']['total_seconds']['mean']==2.
    assert value['inference']['joint_total']['total_seconds']['mean']==8.


def test_cli_saves_thresholds_before_first_evaluation_record_read(monkeypatch,tmp_path):
    cal=phase([record(i,'calibration') for i in range(108)])
    ev=phase([record(1000+i,'evaluation') for i in range(72)])
    output=tmp_path/'analysis';events=[]
    def read(directory,*,expected_phase=None):
        events.append(expected_phase)
        if expected_phase=='evaluation':assert (output/'thresholds.json').exists()
        return cal if expected_phase=='calibration' else ev
    monkeypatch.setattr(analysis,'read_phase',read)
    monkeypatch.setattr('sys.argv',['analysis','--calibration','cal','--evaluation','ev','--output',str(output)])
    analysis.main()
    assert events==['calibration','evaluation']
    assert (output/'summary.md').exists()
