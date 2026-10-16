"""Mock decision records only; no formal result or AC computation is used."""
from copy import deepcopy
from collections import Counter
import pytest
from . import ev_mobile_v2_uncertainty as uncertainty
from . import ev_reliability_v2 as rel
from .test_ev_mobile_v2_analysis_audit import fixture


def data(n=4):
    records=[fixture(100+i,'evaluation')[0] for i in range(n)]
    _,phase=fixture(100,'evaluation');phase['records']=records
    rows=[]
    for record in records:
        row=dict(case_id=record['case_id'],seed=record['seed'],true_name=record['true_name'],rules={})
        for rule in rel.RULES:
            row['rules'][rule]={method:{mode:dict(category='unique_correct',retained=['star'],false_reject=False)
                for mode in uncertainty.MODES} for method in phase['methods']}
        rows.append(row)
    report=dict(schema='ev_reliability_v2_evaluation_v1',phase='evaluation',identity=phase['identity'],
        thresholds_development_only=False,by_history=rows,overall={})
    recount(report,phase)
    return report,phase


def change(report,index,rule,method,mode,category):
    kept=['star'] if category=='unique_correct' else ['cherry'] if category=='unique_wrong' else ['cherry','star']
    report['by_history'][index]['rules'][rule][method][mode]=dict(category=category,retained=kept,false_reject='star' not in kept)


def recount(report,phase):
    report['overall']={rule:{method:{mode:dict(denominator=len(report['by_history']),
        unique_correct=sum(row['rules'][rule][method][mode]['category']=='unique_correct' for row in report['by_history']),
        false_reject=sum(row['rules'][rule][method][mode]['false_reject'] for row in report['by_history']))
        for mode in uncertainty.MODES} for method in phase['methods']} for rule in rel.RULES}


def test_wilson_boundaries_and_half_rate_for_N48():
    zero=uncertainty.wilson(0,48);full=uncertainty.wilson(48,48);half=uncertainty.wilson(24,48)
    assert zero['lower95']==pytest.approx(0.,abs=1e-15)
    assert zero['upper95']==pytest.approx(.07410012966611754)
    assert full['lower95']==pytest.approx(1-zero['upper95']) and full['upper95']==1.
    assert half['lower95']==pytest.approx(1-half['upper95'])
    assert half['lower95']<.37 and half['upper95']>.63
    for count,n in ((True,48),(-1,48),(49,48),(0,0)):
        with pytest.raises(ValueError):uncertainty.wilson(count,n)


def test_paired_improvement_sign_depends_on_event():
    value=uncertainty.paired([True,True,False,False],[False,True,True,False],event='false_reject')
    assert (value['improvement'],value['regression'],value['same'])==(1,1,2)
    assert value['net_improvement_percentage_points']==0.
    correct=uncertainty.paired([False,False,True,True],[True,True,False,True],event='unique_correct')
    assert (correct['improvement'],correct['regression'],correct['same'])==(2,1,1)
    assert correct['net_improvement_percentage_points']==25.


def test_all_rules_modes_methods_are_reported_and_failure_stays_in_N():
    report,phase=data()
    change(report,0,'label_AIC','none','joint_total','failed_keep_all')
    recount(report,phase);value=uncertainty.summarize(report,phase,bank='n3',expected_count=4)
    assert len(value['intervals'])==11
    assert sum(len(rules) for modes in value['intervals'].values() for rules in modes.values())==11*3*5
    cell=value['intervals']['none']['joint_total']['label_AIC']
    assert cell['unique_correct']['count']==3 and cell['unique_correct']['N']==4
    assert cell['false_reject']['count']==0 and cell['failed_keep_all']==1
    assert not value['significance_tests_performed'] and not value['best_rule_selected']


def test_predeclared_rule_pair_and_mobile_pair_use_same_histories():
    report,phase=data()
    change(report,0,'H_R','scalar_fixed','joint_total','unique_wrong')
    change(report,1,'H_R','scalar_fixed','joint_total','unique_wrong')
    change(report,1,'label_AIC','scalar_mobile','joint_total','unique_wrong')
    recount(report,phase);value=uncertainty.summarize(report,phase,bank='n3',expected_count=4)
    pair=value['main_rule_comparison']['paired']['scalar_fixed']['joint_total']
    assert pair['unique_correct']['improvement']==2 and pair['false_reject']['improvement']==2
    mobile=value['mobility_comparison']['paired']['scalar']['joint_total']['label_AIC']
    assert mobile['unique_correct']['regression']==1 and mobile['false_reject']['regression']==1
    assert value['main_rule_comparison']['reference']=='H_R' and value['main_rule_comparison']['challenger']=='label_AIC'


def test_wrong_counts_seeds_bank_or_development_data_are_rejected():
    report,phase=data()
    with pytest.raises(ValueError,match='expected evaluation count'):uncertainty.summarize(report,phase,bank='n3')
    bad=deepcopy(report);bad['by_history'][1]['seed']=bad['by_history'][0]['seed']
    with pytest.raises(ValueError,match='unique seeds'):uncertainty.summarize(bad,phase,bank='n3',expected_count=4)
    with pytest.raises(ValueError,match='physical bank'):uncertainty.summarize(report,phase,bank='n4',expected_count=4)
    bad=deepcopy(report);bad['thresholds_development_only']=True
    with pytest.raises(ValueError,match='formal saved'):uncertainty.summarize(bad,phase,bank='n3',expected_count=4)
    bad=deepcopy(report);bad['overall']['label_AIC']['none']['joint_total']['denominator']=3
    with pytest.raises(ValueError,match='overall count differs'):uncertainty.summarize(bad,phase,bank='n3',expected_count=4)


def test_rejection_consistency_and_all_denominators_checked_without_input_mutation():
    report,phase=data();before=deepcopy((report,phase))
    value=uncertainty.summarize(report,phase,bank='n3',expected_count=4)
    assert (report,phase)==before
    assert 'Wilson' in uncertainty.markdown(value)
    bad=deepcopy(report);bad['by_history'][0]['rules']['label_AIC']['none']['joint_total']['false_reject']=True
    with pytest.raises(ValueError,match='rejection event disagrees'):uncertainty.summarize(bad,phase,bank='n3',expected_count=4)


def test_N48_reference_resolution_is_one_over_48_percentage_points():
    report,phase=data(48);value=uncertainty.summarize(report,phase,bank='n3')
    assert value['N']==48 and value['seed_count']==48
    assert value['resolution']['one_history_percentage_points']==pytest.approx(2.0833333333333335)
    assert {r['count'] for r in value['resolution']['examples']}=={0,1,3,24,47,48}


def test_repeated_retained_candidate_cannot_hide_a_unique_correct_decision():
    report,phase=data()
    decision=report['by_history'][0]['rules']['label_AIC']['none']['joint_total']
    decision.update(retained=['star','star'],category='unresolved')
    recount(report,phase)
    with pytest.raises(ValueError,match='unique in-bank list'):
        uncertainty.summarize(report,phase,bank='n3',expected_count=4)


@pytest.mark.parametrize('missing',[False,True])
def test_null_or_missing_rejection_event_is_rejected_without_dropping_history(missing):
    report,phase=data()
    decision=report['by_history'][0]['rules']['label_AIC']['none']['joint_total']
    if missing:decision.pop('false_reject')
    else:decision['false_reject']=None
    with pytest.raises(ValueError,match='explicit Boolean rejection'):
        uncertainty.summarize(report,phase,bank='n3',expected_count=4)
