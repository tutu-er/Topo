"""Synthetic audit tests: no AC, fit, scientific samples or main data."""
from copy import deepcopy
from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

spec=spec_from_file_location('B_stored_probability_audit',Path(__file__).with_name('b_statistical_audit.py'))
audit=module_from_spec(spec);spec.loader.exec_module(audit)
SETTINGS=dict(propagation_mean_MCSE_limit=.2,propagation_covariance_MCSE_relative_limit=.05,noise_repeats=1)


def record(index=0,values=None):
    values=np.zeros((1,5)) if values is None else np.asarray(values)
    bank=dict(means=np.array([[np.zeros(5),np.ones(5)*4]]).tolist(),
        covariances=np.array([[np.eye(5)*3,np.eye(5)]]).tolist(),log_prior=np.log([.5,.5]).tolist())
    actual=dict(values=values.tolist(),clean=np.zeros(5).tolist(),scores={})
    return dict(spec=dict(index=index,seed=100+index,bank='n1'),status='complete',
        config=dict(horizon=1,noise_repeats=len(values)),candidate_names=['a','b'],true_name='a',true_index=0,
        plans={plan:dict(action_key='shared',power=[[1.]]) for plan in audit.prob.PLANS},
        actions=dict(shared=dict(power=[[1.]],propagation=dict(status='ok',rows=[],banks=dict(G00=bank,G11=bank)),
            regimes={regime:deepcopy(actual) for regime in audit.prob.REGIMES})))


def test_shared_independent_Cholesky_scores_keep_covariance_and_channel_order():
    source=record();source['actions']['shared']['regimes']['regular']['values'][0][2]=10.
    audited=audit.prob.audit_history(source,SETTINGS)
    row=next(x for x in audited['rows'] if x['plan']=='SQMD' and x['model']=='G11' and x['regime']=='regular')
    assert row['available'] and row['coverage90_voltage']==0.
    assert row['coverage90_terminal_P']==1. and row['coverage90_terminal_Q']==1.
    assert row['historical_log_weight_normalizer']==pytest.approx(0.)


def test_independent_calibration_weights_each_history_once_and_keeps_all32():
    first=audit.prob.audit_history(record(0),SETTINGS,calibration_mode=True)
    other=audit.prob.audit_history(record(1,np.ones((9,5))*10),SETTINGS,calibration_mode=True)
    failures=[audit.prob.audit_history(dict(spec=dict(index=i,seed=100+i,bank='n1'),
        status='process_failure'),SETTINGS,calibration_mode=True) for i in range(2,32)]
    calibrated=audit.prob.calibration_from_audits([first,other,*failures])
    assert calibrated['histories']==32 and calibrated['available_histories']==2
    for a in audit.prob.GRID:
        l=first['calibration_row']['losses'][str(a)];r=other['calibration_row']['losses'][str(a)]
        assert calibrated['objective'][str(a)]==pytest.approx((l+r)/2)
        assert calibrated['objective'][str(a)]!=pytest.approx((l+9*r)/10)


def test_all32_minimal_worker_failures_remain_unresolved_and_missing_MAP():
    records=[dict(spec=dict(index=i,seed=100+i,bank='n1'),status='process_failure') for i in range(32)]
    checked=[audit.prob.audit_history(r,SETTINGS) for r in records]
    summary=audit.augment_operational(audit.prob.summary(checked,calibrated=True),checked)
    for plan in audit.prob.PLANS:
        for model in audit.TARGET_MODELS:
            row=summary['summaries'][plan+'__'+model]
            assert row['availability']['total_histories']==32 and row['availability']['histories']==0
            assert row['availability']['error_all_histories_range']==[0.,1.]
            metrics=row['operational_all_histories']['metrics']
            assert metrics['unresolved']['mean']==1. and metrics['false_rejection']['mean']==0.
            assert row['operational_all_histories']['tasks']==64
            assert row['operational_all_histories']['available_tasks']==0
            assert row['operational_all_histories']['failed_tasks']==64
            assert metrics['unresolved']['bootstrap95']==[1.,1.]


def test_common_history_pair_requires_both_regimes_and_reports_no_pairs():
    rows=[]
    for regime in audit.prob.REGIMES:
        rows.append(dict(index=0,plan='SQMD',model='G11',regime=regime,available=True,error=1.,coverage90_voltage=.4))
    new=deepcopy(rows);new[0].update(error=0.,coverage90_voltage=.9)
    new[1].update(error=0.,coverage90_voltage=.7)
    pair=audit.paired(rows,new,32)['SQMD:A_G11->B_G11']
    assert pair['error']['histories']==1 and pair['error']['missing_pairs']==31
    assert pair['error']['B_minus_A']==-1.
    assert pair['coverage90_voltage']['B_minus_A']==pytest.approx(.4)
    new[1]['available']=False
    missing=audit.paired(rows,new,32)['SQMD:A_G11->B_G11']['error']
    assert missing['histories']==0 and missing['missing_pairs']==32 and missing['B_minus_A'] is None


def test_saved_score_and_fixed_data_corruption_are_detected():
    source=record();item=audit.prob.audit_history(source,SETTINGS)
    for row in item['rows']:
        if row['model'] not in ('G00','G11') or row['plan']!='constant':continue
        source['actions']['shared']['regimes'][row['regime']]['scores'][row['model']]=dict(
            available=True,noisy=dict(error=row['error']),clean=dict(error=row['clean']['error']))
    assert not audit.stored_score_comparison(source,item)
    source['actions']['shared']['regimes']['regular']['scores']['G11']['noisy']['error']=.5
    assert audit.stored_score_comparison(source,item)
    other=deepcopy(source);other['plans']['SQMD']['power']=[[2.]]
    assert 'SQMD: declared/action power differs' in audit.fixed_data_comparison(source,other)
    other=deepcopy(source);other['actions']['shared']['regimes']['shift']['values'][0][0]=4.
    assert audit.fixed_data_comparison(source,other)


def test_faults_distinguish_original_missing_plan_and_historical_optimizer_failure():
    original=record();original['plans'].pop('BCMD')
    new=record();new['history_refresh']=dict(status='history_refresh_failure',candidates=dict(a=dict(
        status='candidate_failure',MAP=dict(success=False,status='maximum_iterations'),
        posterior=dict(success=False,status='map_failure'))))
    summary=audit.scientific_failures(new,original)
    assert 'original_A_plan_missing' in summary['categories']
    assert 'historical_MAP_not_accepted' in summary['categories']
    assert summary['MAP_status_counts']=={'maximum_iterations':1}
    assert 'historical_posterior_failure' not in summary['categories']


def test_running_batch_is_not_opened_for_main_statistics(monkeypatch,tmp_path):
    def read(path):
        assert Path(path).name=='controller.json'
        return dict(status='running')
    monkeypatch.setattr(audit,'read',read)
    with pytest.raises(RuntimeError,match='statistical_complete'):
        audit.run(tmp_path,tmp_path/'report',expected_policy='frozen',A_audit_path=tmp_path/'Aaudit')
