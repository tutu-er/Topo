import numpy as np
import pytest

from research_experiments.mobile_ev_mle.ac_history_refresh import report
from research_experiments.mobile_ev_mle.likelihood import Observations
from research_experiments.mobile_ev_mle.symmetric_ac_local import whiten_observations
from types import SimpleNamespace
from hashlib import sha256
from research_experiments.mobile_ev_mle.ac_history_refresh import io


def test_paired_history_unit_and_missing_denominator():
    a=[];b=[]
    for i, va, vb in ((0, 1., 0.), (1, 0., .5)):
        for regime in ('regular','shift'):
            a.append(dict(index=i,plan='SQMD',model='G11',regime=regime,available=True,error=va))
            b.append(dict(index=i,plan='SQMD',model='G11',regime=regime,available=True,error=vb))
    b[-1]['available']=False
    row=report.paired_rows(a,b,3)['SQMD:A_G11->B_G11']['error']
    assert row['histories']==1 and row['total_histories']==3 and row['missing_pairs']==2
    assert row['B_minus_A']==-1 and row['paired_indices']==[0]


def test_channel_layout_and_equal_history_weight():
    n=3;d=8*(3*n+2)
    mean=np.zeros((1,2,d));cov=np.array([[np.eye(d),np.eye(d)]])
    mean[:,:,2*8*n:3*8*n]=1.
    obs=Observations(p=np.zeros((8,n)),q=np.full((8,n),2.),v=np.ones((8,n)),
        root_v=np.ones(8),master_p=np.zeros(8),master_q=np.zeros(8),source_p=None,source_q=None)
    meter=SimpleNamespace(**{f'{c}_{s}':v for c in ('p','q','v','master_p','master_q')
                            for s,v in (('sigma',1.),('bias',0.))})
    values=whiten_observations(obs,meter)[None]
    bank=dict(means=mean,covariances=cov,log_prior=np.log([.5,.5]))
    record=dict(spec=dict(index=0,bank='n3'),candidate_names=['a','b'],true_name='a',
        plans={'SQMD':{'action_key':'x'}},actions={'x':dict(propagation=dict(banks={'G00':bank},rows=[]),
            regimes={r:dict(values=values,clean=values[0]) for r in ('regular','shift')})})
    rows=report.channel_coverage(record)
    assert len(rows)==2 and all(r['coverage']['P']==1 and r['coverage']['Q']==0 and r['coverage']['V']==1 for r in rows)
    summary=report.summarize_channels(rows)
    assert summary['SQMD__G00__Q']['histories']==1
    assert summary['SQMD__G00__Q']['mean']==0


def test_channel_mismatched_bank_fails():
    bank=dict(means=np.zeros((1,2,2)),covariances=np.array([[np.eye(2),np.eye(2)]]),log_prior=[0,0])
    record=dict(spec=dict(index=0,bank='n3'),candidate_names=['a','b'],true_name='a',
        plans={'SQMD':{'action_key':'x'}},actions={'x':dict(propagation=dict(banks={'G00':bank},rows=[]))})
    with pytest.raises(ValueError):report.channel_coverage(record)


def test_minimal_process_failures_keep_all32_and_missing_plan_denominators(tmp_path):
    """A worker failure can legitimately omit labels, plans and source_A."""
    A=tmp_path/'A';B=tmp_path/'B'
    for root,policy in ((A,'A-policy'),(B,'B-policy')):
        io.save_record(root/'freeze.json',dict(policy_id=policy))
        io.save_record(root/'calibration.json',dict(G11_multiplier=1.,SSE_thresholds={}))
    binding=dict(policy_id='B-policy',calibration_sha256=sha256((B/'calibration.json').read_bytes()).hexdigest())
    io.save_record(B/'evaluation_calibration_binding.json',binding)
    for index in range(32):
        spec=dict(index=index,seed=100+index,phase='evaluation',bank='n3')
        pa=A/'evaluation'/f'history_{index:03d}.json.gz'
        pb=B/'evaluation'/f'history_{index:03d}.json.gz'
        io.save_record(pa,dict(spec=spec,status='process_failure',failures={'worker':'saved failure'}))
        io.save_record(pb,dict(spec=spec,status='process_failure',failures={'worker':'replay failure'},
                               keep_all_on_failure=True,seconds=.01))
        # These fields are written by the B controller even for its minimal
        # exception record, which has no inner source_A/candidate/plan data.
        io.save_record(pb.with_name(f'history_{index:03d}.metadata.json'),dict(
            policy_id='B-policy',result_sha256=sha256(pb.read_bytes()).hexdigest(),
            spec=spec,calibration_binding=binding,
            source_A_sha256=sha256(pa.read_bytes()).hexdigest(),science_source_unchanged=True))
    result=report.compare(A,B,expected=32)
    assert result['histories']==32 and len(result['history_statuses'])==32
    for version in ('A','B'):
        for plan in ('constant','SQMD','BCMD'):
            for model in ('G00','G11','G11_cal'):
                availability=result[version]['summaries'][plan+'__'+model]['availability']
                assert availability['total_histories']==32
                assert availability['histories']==0 and availability['failed_histories']==32
                assert availability['failed_history_indices']==list(range(32))
                assert availability['error_all_histories_range']==[0.,1.]


def test_zero_complete_pairs_still_report_missing_pair_count():
    pairs=report.paired_rows([],[],32)
    for plan in ('constant','SQMD','BCMD'):
        row=pairs[plan+':A_G11->B_G11']['error']
        assert row['histories']==0 and row['total_histories']==32
        assert row['missing_pairs']==32 and row['paired_indices']==[]
        assert row['B_minus_A'] is None and row['bootstrap95'] is None
