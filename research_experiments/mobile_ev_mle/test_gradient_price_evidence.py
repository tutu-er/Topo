"""Small real-AC fixtures and independent compact/statistic integrity tests."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from . import gradient_price_evidence as evidence
from .likelihood import Observations,RadialCandidate,ac_predict,concat_observations
from .simulation import model
from .single_site_milp import SingleSiteConstraints,SingleSitePlan,constant_site_plan


def fixture():
    c=RadialCandidate.from_supports('pair01',3,((0,1),));meter=model(3)
    r=np.linspace(.008,.026,c.m);x=np.linspace(.004,.011,c.m)
    p=np.array([[.1,.12,.09],[.11,.1,.13]]);q=np.array([[.03,.04,.02],[.025,.035,.045]])
    sp=np.array([.07,.08]);sq=np.array([.02,.03]);root=np.array([1.,1.001])
    def observed(power):
        physical=ac_predict(c,r,x,power,q,sp,sq,root)
        means={k:physical[k]+getattr(meter,'bias_'+k) for k in ('p','q','v')}
        means.update(master_p=physical['master_p'],master_q=physical['master_q'])
        values={k:np.asarray(v).copy() for k,v in means.items()}
        values['p'][0,0]+=.0006;values['v'][1,2]+=.00012
        return Observations(root_v=root,**values),means,physical
    h,means,physical=observed(p)
    # Exactly one standardized P error and one standardized V error.
    normalizer=2*(3*np.log(2*np.pi*.0006**2)+3*np.log(2*np.pi*.0004**2)+3*np.log(2*np.pi*.00012**2)+2*np.log(2*np.pi*.001**2))
    row=dict(name=c.name,candidate=c.name,supports=[list(s) for s in c.supports],success=True,
        r=r.tolist(),x=x.tolist(),p=p.tolist(),q=q.tolist(),source_p=sp.tolist(),source_q=sq.tolist(),
        bias_p=np.asarray(meter.bias_p).tolist(),bias_q=np.asarray(meter.bias_q).tolist(),bias_v=np.asarray(meter.bias_v).tolist(),
        loss=2.,half_loss=1.,loglik=-1.-.5*normalizer,predicted={k:v.tolist() for k,v in means.items()},
        diagnostics=dict(local_scaled_singular_values=[1,2,3],local_projected_shared_singular_values=[4,5],
                         min_voltage=float(abs(physical['voltage']).min()),max_voltage=float(abs(physical['voltage']).max())))
    constraints=SingleSiteConstraints(energy=.075)
    zero=SingleSitePlan(0,np.zeros((2,3)),np.zeros((2,3)),np.zeros(2,int),np.full(3,.4))
    charged=constant_site_plan(0,2,3,constraints)['plan']
    records=[]
    for method,plan in [('none',zero),('constant_fixed',charged)]:
        future,fmeans,fphysical=observed(p+plan.p)
        joint=deepcopy(row)
        for k in ('p','q'):joint[k]=np.concatenate((np.asarray(row[k]),np.asarray(row[k])+getattr(plan,k))).tolist()
        for k in ('source_p','source_q'):joint[k]=np.tile(row[k],2).tolist()
        joint['predicted']={k:np.concatenate((means[k],fmeans[k])).tolist() for k in means}
        joint.update(loss=4.,half_loss=2.,loglik=-2.-normalizer)
        joint['diagnostics'].update(min_voltage=min(float(abs(physical['voltage']).min()),float(abs(fphysical['voltage']).min())),
                                    max_voltage=max(float(abs(physical['voltage']).max()),float(abs(fphysical['voltage']).max())))
        records.append(dict(method=method,status='complete',action=plan.summary(),observations={k:np.asarray(getattr(future,k)).tolist() if getattr(future,k) is not None else None for k in
            ('p','q','v','root_v','master_p','master_q','source_p','source_q')},fits=[joint],losses={c.name:4.},gaps={c.name:0.},
            winner=c.name,true_gap=0.,true_name=c.name,true_covered=True))
    config=dict(future_points=2,history_points=2,alpha=.1)
    source={};pid=hashlib.sha256(json.dumps(dict(config=config,source_sha256=source),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    trial=dict(history_id='1',seed=1,config=config,source_sha256=source,policy_id=pid,
        history_observations={k:np.asarray(getattr(h,k)).tolist() if getattr(h,k) is not None else None for k in
            ('p','q','v','root_v','master_p','master_q','source_p','source_q')},historical_fits=[row],historical_failures={},
        candidate_names=[c.name],true_name=c.name,true_covered=True,truth=dict(supports=[[0,1]],r_pu=r.tolist(),x_pu=x.tolist()),
        device_constraints=asdict(constraints),records=records,complete=True,relative_score=0.,calibration_score_is_infinite=False,
        selection_uses_truth=False,selection_uses_future_noise=False,selection=dict(searches={},methods={}),
        independent_history_unit=True)
    return row,h,trial


def test_compact_removes_only_two_named_diagnostics_and_keeps_states():
    value=dict(local_scaled_singular_values=[1,2],local_projected_shared_singular_values=[3],
        nested=[dict(source_p=[.1],p=[[.2]],cost=4.,incumbent=-1.,best_bound=2.,
            local_scaled_singular_values=[6],failure='kept',samples=[dict(stat=7.)])],
        unrelated_singular_values=[8],site_evaluations=dict(duplicate='kept'))
    expected=dict(nested=[dict(source_p=[.1],p=[[.2]],cost=4.,incumbent=-1.,best_bound=2.,failure='kept',samples=[dict(stat=7.)])],
                  unrelated_singular_values=[8],site_evaluations=dict(duplicate='kept'))
    assert evidence.compact_value(value)==expected
    assert value['local_scaled_singular_values']==[1,2]


def test_real_ac_replay_recovers_hand_calculated_sse_and_gaussian_loglik():
    row,obs,trial=fixture()
    replay=evidence.replay_fit(row,obs)
    assert replay['success'],replay['errors']
    assert replay['full_loss']==pytest.approx(2.,abs=1e-8)
    assert replay['half_loss']==pytest.approx(1.,abs=1e-8)
    assert replay['loglik']==pytest.approx(row['loglik'],abs=1e-8)
    assert replay['parameter_domain'] and replay['power_balance_residual']<1e-10
    audited=evidence.audit_trial(trial)
    assert audited['success'],audited['errors']
    assert audited['actual_fit_replay_count']==3
    assert audited['recomputed_relative_score']==0.


def test_replay_detects_changed_cost_source_bias_and_parameter_domain():
    row,obs,_=fixture()
    for key,value in [('loss',3.),('loglik',0.),('source_p',[.1,.1]),('bias_v',[0,0,0]),('r',[.2]*len(row['r']))]:
        altered=deepcopy(row);altered[key]=value
        result=evidence.replay_fit(altered,obs)
        assert not result['success'],key


def test_support_permutation_replays_same_physical_parameters():
    row,obs,_=fixture();indices=list(reversed(range(len(row['supports']))))
    for key in ('supports','r','x'):row[key]=[row[key][i] for i in indices]
    result=evidence.replay_fit(row,obs)
    assert result['success'],result['errors']
    assert result['full_loss']==pytest.approx(2.,abs=1e-8)


def test_recorded_nonfinite_failure_is_retained_and_not_recast_as_success():
    row,obs,_=fixture();row.update(success=False,loss=None,half_loss=None,loglik=None,r=[None]*len(row['r']))
    result=evidence.replay_fit(row,obs)
    assert result['success'] and not result['replayed'] and result['recorded_failed_fit']
    row['success']=True
    assert not evidence.replay_fit(row,obs)['success']


def test_calibration_rank_and_infinite_failure_mass_recomputed():
    audits=[dict(history_id=str(i),policy_id='p',calibration_score_is_infinite=False,recomputed_relative_score=float(i)) for i in range(12)]
    saved=dict(calibration_history_ids=[str(i) for i in range(12)],relative=dict(rank=12,sample_count=12,threshold=11.,vacuous=False),
               alpha=.1,failed_history_count=0,all_history_count=12)
    assert evidence.audit_calibration(audits,saved,.1)['success']
    saved['relative']['threshold']=10.
    assert not evidence.audit_calibration(audits,saved,.1)['success']
    saved['relative'].update(threshold=None,vacuous=True);saved['failed_history_count']=1
    audits[0].update(calibration_score_is_infinite=True,recomputed_relative_score=None)
    result=evidence.audit_calibration(audits,saved,.1)
    assert result['success'] and result['rank']==12 and result['vacuous']


def test_trial_operation_and_retained_decisions_are_independently_checked():
    _,_,trial=fixture()
    gate=dict(policy_id=trial['policy_id'],calibration_history_ids=['old'],relative=dict(vacuous=False,threshold=0.))
    for r in trial['records']:
        r.update(retained_candidates=['pair01'],false_rejection=False,only_truth_retained=True,unresolved=False,wrong_candidates_removed=0)
    trial['any_method_false_rejection']=False
    assert evidence.audit_trial(trial,calibration=gate)['success']
    altered=deepcopy(trial);altered['records'][1]['action']['soc'][-1]+=.01
    assert not evidence.audit_trial(altered,calibration=gate)['success']
    altered=deepcopy(trial);altered['records'][0]['retained_candidates']=[]
    assert not evidence.audit_trial(altered,calibration=gate)['success']
    gate['calibration_history_ids']=['1']
    assert not evidence.audit_trial(trial,calibration=gate)['success']


def test_compact_export_keeps_raw_archive_hash_and_all_replay_states(tmp_path):
    _,_,trial=fixture();raw=tmp_path/'raw';compact=tmp_path/'compact'
    config=dict(count=1,seeds=[1],config=trial['config'],source_sha256=trial['source_sha256'],policy_id=trial['policy_id'])
    evidence.write_json(raw/'pilot'/'config.json',config)
    file=raw/'pilot'/'trial_1.json.gz';evidence.write_json(file,trial,compress=True);digest=evidence.sha(file)
    report=evidence.export_phase(raw,compact,'pilot',source_root=None)
    assert report['success'],report['errors']
    assert evidence.sha(file)==digest and report['manifest'][0]['raw_sha256']==digest
    copied=evidence.read_json(compact/'pilot'/'trial_1.json.gz')
    assert copied==evidence.compact_value(trial)
    assert copied['records'][1]['fits'][0]['source_p']==trial['records'][1]['fits'][0]['source_p']
    config['count']=2;evidence.write_json(raw/'pilot'/'config.json',config)
    with pytest.raises(ValueError,match='incomplete'):evidence.export_phase(raw,compact,'pilot',source_root=None)


def test_summary_counts_and_seed_annotations_checked_independently():
    _,_,trial=fixture()
    gate=dict(policy_id=trial['policy_id'],calibration_history_ids=['old'],relative=dict(vacuous=False,threshold=0.))
    for row in trial['records']:
        row.update(retained_candidates=['pair01'],false_rejection=False,only_truth_retained=True,unresolved=False,wrong_candidates_removed=0)
    trial['any_method_false_rejection']=False
    report=evidence.audit_trial(trial,calibration=gate)
    saved=dict(all_history_count=1,complete_history_count=1,failed_history_count=0,candidate_coverage_failures=0,
        any_method_false_rejections=0,policy_false_rejection_denominator=1,
        method_summary=[dict(method=name,all_history_count=1,complete_count=1,failed_count=0,covered_complete_count=1,
            false_rejections=0,only_truth_retained=1,unresolved=0,false_rejection_denominator=1,incomplete_retained_all_count=0)
            for name in ('none','constant_fixed')])
    assert evidence.audit_summary([report],saved,True)['success']
    saved['method_summary'][0]['false_rejection_denominator']=0
    assert not evidence.audit_summary([report],saved,True)['success']
    trial['seeds']=dict(generator=1,policy=1,evaluation_noise=1)
    assert 'independent_seed_streams' in evidence.audit_trial(trial,calibration=gate)['errors']
