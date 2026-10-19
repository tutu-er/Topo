"""Observed-only reconstruction and fixed-plan replay isolation checks."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from research_experiments.mobile_ev_mle.ac_history_refresh import io,replay
from research_experiments.mobile_ev_mle import ev_bayes_response_ac as old
from research_experiments.mobile_ev_mle.information_context import meter_to_record
from research_experiments.mobile_ev_mle.likelihood import MeasurementModel


def _raw():
    H,V,F,n=12,3,2,3
    def obs(T,offset):
        time=np.arange(offset,offset+T)[:,None]
        p=.11+.01*np.sin(time/3+np.arange(n))
        q=.04+.003*np.cos(time/5+np.arange(n))
        return dict(p=p.tolist(),q=q.tolist(),v=np.full((T,n),.998).tolist(),
            root_v=np.ones(T).tolist(),master_p=(p.sum(1)+.05).tolist(),
            master_q=(q.sum(1)+.02).tolist(),source_p=None,source_q=None)
    structures=[dict(name='star',n=n,supports=[[0,1,2],[0],[1],[2]]),
                dict(name='clade_01',n=n,supports=[[0,1,2],[0,1],[0],[1],[2]])]
    meter=meter_to_record(MeasurementModel(.001,.001,.0001,.002,.002))
    initial={};fits=[]
    for row in structures:
        m=len(row['supports']);state=dict(r=[.002]*m,x=[.0006]*m,
            source_p=[.05]*(H+V),source_q=[.02]*(H+V))
        initial[row['name']]=dict(state=state,usable=True,success=True)
        fits.append(dict(candidate=row['name'],success=True,r=list(state['r']),x=list(state['x']),
                         source_p=[.05]*H,source_q=[.02]*H))
    plans={label:dict(action_key=key,domain=domain,power=np.full((F,n),power).tolist())
        for label,key,domain,power in (('constant','a','fixed_charge',1.),
            ('SQMD','b','mobile_bidir',2.),('BCMD','b','mobile_bidir',2.))}
    dimension=F*(3*n+2)
    actions={key:dict(regimes={regime:dict(values=np.zeros((1,dimension)).tolist(),
        clean=np.zeros(dimension).tolist(),refits='future_joint_forbidden',
        truth_inputs_evaluation_only='future_truth_forbidden') for regime in ('regular','shift')})
        for key in ('a','b')}
    return dict(status='complete',spec=dict(phase='development',index=0,seed=1),
        config=dict(horizon=F,posterior_options={}),observed_training=obs(H,0),
        observed_evidence=obs(V,H),train_meter=meter,evidence_meter=deepcopy(meter),
        future_meter=deepcopy(meter),candidate_names=[r['name'] for r in structures],
        candidate_structures=structures,refit_options=dict(r_bounds=[.0001,.01],
            x_bounds=[.00001,.003],ratio_bounds=[.1,.5],source_p_bounds=[-.5,.5],source_q_bounds=[-.5,.5]),
        refit_history=dict(observed_history_only=True,candidates=initial),
        training_result=dict(fits=fits),plans=plans,actions=actions,
        true_name='clade_01',true_index=1,posterior_centers='old_posterior_forbidden',
        prediction='old_prediction_forbidden',actual_white_noise='future_noise_forbidden',
        true_lines_evaluation_only='true_lines_forbidden')


class Guarded(dict):
    def __init__(self,value,forbidden=(),ready=None):
        super().__init__(value);self.forbidden=set(forbidden);self.ready=ready
    def _check(self,key):
        if key in self.forbidden and (self.ready is None or not self.ready()):
            raise AssertionError('forbidden early access: '+key)
    def __getitem__(self,key):self._check(key);return super().__getitem__(key)
    def get(self,key,default=None):self._check(key);return super().get(key,default)


def test_observed_only_inputs_exclude_old_and_future_containers():
    raw=Guarded(_raw(),forbidden=('actions','true_name','true_index','prediction',
        'posterior_centers','actual_white_noise','true_lines_evaluation_only'))
    prepared=io.prepare_history(raw)
    assert not set(raw.forbidden)&set(prepared)
    assert prepared['history'].t==15 and prepared['K_D'].shape==(120,120)
    assert prepared['K_FD'].shape==(16,120) and prepared['K_FF'].shape==(16,16)
    assert prepared['old_posterior_required'] is False
    assert prepared['old_topology_weights_used'] is False
    assert prepared['source_interface']==1
    for value in prepared['initializations'].values():
        assert value['source']=='observed_H_plus_V_conditional_fit'
        assert value['latent'].shape==(15,8)
        np.testing.assert_array_equal(value['latent'][:,-2:],np.tile([.05,.02],(15,1)))


def test_original_training_only_process_reconstructed_without_A_posterior():
    raw=_raw();raw.pop('prediction');raw.pop('posterior_centers')
    prepared=io.prepare_history(raw)
    hyper=old.estimate_process_hyperparameters(prepared['training'],prepared['train_meter'],
        options=prepared['options'])
    _,K=old.process_kernel(hyper,17)
    np.testing.assert_array_equal(prepared['K_D'],K[:120,:120])
    np.testing.assert_array_equal(prepared['K_FD'],K[120:,:120])
    changed=deepcopy(raw)
    changed['observed_evidence']['p']=(np.asarray(changed['observed_evidence']['p'])+.02).tolist()
    prepared2=io.prepare_history(changed)
    np.testing.assert_array_equal(prepared['K_D'],prepared2['K_D'])
    assert prepared['observed_input_sha256']!=prepared2['observed_input_sha256']


def test_training_initial_fallback_uses_only_historical_source_and_readings():
    raw=_raw();raw['refit_history']['observed_history_only']=False
    prepared=io.prepare_history(raw)
    initial=prepared['initializations']['star']
    assert initial['source']=='training_only_fit_with_observed_evidence_proxy'
    assert initial['source_proxy_in_likelihood'] is False
    np.testing.assert_allclose(initial['latent'][:,-2:],np.tile([.05,.02],(15,1)),atol=1e-15)
    assert initial['uses_future_data'] is False
    raw['training_result']['fits']=[]
    assert io.prepare_history(raw)['initializations']['star']['status']=='initialization_failure'


def test_invalid_history_R_X_falls_back_without_silent_domain_projection():
    raw=_raw();raw['refit_history']['candidates']['star']['state']['r'][0]=.1
    initial=io.prepare_history(raw)['initializations']['star']
    assert initial['source']=='training_only_fit_with_observed_evidence_proxy'
    assert initial['rejected_initializations']
    np.testing.assert_allclose(np.exp(initial['eta'][0::2]),.002)


def test_saved_process_failure_preserved_not_regenerated():
    raw=dict(spec=dict(phase='development',index=0),status='process_failure',failures=['worker'])
    with pytest.raises(io.InputFailure,match='missing observed inputs'):io.prepare_history(raw)
    result=replay.replay_record(raw)
    assert result['status']=='input_failure' and result['n_independent_histories']==1
    assert result['failures']['input'].startswith('InputFailure:')
    assert result['plans']=={} and result['actions']=={}


def test_saved_source_telemetry_is_rejected_instead_of_silently_ignored():
    raw=_raw();raw['observed_training']['source_q']=[.02]*12
    with pytest.raises(io.InputFailure,match='unmetered'):io.prepare_history(raw)


def test_future_evaluation_reads_only_labels_values_and_clean():
    raw=_raw();prepared=io.prepare_history(raw)
    for action in raw['actions'].values():
        for regime,value in action['regimes'].items():
            action['regimes'][regime]=Guarded(value,forbidden=('refits','truth_inputs_evaluation_only'))
    actual=io.evaluation_payload(raw,prepared)
    assert actual['true_index']==1 and actual['candidate_miss'] is False
    assert set(actual['actions']['a']['regular'])=={'values','clean'}
    raw['true_index']=0
    with pytest.raises(io.InputFailure,match='index/name mismatch'):io.evaluation_payload(raw,prepared)


def test_fit_once_per_candidate_and_no_candidate_removed_on_failure():
    prepared=io.prepare_history(_raw());calls=[]
    def problem(candidate,*args,**kwargs):
        assert set(kwargs['options'])=={'source_p_bounds','source_q_bounds','voltage_bounds'}
        return SimpleNamespace(candidate=candidate)
    def solve(problem,eta,latent,**kwargs):
        calls.append(problem.candidate.name)
        return dict(success=problem.candidate.name=='star',status='ok',map_w=np.r_[eta,latent.ravel()])
    def local(problem,fitted,**kwargs):return dict(success=fitted['success'],status='ok')
    def future(local,*args):return dict(success=local['success'],status='ok')
    result=replay.fit_observed(prepared,problem_factory=problem,solve=solve,
        local_update=local,future_update=future,context_builder=lambda *a,**k:pytest.fail('failed bank must not normalize'))
    assert calls==prepared['candidate_names']
    assert result['context'] is None and result['record']['MAP_solves']==2
    assert set(result['record']['candidates'])==set(prepared['candidate_names'])
    assert 'clade_01' in result['record']['failures']


def test_replay_fits_before_forecasting_then_reads_actuals_and_shares_cache():
    events=[];raw=Guarded(_raw(),forbidden=('actions','true_name','true_index'),
                         ready=lambda:len([e for e in events if e[0]=='forecast'])==2)
    def fit(inputs,**kwargs):
        assert 'actions' not in inputs and 'true_index' not in inputs
        events.append(('fit',inputs['candidate_names']))
        return dict(record=dict(failures={},status='ok',MAP_solves=2),context={},context_record={})
    cache_ids=[]
    def forecast(context,record,power,**kwargs):
        events.append(('forecast',float(power[0,0])));cache_ids.append(id(kwargs['posterior_cache']))
        kwargs['posterior_cache'].setdefault('shared',dict(status='ok',samples=np.zeros((2,2,2))))
        D=22;bank=dict(means=np.zeros((1,2,D)),covariances=np.tile(np.eye(D),(1,2,1,1)),
                      log_prior=np.log([.5,.5]))
        return dict(status='ok',rows=[],banks={name:deepcopy(bank) for name in replay.MODEL_NAMES})
    def score(bank,values,true_index,**kwargs):
        assert len([e for e in events if e[0]=='forecast'])==2
        assert true_index==1
        return dict(error=0.)
    result=replay.replay_record(raw,fit_function=fit,forecast_function=forecast,score_function=score)
    assert result['status']=='complete'
    assert [e[0] for e in events]==['fit','forecast','forecast']
    assert len(set(cache_ids))==1 and result['posterior_cache_candidates']==['shared']
    assert result['actions']['b']['methods']==['SQMD','BCMD']
    assert set(result['posterior_sampling']['shared'])=={'status'}
    for action in result['actions'].values():
        assert all(row['scores']['G11']['available'] for row in action['regimes'].values())


def test_replay_missing_plan_or_measurement_retains_failure_denominator():
    raw=_raw();raw['plans'].pop('BCMD');raw['actions']['a']['regimes'].pop('shift')
    def fit(inputs,**kwargs):
        return dict(record=dict(failures={'star':'failure'},status='failure',MAP_solves=2),
                    context=None,context_record=None)
    result=replay.replay_record(raw,fit_function=fit)
    assert result['status']=='partial_failure' and result['n_independent_histories']==1
    assert result['failures']['plan_BCMD']=='saved_plan_missing'
    score=result['actions']['a']['regimes']['shift']['scores']['G11']
    assert score['retained_candidates']==raw['candidate_names']
    assert score['unresolved']==1 and score['false_rejection']==0
    assert score['MAP_error'] is None and score['MAP_error_missing_range']==[0,1]


def test_compacted_history_matrices_are_descriptors_and_future_values_unchanged():
    matrix=np.arange(16.).reshape(4,4)
    compact=replay._compact_MAP(dict(J_eta=matrix,eta=np.ones(2)))
    assert compact['J_eta']['shape']==[4,4]
    assert 'sha256' in compact['J_eta']
    posterior=replay._compact_local(dict(line_dimension=2,success=True,
        untruncated=dict(mean=np.zeros(4),cov=matrix),bounded=None,affine_jacobian=matrix))
    np.testing.assert_array_equal(posterior['untruncated']['line_covariance'],matrix[:2,:2])
    assert posterior['untruncated']['covariance']['shape']==[4,4]


def test_save_roundtrip_refuses_overwrite_and_preserves_original(tmp_path):
    original=tmp_path/'A.json.gz';new=tmp_path/'B.json.gz'
    io.save_record(original,dict(status='original',value=np.arange(3)))
    initial=original.read_bytes()
    io.save_record(new,dict(status='B',value=np.arange(4)))
    assert io.load_record(new)['value']==[0,1,2,3]
    with pytest.raises(FileExistsError):io.save_record(original,dict(status='changed'))
    assert original.read_bytes()==initial
    with pytest.raises(ValueError,match='must differ'):replay.replay_file(original,original)


def test_nonfinite_failure_diagnostics_save_without_losing_failure(tmp_path):
    path=tmp_path/'failure.json.gz'
    io.save_record(path,dict(status='integration_failure',diagnostic=np.array([np.inf,np.nan])))
    saved=io.load_record(path)
    assert saved['status']=='integration_failure' and saved['diagnostic']==[None,None]
    assert saved['serialization_nonfinite']['count']==2
    assert saved['serialization_nonfinite']['paths']==['/diagnostic/0','/diagnostic/1']
