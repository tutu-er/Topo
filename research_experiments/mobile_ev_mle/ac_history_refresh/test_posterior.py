"""Affine closed forms, proper evidence and GP cross-covariance checks."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.special import ndtr,logsumexp
from scipy.stats import multivariate_normal,truncnorm

from research_experiments.mobile_ev_mle.ac_history_refresh import posterior as tool
from research_experiments.mobile_ev_mle import ev_bayes_response_ac as old


FAST=dict(seed=31,power=10,max_power=13,replicates=4,min_ess=256.)


def prior(domain=None):
    bounds=dict(edges=1,log_R=np.array([[-1.,1.]]),log_ratio=np.array([[-1.,1.]]),log_X=np.array([[-3.,3.]]))
    if domain is not None:bounds.update(domain)
    mean=np.array([-.1,.05]);cov=np.diag([.2,.1])
    mass=1.
    for j,key in enumerate(('log_R','log_ratio')):
        a,b=bounds[key][0];sd=np.sqrt(cov[j,j]);mass*=ndtr((b-mean[j])/sd)-ndtr((a-mean[j])/sd)
    return dict(mean=mean,cov=cov,bounds=bounds,log_mass=float(np.log(mass)))


def example():
    line=prior();m0=np.r_[line['mean'],np.zeros(4)];C0=block_diag(line['cov'],np.eye(4)*.3)
    J=np.array([[1.,.3,1.,0,0,0],[0.,1.,0.,.5,0,0],[.2,0.,0.,0.,1.,.2]])
    point=np.array([.2,-.2,.3,-.1,.02,.1]);offset=np.array([.02,-.1,.2]);y=np.array([.4,-.2,.1])
    return line,m0,C0,J,point,offset,y


def calculate(data=None,**kwargs):
    line,m0,C0,J,point,offset,y=example() if data is None else data
    return tool.affine_local_posterior(m0,C0,y,point,offset+J@point,J,line,
        truncation_options=FAST,**kwargs)


def test_linear_gaussian_closed_form_and_logdet():
    line,m0,C0,J,point,offset,y=example();got=calculate()
    expected=old.gaussian_linear_update(m0,C0,y,offset,J)
    assert got['success']
    np.testing.assert_allclose(got['untruncated']['mean'],expected['mean'],atol=2e-15)
    np.testing.assert_allclose(got['untruncated']['cov'],expected['cov'],atol=2e-15)
    assert got['gaussian_log_density']==pytest.approx(expected['log_density'],abs=2e-15)
    S=np.eye(len(y))+J@C0@J.T
    assert got['gaussian_log_density']==pytest.approx(multivariate_normal.logpdf(y,mean=offset+J@m0,cov=S),abs=2e-15)
    assert got['gaussian_logdet']==pytest.approx(np.linalg.slogdet(S)[1],abs=2e-15)
    assert got['log_evidence_white']==pytest.approx(got['gaussian_log_density']+got['logZpost']-line['log_mass'])


def test_map_expansion_location_does_not_double_condition_affine_likelihood():
    data=list(example());one=calculate(tuple(data));data[4]=data[4]+np.arange(6)*.2
    two=calculate(tuple(data))
    np.testing.assert_allclose(one['untruncated']['mean'],two['untruncated']['mean'],atol=2e-15)
    np.testing.assert_allclose(one['bounded']['mean'],two['bounded']['mean'],atol=2e-15)
    assert one['log_evidence_white']==pytest.approx(two['log_evidence_white'],abs=2e-15)


def test_active_line_boundary_distinguishes_map_untruncated_and_truncated_mean():
    line=prior(dict(log_R=np.array([[-1.,0.]])))
    m0=np.r_[line['mean'],0.];C0=block_diag(line['cov'],.3)
    J=np.array([[3.,0.,0.]]);y=np.array([1.]);point=np.array([0.,.05,0.])
    result=tool.affine_local_posterior(m0,C0,y,point,J@point,J,line,truncation_options=FAST)
    mu=result['untruncated']['mean'][0];sd=np.sqrt(result['untruncated']['cov'][0,0])
    assert mu>0. and point[0]==0.
    assert -1.<result['bounded']['mean'][0]<0.
    expected=truncnorm.mean((-1.-mu)/sd,-mu/sd,loc=mu,scale=sd)
    assert result['bounded']['mean'][0]==pytest.approx(expected,abs=2e-4)


def test_prior_normalization_counts_once_and_raw_meter_jacobian():
    data=list(example());data[3]=np.zeros_like(data[3]);line,m0,C0,J,point,offset,y=data
    result=calculate(tuple(data),log_whitening_jacobian=7.3)
    assert result['log_evidence_white']==pytest.approx(multivariate_normal.logpdf(y,mean=offset,cov=np.eye(len(y))),abs=3e-14)
    assert result['log_evidence_raw']==pytest.approx(result['log_evidence_white']+7.3)
    assert result['historical_likelihood_used_once'] and result['original_prior_used_once']
    assert not result['old_topology_weights_used']


def test_future_gp_mean_covariance_and_line_state_cross_retained():
    local=calculate();KD=np.eye(4)*.3;KFD=.7*KD;KFF=KD.copy()
    future=tool.future_marginal(local,KD,KFD,KFF);d=2
    assert future['success']
    for label in ('untruncated','bounded'):
        history=local[label];out=future[label]
        np.testing.assert_allclose(out['mean'][d:],.7*history['mean'][d:],atol=1e-14)
        np.testing.assert_allclose(out['cov'][:d,d:],.7*history['cov'][:d,d:],atol=1e-14)
        np.testing.assert_allclose(out['cov'][d:,d:],.49*history['cov'][d:,d:]+.51*KD,atol=1e-14)
    assert abs(future['bounded']['cov'][0,2])>.01
    assert not future['history_used_again']


def test_fresh_complete_topology_bank_uses_only_current_proper_evidence():
    local=calculate();future=tool.future_marginal(local,np.eye(4)*.3,np.eye(4)*.21,np.eye(4)*.3)
    other=deepcopy(future);other['log_evidence_white']+=np.log(3.)
    candidates=[SimpleNamespace(name='one',n=1,m=1),SimpleNamespace(name='two',n=1,m=1)]
    context,record=tool.build_forecast_context(candidates,dict(one=future,two=other),None,horizon=1)
    np.testing.assert_allclose(context['prior'],[.25,.75],atol=2e-15)
    assert record['topology_prior']=='uniform fresh original prior'
    assert not record['old_topology_weights_used']
    assert set(record['candidate_updates'])=={'one','two'}
    for name,row in record['candidate_updates'].items():
        assert row['TV_untruncated_future']['mean'].shape==(6,)
        assert row['future_latent_mean'].shape==(1,4)
    with pytest.raises(ValueError,match='complete candidate'):
        tool.build_forecast_context(candidates,dict(one=future),None,horizon=1)
    failed=deepcopy(future);failed.update(success=False,status='precision_not_met')
    with pytest.raises(ValueError,match='posterior failure'):
        tool.build_forecast_context(candidates,dict(one=future,two=failed),None,horizon=1)


def test_integration_failure_retains_untruncated_evidence_and_candidate():
    line,m0,C0,J,point,offset,y=example()
    result=tool.affine_local_posterior(m0,C0,y,point,offset+J@point,J,line,
        truncation_options=dict(power=3,max_power=3,replicates=2,min_ess=1e9))
    assert not result['success'];assert result['status']=='precision_not_met'
    assert result['bounded'] is None and result['log_evidence_white'] is None
    assert np.isfinite(result['gaussian_log_density'])
    assert result['untruncated']['mean'].shape==m0.shape


def test_future_invalid_process_kernel_is_not_silently_clipped():
    local=calculate()
    with pytest.raises(ValueError,match='indefinite'):
        tool.future_marginal(local,np.eye(4),np.eye(4)*2,np.eye(4))


def test_line_prior_normalizer_must_match_the_original_gaussian():
    data=list(example());data[1]=data[1].copy();data[1][0]+=.1
    with pytest.raises(ValueError,match='original line-prior'):
        calculate(tuple(data))


def test_not_converged_map_candidate_is_preserved_as_failure():
    assert tool.final_local_posterior(None,dict(success=True,converged=False,status='budget'))['status']=='map_failure'
    local=calculate()
    assert not local['nonlinear_local_minimum_certified']
    assert not local['global_map_certified'] and not local['true_Hessian_used']


def test_problem_adapter_whitening_and_final_affine_historical_check():
    line,m0,C0,_,_,_,_=example();J=np.zeros((5,6));J[:,:5]=np.eye(5)
    history=SimpleNamespace(p=np.array([[.1]]),q=np.array([[.2]]),v=np.array([[1.]]),
        master_p=np.array([.3]),master_q=np.array([.2]))
    meter=SimpleNamespace(p_sigma=.1,q_sigma=.2,v_sigma=.02,master_p_sigma=.1,master_q_sigma=.2)
    def evaluate(w):return dict(mean=J@w,Jacobian=J,physical_ok=True)
    problem=SimpleNamespace(prior_mean=m0,prior_covariance=C0,line_prior=line,
        candidate=SimpleNamespace(name='affine'),history=history,meter=meter,evaluate=evaluate)
    result=tool.final_local_posterior(problem,dict(success=True,converged=True,map_w=m0),truncation_options=FAST)
    expected=-sum(np.log(s) for s in (.1,.2,.02,.1,.2))
    assert result['success'] and result['candidate']=='affine'
    assert result['log_whitening_jacobian']==pytest.approx(expected)
    assert result['log_evidence_raw']-result['log_evidence_white']==pytest.approx(expected)
    assert result['bounded_center_AC_check']['affine_error_whitened_max']<1e-14


def test_future_record_reconstructs_original_bounded_sampling_target():
    from research_experiments.mobile_ev_mle.ev_ac_repair_propagation import sample_posterior
    local=calculate();future=tool.future_marginal(local,np.eye(4)*.3,np.eye(4)*.21,np.eye(4)*.3)
    candidate=SimpleNamespace(name='one',n=1,m=1)
    _,record=tool.build_forecast_context([candidate],dict(one=future),None,horizon=1)
    sampled=sample_posterior(record['candidate_updates']['one'],draws=256,replicates=2,seed=99)
    assert sampled['status']=='ok' and sampled['samples'].shape==(2,256,6)
    assert sampled['line_dimension']==2
