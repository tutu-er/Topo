"""Independent AC replay, conditional sampling and moment identities."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import qmc

from . import ev_ac_repair_propagation as tool
from . import ev_bayes_response_ac as old
from . import ev_bayes_response_truncation as trunc
from .candidates import RadialCandidate
from .simulation import model
from .symmetric_ac_local import mean_and_jacobian


@pytest.fixture
def inputs():
    candidate=RadialCandidate.from_supports('test',3,((0,1),))
    rng=np.random.default_rng(21); N,T,n,m=5,3,candidate.n,candidate.m
    r=rng.uniform(.004,.018,(N,m));x=r*.25
    latent=np.tile(np.r_[np.full(n,.12),np.full(n,.035),.1,.04],(N,T,1))
    latent+=rng.normal(0,.002,latent.shape)
    power=np.array([[2.,0,0],[-1.,0,0],[0.,3.,0.]])
    return candidate,r,x,latent,np.linspace(.999,1.001,T),model(n),power


def test_batched_complete_ac_matches_scalar_bias_and_channel_order(inputs):
    c,r,x,z,root,meter,u=inputs
    actual=tool.batched_ac(c,r,x,z,root,meter,power=u)
    assert actual['status']=='ok'
    for j in range(len(r)):
        s=z[j].copy();s[:,:c.n]+=u/20.
        expected=mean_and_jacobian(c,r[j],x[j],s[:,:c.n],s[:,c.n:2*c.n],s[:,-2],s[:,-1],root,meter,with_jacobian=False)
        np.testing.assert_allclose(actual['mean'][j],expected['mean'],rtol=2e-13,atol=2e-10)
    assert actual['diagnostics']['equation_residual'].max()<2e-10
    assert actual['diagnostics']['power_balance_residual'].max()<2e-10


def test_ev_power_added_once_and_source_reactive_is_retained(inputs):
    c,r,x,z,root,meter,u=inputs
    original=z.copy();a=tool.batched_ac(c,r,x,z,root,meter,power=u)
    b=tool.batched_ac(c,r,x,z,root,meter,power=np.zeros_like(u))
    np.testing.assert_array_equal(z,original)
    np.testing.assert_allclose(a['physical']['p']-b['physical']['p'],np.broadcast_to(u/20.,a['physical']['p'].shape))
    np.testing.assert_array_equal(a['physical']['q'],b['physical']['q'])
    altered=z.copy();altered[:,:,-1]+=.01
    reactive=tool.batched_ac(c,r,x,altered,root,meter,power=u)
    assert np.all(reactive['physical']['master_q']>a['physical']['master_q'])
    assert np.max(abs(reactive['physical']['v']-a['physical']['v']))>0


def test_control_variate_exact_for_affine_outputs():
    rng=np.random.default_rng(9);m=np.array([2.,-1.,.3]);C=np.array([[1,.2,0],[.2,2.,.3],[0,.3,.5]])
    F=rng.normal(size=(5,3));mu=rng.normal(size=5)
    # Deliberately biased small sample; the known linear moments correct it.
    W=rng.normal(size=(11,3))+8.
    Y=mu+(W-m)@F.T
    mm,cc,diag=tool.control_variate_moments(W,Y,m,mu,F,C)
    np.testing.assert_allclose(mm,mu,atol=2e-14)
    np.testing.assert_allclose(cc,F@C@F.T,atol=2e-13)
    assert diag['remainder_RMS']<1e-13


def test_nonlinear_control_variate_moments_include_cross_terms():
    uniforms=qmc.Sobol(d=1,scramble=True,seed=410).random_base2(17)
    W=ndtri(uniforms)
    # f(w)=w+.2w^2 has E f=.2 and Var f=1+.04*Var(w^2)=1.08.
    # Moving the expansion center away from zero makes Cov(g,r) nonzero.
    center=np.array([.7]);mu0=np.array([.7+.2*.7**2]);F=np.array([[1.+.4*.7]])
    # The known linear mean at Gaussian mean zero is -.098. Pass a zero mean
    # center representation so the moment identity E f=mu0+E r remains exact.
    linear_mean=mu0-F@center
    Y=W+.2*W**2
    mu,cov,diag=tool.control_variate_moments(W,Y,np.zeros(1),linear_mean,F,np.eye(1))
    np.testing.assert_allclose(mu,[.2],atol=7e-5)
    np.testing.assert_allclose(cov,[[1.08]],atol=6e-4)
    g=W@F.T;r=Y-(linear_mean+g)
    assert abs(np.cov(g[:,0],r[:,0])[0,1])>.2


def sampling_update():
    bounds=dict(edges=1,log_R=np.array([[-.5,.5]]),log_ratio=np.array([[-.5,.5]]),log_X=np.array([[-2.,2.]]))
    C=np.array([[.04,.005,-.035],[.005,.025,.008],[-.035,.008,.1]])
    settings=dict(seed=31,power=10,max_power=13,replicates=4,min_ess=256.)
    return dict(line_prior=dict(mean=np.zeros(2),bounds=bounds),
        TV_untruncated_future=dict(mean=np.zeros(3),cov=C),TV_integration_options=settings)


def test_bounded_joint_sampling_preserves_cross_and_seed():
    update=sampling_update();a=tool.sample_posterior(update,draws=8192,replicates=2,seed=71)
    b=tool.sample_posterior(update,draws=8192,replicates=2,seed=71)
    assert a['status']=='ok';np.testing.assert_array_equal(a['samples'],b['samples'])
    W=a['samples'].reshape(-1,3)
    assert trunc.contains_eta(W[:,:2],update['line_prior']['bounds'],tolerance=1e-13).all()
    C=update['TV_untruncated_future']['cov'];gain=C[2:,:2]@np.linalg.inv(C[:2,:2])
    residual=W[:,2:]-W[:,:2]@gain.T
    assert abs(residual.mean())<.002
    np.testing.assert_allclose(np.cov(W,rowvar=False)[0,2],C[0,2],atol=.004)
    c=tool.sample_posterior(update,draws=8192,replicates=2,seed=72)
    assert not np.array_equal(a['samples'],c['samples'])


def test_invalid_physical_draws_retained_and_no_renormalization(inputs):
    c,r,x,z,root,meter,u=inputs
    z[1,:,-1]=.6
    domain=tool.batched_ac(c,r,x,z,root,meter,power=u)
    assert domain['mean'].shape[0]==len(z)
    assert domain['diagnostics']['source_q_violations'][1]
    assert domain['diagnostics']['discarded_draws']==0
    r[2]=30.;x[2]=10.
    failed=tool.batched_ac(c,r,x,z,root,meter,power=u,max_iter=60)
    assert failed['status']=='ac_failure'
    assert failed['diagnostics']['ac_failure'][2]
    assert np.isnan(failed['mean'][2]).all()
    assert np.isfinite(failed['mean'][0]).all()
    assert len(failed['mean'])==len(z)


def small_context(inputs):
    c,r,x,z,root,meter,u=inputs;T=len(root);d=2*c.m
    eta=np.c_[np.log(r[0]),np.log(x[0]/r[0])].ravel();center=np.r_[eta,z[0].ravel()]
    C=np.zeros((len(center),len(center)));C[d:,d:]=np.eye(len(center)-d)*1e-8
    prior=trunc.line_prior(c.m,(.001,.03),(.0001,.02),(.1,.4))
    update=dict(line_prior=prior,TV_untruncated_future=dict(mean=center,cov=C),
        TV_integration_options=dict(seed=0,power=10,max_power=10,replicates=2))
    row=dict(candidate=c,r=r[0],x=x[0],line_sha256=old._line_id(r[0],x[0]),
        line_dimension=d,joint_mean=center,joint_covariance=C,latent_mean=z[0])
    context=dict(rows=[row],horizon=T,n=c.n,root_v=root,future_meter=meter,base_kw=20.,
        log_prior_all_history=np.zeros(1),options=old._options(None))
    return context,dict(candidate_updates={c.name:update}),u


def test_forecast_four_banks_same_prior_and_zero_parameter_uncertainty(inputs):
    context,record,u=small_context(inputs);T=context['horizon']
    result=tool.forecast_moments(context,record,u,draws=128,replicates=2,seed=81)
    assert result['status']=='ok';assert set(result['banks'])=={'G00','G10','G01','G11'}
    for bank in result['banks'].values():
        np.testing.assert_array_equal(bank['log_prior'],np.zeros(1))
        np.linalg.cholesky(bank['covariances'][0,0])
    assert result['diagnostics']['AC_states']==2*128*T
    assert result['diagnostics']['historical_posterior_unchanged']
    assert result['diagnostics']['output_density_Gaussian_shape_approximation']
    assert result['rows'][0]['discarded_draws']==0


def test_negative_cv_falls_back_to_unfiltered_sample_covariance(inputs,monkeypatch):
    context,record,u=small_context(inputs);original=tool.control_variate_moments
    def negative(*args):
        mu,cov,diag=original(*args)
        return mu,-np.eye(len(mu))*2.,diag
    monkeypatch.setattr(tool,'control_variate_moments',negative)
    result=tool.forecast_moments(context,record,u,draws=32,replicates=2,seed=81)
    assert result['status']=='ok'
    assert result['rows'][0]['covariance_roundoff']['ordinary_AC_sample_covariance_fallback']
    np.linalg.cholesky(result['banks']['G11']['covariances'][0,0])


def test_one_physical_violation_blocks_corrected_banks(inputs,monkeypatch):
    context,record,u=small_context(inputs);original=tool.batched_ac
    def violated(*args,**kwargs):
        result=original(*args,**kwargs)
        result['diagnostics']['source_q_violations'][0]=True
        return result
    monkeypatch.setattr(tool,'batched_ac',violated)
    result=tool.forecast_moments(context,record,u,draws=32,replicates=2,seed=81)
    assert result['status']=='propagation_failure'
    assert set(result['banks'])=={'G00'}
    assert result['rows'][0]['status']=='physical_domain_failure'
    assert result['rows'][0]['violation_fractions']['source_q_violations']==pytest.approx(1/64)
    assert result['rows'][0]['discarded_draws']==0


def test_history_cache_reuse_requires_exact_posterior_fingerprint(inputs):
    from copy import deepcopy
    context,record,u=small_context(inputs);cache={}
    tool.forecast_moments(context,record,u,draws=32,replicates=2,seed=81,posterior_cache=cache)
    assert len(next(iter(cache.values()))['posterior_fingerprint'])==64
    # Different draws is allowed and reconstructs the same posterior.
    tool.forecast_moments(context,record,u,draws=64,replicates=2,seed=81,posterior_cache=cache)
    assert next(iter(cache.values()))['diagnostics']['draws']==64
    changed=deepcopy(record);name=context['rows'][0]['candidate'].name
    changed['candidate_updates'][name]['TV_untruncated_future']['mean'][0]+=.001
    with pytest.raises(ValueError,match='fingerprint mismatch'):
        tool.forecast_moments(context,changed,u,draws=64,replicates=2,seed=81,posterior_cache=cache)


def test_internal_voltage_is_audit_only_while_terminal_domain_is_original(inputs,monkeypatch):
    c,r,x,z,root,meter,u=inputs
    normal=tool.batched_ac(c,r,x,z,root,meter,power=u)
    assert 'internal_voltage_violations' in normal['diagnostics']
    np.testing.assert_array_equal(normal['diagnostics']['voltage_violations'],
        normal['diagnostics']['terminal_voltage_violations'])
    original=tool.batched_ac
    def internal_only(*args,**kwargs):
        result=original(*args,**kwargs)
        result['diagnostics']['internal_voltage_violations'][:]=True
        result['diagnostics']['all_node_voltage_violations'][:]=True
        return result
    monkeypatch.setattr(tool,'batched_ac',internal_only)
    context,record,u=small_context(inputs)
    forecast=tool.forecast_moments(context,record,u,draws=32,replicates=2,seed=81)
    assert forecast['status']=='ok'
    assert forecast['rows'][0]['violation_fractions']['internal_voltage_violations']==1.
    assert forecast['rows'][0]['violation_fractions']['terminal_voltage_violations']==0.
