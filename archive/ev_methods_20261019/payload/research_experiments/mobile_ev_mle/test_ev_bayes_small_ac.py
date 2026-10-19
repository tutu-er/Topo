from types import SimpleNamespace
import numpy as np
import pytest
from scipy.linalg import cho_factor,cho_solve
from scipy.stats import multivariate_normal

from . import ev_bayes_small_ac as ac
from . import ev_mobile_v4_cases as cases
from . import likelihood as lk
from . import ev_systematic_refit as refit
from .ev_fixed_train_fit import subset_observations,subset_model
from .ev_bayes_small_study import complete_curve,serial_record
from . import ev_mobile_v3_cases as public
from . import ev_mobile_v4_domain as nested


@pytest.fixture(scope='module')
def context():
    case=cases.make_case('n3',2026102500,1,'regular')
    D=lk.concat_observations(case['train'],case['validation'])
    dm=refit.concat_models((case['train'],case['validation']),(case['tm'],case['vm']))
    train=subset_observations(D,slice(0,80));evidence=subset_observations(D,slice(80,None))
    tm=subset_model(dm,D,slice(0,80));em=subset_model(dm,D,slice(80,None))
    fits={c.name:SimpleNamespace(success=True,r=np.full(c.m,.005),x=np.full(c.m,.002),
        source_p=np.full(80,.1)+.002*np.sin(np.arange(80)/8),
        source_q=np.full(80,.035)+.001*np.cos(np.arange(80)/7)) for c in case['bank']}
    from dataclasses import replace
    fm=replace(case['fm'],v_sigma=np.broadcast_to(np.asarray(case['fm'].v_sigma)[0],(6,3)))
    ctx,record=ac.build_context(train,evidence,tm,em,case['bank'],fits,fm,horizon=6)
    return ctx,record


def test_evidence_is_scored_before_state_update(context):
    ctx,record=context
    log_likelihoods=[]
    for c,row in record['candidate_updates'].items():
        expected=multivariate_normal.logpdf(row['evidence_observed_whitened'],
            mean=row['evidence_predicted_mean'],cov=row['evidence_covariance'])
        assert row['evidence_log_predictive']==pytest.approx(expected,abs=1e-7)
        log_likelihoods.append(expected)
    from scipy.special import logsumexp
    expected=np.array(log_likelihoods)-logsumexp(log_likelihoods)
    np.testing.assert_allclose(ctx['log_prior'],expected,atol=1e-7)
    assert record['training_used_for_evidence'] is False


def test_conditioned_covariance_and_ac_physics(context):
    ctx,record=context
    for row in record['candidate_updates'].values():
        assert np.linalg.eigvalsh(row['posterior_latent_covariance']).min()>-1e-14
        assert np.max(np.diag(row['posterior_latent_covariance'])-np.diag(row['prior_latent_covariance']))<1e-12
    evaluated=ac.forecast_bank(ctx,np.zeros((6,3)),with_jacobian=True)
    assert evaluated['bank']['means'].shape==(1,4,66)
    assert evaluated['jacobian'].shape==(1,4,66,18)
    assert all(p['ac_equation_residual']<2e-10 for p in evaluated['physical'])
    assert any(np.linalg.norm(r['latent_cov']-np.diag(np.diag(r['latent_cov'])))>0 for r in ctx['rows'])


def test_ev_once_and_mean_directional_derivative(context):
    ctx,_=context
    u=np.zeros((6,3));u[:,0]=2
    central=ac.forecast_bank(ctx,u,with_jacobian=True)
    direction=np.zeros_like(u);direction[:,1]=np.linspace(-.5,.5,6)
    h=1e-3
    plus=ac.forecast_bank(ctx,u+h*direction);minus=ac.forecast_bank(ctx,u-h*direction)
    finite=(plus['bank']['means']-minus['bank']['means'])/(2*h)
    predicted=np.einsum('lkdp,p->lkd',central['jacobian'],direction.ravel())
    np.testing.assert_allclose(finite,predicted,atol=2e-6,rtol=1e-6)
    for i,row in enumerate(ctx['rows']):
        np.testing.assert_allclose(central['physical'][i]['latent_inputs_with_EV'][:,:3]-row['latent_mean'][:,:3],u/ctx['base_kw'])
        np.testing.assert_array_equal(central['physical'][i]['latent_inputs_with_EV'][:,-2:],row['latent_mean'][:,-2:])


def test_joint_density_equals_pq_times_conditional_density(context):
    ctx,_=context;bank=ac.forecast_bank(ctx,np.zeros((6,3)))['bank'];rng=np.random.default_rng(12)
    for mu,S in zip(bank['means'][0],bank['covariances'][0]):
        y=rng.multivariate_normal(mu,S);d=36
        factor=cho_factor(S[:d,:d],lower=True)
        cm=mu[d:]+S[d:,:d]@cho_solve(factor,y[:d]-mu[:d])
        cs=S[d:,d:]-S[d:,:d]@cho_solve(factor,S[:d,d:])
        joint=multivariate_normal.logpdf(y,mean=mu,cov=S)
        factored=multivariate_normal.logpdf(y[:d],mean=mu[:d],cov=S[:d,:d])+multivariate_normal.logpdf(y[d:],mean=cm,cov=cs)
        assert joint==pytest.approx(factored,abs=1e-7)


def test_training_meter_noise_subtracted_from_process():
    rng=np.random.default_rng(100);sigma=.002;values=.1+rng.normal(0,sigma,(2000,1))
    _,C,record=ac.process_forecast(values,8,[sigma])
    row=record['channels'][0]
    assert row['subtracted_innovation_meter_variance']>=sigma**2
    assert row['innovation_variance']<.2*sigma**2
    assert np.linalg.eigvalsh(C).min()>0


def test_constant_plan_and_nested_complete_path():
    domains=nested.build_nested_domains(public.service_domain(6,3,mobile=True).config)
    u=np.zeros((6,3));u[:,0]=2.
    x=complete_curve(domains['fixed_charge'],u)
    import json
    assert json.loads(json.dumps(serial_record(domains['fixed_charge'].decode(x))))['u_kw']==u.tolist()
    assert domains['fixed_charge'].check(x)['feasible']
    for child in ('fixed_bidir','mobile_charge','mobile_bidir'):
        lifted=nested.lift_operation(domains['fixed_charge'],domains[child],x)
        assert domains[child].check(lifted)['feasible']
        np.testing.assert_allclose(domains[child].decode(lifted)['u_kw'],u)


def test_process_forecast_matches_proper_gaussian_kernel_conditioning():
    from .ev_mobile_v4_cases import harmonic_features
    rng=np.random.default_rng(55);H=40;F=4;noise=.002
    observed=.1+.01*np.sin(np.arange(H)/6)+rng.normal(0,noise,H)
    mean,C,record=ac.process_forecast(observed[:,None],F,[noise]);r=record['channels'][0]
    features=harmonic_features(np.arange(H+F));times=np.arange(H+F)
    kernel=r['coefficient_variance']*features@features.T+r['innovation_variance']/(1-r['phi']**2)*r['phi']**abs(times[:,None]-times[None,:])
    Ktt=kernel[:H,:H]+noise**2*np.eye(H);cross=kernel[H:,:H]
    expected_mean=cross@np.linalg.solve(Ktt,observed)
    expected_cov=kernel[H:,H:]-cross@np.linalg.solve(Ktt,cross.T)
    np.testing.assert_allclose(mean[:,0],expected_mean,atol=1e-9)
    np.testing.assert_allclose(C,expected_cov,atol=1e-10)
    assert r['conditional_AR_state_variance_retained']
