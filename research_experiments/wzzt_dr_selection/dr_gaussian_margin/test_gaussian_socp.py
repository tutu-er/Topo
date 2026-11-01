"""Small mathematical checks, independent of any physical grid case."""
import math
import numpy as np
import pytest
from scipy.stats import norm
from gaussian_socp import prepare, evaluate_coefficients, solve_coefficients, box_bound


def problem(**updates):
    p=dict(G_mean_zero=0., G_gradient=[[1.]], G_noise_base=[0.],
           reference_mean=[1.], reference_noise=[[0.]], reference_sign=[1.],
           alpha=1., beta=1.02, target=1., cv=0., elasticities=.5,
           u_lower=[[-3.]], u_upper=[[3.]], slot_hours=.25)
    p.update(updates)
    return p


def test_sign_counterexample_is_repaired():
    p=problem(G_mean_zero=1.02, reference_mean=[-1.], target=0.)
    v=evaluate_coefficients(p, [[0.]])
    assert v['signed_quantile_bound']==pytest.approx(1.02)
    assert v['flip_penalty']==pytest.approx(2.04)
    assert v['corrected_margin_bound']==pytest.approx(-1.02)


def test_residual_identity_and_lower_bound_all_signs():
    rng=np.random.default_rng(2026112401)
    c=rng.normal(size=(2500,7)); f=rng.normal(size=(2500,7))
    sc=np.where(rng.normal(size=7)>=0,1.,-1.); sf=np.where(rng.normal(size=7)>=0,1.,-1.)
    exact=np.abs(c).sum(1)-1.02*np.abs(f).sum(1)
    signed=c@sc-1.02*(f@sf)
    fc=np.maximum(-c*sc,0).sum(1);ff=np.maximum(-f*sf,0).sum(1)
    assert np.max(np.abs(exact-(signed+2*fc-2.04*ff)))<1e-13
    assert np.min(exact-(signed-2.04*ff))>=-1e-13


def test_correlated_reference_uses_row_norm_and_no_independence_claim():
    p=problem(reference_mean=[0.,0.], reference_noise=[[1.,0.],[-1.,0.]], reference_sign=[1.,1.])
    q=prepare(p)
    assert np.allclose(q['reference_std'],[1.,1.])
    assert q['reference_flip_bounds'][0]==pytest.approx(norm.ppf(.975))


def test_independent_slot_variance_and_positive_zero_price_semantics():
    p=problem(G_gradient=[[2.,-3.]], G_noise_base=[.2,-.1], cv=[[.1,.2]],
              u_lower=[[-2.,-2.]],u_upper=[[2.,2.]])
    q=evaluate_coefficients(p,[[1.,-.5]])
    assert q['G_variance_base']==pytest.approx(.05)
    assert q['G_variance_DR']==pytest.approx(.04+.09)
    assert q['cost']==pytest.approx(.625)
    zero=evaluate_coefficients(p,[[0.,0.]])
    assert zero['G_variance_DR']==0 and zero['cost']==0


def test_standard_normal_quantile_probability():
    p=problem(G_mean_zero=2.,G_noise_base=[.4],cv=.2,target=1.)
    ev=evaluate_coefficients(p,[[.5]])
    analytic=norm.cdf((ev['G_mean']-ev['signed_quantile_bound'])/ev['G_std'])
    assert analytic==pytest.approx(.95)
    rng=np.random.default_rng(2026112402)
    samples=ev['G_mean']+.4*rng.normal(size=100000)+.1*rng.normal(size=100000)
    assert abs(np.mean(samples>=ev['signed_quantile_bound'])-.95)<.003


def test_joint_sign_correction_monte_carlo_original_margin():
    # Same z drives both candidate errors: no independence of the two events.
    p=problem(reference_mean=[.4],reference_noise=[[.2]],G_noise_base=[.3],
              G_mean_zero=1.5,target=0.,cv=.1)
    ev=evaluate_coefficients(p,[[.3]])
    rng=np.random.default_rng(2026112403)
    z=rng.normal(size=150000);w=rng.normal(size=z.size)
    rf=.4+.2*z
    G=1.5+.3+.3*z+.03*w
    rc=G+1.02*rf
    true=np.abs(rc)-1.02*np.abs(rf)
    assert np.mean(true>=ev['corrected_margin_bound'])>=.9-.003


def test_solver_zero_fee_when_zero_is_feasible():
    q=solve_coefficients(problem(G_mean_zero=2.),seconds=10.)
    assert q['status']=='optimal'
    assert q['audit']['cost']<1e-7
    assert q['binary_variables']==0 and q['scenarios']==0
    assert q['original_form_target_feasible']


def test_solver_closed_form_one_control():
    cv=.1; p=problem(cv=cv)
    q=solve_coefficients(p,seconds=10.)
    expected=1/(1-norm.ppf(.95)*cv)
    assert q['status']=='optimal'
    assert q['audit']['mean_u'][0][0]==pytest.approx(expected,rel=2e-5,abs=2e-6)
    assert q['audit']['cost']==pytest.approx(.5*expected**2,rel=3e-5)
    assert q['original_form_target_feasible']


def test_solver_rejects_infeasible_corrected_but_keeps_naive_ablation():
    p=problem(G_mean_zero=1.02,reference_mean=[-1.],target=0.,u_lower=[[0.]],u_upper=[[0.]])
    corrected=solve_coefficients(p,seconds=10.)
    naive=solve_coefficients(p,corrected=False,seconds=10.)
    assert corrected['status']=='infeasible'
    assert naive['status']=='optimal' and naive['audit']['cost']<1e-8
    assert 'no original-margin guarantee' in naive['audit']['scope']


def test_maximum_bound_matches_analytic_box_optimum():
    p=problem(cv=.1)
    q=solve_coefficients(p,mode='max_bound',seconds=10.)
    assert q['status']=='optimal'
    assert q['audit']['corrected_margin_bound']==pytest.approx(3*(1-norm.ppf(.95)*.1),rel=2e-5)


def test_invalid_risk_budget_rejected():
    with pytest.raises(ValueError):prepare(problem(epsilon_G=.08,epsilon_F=.05))


def test_signed_power_does_not_impose_unrequested_daily_zero_energy():
    p=problem(G_gradient=[[1.,2.]],u_lower=[[0.,0.]],u_upper=[[2.,2.]])
    q=solve_coefficients(p,seconds=10.)
    assert q['status']=='optimal'
    assert q['audit']['mean_net_energy_kwh'][0]>.1
    assert not q['exact_energy_equality_enforced']


def test_cheap_bound_uses_physical_box_and_only_excludes_sufficient_condition():
    p=problem(G_mean_zero=1.02,reference_mean=[-1.],u_lower=[[-.1]],u_upper=[[.1]],target=0.)
    b=box_bound(p)
    assert b['sufficient_margin_upper_bound']==pytest.approx(-.92)
    assert b['proves_sufficient_model_infeasible']
    assert not box_bound(p,corrected=False)['proves_sufficient_model_infeasible']
