import numpy as np
import pytest
from .day_model import canonical_prices,payment,resource_audit,margin_values,scenario_summary


def small_problem():
    return dict(slots=2,n=1,elasticities=np.array([.5]),price_caps=np.array([2.]),
        factors=np.array([[0.,0.]]),contrast=np.array([[10.]]),
        history_component_micro=-2.,day_weight=1.,new_tolerance=.02,epsilon_micro=0.,
        power_lower=-np.ones((1,2)),power_upper=np.ones((1,2)),
        energy_lower=-2*np.ones((1,2)),energy_upper=2*np.ones((1,2)),
        available=np.ones((1,2),bool),budget=.2)


def small_scenarios():
    return [dict(z=[0.,0.],base_bad_micro=np.array([[-1.],[1.]]),
        reference_mae_micro=0.,margin_zero_micro=-1.,target_micro=.2001)]


def test_quote_scaling_is_not_free():
    u=np.array([[-.2,.2]])
    assert canonical_prices(u,[.5])==pytest.approx([.4])
    assert payment(u,[.4])==pytest.approx(.02)
    assert not resource_audit(small_problem(),u,[.01])['feasible']


def test_margin_and_remaining_gap():
    p=small_problem();s=small_scenarios();u=np.array([[-.2,.2]])
    a=margin_values(p,s,u)[0]
    assert a['margin_micro']==pytest.approx(1.)
    assert a['increment_micro']==pytest.approx(2.)
    assert a['passed']
    assert scenario_summary(p,s,np.zeros((1,2)))['feasibility_slack_micro']==pytest.approx(1.2001)


def test_energy_and_availability():
    p=small_problem()
    assert not resource_audit(p,np.array([[0.,.1]]))['feasible']
    p['available'][0,0]=False
    assert not resource_audit(p,np.array([[-.1,.1]]))['feasible']


def test_budget_uses_real_reduction_energy():
    p=small_problem();p['budget']=.001
    assert not resource_audit(p,np.array([[-.1,.1]]))['feasible']


def test_no_payment_cap_still_checks_physical_limits():
    p=small_problem();p['budget']=None
    good=resource_audit(p,np.array([[-.5,.5]]))
    assert good['feasible'] and not good['payment_cap_enforced']
    assert not resource_audit(p,np.array([[-2.,2.]]))['feasible']


def test_minimum_cost_solver_does_not_keep_the_old_budget():
    from .day_solver import solve
    p=small_problem();p['budget']=None
    s=small_scenarios();s[0]['target_micro']=5.
    # Need a=.6, c=1.2, shifted energy=.15: fee=.18, even
    # though the former optional cap can be chosen below that value.
    r=solve(p,s,seconds=5.,cost_seconds=5.)
    assert r['status']=='target_feasible_cost_incumbent'
    assert r['resource_audit']['expected_payment']==pytest.approx(.18,abs=2.e-5)
    p['budget']=.01
    limited=solve(p,s,seconds=5.,cost_seconds=5.)
    assert limited['status']=='sample_target_infeasible'


def test_timewise_independent_multiplier_breaks_daily_energy():
    u=np.array([-1.,1.])
    assert u.sum()==0
    assert not np.isclose((u*np.array([.8,1.2])).sum(),0.)


def test_actual_cost_is_nonconvex_in_daily_curve():
    a=np.array([[.21,.07,-.21,-.07]])
    b=np.array([[.19,.13,-.19,-.13]])
    c=(a+b)/2
    assert payment(c,canonical_prices(c,[.5])) > (payment(a,canonical_prices(a,[.5]))+payment(b,canonical_prices(b,[.5])))/2


def test_zero_cost_already_sufficient_is_retained():
    from .day_solver import solve
    p=small_problem();p['history_component_micro']=1.
    s=small_scenarios();s[0]['margin_zero_micro']=2.
    s[0]['target_micro']=.0001
    r=solve(p,s,seconds=5.,cost_seconds=5.)
    assert r['status']=='zero_cost_optimal'
    assert r['resource_audit']['expected_payment']==0.


def test_solver_has_exact_minimum_in_two_slot_example():
    from .day_solver import solve
    p=small_problem();s=small_scenarios();a=solve(p,s,seconds=10.,cost_seconds=10.)
    assert a['status']=='target_feasible_cost_incumbent'
    assert a['training']['passed']==1
    # At optimum u=(-a,+a), 10a-1=.2001 and fee=(2a)*(.25a).
    exact=.5*(1.2001/10.)**2
    assert a['resource_audit']['expected_payment']==pytest.approx(exact,abs=2.e-6)
    assert a['secondary']['global_bound']<=a['resource_audit']['expected_payment']+2.e-6


def test_solver_proves_zero_effect_target_unreachable():
    from .day_solver import solve
    p=small_problem();p['contrast'][:]=0.
    a=solve(p,small_scenarios(),seconds=5.,cost_seconds=5.)
    assert a['status']=='sample_target_infeasible'
    assert a['primary']['global_bound']==pytest.approx(1.2001,abs=1.e-5)
