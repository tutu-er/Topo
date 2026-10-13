"""Independent checks of single-site EV operations and affine MILP bounds."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from research_experiments.mobile_ev_mle import single_site_milp as module
from research_experiments.mobile_ev_mle.single_site_milp import (
    SingleSiteConstraints, SingleSitePlan, check_single_site_plan,
    constant_site_plan, solve_site_step,
)


def constraints(**changes):
    return replace(SingleSiteConstraints(),**changes)


def test_linear_price_selects_power_continuously_without_profile_library():
    c=constraints(energy=.075)
    result=solve_site_step(0,2,1,c,intercepts=[3.],gradients=[[2.,1.]])
    assert result['success'] and result['status']=='optimal'
    np.testing.assert_allclose(result['plan'].p[:,0],[.3,0],atol=1e-8)
    assert result['incumbent']==pytest.approx(3.6)
    assert result['best_bound']>=result['incumbent']-1e-8
    assert abs(result['absolute_gap'])<1e-8
    assert not result['binary_modes']
    assert result['validation']['feasible']


def test_max_min_two_prices_has_known_interior_optimum():
    c=constraints(energy=.075)
    result=solve_site_step(0,2,1,c,intercepts=[0.,0.],gradients=[[2.,1.],[1.,2.]])
    assert result['success']
    np.testing.assert_allclose(result['plan'].p[:,0],[.15,.15],atol=1e-8)
    assert result['incumbent']==pytest.approx(.45)
    assert result['best_bound']==pytest.approx(.45)


def test_intercept_and_base_offset_use_same_affine_expression():
    c=constraints(energy=.075)
    base=np.array([.1,.2])
    result=solve_site_step(0,2,1,c,base_power=base,intercepts=[-2.],gradients=[[1.,0.]])
    assert result['success']
    assert result['incumbent']==pytest.approx(-1.8)
    assert result['best_bound']==pytest.approx(-1.8)
    assert result['relative_gap']==pytest.approx(0.)


def test_trust_region_is_a_constraint_on_net_power():
    c=constraints(energy=.075)
    result=solve_site_step(0,2,1,c,base_power=[.15,.15],trust_radius=.05,
                           intercepts=[0.],gradients=[[1.,0.]])
    assert result['success']
    np.testing.assert_allclose(result['plan'].p[:,0],[.2,.1],atol=1e-8)
    assert result['trust_ok']


def test_zero_radius_returns_base_and_zero_price_preserves_task():
    c=constraints(energy=.075)
    result=solve_site_step(0,2,1,c,base_power=[.12,.18],trust_radius=0,
                          intercepts=[2.],gradients=[[0.,0.]])
    assert result['success']
    np.testing.assert_allclose(result['plan'].p[:,0],[.12,.18],atol=1e-8)
    assert result['incumbent']==pytest.approx(2.)


def test_travel_access_soc_and_grid_energy_recomputed():
    access=np.ones((4,2),bool);access[2,1]=False
    c=constraints(energy=.075,availability=access,q_over_p=.2)
    result=solve_site_step(1,4,2,c,intercepts=[0.],gradients=[[50.,1.,10.,2.]])
    assert result['success']
    plan=result['plan']
    np.testing.assert_allclose(plan.p[:,0],0,atol=1e-8)
    np.testing.assert_allclose(plan.p[:,1],[0,0,0,.3],atol=1e-8)
    np.testing.assert_array_equal(plan.sites,[-1,1,1,1])
    np.testing.assert_allclose(plan.q,.2*plan.p,atol=1e-8)
    assert plan.soc[0]==pytest.approx(.4-.01/3)
    assert plan.soc[-1]==pytest.approx(.4-.01/3+.95*.075/3)
    assert result['validation']['energy_pu_hours']==pytest.approx(.075)
    assert result['validation']['travel_energy_pu_hours']==pytest.approx(.01)


def test_uniform_baseline_uses_identical_task_and_arrival():
    c=constraints(energy=.15)
    constant=constant_site_plan(1,4,2,c)
    optimized=solve_site_step(1,4,2,c,intercepts=[0.],gradients=[[0,1,2,3]])
    assert constant['success'] and optimized['success']
    np.testing.assert_allclose(constant['plan'].p[:,1],[0,.2,.2,.2])
    assert constant['validation']['energy_pu_hours']==optimized['validation']['energy_pu_hours']
    assert constant['plan'].soc[0]==optimized['plan'].soc[0]


def test_uniform_baseline_respects_missing_access_slot():
    access=np.ones((4,2),bool);access[2,1]=False
    c=constraints(energy=.15,availability=access)
    result=constant_site_plan(1,4,2,c)
    assert result['success']
    np.testing.assert_allclose(result['plan'].p[:,1],[0,.3,0,.3])


def test_variable_power_can_be_feasible_when_uniform_power_is_not():
    cap=np.array([[.1],[.3]])
    c=constraints(energy=.075,p_charge_max=cap)
    baseline=constant_site_plan(0,2,1,c)
    result=solve_site_step(0,2,1,c,intercepts=[0.],gradients=[[1.,0.]])
    assert not baseline['success'] and baseline['status']=='constant_infeasible'
    assert result['success']
    np.testing.assert_allclose(result['plan'].p[:,0],[.1,.2],atol=1e-8)


def test_discharge_has_binary_modes_and_physical_efficiency():
    c=constraints(energy=0.,p_discharge_max=.3)
    result=solve_site_step(0,2,1,c,intercepts=[0.],gradients=[[1.,-1.]])
    assert result['success'] and result['binary_modes']
    np.testing.assert_allclose(result['plan'].p[:,0],[.3,-.3],atol=1e-8)
    expected=.4+.25/3*(.95*.3-.3/.95)
    assert result['plan'].soc[-1]==pytest.approx(expected)
    assert result['incumbent']==pytest.approx(.6)


def test_negative_energy_uniform_discharge():
    c=constraints(energy=-.075,p_discharge_max=.3)
    result=constant_site_plan(0,2,1,c)
    assert result['success']
    np.testing.assert_allclose(result['plan'].p[:,0],[-.15,-.15])
    assert result['plan'].soc[-1]==pytest.approx(.4-.075/(.95*3))


def test_discharge_cannot_hide_simultaneous_flow_in_soc():
    c=constraints(energy=0.,p_discharge_max=.3)
    plan=SingleSitePlan(0,np.zeros((2,1)),np.zeros((2,1)),np.array([0,0]),np.array([.4,.39,.38]))
    checked=check_single_site_plan(plan,c)
    assert not checked['feasible'] and 'soc_dynamics' in checked['issues']


def test_driving_soc_checked_before_any_charging_recovery():
    c=constraints(initial_soc=.101,soc_min=.1)
    for result in (constant_site_plan(1,4,2,c),solve_site_step(1,4,2,c)):
        assert not result['success']
        assert result['status']=='operational_infeasible'
        assert 'driving' in result['termination']


def test_arrival_outside_window_is_retained_failure():
    c=constraints(travel_slots=np.array([[0,3],[3,0]]))
    result=solve_site_step(1,3,2,c)
    assert not result['success'] and result['status']=='operational_infeasible'
    assert result['plan'] is None


def test_grid_task_infeasible_and_soc_saturation_retained():
    result=solve_site_step(0,2,1,constraints(energy=.16))
    assert not result['success'] and result['status']=='infeasible'
    result=solve_site_step(0,4,1,constraints(initial_soc=.89,soc_max=.9,energy=.15))
    assert not result['success'] and result['status']=='infeasible'


def test_no_access_and_zero_energy_behaviors():
    access=np.zeros((2,1),bool)
    bad=constant_site_plan(0,2,1,constraints(energy=.01,availability=access))
    assert not bad['success']
    zero=constant_site_plan(0,2,1,constraints(energy=0,availability=access))
    assert zero['success']
    np.testing.assert_allclose(zero['plan'].p,0)


def test_independent_validator_catches_wrong_site_and_soc():
    c=constraints(energy=.075)
    good=constant_site_plan(0,2,2,c)['plan']
    good.p[0,1]=.01
    check=check_single_site_plan(good,c)
    assert not check['feasible'] and 'multiple_sites' in check['issues']
    good=constant_site_plan(0,2,2,c)['plan'];good.soc[-1]+=.01
    check=check_single_site_plan(good,c)
    assert not check['feasible'] and 'soc_dynamics' in check['issues']


def test_two_slot_discrete_audit_only_checks_optimizer_result():
    # This four-value audit is independent of the continuous MILP selector.
    c=constraints(energy=.075)
    gs=np.array([[2.,1.],[1.,2.]])
    result=solve_site_step(0,2,1,c,intercepts=[0.,0.],gradients=gs)
    grid=np.array([0.,.1,.2,.3])
    discrete=max(min(gs@np.array([x,.3-x])) for x in grid)
    assert discrete==pytest.approx(.4)
    assert result['incumbent']==pytest.approx(.45)
    assert result['incumbent']>=discrete


def test_solver_failure_is_recorded(monkeypatch):
    def fail(*a,**k):raise RuntimeError('declared solver defect')
    monkeypatch.setattr(module,'milp',fail)
    result=solve_site_step(0,2,1,constraints(energy=.075))
    assert not result['success'] and result['status']=='solver_failure'
    assert 'declared solver defect' in result['termination']


def test_feasible_time_limit_incumbent_keeps_bound_and_termination(monkeypatch):
    original=module.milp
    def limited(*a,**k):
        result=original(*a,**k);result.status=1;result.success=False
        result.message='Time limit reached';result.mip_dual_bound=-.9
        return result
    monkeypatch.setattr(module,'milp',limited)
    result=solve_site_step(0,2,1,constraints(energy=.075),intercepts=[0],gradients=[[1,0]])
    assert result['success'] and result['status']=='feasible_incumbent'
    assert not result['solver_success']
    assert result['incumbent']==pytest.approx(.3)
    assert result['best_bound']==pytest.approx(.9)
    assert result['absolute_gap']==pytest.approx(.6)
    assert result['termination']=='Time limit reached'


def test_invalid_solver_incumbent_is_preserved_not_executed(monkeypatch):
    def invalid(*a,**k):
        return SimpleNamespace(x=np.array([.3,.3,0,0,.4,.4,.4,0]),fun=0,
                               status=1,success=False,message='synthetic invalid point')
    monkeypatch.setattr(module,'milp',invalid)
    result=solve_site_step(0,2,1,constraints(energy=.075))
    assert not result['success'] and result['status']=='invalid_incumbent'
    assert result['plan'] is not None
    assert not result['validation']['feasible']


def test_gap_conflict_retains_raw_negative_gap(monkeypatch):
    original=module.milp
    def invalid_bound(*a,**k):
        result=original(*a,**k);result.mip_dual_bound=0.;return result
    monkeypatch.setattr(module,'milp',invalid_bound)
    result=solve_site_step(0,2,1,constraints(energy=.075),intercepts=[0],gradients=[[1,0]])
    assert not result['success']
    assert result['absolute_gap']==pytest.approx(-.3)
    assert not result['bound_consistent']


@pytest.mark.parametrize('changes,match',[
    ({'eta_charge':0},'efficiency'),({'p_charge_max':-.1},'nonnegative'),
    ({'initial_soc':1.1},'SOC'),({'travel_slots':[[1]]},'zero diagonal'),
    ({'energy':-.1},'discharge'),({'availability':[[2],[1]]},'availability'),
])
def test_invalid_public_constraints_raise(changes,match):
    with pytest.raises(ValueError,match=match):solve_site_step(0,2,1,constraints(**changes))


@pytest.mark.parametrize('changes,match',[
    ({'base_power':[0]},'base_power'),({'trust_radius':-.1},'trust_radius'),
    ({'gradients':[[0]]},'gradients'),({'intercepts':[float('nan')]},'finite'),
    ({'time_limit':0},'time limit'),
])
def test_invalid_step_inputs_raise(changes,match):
    with pytest.raises(ValueError,match=match):solve_site_step(0,2,1,constraints(energy=.075),**changes)


def test_summary_is_json_serializable():
    import json
    plan=constant_site_plan(0,2,1,constraints(energy=.075))['plan']
    value=json.loads(json.dumps(plan.summary()))
    assert value['site']==0 and len(value['soc'])==3


def test_time_limit_slack_eta_is_valid_and_score_is_recomputed(monkeypatch):
    original=module.milp
    def slack(*a,**k):
        result=original(*a,**k);result.status=1;result.success=False
        result.message='Time limit reached with slack eta'
        result.x[-1]-=.01;result.fun+=.01;result.mip_dual_bound=-.9
        return result
    monkeypatch.setattr(module,'milp',slack)
    result=solve_site_step(0,2,1,constraints(energy=.075),intercepts=[0],gradients=[[1,0]])
    assert result['success']
    assert result['raw_incumbent']==pytest.approx(.29)
    assert result['incumbent']==pytest.approx(.3)
    assert result['best_bound']==pytest.approx(.9)
    assert result['solve_seconds']>=0
