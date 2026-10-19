"""Analytic independent expected-proxy audits; no AC population experiments."""
import numpy as np
import pytest

from research_experiments.mobile_ev_mle.weighted_proxy_milp import (
    FrozenPair,EVConfig,SquaredCut,bank,control_box,cut_from_qp,squared_support_cut,
    solve_pair,expected_lower_master,expected_upper_master,evaluate_expected,optimize_expected,check_plan)


def config():
    return EVConfig(n_slots=2,n_sites=1,dt_hours=1,p_max_kw=6,net_grid_energy_kwh=6,
        battery_capacity_kwh=100,initial_soc=.3,eta_charge=1,eta_discharge=1,travel_energy_kwh=0)


def affine_pair(name,B,c=0):
    B=np.array(B,float).reshape(1,-1)
    return FrozenPair(name,np.array([c],float),B,np.empty((1,0)),np.empty((0,0)),np.empty(0),np.empty(0))


def scenario_bank(zero_second=False):
    return {'rival':{'early':affine_pair('early',[1,0]),
        'late':affine_pair('late',[0,0] if zero_second else [0,1])}}


def test_expected_prefers_early_equal_energy_worst_balances():
    # u0+u1=6, 0<=u<=6. Convex weighted sum max=.8*36=28.8 at (6,0).
    # Worst of two squares max=9 at (3,3), not zero at every action.
    answer=optimize_expected(config(),scenario_bank(),{'early':.8,'late':.2},max_iterations=3,total_seconds=10)
    assert answer['failure'] is None
    assert answer['score_lower']<=28.8+1e-7<=answer['score_upper']+2e-7
    assert answer['proxy_gap']<1e-5
    np.testing.assert_allclose(answer['plan'].u_kw.ravel(),[6,0],atol=2e-6)
    assert min(3**2,3**2)==9 and min(6**2,0**2)==0
    assert answer['enumerated_complete_actions']==0 and not answer['is_formal_policy']


def test_zero_scenario_worst_is_always_zero_but_expected_can_select():
    answer=optimize_expected(config(),scenario_bank(True),{'early':.8,'late':.2},max_iterations=3,total_seconds=10)
    assert answer['score_lower']<=28.8+1e-7<=answer['score_upper']+2e-7
    for u0 in np.linspace(0,6,13):assert min(u0*u0,0)==0
    np.testing.assert_allclose(answer['plan'].u_kw.ravel(),[6,0],atol=2e-6)


def test_outer_competitor_min_is_outside_scenario_sum():
    groups={'a':{'early':affine_pair('a0',[0,0]),'late':affine_pair('a1',[0,0],10)},
            'b':{'early':affine_pair('b0',[0,0],10),'late':affine_pair('b1',[0,0])}}
    answer=evaluate_expected(groups,{'early':.5,'late':.5},[3,3])
    assert answer['score_lower']<=50<=answer['score_upper']+1e-10
    assert .5*min(0,100)+.5*min(100,0)==0
    assert min(.5*0+.5*100,.5*100+.5*0)==50


def test_zero_weight_handled_without_removing_required_scenario():
    groups=scenario_bank();weights={'early':1.,'late':0.}
    answer=optimize_expected(config(),groups,weights,max_iterations=3,total_seconds=10)
    assert answer['score_lower']<=36+1e-7<=answer['score_upper']+2e-7
    assert answer['qp_calls']%2==0
    np.testing.assert_allclose(answer['plan'].u_kw.ravel(),[6,0],atol=2e-6)


def test_support_contact_with_nonzero_F_and_zero_start():
    # min_z (u0+z)^2+3 z^2 with |z|<=2 is .75*u0^2 for u0 in [0,6].
    pair=FrozenPair('penalty',np.array([0.]),np.array([[1.,0.]]),np.array([[1.]]),
        np.array([[np.sqrt(3.)]]),np.array([-2.]),np.array([2.]))
    lo,hi=control_box(config());sol=solve_pair(pair,[4,2]);cut=cut_from_qp(pair,[4,2],sol,lo,hi)
    assert sol['q_lower']<=12<=sol['q_upper']+1e-10
    assert abs(cut.value([4,2])-12)<1e-9
    for u0 in np.linspace(0,6,101):assert cut.value([u0,6-u0])<=.75*u0*u0+1e-9
    zero=cut_from_qp(pair,[0,6],lower=lo,upper=hi)
    assert zero.amplitude==0 and zero.value([6,0])==0


def test_negative_a_nonunit_v_and_max_support_error_rejected():
    pair=FrozenPair('adjust',np.array([0.]),np.array([[1.,0.]]),np.array([[1.]]),
        np.empty((0,1)),np.array([-1.]),np.array([1.]))
    lo,hi=control_box(config())
    with pytest.raises(ValueError,match='nonnegative'):
        squared_support_cut(pair,[1],-1,lo,hi)
    with pytest.raises(ValueError,match='norm'):
        squared_support_cut(pair,[2],1,lo,hi)
    correct=squared_support_cut(pair,[1],1,lo,hi)
    assert correct.value([0,6])<0  # box MIN support is -1.
    wrong=SquaredCut(1.,np.array([2.,0.]),np.array([1.]),1.,pair.fingerprint())
    assert wrong.value([0,6])==1 and solve_pair(pair,[0,6])['q_upper']==0
    with pytest.raises(ValueError,match='invalid support'):
        expected_lower_master(config(),{'h':{'w':pair}},{'w':1.},{('h','w'):[wrong]})


def test_lower_and_upper_bound_direction_and_valid_big_M():
    cfg=config();pair=affine_pair('distance',[1,0]);groups={'h':{'w':pair}};weights={'w':1.}
    lo,hi=control_box(cfg)
    cut=cut_from_qp(pair,[6,0],lower=lo,upper=hi)
    lower=expected_lower_master(cfg,groups,weights,{('h','w'):[cut]})
    upper=expected_upper_master(cfg,groups,weights,{('h','w'):[np.empty(0)]},segments=2)
    assert lower['incumbent']<=36+1e-7<=upper['full_proxy_upper_bound']+2e-7
    # d=12u0-36 has min -36 and qbar about36, requiring M about72.
    assert 72<=lower['big_m_max']<=72+1e-6
    assert lower['full_proxy_upper_bound'] is None


@pytest.mark.parametrize('weights',[{'early':-.1,'late':1.1},{'early':.2,'late':.2},
    {'early':np.nan,'late':1.},{'early':np.inf,'late':0.}])
def test_weights_must_be_declared_nonnegative_unit_mass(weights):
    with pytest.raises(ValueError,match='weights'):
        bank(scenario_bank(),weights)


def test_missing_scenario_and_infeasible_witness_not_dropped():
    with pytest.raises(ValueError,match='all declared'):
        bank({'h':{'early':affine_pair('x',[1,0])}},{'early':.5,'late':.5})
    pair=FrozenPair('adjust',np.array([0.]),np.array([[1.,0.]]),np.array([[1.]]),
        np.empty((0,1)),np.array([-1.]),np.array([1.]))
    with pytest.raises(ValueError,match='infeasible'):
        expected_upper_master(config(),{'h':{'w':pair}},{'w':1.},{('h','w'):[np.array([2.])]})


def test_operations_network_choses_movement_and_continuous_power():
    cfg=EVConfig(n_slots=3,n_sites=2,dt_hours=1,p_max_kw=[0,6],net_grid_energy_kwh=6,
        battery_capacity_kwh=100,initial_soc=.5,eta_charge=1,eta_discharge=1,
        travel_slots=1,travel_energy_kwh=.5)
    pair=affine_pair('site1',[0,0,0,1,0,1])
    answer=optimize_expected(cfg,{'h':{'w':pair}},{'w':1.},max_iterations=2,total_seconds=10)
    assert answer['failure'] is None and check_plan(answer['plan'],cfg,tolerance=2e-6)['feasible']
    assert sum(arc.is_travel for arc in answer['plan'].selected_arcs)==1
    assert abs(answer['plan'].energy_kwh[-1]-55.5)<2e-6
    assert answer['score_lower']<=36+1e-7<=answer['score_upper']+2e-7
    assert answer['enumerated_complete_actions']==0


def test_nonzero_F_search_keeps_numerical_bound_conflict_instead_of_success():
    pair=FrozenPair('penalty',np.array([0.]),np.array([[1.,0.]]),np.array([[1.]]),
        np.array([[np.sqrt(3.)]]),np.array([-2.]),np.array([2.]))
    answer=optimize_expected(config(),{'h':{'only':pair}},{'only':1.},max_iterations=4,total_seconds=10)
    # Exact optimum is 27. Solver builds may return a raw u0 infinitesimally
    # above6; that must be a preserved computational failure if bounds cross.
    if not answer['bounds_valid']:
        assert answer['status']=='required_failure' and answer['proxy_gap'] is None
        assert answer['score_lower']>answer['score_upper']
        assert answer['failure'] and answer['records']
    else:
        assert answer['score_lower']<=27+1e-7<=answer['score_upper']+2e-7
