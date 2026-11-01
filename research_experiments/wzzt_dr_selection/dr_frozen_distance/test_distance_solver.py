"""Hand-computable frozen distance price tests; no case data or future read."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from distance_solver import solve_distance,evaluate_distance
from research_experiments.wzzt_dr_selection.dr_conic_pricing.model import PriceProblem


def problem(j=10,*,budget=1.,groups=(),device=None,rhs=None,cap=1.,factors=None):
    return PriceProblem(mu0=np.zeros(1),background_factors=np.zeros((1,1)),
        profiles=np.zeros((1,j)),response_factors=np.zeros((j,1)) if factors is None else factors,
        elasticities=np.ones(j),reduction_hours=np.ones(j),price_upper=np.full(j,cap),
        actual_amplitude_upper=np.ones(j),benefit_directions=np.ones((1,1)),
        benefit_offsets=[0.],risk_allocations=[.1],budget=budget,
        exclusive_groups=groups,device_matrix=device,device_rhs=rhs)


def scenario(base,columns,j=10,normalizer=1.):
    coefficient=np.zeros((len(base),j));coefficient[:,:np.asarray(columns).shape[1]]=columns
    return dict(base=np.asarray(base,float),price_coeff=coefficient,normalizer=normalizer)


def test_exact_convex_absolute_crossing_can_create_positive_gain():
    scenes=[scenario([-.4],[[1.]])]
    r=solve_distance(problem(),scenes,seconds=10)
    assert r['feasible_incumbent'] and r['conditional_finite_scenario_optimality']
    assert r['prices'][0]==pytest.approx(1.,abs=2.e-6)
    assert r['scenario_gain_native'][0]==pytest.approx(.2,abs=2.e-6)
    assert r['compilation']['exact_abs_coordinates']==1


def test_energy_neutral_constant_sign_prediction_gain_is_zero():
    # h=(+1,-1), base positive and both affine values remain nonnegative.
    scenes=[scenario([1.,1.],[[1.],[-1.]],normalizer=2.)]
    r=solve_distance(problem(),scenes,seconds=10)
    assert r['quantile_micro']==pytest.approx(0.,abs=1.e-5)
    assert r['expected_cost']==pytest.approx(0.,abs=1.e-7)
    assert not r['sample_requirement_met']


def test_ninety_percent_quantile_keeps_one_failed_scene():
    scenes=[scenario([0.],[[1.]]) for _ in range(9)]+[scenario([1.],[[-1.]])]
    r=solve_distance(problem(),scenes,seconds=10)
    assert r['required_scenarios']==9 and r['scenario_count']==10
    assert r['positive_pass_count']==9 and r['sample_requirement_met']
    assert r['scenario_gain_native'][-1]==pytest.approx(-1.,abs=2.e-6)
    assert r['quantile_micro']==pytest.approx(1.e6,abs=2.)


def test_original_budget_and_price_cap_apply_to_continuous_quotes():
    scenes=[scenario([0.],[[1.]])]
    r=solve_distance(problem(budget=.25,cap=.8),scenes,seconds=10)
    assert r['prices'][0]==pytest.approx(.5,abs=2.e-6)
    assert r['hard_budget_bound']<=.25+1.e-7
    assert r['original_evaluation']['numerical_feasible']


def test_original_exclusive_module_types_cannot_both_execute():
    scenes=[scenario([0.],[[1.,2.]])]
    r=solve_distance(problem(budget=2.,groups=[(0,1)]),scenes,seconds=10)
    assert r['prices'][:2]==pytest.approx([0.,1.],abs=2.e-6)
    assert r['original_evaluation']['exclusive_group_positive_counts']==[1]


def test_exact_shared_factor_device_constraint_allows_cancellation():
    row=np.zeros((1,10));row[0,:2]=1.
    d=np.zeros((10,1));d[0,0]=.2;d[1,0]=-.2
    p=problem(budget=2.,device=row,rhs=[.6],factors=d)
    scenes=[scenario([0.],[[1.,1.]])]
    r=solve_distance(p,scenes,seconds=10)
    assert r['prices'][:2]==pytest.approx([.3,.3],abs=2.e-6)
    assert r['maximum_violation']<=1.e-7


def test_original_robust_actual_amplitude_cap_is_retained():
    d=np.zeros((10,1));d[0,0]=.2
    r=solve_distance(problem(budget=2.,factors=d),[scenario([0.],[[1.]])],seconds=10)
    assert r['amplitudes'][0]==pytest.approx(1/1.2,abs=2.e-6)
    assert r['original_evaluation']['actual_amplitude_upper'][0]<=1.+1.e-7


def test_arithmetic_uses_the_same_baseline_and_full_normalizer():
    scenes=[scenario([2.,-.4],[[0.],[1.]],normalizer=4.)]
    r=evaluate_distance(scenes,np.r_[1.,np.zeros(9)])
    assert r['scenario_base_distance_native'][0]==pytest.approx(.6)
    assert r['scenario_prediction_distance_native'][0]==pytest.approx(.65)
    assert r['scenario_gain_native'][0]==pytest.approx(.05)
    assert evaluate_distance(scenes,np.zeros(10))['scenario_gain_native']==[0.]


def test_zero_budget_has_a_complete_feasible_zero_price_start():
    r=solve_distance(problem(budget=0.),[scenario([.4],[[1.]])],seconds=10)
    assert r['complete_zero_start_supplied']
    assert r['prices']==pytest.approx(np.zeros(10),abs=1.e-8)
    assert r['quantile_micro']==pytest.approx(0.,abs=1.e-5)
    assert r['expected_cost']==0.


def test_verified_nonzero_start_and_zero_fallback_are_both_complete():
    candidate=np.zeros(10);candidate[0]=1.
    scenes=[scenario([-.4],[[1.]])]
    r=solve_distance(problem(),scenes,seconds=10,warm_start_prices=[candidate])
    assert r['complete_verified_nonzero_start_supplied']
    assert r['complete_zero_start_supplied'] and r['mip_start_count']==2
    assert r['provided_warm_start_count']==r['verified_warm_start_count']==1
    assert r['warm_start_best_quantile_micro']==pytest.approx(.2e6)
    assert r['quantile_micro']>=r['warm_start_best_quantile_micro']-2.
    assert r['original_evaluation']['numerical_feasible']


def test_invalid_type_warm_start_is_retained_as_rejection_not_domain_change():
    invalid=np.zeros(10);invalid[:2]=.5
    scenes=[scenario([0.],[[1.,2.]])]
    r=solve_distance(problem(budget=2.,groups=[(0,1)]),scenes,seconds=10,
                     warm_start_prices=[invalid])
    assert r['provided_warm_start_count']==1 and r['verified_warm_start_count']==0
    assert len(r['warm_start_rejections'])==1
    assert r['prices'][:2]==pytest.approx([0.,1.],abs=2.e-6)
