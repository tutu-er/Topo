"""Small exact price/absolute/minimum/quantile checks; no case or future read."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from affine_solver import solve_affine,evaluate_affine
from research_experiments.wzzt_dr_selection.dr_conic_pricing.model import PriceProblem


def problem(*,budget=1.,groups=(),device=None,rhs=None):
    return PriceProblem(mu0=np.zeros(1),background_factors=np.zeros((1,1)),
        profiles=np.zeros((1,16)),response_factors=np.zeros((16,1)),
        elasticities=np.ones(16),reduction_hours=np.ones(16),price_upper=np.ones(16),
        actual_amplitude_upper=np.ones(16),benefit_directions=np.ones((1,1)),
        benefit_offsets=[0.],risk_allocations=[.1],budget=budget,
        exclusive_groups=groups,device_matrix=device,device_rhs=rhs)


def family(identity,target,base,coef,constant=0.):
    coeff=np.zeros((len(base),8));coeff[:,:np.asarray(coef).shape[1]]=coef
    return dict(family_id=identity,has_target=target,constant_loss=constant,
        residual_base=np.asarray(base,float),residual_price=coeff)


def scene(families,n=1,gain=1.):
    return dict(families=families,validation_size=n,acceptance_margin=gain)


def test_crossing_absolute_is_exact_even_on_negative_side():
    # max |c-.4|, c in[0,1] is .6 at c=1; negative fine absolute must remain .1.
    data=[scene([family(0,False,[-.4],[[1.]]),family(1,True,[-.1],[[0.]])])]
    r=solve_affine(problem(),data,np.zeros(16),seconds=10)
    assert r['feasible_incumbent']
    assert r['prices'][8]==pytest.approx(1.,abs=2.e-6)
    assert r['scenario_margins'][0]==pytest.approx(.6-1.03*.1-1.e-12,abs=2.e-6)
    assert r['compilation']['exact_abs_residuals']==1


def test_ninety_percent_quantile_retains_one_bad_scenario():
    tables=[]
    for constant in [0.]*9+[2.]:
        tables.append(scene([family(0,False,[0.],[[1.]]),family(1,True,[],np.empty((0,1)),constant)]))
    r=solve_affine(problem(),tables,np.zeros(16),seconds=10)
    assert r['required_scenarios']==9
    assert r['quantile_micro']==pytest.approx(1.e6,abs=2.)
    assert r['prices'][8]==pytest.approx(1.,abs=2.e-6)
    assert len(r['scenario_margins'])==10
    assert r['excitation_pass_count']==9


def test_exact_family_minimum_and_negative_quantile():
    # Budget caps c at .6, so coarse min(c,|.6-c|) peaks at c=.3.
    # Fine is .4, so even the optimal exclusion margin remains negative.
    tables=[scene([family(0,False,[0.],[[1.]]),family(2,False,[.6],[[-1.]]),
                   family(1,True,[],np.empty((0,1)),.4)])]
    r=solve_affine(problem(budget=.36),tables,np.zeros(16),seconds=10)
    assert r['prices'][8]==pytest.approx(.3,abs=2.e-6)
    assert r['scenario_margins'][0]==pytest.approx(.3-1.03*.4-1.e-12,abs=2.e-6)
    assert r['quantile_micro']<0 and not r['sample_requirement_met']
    assert r['compilation']['family_min_constraints']==1


def test_price_dependent_fine_absolute_is_exact():
    data=[scene([family(0,False,[],np.empty((0,1)),.5),
                 family(1,True,[-.5],[[1.]])])]
    r=solve_affine(problem(),data,np.zeros(16),seconds=10)
    assert r['prices'][8]==pytest.approx(.5,abs=2.e-6)
    assert r['scenario_margins'][0]==pytest.approx(.5-1.e-12,abs=2.e-6)


def test_fixed_training_prices_caps_and_budget_are_original():
    fixed=np.zeros(16);fixed[0]=.6
    tables=[scene([family(0,False,[0.],[[1.]]),family(1,True,[],np.empty((0,1)),0.)])]
    r=solve_affine(problem(budget=1.),tables,fixed,seconds=10)
    assert r['amplitudes'][0]==.6 and r['prices'][0]==.6
    assert r['prices'][8]==pytest.approx(.8,abs=3.e-6)
    assert r['original_evaluation']['hard_budget_bound']<=1.+1.e-7
    assert r['original_evaluation']['numerical_feasible']


def test_exclusive_types_and_robust_device_rows():
    row=np.zeros((1,16));row[0,8]=1.
    p=problem(budget=2.,groups=[(8,9)],device=row,rhs=[.4])
    # c8+2c9 maximized by choosing c9=1. They cannot be combined.
    data=[scene([family(0,False,[0.],[[1.,2.]]),family(1,True,[],np.empty((0,1)),0.)])]
    r=solve_affine(p,data,np.zeros(16),seconds=10)
    assert r['prices'][8]==pytest.approx(0.,abs=1.e-7)
    assert r['prices'][9]==pytest.approx(1.,abs=2.e-6)
    assert r['original_evaluation']['exclusive_group_positive_counts']==[1]


def test_acceptance_ineligible_mass_is_not_renormalized():
    tables=[scene([family(0,False,[1.],[[0.]]),family(1,True,[],np.empty((0,1)),0.)],gain=-1.) for _ in range(2)]
    tables += [scene([family(0,False,[1.],[[0.]]),family(1,True,[],np.empty((0,1)),0.)]) for _ in range(8)]
    r=solve_affine(problem(),tables,np.zeros(16),seconds=10)
    assert r['status']=='acceptance_infeasible'
    assert r['required_scenarios']==9 and not r['sample_requirement_met']
    assert r['solver_calls']==0


def test_exact_shared_factor_device_uncertainty_can_cancel():
    row=np.zeros((1,16));row[0,8:10]=1.
    payload=problem(budget=2.,device=row,rhs=[.6]).to_dict()
    payload['response_factors'][8][0]=.2
    payload['response_factors'][9][0]=-.2
    p=PriceProblem.from_dict(payload)
    # x8+x9+.2|x8-x9|<=.6: sum .6 needs x8=x9=.3.
    data=[scene([family(0,False,[0.],[[1.,1.]]),family(1,True,[],np.empty((0,1)),0.)])]
    r=solve_affine(p,data,np.zeros(16),seconds=10)
    assert r['prices'][8:10]==pytest.approx([.3,.3],abs=2.e-6)
    assert r['original_evaluation']['numerical_feasible']


def test_arithmetic_uses_full_validation_normalizer():
    tables=[scene([family(0,False,[.8],[[1.]],.2),family(1,True,[],np.empty((0,1)),.1)],n=4)]
    r=evaluate_affine(tables,np.zeros(16))
    assert r['scenario_margins'][0]==pytest.approx(.2+.8/4-1.03*.1-1.e-12)


def test_zero_effect_target_unavailable_keeps_full_denominator():
    available=scene([family(0,False,[0.],[[1.]]),family(1,True,[],np.empty((0,1)),0.)])
    missing=scene([family(0,False,[0.],[[1.]])])
    data=[available]*9+[missing]
    r=solve_affine(problem(),data,np.zeros(16),seconds=10)
    assert r['feasible_incumbent'] and r['sample_requirement_met']
    assert r['scenario_margins'][-1] is None and not r['comparison_available'][-1]
    assert r['training_acceptance_passes'][-1] and not r['acceptance_passes'][-1]
    assert r['excitation_pass_count']==9 and r['scenario_count']==10
    assert r['zero_start_supplied']
    assert r['compilation']['unavailable_comparison_scenarios']==1
    fail=solve_affine(problem(),[available]*8+[missing]*2,np.zeros(16),seconds=10)
    assert fail['status']=='acceptance_infeasible' and fail['solver_calls']==0
