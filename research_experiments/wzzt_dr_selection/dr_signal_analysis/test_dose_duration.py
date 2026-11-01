"""Pure ray/domain/norm checks; no historical or physical test inputs."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dose_duration import largest_common_scale,paired_metrics,build_problem,check_four_corners,historical_gradient
from research_experiments.wzzt_dr_selection.dr_day_modules.day_schedule import ResourceDay
from research_experiments.wzzt_dr_selection.dr_frozen_distance.design import contrast_shape


def problem(slots=8,budget=1.6):
    resource=ResourceDay(resource_id='load',node_label=1,baseline_kw=1.,power_upper_kw=2.,
        prefix_delta_lower_kwh=-2.,prefix_delta_upper_kwh=2.,available=True,reactive_ratio=.35)
    shape=np.zeros(96);shape[:slots]=-1;shape[slots:2*slots]=1
    p=build_problem([resource],[shape],[1],[.5],[[.15,.05]],[2.],np.zeros(192),budget=budget)
    return p,shape


def test_budget_ray_intersection_and_supported_vertices():
    p,_=problem(8)
    r=largest_common_scale(p,[1.])
    assert r['beta']==pytest.approx(np.sqrt(1.6/1.2))
    assert r['audit']['hard_budget_bound']==pytest.approx(1.6)
    assert 'hard_budget' in r['binding']
    assert check_four_corners(p,r['amplitude'])<=1e-8


def test_long_window_energy_bound_can_prevent_spending_full_budget():
    p,_=problem(48)
    r=largest_common_scale(p,[1.])
    assert r['amplitude'][0]==pytest.approx(2/(12*1.2))
    assert r['audit']['hard_budget_bound']<1.6
    assert any(v.startswith('device:') for v in r['binding'])
    assert check_four_corners(p,r['amplitude'])<=1e-8


def test_l1_can_increase_while_rms_and_quadratic_dose_stay_constant():
    effects=[]
    for active in (4,16):
        u=np.zeros(96);u[:active//2]=1/np.sqrt(active);u[active//2:active]=-1/np.sqrt(active)
        effects.append(u)
    a=paired_metrics(np.zeros(96),effects[0]);b=paired_metrics(np.zeros(96),effects[1])
    assert b['D1_gain']==pytest.approx(2*a['D1_gain'])
    assert b['D2_gain']==pytest.approx(a['D2_gain'])
    assert np.sum(effects[0]**2)==pytest.approx(np.sum(effects[1]**2))


def test_prediction_distance_scaling_is_convex_not_linear():
    base=np.array([.2,-.3,2.]);effect=np.array([.4,.8,-1.])
    q1=paired_metrics(base,effect)['D1_gain']
    for alpha in (1.,np.sqrt(2),2.):
        assert paired_metrics(base,alpha*effect)['D1_gain']>=alpha*q1-1e-12


@pytest.mark.parametrize('m',[4,8,16,24,32,48])
def test_historical_shapes_preserve_equal_reduction_and_rebound(m):
    g=np.cos(np.linspace(0,2*np.pi,96))
    h=contrast_shape(g,m)
    assert np.count_nonzero(h<0)==m and np.count_nonzero(h>0)==m
    assert h.sum()==0
    assert np.abs(h).max()==1


def test_invalid_zero_direction_rejected():
    p,_=problem()
    with pytest.raises(ValueError):largest_common_scale(p,[0.])


def test_resource_day_expanded_reactive_ratio_gradient_matches_scalar_formula():
    p,h=problem();direction=np.column_stack((h,np.ones(96)))
    dr=np.array([.1,.2]);dx=np.array([.4,.3]);ratio=np.full(96,.35)
    result=historical_gradient(direction,dr,dx,ratio,200.)
    assert np.allclose(result,direction@(dr+.35*dx)/(96*2*200.))
