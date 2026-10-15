"""Independent small MILP audits and actual AC contrast integration."""
from itertools import product
from types import SimpleNamespace
from copy import deepcopy
from inspect import signature
import numpy as np
import pytest

from . import ev_response_metric_design as design
from . import ev_fixed_train_design as fixed
from .test_ev_gradient_flow_design import context


class Box:
    def __init__(self, low, high):
        self.low=np.array(low,float);self.high=np.array(high,float)
        self.indices=SimpleNamespace(u=np.zeros((1,len(low))))
    def as_dict(self):
        n=len(self.low)
        return dict(lower=self.low,upper=self.high,control_lower=self.low,control_upper=self.high,
            u_map=np.eye(n),integrality=np.zeros(n,int),constraints=[])
    def check(self,x):
        return dict(feasible=bool(x is not None and np.all(x>=self.low-1e-7) and np.all(x<=self.high+1e-7)))


@pytest.mark.parametrize('d,lo,hi,expected',[(0,-1,2,0),(3,-1,2,2),(-4,-1,2,2),(2,1,3,3)])
def test_scalar_interval_distance(d,lo,hi,expected):
    assert design.interval_gap(d,lo,hi)==pytest.approx(expected)


def test_support_interval_matches_all_box_corners():
    c=np.array([2.,-3.,0.]);lo=np.array([-2.,1.,-5.]);hi=np.array([3.,4.,9.])
    vals=[c@np.array(v) for v in product(*zip(lo,hi))]
    actual=design.box_support_interval(c,lo,hi)
    assert actual==pytest.approx((min(vals),max(vals)))


def test_exact_MILP_agrees_with_independent_tiny_box_vertex_audit():
    box=Box([-2.,-1.],[3.,4.]);alpha=np.array([.3,-.7,1.])
    beta=np.array([[1.,2.],[-3.,.2],[.7,-1.4]])
    lo=np.array([-.4,-2.,-.1]);hi=np.array([.8,1.,1.4])
    vertices=list(product(*zip(box.low,box.high)))
    expected=max(float(np.sum(design.interval_gap(alpha+beta@v,lo,hi))) for v in vertices)
    r=design.solve_interval_gap_milp(box,alpha,beta,lo,hi,backend='scipy',solver_seconds=3.)
    assert r['success']
    assert r['proxy_value']==pytest.approx(expected)
    assert r['solver_value']==pytest.approx(expected)
    assert r['proxy_best_bound']==pytest.approx(expected)
    assert np.min(r['big_M'])>=0
    assert r['added_binary_count']==9


def test_complete_overlap_has_zero_gap_not_invented_separation():
    r=design.solve_interval_gap_milp(Box([-1.],[1.]),[0.],[[1.]],[-2.],[2.],backend='scipy')
    assert r['success'] and r['proxy_value']==pytest.approx(0.)


def test_worst_scenario_MILP_finds_interior_optimum_not_just_box_vertices():
    # max min(|x-1|,|x+1|) on [-1,1] is 1 at x=0; both vertices give zero.
    r=design.solve_interval_gap_milp(Box([-1.],[1.]),[-1.,1.],[[1.],[1.]],
        [0.,0.],[0.,0.],scenario_groups=[[0],[1]],backend='scipy',solver_seconds=3.)
    assert r['success'] and r['proxy_value']==pytest.approx(1.)
    assert r['power_kw'][0,0]==pytest.approx(0.,abs=1e-7)
    assert r['proxy_best_bound']==pytest.approx(1.)


def test_source_projection_can_remove_all_discriminating_signal():
    Ga=np.array([[1.,0.],[0.,1.],[0.,0.]])
    w,r=design.contrast_direction([1.,0.,0.],np.zeros((3,1)),Ga,Ga)
    assert np.allclose(w,0.)
    assert r['source_rank']==2
    assert r['direction_origin']=='no_resolved_contrast_signal'


def test_truncated_source_column_still_pays_original_box_support():
    Ga=np.array([[1.,0.],[0.,1e-9],[0.,0.]])
    w,r=design.contrast_direction([0.,1.,0.],np.zeros((3,1)),Ga,Ga,rcond=1e-6)
    assert r['source_rank']==1 and abs(w@Ga[:,1])>0
    l,h=design.box_support_interval(np.r_[w@Ga,-w@Ga],-np.ones(4),np.ones(4))
    assert h>0 and l<0
    assert np.linalg.norm(w)==pytest.approx(1.)


@pytest.mark.parametrize('projection',['joint','bounded_average'])
def test_actual_AC_anchor_and_input_derivatives(context,projection):
    anchor=context['reference_kw']
    m=design.build_contrast(context,anchor,projection=projection)
    r=design.evaluate_contrast(m,anchor)
    assert r['success']
    assert np.max(abs(r['scalar_affine_error']))<1e-10
    assert all(row['source_support_low']<=1e-9 and row['source_support_high']>=-1e-9 for row in m['direction_records'])
    step=1e-3;plus=anchor.copy();minus=anchor.copy();plus[1,0]+=step;minus[1,0]-=step
    derivative=(design.evaluate_contrast(m,plus)['scalar_AC_contrast']-design.evaluate_contrast(m,minus)['scalar_AC_contrast'])/(2*step)
    assert derivative==pytest.approx(m['beta'][:,3],rel=1e-4,abs=1e-7)
    assert r['full_AC_source_distance_certificate'] is False


def test_actual_AC_selector_operation_and_no_truth_interface(context):
    domain=fixed._service(3,3,mobile=False,discharge=True)
    power,record=design.select_contrast_controls(context,domain,dict(square=context['reference_kw']),
        projection='bounded_average',options=dict(backend='scipy',solver_seconds=2.))
    assert power is not None and record['success']
    assert record['selected_operation']['feasible']
    assert record['selected_score']>=record['initial_best_score']-1e-8
    assert record['source_fits_in_selector']==0
    assert record['MILP']['added_binary_count']==9
    assert all(name not in signature(design.select_contrast_controls).parameters for name in ('truth','validation','future_noise'))
    changed=deepcopy(context)
    changed['lines'][context['bank'][0].name]['r']=changed['lines'][context['bank'][0].name]['r'].copy()
    changed['lines'][context['bank'][0].name]['r'][0]*=1.01
    with pytest.raises(ValueError,match='snapshot'):
        design.build_contrast(changed,context['reference_kw'])


def test_training_scenarios_share_one_direction_and_reject_truth_provenance(context):
    domain=fixed._service(3,3,mobile=False,discharge=True)
    scenario=dict(name='stress',p=context['background']['p']+.001,q=context['background']['q'],
        metadata=dict(training_only=True,uses_truth=False,uses_validation=False,uses_future_noise=False))
    power,record=design.select_contrast_controls(context,domain,dict(square=context['reference_kw']),
        projection='bounded_average',options=dict(backend='scipy',solver_seconds=2.),prediction_scenarios=[scenario])
    assert power is not None and record['success']
    assert len(record['scenario_models'])==2
    assert np.array_equal(record['scenario_models'][0]['weights'],record['scenario_models'][1]['weights'])
    assert record['MILP']['scenario_count']==2
    assert record['selected_score']>=record['initial_best_score']-1e-8
    scenario['metadata']['uses_truth']=True
    with pytest.raises(ValueError,match='provenance'):
        design.select_contrast_controls(context,domain,dict(square=context['reference_kw']),prediction_scenarios=[scenario])


def test_frozen_affine_bound_uses_affine_values_and_AC_proxy_is_separate(context):
    m=design.build_contrast(context,context['reference_kw'],projection='bounded_average')
    # Deliberately distinct affine/AC payloads audit the reported certificate's
    # arithmetic, independent of how an expansion point was constructed.
    m['alpha']=m['alpha']+100.
    r=design.evaluate_contrast(m,context['reference_kw'])
    d=m['alpha']+m['beta']@context['reference_kw'].ravel()
    g=design.interval_gap(d,m['source_low'],m['source_high'])
    assert r['affine_source_distance_lower_bound_from_L1']==pytest.approx(float(g.sum()**2/(2*len(g))))
    assert r['affine_source_distance_lower_bound_from_L1']!=pytest.approx(r['source_adjusted_AC_contrast_proxy_squared'])


def test_expired_contrast_budget_never_uses_partial_scenario_score(context):
    domain=fixed._service(3,3,mobile=False,discharge=True)
    u,r=design.select_contrast_controls(context,domain,dict(square=context['reference_kw']),options=dict(total_seconds=1e-12))
    assert u is None and not r['success']
    assert r.get('initialization_budget_exhausted')


def test_invalid_source_interval_rejected():
    with pytest.raises(ValueError,match='ordered'):
        design.interval_gap([0.],[1.],[-1.])
