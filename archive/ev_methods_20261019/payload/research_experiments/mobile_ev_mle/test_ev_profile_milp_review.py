"""Tamper detection and independent-unit checks for the saved-record review."""
from copy import deepcopy
import numpy as np
import pytest
from .ev_profile_milp_review import (replay_projection,replay_bounded,replay_solve,
    summarize,domain_covered,floor_monotonicity)


def test_projection_detects_changed_compressed_constant():
    model=dict(name='x',d=[2.],T=[[1.]],offset=0.,
        full=dict(c=[0.,2.],B=[[0.],[1.]],C=[[1.],[0.]],scale=[1.],
            parameter_center=[.5],lower=[-.5],upper=[.5]))
    assert replay_projection(model,{'one':[1.]})['success']
    bad=deepcopy(model);bad['d']=[3.]
    with pytest.raises(ValueError,match='Gram'):replay_projection(bad,{'one':[1.]})


def test_bounded_cost_tamper_is_rejected():
    model=dict(full=dict(c=[1.,2.],B=[[0.],[0.]],C=[[1.],[1.]],
        lower=[-.2],upper=[.3],parameter_center=[2.]))
    saved=dict(step=[-.2],feasible=True,primal_upper=3.88,tangent_lower_numeric=3.88,
        free_step=[-1.5],free_value=.5,free_bound_violation=1.3,parameter=[1.8])
    assert replay_bounded(model,[0.],saved)['upper']==pytest.approx(3.88)
    saved['primal_upper']+=1.
    with pytest.raises(ValueError,match='upper'):replay_bounded(model,[0.],saved)


def test_MILP_lower_is_recomputed_from_actual_quadratic():
    model=dict(d=[0.],T=[[1.,0.,0.,0.]],offset=0.)
    saved=dict(u=[3.,3.,3.,3.],quadratic_values=[4.5],objective_lower=4.5,
        objective_upper=18.,original_gap=13.5,segments=4,
        secant_error_by_quadratic=[.28125],chord_value_at_returned_u=4.5,solver_success=True)
    assert replay_solve([model],saved)['lower']==pytest.approx(4.5)
    saved['objective_lower']=5.5
    with pytest.raises(ValueError,match='lower'):replay_solve([model],saved)


def test_three_repetitions_are_one_independent_base():
    base=dict(seed=7,covered=True,selection_fallback=False,activation_would_occur=True,
        saved_fit_states_replayed=4,physical_responses_replayed=4,profile_minus_constant_margin=1.,
        selected_pair=['a','b'],pair_includes_truth=True,
        methods=[dict(method=m,status='complete',margin=1.,correct_winner=True)
            for m in ('none','constant','profile_milp','fixed_mle_milp')])
    result=summarize([deepcopy(base) for _ in range(3)])
    assert result['completed_conditions']==3
    assert result['independent_base_seeds']==1
    assert result['groups'][0]['methods'][0]['conditions']==3


def test_coverage_and_nested_domain_violation_are_distinct():
    supports=[[0,1,2],[0,1],[0],[1],[2]]
    r=[.005,.0003,.03,.03,.03];x=[.003,.0002,.01,.01,.01]
    assert domain_covered(supports,r,x,1e-6)
    assert not domain_covered(supports,r,x,.001)
    low=dict(seed=1,truth_r=.003,fit_floor=1e-6,history={'same':1},rows=[],
        historical_fits=[dict(name='a',success=True,loss=10.)])
    high=deepcopy(low);high['fit_floor']=.001;high['historical_fits'][0]['loss']=9.
    checked=floor_monotonicity([low,high])
    assert checked['checks']==1 and checked['violation_count']==1
