"""Small independent checks of the outside-freeze saved-data reviewers."""
import copy
import numpy as np
import pytest
from .ev_original_square_structure_audit import audit_witness,gamma
from .ev_original_location_review import signed_margin,channel_summary,curve_hash


def model_and_witness():
    model=dict(name='a_to_b',c=[1.,2.],B=[[1.,2.,0.,0.],[0.,0.,3.,4.]],
        A0=[[0.],[0.]],At=np.zeros((4,2,1)).tolist(),history_rows=0,
        control_time_site=[[0,0],[0,1],[1,0],[1,1]],
        domain=dict(base_center=[0.],shift=np.zeros((1,4)).tolist(),
            action_lower=[-2.]*4,action_upper=[2.]*4))
    witness=dict(origin='test',physical_intercept=[1.],physical_slope=np.zeros((1,4)).tolist(),validation=dict(valid=True))
    return model,witness


def test_signed_legal_control_diagonal_identity():
    model,witness=model_and_witness()
    result=audit_witness(model,witness,[[1.,0.,0.,-1.]])
    check=result['saved_action_checks'][0]
    assert check['full_square']==pytest.approx(8.)
    assert check['diagonal_square']==pytest.approx(8.)
    assert result['cross_time_absolute_overlap_max']==0.
    assert result['same_time_gram_max']==12.
    assert check['largest_same_time_product']==0.


def test_same_slot_cross_coefficient_is_not_discarded_as_small():
    model,witness=model_and_witness()
    r=audit_witness(model,witness,[[1.,1.,0.,0.]])
    check=r['saved_action_checks'][0]
    assert check['difference']==pytest.approx(4.)
    assert check['same_time_cross_contribution']==pytest.approx(4.)
    assert check['largest_same_time_product']==1.


def test_cross_time_overlap_requires_guard_or_full_terms():
    model,witness=model_and_witness();model['B'][0][2]=.1
    result=audit_witness(model,witness,[[1.,0.,1.,0.]])
    assert result['cross_time_absolute_overlap_max']==pytest.approx(.2)
    assert result['numerical_guard_parts']['cross_time']>=.8
    assert result['saved_action_checks'][0]['guarded_diagonal_dominates']


def test_different_witness_slope_fails_required_condition():
    model,witness=model_and_witness();witness['physical_slope'][0][0]=.01
    result=audit_witness(model,witness,[[1.,0.,0.,0.]])
    assert not result['affine_slope_equals_shift']
    assert result['max_affine_slope_error']==pytest.approx(.01)


def test_roundoff_guard_is_upward_and_small():
    model,witness=model_and_witness();result=audit_witness(model,witness,[[.3,0.,-.7,0.]])
    assert 0<result['squared_cost_numerical_upward_guard']<1e-9
    assert result['saved_action_checks'][0]['guarded_diagonal_dominates']
    assert gamma(100)>0


def test_curve_cache_normalizes_signed_zero_without_approximation():
    assert curve_hash([[1.,0.,-0.]])==curve_hash([[1.,-0.,0.]])
    assert curve_hash([[1.,0.,0.]])!=curve_hash([[1.+1e-12,0.,0.]])


def test_margin_keeps_negative_values_and_tie_order():
    assert signed_margin({'true':3.,'b':2.,'a':2.},'true')==(-1.,'a')


def test_fixed_wrong_channel_identity_keeps_nearest_switch():
    rows=[]
    for method in ('charge_mobile','v2g_mobile'):
        for kind in ('noisy','noiseless'):
            for variant in ('original','remap'):
                for candidate in ('a','b','c'):
                    v={('original','a'):1.,('original','b'):9.,('original','c'):12.,
                       ('remap','a'):1.,('remap','b'):8.,('remap','c'):5.}[variant,candidate]
                    rows.append(dict(seed=1,method=method,kind=kind,variant=variant,candidate=candidate,
                        true_candidate='a',nearest_wrong='b' if variant=='original' else 'c',
                        joint_SSE=v+2.,historical_SSE=2.,future_channel_SSE=dict(v=v)))
    summary,records=channel_summary(rows)
    assert len(records)==8
    assert all(s['nearest_switch_histories']==1 for s in summary)
    assert all(r['fixed_wrong_margin_delta']==pytest.approx(r['future_v_margin_delta']) for r in records)
