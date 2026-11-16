import numpy as np
import pytest

from rnj_wzzt_dr.experiments.internal_contraction_20261116.residual_analysis import (
    error_classification,parameter_errors,physical_coefficients,point_matrix)


def candidate(i,supports,weights=None):
    return dict(source_candidate_id=i,point=dict(supports=supports,r=weights or [1.]*len(supports),x=[0.]*len(supports)))


@pytest.mark.parametrize('bank,selected,expected',[
    ([candidate(0,[[0,1],[2,3]])],0,'correct'),
    ([candidate(0,[[0,1]])],0,'truth_support_missing_from_active_union'),
    ([candidate(0,[[0,1]]),candidate(1,[[2,3]])],0,'union_covers_truth_but_no_exact_candidate'),
    ([candidate(0,[[0,1]]),candidate(1,[[0,1],[2,3]])],0,'exact_candidate_available_but_selected_wrong')])
def test_mutually_exclusive_error_categories(bank,selected,expected):
    r=error_classification(bank,selected,[[0,1],[2,3]],4,1e-9,internal=True)
    assert r['category']==expected


def test_zero_weight_is_not_structural_pool_absence():
    bank=[candidate(0,[[0,1],[2,3]],[1.,0.])]
    r=error_classification(bank,0,[[0,1],[2,3]],4,1e-9,internal=True)
    assert r['union_missing']==[[2,3]] and r['stored_union_missing']==[]
    assert r['union_missing_but_stored_zero']==[[2,3]]


def test_internal_correct_does_not_hide_missing_leaf_or_common_extra():
    bank=[candidate(0,[[0,1],[0,1,2]])]
    truth=[[0],[1],[2],[0,1]]
    assert error_classification(bank,0,truth,3,1e-9,internal=True)['correct']
    full=error_classification(bank,0,truth,3,1e-9)
    assert not full['correct'] and full['fn']==3 and full['fp']==1


def test_series_same_support_aggregates_ohm_and_response():
    edges=[dict(from_bus=0,to_bus=1,r_ohm=.1,x_ohm=.2),dict(from_bus=1,to_bus=101,r_ohm=.3,x_ohm=.4)]
    weights,counts,h=physical_coefficients(edges,[101],0,.8)
    np.testing.assert_allclose(weights[(0,)],[.4,.6]);assert counts[(0,)]==2
    np.testing.assert_allclose(h,[[1.,1.5]])


def test_line_mape_and_matrix_error_are_distinct_and_missing_explicit():
    truth={(0,):np.array([1.,2.]),(1,):np.array([3.,4.])}
    hstar=np.array([[2.5,0.,5.,0.],[0.,7.5,0.,10.]])
    point=dict(supports=[[0],[0,1]],r=[5.,1.],x=[10.,1.])
    result=parameter_errors(point,hstar,truth,[101,102],.8,1e-9,{'P':[101],'Q':[102]})
    assert result['matched_line']['matched_support_count']==1
    assert result['matched_line']['missing_support_count']==1
    assert result['matched_line']['extra_support_count']==1
    assert result['matched_line']['R_MAPE']==pytest.approx(1.)
    assert result['matrix']['R_relative_frobenius']!=pytest.approx(1.)
    assert result['resource_columns']['Q']['labels']==[102]


def test_incorrect_physical_tree_fails():
    with pytest.raises(ValueError,match='Unreachable'):
        physical_coefficients([dict(from_bus=9,to_bus=101,r_ohm=1,x_ohm=1)],[101],0,.8)
