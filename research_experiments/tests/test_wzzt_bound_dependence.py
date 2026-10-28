import numpy as np
import pandas as pd
import pytest
from research_experiments.wzzt_dr_selection.baseline_alignment.bound_dependence import design,solve_nonnegative_l1


def test_unbounded_nonnegative_l1_recovers_above_two_and_dual_matches():
    a=np.array([[1.],[2.],[3.]])
    y=3.*a[:,0]
    free=solve_nonnegative_l1(a,y,time_limit=10.)
    capped=solve_nonnegative_l1(a,y,time_limit=10.,upper_bound=2.)
    assert free['status']=='complete'
    assert free['weights'][0]==pytest.approx(3.)
    assert free['objective']==pytest.approx(0.,abs=1e-10)
    assert capped['objective']==pytest.approx(2.)
    assert capped['checks']['primal_dual_gap']==pytest.approx(0.,abs=1e-10)


def test_independent_feature_columns_preserve_terminal_output_rows():
    p=pd.DataFrame([[1.,2.,4.],[2.,3.,5.]],columns=[101,102,103])
    q=pd.DataFrame([[2.,1.,3.],[4.,2.,1.]],columns=p.columns)
    y=pd.DataFrame(np.zeros((2,3)),columns=p.columns)
    a,target=design([{'P_terminal':p,'Q_terminal':q,'drop_target':y}],[[101,102],[103]])
    assert a.shape==(6,4)
    np.testing.assert_allclose(a[:,0],[3,3,0,5,5,0])
    np.testing.assert_allclose(a[:,1],[0,0,4,0,0,5])
    np.testing.assert_allclose(a[:,2],[3,3,0,6,6,0])
    assert len(target)==6


def test_constant_prediction_uses_l1_median_not_mean():
    fit=solve_nonnegative_l1(np.ones((3,1)),np.array([1.,2.,30.]),time_limit=10.)
    assert fit['status']=='complete'
    assert fit['weights'][0]==pytest.approx(2.)
    assert fit['objective']==pytest.approx(29./3)
