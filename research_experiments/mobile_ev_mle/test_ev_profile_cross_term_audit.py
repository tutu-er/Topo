"""Algebraic diagnostics independent of the saved AC experiment outcomes."""
import numpy as np
import pytest
from research_experiments.mobile_ev_mle.ev_profile_cross_term_audit import decomposition, state_vector


def test_bilinear_response_has_only_cross_remainder():
    t0,t1,w0,w1=2.,5.,3.,7.
    out=decomposition([t0*w0],[t1*w0],[t0*w1],[t1*w1],[[w0]],[[t0]],[t1-t0],[w1-w0])
    assert out['norms']['theta']==0
    assert out['norms']['state']==0
    assert out['norms']['cross']==12
    np.testing.assert_array_equal(out['vectors']['total'],out['vectors']['cross'])
    assert out['closure_norm']==0


def test_linear_response_has_no_remainder():
    out=decomposition([1.,2.],[3.,5.],[5.,7.],[7.,10.],[[2.],[3.]],[[4.],[5.]],[1.],[1.])
    assert all(value==0 for value in out['norms'].values())


def test_pure_curvature_is_not_attributed_to_interaction():
    # f(theta,w)=theta^2+w^2, from (2,3) to (5,7).
    out=decomposition([13.],[34.],[53.],[74.],[[4.]],[[6.]],[3.],[4.])
    assert out['norms']['theta']==9
    assert out['norms']['state']==16
    assert out['norms']['cross']==0
    assert out['norms']['total']==25


def test_vector_remainders_can_cancel():
    # f(theta,w)=theta^2-w^2, from (0,0) to (1,1).
    out=decomposition([0.],[1.],[-1.],[0.],[[0.]],[[0.]],[1.],[1.])
    assert out['norms']['total']==0
    assert out['norm_sum']==2
    assert out['pairwise_inner_products']['theta__state']==-1


def test_state_order_is_timewise_p_q_source_p_source_q():
    values=dict(p=np.array([[1,2],[3,4]]),q=np.array([[5,6],[7,8]]),source_p=np.array([9,10]),source_q=np.array([11,12]))
    np.testing.assert_array_equal(state_vector(values),[1,2,5,6,9,11,3,4,7,8,10,12])


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_nonfinite_mean_is_rejected(bad):
    with pytest.raises(ValueError,match='nonfinite'):
        decomposition([bad],[0.],[0.],[0.],[[0.]],[[0.]],[0.],[0.])


def test_inconsistent_jacobian_dimension_is_rejected():
    with pytest.raises(ValueError,match='dimensions'):
        decomposition([0.],[0.],[0.],[0.],[[0.,1.]],[[0.]],[0.],[0.])
