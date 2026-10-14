"""Closed-form and corruption checks for the outside-freeze QP reviewer."""
from copy import deepcopy
import numpy as np
import pytest
from .ev_compact_qp_review import replay_exchange
from .ev_original_review import replay_qp


def scalar():
    # min_{-1<=xi<=1} (3-(1+u)xi)^2, u=.5.
    # The optimum is the active upper endpoint xi=1; r=1.5 and SSE=2.25.
    model=dict(name='a__to__b',c=np.array([3.]),B=np.array([[0.]]),A0=np.array([[-1.]]),
        At=np.array([[[-1.]]]),task_weight=1.,competitor_history_cost=.2,
        domain=dict(physical_lower=np.array([-1.]),physical_upper=np.array([1.]),
            base_center=np.array([0.]),shift=np.array([[0.]])))
    primal=2.25;g=-4.5
    guard=256*np.finfo(float).eps*(1+primal+abs(g)*(1+1+2))
    qp=dict(physical_parameter=[1.],z=[1.],residual=[1.5],primal_upper=primal,
        tangent_lower_numeric=primal-guard,numeric_guard=guard,success=True)
    return model,qp


def exchange(model,qp,cap=None):
    lower=.5*(qp['tangent_lower_numeric']-.2);upper=.5*(qp['primal_upper']-.2)
    if cap is not None:lower=min(lower,cap);upper=min(upper,cap)
    return dict(required_pairs=[model['name']],aggregation='mean',
        weights={model['name']:1.},offsets={model['name']:.2},score_scale=.5,
        score_caps=None if cap is None else {model['name']:cap},status='test_active_bound',
        evaluations=[dict(u=[.5],origin='analytic',qp_results=[qp],score_lower=lower,score_upper=upper)])


def test_closed_form_active_boundary_tangent_bound():
    model,qp=scalar()
    audit=replay_qp(model,np.array([.5]),qp)
    assert audit['upper']==pytest.approx(2.25)
    assert audit['lower']<=2.25 and 2.25-audit['lower']<1e-10
    result=replay_exchange(exchange(model,qp),[model])
    assert result['counts']['QP_states_replayed']==1
    assert result['evaluations'][0]['score_upper']==pytest.approx(1.025)


def test_same_residual_norm_does_not_hide_wrong_matrix_sign():
    model,qp=scalar();bad=deepcopy(model)
    for key in ('c','B','A0','At'):bad[key]=-bad[key]
    # Wrong reconstructed residual is -1.5: a norm-only audit would still pass.
    with pytest.raises(ValueError,match='original QP residual'):
        replay_qp(bad,np.array([.5]),qp)


def test_corrupted_lower_certificate_is_rejected():
    model,qp=scalar();bad=deepcopy(qp);bad['tangent_lower_numeric']=.1
    with pytest.raises(ValueError,match='tangent lower'):replay_qp(model,np.array([.5]),bad)


def test_saved_cap_and_model_identity_are_checked():
    model,qp=scalar()
    result=replay_exchange(exchange(model,qp,.5),[model])
    assert result['evaluations'][0]['score_lower']==pytest.approx(.5)
    bad=exchange(model,qp);bad['required_pairs']=['other']
    with pytest.raises(ValueError,match='order'):replay_exchange(bad,[model])


def test_common_H_baseline_is_not_reset_to_expansion_cost():
    model,qp=scalar();bad=exchange(model,qp);bad['offsets'][model['name']]=.7
    with pytest.raises(ValueError,match='historical baseline'):replay_exchange(bad,[model])


def test_exception_fallback_retains_feasible_upper_and_zero_lower():
    model,qp=scalar();fallback=dict(qp,fallback=True,success=False,tangent_lower_numeric=0.)
    result=replay_qp(model,np.array([.5]),fallback)
    assert result['upper']==pytest.approx(2.25) and result['lower']==0.
    assert result['declared_exception_fallback'] and not result['success']
