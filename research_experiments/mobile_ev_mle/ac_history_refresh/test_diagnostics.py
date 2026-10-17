"""Closed-form local approximation and compensated-curvature checks."""
from __future__ import annotations

import numpy as np
import pytest

from research_experiments.mobile_ev_mle.ac_history_refresh import diagnostics as tool
from research_experiments.mobile_ev_mle.ac_history_refresh.optimizer import GaussianHistoryProblem


def _prior(d=1,upper=10.,lower=-10.):
    return dict(mean=np.zeros(d),cov=np.eye(d),A=np.r_[np.eye(d),-np.eye(d)],
                b=np.r_[np.broadcast_to(upper,(d,)),-np.broadcast_to(lower,(d,))])


def _linear(J,*,physical=None,prior=None):
    J=np.asarray(J,float);d=1 if prior is None else len(prior['mean'])
    def evaluate(w,with_jacobian=True):
        good=True if physical is None else bool(physical(w))
        row=dict(mean=J@w,physical_ok=good,physical_failures=[] if good else ['state_boundary'])
        if with_jacobian:row['Jacobian']=J
        return row
    return GaussianHistoryProblem(evaluate,np.zeros(len(J)),_prior() if prior is None else prior,
                                  np.eye(J.shape[1]-d))


def test_affine_closure_is_zero_and_displacement_has_dimensionless_scale():
    J=np.array([[2.,1.],[1.,-.5]])
    problem=_linear(J);point=np.array([.3,-.1]);center=np.array([.5,.2]);delta=center-point
    result=tool.center_closure(problem,point,center,channel_sizes={'p':1,'v':1})
    assert result['overall_closure']['maximum_absolute']<1e-15
    assert abs(result['likelihood_difference']['AC_minus_affine_half_SSE'])<1e-15
    assert result['displacement']['prior_standardized_squared_norm']==pytest.approx(delta@delta)
    assert result['displacement']['GN_squared_norm']==pytest.approx(delta@(np.eye(2)+J.T@J)@delta)
    assert result['inference_acceptance_gate'] is False


def _quadratic_problem():
    def evaluate(w,with_jacobian=True):
        a,z=w;row=dict(mean=np.array([z,a*a+z]))
        if with_jacobian:row['Jacobian']=np.array([[0.,1.],[2*a,1.]])
        return row
    return GaussianHistoryProblem(evaluate,[.1,1.2],_prior(),np.eye(1))


def test_nonlinear_closure_reports_linear_channel_and_exact_likelihood_identity():
    problem=_quadratic_problem();point=np.array([1.,0.]);center=np.array([1.2,.1])
    result=tool.center_closure(problem,point,center,channel_sizes={'p':1,'v':1})
    assert result['channels']['p']['closure']['maximum_absolute']<1e-15
    assert result['channels']['v']['closure']['RMS']==pytest.approx(.04)
    # Affine residual=[0,.3], error=[0,.04].
    assert result['likelihood_difference']['AC_minus_affine_half_SSE']==pytest.approx(.3*.04+.04**2/2)
    assert abs(result['likelihood_difference']['identity_roundoff'])<1e-15
    assert result['likelihood_difference']['prior_terms_identical'] is True
    assert 'not posterior or predictive calibration' in result['scope']


def test_line_profile_restores_exact_state_compensation_and_linear_curvature():
    J=np.array([[2.,1.]]);problem=_linear(J);C=np.linalg.inv(np.eye(2)+J.T@J)
    result=tool.weak_profile_curvature(problem,[0.,0.],C)
    assert result['Schur_line_marginal_covariance_identity_relative_error']<1e-14
    row=result['directions'][0]
    assert row['line_profile_eigenvalue']==pytest.approx(3.)
    assert row['line_marginal_SD_prior_units']==pytest.approx(np.sqrt(1/3))
    np.testing.assert_allclose(row['state_compensation_prior_units'],[-1.])
    assert row['GN_curvature']==pytest.approx(1.)
    for step in row['steps']:
        assert step['result']['status']=='ok'
        assert step['result']['complete_objective_curvature']==pytest.approx(1.)
        assert step['result']['h_standard_deviations']==step['requested_h_standard_deviations']
    assert not result['full_true_Hessian_computed']


def test_bilinear_curvature_difference_has_known_residual_and_finite_step_terms():
    def evaluate(w,with_jacobian=True):
        a,z=w;row=dict(mean=np.array([a*z]))
        if with_jacobian:row['Jacobian']=np.array([[z,a]])
        return row
    problem=GaussianHistoryProblem(evaluate,[2.],_prior(),np.eye(1))
    point=np.ones(2);J=np.ones((1,2));C=np.linalg.inv(np.eye(2)+J.T@J)
    row=tool.weak_profile_curvature(problem,point,C)['directions'][0]
    assert row['first_derivative']==pytest.approx(0.,abs=1e-15)
    assert row['GN_curvature']==pytest.approx(1.)
    values=[]
    for step in row['steps']:
        h=step['result']['h_standard_deviations']
        curvature=step['result']['complete_objective_curvature']
        assert curvature==pytest.approx(5/3+h*h/9,rel=1e-13,abs=1e-13)
        values.append(curvature)
    assert abs(values[1]-5/3)<abs(values[0]-5/3)


def test_physical_shrink_records_both_sides_and_requested_sd_fraction():
    J=np.array([[2.,1.]]);problem=_linear(J,physical=lambda w:w[1]<=1.)
    C=np.linalg.inv(np.eye(2)+J.T@J)
    result=tool.weak_profile_curvature(problem,[0.,.9],C,standard_deviation_steps=(.5,))
    step=result['directions'][0]['steps'][0]
    assert step['requested_h_standard_deviations']==.5
    assert step['result']['h_standard_deviations']==.125
    assert len(step['attempts'])==3
    assert not step['attempts'][0]['minus']['physical_ok']
    assert step['attempts'][-1]['minus']['physical_ok']
    assert step['result']['complete_objective_curvature']==pytest.approx(1.)


def test_active_line_boundary_uses_symmetric_tangent_not_impossible_normal_steps():
    line=_prior(d=2,upper=[0.,10.],lower=[-10.,-10.]);J=np.array([[2.,1.,.5]])
    problem=_linear(J,prior=line);C=np.linalg.inv(np.eye(3)+J.T@J)
    result=tool.weak_profile_curvature(problem,np.zeros(3),C)
    assert result['line_tangent_dimension']==1
    assert len(result['directions'])==1 and list(result['active_line_indices'])==[0]
    assert abs(result['directions'][0]['direction_original_units'][0])<1e-15
    assert all(s['result']['status']=='ok' for s in result['directions'][0]['steps'])


def test_physical_block_retains_trials_and_does_not_claim_curvature_or_failure_gate():
    J=np.array([[2.,1.]]);problem=_linear(J,physical=lambda w:w[1]==0.)
    C=np.linalg.inv(np.eye(2)+J.T@J)
    result=tool.weak_profile_curvature(problem,[0.,0.],C,max_backtracks=2)
    for step in result['directions'][0]['steps']:
        assert step['result']['status']=='no_symmetric_physical_step'
        assert len(step['attempts'])==3
    assert result['posterior_calibration_claim'] is False and result['inference_acceptance_gate'] is False


def test_no_line_tangent_and_bad_channel_sizes_are_explicit():
    J=np.array([[2.,1.]]);line=_prior(upper=0.,lower=-10.)
    problem=_linear(J,prior=line);C=np.linalg.inv(np.eye(2)+J.T@J)
    result=tool.weak_profile_curvature(problem,[0.,0.],C)
    assert result['status']=='no_symmetric_line_tangent' and result['directions']==[]
    with pytest.raises(ValueError,match='cover every'):
        tool.center_closure(problem,[0.,0.],[0.,0.],channel_sizes={'p':2})
