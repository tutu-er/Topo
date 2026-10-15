"""Independent analytic checks of fixed-contrast/L1 EV design mathematics.

Finite vertex/sign sets below audit identities only. They are not an EV
selector or an action-library implementation; no AC fits or MILPs are run.
"""
from __future__ import annotations

import itertools
from math import erf, sqrt

import numpy as np
import pytest


def _normal_moments(function, mean, covariance):
    # Three-node Gauss-Hermite integrates all degree-four Gaussian polynomials.
    nodes, weights = np.polynomial.hermite.hermgauss(3)
    factor = np.linalg.cholesky(covariance)
    first = second = 0.
    for inds in itertools.product(range(3), repeat=len(mean)):
        z = sqrt(2.)*nodes[list(inds)]
        w = float(np.prod(weights[list(inds)]))/np.pi**(len(mean)/2.)
        value = float(function(mean+factor@z))
        first += w*value
        second += w*value*value
    return first, second-first*first


def _orthogonal_projector(G):
    return np.eye(G.shape[0])-G@np.linalg.pinv(G)


def test_Gaussian_point_log_ratio_mean_variance_with_prediction_bias():
    a = np.array([1., -.5, .2])
    b = np.array([-.3, .6, -.2])
    error = np.array([-.2, .1, .05])
    sigma = np.array([[1., .2, 0.], [.2, 1.3, .1], [0., .1, .8]])
    delta = a-b
    mean, variance = _normal_moments(lambda y:delta@(y-(a+b)/2.), a+error, sigma)
    assert mean == pytest.approx(.5*(delta@delta)+delta@error, abs=1e-14)
    assert variance == pytest.approx(delta@sigma@delta, abs=1e-13)


def test_increasing_nominal_distance_can_increase_wrong_model_confidence():
    # a_hat=u,b_hat=0 but a_true=-u: the action amplifies a wrong response model.
    for u in (1., 2., 4.):
        delta, error = u, -2.*u
        nominal_mean = .5*delta*delta
        actual_mean = nominal_mean+delta*error
        assert nominal_mean > 0.
        assert actual_mean == pytest.approx(-1.5*u*u)


def test_correct_point_model_classification_error_is_monotone_in_distance():
    phi = lambda x:.5*(1.+erf(x/sqrt(2.)))
    errors = [phi(-d/2.) for d in (0., .5, 1., 2., 4.)]
    assert all(a > b for a, b in zip(errors, errors[1:]))


def test_profile_Gaussian_log_ratio_quadratic_moments_include_rank_term():
    a = np.array([.4, -.2, .7])
    b = np.array([-.3, .1, .2])
    ga = np.array([[1.], [0.], [0.]])
    gb = np.array([[0., 0.], [1., 0.], [0., 1.]])
    pa, pb = _orthogonal_projector(ga), _orthogonal_projector(gb)
    mean_true = a+ga[:,0]*.3+np.array([0., .05, -.03])
    statistic = lambda y:.5*((y-b)@pb@(y-b)-(y-a)@pa@(y-a))
    mean, variance = _normal_moments(statistic, mean_true, np.eye(3))
    expected = .5*((mean_true-b)@pb@(mean_true-b)-(mean_true-a)@pa@(mean_true-a)
                     +np.trace(pb-pa))
    H = pb-pa
    expected_var = .5*np.trace(H@H)+np.linalg.norm(pb@(mean_true-b)-pa@(mean_true-a))**2
    assert mean == pytest.approx(expected, abs=1e-14)
    assert variance == pytest.approx(expected_var, abs=1e-13)
    assert np.trace(pb-pa) == -1.


def test_common_source_space_reduces_profile_ratio_to_linear_statistic():
    g = np.array([[1.], [1.], [0.]])
    P = _orthogonal_projector(g)
    a, b = np.array([1., .2, .3]), np.array([.1, -.2, .8])
    y = np.array([.4, .2, -.1])
    quadratic = .5*((y-b)@P@(y-b)-(y-a)@P@(y-a))
    linear = (P@(a-b))@(y-(a+b)/2.)
    assert quadratic == pytest.approx(linear, abs=1e-15)


def test_unit_joint_source_contrast_bounds_both_directed_profile_distances():
    ga = np.array([[1.], [1.], [0.], [0.]])
    gb = np.array([[1.], [0.], [1.], [0.]])
    G = np.column_stack((ga, gb))
    delta = np.array([.2, -.3, .4, 1.])
    P = _orthogonal_projector(G)
    w = P@delta/np.linalg.norm(P@delta)
    np.testing.assert_allclose(w@G, np.zeros(2), atol=3e-16)
    contrast = w@delta
    dab = .5*np.linalg.norm(_orthogonal_projector(gb)@delta)**2
    dba = .5*np.linalg.norm(_orthogonal_projector(ga)@delta)**2
    assert dab >= .5*contrast**2-1e-14
    assert dba >= .5*contrast**2-1e-14


def test_public_response_direction_avoids_zero_residual_gradient():
    g = np.array([[1.], [0.], [0.]])
    H = np.array([[0.], [2.], [0.]])
    public_v = np.array([1.])
    response = _orthogonal_projector(g)@H@public_v
    w = response/np.linalg.norm(response)
    beta = float(w@H[:,0])
    assert beta == 2.
    d_at_origin = 0.
    assert d_at_origin*beta == 0.  # squared-distance gradient stalls
    assert max(beta*(-1.), beta*1.) == 2.  # signed contrast has a direction


def test_nearly_common_weak_source_columns_can_destroy_joint_projection_signal():
    epsilon = 1e-7
    eye = np.eye(5)
    ga = eye[:,:2]
    gb = np.column_stack((eye[:,0]+epsilon*eye[:,2], eye[:,1]+epsilon*eye[:,3]))
    G = np.column_stack((ga,gb))
    U, singular, _ = np.linalg.svd(G, full_matrices=True)
    assert singular[-1] > 0.
    assert singular[-1] < 1e-6*singular[0]
    exact_joint = np.eye(5)-U[:,:4]@U[:,:4].T
    signal = eye[:,2]
    assert np.linalg.norm(exact_joint@signal) < 2e-9
    assert np.linalg.norm(_orthogonal_projector(gb)@signal) > .999999
    # A retained physical contrast has tiny, rather than exactly zero, leakage.
    assert np.linalg.norm(signal@ga) == 0.
    assert np.max(abs(signal@gb)) == epsilon
    # +/-0.5 source bounds limit this weak column to epsilon/2 response.
    assert .5*np.sum(abs(signal@gb)) == epsilon/2.


def test_joint_projection_zero_signal_remains_a_declared_failure():
    G = np.array([[1.,0.],[0.,1.],[0.,0.]])
    signal = np.array([.2,.3,0.])
    assert np.linalg.norm(_orthogonal_projector(G)@signal) == 0.


def _support(g, low, high):
    return float(np.maximum(np.asarray(g)*low, np.asarray(g)*high).sum())


@pytest.mark.parametrize('sign', [-1.,1.])
def test_physical_source_box_leak_bound_without_rank_assumptions(sign):
    nominal = .8
    ga = np.array([.02,-.03]); gb = np.array([.04,.01])
    lo, hi = np.array([-.5,-.2]), np.array([.3,.4])
    lower = sign*nominal-_support(-sign*ga,lo,hi)-_support(sign*gb,lo,hi)
    attained = []
    for sa in itertools.product(*zip(lo,hi)):
        for sb in itertools.product(*zip(lo,hi)):
            value = sign*(nominal+ga@sa-gb@sb)
            assert value >= lower-1e-15
            attained.append(value)
    assert min(attained) == pytest.approx(lower, abs=1e-15)


@pytest.mark.parametrize('low,high', [(-3.,2.),(-.2,5.),(-1.,1.)])
def test_binary_absolute_value_constraints_are_exact_on_valid_bounds(low,high):
    for d in np.linspace(low,high,31):
        possible = []
        for binary in (0,1):
            lower = max(0.,d,-d)
            upper = min(max(-low,high),d-2*low*(1-binary),-d+2*high*binary)
            if lower <= upper+1e-14:
                assert lower == pytest.approx(abs(d), abs=1e-14)
                assert upper == pytest.approx(abs(d), abs=1e-14)
                possible.append(binary)
        assert possible


def test_maximizing_only_absolute_value_epigraph_is_wrong():
    d, v = .2,100.
    assert v >= d and v >= -d
    assert v != abs(d)


def test_affine_response_bounds_from_power_box_are_valid_and_attained():
    alpha = -.2
    beta = np.array([.3,-.7,.2])
    lo, hi = np.array([-2.,0.,-1.]),np.array([3.,2.,1.])
    lower = alpha+np.minimum(beta*lo,beta*hi).sum()
    upper = alpha+np.maximum(beta*lo,beta*hi).sum()
    values = [alpha+beta@u for u in itertools.product(*zip(lo,hi))]
    assert min(values) == pytest.approx(lower)
    assert max(values) == pytest.approx(upper)


def test_L1_L2_bounds_and_different_rankings():
    for d in (np.array([2.,0.]),np.array([1.2,1.2]),np.array([-.4,.2,1.])):
        one,two = np.linalg.norm(d,1),np.linalg.norm(d)
        assert two <= one+1e-15
        assert one <= sqrt(len(d))*two+1e-15
        assert .5*two**2 >= one**2/(2*len(d))-1e-15
    first,second = np.array([2.,0.]),np.array([1.2,1.2])
    assert np.linalg.norm(second,1) > np.linalg.norm(first,1)
    assert np.linalg.norm(second) < np.linalg.norm(first)


def test_L1_optimum_has_one_over_T_projected_squared_distance_bound():
    responses = np.array([[2.,0.,0.],[1.1,1.1,1.1],[0.,-.7,1.9],[-.5,-.5,-.5]])
    chosen = responses[np.argmax(np.linalg.norm(responses,ord=1,axis=1))]
    best_l2_squared = max(np.sum(responses**2,axis=1))
    assert np.sum(chosen**2) >= best_l2_squared/responses.shape[1]-1e-15


def test_scalar_squared_affine_criterion_needs_only_two_linear_extrema():
    vertices = np.array([[0.,0.],[1.,0.],[0.,2.],[.5,1.5]])
    d = .3+vertices@np.array([.7,-.9])
    from_two_signs = max(max(d),max(-d))
    assert from_two_signs**2 == pytest.approx(max(d*d))


def test_predicted_L1_lower_bound_with_valid_action_error_penalty():
    nominal = np.array([1.,-.4,.2])
    eta = np.array([.2,.1,.3])
    lower = max(0.,np.linalg.norm(nominal,1)-eta.sum())
    for error in itertools.product(*zip(-eta,eta)):
        actual = nominal+error
        assert np.linalg.norm(actual,1) >= lower-1e-15
        assert .5*np.linalg.norm(actual)**2 >= lower**2/(2*len(actual))-1e-15


def test_temporal_pairing_cancels_fixed_bias_but_not_free_per_time_source():
    ua, ub = np.array([0.,1.]),np.array([0.,1.])
    slope_a,slope_b = 1.,2.
    source_a = np.array([.2,.5])
    fixed_bias = .7
    # Same physical source interface, but unconstrained changes can absorb slope.
    source_b = source_a+(slope_a-slope_b)*ua
    ya = slope_a*ua+source_a+fixed_bias
    yb = slope_b*ub+source_b+fixed_bias
    np.testing.assert_allclose(ya,yb,atol=1e-15)
    assert ya[1]-ya[0] == pytest.approx(slope_a+source_a[1]-source_a[0])
    assert ya[1]-ya[0] != pytest.approx(slope_a)
    # Independent unit noises add under differencing.
    difference = np.array([-1.,1.])
    assert difference@np.eye(2)@difference == 2.


def test_spatial_contrast_removes_independent_sources_at_each_time():
    g = np.array([1.,1.]);w = np.array([1.,-1.])/sqrt(2.)
    for source in (-2.,.3,5.):
        signal = np.array([.4,-.1])
        assert w@(signal+g*source) == pytest.approx(w@signal,abs=1e-15)


def _interval_gap(d, low, high):
    return max(0., d+low, -d-high)


@pytest.mark.parametrize('low,high', [(-.5,.4),(-2.,0.),(0.,0.),(-.1,3.)])
def test_scalar_source_interval_gap_is_exact_minimum_absolute_response(low,high):
    for d in np.linspace(-4.,4.,33):
        closest = np.clip(-d,low,high)
        assert _interval_gap(d,low,high) == pytest.approx(abs(d+closest),abs=1e-15)


@pytest.mark.parametrize('dlo,dhi,low,high', [
    (-3.,2.,-.5,.4),(-1.,4.,0.,0.),(1.,2.,-.3,.1),(-4.,-2.,-.2,.5)])
def test_three_branch_big_M_is_valid_and_forces_exact_max(dlo,dhi,low,high):
    upper = max(0.,dhi+low,-dlo-high)
    component_lower = np.array([0.,dlo+low,-dhi-high])
    M = upper-component_lower
    assert np.all(M>=0.)
    for d in np.linspace(dlo,dhi,31):
        pieces = np.array([0.,d+low,-d-high])
        true_gap = float(max(pieces))
        possible = []
        for selected in range(3):
            bits = np.eye(3)[selected]
            vmin = max(0.,max(pieces))
            vmax = min(upper,min(pieces+M*(1.-bits)))
            if vmin<=vmax+1e-14:
                assert vmin == pytest.approx(true_gap,abs=1e-14)
                assert vmax == pytest.approx(true_gap,abs=1e-14)
                possible.append(selected)
        assert possible


def test_interval_gap_bounds_frozen_full_output_distance_with_source_boxes():
    # Two periods, unit output contrasts; nuisance need not be projected away.
    residual0 = np.array([[1.,.2],[-.4,.6]])
    w = np.array([[1.,0.],[0.,1.]])
    nuisance = np.array([.3,.2])
    low,high = -.5*nuisance,.5*nuisance
    d = np.einsum('tm,tm->t',w,residual0)
    gaps = np.array([_interval_gap(d[t],low[t],high[t]) for t in range(2)])
    lower = .5*float(gaps@gaps)
    assert lower >= float(gaps.sum()**2)/(2*len(gaps))-1e-15
    for source in itertools.product((-.5,0.,.5),repeat=2):
        residual = residual0+w*(nuisance*np.asarray(source))[:,None]
        assert .5*np.sum(residual**2)>=lower-1e-15


def test_disjoint_scalar_mean_intervals_have_uniform_Gaussian_separator_margin():
    # Closest endpoints define a detector; any allowed mean is farther away.
    a = np.array([[1.,2.],[-1.,0.]])
    b = np.array([[-1.,0.],[1.,2.]])
    nearest_a = np.array([1.,0.])
    nearest_b = np.array([0.,1.])
    gap_vector = nearest_a-nearest_b
    direction = gap_vector/np.linalg.norm(gap_vector)
    threshold = direction@((nearest_a+nearest_b)/2.)
    margin = np.linalg.norm(gap_vector)/2.
    for mean in itertools.product(*a):
        assert direction@mean-threshold>=margin-1e-15
    for mean in itertools.product(*b):
        assert threshold-direction@mean>=margin-1e-15


def test_nominal_AC_contrast_with_frozen_source_support_is_not_AC_source_certificate():
    # At anchor u=0 source derivative vanishes, but source x control exists.
    # Frozen source support [0,0] predicts gap1 at u=1. Actual s can cancel it.
    anchor_source_derivative = 0.
    nominal_difference = lambda u: u
    actual_difference = lambda u,s:u+u*s
    frozen_gap = _interval_gap(nominal_difference(1.),0.,0.)
    assert anchor_source_derivative == 0.
    assert frozen_gap == 1.
    assert actual_difference(1.,-1.) == 0.


def test_scenario_min_uses_same_action_and_gives_only_finite_scenario_bound():
    # Two declared load scenarios favor different binary actions.
    actions = (-1.,0.,1.)
    scores = {}
    for u in actions:
        d = np.array([[1.+u,.2], [1.-u,.2]])
        gap = np.maximum(0.,np.abs(d)-.1)
        J = gap.sum(axis=1)
        z = float(J.min())
        scores[u] = z
        finite_model_distances = .5*np.sum(gap**2,axis=1)
        assert finite_model_distances.min()>=z*z/(2*gap.shape[1])-1e-15
    assert max(scores,key=scores.get) == 0.
    # A future load not represented by either scenario can have zero signal.
    omitted_actual_gap = np.zeros(2)
    assert np.sum(omitted_actual_gap) < scores[0.]


def test_scenario_source_interpretations_are_not_forced_to_share_values():
    # Independent per-scenario compensation differs from an extra common-source
    # restriction: min(s+1)^2 and min(s-1)^2 each zero, common s cannot do both.
    independent = min((s+1.)**2 for s in (-1.,0.,1.))+min((s-1.)**2 for s in (-1.,0.,1.))
    shared = min((s+1.)**2+(s-1.)**2 for s in (-1.,0.,1.))
    assert independent == 0.
    assert shared == 2.


def test_AC_nominal_gap_square_is_not_original_affine_proxy_lower_bound():
    # d_AC=u+u^2 has value/Jacobian 0/1 at the anchor0; d_affine=u.
    u = 1.
    affine_d, ac_d = u,u+u*u
    frozen_distance = .5*affine_d*affine_d
    AC_contrast_quantity = .5*ac_d*ac_d
    assert frozen_distance == .5
    assert AC_contrast_quantity == 2.
    assert AC_contrast_quantity > frozen_distance


def test_per_time_contrasts_avoid_cross_time_signed_cancellation():
    means = np.array([1.,-1.])
    single_time_average = np.ones(2)/sqrt(2.)
    assert single_time_average@means == 0.
    assert np.linalg.norm(means,1) == 2.
    assert np.linalg.norm(means)**2 == pytest.approx(2.)
