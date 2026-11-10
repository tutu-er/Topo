"""Small analytic model checks, with no archived load, truth, or future data."""
import numpy as np
import pytest
from scipy.special import logit

try:
    from .core import (QuadraticEvidence, build_expected, build_separation,
                       posterior_pair, separation_threshold, event_probability)
except ImportError:
    from core import (QuadraticEvidence, build_expected, build_separation,
                      posterior_pair, separation_threshold, event_probability)


def fixture():
    bc = np.array([[1., -.4], [.2, .8], [-.3, 1.2]])
    bf = np.array([[.5, -.1], [-.3, 1.], [.4, .6]])
    gc = np.array([[.7, -.2], [.1, .8]])
    gf = np.array([[.3, .4], [-.1, .5]])
    sd = np.array([[.8, 1.1], [.9, .7], [1.2, 1.]])
    u = np.array([[.2, -.1, .4], [-.3, .6, .2]])
    return bc, bf, gc, gf, sd, u


def test_expected_quadratic_matches_direct_weighted_squares():
    bc, bf, gc, gf, sd, u = fixture()
    model = build_expected(bc, bf, gc, gf, sd)
    rc = bc + (gc @ u).T
    rf = bf + (gf @ u).T
    expected = .5*np.sum((rc**2-rf**2)/sd**2)
    assert model.kind == "expected_log_bayes_factor"
    assert model.shape == u.shape
    assert model.value(u) == pytest.approx(expected, abs=1e-13)
    assert posterior_pair(rc, rf, sd)["log_bayes_factor"] == pytest.approx(expected)
    assert model.scaled(-1).value(u) == pytest.approx(-expected)
    np.testing.assert_allclose(model.scaled(-1).gradient(u), -model.gradient(u))


def test_nominal_separation_matches_direct_squared_mean_difference():
    bc, bf, gc, gf, sd, u = fixture()
    model = build_separation(bc, bf, gc, gf, sd)
    difference = bc-bf+((gc-gf)@u).T
    assert model.kind == "nominal_mahalanobis_separation"
    assert model.value(u) == pytest.approx(np.sum(difference**2/sd**2), abs=1e-13)
    assert np.linalg.eigvalsh(model.Q).min() >= -1e-13


def test_quadratic_gradient_matches_independent_central_difference():
    bc, bf, gc, gf, sd, u = fixture()
    for builder in (build_expected, build_separation):
        model = builder(bc, bf, gc, gf, sd)
        numerical = np.zeros_like(u)
        step = 1e-6
        for index in np.ndindex(u.shape):
            up, down = u.copy(), u.copy()
            up[index] += step
            down[index] -= step
            numerical[index] = (model.value(up)-model.value(down))/(2*step)
        np.testing.assert_allclose(model.gradient(u), numerical, atol=3e-9, rtol=3e-9)


def test_identical_models_have_zero_evidence_and_separation():
    bc, _, gc, _, sd, u = fixture()
    for builder in (build_expected, build_separation):
        model = builder(bc, bc, gc, gc, sd)
        assert model.value(u) == 0.
        np.testing.assert_array_equal(model.gradient(u), np.zeros_like(u))


def test_expectation_uses_response_variance_with_shared_response_innovations():
    bc, bf, gc, gf, sd, u = fixture()
    bc, bf, sd, u = bc[:1], bf[:1], sd[:1], u[:, :1]
    relative = np.array([[.25], [.4]])
    model = build_expected(bc, bf, gc, gf, sd, response_relative_sd=relative)
    nodes, weights = np.polynomial.hermite.hermgauss(5)
    integrated = 0.
    # Tensor Gauss-Hermite is exact for this quadratic in two Gaussian variables.
    for i in range(nodes.size):
        for j in range(nodes.size):
            z = np.sqrt(2)*np.array([[nodes[i]], [nodes[j]]])
            actual = u*(1+relative*z)
            rc, rf = bc+(gc@actual).T, bf+(gf@actual).T
            integrated += weights[i]*weights[j]/np.pi*.5*np.sum((rc**2-rf**2)/sd**2)
    assert model.value(u) == pytest.approx(integrated, abs=1e-13)
    assert abs(model.value(u)-build_expected(bc,bf,gc,gf,sd).value(u)) > 1e-5


def test_background_second_moments_and_shared_voltage_variance_cancel():
    bc, bf, gc, gf, sd, u = fixture()
    vc, vf = np.full_like(bc,.4), np.full_like(bf,.1)
    model = build_expected(bc,bf,gc,gf,sd,
        background_residual_variance_C=vc, background_residual_variance_F=vf)
    mean_only = build_expected(bc,bf,gc,gf,sd)
    assert model.value(u)-mean_only.value(u) == pytest.approx(.5*np.sum((vc-vf)/sd**2))
    extra = np.arange(bc.size).reshape(bc.shape)*.2
    same = build_expected(bc,bf,gc,gf,sd,
        background_residual_variance_C=vc+extra,
        background_residual_variance_F=vf+extra)
    assert same.value(u) == pytest.approx(model.value(u), abs=1e-13)


def test_expected_quadratic_can_be_indefinite_or_negative():
    model = build_expected([[0.]], [[0.]], [[0.]], [[1.]], [[1.]])
    assert model.Q[0,0,0] == -.5
    assert model.value([[2.]]) == -2.


def test_pair_posterior_keeps_prior_history_and_batch_axes():
    bc, bf, gc, gf, sd, u = fixture()
    rc, rf = bc+(gc@u).T, bf+(gf@u).T
    result = posterior_pair(np.stack([rc,rf]), np.stack([rf,rc]), sd,
                            prior_log_odds=.3, history_log_bayes_factor=-.2)
    assert result["log_odds"].shape == (2,)
    assert result["log_odds"][0]+result["log_odds"][1] == pytest.approx(.2)
    np.testing.assert_allclose(result["posterior_F"]+result["posterior_C"], 1.)
    swapped = posterior_pair(rf,rc,sd,prior_log_odds=-.3,history_log_bayes_factor=.2)
    assert swapped["posterior_F"] == pytest.approx(result["posterior_C"][0])


def test_posterior_pair_only_normalizes_two_candidates_and_is_stable():
    zero = np.zeros((2,3))
    assert posterior_pair(zero,zero,1.)["posterior_F"] == .5
    # Three identical candidates would have posterior 1/3; this pair API is not that model.
    assert posterior_pair(zero,zero,1.,prior_log_odds=1000.)["posterior_F"] == 1.
    assert posterior_pair(zero,zero,1.,prior_log_odds=-1000.)["posterior_C"] == 1.


def test_expected_log_odds_threshold_alone_gives_only_half_event_probability():
    eta = .9
    expected_only_separation = 2*logit(eta)
    assert event_probability(expected_only_separation,eta,state="F") == pytest.approx(.5)
    assert event_probability(expected_only_separation,eta,state="C") == pytest.approx(.5)


def test_exact_threshold_controls_both_states_with_asymmetric_prior_history():
    for eta,beta,initial in ((.9,.9,0.),(.95,.975,1.4),(.8,.2,-.6)):
        separation = separation_threshold(eta,beta,initial)
        p_f = event_probability(separation,eta,initial,"F")
        p_c = event_probability(separation,eta,initial,"C")
        assert min(p_f,p_c) == pytest.approx(beta,abs=2e-14)
        assert event_probability(separation*.99,eta,initial,
            "C" if initial>=0 else "F") < beta


def test_zero_separation_uses_the_deterministic_posterior_not_a_normal_limit():
    eta = .9
    boundary = float(logit(eta))
    assert event_probability(0.,eta,boundary,"F") == 1.
    assert event_probability(0.,eta,boundary,"C") == 0.
    assert event_probability(0.,eta,0.,"F") == 0.
    assert event_probability(0.,eta,-boundary,"C") == 1.
    p = event_probability(np.array([0.,separation_threshold(.9,.9)]),.9,state="F")
    np.testing.assert_allclose(p,[0.,.9])


def test_log_bayes_factor_gaussian_moments_match_exact_model():
    separation = 6.
    nodes, weights = np.polynomial.hermite.hermgauss(7)
    z = np.sqrt(2)*nodes
    rc = (np.sqrt(separation)+z)[:,None,None]
    rf = z[:,None,None]
    odds = posterior_pair(rc,rf,[[1.]])["log_odds"]
    weight = weights/np.sqrt(np.pi)
    assert weight@odds == pytest.approx(separation/2,abs=1e-13)
    assert weight@((odds-separation/2)**2) == pytest.approx(separation,abs=1e-13)


def test_input_contract_rejects_unsupported_or_invalid_values():
    bc,bf,gc,gf,sd,u = fixture()
    with pytest.raises(ValueError):
        build_expected(bc,bf,gc,gf,np.zeros_like(sd))
    with pytest.raises(ValueError):
        build_expected(bc,bf,gc,gf,sd,response_relative_sd=-.1)
    with pytest.raises(ValueError):
        build_expected(bc,bf,gc,gf,sd,background_residual_variance_C=-1.)
    with pytest.raises(ValueError):
        build_separation(bc,bf,gc,gf,sd).value(u.T)
    with pytest.raises(ValueError):
        QuadraticEvidence(np.zeros((1,2,2)),np.zeros((2,1)),0.,"wrong")
    with pytest.raises(ValueError):
        event_probability(-1.,.9)
    with pytest.raises(ValueError):
        event_probability(1.,.9,state="other")
    with pytest.raises(ValueError):
        separation_threshold(.5,.9)
    with pytest.raises(ValueError):
        separation_threshold(.9,1.)
    with pytest.raises(ValueError):
        posterior_pair(np.zeros((2,2)),np.zeros((2,3)),1.)
