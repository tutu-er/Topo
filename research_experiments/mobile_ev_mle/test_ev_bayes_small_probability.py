"""Independent analytic/quadrature/counterexample checks for Gaussian EV risk."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm

from . import ev_bayes_small_probability as probability


def bank(means, covariance=None, prior=None, weights=None):
    means = np.asarray(means, float)
    L, K, d = means.shape
    return dict(means=means, covariances=(np.broadcast_to(np.eye(d), (L, K, d, d)).copy()
        if covariance is None else np.asarray(covariance, float)),
        prior=np.ones(K) / K if prior is None else np.asarray(prior, float),
        weights=np.ones(L) / L if weights is None else np.asarray(weights, float))


def test_unequal_covariance_normalizer_changes_posterior_at_identical_means():
    value = probability.gaussian_posterior([0.], [[0.], [0.]], [[[1.]], [[4.]]], [.5, .5])
    np.testing.assert_allclose(value["posterior"], [2 / 3, 1 / 3], atol=1e-14)
    assert value["log_predictive"] == pytest.approx(np.log(.5 * norm.pdf(0) + .5 * norm.pdf(0, scale=2)))


def test_finite_log_prior_retains_underflowed_candidate_and_future_evidence_can_restore_it():
    lp = np.array([0., -1000.])
    # exp(-1000) is zero in doubles.  At y=50, candidate 1 gains 1250 log
    # likelihood units and must recover, rather than being removed permanently.
    b = bank([[[0.], [50.]]], prior=[1., 0.]); b["log_prior"] = lp
    checked = probability.validate_bank(b)
    assert checked["prior_underflow_count"] == 1
    post = probability.gaussian_posterior([50.], [[0.], [50.]], [[[1.]], [[1.]]],
        [1., 0.], log_prior=lp)
    assert post["posterior"][1] > .999999
    assert np.isfinite(post["log_posterior"]).all()
    result = probability.evaluate_bank(b, np.zeros((1, 2, 4, 1)))
    assert result["stratum_count"] == 2 and len(result["strata"]) == 2
    assert result["prior_underflow_count"] == 1
    overlap = probability.bhattacharyya_bank(b)
    assert len(overlap["pairs"]) == 1
    assert overlap["pairs"][0]["coefficient"] > 0  # exp(-500), not sqrt(0)
    assert overlap["pairs"][0]["log_coefficient"] == -500


def test_multivariate_density_and_posterior_match_independent_scipy_density():
    means = np.array([[.1, -.2], [1., .4], [-.7, .6]])
    cov = np.array([[[1., .2], [.2, .6]], [[.3, -.1], [-.1, 1.2]], [[2., .5], [.5, .9]]])
    prior = np.array([.2, .3, .5]); y = np.array([[0., 0.], [1., -2.]])
    result = probability.gaussian_posterior(y, means, cov, prior)
    densities = np.stack([multivariate_normal.pdf(y, m, c) for m, c in zip(means, cov)], axis=-1)
    expected = densities * prior
    np.testing.assert_allclose(result["log_likelihood"], np.log(densities), atol=1e-13)
    np.testing.assert_allclose(result["posterior"], expected / expected.sum(-1, keepdims=True), atol=1e-14)


@pytest.mark.parametrize("mutation", ["zero_prior", "bad_weights", "nonsymmetric", "singular", "nonfinite"])
def test_invalid_banks_are_rejected_without_repair_or_candidate_removal(mutation):
    b = bank([[[0., 0.], [1., 1.]]])
    if mutation == "zero_prior": b["prior"] = np.array([0., 1.])
    if mutation == "bad_weights": b["weights"] = np.array([.9])
    if mutation == "nonsymmetric": b["covariances"][0, 0, 0, 1] = .1
    if mutation == "singular": b["covariances"][0, 0] = 0
    if mutation == "nonfinite": b["means"][0, 0, 0] = np.nan
    with pytest.raises(ValueError): probability.validate_bank(b)


def test_pair_overlap_matches_independent_quadrature_with_unequal_variances():
    b = bank([[[-.6], [1.3]]], [[[[.7]], [[2.1]]]], [.3, .7])
    result = probability.bhattacharyya_bank(b)
    overlap = quad(lambda y: np.sqrt(norm.pdf(y, -.6, np.sqrt(.7)) * norm.pdf(y, 1.3, np.sqrt(2.1))),
                   -20., 20., epsabs=1e-12)[0]
    assert result["pairs"][0]["overlap"] == pytest.approx(overlap, abs=2e-12)
    exact_risk = quad(lambda y: min(.3 * norm.pdf(y, -.6, np.sqrt(.7)),
                                   .7 * norm.pdf(y, 1.3, np.sqrt(2.1))), -20, 20, epsabs=1e-10)[0]
    assert result["bound"] >= exact_risk
    assert result["bound"] == pytest.approx(np.sqrt(.21) * overlap, abs=1e-12)


def test_multicandidate_bhattacharyya_bound_dominates_true_bayes_error():
    b = bank([[[-2.], [.2], [1.6]]], [[[[.5]], [[1.3]], [[.8]]]], [.2, .5, .3])
    def density(y):
        return b["prior"] * norm.pdf(y, b["means"][0, :, 0], np.sqrt(b["covariances"][0, :, 0, 0]))
    risk = quad(lambda y: density(y).sum() - density(y).max(), -20, 20, epsabs=1e-9)[0]
    upper = probability.bhattacharyya_bank(b)
    assert upper["bound"] >= risk
    assert upper["capped_bound"] >= risk
    assert len(upper["pairs"]) == 3


def test_binary_gaussian_risk_matches_analytic_correct_selection_probability():
    b = bank([[[-1.2], [1.2]]])
    noise = np.random.default_rng(601).normal(size=(1, 2, 30000, 1))
    result = probability.evaluate_bank(b, noise)
    exact = norm.cdf(-1.2)
    assert abs(result["risk"] - exact) < 5 * result["risk_se"] + 1e-4
    assert abs(result["map_error"] - exact) < 5 * result["map_error_se"] + 1e-4
    assert result["accuracy"] == pytest.approx(1 - result["risk"])


def test_input_scenario_is_known_and_never_mixed_as_an_unobserved_topology_signal():
    # Within each l the models coincide.  Swapping l between candidate labels
    # would create an artificial signal if the density summed over l incorrectly.
    b = bank([[[0.], [0.]], [[10.], [10.]]], prior=[.3, .7], weights=[.2, .8])
    result = probability.evaluate_bank(b, np.random.default_rng(7).normal(size=(2, 2, 50, 1)))
    assert result["risk"] == pytest.approx(.3, abs=1e-14)
    assert result["eig"] == pytest.approx(0., abs=1e-14)
    assert result["brier"] == pytest.approx(.58, abs=1e-14)
    assert result["abstention_risk"] == pytest.approx(.1, abs=1e-14)
    assert result["abstention_rate"] == pytest.approx(1.)
    np.testing.assert_allclose(result["posterior_mean"], [.3, .7], atol=1e-14)


def test_posterior_martingale_and_information_identities_on_unequal_covariances():
    b = bank([[[0.], [1.5], [-1.]]], [[[[1.]], [[.4]], [[2.]]]], [.2, .5, .3])
    result = probability.evaluate_bank(b, np.random.default_rng(91).normal(size=(1, 3, 22000, 1)))
    np.testing.assert_array_less(np.abs(result["posterior_mean"] - b["prior"]),
                                 5 * result["posterior_mean_se"] + 2e-4)
    assert abs(result["eig_identity_difference"]) < 5 * result["eig_identity_difference_se"] + 1e-4
    assert result["brier"] >= np.sum(b["prior"] ** 2) - 5 * result["brier_se"]


def test_paired_common_noise_se_uses_covariance_and_identical_action_has_zero_se():
    old = bank([[[-.5], [.5]], [[-.2], [.2]]], weights=[.25, .75], prior=[.3, .7])
    new = bank([[[-.8], [.8]], [[-.4], [.4]]], weights=[.25, .75], prior=[.3, .7])
    noise = np.random.default_rng(31).normal(size=(2, 2, 300, 1))
    a = probability.evaluate_bank(new, noise); b = probability.evaluate_bank(old, noise)
    paired = probability.compare_evaluations(a, b)
    delta = a["samples"]["risk"] - b["samples"]["risk"]
    weights = new["weights"][:, None] * new["prior"][None, :]
    exact_se = np.sqrt(np.sum(weights ** 2 * delta.var(axis=2, ddof=1)) / 300)
    assert paired["risk_se"] == pytest.approx(exact_se, abs=1e-15)
    assert paired["risk"] == pytest.approx(a["risk"] - b["risk"], abs=1e-15)
    assert paired["risk_se"] < np.sqrt(a["risk_se"] ** 2 + b["risk_se"] ** 2)
    same = probability.compare_banks(old, old, noise)
    assert same["risk"] == same["risk_se"] == same["eig"] == same["eig_se"] == 0.
    with pytest.raises(ValueError, match="same noise"):
        probability.compare_evaluations(a, probability.evaluate_bank(old, -noise))


def test_fixed_observation_gradient_and_total_path_gradient_are_distinct_and_correct():
    means = np.array([[-.4, .3], [.8, -.2]])
    covariance = np.array([[[.8, .1], [.1, 1.]], [[1.4, -.2], [-.2, .7]]])
    jac = np.array([[[.5, -.3], [.1, .2]], [[-.2, .1], [.7, -.1]]])
    y = np.array([[.2, -.1], [.7, .8]])
    result = probability.posterior_gradient(y, means, covariance, [.4, .6], jac)
    for p in range(2):
        h = 1e-6
        plus = probability.gaussian_posterior(y, means + h * jac[..., p], covariance, [.4, .6])
        minus = probability.gaussian_posterior(y, means - h * jac[..., p], covariance, [.4, .6])
        np.testing.assert_allclose(result["gradient"][..., p],
            (plus["posterior"] - minus["posterior"]) / (2 * h), atol=2e-10, rtol=1e-8)


def test_eig_risk_brier_and_abstention_path_gradients_match_fixed_noise_finite_differences():
    b = bank([[[-.6, .1], [.8, -.2], [.1, .9]], [[-.3, -.5], [.7, .3], [.8, -.6]]],
             prior=[.2, .5, .3], weights=[.4, .6])
    b["covariances"][:, 1] *= .7
    jac = np.random.default_rng(2).normal(scale=.2, size=(2, 3, 2, 2))
    noise = np.random.default_rng(9).normal(size=(2, 3, 150, 2))
    result = probability.evaluate_bank(b, noise, mean_jacobian=jac)
    assert all(result["gradient_valid"].values())
    for p in range(2):
        h = 1e-6
        plus = dict(b, means=b["means"] + h * jac[..., p])
        minus = dict(b, means=b["means"] - h * jac[..., p])
        a = probability.evaluate_bank(plus, noise); c = probability.evaluate_bank(minus, noise)
        for key in ("eig", "risk", "brier", "abstention_risk"):
            assert result["gradients"][key][p] == pytest.approx((a[key] - c[key]) / (2 * h), abs=3e-10)


def test_nonunique_map_gradient_is_reported_but_eig_gradient_remains_zero():
    b = bank([[[0.], [0.]]])
    noise = np.random.default_rng(8).normal(size=(1, 2, 30, 1))
    result = probability.evaluate_bank(b, noise, mean_jacobian=np.array([[[[1.]], [[-1.]]]]))
    assert result["map_tie_count"] == 60
    assert not result["gradient_valid"]["risk"]
    assert result["gradient_valid"]["eig"]
    np.testing.assert_allclose(result["gradients"]["eig"], 0, atol=1e-14)


def test_common_output_translation_is_not_information_even_with_unequal_covariances():
    b = bank([[[-.5, .1], [.6, -.3], [.2, .9]]], prior=[.2, .5, .3])
    b["covariances"][0, 1] *= .6
    noise = np.random.default_rng(803).normal(size=(1, 3, 50, 2))
    # Every candidate and the generated observation move together.  A fixed-y
    # derivative would give a spurious information price for this translation.
    jac = np.broadcast_to(np.array([[[[.7], [-.2]]]]), (1, 3, 2, 1))
    result = probability.evaluate_bank(b, noise, mean_jacobian=jac)
    for key in ("risk", "eig", "brier", "abstention_risk"):
        np.testing.assert_array_equal(result["gradients"][key], [0.])
    shifted = probability.evaluate_bank(dict(b, means=b["means"] + 10 * jac[..., 0]), noise)
    for key in ("risk", "eig", "brier", "abstention_risk"):
        assert shifted[key] == pytest.approx(result[key], abs=1e-14)


def test_candidate_and_scenario_relabeling_preserve_the_same_integral():
    b = bank([[[-.5], [.6], [1.2]], [[-.3], [.2], [.7]]],
             prior=[.2, .3, .5], weights=[.4, .6])
    noise = np.random.default_rng(804).normal(size=(2, 3, 30, 1))
    first = probability.evaluate_bank(b, noise)
    order = [2, 0, 1]
    permuted = dict(means=b["means"][::-1, order], covariances=b["covariances"][::-1, order],
                    prior=b["prior"][order], weights=b["weights"][::-1])
    second = probability.evaluate_bank(permuted, noise[::-1, order])
    for key in ("risk", "eig", "brier", "abstention_risk", "risk_se", "eig_se"):
        assert first[key] == pytest.approx(second[key], abs=1e-14)


@pytest.mark.parametrize("prior,cost,expected,abstain", [([.95, .05], .1, .05, 0),
    ([.5, .5], .1, .1, 1), ([.9, .1], .1, .1, 0)])
def test_abstention_loss_has_the_declared_conditional_decision_meaning(prior, cost, expected, abstain):
    b = bank([[[0.], [0.]]], prior=prior)
    result = probability.evaluate_bank(b, np.zeros((1, 2, 4, 1)), abstention_cost=cost)
    assert result["abstention_risk"] == pytest.approx(expected)
    assert result["abstention_rate"] == abstain


def test_single_candidate_boundary_has_zero_risk_and_zero_information():
    b = bank([[[2.]]])
    result = probability.evaluate_bank(b, np.zeros((1, 1, 4, 1)))
    assert result["risk"] == result["eig"] == 0
    assert result["brier"] == 1
    assert probability.bhattacharyya_bank(b)["bound"] == 0


def test_gaussian_input_conditioning_matches_known_posterior_and_counts_meter_error_once():
    result = probability.condition_gaussian_inputs([0.], [[4.]], [3.], [[1.]])
    np.testing.assert_allclose(result["mean"], [2.4])
    np.testing.assert_allclose(result["covariance"], [[.8]])
    assert result["log_input_density"] == pytest.approx(norm.logpdf(3, scale=np.sqrt(5)))
    assert result["meter_error_conditioned_once"]
    # Propagated conditional input error is .8*G^2, not (.8+1)*G^2.
    assert 2 ** 2 * result["covariance"][0, 0] == pytest.approx(3.2)
    deterministic = probability.condition_gaussian_inputs([2.], [[0.]], [[1.], [3.]], [[1.]])
    np.testing.assert_allclose(deterministic["mean"], [[2.], [2.]])
    np.testing.assert_allclose(deterministic["covariance"], [[0.]])


def test_multivariate_input_conditioning_matches_joint_gaussian_schur_complement():
    mean = np.array([1., -.5]); C = np.array([[2., .4], [.4, 1.]])
    R = np.array([[.3, -.1], [-.1, .6]]); y = np.array([.2, .7])
    result = probability.condition_gaussian_inputs(mean, C, y, R)
    expected_mean = mean + C @ np.linalg.solve(C + R, y - mean)
    expected_cov = C - C @ np.linalg.solve(C + R, C)
    np.testing.assert_allclose(result["mean"], expected_mean, atol=1e-14)
    np.testing.assert_allclose(result["covariance"], expected_cov, atol=1e-14)
    assert result["log_input_density"] == pytest.approx(multivariate_normal.logpdf(y, mean, C + R))


@pytest.mark.parametrize("kwargs", [dict(prior_cov=[[-1.]]), dict(measurement_cov=[[0.]]),
    dict(observed=[np.nan]), dict(observed=[1., 2.])])
def test_invalid_input_conditioning_is_rejected(kwargs):
    args = dict(prior_mean=[0.], prior_cov=[[1.]], observed=[1.], measurement_cov=[[.5]])
    args.update(kwargs)
    with pytest.raises(ValueError): probability.condition_gaussian_inputs(**args)
