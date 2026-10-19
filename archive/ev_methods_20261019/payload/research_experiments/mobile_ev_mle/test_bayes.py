"""Independent density/quadrature checks for finite Bayesian EV design."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

try:
    from .bayes import posterior_weights, predictive_logpdf, eig_scores
except ImportError:
    from bayes import posterior_weights, predictive_logpdf, eig_scores


def binary_information_quadrature(gamma):
    """Independent one-dimensional Gaussian-channel entropy integral."""
    return quad(lambda z: norm.pdf(z) * (
        np.log(2.0) - np.logaddexp(0.0, -2.0 * gamma * (gamma + z))),
        -12.0, 12.0, epsabs=1e-11, epsrel=1e-11)[0]


def binary_posterior(prior=(0.5, 0.5)):
    return posterior_weights([0.0], [[0.0], [0.0]], [0, 1], prior)


def test_posterior_is_full_gaussian_density_and_has_no_implicit_topology_prior():
    y = np.array([0.2, -0.4])
    means = np.array([[0.0, 0.0], [1.0, -0.5], [-0.4, 0.1]])
    prior = np.array([0.1, 0.2, 0.7])
    labels = np.array([2, 2, 9])
    posterior = posterior_weights(y, means, labels, prior)
    likelihood = np.prod(norm.pdf(y[None, :], loc=means), axis=1)
    evidence = np.dot(prior, likelihood)
    expected = prior * likelihood / evidence
    np.testing.assert_allclose(posterior.weights, expected, rtol=1e-13)
    np.testing.assert_allclose(posterior.topology_probabilities,
                               [expected[:2].sum(), expected[2]], rtol=1e-13)
    assert posterior.log_evidence == pytest.approx(np.log(evidence), abs=1e-13)
    corrected = posterior_weights(y, means, labels, prior,
                                  log_whitening_jacobian=3.0)
    np.testing.assert_array_equal(corrected.log_weights, posterior.log_weights)
    assert corrected.log_evidence == pytest.approx(posterior.log_evidence + 3.0)
    assert posterior.summary()["historical_screening"] is False


@pytest.mark.parametrize("prior", [[0.0, 1.0], [-0.1, 1.1], [0.2, 0.2],
                                    [np.nan, 1.0], [1.0]])
def test_requires_a_positive_normalized_external_atom_prior(prior):
    with pytest.raises(ValueError, match="proper_prior"):
        posterior_weights([0.0], [[0.0], [1.0]], [0, 1], prior)


def test_input_dimensions_and_whitening_are_explicit():
    with pytest.raises(ValueError, match="history_y"):
        posterior_weights([0.0, 1.0], [[0.0], [1.0]], [0, 1], [0.5, 0.5])
    with pytest.raises(ValueError, match="integer"):
        posterior_weights([0.0], [[0.0], [1.0]], [0.0, 1.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="finite"):
        posterior_weights([0.0], [[np.nan], [1.0]], [0, 1], [0.5, 0.5])
    with pytest.raises(ValueError, match="log_whitening_jacobian"):
        posterior_weights([0.0], [[0.0], [1.0]], [0, 1], [0.5, 0.5],
                          log_whitening_jacobian=np.inf)
    posterior = binary_posterior()
    with pytest.raises(ValueError, match="same future dimension"):
        eig_scores(posterior, {"a": np.zeros((2, 1)), "b": np.zeros((2, 2))})
    with pytest.raises(ValueError, match="samples_per_topology"):
        eig_scores(posterior, {"a": np.zeros((2, 1))}, samples_per_topology=1)


def test_log_weights_retain_a_tiny_topology_without_a_historical_cutoff():
    posterior = posterior_weights([0.0], [[0.0], [40.0]], [0, 1], [0.5, 0.5])
    assert np.isfinite(posterior.log_weights).all()
    assert posterior.log_weights[1] == pytest.approx(-800.0)
    assert posterior.log_topology_probabilities[1] == pytest.approx(-800.0)
    means = np.array([[0.0], [10.0]])
    assert predictive_logpdf(posterior, means, [10.0], topology=1) == pytest.approx(
        norm.logpdf(0.0), abs=1e-12)
    score = eig_scores(posterior, {"a": means}, samples_per_topology=128, seed=17)["a"]
    assert [row["sample_count"] for row in score["strata"]] == [128, 128]
    assert np.isfinite(score["eig"])


def test_mixture_normalization_and_atom_permutation_and_split_invariance():
    posterior = posterior_weights([0.3], [[0.0], [0.5], [-0.5]],
                                  [0, 0, 1], [0.2, 0.3, 0.5])
    means = np.array([[-1.0], [0.7], [2.0]])
    for topology in (None, 0, 1):
        integral = quad(lambda y: np.exp(predictive_logpdf(
            posterior, means, [y], topology=topology)), -12.0, 14.0,
            epsabs=1e-10)[0]
        assert integral == pytest.approx(1.0, abs=1e-9)
    perm = np.array([2, 0, 1])
    reordered = posterior_weights([0.3], np.array([[0.0], [0.5], [-0.5]])[perm],
                                  np.array([0, 0, 1])[perm], np.array([0.2, 0.3, 0.5])[perm])
    grid = np.linspace(-6.0, 6.0, 31)[:, None]
    np.testing.assert_allclose(predictive_logpdf(posterior, means, grid),
                               predictive_logpdf(reordered, means[perm], grid), atol=1e-13)
    split = posterior_weights([0.3], [[0.0], [0.0], [0.5], [-0.5]],
                              [0, 0, 0, 1], [0.07, 0.13, 0.3, 0.5])
    split_means = np.array([[-1.0], [-1.0], [0.7], [2.0]])
    np.testing.assert_allclose(posterior.topology_probabilities,
                               split.topology_probabilities, atol=1e-13)
    np.testing.assert_allclose(predictive_logpdf(posterior, means, grid),
                               predictive_logpdf(split, split_means, grid), atol=1e-13)


def test_identical_model_mixtures_have_zero_topology_information_but_line_information():
    posterior = posterior_weights([0.0], np.zeros((4, 1)), [0, 0, 1, 1],
                                  [0.25, 0.25, 0.25, 0.25])
    means = np.array([[-2.0], [2.0], [-2.0], [2.0]])
    score = eig_scores(posterior, {"same": means}, samples_per_topology=128, seed=20)["same"]
    assert abs(score["eig"]) < 1e-13
    assert abs(score["entropy_reduction"]) < 1e-13
    # The two line-state means are distinguishable even though h is not.
    assert binary_information_quadrature(2.0) > 0.5
    # Marginalizing unknown line state is neither plug-in N(0,1) nor N(0,5).
    density = np.exp(predictive_logpdf(posterior, means, [0.0], topology=0))
    assert density == pytest.approx(norm.pdf(2.0), abs=1e-13)
    assert abs(density - norm.pdf(0.0)) > 0.1
    assert abs(density - norm.pdf(0.0, scale=np.sqrt(5.0))) > 0.1


def test_one_topology_has_zero_model_information_even_with_unknown_atoms():
    posterior = posterior_weights([0.0], [[0.0], [0.0]], [7, 7], [0.2, 0.8])
    score = eig_scores(posterior, {"one": np.array([[-2.0], [2.0]])},
                       samples_per_topology=128, seed=8)["one"]
    assert score["entropy_upper_bound"] == 0.0
    assert score["eig"] == 0.0
    assert score["eig_se"] == 0.0
    assert score["entropy_reduction"] == 0.0


def test_binary_gaussian_quadrature_and_entropy_identity_at_frozen_budgets():
    posterior = binary_posterior()
    gamma = 1.0
    exact = binary_information_quadrature(gamma)
    assert 0.0 < exact < np.log(2.0)
    future = {"a": np.array([[-gamma], [gamma]])}
    errors = []
    for samples in (128, 512, 2048):
        for seed in (201, 202, 203):
            score = eig_scores(posterior, future, samples_per_topology=samples, seed=seed)["a"]
            assert abs(score["eig"] - exact) < 6.0 * score["eig_se"] + 1e-4
            assert abs(score["entropy_reduction"] - exact) < 6.0 * score["entropy_se"] + 1e-4
            assert abs(score["identity_difference"]) < (
                6.0 * score["identity_difference_se"] + 1e-4)
            assert score["eig"] <= score["entropy_upper_bound"] + 1e-12
            if seed == 201:
                errors.append(score["eig_se"])
    assert errors[-1] < errors[0]


def test_stratified_standard_error_matches_independent_gaussian_integrals():
    probabilities = np.array([0.3, 0.7])
    means = np.array([-1.0, 1.0])
    posterior = binary_posterior(probabilities)
    analytic_means, analytic_variances = [], []
    for h in range(2):
        def information(z):
            y = means[h] + z
            mix = np.logaddexp(np.log(probabilities[0]) + norm.logpdf(y, means[0]),
                               np.log(probabilities[1]) + norm.logpdf(y, means[1]))
            return norm.logpdf(z) - mix
        average = quad(lambda z: norm.pdf(z) * information(z), -12.0, 12.0,
                       epsabs=1e-11)[0]
        second = quad(lambda z: norm.pdf(z) * information(z) ** 2, -12.0, 12.0,
                      epsabs=1e-11)[0]
        analytic_means.append(average)
        analytic_variances.append(second - average ** 2)
    samples = 8192
    exact = np.dot(probabilities, analytic_means)
    exact_se = np.sqrt(np.dot(probabilities ** 2, analytic_variances) / samples)
    score = eig_scores(posterior, {"a": means[:, None]},
                       samples_per_topology=samples, seed=33)["a"]
    assert abs(score["eig"] - exact) < 5.0 * exact_se
    assert score["eig_se"] == pytest.approx(exact_se, rel=0.08)


def test_common_noise_is_action_order_and_batch_size_independent():
    posterior = binary_posterior()
    future = {"b": np.array([[-1.0], [1.0]]), "a": np.array([[-0.5], [0.5]])}
    first = eig_scores(posterior, future, samples_per_topology=128, seed=44, batch_size=17)
    second = eig_scores(posterior, dict(reversed(list(future.items()))),
                        samples_per_topology=128, seed=44, batch_size=256)
    for action in future:
        for key in ("eig", "eig_se", "entropy_reduction", "entropy_se",
                    "identity_difference", "identity_difference_se"):
            assert first[action][key] == pytest.approx(second[action][key], abs=1e-14)
    doubled = eig_scores(posterior, {"a": future["a"], "copy": future["a"]},
                         samples_per_topology=128, seed=44)
    assert doubled["a"] == doubled["copy"]


def test_small_negative_mc_estimates_are_retained():
    posterior = binary_posterior()
    future = {"weak": np.array([[-0.02], [0.02]])}
    scores = [eig_scores(posterior, future, samples_per_topology=2, seed=seed)["weak"]
              for seed in range(16)]
    assert any(score["eig"] < 0.0 for score in scores)
    assert all(score["clipped"] is False for score in scores)


def test_analytic_line_uncertainty_reduces_model_information():
    # Gaussian line coefficient uncertainty inflates s_a, with gamma=c0*a/s_a.
    plugin = binary_information_quadrature(1.0)
    marginalized = binary_information_quadrature(1.0 / np.sqrt(2.0))
    assert 0.0 < marginalized < plugin < np.log(2.0)
    assert binary_information_quadrature(0.0) == pytest.approx(0.0, abs=1e-14)


def test_paired_action_standard_error_retains_common_noise_covariance():
    posterior = binary_posterior()
    future = {"weak": np.array([[-0.5], [0.5]]),
              "strong": np.array([[-1.0], [1.0]])}
    def difference(z):
        return (np.logaddexp(0.0, -(0.5 + z))
                - np.logaddexp(0.0, -2.0 * (1.0 + z)))
    average = quad(lambda z: norm.pdf(z) * difference(z), -12.0, 12.0,
                   epsabs=1e-11)[0]
    second = quad(lambda z: norm.pdf(z) * difference(z) ** 2, -12.0, 12.0,
                  epsabs=1e-11)[0]
    samples = 8192
    exact_se = np.sqrt(0.5 * (second - average ** 2) / samples)
    scores = eig_scores(posterior, future, samples_per_topology=samples, seed=330)
    comparison = scores["strong"]["paired_comparisons"]["weak"]
    assert average == pytest.approx(binary_information_quadrature(1.0)
                                    - binary_information_quadrature(0.5), abs=1e-10)
    assert abs(comparison["eig_gain"] - average) < 5.0 * exact_se
    assert comparison["eig_gain_se"] == pytest.approx(exact_se, rel=0.08)
    inverse = scores["weak"]["paired_comparisons"]["strong"]
    assert inverse["eig_gain"] == pytest.approx(-comparison["eig_gain"], abs=1e-14)
    assert inverse["eig_gain_se"] == comparison["eig_gain_se"]
    self_comparison = scores["weak"]["paired_comparisons"]["weak"]
    assert self_comparison["eig_gain"] == self_comparison["eig_gain_se"] == 0.0
