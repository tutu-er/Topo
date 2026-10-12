"""Independent quadrature and variance checks for posterior-KL EIG."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import norm

try:
    from .bayes import posterior_weights, eig_scores
    from .bayes_rb import pointwise_topology_kl, rb_eig_scores
except ImportError:
    from bayes import posterior_weights, eig_scores
    from bayes_rb import pointwise_topology_kl, rb_eig_scores


def posterior(prior=(0.5, 0.5)):
    return posterior_weights([0.0], [[0.0], [0.0]], [0, 1], prior)


def gaussian_statistics(z, h, probabilities, gamma):
    """Direct scalar densities, independently of the finite-mixture module."""
    means = np.array([-gamma, gamma])
    observation = means[h] + z
    log_densities = norm.logpdf(observation, loc=means)
    log_joint = np.log(probabilities) + log_densities
    log_density = np.logaddexp(log_joint[0], log_joint[1])
    log_updated = log_joint - log_density
    updated = np.exp(log_updated)
    f = norm.logpdf(z) - log_density
    rb = np.dot(updated, log_updated - np.log(probabilities))
    entropy = -np.dot(probabilities, np.log(probabilities)) + np.dot(updated, log_updated)
    return np.array([f, rb, entropy])


def gaussian_integrals(probabilities, gamma):
    probabilities = np.asarray(probabilities)
    averages = np.empty((2, 3))
    variances = np.empty((2, 3))
    for h in range(2):
        for k in range(3):
            average = quad(lambda z: norm.pdf(z) * gaussian_statistics(
                z, h, probabilities, gamma)[k], -12.0, 12.0,
                epsabs=1e-10)[0]
            second = quad(lambda z: norm.pdf(z) * gaussian_statistics(
                z, h, probabilities, gamma)[k] ** 2, -12.0, 12.0,
                epsabs=1e-10)[0]
            averages[h, k] = average
            variances[h, k] = second - average ** 2
    return probabilities @ averages, averages, variances


def test_pointwise_kl_matches_bayes_rule_and_can_exceed_prior_entropy():
    probabilities = np.array([0.2, 0.8])
    post = posterior(probabilities)
    means = np.array([[-0.8], [0.8]])
    observations = np.array([-3.0, -1.0, 0.0, 2.0])
    expected = []
    for y in observations:
        densities = norm.pdf(y, loc=means[:, 0])
        updated = probabilities * densities / np.dot(probabilities, densities)
        expected.append(np.dot(updated, np.log(updated / probabilities)))
    actual = pointwise_topology_kl(post, means, observations[:, None])
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    assert np.min(actual) >= 0.0
    assert actual[0] > post.topology_entropy


def test_pointwise_information_is_invariant_to_topology_label_permutation():
    post = posterior((0.2, 0.8))
    changed = posterior_weights([0.0], [[0.0], [0.0]], [9, 2], [0.2, 0.8])
    means = np.array([[-1.0], [1.0]])
    grid = np.linspace(-3.0, 3.0, 25)[:, None]
    np.testing.assert_allclose(pointwise_topology_kl(post, means, grid),
                               pointwise_topology_kl(changed, means, grid), atol=1e-13)


def test_identical_predictive_mixtures_have_zero_information_for_unequal_priors():
    post = posterior_weights([0.0], np.zeros((4, 1)), [0, 0, 1, 1],
                             [0.05, 0.15, 0.2, 0.6])
    means = np.array([[-2.0], [2.0], [-2.0], [2.0]])
    grid = np.linspace(-5.0, 5.0, 37)[:, None]
    values = pointwise_topology_kl(post, means, grid)
    assert np.max(np.abs(values)) < 1e-13
    score = rb_eig_scores(post, {"a": means}, samples_per_topology=128, seed=10)["a"]
    assert abs(score["eig"]) < 1e-13
    assert abs(score["entropy_reduction"]) < 1e-13
    assert score["clipped"] is False


def test_one_topology_has_zero_information_with_multiple_line_states():
    post = posterior_weights([0.0], [[0.0], [0.0]], [7, 7], [0.2, 0.8])
    means = np.array([[-2.0], [2.0]])
    score = rb_eig_scores(post, {"a": means}, samples_per_topology=128, seed=11)["a"]
    assert score["eig"] == score["eig_se"] == score["entropy_upper_bound"] == 0.0


@pytest.mark.parametrize("probabilities", [(0.5, 0.5), (0.2, 0.8)])
def test_rb_mean_and_stratified_se_against_independent_binary_quadrature(probabilities):
    probabilities = np.asarray(probabilities)
    exact, _, variances = gaussian_integrals(probabilities, gamma=0.8)
    np.testing.assert_allclose(exact, exact[0], atol=1e-10)
    post = posterior(probabilities)
    future = {"a": np.array([[-0.8], [0.8]])}
    samples = 4096
    exact_se = np.sqrt(np.dot(probabilities ** 2, variances[:, 1]) / samples)
    for seed in (871, 872, 873):
        score = rb_eig_scores(post, future, samples_per_topology=samples, seed=seed)["a"]
        assert abs(score["eig"] - exact[0]) < 5.0 * exact_se
        assert score["eig_se"] == pytest.approx(exact_se, rel=0.10)
        assert abs(score["identity_difference"]) < (
            5.0 * score["identity_difference_se"] + 1e-12)
        assert 0.0 < score["eig"] < score["entropy_upper_bound"]
        if probabilities[0] == 0.5:
            assert score["eig"] == pytest.approx(score["entropy_reduction"], abs=1e-13)
            assert score["eig_se"] == pytest.approx(score["entropy_se"], abs=1e-13)


def test_iid_rao_blackwell_variance_identity_by_quadrature():
    probabilities = np.array([0.2, 0.8])
    gamma = 0.8
    exact, averages, variances = gaussian_integrals(probabilities, gamma)
    mixture_variance = probabilities @ (variances + averages ** 2) - exact ** 2
    def conditional_variance_term(y):
        log_densities = norm.logpdf(y, loc=[-gamma, gamma])
        log_joint = np.log(probabilities) + log_densities
        log_density = logsumexp(log_joint)
        updated = np.exp(log_joint - log_density)
        ratios = log_densities - log_density
        average = np.dot(updated, ratios)
        variance = np.dot(updated, (ratios - average) ** 2)
        return np.exp(log_density) * variance
    lost_variance = quad(conditional_variance_term, -12.0, 12.0, epsabs=1e-10)[0]
    assert lost_variance > 0.0
    assert mixture_variance[0] - mixture_variance[1] == pytest.approx(
        lost_variance, abs=1e-9)


def test_measured_variance_reduction_on_symmetric_gaussian_toy():
    # For this symmetric equal-allocation toy the conditional distributions of
    # F and G are the same in both strata, so the iid variance order carries over.
    post = posterior()
    future = {"a": np.array([[-0.5], [0.5]])}
    ordinary, rb = [], []
    for seed in range(5000, 5032):
        ordinary.append(eig_scores(post, future, samples_per_topology=128, seed=seed)["a"]["eig"])
        rb.append(rb_eig_scores(post, future, samples_per_topology=128, seed=seed)["a"]["eig"])
    assert np.var(rb, ddof=1) < 0.5 * np.var(ordinary, ddof=1)


def test_paired_rb_action_error_against_joint_common_noise_quadrature():
    probabilities = np.array([0.2, 0.8])
    post = posterior(probabilities)
    future = {"weak": np.array([[-0.25], [0.25]]),
              "strong": np.array([[-0.5], [0.5]])}
    conditional_means, conditional_variances = [], []
    for h in range(2):
        def difference(z):
            return (gaussian_statistics(z, h, probabilities, 0.5)[1]
                    - gaussian_statistics(z, h, probabilities, 0.25)[1])
        average = quad(lambda z: norm.pdf(z) * difference(z), -12.0, 12.0,
                       epsabs=1e-10)[0]
        second = quad(lambda z: norm.pdf(z) * difference(z) ** 2, -12.0, 12.0,
                      epsabs=1e-10)[0]
        conditional_means.append(average)
        conditional_variances.append(second - average ** 2)
    samples = 8192
    exact_gain = np.dot(probabilities, conditional_means)
    exact_se = np.sqrt(np.dot(probabilities ** 2, conditional_variances) / samples)
    scores = rb_eig_scores(post, future, samples_per_topology=samples, seed=777)
    comparison = scores["strong"]["paired_comparisons"]["weak"]
    assert abs(comparison["eig_gain"] - exact_gain) < 5.0 * exact_se
    assert comparison["eig_gain_se"] == pytest.approx(exact_se, rel=0.10)
    inverse = scores["weak"]["paired_comparisons"]["strong"]
    assert inverse["eig_gain"] == -comparison["eig_gain"]
    assert inverse["eig_gain_se"] == comparison["eig_gain_se"]


def test_action_order_and_batch_size_invariance_with_tiny_mass_retained():
    post = posterior_weights([0.0], [[0.0], [40.0]], [0, 1], [0.5, 0.5])
    future = {"a": np.array([[0.0], [2.0]]), "b": np.array([[0.0], [3.0]])}
    first = rb_eig_scores(post, future, samples_per_topology=128, seed=12, batch_size=17)
    second = rb_eig_scores(post, dict(reversed(list(future.items()))),
                           samples_per_topology=128, seed=12, batch_size=256)
    assert [r["sample_count"] for r in first["a"]["strata"]] == [128, 128]
    for action in future:
        for key in ("eig", "eig_se", "entropy_reduction", "entropy_se"):
            assert first[action][key] == pytest.approx(second[action][key], abs=1e-13)
    assert post.log_topology_probabilities[1] == pytest.approx(-800.0)


def test_checks_future_input_dimension():
    post = posterior()
    with pytest.raises(ValueError, match="same future dimension"):
        rb_eig_scores(post, {"a": np.zeros((2, 1)), "b": np.zeros((2, 2))})
    with pytest.raises(ValueError, match="samples_per_topology"):
        rb_eig_scores(post, {"a": np.zeros((2, 1))}, samples_per_topology=1)
    with pytest.raises(ValueError, match="values"):
        pointwise_topology_kl(post, np.zeros((2, 1)), [np.nan])


def test_signed_roundoff_is_retained_for_identical_three_model_predictions():
    post = posterior_weights([0.0], np.zeros((3, 1)), [0, 1, 2], [0.1, 0.2, 0.7])
    value = pointwise_topology_kl(post, np.zeros((3, 1)), [10.0])
    # The exact value is zero; this frozen numerical fixture produces a small
    # negative cancellation remainder, which the estimator must preserve.
    assert -1e-12 < value < 0.0
