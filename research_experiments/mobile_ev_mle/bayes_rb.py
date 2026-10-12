"""Posterior-KL estimator of finite Bayesian topology information.

This separate module leaves the phase-1 log-ratio estimator unchanged. Given
each simulated observation it averages over the conditional topology posterior.
For ordinary iid mixture draws this is a Rao--Blackwell variance reduction.
For fixed topology strata the standard error is computed directly; its variance
order relative to the phase-1 stratified estimator depends on the configuration.
"""

from collections.abc import Mapping

import numpy as np
from scipy.special import logsumexp

try:
    from .bayes import _future_matrix, _positive_integer, _log_predictive_parts
except ImportError:
    from bayes import _future_matrix, _positive_integer, _log_predictive_parts


def _posterior_statistics(posterior, means, observations):
    joint, marginal = _log_predictive_parts(posterior, means, observations)
    log_mass = joint - marginal[:, None]
    mass = np.exp(log_mass)
    # Signed floating-point roundoff is preserved, rather than clipped.
    information = np.sum(mass * (
        log_mass - posterior.log_topology_probabilities[None, :]), axis=1)
    reduction = posterior.topology_entropy + np.sum(mass * log_mass, axis=1)
    return information, reduction


def pointwise_topology_kl(posterior, future_means, values, *, batch_size=256):
    """Return KL(topology posterior after y || historical topology posterior).

    values[DN] returns a float; values[M,DN] returns an array. Predictive means
    and observations use the frozen phase-1 Gaussian whitening convention.
    """
    means = _future_matrix(posterior, future_means)
    batch_size = _positive_integer(batch_size, "batch_size")
    observations = np.asarray(values, dtype=float)
    scalar = observations.ndim == 1
    if scalar:
        observations = observations[None, :]
    if (observations.ndim != 2 or observations.shape[1] != means.shape[1]
            or not np.all(np.isfinite(observations))):
        raise ValueError("values must be finite with shape [DN] or [M,DN]")
    information = np.empty(len(observations))
    for start in range(0, len(observations), batch_size):
        information[start:start + batch_size] = _posterior_statistics(
            posterior, means, observations[start:start + batch_size])[0]
    return float(information[0]) if scalar else information


def rb_eig_scores(posterior, future_means, *, samples_per_topology=2048,
                  seed=0, batch_size=256):
    """Estimate E_y KL(pi(.|y)||pi) with independent fixed topology strata.

    All topology labels receive the same fixed sample count; within each label
    the atom is sampled using its posterior conditional mass. Atom selections
    and Gaussian noise are common across actions and independent across labels.
    Per-action and paired comparison errors use weighted stratum variances.
    The entropy-reduction estimator uses the same observations as an identity
    check. Neither estimator nor its paired comparisons are clipped.
    """
    samples = _positive_integer(samples_per_topology, "samples_per_topology", minimum=2)
    batch_size = _positive_integer(batch_size, "batch_size")
    if not isinstance(future_means, Mapping) or not future_means:
        raise ValueError("future_means must be a nonempty action-name mapping")
    if not all(isinstance(name, str) for name in future_means):
        raise ValueError("action names must be strings")
    actions = {name: _future_matrix(posterior, future_means[name])
               for name in sorted(future_means)}
    dimensions = {means.shape[1] for means in actions.values()}
    if len(dimensions) != 1:
        raise ValueError("all actions must have the same future dimension for common noise")
    dimension = dimensions.pop()
    rng = np.random.default_rng(seed)
    strata = []
    for h in posterior.topology_labels:
        indices = np.flatnonzero(posterior.labels == h)
        log_conditional = posterior.log_weights[indices]
        conditional = np.exp(log_conditional - logsumexp(log_conditional))
        conditional /= conditional.sum()
        selections = rng.choice(indices, size=samples, replace=True, p=conditional)
        noise = rng.standard_normal((samples, dimension))
        strata.append((selections, noise))

    scores, retained = {}, {}
    for name, means in actions.items():
        information_mean = entropy_mean = difference_mean = 0.0
        information_variance = entropy_variance = difference_variance = 0.0
        rows, action_draws = [], []
        for h_index, (selections, noise) in enumerate(strata):
            observations = means[selections] + noise
            information, reduction = np.empty(samples), np.empty(samples)
            for start in range(0, samples, batch_size):
                stop = start + batch_size
                information[start:stop], reduction[start:stop] = _posterior_statistics(
                    posterior, means, observations[start:stop])
            difference = information - reduction
            action_draws.append((information, reduction))
            information_avg, reduction_avg, difference_avg = map(
                float, (information.mean(), reduction.mean(), difference.mean()))
            information_var, reduction_var, difference_var = map(
                float, (information.var(ddof=1), reduction.var(ddof=1),
                        difference.var(ddof=1)))
            probability = float(posterior.topology_probabilities[h_index])
            information_mean += probability * information_avg
            entropy_mean += probability * reduction_avg
            difference_mean += probability * difference_avg
            information_variance += probability ** 2 * information_var / samples
            entropy_variance += probability ** 2 * reduction_var / samples
            difference_variance += probability ** 2 * difference_var / samples
            rows.append(dict(
                topology=int(posterior.topology_labels[h_index]),
                posterior_probability=probability, sample_count=samples,
                rb_mean=information_avg, rb_sample_variance=information_var,
                entropy_reduction_mean=reduction_avg,
                entropy_sample_variance=reduction_var,
                identity_difference_mean=difference_avg,
                identity_sample_variance=difference_var,
                min_pointwise_kl=float(information.min()),
                max_pointwise_kl=float(information.max()),
            ))
        retained[name] = action_draws
        scores[name] = dict(
            estimator="posterior_KL_Rao_Blackwell",
            eig=float(information_mean), eig_se=float(np.sqrt(information_variance)),
            entropy_reduction=float(entropy_mean),
            entropy_se=float(np.sqrt(entropy_variance)),
            identity_difference=float(difference_mean),
            identity_difference_se=float(np.sqrt(difference_variance)),
            entropy_upper_bound=posterior.topology_entropy,
            samples_per_topology=samples, total_samples=samples * len(strata),
            integration_seed=int(seed) if isinstance(seed, (int, np.integer)) else None,
            strata=rows, common_noise_across_actions=True,
            independent_topology_strata=True, clipped=False,
            variance_comparison_scope="iid mixture theorem; fixed strata compared empirically",
        )
    for name in scores:
        paired = {}
        for baseline in scores:
            gain = entropy_gain = gain_variance = entropy_gain_variance = 0.0
            for probability, current, control in zip(
                    posterior.topology_probabilities, retained[name], retained[baseline]):
                information_difference = current[0] - control[0]
                entropy_difference = current[1] - control[1]
                gain += probability * information_difference.mean()
                entropy_gain += probability * entropy_difference.mean()
                gain_variance += probability ** 2 * information_difference.var(ddof=1) / samples
                entropy_gain_variance += probability ** 2 * entropy_difference.var(ddof=1) / samples
            paired[baseline] = dict(
                eig_gain=float(gain), eig_gain_se=float(np.sqrt(gain_variance)),
                entropy_gain=float(entropy_gain),
                entropy_gain_se=float(np.sqrt(entropy_gain_variance)),
            )
        scores[name]["paired_comparisons"] = paired
    return scores
