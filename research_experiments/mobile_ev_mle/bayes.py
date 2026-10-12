"""Exact finite-prior Gaussian updating and topology-information design.

All observations and means are whitened by the same known measurement noise
covariance. Each row is one *complete* atom, retaining its shared line vector
and its historical/future latent path. No atom is screened by historical loss.
The finite sums are exact; only the prospective observation integral is Monte
Carlo. Random draws are independent between topology strata and shared between
actions, so the reported stratified standard errors include the right weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


_LOG_TWO_PI = float(np.log(2.0 * np.pi))


def _readonly(values):
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def _matrix(values, name):
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or not all(result.shape):
        raise ValueError(f"{name} must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class PosteriorWeights:
    """Posterior on a declared finite atom support, with stable log masses."""

    log_weights: np.ndarray
    weights: np.ndarray
    labels: np.ndarray
    topology_labels: np.ndarray
    log_topology_probabilities: np.ndarray
    topology_probabilities: np.ndarray
    history_loss: np.ndarray
    log_evidence: float
    atom_ess: float
    conditional_atom_ess: np.ndarray

    @property
    def topology_entropy(self):
        return float(-np.dot(self.topology_probabilities,
                             self.log_topology_probabilities))

    def summary(self):
        """JSON-friendly concentration summary; ESS is not an MCMC check."""
        return dict(
            topology_probabilities={int(h): float(p) for h, p in
                                    zip(self.topology_labels, self.topology_probabilities)},
            log_topology_probabilities={int(h): float(p) for h, p in
                                        zip(self.topology_labels, self.log_topology_probabilities)},
            topology_entropy=self.topology_entropy,
            log_evidence=self.log_evidence,
            atom_ess=self.atom_ess,
            conditional_atom_ess={int(h): float(p) for h, p in
                                  zip(self.topology_labels, self.conditional_atom_ess)},
            atom_count=len(self.weights),
            atom_log_weight_min=float(self.log_weights.min()),
            floating_underflow_count=int(np.count_nonzero(self.weights == 0.0)),
            historical_screening=False,
        )


def posterior_weights(history_y, history_means, labels, proper_prior, *,
                      log_whitening_jacobian=0.0):
    """Update positive, externally specified atom masses with Gaussian meters.

    Parameters are y[DH], means[G,DH], integer labels[G], and prior[G]. The
    prior must already sum to one; this function does not invent topology
    masses from atom counts. ``log_evidence`` includes the unit-Gaussian
    normalizer in whitened coordinates. To recover evidence in the original
    coordinates, supply log(abs(det(whitening_matrix))). That shared constant
    does not affect posterior weights. No small log mass is truncated.
    """
    means = _matrix(history_means, "history_means")
    observation = np.asarray(history_y, dtype=float)
    if observation.ndim != 1 or observation.shape != (means.shape[1],):
        raise ValueError("history_y must have shape [DH]")
    if not np.all(np.isfinite(observation)):
        raise ValueError("history_y must be finite")
    group = np.asarray(labels)
    if (group.ndim != 1 or len(group) != len(means)
            or not np.issubdtype(group.dtype, np.integer)):
        raise ValueError("labels must have shape [G] and integer values")
    prior = np.asarray(proper_prior, dtype=float)
    if prior.shape != (len(means),) or not np.all(np.isfinite(prior)):
        raise ValueError("proper_prior must be a finite array with shape [G]")
    if np.any(prior <= 0.0):
        raise ValueError("proper_prior must give every declared atom positive mass")
    if not np.isclose(prior.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("proper_prior must already sum to one")
    if not np.isfinite(log_whitening_jacobian):
        raise ValueError("log_whitening_jacobian must be finite")

    residual = means - observation
    with np.errstate(over="ignore", invalid="ignore"):
        loss = np.einsum("ij,ij->i", residual, residual)
    if not np.all(np.isfinite(loss)):
        raise ValueError("history squared residuals overflowed")
    log_score = np.log(prior) - 0.5 * loss
    log_partition = float(logsumexp(log_score))
    log_weight = log_score - log_partition
    weight = np.exp(log_weight)
    topologies = np.unique(group)
    log_mass = np.array([logsumexp(log_weight[group == h]) for h in topologies])
    log_mass -= logsumexp(log_mass)
    mass = np.exp(log_mass)
    conditional_ess = []
    for h in topologies:
        log_conditional = log_weight[group == h]
        conditional = np.exp(log_conditional - logsumexp(log_conditional))
        conditional_ess.append(1.0 / np.dot(conditional, conditional))
    return PosteriorWeights(
        log_weights=_readonly(log_weight), weights=_readonly(weight),
        labels=_readonly(group), topology_labels=_readonly(topologies),
        log_topology_probabilities=_readonly(log_mass),
        topology_probabilities=_readonly(mass), history_loss=_readonly(loss),
        log_evidence=float(log_partition - 0.5 * means.shape[1] * _LOG_TWO_PI
                           + log_whitening_jacobian),
        atom_ess=float(1.0 / np.dot(weight, weight)),
        conditional_atom_ess=_readonly(conditional_ess),
    )


def _future_matrix(posterior, values):
    if not isinstance(posterior, PosteriorWeights):
        raise TypeError("posterior must be returned by posterior_weights")
    result = _matrix(values, "future_means")
    if len(result) != len(posterior.log_weights):
        raise ValueError("future_means must have the same atom rows as the posterior")
    return result


def _log_predictive_parts(posterior, means, observations):
    # Direct residuals avoid cancellation for large, almost equal means.
    with np.errstate(over="ignore", invalid="ignore"):
        residual = observations[:, None, :] - means[None, :, :]
        squared = np.einsum("bgd,bgd->bg", residual, residual)
    if not np.all(np.isfinite(squared)):
        raise ValueError("future squared residuals overflowed")
    log_atom = (-0.5 * squared - 0.5 * means.shape[1] * _LOG_TWO_PI
                + posterior.log_weights[None, :])
    joint = np.column_stack([
        logsumexp(log_atom[:, posterior.labels == h], axis=1)
        for h in posterior.topology_labels
    ])
    marginal = logsumexp(joint, axis=1)
    return joint, marginal


def predictive_logpdf(posterior, future_means, values, *, topology=None,
                      batch_size=256):
    """Exact unit-noise finite-mixture log density, marginal or conditional.

    values[DN] describes one observation and returns a float. values[M,DN]
    returns an array. All Gaussian normalizing constants are retained.
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
    index = None
    if topology is not None:
        match = np.flatnonzero(posterior.topology_labels == topology)
        if len(match) != 1:
            raise ValueError("topology is not a declared label")
        index = int(match[0])
    result = np.empty(len(observations))
    for start in range(0, len(observations), batch_size):
        stop = start + batch_size
        joint, marginal = _log_predictive_parts(posterior, means, observations[start:stop])
        result[start:stop] = (marginal if index is None else
                             joint[:, index] - posterior.log_topology_probabilities[index])
    return float(result[0]) if scalar else result


def eig_scores(posterior, future_means, *, samples_per_topology=512,
               seed=0, batch_size=256):
    """Estimate conditional topology EIG and the entropy identity by strata.

    ``future_means`` maps action names to means[G,DN] on exactly the historical
    atom support. Each topology receives the fixed requested sample count,
    including tiny posterior-mass topologies. Conditional atom selections and
    standard Gaussian errors are shared across actions, but are independent
    between topology strata. Therefore Var(Uhat)=sum_h pi_h**2*s_h**2/M_h.
    Negative estimates and identity differences are retained without clipping.
    """
    samples = _positive_integer(samples_per_topology, "samples_per_topology", minimum=2)
    batch_size = _positive_integer(batch_size, "batch_size")
    if not isinstance(future_means, Mapping) or not future_means:
        raise ValueError("future_means must be a nonempty action-name mapping")
    if not all(isinstance(name, str) for name in future_means):
        raise ValueError("action names must be strings")
    action_means = {name: _future_matrix(posterior, future_means[name])
                    for name in sorted(future_means)}
    dimensions = {value.shape[1] for value in action_means.values()}
    if len(dimensions) != 1:
        raise ValueError("all actions must have the same future dimension for common noise")
    dimension = dimensions.pop()
    rng = np.random.default_rng(seed)
    strata = []
    for h in posterior.topology_labels:
        indices = np.flatnonzero(posterior.labels == h)
        log_conditional = posterior.log_weights[indices]
        conditional = np.exp(log_conditional - logsumexp(log_conditional))
        # Only roundoff normalization, with no loss/probability cutoff.
        conditional /= conditional.sum()
        selections = rng.choice(indices, size=samples, replace=True, p=conditional)
        noise = rng.standard_normal((samples, dimension))
        strata.append((selections, noise))

    entropy = posterior.topology_entropy
    scores = {}
    retained_draws = {}
    for name, means in action_means.items():
        eig_mean = entropy_mean = difference_mean = 0.0
        eig_variance = entropy_variance = difference_variance = 0.0
        rows = []
        action_draws = []
        for h_index, (selections, noise) in enumerate(strata):
            observations = means[selections] + noise
            information = np.empty(samples)
            reduction = np.empty(samples)
            for start in range(0, samples, batch_size):
                stop = start + batch_size
                joint, marginal = _log_predictive_parts(
                    posterior, means, observations[start:stop])
                log_updated_mass = joint - marginal[:, None]
                updated_mass = np.exp(log_updated_mass)
                information[start:stop] = (
                    joint[:, h_index] - posterior.log_topology_probabilities[h_index]
                    - marginal)
                updated_entropy = -np.sum(updated_mass * log_updated_mass, axis=1)
                reduction[start:stop] = entropy - updated_entropy
            difference = information - reduction
            action_draws.append((information, reduction))
            information_avg, reduction_avg, difference_avg = map(
                float, (information.mean(), reduction.mean(), difference.mean()))
            information_var, reduction_var, difference_var = map(
                float, (information.var(ddof=1), reduction.var(ddof=1),
                        difference.var(ddof=1)))
            probability = float(posterior.topology_probabilities[h_index])
            eig_mean += probability * information_avg
            entropy_mean += probability * reduction_avg
            difference_mean += probability * difference_avg
            eig_variance += probability ** 2 * information_var / samples
            entropy_variance += probability ** 2 * reduction_var / samples
            difference_variance += probability ** 2 * difference_var / samples
            rows.append(dict(
                topology=int(posterior.topology_labels[h_index]),
                posterior_probability=probability, sample_count=samples,
                eig_mean=information_avg, eig_sample_variance=information_var,
                entropy_reduction_mean=reduction_avg,
                entropy_sample_variance=reduction_var,
                identity_difference_mean=difference_avg,
                identity_sample_variance=difference_var,
            ))
        scores[name] = dict(
            eig=float(eig_mean), eig_se=float(np.sqrt(eig_variance)),
            entropy_reduction=float(entropy_mean),
            entropy_se=float(np.sqrt(entropy_variance)),
            identity_difference=float(difference_mean),
            identity_difference_se=float(np.sqrt(difference_variance)),
            entropy_upper_bound=entropy,
            samples_per_topology=samples,
            total_samples=samples * len(strata),
            integration_seed=int(seed) if isinstance(seed, (int, np.integer)) else None,
            strata=rows, common_noise_across_actions=True,
            independent_topology_strata=True, clipped=False,
        )
        retained_draws[name] = action_draws
    # Use the same draw differences, retaining the covariance created by CRN.
    # Adding the separate action variances would discard that covariance.
    for name in scores:
        paired = {}
        for baseline in scores:
            gain = entropy_gain = gain_variance = entropy_gain_variance = 0.0
            for probability, current, control in zip(
                    posterior.topology_probabilities, retained_draws[name],
                    retained_draws[baseline]):
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
