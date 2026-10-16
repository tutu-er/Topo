"""Normalized conditional-Gaussian topology probabilities for a small EV study.

``means[l,k,d]`` and ``covariances[l,k,d,d]`` describe outputs conditional on
the *observed common input scenario* l.  Its weight is ``weights[l]``.  The
likelihood sums over candidate k, never over l: the P/Q input scenario is known
when classifying a future observation.  Lines, Gaussian prediction banks and
scenario/prior weights are supplied by the caller; this module performs no AC
fit, candidate screening, or EV-action enumeration.

Integration noise has shape [L,K,R,d].  Strata (l,k) are independent, while two
actions use the same array.  Reported SEs concern this conditional integral,
not independent historical samples or predictive model error.  ``brier`` is
posterior concentration E sum_k posterior_k**2; ``brier_risk`` is its complement.
"""

from __future__ import annotations

from hashlib import sha256
import numpy as np
from scipy.special import logsumexp


_METRICS = ("risk", "eig", "brier", "brier_risk", "abstention_risk",
            "abstention_rate", "map_error", "eig_direct", "eig_identity_difference")


def _finite(value, name):
    a = np.array(value, dtype=float, copy=True)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    return a


def _probability(value, size, name, *, positive):
    a = _finite(value, name)
    if a.shape != (size,) or np.any(a <= 0 if positive else a < 0):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} vector")
    if not np.isclose(a.sum(), 1., atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} must sum to one")
    return a


def _prior(prior, size, log_prior=None):
    """Finite log masses retain candidates whose displayed mass underflows."""
    if log_prior is None:
        p = _probability(prior, size, "prior", positive=True)
        return p, np.log(p)
    lp = _finite(log_prior, "log_prior")
    if lp.shape != (size,) or not np.isclose(logsumexp(lp), 0., atol=1e-10, rtol=0.):
        raise ValueError("log_prior must have shape [K] and logsumexp zero")
    lp -= logsumexp(lp)
    p = np.exp(lp)
    if prior is not None:
        shown = _probability(prior, size, "prior", positive=False)
        if not np.allclose(shown, p, atol=1e-14, rtol=1e-10):
            raise ValueError("prior must agree with exp(log_prior)")
    return p, lp


def _symmetric(value, shape, name):
    a = _finite(value, name)
    if a.shape != shape or not np.allclose(a, a.swapaxes(-1, -2), atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must have the required symmetric shape {shape}")
    return .5 * (a + a.swapaxes(-1, -2))


def _gaussians(means, covariances):
    means = _finite(means, "means")
    if means.ndim != 2 or min(means.shape) < 1:
        raise ValueError("means must have shape [K,d] with positive dimensions")
    K, d = means.shape
    covariance = _symmetric(covariances, (K, d, d), "covariances")
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariances must be positive definite; no jitter is added") from exc
    inverse_chol = np.linalg.solve(chol, np.broadcast_to(np.eye(d), (K, d, d)))
    precision = np.einsum("kji,kjl->kil", inverse_chol, inverse_chol)
    logdet = 2 * np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(-1)
    return dict(means=means, covariances=covariance, cholesky=chol,
                inverse_cholesky=inverse_chol, precision=precision, logdet=logdet)


def validate_bank(bank):
    """Copy and validate a bank; reject nonproper weights and singular covariance.

    Finite normalized log_prior is authoritative when present; displayed prior
    may then underflow to zero.  Without it positive prior is required.  Scenario
    weights may be zero.  No candidate is removed and no covariance is repaired.
    """
    if not isinstance(bank, dict):
        raise ValueError("bank must be a dict")
    means = _finite(bank.get("means"), "means")
    if means.ndim != 3 or min(means.shape) < 1:
        raise ValueError("means must have shape [L,K,d] with positive dimensions")
    L, K, d = means.shape
    covariance = _symmetric(bank.get("covariances"), (L, K, d, d), "covariances")
    models = [_gaussians(means[l], covariance[l]) for l in range(L)]
    prior, log_prior = _prior(bank.get("prior"), K, bank.get("log_prior"))
    return dict(means=means, covariances=covariance,
                prior=prior, log_prior=log_prior, prior_underflow_count=int(np.sum(prior == 0)),
                weights=_probability(bank.get("weights"), L, "weights", positive=False),
                cholesky=np.stack([m["cholesky"] for m in models]),
                logdet=np.stack([m["logdet"] for m in models]), _models=models)


def _posterior(values, model, log_prior):
    y = _finite(values, "values")
    d = model["means"].shape[-1]
    if y.ndim < 1 or y.shape[-1] != d:
        raise ValueError("values must end in output dimension d")
    residual = y[..., None, :] - model["means"]
    white = np.einsum("kij,...kj->...ki", model["inverse_cholesky"], residual)
    loglike = -.5 * (d * np.log(2 * np.pi) + model["logdet"] + (white ** 2).sum(-1))
    if not np.isfinite(loglike).all():
        raise ValueError("Gaussian log likelihood overflow; rescale the declared inputs")
    joint = loglike + log_prior
    logpred = logsumexp(joint, axis=-1)
    logpost = joint - logpred[..., None]
    return dict(posterior=np.exp(logpost), log_posterior=logpost,
                log_predictive=logpred, log_likelihood=loglike, residual=residual)


def gaussian_posterior(values, means, covariances, prior=None, *, log_prior=None):
    """Full Gaussian density and candidate posterior, including every logdet.

    ``values[...,d]`` is one or several observations; returns posterior[...,K].
    The prior is external.  No maximized fit loss is substituted for a density.
    """
    model = _gaussians(means, covariances)
    p, lp = _prior(prior, len(model["means"]), log_prior)
    result = _posterior(values, model, lp)
    result.pop("residual")
    result.update(log_prior=lp, prior=p, prior_underflow_count=int(np.sum(p == 0)))
    return result


def fixed_observation_posterior_gradient(values, means, covariances, prior, mean_jacobian, *, log_prior=None):
    """Derivative at fixed values, fixed covariance and fixed prior.

    mean_jacobian[k,d,p] gives p continuous-control directions.  This is not the
    total planning derivative, because EV changes the generated observation too.
    """
    model = _gaussians(means, covariances)
    p, lp = _prior(prior, len(model["means"]), log_prior)
    jac = _finite(mean_jacobian, "mean_jacobian")
    if jac.ndim != 3 or jac.shape[:2] != model["means"].shape or jac.shape[-1] < 1:
        raise ValueError("mean_jacobian must have shape [K,d,p]")
    result = _posterior(values, model, lp)
    precision_residual = np.einsum("kij,...kj->...ki", model["precision"], result.pop("residual"))
    score = np.einsum("kdp,...kd->...kp", jac, precision_residual)
    average = np.einsum("...k,...kp->...p", result["posterior"], score)
    result.update(gradient=result["posterior"][..., :, None] * (score - average[..., None, :]),
                  log_density_gradient=score, derivative_scope="fixed observation/covariance/prior")
    return result


# Short alias for callers that already state the fixed-observation convention.
posterior_gradient = fixed_observation_posterior_gradient


def bhattacharyya_bank(bank):
    """Gaussian pair overlaps and multicandidate Bayes-error upper bound.

    B_ij=delta' ((Sigma_i+Sigma_j)/2)^-1 delta/8
          + log(det(Sigma_bar)/sqrt(det(Sigma_i)det(Sigma_j)))/2.
    Pointwise sum f-max(f) <= sum_{i<j} min(f_i,f_j)
    <= sum_{i<j} sqrt(f_i*f_j), so integration yields this bound.
    The raw pair sum is retained even if a trivial bound is smaller.
    """
    b = validate_bank(bank)
    L, K, _ = b["means"].shape
    rows, by_scenario = [], np.zeros(L)
    for l in range(L):
        for i in range(K):
            for j in range(i + 1, K):
                average = .5 * (b["covariances"][l, i] + b["covariances"][l, j])
                chol = np.linalg.cholesky(average)
                delta = b["means"][l, i] - b["means"][l, j]
                white = np.linalg.solve(chol, delta)
                mean_term = float(white @ white / 8.)
                covariance_term = float(.5 * (2 * np.log(np.diag(chol)).sum()
                    - .5 * (b["logdet"][l, i] + b["logdet"][l, j])))
                raw_B = mean_term + covariance_term
                if raw_B < -1e-10:
                    raise ValueError("negative Bhattacharyya distance exceeds rounding tolerance")
                distance = max(0., raw_B)
                log_coefficient = float(.5 * (b["log_prior"][i] + b["log_prior"][j]))
                coefficient = float(np.exp(log_coefficient))
                contribution = float(np.exp(log_coefficient - distance))
                by_scenario[l] += contribution
                rows.append(dict(l=l, i=i, j=j, B=distance, raw_B=raw_B,
                    mean_term=mean_term, covariance_term=covariance_term,
                    coefficient=coefficient, log_coefficient=log_coefficient, overlap=float(np.exp(-distance)),
                    contribution=contribution,
                    weighted_contribution=float(b["weights"][l] * contribution)))
    total = float(b["weights"] @ by_scenario)
    trivial = float(1 - b["prior"].max())
    return dict(pairs=rows, by_scenario=by_scenario, bound=total, risk_upper_bound=total,
                trivial_bound=trivial, capped_bound=min(total, trivial),
                log_prior=b["log_prior"].copy(), prior_underflow_count=b["prior_underflow_count"],
                scope="declared conditional Gaussian model only")


def _summary(samples, stratum_weights):
    """Independent (l,k) strata, R shared-noise replications within each stratum."""
    R = samples.shape[2]
    mean = np.einsum("lk,lk...->...", stratum_weights, samples.mean(axis=2))
    variance = np.einsum("lk,lk...->...", stratum_weights ** 2, samples.var(axis=2, ddof=1)) / R
    return mean, np.sqrt(variance)


def evaluate_bank(bank, noise, *, abstention_cost=.1, mean_jacobian=None,
                  return_posterior=False):
    """Conditional risk/EIG/Brier and stratified integration SE on common noise.

    EIG uses pointwise posterior KL, a Rao--Blackwell representation.  No variance
    improvement is claimed for the fixed stratified sampling scheme.  The direct
    log-ratio estimate and paired identity difference are retained for audit.

    If mean_jacobian[L,K,d,p] is supplied, gradients are the exact path derivative
    of these sample objectives for fixed covariance, prior and scenario weights.
    They include dy/du for the generating candidate.  MAP/abstention ties are
    recorded; no smooth gradient is asserted at a nonunique maximum/threshold.
    """
    b = validate_bank(bank)
    L, K, d = b["means"].shape
    e = _finite(noise, "noise")
    if e.ndim != 4 or e.shape[:2] != (L, K) or e.shape[-1] != d or e.shape[2] < 2:
        raise ValueError("noise must have shape [L,K,R,d] with R>=2")
    if not np.isscalar(abstention_cost) or not np.isfinite(abstention_cost) or not 0 < abstention_cost < 1:
        raise ValueError("abstention_cost must be finite and strictly between zero and one")
    R = e.shape[2]
    jac = None
    if mean_jacobian is not None:
        jac = _finite(mean_jacobian, "mean_jacobian")
        if jac.ndim != 4 or jac.shape[:3] != (L, K, d) or jac.shape[-1] < 1:
            raise ValueError("mean_jacobian must have shape [L,K,d,p]")
    samples = {key: np.empty((L, K, R)) for key in _METRICS}
    posteriors = np.empty((L, K, R, K))
    grads = ({key: np.empty((L, K, R, jac.shape[-1]))
              for key in ("risk", "eig", "brier", "abstention_risk")} if jac is not None else {})
    tie_count = threshold_count = 0
    logprior = b["log_prior"]
    for l in range(L):
        model = b["_models"][l]
        for k in range(K):
            y = b["means"][l, k] + e[l, k] @ b["cholesky"][l, k].T
            value = _posterior(y, model, logprior)
            p, lp = value["posterior"], value["log_posterior"]
            posteriors[l, k] = p
            winner = np.argmax(p, axis=-1)
            maximum = p[np.arange(R), winner]
            risk = 1 - maximum
            concentration = (p ** 2).sum(-1)
            information = (p * (lp - logprior)).sum(-1)
            direct = lp[:, k] - logprior[k]
            samples["risk"][l, k] = risk
            samples["eig"][l, k] = information
            samples["brier"][l, k] = concentration
            samples["brier_risk"][l, k] = 1 - concentration
            samples["abstention_risk"][l, k] = np.minimum(abstention_cost, risk)
            at_threshold = np.isclose(risk, abstention_cost, atol=1e-12, rtol=1e-10)
            samples["abstention_rate"][l, k] = (risk > abstention_cost) & ~at_threshold
            samples["map_error"][l, k] = winner != k
            samples["eig_direct"][l, k] = direct
            samples["eig_identity_difference"][l, k] = information - direct
            tied = np.sum(np.isclose(p, maximum[:, None], atol=1e-12, rtol=1e-10), axis=-1) > 1
            tie_count += int(tied.sum())
            threshold_count += int(at_threshold.sum())
            if jac is not None:
                precision_residual = np.einsum("bij,rbj->rbi", model["precision"], value["residual"])
                path_loglike = np.einsum("bdp,rbd->rbp", jac[l] - jac[l, k], precision_residual)
                mean_score = np.einsum("rb,rbp->rp", p, path_loglike)
                path_posterior = p[..., None] * (path_loglike - mean_score[:, None, :])
                risk_gradient = -path_posterior[np.arange(R), winner]
                grads["risk"][l, k] = risk_gradient
                grads["eig"][l, k] = np.einsum("rbp,rb->rp", path_posterior, lp - logprior)
                grads["brier"][l, k] = 2 * np.einsum("rb,rbp->rp", p, path_posterior)
                grads["abstention_risk"][l, k] = np.where(
                    (risk < abstention_cost)[:, None], risk_gradient, 0.)
    sw = b["weights"][:, None] * b["prior"][None, :]
    result = dict(sample_count_per_stratum=R, stratum_count=L * K,
                  sample_count=L * K * R, samples=samples, strata=[],
                  prior=b["prior"].copy(), weights=b["weights"].copy(),
                  log_prior=b["log_prior"].copy(), prior_underflow_count=b["prior_underflow_count"],
                  noise_sha256=sha256(np.asarray(e, dtype="<f8").tobytes()).hexdigest(),
                  abstention_cost=float(abstention_cost), map_tie_count=tie_count,
                  abstention_boundary_count=threshold_count,
                  density_scope="Gaussian topology mixture conditional on known input scenario l",
                  se_scope="independent integration strata; not independent histories")
    for key, values in samples.items():
        mean, se = _summary(values, sw)
        result[key], result[key + "_se"] = float(mean), float(se)
    result["accuracy"] = 1 - result["risk"]
    result["accuracy_se"] = result["risk_se"]
    result["brier_concentration"] = result["brier"]
    result["posterior_mean"], result["posterior_mean_se"] = _summary(posteriors, sw)
    for l in range(L):
        for k in range(K):
            result["strata"].append(dict(l=l, k=k, weight=float(sw[l, k]), sample_count=R,
                means={key: float(v[l, k].mean()) for key, v in samples.items()},
                sample_variances={key: float(v[l, k].var(ddof=1)) for key, v in samples.items()}))
    if jac is not None:
        result["gradients"], result["gradient_ses"] = {}, {}
        for key, values in grads.items():
            result["gradients"][key], result["gradient_ses"][key] = _summary(values, sw)
        result["gradient_valid"] = dict(risk=tie_count == 0, eig=True, brier=True,
            abstention_risk=tie_count == 0 and threshold_count == 0)
        result["gradient_scope"] = "fixed covariance, prior, input weights; observation path included"
    if return_posterior:
        result["posterior_samples"] = posteriors
    return result


def compare_evaluations(proposed, baseline):
    """Paired integration differences; callers must use the same noise array.

    Different prior/scenario weights would change the stratification and are
    rejected.  The method retains CRN covariance by differencing samples first.
    A negative risk change or positive EIG/Brier change favors the proposal.
    """
    p, b = proposed, baseline
    if (not np.array_equal(p["log_prior"], b["log_prior"]) or not np.array_equal(p["weights"], b["weights"])):
        raise ValueError("paired actions must use identical prior and scenario weights")
    if p.get("noise_sha256") != b.get("noise_sha256"):
        raise ValueError("paired actions must use the same noise array")
    sw = p["weights"][:, None] * p["prior"][None, :]
    result = {"direction": "proposed minus baseline", "se_scope": p["se_scope"]}
    for key in _METRICS:
        left, right = np.asarray(p["samples"][key]), np.asarray(b["samples"][key])
        if left.shape != right.shape:
            raise ValueError("paired actions must have identical noise/stratum shapes")
        mean, se = _summary(left - right, sw)
        result[key], result[key + "_se"] = float(mean), float(se)
    return result


def compare_banks(proposed, baseline, noise, *, abstention_cost=.1):
    """Evaluate two actions with a common declared noise array and pair them."""
    left = evaluate_bank(proposed, noise, abstention_cost=abstention_cost)
    right = evaluate_bank(baseline, noise, abstention_cost=abstention_cost)
    return compare_evaluations(left, right)


def condition_gaussian_inputs(prior_mean, prior_cov, observed, measurement_cov):
    """Condition true input x on measured input x+meter noise exactly once.

    x~N(prior_mean,prior_cov), independent meter noise~N(0,measurement_cov).
    The returned input covariance is already conditional; do not add the meter
    covariance again when propagating it through AC.  A zero prior covariance
    is allowed.  log_input_density is common across candidates only if this
    entire input model is common.
    """
    mean = _finite(prior_mean, "prior_mean")
    if mean.ndim != 1 or len(mean) < 1:
        raise ValueError("prior_mean must be a nonempty vector")
    d = len(mean)
    covariance = _symmetric(prior_cov, (d, d), "prior_cov")
    if np.linalg.eigvalsh(covariance).min() < -1e-12 * (1 + np.linalg.norm(covariance)):
        raise ValueError("prior_cov must be positive semidefinite")
    meter = _symmetric(measurement_cov, (d, d), "measurement_cov")
    try:
        np.linalg.cholesky(meter)
        total_chol = np.linalg.cholesky(covariance + meter)
    except np.linalg.LinAlgError as exc:
        raise ValueError("measurement_cov must be positive definite") from exc
    value = _finite(observed, "observed")
    if value.ndim < 1 or value.shape[-1] != d:
        raise ValueError("observed must end in input dimension d")
    gain = np.linalg.solve(total_chol.T, np.linalg.solve(total_chol, covariance)).T
    conditional_mean = mean + np.einsum("ij,...j->...i", gain, value - mean)
    conditional_covariance = covariance - gain @ covariance
    conditional_covariance = .5 * (conditional_covariance + conditional_covariance.T)
    white = np.linalg.solve(total_chol, (value - mean).reshape(-1, d).T).T
    log_density = -.5 * (d * np.log(2 * np.pi) + 2 * np.log(np.diag(total_chol)).sum()
                          + (white ** 2).sum(-1))
    return dict(mean=conditional_mean, covariance=conditional_covariance, gain=gain,
                log_input_density=log_density.reshape(value.shape[:-1]),
                meter_error_conditioned_once=True)
