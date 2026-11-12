"""Posterior exponential moments, retaining the original Gaussian power law.

The spectral calculation is the retained dr.zero Kronecker/SVD integral with
the newly agreed common residual variance and posterior exponent exposed.
It is not a new estimator or a surrogate identification score.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta as beta_distribution


def binomial_interval(success, count, confidence=.95):
    alpha = 1-confidence
    return (float(beta_distribution.ppf(alpha/2, success, count-success+1)) if success else 0.,
            float(beta_distribution.ppf(1-alpha/2, success+1, count-success)) if success < count else 1.)


def stream(namespace, identity, target, study_seed=2026111201):
    digest = hashlib.sha256((namespace+"/"+identity+"/"+str(target)).encode()).digest()
    entropy = [study_seed, *np.frombuffer(digest[:16], dtype="<u4").tolist()]
    return np.random.default_rng(np.random.SeedSequence(entropy))


@dataclass
class ForecastLaw:
    mean: np.ndarray
    relative_sd: float = .05
    time_ar: float = .7

    def __post_init__(self):
        self.mean = np.asarray(self.mean, float)
        if self.mean.ndim != 2 or not np.isfinite(self.mean).all() or self.mean.shape[1] % 2:
            raise ValueError("Finite [time,P then Q] forecast required")
        if not 0 <= self.time_ar < 1 or not np.isfinite(self.relative_sd) or self.relative_sd < 0:
            raise ValueError("Invalid frozen Gaussian power-error law")
        self.sd = self.relative_sd*np.max(np.abs(self.mean), axis=0)
        T = len(self.mean)
        covariance = self.time_ar**np.abs(np.arange(T)[:, None]-np.arange(T)[None, :])
        values, self.U = np.linalg.eigh(covariance)
        self.values = np.maximum(values, 0.)

    def sample(self, rng, count):
        T, channels = self.mean.shape; n = channels//2
        z = rng.standard_normal((count, T, n))
        for ti in range(1, T):
            z[:, ti] = self.time_ar*z[:, ti-1]+np.sqrt(1-self.time_ar**2)*z[:, ti]
        noise = np.concatenate((z*self.sd[:n], z*self.sd[n:]), axis=2)
        return self.mean[None]+noise


def spectral_moment(difference, law, sigma2, exponent=.5):
    """-log E exp[-lambda*(new log odds)], averaged over power and e.

    e|P ~ N(0,sigma2*I); 0<lambda<1. Old normalized SSE exponent .25
    becomes posterior exponent .5 when the residual scales agree.
    """
    if not np.isfinite(sigma2) or sigma2 <= 0 or not 0 < exponent < 1:
        raise ValueError("Positive fixed residual variance and exponent in (0,1) required")
    dh = np.asarray(difference, float); n = dh.shape[0]
    if dh.shape != (n, 2*n) or law.mean.shape[1] != 2*n:
        raise ValueError("Difference and forecast dimensions differ")
    scale = np.sqrt(sigma2); coefficient = exponent*(1-exponent)/2
    power_factor = (dh[:, :n]*law.sd[:n]+dh[:, n:]*law.sd[n:])/scale
    V, singular, _ = np.linalg.svd(np.sqrt(coefficient)*power_factor, full_matrices=True)
    diagonal = 1+2*law.values[:, None]*singular[None, :]**2
    mean = law.mean@dh.T/scale
    transformed = law.U.T@(np.sqrt(coefficient)*mean)@V
    quadratic = float(np.sum(transformed**2/diagonal))
    half_logdet = float(.5*np.log(diagonal).sum())
    return dict(T=quadratic+half_logdet, quadratic=quadratic, half_logdet=half_logdet)


def failure_bound(log_history_weights, moments, target, beta, exponent=.5):
    lp = np.asarray(log_history_weights, float)
    competitors = [j for j in range(len(lp)) if j != target]
    if not competitors:
        return dict(success_lower=1., log_failure_upper=None, pairs=[], active=[], log_tail=None)
    log_c = exponent*(np.log(beta/(1-beta))-lp[target]+lp[competitors])
    T = np.asarray([moments[j]["T"] for j in competitors])
    log_terms = log_c-T
    total = float(logsumexp(log_terms))
    pairs = [dict(competitor=j, log_c=float(c), T=float(t), log_contribution=float(f))
             for j, c, t, f in zip(competitors, log_c, T, log_terms, strict=True)]
    return dict(success_lower=float(-np.expm1(min(0., total))), log_failure_upper=min(0., total),
                log_failure_sum_unclipped=total, pairs=pairs)


def active_allocation(pairs, eta=.9):
    ordered = sorted(pairs, key=lambda p: (-p["log_c"], p["competitor"]))
    count = 0; allowed = np.log(.1*(1-eta))
    def tail(size):
        return float(logsumexp([p["log_c"] for p in ordered[size:]])) if size < len(ordered) else -np.inf
    while count < len(ordered) and tail(count) > allowed:
        count += 1
    log_tail = tail(count)
    remaining = 1-eta-(np.exp(log_tail) if np.isfinite(log_tail) else 0.)
    epsilon = remaining/count if count else None
    return dict(active=[p["competitor"] for p in ordered[:count]],
        log_tail=None if not np.isfinite(log_tail) else log_tail, remaining_failure=remaining,
        epsilon=epsilon, thresholds=[dict(competitor=p["competitor"], h=float(p["log_c"]-np.log(epsilon)))
                                   for p in ordered[:count]])


def simulate_target(H, log_history, target, law, sigma2, rng, count, chunk=64):
    """Conditional target data; log odds use squared-distance identities.

    Power forecast uncertainty is the original Gaussian AR/same-node-PQ law.
    Actual physical device errors/AC are deliberately a separate experiment.
    """
    H, log_history = np.asarray(H, float), np.asarray(log_history, float)
    differences = H[target][None]-H
    values = []
    for start in range(0, count, chunk):
        size = min(chunk, count-start)
        power = law.sample(rng, size)
        error = np.sqrt(sigma2)*rng.standard_normal((size, len(law.mean), H.shape[1]))
        relative = np.empty((size, len(H)))
        for j, dh in enumerate(differences):
            d = power@dh.T
            increment = np.sum(d*d+2*error*d, axis=(1, 2))/(2*sigma2)
            relative[:, j] = log_history[j]-log_history[target]-increment
        values.extend((-logsumexp(relative, axis=1)).tolist())
    return np.asarray(values)
