"""Frozen two-candidate Gaussian evidence and its quadratic planning forms.

Residual arrays have shape (slots, observations), effects (observations,
resources), and delivery (resources, slots). Likelihood standard deviations
use the same residual units. Evidence is a SUM over observations, as required
by the declared independent Gaussian likelihood, not a mean absolute loss.

These functions define a working two-candidate posterior. They do not estimate
priors/covariances, fit a topology, or certify that either candidate is true.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import expit, logit, ndtr, ndtri


def _finite(name, value):
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(name + " must contain finite numbers")
    return array


def _broadcast(name, value, shape, *, positive=False, nonnegative=False):
    array = _finite(name, value)
    try:
        array = np.array(np.broadcast_to(array, shape), copy=True)
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to {shape}") from exc
    if positive and np.any(array <= 0):
        raise ValueError(name + " must be strictly positive")
    if nonnegative and np.any(array < 0):
        raise ValueError(name + " must be nonnegative")
    return array


def _inputs(baseC, baseF, effectC, effectF, likelihood_sd):
    coarse, fine = _finite("baseC", baseC), _finite("baseF", baseF)
    if (coarse.ndim != 2 or not coarse.size or coarse.shape != fine.shape):
        raise ValueError("Candidate bases must share nonempty (slots, observations) shape")
    slots, nodes = coarse.shape
    gc, gf = _finite("effectC", effectC), _finite("effectF", effectF)
    if (gc.ndim != 2 or not gc.size or gc.shape != gf.shape or gc.shape[0] != nodes):
        raise ValueError("Effects must share nonempty (observations, resources) shape")
    sd = _broadcast("likelihood_sd", likelihood_sd, (slots, nodes), positive=True)
    return coarse, fine, gc, gf, 1. / sd**2


@dataclass(frozen=True)
class QuadraticEvidence:
    """Scalar c + sum_t(linear[t] @ u[:,t] + u[:,t].T @ Q[t] @ u[:,t]).

    Q is stored symmetric, and gradient is returned in resource-by-slot order.
    ``kind`` distinguishes expected log Bayes factor from nominal separation.
    Prior/history log odds are deliberately external to this quadratic form.
    """

    Q: np.ndarray
    linear: np.ndarray
    constant: float
    kind: str

    def __post_init__(self):
        q = np.array(_finite("Q", self.Q), copy=True)
        if q.ndim != 3 or not q.size or q.shape[1] != q.shape[2]:
            raise ValueError("Q must have shape (slots, resources, resources)")
        if not np.allclose(q, q.swapaxes(1, 2), atol=1e-12, rtol=1e-12):
            raise ValueError("Q must be symmetric in its last two axes")
        linear = np.array(_finite("linear", self.linear), copy=True)
        if linear.shape != q.shape[:2]:
            raise ValueError("linear must have shape (slots, resources)")
        constant = float(self.constant)
        if not np.isfinite(constant) or not isinstance(self.kind, str) or not self.kind:
            raise ValueError("Finite constant and explicit nonempty kind are required")
        q = .5 * (q + q.swapaxes(1, 2))
        q.setflags(write=False)
        linear.setflags(write=False)
        object.__setattr__(self, "Q", q)
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "constant", constant)

    @property
    def T(self):
        return self.Q.shape[0]

    @property
    def R(self):
        return self.Q.shape[1]

    @property
    def shape(self):
        return self.R, self.T

    def _delivery(self, u):
        u = _finite("delivery", u)
        if u.shape != self.shape:
            raise ValueError(f"Delivery must have shape {self.shape}")
        return u

    def value(self, u):
        u = self._delivery(u)
        return float(self.constant + np.einsum("tr,rt->", self.linear, u)
                     + np.einsum("rt,trq,qt->", u, self.Q, u))

    def gradient(self, u):
        u = self._delivery(u)
        return 2 * np.einsum("trq,qt->rt", self.Q, u) + self.linear.T

    def scaled(self, factor):
        """Return a new form, e.g. the C-oriented branch of expected evidence."""
        factor = float(factor)
        if not np.isfinite(factor):
            raise ValueError("Evidence scale must be finite")
        return QuadraticEvidence(factor*self.Q, factor*self.linear,
                                 factor*self.constant, self.kind)


def build_expected(baseC, baseF, effectC, effectF, likelihood_sd,
                   response_relative_sd=0.,
                   background_residual_variance_C=0.,
                   background_residual_variance_F=0.):
    """Build E[log(likelihood_F / likelihood_C)] under supplied moments.

    Means are base_k[t] + effect_k @ u[:,t]. Independent multiplicative
    response innovations are u[r,t] * response_relative_sd[r,t] * Z[r,t];
    the SAME innovation acts on both candidates through their own effects.
    Supplied background variances are variances of candidate residuals from
    background power forecast uncertainty, independent of response innovations.
    Any shared zero-mean voltage innovation cancels from this expectation and
    must not be counted a second time. Correlated response innovations are not
    represented by this diagonal-resource variance interface.

    This expectation is a quadratic form, not a posterior-threshold probability.
    The returned form contains no prior or historical evidence contribution.
    """
    bc, bf, gc, gf, weight = _inputs(baseC, baseF, effectC, effectF, likelihood_sd)
    slots, _ = bc.shape
    resources = gc.shape[1]
    relative_sd = _broadcast("response_relative_sd", response_relative_sd,
                             (resources, slots), nonnegative=True)
    vc = _broadcast("background_residual_variance_C", background_residual_variance_C,
                    bc.shape, nonnegative=True)
    vf = _broadcast("background_residual_variance_F", background_residual_variance_F,
                    bf.shape, nonnegative=True)
    q = .5 * (np.einsum("nr,tn,nq->trq", gc, weight, gc)
              - np.einsum("nr,tn,nq->trq", gf, weight, gf))
    variance_diagonal = .5 * relative_sd.T**2 * np.einsum(
        "tn,nr->tr", weight, gc**2 - gf**2)
    indices = np.arange(resources)
    q[:, indices, indices] += variance_diagonal
    linear = np.einsum("tn,nr->tr", weight * bc, gc) - np.einsum(
        "tn,nr->tr", weight * bf, gf)
    constant = .5 * np.sum(weight * (bc**2 - bf**2 + vc - vf))
    return QuadraticEvidence(q, linear, float(constant), "expected_log_bayes_factor")


def build_separation(baseC, baseF, effectC, effectF, likelihood_sd):
    """Build nominal S=sum((mean_C-mean_F)/likelihood_sd)**2.

    This contains no random-response or background-variance addition. Exact
    two-Gaussian confidence formulas condition on a fixed power/design, so an
    expected squared separation must not silently replace this nominal S.
    """
    bc, bf, gc, gf, weight = _inputs(baseC, baseF, effectC, effectF, likelihood_sd)
    base, effect = bc - bf, gc - gf
    q = np.einsum("nr,tn,nq->trq", effect, weight, effect)
    linear = 2 * np.einsum("tn,nr->tr", weight * base, effect)
    constant = float(np.sum(weight * base**2))
    return QuadraticEvidence(q, linear, constant, "nominal_mahalanobis_separation")


def posterior_pair(rC, rF, likelihood_sd, prior_log_odds=0.,
                   history_log_bayes_factor=0.):
    """Two-candidate working posterior; last two residual axes are slots/nodes.

    The common independent Gaussian likelihood uses one fixed covariance for
    both candidates. log_odds = log(posterior_F / posterior_C). Priors and
    historical log likelihood ratios are supplied separately and counted once.
    Leading axes are independent evaluation paths. This function does not
    normalize a pool of more than two candidate topologies.
    """
    coarse, fine = _finite("rC", rC), _finite("rF", rF)
    if coarse.ndim < 2 or coarse.shape != fine.shape or not coarse.size:
        raise ValueError("Pair residuals require equal nonempty (..., slots, observations) shape")
    sd = _broadcast("likelihood_sd", likelihood_sd, coarse.shape[-2:], positive=True)
    prior, history = float(prior_log_odds), float(history_log_bayes_factor)
    if not np.isfinite([prior, history]).all():
        raise ValueError("Prior and historical log odds must be finite")
    # Factored difference of squares avoids cancellation of two large sums.
    log_bayes_factor = .5 * np.sum((coarse - fine) * (coarse + fine) / sd**2,
                                   axis=(-2, -1))
    odds = prior + history + log_bayes_factor
    if not np.isfinite(odds).all():
        raise ArithmeticError("Unresolved numerical log likelihood ratio")
    def output(value):
        return float(value) if np.ndim(value) == 0 else value
    return dict(log_odds=output(odds), log_bayes_factor=output(log_bayes_factor),
                posterior_F=output(expit(odds)), posterior_C=output(expit(-odds)))


def _eta(eta):
    eta = float(eta)
    if not np.isfinite(eta) or not .5 < eta < 1:
        raise ValueError("eta must lie strictly between 0.5 and 1")
    return eta


def separation_threshold(eta, beta, log_odds0=0.):
    """Exact nominal S threshold for BOTH possible correct-candidate events.

    Under either of two Gaussian candidate models with a shared covariance and
    fixed observed power, require probability >= beta that the TRUE candidate's
    posterior is >= eta. log_odds0 combines prior and already-frozen history.
    It is not a guarantee under random future power or model misspecification.
    """
    eta, beta, initial = _eta(eta), float(beta), float(log_odds0)
    if not np.isfinite([beta, initial]).all() or not 0 < beta < 1:
        raise ValueError("beta must lie in (0,1) and initial log odds must be finite")
    z = float(ndtri(beta))
    gamma = 2 * (float(logit(eta)) + abs(initial))
    root = np.sqrt(z*z + gamma)
    # Stable positive root if z is a large negative normal quantile.
    amplitude = z + root if z >= 0 else gamma / (root - z)
    return float(amplitude**2)


def event_probability(S, eta, log_odds0=0., state="F"):
    """Probability that the TRUE state's own posterior is >= eta.

    state='F' evaluates posterior_F under candidate F as data generator;
    state='C' evaluates posterior_C under C. These are different confidence
    events, not the probability of always choosing F under both states.
    S is nominal squared Mahalanobis separation for a deterministic design.
    """
    eta, initial = _eta(eta), float(log_odds0)
    separation = _finite("S", S)
    if np.any(separation < 0) or not np.isfinite(initial):
        raise ValueError("S must be nonnegative and initial log odds finite")
    if state not in ("C", "F"):
        raise ValueError("state must be 'C' or 'F'")
    signed = initial if state == "F" else -initial
    threshold = float(logit(eta))
    positive = separation > 0
    root = np.sqrt(separation)
    standardized = np.divide(.5*separation + signed - threshold, root,
                             out=np.zeros_like(separation), where=positive)
    result = np.where(positive, ndtr(standardized), float(signed >= threshold))
    return float(result) if result.ndim == 0 else result
