"""Fixed-covariance Gaussian posterior bounds as quadratic constraints.

All inputs are already whitened by the common voltage likelihood standard
deviations. ``mean_base`` is (slots, observations), ``mean_effect`` is
(slots, observations, resources), and ``covariance`` is the PRICE-INDEPENDENT
covariance of the whitened candidate-voltage difference, (slots, nodes, nodes).
``build_fixed_f`` declares independent time blocks. ``build_fixed_f_full``
accepts a full time-by-node covariance and preserves cross-slot correlation.
Singular power covariance is allowed; I + covariance/4 is positive definite.
The returned score is a SUM over all slots, not a mean.

These are sufficient bounds under a frozen two-candidate working model with
the fine candidate as generator. They do not calibrate topology truth or make
the old MAE acceptance event a Gaussian posterior event.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh, solve_triangular
from scipy.special import expit, logit

try:
    from .core import QuadraticEvidence
except ImportError:
    from core import QuadraticEvidence


def _finite(name, value):
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(name + " must contain finite numbers")
    return array


def _probability(name, value, *, lower=0.):
    value = float(value)
    if not np.isfinite(value) or not lower < value < 1.:
        raise ValueError(f"{name} must lie in ({lower},1)")
    return value


def fixed_covariance(covariance):
    """Validate and copy fixed block-time PSD covariance.

    Roundoff eigenvalues within 1e-12 of the block spectral scale are projected
    to zero. Material negative eigenvalues or asymmetric blocks are rejected.
    No diagonal, positive-variance, or positive line-weight floor is added.
    """
    value = np.array(_finite("covariance", covariance), copy=True)
    if (value.ndim != 3 or not value.size or value.shape[1] != value.shape[2]):
        raise ValueError("covariance must have nonempty (slots,nodes,nodes) shape")
    if not np.allclose(value, value.swapaxes(1, 2), atol=1e-12, rtol=1e-12):
        raise ValueError("Each covariance block must be symmetric")
    value = .5 * (value + value.swapaxes(1, 2))
    for index, block in enumerate(value):
        eig, vectors = np.linalg.eigh(block)
        tolerance = 1e-12 * max(1., float(np.max(np.abs(eig))))
        if eig.min() < -tolerance:
            raise ValueError("Each covariance block must be positive semidefinite")
        if eig.min() < 0.:
            value[index] = (vectors * np.maximum(eig, 0.)) @ vectors.T
    return value


def fixed_covariance_sha256(covariance):
    """Hash canonical fixed covariance, independent of delivery or price."""
    value = np.ascontiguousarray(fixed_covariance(covariance), dtype="<f8")
    digest = hashlib.sha256(json.dumps(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def full_covariance(covariance):
    """Copy and validate full time-major covariance, including cross-slot terms.

    A negative minimum eigenvalue within the 1e-12 matrix-infinity-norm
    roundoff tolerance is shifted to zero. This is a numerical covariance
    repair, not a positive physical-noise floor or a line-weight constraint.
    """
    value = np.array(_finite("covariance_full", covariance), copy=True)
    if value.ndim != 2 or not value.size or value.shape[0] != value.shape[1]:
        raise ValueError("covariance_full must be a nonempty square matrix")
    if not np.allclose(value,value.T,atol=1e-12,rtol=1e-12):
        raise ValueError("covariance_full must be symmetric")
    value = .5*(value+value.T)
    minimum = float(eigh(value,subset_by_index=(0,0),eigvals_only=True,
                         check_finite=False)[0])
    tolerance = 1e-12*max(1.,float(np.linalg.norm(value,ord=np.inf)))
    if minimum < -tolerance:
        raise ValueError("covariance_full must be positive semidefinite")
    if minimum < 0.:
        value.flat[::len(value)+1] -= minimum
    return value


@dataclass(frozen=True)
class DenseQuadraticEvidence:
    """Dense score c + l.T vec(u) + vec(u).T Q vec(u).

    Delivery u has shape (resources,slots); vec(u)=u.ravel() is resource-major.
    This ordering differs from the time-major stacked voltage observations.
    It permits full time correlation without pretending independent blocks.
    """
    Q: np.ndarray
    linear: np.ndarray
    constant: float
    resources: int
    slots: int
    kind: str = "fixed_f_full_negative_log_laplace_transform"

    def __post_init__(self):
        if (isinstance(self.resources,bool) or isinstance(self.slots,bool)
                or int(self.resources)!=self.resources or int(self.slots)!=self.slots
                or self.resources<1 or self.slots<1):
            raise ValueError("Positive integer resource and slot counts required")
        size=int(self.resources)*int(self.slots)
        q=np.array(_finite("Q",self.Q),copy=True)
        linear=np.array(_finite("linear",self.linear),copy=True)
        if q.shape!=(size,size) or linear.shape!=(size,):
            raise ValueError("Dense quadratic dimensions differ from resources*slots")
        if not np.allclose(q,q.T,atol=1e-12,rtol=1e-12):
            raise ValueError("Dense Q must be symmetric")
        constant=float(self.constant)
        if not np.isfinite(constant) or not isinstance(self.kind,str) or not self.kind:
            raise ValueError("Finite constant and explicit kind required")
        q=.5*(q+q.T);q.setflags(write=False);linear.setflags(write=False)
        object.__setattr__(self,"Q",q)
        object.__setattr__(self,"linear",linear)
        object.__setattr__(self,"constant",constant)
        object.__setattr__(self,"resources",int(self.resources))
        object.__setattr__(self,"slots",int(self.slots))

    @property
    def R(self):return self.resources

    @property
    def T(self):return self.slots

    @property
    def shape(self):return self.R,self.T

    def _delivery(self,u):
        value=_finite("delivery",u)
        if value.shape!=self.shape:
            raise ValueError(f"Delivery must have shape {self.shape}")
        return value.ravel()

    def value(self,u):
        vector=self._delivery(u)
        return float(self.constant+self.linear@vector+vector@self.Q@vector)

    def gradient(self,u):
        vector=self._delivery(u)
        return (self.linear+2*self.Q@vector).reshape(self.shape)

    def scaled(self,factor):
        factor=float(factor)
        if not np.isfinite(factor):raise ValueError("Finite evidence scale required")
        return DenseQuadraticEvidence(factor*self.Q,factor*self.linear,
            factor*self.constant,self.R,self.T,self.kind)


def build_fixed_f_full(mean_base,mean_effect,covariance_full):
    """Full fixed covariance score, retaining all cross-time correlation.

    Inputs mean_base[T,N], mean_effect[T,N,R], covariance_full[T*N,T*N].
    Covariance rows use time-major observation order. Dense evidence variables
    use resource-major delivery order. No correlation or covariance blocks are
    discarded. This is the dense counterpart of build_fixed_f.
    """
    base=_finite("mean_base",mean_base)
    effect=_finite("mean_effect",mean_effect)
    if base.ndim!=2 or not base.size:
        raise ValueError("mean_base must have nonempty (slots,nodes) shape")
    slots,nodes=base.shape
    if effect.ndim!=3 or effect.shape[:2]!=base.shape or effect.shape[2]<1:
        raise ValueError("mean_effect must have (slots,nodes,resources) shape")
    resources=effect.shape[2]
    cov=full_covariance(covariance_full)
    size=slots*nodes
    if cov.shape!=(size,size):
        raise ValueError("Full covariance differs from stacked observation shape")
    design=np.zeros((size,resources*slots))
    for t in range(slots):
        rows=slice(t*nodes,(t+1)*nodes)
        columns=np.arange(resources)*slots+t
        design[rows,columns]=effect[t]
    chol=np.linalg.cholesky(np.eye(size)+cov/4.)
    transformed=solve_triangular(chol,np.column_stack((base.ravel(),design)),
                                 lower=True,check_finite=False)
    b,g=transformed[:,0],transformed[:,1:]
    return DenseQuadraticEvidence(g.T@g/8.,g.T@b/4.,
        float(b@b)/8.+float(np.log(np.diag(chol)).sum()),resources,slots)


def build_fixed_f(mean_base, mean_effect, covariance):
    """Return T(u) = -log E exp(-||Z(u)||^2/8) as QuadraticEvidence.

    Z_t has mean mean_base[t] + mean_effect[t] @ u[:,t] and fixed covariance
    covariance[t]. Cholesky solves are used; no explicit matrix inverse is
    formed. With H_t=I+C_t/4,

      T = sum_t [m_t.T H_t^-1 m_t / 8 + logdet(H_t) / 2].

    Fixed C is part of the input contract. Passing a different C for different
    prices is a different model, not an evaluation of this quadratic.
    """
    base = _finite("mean_base", mean_base)
    effect = _finite("mean_effect", mean_effect)
    if base.ndim != 2 or not base.size:
        raise ValueError("mean_base must have nonempty (slots,nodes) shape")
    slots, nodes = base.shape
    if effect.ndim != 3 or effect.shape[:2] != base.shape or effect.shape[2] < 1:
        raise ValueError("mean_effect must have (slots,nodes,resources) shape")
    cov = fixed_covariance(covariance)
    if cov.shape != (slots, nodes, nodes):
        raise ValueError("Covariance blocks differ from the mean observation shape")
    resources = effect.shape[2]
    q = np.empty((slots, resources, resources))
    linear = np.empty((slots, resources))
    constant = 0.
    identity = np.eye(nodes)
    for t in range(slots):
        chol = np.linalg.cholesky(identity + cov[t]/4.)
        whitened_base = np.linalg.solve(chol, base[t])
        whitened_effect = np.linalg.solve(chol, effect[t])
        q[t] = whitened_effect.T @ whitened_effect / 8.
        linear[t] = whitened_effect.T @ whitened_base / 4.
        constant += (float(whitened_base @ whitened_base)/8.
                     + float(np.log(np.diag(chol)).sum()))
    return QuadraticEvidence(q, linear, constant,
        "fixed_f_negative_log_laplace_transform")


def thresholds(q_star=.9, eta=.9, beta=.9, log_odds0=0.):
    """Score thresholds for two DIFFERENT sufficient posterior requirements.

    mean: E[q_F | F] >= q_star.
    probability: Pr(q_F >= eta | F) >= beta.
    log_odds0 contains prior plus historical evidence, each counted once.
    """
    q_star = _probability("q_star", q_star)
    eta = _probability("eta", eta, lower=.5)
    beta = _probability("beta", beta)
    initial = float(log_odds0)
    if not np.isfinite(initial):
        raise ValueError("log_odds0 must be finite")
    return dict(mean=-math.log(2.)-math.log1p(-q_star)-initial/2.,
                probability=(float(logit(eta))-initial)/2.-math.log1p(-beta))


def guarantees(score, log_odds0=0., eta=.9):
    """Stable sufficient lower bounds, scalar or array, conditional on F.

    score is T=-log B, not a posterior or a probability. Lower bounds are
    clipped to [0,1]; their raw log failure bounds are retained for audit.
    """
    value = _finite("score", score)
    if np.any(value < -1e-10):
        raise ValueError("A fixed-F score cannot be materially negative")
    value = np.maximum(value, 0.)
    initial = float(log_odds0)
    if not np.isfinite(initial):
        raise ValueError("log_odds0 must be finite")
    eta = _probability("eta", eta, lower=.5)
    log_mean_failure = -math.log(2.)-initial/2.-value
    log_event_failure = (float(logit(eta))-initial)/2.-value
    mean_lower = -np.expm1(np.minimum(log_mean_failure, 0.))
    event_lower = -np.expm1(np.minimum(log_event_failure, 0.))
    scalar = lambda array: float(array) if np.ndim(array) == 0 else array
    return dict(mean_posterior_lower_bound=scalar(mean_lower),
                posterior_event_lower_bound=scalar(event_lower),
                log_mean_failure_upper_bound=scalar(log_mean_failure),
                log_event_failure_upper_bound=scalar(log_event_failure))


def conditional_mean_posterior(S, log_odds0=0., state="F", *, epsabs=1e-11):
    """Adaptive deterministic integral for a fixed realized separation S.

    state='F': E[q_F | F,S]. state='C': E[q_C | C,S]. This matches the
    correct-candidate semantics of core.event_probability. It is not the
    posterior at expected log odds. A random-power evaluation still needs its
    own outer expectation over S; this routine is not used by the QCQP.
    """
    values = _finite("S", S)
    if np.any(values < 0.):
        raise ValueError("S must be nonnegative")
    initial = float(log_odds0)
    if not np.isfinite(initial) or not np.isfinite(epsabs) or epsabs <= 0.:
        raise ValueError("Finite log odds and positive integration tolerance required")
    if state not in ("F", "C"):
        raise ValueError("state must be F or C")
    initial = initial if state == "F" else -initial
    result = np.empty_like(values)
    normalizer = math.sqrt(2.*math.pi)
    for index in np.ndindex(values.shape):
        separation = float(values[index])
        if separation == 0.:
            result[index] = expit(initial)
            continue
        mean = initial + separation/2.
        scale = math.sqrt(separation)
        integrand = lambda z: (float(expit(mean+scale*z))
                               * math.exp(-z*z/2.)/normalizer)
        integral, error = quad(integrand, -np.inf, np.inf,
                               epsabs=epsabs, epsrel=epsabs, limit=200)
        if error > max(1e-10, 10.*epsabs):
            raise ArithmeticError("Conditional posterior quadrature did not converge")
        result[index] = integral
    return float(result) if result.ndim == 0 else result
