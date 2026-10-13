"""Proper continuous nominal-population EIG diagnostic, research only.

The target is the declared synthetic prior; H-only fitting can define an
importance proposal but cannot alter the prior. All likelihood constants,
proposal densities, latent time independence and shared H/N beta are explicit.
An accepted inverse-AC proof is required before building an actual H bank.
No caller is permitted to supply evaluation truth or actual future states.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.special import logsumexp, ndtr, ndtri

TOPOLOGY_NAMES = ("star", "clade_01", "clade_02", "clade_12")
CHANNELS = ("p", "q", "v", "master_p", "master_q")
HISTORY_KEYS = tuple("history_" + s for s in (*CHANNELS, "root_v"))
DEFAULT_ACTION_NAMES = ("constant_fixed", "fixed_mle_milp", "refit_milp")
LOG2PI = np.log(2 * np.pi)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _frozen(value, shape=None, name="array"):
    value = np.array(value, dtype=float, copy=True)
    if shape is not None:
        value = np.broadcast_to(value, shape).copy()
    if not np.isfinite(value).all():
        raise ValueError(name + " must be finite")
    value.setflags(write=False)
    return value


def array_hash(*arrays):
    result = sha256()
    for value in arrays:
        a = np.ascontiguousarray(value, dtype="<f8")
        result.update(str(a.shape).encode())
        result.update(a.tobytes())
    return result.hexdigest()


def log_ess(logweights, axis=-1):
    """ESS for nonnegative unnormalized weights; zero total gives exactly 0."""
    a = np.asarray(logweights, float)
    l1, l2 = logsumexp(a, axis=axis), logsumexp(2 * a, axis=axis)
    with np.errstate(invalid="ignore", over="ignore"):
        result = np.exp(2 * l1 - l2)
    return np.where(np.isfinite(l1), result, 0.)


def entropy(probabilities):
    p = np.asarray(probabilities, float)
    if np.any(p < 0) or not np.isclose(p.sum(), 1.):
        raise ValueError("entropy requires a normalized distribution")
    return float(-np.sum(p[p > 0] * np.log(p[p > 0])))


@dataclass(frozen=True)
class MeterObservation:
    p: np.ndarray
    q: np.ndarray
    v: np.ndarray
    master_p: np.ndarray
    master_q: np.ndarray
    root_v: np.ndarray | float = 1.

    def __post_init__(self):
        p = np.asarray(self.p, float)
        if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 1:
            raise ValueError("meter p must have shape (time,3)")
        for name in ("p", "q", "v"):
            object.__setattr__(self, name, _frozen(getattr(self, name), p.shape, name))
        for name in ("master_p", "master_q", "root_v"):
            object.__setattr__(self, name, _frozen(getattr(self, name), (len(p),), name))
        # The latent physical voltage magnitude is positive. An additive
        # Gaussian meter reading has support on all R, including negative
        # values: retaining it is necessary for the declared proper model.
        if np.any(self.root_v <= 0):
            raise ValueError("the known physical root magnitude must be positive")

    @property
    def t(self):
        return len(self.p)

    def identity(self):
        return array_hash(*[getattr(self, name) for name in (*CHANNELS, "root_v")])


@dataclass(frozen=True)
class MeterSpec:
    p_sigma: np.ndarray | float = .0006
    q_sigma: np.ndarray | float = .0004
    v_sigma: np.ndarray | float = .00012
    master_p_sigma: float = .001
    master_q_sigma: float = .001
    bias_p: np.ndarray | float = field(default_factory=lambda: np.linspace(-.00015, .00015, 3))
    bias_q: np.ndarray | float = field(default_factory=lambda: np.linspace(.0001, -.0001, 3))
    bias_v: np.ndarray | float = field(default_factory=lambda: np.linspace(-.00004, .00004, 3))

    def __post_init__(self):
        for name in ("p_sigma", "q_sigma", "v_sigma", "bias_p", "bias_q", "bias_v"):
            object.__setattr__(self, name, _frozen(getattr(self, name), (3,), name))
        for name in ("master_p_sigma", "master_q_sigma"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + " must be positive finite")
            object.__setattr__(self, name, value)
        if any(np.any(getattr(self, name) <= 0) for name in ("p_sigma", "q_sigma", "v_sigma")):
            raise ValueError("meter standard deviations must be positive")

    @classmethod
    def from_model(cls, model):
        if getattr(model, "fit_bias", False):
            raise ValueError("this diagnostic requires known externally calibrated bias")
        return cls(**{name: getattr(model, name) for name in cls.__dataclass_fields__})

    def identity(self):
        return array_hash(*[getattr(self, name) for name in self.__dataclass_fields__])


@dataclass(frozen=True)
class PlannedAction:
    name: str
    p: np.ndarray
    q: np.ndarray

    def __post_init__(self):
        p = np.asarray(self.p, float)
        if not isinstance(self.name, str) or not self.name or p.ndim != 2 or p.shape[1] != 3 or not len(p):
            raise ValueError("action requires a name and (time,3) p/q")
        object.__setattr__(self, "p", _frozen(p))
        object.__setattr__(self, "q", _frozen(self.q, p.shape, "action q"))

    def identity(self):
        return array_hash(self.p, self.q)

    def validate_nominal_domain(self):
        if np.any(self.q != 0) or np.any(self.p < 0) or np.any(self.p > .3):
            raise ValueError("proper nominal inverse proof requires QEV=0 and PEV in [0,.3]")
        if np.any((self.p > 0).sum(axis=1) > 1):
            raise ValueError("proper nominal inverse proof requires a single EV terminal per time")


def history_from_arrays(arrays: Mapping):
    """Exactly six lazy NPZ/Mapping accesses. Never enumerate or load future."""
    return MeterObservation(**{key.removeprefix("history_"): np.asarray(arrays[key], float).copy()
                               for key in HISTORY_KEYS})


def read_history(path):
    with np.load(path, allow_pickle=False) as given:
        return history_from_arrays(given)


def sanitized_input(payload: Mapping, action_names=DEFAULT_ACTION_NAMES):
    """Public schema has no case, true_index, truth, future, latent or noise."""
    history = MeterObservation(**payload["history"])
    meter = MeterSpec(**payload["meter"])
    actions = tuple(PlannedAction(name, payload["actions"][name]["p"], payload["actions"][name]["q"])
                    for name in action_names)
    if len({a.p.shape for a in actions}) != 1:
        raise ValueError("actions must share a future horizon")
    centers = {name: np.asarray(payload["history_centers"][name], float)
               for name in TOPOLOGY_NAMES}
    return history, meter, actions, centers


def read_sanitized_bundle(json_path, npz_path, *, meter=None, action_names=DEFAULT_ACTION_NAMES):
    """Root-sanitized H/actions only; this routine never opens original trial."""
    given = json.loads(Path(json_path).read_text(encoding="utf-8"))
    history = read_history(npz_path)
    centers = {name: given["centres"][name] for name in TOPOLOGY_NAMES}
    actions = tuple(PlannedAction(name, given["actions"][name]["p"], given["actions"][name]["q"])
                    for name in action_names)
    if len({a.p.shape for a in actions}) != 1:
        raise ValueError("sanitized actions must share a horizon")
    return history, MeterSpec() if meter is None else meter, actions, centers


@dataclass(frozen=True)
class PriorComponent:
    name: str
    topology: int
    strength: str | None
    candidate: object
    mass: float
    nominal_beta: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        dim = 2 * self.candidate.m
        for name in ("nominal_beta", "lower", "upper"):
            object.__setattr__(self, name, _frozen(getattr(self, name), (dim,), name))
        if not 0 < self.mass < 1 or np.any(self.lower >= self.upper):
            raise ValueError("component mass and box must be proper")

    @property
    def prior_logconstant(self):
        return float(-np.log(self.upper - self.lower).sum())

    def prior_logpdf(self, beta):
        b = np.asarray(beta, float)
        if b.shape[-1] != len(self.lower):
            raise ValueError("wrong beta dimension")
        supported = np.all(np.isfinite(b) & (b >= self.lower) & (b <= self.upper), axis=-1)
        return np.where(supported, self.prior_logconstant, -np.inf)

    def payload(self):
        return dict(name=self.name, topology=self.topology, strength=self.strength, mass=self.mass,
                    supports=[list(s) for s in self.candidate.supports],
                    nominal_beta=self.nominal_beta.tolist(), lower=self.lower.tolist(), upper=self.upper.tolist())


def declared_components(candidates, *, star_mass=.25, weak_mass=.5):
    """Four topologies, seven modes; star strength label integrated out."""
    if not 0 < star_mass < 1 or not 0 < weak_mass < 1:
        raise ValueError("prior masses must be interior probabilities")
    by_name = {c.name: c for c in candidates}
    if set(by_name) != set(TOPOLOGY_NAMES):
        raise ValueError("all four declared topology candidates are required")
    components = []
    for h, name in enumerate(TOPOLOGY_NAMES):
        c = by_name[name]
        strengths = (None,) if h == 0 else ("weak", "strong")
        for strength in strengths:
            internal = (.00003, .00002) if strength == "weak" else (.012, .007)
            r = [.005 if len(s) == 3 else .028 + .003 * s[0] if len(s) == 1 else internal[0]
                 for s in c.supports]
            x = [.0035 if len(s) == 3 else .011 + .002 * s[0] if len(s) == 1 else internal[1]
                 for s in c.supports]
            beta = np.r_[r, x]
            mass = star_mass if h == 0 else (1-star_mass)/3 * (weak_mass if strength == "weak" else 1-weak_mass)
            components.append(PriorComponent(name if h == 0 else name+"_"+strength,
                h, strength, c, float(mass), beta, .8 * beta, 1.2 * beta))
    if not np.isclose(sum(c.mass for c in components), 1., atol=1e-14):
        raise AssertionError("prior mass not normalized")
    return tuple(components)


@dataclass(frozen=True)
class MixtureBoxProposal:
    component: PriorComponent
    mean: np.ndarray
    sigma: np.ndarray
    uniform_mass: float = .2

    def __post_init__(self):
        d = len(self.component.lower)
        object.__setattr__(self, "mean", _frozen(self.mean, (d,), "proposal mean"))
        object.__setattr__(self, "sigma", _frozen(self.sigma, (d,), "proposal sigma"))
        if not 0 < self.uniform_mass <= 1 or np.any(self.sigma <= 0):
            raise ValueError("proposal needs positive sigma and uniform coverage mass")
        if np.any(self.mean < self.component.lower) or np.any(self.mean > self.component.upper):
            raise ValueError("proposal mean must be clipped into the prior box")

    @classmethod
    def from_history_center(cls, component, center, *, relative_sd=.08, uniform_mass=.2):
        center = np.asarray(center, float)
        if center.shape != component.nominal_beta.shape or not np.isfinite(center).all():
            raise ValueError("H-only center must match beta")
        return cls(component, np.clip(center, component.lower, component.upper),
                   relative_sd * component.nominal_beta, uniform_mass)

    def _cdf_limits(self):
        c = self.component
        lo = ndtr((c.lower-self.mean)/self.sigma)
        hi = ndtr((c.upper-self.mean)/self.sigma)
        if np.any(hi <= lo):
            raise ValueError("degenerate truncated normal normalization")
        return lo, hi

    def logpdf(self, beta):
        beta = np.asarray(beta, float)
        c = self.component
        uniform = c.prior_logpdf(beta)
        if self.uniform_mass == 1:
            return uniform
        lo, hi = self._cdf_limits()
        z = (beta-self.mean)/self.sigma
        normal = np.sum(-.5*z*z - .5*LOG2PI - np.log(self.sigma) - np.log(hi-lo), axis=-1)
        normal = np.where(np.isfinite(uniform), normal, -np.inf)
        return np.logaddexp(np.log(self.uniform_mass)+uniform,
                            np.log1p(-self.uniform_mass)+normal)

    def sample(self, count, rng):
        count = _integer(count, "theta count")
        c = self.component
        u = rng.random((count, len(c.lower)))
        beta = c.lower+(c.upper-c.lower)*u
        if self.uniform_mass < 1:
            lo, hi = self._cdf_limits()
            a = lo+(hi-lo)*rng.random(beta.shape)
            tn = self.mean+self.sigma*ndtri(np.clip(a, np.finfo(float).eps, 1-np.finfo(float).eps))
            use_normal = rng.random(count) >= self.uniform_mass
            beta[use_normal] = tn[use_normal]
        if not np.isfinite(c.prior_logpdf(beta)).all():
            raise RuntimeError("proposal generated a point outside the proper prior box")
        logq = self.logpdf(beta)
        return beta, logq

    def payload(self):
        return dict(mean=self.mean.tolist(), sigma=self.sigma.tolist(), uniform_mass=self.uniform_mass,
                    density="uniform_plus_product_truncated_normal", target_prior=self.component.name)


@dataclass(frozen=True)
class GaussianUniformProposal:
    """H-only correlated untruncated Gaussian mixture plus uniform coverage.

    Gaussian samples outside the prior are retained with target weight zero.
    No redraw or unknown multivariate truncation normalization is introduced.
    Every Gaussian is defined in phi=beta/component.nominal_beta coordinates;
    logpdf includes the phi-to-beta Jacobian exactly once.
    """
    component: PriorComponent
    means_phi: np.ndarray
    covariances_phi: np.ndarray
    uniform_mass: float = .2

    def __post_init__(self):
        means = np.asarray(self.means_phi, float)
        covariance = np.asarray(self.covariances_phi, float)
        d = len(self.component.lower)
        if means.ndim != 2 or means.shape[1] != d or not len(means):
            raise ValueError("scaled Gaussian means must have shape (center,beta_dimension)")
        if covariance.shape != (len(means), d, d) or not np.isfinite(covariance).all():
            raise ValueError("scaled covariances must match all centers")
        if not 0 < self.uniform_mass <= 1:
            raise ValueError("uniform coverage mass must be in (0,1]")
        for matrix in covariance:
            if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-10):
                raise ValueError("covariance must be symmetric")
            np.linalg.cholesky(matrix)
        object.__setattr__(self, "means_phi", _frozen(means))
        object.__setattr__(self, "covariances_phi", _frozen(covariance))

    @classmethod
    def from_history_centers(cls, component, centers, *, ridge=75., uniform_mass=.2):
        if ridge <= 0 or not np.isfinite(ridge):
            raise ValueError("positive finite scaled engineering ridge required")
        d, scale = len(component.lower), component.nominal_beta
        means, matrices = [], []
        for row in centers:
            beta = np.asarray(row["beta"], float)
            precision = np.asarray(row["profile_precision"], float)
            if beta.shape != (d,) or precision.shape != (d,d) or not np.isfinite(beta).all() or not np.isfinite(precision).all():
                raise ValueError("H-only beta/profile_precision has wrong shape")
            precision = .5*(precision+precision.T)
            scaled = scale[:, None]*precision*scale[None, :]
            regularized = scaled+ridge*np.eye(d)
            np.linalg.cholesky(regularized)
            covariance = np.linalg.solve(regularized, np.eye(d))
            # Freeze engineering center inside this mode's target box; this
            # changes only q, whose exact density is retained.
            means.append(np.clip(beta, component.lower, component.upper)/scale)
            matrices.append(.5*(covariance+covariance.T))
        return cls(component, np.array(means), np.array(matrices), uniform_mass)

    def logpdf(self, beta):
        beta = np.asarray(beta, float)
        c = self.component
        uniform = c.prior_logpdf(beta)
        if self.uniform_mass == 1:
            return uniform
        phi = beta/c.nominal_beta
        terms = []
        jac = -np.log(c.nominal_beta).sum()
        for mean, covariance in zip(self.means_phi, self.covariances_phi):
            cholesky = np.linalg.cholesky(covariance)
            delta = phi-mean
            flat = delta.reshape(-1, len(mean))
            white = np.linalg.solve(cholesky, flat.T).T.reshape(delta.shape)
            terms.append(-.5*np.sum(white**2, axis=-1)-.5*len(mean)*LOG2PI-
                         np.log(np.diag(cholesky)).sum()+jac)
        gaussian = logsumexp(np.stack(terms), axis=0)-np.log(len(terms))
        return np.logaddexp(np.log(self.uniform_mass)+uniform,
                            np.log1p(-self.uniform_mass)+gaussian)

    def sample(self, count, rng):
        count = _integer(count, "theta count")
        c = self.component
        beta = c.lower+(c.upper-c.lower)*rng.random((count, len(c.lower)))
        if self.uniform_mass < 1:
            normal_rows = rng.random(count) >= self.uniform_mass
            center = rng.integers(len(self.means_phi), size=count)
            noise = rng.standard_normal(beta.shape)
            for k, (mean, covariance) in enumerate(zip(self.means_phi, self.covariances_phi)):
                rows = normal_rows & (center == k)
                beta[rows] = (mean+noise[rows]@np.linalg.cholesky(covariance).T)*c.nominal_beta
        return beta, self.logpdf(beta)

    def payload(self):
        return dict(means_phi=self.means_phi.tolist(), covariances_phi=self.covariances_phi.tolist(),
                    uniform_mass=self.uniform_mass, density="uniform_plus_untruncated_correlated_Gaussian",
                    phi_scale=self.component.nominal_beta.tolist(), Gaussian_outside_prior="retained_zero_target_weight",
                    H_center_clipped_to_component_prior_box=True,
                    engineering_covariance_is_prior_or_confidence_region=False, target_prior=self.component.name)


@dataclass(frozen=True)
class LatentEstimate:
    loglik: np.ndarray
    log_by_time: np.ndarray
    time_ess: np.ndarray
    time_positive_count: np.ndarray
    invalid_inverse_count: int
    samples_per_time: int
    prior_outside_theta_count: int = 0
    background_outside_point_count: int = 0
    master_rectangle_outside_point_count: int = 0
    source_prior_outside_supported_point_count: int = 0

    def diagnostics(self):
        positive = self.time_positive_count > 0
        ess = self.time_ess[positive]
        with np.errstate(divide="ignore", invalid="ignore"):
            relse = np.sqrt(np.maximum(self.samples_per_time/self.time_ess-1.,0.)/
                            max(self.samples_per_time-1,1))
        joint_finite = np.isfinite(self.loglik)
        relse_finite = relse[joint_finite]
        return dict(theta_count=len(self.loglik), latent_per_time=self.samples_per_time,
            finite_joint_theta_count=int(np.isfinite(self.loglik).sum()),
            zero_theta_time_count=int((~positive).sum()), invalid_inverse_count=self.invalid_inverse_count,
            prior_outside_theta_count=self.prior_outside_theta_count,
            inverse_failure_in_supported_domain_count=0,
            inverse_failure_in_supported_domain_handling="raise; cannot become zero probability",
            invalid_inverse_count_scope="outside background/master support only, after proof-domain contract",
            background_outside_point_count=self.background_outside_point_count,
            master_rectangle_outside_point_count=self.master_rectangle_outside_point_count,
            source_prior_outside_supported_point_count=self.source_prior_outside_supported_point_count,
            min_positive_time_ess=None if not len(ess) else float(ess.min()),
            median_positive_time_ess=None if not len(ess) else float(np.median(ess)),
            min_positive_count=int(self.time_positive_count.min()),
            median_time_MC_relative_SE_on_finite_joint_theta=None if not relse_finite.size else float(np.median(relse_finite)),
            maximum_time_MC_relative_SE_on_finite_joint_theta=None if not relse_finite.size else float(relse_finite.max()),
            per_time_relative_SE_formula="sqrt((L/ESS-1)/(L-1)); sample-moment diagnostic only")


def latent_likelihood(component, beta, observation, meter, latent_normals, engine, *, action=None):
    """Unbiased product of independent-time means, if the inverse proof holds.

    latent_normals shape (T,L,8), independent across T, common across theta.
    Exactly eight metering Gaussian densities cancel their proposal.
    """
    beta = np.asarray(beta, float)
    if beta.ndim != 2 or beta.shape[1] != 2*component.candidate.m:
        raise ValueError("beta must have shape (theta,2*edge_count)")
    z = np.asarray(latent_normals, float)
    if z.ndim != 3 or z.shape[0] != observation.t or z.shape[2] != 8 or z.shape[1] < 1 or not np.isfinite(z).all():
        raise ValueError("latent normals must have shape (time,latent,8)")
    if action is not None and action.p.shape != observation.p.shape:
        raise ValueError("command must match observed future horizon")
    if np.any(observation.root_v != 1.):
        raise ValueError("proper nominal inverse proof requires the known physical root=1")
    if action is not None:
        action.validate_nominal_domain()
    supported_beta = np.isfinite(component.prior_logpdf(beta))
    if not np.all(supported_beta):
        # Untruncated Gaussian points outside the target prior remain in the
        # importance bank, but do not enter an engine certified only on target.
        n, t = len(beta), observation.t
        log_by_time = np.full((n,t), -np.inf)
        time_ess, time_count = np.zeros((n,t)), np.zeros((n,t), int)
        invalid = background_outside = master_outside = source_outside = 0
        if np.any(supported_beta):
            sub = latent_likelihood(component, beta[supported_beta], observation, meter, z, engine, action=action)
            log_by_time[supported_beta] = sub.log_by_time
            time_ess[supported_beta] = sub.time_ess
            time_count[supported_beta] = sub.time_positive_count
            invalid = sub.invalid_inverse_count
            background_outside = sub.background_outside_point_count
            master_outside = sub.master_rectangle_outside_point_count
            source_outside = sub.source_prior_outside_supported_point_count
        return LatentEstimate(log_by_time.sum(axis=1), log_by_time, time_ess, time_count,
                              invalid, z.shape[1], int((~supported_beta).sum()),
                              background_outside,master_outside,source_outside)
    total_p = (observation.p-meter.bias_p)[:, None, :] + meter.p_sigma*z[:, :, :3]
    total_q = (observation.q-meter.bias_q)[:, None, :] + meter.q_sigma*z[:, :, 3:6]
    master_p = observation.master_p[:, None] + meter.master_p_sigma*z[:, :, 6]
    master_q = observation.master_q[:, None] + meter.master_q_sigma*z[:, :, 7]
    m = component.candidate.m
    inv = engine.inverse_ac(component.candidate, beta[:, None, None, :m], beta[:, None, None, m:],
        total_p[None], total_q[None], master_p[None], master_q[None],
        root_v=observation.root_v[None, :, None])
    bgp = total_p if action is None else total_p-action.p[:, None, :]
    bgq = total_q if action is None else total_q-action.q[:, None, :]
    logbg = engine.background_logpdf(bgp, bgq)
    logsrc = engine.source_logpdf(inv["source_p"], inv["source_q"])
    jac = np.asarray(inv["jacobian_abs"], float)
    valid = np.asarray(inv["valid"], bool) & np.asarray(inv.get("high_branch", inv["valid"]), bool)
    if "certified_domain" in inv:
        valid &= np.asarray(inv["certified_domain"], bool)
    master_rectangle = (master_p[None] >= 0) & (master_p[None] <= 1) & (master_q[None] >= 0) & (master_q[None] <= .5)
    eligible = np.isfinite(logbg)[None] & master_rectangle
    physical_response_valid = (np.isfinite(inv["v"]).all(axis=-1) & (np.asarray(inv["v"]) > 0).all(axis=-1)
                               & np.isfinite(inv["source_p"]) & np.isfinite(inv["source_q"])
                               & np.isfinite(jac) & (jac > 0))
    numerical_failures = eligible & ~(valid & physical_response_valid)
    if np.any(numerical_failures):
        raise RuntimeError("inverse numerical/branch/certificate failure inside proven background/master/parameter domain; "
                           f"{int(numerical_failures.sum())} supported proposal points cannot be assigned zero target probability")
    valid &= master_rectangle
    bg_full = np.broadcast_to(logbg[None],valid.shape)
    target_supported = valid & np.isfinite(bg_full) & np.isfinite(logsrc)
    kernel = np.full(valid.shape,-np.inf)
    if np.any(target_supported):
        measured_v = np.broadcast_to((observation.v-meter.bias_v)[None,:,None,:],inv["v"].shape)
        with np.errstate(over="raise",invalid="raise",divide="raise"):
            residual_v = (measured_v[target_supported]-inv["v"][target_supported])/meter.v_sigma
            logv = np.sum(-.5*residual_v**2-np.log(meter.v_sigma)-.5*LOG2PI,axis=-1)
            target = logv+bg_full[target_supported]+logsrc[target_supported]+np.log(jac[target_supported])
        if not np.isfinite(logv).all() or not np.isfinite(target).all():
            raise FloatingPointError("nonfinite supported Gaussian target; numerical_failure, not probability zero")
        kernel[target_supported] = target
    pertime = logsumexp(kernel, axis=-1)-np.log(z.shape[1])
    return LatentEstimate(pertime.sum(axis=1), pertime, log_ess(kernel),
        np.isfinite(kernel).sum(axis=-1), int((~valid).sum()), z.shape[1],0,
        int((~np.isfinite(logbg)).sum())*len(beta),int((~master_rectangle).sum())*len(beta),
        int((eligible & valid & ~np.isfinite(logsrc)).sum()))


@dataclass(frozen=True)
class ComponentParticles:
    component: PriorComponent
    proposal: object
    beta: np.ndarray
    logq: np.ndarray
    history: LatentEstimate
    logweight: np.ndarray


@dataclass(frozen=True)
class HistoryBank:
    blocks: tuple[ComponentParticles, ...]
    log_evidence: float
    topology_mass: np.ndarray
    diagnostics: dict
    history_hash: str
    meter_hash: str
    seed: int
    proof_accepted: bool

    @property
    def posterior_entropy(self):
        return None if not np.isfinite(self.log_evidence) else entropy(self.topology_mass)

    @property
    def topology_logmass(self):
        if not np.isfinite(self.log_evidence):
            return np.full(4,-np.inf)
        return np.array([logsumexp(np.concatenate([b.logweight for b in self.blocks if b.component.topology == h]))-
                         self.log_evidence for h in range(4)])

    def weights(self):
        if not np.isfinite(self.log_evidence):
            return tuple(np.zeros(len(b.beta)) for b in self.blocks)
        return tuple(np.exp(b.logweight-self.log_evidence) for b in self.blocks)

    def summary(self):
        return dict(log_evidence=self.log_evidence, topology_mass=None if not np.isfinite(self.log_evidence) else self.topology_mass.tolist(),
            log_topology_mass=None if not np.isfinite(self.log_evidence) else self.topology_logmass.tolist(),
            topology_names=list(TOPOLOGY_NAMES), posterior_entropy=self.posterior_entropy,
            diagnostics=self.diagnostics, history_hash=self.history_hash, meter_hash=self.meter_hash,
            seed=self.seed, inverse_proof_accepted=self.proof_accepted,
            target="declared proper nominal synthetic model", H_updates=1,
            WZZT_extra_likelihood=False, mode_count=len(self.blocks),
            posterior_available=bool(np.isfinite(self.log_evidence)),
            zero_MC_mass_is_exact_posterior_probability=False)


def build_history_bank(components, proposals, history, meter, engine, *, n_theta=256,
                       n_latent=16, seed=221013, proof_accepted=False,
                       minimum_mode_ess=20., minimum_total_ess=100., maximum_weight=.05,
                       latent_evaluator=None):
    if not proof_accepted:
        raise ValueError("inverse change-of-variables proof must be accepted before actual-H integration")
    n_theta, n_latent = _integer(n_theta, "theta count"), _integer(n_latent, "latent count")
    if len(components) != 7 or len(proposals) != 7 or any(p.component is not c for c,p in zip(components, proposals)):
        raise ValueError("all seven prior modes and their matching proposals are required")
    if not np.isclose(sum(c.mass for c in components), 1., atol=1e-14):
        raise ValueError("component prior mass must sum to one")
    streams = np.random.SeedSequence(seed).spawn(8)
    # Independent time clouds, shared across beta and modes for paired diagnostics.
    z = np.random.default_rng(streams[0]).standard_normal((history.t, n_latent, 8))
    evaluate_latent = latent_likelihood if latent_evaluator is None else latent_evaluator
    blocks, rows = [], []
    for c, p, stream in zip(components, proposals, streams[1:]):
        beta, logq = p.sample(n_theta, np.random.default_rng(stream))
        result = evaluate_latent(c, beta, history, meter, z, engine)
        lw = np.log(c.mass)-np.log(n_theta)+c.prior_logpdf(beta)-logq+result.loglik
        block = ComponentParticles(c, p, _frozen(beta), _frozen(logq), result, lw)
        blocks.append(block)
        row = result.diagnostics()
        row.update(component=c.name, topology=c.topology, mode_ess=float(log_ess(lw)),
            log_mode_evidence=float(logsumexp(lw)-np.log(c.mass)),
            proposal=p.payload(), mode_preserved=True,
            numerical_zero_mode_estimate=bool(not np.isfinite(logsumexp(lw))),
            numerical_zero_is_exact_model_rejection=False)
        time_terms = (c.prior_logpdf(beta)-logq-np.log(n_theta))[:,None]+result.log_by_time
        row["log_single_time_marginal_evidence"] = logsumexp(time_terms,axis=0).tolist()
        row["single_time_marginals_are_joint_history_evidence"] = False
        rows.append(row)
    flat = np.concatenate([b.logweight for b in blocks])
    logz = float(logsumexp(flat))
    topology = np.zeros(4) if not np.isfinite(logz) else np.array([
        sum(np.exp(logsumexp(b.logweight)-logz) for b in blocks if b.component.topology == h) for h in range(4)])
    global_ess = float(log_ess(flat))
    maximum = 0. if not np.isfinite(logz) else float(np.exp(flat.max()-logz))
    problems = []
    if not np.isfinite(logz):
        problems.append("all_mode_likelihood_estimates_zero_no_posterior_available")
    if any(r["numerical_zero_mode_estimate"] for r in rows):
        problems.append("one_or_more_mode_likelihood_estimates_zero_not_exact_rejection")
    if any(r["mode_ess"] < minimum_mode_ess for r in rows):
        problems.append("one_or_more_mode_ESS_below_frozen_minimum")
    if global_ess < minimum_total_ess:
        problems.append("overall_ESS_below_frozen_minimum")
    if maximum > maximum_weight:
        problems.append("maximum_particle_weight_above_frozen_maximum")
    for b,row in zip(blocks,rows):
        posterior_weight = np.zeros(len(b.beta)) if not np.isfinite(logz) else np.exp(b.logweight-logz)
        row["numerical_posterior_mode_mass"] = float(posterior_weight.sum())
        row["conditional_particle_ESS"] = row["mode_ess"]
        mode_logtotal = float(logsumexp(b.logweight))
        if np.isfinite(mode_logtotal):
            # Conditional mode diagnostics survive global display underflow.
            # Every mode is retained even when exp(logmode-logZ) rounds to 0.
            w = np.exp(b.logweight-mode_logtotal)
            with np.errstate(divide="ignore", invalid="ignore"):
                relse = np.sqrt(np.maximum(n_latent/b.history.time_ess-1.,0.)/max(n_latent-1,1))
            # Mask zero-weight particles before summation to avoid 0*inf.
            active = w > 0
            row["H_weighted_mean_time_MC_relative_SE"] = np.sum(w[active,None]*relse[active],axis=0).tolist()
            row["time_MC_diagnostics_conditioned_on_mode_not_global_display_mass"] = True
        else:
            row["H_weighted_mean_time_MC_relative_SE"] = None
    diagnostics = dict(status="failed" if problems else "passed_initial_ESS",
        failure_reasons=problems, total_ess=global_ess, maximum_particle_weight=maximum,
        minimum_mode_ess=minimum_mode_ess, minimum_total_ess=minimum_total_ess,
        maximum_weight_limit=maximum_weight, modes=rows,
        independent_latent_clouds_across_time=True, latent_cloud_shared_across_beta=True,
        estimated_likelihood_product_unbiased=True, posterior_self_normalization_bias_present=True,
        latent_evaluator="baseline Gaussian cancellation" if latent_evaluator is None else "explicit supplied importance evaluator; inspect per-mode ratio/proposal ledger",
        finite_bank_discretization_present=True, integral_doubling_required=True)
    return HistoryBank(tuple(blocks), logz, _frozen(topology), diagnostics,
                       history.identity(), meter.identity(), int(seed), True)


def reweight_history_bank(bank, *, star_mass=.25, weak_mass=.5):
    """Only mass sensitivity: no changed beta bounds or changed densities."""
    if not 0 < star_mass < 1 or not 0 < weak_mass < 1:
        raise ValueError("mass sensitivity requires proper probabilities")
    logs, rows = [], []
    for b in bank.blocks:
        c = b.component
        mass = star_mass if c.topology == 0 else (1-star_mass)/3*(weak_mass if c.strength == "weak" else 1-weak_mass)
        logs.append(b.logweight+np.log(mass/c.mass))
        rows.append(dict(component=c.name, changed_mass=float(mass)))
    logz = float(logsumexp(np.concatenate(logs)))
    posterior = np.array([sum(np.exp(logsumexp(lw)-logz) for b,lw in zip(bank.blocks, logs) if b.component.topology == h)
                          for h in range(4)])
    return dict(star_mass=star_mass, conditional_weak_mass=weak_mass,
                log_evidence=logz, topology_mass=posterior.tolist(), posterior_entropy=entropy(posterior),
                component_masses=rows, densities_and_support_unchanged=True)


@dataclass(frozen=True)
class FutureCRN:
    block_index: np.ndarray
    theta_index: np.ndarray
    topology: np.ndarray
    base_p: np.ndarray
    base_q: np.ndarray
    source_p: np.ndarray
    source_q: np.ndarray
    meter_normals: np.ndarray


def sample_future_crn(bank, outer, horizon, rng):
    outer, horizon = _integer(outer, "outer"), _integer(horizon, "horizon")
    if not bank.proof_accepted:
        raise ValueError("unaccepted inverse bank")
    if not np.isfinite(bank.log_evidence):
        raise ValueError("all H estimates zero; no posterior predictive distribution available")
    weights = np.concatenate(bank.weights())
    selected = rng.choice(len(weights), size=outer, p=weights/weights.sum())
    limits = np.cumsum([len(b.beta) for b in bank.blocks])
    block_index = np.searchsorted(limits, selected, side="right")
    offsets = np.r_[0, limits[:-1]]
    theta_index = selected-offsets[block_index]
    topology = np.array([bank.blocks[i].component.topology for i in block_index])
    common = rng.uniform(-.012, .012, (outer, horizon, 1))
    bp = .12+common+rng.uniform(-.004, .004, (outer, horizon, 3))
    bq = .04+.25*common+rng.uniform(-.004, .004, (outer, horizon, 3))
    return FutureCRN(block_index, theta_index, topology, bp, bq,
        rng.uniform(.065, .135, (outer, horizon)), rng.uniform(.02, .05, (outer, horizon)),
        rng.standard_normal((outer, horizon, 11)))


def predictive_observations(bank, crn, action, meter, engine, *, root_v=1.):
    """Fresh hypothetical future; total P adds action once, every channel noisy."""
    action.validate_nominal_domain()
    if np.any(np.asarray(root_v) != 1.):
        raise ValueError("proper nominal forward proof requires known physical root=1")
    observations = []
    for j, (block_i, theta_i) in enumerate(zip(crn.block_index, crn.theta_index)):
        b = bank.blocks[int(block_i)]
        m, beta = b.component.candidate.m, b.beta[int(theta_i)]
        p, q = crn.base_p[j]+action.p, crn.base_q[j]+action.q
        physical = engine.forward_ac(b.component.candidate, beta[:m], beta[m:], p, q,
            crn.source_p[j], crn.source_q[j], root_v=root_v)
        physical_positive = np.isfinite(physical["v"]).all() and np.all(np.asarray(physical["v"]) > 0)
        if not np.all(physical["valid"]) or not physical_positive:
            raise RuntimeError("forward AC failed for a supported proper-prior predictive state")
        z = crn.meter_normals[j]
        observations.append(MeterObservation(
            p+meter.bias_p+meter.p_sigma*z[:, :3], q+meter.bias_q+meter.q_sigma*z[:, 3:6],
            physical["v"]+meter.bias_v+meter.v_sigma*z[:, 6:9],
            physical["master_p"]+meter.master_p_sigma*z[:, 9],
            physical["master_q"]+meter.master_q_sigma*z[:, 10], root_v))
    return tuple(observations)


def predictive_logdensities(bank, observation, action, meter, engine, latent_normals):
    """Future likelihood on the same beta particles already updated by H."""
    joint_terms, topology_terms, diagnostics = [], [[] for _ in range(4)], []
    for block in bank.blocks:
        result = latent_likelihood(block.component, block.beta, observation, meter, latent_normals,
                                   engine, action=action)
        terms = block.logweight-bank.log_evidence+result.loglik
        mode_log = float(logsumexp(terms))
        joint_terms.append(mode_log)
        topology_terms[block.component.topology].append(mode_log)
        diagnostics.append(dict(component=block.component.name, **result.diagnostics()))
    joint_log = float(logsumexp(joint_terms))
    topology_joint_log = np.array([logsumexp(terms) for terms in topology_terms])
    with np.errstate(divide="ignore", invalid="ignore"):
        # Keep conditional densities valid even if an extremely small
        # mathematical topology mass underflows when exponentiated for display.
        conditional_log = topology_joint_log-bank.topology_logmass
    return dict(log_mixture=joint_log, log_by_topology=conditional_log,
                log_topology_joint=topology_joint_log, modes=diagnostics)


def evaluate_paired_eig(bank, actions, meter, engine, *, n_outer=128, n_latent=16,
                        seed=321013, allow_failed_history=False):
    """One independent repeat, action-paired CRN, raw negative EIG retained.

    This routine is deliberately gated: a failed H bank does not trigger an
    expensive predictive run unless root explicitly requests a failure audit.
    """
    if bank.diagnostics["status"] == "failed" and not allow_failed_history:
        raise ValueError("H integration failed frozen ESS checks; predictive EIG not authorized")
    n_outer, n_latent = _integer(n_outer, "outer"), _integer(n_latent, "latent")
    actions = tuple(actions)
    if not actions or len({a.p.shape for a in actions}) != 1 or len({a.name for a in actions}) != len(actions):
        raise ValueError("a distinct-name common-horizon action set is required")
    streams = np.random.SeedSequence(seed).spawn(2)
    crn = sample_future_crn(bank, n_outer, len(actions[0].p), np.random.default_rng(streams[0]))
    z = np.random.default_rng(streams[1]).standard_normal((n_outer, len(actions[0].p), n_latent, 8))
    rows = {}
    for action in actions:
        observations = predictive_observations(bank, crn, action, meter, engine)
        scores, reductions, posterior_probabilities, density_rows = [], [], [], []
        failed = []
        for j, obs in enumerate(observations):
            given = predictive_logdensities(bank, obs, action, meter, engine, z[j])
            if not np.isfinite(given["log_mixture"]) or not np.isfinite(given["log_by_topology"][crn.topology[j]]):
                failed.append(j)
                scores.append(float("nan")); reductions.append(float("nan"))
                posterior_probabilities.append([float("nan")]*4)
            else:
                prob = np.exp(given["log_topology_joint"]-given["log_mixture"])
                scores.append(float(given["log_by_topology"][crn.topology[j]]-given["log_mixture"]))
                reductions.append(bank.posterior_entropy-entropy(prob))
                posterior_probabilities.append(prob.tolist())
            density_rows.append(dict(log_mixture=given["log_mixture"],
                log_by_topology=given["log_by_topology"].tolist(), modes=given["modes"]))
        value = np.asarray(scores)
        # A failed sample is retained; the reported estimator is unavailable,
        # rather than conditioned on a success subset.
        usable = not failed
        rows[action.name] = dict(status="completed_needs_doubling" if usable else "failed",
            eig=None if not usable else float(value.mean()),
            outer_standard_error=None if not usable or n_outer < 2 else float(value.std(ddof=1)/np.sqrt(n_outer)),
            entropy_reduction_estimate=None if not usable else float(np.mean(reductions)),
            raw_logratio_scores=scores, raw_entropy_reductions=reductions,
            future_topology_posterior=posterior_probabilities, density_diagnostics=density_rows,
            failed_outer_indices=failed, action_hash=action.identity(), negative_EIG_clipped=False)
    differences = {}
    for i, left in enumerate(actions):
        for right in actions[i+1:]:
            a, b = rows[left.name], rows[right.name]
            if a["status"] == "failed" or b["status"] == "failed":
                differences[left.name+"_minus_"+right.name] = dict(status="failed", mean=None)
            else:
                delta = np.asarray(a["raw_logratio_scores"])-np.asarray(b["raw_logratio_scores"])
                differences[left.name+"_minus_"+right.name] = dict(status="needs_independent_repeats_and_doubling",
                    mean=float(delta.mean()), standard_error=None if n_outer < 2 else float(delta.std(ddof=1)/np.sqrt(n_outer)),
                    raw_difference=delta.tolist())
    return dict(seed=int(seed), outer=n_outer, latent_per_time=n_latent,
        status="failed" if any(row["status"] == "failed" for row in rows.values()) else "completed_needs_doubling",
        history=bank.summary(), actions=rows, paired_differences=differences,
        sampled_predictive_topology=crn.topology.tolist(), action_paired_CRN=True,
        hypothetical_future_only=True, actual_evaluation_future_used=False,
        nested_log_bias_present=True, posterior_particle_discretization_present=True,
        outer_SE_does_not_bound_nested_bias=True, negative_EIG_clipped=False)


def save_particle_bank(path, bank):
    """Exact random bank for replay/doubling diagnostics; no latent truth."""
    arrays = {}
    for i, b in enumerate(bank.blocks):
        arrays[f"beta_{i}"] = b.beta
        arrays[f"logq_{i}"] = b.logq
        arrays[f"loghistory_{i}"] = b.history.loglik
        arrays[f"loghistory_time_{i}"] = b.history.log_by_time
        arrays[f"logweight_{i}"] = b.logweight
        arrays[f"latent_ess_{i}"] = b.history.time_ess
        arrays[f"latent_positive_count_{i}"] = b.history.time_positive_count
    np.savez_compressed(path, **arrays)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
