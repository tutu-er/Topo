"""Correlated Gaussian delivery multipliers with shared underlying innovations.

All nodes are drawn before any active-node subset is selected.  One innovation
array can be reused for every policy and for different response-law parameters.
The multiplier is mean + relative_sd * Z, where Var(Z_it)=1.  `relative_sd`
is the standard deviation relative to the nominal multiplier one; it is NOT
the coefficient of variation when the multiplier mean differs from one.

Innovation columns are ordered as day, node effects, common-AR innovations,
then node-major node-AR innovations.  Each AR starts at N(0,1) and is stationary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping
import math
import numbers

import numpy as np


BASE_RELATIVE_SD = math.sqrt(0.025 / 3.0)


@dataclass(frozen=True)
class ResponseLaw:
    name: str
    mean: float
    weights: tuple[float, float, float, float]
    phi: float
    relative_sd: float = BASE_RELATIVE_SD

    def __post_init__(self):
        weights = tuple(float(w) for w in self.weights)
        if len(weights) != 4 or not all(math.isfinite(w) and w >= 0 for w in weights):
            raise ValueError("four finite nonnegative variance weights required")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("variance weights must sum to one")
        if not math.isfinite(self.phi) or abs(self.phi) >= 1:
            raise ValueError("stationary AR coefficient must satisfy abs(phi)<1")
        if not math.isfinite(self.mean) or self.mean <= 0:
            raise ValueError("positive finite multiplier mean required")
        if not math.isfinite(self.relative_sd) or self.relative_sd < 0:
            raise ValueError("finite nonnegative nominal-relative SD required")
        object.__setattr__(self, "weights", weights)

    @property
    def coefficient_of_variation(self):
        return self.relative_sd / self.mean

    def to_dict(self):
        result = asdict(self)
        result["weights"] = list(self.weights)
        result["coefficient_of_variation"] = self.coefficient_of_variation
        result["multiplier_formula"] = "mean + relative_sd * Z"
        return result


MAIN_LAW = ResponseLaw("main", 1.0, (.15, .25, .20, .40), .8)
MISSPEC_LAW = ResponseLaw("misspec", .9, (.30, .30, .25, .15), .9)


def resolve_law(law=MAIN_LAW):
    if isinstance(law, ResponseLaw):
        return law
    if isinstance(law, str):
        if law == "main":
            return MAIN_LAW
        if law == "misspec":
            return MISSPEC_LAW
        raise ValueError("unknown response law")
    if isinstance(law, Mapping):
        # The two derived fields emitted by to_dict are not input parameters.
        fields = {k: v for k, v in law.items() if k not in ("coefficient_of_variation", "multiplier_formula")}
        return ResponseLaw(**fields)
    raise TypeError("law must be ResponseLaw, a declared name, or a mapping")


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def innovation_dimension(n, T):
    n = _positive_integer(n, "n")
    T = _positive_integer(T, "T")
    return 1 + n + T + n * T


def draw_base_normals(seed, samples, n, T):
    """Generate independent underlying innovations, without selecting nodes."""
    samples = _positive_integer(samples, "samples")
    dimension = innovation_dimension(n, T)
    return np.random.default_rng(seed).standard_normal((samples, dimension))


def _stationary_ar(innovations, phi):
    """The last axis is time; its first input is the stationary initial state."""
    output = np.empty_like(innovations)
    output[..., 0] = innovations[..., 0]
    scale = math.sqrt(1.0 - phi * phi)
    for t in range(1, innovations.shape[-1]):
        output[..., t] = phi * output[..., t - 1] + scale * innovations[..., t]
    return output


def standardized_from_base(base_normal, n, T, law=MAIN_LAW):
    """Deterministic linear transform to standardized Z, shape (samples,n,T)."""
    n = _positive_integer(n, "n")
    T = _positive_integer(T, "T")
    law = resolve_law(law)
    base = np.asarray(base_normal, dtype=float)
    if base.ndim != 2 or base.shape[1] != innovation_dimension(n, T):
        raise ValueError("base_normal must have shape (samples,1+n+T+n*T)")
    if not np.isfinite(base).all():
        raise ValueError("base_normal must be finite")
    g_day = base[:, :1, None]
    g_node = base[:, 1:1+n, None]
    ar_common = _stationary_ar(base[:, 1+n:1+n+T], law.phi)[:, None, :]
    ar_node = _stationary_ar(base[:, 1+n+T:].reshape(len(base), n, T), law.phi)
    wg, wi, wc, we = law.weights
    return math.sqrt(wg) * g_day + math.sqrt(wi) * g_node + math.sqrt(wc) * ar_common + math.sqrt(we) * ar_node


def multipliers_from_base(base_normal, n, T, law=MAIN_LAW):
    """Delivery multiplier; neither clipped nor resampled at physical bounds."""
    law = resolve_law(law)
    return law.mean + law.relative_sd * standardized_from_base(base_normal, n, T, law)


def draw(seed, samples, n, T, law=MAIN_LAW):
    """Return multiplier samples, shape (samples,n,T), using full-node draws.

    Use the same seed, dimensions, and sample count for paired law comparisons.
    For multiple policies, reuse this returned array instead of redrawing it.
    """
    return multipliers_from_base(draw_base_normals(seed, samples, n, T), n, T, law)


def linear_factor(n, T, law=MAIN_LAW):
    """Exact factor F for node-major flattened Z=F*xi; intended for small audits."""
    dimension = innovation_dimension(n, T)
    return standardized_from_base(np.eye(dimension), n, T, law).reshape(dimension, n*T).T


def analytic_covariance(n, T, law=MAIN_LAW, *, multiplier=False):
    """Exact covariance in node-major order; no Monte Carlo estimation."""
    innovation_dimension(n, T)
    law = resolve_law(law)
    nodes = np.repeat(np.arange(n), T)
    slots = np.tile(np.arange(T), n)
    same_node = nodes[:, None] == nodes[None, :]
    lag = np.abs(slots[:, None] - slots[None, :])
    wg, wi, wc, we = law.weights
    covariance = wg + wi * same_node + (wc + we * same_node) * np.power(law.phi, lag)
    return covariance * law.relative_sd**2 if multiplier else covariance
