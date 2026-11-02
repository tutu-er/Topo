"""One-dimensional conditional Gaussian integration for signed L1 sums.

Compute P(sum_i w_i * abs(a_i + b_i Z) > h), Z ~ N(0, 1).
The event is integrated exactly by linear pieces, up to floating-point error.
No residual signs are frozen and small nonzero slopes are not rounded to zero.

The returned derivatives are conditional derivatives only.  Differentiating an
outer expectation additionally requires a valid interchange argument; pointwise
almost-everywhere differentiability alone is insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.special import erf, log_ndtr


@dataclass(frozen=True, slots=True)
class ConditionalIntegralResult:
    probability: float
    gradient: np.ndarray | None
    coeff_grad_a: np.ndarray | None
    coeff_grad_b: np.ndarray | None
    gradient_valid: bool
    intervals: tuple[tuple[float, float], ...]
    threshold_roots: tuple[float, ...]
    root_slopes: tuple[float, ...]
    diagnostics: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        if key not in self.__dataclass_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class _CompensatedSum:
    """Neumaier accumulator; retains small updates without zero tolerances."""

    __slots__ = ("total", "correction")

    def __init__(self, value: float):
        self.total = float(value)
        self.correction = 0.0

    def add(self, value: float) -> None:
        value = float(value)
        new = self.total + value
        if abs(self.total) >= abs(value):
            self.correction += (self.total - new) + value
        else:
            self.correction += (value - new) + self.total
        self.total = new

    @property
    def value(self) -> float:
        return self.total + self.correction


def _normal_mass(left: float, right: float) -> float:
    """Stable Phi(right)-Phi(left), including positive/negative far tails."""
    if not left < right:
        return 0.0
    if left == -math.inf and right == math.inf:
        return 1.0
    if -1.0 <= left <= right <= 1.0:
        return float(0.5 * (erf(right / math.sqrt(2.0)) - erf(left / math.sqrt(2.0))))
    if left >= 0.0:
        log_large, log_small = float(log_ndtr(-left)), float(log_ndtr(-right))
    elif right <= 0.0:
        log_large, log_small = float(log_ndtr(right)), float(log_ndtr(left))
    else:
        return float(0.5 * (erf(right / math.sqrt(2.0)) - erf(left / math.sqrt(2.0))))
    if log_large == -math.inf:
        return 0.0
    return math.exp(log_large) * (-math.expm1(log_small - log_large))


def _vector(value: Any, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return value


def conditional_probability(
    a: Any,
    b: Any,
    w: Any,
    h: float,
    da: Any | None = None,
    db: Any | None = None,
    *,
    coefficient_gradients: bool = True,
) -> ConditionalIntegralResult:
    """Integrate a strict signed-L1 event and optionally differentiate it.

    Parameters
    ----------
    a, b, w : finite arrays, shape (n,)
        Coefficients of sum(w * abs(a + b*z)).  Zero weights are retained in
        derivative output and omitted from the event scan.
    h : finite float
        Fixed threshold, in the same units as the signed sum.
    da, db : optional finite arrays, shape (n, d)
        Derivatives of a/b with respect to d controls.  A missing member is
        interpreted as zero if the other is supplied.  w and h are fixed.
    coefficient_gradients : bool
        Return dP/da and dP/db.  Their construction uses sorted roots and prefix
        sums, not an n-by-number_of_roots sign matrix.  Disable for probability
        only calls.  With da/db supplied the internal coefficient derivatives
        are still computed for the chain rule.

    Returns
    -------
    ConditionalIntegralResult
        gradient_valid is false and derivative arrays are None when a threshold
        platform, a threshold at/within roundoff of a residual kink, or a
        relevant zero residual prevents this smooth derivative formula.
        Such diagnostics do not replace the strict probability by a tolerance
        test.  Nonzero slopes are never set to zero.

    Complexity
    ----------
    O(n log n) probability and coefficient gradients, O(n*d) optional chain
    rule.  The outer expectation gradient has additional regularity conditions.
    """
    a = _vector(a, "a")
    b = _vector(b, "b")
    w = _vector(w, "w")
    if a.shape != b.shape or a.shape != w.shape:
        raise ValueError("a, b and w must have identical shapes")
    h = float(h)
    if not math.isfinite(h):
        raise ValueError("h must be finite")
    n = a.size
    d = None
    derivatives: list[np.ndarray | None] = []
    for value, name in ((da, "da"), (db, "db")):
        if value is None:
            derivatives.append(None)
            continue
        value = np.asarray(value, dtype=float)
        if value.ndim != 2 or value.shape[0] != n or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite and have shape (n,d)")
        if d is not None and value.shape[1] != d:
            raise ValueError("da and db must have the same control dimension")
        d = value.shape[1]
        derivatives.append(value)
    da, db = derivatives
    need_gradient = coefficient_gradients or d is not None

    # A common positive scaling leaves the event invariant.  It is treated as
    # a fixed numerical scale in the derivative chain rule at this evaluation.
    scale = max(float(np.max(np.abs(a), initial=0.0)),
                float(np.max(np.abs(b), initial=0.0)), abs(h))
    if scale == 0.0:
        scale = 1.0
    aa, bb, hh = a / scale, b / scale, h / scale
    active = w != 0.0
    variable = active & (bb != 0.0)
    indices = np.flatnonzero(variable)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        knot_values = -aa[indices] / bb[indices]
    if not np.all(np.isfinite(knot_values)):
        raise FloatingPointError("a/b produces non-finite knots; rescale/reparameterize coefficients")
    order = np.argsort(knot_values, kind="stable")
    indices, knots = indices[order], knot_values[order]
    knot_list = knots.tolist()
    slope_updates = (2.0 * w[indices] * np.abs(bb[indices])).tolist()
    intercept_updates = (2.0 * w[indices] * np.sign(bb[indices]) * aa[indices]).tolist()

    initial_slope = math.fsum((-w[variable] * np.abs(bb[variable])).tolist())
    initial_intercept = math.fsum(
        np.concatenate((-w[variable] * np.sign(bb[variable]) * aa[variable],
                        w[active & ~variable] * np.abs(aa[active & ~variable]))).tolist())
    slope_sum, intercept_sum = _CompensatedSum(initial_slope), _CompensatedSum(initial_intercept)
    if not math.isfinite(initial_slope) or not math.isfinite(initial_intercept):
        raise FloatingPointError("weighted coefficients overflow")

    intervals: list[tuple[float, float]] = []
    roots: list[float] = []
    slopes: list[float] = []
    reasons: set[str] = set()
    exact_platform_count = 0
    near_platform_count = 0
    threshold_kink_count = 0
    unique_knot_count = 0
    eps = np.finfo(float).eps

    zero_residual = active & (aa == 0.0) & (bb == 0.0)
    if need_gradient and np.any(zero_residual):
        # Full coefficient derivatives do not exist in general at an identically
        # zero residual.  For a restricted chain rule, fixed zero rows are safe.
        moving = coefficient_gradients
        if not moving and da is not None:
            moving = bool(np.any(da[zero_residual] != 0.0))
        if not moving and db is not None:
            moving = bool(np.any(db[zero_residual] != 0.0))
        if moving:
            reasons.add("identically_zero_residual")

    def add_interval(left: float, right: float) -> None:
        if not left < right:
            return
        if intervals and intervals[-1][1] == left:
            intervals[-1] = (intervals[-1][0], right)
        else:
            intervals.append((left, right))

    def scan_interval(left: float, right: float, slope: float, intercept: float) -> None:
        nonlocal exact_platform_count, near_platform_count
        if slope == 0.0:
            if intercept == hh:
                exact_platform_count += 1
                reasons.add("threshold_platform")
            elif abs(intercept - hh) <= 32.0 * eps * (abs(intercept) + abs(hh)):
                near_platform_count += 1
                reasons.add("near_threshold_platform")
            if intercept > hh:
                add_interval(left, right)
            return
        root = (hh - intercept) / slope
        if left < root < right:
            roots.append(root)
            slopes.append(slope)
        if slope > 0.0:
            add_interval(max(left, root), right)
        else:
            add_interval(left, min(right, root))

    left = -math.inf
    j = 0
    while j < indices.size:
        knot = knot_list[j]
        slope, intercept = slope_sum.value, intercept_sum.value
        scan_interval(left, knot, slope, intercept)
        value = slope * knot + intercept
        # Conservative derivative diagnostic only; no probability snapping.
        error_scale = abs(slope * knot) + abs(intercept) + abs(hh)
        if value == hh or abs(value - hh) <= 32.0 * eps * error_scale:
            threshold_kink_count += 1
            reasons.add("threshold_at_or_near_kink")
        end = j + 1
        while end < indices.size and knot_list[end] == knot:
            end += 1
        if end == j + 1:
            slope_sum.add(slope_updates[j])
            intercept_sum.add(intercept_updates[j])
        else:
            slope_sum.add(math.fsum(slope_updates[j:end]))
            intercept_sum.add(math.fsum(intercept_updates[j:end]))
        unique_knot_count += 1
        left = knot
        j = end
    scan_interval(left, math.inf, slope_sum.value, intercept_sum.value)

    probability_unclipped = math.fsum(_normal_mass(l, r) for l, r in intervals)
    probability = min(1.0, max(0.0, probability_unclipped))
    ga = gb = gradient = None
    gradient_valid = bool(need_gradient and not reasons)
    if gradient_valid:
        ga, gb = np.zeros(n), np.zeros(n)
        if roots:
            rr, ss = np.asarray(roots), np.asarray(slopes)
            # These roots are already ordered by the interval scan.  Coefficient
            # signs change only at their own knots.  Prefix sums thus replace an
            # O(n*number_of_roots) root-sign matrix by O(n log n) queries.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                densities = np.exp(-0.5 * rr * rr) / math.sqrt(2.0 * math.pi)
                factors = densities / np.abs(ss)
                weighted_roots = factors * rr
            if not np.all(np.isfinite(factors)) or not np.all(np.isfinite(weighted_roots)):
                reasons.add("nonfinite_gradient_factor")
                gradient_valid = False
            else:
                prefix = np.concatenate(([0.0], np.cumsum(factors)))
                prefix_z = np.concatenate(([0.0], np.cumsum(weighted_roots)))
                count = np.searchsorted(rr, knots, side="left")
                signs = np.sign(bb[indices])
                ga[indices] = w[indices] * signs * (prefix[-1] - 2.0 * prefix[count])
                gb[indices] = w[indices] * signs * (prefix_z[-1] - 2.0 * prefix_z[count])
                const = active & ~variable
                ga[const] = w[const] * np.sign(aa[const]) * prefix[-1]
                gb[const] = w[const] * np.sign(aa[const]) * prefix_z[-1]
                ga /= scale
                gb /= scale
        if gradient_valid and d is not None:
            gradient = np.zeros(d)
            if da is not None:
                gradient += ga @ da
            if db is not None:
                gradient += gb @ db
        if gradient_valid and not (np.all(np.isfinite(ga)) and np.all(np.isfinite(gb))
                                   and (gradient is None or np.all(np.isfinite(gradient)))):
            reasons.add("nonfinite_gradient")
            gradient_valid = False
        if not gradient_valid:
            ga = gb = gradient = None

    if not coefficient_gradients:
        ga = gb = None
    diagnostics = {
        "n_terms": n,
        "active_terms": int(np.count_nonzero(active)),
        "nonzero_slopes": int(indices.size),
        "unique_knots": unique_knot_count,
        "transverse_roots": len(roots),
        "threshold_platforms": exact_platform_count,
        "near_threshold_platforms": near_platform_count,
        "threshold_kinks": threshold_kink_count,
        "identically_zero_residuals": int(np.count_nonzero(zero_residual)),
        "gradient_requested": need_gradient,
        "gradient_invalid_reasons": sorted(reasons),
        "coefficient_scale": scale,
        "strict_event": True,
        "outer_gradient_interchange_guaranteed": False,
    }
    return ConditionalIntegralResult(
        probability, gradient, ga, gb, gradient_valid, tuple(intervals),
        tuple(roots), tuple(s * scale for s in slopes), diagnostics)
