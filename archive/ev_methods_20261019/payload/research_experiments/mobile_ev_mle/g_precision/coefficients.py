"""Fixed-response pair distances for G-weighted EV power planning.

The affine AC means/Jacobians and source forecasts supplied by the caller are
fixed.  Noise is independent between times and channels, with ``sigmas`` the
per-channel standard deviation.  With at most one active EV site per time,
the squared affine distance has no same-time cross-site power products.
The decomposition uses *absolute power*: a change of connection site may
make the increment from the anchor have two nonzero components.

This module does not profile sources or shared R/X, infer probabilities,
read simulator truth, select routes, or claim an AC approximation guarantee.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np


def _finite_matrix(value, name):
    try:
        result = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite matrix") from exc
    if result.ndim != 2 or min(result.shape, default=0) < 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must have nonempty finite shape [time,site]")
    return result


@dataclass(frozen=True)
class QuadraticTask:
    """``constant + sum(linear*u) + sum(quadratic*u**2)``.

    Arrays have the same [time,site] shape, are copied, and made read-only.
    ``quadratic`` already includes the factor 1/2 of the squared distance.
    A zero square coefficient requires a zero linear coefficient, as for a
    squared affine response whose sensitivity at that control is zero.
    """

    name: str
    constant: float
    linear: np.ndarray
    quadratic: np.ndarray

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Task name must be a nonempty string")
        try:
            constant = float(self.constant)
        except (TypeError, ValueError) as exc:
            raise ValueError("Task constant must be finite") from exc
        if not math.isfinite(constant):
            raise ValueError("Task constant must be finite")
        linear = _finite_matrix(self.linear, "linear")
        quadratic = _finite_matrix(self.quadratic, "quadratic")
        if linear.shape != quadratic.shape:
            raise ValueError("linear and quadratic must have the same [time,site] shape")
        if np.any(quadratic < 0):
            raise ValueError("quadratic coefficients must be nonnegative")
        if np.any((quadratic == 0) & (linear != 0)):
            raise ValueError("zero quadratic response must have zero linear response")
        linear.setflags(write=False)
        quadratic.setflags(write=False)
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "quadratic", quadratic)

    def value(self, power) -> float:
        u = _finite_matrix(power, "power")
        if u.shape != self.linear.shape:
            raise ValueError("power must have the task's [time,site] shape")
        with np.errstate(over="raise", invalid="raise"):
            value = self.constant + np.sum(self.linear * u + self.quadratic * u * u)
        if not np.isfinite(value):
            raise ValueError("Quadratic task value is nonfinite")
        return float(value)

    def record(self) -> dict:
        """Lossless finite JSON-friendly constructor fields."""
        return dict(name=self.name, constant=self.constant,
                    linear=self.linear.tolist(), quadratic=self.quadratic.tolist())


def build_tasks_from_responses(responses, sigmas, *, include_intercept: bool,
                               anchor_power=None) -> list[QuadraticTask]:
    """Build every unordered candidate-pair task using fixed affine responses.

    ``responses[name]`` has mean [T,M] and Jacobian [T,M,J], both at the same
    anchor power [T,J].  Names are sorted only to make pair order stable.
    For each pair, g=Jacobian_a-Jacobian_b and
    d0=mean_a-mean_b-g@anchor.  Intercept mode yields
      a=.5*sum((d0/sigma)**2), b=sum_M(d0*g/sigma**2),
      d=.5*sum_M((g/sigma)**2).
    Without the intercept, a and b are zero: this is response-only GSP.

    The returned diagonal quadratic equals the complete squared affine
    distance on the single-EV operation domain, not for simultaneous powers
    at multiple sites within one time slot.  There is no source projection.
    """
    if not isinstance(include_intercept, (bool, np.bool_)):
        raise ValueError("include_intercept must be boolean")
    if not isinstance(responses, Mapping) or len(responses) < 2:
        raise ValueError("At least two named candidate responses are required")
    if any(not isinstance(name, str) or not name for name in responses):
        raise ValueError("Candidate names must be nonempty strings")
    sigma = _finite_matrix(sigmas, "sigmas")
    if np.any(sigma <= 0):
        raise ValueError("sigmas must be strictly positive")
    T, M = sigma.shape
    checked, n_sites = {}, None
    for name in sorted(responses):
        record = responses[name]
        if not isinstance(record, Mapping) or not {"mean", "jacobian"} <= set(record):
            raise ValueError("Each candidate needs mean and jacobian arrays")
        mean = _finite_matrix(record["mean"], name + " mean")
        try:
            jacobian = np.array(record["jacobian"], dtype=float, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("Candidate Jacobian must be finite [time,channel,site]") from exc
        if (mean.shape != (T, M) or jacobian.ndim != 3
                or jacobian.shape[:2] != (T, M) or jacobian.shape[2] < 1
                or not np.isfinite(jacobian).all()):
            raise ValueError("Response dimensions must agree with sigmas [time,channel]")
        if n_sites is None:
            n_sites = jacobian.shape[2]
        elif jacobian.shape[2] != n_sites:
            raise ValueError("All candidate Jacobians need the same site dimension")
        checked[name] = (mean, jacobian)
    shape = (T, n_sites)
    anchor = np.zeros(shape) if anchor_power is None else _finite_matrix(anchor_power, "anchor_power")
    if anchor.shape != shape:
        raise ValueError("anchor_power must have shape [time,site]")
    result, used_names = [], set()
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for left, right in combinations(checked, 2):
            task_name = left + "__vs__" + right
            if task_name in used_names:
                raise ValueError("Candidate names produce a duplicate pair task name")
            used_names.add(task_name)
            mean_left, jacobian_left = checked[left]
            mean_right, jacobian_right = checked[right]
            g = jacobian_left - jacobian_right
            white_g = g / sigma[:, :, None]
            quadratic = .5 * np.einsum("tmj,tmj->tj", white_g, white_g)
            if include_intercept:
                d0 = mean_left - mean_right - np.einsum("tmj,tj->tm", g, anchor)
                white_d0 = d0 / sigma
                constant = .5 * float(np.sum(white_d0 * white_d0))
                linear = np.einsum("tm,tmj->tj", white_d0, white_g)
            else:
                constant, linear = 0., np.zeros(shape)
            result.append(QuadraticTask(task_name, constant, linear, quadratic))
    return result


def score_tasks(tasks, power) -> float:
    """Minimum unordered-pair score; no truth-based selection or pair deletion."""
    tasks = list(tasks)
    if not tasks:
        raise ValueError("At least one task is required")
    if any(not isinstance(task, QuadraticTask) for task in tasks):
        raise TypeError("All tasks must be QuadraticTask objects")
    return min(task.value(power) for task in tasks)


__all__ = ["QuadraticTask", "build_tasks_from_responses", "score_tasks"]
