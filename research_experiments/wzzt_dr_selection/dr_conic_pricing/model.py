"""Inputs and numerical checks for reliable-excitation price planning.

The probability statement concerns declared linear excitation benefits.  It is
not a statement about correctness of a topology identifier.  Common factors are
independent U[-1, 1]; correlations arise through the loading matrices.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import copy
import json
from typing import Mapping

import numpy as np


def _array(value, name, ndim=None, shape=None):
    a = np.array(value, dtype=float, copy=True)
    if (ndim is not None and a.ndim != ndim) or (shape is not None and a.shape != shape):
        raise ValueError(f"{name} has shape {a.shape}; expected {shape or str(ndim)+' dimensions'}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must contain only finite numbers")
    a.flags.writeable = False
    return a


@dataclass(frozen=True)
class PriceProblem:
    """One frozen planning instance; x is expected response amplitude in kW.

    profiles has shape (power coordinates, contracts); response_factors has
    shape (contracts, common factors).  Power coordinates can concatenate P/Q
    and all planning time slots, with an explicitly documented ordering.
    device_matrix acts on realized contract amplitudes, not on total load.
    """

    mu0: object
    background_factors: object
    profiles: object
    response_factors: object
    elasticities: object
    reduction_hours: object
    price_upper: object
    actual_amplitude_upper: object
    benefit_directions: object
    benefit_offsets: object
    risk_allocations: object
    budget: float
    confidence: float = .9
    device_matrix: object = None
    device_rhs: object = None
    contract_ids: object = ()
    exclusive_groups: object = ()
    benefit_mode: str = "total"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        mu = _array(self.mu0, "mu0", ndim=1)
        if not mu.size:
            raise ValueError("at least one power coordinate is required")
        h = _array(self.profiles, "profiles", ndim=2)
        if h.shape[0] != mu.size or h.shape[1] == 0:
            raise ValueError("profiles must have one row per power coordinate and at least one contract")
        j = h.shape[1]
        f = _array(self.background_factors, "background_factors", ndim=2)
        if f.shape[0] != mu.size:
            raise ValueError("background_factors row count must match mu0")
        d = _array(self.response_factors, "response_factors", shape=(j, f.shape[1]))
        if np.any(np.sum(np.abs(d), axis=1) > 1.):
            raise ValueError("each response_factors row must have L1 norm <= 1")
        g = _array(self.benefit_directions, "benefit_directions", ndim=2)
        if g.shape[1] != mu.size or g.shape[0] == 0:
            raise ValueError("benefit_directions must have one column per power coordinate and at least one row")
        for name, value in (("mu0", mu), ("profiles", h), ("background_factors", f),
                            ("response_factors", d), ("benefit_directions", g)):
            object.__setattr__(self, name, value)
        for name in ("elasticities", "reduction_hours", "price_upper", "actual_amplitude_upper"):
            a = _array(getattr(self, name), name, shape=(j,))
            if np.any(a <= 0) if name in ("elasticities", "reduction_hours") else np.any(a < 0):
                raise ValueError(f"{name} must be {'strictly positive' if name in ('elasticities','reduction_hours') else 'nonnegative'}")
            object.__setattr__(self, name, a)
        k = g.shape[0]
        offsets = _array(self.benefit_offsets, "benefit_offsets", shape=(k,))
        eps = _array(self.risk_allocations, "risk_allocations", shape=(k,))
        confidence, budget = float(self.confidence), float(self.budget)
        if not np.isfinite((confidence, budget)).all() or not 0 < confidence < 1 or budget < 0:
            raise ValueError("confidence must be in (0,1), and budget must be finite and nonnegative")
        if np.any(eps <= 0) or np.any(eps >= 1) or eps.sum() > 1-confidence+1e-12:
            raise ValueError("positive fixed risk allocations must sum to at most 1-confidence")
        if self.benefit_mode not in ("total", "incremental"):
            raise ValueError("benefit_mode must be 'total' or 'incremental'")
        if self.benefit_mode == "incremental" and np.any(offsets != 0):
            raise ValueError("incremental benefit uses a paired difference and requires zero benefit_offsets")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "budget", budget)
        object.__setattr__(self, "benefit_offsets", offsets)
        object.__setattr__(self, "risk_allocations", eps)
        if self.device_matrix is None:
            a = np.empty((0, j))
        else:
            a = np.asarray(self.device_matrix, dtype=float)
            if a.size == 0:
                a = np.empty((0, j))
        a = _array(a, "device_matrix", ndim=2)
        if a.shape[1] != j:
            raise ValueError("device_matrix must have one column per contract")
        b = _array(np.empty(0) if self.device_rhs is None else self.device_rhs,
                   "device_rhs", shape=(a.shape[0],))
        object.__setattr__(self, "device_matrix", a)
        object.__setattr__(self, "device_rhs", b)
        ids = tuple(self.contract_ids) or tuple(f"contract_{i}" for i in range(j))
        if len(ids) != j or len(set(ids)) != j or any(not isinstance(s, str) or not s for s in ids):
            raise ValueError("contract_ids must be distinct nonempty strings, one per contract")
        object.__setattr__(self, "contract_ids", ids)
        groups, used = [], set()
        for group in self.exclusive_groups:
            group = tuple(group)
            if (not group or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                                  or not 0 <= i < j for i in group) or len(set(group)) != len(group)):
                raise ValueError("exclusive_groups require nonempty groups of distinct contract indices")
            if used.intersection(group):
                raise ValueError("exclusive_groups must be disjoint physical contract-type groups")
            used.update(group)
            groups.append(tuple(int(i) for i in group))
        object.__setattr__(self, "exclusive_groups", tuple(groups))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a JSON-compatible mapping")
        try:
            metadata = json.loads(json.dumps(dict(self.metadata), allow_nan=False))
        except (ValueError, TypeError) as error:
            raise ValueError("metadata must be JSON-compatible and finite") from error
        object.__setattr__(self, "metadata", metadata)

    @property
    def n_contracts(self):
        return self.profiles.shape[1]

    @property
    def n_factors(self):
        return self.background_factors.shape[1]

    @property
    def q(self):
        return np.sum(np.abs(self.response_factors), axis=1)

    @property
    def amplitude_upper(self):
        return np.minimum(self.elasticities*self.price_upper,
                          self.actual_amplitude_upper/(1+self.q))

    @property
    def expected_cost_coefficients(self):
        return self.reduction_hours/self.elasticities

    @property
    def hard_cost_coefficients(self):
        return self.expected_cost_coefficients*(1+self.q)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping):
            raise TypeError("problem must be a mapping")
        names = {f.name for f in fields(cls)}
        unknown = set(value)-names
        if unknown:
            raise ValueError(f"unknown problem fields: {sorted(unknown)}")
        return cls(**dict(value))

    def to_dict(self):
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, tuple):
                value = [list(v) if isinstance(v, tuple) else v for v in value]
            else:
                value = copy.deepcopy(value)
            result[f.name] = value
        return result


def evaluate_plan(problem: PriceProblem, amplitudes, *, tolerance=1e-7, include_covariance=False):
    """Recompute moments and constraint residuals without using solver variables.

    Feasible is a numerical input/model check.  The lower bound is the Cantelli
    sufficient bound for the declared moment model, not an exact quantile.
    """
    x = _array(amplitudes, "amplitudes", shape=(problem.n_contracts,))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    h, d = problem.profiles, problem.response_factors
    response_factor = (h*x)@d
    net_mean = problem.mu0+h@x
    net_factor = problem.background_factors+response_factor
    if problem.benefit_mode == "incremental":
        benefit_mean = problem.benefit_directions@(h@x)
        benefit_factor = problem.benefit_directions@response_factor
    else:
        benefit_mean = problem.benefit_offsets+problem.benefit_directions@net_mean
        benefit_factor = problem.benefit_directions@net_factor
    std = np.linalg.norm(benefit_factor, axis=1)/np.sqrt(3.)
    gamma = np.sqrt((1-problem.risk_allocations)/problem.risk_allocations)
    lcb = benefit_mean-gamma*std
    device_worst = problem.device_matrix@x + np.sum(
        np.abs((problem.device_matrix*x)@d), axis=1)
    expected_cost = float(problem.expected_cost_coefficients@(x*x))
    hard_cost = float(problem.hard_cost_coefficients@(x*x))
    # The exact shared-factor worst cost can be smaller than the convex cap.
    payment_factor = (problem.expected_cost_coefficients*x*x)@d
    exact_box_worst = expected_cost+float(np.sum(np.abs(payment_factor)))
    exclusive_counts = [int(np.count_nonzero(x[list(group)] > tolerance))
                        for group in problem.exclusive_groups]
    device_viol = float(np.max(device_worst-problem.device_rhs, initial=0.))
    upper_viol = float(np.max(x-problem.amplitude_upper, initial=0.))
    lower_viol = float(np.max(-x, initial=0.))
    budget_viol = max(0., hard_cost-problem.budget)
    exclusive_viol = max((count-1 for count in exclusive_counts), default=0)
    max_viol = max(0., device_viol, upper_viol, lower_viol, budget_viol, exclusive_viol)
    net_covariance_factor = net_factor/np.sqrt(3.)
    response_covariance_factor = (x[:, None]*d)/np.sqrt(3.)
    result = dict(
        amplitudes=x.tolist(), prices=(x/problem.elasticities).tolist(),
        net_mean=net_mean.tolist(), net_factors=net_factor.tolist(),
        net_covariance_factor=net_covariance_factor.tolist(),
        net_covariance_diagonal=np.sum(net_covariance_factor**2, axis=1).tolist(),
        response_mean=x.tolist(),
        response_covariance_factor=response_covariance_factor.tolist(),
        response_covariance_diagonal=np.sum(response_covariance_factor**2, axis=1).tolist(),
        benefit_means=benefit_mean.tolist(), benefit_standard_deviations=std.tolist(),
        benefit_lower_bounds=lcb.tolist(), reliable_benefit=float(np.min(lcb)),
        expected_cost=expected_cost, hard_budget_bound=hard_cost,
        exact_box_worst_cost=exact_box_worst,
        device_worst_values=device_worst.tolist(), device_residuals=(device_worst-problem.device_rhs).tolist(),
        actual_amplitude_lower=((1-problem.q)*x).tolist(),
        actual_amplitude_upper=((1+problem.q)*x).tolist(),
        exclusive_group_positive_counts=exclusive_counts,
        maximum_violation=max_viol, numerical_feasible=bool(max_viol <= tolerance),
        violations=dict(device=max(0., device_viol), amplitude_upper=max(0., upper_viol),
                        amplitude_lower=max(0., lower_viol), budget=budget_viol,
                        exclusivity=max(0, exclusive_viol)),
        probability_scope="declared linear excitation benefits; not topology recovery",
        confidence=problem.confidence, bound_kind="Cantelli plus fixed risk allocation union bound",
    )
    if include_covariance:
        result["net_covariance"] = (net_covariance_factor@net_covariance_factor.T).tolist()
        result["response_covariance"] = (response_covariance_factor@response_covariance_factor.T).tolist()
    return result
