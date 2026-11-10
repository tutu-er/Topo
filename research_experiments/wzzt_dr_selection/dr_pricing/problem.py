"""Portable fixed-pair pricing model and the original causal device map.

Only NumPy/SciPy and local probability.py are needed. Case inputs are a JSON
description plus a numeric NPZ, with no historical experiment imports or pickle.
The explicit selection settings determine the crossing threshold; no new-day
weight, topology prior, or positive line-weight bound is supplied implicitly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import numpy as np

try:
    from .probability import IndependentResiduals, probability_and_gradient
except ImportError:
    from probability import IndependentResiduals, probability_and_gradient


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array(arrays, name, shape=None, *, finite=True, dtype=float):
    if name not in arrays:
        raise ValueError("Missing input array: " + name)
    value = np.array(arrays[name], dtype=dtype, copy=True)
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if np.isnan(value).any() or (finite and not np.isfinite(value).all()):
        raise ValueError(name + " contains invalid numbers")
    return value


class Devices:
    """Power/state-feasible delivery from explicit per-resource capabilities."""

    def __init__(self, arrays, dt_hours, decision_slots):
        self.delta_lower_kw = _array(arrays, "resource_lower_kw")
        if self.delta_lower_kw.ndim != 2 or not self.delta_lower_kw.size:
            raise ValueError("Resource powers must be nonempty (resources, slots) arrays")
        self.R, self.T = self.delta_lower_kw.shape
        self.delta_upper_kw = _array(arrays, "resource_upper_kw", (self.R, self.T))
        self.cost_full_activation = _array(arrays, "resource_cost", (self.R, self.T))
        self.eta = _array(arrays, "resource_efficiency", (self.R,))
        self.energy_low = _array(arrays, "resource_energy_lower_kwh", (self.R,), finite=False)
        self.energy_high = _array(arrays, "resource_energy_upper_kwh", (self.R,), finite=False)
        for key in ("resource_reset", "resource_terminal"):
            raw = _array(arrays, key, (self.R, self.T))
            if not np.isin(raw, (0, 1)).all():
                raise ValueError(key + " must be boolean or numeric 0/1")
            setattr(self, "reset" if key.endswith("reset") else "terminal", raw.astype(bool))
        self.dt, self.decision_slots = float(dt_hours), int(decision_slots)
        if (not np.isfinite(self.dt) or self.dt <= 0 or self.decision_slots != decision_slots
                or self.decision_slots <= 0 or self.T % self.decision_slots):
            raise ValueError("Positive dt and integer decision_slots dividing slot count are required")
        self.H = self.T // self.decision_slots
        if (np.any(self.delta_lower_kw > 0) or np.any(self.delta_upper_kw < 0)
                or np.any(self.cost_full_activation <= 0)
                or np.any(self.eta <= 0) or np.any(self.eta > 1)
                or np.any(self.energy_low > 0) or np.any(self.energy_high < 0)
                or not self.reset[:, 0].all()):
            raise ValueError("Capabilities must admit zero deviation and positive efficiency/cost")
        for r in range(self.R):
            starts = np.flatnonzero(self.reset[r])
            for previous, start in zip(starts[:-1], starts[1:]):
                completed = np.flatnonzero(self.terminal[r, previous:start]) + previous
                if not completed.size:
                    raise ValueError("A resource reset would erase energy before its previous task deadline")
                gap = slice(int(completed[-1]) + 1, int(start))
                if (np.any(self.delta_lower_kw[r, gap] != 0)
                        or np.any(self.delta_upper_kw[r, gap] != 0)):
                    raise ValueError("Nonzero capability between a completed task and reset would erase energy")
        self.scale = np.maximum(-self.delta_lower_kw, self.delta_upper_kw)
        lower_name = "resource_future_lower_kwh" if "resource_future_lower_kwh" in arrays else "future_lower"
        upper_name = "resource_future_upper_kwh" if "resource_future_upper_kwh" in arrays else "future_upper"
        if lower_name in arrays or upper_name in arrays:
            self.future_lo = _array(arrays, lower_name, (self.R, self.T), finite=False)
            self.future_hi = _array(arrays, upper_name, (self.R, self.T), finite=False)
        else:
            self.future_lo, self.future_hi = self._future_capabilities()
        if np.any(self.future_lo > 0) or np.any(self.future_hi < 0):
            raise ValueError("Future capability intervals must include zero")
        self.next_lo = np.maximum(self.energy_low[:, None], -self.future_hi)
        self.next_hi = np.minimum(self.energy_high[:, None], -self.future_lo)

    def _future_capabilities(self):
        """Derive remaining energy bounds up to each explicitly declared deadline.

        Gaps after a completed task retain zero debt. A resource with no terminal
        task (such as curtailment) has no future repayment constraint.
        """
        lower = np.zeros((self.R, self.T))
        upper = np.zeros_like(lower)
        for r in range(self.R):
            deadlines = np.flatnonzero(self.terminal[r])
            if not deadlines.size:
                lower[r], upper[r] = -np.inf, np.inf
                continue
            previous = -1
            for deadline in deadlines:
                resets = np.flatnonzero(self.reset[r, :deadline + 1])
                start = max(previous + 1, int(resets[-1]))
                end = int(deadline) + 1
                lo, hi = self.delta_lower_kw[r, start:end], self.delta_upper_kw[r, start:end]
                energy_lo = np.where(lo < 0, lo / self.eta[r], lo * self.eta[r]) * self.dt
                energy_hi = np.where(hi < 0, hi / self.eta[r], hi * self.eta[r]) * self.dt
                lower[r, start:end] = np.cumsum(energy_lo[::-1])[::-1] - energy_lo
                upper[r, start:end] = np.cumsum(energy_hi[::-1])[::-1] - energy_hi
                previous = int(deadline)
        return lower, upper

    def _inverse(self, energy_rate):
        return np.where(energy_rate >= 0, energy_rate / self.eta, energy_rate * self.eta)

    def project(self, raw, cache=False):
        """Map (..., resources, slots) requests to causal feasible kW delivery."""
        raw = np.asarray(raw, float)
        if raw.shape[-2:] != (self.R, self.T) or not np.isfinite(raw).all():
            raise ValueError("Finite resource-by-slot requests required")
        shape = raw.shape
        values = raw.reshape(-1, self.R, self.T)
        energy = np.zeros(values.shape[:2])
        output = np.zeros_like(values)
        direct = np.zeros_like(values)
        state = np.zeros_like(values)
        update = np.zeros_like(values)
        for t in range(self.T):
            energy = np.where(self.reset[:, t], 0., energy)
            energy_lower = self._inverse((self.next_lo[:, t] - energy) / self.dt)
            energy_upper = self._inverse((self.next_hi[:, t] - energy) / self.dt)
            lower = np.maximum(self.delta_lower_kw[:, t], energy_lower)
            upper = np.minimum(self.delta_upper_kw[:, t], energy_upper)
            if np.any(lower > upper + 1e-8):
                raise ArithmeticError("Device state left its feasible interval")
            upper = np.maximum(upper, lower)
            power = np.minimum(np.maximum(values[:, :, t], lower), upper)
            slope = np.where(power >= 0, self.eta, 1 / self.eta)
            interior = (values[:, :, t] > lower) & (values[:, :, t] < upper)
            interior |= ((values[:, :, t] == 0) & (lower < upper)
                         & (values[:, :, t] >= lower) & (values[:, :, t] <= upper))
            energy_face = (((values[:, :, t] <= lower) & (energy_lower >= self.delta_lower_kw[:, t]))
                           | ((values[:, :, t] >= upper) & (energy_upper <= self.delta_upper_kw[:, t])))
            direct[:, :, t] = np.where(energy_face, 0., interior)
            state[:, :, t] = np.where(energy_face, -1 / (self.dt * slope), 0.)
            update[:, :, t] = self.dt * slope
            output[:, :, t] = power
            energy += self.dt * slope * power
        if cache:
            return output.reshape(shape), (direct, state, update, shape)
        return output.reshape(shape)

    def vjp(self, gradient, cache):
        """Back-propagate a delivery gradient through the selected device faces."""
        direct, state, update, shape = cache
        gradient = np.asarray(gradient).reshape(direct.shape)
        previous = np.zeros(direct.shape[:2])
        output = np.zeros_like(direct)
        for t in range(self.T - 1, -1, -1):
            combined = gradient[:, :, t] + previous * update[:, :, t]
            output[:, :, t] = combined * direct[:, :, t]
            previous = previous + combined * state[:, :, t]
            previous = np.where(self.reset[:, t], 0., previous)
        return output.reshape(shape)

    def nominal(self, requests, cache=False):
        requests = np.asarray(requests, float).reshape(self.R, self.H)
        return self.project(np.repeat(requests, self.decision_slots, axis=1) * self.scale, cache=cache)

    def hourly_vjp(self, gradient, cache):
        return (self.vjp(gradient, cache) * self.scale).reshape(self.R, self.H, self.decision_slots).sum(-1)

    def fees(self, delivery, gradient=False):
        delivery = np.asarray(delivery, float)
        headroom = np.where(delivery >= 0, self.delta_upper_kw, -self.delta_lower_kw)
        coefficient = np.divide(self.dt * self.cost_full_activation, headroom,
                                out=np.zeros_like(headroom), where=headroom > 0)
        if np.any((headroom == 0) & (np.abs(delivery) > 1e-8)):
            raise ValueError("Zero power capability was used")
        fee = float(np.sum(coefficient * delivery * delivery))
        return (fee, 2 * coefficient * delivery) if gradient else fee

    def audit(self, delivery):
        delivery = np.asarray(delivery, float)
        if delivery.shape != (self.R, self.T) or not np.isfinite(delivery).all():
            raise ValueError("Finite resource-by-slot delivery required")
        power = max(float(np.maximum(self.delta_lower_kw - delivery, 0).max()),
                    float(np.maximum(delivery - self.delta_upper_kw, 0).max()))
        energy = np.zeros(self.R)
        energy_error = deadline_error = 0.
        for t in range(self.T):
            energy = np.where(self.reset[:, t], 0., energy)
            energy += self.dt * np.where(delivery[:, t] >= 0, self.eta, 1 / self.eta) * delivery[:, t]
            energy_error = max(energy_error, float(np.maximum(self.energy_low - energy, 0).max()),
                               float(np.maximum(energy - self.energy_high, 0).max()))
            due = self.terminal[:, t]
            if np.any(due):
                deadline_error = max(deadline_error, float(np.max(np.abs(energy[due]))))
        return dict(power_violation_kw=power, energy_violation_kwh=energy_error,
                    deadline_violation_kwh=deadline_error,
                    passed=max(power, energy_error, deadline_error) < 1e-7,
                    absolute_DR_kwh=float(self.dt * np.abs(delivery).sum()),
                    net_DR_kwh=float(self.dt * delivery.sum()),
                    max_sum_abs_kw=float(np.abs(delivery).sum(axis=0).max()))


class PricingProblem:
    """A frozen candidate pair, explicit validation rule, and resource schedule."""

    def __init__(self, config, arrays):
        self.config = self.metadata = dict(config)
        if not isinstance(config.get("residual_unit"), str) or not config["residual_unit"].strip():
            raise ValueError("Explicit nonempty residual_unit is required")
        information = config.get("information", {})
        for name, expected in (("parameters_frozen", True), ("actual_future_used", False), ("truth_used", False)):
            if information.get(name) is not expected:
                raise ValueError(f"information.{name} must be explicitly declared {expected}")
        self.base = {"C": _array(arrays, "baseline_coarse"), "F": _array(arrays, "baseline_fine")}
        if self.base["C"].ndim != 2 or not self.base["C"].size or self.base["F"].shape != self.base["C"].shape:
            raise ValueError("Candidate baselines must share nonempty (slots, observations) shape")
        self.T, self.N = self.base["C"].shape
        self.device = Devices(arrays, config["dt_hours"], config["decision_slots"])
        if self.device.T != self.T:
            raise ValueError("Resources and residuals have different slot counts")
        self.shape = (self.device.R, self.device.H)
        self.effect = {"C": _array(arrays, "effect_coarse", (self.N, self.device.R)),
                       "F": _array(arrays, "effect_fine", (self.N, self.device.R))}
        self.sigma = _array(arrays, "sigma", (self.T, self.N))
        if np.any(self.sigma < 0):
            raise ValueError("Residual standard deviations must be nonnegative")
        self._selection_rule(config["selection"])
        probability = config["probability"]
        if (probability.get("independent_cells") is not True
                or probability.get("shared_candidate_noise") is not True
                or probability.get("scope") != "conditional_voltage"):
            raise ValueError("Unsupported probability law: declare independent cells/shared candidate voltage noise")
        self.target = float(probability["target"])
        if not 0 < self.target < 1:
            raise ValueError("Target probability must lie strictly between zero and one")
        self.options = dict(cutoff=512., order=32, tolerance=1e-7, chunk=64)
        numerics = dict(config.get("numerics", {}))
        self.audit_orders = tuple(numerics.pop("audit_orders", (16, 32, 64)))
        if (len(self.audit_orders) < 2 or any(not isinstance(order, int) or order < 2
                for order in self.audit_orders) or any(right <= left
                for left, right in zip(self.audit_orders, self.audit_orders[1:]))):
            raise ValueError("audit_orders must contain at least two increasing integer orders >=2")
        if set(numerics) - set(self.options):
            raise ValueError("Unknown numerical option: " + ", ".join(sorted(set(numerics) - set(self.options))))
        self.options.update(numerics)
        if (not np.isfinite([self.options["cutoff"], self.options["tolerance"]]).all()
                or self.options["cutoff"] <= 0 or self.options["tolerance"] <= 0
                or self.options["order"] < 2 or int(self.options["order"]) != self.options["order"]
                or self.options["chunk"] < 1 or int(self.options["chunk"]) != self.options["chunk"]):
            raise ValueError("Invalid cutoff, quadrature order, tolerance, or frequency chunk")
        self.node_labels = config.get("node_labels", list(range(self.N)))
        self.resource_ids = config.get("resource_ids", [str(r) for r in range(self.device.R)])
        if len(self.node_labels) != self.N or len(set(self.node_labels)) != self.N:
            raise ValueError("Unique observation node labels required")
        if len(self.resource_ids) != self.device.R or len(set(self.resource_ids)) != self.device.R:
            raise ValueError("Unique resource identifiers required")
        scale = self.device.scale.reshape(self.device.R, self.device.H, self.device.decision_slots)
        self.active = np.flatnonzero(scale.max(axis=2).ravel() > 0)
        if not self.active.size:
            raise ValueError("No controllable resource intervals")
        low = (self.device.delta_lower_kw.reshape(scale.shape).min(axis=2) < 0)
        high = (self.device.delta_upper_kw.reshape(scale.shape).max(axis=2) > 0)
        self.lower = np.where(low.ravel()[self.active], -1., 0.)
        self.upper = np.where(high.ravel()[self.active], 1., 0.)
        self.bounds = list(zip(self.lower, self.upper))
        self.zero = np.zeros(self.active.size)
        warm_name = "warm_requests" if "warm_requests" in arrays else "warm_x"
        warm = _array(arrays, warm_name, self.shape) if warm_name in arrays else np.zeros(self.shape)
        self.warm = self.old = self.requests_to_values(warm)
        point = self.nominal(self.warm)
        model = IndependentResiduals(point["a_C"], point["a_F"], self.sigma.ravel(), self.kappa)
        normalization = config.get("normalization", (model.sum_mean, model.sum_sd or 1.))
        self.normalization = tuple(map(float, normalization))
        if len(self.normalization) != 2 or not np.isfinite(self.normalization).all() or self.normalization[1] <= 0:
            raise ValueError("Finite normalization center and positive scale required")
        self.last = self.last_point = None
        self.case_path = self.arrays_path = None
        self.input_references = {}

    def _selection_rule(self, rule):
        required = ("history_count", "new_count", "new_day_weight", "historical_loss_coarse",
                    "historical_loss_fine", "relative_band", "absolute_tolerance", "candidate_ids",
                    "support_counts", "historically_eligible", "candidate_count")
        missing = [key for key in required if key not in rule]
        if missing:
            raise ValueError("Explicit selection settings required: " + ", ".join(missing))
        self.candidate_ids = tuple(rule["candidate_ids"])
        self.support_counts = tuple(rule["support_counts"])
        if (rule["candidate_count"] != 2 or len(self.candidate_ids) != 2
                or len(set(self.candidate_ids)) != 2 or len(self.support_counts) != 2
                or not 0 <= self.support_counts[0] < self.support_counts[1]
                or list(rule["historically_eligible"]) != [True, True]):
            raise ValueError("Unsupported selection: requires an eligible pair with coarse raw support smaller than fine")
        self.history_count, self.new_count = float(rule["history_count"]), float(rule["new_count"])
        self.new_weight = float(rule["new_day_weight"])
        self.kappa = 1. + float(rule["relative_band"])
        self.epsilon = float(rule["absolute_tolerance"])
        self.history_losses = np.array([rule["historical_loss_coarse"], rule["historical_loss_fine"]], float)
        numbers = [self.history_count, self.new_count, self.new_weight, self.kappa, self.epsilon, *self.history_losses]
        if (not np.isfinite(numbers).all() or self.history_count <= 0 or self.new_count != self.T
                or self.new_weight <= 0 or self.kappa <= 1 or self.epsilon < 0
                or np.any(self.history_losses < 0)):
            raise ValueError("Invalid selection values or new_count differs from complete slot count")
        if np.any(self.history_losses > self.kappa * self.history_losses.min() + self.epsilon):
            raise ValueError("Candidate pair fails the declared historical validation gate")
        new_mass = self.new_weight * self.new_count
        self.new_fraction = new_mass / (self.history_count + new_mass)
        self.threshold = (self.history_count / new_mass
            * (self.kappa * self.history_losses[1] - self.history_losses[0])
            + (self.history_count + new_mass) / new_mass * self.epsilon)

    def expand(self, values):
        values = np.asarray(values, float)
        if values.shape != self.active.shape or not np.isfinite(values).all():
            raise ValueError("Finite active request vector with the declared shape required")
        requests = np.zeros(self.device.R * self.device.H)
        requests[self.active] = values
        return requests.reshape(self.shape)

    def requests_to_values(self, requests):
        requests = np.asarray(requests, float)
        if requests.shape != self.shape or not np.isfinite(requests).all():
            raise ValueError("Finite resource-by-decision request array required")
        values = requests.ravel()[self.active]
        if (np.any(values < self.lower) or np.any(values > self.upper)
                or not np.array_equal(self.expand(values), requests)):
            raise ValueError("Requests violate directional bounds or use an unavailable interval")
        return values.copy()

    def nominal(self, values):
        requests = self.expand(values)
        delivery, cache = self.device.nominal(requests, cache=True)
        means = {k:(self.base[k] + np.einsum("ir,rt->ti", self.effect[k], delivery)).ravel()
                 for k in ("C", "F")}
        fee, derivative = self.device.fees(delivery, gradient=True)
        return dict(x=requests, mean_u=delivery, cache=cache, fee=fee, fee_gradient=derivative,
                    a_C=means["C"], a_F=means["F"])

    def evaluate(self, values):
        values = np.asarray(values, float)
        if self.last_point is not None and np.array_equal(values, self.last_point):
            return self.last
        if np.any(values < self.lower) or np.any(values > self.upper):
            raise ValueError("Requests lie outside their declared bounds")
        began = time.perf_counter()
        point = self.nominal(values)
        result = probability_and_gradient(point["a_C"], point["a_F"], self.sigma.ravel(), self.threshold,
            kappa=self.kappa, normalization=self.normalization, refinements=0, **self.options)
        coarse = result["gradient_coarse"].reshape(self.T, self.N)
        fine = result["gradient_fine"].reshape(self.T, self.N)
        delivery_derivative = (coarse @ self.effect["C"] + fine @ self.effect["F"]).T
        probability_gradient = self.device.hourly_vjp(delivery_derivative, point["cache"]).ravel()[self.active]
        fee_gradient = self.device.hourly_vjp(point["fee_gradient"], point["cache"]).ravel()[self.active]
        audit = self.device.audit(point["mean_u"])
        probability = float(result["probability"])
        if (not audit["passed"] or not np.isfinite(probability) or not 0 <= probability <= 1
                or not np.isfinite(probability_gradient).all() or not np.isfinite(fee_gradient).all()):
            raise ArithmeticError("Unresolved device or numerical probability evaluation")
        self.last_point = values.copy()
        self.last = dict(values=values.copy(), x=point["x"], mean_u=point["mean_u"],
            probability=probability, fee=point["fee"], gradient=probability_gradient,
            fee_gradient=fee_gradient, moments=result["moments"],
            analytic_tail_bound=result["analytic_tail_bound"], nominal_audit=audit,
            seconds=time.perf_counter() - began)
        return self.last

    def audit_probability(self, values, orders=None):
        orders = self.audit_orders if orders is None else tuple(orders)
        if len(orders) < 2:
            raise ValueError("Probability audit requires at least two quadrature orders")
        point = self.nominal(values)
        results = [probability_and_gradient(point["a_C"], point["a_F"], self.sigma.ravel(), self.threshold,
            kappa=self.kappa, normalization=self.normalization,
            **dict(self.options, order=order, refinements=0)) for order in orders]
        probabilities = [row["probability"] for row in results]
        spread = max(probabilities) - min(probabilities)
        tolerance = self.options["tolerance"]
        return dict(probability=probabilities[-1], numerical_pass=bool(spread <= tolerance and all(
            0 <= row["probability"] <= 1 and row.get("analytic_tail_bound", 0.) <= tolerance for row in results)),
            orders=list(orders), probabilities=probabilities, maximum_difference=spread,
            moments=results[-1].get("moments", IndependentResiduals(
                point["a_C"], point["a_F"], self.sigma.ravel(), self.kappa).moments()),
            analytic_tail_bound=results[-1].get("analytic_tail_bound", 0.),
            certified_error=0. if all(row.get("certified_error") == 0. for row in results) else None)

    def unit_reward(self, delivery):
        delivery = np.asarray(delivery, float)
        headroom = np.where(delivery >= 0, self.device.delta_upper_kw, -self.device.delta_lower_kw)
        return np.divide(self.device.cost_full_activation * np.abs(delivery), headroom,
                         out=np.zeros_like(headroom), where=headroom > 0)

    def audit_delivery(self, delivery):
        return self.device.audit(delivery)

    def score_observations(self, power_channels, observed_drop,
                           coarse_response, fine_response):
        """Score actual observations with externally fitted, frozen responses.

        power_channels is (slots, channels), observed_drop (slots, nodes), and
        responses (nodes, channels). The caller freezes the response matrices
        before planning and uses the same residual units as this case. This
        adapter applies no parameter fitting or simulator-specific defaults.
        """
        power = np.asarray(power_channels, float)
        drop = np.asarray(observed_drop, float)
        responses = [np.asarray(coarse_response, float), np.asarray(fine_response, float)]
        if (power.ndim != 2 or power.shape[0] != self.T
                or drop.shape != (self.T, self.N) or not np.isfinite(power).all()
                or not np.isfinite(drop).all()
                or any(h.shape != (self.N, power.shape[1]) or not np.isfinite(h).all()
                       for h in responses)):
            raise ValueError("Actual observations and frozen responses have incompatible shape/units")
        return self.score_residuals(drop - power @ responses[0].T,
                                    drop - power @ responses[1].T)

    def score_residuals(self, coarse, fine):
        """Apply the same frozen-pair selector to actual recorded residuals."""
        residuals = [np.asarray(coarse, float), np.asarray(fine, float)]
        if any(value.shape != (self.T, self.N) or not np.isfinite(value).all() for value in residuals):
            raise ValueError("Actual residuals must match the complete (slots, observations) shape")
        losses = np.array([np.abs(value).mean() for value in residuals])
        combined = ((1 - self.new_fraction) * self.history_losses + self.new_fraction * losses)
        eligible = combined <= self.kappa * combined.min() + self.epsilon
        selected = min(np.flatnonzero(eligible), key=lambda j:(self.support_counts[j], combined[j], j))
        gap = float(losses[0] - self.kappa * losses[1])
        return dict(selected=self.candidate_ids[selected], eligible=[self.candidate_ids[j] for j in np.flatnonzero(eligible)],
            coarse_excluded=not bool(eligible[0]), D=gap, threshold=self.threshold,
            new_losses=dict(zip(self.candidate_ids, losses.tolist())),
            combined_losses=dict(zip(self.candidate_ids, combined.tolist())),
            selection_margin=float(combined[0] - self.kappa * combined[1] - self.epsilon),
            identity_error=abs(float(combined[0] - self.kappa * combined[1] - self.epsilon)
                               - self.new_fraction * (gap - self.threshold)),
            coverage="declared frozen pair only", true_topology_probability=False)


def aggregate_delivery(delivery, node_index, q_over_p, node_count):
    """Return (slots, nodes) kW/kvar increments from resource-by-slot kW.

    Node indices start at zero. Add these increments to the natural load once.
    """
    delivery, ratios = np.asarray(delivery, float), np.asarray(q_over_p, float)
    nodes = np.asarray(node_index)
    if (delivery.ndim != 2 or ratios.shape != (delivery.shape[0],)
            or nodes.shape != ratios.shape or not np.isfinite(delivery).all()
            or not np.isfinite(ratios).all() or not np.isfinite(nodes).all()
            or node_count <= 0 or int(node_count) != node_count
            or not np.equal(nodes, nodes.astype(int)).all()
            or np.any(nodes < 0) or np.any(nodes >= node_count)):
        raise ValueError("Invalid resource-to-node increment mapping")
    active = np.zeros((delivery.shape[1], int(node_count)))
    reactive = np.zeros_like(active)
    for resource, node in enumerate(nodes.astype(int)):
        active[:, node] += delivery[resource]
        reactive[:, node] += ratios[resource] * delivery[resource]
    return active, reactive


def load_problem(path):
    """Read a portable numeric case; arrays_file must remain inside its folder."""
    path = Path(path).resolve()
    case_path = path / "case.json" if path.is_dir() else path
    config = json.loads(case_path.read_text(encoding="utf-8"))
    if config.get("format_version") != 1:
        raise ValueError("Unsupported case format_version")
    arrays_path = (case_path.parent / config.get("arrays_file", "arrays.npz")).resolve()
    if not arrays_path.is_relative_to(case_path.parent):
        raise ValueError("arrays_file must be local to the portable case folder")
    with np.load(arrays_path, allow_pickle=False) as saved:
        arrays = {key:saved[key] for key in saved.files}
    problem = PricingProblem(config, arrays)
    problem.case_path, problem.arrays_path = case_path, arrays_path
    problem.input_references = {"case": dict(path=case_path.name, sha256=_hash(case_path)),
                                "arrays": dict(path=str(arrays_path.relative_to(case_path.parent)), sha256=_hash(arrays_path))}
    return problem
