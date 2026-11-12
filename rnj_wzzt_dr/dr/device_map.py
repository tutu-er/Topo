"""Original causal device map, extracted without legacy pricing models."""
from __future__ import annotations
import numpy as np

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
