"""Original resource-only adapter, without candidate or runtime imports."""
from __future__ import annotations
import numpy as np
from .device_map import Devices as SolverDevices

def _finite(value, name, shape=None):
    a = np.asarray(value, dtype=float)
    if not np.isfinite(a).all() or (shape is not None and a.shape != shape):
        raise ValueError(f"Invalid finite shape for {name}: {a.shape}, expected {shape}")
    return a


class ResourceOnly:
    """Original device/fee contract without any MAE candidate qualification.

    The resource controller is the existing vendor.problem.Devices; the small
    active-hour mapping below has the same semantics as PricingProblem. This
    adapter never constructs its MAE eligibility or probability model.
    """
    def __init__(self, original, labels):
        terminal = np.zeros_like(original.reset, dtype=bool)
        terminal[original.kind != 1, -1] = True
        terminal[original.kind == 2, 31] = True
        arrays = dict(resource_lower_kw=original.delta_lower_kw,
            resource_upper_kw=original.delta_upper_kw,
            resource_cost=original.cost_full_activation,
            resource_efficiency=original.eta,
            resource_energy_lower_kwh=original.energy_low,
            resource_energy_upper_kwh=original.energy_high,
            resource_reset=original.reset, resource_terminal=terminal,
            resource_future_lower_kwh=original.future_lo,
            resource_future_upper_kwh=original.future_hi)
        self.device = SolverDevices(arrays, original.dt, 4)
        self.T, self.H, self.N = self.device.T, self.device.H, len(labels)
        self.shape = (self.device.R, self.device.H)
        self.node_labels, self.resource_ids = list(labels), list(original.ids)
        self.config = self.metadata = dict(node_labels=self.node_labels,
            resource_ids=self.resource_ids, dt_hours=original.dt, decision_slots=4,
            resource_contract="unchanged heterogeneous full inventory",
            selection="external strict unweighted physical MSE; no MAE gate")
        scale = self.device.scale.reshape(self.device.R, self.H, 4)
        self.active = np.flatnonzero(scale.max(axis=2).ravel() > 0)
        if not self.active.size:
            raise ValueError("No available control intervals in the original resource inventory")
        low = self.device.delta_lower_kw.reshape(scale.shape).min(axis=2) < 0
        high = self.device.delta_upper_kw.reshape(scale.shape).max(axis=2) > 0
        self.lower = np.where(low.ravel()[self.active], -1., 0.)
        self.upper = np.where(high.ravel()[self.active], 1., 0.)
        self.bounds = list(zip(self.lower, self.upper))
        self.zero = np.zeros(self.active.size)
        self.warm = self.old = self.zero.copy()
        for name in ("delta_lower_kw", "delta_upper_kw", "cost_full_activation",
                     "eta", "energy_low", "energy_high", "reset", "future_lo", "future_hi"):
            if not np.array_equal(getattr(self.device, name), getattr(original, name)):
                raise ValueError("Solver and original device differ: " + name)
        if not self.audit_delivery(np.zeros((self.device.R, self.T)))["passed"]:
            raise ValueError("Zero delivery is not feasible for the declared device contract")

    def expand(self, values):
        values = _finite(values, "active hourly requests", self.active.shape)
        x = np.zeros(self.device.R * self.H)
        x[self.active] = values
        return x.reshape(self.shape)

    def requests_to_values(self, requests):
        x = _finite(requests, "hourly requests", self.shape)
        values = x.ravel()[self.active]
        if (np.any(values < self.lower) or np.any(values > self.upper)
                or not np.array_equal(self.expand(values), x)):
            raise ValueError("Requests use an unavailable interval or violate original bounds")
        return values.copy()

    def nominal(self, values):
        x = self.expand(values)
        u, cache = self.device.nominal(x, cache=True)
        fee, derivative = self.device.fees(u, gradient=True)
        return dict(x=x, mean_u=u, cache=cache, fee=fee, fee_gradient=derivative)

    def audit_delivery(self, delivery):
        return self.device.audit(delivery)

    def unit_reward(self, delivery):
        u = _finite(delivery, "delivery", (self.device.R, self.T))
        headroom = np.where(u >= 0, self.device.delta_upper_kw, -self.device.delta_lower_kw)
        return np.divide(self.device.cost_full_activation * np.abs(u), headroom,
                         out=np.zeros_like(u), where=headroom > 0)
