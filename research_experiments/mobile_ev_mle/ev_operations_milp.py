"""Single-EV time-expanded operations MILP, in kW and kWh.

This module does not enumerate routes or power profiles. A binary arc flow
chooses one path in an acyclic time/site network. Only one-slot stay arcs
allow charging; a travel arc skips ``travel_slots`` complete unconnected
slots. Driving energy is deducted once, in the departure slot. Every battery
boundary is constrained, including arrival before the next charging slot.

``u_index[t, site]`` is time-major. Q is identically zero. The operation
builder has no topology, measurement, forecast, or future-noise inputs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


class LinearMILP:
    """Append-only linear MILP builder; objectives default to maximization."""

    def __init__(self):
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.integer: list[int] = []
        self.variables: dict[str, np.ndarray] = {}
        self.rows: list[dict[int, float]] = []
        self.row_lower: list[float] = []
        self.row_upper: list[float] = []

    @property
    def n_variables(self):
        return len(self.lower)

    def clone(self):
        return deepcopy(self)

    def add_variables(self, name, shape=(), lower=0.0, upper=np.inf,
                      integer=False):
        if name in self.variables:
            raise ValueError(f"Duplicate variable group: {name}")
        if isinstance(shape, (int, np.integer)):
            shape = (int(shape),)
        shape = tuple(shape)
        if any(int(s) != s or s < 0 for s in shape):
            raise ValueError("Variable dimensions must be nonnegative integers")
        shape = tuple(int(s) for s in shape)
        size = int(np.prod(shape)) if shape else 1
        lo = np.broadcast_to(np.asarray(lower, float), shape).ravel()
        hi = np.broadcast_to(np.asarray(upper, float), shape).ravel()
        ints = np.broadcast_to(np.asarray(integer, int), shape).ravel()
        if (np.isnan(lo).any() or np.isnan(hi).any() or (lo > hi).any()
                or not np.isin(ints, (0, 1)).all()):
            raise ValueError("Invalid variable bounds or integrality")
        start = self.n_variables
        indices = np.arange(start, start + size, dtype=int).reshape(shape)
        self.lower.extend(lo.tolist())
        self.upper.extend(hi.tolist())
        self.integer.extend(ints.tolist())
        self.variables[name] = indices
        return indices

    def add_row(self, coefficients: Mapping[int, float], lower=-np.inf,
                upper=np.inf):
        if np.isnan(lower) or np.isnan(upper) or lower > upper:
            raise ValueError("Invalid constraint bounds")
        row = {}
        for index, coefficient in coefficients.items():
            if int(index) != index:
                raise ValueError("Constraint variable indices must be integers")
            index, coefficient = int(index), float(coefficient)
            if not 0 <= index < self.n_variables or not np.isfinite(coefficient):
                raise ValueError("Invalid constraint variable or coefficient")
            if coefficient:
                row[index] = coefficient
        self.rows.append(row)
        self.row_lower.append(float(lower))
        self.row_upper.append(float(upper))
        return len(self.rows) - 1

    def arrays(self, objective=None, cmax=True):
        c = np.zeros(self.n_variables)
        if objective is not None:
            if isinstance(objective, Mapping):
                for index, coefficient in objective.items():
                    if int(index) != index or not 0 <= int(index) < self.n_variables:
                        raise ValueError("Objective variable out of range")
                    c[int(index)] = float(coefficient)
            else:
                c = np.asarray(objective, float)
                if c.shape != (self.n_variables,):
                    raise ValueError("Objective shape mismatch")
            if not np.isfinite(c).all():
                raise ValueError("Objective must be finite")
        rr, cc, vv = [], [], []
        for i, row in enumerate(self.rows):
            for j, value in row.items():
                rr.append(i); cc.append(j); vv.append(value)
        matrix = coo_matrix((vv, (rr, cc)),
                            shape=(len(self.rows), self.n_variables)).tocsr()
        constraints = LinearConstraint(matrix, self.row_lower, self.row_upper)
        return dict(c=-c if cmax else c,
                    integrality=np.asarray(self.integer, int),
                    bounds=Bounds(self.lower, self.upper), constraints=constraints)

    def solve(self, objective=None, cmax=True, time_limit=60.0,
              mip_rel_gap=1e-8, **options):
        """Return SciPy result plus objective/bound in the caller's direction.

        For maximization ``objective_bound`` is an upper bound; for
        minimization it is a lower bound. A status code alone is not a gap
        certificate. ``fun`` and ``mip_dual_bound`` retain SciPy's raw signs.
        """
        settings = dict(time_limit=float(time_limit),
                        mip_rel_gap=float(mip_rel_gap), presolve=True)
        settings.update(options)
        result = milp(**self.arrays(objective, cmax), options=settings)
        sign = -1.0 if cmax else 1.0
        result.objective_value = (None if result.fun is None
                                  else sign * float(result.fun))
        bound = getattr(result, "mip_dual_bound", None)
        result.objective_bound = None if bound is None else sign * float(bound)
        return result


@dataclass(frozen=True)
class EVConfig:
    n_slots: int = 8
    n_sites: int = 3
    dt_hours: float = 0.25
    p_max_kw: Any = 6.0
    p_discharge_max_kw: Any = 6.0
    net_grid_energy_kwh: float = 6.0
    battery_capacity_kwh: float = 60.0
    initial_soc: float = 0.4
    soc_min: float = 0.1
    soc_max: float = 0.9
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    initial_site: int = 0
    access: Any = None  # bool [slot, site], connection availability
    travel_slots: Any = 1  # scalar or [origin, destination], diagonal zero
    travel_energy_kwh: Any = 0.2  # deducted once per selected travel arc
    travel_allowed: Any = None  # None means all off-diagonal moves allowed
    allow_discharge: bool = False
    max_throughput_kwh: float | None = None
    terminal_energy_kwh: float | None = None
    terminal_site: int | None = None
    require_terminal_connection: bool = True

    def arrays(self):
        if (not isinstance(self.n_slots, (int, np.integer))
                or not isinstance(self.n_sites, (int, np.integer))
                or self.n_slots < 1 or self.n_sites < 1):
            raise ValueError("Positive integer horizon and number of sites required")
        positive = [self.dt_hours, self.battery_capacity_kwh,
                    self.eta_charge, self.eta_discharge]
        if (not np.isfinite(positive).all() or min(positive) <= 0
                or self.eta_charge > 1 or self.eta_discharge > 1):
            raise ValueError("Invalid time, battery capacity, or efficiency")
        if not 0 <= self.soc_min <= self.initial_soc <= self.soc_max <= 1:
            raise ValueError("SOC limits must contain the initial SOC")
        if not 0 <= self.initial_site < self.n_sites:
            raise ValueError("Initial site is outside the site set")
        if self.terminal_site is not None and not 0 <= self.terminal_site < self.n_sites:
            raise ValueError("Terminal site is outside the site set")
        if not np.isfinite(self.net_grid_energy_kwh):
            raise ValueError("Grid energy target must be finite")
        if not self.allow_discharge and self.net_grid_energy_kwh < 0:
            raise ValueError("Charge-only grid energy must be nonnegative")
        shape = (self.n_slots, self.n_sites)
        access = (np.ones(shape, bool) if self.access is None
                  else np.asarray(self.access, bool))
        if access.shape != shape:
            raise ValueError("Access must have shape (n_slots,n_sites)")
        cap = np.broadcast_to(np.asarray(self.p_max_kw, float),
                              (self.n_sites,)).copy()
        discharge = np.broadcast_to(np.asarray(self.p_discharge_max_kw, float),
                                    (self.n_sites,)).copy()
        if not np.isfinite(np.r_[cap, discharge]).all() or (cap < 0).any() or (discharge < 0).any():
            raise ValueError("Power limits must be finite and nonnegative")
        if not self.allow_discharge:
            discharge[:] = 0.0
        raw_slots = np.asarray(self.travel_slots)
        slots = np.broadcast_to(raw_slots, (self.n_sites, self.n_sites)).copy()
        raw_energy = np.asarray(self.travel_energy_kwh, float)
        energy = np.broadcast_to(raw_energy, slots.shape).copy()
        # A scalar specifies off-diagonal entries, with an automatic zero diagonal.
        if raw_slots.ndim == 0:
            np.fill_diagonal(slots, 0)
        if raw_energy.ndim == 0:
            np.fill_diagonal(energy, 0.0)
        allowed = (np.ones(slots.shape, bool) if self.travel_allowed is None
                   else np.asarray(self.travel_allowed, bool).copy())
        if allowed.shape != slots.shape:
            raise ValueError("Travel permission matrix has the wrong shape")
        np.fill_diagonal(allowed, False)
        if (not np.isfinite(slots).all() or not np.equal(slots, np.floor(slots)).all()
                or (slots < 0).any() or np.any(np.diag(slots) != 0)
                or np.any(slots[allowed] < 1)):
            raise ValueError("Permitted moves need at least one integer travel slot")
        if (not np.isfinite(energy).all() or (energy < 0).any()
                or np.any(np.diag(energy) != 0)):
            raise ValueError("Driving energy must be finite, nonnegative, zero on diagonal")
        for value in (self.max_throughput_kwh, self.terminal_energy_kwh):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError("Optional energy bounds must be finite and nonnegative")
        return access, cap, discharge, slots.astype(int), energy, allowed


@dataclass(frozen=True)
class Arc:
    departure: int
    arrival: int
    origin: int
    destination: int
    travel_energy_kwh: float = 0.0

    @property
    def is_travel(self):
        return self.origin != self.destination


@dataclass(frozen=True)
class EVIndices:
    u_index: np.ndarray
    p_plus_index: np.ndarray
    p_minus_index: np.ndarray
    energy_index: np.ndarray
    arc_index: np.ndarray
    stay_arc_index: np.ndarray
    arcs: tuple[Arc, ...]
    mode_index: np.ndarray | None


def operation_model(config: EVConfig = EVConfig()):
    """Build operations only; return ``(LinearMILP, EVIndices)``.

    Number of network arcs is O(T*n_sites**2), not the number of complete
    paths. There are no zero-time moves. Every selected path has one outgoing
    arc per visited node and can skip intermediate nodes only while travelling.
    """
    access, cap, discharge, travel, drive, allowed = config.arrays()
    T, n = config.n_slots, config.n_sites
    model = LinearMILP()
    arcs, stays = [], np.empty((T, n), int)
    for t in range(T):
        for i in range(n):
            stays[t, i] = len(arcs)
            arcs.append(Arc(t, t + 1, i, i))
        for i in range(n):
            for j in range(n):
                if allowed[i, j] and t + travel[i, j] <= T:
                    arcs.append(Arc(t, t + int(travel[i, j]), i, j, float(drive[i, j])))
    arc_index = model.add_variables("arc", len(arcs), 0, 1, integer=True)
    stay_index = arc_index[stays]
    pp = model.add_variables("p_plus", (T, n), 0, np.broadcast_to(cap, (T, n)))
    pm = model.add_variables("p_minus", (T, n), 0, np.broadcast_to(discharge, (T, n)))
    u = model.add_variables("u", (T, n), -np.broadcast_to(discharge, (T, n)),
                            np.broadcast_to(cap, (T, n)))
    e = model.add_variables("energy", T + 1,
                            config.battery_capacity_kwh * config.soc_min,
                            config.battery_capacity_kwh * config.soc_max)
    mode = (model.add_variables("charge_mode", (T, n), 0, 1, integer=True)
            if config.allow_discharge else None)
    incoming, outgoing = {}, {}
    for k, arc in enumerate(arcs):
        outgoing.setdefault((arc.departure, arc.origin), []).append(int(arc_index[k]))
        incoming.setdefault((arc.arrival, arc.destination), []).append(int(arc_index[k]))
    for t in range(T):
        for i in range(n):
            row = {index: 1.0 for index in outgoing.get((t, i), [])}
            row.update({index: -1.0 for index in incoming.get((t, i), [])})
            rhs = float(t == 0 and i == config.initial_site)
            model.add_row(row, rhs, rhs)
    terminal = {index: 1.0 for i in range(n) for index in incoming.get((T, i), [])}
    model.add_row(terminal, 1.0, 1.0)
    if config.terminal_site is not None:
        model.add_row({index: 1.0 for index in incoming.get((T, config.terminal_site), [])},
                      1.0, 1.0)
    if config.require_terminal_connection:
        model.add_row({int(index): 1.0 for index in stay_index[T - 1]}, 1.0, 1.0)
    for t in range(T):
        for i in range(n):
            stay = int(stay_index[t, i])
            model.add_row({int(pp[t, i]): 1, stay: -cap[i] * access[t, i]}, upper=0)
            model.add_row({int(pm[t, i]): 1, stay: -discharge[i] * access[t, i]}, upper=0)
            model.add_row({int(u[t, i]): 1, int(pp[t, i]): -1, int(pm[t, i]): 1}, 0, 0)
            if mode is not None:
                model.add_row({int(pp[t, i]): 1, int(mode[t, i]): -cap[i]}, upper=0)
                model.add_row({int(pm[t, i]): 1, int(mode[t, i]): discharge[i]},
                              upper=discharge[i])
    initial_energy = config.initial_soc * config.battery_capacity_kwh
    model.add_row({int(e[0]): 1}, initial_energy, initial_energy)
    for t in range(T):
        row = {int(e[t + 1]): 1, int(e[t]): -1}
        for i in range(n):
            row[int(pp[t, i])] = -config.dt_hours * config.eta_charge
            row[int(pm[t, i])] = config.dt_hours / config.eta_discharge
        for k, arc in enumerate(arcs):
            if arc.is_travel and arc.departure == t:
                row[int(arc_index[k])] = arc.travel_energy_kwh
        model.add_row(row, 0, 0)
    grid_row = {int(index): config.dt_hours for index in u.ravel()}
    model.add_row(grid_row, config.net_grid_energy_kwh, config.net_grid_energy_kwh)
    if config.max_throughput_kwh is not None:
        model.add_row({int(index): config.dt_hours for index in np.r_[pp.ravel(), pm.ravel()]},
                      upper=config.max_throughput_kwh)
    if config.terminal_energy_kwh is not None:
        model.add_row({int(e[-1]): 1}, config.terminal_energy_kwh, config.terminal_energy_kwh)
    indices = EVIndices(u, pp, pm, e, arc_index, stay_index, tuple(arcs), mode)
    return model, indices


@dataclass(frozen=True)
class EVPlan:
    selected_arcs: tuple[Arc, ...]
    arc_values: np.ndarray
    p_plus_kw: np.ndarray
    p_minus_kw: np.ndarray
    u_kw: np.ndarray
    energy_kwh: np.ndarray

    def as_dict(self):
        return dict(arcs=[vars(arc) for arc in self.selected_arcs],
                    p_plus_kw=self.p_plus_kw.tolist(), p_minus_kw=self.p_minus_kw.tolist(),
                    u_kw=self.u_kw.tolist(), energy_kwh=self.energy_kwh.tolist())


def decode_plan(solution, indices: EVIndices):
    """Decode either an OptimizeResult or its x vector, retaining raw flow."""
    x = getattr(solution, "x", solution)
    if x is None:
        raise ValueError("Cannot decode a solve without an incumbent")
    x = np.asarray(x, float)
    values = x[indices.arc_index].copy()
    selected = tuple(arc for arc, value in zip(indices.arcs, values) if value > 0.5)
    return EVPlan(selected, values, x[indices.p_plus_index].copy(),
                  x[indices.p_minus_index].copy(), x[indices.u_index].copy(),
                  x[indices.energy_index].copy())


def check_plan(plan: EVPlan, config: EVConfig, tolerance=1e-7):
    """Recompute physical constraints independently of the MILP row matrix."""
    access, cap, discharge, travel, drive, allowed = config.arrays()
    T, n = config.n_slots, config.n_sites
    issues, errors = [], []

    def check(label, value):
        value = float(max(0.0, value))
        errors.append(value)
        if value > tolerance:
            issues.append(label)

    pp, pm, u, energy = (np.asarray(plan.p_plus_kw), np.asarray(plan.p_minus_kw),
                         np.asarray(plan.u_kw), np.asarray(plan.energy_kwh))
    if (pp.shape != (T, n) or pm.shape != (T, n) or u.shape != (T, n)
            or energy.shape != (T + 1,) or not all(np.isfinite(a).all()
                                                 for a in (pp, pm, u, energy, plan.arc_values))):
        return dict(feasible=False, issues=["shape_or_nonfinite"], max_violation=np.inf)
    check("arc_integrality", np.max(abs(plan.arc_values - np.round(plan.arc_values)), initial=0))
    check("arc_bounds", max(np.max(-plan.arc_values, initial=0),
                             np.max(plan.arc_values - 1, initial=0)))
    check("selected_arc_count", abs(np.sum(np.round(plan.arc_values)) - len(plan.selected_arcs)))
    # Follow selected arcs, rejecting extraneous arcs and time/site discontinuities.
    remaining, ordered = list(plan.selected_arcs), []
    node = (0, config.initial_site)
    route_valid = True
    while node[0] < T:
        matches = [arc for arc in remaining if (arc.departure, arc.origin) == node]
        if len(matches) != 1:
            issues.append("path_not_single_connected"); route_valid = False; break
        arc = matches[0]
        remaining.remove(arc); ordered.append(arc)
        if (not 0 <= arc.origin < n or not 0 <= arc.destination < n
                or not 0 <= arc.departure < arc.arrival <= T):
            issues.append("arc_out_of_range"); route_valid = False; break
        if arc.is_travel:
            if (not allowed[arc.origin, arc.destination]
                    or arc.arrival - arc.departure != travel[arc.origin, arc.destination]):
                issues.append("invalid_travel_arc"); route_valid = False; break
            check("arc_travel_energy", abs(arc.travel_energy_kwh - drive[arc.origin, arc.destination]))
        else:
            check("stay_driving_energy", abs(arc.travel_energy_kwh))
            if arc.arrival != arc.departure + 1:
                issues.append("invalid_stay_arc"); route_valid = False; break
        node = (arc.arrival, arc.destination)
    if remaining or node[0] != T:
        issues.append("extraneous_or_unfinished_path"); route_valid = False
    if config.terminal_site is not None and node != (T, config.terminal_site):
        issues.append("terminal_site")
    if config.require_terminal_connection and (not ordered or ordered[-1].is_travel):
        issues.append("terminal_connection")
    connected = np.zeros((T, n), bool)
    movement = np.zeros(T)
    sites = np.full(T, -1, int)
    if route_valid:
        for arc in ordered:
            if arc.is_travel:
                movement[arc.departure] += drive[arc.origin, arc.destination]
            else:
                connected[arc.departure, arc.origin] = True
                sites[arc.departure] = arc.origin
    check("negative_power", max(np.max(-pp, initial=0), np.max(-pm, initial=0)))
    check("charging_access_or_location", np.max(pp - connected * access * cap, initial=0))
    check("discharging_access_or_location", np.max(pm - connected * access * discharge, initial=0))
    check("charge_discharge_overlap", np.max(np.minimum(pp, pm), initial=0))
    check("net_power_definition", np.max(abs(u - pp + pm), initial=0))
    grid_energy = float(config.dt_hours * np.sum(u))
    check("grid_energy", abs(grid_energy - config.net_grid_energy_kwh))
    throughput = float(config.dt_hours * np.sum(pp + pm))
    if config.max_throughput_kwh is not None:
        check("throughput", throughput - config.max_throughput_kwh)
    expected = np.r_[config.initial_soc * config.battery_capacity_kwh,
                      config.initial_soc * config.battery_capacity_kwh + np.cumsum(
                          config.dt_hours * (config.eta_charge * pp.sum(axis=1)
                                             - pm.sum(axis=1) / config.eta_discharge) - movement)]
    check("energy_balance", np.max(abs(energy - expected), initial=0))
    check("soc_min", config.soc_min * config.battery_capacity_kwh - np.min(expected))
    check("soc_max", np.max(expected) - config.soc_max * config.battery_capacity_kwh)
    if config.terminal_energy_kwh is not None:
        check("terminal_energy", abs(expected[-1] - config.terminal_energy_kwh))
    return dict(feasible=not issues, issues=sorted(set(issues)),
                max_violation=max(errors, default=0.0), sites=sites.tolist(),
                grid_energy_kwh=grid_energy, throughput_kwh=throughput,
                travel_energy_kwh=float(movement.sum()),
                energy_kwh=expected.tolist(), soc=(expected / config.battery_capacity_kwh).tolist(),
                route=[vars(arc) for arc in ordered])
