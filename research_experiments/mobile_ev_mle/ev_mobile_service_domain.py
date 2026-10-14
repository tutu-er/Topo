"""One mobile battery's linear operations domain, in kW and kWh.

A single binary path in a time/site/connection-state DAG chooses movement
and setup. Continuous charging/discharging has an explicit time-major net
power projection ``u = u_map @ x``. No complete routes or power curves are
enumerated. There are no topology, AC-truth, or future-noise inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ev_operations_milp import LinearMILP


def _array(value, shape, label, *, nonnegative=True):
    try:
        a = np.broadcast_to(np.asarray(value, float), shape).copy()
    except ValueError as exc:
        raise ValueError(f"{label} must broadcast to {shape}") from exc
    if not np.isfinite(a).all() or (nonnegative and np.any(a < 0)):
        raise ValueError(f"Invalid {label}")
    return a


def _boolean(value, shape, label):
    a = _array(value, shape, label)
    if not np.isin(a, (0, 1)).all():
        raise ValueError(f"{label} must contain booleans or 0/1")
    return a.astype(bool)


@dataclass(frozen=True)
class MobileServiceConfig:
    """Public operations data. ``terminal_energy_kwh`` is battery service.

    ``device_kind`` is a label only: identical EV and ESS inputs yield exactly
    the same operations domain. To compare a stationary device set
    ``mobile=False``. Driver costs are separately budgeted/reported; the
    information objective is not automatically mixed with monetary units.
    """

    n_slots: int = 6
    n_sites: int = 3
    dt_hours: float = .25
    p_charge_max_kw: object = 6.
    p_discharge_max_kw: object = 6.
    allow_discharge: bool = True
    battery_capacity_kwh: float = 60.
    initial_soc: float = .4
    soc_min: float = .1
    soc_max: float = .9
    eta_charge: float = .95
    eta_discharge: float = .95
    terminal_energy_kwh: float = 27.
    initial_site: int = 0
    initial_connected: bool = True
    mobile: bool = True
    access: object = True                 # [time, site]
    worker_available: object = True       # [time, site], setup slots only
    setup_slots: object = 1               # [site], zero allowed
    setup_cost: object = 0.               # [site], user-declared cost units
    travel_slots: object = 1              # [origin, destination], off-diagonal >=1
    travel_energy_kwh: object = .2        # [origin, destination], deduct at departure
    travel_allowed: object = True        # [origin, destination]
    travel_cost: object = 0.              # [origin, destination]
    driver_budget: float | None = None
    max_moves: int | None = None
    max_throughput_kwh: float | None = None
    terminal_site: int | None = None
    terminal_connected: bool = False
    device_kind: str = "ev"
    cost_units: str = "declared_cost_unit"

    def arrays(self):
        for label in ("n_slots", "n_sites"):
            value = getattr(self, label)
            if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        T, J = int(self.n_slots), int(self.n_sites)
        scalars = (self.dt_hours, self.battery_capacity_kwh, self.initial_soc,
                   self.soc_min, self.soc_max, self.eta_charge,
                   self.eta_discharge, self.terminal_energy_kwh)
        if not np.isfinite(scalars).all():
            raise ValueError("Nonfinite battery/time input")
        if not (self.dt_hours > 0 and self.battery_capacity_kwh > 0
                and 0 <= self.soc_min <= self.initial_soc <= self.soc_max <= 1
                and 0 < self.eta_charge <= 1 and 0 < self.eta_discharge <= 1
                and self.soc_min * self.battery_capacity_kwh <= self.terminal_energy_kwh
                <= self.soc_max * self.battery_capacity_kwh):
            raise ValueError("Invalid battery/time bounds or terminal service")
        for label in ("initial_site", "terminal_site"):
            value = getattr(self, label)
            if value is not None and (int(value) != value or not 0 <= value < J):
                raise ValueError(f"Invalid {label}")
        if self.initial_site is None:
            raise ValueError("initial_site is required")
        for label in ("allow_discharge", "initial_connected", "mobile", "terminal_connected"):
            if not isinstance(getattr(self, label), (bool, np.bool_)):
                raise ValueError(f"{label} must be boolean")
        for label in ("driver_budget", "max_throughput_kwh"):
            value = getattr(self, label)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"Invalid {label}")
        if self.max_moves is not None and (isinstance(self.max_moves, bool)
                or int(self.max_moves) != self.max_moves or self.max_moves < 0):
            raise ValueError("max_moves must be a nonnegative integer")
        charge = _array(self.p_charge_max_kw, (T, J), "p_charge_max_kw")
        discharge = _array(self.p_discharge_max_kw, (T, J), "p_discharge_max_kw")
        if not self.allow_discharge:
            discharge[:] = 0
        access = _boolean(self.access, (T, J), "access")
        worker = _boolean(self.worker_available, (T, J), "worker_available")
        setup = _array(self.setup_slots, (J,), "setup_slots")
        travel = _array(self.travel_slots, (J, J), "travel_slots")
        if np.any(setup != np.round(setup)) or np.any(travel != np.round(travel)):
            raise ValueError("Travel/setup durations must be integer slots")
        offdiag = ~np.eye(J, dtype=bool)
        if np.any(travel[offdiag] < 1):
            raise ValueError("Every off-diagonal travel duration must be at least one slot")
        drive = _array(self.travel_energy_kwh, (J, J), "travel_energy_kwh")
        travel_cost = _array(self.travel_cost, (J, J), "travel_cost")
        setup_cost = _array(self.setup_cost, (J,), "setup_cost")
        allowed = _boolean(self.travel_allowed, (J, J), "travel_allowed") & offdiag
        if not self.mobile:
            allowed[:] = False
        # Diagonals are never travel arcs, independent of scalar input broadcasts.
        for a in (travel, drive, travel_cost):
            np.fill_diagonal(a, 0)
        return dict(charge=charge, discharge=discharge, access=access, worker=worker,
                    setup=setup.astype(int), travel=travel.astype(int), drive=drive,
                    travel_cost=travel_cost, setup_cost=setup_cost, allowed=allowed)


@dataclass(frozen=True)
class ServiceArc:
    departure: int
    arrival: int
    origin: int
    destination: int
    origin_connected: bool
    destination_connected: bool
    kind: str                        # wait, connected, setup, travel
    driving_energy_kwh: float = 0.
    driver_cost: float = 0.

    @property
    def start(self):
        return self.departure, self.origin, self.origin_connected

    @property
    def end(self):
        return self.arrival, self.destination, self.destination_connected


@dataclass(frozen=True)
class MobileServiceIndices:
    u: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    energy: np.ndarray
    mode: np.ndarray
    arc: np.ndarray
    connected_arc: np.ndarray
    arcs: tuple[ServiceArc, ...]


@dataclass
class MobileServiceDomain:
    config: MobileServiceConfig
    model: LinearMILP
    indices: MobileServiceIndices
    cost_vector: np.ndarray

    def as_dict(self):
        """Generic MILP selector API; u is time-major and measured in kW."""
        arrays = self.model.arrays()
        controls = self.indices.u.ravel()
        mapping = np.zeros((len(controls), self.model.n_variables))
        mapping[np.arange(len(controls)), controls] = 1.
        lower, upper = np.asarray(self.model.lower), np.asarray(self.model.upper)
        return dict(lower=lower.copy(), upper=upper.copy(),
                    integrality=np.asarray(self.model.integer, int),
                    constraints=[arrays["constraints"]], control_indices=controls.copy(),
                    u_map=mapping, control_lower=lower[controls].copy(),
                    control_upper=upper[controls].copy(),
                    zero_product_pairs=[(t * self.config.n_sites + j,
                                         t * self.config.n_sites + k)
                                        for t in range(self.config.n_slots)
                                        for j in range(self.config.n_sites)
                                        for k in range(j + 1, self.config.n_sites)],
                    zero_product_proof="single_EV_one_site_per_slot; u linked occupancy")

    def solve_prices(self, prices, *, driver_cost_weight=0., **options):
        """Audit/helper solve: maximize sum(price*u) minus declared cost weight.

        A nonzero cost weight must have units score/cost. Domain budgets allow
        a score-only objective without requiring such a conversion.
        """
        prices = np.asarray(prices, float)
        if prices.shape != self.indices.u.shape or not np.isfinite(prices).all():
            raise ValueError("prices must be finite [time,site]")
        if not np.isfinite(driver_cost_weight) or driver_cost_weight < 0:
            raise ValueError("driver_cost_weight must be finite and nonnegative")
        objective = -float(driver_cost_weight) * self.cost_vector.copy()
        objective[self.indices.u.ravel()] += prices.ravel()
        return self.model.solve(objective, **options)

    def decode(self, result):
        x = getattr(result, "x", result)
        if x is None:
            raise ValueError("No incumbent to decode")
        x = np.asarray(x, float)
        if x.shape != (self.model.n_variables,) or not np.isfinite(x).all():
            raise ValueError("Invalid full operation vector")
        ix = self.indices
        return dict(u_kw=x[ix.u].copy(), charge_kw=x[ix.charge].copy(),
                    discharge_kw=x[ix.discharge].copy(), energy_kwh=x[ix.energy].copy(),
                    arc_values=x[ix.arc].copy(),
                    selected_arcs=[a for a, y in zip(ix.arcs, x[ix.arc]) if y > .5],
                    driver_cost=float(self.cost_vector @ x))

    def check(self, result, tolerance=1e-7):
        return check_mobile_service_plan(self, result, tolerance=tolerance)


def build_mobile_service_domain(config=MobileServiceConfig()):
    """Build O(T*J^2) network arcs, never complete routes.

    A zero-slot setup arc is directed unplugged -> connected. All other arcs
    increase time, so these arcs do not introduce a directed cycle.
    """
    data = config.arrays()
    T, J = int(config.n_slots), int(config.n_sites)
    model, arcs = LinearMILP(), []
    connected_ids = np.empty((T, J), int)
    for t in range(T):
        for j in range(J):
            arcs.append(ServiceArc(t, t + 1, j, j, False, False, "wait"))
            connected_ids[t, j] = len(arcs)
            arcs.append(ServiceArc(t, t + 1, j, j, True, True, "connected"))
            duration = int(data["setup"][j])
            stop = t + duration
            # Zero-duration setup uses that instant's public access/worker.
            slots = slice(t, max(t + 1, stop))
            if (stop <= T and data["access"][slots, j].all()
                    and data["worker"][slots, j].all()):
                arcs.append(ServiceArc(t, stop, j, j, False, True, "setup",
                                       driver_cost=float(data["setup_cost"][j])))
            for dest in range(J):
                arrival = t + int(data["travel"][j, dest])
                if data["allowed"][j, dest] and arrival <= T:
                    for connected in (False, True):
                        arcs.append(ServiceArc(t, arrival, j, dest, connected, False,
                                               "travel", float(data["drive"][j, dest]),
                                               float(data["travel_cost"][j, dest])))
    arc = model.add_variables("arc", len(arcs), 0, 1, integer=True)
    pp = model.add_variables("charge", (T, J), 0, data["charge"] * data["access"])
    pm = model.add_variables("discharge", (T, J), 0, data["discharge"] * data["access"])
    u = model.add_variables("u", (T, J), -data["discharge"] * data["access"],
                            data["charge"] * data["access"])
    energy = model.add_variables("energy", T + 1,
                                 config.battery_capacity_kwh * config.soc_min,
                                 config.battery_capacity_kwh * config.soc_max)
    mode = model.add_variables("charge_mode", T, 0, 1, integer=True)
    incoming, outgoing = {}, {}
    for index, a in zip(arc, arcs):
        incoming.setdefault(a.end, []).append(int(index))
        outgoing.setdefault(a.start, []).append(int(index))
    source = (0, int(config.initial_site), bool(config.initial_connected))
    for t in range(T):
        for j in range(J):
            for state in (False, True):
                node = t, j, state
                row = {i: 1. for i in outgoing.get(node, [])}
                for i in incoming.get(node, []):
                    row[i] = row.get(i, 0.) - 1.
                rhs = float(node == source)
                model.add_row(row, rhs, rhs)
    terminal = [int(i) for i, a in zip(arc, arcs) if a.arrival == T]
    model.add_row({i: 1 for i in terminal}, 1, 1)
    if config.terminal_site is not None:
        model.add_row({int(i): 1 for i, a in zip(arc, arcs)
                       if a.arrival == T and a.destination == config.terminal_site}, 1, 1)
    if config.terminal_connected:
        model.add_row({int(i): 1 for i, a in zip(arc, arcs)
                       if a.arrival == T and a.destination_connected}, 1, 1)
    connected_arc = arc[connected_ids]
    for t in range(T):
        for j in range(J):
            C, D = data["charge"][t, j], data["discharge"][t, j]
            y = int(connected_arc[t, j])
            model.add_row({int(pp[t, j]): 1, y: -C}, upper=0)
            model.add_row({int(pm[t, j]): 1, y: -D}, upper=0)
            model.add_row({int(pp[t, j]): 1, int(mode[t]): -C}, upper=0)
            model.add_row({int(pm[t, j]): 1, int(mode[t]): D}, upper=D)
            model.add_row({int(u[t, j]): 1, int(pp[t, j]): -1,
                           int(pm[t, j]): 1}, 0, 0)
    E0 = config.initial_soc * config.battery_capacity_kwh
    model.add_row({int(energy[0]): 1}, E0, E0)
    for t in range(T):
        row = {int(energy[t + 1]): 1, int(energy[t]): -1}
        for j in range(J):
            row[int(pp[t, j])] = -config.dt_hours * config.eta_charge
            row[int(pm[t, j])] = config.dt_hours / config.eta_discharge
        for i, a in zip(arc, arcs):
            if a.kind == "travel" and a.departure == t:
                row[int(i)] = a.driving_energy_kwh
        model.add_row(row, 0, 0)
    model.add_row({int(energy[-1]): 1}, config.terminal_energy_kwh,
                  config.terminal_energy_kwh)
    cost = np.zeros(model.n_variables)
    for i, a in zip(arc, arcs):
        cost[int(i)] = a.driver_cost
    if config.driver_budget is not None:
        model.add_row({int(i): a.driver_cost for i, a in zip(arc, arcs)},
                      upper=config.driver_budget)
    if config.max_moves is not None:
        model.add_row({int(i): 1 for i, a in zip(arc, arcs) if a.kind == "travel"},
                      upper=config.max_moves)
    if config.max_throughput_kwh is not None:
        model.add_row({int(i): config.dt_hours for i in np.r_[pp.ravel(), pm.ravel()]},
                      upper=config.max_throughput_kwh)
    indices = MobileServiceIndices(u, pp, pm, energy, mode, arc, connected_arc, tuple(arcs))
    return MobileServiceDomain(config, model, indices, cost)


def check_mobile_service_plan(domain, result, *, tolerance=1e-7):
    """Physical replay from selected path and powers, plus full linear audit."""
    x = getattr(result, "x", result)
    if x is None:
        return dict(feasible=False, issues=["no_incumbent"], max_violation=np.inf)
    x = np.asarray(x, float)
    if x.shape != (domain.model.n_variables,) or not np.isfinite(x).all():
        return dict(feasible=False, issues=["shape_or_nonfinite"], max_violation=np.inf)
    cfg, ix, data = domain.config, domain.indices, domain.config.arrays()
    T, J = int(cfg.n_slots), int(cfg.n_sites)
    errors, issues = [], []

    def check(label, error):
        error = float(max(0., error))
        errors.append(error)
        if error > tolerance:
            issues.append(label)

    low, high = np.asarray(domain.model.lower), np.asarray(domain.model.upper)
    integers = np.asarray(domain.model.integer, bool)
    check("variable_bounds", max(np.max(low - x, initial=0), np.max(x - high, initial=0)))
    check("integrality", np.max(abs(x[integers] - np.round(x[integers])), initial=0))
    constraints = domain.model.arrays()["constraints"]
    ax = constraints.A @ x
    check("linear_constraints", max(np.max(constraints.lb - ax, initial=0),
                                     np.max(ax - constraints.ub, initial=0)))
    remaining = [a for a, value in zip(ix.arcs, x[ix.arc]) if value > .5]
    ordered, node = [], (0, int(cfg.initial_site), bool(cfg.initial_connected))
    connected = np.zeros((T, J), bool)
    driving = np.zeros(T)
    cost, moves = 0., 0
    while node[0] < T and len(ordered) <= len(ix.arcs):
        matches = [a for a in remaining if a.start == node]
        if len(matches) != 1:
            issues.append("path_not_single_connected")
            break
        a = matches[0]
        remaining.remove(a)
        ordered.append(a)
        if a.kind == "connected":
            connected[a.departure, a.origin] = True
        elif a.kind == "travel":
            moves += 1
            driving[a.departure] += data["drive"][a.origin, a.destination]
            check("travel_duration", abs(a.arrival - a.departure
                                          - data["travel"][a.origin, a.destination]))
            if not data["allowed"][a.origin, a.destination]:
                issues.append("travel_not_allowed")
        elif a.kind == "setup":
            sl = slice(a.departure, max(a.departure + 1, a.arrival))
            if not (data["access"][sl, a.origin].all()
                    and data["worker"][sl, a.origin].all()):
                issues.append("setup_access_or_worker")
        cost += a.driver_cost
        node = a.end
    if remaining or node[0] != T:
        issues.append("extraneous_or_unfinished_path")
    if cfg.terminal_site is not None and node[1] != cfg.terminal_site:
        issues.append("terminal_site")
    if cfg.terminal_connected and not node[2]:
        issues.append("terminal_connection")
    pp, pm, u, E = x[ix.charge], x[ix.discharge], x[ix.u], x[ix.energy]
    check("nonnegative_powers", max(np.max(-pp, initial=0), np.max(-pm, initial=0)))
    check("charging_location_or_access", np.max(pp - connected * data["access"]
                                               * data["charge"], initial=0))
    check("discharging_location_or_access", np.max(pm - connected * data["access"]
                                                  * data["discharge"], initial=0))
    check("charge_discharge_overlap", np.max(np.minimum(pp, pm), initial=0))
    check("net_power", np.max(abs(u - pp + pm), initial=0))
    expected = np.r_[cfg.initial_soc * cfg.battery_capacity_kwh,
                     cfg.initial_soc * cfg.battery_capacity_kwh + np.cumsum(
                         cfg.dt_hours * (cfg.eta_charge * pp.sum(axis=1)
                                         - pm.sum(axis=1) / cfg.eta_discharge) - driving)]
    check("energy_balance", np.max(abs(E - expected), initial=0))
    check("soc_min", cfg.soc_min * cfg.battery_capacity_kwh - np.min(expected))
    check("soc_max", np.max(expected) - cfg.soc_max * cfg.battery_capacity_kwh)
    check("terminal_energy", abs(expected[-1] - cfg.terminal_energy_kwh))
    throughput = cfg.dt_hours * float(np.sum(pp + pm))
    if cfg.max_throughput_kwh is not None:
        check("throughput", throughput - cfg.max_throughput_kwh)
    if cfg.driver_budget is not None:
        check("driver_budget", cost - cfg.driver_budget)
    if cfg.max_moves is not None:
        check("max_moves", moves - cfg.max_moves)
    return dict(feasible=not issues, issues=sorted(set(issues)),
                max_violation=max(errors, default=0.),
                u_kw=u.tolist(), energy_kwh=expected.tolist(),
                soc=(expected / cfg.battery_capacity_kwh).tolist(),
                connected=connected.tolist(),
                grid_energy_kwh=cfg.dt_hours * float(np.sum(u)),
                throughput_kwh=throughput, travel_energy_kwh=float(driving.sum()),
                conversion_loss_kwh=cfg.dt_hours * float(np.sum(
                    (1 - cfg.eta_charge) * pp + (1 / cfg.eta_discharge - 1) * pm)),
                driver_cost=cost, cost_units=cfg.cost_units, moves=moves,
                route=[vars(a) for a in ordered])
