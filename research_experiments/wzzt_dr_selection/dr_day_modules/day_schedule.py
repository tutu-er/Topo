"""Resource-day DR modules and a linear operating subproblem.

The identifier is not imported. A module amplitude is in kW. Power/energy
capability comes from explicit device declarations, never aggregate net load.
The LP below optimizes declared operating costs/tasks, not WZZT final risk.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
from typing import Mapping

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from research_experiments.wzzt_dr_selection.dr_competition_bridge.coded_shapes import (
    contract_quantities, feasible_on_declared_device_band,
)

SLOTS = 96
DT = .25


def _vector(value, name, *, nonnegative=False):
    try:
        a = np.broadcast_to(np.asarray(value, float), (SLOTS,)).copy()
    except ValueError as error:
        raise ValueError(name+" must broadcast to 96 slots") from error
    if not np.isfinite(a).all() or (nonnegative and (a < 0).any()):
        raise ValueError(name+" must be finite"+(" and nonnegative" if nonnegative else ""))
    a.flags.writeable = False
    return a


def _name(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(label+" must be a nonempty string")
    return value


def _slots(values, label):
    values = list(values)
    if (not values or any(isinstance(v, (bool, np.bool_)) or
                         not isinstance(v, (int, np.integer)) for v in values) or
            len(set(values)) != len(values) or min(values) < 0 or max(values) >= SLOTS):
        raise ValueError(label+" requires unique integer slots in [0, 95]")
    return np.asarray(values, int)


@dataclass(frozen=True)
class Battery:
    capacity_kwh: float
    initial_kwh: float
    terminal_kwh: float
    soc_min: float = 0.
    soc_max: float = 1.
    eta_charge: float = 1.

    def __post_init__(self):
        values = (self.capacity_kwh, self.initial_kwh, self.terminal_kwh,
                  self.soc_min, self.soc_max, self.eta_charge)
        if not np.isfinite(values).all() or self.capacity_kwh <= 0:
            raise ValueError("invalid battery declaration")
        if not (0 <= self.soc_min <= self.soc_max <= 1 and 0 < self.eta_charge <= 1):
            raise ValueError("invalid SOC/efficiency")
        low, high = self.capacity_kwh*self.soc_min, self.capacity_kwh*self.soc_max
        if not (low <= self.initial_kwh <= high and low <= self.terminal_kwh <= high):
            raise ValueError("initial/terminal energy outside battery bounds")


@dataclass(frozen=True)
class ResourceDay:
    resource_id: str
    node_label: int
    baseline_kw: object
    power_upper_kw: object
    prefix_delta_lower_kwh: object
    prefix_delta_upper_kwh: object
    available: object = True
    power_lower_kw: object = 0.
    reactive_ratio: object = .35
    incentive_per_kwh: object = 0.
    electricity_price_per_kwh: object = 0.
    kind: str = "shift"
    battery: Battery | None = None
    service_deadline_slot: int | None = None
    service_deadline_source: str = "explicit planned declaration"
    declaration_source: str = "explicit synthetic resource declaration"

    def __post_init__(self):
        _name(self.resource_id, "resource_id")
        _name(self.declaration_source, "declaration_source")
        if self.battery is not None and not isinstance(self.battery, Battery):
            raise TypeError("battery must be an explicit Battery declaration")
        if self.kind == "ev" and self.battery is None:
            raise ValueError("EV resources require a battery declaration")
        deadline = self.service_deadline_slot
        if deadline is not None:
            if (isinstance(deadline, (bool, np.bool_)) or
                    not isinstance(deadline, (int, np.integer)) or not 1 <= deadline <= SLOTS):
                raise ValueError("service_deadline_slot is an integer boundary in [1, 96]")
            object.__setattr__(self, "service_deadline_slot", int(deadline))
            _name(self.service_deadline_source, "service_deadline_source")
        elif self.kind == "ev":
            raise ValueError("EV resources require a planned service_deadline_slot")
        if (isinstance(self.node_label, (bool, np.bool_)) or
                not isinstance(self.node_label, (int, np.integer))):
            raise ValueError("node_label must be an integer")
        for key in ("baseline_kw", "power_upper_kw", "power_lower_kw", "reactive_ratio",
                    "incentive_per_kwh"):
            object.__setattr__(self, key, _vector(getattr(self, key), key, nonnegative=True))
        for key in ("prefix_delta_lower_kwh", "prefix_delta_upper_kwh",
                    "electricity_price_per_kwh"):
            object.__setattr__(self, key, _vector(getattr(self, key), key))
        available = np.asarray(self.available)
        if available.dtype.kind != "b":
            raise ValueError("availability must be an explicit boolean mask")
        available = np.broadcast_to(available, (SLOTS,)).copy()
        available.flags.writeable = False
        object.__setattr__(self, "available", available)
        if (self.power_lower_kw > self.power_upper_kw).any():
            raise ValueError("power lower bound exceeds upper bound")
        if ((self.baseline_kw < self.power_lower_kw-1e-12) |
                (self.baseline_kw > self.power_upper_kw+1e-12)).any():
            raise ValueError("device baseline itself violates declared power bounds")
        lo, hi = self.energy_bounds()
        if (lo > hi+1e-12).any() or (lo > 1e-12).any() or (hi < -1e-12).any():
            raise ValueError("baseline must be feasible in the declared prefix-energy band")
        if self.battery is not None:
            if self.kind != "ev" or (self.baseline_kw[~available] > 1e-12).any():
                raise ValueError("EV baseline must respect its connection window")
            energy = self.battery.initial_kwh + self.battery.eta_charge*DT*np.cumsum(self.baseline_kw)
            if abs(energy[-1]-self.battery.terminal_kwh) > 1e-8:
                raise ValueError("EV baseline must deliver the declared terminal task")
            if abs(energy[self.service_deadline_slot-1]-self.battery.terminal_kwh) > 1e-8:
                raise ValueError("EV baseline must deliver the declared task by its planned deadline")

    def energy_bounds(self):
        lo = np.asarray(self.prefix_delta_lower_kwh, float).copy()
        hi = np.asarray(self.prefix_delta_upper_kwh, float).copy()
        if self.battery is not None:
            b = self.battery
            reference = b.initial_kwh+b.eta_charge*DT*np.cumsum(self.baseline_kw)
            lo = np.maximum(lo, (b.soc_min*b.capacity_kwh-reference)/b.eta_charge)
            hi = np.minimum(hi, (b.soc_max*b.capacity_kwh-reference)/b.eta_charge)
        if self.service_deadline_slot is not None:
            index = self.service_deadline_slot-1
            lo[index] = max(lo[index], 0.)
            hi[index] = min(hi[index], 0.)
        return lo, hi


def ev_resource(resource_id, node_label, baseline_kw, power_upper_kw, available, battery,
                *, incentive_per_kwh=0., electricity_price_per_kwh=0.,
                service_deadline_slot=None,
                declaration_source="explicit synthetic EV declaration"):
    """Charge-only EV with a planned service deadline, distinct from actual departure.

    The helper may infer the deadline from one explicit planned connection window.
    Multiple windows require the caller's planned deadline. Upstream must freeze
    these planning inputs before any actual availability or disconnect is read.
    """
    if not isinstance(battery, Battery):
        raise TypeError("an explicit Battery declaration is required")
    base = _vector(baseline_kw, "EV baseline", nonnegative=True)
    raw_available = np.asarray(available)
    if raw_available.dtype.kind != "b":
        raise ValueError("planned EV availability must be an explicit boolean mask")
    planned_available = np.broadcast_to(raw_available, (SLOTS,)).copy()
    if service_deadline_slot is None:
        positions = np.flatnonzero(planned_available)
        if not len(positions) or (np.diff(positions) != 1).any():
            raise ValueError("multiple or empty planned connection windows require an explicit deadline")
        service_deadline_slot = int(positions[-1]+1)
        deadline_source = "provided planned single connection-window end"
    else:
        deadline_source = "explicit planned service deadline"
    reference = battery.initial_kwh+battery.eta_charge*DT*np.cumsum(base)
    return ResourceDay(
        resource_id, node_label, base, power_upper_kw,
        (battery.soc_min*battery.capacity_kwh-reference)/battery.eta_charge,
        (battery.soc_max*battery.capacity_kwh-reference)/battery.eta_charge,
        available=planned_available, reactive_ratio=0., incentive_per_kwh=incentive_per_kwh,
        electricity_price_per_kwh=electricity_price_per_kwh, kind="ev", battery=battery,
        service_deadline_slot=service_deadline_slot, service_deadline_source=deadline_source,
        declaration_source=declaration_source)


@dataclass(frozen=True)
class DayModule:
    module_id: str
    profiles: Mapping[str, object]
    amplitude_upper_kw: float
    description: str = ""

    def __post_init__(self):
        _name(self.module_id, "module_id")
        if not self.profiles:
            raise ValueError("module must refer to at least one physical resource")
        if not np.isfinite(self.amplitude_upper_kw) or self.amplitude_upper_kw < 0:
            raise ValueError("invalid amplitude upper bound")
        profiles = {}
        for resource_id, value in self.profiles.items():
            _name(resource_id, "module resource_id")
            profile = _vector(value, "module profile")
            if abs(profile.sum()*DT) > 1e-10*max(1., np.abs(profile).sum()*DT):
                raise ValueError("each resource module must have zero daily incremental energy")
            profiles[resource_id] = profile
        object.__setattr__(self, "profiles", types.MappingProxyType(profiles))


def shift_shape(reduction_slots, rebound_slots):
    reduction, rebound = _slots(reduction_slots, "reduction"), _slots(rebound_slots, "rebound")
    if set(reduction) & set(rebound):
        raise ValueError("reduction and rebound slots must be disjoint")
    profile = np.zeros(SLOTS)
    profile[reduction] = -1.
    profile[rebound] = len(reduction)/len(rebound)
    return profile


def shift_module(module_id, resource_id, reduction_slots, rebound_slots, amplitude_upper_kw):
    return DayModule(module_id, {resource_id:shift_shape(reduction_slots, rebound_slots)},
                     amplitude_upper_kw, "single-resource reduction and rebound")


def ev_reallocation_module(module_id, resource, reduction_slots, rebound_slots, amplitude_upper_kw):
    if not isinstance(resource, ResourceDay) or resource.kind != "ev":
        raise ValueError("EV reallocation requires an EV resource")
    slots = np.r_[_slots(reduction_slots, "reduction"), _slots(rebound_slots, "rebound")]
    if not resource.available[slots].all():
        raise ValueError("EV module extends outside the declared connection window")
    return shift_module(module_id, resource.resource_id, reduction_slots, rebound_slots,
                        amplitude_upper_kw)


def spatial_module(module_id, waveform, resource_signs, amplitude_upper_kw):
    """Same/opposed pattern; actual joint participation is a separate model."""
    if not resource_signs or any(isinstance(v, (bool, np.bool_)) or v not in (-1, 1)
                                 for v in resource_signs.values()):
        raise ValueError("spatial signs must be explicit -1 or +1")
    profile = _vector(waveform, "spatial waveform")
    return DayModule(module_id, {key:float(sign)*profile for key,sign in resource_signs.items()},
                     amplitude_upper_kw, "declared spatial pattern; no topology input")


def complementary_modules(prefix, resources, waveforms, amplitude_upper_kw):
    """Two candidate spatial bases with explicitly distinct temporal profiles.

    Nonzero amplitudes and sufficient resource delivery are still needed for
    two executed directions. This constructor does not certify observed input
    rank, useful information, or topology recovery. Zero amplitudes remain legal.
    """
    if len(resources) != 3 or len(set(resources)) != 3:
        raise ValueError("three distinct resource IDs required")
    try:
        waveforms = list(waveforms)
    except TypeError as error:
        raise ValueError("two explicit temporal waveforms are required") from error
    if len(waveforms) != 2:
        raise ValueError("two explicit temporal waveforms are required")
    first = _vector(waveforms[0], "first temporal waveform")
    second = _vector(waveforms[1], "second temporal waveform")
    if np.linalg.matrix_rank(np.vstack((first, second))) < 2:
        raise ValueError("temporal waveforms are numerically collinear; not complementary execution")
    i,j,k = resources
    return (
        spatial_module(prefix+"_first", first, {i:-1,j:1}, amplitude_upper_kw),
        spatial_module(prefix+"_second", second, {j:1,k:-1}, amplitude_upper_kw),
    )


@dataclass(frozen=True)
class LinearTask:
    name: str
    coefficients: Mapping[str, object]
    lower: float = -np.inf
    upper: float = np.inf
    units: str = "caller-declared linear functional"

    def __post_init__(self):
        _name(self.name, "task name")
        if np.isnan((self.lower,self.upper)).any() or self.lower > self.upper:
            raise ValueError("invalid linear task bounds")
        if not np.isfinite((self.lower,self.upper)).any():
            raise ValueError("task needs at least one finite bound")
        object.__setattr__(self, "coefficients",
                           {key:_vector(value, "task coefficient") for key,value in self.coefficients.items()})


def _layout(resources, modules):
    resources, modules = list(resources), list(modules)
    if not resources or any(not isinstance(r, ResourceDay) for r in resources):
        raise ValueError("explicit ResourceDay declarations are required")
    if len({r.resource_id for r in resources}) != len(resources):
        raise ValueError("resource_id has one unique baseline; DR and EV cannot duplicate it")
    if any(not isinstance(m, DayModule) for m in modules):
        raise ValueError("DayModule declarations are required")
    if len({m.module_id for m in modules}) != len(modules):
        raise ValueError("duplicate module_id")
    lookup = {r.resource_id:i for i,r in enumerate(resources)}
    matrix = np.zeros((len(resources)*SLOTS, len(modules)))
    for j,module in enumerate(modules):
        for key,profile in module.profiles.items():
            if key not in lookup:
                raise ValueError("module refers to an undeclared resource: "+key)
            i = lookup[key]
            matrix[i*SLOTS:(i+1)*SLOTS,j] = profile
    return resources, modules, lookup, matrix


def evaluate_schedule(resources, modules, amplitudes_kw, *, rho=None, tolerance=1e-7):
    """Audit aggregate resource trajectories; same rho scales every module of a resource.

    Network feasibility is deliberately unavailable here. No clipping or
    resampling is performed, and billing uses aggregate net resource reduction.
    """
    resources, modules, _, _ = _layout(resources, modules)
    amplitudes = np.asarray(amplitudes_kw, float)
    if amplitudes.shape != (len(modules),) or not np.isfinite(amplitudes).all():
        raise ValueError("amplitudes must match the module list")
    rho = {} if rho is None else dict(rho)
    unknown = set(rho)-{r.resource_id for r in resources}
    if unknown:
        raise ValueError("rho refers to an undeclared resource")
    records, nodes = {}, {}
    payment = full_payment = bill = maximum = 0.
    for resource in resources:
        key = resource.resource_id
        scale = float(rho.get(key, 1.))
        if not np.isfinite(scale) or not 0 <= scale <= 1:
            raise ValueError("rho is one resource-day delivery fraction in [0,1]")
        planned = sum((a*m.profiles[key] for a,m in zip(amplitudes,modules) if key in m.profiles),
                      np.zeros(SLOTS))
        delta = scale*planned
        actual = resource.baseline_kw+delta
        lo,hi = resource.energy_bounds()
        prefix = DT*np.cumsum(delta)
        quantities = contract_quantities(delta[:,None], DT)
        full_prefix = DT*np.cumsum(planned)
        full_power = resource.baseline_kw+planned
        full_violation = max(
            float(np.max(resource.power_lower_kw-full_power, initial=0.)),
            float(np.max(full_power-resource.power_upper_kw, initial=0.)),
            float(np.max(lo-full_prefix, initial=0.)), float(np.max(full_prefix-hi, initial=0.)),
            float(np.max(np.abs(planned[~resource.available]), initial=0.)), abs(float(full_prefix[-1])))
        violation = max(
            float(np.max(resource.power_lower_kw-actual, initial=0.)),
            float(np.max(actual-resource.power_upper_kw, initial=0.)),
            float(np.max(lo-prefix, initial=0.)), float(np.max(prefix-hi, initial=0.)),
            float(np.max(np.abs(delta[~resource.available]), initial=0.)), abs(float(prefix[-1])))
        strict = feasible_on_declared_device_band(
            delta[:,None], resource.baseline_kw[:,None], resource.power_upper_kw[:,None],
            lo[:,None], hi[:,None], DT)
        incentive = float(DT*np.dot(resource.incentive_per_kwh, np.maximum(-delta,0.)))
        full_incentive = float(DT*np.dot(resource.incentive_per_kwh, np.maximum(-planned,0.)))
        delta_bill = float(DT*np.dot(resource.electricity_price_per_kwh, delta))
        row = dict(rho=scale, baseline_kw=resource.baseline_kw.tolist(), planned_delta_kw=planned.tolist(),
                   executed_delta_kw=delta.tolist(), actual_kw=actual.tolist(),
                   prefix_incremental_grid_kwh=prefix.tolist(),
                   reduced_kwh=float(quantities["reduced_energy_kWh"][0]),
                   rebound_kwh=float(quantities["added_energy_kWh"][0]),
                   net_kwh=float(quantities["net_energy_kWh"][0]),
                   incentive_payment=incentive, incremental_electricity_bill=delta_bill,
                   maximum_violation=max(0.,violation,full_violation),
                   full_plan_feasible=full_violation <= tolerance,
                   actual_resource_feasible=violation <= tolerance,
                   feasible=max(violation,full_violation) <= tolerance,
                   full_delivery_incentive_payment=full_incentive, baseline_feasible=True,
                   strict_provider_band_pass=bool(strict), declaration_source=resource.declaration_source)
        if resource.battery is not None:
            b = resource.battery
            energy = b.initial_kwh+b.eta_charge*DT*np.cumsum(actual)
            full_energy = b.initial_kwh+b.eta_charge*DT*np.cumsum(full_power)
            full_violation = max(full_violation,
                                 float(np.max(b.capacity_kwh*b.soc_min-full_energy, initial=0.)),
                                 float(np.max(full_energy-b.capacity_kwh*b.soc_max, initial=0.)),
                                 abs(float(full_energy[-1]-b.terminal_kwh)))
            violation = max(violation, float(np.max(b.capacity_kwh*b.soc_min-energy, initial=0.)),
                            float(np.max(energy-b.capacity_kwh*b.soc_max, initial=0.)),
                            abs(float(energy[-1]-b.terminal_kwh)))
            deadline = resource.service_deadline_slot
            full_deadline_energy = float(full_energy[deadline-1])
            actual_deadline_energy = float(energy[deadline-1])
            full_violation = max(full_violation, abs(full_deadline_energy-b.terminal_kwh))
            violation = max(violation, abs(actual_deadline_energy-b.terminal_kwh))
            row.update(planned_service_deadline_slot=deadline,
                       service_deadline_source=resource.service_deadline_source,
                       service_target_kwh=b.terminal_kwh,
                       planned_deadline_energy_kwh=full_deadline_energy,
                       actual_deadline_energy_kwh=actual_deadline_energy,
                       planned_deadline_pass=abs(full_deadline_energy-b.terminal_kwh) <= tolerance,
                       actual_deadline_pass=abs(actual_deadline_energy-b.terminal_kwh) <= tolerance)
            row.update(battery_energy_kwh=np.r_[b.initial_kwh,energy].tolist(),
                       planned_battery_energy_kwh=np.r_[b.initial_kwh,full_energy].tolist(),
                       planned_soc=(np.r_[b.initial_kwh,full_energy]/b.capacity_kwh).tolist(),
                       full_plan_feasible=full_violation <= tolerance,
                       terminal_energy_kwh=float(energy[-1]))
            row["soc"] = (np.r_[b.initial_kwh,energy]/b.capacity_kwh).tolist()
            row.update(maximum_violation=max(0.,violation,full_violation),
                       actual_resource_feasible=violation <= tolerance,
                       feasible=max(violation,full_violation) <= tolerance)
        records[key] = row
        entry = nodes.setdefault(int(resource.node_label), dict(delta_kw=np.zeros(SLOTS),delta_kvar=np.zeros(SLOTS)))
        entry["delta_kw"] += delta
        entry["delta_kvar"] += resource.reactive_ratio*delta
        payment += incentive; full_payment += full_incentive; bill += delta_bill
        maximum = max(maximum,violation,full_violation)
    amp_violation = max(float(np.max(-amplitudes, initial=0.)),
                        float(np.max(amplitudes-np.asarray([m.amplitude_upper_kw for m in modules]), initial=0.)))
    maximum = max(maximum,amp_violation,0.)
    return dict(resource_feasible=maximum <= tolerance,maximum_violation=maximum,
                resources=records,node_increments={str(k):{key:a.tolist() for key,a in v.items()} for k,v in nodes.items()},
                total_incentive_payment=payment,full_delivery_incentive_payment=full_payment,
                total_incremental_electricity_bill=bill,
                amplitude_bounds_violation=max(0.,amp_violation),rho_scope="one fraction per resource-day",
                network_feasibility="not evaluated; requires AC execution",
                availability_scope="declared fixed windows; actual window changes need a new resource audit",
                increment_only_payload=True,
                terminal_payload_semantics="node_increments contains deltas; resource audit also includes baselines",
                double_counting_bridge_check="not evaluated; bridge must verify total_after-total_before equals increments",
                no_clipping=True)


def optimize_amplitudes(resources, modules, *, incentive_budget=1.6, selected_module_ids=None,
                        tasks=(), amplitude_cost_per_kw=None, incentive_weight=1.,
                        time_limit_seconds=30., tolerance=1e-7):
    """Continuous LP for specified operating costs/tasks, not topology-risk optimization.

    Nonzero amplitudes select modules when no real setup/cardinality restriction
    exists. The budget is on net resource reductions, not gross module labels.
    """
    resources, modules, lookup, B = _layout(resources, modules)
    if not np.isfinite((incentive_budget,incentive_weight,time_limit_seconds)).all() or (
            incentive_budget < 0 or incentive_weight < 0 or time_limit_seconds <= 0):
        raise ValueError("invalid budget, cost weight or time limit")
    selected = {m.module_id for m in modules} if selected_module_ids is None else set(selected_module_ids)
    if selected-{m.module_id for m in modules}:
        raise ValueError("unknown selected module")
    n, rows = len(modules), len(resources)*SLOTS
    costs = np.zeros(n) if amplitude_cost_per_kw is None else np.asarray(amplitude_cost_per_kw,float)
    if costs.shape != (n,) or not np.isfinite(costs).all():
        raise ValueError("amplitude costs must match modules")
    upper = np.asarray([m.amplitude_upper_kw if m.module_id in selected else 0. for m in modules])
    # Variables: amplitudes, net reduction epigraphs, incremental grid-energy states.
    reduction = n+np.arange(rows); energy = n+rows+np.arange(rows); count = n+2*rows
    incentive = np.concatenate([r.incentive_per_kwh for r in resources])*DT
    tariffs = np.concatenate([r.electricity_price_per_kwh for r in resources])*DT
    objective = np.zeros(count); objective[:n] = costs+B.T@tariffs
    objective[reduction] = incentive_weight*incentive
    bounds = list(zip(np.zeros(n),upper))
    max_reduction = np.maximum(0., -(np.minimum(B,0.)@upper))
    bounds += list(zip(np.zeros(rows),max_reduction))
    for r in resources:
        lo,hi = r.energy_bounds();bounds += list(zip(lo,hi))
    eq_rows=[];eq_rhs=[];ub_rows=[];ub_rhs=[]
    def add(entries,rhs,equality=False):
        (eq_rows if equality else ub_rows).append(entries)
        (eq_rhs if equality else ub_rhs).append(float(rhs))
    for i,r in enumerate(resources):
        for t in range(SLOTS):
            row = i*SLOTS+t; profile = B[row]
            terms={j:float(v) for j,v in enumerate(profile) if v}
            add(terms,r.power_upper_kw[t]-r.baseline_kw[t])
            add({j:-v for j,v in terms.items()},r.baseline_kw[t]-r.power_lower_kw[t])
            epigraph={j:-v for j,v in terms.items()};epigraph[int(reduction[row])]=-1.
            add(epigraph,0.)
            state={j:-DT*v for j,v in terms.items()};state[int(energy[row])]=1.
            if t:state[int(energy[row-1])]=-1.
            add(state,0.,True)
            if not r.available[t]:add(terms,0.,True)
        add({int(energy[(i+1)*SLOTS-1]):1.},0.,True)
    add({int(reduction[k]):float(v) for k,v in enumerate(incentive) if v},incentive_budget)
    for task in tasks:
        if not isinstance(task,LinearTask) or set(task.coefficients)-set(lookup):
            raise ValueError("task must refer to declared resources")
        row=np.zeros(n)
        for key,coefficient in task.coefficients.items():
            i=lookup[key];row += coefficient@B[i*SLOTS:(i+1)*SLOTS]
        entries={j:float(v) for j,v in enumerate(row) if v}
        if np.isfinite(task.upper):add(entries,task.upper)
        if np.isfinite(task.lower):add({j:-v for j,v in entries.items()},-task.lower)
    def sparse(entries):
        matrix=lil_matrix((len(entries),count))
        for i,row in enumerate(entries):
            for j,value in row.items():matrix[i,j]=value
        return matrix.tocsr()
    Aeq,Aub=sparse(eq_rows),sparse(ub_rows)
    raw=linprog(objective,A_ub=Aub,b_ub=ub_rhs,A_eq=Aeq,b_eq=eq_rhs,bounds=bounds,method="highs",
                options=dict(time_limit=float(time_limit_seconds),primal_feasibility_tolerance=1e-9,
                             dual_feasibility_tolerance=1e-9))
    result=dict(status="no_feasible_incumbent",solver_status=int(raw.status),solver_message=str(raw.message),
                solver_reported_optimal=bool(raw.status==0),objective_bound=None,
                objective_scope="declared linear operating cost/tasks, not WZZT final risk",
                lp_optimizer_calls=1,QP_calls=0,AC_calls=0,WZZT_calls=0,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                incentive_budget=float(incentive_budget),module_amplitudes_kw=None,execution=None)
    if raw.x is not None and np.isfinite(raw.x).all():
        a=np.asarray(raw.x[:n]);audit=evaluate_schedule(resources,modules,a,tolerance=tolerance)
        equality=float(np.max(np.abs(Aeq@raw.x-np.asarray(eq_rhs)),initial=0.))
        inequality=float(np.max(Aub@raw.x-np.asarray(ub_rhs),initial=0.))
        direct=float(costs@a+audit["total_incremental_electricity_bill"]+
                     incentive_weight*audit["total_incentive_payment"])
        passed=bool(audit["resource_feasible"] and equality<=tolerance and inequality<=tolerance and
                    audit["total_incentive_payment"]<=incentive_budget+tolerance and
                    abs(direct-float(raw.fun))<=tolerance*max(1.,abs(direct)))
        result.update(status=("optimal" if raw.status==0 else "feasible_unproven") if passed else "numerically_invalid",
                      module_amplitudes_kw={m.module_id:float(v) for m,v in zip(modules,a)},execution=audit,
                      solver_objective=float(raw.fun),direct_operating_objective=direct,
                      equality_violation=equality,inequality_violation=max(0.,inequality),
                      independent_primal_checks_pass=passed,
                      objective_bound=float(raw.fun) if raw.status==0 and passed else None)
    return result


def decompose_terminal_baseline(total_kw, node_labels, resources):
    """Resource bases are subtracted once; the nonflexible DER residual may be negative."""
    resources,_,_,_=_layout(resources,())
    labels=list(node_labels);total=np.asarray(total_kw,float)
    if total.shape!=(SLOTS,len(labels)) or not np.isfinite(total).all() or len(set(labels))!=len(labels):
        raise ValueError("aligned finite 96-point terminal totals required")
    device=np.zeros_like(total)
    for r in resources:
        if r.node_label not in labels:raise ValueError("resource node absent from terminal labels")
        device[:,labels.index(r.node_label)] += r.baseline_kw
    return dict(resource_baseline_kw=device,nonflexible_kw=total-device,
                negative_nonflexible_allowed=True,resource_capability_inferred_from_total=False)


def load_archived_ev_domain(source_directory, expected_sha256, **config_values):
    """Explicit optional adapter; the external archive is not a packaged dependency.

    Only the two reviewed operation-domain source files are imported. No solve
    occurs here. Default/accepted scope is a fixed-site, non-V2G 96-slot EV.
    """
    directory=Path(source_directory).resolve()
    names=("ev_operations_milp.py","ev_mobile_service_domain.py")
    if set(expected_sha256)!=set(names):
        raise ValueError("explicit hashes for both archived source files are required")
    for name in names:
        file=directory/name
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=expected_sha256[name]:
            raise ValueError("archived EV source hash mismatch: "+name)
    config=dict(n_slots=SLOTS,n_sites=1,dt_hours=DT,mobile=False,allow_discharge=False)
    config.update(config_values)
    if config["n_slots"]!=SLOTS or config["n_sites"]!=1 or config["dt_hours"]!=DT or (
            config["mobile"] or config["allow_discharge"]):
        raise ValueError("this adapter is fixed-site, charge-only, 96-slot")
    token=hashlib.sha256((str(directory)+json.dumps(expected_sha256,sort_keys=True)).encode()).hexdigest()[:16]
    name="_wzzt_explicit_ev_archive_"+token
    if name not in sys.modules:
        package=types.ModuleType(name);package.__path__=[str(directory)];sys.modules[name]=package
    before=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        module=importlib.import_module(name+".ev_mobile_service_domain")
        domain=module.build_mobile_service_domain(module.MobileServiceConfig(**config))
    finally:
        sys.dont_write_bytecode=before
    return domain,dict(source_directory=str(directory),source_sha256=dict(expected_sha256),
                       packaged_dependency=False,solver_calls=0,mobile=False,allow_discharge=False)
