"""Feasible EV schedules and conditional history-anchored model discrimination.

This research module evaluates a finite library. Inner AC fits are local:
their joint objective values are upper bounds on each inner infimum, but a
difference of two local minima is not a bound on the exact T criterion.
No topology truth or future random observations enter the selector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .likelihood import (
        Observations, MeasurementModel, FitResult, ac_predict,
        concat_observations, fit_candidate,
    )
except ImportError:
    from likelihood import (
        Observations, MeasurementModel, FitResult, ac_predict,
        concat_observations, fit_candidate,
    )


@dataclass(frozen=True)
class EVConstraints:
    """Power in pu, energy/capacity in pu-hours, SOC as a fraction.

    base_kw converts metadata to kW and kWh. availability[t,j] means charger
    access in slot t. travel_slots[i,j] is the required number of complete
    unconnected slots between sites. travel_energy[i,j] is battery energy
    consumed by the move (pu-hours); zero explicitly neglects driving energy.
    q_over_p is one declared fixed charger ratio; Q is never independently
    optimized. Each slot is treated as a steady constant-power AC condition.
    """
    dt_hours: float = .25
    p_max: float = .30
    energy: float = .30
    battery_capacity: float = 3.0
    initial_soc: float = .40
    soc_min: float = .20
    soc_max: float = .90
    efficiency: float = .95
    base_kw: float = 20.0
    q_over_p: float = 0.0
    availability: Any = field(default=None, repr=False)
    travel_slots: Any = field(default=None, repr=False)
    travel_energy: Any = field(default=None, repr=False)
    initial_site: int | None = None

    def arrays(self, n_slots: int, n_sites: int):
        positive = (self.dt_hours, self.p_max, self.battery_capacity,
                    self.efficiency, self.base_kw)
        if not all(np.isfinite(positive)) or min(positive) <= 0:
            raise ValueError("Positive finite time, power, capacity, efficiency and base are required")
        if not (np.isfinite(self.energy) and self.energy >= 0
                and 0 < self.efficiency <= 1
                and 0 <= self.soc_min <= self.initial_soc <= self.soc_max <= 1
                and np.isfinite(self.q_over_p)):
            raise ValueError("Invalid energy, SOC, efficiency or fixed charger Q/P")
        if n_slots < 1 or n_sites < 1:
            raise ValueError("A positive number of slots and sites is required")
        access = (np.ones((n_slots, n_sites), dtype=bool) if self.availability is None
                  else np.asarray(self.availability, dtype=bool))
        raw_travel = (np.ones((n_sites, n_sites), dtype=int)
                      if self.travel_slots is None else np.asarray(self.travel_slots))
        if self.travel_slots is None:
            np.fill_diagonal(raw_travel, 0)
        if (raw_travel.shape != (n_sites, n_sites)
                or not np.isfinite(raw_travel).all()
                or (raw_travel < 0).any()
                or not np.equal(raw_travel, np.floor(raw_travel)).all()):
            raise ValueError("Travel slots must be a finite nonnegative integer site matrix")
        travel = raw_travel.astype(int)
        mobility = (np.zeros((n_sites, n_sites), float) if self.travel_energy is None
                    else np.asarray(self.travel_energy, float))
        if (access.shape != (n_slots, n_sites)
                or mobility.shape != (n_sites, n_sites)
                or not np.isfinite(mobility).all() or (mobility < 0).any()
                or np.any(np.diag(travel) != 0) or np.any(np.diag(mobility) != 0)):
            raise ValueError("Invalid access/travel matrix shape or diagonal")
        if self.initial_site is not None and not 0 <= self.initial_site < n_sites:
            raise ValueError("Initial site is outside the terminal set")
        return access, travel, mobility


@dataclass(frozen=True)
class EVAction:
    name: str
    p: np.ndarray
    q: np.ndarray
    sites: np.ndarray
    family: str = "fixed_pulse"

    def metadata(self, constraints: EVConstraints | None = None):
        result = dict(name=self.name, family=self.family,
                      p=np.asarray(self.p).tolist(), q=np.asarray(self.q).tolist(),
                      sites=np.asarray(self.sites).tolist())
        if constraints is not None:
            result.update(
                energy_pu_hours=float(constraints.dt_hours * np.sum(self.p)),
                energy_kwh=float(constraints.base_kw * constraints.dt_hours * np.sum(self.p)),
                max_kw=float(constraints.base_kw * np.max(self.p, initial=0.0)),
                slot_minutes=60 * constraints.dt_hours,
                fixed_q_over_p=constraints.q_over_p,
            )
        return result


def check_action(action: EVAction, constraints: EVConstraints) -> dict:
    """Check access, power/energy, SOC and elapsed travel time explicitly."""
    p, q, raw_sites = (np.asarray(action.p, float), np.asarray(action.q, float),
                       np.asarray(action.sites))
    issues = []
    if p.ndim != 2 or p.shape != q.shape or raw_sites.shape != (len(p),):
        return dict(feasible=False, issues=["action_shape"], soc=[])
    n_slots, n_sites = p.shape
    access, travel, mobility = constraints.arrays(n_slots, n_sites)
    if (not np.isfinite(p).all() or not np.isfinite(q).all()
            or not np.isfinite(raw_sites).all()
            or not np.equal(raw_sites, np.floor(raw_sites)).all()):
        return dict(feasible=False, issues=["nonfinite_or_noninteger_action"], soc=[])
    sites = raw_sites.astype(int)
    tolerance = 1e-10
    if (sites < -1).any() or (sites >= n_sites).any():
        return dict(feasible=False, issues=["site_out_of_range"], soc=[])
    if (p < -tolerance).any() or (p > constraints.p_max + tolerance).any():
        issues.append("charging_power")
    if np.max(np.abs(q - constraints.q_over_p * p), initial=0.0) > tolerance:
        issues.append("unsupported_reactive_power")
    if np.any((p > tolerance) & ~access):
        issues.append("charger_access")
    for t, site in enumerate(sites):
        if site == -1 and np.max(abs(p[t]), initial=0.0) > tolerance:
            issues.append("charging_during_travel")
        elif site >= 0 and np.any(np.delete(np.abs(p[t]), site) > tolerance):
            issues.append("multiple_or_wrong_site")
    delivered = constraints.dt_hours * float(np.sum(p))
    if not np.isclose(delivered, constraints.energy, atol=tolerance, rtol=1e-9):
        issues.append("energy_equality")
    movement_by_slot = np.zeros(n_slots)
    previous_site = constraints.initial_site
    previous_t = -1
    for t, site in enumerate(sites):
        if site < 0:
            continue
        if previous_site is not None and site != previous_site:
            elapsed = t - previous_t - 1
            if elapsed < travel[previous_site, site]:
                issues.append("insufficient_travel_time")
            movement_by_slot[t] += mobility[previous_site, site]
        previous_site, previous_t = site, t
    if sites[-1] == -1:
        issues.append("unfinished_travel")
    increments = (constraints.efficiency * constraints.dt_hours * np.sum(p, axis=1)
                  - movement_by_slot) / constraints.battery_capacity
    soc = np.r_[constraints.initial_soc, constraints.initial_soc + np.cumsum(increments)]
    if np.any(soc < constraints.soc_min - tolerance) or np.any(soc > constraints.soc_max + tolerance):
        issues.append("soc")
    return dict(feasible=not issues, issues=sorted(set(issues)), soc=soc.tolist(),
                energy_pu_hours=delivered, movement_energy_pu_hours=float(sum(movement_by_slot)))


def _allocate(indices, n_slots, power_units, p_max):
    """Charge in the supplied slot order with one fractional final slot."""
    result = np.zeros(n_slots)
    remaining = float(power_units)
    for t in indices:
        amount = min(p_max, max(0.0, remaining))
        result[t] = amount
        remaining -= amount
    return result if abs(remaining) <= 1e-10 else None


def enumerate_actions(n_slots: int, n_sites: int, constraints: EVConstraints,
                      *, max_actions: int = 12) -> tuple[list[EVAction], list[dict]]:
    """A deterministic finite library with constant, early/late and mobile arms.

    Stationary sites with restricted access have constant amplitude in their
    available slots. Mobile arms split the grid-delivered energy equally before
    and after one move, with the declared travel slots unconnected. This library
    is a design restriction, not the set of all physically feasible schedules.
    Rejected proposals are returned for transparent selection-failure reporting.
    """
    access, travel, _ = constraints.arrays(n_slots, n_sites)
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
    units = constraints.energy / constraints.dt_hours
    proposals = []
    for site in range(n_sites):
        positions = np.full(n_slots, site, int)
        if constraints.initial_site is not None and site != constraints.initial_site:
            positions[:travel[constraints.initial_site, site]] = -1
        slots = np.flatnonzero(access[:, site] & (positions == site))
        if len(slots):
            profile = np.zeros(n_slots)
            profile[slots] = units / len(slots)
            proposals.append((f"constant_site_{site}", "constant", positions, profile, site))
        for label, order in (("early", slots), ("late", slots[::-1])):
            profile = _allocate(order, n_slots, units, constraints.p_max)
            if profile is not None:
                proposals.append((f"{label}_site_{site}", "fixed_pulse", positions, profile, site))
    actions, rejected, keys = [], [], set()
    for name, family, sites, profile, site in proposals:
        p = np.zeros((n_slots, n_sites))
        p[:, site] = profile
        action = EVAction(name, p, constraints.q_over_p * p, sites.copy(), family)
        check = check_action(action, constraints)
        key = (p.round(12).tobytes(), sites.tobytes())
        if not check["feasible"]:
            rejected.append(dict(name=name, **check))
        elif key not in keys:
            actions.append(action)
            keys.add(key)
    # Round-robin directed routes supply mobile options before the size cap.
    for first in range(n_sites):
        for second in range(n_sites):
            if first == second:
                continue
            departure = max(1, n_slots // 2 - travel[first, second] // 2)
            arrival = departure + travel[first, second]
            if arrival >= n_slots:
                rejected.append(dict(name=f"mobile_{first}_{second}",
                                     feasible=False, issues=["travel_consumes_horizon"]))
                continue
            sites = np.r_[np.full(departure, first, int),
                          np.full(travel[first, second], -1, int),
                          np.full(n_slots - arrival, second, int)]
            if constraints.initial_site is not None and first != constraints.initial_site:
                sites[:travel[constraints.initial_site, first]] = -1
            before = np.flatnonzero(access[:, first] & (sites == first))
            after = np.flatnonzero(access[:, second] & (sites == second))
            left = _allocate(before, n_slots, units / 2, constraints.p_max)
            right = _allocate(after, n_slots, units / 2, constraints.p_max)
            name = f"mobile_{first}_{second}"
            if left is None or right is None:
                rejected.append(dict(name=name, feasible=False,
                                     issues=["split_energy_or_access"]))
                continue
            p = np.zeros((n_slots, n_sites))
            p[:, first], p[:, second] = left, right
            action = EVAction(name, p, constraints.q_over_p * p, sites, "mobile")
            check = check_action(action, constraints)
            key = (p.round(12).tobytes(), sites.tobytes())
            if not check["feasible"]:
                rejected.append(dict(name=name, **check))
            elif key not in keys:
                actions.append(action)
                keys.add(key)
    # Keep all site constants, then representative pulse and mobile schedules.
    constants = [a for a in actions if a.family == "constant"]
    pulses = [a for a in actions if a.family == "fixed_pulse"]
    mobiles = [a for a in actions if a.family == "mobile"]
    if mobiles and len(constants) + len(pulses) >= max_actions:
        reserve = min(2, len(mobiles), max(0, max_actions - len(constants)))
        kept = constants[:max_actions] + pulses[:max(0, max_actions-len(constants)-reserve)] + mobiles[:reserve]
    else:
        kept = (constants + pulses + mobiles)[:max_actions]
    kept_names = {a.name for a in kept}
    rejected.extend(dict(name=a.name, feasible=True, issues=["finite_library_size_cap"])
                    for a in actions if a.name not in kept_names)
    return kept, rejected


@dataclass(frozen=True)
class FutureForecast:
    """Conditional latent future baseline; no future noise observations."""
    p: np.ndarray
    q: np.ndarray
    source_p: np.ndarray
    source_q: np.ndarray
    root_v: np.ndarray

    @classmethod
    def from_history(cls, history: Observations, reference_fit: FitResult, n_slots: int):
        if n_slots < 1:
            raise ValueError("Future horizon must be positive")
        p = np.tile(np.mean(reference_fit.p, axis=0), (n_slots, 1))
        q = np.tile(np.mean(reference_fit.q, axis=0), (n_slots, 1))
        source_p = np.full(n_slots, np.mean(reference_fit.source_p))
        source_q = np.full(n_slots, np.mean(reference_fit.source_q))
        root_v = np.full(n_slots, np.mean(np.broadcast_to(history.root_v, (len(history.p),))))
        return cls(p, q, source_p, source_q, root_v)


def _known_bias(model: MeasurementModel, fit: FitResult, channel: str, shape):
    # If a calibrated bias is fitted, its single shared value must be used for
    # both historical and expected future meter means.
    value = getattr(fit, "bias_" + channel, None)
    if value is None:
        value = getattr(model, "bias_" + channel, None)
    return np.zeros(shape) if value is None else np.broadcast_to(np.asarray(value, float), shape)


def expected_observations(candidate, reference_fit: FitResult, history: Observations,
                          model: MeasurementModel, forecast: FutureForecast,
                          action: EVAction, *, voltage_limits: tuple[float, float] | None = None) -> Observations:
    """Predict noiseless meter means; terminal P/Q contain EV exactly once."""
    p, q = np.asarray(forecast.p, float) + action.p, np.asarray(forecast.q, float) + action.q
    if p.shape != action.p.shape or q.shape != p.shape:
        raise ValueError("Forecast and EV action terminal dimensions differ")
    predicted = ac_predict(candidate, reference_fit.r, reference_fit.x, p, q,
                           source_p=forecast.source_p, source_q=forecast.source_q,
                           root_v=forecast.root_v)
    if voltage_limits is not None:
        low, high = voltage_limits
        if not (0 < low < high):
            raise ValueError("Invalid nominal voltage limits")
        raw_voltage = np.asarray(predicted.get("voltage", predicted["v"]))
        physical_voltage = np.abs(raw_voltage) if np.iscomplexobj(raw_voltage) else raw_voltage
        vmin, vmax = float(np.min(physical_voltage)), float(np.max(physical_voltage))
        if vmin < low or vmax > high:
            raise ValueError(f"nominal_voltage_screen:min={vmin:.8g},max={vmax:.8g}")
    return Observations(
        p=predicted["p"] + _known_bias(model, reference_fit, "p", p.shape),
        q=predicted["q"] + _known_bias(model, reference_fit, "q", p.shape),
        v=predicted["v"] + _known_bias(model, reference_fit, "v", p.shape),
        root_v=forecast.root_v,
        master_p=predicted["master_p"] if history.master_p is not None else None,
        master_q=predicted["master_q"] if history.master_q is not None else None,
        source_p=predicted["source_p"] if history.source_p is not None else None,
        source_q=predicted["source_q"] if history.source_q is not None else None,
    )


def select_action(candidates: Sequence, history: Observations,
                  historical_fits: Mapping[str, FitResult], model: MeasurementModel,
                  actions: Sequence[EVAction], *, forecast: FutureForecast | None = None,
                  constraints: EVConstraints | None = None,
                  near_deviance: float | None = None, include_historical_gap: bool = True,
                  multistart: int = 1, max_nfev: int = 120, seed: int = 0,
                  fit_options: Mapping[str, Any] | None = None,
                  voltage_limits: tuple[float, float] | None = (.9, 1.1)) -> dict:
    """Enumerate conditional reference-versus-rival joint refit criteria.

    J_hf is the local Gaussian weighted-square objective on the ORIGINAL
    historical observations concatenated with reference noiseless future
    predictions. An action has score min_rival[J_hf - J_h,rival + gap_h].
    gap_h=J_h,rival-J_h,reference when include_historical_gap is True.
    Rivals must use one set of R/X and fixed biases across both data blocks;
    latent total P/Q and common-source P/Q are re-estimated for every slot.

    Failed refits invalidate that action, rather than increasing its score.
    near_deviance is an explicit finite-candidate screen. Scores are
    conditional design quantities, neither rejection probabilities nor
    certified optima. No truth/noise argument is accepted by this function.
    """
    started = perf_counter()
    candidate_by_name = {candidate.name: candidate for candidate in candidates}
    if len(candidate_by_name) != len(candidates):
        raise ValueError("Candidate names must be unique")
    usable, historical_failures = {}, []
    for name in candidate_by_name:
        fitted = historical_fits.get(name)
        if fitted is None or not fitted.success or not np.isfinite(fitted.loss):
            historical_failures.append(name)
        else:
            usable[name] = fitted
    common = dict(
        criterion="conditional_local_history_future_joint_refit",
        score_interpretation="uncertified_local_objective_difference_not_topology_probability",
        inner_joint_cost_bound="upper_bound_on_inner_infimum_if_feasible",
        incremental_score_bound="uncertified_difference_of_local_objective_values",
        historical_fit_failures=historical_failures,
        finite_library_size=len(actions),
        include_historical_gap=include_historical_gap,
        local_inner_optimization=True,
        nominal_voltage_limits=voltage_limits,
        grid_safety_interpretation="reference_nominal_forecast_screen_only",
    )
    if not usable:
        return dict(common, status="historical_fit_failure", selected=None,
                    scores=[], elapsed_seconds=perf_counter()-started)
    reference_name = min(usable, key=lambda name: (usable[name].loss, name))
    reference_fit = usable[reference_name]
    gaps = {name: float(fit.loss - reference_fit.loss) for name, fit in usable.items()}
    rivals = [name for name in usable if name != reference_name
              and (near_deviance is None or gaps[name] <= near_deviance)]
    common.update(reference=reference_name, historical_deviance_gaps=gaps, rivals=rivals,
                  screened_candidates=[name for name in usable if name != reference_name and name not in rivals],
                  near_deviance=near_deviance)
    if not rivals:
        return dict(common, status="no_competing_candidate", selected=None,
                    scores=[], elapsed_seconds=perf_counter()-started)
    if not actions:
        return dict(common, status="no_feasible_action", selected=None,
                    scores=[], elapsed_seconds=perf_counter()-started)
    if forecast is None:
        forecast = FutureForecast.from_history(history, reference_fit, len(actions[0].p))
    options = dict(fit_options or {})
    for reserved in ("initial", "multistart", "max_nfev", "seed"):
        if reserved in options:
            raise ValueError(f"Use the explicit {reserved} selection argument")
    scores = []
    for index, action in enumerate(actions):
        action_started = perf_counter()
        row = dict(action=action.name, family=action.family, success=False,
                   score=None, competitor_fits=[], failures=[])
        if constraints is not None:
            row["feasibility"] = check_action(action, constraints)
            if not row["feasibility"]["feasible"]:
                row["failures"].append("infeasible_ev_action")
                row["elapsed_seconds"] = perf_counter() - action_started
                scores.append(row)
                continue
        try:
            future = expected_observations(candidate_by_name[reference_name],
                                           reference_fit, history, model, forecast, action,
                                           voltage_limits=voltage_limits)
            joined = concat_observations(history, future)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            row["failures"].append(f"reference_prediction:{type(error).__name__}:{error}")
            row["elapsed_seconds"] = perf_counter() - action_started
            scores.append(row)
            continue
        for rival_index, name in enumerate(rivals):
            inner_started = perf_counter()
            try:
                fitted = fit_candidate(candidate_by_name[name], joined, model,
                                       initial=usable[name], multistart=multistart,
                                       max_nfev=max_nfev, seed=seed + 1009*index + rival_index,
                                       **options)
                success = bool(fitted.success and np.isfinite(fitted.loss))
                increment = float(fitted.loss - usable[name].loss) if success else None
                value = (increment + gaps[name] if include_historical_gap else increment) if success else None
                row["competitor_fits"].append(dict(
                    candidate=name, success=success, historical_loss=float(usable[name].loss),
                    joint_loss=float(fitted.loss) if np.isfinite(fitted.loss) else None,
                    historical_deviance_gap=gaps[name], increment=increment, criterion=value,
                    negative_increment=bool(success and increment < -1e-7),
                    diagnostics=fitted.diagnostics,
                    elapsed_seconds=perf_counter()-inner_started,
                ))
                if success and increment < -1e-7:
                    row["failures"].append(f"historical_baseline_instability:{name}")
                if not success:
                    row["failures"].append(f"inner_fit:{name}")
            except (ValueError, RuntimeError, FloatingPointError) as error:
                row["failures"].append(f"inner_fit:{name}:{type(error).__name__}:{error}")
                row["competitor_fits"].append(dict(candidate=name, success=False,
                                                  elapsed_seconds=perf_counter()-inner_started))
        if not row["failures"]:
            row["success"] = True
            row["score"] = float(min(fit["criterion"] for fit in row["competitor_fits"]))
        row["elapsed_seconds"] = perf_counter() - action_started
        scores.append(row)
    valid = [row for row in scores if row["success"]]
    selected = max(valid, key=lambda row: (row["score"], row["action"]))["action"] if valid else None
    status = ("selected_with_historical_failures" if historical_failures else "selected") if selected else "all_action_fits_failed"
    return dict(common, status=status, selected=selected, scores=scores,
                failed_action_count=sum(not row["success"] for row in scores),
                elapsed_seconds=perf_counter()-started)

