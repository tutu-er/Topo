"""Theft (unmetered load) scenario generation with a feeder-head master meter.

The theft load is added to the PHYSICAL injections before the AC power flow, but
never enters the reporting terminal meter. At most one location may be active at
any time step (validated). Truth channels are stored for evaluation only and must
not feed any fitting or threshold selection.

Units: theft amplitudes are given in kW and converted to per unit on
``net.base_mva``; all scenario power channels remain per unit, matching
``theft_wzzt.scenario.simulation``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from theft_wzzt.data.paper_style_case_bank import CASE_BUILDERS
from theft_wzzt.estimation.preprocessing import squared_voltage_drop_from_observed_root
from theft_wzzt.models.ac_powerflow import solve_ac_power_flow_timeseries
from theft_wzzt.scenario.profiles import _build_terminal_profiles
from theft_wzzt.scenario.settings import _finite_number, resolve_scenario_settings
from theft_wzzt.scenario.simulation import (
    SCENARIO_DEFINITIONS,
    _diagnostics,
    _integer,
    _resource_assignment,
    _root_voltage,
    _terminal_buses,
)


def add_unobserved_spur(net, parent_bus: int, spur_bus: int,
                        r_ohm: float = 0.05, x_ohm: float = 0.04):
    """Attach an unmetered spur branch; its response column equals the parent's."""

    buses = net.buses
    row = {col: 0 for col in buses.columns}
    row.update({"bus_id": spur_bus, "original_bus_id": spur_bus,
                "bus_type": "hidden_internal", "pd_kw": 0.0, "qd_kvar": 0.0,
                "is_observed": False, "has_load": False, "is_terminal": False,
                "is_original_case33_bus": False, "hidden_degree": 1,
                "note": "unobserved spur for ambiguity experiments"})
    net.buses = pd.concat([buses, pd.DataFrame([row])], ignore_index=True)
    branch = {col: None for col in net.branches.columns}
    branch.update({"from_bus": parent_bus, "to_bus": spur_bus, "r_ohm": r_ohm,
                   "x_ohm": x_ohm, "branch_type": "spur", "is_true_closed": True,
                   "is_candidate": True})
    net.branches = pd.concat([net.branches, pd.DataFrame([branch])], ignore_index=True)
    return net


@dataclass(frozen=True)
class TheftSpec:
    """One unmetered-load source: a fixed bus with a nonnegative kW profile."""

    bus_id: int
    amplitude_kw: tuple[float, ...]
    kappa: float = 0.0

    def validate(self, net, t_count: int) -> np.ndarray:
        buses = net.buses.set_index("bus_id")
        if int(self.bus_id) not in buses.index or int(self.bus_id) == int(net.root_bus):
            raise ValueError(f"theft bus {self.bus_id} is not a non-root bus")
        amplitude = np.asarray(self.amplitude_kw, dtype=float)
        if amplitude.shape != (t_count,) or not np.all(np.isfinite(amplitude)):
            raise ValueError(f"theft amplitude must be a finite length-{t_count} profile")
        if np.any(amplitude < 0):
            raise ValueError("theft only adds load: amplitude must be nonnegative")
        if not np.isfinite(self.kappa) or self.kappa < 0:
            raise ValueError("kappa must be finite and nonnegative")
        return amplitude


def _head_power(ac: dict, net) -> tuple[pd.Series, pd.Series]:
    """Feeder-head P/Q from root-outgoing branch 'from' flows, plus total loss."""

    edges = ac["metadata"]["branch_edges"]
    root = int(net.root_bus)
    head = [e["label"] for e in edges if e["from_bus"] == root]
    p0 = ac["branch_p_from_pu"][head].sum(axis=1)
    q0 = ac["branch_q_from_pu"][head].sum(axis=1)
    return p0.rename("P0_true"), q0.rename("Q0_true")


def simulate_theft_scenarios(
    case_key: str,
    thefts: tuple[TheftSpec, ...] | None = None,
    *,
    scenario_index: int = 0,
    t_count: int = 96,
    replicate: int = 0,
    scenario_suite: str = "reference",
    root_observation: str | None = None,
    impedance_scale: float | None = None,
    case_modifier=None,
    pq_noise_rel: float = 0.005,
    v_noise_rel: float = 0.0002,
    master_noise_rel: float = 0.002,
) -> tuple[object, dict]:
    """Simulate one scenario with optional theft and a feeder-head master meter.

    Returns ``(net, scenario)``. In addition to the keys produced by
    ``theft_wzzt.scenario.simulation._simulate_pool``, the scenario carries:

    - ``P0_measured`` / ``Q0_measured``: noisy feeder-head master meter (pu).
    - ``P0_true`` / ``Q0_true``: true feeder-head power (evaluation only).
    - ``loss_p_true`` / ``loss_q_true``: true technical loss of the ACTUAL
      (theft-carrying) network, per sample (evaluation only).
    - ``loss_p_notheft`` / ``loss_q_notheft``: counterfactual loss without
      theft, i.e. the external loss estimate's ideal target (evaluation only).
    - ``theft_truth``: list of {"bus_id", "amplitude_pu", "kappa"} entries
      (evaluation only; never a fitting input).

    ``case_modifier`` (if given) transforms the freshly built network before any
    simulation, e.g. to attach an unobserved spur branch. ``impedance_scale``
    overrides the suite default physics scaling; model-side bounds must then be
    derived from a SEPARATE nominal network to emulate parameter error.
    """

    settings = resolve_scenario_settings(
        scenario_suite, root_observation=root_observation,
        impedance_scale=impedance_scale,
    )
    if not isinstance(case_key, str) or case_key not in CASE_BUILDERS:
        raise ValueError(f"unknown case_key: {case_key!r}")
    t_count = _integer(t_count, "t_count", 1)
    if scenario_suite != "legacy" and (t_count < 4 or 96 % t_count):
        raise ValueError("non-legacy t_count must be >= 4 and divide 96")
    physical_count = t_count if scenario_suite == "legacy" else 96
    sample_indices = np.arange(0, physical_count, physical_count // t_count, dtype=int)
    master_noise_rel = _finite_number(master_noise_rel, "master_noise_rel")

    net = CASE_BUILDERS[case_key]()
    if case_modifier is not None:
        net = case_modifier(net)
    if settings["impedance_scale"] is not None:
        original_scale = _finite_number(
            net.metadata["impedance_scale"], "original impedance_scale", positive=True
        )
        target_scale = settings["impedance_scale"]
        if target_scale != original_scale:
            net.branches.loc[:, ["r_ohm", "x_ohm"]] = (
                net.branches.loc[:, ["r_ohm", "x_ohm"]] / original_scale * target_scale
            )
        net.metadata["impedance_scale"] = target_scale

    theft_list = list(thefts or ())
    amplitudes = []
    for spec in theft_list:
        amplitudes.append(spec.validate(net, physical_count))
    if amplitudes:
        active_count = sum((a > 0).astype(int) for a in amplitudes)
        if np.any(active_count > 1):
            raise ValueError("at most one theft location may be active per time step")

    definition = SCENARIO_DEFINITIONS[scenario_index]
    shifted_seed = int(definition["seed"]) + 100_003 * replicate
    profile = settings["profiles"][scenario_index]
    assignment = _resource_assignment(case_key, net)
    p_true, q_true, _ = _build_terminal_profiles(
        net, assignment, physical_count, shifted_seed, profile_scenario=profile,
    )
    root_true = _root_voltage(p_true.index, shifted_seed + 1000, settings)

    # Actual physical injections = reported truth + unmetered theft.
    p_actual = p_true.copy()
    q_actual = q_true.copy()
    pu_per_kw = 1.0 / (net.base_mva * 1000.0)
    truth_records = []
    for spec, amplitude in zip(theft_list, amplitudes):
        column = int(spec.bus_id)
        p_actual[column] = p_actual.get(column, 0.0) + amplitude * pu_per_kw
        q_actual[column] = q_actual.get(column, 0.0) + spec.kappa * amplitude * pu_per_kw
        truth_records.append({
            "bus_id": column,
            "amplitude_pu": pd.Series(amplitude * pu_per_kw, index=p_true.index),
            "kappa": float(spec.kappa),
        })

    ac = solve_ac_power_flow_timeseries(
        net, p_actual, q_actual, v_root=root_true, max_iter=100, tol=1e-10
    )
    if not bool(ac["converged"].all()):
        raise RuntimeError(f"AC power flow did not converge for {case_key} (theft scenario)")
    p0_true, q0_true = _head_power(ac, net)
    loss_p_true = ac["branch_loss_p_pu"].sum(axis=1)
    loss_q_true = ac["branch_loss_q_pu"].sum(axis=1)

    ac_clean = solve_ac_power_flow_timeseries(
        net, p_true, q_true, v_root=root_true, max_iter=100, tol=1e-10
    )
    if not bool(ac_clean["converged"].all()):
        raise RuntimeError(f"AC power flow did not converge for {case_key} (clean reference)")
    loss_p_notheft = ac_clean["branch_loss_p_pu"].sum(axis=1)
    loss_q_notheft = ac_clean["branch_loss_q_pu"].sum(axis=1)

    terminals = _terminal_buses(net)
    voltage_true = ac["V_bus_mag"].loc[:, terminals]

    # Meters: terminal meters report the METERED load only (theft excluded);
    # voltages and the master meter see the actual physics.
    rng = np.random.default_rng(shifted_seed + 2000)
    p_measured = p_true + rng.normal(
        0.0, pq_noise_rel * np.maximum(np.abs(p_true), 1e-12), size=p_true.shape
    )
    q_measured = q_true + rng.normal(
        0.0, pq_noise_rel * np.maximum(np.abs(q_true), 1e-12), size=q_true.shape
    )
    voltage_measured = voltage_true + rng.normal(
        0.0, v_noise_rel * np.maximum(np.abs(voltage_true), 1e-12), size=voltage_true.shape
    )
    master_rng = np.random.default_rng(shifted_seed + 3000)
    p0_measured = p0_true + master_rng.normal(
        0.0, master_noise_rel * np.maximum(np.abs(p0_true), 1e-12), size=p0_true.shape
    )
    q0_measured = q0_true + master_rng.normal(
        0.0, master_noise_rel * np.maximum(np.abs(q0_true), 1e-12), size=q0_true.shape
    )
    p0_measured.name, q0_measured.name = "P0_measured", "Q0_measured"

    observation = settings["root_observation"]
    root_meter_seed = 70_000_019 + 100_003 * replicate + scenario_index
    if observation == "exact":
        root_observed = root_true.copy()
    elif observation == "noisy":
        root_rng = np.random.default_rng(root_meter_seed)
        root_observed = root_true + root_rng.normal(
            0.0,
            settings["root_meter_noise_rel"] * np.maximum(np.abs(root_true), 1e-12),
            size=physical_count,
        )
        root_observed.name = "V_root_measured"
    else:
        raise ValueError("theft identification requires an observed root voltage")
    drop_target = squared_voltage_drop_from_observed_root(voltage_measured, root_observed)

    actual_settings = deepcopy(settings)
    actual_settings.update({
        "profile_scenario": profile, "replicate": replicate,
        "scenario_index": scenario_index, "profile_seed": shifted_seed,
        "root_seed": shifted_seed + 1000, "meter_seed": shifted_seed + 2000,
        "master_meter_seed": shifted_seed + 3000,
        "root_meter_seed": root_meter_seed,
        "pq_noise_rel": pq_noise_rel, "v_noise_rel": v_noise_rel,
        "master_noise_rel": master_noise_rel,
        "observed_sample_count": t_count,
        "physical_sample_count": physical_count,
    })
    diagnostics = _diagnostics(
        p_true, q_true, voltage_true, root_true, p_measured, q_measured,
        voltage_measured, root_observed, drop_target, actual_settings,
    )
    diagnostics["theft_bus_ids"] = [r["bus_id"] for r in truth_records]
    diagnostics["theft_total_kwh"] = float(
        sum(r["amplitude_pu"].sum() for r in truth_records)
        * net.base_mva * 1000.0 * (24.0 / physical_count)
    )
    name = f"theft_{case_key}_{profile}_seed{definition['seed']}_rep{replicate}"
    scenario = {
        "name": name,
        "P_terminal": p_measured.iloc[sample_indices].copy(),
        "Q_terminal": q_measured.iloc[sample_indices].copy(),
        "V_terminal": voltage_measured.iloc[sample_indices].copy(),
        "root_voltage": root_observed.iloc[sample_indices].copy(),
        "drop_target": drop_target.iloc[sample_indices].copy(),
        "P_true": p_true.iloc[sample_indices].copy(),
        "Q_true": q_true.iloc[sample_indices].copy(),
        "V_terminal_true": voltage_true.iloc[sample_indices].copy(),
        "root_voltage_true": root_true.iloc[sample_indices].copy(),
        "root_observation": observation,
        "source_sample_indices": sample_indices.tolist(),
        "scenario_settings": actual_settings,
        "diagnostics": diagnostics,
        "P0_measured": p0_measured.iloc[sample_indices].copy(),
        "Q0_measured": q0_measured.iloc[sample_indices].copy(),
        "P0_true": p0_true.iloc[sample_indices].copy(),
        "Q0_true": q0_true.iloc[sample_indices].copy(),
        "loss_p_true": loss_p_true.iloc[sample_indices].copy(),
        "loss_q_true": loss_q_true.iloc[sample_indices].copy(),
        "loss_p_notheft": loss_p_notheft.iloc[sample_indices].copy(),
        "loss_q_notheft": loss_q_notheft.iloc[sample_indices].copy(),
        "theft_truth": [
            {**r, "amplitude_pu": r["amplitude_pu"].iloc[sample_indices].copy()}
            for r in truth_records
        ],
    }
    return net, scenario
