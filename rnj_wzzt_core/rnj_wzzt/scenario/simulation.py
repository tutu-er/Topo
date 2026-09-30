"""Reproducible noisy AC scenarios with explicit physical and meter settings."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rnj_wzzt.data.paper_style_case_bank import (
    CASE_BUILDERS,
    build_case_bank_resource_assignment,
)
from rnj_wzzt.data.paper_style_terminal_lv import build_paper_style_resource_assignment
from rnj_wzzt.estimation.preprocessing import squared_voltage_drop_from_observed_root
from rnj_wzzt.models.ac_powerflow import solve_ac_power_flow_timeseries
from rnj_wzzt.models.network import TerminalizedNetwork
from rnj_wzzt.scenario.profiles import _build_terminal_profiles
from rnj_wzzt.scenario.settings import _finite_number, resolve_scenario_settings


# Retained for compatibility with the original nested three-scenario pool.
SCENARIO_DEFINITIONS = (
    {"name": "oneday_96pts_default_seed42", "seed": 42,
     "profile_scenario": "default", "root_voltage_sigma": 0.0008},
    {"name": "oneday_96pts_default_seed7", "seed": 7,
     "profile_scenario": "default", "root_voltage_sigma": 0.0008},
    {"name": "oneday_96pts_default_seed21", "seed": 21,
     "profile_scenario": "default", "root_voltage_sigma": 0.0008},
)


def _terminal_buses(net) -> list[int]:
    return net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"].astype(int).tolist()


def _resource_assignment(case_key: str, net) -> pd.DataFrame:
    if case_key == "paper15":
        return build_paper_style_resource_assignment(net)
    return build_case_bank_resource_assignment(net)


def _integer(value, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _root_voltage(index: pd.Index, seed: int, settings: dict) -> pd.Series:
    """Generate physical voltage without clipping or sample-variance rescaling."""

    count = len(index)
    rng = np.random.default_rng(seed)
    sigma = settings["root_sigma"]
    if settings["root_process"] == "iid":
        fluctuation = rng.normal(0.0, sigma, size=count)
    else:
        hours = np.arange(count) * settings["day_hours"] / count
        phase = rng.uniform(0.0, 2.0 * np.pi)
        fraction = settings["daily_fraction"]
        daily = np.sqrt(2.0) * fraction * sigma * np.sin(2.0 * np.pi * hours / 24.0 + phase)
        stationary_std = sigma * np.sqrt(1.0 - fraction**2)
        rho = np.exp(-(settings["day_hours"] / count) / settings["ou_tau_hours"])
        correlated = np.empty(count)
        correlated[0] = rng.normal(0.0, stationary_std)
        innovations = rng.normal(0.0, stationary_std * np.sqrt(1.0 - rho**2), size=count - 1)
        for position in range(1, count):
            correlated[position] = rho * correlated[position - 1] + innovations[position - 1]
        fluctuation = daily + correlated
        pulse = settings["tap_step_amplitude"] * (
            (hours >= settings["tap_start_hour"]) & (hours < settings["tap_end_hour"])
        )
        fluctuation = fluctuation + pulse - pulse.mean()
    return pd.Series(settings["root_mean"] + fluctuation, index=index, name="V_root_true")


def _dynamic_rms(frame: pd.DataFrame) -> float:
    values = frame.to_numpy(dtype=float)
    return float(np.sqrt(np.mean((values - values.mean(axis=0))**2)))


def _diagnostics(p_true, q_true, voltage_true, root_true, p_measured, q_measured,
                 voltage_measured, root_observed, drop_target, settings: dict) -> dict:
    """Keep truth-based checks in diagnostics, separate from estimator inputs."""

    physical_drop = squared_voltage_drop_from_observed_root(voltage_true, root_true)
    signal = _dynamic_rms(physical_drop)
    target_error = _dynamic_rms(drop_target - physical_drop)
    meter_error = target_error
    root_error = float(np.std(root_observed - root_true))
    snr = signal / target_error if target_error > 0.0 else None
    return {
        "sample_basis": "full_physical_grid",
        "physical_sample_count": len(p_true),
        "observed_sample_count": settings["observed_sample_count"],
        "root_voltage_mean_pu": float(root_true.mean()),
        "root_voltage_std_pu": float(np.std(root_true)),
        "root_voltage_min_pu": float(root_true.min()),
        "root_voltage_max_pu": float(root_true.max()),
        "root_observed_std_pu": float(np.std(root_observed)),
        "root_meter_noise_std_pu": root_error,
        "terminal_voltage_true_min_pu": float(voltage_true.to_numpy().min()),
        "terminal_voltage_true_max_pu": float(voltage_true.to_numpy().max()),
        "terminal_voltage_measured_min_pu": float(voltage_measured.to_numpy().min()),
        "terminal_voltage_measured_max_pu": float(voltage_measured.to_numpy().max()),
        "p_noise_std_pu": float(np.std((p_measured - p_true).to_numpy())),
        "q_noise_std_pu": float(np.std((q_measured - q_true).to_numpy())),
        "terminal_voltage_noise_std_pu": float(np.std((voltage_measured - voltage_true).to_numpy())),
        "dynamic_drop_signal_rms_pu2": signal,
        "dynamic_measurement_error_rms_pu2": meter_error,
        "dynamic_target_error_rms_pu2": target_error,
        "dynamic_drop_snr": snr,
        "dynamic_drop_snr_db": float(20.0 * np.log10(snr)) if snr is not None and snr > 0.0 else None,
        "ac_converged_all_physical_samples": True,
    }


@dataclass
class _ScenarioContext:
    case_key: str
    t_count: int
    net: TerminalizedNetwork
    settings: dict
    assignment: pd.DataFrame
    terminals: list[int]
    physical_count: int
    sample_indices: np.ndarray


@dataclass
class _BaseTrajectory:
    p_base: pd.DataFrame
    q_base: pd.DataFrame
    root_true: pd.Series
    name: str
    profile: str
    shifted_seed: int
    scenario_index: int
    replicate: int


def _prepare_case_context(
    case_key: str,
    t_count: int,
    *,
    scenario_suite: str = "legacy",
    root_observation: str | None = None,
    root_meter_noise_rel: float | None = None,
    root_sigma: float | None = None,
    impedance_scale: float | None = None,
) -> _ScenarioContext:
    """Prepare one network, settings, and the physical/observed time grids."""

    settings = resolve_scenario_settings(
        scenario_suite, root_observation=root_observation,
        root_meter_noise_rel=root_meter_noise_rel, root_sigma=root_sigma,
        impedance_scale=impedance_scale,
    )
    if not isinstance(case_key, str) or case_key not in CASE_BUILDERS:
        raise ValueError(f"unknown case_key: {case_key!r}")
    t_count = _integer(t_count, "t_count", 1)
    if scenario_suite != "legacy" and (t_count < 4 or 96 % t_count):
        raise ValueError("non-legacy t_count must be >= 4 and divide 96")
    physical_count = t_count if scenario_suite == "legacy" else 96
    sample_indices = np.arange(0, physical_count, physical_count // t_count, dtype=int)
    net = CASE_BUILDERS[case_key]()
    original_scale = _finite_number(net.metadata["impedance_scale"], "original impedance_scale", positive=True)
    target_scale = settings["impedance_scale"]
    if target_scale is not None and target_scale != original_scale:
        net.branches.loc[:, ["r_ohm", "x_ohm"]] = (
            net.branches.loc[:, ["r_ohm", "x_ohm"]] / original_scale * target_scale
        )
    net.metadata["original_impedance_scale"] = original_scale
    net.metadata["impedance_scale"] = original_scale if target_scale is None else target_scale
    if scenario_suite != "legacy" or net.metadata["impedance_scale"] != original_scale:
        # Builder notes describe the original benchmark, not this simulation.
        if "line_parameter_note" in net.metadata:
            source_note = net.metadata["line_parameter_note"]
            net.metadata["original_line_parameter_note"] = source_note
            library_note = source_note.partition("; impedances multiplied by scale factor ")[0]
            net.metadata["line_parameter_note"] = (
                f"{library_note}; impedances multiplied by scale factor {net.metadata['impedance_scale']:g}"
            )
        if "voltage_design_note" in net.metadata:
            net.metadata["original_voltage_design_note"] = net.metadata["voltage_design_note"]
            net.metadata["voltage_design_note"] = (
                "No terminal-voltage range is imposed; actual simulated ranges "
                "are reported in each scenario's diagnostics."
            )
    settings.update({
        "original_impedance_scale": original_scale,
        "impedance_scale": net.metadata["impedance_scale"],
        "physical_sample_count": physical_count,
        "observed_sample_count": t_count,
        "physical_interval_hours": 24.0 / physical_count,
        "observation_interval_hours": 24.0 / t_count,
    })
    return _ScenarioContext(
        case_key=case_key, t_count=t_count, net=net, settings=settings,
        assignment=_resource_assignment(case_key, net), terminals=_terminal_buses(net),
        physical_count=physical_count, sample_indices=sample_indices,
    )


def _build_base_trajectory(
    context: _ScenarioContext, scenario_index: int, replicate: int,
) -> _BaseTrajectory:
    """Generate only the requested scenario's normal terminal P/Q and root V."""

    scenario_index = _integer(scenario_index, "scenario_index", 0)
    replicate = _integer(replicate, "replicate", 0)
    if scenario_index >= len(SCENARIO_DEFINITIONS):
        raise ValueError("scenario_index is outside the available scenario definitions")
    definition = SCENARIO_DEFINITIONS[scenario_index]
    shifted_seed = int(definition["seed"]) + 100_003 * replicate
    profile = context.settings["profiles"][scenario_index]
    p_base, q_base, _ = _build_terminal_profiles(
        context.net, context.assignment, context.physical_count,
        shifted_seed, profile_scenario=profile,
    )
    root_true = _root_voltage(p_base.index, shifted_seed + 1000, context.settings)
    if context.settings["scenario_suite"] == "legacy":
        name = f"{definition['name']}_rep{replicate}"
    else:
        name = (f"oneday_96pts_{profile}_seed{definition['seed']}_"
                f"{context.settings['scenario_suite']}_rep{replicate}")
    return _BaseTrajectory(
        p_base=p_base, q_base=q_base, root_true=root_true, name=name,
        profile=profile, shifted_seed=shifted_seed,
        scenario_index=scenario_index, replicate=replicate,
    )


def _solve_physical_ac(
    context: _ScenarioContext,
    p_physical: pd.DataFrame,
    q_physical: pd.DataFrame,
    root_true: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    """Solve the actual physical injections on every physical time point."""

    if not p_physical.index.equals(root_true.index) or not q_physical.index.equals(root_true.index):
        raise ValueError("physical loads and root voltage must share a time index")
    unknown_buses = set(p_physical.columns).union(q_physical.columns) - set(context.net.buses["bus_id"])
    if unknown_buses:
        raise ValueError(f"physical loads contain unknown buses: {sorted(unknown_buses, key=str)}")
    ac = solve_ac_power_flow_timeseries(
        context.net, p_physical, q_physical, v_root=root_true, max_iter=100, tol=1e-10,
    )
    if not bool(ac["converged"].all()):
        raise RuntimeError(f"AC power flow did not converge for {context.case_key}")
    return ac, ac["V_bus_mag"].loc[:, context.terminals]


def _observe_terminal_channels(
    context: _ScenarioContext,
    base: _BaseTrajectory,
    p_meter_true: pd.DataFrame,
    q_meter_true: pd.DataFrame,
    voltage_true: pd.DataFrame,
    *,
    pq_noise_rel: float,
    v_noise_rel: float,
) -> dict:
    """Observe terminal P/Q/V and root V without changing physical AC results."""

    root_true = base.root_true
    if not all(frame.index.equals(root_true.index) for frame in
               (p_meter_true, q_meter_true, voltage_true)):
        raise ValueError("meter loads, terminal voltage, and root voltage must share a time index")
    if list(p_meter_true.columns) != context.terminals or list(q_meter_true.columns) != context.terminals:
        raise ValueError("meter loads must use the observed terminal columns in network order")
    # Preserve the legacy P, Q, V draw order. The root meter has its own RNG.
    rng = np.random.default_rng(base.shifted_seed + 2000)
    p_measured = p_meter_true + rng.normal(
        0.0, pq_noise_rel * np.maximum(np.abs(p_meter_true), 1e-12), size=p_meter_true.shape,
    )
    q_measured = q_meter_true + rng.normal(
        0.0, pq_noise_rel * np.maximum(np.abs(q_meter_true), 1e-12), size=q_meter_true.shape,
    )
    voltage_measured = voltage_true + rng.normal(
        0.0, v_noise_rel * np.maximum(np.abs(voltage_true), 1e-12), size=voltage_true.shape,
    )
    root_meter_seed = 70_000_019 + 100_003 * base.replicate + base.scenario_index
    if context.settings["root_observation"] == "exact":
        root_observed = root_true.copy()
    else:
        root_rng = np.random.default_rng(root_meter_seed)
        root_observed = root_true + root_rng.normal(
            0.0, context.settings["root_meter_noise_rel"] * np.maximum(np.abs(root_true), 1e-12),
            size=len(root_true),
        )
        root_observed.name = "V_root_measured"
    drop_target = squared_voltage_drop_from_observed_root(voltage_measured, root_observed)
    actual_settings = deepcopy(context.settings)
    actual_settings.update({
        "profile_scenario": base.profile,
        "replicate": base.replicate,
        "scenario_index": base.scenario_index,
        "profile_seed": base.shifted_seed,
        "root_seed": base.shifted_seed + 1000,
        "meter_seed": base.shifted_seed + 2000,
        "root_meter_seed": root_meter_seed,
    })
    return {
        "p_measured": p_measured,
        "q_measured": q_measured,
        "voltage_measured": voltage_measured,
        "root_observed": root_observed,
        "drop_target": drop_target,
        "root_meter_seed": root_meter_seed,
        "scenario_settings": actual_settings,
    }


def _simulate_pool(
    case_key: str,
    t_count: int,
    replicate: int,
    maximum_scenarios: int,
    pq_noise_rel: float,
    v_noise_rel: float,
    *,
    scenario_suite: str = "legacy",
    root_observation: str | None = None,
    root_meter_noise_rel: float | None = None,
    root_sigma: float | None = None,
    impedance_scale: float | None = None,
) -> tuple[object, list[dict]]:
    """Simulate a nested scenario pool, preserving the six-argument legacy API.

    New suites generate all 96 physical and measured samples before uniform
    subsampling. Legacy generates the requested sample count directly, keeping
    its original seeds, random-draw order, impedance scales, and observations.
    Both observation modes use an available root meter to form the regression
    target. Truth fields remain available for simulation diagnostics.
    """

    context = _prepare_case_context(
        case_key, t_count, scenario_suite=scenario_suite,
        root_observation=root_observation, root_meter_noise_rel=root_meter_noise_rel,
        root_sigma=root_sigma, impedance_scale=impedance_scale,
    )
    replicate = _integer(replicate, "replicate", 0)
    maximum_scenarios = _integer(maximum_scenarios, "maximum_scenarios", 1)
    if maximum_scenarios > len(SCENARIO_DEFINITIONS):
        raise ValueError(f"only {len(SCENARIO_DEFINITIONS)} scenario definitions are available")
    pq_noise_rel = _finite_number(pq_noise_rel, "pq_noise_rel")
    v_noise_rel = _finite_number(v_noise_rel, "v_noise_rel")
    scenarios = []
    for scenario_index in range(maximum_scenarios):
        base = _build_base_trajectory(context, scenario_index, replicate)
        _, voltage_true = _solve_physical_ac(
            context, base.p_base, base.q_base, base.root_true,
        )
        day = _observe_terminal_channels(
            context, base, base.p_base, base.q_base, voltage_true,
            pq_noise_rel=pq_noise_rel, v_noise_rel=v_noise_rel,
        )
        p_measured = day["p_measured"]
        q_measured = day["q_measured"]
        voltage_measured = day["voltage_measured"]
        root_observed = day["root_observed"]
        drop_target = day["drop_target"]
        actual_settings = day["scenario_settings"]
        diagnostics = _diagnostics(base.p_base, base.q_base, voltage_true, base.root_true,
                                   p_measured, q_measured, voltage_measured, root_observed,
                                   drop_target, actual_settings)
        sample_indices = context.sample_indices
        scenarios.append({
            "name": base.name,
            "P_terminal": p_measured.iloc[sample_indices].copy(),
            "Q_terminal": q_measured.iloc[sample_indices].copy(),
            "V_terminal": voltage_measured.iloc[sample_indices].copy(),
            "root_voltage": root_observed.iloc[sample_indices].copy(),
            "drop_target": drop_target.iloc[sample_indices].copy(),
            "P_true": base.p_base.iloc[sample_indices].copy(),
            "Q_true": base.q_base.iloc[sample_indices].copy(),
            "V_terminal_true": voltage_true.iloc[sample_indices].copy(),
            "root_voltage_true": base.root_true.iloc[sample_indices].copy(),
            "root_observation": context.settings["root_observation"],
            "source_sample_indices": sample_indices.tolist(),
            "scenario_settings": actual_settings,
            "diagnostics": diagnostics,
        })
    return context.net, scenarios
