"""Export current terminalized case33 PQV data and node load-curve assignment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from terminal_case33.data.case33_der_assignment import assignment_summary, build_case33_der_assignment
from terminal_case33.data.case33bw_raw import load_raw_case33bw
from terminal_case33.models.ac_powerflow import solve_ac_power_flow_timeseries
from terminal_case33.scenario.terminalize import terminalize_case33
from terminal_case33.scenario.validate_scenario import validate_terminal_load_only_scenario
from terminal_case33.utils.io import ensure_dir, write_json
from terminal_case33.visualization.plots import plot_network


def _daily_demand_multiplier(t_count: int, customer_type: str, phase: float = 0.0) -> np.ndarray:
    """Return a dataset-inspired normalized demand curve."""

    h = (np.arange(t_count) / t_count * 24.0 + phase) % 24.0
    if customer_type == "commercial":
        workday = 0.34 * np.exp(-0.5 * ((h - 13.0) / 4.2) ** 2)
        shoulder = 0.10 * np.exp(-0.5 * ((h - 8.5) / 1.9) ** 2)
        return 0.50 + workday + shoulder
    if customer_type == "industrial":
        shift = 0.22 * ((h >= 7.0) & (h <= 18.0)).astype(float)
        process = 0.07 * np.sin(2 * np.pi * (h - 3.0) / 8.0)
        return np.clip(0.68 + shift + process, 0.45, 1.05)
    morning = 0.18 * np.exp(-0.5 * ((h - 7.5) / 2.2) ** 2)
    evening = 0.32 * np.exp(-0.5 * ((h - 19.0) / 3.0) ** 2)
    night = 0.06 * np.exp(-0.5 * ((h - 1.0) / 3.5) ** 2)
    return 0.58 + morning + evening + night


def _pv_profile(t_count: int, rng: np.random.Generator, orientation: str, cloud_offset: float) -> np.ndarray:
    """Return a site-specific PV-like profile with orientation and cloud diversity."""

    h = np.arange(t_count) / t_count * 24.0
    peak_hour = {"east": 10.4, "south": 12.6, "west": 15.0, "flat": 12.8}.get(orientation, 12.5)
    width = {"east": 3.9, "south": 4.6, "west": 4.1, "flat": 5.2}.get(orientation, 4.5)
    clear = np.exp(-0.5 * ((h - peak_hour) / width) ** 2)
    daylight = ((h >= 5.8) & (h <= 19.0)).astype(float)
    cloud = np.ones(t_count)
    for center, depth, span in [
        (9.0 + cloud_offset, 0.10 + 0.10 * rng.random(), 0.9 + 0.4 * rng.random()),
        (13.0 - 0.4 * cloud_offset, 0.08 + 0.18 * rng.random(), 1.0 + 0.7 * rng.random()),
        (16.0 + 0.2 * cloud_offset, 0.04 + 0.10 * rng.random(), 0.7 + 0.5 * rng.random()),
    ]:
        cloud -= depth * np.exp(-0.5 * ((h - center) / span) ** 2)
    fast = 1.0 + rng.normal(0.0, 0.015, size=t_count)
    return np.clip(clear * daylight * cloud * fast, 0.0, 1.0)


def _wind_profile(t_count: int, rng: np.random.Generator, site_phase: float, roughness: float) -> np.ndarray:
    """Return a site-specific wind-like profile with visible intra-day variation."""

    h = np.arange(t_count) / t_count * 24.0
    ar = np.zeros(t_count)
    shocks = rng.normal(0.0, 0.09 + roughness, size=t_count)
    for idx in range(1, t_count):
        ar[idx] = 0.86 * ar[idx - 1] + shocks[idx]
    synoptic = 0.38 + 0.23 * np.sin(2 * np.pi * (h - site_phase) / 24.0)
    local = 0.12 * np.sin(2 * np.pi * (h + 0.7 * site_phase) / 8.0)
    ramp_center = 5.0 + 13.0 * rng.random()
    ramp = (0.18 + 0.10 * rng.random()) * np.exp(-0.5 * ((h - ramp_center) / (1.1 + rng.random())) ** 2)
    lull = (0.10 + 0.08 * rng.random()) * np.exp(-0.5 * ((h - (ramp_center + 5.0) % 24.0) / 1.8) ** 2)
    return np.clip(synoptic + local + ar + ramp - lull, 0.02, 0.98)


def _small_generator_profile(t_count: int, rng: np.random.Generator, mode: str) -> np.ndarray:
    """Return a heterogeneous dispatchable small-generator profile."""

    h = np.arange(t_count) / t_count * 24.0
    if mode == "chp_flat":
        return np.clip(0.48 + 0.08 * np.sin(2 * np.pi * (h - 6.0) / 24.0), 0.25, 0.70)
    if mode == "backup_test":
        pulse = np.exp(-0.5 * ((h - 10.5) / 0.65) ** 2)
        return np.clip(0.05 + 0.75 * pulse, 0.0, 0.90)
    if mode == "industrial_shift":
        shift = ((h >= 8.0) & (h <= 16.5)).astype(float)
        return np.clip(0.20 + 0.55 * shift + 0.05 * rng.normal(size=t_count), 0.0, 0.88)
    if mode == "two_peak":
        morning = 0.34 * np.exp(-0.5 * ((h - 8.0) / 1.2) ** 2)
        evening = 0.58 * np.exp(-0.5 * ((h - 19.0) / 1.9) ** 2)
        return np.clip(0.08 + morning + evening, 0.0, 0.92)
    evening_dispatch = ((h >= 17.0) & (h <= 22.0)).astype(float)
    ramp = np.exp(-0.5 * ((h - 19.5) / 1.7) ** 2)
    return np.clip(0.12 + 0.70 * evening_dispatch * ramp, 0.0, 0.95)


def _curve_description(profile_class: str) -> str:
    """Human-readable profile class description."""

    return {
        "base_load": "daily demand multiplier only; no DER subtraction",
        "pv": "daily demand minus PV generation with noon clear-sky peak",
        "wind": "daily demand minus smooth weather-correlated wind generation",
        "small_generator": "daily demand minus dispatchable generator output during peak-support windows",
    }.get(profile_class, "root or unassigned")


def _customer_type(original_bus: int, profile_class: str) -> str:
    """Assign a demand-shape family to an original case33 customer node."""

    if profile_class == "small_generator" or original_bus in {6, 24, 25, 30, 31}:
        return "industrial"
    if original_bus in {2, 3, 4, 5, 23, 24, 25}:
        return "commercial"
    return "residential"


def _resource_components(original_bus: int, profile_class: str, der_kw: float) -> dict[str, float]:
    """Split DER capacity into mixed PV/wind/generator components."""

    if profile_class == "pv":
        if original_bus in {7, 8, 18}:
            return {"pv_kw": der_kw, "wind_kw": 0.0, "small_generator_kw": 0.0}
        if original_bus in {4, 5, 23}:
            return {"pv_kw": 0.78 * der_kw, "wind_kw": 0.0, "small_generator_kw": 0.22 * der_kw}
        return {"pv_kw": 0.70 * der_kw, "wind_kw": 0.30 * der_kw, "small_generator_kw": 0.0}
    if profile_class == "wind":
        if original_bus in {26, 27, 32, 33}:
            return {"pv_kw": 0.25 * der_kw, "wind_kw": 0.75 * der_kw, "small_generator_kw": 0.0}
        if original_bus in {11, 12, 29}:
            return {"pv_kw": 0.0, "wind_kw": 0.72 * der_kw, "small_generator_kw": 0.28 * der_kw}
        return {"pv_kw": 0.0, "wind_kw": der_kw, "small_generator_kw": 0.0}
    if profile_class == "small_generator":
        if original_bus in {24, 30}:
            return {"pv_kw": 0.25 * der_kw, "wind_kw": 0.0, "small_generator_kw": 0.75 * der_kw}
        if original_bus == 31:
            return {"pv_kw": 0.0, "wind_kw": 0.25 * der_kw, "small_generator_kw": 0.75 * der_kw}
        return {"pv_kw": 0.0, "wind_kw": 0.0, "small_generator_kw": der_kw}
    return {"pv_kw": 0.0, "wind_kw": 0.0, "small_generator_kw": 0.0}


def _demand_power_factor(t_count: int, customer_type: str, phase: float) -> np.ndarray:
    """Return a time-varying lagging demand power factor."""

    h = np.arange(t_count) / t_count * 24.0
    base = {"residential": 0.955, "commercial": 0.945, "industrial": 0.925}.get(customer_type, 0.95)
    swing = 0.018 * np.sin(2 * np.pi * (h - 15.0 + phase) / 24.0)
    motor = -0.025 * np.exp(-0.5 * ((h - 18.5) / 2.3) ** 2) if customer_type == "residential" else 0.0
    return np.clip(base + swing + motor, 0.86, 0.99)


def _q_from_pf(p_kw: np.ndarray, pf: np.ndarray | float) -> np.ndarray:
    """Convert active power and power factor into reactive power magnitude."""

    pf_arr = np.clip(np.asarray(pf, dtype=float), 0.2, 0.9999)
    return p_kw * np.tan(np.arccos(pf_arr))


def _der_q_injection(p_gen_kw: np.ndarray, capacity_kw: float, pf: float, sign: float) -> np.ndarray:
    """Reactive DER injection limited by inverter/generator apparent rating."""

    if capacity_kw <= 0.0:
        return np.zeros_like(p_gen_kw)
    s_rating = 1.10 * capacity_kw
    q_desired = sign * _q_from_pf(np.maximum(p_gen_kw, 0.0), pf)
    q_limit = np.sqrt(np.maximum(s_rating**2 - np.minimum(np.maximum(p_gen_kw, 0.0), s_rating) ** 2, 0.0))
    return np.clip(q_desired, -q_limit, q_limit)


def _scenario_shape(
    t_count: int,
    original_bus: int,
    terminal_index: int,
    profile_scenario: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Return demand/PV/wind/generator scale vectors for a weather-load scenario."""

    h = np.arange(t_count) / t_count * 24.0
    ones = np.ones(t_count)
    if profile_scenario in {"default", "base", ""}:
        return ones, ones, ones, ones, "default diversified daily profiles"
    if profile_scenario == "high_wind":
        phase = (original_bus * 0.9 + terminal_index * 1.7) % 24.0
        wind = 1.45 + 0.35 * np.sin(2 * np.pi * (h - phase) / 24.0)
        wind += 0.18 * np.sin(2 * np.pi * (h - 0.4 * phase) / 6.0)
        return ones, 0.85 * ones, np.clip(wind, 0.35, 2.10), ones, "high-wind day with site-specific wind ramps"
    if profile_scenario == "low_wind":
        wind = 0.18 + 0.08 * np.sin(2 * np.pi * (h + original_bus) / 24.0)
        return ones, ones, np.clip(wind, 0.04, 0.35), ones, "low-wind day with weak wind production"
    if profile_scenario == "no_solar":
        cloudy = 0.04 + 0.03 * ((original_bus + terminal_index) % 3)
        return 1.08 * ones, cloudy * ones, 0.95 * ones, ones, "dark cloudy day with almost no PV output"
    if profile_scenario == "cloudy_variable_pv":
        cloud_center = 10.0 + ((original_bus * 1.3 + terminal_index) % 7.0)
        dip = 0.70 * np.exp(-0.5 * ((h - cloud_center) / 2.0) ** 2)
        fast = 0.10 * np.sin(2 * np.pi * (h + original_bus) / 3.4)
        pv = np.clip(0.78 - dip + fast, 0.03, 0.95)
        return 1.02 * ones, pv, 0.90 * ones, ones, "variable cloudy PV day with node-specific cloud passage"
    if profile_scenario == "high_load":
        morning = 0.10 * np.exp(-0.5 * ((h - 7.5) / 2.0) ** 2)
        evening = 0.18 * np.exp(-0.5 * ((h - 19.0) / 2.5) ** 2)
        return 1.28 * ones + morning + evening, 0.90 * ones, 0.90 * ones, 0.85 * ones, "high-load day with slightly reduced DER availability"
    if profile_scenario == "evening_peak_high_load":
        peak = 0.42 * np.exp(-0.5 * ((h - 19.0) / 2.0) ** 2)
        shoulder = 0.12 * np.exp(-0.5 * ((h - 8.0) / 1.8) ** 2)
        return 1.10 * ones + peak + shoulder, 0.80 * ones, 0.85 * ones, 1.15 * ones, "strong evening peak with dispatchable generation response"
    if profile_scenario == "storm_front":
        front = 8.0 + ((original_bus + 2 * terminal_index) % 9.0)
        wind = 0.45 + 1.15 / (1.0 + np.exp(-(h - front) / 0.8))
        wind -= 0.45 * np.exp(-0.5 * ((h - ((front + 5.0) % 24.0)) / 1.4) ** 2)
        pv = 0.55 - 0.40 / (1.0 + np.exp(-(h - front) / 0.9))
        demand = 1.08 + 0.10 * np.exp(-0.5 * ((h - (front + 2.0)) / 2.5) ** 2)
        return demand, np.clip(pv, 0.03, 0.65), np.clip(wind, 0.10, 2.15), ones, "storm-front day: PV collapses while wind ramps by site"
    if profile_scenario == "mixed_cloud_wind":
        pv = 0.55 + 0.25 * np.sin(2 * np.pi * (h - 7.0 - terminal_index * 0.2) / 24.0)
        pv -= 0.35 * np.exp(-0.5 * ((h - (12.0 + original_bus % 5)) / 1.6) ** 2)
        wind = 0.80 + 0.42 * np.sin(2 * np.pi * (h - original_bus * 0.6) / 12.0)
        return 1.04 * ones, np.clip(pv, 0.02, 0.95), np.clip(wind, 0.20, 1.55), ones, "mixed cloudy and windy day with diverse DER ramps"
    raise ValueError(f"unknown profile_scenario: {profile_scenario}")


def _build_terminal_profiles(
    net,
    raw_assignment: pd.DataFrame,
    t_count: int,
    seed: int,
    profile_scenario: str = "default",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create terminal net-load P/Q profiles from deterministic node classes."""

    rng = np.random.default_rng(seed)
    terminals = net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"].astype(int).tolist()
    bus_df = net.buses.set_index("bus_id")
    assignment = raw_assignment.set_index("bus_id")
    idx = pd.RangeIndex(t_count, name="t")
    p = pd.DataFrame(0.0, index=idx, columns=terminals)
    q = pd.DataFrame(0.0, index=idx, columns=terminals)
    records = []
    orientations = ["east", "south", "west", "flat"]
    gen_modes = ["evening_peak", "chp_flat", "backup_test", "industrial_shift", "two_peak"]

    for k, terminal in enumerate(terminals):
        original = int(bus_df.loc[terminal, "original_bus_id"])
        row = assignment.loc[original]
        p0_kw = float(bus_df.loc[terminal, "pd_kw"])
        q0_kvar = float(bus_df.loc[terminal, "qd_kvar"])
        profile_class = str(row["profile_class"])
        der_kw = float(row["der_capacity_kw"])
        customer = str(row["customer_type"]) if "customer_type" in row.index and pd.notna(row["customer_type"]) else _customer_type(original, profile_class)
        demand_scale, pv_scale, wind_scale, generator_scale, scenario_note = _scenario_shape(t_count, original, k, profile_scenario)
        phase_hours = float((original * 0.37 + k * 0.11) % 2.8 - 1.4)
        demand = _daily_demand_multiplier(t_count, customer, phase=phase_hours)
        local_variation = 1.0 + 0.045 * np.sin(2 * np.pi * (np.arange(t_count) / t_count + k / max(1, len(terminals))))
        local_variation += pd.Series(rng.normal(0.0, 0.02, size=t_count)).rolling(5, min_periods=1, center=True).mean().to_numpy()
        demand_kw = np.maximum(0.05 * p0_kw, p0_kw * demand * local_variation * demand_scale)

        if {"pv_capacity_kw", "wind_capacity_kw", "small_generator_capacity_kw"}.issubset(set(assignment.columns)):
            components = {
                "pv_kw": float(row["pv_capacity_kw"]),
                "wind_kw": float(row["wind_capacity_kw"]),
                "small_generator_kw": float(row["small_generator_capacity_kw"]),
            }
        else:
            components = _resource_components(original, profile_class, der_kw)
        orientation = orientations[(original + k) % len(orientations)]
        gen_mode = gen_modes[(original + 2 * k) % len(gen_modes)]
        pv_shape = _pv_profile(t_count, rng, orientation=orientation, cloud_offset=((original % 7) - 3) * 0.25)
        wind_shape = _wind_profile(
            t_count,
            rng,
            site_phase=float((original * 1.7 + k * 0.8) % 24.0),
            roughness=0.015 * ((original % 5) + 1),
        )
        generator_shape = _small_generator_profile(t_count, rng, gen_mode)
        pv_gen_kw = components["pv_kw"] * pv_shape * pv_scale
        wind_gen_kw = components["wind_kw"] * wind_shape * wind_scale
        small_gen_kw = components["small_generator_kw"] * generator_shape * generator_scale
        der_generation = pv_gen_kw + wind_gen_kw + small_gen_kw

        demand_pf = _demand_power_factor(t_count, customer, phase_hours)
        q_demand = _q_from_pf(demand_kw, demand_pf)
        pv_pf = 0.98 - 0.015 * ((original + 1) % 3)
        wind_pf = 0.97 - 0.02 * (original % 3)
        gen_pf = 0.94 + 0.02 * ((original + k) % 3)
        pv_q = _der_q_injection(pv_gen_kw, components["pv_kw"], pv_pf, sign=1.0 if original % 4 else -1.0)
        wind_q = _der_q_injection(wind_gen_kw, components["wind_kw"], wind_pf, sign=1.0 if original % 5 else -1.0)
        generator_q = _der_q_injection(small_gen_kw, components["small_generator_kw"], gen_pf, sign=1.0)

        net_p_kw = demand_kw - der_generation
        net_q_kvar = q_demand - pv_q - wind_q - generator_q
        p.loc[:, terminal] = net_p_kw / (net.base_mva * 1000.0)
        q.loc[:, terminal] = net_q_kvar / (net.base_mva * 1000.0)
        records.append(
            {
                "terminal_bus": terminal,
                "original_bus": original,
                "profile_class": profile_class,
                "profile_scenario": profile_scenario,
                "base_pd_kw": p0_kw,
                "base_qd_kvar": q0_kvar,
                "der_capacity_kw": der_kw,
                "customer_type": customer,
                "pv_capacity_kw": round(float(components["pv_kw"]), 6),
                "wind_capacity_kw": round(float(components["wind_kw"]), 6),
                "small_generator_capacity_kw": round(float(components["small_generator_kw"]), 6),
                "pv_orientation": orientation if components["pv_kw"] > 0.0 else "",
                "small_generator_mode": gen_mode if components["small_generator_kw"] > 0.0 else "",
                "demand_pf_mean": float(demand_pf.mean()),
                "pv_pf": pv_pf if components["pv_kw"] > 0.0 else np.nan,
                "wind_pf": wind_pf if components["wind_kw"] > 0.0 else np.nan,
                "small_generator_pf": gen_pf if components["small_generator_kw"] > 0.0 else np.nan,
                "curve_description": _curve_description(profile_class),
                "scenario_description": scenario_note,
                "component_description": (
                    f"demand={customer}; PV={components['pv_kw']:.3f} kW; "
                    f"wind={components['wind_kw']:.3f} kW; generator={components['small_generator_kw']:.3f} kW"
                ),
                "demand_scenario_scale_mean": float(np.mean(demand_scale)),
                "pv_scenario_scale_mean": float(np.mean(pv_scale)),
                "wind_scenario_scale_mean": float(np.mean(wind_scale)),
                "generator_scenario_scale_mean": float(np.mean(generator_scale)),
                "demand_p_min_kw": float(demand_kw.min()),
                "demand_p_mean_kw": float(demand_kw.mean()),
                "demand_p_max_kw": float(demand_kw.max()),
                "pv_p_mean_kw": float(pv_gen_kw.mean()),
                "wind_p_mean_kw": float(wind_gen_kw.mean()),
                "small_generator_p_mean_kw": float(small_gen_kw.mean()),
                "net_p_min_kw": float(net_p_kw.min()),
                "net_p_mean_kw": float(net_p_kw.mean()),
                "net_p_max_kw": float(net_p_kw.max()),
                "q_demand_mean_kvar": float(q_demand.mean()),
                "q_der_injection_mean_kvar": float((pv_q + wind_q + generator_q).mean()),
                "net_q_min_kvar": float(net_q_kvar.min()),
                "net_q_mean_kvar": float(net_q_kvar.mean()),
                "net_q_max_kvar": float(net_q_kvar.max()),
            }
        )
    return p, q, pd.DataFrame(records)


def _plot_curves(frame: pd.DataFrame, assignment: pd.DataFrame, path: Path, ylabel: str, title: str) -> None:
    """Plot terminal curves colored by profile class."""

    colors = {"base_load": "#7f7f7f", "pv": "#d4a017", "wind": "#1f77b4", "small_generator": "#2ca02c"}
    class_by_terminal = assignment.set_index("terminal_bus")["profile_class"].to_dict()
    plt.figure(figsize=(12, 6))
    for col in frame.columns:
        cls = class_by_terminal[int(col)]
        plt.plot(frame.index, frame[col], color=colors.get(cls, "#333333"), alpha=0.32, linewidth=1.0)
    handles = [
        plt.Line2D([0], [0], color=color, lw=2, label=cls)
        for cls, color in colors.items()
        if cls in set(class_by_terminal.values())
    ]
    plt.legend(handles=handles, loc="best")
    plt.xlabel("time step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run(output_dir: str | Path = "outputs/current_case_pqv_load_curves", T: int = 96, seed: int = 7, noise_rel: float = 0.005) -> dict:
    """Export current terminalized case33 PQV and load-curve assignment."""

    out_dir = ensure_dir(output_dir)
    raw = load_raw_case33bw(include_tie_lines=False)
    net = terminalize_case33(raw, mode="hybrid_leaf", service_impedance_mode="scaled_original", seed=seed)
    scenario = validate_terminal_load_only_scenario(net, strict=True, output_path=out_dir / "scenario_summary.json")
    raw_assignment = build_case33_der_assignment(raw)
    P, Q, curve_assignment = _build_terminal_profiles(net, raw_assignment, T, seed)
    root = pd.Series(1.0 + np.random.default_rng(seed + 100).normal(0.0, 0.001, size=T), index=P.index, name="V_root_true")
    ac = solve_ac_power_flow_timeseries(net, P, Q, v_root=root, max_iter=100, tol=1e-10)

    terminals = list(P.columns)
    V = ac["V_bus_mag"].loc[:, terminals]
    rng = np.random.default_rng(seed + 200)
    P_meas = P + rng.normal(0.0, noise_rel * np.maximum(np.abs(P), 1e-12), size=P.shape)
    Q_meas = Q + rng.normal(0.0, noise_rel * np.maximum(np.abs(Q), 1e-12), size=Q.shape)
    V_meas = V + rng.normal(0.0, noise_rel * np.maximum(np.abs(V), 1e-12), size=V.shape)

    scale = net.base_mva * 1000.0
    P_kw = P * scale
    Q_kvar = Q * scale
    P_meas_kw = P_meas * scale
    Q_meas_kvar = Q_meas * scale
    P_kw.to_csv(out_dir / "P_terminal_true_kw.csv")
    Q_kvar.to_csv(out_dir / "Q_terminal_true_kvar.csv")
    V.to_csv(out_dir / "V_terminal_true_pu.csv")
    P_meas_kw.to_csv(out_dir / "P_terminal_meas_kw.csv")
    Q_meas_kvar.to_csv(out_dir / "Q_terminal_meas_kvar.csv")
    V_meas.to_csv(out_dir / "V_terminal_meas_pu.csv")
    ac["branch_p_from_pu"].to_csv(out_dir / "branch_p_from_pu.csv")
    ac["branch_q_from_pu"].to_csv(out_dir / "branch_q_from_pu.csv")
    ac["branch_loss_p_pu"].to_csv(out_dir / "branch_loss_p_pu.csv")
    curve_assignment.to_csv(out_dir / "terminal_load_curve_assignment.csv", index=False)
    raw_assignment.to_csv(out_dir / "raw_case33_node_profile_assignment.csv", index=False)
    pd.DataFrame(assignment_summary(raw_assignment)).to_csv(out_dir / "raw_case33_profile_summary.csv", index=False)

    summary = curve_assignment.copy()
    summary["v_min_pu"] = [float(V[col].min()) for col in terminals]
    summary["v_mean_pu"] = [float(V[col].mean()) for col in terminals]
    summary["v_max_pu"] = [float(V[col].max()) for col in terminals]
    summary.to_csv(out_dir / "terminal_pqv_summary.csv", index=False)

    long = []
    for terminal in terminals:
        original = int(curve_assignment.set_index("terminal_bus").loc[terminal, "original_bus"])
        cls = str(curve_assignment.set_index("terminal_bus").loc[terminal, "profile_class"])
        for t in P.index:
            long.append(
                {
                    "t": int(t),
                    "terminal_bus": int(terminal),
                    "original_bus": original,
                    "profile_class": cls,
                    "P_true_kw": float(P_kw.loc[t, terminal]),
                    "Q_true_kvar": float(Q_kvar.loc[t, terminal]),
                    "V_true_pu": float(V.loc[t, terminal]),
                    "P_meas_kw": float(P_meas_kw.loc[t, terminal]),
                    "Q_meas_kvar": float(Q_meas_kvar.loc[t, terminal]),
                    "V_meas_pu": float(V_meas.loc[t, terminal]),
                }
            )
    pd.DataFrame(long).to_csv(out_dir / "terminal_pqv_long.csv", index=False)

    feeder = pd.DataFrame(
        {
            "P_total_true_kw": P_kw.sum(axis=1),
            "Q_total_true_kvar": Q_kvar.sum(axis=1),
            "V_terminal_min_pu": V.min(axis=1),
            "V_terminal_mean_pu": V.mean(axis=1),
            "V_terminal_max_pu": V.max(axis=1),
            "root_voltage_pu": root,
        }
    )
    feeder.to_csv(out_dir / "feeder_total_pqv.csv")

    _plot_curves(P_kw, curve_assignment, out_dir / "terminal_P_curves_kw.png", "P net load [kW]", "terminal active-power curves")
    _plot_curves(Q_kvar, curve_assignment, out_dir / "terminal_Q_curves_kvar.png", "Q net load [kvar]", "terminal reactive-power curves")
    _plot_curves(V, curve_assignment, out_dir / "terminal_V_curves_pu.png", "V magnitude [pu]", "terminal voltage curves from AC power flow")
    plot_network(net, out_dir / "topology_terminalized.png", "terminalized case33")

    p_corr = P_kw.corr().to_numpy()
    corr_vals = p_corr[~np.eye(p_corr.shape[0], dtype=bool)]
    component_presence = (
        (curve_assignment["pv_capacity_kw"] > 0).astype(int)
        + (curve_assignment["wind_capacity_kw"] > 0).astype(int)
        + (curve_assignment["small_generator_capacity_kw"] > 0).astype(int)
    )

    metrics = {
        "scenario": scenario,
        "T": T,
        "seed": seed,
        "noise_rel": noise_rel,
        "profile_summary": assignment_summary(raw_assignment),
        "terminal_profile_counts": curve_assignment["profile_class"].value_counts().to_dict(),
        "component_counts": {
            "mixed_component_nodes": int((component_presence > 1).sum()),
            "pv_component_nodes": int((curve_assignment["pv_capacity_kw"] > 0).sum()),
            "wind_component_nodes": int((curve_assignment["wind_capacity_kw"] > 0).sum()),
            "small_generator_component_nodes": int((curve_assignment["small_generator_capacity_kw"] > 0).sum()),
        },
        "p_curve_correlation": {
            "mean_offdiag": float(corr_vals.mean()),
            "min_offdiag": float(corr_vals.min()),
            "max_offdiag": float(corr_vals.max()),
        },
        "total_p_min_kw": float(feeder["P_total_true_kw"].min()),
        "total_p_mean_kw": float(feeder["P_total_true_kw"].mean()),
        "total_p_max_kw": float(feeder["P_total_true_kw"].max()),
        "terminal_v_min_pu": float(V.min().min()),
        "terminal_v_mean_pu": float(V.mean().mean()),
        "terminal_v_max_pu": float(V.max().max()),
        "all_ac_converged": bool(ac["converged"].all()),
        "max_ac_iterations": int(ac["iterations"].max()),
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# Current case PQV and load-curve assignment\n\n"
        "The current export uses terminalized case33 in `hybrid_leaf` mode. Each observed terminal inherits the deterministic "
        "profile class assigned to its original case33 bus. Net terminal load is demand minus DER generation for PV, wind, "
        "and small-generator classes. Voltages and branch flows are generated by nonlinear radial AC power flow.\n\n"
        f"- observed terminals: {scenario['observed_terminal_count']}\n"
        f"- hidden internal nodes: {scenario['hidden_internal_count']}\n"
        f"- time steps: {T}\n"
        f"- relative measurement noise: {noise_rel:.3%}\n"
        f"- total P range: {metrics['total_p_min_kw']:.3f} to {metrics['total_p_max_kw']:.3f} kW\n"
        f"- terminal V range: {metrics['terminal_v_min_pu']:.6f} to {metrics['terminal_v_max_pu']:.6f} pu\n\n"
        f"- mixed DER component nodes: {metrics['component_counts']['mixed_component_nodes']}\n"
        f"- P-curve off-diagonal correlation mean/min/max: "
        f"{metrics['p_curve_correlation']['mean_offdiag']:.3f} / "
        f"{metrics['p_curve_correlation']['min_offdiag']:.3f} / "
        f"{metrics['p_curve_correlation']['max_offdiag']:.3f}\n\n"
        "Main files: `terminal_pqv_long.csv`, `terminal_pqv_summary.csv`, "
        "`terminal_load_curve_assignment.csv`, `P_terminal_true_kw.csv`, `Q_terminal_true_kvar.csv`, `V_terminal_true_pu.csv`.\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/current_case_pqv_load_curves")
    parser.add_argument("--T", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-rel", type=float, default=0.005)
    args = parser.parse_args()
    run(args.output, T=args.T, seed=args.seed, noise_rel=args.noise_rel)


if __name__ == "__main__":
    main()
