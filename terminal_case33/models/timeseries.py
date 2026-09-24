"""Terminal load and voltage time-series simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from terminal_case33.models.ac_powerflow import solve_ac_power_flow_timeseries
from terminal_case33.models.network import TerminalizedNetwork


def _profiles(rng: np.random.Generator, n: int, t_count: int, mode: str) -> np.ndarray:
    """Generate dimensionless load multipliers."""

    time = np.arange(t_count)
    base = 1.0 + 0.18 * np.sin(2 * np.pi * time / max(t_count, 2))
    if mode == "independent_residential":
        return np.maximum(0.05, base[:, None] + rng.normal(0.0, 0.22, size=(t_count, n)))
    if mode == "correlated_residential":
        common = rng.normal(0.0, 0.25, size=(t_count, 1))
        return np.maximum(0.05, base[:, None] + common + rng.normal(0.0, 0.04, size=(t_count, n)))
    if mode == "pv_ev_mixed":
        pv = -0.35 * np.maximum(0.0, np.sin(2 * np.pi * (time / max(t_count, 2) - 0.2)))[:, None]
        ev = (rng.random((t_count, n)) < 0.04).astype(float) * rng.uniform(0.3, 1.0, size=(t_count, n))
        return np.maximum(-0.4, base[:, None] + pv + ev + rng.normal(0.0, 0.12, size=(t_count, n)))
    if mode == "low_excitation":
        return np.maximum(0.05, 1.0 + rng.normal(0.0, 0.015, size=(t_count, n)))
    if mode == "step_probe":
        prof = np.ones((t_count, n))
        for k in range(n):
            start = (k * max(2, t_count // max(n, 1))) % max(t_count, 1)
            prof[start : min(t_count, start + max(2, t_count // 8)), k] += 0.45
        return prof + rng.normal(0.0, 0.01, size=(t_count, n))
    raise ValueError(f"unknown profile_mode: {mode}")


def simulate_terminal_load_timeseries(
    net: TerminalizedNetwork,
    T: int,
    dt_seconds: int,
    profile_mode: str,
    load_scale: float = 1.0,
    pf_mode: str = "original_qp_ratio",
    noise_config: dict | None = None,
    v0_config: dict | None = None,
    delay_config: dict | None = None,
    hidden_load_leakage: float = 0.0,
    seed: int = 0,
) -> dict:
    """Simulate terminal P/Q/V data from nonlinear radial AC power flow.

    P and Q outputs are in per unit on ``net.base_mva``. Hidden internal buses
    remain zero unless ``hidden_load_leakage`` is positive, in which case the
    metadata is marked as a strict-scenario violation.
    """

    del dt_seconds
    rng = np.random.default_rng(seed)
    terminals = net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"].astype(int).tolist()
    n = len(terminals)
    terminal_df = net.buses.set_index("bus_id").loc[terminals]
    p0 = terminal_df["pd_kw"].to_numpy(dtype=float) / (net.base_mva * 1000.0) * load_scale
    q0 = terminal_df["qd_kvar"].to_numpy(dtype=float) / (net.base_mva * 1000.0) * load_scale
    mult = _profiles(rng, n, T, profile_mode)
    p = mult * p0[None, :]

    if pf_mode == "original_qp_ratio":
        ratio = np.divide(q0, p0, out=np.zeros_like(q0), where=p0 != 0)
        q = p * ratio[None, :]
    elif pf_mode == "fixed_pf":
        ratio = np.tan(np.arccos(0.95))
        q = p * ratio
    elif pf_mode == "time_varying_pf":
        ratio = np.divide(q0, p0, out=np.zeros_like(q0), where=p0 != 0)
        phases = np.linspace(0.0, np.pi, n, endpoint=False)
        waves = np.sin(np.linspace(0, 6 * np.pi, T)[:, None] + phases[None, :])
        q = p * np.maximum(0.05, ratio[None, :] + 0.15 * waves)
    elif pf_mode == "noisy_q_ratio":
        ratio = np.divide(q0, p0, out=np.zeros_like(q0), where=p0 != 0)
        q = p * (ratio[None, :] + rng.normal(0.0, 0.05, size=(T, n)))
    else:
        raise ValueError(f"unknown pf_mode: {pf_mode}")

    v0_config = v0_config or {}
    root_sigma = float(v0_config.get("sigma", 0.002))
    root = 1.0 + rng.normal(0.0, root_sigma, size=T)

    all_bus_ids = net.buses["bus_id"].astype(int).tolist()
    p_all = pd.DataFrame(0.0, index=pd.RangeIndex(T, name="t"), columns=all_bus_ids)
    q_all = pd.DataFrame(0.0, index=pd.RangeIndex(T, name="t"), columns=all_bus_ids)
    for col_idx, bus in enumerate(terminals):
        p_all.loc[:, bus] = p[:, col_idx]
        q_all.loc[:, bus] = q[:, col_idx]

    leakage_buses: list[int] = []
    if hidden_load_leakage > 0.0:
        hidden = net.buses.loc[net.buses["bus_type"].eq("hidden_internal"), "bus_id"].astype(int).tolist()
        if hidden:
            count = max(1, min(len(hidden), int(np.ceil(0.1 * len(hidden)))))
            leakage_buses = sorted(map(int, rng.choice(hidden, size=count, replace=False)))
            weights = rng.dirichlet(np.ones(count))
            p_hidden_total = hidden_load_leakage * p.sum(axis=1)
            q_hidden_total = hidden_load_leakage * q.sum(axis=1)
            for bus, weight in zip(leakage_buses, weights):
                p_all.loc[:, bus] = p_hidden_total * weight
                q_all.loc[:, bus] = q_hidden_total * weight

    ac = solve_ac_power_flow_timeseries(
        net,
        p_all,
        q_all,
        v_root=pd.Series(root, index=p_all.index),
        max_iter=int(v0_config.get("ac_max_iter", 100)),
        tol=float(v0_config.get("ac_tol", 1e-10)),
    )
    v = ac["V_bus_mag"].loc[:, terminals].to_numpy(dtype=float)

    noise_config = noise_config or {}
    p_noise = float(noise_config.get("p_std", 0.0))
    q_noise = float(noise_config.get("q_std", 0.0))
    v_noise = float(noise_config.get("v_std", 0.0))
    p_rel_noise = float(noise_config.get("p_rel_std", 0.0))
    q_rel_noise = float(noise_config.get("q_rel_std", 0.0))
    v_rel_noise = float(noise_config.get("v_rel_std", 0.0))
    p_scale = np.maximum(np.abs(p), 1e-12)
    q_scale = np.maximum(np.abs(q), 1e-12)
    v_scale = np.maximum(np.abs(v), 1e-12)
    p_meas = p + rng.normal(0.0, p_noise, size=p.shape)
    p_meas += rng.normal(0.0, p_rel_noise * p_scale, size=p.shape)
    q_meas = q + rng.normal(0.0, q_noise, size=q.shape)
    q_meas += rng.normal(0.0, q_rel_noise * q_scale, size=q.shape)
    v_meas = v + rng.normal(0.0, v_noise, size=v.shape)
    v_meas += rng.normal(0.0, v_rel_noise * v_scale, size=v.shape)

    delay_config = delay_config or {}
    max_delay = int(delay_config.get("max_delay_steps", 0))
    delays = np.zeros(n, dtype=int)
    if max_delay > 0:
        delays = rng.integers(-max_delay, max_delay + 1, size=n)
        for col_idx, lag in enumerate(delays):
            p_meas[:, col_idx] = np.roll(p_meas[:, col_idx], lag)
            q_meas[:, col_idx] = np.roll(q_meas[:, col_idx], lag)

    columns = list(map(int, terminals))
    idx = pd.RangeIndex(T, name="t")
    metadata = {
        "terminals": columns,
        "profile_mode": profile_mode,
        "pf_mode": pf_mode,
        "hidden_load_leakage": hidden_load_leakage,
        "violation": hidden_load_leakage > 0.0,
        "voltage_generation_model": "nonlinear_radial_ac_power_flow",
        "power_flow_solver": ac["metadata"],
        "hidden_leakage_buses": leakage_buses,
        "measurement_noise": {
            "p_std": p_noise,
            "q_std": q_noise,
            "v_std": v_noise,
            "p_rel_std": p_rel_noise,
            "q_rel_std": q_rel_noise,
            "v_rel_std": v_rel_noise,
        },
        "delays": {int(bus): int(delay) for bus, delay in zip(columns, delays)},
    }
    return {
        "P_terminal_true": pd.DataFrame(p, index=idx, columns=columns),
        "Q_terminal_true": pd.DataFrame(q, index=idx, columns=columns),
        "V_terminal_true": pd.DataFrame(v, index=idx, columns=columns),
        "P_terminal_meas": pd.DataFrame(p_meas, index=idx, columns=columns),
        "Q_terminal_meas": pd.DataFrame(q_meas, index=idx, columns=columns),
        "V_terminal_meas": pd.DataFrame(v_meas, index=idx, columns=columns),
        "V_root_true": pd.Series(root, index=idx, name="V_root_true"),
        "V_bus_true": ac["V_bus_mag"],
        "V_bus_angle_deg_true": ac["V_bus_angle_deg"],
        "I_branch_mag_true": ac["I_branch_mag"],
        "branch_p_from_pu_true": ac["branch_p_from_pu"],
        "branch_q_from_pu_true": ac["branch_q_from_pu"],
        "branch_p_to_pu_true": ac["branch_p_to_pu"],
        "branch_q_to_pu_true": ac["branch_q_to_pu"],
        "branch_loss_p_pu_true": ac["branch_loss_p_pu"],
        "branch_loss_q_pu_true": ac["branch_loss_q_pu"],
        "ac_converged": ac["converged"],
        "ac_iterations": ac["iterations"],
        "ac_max_voltage_error": ac["max_voltage_error"],
        "topology_metadata": metadata,
    }

