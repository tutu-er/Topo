"""Shared experiment helpers for the terminal-load-only research line."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from experiments.export_current_pqv_load_curves import _build_terminal_profiles
from terminal_case33.data.paper_style_case_bank import CASE_BUILDERS, build_case_bank_resource_assignment
from terminal_case33.data.paper_style_terminal_lv import build_paper_style_resource_assignment
from terminal_case33.estimation.preprocessing import squared_voltage_drop_from_observed_root
from terminal_case33.models.ac_powerflow import solve_ac_power_flow_timeseries
from terminal_case33.models.network import TerminalizedNetwork


def terminal_buses(net: TerminalizedNetwork) -> list[int]:
    """Return observed terminal bus ids in stable order."""

    return net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"].astype(int).tolist()


def terminal_buses_from_scenarios(scenarios: list[dict]) -> list[int]:
    """Return terminal or pseudo-terminal column ids from scenario data."""

    return [int(col) for col in scenarios[0]["P_terminal"].columns]


def resource_assignment(case_key: str, net: TerminalizedNetwork) -> pd.DataFrame:
    """Return the case-specific heterogeneous terminal assignment."""

    if case_key == "paper15":
        return build_paper_style_resource_assignment(net)
    return build_case_bank_resource_assignment(net)


def default_scenario_defs() -> list[dict]:
    """Return noisy one-day operating scenarios used by the main benchmarks."""

    return [
        {"name": "oneday_96pts_default_seed42", "seed": 42, "T": 96, "profile_scenario": "default", "v_noise_rel": 0.0002, "root_voltage_sigma": 0.0008},
        {"name": "oneday_288pts_default_seed42", "seed": 42, "T": 288, "profile_scenario": "default", "v_noise_rel": 0.0002, "root_voltage_sigma": 0.0008},
        {"name": "oneday_960pts_default_seed42", "seed": 42, "T": 960, "profile_scenario": "default", "v_noise_rel": 0.0002, "root_voltage_sigma": 0.0008},
        {"name": "oneday_960pts_default_seed7", "seed": 7, "T": 960, "profile_scenario": "default", "v_noise_rel": 0.0002, "root_voltage_sigma": 0.0008},
        {"name": "oneday_960pts_default_seed21", "seed": 21, "T": 960, "profile_scenario": "default", "v_noise_rel": 0.0002, "root_voltage_sigma": 0.0008},
        {"name": "oneday_960pts_high_wind_seed42", "seed": 42, "T": 960, "profile_scenario": "high_wind", "v_noise_rel": 0.0002, "root_voltage_sigma": 0.0008},
        {"name": "oneday_960pts_high_wind_seed88", "seed": 88, "T": 960, "profile_scenario": "high_wind", "v_noise_rel": 0.0002, "root_voltage_sigma": 0.0008},
        {"name": "oneday_960pts_low_wind_seed42", "seed": 42, "T": 960, "profile_scenario": "low_wind", "v_noise_rel": 0.0002, "root_voltage_sigma": 0.0008},
        {"name": "oneday_960pts_no_solar_seed42", "seed": 42, "T": 960, "profile_scenario": "no_solar", "v_noise_rel": 0.0002, "root_voltage_sigma": 0.0008},
        {"name": "oneday_960pts_no_solar_seed123", "seed": 123, "T": 960, "profile_scenario": "no_solar", "v_noise_rel": 0.0002, "root_voltage_sigma": 0.0008},
        {"name": "oneday_960pts_cloudy_variable_pv", "seed": 42, "T": 960, "profile_scenario": "cloudy_variable_pv", "v_noise_rel": 0.0002, "root_voltage_sigma": 0.0008},
        {"name": "oneday_960pts_high_load_seed42", "seed": 42, "T": 960, "profile_scenario": "high_load", "v_noise_rel": 0.0002, "root_voltage_sigma": 0.0008},
        {"name": "oneday_960pts_high_load_seed88", "seed": 88, "T": 960, "profile_scenario": "high_load", "v_noise_rel": 0.0002, "root_voltage_sigma": 0.0008},
        {"name": "oneday_960pts_evening_peak_high_load", "seed": 42, "T": 960, "profile_scenario": "evening_peak_high_load", "v_noise_rel": 0.0002, "root_voltage_sigma": 0.0008},
        {"name": "oneday_960pts_storm_front_seed42", "seed": 42, "T": 960, "profile_scenario": "storm_front", "v_noise_rel": 0.0002, "root_voltage_sigma": 0.0008},
        {"name": "oneday_960pts_storm_front_seed21", "seed": 21, "T": 960, "profile_scenario": "storm_front", "v_noise_rel": 0.0002, "root_voltage_sigma": 0.0008},
        {"name": "oneday_960pts_mixed_cloud_wind", "seed": 42, "T": 960, "profile_scenario": "mixed_cloud_wind", "v_noise_rel": 0.0002, "root_voltage_sigma": 0.0008},
    ]


def selected_scenario_defs(max_scenarios: int | None = None, include_sampling_variants: bool = False) -> list[dict]:
    """Return the multi-scenario set used by the main case-bank benchmarks."""

    scenarios = []
    for item in default_scenario_defs():
        if int(item["T"]) != 960 and not include_sampling_variants:
            continue
        if item["name"] in {"oneday_960pts_low_vnoise", "oneday_960pts_high_root_swing"}:
            continue
        scenarios.append(item)
    return scenarios[:max_scenarios] if max_scenarios is not None else scenarios


def simulate_case_scenario(
    *,
    net: TerminalizedNetwork,
    assignment: pd.DataFrame,
    T: int,
    seed: int,
    pq_noise_rel: float,
    v_noise_rel: float,
    root_voltage_mean: float,
    root_voltage_sigma: float,
    profile_scenario: str,
) -> dict:
    """Generate one noisy AC power-flow scenario for a case-bank feeder."""

    terminals = terminal_buses(net)
    P, Q, curve_assignment = _build_terminal_profiles(net, assignment, T, seed, profile_scenario=profile_scenario)
    root = pd.Series(
        root_voltage_mean + np.random.default_rng(seed + 1000).normal(0.0, root_voltage_sigma, size=T),
        index=P.index,
        name="V_root_true",
    )
    ac = solve_ac_power_flow_timeseries(net, P, Q, v_root=root, max_iter=100, tol=1e-10)
    V = ac["V_bus_mag"].loc[:, terminals]
    rng = np.random.default_rng(seed + 2000)
    P_meas = P + rng.normal(0.0, pq_noise_rel * np.maximum(np.abs(P), 1e-12), size=P.shape)
    Q_meas = Q + rng.normal(0.0, pq_noise_rel * np.maximum(np.abs(Q), 1e-12), size=Q.shape)
    V_meas = V + rng.normal(0.0, v_noise_rel * np.maximum(np.abs(V), 1e-12), size=V.shape)
    return {
        "V_terminal": V_meas,
        "P_terminal": P_meas,
        "Q_terminal": Q_meas,
        "root_voltage": root,
        "drop_target": squared_voltage_drop_from_observed_root(V_meas, root),
        "P_true": P,
        "Q_true": Q,
        "V_true": V,
        "V_bus_true": ac["V_bus_mag"],
        "branch_p_from_true": ac["branch_p_from_pu"],
        "branch_q_from_true": ac["branch_q_from_pu"],
        "curve_assignment": curve_assignment,
        "ac_converged": bool(ac["converged"].all()),
        "max_ac_iterations": int(ac["iterations"].max()),
    }


def simulate_case(
    case_key: str,
    scenario_count: int,
    t_count: int,
    pq_noise_rel: float,
    root_voltage_mean: float,
) -> tuple[TerminalizedNetwork, list[dict]]:
    """Simulate noisy measured scenarios for one case-bank feeder."""

    net = CASE_BUILDERS[case_key]()
    assignment = resource_assignment(case_key, net)
    scenarios = []
    for item in selected_scenario_defs(max_scenarios=scenario_count):
        sim = simulate_case_scenario(
            net=net,
            assignment=assignment,
            T=t_count,
            seed=int(item["seed"]),
            pq_noise_rel=pq_noise_rel,
            v_noise_rel=float(item["v_noise_rel"]),
            root_voltage_mean=root_voltage_mean,
            root_voltage_sigma=float(item["root_voltage_sigma"]),
            profile_scenario=str(item["profile_scenario"]),
        )
        scenarios.append(
            {
                "name": str(item["name"]),
                "profile_scenario": str(item["profile_scenario"]),
                "T": int(t_count),
                "V_terminal": sim["V_terminal"],
                "P_terminal": sim["P_terminal"],
                "Q_terminal": sim["Q_terminal"],
                "root_voltage": sim["root_voltage"],
                "drop_target": sim["drop_target"],
            }
        )
    return net, scenarios


def oriented_parent_map(net: TerminalizedNetwork) -> dict[int, int]:
    """Return root-oriented parent mapping for a physical case."""

    graph = net.to_networkx_graph()
    return {int(child): int(parent) for child, parent in nx.bfs_predecessors(graph, net.root_bus)}


def terminal_parent_map(net: TerminalizedNetwork) -> dict[int, int]:
    """Return each observed terminal's immediate physical parent."""

    parent = oriented_parent_map(net)
    return {bus: parent[bus] for bus in terminal_buses(net)}

