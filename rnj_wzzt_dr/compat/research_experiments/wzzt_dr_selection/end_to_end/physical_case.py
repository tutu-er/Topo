"""Research adapter: external terminal loads -> production AC -> meter channels.

The physical network and AC truth stay outside the observed estimator payload.
paper15 is a repository-built synthetic feeder, not a measured public network.
"""
from __future__ import annotations

from copy import deepcopy
import networkx as nx
import numpy as np
import pandas as pd

from rnj_wzzt.estimation.preprocessing import squared_voltage_drop_from_observed_root
from rnj_wzzt.models.ac_powerflow import solve_ac_power_flow_timeseries
from rnj_wzzt.models.lin_distflow import ohm_to_pu
from rnj_wzzt.scenario.profiles import _demand_power_factor
from rnj_wzzt.scenario.settings import resolve_scenario_settings
from rnj_wzzt.scenario.simulation import (
    _build_base_trajectory, _prepare_case_context, _resource_assignment,
    _root_voltage, _terminal_buses,
)


def build_reference_case(case_key: str = "paper15"):
    """Reuse normal reference-suite scaling and terminal-load-only contract."""
    context = _prepare_case_context(case_key, 24, scenario_suite="reference")
    context.net.check_terminal_load_only(strict=True)
    if context.net.metadata["impedance_scale"] != 1.0:
        raise AssertionError("reference impedance scale must equal one")
    return context.net


def effective_truth_supports(net) -> list[list[int]]:
    """Evaluator-only descendant supports of physical nonzero-impedance lines.

    Shared descendants are deduplicated; zero-impedance edges have no effect in
    the fitted R/X model. This must never supply a planning candidate target.
    """
    graph = net.to_networkx_graph()
    tree = nx.bfs_tree(graph, net.root_bus)
    terminals = _terminal_buses(net)
    supports = set()
    for parent, child in tree.edges:
        edge = graph.edges[parent, child]
        if edge['r_ohm'] == 0 and edge['x_ohm'] == 0:
            continue
        descendants = nx.descendants(tree, child) | {child}
        support = tuple(node for node in terminals if node in descendants)
        if support:
            supports.add(support)
    return [list(s) for s in sorted(supports, key=lambda x: (len(x), x))]


def reference_terminal_profiles(
    *, case_key: str = "paper15", scenario_index: int = 0,
    replicate: int = 0, hourly: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reuse native 96-point P/Q profiles; optional hourly energy averaging.

    Existing mixed demand/DER curves are synthetic and may export active power.
    Hourly averaging creates 24 hourly power levels, not 96 extra observations.
    """
    context = _prepare_case_context(case_key, 24, scenario_suite="reference")
    base = _build_base_trajectory(context, scenario_index, replicate)
    p = base.p_base * (context.net.base_mva * 1000.0)
    q = base.q_base * (context.net.base_mva * 1000.0)
    if hourly:
        p = pd.DataFrame(p.to_numpy().reshape(24, 4, -1).mean(axis=1), columns=p.columns)
        q = pd.DataFrame(q.to_numpy().reshape(24, 4, -1).mean(axis=1), columns=q.columns)
        p.index = q.index = pd.RangeIndex(24, name="hour_start")
    return p, q, {
        "source": "production reference-suite synthetic terminal profiles",
        "case_key": case_key, "scenario_index": scenario_index,
        "replicate": replicate, "profile_seed": base.shifted_seed,
        "profile": base.profile, "native_count": 96,
        "output_count": len(p), "interval_hours": 1.0 if hourly else 0.25,
        "hourly_conversion": "mean of four consecutive native power levels" if hourly else "none",
        "measured_load": False, "includes_existing_der_profiles": True,
        "impedance_scale": 1.0,
    }


def _validate_frame(frame: pd.DataFrame, name: str, terminals: list[int]) -> None:
    if not isinstance(frame, pd.DataFrame) or len(frame) == 0:
        raise ValueError(f"{name} must be a nonempty DataFrame")
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError(f"{name} has duplicate timestamps or terminal IDs")
    if list(frame.columns) != terminals:
        raise ValueError(f"{name} must use exact terminal order {terminals}")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains nonfinite values")


def synthetic_q_from_reference_pf(
    p_kw: pd.DataFrame, *, case_key: str = "paper15",
    customer_mode: str = "residential", interval_hours: float = 1.0,
) -> pd.DataFrame:
    """Generate explicitly synthetic household Q from existing demand PF curves.

    Use only for nonnegative demand, not mixed net DER injections. Values repeat
    the reference 24-hour demand-PF shape; they are not measured iFlex Q.
    """
    net = build_reference_case(case_key)
    terminals = _terminal_buses(net)
    _validate_frame(p_kw, "p_kw", terminals)
    if (p_kw.to_numpy() < 0).any():
        raise ValueError("synthetic household PF requires nonnegative demand")
    if customer_mode not in {"residential", "reference_customer_types"}:
        raise ValueError("unknown customer_mode")
    if interval_hours not in {1.0, 0.25}:
        raise ValueError("PF profile supports hourly or 15-minute intervals")
    count = int(24 / interval_hours)
    assignment = _resource_assignment(case_key, net).set_index("bus_id")
    result = p_kw.copy()
    for k, terminal in enumerate(terminals):
        original = net.terminal_to_original[terminal]
        customer = "residential" if customer_mode == "residential" else str(assignment.loc[terminal, "customer_type"])
        phase = float((original * 0.37 + k * 0.11) % 2.8 - 1.4)
        pf_day = _demand_power_factor(count, customer, phase)
        if isinstance(p_kw.index, pd.DatetimeIndex):
            slots = (p_kw.index.hour + p_kw.index.minute / 60) / interval_hours
            if not np.allclose(slots, np.round(slots)):
                raise ValueError('timestamps must align with PF interval grid')
            pf = pf_day[np.asarray(np.round(slots), dtype=int)]
        else:
            pf = np.tile(pf_day, int(np.ceil(len(p_kw) / count)))[:len(p_kw)]
        result.loc[:, terminal] = p_kw[terminal].to_numpy() * np.tan(np.arccos(pf))
    return result


def _physical_checks(net, p_pu, q_pu, ac, voltage_limits_pu, equation_tolerance_pu):
    """Independent nodal/branch power checks from returned physical quantities."""
    all_buses = net.buses.bus_id.astype(int).tolist()
    local_p = p_pu.reindex(columns=all_buses, fill_value=0.0)
    local_q = q_pu.reindex(columns=all_buses, fill_value=0.0)
    inflow_p = pd.DataFrame(0.0, index=p_pu.index, columns=all_buses)
    inflow_q = inflow_p.copy()
    outflow_p = inflow_p.copy()
    outflow_q = inflow_p.copy()
    graph = net.to_networkx_graph()
    branch_loss_error, distflow_error = 0.0, 0.0
    for edge in ac["metadata"]["branch_edges"]:
        label, u, v = edge["label"], edge["from_bus"], edge["to_bus"]
        inflow_p.loc[:, v] += ac["branch_p_to_pu"][label]
        inflow_q.loc[:, v] += ac["branch_q_to_pu"][label]
        outflow_p.loc[:, u] += ac["branch_p_from_pu"][label]
        outflow_q.loc[:, u] += ac["branch_q_from_pu"][label]
        r = ohm_to_pu(net, graph.edges[u, v]["r_ohm"])
        x = ohm_to_pu(net, graph.edges[u, v]["x_ohm"])
        ell = ac["I_branch_mag"][label].to_numpy() ** 2
        branch_loss_error = max(branch_loss_error,
            float(np.max(np.abs(ac["branch_loss_p_pu"][label].to_numpy() - r * ell))),
            float(np.max(np.abs(ac["branch_loss_q_pu"][label].to_numpy() - x * ell))))
        d = (ac["V_bus_mag"][u].to_numpy()**2 - ac["V_bus_mag"][v].to_numpy()**2
             - 2 * (r * ac["branch_p_from_pu"][label].to_numpy() + x * ac["branch_q_from_pu"][label].to_numpy())
             + (r*r + x*x) * ell)
        distflow_error = max(distflow_error, float(np.max(np.abs(d))))
    nonroot = [x for x in all_buses if x != net.root_bus]
    nodal_p_error = float(np.max(np.abs((inflow_p - local_p - outflow_p)[nonroot].to_numpy())))
    nodal_q_error = float(np.max(np.abs((inflow_q - local_q - outflow_q)[nonroot].to_numpy())))
    slack_p = outflow_p[net.root_bus] + local_p[net.root_bus]
    slack_q = outflow_q[net.root_bus] + local_q[net.root_bus]
    balance_p = slack_p - local_p.sum(axis=1) - ac["branch_loss_p_pu"].sum(axis=1)
    balance_q = slack_q - local_q.sum(axis=1) - ac["branch_loss_q_pu"].sum(axis=1)
    v = ac["V_bus_mag"].to_numpy()
    lo, hi = voltage_limits_pu
    convergence = ac["converged"].to_numpy(dtype=bool)
    physical_finite = all(np.isfinite(ac[key].to_numpy()).all() for key in (
        "V_bus_mag", "V_bus_angle_deg", "I_branch_mag", "branch_p_from_pu",
        "branch_q_from_pu", "branch_p_to_pu", "branch_q_to_pu",
        "branch_loss_p_pu", "branch_loss_q_pu", "max_voltage_error"))
    residual = max(nodal_p_error, nodal_q_error, branch_loss_error,
                   float(np.max(np.abs(balance_p))), float(np.max(np.abs(balance_q))))
    return {
        "sample_count": len(p_pu), "converged_samples": int(convergence.sum()),
        "nonconverged_samples": int((~convergence).sum()),
        "physical_outputs_finite": bool(physical_finite),
        "max_nodal_p_residual_pu": nodal_p_error, "max_nodal_q_residual_pu": nodal_q_error,
        "max_system_p_balance_residual_pu": float(np.max(np.abs(balance_p))),
        "max_system_q_balance_residual_pu": float(np.max(np.abs(balance_q))),
        "max_branch_loss_formula_error_pu": branch_loss_error,
        "max_distflow_identity_error_pu2": distflow_error,
        "equation_tolerance_pu": equation_tolerance_pu,
        "electrical_equations_pass": bool(physical_finite and residual <= equation_tolerance_pu and distflow_error <= equation_tolerance_pu),
        "physical_voltage_min_pu": float(v.min()), "physical_voltage_max_pu": float(v.max()),
        "voltage_limits_pu": [lo, hi],
        "voltage_violation_cells": int(((v < lo) | (v > hi)).sum()),
        "voltage_limits_pass": bool(((v >= lo) & (v <= hi)).all()),
        "voltage_limit_meaning": "declared scenario screening range, not a regulatory certificate",
        "thermal_limits_checked": False,
        "thermal_reason": "paper-style builder has no branch ampacity ratings",
        "max_iterations": int(ac["iterations"].max()),
        "max_solver_voltage_update_pu": float(ac["max_voltage_error"].max()),
        "no_sample_dropped": True,
    }


def simulate_observations(
    net, p_kw: pd.DataFrame, q_kvar: pd.DataFrame, *, seed: int,
    name: str = "external_terminal_day", root_true: pd.Series | None = None,
    interval_hours: float = 1.0, pq_noise_rel: float = 0.005,
    v_noise_rel: float = 0.0002, root_meter_noise_rel: float = 0.0002,
    root_sigma: float = 0.003, voltage_limits_pu: tuple[float, float] = (0.90, 1.10),
    max_iter: int = 100, tol: float = 1e-10, equation_tolerance_pu: float = 1e-8,
) -> dict:
    """Run production balanced radial AC and its relative Gaussian meter model.

    Positive kW/kvar means consumption. Paired strategies can reuse the same seed
    to share standardized meter draws and physical root realization. Invalid input
    raises; nonconverged AC is retained in truth/checks and exposes no observed
    estimator payload. Operational voltage violations remain explicitly reported.
    """
    net.check_terminal_load_only(strict=True)
    terminals = _terminal_buses(net)
    _validate_frame(p_kw, "p_kw", terminals); _validate_frame(q_kvar, "q_kvar", terminals)
    if not p_kw.index.equals(q_kvar.index):
        raise ValueError("P and Q must have identical timestamps")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for key, value in {"interval_hours": interval_hours, "pq_noise_rel": pq_noise_rel,
                       "v_noise_rel": v_noise_rel, "root_meter_noise_rel": root_meter_noise_rel,
                       "root_sigma": root_sigma, "equation_tolerance_pu": equation_tolerance_pu}.items():
        if not np.isfinite(value) or value < 0 or (key in {"interval_hours", "equation_tolerance_pu"} and value == 0):
            raise ValueError(f"invalid {key}")
    if not 0 < voltage_limits_pu[0] < voltage_limits_pu[1]:
        raise ValueError("invalid voltage limits")
    settings = resolve_scenario_settings("reference", root_sigma=root_sigma,
                                         root_meter_noise_rel=root_meter_noise_rel)
    settings["day_hours"] = len(p_kw) * interval_hours
    root_source = 'reference root-voltage generator' if root_true is None else 'explicit external physical root'
    if root_true is None:
        root_true = _root_voltage(p_kw.index, int(seed) + 1000, settings)
    else:
        if not isinstance(root_true, pd.Series) or not root_true.index.equals(p_kw.index):
            raise ValueError("root_true must share the full P/Q index")
        root_true = root_true.copy()
    if not np.isfinite(root_true).all() or (root_true <= 0).any():
        raise ValueError("physical root magnitude must be finite and positive")
    base_kw = net.base_mva * 1000.0
    p_pu, q_pu = p_kw / base_kw, q_kvar / base_kw
    ac = solve_ac_power_flow_timeseries(net, p_pu, q_pu, v_root=root_true,
                                      max_iter=max_iter, tol=tol)
    checks = _physical_checks(net, p_pu, q_pu, ac, voltage_limits_pu, equation_tolerance_pu)
    valid = checks["nonconverged_samples"] == 0 and checks["electrical_equations_pass"]
    checks["usable_ac_solution"] = valid
    truth = {"P_kw": p_kw.copy(), "Q_kvar": q_kvar.copy(), "P_true": p_pu,
             "Q_true": q_pu, "root_voltage_true": root_true,
             "V_terminal_true": ac["V_bus_mag"].loc[:, terminals].copy(), "ac": ac,
             "truth_supports": effective_truth_supports(net)}
    if not valid:
        return {"observed": None, "truth": truth, "checks": checks}
    v_true = truth["V_terminal_true"]
    rng = np.random.default_rng(int(seed) + 2000)
    p = p_pu + rng.normal(0, pq_noise_rel * np.maximum(np.abs(p_pu), 1e-12), size=p_pu.shape)
    q = q_pu + rng.normal(0, pq_noise_rel * np.maximum(np.abs(q_pu), 1e-12), size=q_pu.shape)
    v = v_true + rng.normal(0, v_noise_rel * np.maximum(np.abs(v_true), 1e-12), size=v_true.shape)
    root_rng = np.random.default_rng(int(seed) + 70_000_019)
    root = root_true + root_rng.normal(0, root_meter_noise_rel * np.maximum(np.abs(root_true), 1e-12), len(root_true))
    root.name = "V_root_measured"
    settings.update({"interval_hours": interval_hours, "sample_count": len(p),
                     "root_source": root_source,
                     "root_seed": int(seed) + 1000 if root_source == 'reference root-voltage generator' else None,
                     "meter_seed": int(seed) + 2000,
                     "root_meter_seed": int(seed) + 70_000_019,
                     "pq_noise_rel": pq_noise_rel, "v_noise_rel": v_noise_rel,
                     "input_units": "kW/kvar", "estimator_power_units": "pu",
                     "voltage_units": "pu", "base_mva": net.base_mva,
                     "impedance_scale": float(net.metadata["impedance_scale"]),
                     "source": "external terminal loads; production AC and reference meter model"})
    observed = {"name": name, "P_terminal": p, "Q_terminal": q,
                "V_terminal": v, "root_voltage": root,
                "drop_target": squared_voltage_drop_from_observed_root(v, root),
                "root_observation": "noisy" if root_meter_noise_rel else "exact",
                "scenario_settings": deepcopy(settings)}
    return {"observed": observed, "truth": truth, "checks": checks}
