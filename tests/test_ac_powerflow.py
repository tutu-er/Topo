"""Tests for nonlinear radial AC power-flow calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from terminal_case33.data.small_terminal_lv import build_small_terminal_lv_case
from terminal_case33.models.ac_powerflow import solve_ac_power_flow_snapshot, solve_ac_power_flow_timeseries


def test_ac_snapshot_converges_and_has_positive_losses() -> None:
    """A positive-load LV feeder should converge and dissipate active losses."""

    net = build_small_terminal_lv_case()
    terminals = net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"].astype(int).tolist()
    p = {bus: 0.65 * float(net.buses.set_index("bus_id").loc[bus, "pd_kw"]) / (net.base_mva * 1000.0) for bus in terminals}
    q = {bus: 0.65 * float(net.buses.set_index("bus_id").loc[bus, "qd_kvar"]) / (net.base_mva * 1000.0) for bus in terminals}

    result = solve_ac_power_flow_snapshot(net, p, q, v_root=1.0 + 0.0j)

    assert result.converged
    assert result.iterations < 50
    assert abs(result.bus_voltage[net.root_bus] - 1.0) < 1e-12
    assert min(abs(v) for v in result.bus_voltage.values()) > 0.9
    assert all(loss.real >= -1e-12 for loss in result.branch_loss.values())


def test_ac_timeseries_power_balance_at_root() -> None:
    """Root sending power should equal total terminal load plus AC line losses."""

    net = build_small_terminal_lv_case()
    terminals = net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"].astype(int).tolist()
    base = net.buses.set_index("bus_id").loc[terminals]
    p0 = base["pd_kw"].to_numpy(dtype=float) / (net.base_mva * 1000.0)
    q0 = base["qd_kvar"].to_numpy(dtype=float) / (net.base_mva * 1000.0)
    multipliers = np.array([[0.7], [1.0], [1.2]])
    P = pd.DataFrame(multipliers * p0[None, :], columns=terminals)
    Q = pd.DataFrame(multipliers * q0[None, :], columns=terminals)

    result = solve_ac_power_flow_timeseries(net, P, Q, v_root=1.0)

    root_cols = [
        item["label"]
        for item in result["metadata"]["branch_edges"]
        if item["from_bus"] == net.root_bus
    ]
    root_p = result["branch_p_from_pu"][root_cols].sum(axis=1)
    root_q = result["branch_q_from_pu"][root_cols].sum(axis=1)
    assert bool(result["converged"].all())
    assert np.allclose(root_p, P.sum(axis=1) + result["branch_loss_p_pu"].sum(axis=1), atol=1e-10)
    assert np.allclose(root_q, Q.sum(axis=1) + result["branch_loss_q_pu"].sum(axis=1), atol=1e-10)
    assert result["V_bus_mag"].shape == (3, len(net.buses))


def test_batched_timeseries_matches_independent_snapshots() -> None:
    """Batched sweeps must preserve the independent snapshot solution."""

    net = build_small_terminal_lv_case()
    terminals = net.load_buses()
    base = net.buses.set_index("bus_id").loc[terminals]
    scales = np.array([[0.55], [0.83], [1.11], [0.97]])
    P = pd.DataFrame(
        scales * base["pd_kw"].to_numpy(dtype=float)[None, :] / (net.base_mva * 1000.0),
        columns=terminals,
    )
    Q = pd.DataFrame(
        scales * base["qd_kvar"].to_numpy(dtype=float)[None, :] / (net.base_mva * 1000.0),
        columns=terminals,
    )
    root = pd.Series([1.018, 1.021, 1.017, 1.023], index=P.index)

    batched = solve_ac_power_flow_timeseries(net, P, Q, v_root=root)
    edge_labels = {
        (item["from_bus"], item["to_bus"]): item["label"]
        for item in batched["metadata"]["branch_edges"]
    }
    for row in P.index:
        snapshot = solve_ac_power_flow_snapshot(
            net,
            P.loc[row].to_dict(),
            Q.loc[row].to_dict(),
            v_root=float(root.loc[row]),
        )
        expected_voltage = np.array(
            [abs(snapshot.bus_voltage[int(bus)]) for bus in batched["V_bus_mag"].columns]
        )
        expected_p_from = np.array(
            [snapshot.branch_power_from[edge].real for edge in edge_labels]
        )
        assert np.allclose(batched["V_bus_mag"].loc[row], expected_voltage, atol=1e-12)
        assert np.allclose(batched["branch_p_from_pu"].loc[row], expected_p_from, atol=1e-12)
        assert int(batched["iterations"].loc[row]) == snapshot.iterations