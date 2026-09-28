"""Radial AC power-flow solver for terminal-load-only feeders.

The solver uses the feeder root as the slack/reference node and applies the
standard backward-forward sweep equations on a balanced single-phase equivalent
radial network. Loads are positive consumptions in per unit on ``net.base_mva``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import networkx as nx
import numpy as np
import pandas as pd

from rnj_wzzt.models.lin_distflow import ohm_to_pu
from rnj_wzzt.models.network import TerminalizedNetwork


@dataclass
class ACPowerFlowSnapshot:
    """One-time-step AC power-flow result."""

    bus_voltage: dict[int, complex]
    branch_current: dict[tuple[int, int], complex]
    branch_power_from: dict[tuple[int, int], complex]
    branch_power_to: dict[tuple[int, int], complex]
    branch_loss: dict[tuple[int, int], complex]
    converged: bool
    iterations: int
    max_voltage_error: float


def _validate_solver_inputs(max_iter: int, tol: float, *values) -> float:
    """Reject invalid stopping rules and nonfinite electrical inputs."""

    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if isinstance(tol, (bool, np.bool_)):
        raise ValueError("tol must be finite and positive")
    try:
        tol = float(tol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("tol must be finite and positive") from exc
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    if any(not np.isfinite(np.asarray(value, dtype=complex)).all() for value in values):
        raise ValueError("power-flow inputs must be finite")
    return tol


def _oriented_tree(net: TerminalizedNetwork) -> tuple[list[int], dict[int, int], dict[int, list[int]], dict[tuple[int, int], complex]]:
    """Return root-oriented nodes, parents, children, and per-unit impedances."""

    graph = net.to_networkx_graph()
    if not nx.is_tree(graph):
        raise ValueError("AC radial power flow requires a connected acyclic closed network")
    if net.root_bus not in graph:
        raise ValueError(f"root bus {net.root_bus} is not present in the network graph")

    bfs_nodes = list(nx.bfs_tree(graph, net.root_bus).nodes())
    parent = dict(nx.bfs_predecessors(graph, net.root_bus))
    children: dict[int, list[int]] = {int(node): [] for node in bfs_nodes}
    z_pu: dict[tuple[int, int], complex] = {}
    for child, par in parent.items():
        child_i = int(child)
        par_i = int(par)
        children[par_i].append(child_i)
        edge = graph.edges[par_i, child_i]
        z_pu[(par_i, child_i)] = complex(
            ohm_to_pu(net, float(edge["r_ohm"])),
            ohm_to_pu(net, float(edge["x_ohm"])),
        )
    if not np.isfinite(list(z_pu.values())).all():
        raise ValueError("branch impedances must be finite")
    return [int(node) for node in bfs_nodes], {int(k): int(v) for k, v in parent.items()}, children, z_pu


def solve_ac_power_flow_snapshot(
    net: TerminalizedNetwork,
    p_load_pu: Mapping[int, float],
    q_load_pu: Mapping[int, float],
    v_root: complex | float = 1.0 + 0.0j,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> ACPowerFlowSnapshot:
    """Solve one radial AC power-flow snapshot by backward-forward sweep.

    Args:
        net: Terminalized radial feeder.
        p_load_pu: Positive active load at each bus in per unit.
        q_load_pu: Positive reactive load at each bus in per unit.
        v_root: Root/slack voltage phasor. A real value means zero angle.
        max_iter: Maximum backward-forward sweep iterations.
        tol: Convergence tolerance on maximum complex-voltage update.

    Returns:
        A snapshot with bus phasors, oriented branch currents, branch complex
        powers at sending/receiving ends, losses, and convergence metadata.
    """

    tol = _validate_solver_inputs(max_iter, tol, list(p_load_pu.values()), list(q_load_pu.values()), v_root)
    nodes, parent, children, z_pu = _oriented_tree(net)
    return _solve_ac_power_flow_snapshot_oriented(
        net,
        nodes,
        parent,
        children,
        z_pu,
        p_load_pu,
        q_load_pu,
        v_root=v_root,
        max_iter=max_iter,
        tol=tol,
    )


def _solve_ac_power_flow_snapshot_oriented(
    net: TerminalizedNetwork,
    nodes: list[int],
    parent: dict[int, int],
    children: dict[int, list[int]],
    z_pu: dict[tuple[int, int], complex],
    p_load_pu: Mapping[int, float],
    q_load_pu: Mapping[int, float],
    v_root: complex | float,
    max_iter: int,
    tol: float,
) -> ACPowerFlowSnapshot:
    """Solve one snapshot using a precomputed root-oriented tree."""

    slack = complex(v_root)
    voltages = {node: slack for node in nodes}
    branch_current = {edge: 0.0 + 0.0j for edge in z_pu}
    converged = False
    max_error = float("inf")

    s_load = {
        node: complex(float(p_load_pu.get(node, 0.0)), float(q_load_pu.get(node, 0.0)))
        for node in nodes
    }

    for iteration in range(1, max_iter + 1):
        old_voltages = voltages.copy()
        downstream_current: dict[int, complex] = {}

        for node in reversed(nodes):
            v_node = voltages[node]
            if abs(v_node) < 1e-8:
                raise FloatingPointError(f"voltage collapsed near zero at bus {node}")
            i_load = np.conj(s_load[node] / v_node)
            i_downstream = sum(branch_current[(node, child)] for child in children.get(node, []))
            downstream_current[node] = complex(i_load + i_downstream)
            if node != net.root_bus:
                par = parent[node]
                branch_current[(par, node)] = downstream_current[node]

        voltages[net.root_bus] = slack
        for node in nodes:
            for child in children.get(node, []):
                voltages[child] = voltages[node] - z_pu[(node, child)] * branch_current[(node, child)]

        max_error = float(np.max([abs(voltages[node] - old_voltages[node]) for node in nodes]))
        if max_error <= tol:
            converged = True
            break

    branch_power_from: dict[tuple[int, int], complex] = {}
    branch_power_to: dict[tuple[int, int], complex] = {}
    branch_loss: dict[tuple[int, int], complex] = {}
    for edge, current in branch_current.items():
        u, v = edge
        s_from = voltages[u] * np.conj(current)
        s_to = voltages[v] * np.conj(current)
        branch_power_from[edge] = complex(s_from)
        branch_power_to[edge] = complex(s_to)
        branch_loss[edge] = complex(s_from - s_to)

    return ACPowerFlowSnapshot(
        bus_voltage=voltages,
        branch_current=branch_current,
        branch_power_from=branch_power_from,
        branch_power_to=branch_power_to,
        branch_loss=branch_loss,
        converged=converged,
        iterations=iteration,
        max_voltage_error=float(max_error),
    )


def solve_ac_power_flow_timeseries(
    net: TerminalizedNetwork,
    P_load_pu: pd.DataFrame,
    Q_load_pu: pd.DataFrame,
    v_root: float | complex | pd.Series | np.ndarray = 1.0,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> dict[str, pd.DataFrame | pd.Series | dict]:
    """Solve radial AC power flow for every row of a load time series.

    ``P_load_pu`` and ``Q_load_pu`` may contain only terminal columns or all bus
    columns. Missing buses are treated as zero load. Output branch quantities
    are oriented away from the root and expressed in per unit.
    """

    if not P_load_pu.index.equals(Q_load_pu.index):
        raise ValueError("P_load_pu and Q_load_pu must use the same time index")
    if list(P_load_pu.columns) != list(Q_load_pu.columns):
        raise ValueError("P_load_pu and Q_load_pu must use the same bus columns")

    nodes, parent, children, z_pu = _oriented_tree(net)
    edges = list(z_pu.keys())
    edge_cols = [f"{u}->{v}" for u, v in edges]
    idx = P_load_pu.index
    t_count = len(idx)
    n_nodes = len(nodes)
    n_edges = len(edges)

    node_pos = {node: pos for pos, node in enumerate(nodes)}
    edge_parent_idx = np.array([node_pos[u] for u, _ in edges], dtype=int)
    edge_child_idx = np.array([node_pos[v] for _, v in edges], dtype=int)
    z_arr = np.array([z_pu[edge] for edge in edges], dtype=complex)
    root_idx = node_pos[int(net.root_bus)]
    in_edge = np.full(n_nodes, -1, dtype=int)
    child_edges: list[list[int]] = [[] for _ in range(n_nodes)]
    for edge_idx, (parent_idx, child_idx) in enumerate(zip(edge_parent_idx, edge_child_idx)):
        in_edge[child_idx] = edge_idx
        child_edges[parent_idx].append(edge_idx)

    p_arr = np.zeros((t_count, n_nodes), dtype=float)
    q_arr = np.zeros((t_count, n_nodes), dtype=float)
    for col in P_load_pu.columns:
        bus = int(col)
        if bus in node_pos:
            p_arr[:, node_pos[bus]] = P_load_pu[col].to_numpy(dtype=float)
            q_arr[:, node_pos[bus]] = Q_load_pu[col].to_numpy(dtype=float)

    if isinstance(v_root, pd.Series):
        slack = v_root.loc[idx].to_numpy(dtype=complex)
    elif isinstance(v_root, np.ndarray):
        if len(v_root) != t_count:
            raise ValueError("v_root array length must match the time-series length")
        slack = np.asarray(v_root, dtype=complex)
    else:
        slack = np.full(t_count, complex(v_root), dtype=complex)

    tol = _validate_solver_inputs(max_iter, tol, P_load_pu, Q_load_pu, slack)
    voltage = np.repeat(slack[:, None], n_nodes, axis=1)
    current = np.zeros((t_count, n_edges), dtype=complex)
    s_load = p_arr + 1j * q_arr
    converged_arr = np.zeros(t_count, dtype=bool)
    iterations_arr = np.zeros(t_count, dtype=int)
    max_error_arr = np.full(t_count, np.inf, dtype=float)
    active = np.ones(t_count, dtype=bool)

    for iteration in range(1, max_iter + 1):
        old_voltage = voltage.copy()
        next_current = current.copy()
        for node_idx in range(n_nodes - 1, -1, -1):
            if np.any(np.abs(voltage[active, node_idx]) < 1e-8):
                raise FloatingPointError(f"voltage collapsed near zero at bus {nodes[node_idx]}")
            total_current = np.conj(s_load[:, node_idx] / voltage[:, node_idx])
            for edge_idx in child_edges[node_idx]:
                total_current += next_current[:, edge_idx]
            edge_idx = in_edge[node_idx]
            if edge_idx >= 0:
                next_current[:, edge_idx] = total_current

        next_voltage = voltage.copy()
        next_voltage[:, root_idx] = slack
        for edge_idx in range(n_edges):
            next_voltage[:, edge_child_idx[edge_idx]] = (
                next_voltage[:, edge_parent_idx[edge_idx]]
                - z_arr[edge_idx] * next_current[:, edge_idx]
            )

        errors = np.max(np.abs(next_voltage - old_voltage), axis=1)
        voltage[active] = next_voltage[active]
        current[active] = next_current[active]
        max_error_arr[active] = errors[active]
        newly_converged = active & (errors <= tol)
        iterations_arr[newly_converged] = iteration
        converged_arr[newly_converged] = True
        active &= ~newly_converged
        if not np.any(active):
            break

    iterations_arr[active] = max_iter
    s_from = voltage[:, edge_parent_idx] * np.conj(current)
    s_to = voltage[:, edge_child_idx] * np.conj(current)
    loss = s_from - s_to
    return {
        "V_bus_mag": pd.DataFrame(np.abs(voltage), index=idx, columns=nodes),
        "V_bus_angle_deg": pd.DataFrame(np.degrees(np.angle(voltage)), index=idx, columns=nodes),
        "I_branch_mag": pd.DataFrame(np.abs(current), index=idx, columns=edge_cols),
        "branch_p_from_pu": pd.DataFrame(s_from.real, index=idx, columns=edge_cols),
        "branch_q_from_pu": pd.DataFrame(s_from.imag, index=idx, columns=edge_cols),
        "branch_p_to_pu": pd.DataFrame(s_to.real, index=idx, columns=edge_cols),
        "branch_q_to_pu": pd.DataFrame(s_to.imag, index=idx, columns=edge_cols),
        "branch_loss_p_pu": pd.DataFrame(loss.real, index=idx, columns=edge_cols),
        "branch_loss_q_pu": pd.DataFrame(loss.imag, index=idx, columns=edge_cols),
        "converged": pd.Series(converged_arr, index=idx, name="converged"),
        "iterations": pd.Series(iterations_arr, index=idx, name="iterations"),
        "max_voltage_error": pd.Series(max_error_arr, index=idx, name="max_voltage_error"),
        "metadata": {
            "solver": "radial_backward_forward_sweep",
            "root_bus": int(net.root_bus),
            "root_reference": "slack phasor, angle 0 unless v_root is complex",
            "load_convention": "positive P/Q are consumptions in per unit on base_mva",
            "branch_orientation": "away from root",
            "branch_edges": [{"label": col, "from_bus": int(u), "to_bus": int(v)} for (u, v), col in zip(edges, edge_cols)],
        },
    }
