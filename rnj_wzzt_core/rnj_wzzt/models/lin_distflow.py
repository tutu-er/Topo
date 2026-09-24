"""Reduced LinDistFlow sensitivity matrices for terminal buses."""

from __future__ import annotations

import networkx as nx
import numpy as np

from rnj_wzzt.models.network import TerminalizedNetwork


def ohm_to_pu(net: TerminalizedNetwork, value_ohm: float) -> float:
    """Convert Ohm to per-unit impedance using ``Z_base = kV^2 / MVA``."""

    z_base = (net.base_kv**2) / net.base_mva
    return float(value_ohm) / z_base


def build_reduced_sensitivity_matrices(
    net: TerminalizedNetwork,
    observed_terminals: list[int] | None = None,
    injection_terminals: list[int] | None = None,
    convention: str = "load_positive",
    voltage_model: str = "magnitude",
) -> tuple[np.ndarray, np.ndarray]:
    """Build reduced R and X matrices over observed terminal buses.

    For terminals ``i,j``, each matrix entry is the sum of per-unit resistance
    or reactance along the common root-to-node path. ``load_positive`` uses
    ``Delta V = -R Delta P_load - X Delta Q_load``; ``net_injection`` flips the
    sign convention in the caller's regression interpretation, not the matrix.
    """

    del injection_terminals
    if convention not in {"load_positive", "net_injection"}:
        raise ValueError(f"unknown convention: {convention}")
    if voltage_model not in {"magnitude", "squared-voltage"}:
        raise ValueError(f"unknown voltage_model: {voltage_model}")

    if observed_terminals is None:
        observed_terminals = (
            net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"].astype(int).tolist()
        )
    terminals = list(map(int, observed_terminals))
    graph = net.to_networkx_graph()
    paths = {node: nx.shortest_path(graph, net.root_bus, node) for node in terminals}
    edge_attrs = {(min(u, v), max(u, v)): data for u, v, data in graph.edges(data=True)}
    n = len(terminals)
    r_mat = np.zeros((n, n), dtype=float)
    x_mat = np.zeros((n, n), dtype=float)
    for a, i in enumerate(terminals):
        edges_i = {(min(u, v), max(u, v)) for u, v in zip(paths[i][:-1], paths[i][1:])}
        for b, j in enumerate(terminals):
            edges_j = {(min(u, v), max(u, v)) for u, v in zip(paths[j][:-1], paths[j][1:])}
            common = edges_i & edges_j
            r_mat[a, b] = sum(ohm_to_pu(net, edge_attrs[e]["r_ohm"]) for e in common)
            x_mat[a, b] = sum(ohm_to_pu(net, edge_attrs[e]["x_ohm"]) for e in common)
    if voltage_model == "squared-voltage":
        r_mat = 2.0 * r_mat
        x_mat = 2.0 * x_mat
    return r_mat, x_mat


def impedance_distance_from_reduced_R(R: np.ndarray) -> np.ndarray:
    """Convert a reduced R matrix to additive terminal-terminal distances."""

    diag = np.diag(R)
    return diag[:, None] + diag[None, :] - 2.0 * R


def impedance_distance_from_reduced_X(X: np.ndarray) -> np.ndarray:
    """Convert a reduced X matrix to additive terminal-terminal distances."""

    return impedance_distance_from_reduced_R(X)
