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
    """Build R/X with observed buses as rows and injection buses as columns.

    For terminals ``i,j``, each matrix entry is the sum of per-unit resistance
    or reactance along the common root-to-node path. ``load_positive`` uses
    ``Delta V = -R Delta P_load - X Delta Q_load``; ``net_injection`` flips the
    sign convention in the caller's regression interpretation, not the matrix.
    Omitted injection buses default to the observed buses, giving square R/X.
    """

    if convention not in {"load_positive", "net_injection"}:
        raise ValueError(f"unknown convention: {convention}")
    if voltage_model not in {"magnitude", "squared-voltage"}:
        raise ValueError(f"unknown voltage_model: {voltage_model}")

    if observed_terminals is None:
        observed_terminals = (
            net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"].astype(int).tolist()
        )
    observed = list(map(int, observed_terminals))
    injections = observed if injection_terminals is None else list(map(int, injection_terminals))
    graph = net.to_networkx_graph()
    path_edges = {}
    for node in dict.fromkeys(observed + injections):
        path = nx.shortest_path(graph, net.root_bus, node)
        path_edges[node] = {(min(u, v), max(u, v)) for u, v in zip(path[:-1], path[1:])}
    impedance = {
        (min(u, v), max(u, v)): (ohm_to_pu(net, data["r_ohm"]), ohm_to_pu(net, data["x_ohm"]))
        for u, v, data in graph.edges(data=True)
    }
    r_mat = np.zeros((len(observed), len(injections)), dtype=float)
    x_mat = np.zeros_like(r_mat)
    for a, i in enumerate(observed):
        for b, j in enumerate(injections):
            common = path_edges[i] & path_edges[j]
            r_mat[a, b] = sum(impedance[e][0] for e in common)
            x_mat[a, b] = sum(impedance[e][1] for e in common)
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
