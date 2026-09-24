"""Probabilistic spanning trees via the matrix-tree theorem."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class TreePosterior:
    """Spanning-tree posterior summary."""

    map_tree: list[tuple[int, int]]
    edge_marginals: pd.DataFrame
    logZ: float
    sum_marginals: float


def build_edge_scores_from_distances(
    terminals: list[int],
    dR: np.ndarray,
    dX: np.ndarray,
    beta: float = 1.0,
    uncertainty: np.ndarray | None = None,
    candidate_edges: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Convert R/X distances to positive edge weights."""

    idx = {int(bus): k for k, bus in enumerate(terminals)}
    edges = candidate_edges
    if edges is None:
        edges = [(int(u), int(v)) for i, u in enumerate(terminals) for v in terminals[i + 1 :]]
    rows = []
    scale = float(np.nanmedian(dR + dX)) or 1.0
    for u, v in edges:
        if int(u) not in idx or int(v) not in idx:
            continue
        i, j = idx[int(u)], idx[int(v)]
        distance = float(max(dR[i, j] + dX[i, j], 0.0))
        unc = 0.0 if uncertainty is None else float(uncertainty[i, j])
        weight = float(np.exp(-beta * distance / scale) / (1.0 + unc))
        rows.append({"u": int(u), "v": int(v), "distance": distance, "weight": weight})
    return pd.DataFrame(rows)


def matrix_tree_edge_marginals(
    nodes: list[int],
    candidate_edges: list[tuple[int, int]],
    weights: dict[tuple[int, int], float],
    root: int | None = None,
) -> TreePosterior:
    """Compute MAP tree, edge marginals, and log-partition over spanning trees."""

    del root
    nodes = list(map(int, nodes))
    idx = {node: k for k, node in enumerate(nodes)}
    n = len(nodes)
    L = np.zeros((n, n), dtype=float)
    clean_weights: dict[tuple[int, int], float] = {}
    for u, v in candidate_edges:
        u, v = int(u), int(v)
        key = (min(u, v), max(u, v))
        w = float(weights.get((u, v), weights.get((v, u), weights.get(key, 1.0))))
        clean_weights[key] = w
        i, j = idx[u], idx[v]
        L[i, i] += w
        L[j, j] += w
        L[i, j] -= w
        L[j, i] -= w
    cofactor = L[1:, 1:] if n > 1 else np.ones((1, 1))
    sign, logdet = np.linalg.slogdet(cofactor)
    logZ = float(logdet if sign > 0 else -np.inf)
    L_pinv = np.linalg.pinv(L)
    rows = []
    for u, v in sorted(clean_weights):
        i, j = idx[u], idx[v]
        eff = L_pinv[i, i] + L_pinv[j, j] - 2.0 * L_pinv[i, j]
        rows.append({"u": u, "v": v, "weight": clean_weights[(u, v)], "edge_marginal": clean_weights[(u, v)] * eff})
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for (u, v), w in clean_weights.items():
        graph.add_edge(u, v, weight=w)
    tree = nx.maximum_spanning_tree(graph, weight="weight")
    map_edges = [(int(u), int(v)) for u, v in tree.edges()]
    marginals = pd.DataFrame(rows)
    return TreePosterior(
        map_tree=map_edges,
        edge_marginals=marginals,
        logZ=logZ,
        sum_marginals=float(marginals["edge_marginal"].sum()) if not marginals.empty else 0.0,
    )

