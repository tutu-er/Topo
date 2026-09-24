"""Diagnostics for rooted and terminal-local latent-tree recovery."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class RootPlacement:
    """Best weighted-tree location inferred from root-to-terminal depths."""

    kind: str
    node: int | None
    edge: tuple[int, int] | None
    offset_from_edge_start: float
    rmse: float
    max_abs_error: float


def _weighted_tree(edges: list[tuple] | tuple[tuple, ...]) -> nx.Graph:
    graph = nx.Graph()
    for edge in edges:
        length = float(edge[2]) if len(edge) >= 3 else 1.0
        graph.add_edge(int(edge[0]), int(edge[1]), weight=max(length, 0.0))
    if graph.number_of_nodes() and not nx.is_tree(graph):
        raise ValueError("edges must form a tree")
    return graph


def locate_root_from_terminal_depths(
    edges: list[tuple] | tuple[tuple, ...],
    terminals: list[int] | tuple[int, ...],
    terminal_depths: np.ndarray,
) -> RootPlacement:
    """Locate the root on a weighted unrooted tree using root-terminal depths."""

    graph = _weighted_tree(edges)
    labels = [int(node) for node in terminals]
    depths = np.asarray(terminal_depths, dtype=float)
    if depths.shape != (len(labels),):
        raise ValueError("terminal_depths shape must match terminals")
    if not np.all(np.isfinite(depths)):
        raise ValueError("terminal_depths must be finite")

    best: RootPlacement | None = None

    def consider(kind: str, node: int | None, edge: tuple[int, int] | None, offset: float, predicted: np.ndarray) -> None:
        nonlocal best
        errors = depths - predicted
        candidate = RootPlacement(
            kind=kind,
            node=node,
            edge=edge,
            offset_from_edge_start=float(offset),
            rmse=float(np.sqrt(np.mean(errors**2))),
            max_abs_error=float(np.max(np.abs(errors))),
        )
        if best is None or candidate.rmse < best.rmse - 1e-12:
            best = candidate

    for node in graph.nodes():
        predicted = np.asarray([nx.shortest_path_length(graph, node, terminal, weight="weight") for terminal in labels])
        consider("node", int(node), None, 0.0, predicted)

    for left, right, data in list(graph.edges(data=True)):
        length = float(data["weight"])
        if length <= 1e-12:
            continue
        graph.remove_edge(left, right)
        left_component = nx.node_connected_component(graph, left)
        base = []
        slope = []
        for terminal in labels:
            if terminal in left_component:
                base.append(nx.shortest_path_length(graph, left, terminal, weight="weight"))
                slope.append(1.0)
            else:
                base.append(nx.shortest_path_length(graph, right, terminal, weight="weight") + length)
                slope.append(-1.0)
        graph.add_edge(left, right, weight=length)
        base_array = np.asarray(base, dtype=float)
        slope_array = np.asarray(slope, dtype=float)
        offset = float(np.clip(np.mean(slope_array * (depths - base_array)), 0.0, length))
        predicted = base_array + slope_array * offset
        if offset <= 1e-10:
            continue
        if length - offset <= 1e-10:
            continue
        consider("edge", None, (int(left), int(right)), offset, predicted)

    if best is None:
        raise ValueError("cannot locate a root on an empty tree")
    return best


def terminal_partition_at_node(
    edges: list[tuple] | tuple[tuple, ...],
    node: int,
    terminals: list[int] | tuple[int, ...],
) -> frozenset[frozenset[int]]:
    """Return terminal branch groups incident to one tree node."""

    graph = _weighted_tree(edges)
    terminal_set = {int(terminal) for terminal in terminals}
    if node not in graph:
        return frozenset()
    graph.remove_node(node)
    groups = []
    for component in nx.connected_components(graph):
        group = frozenset(terminal_set.intersection(component))
        if group:
            groups.append(group)
    return frozenset(groups)


def root_partition_is_correct(
    placement: RootPlacement,
    predicted_edges: list[tuple] | tuple[tuple, ...],
    true_edges: list[tuple] | tuple[tuple, ...],
    true_root: int,
    terminals: list[int] | tuple[int, ...],
) -> bool:
    """Check whether a located node induces the true root terminal partition."""

    if placement.kind == "node" and placement.node is not None:
        predicted = terminal_partition_at_node(predicted_edges, placement.node, terminals)
    elif placement.kind == "edge" and placement.edge is not None:
        graph = _weighted_tree(predicted_edges)
        graph.remove_edge(*placement.edge)
        terminal_set = {int(terminal) for terminal in terminals}
        predicted = frozenset(
            frozenset(terminal_set.intersection(component))
            for component in nx.connected_components(graph)
            if terminal_set.intersection(component)
        )
    else:
        return False
    truth = terminal_partition_at_node(true_edges, true_root, terminals)
    return predicted == truth


def terminal_sibling_pairs(
    edges: list[tuple] | tuple[tuple, ...],
    terminals: list[int] | tuple[int, ...],
) -> set[tuple[int, int]]:
    """Return terminal pairs attached to the same nonterminal node."""

    graph = _weighted_tree(edges)
    terminal_set = {int(node) for node in terminals}
    pairs = set()
    for node in graph.nodes():
        if node in terminal_set:
            continue
        adjacent = sorted(neighbor for neighbor in graph.neighbors(node) if neighbor in terminal_set)
        pairs.update((int(left), int(right)) for left, right in combinations(adjacent, 2))
    return pairs


def terminal_limb_lengths(
    edges: list[tuple] | tuple[tuple, ...],
    terminals: list[int] | tuple[int, ...],
) -> dict[int, float]:
    """Return the reconstructed pendant-edge length of each terminal."""

    graph = _weighted_tree(edges)
    result = {}
    for terminal in terminals:
        node = int(terminal)
        if node in graph and graph.degree(node) == 1:
            neighbor = next(iter(graph.neighbors(node)))
            result[node] = float(graph.edges[node, neighbor]["weight"])
    return result
