"""Interval-meter placement stage from Pengwah et al. (2024)."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


WeightedEdge = tuple[int, int, float]


@dataclass(frozen=True)
class IntervalPlacement:
    """Diagnostic record for one inserted interval meter."""

    interval_node: int
    closest_smart_meter: int
    common_path_depth: float
    mode: str
    attachment_node: int


@dataclass(frozen=True)
class PartialMeterTreeResult:
    """Smart-meter reduced tree after interval-meter insertion."""

    root: int
    smart_meters: tuple[int, ...]
    interval_meters: tuple[int, ...]
    edges: tuple[WeightedEdge, ...]
    placements: tuple[IntervalPlacement, ...]


def _root_depths(graph: nx.Graph, root: int) -> dict[int, float]:
    return {
        int(node): float(length)
        for node, length in nx.single_source_dijkstra_path_length(graph, root, weight="weight").items()
    }


def insert_interval_meters(
    smart_tree_edges: list[WeightedEdge] | tuple[WeightedEdge, ...],
    root: int,
    smart_meters: list[int],
    interval_meters: list[int],
    R_SI: np.ndarray,
    equality_tolerance: float = 1e-6,
    service_edge_length: float | None = None,
) -> PartialMeterTreeResult:
    """Insert interval meters using their smart/interval common-path depths.

    For interval meter ``i``, ``max_s R_SI[s,i]`` estimates the depth of its
    most recent common ancestor with the closest smart meter.  The meter is
    attached to an existing node at that depth or a new hidden node inserted
    into the bracketing edge, exactly matching the two cases in Algorithm 1 of
    Pengwah et al. (2024).
    """

    smart = [int(node) for node in smart_meters]
    interval = [int(node) for node in interval_meters]
    cross = np.asarray(R_SI, dtype=float)
    if cross.shape != (len(smart), len(interval)):
        raise ValueError("R_SI shape must be (smart meters, interval meters)")
    graph = nx.Graph()
    graph.add_weighted_edges_from((int(left), int(right), float(length)) for left, right, length in smart_tree_edges)
    if not nx.is_tree(graph) or root not in graph or not set(smart).issubset(graph.nodes):
        raise ValueError("smart_tree_edges must form a rooted tree containing all smart meters")
    next_hidden = min([-1, *(int(node) for node in graph.nodes if int(node) < 0)]) - 1
    placements: list[IntervalPlacement] = []
    positive_edges = [float(data["weight"]) for _, _, data in graph.edges(data=True) if data["weight"] > 0.0]
    default_service = 10.0 * equality_tolerance
    if positive_edges:
        default_service = max(default_service, 0.1 * float(np.median(positive_edges)))
    service_length = default_service if service_edge_length is None else float(service_edge_length)

    for interval_index, interval_node in enumerate(interval):
        closest_index = int(np.argmax(cross[:, interval_index]))
        closest = smart[closest_index]
        target_depth = max(0.0, float(cross[closest_index, interval_index]))
        depths = _root_depths(graph, root)
        path = nx.shortest_path(graph, root, closest)
        exact = min(path, key=lambda node: abs(depths[node] - target_depth))
        if abs(depths[exact] - target_depth) <= equality_tolerance:
            graph.add_edge(exact, interval_node, weight=service_length)
            placements.append(IntervalPlacement(interval_node, closest, target_depth, "existing_node", int(exact)))
            continue

        bracket = None
        for parent, child in zip(path[:-1], path[1:], strict=True):
            if depths[parent] < target_depth < depths[child]:
                bracket = (parent, child)
                break
        if bracket is None:
            attachment = root if target_depth <= depths[root] else closest
            graph.add_edge(attachment, interval_node, weight=service_length)
            placements.append(IntervalPlacement(interval_node, closest, target_depth, "clamped_node", int(attachment)))
            continue
        parent, child = bracket
        old_length = float(graph.edges[parent, child]["weight"])
        hidden = next_hidden
        next_hidden -= 1
        upper = max(equality_tolerance, target_depth - depths[parent])
        lower = max(equality_tolerance, old_length - upper)
        graph.remove_edge(parent, child)
        graph.add_edge(parent, hidden, weight=upper)
        graph.add_edge(hidden, child, weight=lower)
        graph.add_edge(hidden, interval_node, weight=service_length)
        placements.append(IntervalPlacement(interval_node, closest, target_depth, "split_edge", hidden))
    if not nx.is_tree(graph):
        raise RuntimeError("interval insertion did not preserve a tree")
    edges = tuple((int(left), int(right), float(data["weight"])) for left, right, data in graph.edges(data=True))
    return PartialMeterTreeResult(root, tuple(smart), tuple(interval), edges, tuple(placements))
