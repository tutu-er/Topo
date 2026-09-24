"""CLGrouping baseline for latent-tree reconstruction."""

from __future__ import annotations

import networkx as nx
import numpy as np

from terminal_case33.graph.latent_tree import LatentTreeResult, recursive_grouping


def cl_grouping(
    distance_matrix: np.ndarray,
    observed_nodes: list[int],
    tolerance: float = 0.03,
) -> LatentTreeResult:
    """Run the Choi et al. Chow-Liu grouping pattern with RG locally.

    A minimum-distance tree supplies global neighborhoods.  Recursive grouping
    then replaces each rooted parent/children star by a local latent subtree.
    This is the distance-tree specialization of CLGrouping; unlike using the
    initial minimum-distance tree as the output, hidden nodes are explicitly
    introduced by the local RG calls.
    """

    labels = [int(node) for node in observed_nodes]
    distance = np.asarray(distance_matrix, dtype=float)
    if distance.shape != (len(labels), len(labels)):
        raise ValueError("distance_matrix shape must match observed_nodes")
    if len(labels) < 2:
        return LatentTreeResult("cl_grouping", tuple(labels), tuple())
    distance = np.maximum(0.0, 0.5 * (distance + distance.T))
    np.fill_diagonal(distance, 0.0)
    index = {node: position for position, node in enumerate(labels)}
    complete = nx.Graph()
    complete.add_nodes_from(labels)
    for i, left in enumerate(labels):
        for j in range(i + 1, len(labels)):
            complete.add_edge(left, labels[j], weight=float(distance[i, j]))
    guide = nx.minimum_spanning_tree(complete, weight="weight", algorithm="kruskal")
    root = min(labels, key=lambda node: (guide.degree(node) == 1, -guide.degree(node), node))
    parent = {root: None}
    children = {node: [] for node in labels}
    for left, right in nx.bfs_edges(guide, root):
        parent[right] = left
        children[left].append(right)

    output: list[tuple[int, int, float]] = []
    next_hidden = -1
    for node in nx.dfs_postorder_nodes(guide, root):
        family = [node, *children[node]]
        if len(family) == 1:
            continue
        if len(family) == 2:
            child = family[1]
            output.append((node, child, float(distance[index[node], index[child]])))
            continue
        local_index = [index[item] for item in family]
        local_distance = distance[np.ix_(local_index, local_index)]
        local = recursive_grouping(local_distance, family, tolerance=tolerance)
        remap: dict[int, int] = {}
        for left, right, _ in local.edges:
            for endpoint in (left, right):
                if endpoint < 0 and endpoint not in remap:
                    remap[endpoint] = next_hidden
                    next_hidden -= 1
        output.extend((remap.get(left, left), remap.get(right, right), length) for left, right, length in local.edges)

    graph = nx.Graph()
    graph.add_weighted_edges_from(output)
    if not nx.is_tree(graph) or not set(labels).issubset(graph.nodes):
        # Local replacement can be degenerate for a highly non-additive input;
        # global RG remains a deterministic complete fallback.
        fallback = recursive_grouping(distance, labels, tolerance=tolerance)
        return LatentTreeResult("cl_grouping_fallback_rg", fallback.terminals, fallback.edges, fallback.forced_merges)
    return LatentTreeResult("cl_grouping", tuple(labels), tuple(output))
