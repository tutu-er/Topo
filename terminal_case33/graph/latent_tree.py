"""Latent-tree reconstruction from terminal additive distances.

Hidden-node labels are algorithm-local. Topology comparisons therefore use
terminal splits induced by internal edges, not raw hidden-node edge tuples.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np


WeightedEdge = tuple[int, int, float]
TerminalSplit = frozenset[int]


@dataclass(frozen=True)
class LatentTreeResult:
    """A reconstructed unrooted latent tree."""

    method: str
    terminals: tuple[int, ...]
    edges: tuple[WeightedEdge, ...]
    forced_merges: int = 0


def _validate_distance(distance_matrix: np.ndarray, terminals: list[int]) -> np.ndarray:
    distance = np.asarray(distance_matrix, dtype=float)
    if distance.shape != (len(terminals), len(terminals)):
        raise ValueError("distance_matrix shape must match terminals")
    if not np.all(np.isfinite(distance)):
        raise ValueError("distance_matrix must be finite")
    distance = np.maximum(0.0, 0.5 * (distance + distance.T))
    np.fill_diagonal(distance, 0.0)
    return distance


def _key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _contract_zero_hidden_edges(
    edges: list[WeightedEdge],
    terminals: list[int],
    atol: float = 1e-10,
) -> tuple[WeightedEdge, ...]:
    """Suppress zero-length hidden-hidden edges created at multifurcations."""

    terminal_set = set(terminals)
    parent: dict[int, int] = {}

    def find(node: int) -> int:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[min(left_root, right_root)] = max(left_root, right_root)

    for left, right, length in edges:
        if left not in terminal_set and right not in terminal_set and length <= atol:
            union(left, right)
    compact: dict[tuple[int, int], float] = {}
    for left, right, length in edges:
        mapped_left = find(left) if left not in terminal_set else left
        mapped_right = find(right) if right not in terminal_set else right
        if mapped_left == mapped_right:
            continue
        key = _key(mapped_left, mapped_right)
        compact[key] = min(compact.get(key, float("inf")), float(length))
    return tuple((left, right, length) for (left, right), length in sorted(compact.items()))


def neighbor_joining(distance_matrix: np.ndarray, terminals: list[int]) -> LatentTreeResult:
    """Recover an unrooted binary latent tree with neighbor joining."""

    labels = [int(node) for node in terminals]
    distance = _validate_distance(distance_matrix, labels)
    if len(labels) < 2:
        return LatentTreeResult("neighbor_joining", tuple(labels), tuple())
    active = list(labels)
    values = {
        _key(left, right): float(distance[i, j])
        for i, left in enumerate(labels)
        for j, right in enumerate(labels)
        if i < j
    }

    def get(left: int, right: int) -> float:
        return 0.0 if left == right else values[_key(left, right)]

    edges: list[WeightedEdge] = []
    next_hidden = -1
    while len(active) > 2:
        count = len(active)
        row_sum = {node: sum(get(node, other) for other in active if other != node) for node in active}
        left, right = min(
            combinations(active, 2),
            key=lambda pair: ((count - 2) * get(*pair) - row_sum[pair[0]] - row_sum[pair[1]], _key(*pair)),
        )
        separation = get(left, right)
        left_length = 0.5 * separation + (row_sum[left] - row_sum[right]) / (2.0 * (count - 2))
        hidden = next_hidden
        next_hidden -= 1
        edges.append((hidden, left, max(0.0, float(left_length))))
        edges.append((hidden, right, max(0.0, float(separation - left_length))))
        remaining = [node for node in active if node not in {left, right}]
        for node in remaining:
            values[_key(hidden, node)] = max(0.0, 0.5 * (get(left, node) + get(right, node) - separation))
        active = remaining + [hidden]
    edges.append((active[0], active[1], max(0.0, float(get(active[0], active[1])))))
    return LatentTreeResult("neighbor_joining", tuple(labels), _contract_zero_hidden_edges(edges, labels))


def recursive_grouping(
    distance_matrix: np.ndarray,
    terminals: list[int],
    tolerance: float = 0.03,
) -> LatentTreeResult:
    """Recover a latent tree by robust recursive grouping.

    For active nodes i and j, RG examines ``d(i,k)-d(j,k)`` over all other
    active k. A nearly constant value identifies a parent-child or sibling
    family. If no family meets ``tolerance``, the least-inconsistent pair is
    merged so a noisy non-additive matrix still yields a complete tree.
    """

    labels = [int(node) for node in terminals]
    distance = _validate_distance(distance_matrix, labels)
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    if len(labels) < 2:
        return LatentTreeResult("recursive_grouping", tuple(labels), tuple())
    active = list(labels)
    values = {
        _key(left, right): float(distance[i, j])
        for i, left in enumerate(labels)
        for j, right in enumerate(labels)
        if i < j
    }

    def get(left: int, right: int) -> float:
        return 0.0 if left == right else values[_key(left, right)]

    def relation(left: int, right: int) -> tuple[float, float, str]:
        others = [node for node in active if node not in {left, right}]
        if not others:
            return 0.0, 0.0, "siblings"
        phi = np.asarray([get(left, node) - get(right, node) for node in others])
        mean = float(np.median(phi))
        scale = max(
            get(left, right),
            float(np.median([get(left, node) + get(right, node) for node in others])),
            1e-12,
        )
        residual = float(np.max(np.abs(phi - mean)) / scale)
        endpoint = tolerance * scale
        if abs(mean + get(left, right)) <= endpoint:
            kind = "left_parent"
        elif abs(mean - get(left, right)) <= endpoint:
            kind = "right_parent"
        else:
            kind = "siblings"
        return residual, mean, kind

    def star_edges(nodes: list[int], hidden: int) -> list[WeightedEdge]:
        result = []
        for node in nodes:
            others = [item for item in nodes if item != node]
            estimates = [0.5 * (get(node, a) + get(node, b) - get(a, b)) for a, b in combinations(others, 2)]
            result.append((hidden, node, max(0.0, float(np.median(estimates)) if estimates else 0.0)))
        return result

    edges: list[WeightedEdge] = []
    forced_merges = 0
    next_hidden = -1
    while len(active) > 1:
        if len(active) == 2:
            edges.append((active[0], active[1], max(0.0, get(active[0], active[1]))))
            break
        if len(active) == 3:
            edges.extend(star_edges(active, next_hidden))
            break

        pair, pair_relation = min(
            ((pair, relation(*pair)) for pair in combinations(active, 2)),
            key=lambda item: (item[1][0], _key(*item[0])),
        )
        left, right = pair
        accepted = pair_relation[0] <= tolerance
        family = [left, right]
        if accepted:
            family.extend(
                node
                for node in active
                if node not in family
                and relation(left, node)[0] <= tolerance
                and relation(right, node)[0] <= tolerance
            )
        else:
            forced_merges += 1

        parent = None
        if accepted:
            for candidate in family:
                if all(relation(candidate, child)[2] == "left_parent" for child in family if child != candidate):
                    parent = candidate
                    break
        remaining = [node for node in active if node not in family]
        if parent is not None:
            edges.extend((parent, child, max(0.0, get(parent, child))) for child in family if child != parent)
            active = remaining + [parent]
            continue

        hidden = next_hidden
        next_hidden -= 1
        if len(family) >= 3:
            family_edges = star_edges(family, hidden)
            limb = {node: length for _, node, length in family_edges}
        else:
            _, phi, _ = pair_relation
            separation = get(left, right)
            left_limb = float(np.clip(0.5 * (separation + phi), 0.0, separation))
            limb = {left: left_limb, right: separation - left_limb}
            family_edges = [(hidden, node, length) for node, length in limb.items()]
        edges.extend(family_edges)
        for node in remaining:
            values[_key(hidden, node)] = max(0.0, float(np.median([get(member, node) - limb[member] for member in family])))
        active = remaining + [hidden]

    return LatentTreeResult(
        "recursive_grouping", tuple(labels), _contract_zero_hidden_edges(edges, labels), forced_merges
    )


def terminal_splits(edges: list[tuple] | tuple[tuple, ...], terminals: list[int] | tuple[int, ...]) -> set[TerminalSplit]:
    """Return canonical nontrivial terminal splits induced by tree edges."""

    terminal_set = {int(node) for node in terminals}
    graph = nx.Graph()
    graph.add_edges_from((int(edge[0]), int(edge[1])) for edge in edges)
    splits: set[TerminalSplit] = set()
    for left, right in list(graph.edges()):
        graph.remove_edge(left, right)
        components = list(nx.connected_components(graph))
        graph.add_edge(left, right)
        if len(components) != 2:
            continue
        side = terminal_set.intersection(components[0])
        other = terminal_set - side
        if min(len(side), len(other)) <= 1:
            continue
        if len(side) > len(other) or (len(side) == len(other) and tuple(sorted(side)) > tuple(sorted(other))):
            side = other
        splits.add(frozenset(side))
    return splits


def terminal_sibling_groups(edges: list[tuple] | tuple[tuple, ...], terminals: list[int] | tuple[int, ...]) -> set[frozenset[int]]:
    """Return terminal groups sharing one adjacent latent node."""

    terminal_set = {int(node) for node in terminals}
    graph = nx.Graph()
    graph.add_edges_from((int(edge[0]), int(edge[1])) for edge in edges)
    groups = set()
    for node in graph.nodes():
        children = frozenset(int(neighbor) for neighbor in graph.neighbors(node) if neighbor in terminal_set)
        if node not in terminal_set and len(children) >= 2:
            groups.add(children)
    return groups
