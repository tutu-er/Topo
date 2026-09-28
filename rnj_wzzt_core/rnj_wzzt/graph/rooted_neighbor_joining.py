"""Rooted Neighbor-Joining from root-to-LCA shared-path scores.

The RNJ recursion ranks terminal pairs by their shared root-path length. For
reduced power-system sensitivity matrices this score is supplied directly by
the RX75 normalized weighted combination of ``R_ij`` and ``X_ij``. Generic
distance adapters live in the separate research experiments package.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


WeightedEdge = tuple[int, int, float]


@dataclass(frozen=True)
class RootedTreeResult:
    """A rooted latent tree whose root keeps the supplied physical label."""

    root: int
    terminals: tuple[int, ...]
    edges: tuple[WeightedEdge, ...]
    group_tolerance: float



def _contract_zero_internal_edges(
    edges: list[WeightedEdge],
    root: int,
    terminals: list[int],
    atol: float = 1e-10,
) -> tuple[WeightedEdge, ...]:
    terminal_set = set(terminals)
    parent: dict[int, int] = {}

    def find(node: int) -> int:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if root in {left_root, right_root}:
            parent[left_root] = root
            parent[right_root] = root
        else:
            parent[min(left_root, right_root)] = max(left_root, right_root)

    for left, right, length in edges:
        if length <= atol and left not in terminal_set and right not in terminal_set:
            union(left, right)
    compact: dict[tuple[int, int], float] = {}
    for left, right, length in edges:
        mapped_left = find(left) if left not in terminal_set else left
        mapped_right = find(right) if right not in terminal_set else right
        if mapped_left == mapped_right:
            continue
        key = tuple(sorted((mapped_left, mapped_right)))
        compact[key] = min(compact.get(key, float("inf")), float(length))
    return tuple((left, right, length) for (left, right), length in sorted(compact.items()))


def rooted_neighbor_joining(
    shared_path_matrix: np.ndarray,
    root_depths: np.ndarray,
    terminals: list[int],
    root: int,
    group_tolerance: float = 0.0,
) -> RootedTreeResult:
    """Reconstruct a rooted general tree from shared-path scores.

    ``group_tolerance`` is the allowed shared-path difference when attaching
    more than two active nodes to one parent. It is a numerical analogue of the
    minimum-edge-length threshold in general-tree RNJ, not the same theoretical
    threshold unless it is calibrated from a known edge-length lower bound.
    """

    labels = [int(node) for node in terminals]
    root = int(root)
    if len(set(labels)) != len(labels):
        raise ValueError("terminals must contain unique labels")
    if root in labels:
        raise ValueError("root must not be a terminal")
    shared = np.asarray(shared_path_matrix, dtype=float)
    depths_array = np.asarray(root_depths, dtype=float)
    if shared.shape != (len(labels), len(labels)):
        raise ValueError("shared_path_matrix shape must match terminals")
    if depths_array.shape != (len(labels),):
        raise ValueError("root_depths shape must match terminals")
    if not np.all(np.isfinite(shared)) or not np.all(np.isfinite(depths_array)):
        raise ValueError("RNJ inputs must be finite")
    if not np.isfinite(group_tolerance) or group_tolerance < 0.0:
        raise ValueError("group_tolerance must be finite and nonnegative")

    shared = np.maximum(0.0, 0.5 * (shared + shared.T))
    depths: dict[int, float] = {node: max(0.0, float(depths_array[index])) for index, node in enumerate(labels)}
    values: dict[tuple[int, int], float] = {}
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            if i < j:
                values[(min(left, right), max(left, right))] = float(
                    np.clip(shared[i, j], 0.0, min(depths[left], depths[right]))
                )

    def get(left: int, right: int) -> float:
        if left == right:
            return depths[left]
        return values[(min(left, right), max(left, right))]

    active = list(labels)
    edges: list[WeightedEdge] = []
    next_hidden = min(-1, min([root, *labels]) - 1)
    while len(active) > 1:
        left, right = max(
            combinations(active, 2),
            key=lambda pair: (get(*pair), tuple(-value for value in sorted(pair))),
        )
        if get(left, right) <= group_tolerance:
            edges.extend((int(root), node, max(0.0, depths[node])) for node in active)
            active = []
            break
        # Every stored score is clipped to both endpoint depths on ingestion or
        # update, so the selected pair score is already the parent depth.
        parent_depth = get(left, right)
        family = [left, right]
        for node in active:
            if node in family:
                continue
            support = min(get(left, node), get(right, node))
            if parent_depth - support <= group_tolerance:
                family.append(node)

        hidden = next_hidden
        next_hidden -= 1
        depths[hidden] = max(0.0, parent_depth)
        for child in family:
            edges.append((hidden, child, max(0.0, depths[child] - depths[hidden])))
        remaining = [node for node in active if node not in family]
        for node in remaining:
            estimate = float(np.median([get(member, node) for member in family]))
            values[(min(hidden, node), max(hidden, node))] = float(
                np.clip(estimate, 0.0, min(depths[hidden], depths[node]))
            )
        active = remaining + [hidden]

    if active:
        edges.append((int(root), active[0], max(0.0, depths[active[0]])))
    return RootedTreeResult(
        root=int(root),
        terminals=tuple(labels),
        edges=_contract_zero_internal_edges(edges, int(root), labels),
        group_tolerance=float(group_tolerance),
    )
