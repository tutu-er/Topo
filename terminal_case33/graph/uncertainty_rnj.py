"""Uncertainty-aware rooted neighbor joining from distance replicates."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from terminal_case33.graph.rooted_neighbor_joining import (
    RootedTreeResult,
    _contract_zero_internal_edges,
)


@dataclass(frozen=True)
class UncertainRNJDecision:
    """One conservative grouping decision and its bootstrap diagnostics."""

    pair: tuple[int, int]
    family: tuple[int, ...]
    parent_depth: float
    pair_lower_bound: float
    pair_upper_bound: float
    maximum_family_gap: float


@dataclass(frozen=True)
class UncertainRNJResult:
    """Rooted tree reconstructed while propagating shared-path replicates."""

    tree: RootedTreeResult
    decisions: tuple[UncertainRNJDecision, ...]
    confidence_level: float
    replicate_count: int


def rooted_neighbor_joining_with_uncertainty(
    shared_path_samples: np.ndarray,
    root_depth_samples: np.ndarray,
    terminals: list[int],
    root: int,
    group_tolerance: float,
    confidence_level: float = 0.90,
    minimum_parent_depth: float = 0.0,
) -> UncertainRNJResult:
    """Run conservative RNJ using bootstrap distributions of shared paths.

    Pair selection maximizes a lower confidence bound rather than a point
    estimate. A third active node joins a family unless the lower confidence
    bound of its shared-path gap exceeds ``group_tolerance``. Ambiguous hidden
    edges are therefore contracted into a multifurcation instead of appearing
    as unsupported binary refinements.
    """

    samples = np.asarray(shared_path_samples, dtype=float)
    depth_samples = np.asarray(root_depth_samples, dtype=float)
    labels = [int(node) for node in terminals]
    if samples.ndim != 3 or samples.shape[1:] != (len(labels), len(labels)):
        raise ValueError("shared_path_samples must have shape (B, n, n)")
    if depth_samples.shape != (samples.shape[0], len(labels)):
        raise ValueError("root_depth_samples must have shape (B, n)")
    if samples.shape[0] < 2:
        raise ValueError("at least two uncertainty replicates are required")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0.5, 1)")
    if group_tolerance < 0.0 or minimum_parent_depth < 0.0:
        raise ValueError("RNJ thresholds must be nonnegative")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(depth_samples)):
        raise ValueError("uncertainty-aware RNJ inputs must be finite")

    lower_q = 0.5 * (1.0 - confidence_level)
    upper_q = 1.0 - lower_q
    depths: dict[int, np.ndarray] = {
        node: np.maximum(depth_samples[:, index], 0.0)
        for index, node in enumerate(labels)
    }
    values: dict[tuple[int, int], np.ndarray] = {}
    symmetrized = np.maximum(0.0, 0.5 * (samples + samples.transpose(0, 2, 1)))
    for i, left in enumerate(labels):
        for j in range(i + 1, len(labels)):
            right = labels[j]
            values[(min(left, right), max(left, right))] = np.minimum(
                symmetrized[:, i, j],
                np.minimum(depths[left], depths[right]),
            )

    def get(left: int, right: int) -> np.ndarray:
        if left == right:
            return depths[left]
        return values[(min(left, right), max(left, right))]

    def quantile(values_array: np.ndarray, probability: float) -> float:
        return float(np.quantile(values_array, probability))

    active = list(labels)
    edges: list[tuple[int, int, float]] = []
    decisions: list[UncertainRNJDecision] = []
    next_hidden = -1
    while len(active) > 1:
        left, right = max(
            combinations(active, 2),
            key=lambda pair: (
                quantile(get(*pair), lower_q),
                -max(pair),
                -min(pair),
            ),
        )
        pair_samples = get(left, right)
        pair_lower = quantile(pair_samples, lower_q)
        pair_upper = quantile(pair_samples, upper_q)
        if pair_lower <= minimum_parent_depth:
            edges.extend(
                (int(root), node, max(0.0, float(np.median(depths[node]))))
                for node in active
            )
            active = []
            break

        parent_samples = np.minimum(pair_samples, np.minimum(depths[left], depths[right]))
        family = [left, right]
        family_gaps: list[float] = []
        for node in active:
            if node in family:
                continue
            support = np.minimum(get(left, node), get(right, node))
            gap = parent_samples - support
            conservative_gap = quantile(gap, lower_q)
            if conservative_gap <= group_tolerance:
                family.append(node)
                family_gaps.append(conservative_gap)

        hidden = next_hidden
        next_hidden -= 1
        depths[hidden] = np.maximum(parent_samples, 0.0)
        parent_depth = float(np.median(depths[hidden]))
        for child in family:
            length = float(np.median(np.maximum(depths[child] - depths[hidden], 0.0)))
            edges.append((hidden, child, length))
        remaining = [node for node in active if node not in family]
        for node in remaining:
            estimates = np.column_stack([get(member, node) for member in family])
            estimate = np.median(estimates, axis=1)
            values[(min(hidden, node), max(hidden, node))] = np.clip(
                estimate,
                0.0,
                np.minimum(depths[hidden], depths[node]),
            )
        decisions.append(
            UncertainRNJDecision(
                pair=(int(left), int(right)),
                family=tuple(sorted(int(node) for node in family)),
                parent_depth=parent_depth,
                pair_lower_bound=pair_lower,
                pair_upper_bound=pair_upper,
                maximum_family_gap=max(family_gaps, default=0.0),
            )
        )
        active = remaining + [hidden]

    if active:
        node = active[0]
        edges.append((int(root), node, max(0.0, float(np.median(depths[node])))))
    tree = RootedTreeResult(
        root=int(root),
        terminals=tuple(labels),
        edges=_contract_zero_internal_edges(edges, int(root), labels),
        group_tolerance=float(group_tolerance),
    )
    return UncertainRNJResult(
        tree=tree,
        decisions=tuple(decisions),
        confidence_level=float(confidence_level),
        replicate_count=int(samples.shape[0]),
    )
