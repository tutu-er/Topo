"""Rooted peripheral-clade aggregation for hidden-tree reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from rnj_wzzt.graph.rooted_neighbor_joining import (
    RootedTreeResult,
    rooted_neighbor_joining,
)


@dataclass(frozen=True)
class PseudoCluster:
    """A reliable downstream clade contracted to one pseudo boundary node."""

    pseudo_id: int
    members: frozenset[int]
    confidence: float
    frozen_clades: tuple[frozenset[int], ...]
    frozen_sibling_pairs: tuple[tuple[int, int], ...]


def rooted_tree_from_clades(
    clades: set[frozenset[int]] | frozenset[frozenset[int]],
    terminals: list[int] | tuple[int, ...],
    root: int,
    include_hidden_root_stem: bool = False,
) -> RootedTreeResult:
    """Build the unique reduced rooted topology represented by laminar clades.

    Nontrivial rooted clades determine a reduced tree up to hidden-node labels
    and positive edge lengths. ``include_hidden_root_stem`` preserves a common
    root-to-all-terminals edge, which is intentionally absent from nontrivial
    clade metrics. Returned unit lengths are placeholders; fixed-tree R/X
    fitting estimates physical edge coefficients later.
    """

    terminal_tuple = tuple(int(node) for node in terminals)
    root = int(root)
    if len(set(terminal_tuple)) != len(terminal_tuple):
        raise ValueError("terminals must contain unique labels")
    if root in terminal_tuple:
        raise ValueError("root must not be a terminal")
    terminal_set = frozenset(terminal_tuple)
    normalized = {
        frozenset(int(node) for node in clade)
        for clade in clades
    }
    if any(not clade <= terminal_set for clade in normalized):
        raise ValueError("clades must contain only supplied terminals")
    normalized = {clade for clade in normalized if 1 < len(clade) < len(terminal_tuple)}
    ordered = sorted(normalized, key=lambda item: (len(item), tuple(sorted(item))))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if left & right and not (left <= right or right <= left):
                raise ValueError("rooted clades must be laminar")

    first_hidden = min(-1, min((root, *terminal_tuple)) - 1)
    hidden_for_clade = {clade: first_hidden - index for index, clade in enumerate(ordered)}
    stem = first_hidden - len(ordered) if include_hidden_root_stem else root
    containers = [*ordered, terminal_set]
    edges: list[tuple[int, int, float]] = []
    if include_hidden_root_stem:
        edges.append((int(root), stem, 1.0))
    for clade in ordered:
        parent_clade = min(
            (candidate for candidate in containers if clade < candidate),
            key=lambda item: (len(item), tuple(sorted(item))),
        )
        parent = stem if parent_clade == terminal_set else hidden_for_clade[parent_clade]
        edges.append((int(parent), hidden_for_clade[clade], 1.0))
    for terminal in terminal_tuple:
        containing = [clade for clade in ordered if terminal in clade]
        parent = hidden_for_clade[min(containing, key=len)] if containing else stem
        edges.append((int(parent), terminal, 1.0))
    return RootedTreeResult(
        root=int(root),
        terminals=terminal_tuple,
        edges=tuple(edges),
        group_tolerance=0.0,
    )


def rooted_clades(
    edges: list[tuple] | tuple[tuple, ...],
    root: int,
    terminals: list[int] | tuple[int, ...],
) -> set[frozenset[int]]:
    """Return nontrivial downstream terminal clades of a rooted tree."""

    graph = nx.Graph()
    graph.add_edges_from((int(edge[0]), int(edge[1])) for edge in edges)
    terminal_set = {int(node) for node in terminals}
    parent = {
        int(child): int(upstream) for child, upstream in nx.bfs_predecessors(graph, int(root))
    }
    children: dict[int, list[int]] = {int(node): [] for node in graph.nodes()}
    for child, upstream in parent.items():
        children[upstream].append(child)
    clades = set()

    def visit(node: int) -> set[int]:
        found = {node} if node in terminal_set else set()
        for child in children[node]:
            found.update(visit(child))
        if 1 < len(found) < len(terminal_set):
            clades.add(frozenset(found))
        return found

    visit(int(root))
    return clades


def select_peripheral_clusters(
    terminals: list[int],
    clade_confidence: dict[frozenset[int], float],
    sibling_confidence: dict[tuple[int, int], float],
    threshold: float,
    max_cluster_size: int,
) -> list[PseudoCluster]:
    """Select large disjoint reliable clades and freeze their local structures."""

    def has_internal_support(clade: frozenset[int]) -> bool:
        stable_pairs = [
            confidence for pair, confidence in sibling_confidence.items() if set(pair) <= clade
        ]
        stable_nested = [
            confidence for nested, confidence in clade_confidence.items() if nested < clade
        ]
        if len(clade) == 2:
            pair = tuple(sorted(clade))
            return sibling_confidence.get(pair, 0.0) >= threshold
        return max([*stable_pairs, *stable_nested], default=0.0) >= threshold

    candidates = [
        clade
        for clade, confidence in clade_confidence.items()
        if (
            confidence >= threshold
            and 2 <= len(clade) <= max_cluster_size
            and has_internal_support(clade)
        )
    ]
    selected: list[frozenset[int]] = []
    occupied: set[int] = set()
    for clade in sorted(candidates, key=lambda item: (-len(item), tuple(sorted(item)))):
        if occupied.isdisjoint(clade):
            selected.append(clade)
            occupied.update(clade)

    clusters = []
    for index, members in enumerate(selected):
        frozen_clades = tuple(
            sorted(
                (
                    clade
                    for clade, confidence in clade_confidence.items()
                    if confidence >= threshold and clade <= members
                ),
                key=lambda item: (len(item), tuple(sorted(item))),
            )
        )
        frozen_pairs = tuple(
            sorted(
                pair
                for pair, confidence in sibling_confidence.items()
                if confidence >= threshold and set(pair) <= members
            )
        )
        clusters.append(
            PseudoCluster(
                pseudo_id=900000 + index,
                members=members,
                confidence=float(clade_confidence[members]),
                frozen_clades=frozen_clades,
                frozen_sibling_pairs=frozen_pairs,
            )
        )
    return clusters


def _boundary_differential(
    matrix: np.ndarray,
    member_positions: list[int],
) -> np.ndarray:
    """Return descendant-to-boundary coefficients for injections inside a clade."""

    submatrix = matrix[np.ix_(member_positions, member_positions)]
    off_diagonal = submatrix[np.triu_indices(len(member_positions), 1)]
    boundary_depth = np.quantile(off_diagonal, 0.2)
    boundary_depth = np.maximum(np.clip(boundary_depth, 0.0, np.min(np.diag(submatrix))), 0.0)
    return np.maximum(submatrix - boundary_depth, 0.0)


def aggregate_rooted_scenarios(
    scenarios: list[dict],
    terminals: list[int],
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    clusters: list[PseudoCluster],
    voltage_mode: str,
    deembedding_weight: float = 1.0,
) -> tuple[list[dict], dict[int, frozenset[int]]]:
    """Aggregate P/Q and reconstruct pseudo-boundary squared voltage.

    ``deembedded_vsq`` starts from each measured terminal voltage and adds the
    estimated differential R/X drop from the terminal to the clade boundary.
    Tree additivity makes that differential zero for injections outside the
    clade, so de-embedding deliberately uses only descendant P/Q columns.
    ``mean_vsq`` retains the former arithmetic-mean proxy for comparison.
    """

    if voltage_mode not in {"mean_vsq", "deembedded_vsq"}:
        raise ValueError("voltage_mode must be 'mean_vsq' or 'deembedded_vsq'")
    if not 0.0 <= deembedding_weight <= 1.0:
        raise ValueError("deembedding_weight must be in [0, 1]")
    from rnj_wzzt.estimation.preprocessing import (
        squared_voltage_drop_from_observed_root,
    )

    position = {int(node): index for index, node in enumerate(terminals)}
    clustered = set().union(*(cluster.members for cluster in clusters)) if clusters else set()
    pseudo_members = {cluster.pseudo_id: cluster.members for cluster in clusters}
    next_id = 910000
    for terminal in terminals:
        if terminal not in clustered:
            pseudo_members[next_id] = frozenset([terminal])
            next_id += 1

    differentials = {}
    if voltage_mode == "deembedded_vsq":
        for pseudo_id, members in pseudo_members.items():
            if len(members) > 1:
                indices = [position[node] for node in sorted(members)]
                differentials[pseudo_id] = (
                    _boundary_differential(r_matrix, indices),
                    _boundary_differential(x_matrix, indices),
                )

    aggregated = []
    for scenario in scenarios:
        p_terminal = scenario["P_terminal"].loc[:, terminals]
        q_terminal = scenario["Q_terminal"].loc[:, terminals]
        v_terminal = scenario["V_terminal"].loc[:, terminals]
        p_pseudo = pd.DataFrame(index=p_terminal.index)
        q_pseudo = pd.DataFrame(index=q_terminal.index)
        v_pseudo_sq = pd.DataFrame(index=v_terminal.index)
        for pseudo_id, members in pseudo_members.items():
            children = sorted(members)
            p_children = p_terminal[children]
            q_children = q_terminal[children]
            voltage_sq = v_terminal[children].pow(2)
            p_pseudo[pseudo_id] = p_children.sum(axis=1)
            q_pseudo[pseudo_id] = q_children.sum(axis=1)
            if len(children) == 1:
                v_pseudo_sq[pseudo_id] = voltage_sq[children[0]]
                continue
            mean_voltage_sq = voltage_sq.mean(axis=1)
            if voltage_mode == "mean_vsq":
                v_pseudo_sq[pseudo_id] = mean_voltage_sq
                continue
            delta_r, delta_x = differentials[pseudo_id]
            recovered = (
                voltage_sq.to_numpy(dtype=float)
                + p_children.to_numpy(dtype=float) @ delta_r.T
                + q_children.to_numpy(dtype=float) @ delta_x.T
            )
            deembedded = np.median(recovered, axis=1)
            mean_voltage_sq = mean_voltage_sq.to_numpy(dtype=float)
            v_pseudo_sq[pseudo_id] = mean_voltage_sq + deembedding_weight * (
                deembedded - mean_voltage_sq
            )

        v_pseudo = np.sqrt(np.maximum(v_pseudo_sq, 1e-12))
        aggregated.append(
            {
                "name": scenario["name"],
                "P_terminal": p_pseudo,
                "Q_terminal": q_pseudo,
                "V_terminal": v_pseudo,
                "root_voltage": scenario["root_voltage"],
                "drop_target": squared_voltage_drop_from_observed_root(
                    v_pseudo, scenario["root_voltage"]
                ),
            }
        )
    return aggregated, pseudo_members


def build_local_cluster_scenarios(
    scenarios: list[dict],
    pseudo_scenarios: list[dict],
    cluster: PseudoCluster,
) -> list[dict]:
    """Build terminal measurements referenced to one estimated pseudo boundary.

    The pseudo boundary voltage is used as the observed root voltage of the
    local problem. Only terminals inside ``cluster`` remain as local injections,
    so the second-stage reconstruction does not create another pseudo layer.
    """

    if len(scenarios) != len(pseudo_scenarios):
        raise ValueError("original and pseudo scenario counts must match")
    from rnj_wzzt.estimation.preprocessing import (
        squared_voltage_drop_from_observed_root,
    )

    members = sorted(cluster.members)
    local = []
    for scenario, pseudo in zip(scenarios, pseudo_scenarios, strict=True):
        if cluster.pseudo_id not in pseudo["V_terminal"].columns:
            raise ValueError(f"pseudo voltage is missing cluster {cluster.pseudo_id}")
        root_voltage = pseudo["V_terminal"][cluster.pseudo_id].copy()
        voltage = scenario["V_terminal"].loc[:, members].copy()
        local.append(
            {
                "name": scenario["name"],
                "P_terminal": scenario["P_terminal"].loc[:, members].copy(),
                "Q_terminal": scenario["Q_terminal"].loc[:, members].copy(),
                "V_terminal": voltage,
                "root_voltage": root_voltage,
                "drop_target": squared_voltage_drop_from_observed_root(voltage, root_voltage),
            }
        )
    return local


def reidentify_local_clades_from_shared_paths(
    shared_path_matrix: np.ndarray,
    root_depths: np.ndarray,
    terminals: list[int],
    clusters: list[PseudoCluster],
    tolerance_factor: float,
    boundary_quantile: float = 0.2,
) -> dict[int, set[frozenset[int]]]:
    """Re-root estimated shared paths at each pseudo boundary and rerun RNJ.

    This avoids a second sensitivity regression. Pairwise terminal distances
    remain those estimated from the full data set; only the common path above
    each one-layer cluster is removed before local reconstruction.
    """

    if not 0.0 <= boundary_quantile <= 1.0:
        raise ValueError("boundary_quantile must be in [0, 1]")
    shared = np.asarray(shared_path_matrix, dtype=float)
    depths = np.asarray(root_depths, dtype=float)
    if shared.shape != (len(terminals), len(terminals)) or depths.shape != (len(terminals),):
        raise ValueError("shared paths and root depths must match terminals")
    position = {int(node): index for index, node in enumerate(terminals)}
    local_clades = {}
    for cluster in clusters:
        members = sorted(cluster.members)
        indices = [position[node] for node in members]
        submatrix = shared[np.ix_(indices, indices)]
        off_diagonal = submatrix[np.triu_indices(len(indices), 1)]
        boundary_depth = (
            float(np.quantile(off_diagonal, boundary_quantile)) if off_diagonal.size else 0.0
        )
        local_depths = np.maximum(depths[indices] - boundary_depth, 0.0)
        local_shared = np.maximum(submatrix - boundary_depth, 0.0)
        np.fill_diagonal(local_shared, local_depths)
        tolerance = tolerance_factor * max(float(np.median(local_depths)), 1e-12)
        tree = rooted_neighbor_joining(
            local_shared,
            local_depths,
            members,
            cluster.pseudo_id,
            tolerance,
        )
        local_clades[cluster.pseudo_id] = rooted_clades(tree.edges, cluster.pseudo_id, members)
    return local_clades


def expand_pseudo_clades(
    pseudo_edges: tuple[tuple[int, int, float], ...],
    root: int,
    pseudo_members: dict[int, frozenset[int]],
    clusters: list[PseudoCluster],
) -> set[frozenset[int]]:
    """Expand pseudo-level rooted clades while retaining frozen local clades."""

    pseudo_nodes = list(pseudo_members)
    expanded = set()
    for pseudo_clade in rooted_clades(pseudo_edges, root, pseudo_nodes):
        expanded.add(frozenset().union(*(pseudo_members[node] for node in pseudo_clade)))
    for cluster in clusters:
        expanded.add(cluster.members)
        expanded.update(cluster.frozen_clades)
    return expanded


def expand_pseudo_clades_with_local_reidentification(
    pseudo_edges: tuple[tuple[int, int, float], ...],
    root: int,
    pseudo_members: dict[int, frozenset[int]],
    clusters: list[PseudoCluster],
    local_clades: dict[int, set[frozenset[int]]],
) -> set[frozenset[int]]:
    """Combine one pseudo backbone with independently re-identified local trees."""

    pseudo_nodes = list(pseudo_members)
    expanded = {
        frozenset().union(*(pseudo_members[node] for node in pseudo_clade))
        for pseudo_clade in rooted_clades(pseudo_edges, root, pseudo_nodes)
    }
    for cluster in clusters:
        expanded.add(cluster.members)
        expanded.update(local_clades.get(cluster.pseudo_id, set()))
    return expanded


def expand_pseudo_sibling_pairs(
    pseudo_edges: tuple[tuple[int, int, float], ...],
    root: int,
    pseudo_members: dict[int, frozenset[int]],
    clusters: list[PseudoCluster],
) -> set[tuple[int, int]]:
    """Keep frozen local sibling pairs and map singleton pseudo siblings."""

    graph = nx.Graph()
    graph.add_edges_from((int(left), int(right)) for left, right, _ in pseudo_edges)
    pseudo_set = set(pseudo_members)
    pairs = set(pair for cluster in clusters for pair in cluster.frozen_sibling_pairs)
    for node in graph.nodes():
        adjacent = [neighbor for neighbor in graph.neighbors(node) if neighbor in pseudo_set]
        for left, right in combinations(adjacent, 2):
            if len(pseudo_members[left]) == len(pseudo_members[right]) == 1:
                pairs.add(
                    tuple(
                        sorted(
                            (next(iter(pseudo_members[left])), next(iter(pseudo_members[right])))
                        )
                    )
                )
    return pairs
