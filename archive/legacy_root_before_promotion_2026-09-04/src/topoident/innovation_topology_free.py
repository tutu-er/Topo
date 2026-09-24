"""Topology-library-free latent-tree recovery from terminal measurements.

The learner receives only terminal injections/potentials and the slack label.  The
physical network is used solely by the simulator.  A terminal Green matrix is
estimated first; neighbor joining then creates both edges and latent nodes from
its induced additive resistance metric.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np

from .powerflow import build_ybus
from .terminal_pmu import terminal_test_network


WeightedEdge = tuple[int, int, float]
Split = tuple[int, ...]


def _estimate_green(
    terminal_injections: np.ndarray,
    terminal_potentials: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Estimate and project the terminal Green matrix onto the PSD cone."""
    gram = terminal_injections.T @ terminal_injections
    cross = terminal_injections.T @ terminal_potentials
    green = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), cross)
    green = 0.5 * (green + green.T)
    eigenvalues, eigenvectors = np.linalg.eigh(green)
    floor = max(1.0e-10, 1.0e-8 * float(np.max(np.abs(eigenvalues))))
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


def _distance_from_green(green: np.ndarray) -> np.ndarray:
    """Add the grounded slack as leaf zero and form effective resistances."""
    terminals = green.shape[0]
    distance = np.zeros((terminals + 1, terminals + 1))
    diagonal = np.diag(green)
    distance[0, 1:] = diagonal
    distance[1:, 0] = diagonal
    interior = diagonal[:, None] + diagonal[None, :] - 2.0 * green
    distance[1:, 1:] = np.maximum(interior, 0.0)
    return distance


def _neighbor_joining(distance: np.ndarray, labels: list[int]) -> list[WeightedEdge]:
    """Construct an unrooted latent tree without enumerating candidate trees."""
    active = list(labels)
    values: dict[tuple[int, int], float] = {}

    def key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    for i, a in enumerate(active):
        for j, b in enumerate(active[i + 1 :], start=i + 1):
            values[key(a, b)] = float(distance[i, j])

    def get(a: int, b: int) -> float:
        return 0.0 if a == b else values[key(a, b)]

    edges: list[WeightedEdge] = []
    next_latent = -1
    while len(active) > 2:
        count = len(active)
        row_sum = {a: sum(get(a, b) for b in active if b != a) for a in active}
        pair = min(
            combinations(active, 2),
            key=lambda ab: (count - 2) * get(*ab) - row_sum[ab[0]] - row_sum[ab[1]],
        )
        left, right = pair
        separation = get(left, right)
        left_length = 0.5 * separation + (row_sum[left] - row_sum[right]) / (
            2.0 * (count - 2)
        )
        right_length = separation - left_length
        latent = next_latent
        next_latent -= 1
        edges.append((latent, left, max(float(left_length), 0.0)))
        edges.append((latent, right, max(float(right_length), 0.0)))

        remaining = [node for node in active if node not in pair]
        for node in remaining:
            values[key(latent, node)] = 0.5 * (
                get(left, node) + get(right, node) - separation
            )
        active = remaining + [latent]

    edges.append((active[0], active[1], max(get(active[0], active[1]), 0.0)))
    return edges


def _tree_splits(edges: list[tuple[int, int]], observed: set[int]) -> set[Split]:
    """Return unique non-trivial bipartitions induced on observed leaves."""
    adjacency: dict[int, set[int]] = {}
    for u, v in edges:
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)

    splits: set[Split] = set()
    all_observed = set(observed)
    for u, v in edges:
        reached = {u}
        stack = [u]
        while stack:
            node = stack.pop()
            for nxt in adjacency[node]:
                if (node == u and nxt == v) or (node == v and nxt == u):
                    continue
                if nxt not in reached:
                    reached.add(nxt)
                    stack.append(nxt)
        side = reached & all_observed
        other = all_observed - side
        if len(side) < 2 or len(other) < 2:
            continue
        a = tuple(sorted(side))
        b = tuple(sorted(other))
        splits.add(a if (len(a), a) <= (len(b), b) else b)
    return splits


def _leaf_distances(edges: list[WeightedEdge], labels: list[int]) -> np.ndarray:
    adjacency: dict[int, list[tuple[int, float]]] = {}
    for u, v, weight in edges:
        adjacency.setdefault(u, []).append((v, weight))
        adjacency.setdefault(v, []).append((u, weight))
    result = np.zeros((len(labels), len(labels)))
    for row, source in enumerate(labels):
        distances = {source: 0.0}
        stack = [(source, -999_999)]
        while stack:
            node, parent = stack.pop()
            for nxt, weight in adjacency[node]:
                if nxt == parent:
                    continue
                distances[nxt] = distances[node] + weight
                stack.append((nxt, node))
        for column, target in enumerate(labels):
            result[row, column] = distances[target]
    return result


def _simulate_terminal_potentials(
    samples: int,
    rng: np.random.Generator,
    measurement_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[tuple[int, int]]]:
    """Generate a scalar radial experiment; topology is not passed to the learner."""
    n_bus, true_edges, library, terminals = terminal_test_network()
    conductance_branches = []
    for edge in true_edges:
        branch = library[edge]
        conductance_branches.append(type(branch)(branch.u, branch.v, branch.r, 0.0))
    laplacian = build_ybus(n_bus, conductance_branches).real

    base = np.array([0.055, 0.070, 0.060, 0.075])
    injections = -base * rng.lognormal(
        mean=-0.5 * 0.45**2, sigma=0.45, size=(samples, len(terminals))
    )
    full_injections = np.zeros((samples, n_bus))
    full_injections[:, terminals] = injections
    potentials = np.linalg.solve(laplacian[1:, 1:], full_injections[:, 1:].T).T
    terminal_potentials = potentials[:, np.asarray(terminals) - 1]
    observed = terminal_potentials + rng.normal(
        0.0, measurement_sigma, terminal_potentials.shape
    )

    selector = np.asarray(terminals) - 1
    exact_green = np.linalg.inv(laplacian[1:, 1:])[np.ix_(selector, selector)]
    return injections, observed, exact_green, terminals, true_edges


def run_topology_free_latent_tree(
    *,
    seed: int = 707,
    samples: int = 600,
    bootstrap_draws: int = 80,
    measurement_sigma: float = 2.0e-5,
    ridge: float = 1.0e-8,
) -> dict[str, object]:
    """Recover identifiable terminal splits without topology or edge candidates."""
    rng = np.random.default_rng(seed)
    injections, observed, exact_green, terminals, true_edges = _simulate_terminal_potentials(
        samples, rng, measurement_sigma
    )
    labels = [0] + terminals
    observed_set = set(labels)

    estimated_green = _estimate_green(injections, observed, ridge)
    estimated_distance = _distance_from_green(estimated_green)
    exact_distance = _distance_from_green(exact_green)
    estimated_tree = _neighbor_joining(estimated_distance, labels)
    estimated_splits = _tree_splits(
        [(u, v) for u, v, _ in estimated_tree], observed_set
    )
    true_splits = _tree_splits(true_edges, observed_set)

    counts: Counter[Split] = Counter()
    for _ in range(bootstrap_draws):
        indices = rng.integers(0, samples, size=samples)
        green = _estimate_green(injections[indices], observed[indices], ridge)
        tree = _neighbor_joining(_distance_from_green(green), labels)
        counts.update(_tree_splits([(u, v) for u, v, _ in tree], observed_set))

    reconstructed_distance = _leaf_distances(estimated_tree, labels)
    scale = np.maximum(exact_distance[np.triu_indices(len(labels), 1)], 1.0e-12)
    distance_relative_error = float(
        np.mean(
            np.abs(
                estimated_distance[np.triu_indices(len(labels), 1)]
                - exact_distance[np.triu_indices(len(labels), 1)]
            )
            / scale
        )
    )
    tree_fit_rmse = float(np.sqrt(np.mean((reconstructed_distance - estimated_distance) ** 2)))
    matched = estimated_splits & true_splits
    latent_nodes = {node for edge in estimated_tree for node in edge[:2] if node < 0}

    split_support = {
        "|".join(map(str, split)): counts[split] / bootstrap_draws
        for split in sorted(counts)
    }
    true_support = [counts[split] / bootstrap_draws for split in true_splits]
    return {
        "innovation": "topology-library-free terminal Green matrix and latent additive tree",
        "learner_prior": "slack label, terminal labels, radial positive-resistance model only",
        "uses_candidate_topologies": False,
        "uses_candidate_edges": False,
        "uses_hidden_bus_count": False,
        "samples": samples,
        "observed_labels_zero_based": labels,
        "terminal_buses_one_based": [node + 1 for node in terminals],
        "inferred_latent_nodes": len(latent_nodes),
        "true_nontrivial_splits": len(true_splits),
        "recovered_nontrivial_splits": len(estimated_splits),
        "matched_nontrivial_splits": len(matched),
        "split_precision": len(matched) / max(len(estimated_splits), 1),
        "split_recall": len(matched) / max(len(true_splits), 1),
        "minimum_true_split_bootstrap_support": min(true_support, default=1.0),
        "bootstrap_split_support": split_support,
        "terminal_distance_mean_relative_error": distance_relative_error,
        "additive_tree_fit_rmse": tree_fit_rmse,
        "identifiability_note": (
            "degree-2 hidden buses are suppressed; recovered edges represent series-equivalent "
            "segments between identifiable branching points"
        ),
    }
