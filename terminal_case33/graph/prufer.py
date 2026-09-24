"""Integer-length tree reconstruction for the Soumalas et al. baseline.

The 2017 method assumes that feeder lengths are integer multiples of a
minimum span and validates several line-type/span scenarios.  This module
implements that complete scenario loop.  Each quantized additive matrix is
converted to a weighted latent tree, subdivided into unit edges, encoded as a
Pruefer sequence, decoded again, and ranked by terminal-distance residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heapify, heappop, heappush

import networkx as nx
import numpy as np

from terminal_case33.graph.latent_tree import LatentTreeResult, neighbor_joining


UnweightedEdge = tuple[int, int]


@dataclass(frozen=True)
class SoumalasCandidate:
    """One minimum-span scenario and its decoded integer tree."""

    minimum_span: float
    distance_stress: float
    prufer_sequence: tuple[int, ...]
    edges: tuple[UnweightedEdge, ...]


@dataclass(frozen=True)
class SoumalasResult:
    """Best Soumalas-style scenario and all candidates considered."""

    terminals: tuple[int, ...]
    chosen_minimum_span: float
    edges: tuple[UnweightedEdge, ...]
    prufer_sequence: tuple[int, ...]
    candidates: tuple[SoumalasCandidate, ...]


def prufer_encode(edges: list[UnweightedEdge] | tuple[UnweightedEdge, ...]) -> tuple[int, ...]:
    """Encode a labelled tree using the standard smallest-leaf convention."""

    graph = nx.Graph()
    graph.add_edges_from((int(left), int(right)) for left, right in edges)
    if graph.number_of_nodes() <= 2:
        return tuple()
    if not nx.is_tree(graph):
        raise ValueError("edges must form a tree")
    degree = dict(graph.degree())
    neighbours = {node: set(graph.neighbors(node)) for node in graph.nodes()}
    leaves = [node for node, value in degree.items() if value == 1]
    heapify(leaves)
    sequence: list[int] = []
    for _ in range(graph.number_of_nodes() - 2):
        leaf = heappop(leaves)
        parent = next(iter(neighbours[leaf]))
        sequence.append(int(parent))
        neighbours[parent].remove(leaf)
        degree[parent] -= 1
        if degree[parent] == 1:
            heappush(leaves, parent)
    return tuple(sequence)


def prufer_decode(sequence: list[int] | tuple[int, ...], labels: list[int] | tuple[int, ...]) -> tuple[UnweightedEdge, ...]:
    """Decode a Pruefer sequence over an explicitly supplied label set."""

    node_labels = [int(node) for node in labels]
    code = [int(node) for node in sequence]
    if len(node_labels) != len(code) + 2:
        raise ValueError("label count must equal sequence length plus two")
    if len(set(node_labels)) != len(node_labels) or any(node not in node_labels for node in code):
        raise ValueError("labels must be unique and contain every sequence entry")
    degree = {node: 1 for node in node_labels}
    for node in code:
        degree[node] += 1
    leaves = [node for node in node_labels if degree[node] == 1]
    heapify(leaves)
    edges: list[UnweightedEdge] = []
    for parent in code:
        leaf = heappop(leaves)
        edges.append((leaf, parent))
        degree[leaf] -= 1
        degree[parent] -= 1
        if degree[parent] == 1:
            heappush(leaves, parent)
    remaining = sorted(node for node, value in degree.items() if value == 1)
    edges.append((remaining[0], remaining[1]))
    return tuple(sorted((min(left, right), max(left, right)) for left, right in edges))


def _integer_subdivision(tree: LatentTreeResult, terminals: list[int]) -> tuple[UnweightedEdge, ...]:
    """Round latent edge lengths and expand every edge into unit spans."""

    used = {int(node) for edge in tree.edges for node in edge[:2]}
    next_hidden = min([-1, *(node for node in used if node < 0)]) - 1
    result: list[UnweightedEdge] = []
    for left, right, length in tree.edges:
        spans = max(1, int(np.rint(length)))
        previous = int(left)
        for _ in range(spans - 1):
            while next_hidden in used:
                next_hidden -= 1
            result.append((previous, next_hidden))
            used.add(next_hidden)
            previous = next_hidden
            next_hidden -= 1
        result.append((previous, int(right)))
    graph = nx.Graph()
    graph.add_edges_from(result)
    if not nx.is_tree(graph) or not set(terminals).issubset(graph.nodes):
        raise RuntimeError("integer subdivision did not produce a terminal-spanning tree")
    return tuple(result)


def _terminal_distances(edges: tuple[UnweightedEdge, ...], terminals: list[int]) -> np.ndarray:
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return np.asarray(
        [[nx.shortest_path_length(graph, left, right) for right in terminals] for left in terminals],
        dtype=float,
    )


def soumalas_prufer_reconstruction(
    distance_matrix: np.ndarray,
    terminals: list[int],
    minimum_spans: list[float] | tuple[float, ...],
) -> SoumalasResult:
    """Reconstruct and rank integer-span topology scenarios.

    ``minimum_spans`` represents the candidate product of minimum line length
    and resistance (or reactance) per unit length.  In a noiseless homogeneous
    feeder, including the true value recovers the exact unit-edge tree.
    """

    distance = np.asarray(distance_matrix, dtype=float)
    if distance.shape != (len(terminals), len(terminals)):
        raise ValueError("distance_matrix shape must match terminals")
    spans = sorted({float(value) for value in minimum_spans if float(value) > 0.0})
    if not spans:
        raise ValueError("minimum_spans must contain a positive value")
    candidates: list[SoumalasCandidate] = []
    reference_norm = max(float(np.linalg.norm(distance, ord="fro")), 1e-15)
    for span in spans:
        quantized = np.rint(np.maximum(distance, 0.0) / span)
        quantized = 0.5 * (quantized + quantized.T)
        np.fill_diagonal(quantized, 0.0)
        latent = neighbor_joining(quantized, terminals)
        unit_edges = _integer_subdivision(latent, terminals)
        labels = sorted({node for edge in unit_edges for node in edge})
        sequence = prufer_encode(unit_edges)
        decoded = prufer_decode(sequence, labels)
        predicted = _terminal_distances(decoded, terminals) * span
        stress = float(np.linalg.norm(predicted - distance, ord="fro") / reference_norm)
        candidates.append(SoumalasCandidate(span, stress, sequence, decoded))
    # Equal electrical fits are observationally indistinguishable. Prefer the
    # minimal tree, which avoids inventing recoverability of degree-2 chains.
    candidates.sort(key=lambda item: (item.distance_stress, len(item.edges), -item.minimum_span))
    best = candidates[0]
    return SoumalasResult(
        terminals=tuple(int(node) for node in terminals),
        chosen_minimum_span=best.minimum_span,
        edges=best.edges,
        prufer_sequence=best.prufer_sequence,
        candidates=tuple(candidates),
    )
