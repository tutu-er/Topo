"""Measurement objectives for selecting a backtracked Steiner-tree candidate."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from terminal_case33.graph.enhanced_recursive_grouping import GroupingCandidate


@dataclass(frozen=True)
class CandidateObjective:
    """Four Pengwah objective components and their normalized total."""

    candidate_index: int
    sensitivity_error: float
    voltage_error: float
    profile_error: float
    correlation_error: float
    total_score: float


def _fit_nonnegative_edge_lengths(
    edges: tuple[tuple[int, int, float], ...],
    nodes: list[int],
    target_distance: np.ndarray,
    iterations: int = 200,
) -> np.ndarray:
    graph = nx.Graph()
    edge_list = [(left, right) for left, right, _ in edges]
    graph.add_edges_from(edge_list)
    edge_index = {frozenset(edge): index for index, edge in enumerate(edge_list)}
    rows, target = [], []
    for i, left in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            row = np.zeros(len(edge_list), dtype=float)
            path = nx.shortest_path(graph, left, nodes[j])
            for first, second in zip(path[:-1], path[1:], strict=True):
                row[edge_index[frozenset((first, second))]] = 1.0
            rows.append(row)
            target.append(float(target_distance[i, j]))
    design = np.asarray(rows)
    response = np.asarray(target)
    lengths = np.maximum(0.0, np.linalg.lstsq(design, response, rcond=None)[0])
    step = 1.0 / max(float(np.linalg.norm(design, ord=2)) ** 2, 1e-15)
    for _ in range(iterations):
        proposal = np.maximum(0.0, lengths - step * design.T @ (design @ lengths - response))
        if np.linalg.norm(proposal - lengths) <= 1e-10 * max(np.linalg.norm(lengths), 1e-15):
            lengths = proposal
            break
        lengths = proposal
    return lengths


def _shared_path_matrix(
    edges: tuple[tuple[int, int, float], ...],
    edge_lengths: np.ndarray,
    root: int,
    terminals: list[int],
) -> np.ndarray:
    graph = nx.Graph()
    for (left, right, _), length in zip(edges, edge_lengths, strict=True):
        graph.add_edge(left, right, weight=float(length))
    depths = np.asarray([nx.shortest_path_length(graph, root, node, weight="weight") for node in terminals])
    distance = np.asarray(
        [[nx.shortest_path_length(graph, left, right, weight="weight") for right in terminals] for left in terminals]
    )
    shared = 0.5 * (depths[:, None] + depths[None, :] - distance)
    return np.maximum(0.0, 0.5 * (shared + shared.T))


def _profile_matrix(voltage: np.ndarray) -> np.ndarray:
    columns = []
    for left in range(voltage.shape[1]):
        for right in range(left + 1, voltage.shape[1]):
            columns.append(np.sign(voltage[:, left] - voltage[:, right]))
    return np.column_stack(columns) if columns else np.zeros((len(voltage), 0))


def rank_candidates_by_smart_meter_objectives(
    candidates: tuple[GroupingCandidate, ...],
    root: int,
    terminals: list[int],
    resistance_distance_with_root: np.ndarray,
    reactance_distance_with_root: np.ndarray,
    voltage: pd.DataFrame,
    active_power: pd.DataFrame,
    reactive_power: pd.DataFrame,
    root_voltage: pd.Series,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> tuple[int, tuple[CandidateObjective, ...]]:
    """Rank candidates using the four objectives of Pengwah et al. (2022).

    The published logistic coefficients were trained on a private synthetic
    corpus and are unavailable.  This implementation therefore normalizes
    each objective across the generated candidate pool and applies explicit
    user-supplied weights, defaulting to equal weights.
    """

    if not candidates:
        raise ValueError("at least one candidate is required")
    ordered_nodes = [*terminals, int(root)]
    index = voltage.index.intersection(active_power.index).intersection(reactive_power.index)
    index = index.intersection(root_voltage.index)
    v = voltage.loc[index, terminals].to_numpy(dtype=float)
    p = active_power.loc[index, terminals].to_numpy(dtype=float)
    q = reactive_power.loc[index, terminals].to_numpy(dtype=float)
    v0 = root_voltage.loc[index].to_numpy(dtype=float)
    ir = p / np.maximum(v, 1e-6)
    ix = q / np.maximum(v, 1e-6)
    raw: list[tuple[float, float, float, float]] = []
    for candidate in candidates:
        r_lengths = _fit_nonnegative_edge_lengths(candidate.edges, ordered_nodes, resistance_distance_with_root)
        x_lengths = _fit_nonnegative_edge_lengths(candidate.edges, ordered_nodes, reactance_distance_with_root)
        r_shared = _shared_path_matrix(candidate.edges, r_lengths, root, terminals)
        x_shared = _shared_path_matrix(candidate.edges, x_lengths, root, terminals)
        drop_hat = ir @ r_shared.T + ix @ x_shared.T
        voltage_hat = v0[:, None] - drop_hat
        sensitivity_error = float(
            np.linalg.norm(np.diff(v, axis=0) - np.diff(voltage_hat, axis=0), ord="fro")
            / max(np.linalg.norm(np.diff(v, axis=0), ord="fro"), 1e-15)
        )
        voltage_error = float(np.linalg.norm(v - voltage_hat, ord="fro") / max(np.linalg.norm(v, ord="fro"), 1e-15))
        measured_profile = _profile_matrix(v)
        estimated_profile = _profile_matrix(voltage_hat)
        profile_error = float(
            np.linalg.norm(measured_profile - estimated_profile, ord="fro")
            / max(np.sqrt(measured_profile.size), 1.0)
        )
        measured_corr = np.nan_to_num(np.corrcoef(v, rowvar=False))
        estimated_corr = np.nan_to_num(np.corrcoef(voltage_hat, rowvar=False))
        correlation_error = float(
            np.linalg.norm(measured_corr - estimated_corr, ord="fro")
            / max(np.linalg.norm(measured_corr, ord="fro"), 1e-15)
        )
        raw.append((sensitivity_error, voltage_error, profile_error, correlation_error))
    values = np.asarray(raw)
    minimum = values.min(axis=0)
    scale = np.maximum(values.max(axis=0) - minimum, 1e-12)
    normalized = (values - minimum) / scale
    totals = normalized @ np.asarray(weights, dtype=float)
    objectives = tuple(
        CandidateObjective(index, *raw[index], float(totals[index])) for index in range(len(candidates))
    )
    selected = min(range(len(candidates)), key=lambda index: (totals[index], candidates[index].distance_stress))
    return int(selected), objectives
