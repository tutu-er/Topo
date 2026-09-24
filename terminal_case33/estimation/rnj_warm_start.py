"""Stable partial RNJ clades for laminar L1-MILP initialization."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil
from typing import Hashable, Sequence

import numpy as np

from terminal_case33.estimation.multiscenario import fit_projected_sensitivity
from terminal_case33.graph.rooted_hierarchy import rooted_clades
from terminal_case33.graph.rooted_neighbor_joining import rooted_neighbor_joining
from terminal_case33.graph.sensitivity_geometry import sensitivity_geometry


LabelClade = tuple[Hashable, ...]


@dataclass(frozen=True)
class StableRNJSupportSelection:
    """Auditable output of conservative multi-mode bootstrap RNJ screening."""

    terminal_labels: tuple[Hashable, ...]
    modes: tuple[str, ...]
    bootstrap_replicates: int
    confidence_threshold: float
    tolerance_factor: float
    maximum_support_size: int
    maximum_support_count: int
    selected_support_labels: tuple[LabelClade, ...]
    selected_support_indices: tuple[tuple[int, ...], ...]
    selected_confidences: tuple[float, ...]
    full_data_clades_by_mode: tuple[tuple[str, tuple[LabelClade, ...]], ...]
    bootstrap_confidences_by_mode: tuple[
        tuple[str, tuple[tuple[LabelClade, float], ...]], ...
    ]


def _validate_scenarios(scenarios: Sequence[dict]) -> tuple[Hashable, ...]:
    if not scenarios:
        raise ValueError("at least one scenario is required")
    first = scenarios[0]["P_terminal"]
    if not hasattr(first, "columns"):
        raise ValueError("RNJ warm start requires labelled tabular P_terminal data")
    labels = tuple(first.columns.tolist())
    if len(labels) < 2:
        raise ValueError("RNJ warm start requires at least two terminals")
    for scenario in scenarios:
        for key in ("P_terminal", "Q_terminal", "drop_target"):
            value = scenario[key]
            if not hasattr(value, "columns") or tuple(value.columns.tolist()) != labels:
                raise ValueError(f"{key} columns must match across scenarios")
            if len(value) != len(scenario["P_terminal"]):
                raise ValueError("P, Q, and target row counts must agree")
    return labels


def _clades_from_scenarios(
    scenarios: Sequence[dict],
    labels: tuple[Hashable, ...],
    *,
    root_bus: int,
    modes: tuple[str, ...],
    tolerance_factor: float,
    ridge_alpha: float,
) -> dict[str, set[frozenset[Hashable]]]:
    r_matrix, x_matrix, _, _ = fit_projected_sensitivity(
        list(scenarios),
        alpha=ridge_alpha,
        constraint_mode="ordered",
    )
    integer_labels = [int(label) for label in labels]
    output: dict[str, set[frozenset[Hashable]]] = {}
    for mode in modes:
        geometry = sensitivity_geometry(r_matrix, x_matrix, mode)
        scale = max(float(np.median(geometry.root_depths)), 1e-12)
        tree = rooted_neighbor_joining(
            geometry.shared_paths,
            geometry.root_depths,
            integer_labels,
            int(root_bus),
            group_tolerance=tolerance_factor * scale,
        )
        integer_clades = rooted_clades(tree.edges, int(root_bus), integer_labels)
        output[mode] = {
            frozenset(labels[integer_labels.index(node)] for node in clade)
            for clade in integer_clades
        }
    return output


def _bootstrap_copy(scenarios: Sequence[dict], rng: np.random.Generator) -> list[dict]:
    sampled: list[dict] = []
    for scenario in scenarios:
        count = len(scenario["P_terminal"])
        if count < 2:
            raise ValueError("each scenario needs at least two rows for bootstrap RNJ")
        indices = rng.integers(0, count, size=count)
        replicate = {"name": str(scenario.get("name", "scenario"))}
        for key in ("P_terminal", "Q_terminal", "drop_target"):
            replicate[key] = scenario[key].iloc[indices].reset_index(drop=True)
        sampled.append(replicate)
    return sampled


def _label_tuple(
    clade: frozenset[Hashable], labels: tuple[Hashable, ...]
) -> LabelClade:
    return tuple(label for label in labels if label in clade)


def _laminar_with(
    selected: Sequence[frozenset[Hashable]], candidate: frozenset[Hashable]
) -> bool:
    for existing in selected:
        overlap = existing & candidate
        if overlap and not (existing <= candidate or candidate <= existing):
            return False
    return True


def select_stable_rnj_initial_supports(
    scenarios: Sequence[dict],
    *,
    root_bus: int,
    modes: tuple[str, ...] = ("R", "RX_75R_25X"),
    bootstrap_replicates: int = 12,
    confidence_threshold: float = 0.75,
    tolerance_factor: float = 0.16,
    maximum_support_size: int | None = None,
    maximum_support_count: int = 3,
    ridge_alpha: float = 1e-6,
    seed: int = 20260903,
) -> StableRNJSupportSelection:
    """Select small, stable, mutually laminar RNJ clades using training data only.

    A candidate must occur in the full-data RNJ tree for every requested mode.
    Its confidence is the minimum bootstrap frequency across modes.  Truth is
    neither accepted nor consulted by this routine.
    """

    labels = _validate_scenarios(scenarios)
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("modes must be a nonempty sequence without duplicates")
    if not isinstance(bootstrap_replicates, (int, np.integer)) or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    if not np.isfinite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must lie in [0, 1]")
    if not np.isfinite(tolerance_factor) or tolerance_factor < 0.0:
        raise ValueError("tolerance_factor must be nonnegative")
    if not isinstance(maximum_support_count, (int, np.integer)) or maximum_support_count < 0:
        raise ValueError("maximum_support_count must be a nonnegative integer")
    if not np.isfinite(ridge_alpha) or ridge_alpha < 0.0:
        raise ValueError("ridge_alpha must be nonnegative")
    if maximum_support_size is None:
        maximum_support_size = max(2, min(6, ceil(len(labels) / 3)))
    if not isinstance(maximum_support_size, (int, np.integer)):
        raise ValueError("maximum_support_size must be an integer")
    maximum_support_size = int(maximum_support_size)
    if maximum_support_size < 2 or maximum_support_size >= len(labels):
        raise ValueError("maximum_support_size must lie between 2 and n-1")

    full = _clades_from_scenarios(
        scenarios,
        labels,
        root_bus=root_bus,
        modes=modes,
        tolerance_factor=tolerance_factor,
        ridge_alpha=ridge_alpha,
    )
    counters = {mode: Counter() for mode in modes}
    rng = np.random.default_rng(seed)
    for _ in range(int(bootstrap_replicates)):
        sampled = _bootstrap_copy(scenarios, rng)
        replicate = _clades_from_scenarios(
            sampled,
            labels,
            root_bus=root_bus,
            modes=modes,
            tolerance_factor=tolerance_factor,
            ridge_alpha=ridge_alpha,
        )
        for mode in modes:
            counters[mode].update(replicate[mode])

    common_full = set.intersection(*(set(full[mode]) for mode in modes))
    confidences = {
        clade: min(
            counters[mode][clade] / float(bootstrap_replicates) for mode in modes
        )
        for clade in common_full
    }
    candidates = [
        clade
        for clade, confidence in confidences.items()
        if 2 <= len(clade) <= maximum_support_size
        and confidence >= confidence_threshold
    ]
    selected: list[frozenset[Hashable]] = []
    for clade in sorted(
        candidates,
        key=lambda item: (
            -confidences[item],
            len(item),
            tuple(str(value) for value in _label_tuple(item, labels)),
        ),
    ):
        if len(selected) >= int(maximum_support_count):
            break
        if _laminar_with(selected, clade):
            selected.append(clade)

    selected_labels = tuple(_label_tuple(clade, labels) for clade in selected)
    position = {label: index for index, label in enumerate(labels)}
    selected_indices = tuple(
        tuple(position[label] for label in clade) for clade in selected_labels
    )
    full_rows = tuple(
        (
            mode,
            tuple(
                sorted(
                    (_label_tuple(clade, labels) for clade in full[mode]),
                    key=lambda item: (len(item), tuple(str(value) for value in item)),
                )
            ),
        )
        for mode in modes
    )
    confidence_rows = tuple(
        (
            mode,
            tuple(
                sorted(
                    (
                        (_label_tuple(clade, labels), count / float(bootstrap_replicates))
                        for clade, count in counters[mode].items()
                    ),
                    key=lambda item: (
                        -item[1],
                        len(item[0]),
                        tuple(str(value) for value in item[0]),
                    ),
                )
            ),
        )
        for mode in modes
    )
    return StableRNJSupportSelection(
        terminal_labels=labels,
        modes=modes,
        bootstrap_replicates=int(bootstrap_replicates),
        confidence_threshold=float(confidence_threshold),
        tolerance_factor=float(tolerance_factor),
        maximum_support_size=maximum_support_size,
        maximum_support_count=int(maximum_support_count),
        selected_support_labels=selected_labels,
        selected_support_indices=selected_indices,
        selected_confidences=tuple(float(confidences[item]) for item in selected),
        full_data_clades_by_mode=full_rows,
        bootstrap_confidences_by_mode=confidence_rows,
    )
