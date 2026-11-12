"""Honest three-stage cross-fitting for adaptive latent-tree selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from terminal_case33.estimation.baseline import fit_complete_rnj_baseline
from terminal_case33.graph.rooted_hierarchy import rooted_tree_from_clades
from terminal_case33.models.network import TerminalizedNetwork
from terminal_case33.pipeline.ac_likelihood import rank_candidate_trees_ac
from terminal_case33.pipeline.peripheral_edge_proposals import (
    PeripheralProposalResult,
    propose_peripheral_clusters,
)
from terminal_case33.pipeline.rnj_candidates import RNJCandidate
from terminal_case33.pipeline.two_level_aggregation import (
    run_ordered_two_level_aggregation,
)
from terminal_case33.pipeline.uncertainty_aware_rnj import (
    estimate_uncertainty_aware_rnj,
)


Clade = frozenset[int]


@dataclass(frozen=True)
class ThreeStageSplit:
    """Disjoint scenario indices for discovery, estimation, and validation."""

    fold: int
    discovery_indices: tuple[int, ...]
    estimation_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


@dataclass(frozen=True)
class CrossFitFoldResult:
    """Candidate generation and held-out selection result for one rotation."""

    split: ThreeStageSplit
    candidate_count: int
    candidate_sources: tuple[str, ...]
    selected_source: str
    selected_clades: frozenset[Clade]
    selection_score: float
    discovery_proposals: PeripheralProposalResult


@dataclass(frozen=True)
class ThreeStageCrossFitResult:
    """Topology selected by voting over honest three-stage rotations."""

    selected_candidate: RNJCandidate
    selected_clades: frozenset[Clade]
    folds: tuple[CrossFitFoldResult, ...]
    clade_selection_support: dict[Clade, float]
    topology_selection_support: float


def three_stage_splits(scenario_count: int) -> tuple[ThreeStageSplit, ...]:
    """Create three cyclic, disjoint discovery/estimate/validation splits."""

    if scenario_count < 3:
        raise ValueError("three-stage cross-fitting requires at least three scenarios")
    splits = []
    for fold in range(3):
        groups = [
            tuple(index for index in range(scenario_count) if (index + fold) % 3 == group)
            for group in range(3)
        ]
        if not all(groups):
            raise ValueError("every cross-fitting stage must receive a scenario")
        splits.append(
            ThreeStageSplit(
                fold=fold,
                discovery_indices=groups[0],
                estimation_indices=groups[1],
                validation_indices=groups[2],
            )
        )
    return tuple(splits)


def _candidate_from_clades(
    clades: frozenset[Clade],
    terminals: tuple[int, ...],
    root_bus: int,
    source: str,
    include_hidden_root_stem: bool,
) -> RNJCandidate:
    tree = rooted_tree_from_clades(
        clades,
        terminals,
        root_bus,
        include_hidden_root_stem=include_hidden_root_stem,
    )
    return RNJCandidate(
        tree=tree,
        distance_mode=source,
        tolerance_factor=0.0,
        clades=clades,
    )


def _has_hidden_root_stem(candidate: RNJCandidate) -> bool:
    root_neighbors = {
        int(right) if int(left) == candidate.tree.root else int(left)
        for left, right, _weight in candidate.tree.edges
        if candidate.tree.root in {int(left), int(right)}
    }
    return bool(
        len(root_neighbors) == 1
        and next(iter(root_neighbors)) not in set(candidate.tree.terminals)
    )


def identify_topology_three_stage_crossfit(
    scenarios: list[dict],
    reference_net: TerminalizedNetwork,
    pq_noise_relative_std: float = 0.005,
    voltage_noise_relative_std: float = 0.0002,
    distance_mode: str = "RX_75R_25X",
    tolerance_factor: float = 0.16,
    discovery_bootstrap_replicates: int = 4,
    uncertainty_bootstrap_replicates: int = 6,
    seed: int = 0,
) -> ThreeStageCrossFitResult:
    """Run cyclic discovery/estimation/AC-validation without stage reuse.

    Discovery data choose only peripheral regions. Estimation data fit R/X,
    uncertainty-aware RNJ, and joint pseudo-parent candidates. Validation data
    are touched only by the fixed-tree EIV/AC score. The final discrete tree is
    one of the fold winners, selected by full-topology vote.
    """

    splits = three_stage_splits(len(scenarios))
    fold_results = []
    winners: list[tuple[RNJCandidate, float]] = []
    for split in splits:
        discovery = [scenarios[index] for index in split.discovery_indices]
        estimation = [scenarios[index] for index in split.estimation_indices]
        validation = [scenarios[index] for index in split.validation_indices]
        proposals = propose_peripheral_clusters(
            discovery,
            reference_net.root_bus,
            preprocessing_recipe={"name": "daily_demean", "kind": "demean"},
            distance_mode=distance_mode,
            bootstrap_replicates=discovery_bootstrap_replicates,
            maximum_cluster_size=5,
            minimum_support=0.60,
            minimum_boundary_margin=0.0,
            minimum_edge_length_ratio=0.0,
            block_fraction=0.50,
            pq_extra_noise_rel=0.0,
            voltage_extra_noise_rel=0.0,
            seed=seed + 101 * split.fold,
        )
        base = fit_complete_rnj_baseline(
            estimation,
            reference_net.root_bus,
            preprocessing="daily_demean",
            distance_mode=distance_mode,
            constraint_mode="ordered",
            tolerance_factor=tolerance_factor,
        )
        base_candidate = RNJCandidate(
            tree=base.tree,
            distance_mode="crossfit_ordered_base",
            tolerance_factor=base.tolerance_factor,
            clades=base.rooted_clades,
        )
        uncertain = estimate_uncertainty_aware_rnj(
            estimation,
            reference_net.root_bus,
            distance_mode=distance_mode,
            tolerance_factor=tolerance_factor,
            bootstrap_replicates=uncertainty_bootstrap_replicates,
            seed=seed + 1009 * split.fold,
        )
        uncertain_candidate = RNJCandidate(
            tree=uncertain.uncertain.tree,
            distance_mode="crossfit_uncertainty_rnj",
            tolerance_factor=tolerance_factor,
            clades=uncertain.uncertain_clades,
        )
        candidates = [base_candidate, uncertain_candidate]
        include_stem = _has_hidden_root_stem(base_candidate)
        for mode, clusters in proposals.clusters_by_mode.items():
            if not clusters:
                continue
            aggregation = run_ordered_two_level_aggregation(
                estimation,
                reference_net.root_bus,
                list(clusters),
                distance_mode=distance_mode,
                tolerance_factor=tolerance_factor,
                pseudo_voltage_mode="joint_latent_vsq",
            )
            for aggregation_mode, clades in aggregation.candidate_clades_by_mode.items():
                candidates.append(
                    _candidate_from_clades(
                        clades,
                        base.terminals,
                        reference_net.root_bus,
                        f"crossfit_{mode}__{aggregation_mode}__joint_parent",
                        include_stem,
                    )
                )
        unique = {candidate.clades: candidate for candidate in candidates}
        rerank = rank_candidate_trees_ac(
            list(unique.values()),
            estimation,
            validation,
            reference_net,
            pq_noise_relative_std=pq_noise_relative_std,
            voltage_noise_relative_std=voltage_noise_relative_std,
        )
        winner = rerank.selected.candidate
        winners.append((winner, float(rerank.selected.selection_score)))
        fold_results.append(
            CrossFitFoldResult(
                split=split,
                candidate_count=len(unique),
                candidate_sources=tuple(
                    sorted(candidate.distance_mode for candidate in unique.values())
                ),
                selected_source=winner.distance_mode,
                selected_clades=winner.clades,
                selection_score=float(rerank.selected.selection_score),
                discovery_proposals=proposals,
            )
        )

    grouped: dict[frozenset[Clade], list[tuple[RNJCandidate, float]]] = {}
    for candidate, score in winners:
        grouped.setdefault(candidate.clades, []).append((candidate, score))
    def mean_topology_agreement(item: tuple[RNJCandidate, float]) -> float:
        clades = item[0].clades
        similarities = []
        for other, _score in winners:
            union = clades | other.clades
            similarities.append(len(clades & other.clades) / max(len(union), 1))
        return float(np.mean(similarities))

    selected_candidate, _selected_score = max(
        winners,
        key=lambda item: (mean_topology_agreement(item), -item[1]),
    )
    selected_group = grouped[selected_candidate.clades]
    all_selected_clades = set().union(*(candidate.clades for candidate, _ in winners))
    clade_support = {
        clade: float(np.mean([clade in candidate.clades for candidate, _ in winners]))
        for clade in all_selected_clades
    }
    return ThreeStageCrossFitResult(
        selected_candidate=selected_candidate,
        selected_clades=selected_candidate.clades,
        folds=tuple(fold_results),
        clade_selection_support=clade_support,
        topology_selection_support=float(len(selected_group) / len(winners)),
    )
