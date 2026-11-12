"""Block-disjoint three-stage cross-fitting for small scenario banks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from terminal_case33.estimation.baseline import fit_complete_rnj_baseline
from terminal_case33.models.network import TerminalizedNetwork
from terminal_case33.pipeline.ac_likelihood import rank_candidate_trees_ac
from terminal_case33.pipeline.peripheral_edge_proposals import (
    PeripheralProposalResult,
    propose_peripheral_clusters,
)
from terminal_case33.pipeline.rnj_candidates import RNJCandidate
from terminal_case33.pipeline.three_stage_crossfit import (
    _candidate_from_clades,
    _has_hidden_root_stem,
)
from terminal_case33.pipeline.two_level_aggregation import (
    run_ordered_two_level_aggregation,
)
from terminal_case33.pipeline.uncertainty_aware_rnj import (
    estimate_uncertainty_aware_rnj,
)


Clade = frozenset[int]


@dataclass(frozen=True)
class BlockedCrossFitFoldResult:
    """One rotation using disjoint time blocks from every scenario."""

    fold: int
    stage_sample_counts: tuple[int, int, int]
    candidate_count: int
    candidate_sources: tuple[str, ...]
    selected_source: str
    selected_clades: frozenset[Clade]
    selection_score: float
    discovery_proposals: PeripheralProposalResult


@dataclass(frozen=True)
class BlockedThreeStageCrossFitResult:
    """Topology selected by voting over three block-disjoint rotations."""

    selected_candidate: RNJCandidate
    selected_clades: frozenset[Clade]
    folds: tuple[BlockedCrossFitFoldResult, ...]
    clade_selection_support: dict[Clade, float]
    topology_selection_support: float


def blocked_stage_indices(
    count: int,
    block_length: int,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition rows into rotating contiguous blocks for the three stages."""

    if count < 3 or block_length <= 0:
        raise ValueError("count must be at least three and block_length positive")
    block = np.arange(count, dtype=int) // min(block_length, count)
    groups = tuple(
        np.flatnonzero((block + int(fold)) % 3 == stage)
        for stage in range(3)
    )
    if not all(len(indices) for indices in groups):
        raise ValueError("block_length leaves an empty cross-fitting stage")
    return groups


def _subset_scenario(scenario: dict, indices: np.ndarray, label: str) -> dict:
    def frame(name: str) -> pd.DataFrame:
        return scenario[name].iloc[indices].reset_index(drop=True).copy()

    return {
        "name": f"{scenario['name']}__{label}",
        "P_terminal": frame("P_terminal"),
        "Q_terminal": frame("Q_terminal"),
        "V_terminal": frame("V_terminal"),
        "root_voltage": scenario["root_voltage"].iloc[indices].reset_index(drop=True).copy(),
        "drop_target": frame("drop_target"),
    }


def _fold_stage_scenarios(
    scenarios: list[dict],
    block_length: int,
    fold: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    stages: tuple[list[dict], list[dict], list[dict]] = ([], [], [])
    for scenario in scenarios:
        groups = blocked_stage_indices(len(scenario["P_terminal"]), block_length, fold)
        for stage, indices in enumerate(groups):
            stages[stage].append(
                _subset_scenario(scenario, indices, f"fold_{fold}_stage_{stage}")
            )
    return stages


def identify_topology_blocked_three_stage_crossfit(
    scenarios: list[dict],
    reference_net: TerminalizedNetwork,
    pq_noise_relative_std: float = 0.005,
    voltage_noise_relative_std: float = 0.0002,
    distance_mode: str = "RX_75R_25X",
    tolerance_factor: float = 0.16,
    temporal_block_length: int = 8,
    discovery_bootstrap_replicates: int = 4,
    uncertainty_bootstrap_replicates: int = 6,
    seed: int = 0,
) -> BlockedThreeStageCrossFitResult:
    """Cross-fit using nonoverlapping time blocks from every operating day."""

    if len(scenarios) < 1:
        raise ValueError("at least one scenario is required")
    fold_results = []
    winners: list[tuple[RNJCandidate, float]] = []
    for fold in range(3):
        discovery, estimation, validation = _fold_stage_scenarios(
            scenarios,
            temporal_block_length,
            fold,
        )
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
            seed=seed + 101 * fold,
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
            distance_mode="blocked_crossfit_ordered_base",
            tolerance_factor=base.tolerance_factor,
            clades=base.rooted_clades,
        )
        uncertain = estimate_uncertainty_aware_rnj(
            estimation,
            reference_net.root_bus,
            distance_mode=distance_mode,
            tolerance_factor=tolerance_factor,
            bootstrap_replicates=uncertainty_bootstrap_replicates,
            block_length=max(2, temporal_block_length // 2),
            seed=seed + 1009 * fold,
        )
        candidates = [
            base_candidate,
            RNJCandidate(
                tree=uncertain.uncertain.tree,
                distance_mode="blocked_crossfit_uncertainty_rnj",
                tolerance_factor=tolerance_factor,
                clades=uncertain.uncertain_clades,
            ),
        ]
        include_stem = _has_hidden_root_stem(base_candidate)
        clusters = proposals.clusters_by_mode.get("nj_rg_consensus", ())
        if clusters:
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
                        f"blocked_crossfit_nj_rg__{aggregation_mode}__joint_parent",
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
            BlockedCrossFitFoldResult(
                fold=fold,
                stage_sample_counts=(
                    sum(len(item["P_terminal"]) for item in discovery),
                    sum(len(item["P_terminal"]) for item in estimation),
                    sum(len(item["P_terminal"]) for item in validation),
                ),
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
    all_clades = set().union(*(candidate.clades for candidate, _ in winners))
    clade_support = {
        clade: float(np.mean([clade in candidate.clades for candidate, _ in winners]))
        for clade in all_clades
    }
    return BlockedThreeStageCrossFitResult(
        selected_candidate=selected_candidate,
        selected_clades=selected_candidate.clades,
        folds=tuple(fold_results),
        clade_selection_support=clade_support,
        topology_selection_support=float(len(selected_group) / len(winners)),
    )
