"""High-level AC model selection and quartet-supported hidden-edge validation."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from terminal_case33.estimation.multiscenario import fit_projected_sensitivity
from terminal_case33.graph.quartet_significance import (
    QuartetCladeEvidence,
    bootstrap_quartet_significance,
)
from terminal_case33.models.network import TerminalizedNetwork
from terminal_case33.pipeline.ac_likelihood import (
    ACRerankResult,
    CandidateACScore,
    rank_candidate_trees_ac,
    split_representative_scenarios,
)
from terminal_case33.pipeline.rnj_candidates import (
    RNJCandidate,
    generate_rnj_candidates,
)


@dataclass(frozen=True)
class ValidatedTopologyResult:
    """AC-selected topology before and after significant-edge contraction."""

    ac_rerank: ACRerankResult
    raw_clades: frozenset[frozenset[int]]
    quartet_clades: frozenset[frozenset[int]]
    validated_clades: frozenset[frozenset[int]]
    quartet_evidence: dict[frozenset[int], QuartetCladeEvidence]
    clade_edge_strength: dict[frozenset[int], float]
    edge_strength_threshold: float


def clade_edge_strengths(score: CandidateACScore) -> dict[frozenset[int], float]:
    """Map every hidden clade to its fitted squared-voltage edge strength."""

    directed = nx.DiGraph()
    directed.add_edges_from(score.edge_fit.oriented_edges)
    terminals = set(score.candidate.tree.terminals)
    strengths: dict[frozenset[int], float] = {}
    for index, (_parent, child) in enumerate(score.edge_fit.oriented_edges):
        descendants = frozenset(
            terminal
            for terminal in terminals
            if terminal == child or nx.has_path(directed, child, terminal)
        )
        if 1 < len(descendants) < len(terminals):
            strengths[descendants] = float(
                score.edge_fit.r_edge_coefficients[index]
                + score.edge_fit.x_edge_coefficients[index]
            )
    return strengths


def contract_unsupported_hidden_edges(
    score: CandidateACScore,
    quartet_evidence: dict[frozenset[int], QuartetCladeEvidence],
    minimum_edge_strength_ratio: float = 0.02,
    quartet_contradiction_effect: float = 0.05,
) -> tuple[set[frozenset[int]], set[frozenset[int]], dict[frozenset[int], float], float]:
    """Keep effective hidden edges unless quartets significantly contradict them.

    Failure to establish a positive quartet gap is not evidence that an edge is
    absent, especially with little data. An edge is therefore removed only when
    its fitted impedance collapses to zero or the quartet upper confidence bound
    is materially negative. Positive quartet significance remains a diagnostic.
    """

    if minimum_edge_strength_ratio < 0.0:
        raise ValueError("minimum_edge_strength_ratio must be nonnegative")
    if quartet_contradiction_effect < 0.0:
        raise ValueError("quartet_contradiction_effect must be nonnegative")
    strengths = clade_edge_strengths(score)
    positive = np.array([value for value in strengths.values() if value > 0.0])
    reference = float(np.median(positive)) if positive.size else 0.0
    threshold = minimum_edge_strength_ratio * reference
    quartet_clades = {
        clade
        for clade in score.candidate.clades
        if quartet_evidence.get(clade) is not None and quartet_evidence[clade].accepted
    }
    validated = {
        clade
        for clade in score.candidate.clades
        if strengths.get(clade, 0.0) > threshold
        and (
            quartet_evidence.get(clade) is None
            or quartet_evidence[clade].upper_confidence_bound >= -quartet_contradiction_effect
        )
    }
    return quartet_clades, validated, strengths, threshold


def select_and_validate_topology(
    scenarios: list[dict],
    reference_net: TerminalizedNetwork,
    quartet_candidate_clades: set[frozenset[int]] | None = None,
    extra_candidates: list[RNJCandidate] | tuple[RNJCandidate, ...] | None = None,
    precomputed_quartet_evidence: (
        dict[frozenset[int], QuartetCladeEvidence] | None
    ) = None,
    distance_modes: tuple[str, ...] = (
        "R",
        "X",
        "RX_equal_normalized",
        "RX_75R_25X",
    ),
    tolerance_factors: tuple[float, ...] = (0.12, 0.16, 0.20, 0.24, 0.28, 0.32),
    pq_noise_relative_std: float = 0.005,
    voltage_noise_relative_std: float = 0.0002,
    bic_complexity_weight: float = 1.0,
    quartet_replicates: int = 20,
    quartet_confidence: float = 0.90,
    quartet_positive_support: float = 0.80,
    quartet_minimum_effect: float = 0.002,
    quartet_distance_mode: str = "RX_75R_25X",
    quartet_preprocess_recipe: dict | None = None,
    minimum_edge_strength_ratio: float = 0.02,
    quartet_contradiction_effect: float = 0.05,
    validation_scenario_count: int | None = None,
    seed: int = 0,
) -> ValidatedTopologyResult:
    """Generate a shared candidate pool, AC-rerank, and validate hidden edges.

    ``extra_candidates`` may contain aggregation or GTLS proposals. They
    are deduplicated with the ordinary constrained-RNJ grid by rooted clades so
    every proposal is compared under the same fixed-tree fit and held-out AC
    likelihood. Precomputed quartet records are reused and only missing clades
    are bootstrapped.
    """

    if len(scenarios) < 2:
        raise ValueError("topology validation requires at least two scenarios")
    training, validation, _validation_indices = split_representative_scenarios(
        scenarios,
        validation_scenario_count,
    )
    terminals = [int(column) for column in scenarios[0]["P_terminal"].columns]
    r_matrix, x_matrix, _r2, _condition = fit_projected_sensitivity(
        training,
        constraint_mode="ordered",
    )
    generated = generate_rnj_candidates(
        r_matrix,
        x_matrix,
        terminals,
        reference_net.root_bus,
        distance_modes,
        tolerance_factors,
    )
    unique_candidates = {candidate.clades: candidate for candidate in generated}
    for candidate in extra_candidates or ():
        if set(candidate.tree.terminals) != set(terminals):
            raise ValueError("extra candidate terminals do not match scenarios")
        if candidate.tree.root != reference_net.root_bus:
            raise ValueError("extra candidate root does not match reference network")
        unique_candidates.setdefault(candidate.clades, candidate)
    candidates = list(unique_candidates.values())
    rerank = rank_candidate_trees_ac(
        candidates,
        training,
        validation,
        reference_net,
        pq_noise_relative_std=pq_noise_relative_std,
        voltage_noise_relative_std=voltage_noise_relative_std,
        complexity_weight=bic_complexity_weight,
    )
    raw_clades = set(rerank.selected.candidate.clades)
    tested_clades = raw_clades | set(quartet_candidate_clades or ())
    tested_clades = {
        clade for clade in tested_clades if 1 < len(clade) < len(terminals)
    }
    evidence = dict(precomputed_quartet_evidence or {})
    missing_clades = tested_clades - set(evidence)
    if missing_clades:
        _accepted, new_evidence = bootstrap_quartet_significance(
            training,
            missing_clades,
            reference_net.root_bus,
            distance_mode=quartet_distance_mode,
            replicates=quartet_replicates,
            confidence_level=quartet_confidence,
            minimum_positive_support=quartet_positive_support,
            minimum_effect=quartet_minimum_effect,
            preprocess_recipe=quartet_preprocess_recipe,
            seed=seed,
        )
        evidence.update(new_evidence)
    quartet_clades, validated, strengths, threshold = contract_unsupported_hidden_edges(
        rerank.selected,
        evidence,
        minimum_edge_strength_ratio=minimum_edge_strength_ratio,
        quartet_contradiction_effect=quartet_contradiction_effect,
    )
    return ValidatedTopologyResult(
        ac_rerank=rerank,
        raw_clades=frozenset(raw_clades),
        quartet_clades=frozenset(quartet_clades),
        validated_clades=frozenset(validated),
        quartet_evidence=evidence,
        clade_edge_strength=strengths,
        edge_strength_threshold=threshold,
    )
