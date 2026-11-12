"""Unified rooted topology identification without frequency-domain filtering.

Ordered constrained sensitivity fitting provides the deterministic RNJ base.
Root-placed NJ and rooted RG propose stable peripheral regions, a one-layer
aggregate/decompose pass produces additional candidates, GTLS models P/Q
errors, and held-out AC likelihood with quartet checks selects and validates
the final hidden tree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from terminal_case33.estimation.baseline import (
    CompleteBaselineResult,
    fit_complete_rnj_baseline,
)
from terminal_case33.graph.quartet_significance import (
    QuartetCladeEvidence,
    bootstrap_quartet_significance,
)
from terminal_case33.graph.rooted_hierarchy import PseudoCluster, rooted_tree_from_clades
from terminal_case33.models.network import TerminalizedNetwork
from terminal_case33.pipeline.ac_likelihood import (
    representative_validation_count,
    split_representative_scenarios,
)
from terminal_case33.pipeline.gtls_topology_posterior import (
    GTLSTopologyPosterior,
    infer_gtls_topology_posterior,
)
from terminal_case33.pipeline.peripheral_edge_proposals import (
    PeripheralProposalResult,
    propose_peripheral_clusters,
)
from terminal_case33.pipeline.rnj_candidates import RNJCandidate
from terminal_case33.pipeline.topology_validation import (
    ValidatedTopologyResult,
    select_and_validate_topology,
)
from terminal_case33.pipeline.two_level_aggregation import (
    OrderedAggregationResult,
    run_ordered_two_level_aggregation,
)


Clade = frozenset[int]
DAILY_DEMEAN_RECIPE = {"name": "daily_demean", "kind": "demean"}


@dataclass(frozen=True)
class UnifiedTopologyConfig:
    """Label-free thresholds for the unified noisy-measurement pipeline."""

    validation_scenario_count: int | None = None
    gtls_validation_scenario_count: int = 1
    quartet_replicates: int = 20
    quartet_confidence: float = 0.90
    quartet_positive_support: float = 0.80
    quartet_minimum_effect: float = 0.002
    gtls_shrinkage_to_ols: float = 0.99
    gtls_temporal_block_length: int = 12
    gtls_posterior_temperature: float = 1.0
    gtls_map_probability_threshold: float = 0.50
    distance_mode: str = "RX_75R_25X"
    rnj_tolerance_factor: float = 0.16
    gtls_clade_confirmation_threshold: float = 0.50
    aggregation_max_cluster_size: int = 5
    aggregation_deembedding_weight: float = 0.50
    aggregation_pseudo_voltage_mode: str = "deembedded_vsq"
    aggregation_local_tolerance_factor: float = 0.40
    aggregation_block_fraction: float = 0.70
    enable_ordered_aggregation: bool = True
    local_refit_maximum_nrmse: float = 1.25
    local_refit_minimum_stability: float = 0.50
    enable_nj_edge_aggregation: bool = True
    nj_edge_bootstrap_replicates: int = 8
    nj_edge_support_threshold: float = 0.875
    nj_edge_minimum_boundary_margin: float = 0.10
    nj_edge_minimum_length_ratio: float = 0.02
    nj_edge_rg_tolerance: float = 0.03
    nj_edge_require_external_confirmation: bool = True
    nj_edge_candidate_beam_width: int = 6
    quartet_contradiction_effect: float = 0.05
    minimum_fit_r2: float = 0.80
    maximum_condition_number: float = 1e5


@dataclass(frozen=True)
class UnifiedCladeEvidence:
    """Auditable multi-method evidence for one rooted terminal clade."""

    clade: Clade
    gtls_marginal: float
    quartet_estimate: float
    quartet_lower_bound: float
    quartet_upper_bound: float
    quartet_positive_support: float
    quartet_accepted: bool
    in_ac_raw: bool
    in_ac_validated: bool
    in_gtls_map: bool
    in_ordered_base: bool
    source_support_count: int
    diagnostic_support_score: float
    materially_contradicted: bool
    stable_for_aggregation: bool

    def to_dict(self) -> dict:
        """Return a JSON-serializable evidence record."""

        record = asdict(self)
        record["clade"] = sorted(self.clade)
        return record


@dataclass(frozen=True)
class UnifiedTopologyResult:
    """Final clades plus every branch needed to audit the decision."""

    final_clades: frozenset[Clade]
    pre_quartet_clades: frozenset[Clade]
    selection_source: str
    ac_validated: ValidatedTopologyResult
    gtls_posterior: GTLSTopologyPosterior
    base: CompleteBaselineResult
    peripheral_proposals: PeripheralProposalResult | None
    detector_aggregations: dict[str, OrderedAggregationResult]
    confirmed_detector_clusters: dict[str, tuple[PseudoCluster, ...]]
    rejected_detector_clusters: dict[str, tuple[str, ...]]
    clade_evidence: dict[Clade, UnifiedCladeEvidence]
    proposed_aggregation_clades: frozenset[Clade]
    stable_aggregation_clades: frozenset[Clade]
    applied_aggregation_clades: frozenset[Clade]
    aggregation_rerun: bool
    base_trusted: bool
    gtls_map_trusted: bool

    def summary(self) -> dict:
        """Return compact JSON-compatible selection diagnostics."""

        return {
            "selection_source": self.selection_source,
            "final_clades": [sorted(clade) for clade in sorted_clades(self.final_clades)],
            "pre_quartet_clades": [
                sorted(clade) for clade in sorted_clades(self.pre_quartet_clades)
            ],
            "proposed_aggregation_clades": [
                sorted(clade) for clade in sorted_clades(self.proposed_aggregation_clades)
            ],
            "stable_aggregation_clades": [
                sorted(clade) for clade in sorted_clades(self.stable_aggregation_clades)
            ],
            "applied_aggregation_clades": [
                sorted(clade) for clade in sorted_clades(self.applied_aggregation_clades)
            ],
            "aggregation_rerun": self.aggregation_rerun,
            "base_trusted": self.base_trusted,
            "base_r2_score": self.base.r2_score,
            "base_condition_number": self.base.condition_number,
            "gtls_map_trusted": self.gtls_map_trusted,
            "peripheral_proposal_counts": (
                {
                    mode: len(clusters)
                    for mode, clusters in self.peripheral_proposals.clusters_by_mode.items()
                }
                if self.peripheral_proposals is not None
                else {}
            ),
            "confirmed_detector_cluster_counts": {
                mode: len(clusters)
                for mode, clusters in self.confirmed_detector_clusters.items()
            },
            "detector_aggregation_candidate_counts": {
                mode: len(result.candidate_clades_by_mode)
                for mode, result in self.detector_aggregations.items()
            },
            "rejected_detector_clusters": {
                key: list(value) for key, value in self.rejected_detector_clusters.items()
            },
            "gtls_map_probability": self.gtls_posterior.selected.posterior_probability,
            "ac_training_scenario_count": self.ac_validated.ac_rerank.training_scenario_count,
            "ac_validation_scenario_count": self.ac_validated.ac_rerank.validation_scenario_count,
            "ac_refinement_accepted": (
                self.ac_validated.ac_rerank.selected.edge_fit.ac_refinement_accepted
            ),
            "ac_training_rmse_before": (
                self.ac_validated.ac_rerank.selected.edge_fit.training_ac_rmse_before
            ),
            "ac_training_rmse_after": (
                self.ac_validated.ac_rerank.selected.edge_fit.training_ac_rmse_after
            ),
            "ac_predictive_nll": (
                self.ac_validated.ac_rerank.selected.standardized_negative_log_likelihood
            ),
        }


def sorted_clades(clades: set[Clade] | frozenset[Clade]) -> list[Clade]:
    """Return clades in deterministic size and node order."""

    return sorted(clades, key=lambda value: (len(value), tuple(sorted(value))))


def _finite_or_zero(value: float) -> float:
    """Return a finite scalar or zero."""

    return float(value) if np.isfinite(value) else 0.0


def build_clade_evidence(
    clades: set[Clade],
    *,
    ac_raw_clades: set[Clade],
    ac_validated_clades: set[Clade],
    gtls_map_clades: set[Clade],
    gtls_marginals: dict[Clade, float],
    ordered_base_clades: set[Clade],
    quartet_evidence: dict[Clade, QuartetCladeEvidence],
    gtls_confirmation_threshold: float = 0.50,
    quartet_contradiction_effect: float = 0.05,
) -> dict[Clade, UnifiedCladeEvidence]:
    """Fuse base RNJ, GTLS, AC, and quartet evidence with explicit gates."""

    if not 0.0 <= gtls_confirmation_threshold <= 1.0:
        raise ValueError("gtls_confirmation_threshold must be in [0, 1]")
    if quartet_contradiction_effect < 0.0:
        raise ValueError("quartet_contradiction_effect must be nonnegative")

    fused: dict[Clade, UnifiedCladeEvidence] = {}
    for clade in sorted_clades(clades):
        quartet = quartet_evidence.get(clade)
        gtls = float(np.clip(gtls_marginals.get(clade, 0.0), 0.0, 1.0))
        q_estimate = _finite_or_zero(quartet.estimate) if quartet else 0.0
        q_lower = _finite_or_zero(quartet.lower_confidence_bound) if quartet else 0.0
        q_upper = _finite_or_zero(quartet.upper_confidence_bound) if quartet else 0.0
        q_support = (
            float(np.clip(quartet.positive_support, 0.0, 1.0))
            if quartet is not None
            else 0.0
        )
        q_accepted = bool(quartet is not None and quartet.accepted)
        contradicted = bool(
            quartet is not None
            and np.isfinite(quartet.upper_confidence_bound)
            and quartet.upper_confidence_bound < -quartet_contradiction_effect
        )
        in_base = clade in ordered_base_clades
        ac_raw = clade in ac_raw_clades
        ac_validated = clade in ac_validated_clades
        gtls_confirmed = gtls >= gtls_confirmation_threshold
        source_count = sum((in_base, gtls_confirmed, ac_validated, q_accepted))
        ac_score = 1.0 if ac_validated else 0.5 if ac_raw else 0.0
        support_score = (
            0.35 * float(in_base)
            + 0.35 * gtls
            + 0.20 * q_support
            + 0.10 * ac_score
        )
        fused[clade] = UnifiedCladeEvidence(
            clade=clade,
            gtls_marginal=gtls,
            quartet_estimate=q_estimate,
            quartet_lower_bound=q_lower,
            quartet_upper_bound=q_upper,
            quartet_positive_support=q_support,
            quartet_accepted=q_accepted,
            in_ac_raw=ac_raw,
            in_ac_validated=ac_validated,
            in_gtls_map=clade in gtls_map_clades,
            in_ordered_base=in_base,
            source_support_count=int(source_count),
            diagnostic_support_score=float(support_score),
            materially_contradicted=contradicted,
            stable_for_aggregation=bool(source_count >= 2 and not contradicted),
        )
    return fused


def _candidate_clade_union(
    posterior: GTLSTopologyPosterior,
    base: CompleteBaselineResult,
    peripheral: PeripheralProposalResult | None = None,
    detector_aggregations: dict[str, OrderedAggregationResult] | None = None,
) -> set[Clade]:
    """Collect every hidden split proposed before quartet validation."""

    clades: set[Clade] = set(base.rooted_clades)
    for score in posterior.ranked:
        clades.update(score.projection.candidate.clades)
    if peripheral is not None:
        clades.update(peripheral.candidate_clades)
    for aggregation in (detector_aggregations or {}).values():
        for candidate in aggregation.candidate_clades_by_mode.values():
            clades.update(candidate)
    terminal_count = len(base.terminals)
    return {clade for clade in clades if 1 < len(clade) < terminal_count}


def _confirmed_peripheral_clusters(
    proposals: PeripheralProposalResult,
    evidence: dict[Clade, UnifiedCladeEvidence],
    require_external_confirmation: bool,
) -> tuple[dict[str, tuple[PseudoCluster, ...]], dict[str, tuple[str, ...]]]:
    """Confirm NJ regions without freezing the detector's internal topology."""

    accepted: dict[str, tuple[PseudoCluster, ...]] = {}
    rejected: dict[str, tuple[str, ...]] = {}
    for mode, clusters in proposals.clusters_by_mode.items():
        mode_accepted = []
        for cluster in clusters:
            record = evidence.get(cluster.members)
            label = "{" + ",".join(str(node) for node in sorted(cluster.members)) + "}"
            reasons = []
            if record is None:
                reasons.append(f"{label}: no fused evidence")
            elif record.materially_contradicted:
                reasons.append(f"{label}: quartet contradiction")
            elif (
                require_external_confirmation
                and mode != "nj_rg_consensus"
                and record.source_support_count < 2
            ):
                reasons.append(
                    f"{label}: NJ-only support has fewer than two external sources"
                )
            if reasons:
                rejected[f"{mode}:{label}"] = tuple(reasons)
            else:
                mode_accepted.append(cluster)
        accepted[mode] = tuple(mode_accepted)
    return accepted, rejected


def _peripheral_cluster_beam(
    proposals: PeripheralProposalResult,
    confirmed: dict[str, tuple[PseudoCluster, ...]],
    maximum_candidates: int,
) -> dict[str, tuple[PseudoCluster, ...]]:
    """Build a small soft-contraction beam without combinatorial subsets."""

    if maximum_candidates <= 0:
        return {}
    consensus = list(confirmed.get("nj_rg_consensus", ()))
    consensus_members = {cluster.members for cluster in consensus}
    nj_only = [
        cluster
        for cluster in confirmed.get("nj_stable", ())
        if cluster.members not in consensus_members
    ]

    def score(cluster: PseudoCluster) -> tuple[float, float, tuple[int, ...]]:
        record = proposals.evidence[cluster.members]
        return (
            float(cluster.confidence),
            float(record.boundary_margin),
            tuple(-node for node in sorted(cluster.members)),
        )

    consensus.sort(key=score, reverse=True)
    nj_only.sort(key=score, reverse=True)
    proposed: list[tuple[str, tuple[PseudoCluster, ...]]] = []
    if consensus:
        proposed.append(("nj_rg_consensus_all", tuple(consensus)))
        proposed.extend(
            (f"nj_rg_consensus_single_{index}", (cluster,))
            for index, cluster in enumerate(consensus)
        )
    if len(nj_only) >= 2:
        proposed.append(("nj_stable_all", tuple(nj_only)))
    proposed.extend(
        (f"nj_stable_single_{index}", (cluster,))
        for index, cluster in enumerate(nj_only)
    )
    unique: dict[tuple[tuple[int, ...], ...], tuple[str, tuple[PseudoCluster, ...]]] = {}
    for name, clusters in proposed:
        key = tuple(sorted(tuple(sorted(cluster.members)) for cluster in clusters))
        unique.setdefault(key, (name, clusters))
    return {name: clusters for name, clusters in list(unique.values())[:maximum_candidates]}


def _extra_candidates(
    base: CompleteBaselineResult,
    posterior: GTLSTopologyPosterior,
    detector_aggregations: dict[str, OrderedAggregationResult],
) -> list[RNJCandidate]:
    """Expose every proposal family to one fixed-tree AC ranker."""

    candidates = [score.projection.candidate for score in posterior.ranked]
    candidates.append(
        RNJCandidate(
            tree=base.tree,
            distance_mode="ordered_base",
            tolerance_factor=base.tolerance_factor,
            clades=base.rooted_clades,
        )
    )
    root_neighbors = {
        int(right) if int(left) == base.root_bus else int(left)
        for left, right, _weight in base.tree.edges
        if base.root_bus in {int(left), int(right)}
    }
    include_hidden_root_stem = bool(
        len(root_neighbors) == 1 and next(iter(root_neighbors)) not in set(base.terminals)
    )
    for detector, aggregation in detector_aggregations.items():
        for mode, clades in aggregation.candidate_clades_by_mode.items():
            tree = rooted_tree_from_clades(
                clades,
                base.terminals,
                base.root_bus,
                include_hidden_root_stem=include_hidden_root_stem,
            )
            candidates.append(
                RNJCandidate(
                    tree=tree,
                    distance_mode=f"{detector}__{mode}",
                    tolerance_factor=0.0,
                    clades=clades,
                )
            )
    return candidates


def _selected_ac_source(
    selected_clades: set[Clade],
    base: CompleteBaselineResult,
    posterior: GTLSTopologyPosterior,
    detector_aggregations: dict[str, OrderedAggregationResult],
) -> tuple[str, set[Clade]]:
    """Describe the AC-selected proposal family and applied contractions."""

    for detector, aggregation in detector_aggregations.items():
        for mode, clades in aggregation.candidate_clades_by_mode.items():
            if selected_clades == set(clades):
                return (
                    f"ac_{detector}__{mode}",
                    {cluster.members for cluster in aggregation.clusters},
                )
    if selected_clades == set(base.rooted_clades):
        return "ac_ordered_base", set()
    if any(
        selected_clades == set(score.projection.candidate.clades)
        for score in posterior.ranked
    ):
        return "ac_gtls_ols_pool", set()
    return "ac_rnj_grid", set()


def _no_material_contradictions(
    clades: set[Clade] | frozenset[Clade],
    evidence: dict[Clade, UnifiedCladeEvidence],
) -> bool:
    return not any(
        evidence.get(clade) is not None and evidence[clade].materially_contradicted
        for clade in clades
    )


def _validate_config(config: UnifiedTopologyConfig, scenario_count: int) -> None:
    validation_count = (
        representative_validation_count(scenario_count)
        if config.validation_scenario_count is None
        else config.validation_scenario_count
    )
    if not 1 <= validation_count < scenario_count:
        raise ValueError("validation_scenario_count must lie between 1 and len(scenarios)-1")
    training_scenario_count = scenario_count - validation_count
    if not 1 <= config.gtls_validation_scenario_count < training_scenario_count:
        raise ValueError(
            "gtls_validation_scenario_count must leave at least one outer-training scenario"
        )
    if config.quartet_replicates <= 0:
        raise ValueError("quartet_replicates must be positive")
    if not 0.0 < config.quartet_confidence < 1.0:
        raise ValueError("quartet_confidence must be in (0, 1)")
    if not 0.0 <= config.quartet_positive_support <= 1.0:
        raise ValueError("quartet_positive_support must be in [0, 1]")
    if config.quartet_minimum_effect < 0.0:
        raise ValueError("quartet_minimum_effect must be nonnegative")
    if not 0.0 <= config.gtls_shrinkage_to_ols <= 1.0:
        raise ValueError("gtls_shrinkage_to_ols must be in [0, 1]")
    if not 0.0 <= config.gtls_map_probability_threshold <= 1.0:
        raise ValueError("gtls_map_probability_threshold must be in [0, 1]")
    if config.rnj_tolerance_factor < 0.0:
        raise ValueError("rnj_tolerance_factor must be nonnegative")
    if config.aggregation_max_cluster_size < 2:
        raise ValueError("aggregation_max_cluster_size must be at least two")
    if not 0.0 <= config.aggregation_deembedding_weight <= 1.0:
        raise ValueError("aggregation_deembedding_weight must be in [0, 1]")
    if config.aggregation_pseudo_voltage_mode not in {
        "deembedded_vsq",
        "joint_latent_vsq",
        "regularized_joint_vsq",
    }:
        raise ValueError("invalid aggregation_pseudo_voltage_mode")
    if config.aggregation_local_tolerance_factor < 0.0:
        raise ValueError("aggregation_local_tolerance_factor must be nonnegative")
    if not 0.5 <= config.aggregation_block_fraction < 1.0:
        raise ValueError("aggregation_block_fraction must be in [0.5, 1)")
    if config.local_refit_maximum_nrmse <= 0.0:
        raise ValueError("local_refit_maximum_nrmse must be positive")
    if not 0.0 <= config.local_refit_minimum_stability <= 1.0:
        raise ValueError("local_refit_minimum_stability must be in [0, 1]")
    if config.nj_edge_bootstrap_replicates <= 0:
        raise ValueError("nj_edge_bootstrap_replicates must be positive")
    if not 0.0 <= config.nj_edge_support_threshold <= 1.0:
        raise ValueError("nj_edge_support_threshold must be in [0, 1]")
    if config.nj_edge_minimum_boundary_margin < 0.0:
        raise ValueError("nj_edge_minimum_boundary_margin must be nonnegative")
    if config.nj_edge_minimum_length_ratio < 0.0:
        raise ValueError("nj_edge_minimum_length_ratio must be nonnegative")
    if config.nj_edge_rg_tolerance < 0.0:
        raise ValueError("nj_edge_rg_tolerance must be nonnegative")
    if config.nj_edge_candidate_beam_width <= 0:
        raise ValueError("nj_edge_candidate_beam_width must be positive")


def identify_topology_unified(
    scenarios: list[dict],
    reference_net: TerminalizedNetwork,
    pq_noise_relative_std: float,
    voltage_noise_relative_std: float,
    config: UnifiedTopologyConfig | None = None,
    seed: int = 0,
) -> UnifiedTopologyResult:
    """Run Ordered-RNJ, NJ/RG aggregation, GTLS, AC, and quartet inference."""

    if len(scenarios) < 3:
        raise ValueError("unified topology identification requires at least three scenarios")
    selected_config = config or UnifiedTopologyConfig()
    _validate_config(selected_config, len(scenarios))
    training, validation, _validation_indices = split_representative_scenarios(
        scenarios,
        selected_config.validation_scenario_count,
    )

    base = fit_complete_rnj_baseline(
        training,
        reference_net.root_bus,
        preprocessing="daily_demean",
        distance_mode=selected_config.distance_mode,
        constraint_mode="ordered",
        tolerance_factor=selected_config.rnj_tolerance_factor,
    )
    posterior = infer_gtls_topology_posterior(
        training,
        root_bus=reference_net.root_bus,
        pq_noise_relative_std=pq_noise_relative_std,
        voltage_noise_relative_std=voltage_noise_relative_std,
        validation_scenario_count=selected_config.gtls_validation_scenario_count,
        shrinkage_to_ols=selected_config.gtls_shrinkage_to_ols,
        temporal_block_length=selected_config.gtls_temporal_block_length,
        posterior_temperature=selected_config.gtls_posterior_temperature,
    )

    peripheral_proposals = None
    if selected_config.enable_nj_edge_aggregation:
        peripheral_proposals = propose_peripheral_clusters(
            training,
            reference_net.root_bus,
            preprocessing_recipe=DAILY_DEMEAN_RECIPE,
            distance_mode=selected_config.distance_mode,
            bootstrap_replicates=selected_config.nj_edge_bootstrap_replicates,
            rg_tolerance=selected_config.nj_edge_rg_tolerance,
            maximum_cluster_size=selected_config.aggregation_max_cluster_size,
            minimum_support=selected_config.nj_edge_support_threshold,
            minimum_boundary_margin=selected_config.nj_edge_minimum_boundary_margin,
            minimum_edge_length_ratio=selected_config.nj_edge_minimum_length_ratio,
            block_fraction=selected_config.aggregation_block_fraction,
            pq_extra_noise_rel=min(0.5 * pq_noise_relative_std, 0.0025),
            voltage_extra_noise_rel=min(0.5 * voltage_noise_relative_std, 0.0001),
            seed=seed + 7_919,
        )

    proposed_clades = _candidate_clade_union(posterior, base, peripheral_proposals)
    if proposed_clades:
        _accepted, preliminary_quartet = bootstrap_quartet_significance(
            training,
            proposed_clades,
            reference_net.root_bus,
            distance_mode=selected_config.distance_mode,
            replicates=selected_config.quartet_replicates,
            confidence_level=selected_config.quartet_confidence,
            minimum_positive_support=selected_config.quartet_positive_support,
            minimum_effect=selected_config.quartet_minimum_effect,
            preprocess_recipe=DAILY_DEMEAN_RECIPE,
            seed=seed,
        )
    else:
        preliminary_quartet = {}

    gtls_map_clades = set(posterior.selected.projection.candidate.clades)
    preliminary_evidence = build_clade_evidence(
        proposed_clades,
        ac_raw_clades=set(),
        ac_validated_clades=set(),
        gtls_map_clades=gtls_map_clades,
        gtls_marginals=posterior.clade_marginals,
        ordered_base_clades=set(base.rooted_clades),
        quartet_evidence=preliminary_quartet,
        gtls_confirmation_threshold=selected_config.gtls_clade_confirmation_threshold,
        quartet_contradiction_effect=selected_config.quartet_contradiction_effect,
    )

    confirmed_detector_clusters: dict[str, tuple[PseudoCluster, ...]] = {}
    rejected_detector_clusters: dict[str, tuple[str, ...]] = {}
    if peripheral_proposals is not None:
        confirmed_detector_clusters, rejected_detector_clusters = (
            _confirmed_peripheral_clusters(
                peripheral_proposals,
                preliminary_evidence,
                selected_config.nj_edge_require_external_confirmation,
            )
        )

    detector_aggregations: dict[str, OrderedAggregationResult] = {}
    if peripheral_proposals is not None and selected_config.enable_ordered_aggregation:
        detector_cache: dict[
            tuple[tuple[int, ...], ...], OrderedAggregationResult
        ] = {}
        cluster_beam = _peripheral_cluster_beam(
            peripheral_proposals,
            confirmed_detector_clusters,
            selected_config.nj_edge_candidate_beam_width,
        )
        for mode, clusters in cluster_beam.items():
            cache_key = tuple(
                sorted(tuple(sorted(cluster.members)) for cluster in clusters)
            )
            aggregation = detector_cache.get(cache_key)
            if aggregation is None:
                aggregation = run_ordered_two_level_aggregation(
                    training,
                    reference_net.root_bus,
                    list(clusters),
                    distance_mode=selected_config.distance_mode,
                    tolerance_factor=selected_config.rnj_tolerance_factor,
                    local_tolerance_factor=selected_config.aggregation_local_tolerance_factor,
                    deembedding_weight=selected_config.aggregation_deembedding_weight,
                    pseudo_voltage_mode=selected_config.aggregation_pseudo_voltage_mode,
                    local_refit_max_condition_number=1e8,
                    local_refit_max_heldout_nrmse=selected_config.local_refit_maximum_nrmse,
                    local_refit_minimum_stability=selected_config.local_refit_minimum_stability,
                )
                detector_cache[cache_key] = aggregation
            detector_aggregations[mode] = aggregation

    final_proposed_clades = _candidate_clade_union(
        posterior,
        base,
        peripheral_proposals,
        detector_aggregations,
    )
    ac_validated = select_and_validate_topology(
        scenarios,
        reference_net,
        quartet_candidate_clades=proposed_clades | final_proposed_clades,
        extra_candidates=_extra_candidates(base, posterior, detector_aggregations),
        precomputed_quartet_evidence=preliminary_quartet,
        pq_noise_relative_std=pq_noise_relative_std,
        voltage_noise_relative_std=voltage_noise_relative_std,
        quartet_replicates=selected_config.quartet_replicates,
        quartet_confidence=selected_config.quartet_confidence,
        quartet_positive_support=selected_config.quartet_positive_support,
        quartet_minimum_effect=selected_config.quartet_minimum_effect,
        quartet_distance_mode=selected_config.distance_mode,
        quartet_preprocess_recipe=DAILY_DEMEAN_RECIPE,
        quartet_contradiction_effect=selected_config.quartet_contradiction_effect,
        validation_scenario_count=len(validation),
        seed=seed,
    )

    all_clades = proposed_clades | final_proposed_clades | set(ac_validated.raw_clades)
    evidence = build_clade_evidence(
        all_clades,
        ac_raw_clades=set(ac_validated.raw_clades),
        ac_validated_clades=set(ac_validated.validated_clades),
        gtls_map_clades=gtls_map_clades,
        gtls_marginals=posterior.clade_marginals,
        ordered_base_clades=set(base.rooted_clades),
        quartet_evidence=ac_validated.quartet_evidence,
        gtls_confirmation_threshold=selected_config.gtls_clade_confirmation_threshold,
        quartet_contradiction_effect=selected_config.quartet_contradiction_effect,
    )
    selected = set(ac_validated.raw_clades)
    source, applied_clusters = _selected_ac_source(
        selected,
        base,
        posterior,
        detector_aggregations,
    )
    stable_clusters = {
        cluster.members
        for clusters in confirmed_detector_clusters.values()
        for cluster in clusters
    }
    base_trusted = bool(
        base.r2_score >= selected_config.minimum_fit_r2
        and np.isfinite(base.condition_number)
        and base.condition_number <= selected_config.maximum_condition_number
        and _no_material_contradictions(base.rooted_clades, evidence)
    )
    gtls_map_trusted = bool(
        posterior.selected.posterior_probability
        >= selected_config.gtls_map_probability_threshold
        and _no_material_contradictions(gtls_map_clades, evidence)
    )
    return UnifiedTopologyResult(
        final_clades=frozenset(ac_validated.validated_clades),
        pre_quartet_clades=frozenset(selected),
        selection_source=source,
        ac_validated=ac_validated,
        gtls_posterior=posterior,
        base=base,
        peripheral_proposals=peripheral_proposals,
        detector_aggregations=detector_aggregations,
        confirmed_detector_clusters=confirmed_detector_clusters,
        rejected_detector_clusters=rejected_detector_clusters,
        clade_evidence=evidence,
        proposed_aggregation_clades=(
            peripheral_proposals.candidate_clades
            if peripheral_proposals is not None
            else frozenset()
        ),
        stable_aggregation_clades=frozenset(stable_clusters),
        applied_aggregation_clades=frozenset(applied_clusters),
        aggregation_rerun=bool(detector_aggregations),
        base_trusted=base_trusted,
        gtls_map_trusted=gtls_map_trusted,
    )
