"""Reusable two-level aggregation for rooted hidden-tree identification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from terminal_case33.estimation.baseline import (
    CompleteBaselineResult,
    fit_complete_rnj_baseline,
)
from terminal_case33.estimation.multiscenario import (
    fit_projected_sensitivity,
    preprocess_scenarios,
)
from terminal_case33.graph.rooted_hierarchy import (
    PseudoCluster,
    aggregate_rooted_scenarios,
    build_local_cluster_scenarios,
    expand_pseudo_clades_with_local_reidentification,
    rooted_clades,
)
from terminal_case33.graph.rooted_neighbor_joining import (
    RootedTreeResult,
    rooted_neighbor_joining,
)
from terminal_case33.graph.sensitivity_geometry import sensitivity_geometry
from terminal_case33.pipeline.pseudo_parent_voltage import (
    JointPseudoParentFit,
    aggregate_rooted_scenarios_joint,
    regularized_joint_fit_options,
)


Clade = frozenset[int]


@dataclass(frozen=True)
class LocalClusterFit:
    """One pseudo-rooted local sensitivity fit and its label-free diagnostics."""

    pseudo_id: int
    members: tuple[int, ...]
    tree: RootedTreeResult
    clades: frozenset[Clade]
    r_matrix: np.ndarray
    x_matrix: np.ndarray
    r2_score: float
    condition_number: float
    heldout_nrmse: float
    clade_stability: float
    selection_score: float
    admissible: bool


@dataclass(frozen=True)
class OrderedAggregationResult:
    """Ordered-OLS backbone aggregation with independently refitted local trees."""

    base: CompleteBaselineResult
    clusters: tuple[PseudoCluster, ...]
    pseudo_members: dict[int, frozenset[int]]
    pseudo: CompleteBaselineResult
    local_fits: dict[int, LocalClusterFit]
    candidate_clades_by_mode: dict[str, frozenset[Clade]]
    condition_improvement: float
    mean_local_nrmse: float
    mean_local_stability: float
    all_local_fits_admissible: bool
    pseudo_voltage_mode: str
    pseudo_voltage_fits: dict[int, JointPseudoParentFit]
    pseudo_voltage_options: dict[str, object]


def _distance_and_depth(
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one additive distance and matching root-depth vector."""
    geometry = sensitivity_geometry(r_matrix, x_matrix, mode)
    return geometry.distance, geometry.root_depths


def _fit_local_tree(
    scenarios: list[dict],
    root_bus: int,
    recipe: dict,
    distance_mode: str,
    tolerance_factor: float,
) -> tuple[np.ndarray, np.ndarray, RootedTreeResult, float, float]:
    """Fit ordered R/X and RNJ for one pseudo-rooted local problem."""

    terminals = [int(column) for column in scenarios[0]["P_terminal"].columns]
    prepared = preprocess_scenarios(scenarios, recipe)
    r_matrix, x_matrix, r2_score, condition_number = fit_projected_sensitivity(
        prepared,
        constraint_mode="ordered",
    )
    geometry = sensitivity_geometry(r_matrix, x_matrix, distance_mode)
    depths = geometry.root_depths
    tolerance = tolerance_factor * max(float(np.median(depths)), 1e-12)
    tree = rooted_neighbor_joining(
        geometry.shared_paths,
        depths,
        terminals,
        int(root_bus),
        tolerance,
    )
    return r_matrix, x_matrix, tree, float(r2_score), float(condition_number)


def _predictive_nrmse(
    training: list[dict],
    validation: list[dict],
    recipe: dict,
) -> float:
    """Evaluate an ordered sensitivity fit on untouched scenario blocks."""

    if not training or not validation:
        return float("nan")
    prepared_training = preprocess_scenarios(training, recipe)
    r_matrix, x_matrix, _r2, _condition = fit_projected_sensitivity(
        prepared_training,
        constraint_mode="ordered",
    )
    prepared_validation = preprocess_scenarios(validation, recipe)
    residual_sum = 0.0
    reference_sum = 0.0
    for scenario in prepared_validation:
        p = scenario["P_terminal"].to_numpy(dtype=float)
        q = scenario["Q_terminal"].to_numpy(dtype=float)
        target = scenario["drop_target"].to_numpy(dtype=float)
        prediction = p @ r_matrix.T + q @ x_matrix.T
        intercept = np.mean(target - prediction, axis=0, keepdims=True)
        residual_sum += float(np.sum((target - prediction - intercept) ** 2))
        reference_sum += float(
            np.sum((target - target.mean(axis=0, keepdims=True)) ** 2)
        )
    return float(np.sqrt(residual_sum / max(reference_sum, 1e-15)))


def _clade_jaccard(left: set[Clade], right: set[Clade]) -> float:
    """Return Jaccard agreement between two rooted-clade sets."""

    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


def fit_local_cluster_sensitivities(
    scenarios: list[dict],
    pseudo_scenarios: list[dict],
    clusters: list[PseudoCluster] | tuple[PseudoCluster, ...],
    recipe: dict,
    distance_mode: str = "RX_75R_25X",
    tolerance_factor: float = 0.40,
    maximum_condition_number: float = 1e8,
    maximum_heldout_nrmse: float = 1.25,
    minimum_clade_stability: float = 0.50,
    minimum_r2_score: float = 0.50,
) -> dict[int, LocalClusterFit]:
    """Refit each cluster from its pseudo-rooted P/Q/V instead of global distances.

    Region membership is treated as fixed for this conditional problem, but no
    internal clade is frozen. Scenario leave-one-out fits measure whether the
    newly inferred local tree is stable without using topology labels.
    """

    if len(scenarios) != len(pseudo_scenarios):
        raise ValueError("original and pseudo scenario counts must match")
    if not clusters:
        return {}
    if tolerance_factor < 0.0:
        raise ValueError("tolerance_factor must be nonnegative")
    fits: dict[int, LocalClusterFit] = {}
    for cluster in clusters:
        local = build_local_cluster_scenarios(
            scenarios,
            pseudo_scenarios,
            cluster,
        )
        r_matrix, x_matrix, tree, r2_score, condition_number = _fit_local_tree(
            local,
            cluster.pseudo_id,
            recipe,
            distance_mode,
            tolerance_factor,
        )
        members = tuple(sorted(cluster.members))
        full_clades = rooted_clades(tree.edges, cluster.pseudo_id, members)
        if len(local) >= 2:
            heldout_nrmse = _predictive_nrmse(local[:-1], local[-1:], recipe)
            fold_clades = []
            for heldout in range(len(local)):
                subset = [
                    scenario for index, scenario in enumerate(local) if index != heldout
                ]
                if not subset:
                    continue
                fold_tree = _fit_local_tree(
                    subset,
                    cluster.pseudo_id,
                    recipe,
                    distance_mode,
                    tolerance_factor,
                )[2]
                fold_clades.append(
                    rooted_clades(fold_tree.edges, cluster.pseudo_id, members)
                )
            stability = float(
                np.mean(
                    [_clade_jaccard(full_clades, candidate) for candidate in fold_clades]
                )
            )
        else:
            heldout_nrmse = float("nan")
            stability = 1.0
        if np.isfinite(condition_number):
            condition_penalty = max(
                0.0,
                np.log10(max(condition_number, 1.0)) - 4.0,
            ) / 4.0
        else:
            condition_penalty = 10.0
        predictive_term = heldout_nrmse if np.isfinite(heldout_nrmse) else 1.0
        selection_score = (
            predictive_term
            + 0.15 * (1.0 - stability)
            + 0.05 * condition_penalty
            + 0.10 * max(0.0, minimum_r2_score - r2_score)
        )
        admissible = bool(
            np.isfinite(condition_number)
            and condition_number <= maximum_condition_number
            and (
                not np.isfinite(heldout_nrmse)
                or heldout_nrmse <= maximum_heldout_nrmse
            )
            and stability >= minimum_clade_stability
            and r2_score >= minimum_r2_score
        )
        fits[cluster.pseudo_id] = LocalClusterFit(
            pseudo_id=cluster.pseudo_id,
            members=members,
            tree=tree,
            clades=frozenset(full_clades),
            r_matrix=r_matrix,
            x_matrix=x_matrix,
            r2_score=r2_score,
            condition_number=condition_number,
            heldout_nrmse=float(heldout_nrmse),
            clade_stability=stability,
            selection_score=float(selection_score),
            admissible=admissible,
        )
    return fits


def region_only_clusters(
    clusters: list[PseudoCluster] | tuple[PseudoCluster, ...],
) -> list[PseudoCluster]:
    """Drop preliminary internal edges while preserving trusted memberships."""

    return [
        PseudoCluster(
            pseudo_id=cluster.pseudo_id,
            members=cluster.members,
            confidence=cluster.confidence,
            frozen_clades=tuple(),
            frozen_sibling_pairs=tuple(),
        )
        for cluster in clusters
    ]


def run_ordered_two_level_aggregation(
    scenarios: list[dict],
    root_bus: int,
    clusters: list[PseudoCluster] | tuple[PseudoCluster, ...],
    preprocessing: str = "daily_demean",
    distance_mode: str = "RX_75R_25X",
    tolerance_factor: float = 0.16,
    local_tolerance_factor: float = 0.40,
    deembedding_weight: float = 0.50,
    pseudo_voltage_mode: str = "deembedded_vsq",
    joint_fit_options: dict[str, object] | None = None,
    joint_blend_with_deembedded: float = 1.0,
    local_refit_max_condition_number: float = 1e8,
    local_refit_max_heldout_nrmse: float = 1.25,
    local_refit_minimum_stability: float = 0.50,
    local_refit_minimum_r2: float = 0.50,
) -> OrderedAggregationResult:
    """Embed one region-only aggregation layer in the Ordered OLS/RNJ path."""

    selected_clusters = region_only_clusters(clusters)
    if not selected_clusters:
        raise ValueError("ordered aggregation requires at least one cluster")
    base = fit_complete_rnj_baseline(
        scenarios,
        root_bus,
        preprocessing=preprocessing,
        distance_mode=distance_mode,
        tolerance_factor=tolerance_factor,
    )
    effective_joint_options: dict[str, object] = {}
    if pseudo_voltage_mode == "deembedded_vsq":
        pseudo_scenarios, pseudo_members = aggregate_rooted_scenarios(
            scenarios,
            list(base.terminals),
            base.r_matrix.to_numpy(dtype=float),
            base.x_matrix.to_numpy(dtype=float),
            selected_clusters,
            "deembedded_vsq",
            deembedding_weight=deembedding_weight,
        )
        pseudo_voltage_fits: dict[int, JointPseudoParentFit] = {}
    elif pseudo_voltage_mode in {
        "joint_latent_vsq",
        "regularized_joint_vsq",
    }:
        if pseudo_voltage_mode == "regularized_joint_vsq":
            effective_joint_options.update(regularized_joint_fit_options())
        effective_joint_options.update(joint_fit_options or {})
        pseudo_scenarios, pseudo_members, pseudo_voltage_fits = (
            aggregate_rooted_scenarios_joint(
                scenarios,
                list(base.terminals),
                base.r_matrix.to_numpy(dtype=float),
                base.x_matrix.to_numpy(dtype=float),
                selected_clusters,
                fit_options=effective_joint_options,
                blend_with_deembedded=joint_blend_with_deembedded,
                deembedding_weight=deembedding_weight,
            )
        )
    else:
        raise ValueError(
            "pseudo_voltage_mode must be 'deembedded_vsq', "
            "'joint_latent_vsq', or 'regularized_joint_vsq'"
        )
    pseudo = fit_complete_rnj_baseline(
        pseudo_scenarios,
        root_bus,
        preprocessing=preprocessing,
        distance_mode=distance_mode,
        tolerance_factor=tolerance_factor,
    )
    recipe = {
        "name": preprocessing,
        "kind": "demean" if preprocessing == "daily_demean" else preprocessing,
    }
    if preprocessing == "raw":
        recipe["kind"] = "raw"
    elif preprocessing == "difference":
        recipe["kind"] = "difference"
    local_fits = fit_local_cluster_sensitivities(
        scenarios,
        pseudo_scenarios,
        selected_clusters,
        recipe,
        distance_mode=distance_mode,
        tolerance_factor=local_tolerance_factor,
        maximum_condition_number=local_refit_max_condition_number,
        maximum_heldout_nrmse=local_refit_max_heldout_nrmse,
        minimum_clade_stability=local_refit_minimum_stability,
        minimum_r2_score=local_refit_minimum_r2,
    )
    local_refit = {
        pseudo_id: set(local.clades) for pseudo_id, local in local_fits.items()
    }
    reused_local = {
        cluster.pseudo_id: {
            clade
            for clade in base.rooted_clades
            if clade < cluster.members
        }
        for cluster in selected_clusters
    }
    hybrid_local = {
        cluster.pseudo_id: (
            local_refit[cluster.pseudo_id]
            if local_fits[cluster.pseudo_id].admissible
            else reused_local[cluster.pseudo_id]
        )
        for cluster in selected_clusters
    }
    refit_clades = expand_pseudo_clades_with_local_reidentification(
        pseudo.tree.edges,
        root_bus,
        pseudo_members,
        selected_clusters,
        local_refit,
    )
    reuse_clades = expand_pseudo_clades_with_local_reidentification(
        pseudo.tree.edges,
        root_bus,
        pseudo_members,
        selected_clusters,
        reused_local,
    )
    hybrid_clades = expand_pseudo_clades_with_local_reidentification(
        pseudo.tree.edges,
        root_bus,
        pseudo_members,
        selected_clusters,
        hybrid_local,
    )
    all_local_fits_admissible = all(
        fit.admissible for fit in local_fits.values()
    )
    candidate_clades_by_mode = {
        "ordered_aggregation_reuse": frozenset(reuse_clades),
        "ordered_aggregation_hybrid_local": frozenset(hybrid_clades),
    }
    if all_local_fits_admissible:
        candidate_clades_by_mode[
            "ordered_aggregation_local_refit"
        ] = frozenset(refit_clades)
    finite_nrmse = [
        fit.heldout_nrmse
        for fit in local_fits.values()
        if np.isfinite(fit.heldout_nrmse)
    ]
    condition_improvement = float(
        base.condition_number / max(pseudo.condition_number, 1e-15)
        if np.isfinite(base.condition_number) and np.isfinite(pseudo.condition_number)
        else 0.0
    )
    return OrderedAggregationResult(
        base=base,
        clusters=tuple(selected_clusters),
        pseudo_members=pseudo_members,
        pseudo=pseudo,
        local_fits=local_fits,
        candidate_clades_by_mode=candidate_clades_by_mode,
        condition_improvement=condition_improvement,
        mean_local_nrmse=(
            float(np.mean(finite_nrmse)) if finite_nrmse else float("nan")
        ),
        mean_local_stability=float(
            np.mean([fit.clade_stability for fit in local_fits.values()])
        ),
        all_local_fits_admissible=all_local_fits_admissible,
        pseudo_voltage_mode=pseudo_voltage_mode,
        pseudo_voltage_fits=pseudo_voltage_fits,
        pseudo_voltage_options={
            "deembedding_weight": float(deembedding_weight),
            "joint_blend_with_deembedded": float(joint_blend_with_deembedded),
            **effective_joint_options,
        },
    )
