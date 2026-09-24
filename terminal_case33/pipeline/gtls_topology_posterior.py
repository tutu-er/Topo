"""Finite-candidate GTLS topology posterior for noisy smart-meter data.

The topology search remains discrete: ordered OLS and separable GTLS produce
RNJ candidates, then each fixed tree is projected onto the exact additive-tree
cone.  Held-out residuals are scored with both response noise and propagated
P/Q measurement noise.  The final probabilities form a tempered generalized
posterior over the finite candidate set; they are not claimed to be a fully
calibrated Bayesian posterior over every possible latent tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

from terminal_case33.estimation.gtls import (
    GTLSSensitivityEstimate,
    fit_separable_gtls_sensitivity,
)
from terminal_case33.estimation.meter_error import (
    propagated_squared_voltage_variance,
)
from terminal_case33.estimation.multiscenario import fit_projected_sensitivity
from terminal_case33.estimation.nonnegative import solve_nonnegative_ridge
from terminal_case33.pipeline.rnj_candidates import (
    RNJCandidate,
    generate_rnj_candidates,
    rooted_path_incidence,
)


@dataclass(frozen=True)
class CandidateTreeProjection:
    """Nonnegative additive R/X projection for one fixed candidate tree."""

    candidate: RNJCandidate
    sources: frozenset[str]
    r_edge_coefficients: np.ndarray
    x_edge_coefficients: np.ndarray
    r_matrix: np.ndarray
    x_matrix: np.ndarray
    relative_matrix_error: float
    iterations: int


@dataclass(frozen=True)
class GTLSTopologyScore:
    """One candidate's EIV score and generalized posterior probability."""

    projection: CandidateTreeProjection
    predictive_negative_log_likelihood: float
    complexity_penalty: float
    matrix_penalty: float
    selection_loss: float
    posterior_probability: float
    observation_count: int


@dataclass(frozen=True)
class GTLSTopologyPosterior:
    """Ranked finite-candidate posterior and its GTLS diagnostics."""

    selected: GTLSTopologyScore
    ranked: tuple[GTLSTopologyScore, ...]
    estimate: GTLSSensitivityEstimate
    candidate_count: int
    effective_observation_count: int
    posterior_entropy: float
    effective_candidate_count: float
    top_two_probability_gap: float
    model_variance: np.ndarray
    clade_marginals: dict[frozenset[int], float]
    consensus_clades: frozenset[frozenset[int]]


def _projected_nnls(
    design: np.ndarray,
    target: np.ndarray,
    ridge: float = 1e-12,
    max_iterations: int = 5000,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, int]:
    """Solve the shared convex nonnegative ridge subproblem."""

    return solve_nonnegative_ridge(
        design,
        target,
        ridge=ridge,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

def project_gtls_to_candidate_tree(
    candidate: RNJCandidate,
    sources: frozenset[str],
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
) -> CandidateTreeProjection:
    """Project dense GTLS matrices onto one nonnegative additive-tree cone."""

    _oriented_edges, incidence = rooted_path_incidence(candidate.tree)
    basis = np.column_stack([np.outer(column, column).reshape(-1) for column in incidence.T])
    r_edge, r_iterations = _projected_nnls(basis, r_matrix.reshape(-1))
    x_edge, x_iterations = _projected_nnls(basis, x_matrix.reshape(-1))
    projected_r = (incidence * r_edge[None, :]) @ incidence.T
    projected_x = (incidence * x_edge[None, :]) @ incidence.T
    numerator = np.linalg.norm(projected_r - r_matrix, ord="fro") ** 2
    numerator += np.linalg.norm(projected_x - x_matrix, ord="fro") ** 2
    denominator = np.linalg.norm(r_matrix, ord="fro") ** 2
    denominator += np.linalg.norm(x_matrix, ord="fro") ** 2
    return CandidateTreeProjection(
        candidate=candidate,
        sources=sources,
        r_edge_coefficients=r_edge,
        x_edge_coefficients=x_edge,
        r_matrix=projected_r,
        x_matrix=projected_x,
        relative_matrix_error=float(np.sqrt(numerator / max(denominator, 1e-24))),
        iterations=max(r_iterations, x_iterations),
    )


def _scenario_arrays(
    scenario: dict,
    terminals: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract P, Q, squared-voltage drop, and terminal voltage arrays."""

    p = scenario["P_terminal"].loc[:, terminals].to_numpy(dtype=float)
    q = scenario["Q_terminal"].loc[:, terminals].to_numpy(dtype=float)
    drop = scenario["drop_target"].loc[:, terminals].to_numpy(dtype=float)
    voltage = scenario["V_terminal"].loc[:, terminals].to_numpy(dtype=float)
    return p, q, drop, voltage


def estimate_model_variance(
    scenarios: list[dict],
    terminals: list[int],
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    pq_noise_relative_std: float,
    voltage_noise_relative_std: float,
) -> np.ndarray:
    """Estimate a candidate-independent AC/linear mismatch variance floor."""

    residual_blocks = []
    meter_blocks = []
    for scenario in scenarios:
        p, q, drop, voltage = _scenario_arrays(scenario, terminals)
        residual = drop - p @ r_matrix.T - q @ x_matrix.T
        residual -= residual.mean(axis=0, keepdims=True)
        residual_blocks.append(residual**2)
        meter_blocks.append(
            propagated_squared_voltage_variance(
                p,
                q,
                voltage,
                r_matrix,
                x_matrix,
                pq_noise_relative_std,
                voltage_noise_relative_std,
            )
        )
    residual_variance = np.mean(np.vstack(residual_blocks), axis=0)
    meter_variance = np.mean(np.vstack(meter_blocks), axis=0)
    floor = np.maximum(0.0, residual_variance - meter_variance)
    numerical_floor = max(float(np.median(meter_variance)) * 1e-3, 1e-16)
    return np.maximum(floor, numerical_floor)


def fit_structured_gtls_candidate(
    candidate: RNJCandidate,
    sources: frozenset[str],
    training_scenarios: list[dict],
    terminals: list[int],
    initial_r_matrix: np.ndarray,
    initial_x_matrix: np.ndarray,
    model_variance: np.ndarray,
    pq_noise_relative_std: float,
    voltage_noise_relative_std: float,
    max_iterations: int = 12,
    tolerance: float = 1e-6,
) -> CandidateTreeProjection:
    """Fit one fixed additive tree by heteroscedastic EIV reweighting.

    The topology is fixed, so every iterate solves a convex nonnegative least
    squares problem in branch R/X. The variance update propagates P/Q meter
    errors through the current tree matrices. This IRLS scheme is a practical
    structured-GTLS approximation; the full sample-dependent likelihood is
    nonconvex because its variance depends on the branch coefficients.
    """

    _oriented_edges, incidence = rooted_path_incidence(candidate.tree)
    initial = project_gtls_to_candidate_tree(
        candidate,
        sources,
        initial_r_matrix,
        initial_x_matrix,
    )
    edge_count = incidence.shape[1]
    coefficients = np.concatenate([initial.r_edge_coefficients, initial.x_edge_coefficients])
    design_blocks = []
    target_blocks = []
    raw_blocks = []
    for scenario in training_scenarios:
        p, q, drop, voltage = _scenario_arrays(scenario, terminals)
        centered_p = p - p.mean(axis=0, keepdims=True)
        centered_q = q - q.mean(axis=0, keepdims=True)
        centered_drop = drop - drop.mean(axis=0, keepdims=True)
        downstream_p = centered_p @ incidence
        downstream_q = centered_q @ incidence
        r_features = downstream_p[:, None, :] * incidence[None, :, :]
        x_features = downstream_q[:, None, :] * incidence[None, :, :]
        design_blocks.append(
            np.concatenate([r_features, x_features], axis=2).reshape(
                -1,
                2 * edge_count,
            )
        )
        target_blocks.append(centered_drop.reshape(-1))
        raw_blocks.append((p, q, voltage))
    design = np.vstack(design_blocks)
    target = np.concatenate(target_blocks)
    for iteration in range(1, max_iterations + 1):
        r_edge = coefficients[:edge_count]
        x_edge = coefficients[edge_count:]
        r_matrix = (incidence * r_edge[None, :]) @ incidence.T
        x_matrix = (incidence * x_edge[None, :]) @ incidence.T
        variance_blocks = []
        for p, q, voltage in raw_blocks:
            variance = propagated_squared_voltage_variance(
                p,
                q,
                voltage,
                r_matrix,
                x_matrix,
                pq_noise_relative_std,
                voltage_noise_relative_std,
            )
            variance_blocks.append((variance + model_variance[None, :]).reshape(-1))
        standard_deviation = np.sqrt(np.maximum(np.concatenate(variance_blocks), 1e-16))
        proposal, _inner_iterations = _projected_nnls(
            design / standard_deviation[:, None],
            target / standard_deviation,
        )
        proposal = 0.5 * coefficients + 0.5 * proposal
        relative_change = np.linalg.norm(proposal - coefficients)
        relative_change /= max(np.linalg.norm(coefficients), 1e-12)
        coefficients = proposal
        if relative_change <= tolerance:
            break
    r_edge = coefficients[:edge_count]
    x_edge = coefficients[edge_count:]
    r_matrix = (incidence * r_edge[None, :]) @ incidence.T
    x_matrix = (incidence * x_edge[None, :]) @ incidence.T
    numerator = np.linalg.norm(r_matrix - initial_r_matrix, ord="fro") ** 2
    numerator += np.linalg.norm(x_matrix - initial_x_matrix, ord="fro") ** 2
    denominator = np.linalg.norm(initial_r_matrix, ord="fro") ** 2
    denominator += np.linalg.norm(initial_x_matrix, ord="fro") ** 2
    return CandidateTreeProjection(
        candidate=candidate,
        sources=sources,
        r_edge_coefficients=r_edge,
        x_edge_coefficients=x_edge,
        r_matrix=r_matrix,
        x_matrix=x_matrix,
        relative_matrix_error=float(np.sqrt(numerator / max(denominator, 1e-24))),
        iterations=iteration,
    )


def score_gtls_candidate(
    projection: CandidateTreeProjection,
    validation_scenarios: list[dict],
    terminals: list[int],
    model_variance: np.ndarray,
    pq_noise_relative_std: float,
    voltage_noise_relative_std: float,
    effective_observation_count: int,
    complexity_weight: float = 0.25,
    matrix_weight: float = 0.0,
) -> tuple[float, float, float, float, int]:
    """Return EIV predictive NLL and penalties for a fixed tree projection."""

    losses = []
    observation_count = 0
    for scenario in validation_scenarios:
        p, q, drop, voltage = _scenario_arrays(scenario, terminals)
        residual = drop - p @ projection.r_matrix.T - q @ projection.x_matrix.T
        residual -= residual.mean(axis=0, keepdims=True)
        variance = propagated_squared_voltage_variance(
            p,
            q,
            voltage,
            projection.r_matrix,
            projection.x_matrix,
            pq_noise_relative_std,
            voltage_noise_relative_std,
        )
        variance += model_variance[None, :]
        losses.append(0.5 * (np.log(variance) + residual**2 / variance))
        observation_count += residual.size
    predictive_nll = float(np.mean(np.vstack(losses)))
    parameter_count = 2 * len(projection.r_edge_coefficients)
    effective_count = max(int(effective_observation_count), 2)
    complexity_penalty = (
        0.5 * complexity_weight * parameter_count * np.log(effective_count) / effective_count
    )
    matrix_penalty = matrix_weight * projection.relative_matrix_error**2
    total = predictive_nll + complexity_penalty + matrix_penalty
    return (
        predictive_nll,
        float(complexity_penalty),
        float(matrix_penalty),
        float(total),
        observation_count,
    )


def _candidate_union(
    terminals: list[int],
    root_bus: int,
    ols_matrices: tuple[np.ndarray, np.ndarray],
    gtls_matrices: tuple[np.ndarray, np.ndarray],
    distance_modes: tuple[str, ...],
    tolerance_factors: tuple[float, ...],
) -> dict[frozenset[frozenset[int]], tuple[RNJCandidate, frozenset[str]]]:
    """Generate and deduplicate OLS and GTLS RNJ candidates by rooted clades."""

    union: dict[frozenset[frozenset[int]], tuple[RNJCandidate, frozenset[str]]] = {}
    for source, matrices in (("ols", ols_matrices), ("gtls", gtls_matrices)):
        candidates = generate_rnj_candidates(
            matrices[0],
            matrices[1],
            terminals,
            root_bus,
            distance_modes=distance_modes,
            tolerance_factors=tolerance_factors,
        )
        for candidate in candidates:
            existing = union.get(candidate.clades)
            if existing is None:
                union[candidate.clades] = (candidate, frozenset({source}))
            else:
                union[candidate.clades] = (existing[0], existing[1] | {source})
    return union


def infer_gtls_topology_posterior(
    scenarios: list[dict],
    root_bus: int,
    pq_noise_relative_std: float,
    voltage_noise_relative_std: float,
    validation_scenario_count: int = 1,
    distance_modes: tuple[str, ...] = (
        "R",
        "X",
        "RX_equal_normalized",
        "RX_75R_25X",
    ),
    tolerance_factors: tuple[float, ...] = (0.12, 0.16, 0.20, 0.24, 0.28, 0.32),
    shrinkage_to_ols: float = 0.0,
    constraint_mode: str = "ordered",
    temporal_block_length: int = 12,
    posterior_temperature: float = 1.0,
    complexity_weight: float = 0.25,
    matrix_weight: float = 0.0,
    clade_probability_threshold: float = 0.5,
) -> GTLSTopologyPosterior:
    """Infer a tempered GTLS posterior over an OLS/GTLS RNJ candidate union.

    The held-out score profiles one constant voltage-drop offset per scenario.
    Temporal correlation is handled conservatively by counting one independent
    block per ``temporal_block_length`` samples when converting score gaps to
    posterior probabilities.
    """

    if len(scenarios) < 2:
        raise ValueError("GTLS topology posterior requires at least two scenarios")
    if not 1 <= validation_scenario_count < len(scenarios):
        raise ValueError("invalid validation_scenario_count")
    if temporal_block_length <= 0:
        raise ValueError("temporal_block_length must be positive")
    if posterior_temperature <= 0.0:
        raise ValueError("posterior_temperature must be positive")
    if not 0.0 <= clade_probability_threshold <= 1.0:
        raise ValueError("clade_probability_threshold must be in [0, 1]")
    terminals = [int(column) for column in scenarios[0]["P_terminal"].columns]
    training = scenarios[:-validation_scenario_count]
    validation = scenarios[-validation_scenario_count:]
    gtls = fit_separable_gtls_sensitivity(
        training,
        pq_noise_relative_std=pq_noise_relative_std,
        voltage_noise_relative_std=voltage_noise_relative_std,
        constraint_mode=constraint_mode,
        shrinkage_to_ols=shrinkage_to_ols,
    )
    ols_r, ols_x, _ols_r2, _condition = fit_projected_sensitivity(
        training,
        constraint_mode=constraint_mode,
    )
    candidates = _candidate_union(
        terminals,
        root_bus,
        (ols_r, ols_x),
        (gtls.r_matrix, gtls.x_matrix),
        distance_modes,
        tolerance_factors,
    )
    if not candidates:
        raise RuntimeError("RNJ did not generate any topology candidates")
    model_variance = estimate_model_variance(
        training,
        terminals,
        ols_r,
        ols_x,
        pq_noise_relative_std,
        voltage_noise_relative_std,
    )
    independent_blocks = sum(
        max(1, ceil(len(scenario["P_terminal"]) / temporal_block_length)) for scenario in validation
    )
    effective_count = independent_blocks * len(terminals)
    provisional = []
    for candidate, sources in candidates.values():
        projection = fit_structured_gtls_candidate(
            candidate,
            sources,
            training,
            terminals,
            ols_r,
            ols_x,
            model_variance,
            pq_noise_relative_std,
            voltage_noise_relative_std,
        )
        score = score_gtls_candidate(
            projection,
            validation,
            terminals,
            model_variance,
            pq_noise_relative_std,
            voltage_noise_relative_std,
            effective_count,
            complexity_weight=complexity_weight,
            matrix_weight=matrix_weight,
        )
        provisional.append((projection, *score))
    losses = np.array([item[4] for item in provisional], dtype=float)
    log_weights = -effective_count * (losses - float(np.min(losses)))
    log_weights /= posterior_temperature
    log_weights -= float(np.max(log_weights))
    weights = np.exp(np.clip(log_weights, -745.0, 0.0))
    weights /= float(np.sum(weights))
    scores = []
    for item, probability in zip(provisional, weights, strict=True):
        projection, predictive, complexity, matrix_penalty, total, count = item
        scores.append(
            GTLSTopologyScore(
                projection=projection,
                predictive_negative_log_likelihood=predictive,
                complexity_penalty=complexity,
                matrix_penalty=matrix_penalty,
                selection_loss=total,
                posterior_probability=float(probability),
                observation_count=count,
            )
        )
    ranked = tuple(sorted(scores, key=lambda value: value.selection_loss))
    probabilities = np.array([score.posterior_probability for score in ranked], dtype=float)
    positive = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    probability_gap = (
        ranked[0].posterior_probability - ranked[1].posterior_probability
        if len(ranked) > 1
        else 1.0
    )
    all_clades = set().union(*(score.projection.candidate.clades for score in ranked))
    clade_marginals = {
        clade: float(
            sum(
                score.posterior_probability
                for score in ranked
                if clade in score.projection.candidate.clades
            )
        )
        for clade in all_clades
    }
    consensus: list[frozenset[int]] = []
    for clade, marginal in sorted(
        clade_marginals.items(),
        key=lambda item: (-item[1], -len(item[0]), tuple(sorted(item[0]))),
    ):
        if marginal < clade_probability_threshold:
            continue
        if all(
            clade <= accepted or accepted <= clade or clade.isdisjoint(accepted)
            for accepted in consensus
        ):
            consensus.append(clade)
    return GTLSTopologyPosterior(
        selected=ranked[0],
        ranked=ranked,
        estimate=gtls,
        candidate_count=len(ranked),
        effective_observation_count=effective_count,
        posterior_entropy=entropy,
        effective_candidate_count=float(np.exp(entropy)),
        top_two_probability_gap=float(probability_gap),
        model_variance=model_variance,
        clade_marginals=clade_marginals,
        consensus_clades=frozenset(consensus),
    )
