"""Error-aware fixed-tree fitting and predictive radial-AC ranking."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil

import networkx as nx
import numpy as np
import pandas as pd

from terminal_case33.estimation.gtls import fit_separable_gtls_sensitivity
from terminal_case33.estimation.meter_error import (
    propagated_squared_voltage_variance,
)
from terminal_case33.estimation.nonnegative import solve_nonnegative_ridge
from terminal_case33.models.ac_powerflow import solve_ac_power_flow_timeseries
from terminal_case33.models.network import TerminalizedNetwork
from terminal_case33.pipeline.rnj_candidates import (
    RNJCandidate,
    rooted_path_incidence,
)


@dataclass(frozen=True)
class FixedTreeACFit:
    """Structured-EIV branch fit after one accepted AC correction."""

    oriented_edges: tuple[tuple[int, int], ...]
    path_incidence: np.ndarray
    r_edge_coefficients: np.ndarray
    x_edge_coefficients: np.ndarray
    r_matrix: np.ndarray
    x_matrix: np.ndarray
    linear_rmse: float
    condition_number: float
    iterations: int
    ac_refinement_accepted: bool
    training_ac_rmse_before: float
    training_ac_rmse_after: float
    model_voltage_variance: np.ndarray


@dataclass(frozen=True)
class CandidateACScore:
    """Predictive score for one candidate on representative held-out scenarios."""

    candidate: RNJCandidate
    edge_fit: FixedTreeACFit
    voltage_rmse: float
    standardized_negative_log_likelihood: float
    complexity_penalty: float
    scenario_dispersion_penalty: float
    selection_score: float
    converged: bool
    observation_count: int
    effective_observation_count: int
    scenario_negative_log_likelihoods: tuple[float, ...]


@dataclass(frozen=True)
class ACRerankResult:
    """Ranked finite candidates and the minimum predictive-score topology."""

    selected: CandidateACScore
    ranked: tuple[CandidateACScore, ...]
    candidate_count: int
    training_scenario_count: int
    validation_scenario_count: int


def _tree_matrices(
    incidence: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    edge_count = incidence.shape[1]
    return (
        (incidence * coefficients[:edge_count][None, :]) @ incidence.T,
        (incidence * coefficients[edge_count:][None, :]) @ incidence.T,
    )


def _project_matrix(incidence: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    basis = np.column_stack(
        [
            np.outer(incidence[:, edge], incidence[:, edge]).reshape(-1)
            for edge in range(incidence.shape[1])
        ]
    )
    coefficients, _iterations = solve_nonnegative_ridge(
        basis,
        np.asarray(matrix, dtype=float).reshape(-1),
        ridge=1e-10,
        max_iterations=5000,
        tolerance=1e-9,
    )
    return coefficients


def _regression_system(
    candidate: RNJCandidate,
    scenarios: list[dict],
    corrections: list[np.ndarray] | None = None,
) -> tuple[
    tuple[tuple[int, int], ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    terminals = list(candidate.tree.terminals)
    oriented, incidence = rooted_path_incidence(candidate.tree)
    adjustments = corrections or [None] * len(scenarios)
    design_blocks = []
    target_blocks = []
    raw_blocks = []
    for scenario, correction in zip(scenarios, adjustments, strict=True):
        p = scenario["P_terminal"].loc[:, terminals].to_numpy(dtype=float)
        q = scenario["Q_terminal"].loc[:, terminals].to_numpy(dtype=float)
        voltage = scenario["V_terminal"].loc[:, terminals].to_numpy(dtype=float)
        drop = scenario["drop_target"].loc[:, terminals].to_numpy(dtype=float)
        if correction is not None:
            drop = drop - np.asarray(correction, dtype=float)
        downstream_p = p @ incidence
        downstream_q = q @ incidence
        design_blocks.append(
            np.concatenate(
                [
                    downstream_p[:, None, :] * incidence[None, :, :],
                    downstream_q[:, None, :] * incidence[None, :, :],
                ],
                axis=2,
            ).reshape(-1, 2 * len(oriented))
        )
        target_blocks.append(drop.reshape(-1))
        raw_blocks.append((p, q, voltage))
    return oriented, incidence, np.vstack(design_blocks), np.concatenate(target_blocks), raw_blocks


def _fit_eiv_tree(
    candidate: RNJCandidate,
    scenarios: list[dict],
    initial_r: np.ndarray,
    initial_x: np.ndarray,
    model_drop_variance: np.ndarray,
    pq_noise_relative_std: float,
    voltage_noise_relative_std: float,
    corrections: list[np.ndarray] | None = None,
    max_iterations: int = 8,
) -> FixedTreeACFit:
    oriented, incidence, design, target, raw = _regression_system(
        candidate,
        scenarios,
        corrections,
    )
    coefficients = np.concatenate(
        [_project_matrix(incidence, initial_r), _project_matrix(incidence, initial_x)]
    )
    for iteration in range(1, max_iterations + 1):
        r_matrix, x_matrix = _tree_matrices(incidence, coefficients)
        variance = np.concatenate(
            [
                (
                    propagated_squared_voltage_variance(
                        p,
                        q,
                        voltage,
                        r_matrix,
                        x_matrix,
                        pq_noise_relative_std,
                        voltage_noise_relative_std,
                    )
                    + model_drop_variance[None, :]
                ).reshape(-1)
                for p, q, voltage in raw
            ]
        )
        sigma = np.sqrt(np.maximum(variance, 1e-16))
        proposal, _inner = solve_nonnegative_ridge(
            design / sigma[:, None],
            target / sigma,
            ridge=1e-10,
            max_iterations=5000,
            tolerance=1e-9,
        )
        proposal = 0.5 * coefficients + 0.5 * proposal
        change = np.linalg.norm(proposal - coefficients)
        change /= max(np.linalg.norm(coefficients), 1e-12)
        coefficients = proposal
        if change <= 1e-6:
            break
    r_matrix, x_matrix = _tree_matrices(incidence, coefficients)
    residual = target - design @ coefficients
    edge_count = len(oriented)
    return FixedTreeACFit(
        oriented_edges=oriented,
        path_incidence=incidence,
        r_edge_coefficients=coefficients[:edge_count],
        x_edge_coefficients=coefficients[edge_count:],
        r_matrix=r_matrix,
        x_matrix=x_matrix,
        linear_rmse=float(np.sqrt(np.mean(residual**2))),
        condition_number=float(np.linalg.cond(design)),
        iterations=iteration,
        ac_refinement_accepted=False,
        training_ac_rmse_before=float("nan"),
        training_ac_rmse_after=float("nan"),
        model_voltage_variance=np.zeros(incidence.shape[0]),
    )


def _candidate_network(
    candidate: RNJCandidate,
    fit: FixedTreeACFit,
    reference_net: TerminalizedNetwork,
) -> TerminalizedNetwork:
    graph = nx.Graph()
    graph.add_edges_from(fit.oriented_edges)
    terminals = set(candidate.tree.terminals)
    buses = []
    for node in graph.nodes:
        root = node == candidate.tree.root
        terminal = node in terminals
        buses.append(
            {
                "bus_id": int(node),
                "original_bus_id": pd.NA,
                "bus_type": "root" if root else "observed_terminal" if terminal else "hidden_internal",
                "pd_kw": 0.0,
                "qd_kvar": 0.0,
                "is_observed": root or terminal,
                "has_load": terminal,
                "is_terminal": terminal,
                "is_original_case33_bus": False,
                "hidden_degree": int(graph.degree(node)),
                "note": "AC candidate",
            }
        )
    z_base = reference_net.base_kv**2 / reference_net.base_mva
    branches = []
    for index, ((parent, child), r_edge, x_edge) in enumerate(
        zip(
            fit.oriented_edges,
            fit.r_edge_coefficients,
            fit.x_edge_coefficients,
            strict=True,
        ),
        start=1,
    ):
        branches.append(
            {
                "from_bus": parent,
                "to_bus": child,
                "r_ohm": max(0.5 * float(r_edge) * z_base, 1e-12),
                "x_ohm": max(0.5 * float(x_edge) * z_base, 1e-12),
                "status": True,
                "branch_type": "synthetic",
                "original_branch_id": index,
                "length_m": np.nan,
                "is_candidate": True,
                "is_true_closed": True,
            }
        )
    return TerminalizedNetwork(
        buses=pd.DataFrame(buses),
        branches=pd.DataFrame(branches),
        root_bus=candidate.tree.root,
        base_kv=reference_net.base_kv,
        base_mva=reference_net.base_mva,
        original_to_terminal={int(node): int(node) for node in terminals},
        terminal_to_original={int(node): int(node) for node in terminals},
        metadata={"case_name": "ac_candidate"},
    )


def _run_ac(
    candidate: RNJCandidate,
    fit: FixedTreeACFit,
    scenarios: list[dict],
    reference_net: TerminalizedNetwork,
) -> tuple[list[np.ndarray], list[np.ndarray], bool]:
    net = _candidate_network(candidate, fit, reference_net)
    terminals = list(candidate.tree.terminals)
    voltage_residuals = []
    predicted_drops = []
    converged = True
    for scenario in scenarios:
        root = scenario["root_voltage"]
        ac = solve_ac_power_flow_timeseries(
            net,
            scenario["P_terminal"].loc[:, terminals],
            scenario["Q_terminal"].loc[:, terminals],
            v_root=root,
            max_iter=100,
            tol=1e-10,
        )
        predicted = ac["V_bus_mag"].loc[:, terminals].to_numpy(dtype=float)
        observed = scenario["V_terminal"].loc[:, terminals].to_numpy(dtype=float)
        voltage_residuals.append(observed - predicted)
        predicted_drops.append(root.to_numpy(dtype=float)[:, None] ** 2 - predicted**2)
        converged = converged and bool(ac["converged"].all())
    return voltage_residuals, predicted_drops, converged


def _representative_subset(scenario: dict, maximum_count: int) -> dict:
    count = len(scenario["P_terminal"])
    if count <= maximum_count:
        positions = np.arange(count)
    else:
        p = scenario["P_terminal"].to_numpy(dtype=float)
        q = scenario["Q_terminal"].to_numpy(dtype=float)
        order = np.argsort(np.sum(np.hypot(p, q), axis=1), kind="stable")
        positions = np.sort(
            order[np.rint(np.linspace(0, count - 1, maximum_count)).astype(int)]
        )
    result = {"name": scenario.get("name", "scenario")}
    for key in ("P_terminal", "Q_terminal", "V_terminal", "drop_target"):
        result[key] = scenario[key].iloc[positions].copy()
    result["root_voltage"] = scenario["root_voltage"].iloc[positions].copy()
    return result


def _blend_fit(initial: FixedTreeACFit, proposal: FixedTreeACFit, alpha: float) -> FixedTreeACFit:
    coefficients = np.concatenate(
        [
            (1.0 - alpha) * initial.r_edge_coefficients + alpha * proposal.r_edge_coefficients,
            (1.0 - alpha) * initial.x_edge_coefficients + alpha * proposal.x_edge_coefficients,
        ]
    )
    r_matrix, x_matrix = _tree_matrices(initial.path_incidence, coefficients)
    edge_count = len(initial.oriented_edges)
    return replace(
        initial,
        r_edge_coefficients=coefficients[:edge_count],
        x_edge_coefficients=coefficients[edge_count:],
        r_matrix=r_matrix,
        x_matrix=x_matrix,
    )


def _refine_with_ac(
    candidate: RNJCandidate,
    initial: FixedTreeACFit,
    scenarios: list[dict],
    reference_net: TerminalizedNetwork,
    model_drop_variance: np.ndarray,
    pq_noise_relative_std: float,
    voltage_noise_relative_std: float,
    maximum_samples: int,
) -> FixedTreeACFit:
    sampled = [_representative_subset(item, maximum_samples) for item in scenarios]
    residuals, ac_drops, converged = _run_ac(candidate, initial, sampled, reference_net)
    before = float(np.sqrt(np.mean(np.concatenate([x.ravel() for x in residuals]) ** 2)))
    if not converged:
        return replace(initial, training_ac_rmse_before=float("inf"), training_ac_rmse_after=float("inf"))
    terminals = list(candidate.tree.terminals)
    corrections = []
    for scenario, ac_drop in zip(sampled, ac_drops, strict=True):
        p = scenario["P_terminal"].loc[:, terminals].to_numpy(dtype=float)
        q = scenario["Q_terminal"].loc[:, terminals].to_numpy(dtype=float)
        corrections.append(ac_drop - p @ initial.r_matrix.T - q @ initial.x_matrix.T)
    proposal = _fit_eiv_tree(
        candidate,
        sampled,
        initial.r_matrix,
        initial.x_matrix,
        model_drop_variance,
        pq_noise_relative_std,
        voltage_noise_relative_std,
        corrections,
    )
    best = initial
    best_rmse = before
    best_residuals = residuals
    for alpha in (1.0, 0.5, 0.25):
        trial = _blend_fit(initial, proposal, alpha)
        trial_residuals, _drops, trial_converged = _run_ac(
            candidate,
            trial,
            sampled,
            reference_net,
        )
        rmse = float(
            np.sqrt(
                np.mean(np.concatenate([x.ravel() for x in trial_residuals]) ** 2)
            )
        )
        if trial_converged and rmse < best_rmse:
            best, best_rmse, best_residuals = trial, rmse, trial_residuals
            break
    residuals = best_residuals
    meter_variance = []
    for scenario in sampled:
        p = scenario["P_terminal"].loc[:, terminals].to_numpy(dtype=float)
        q = scenario["Q_terminal"].loc[:, terminals].to_numpy(dtype=float)
        voltage = scenario["V_terminal"].loc[:, terminals].to_numpy(dtype=float)
        drop_variance = propagated_squared_voltage_variance(
            p,
            q,
            voltage,
            best.r_matrix,
            best.x_matrix,
            pq_noise_relative_std,
            voltage_noise_relative_std,
        )
        meter_variance.append(drop_variance / np.maximum(4.0 * voltage**2, 1e-12))
    residual_variance = np.mean(np.vstack([x**2 for x in residuals]), axis=0)
    known_variance = np.mean(np.vstack(meter_variance), axis=0)
    floor = max(float(np.median(known_variance)) * 1e-3, 1e-16)
    return replace(
        best,
        ac_refinement_accepted=best is not initial,
        training_ac_rmse_before=before,
        training_ac_rmse_after=best_rmse,
        model_voltage_variance=np.maximum(residual_variance - known_variance, floor),
    )


def _model_drop_variance(
    scenarios: list[dict],
    terminals: list[int],
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    pq_noise_relative_std: float,
    voltage_noise_relative_std: float,
) -> np.ndarray:
    residuals = []
    known = []
    for scenario in scenarios:
        p = scenario["P_terminal"].loc[:, terminals].to_numpy(dtype=float)
        q = scenario["Q_terminal"].loc[:, terminals].to_numpy(dtype=float)
        voltage = scenario["V_terminal"].loc[:, terminals].to_numpy(dtype=float)
        drop = scenario["drop_target"].loc[:, terminals].to_numpy(dtype=float)
        residual = drop - p @ r_matrix.T - q @ x_matrix.T
        residual -= residual.mean(axis=0, keepdims=True)
        residuals.append(residual**2)
        known.append(
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
    residual_variance = np.mean(np.vstack(residuals), axis=0)
    known_variance = np.mean(np.vstack(known), axis=0)
    floor = max(float(np.median(known_variance)) * 1e-3, 1e-16)
    return np.maximum(residual_variance - known_variance, floor)


def _score_candidate(
    candidate: RNJCandidate,
    fit: FixedTreeACFit,
    scenarios: list[dict],
    reference_net: TerminalizedNetwork,
    pq_noise_relative_std: float,
    voltage_noise_relative_std: float,
    complexity_weight: float,
    scenario_dispersion_weight: float,
    temporal_block_length: int,
) -> CandidateACScore:
    terminals = list(candidate.tree.terminals)
    residuals, _drops, converged = _run_ac(candidate, fit, scenarios, reference_net)
    scenario_losses = []
    observation_count = 0
    effective_count = 0
    for scenario, residual in zip(scenarios, residuals, strict=True):
        p = scenario["P_terminal"].loc[:, terminals].to_numpy(dtype=float)
        q = scenario["Q_terminal"].loc[:, terminals].to_numpy(dtype=float)
        voltage = scenario["V_terminal"].loc[:, terminals].to_numpy(dtype=float)
        drop_variance = propagated_squared_voltage_variance(
            p,
            q,
            voltage,
            fit.r_matrix,
            fit.x_matrix,
            pq_noise_relative_std,
            voltage_noise_relative_std,
        )
        variance = drop_variance / np.maximum(4.0 * voltage**2, 1e-12)
        variance += fit.model_voltage_variance[None, :]
        variance = np.maximum(variance, 1e-16)
        scenario_losses.append(float(np.mean(0.5 * (np.log(variance) + residual**2 / variance))))
        observation_count += residual.size
        effective_count += max(1, ceil(len(scenario["P_terminal"]) / temporal_block_length)) * len(terminals)
    predictive = float(np.mean(scenario_losses))
    standard_error = (
        float(np.std(scenario_losses, ddof=1) / np.sqrt(len(scenario_losses)))
        if len(scenario_losses) > 1
        else 0.0
    )
    dispersion = scenario_dispersion_weight * standard_error
    effective_count = max(effective_count, 2)
    complexity = (
        0.5
        * complexity_weight
        * 2
        * len(fit.oriented_edges)
        * np.log(effective_count)
        / effective_count
    )
    raw = np.concatenate([x.ravel() for x in residuals])
    score = predictive + dispersion + complexity if converged else float("inf")
    return CandidateACScore(
        candidate=candidate,
        edge_fit=fit,
        voltage_rmse=float(np.sqrt(np.mean(raw**2))),
        standardized_negative_log_likelihood=predictive,
        complexity_penalty=float(complexity),
        scenario_dispersion_penalty=float(dispersion),
        selection_score=float(score),
        converged=converged,
        observation_count=observation_count,
        effective_observation_count=effective_count,
        scenario_negative_log_likelihoods=tuple(scenario_losses),
    )


def rank_candidate_trees_ac(
    candidates: list[RNJCandidate],
    training_scenarios: list[dict],
    validation_scenarios: list[dict],
    reference_net: TerminalizedNetwork,
    pq_noise_relative_std: float = 0.005,
    voltage_noise_relative_std: float = 0.0002,
    complexity_weight: float = 1.0,
    scenario_dispersion_weight: float = 0.25,
    temporal_block_length: int = 12,
    ac_refinement_max_samples: int = 12,
) -> ACRerankResult:
    """Fit EIV trees, update R/X once with AC, then rank held-out scenarios."""

    if not candidates or not training_scenarios or not validation_scenarios:
        raise ValueError("candidates, training, and validation must be nonempty")
    terminals = list(candidates[0].tree.terminals)
    gtls = fit_separable_gtls_sensitivity(
        training_scenarios,
        pq_noise_relative_std=max(pq_noise_relative_std, 1e-12),
        voltage_noise_relative_std=voltage_noise_relative_std,
        constraint_mode="ordered",
        shrinkage_to_ols=0.95,
    )
    model_variance = _model_drop_variance(
        training_scenarios,
        terminals,
        gtls.r_matrix,
        gtls.x_matrix,
        pq_noise_relative_std,
        voltage_noise_relative_std,
    )
    scores = []
    for candidate in candidates:
        if list(candidate.tree.terminals) != terminals:
            raise ValueError("candidate terminal order mismatch")
        fit = _fit_eiv_tree(
            candidate,
            training_scenarios,
            gtls.r_matrix,
            gtls.x_matrix,
            model_variance,
            pq_noise_relative_std,
            voltage_noise_relative_std,
        )
        fit = _refine_with_ac(
            candidate,
            fit,
            training_scenarios,
            reference_net,
            model_variance,
            pq_noise_relative_std,
            voltage_noise_relative_std,
            ac_refinement_max_samples,
        )
        scores.append(
            _score_candidate(
                candidate,
                fit,
                validation_scenarios,
                reference_net,
                pq_noise_relative_std,
                voltage_noise_relative_std,
                complexity_weight,
                scenario_dispersion_weight,
                temporal_block_length,
            )
        )
    ranked = tuple(sorted(scores, key=lambda item: item.selection_score))
    return ACRerankResult(
        selected=ranked[0],
        ranked=ranked,
        candidate_count=len(ranked),
        training_scenario_count=len(training_scenarios),
        validation_scenario_count=len(validation_scenarios),
    )


def representative_validation_count(scenario_count: int) -> int:
    """Choose one to three held-out scenarios as the case bank grows."""

    if scenario_count < 2:
        raise ValueError("at least two scenarios are required")
    return min(3, max(1, int(round(scenario_count / 3.0))))


def split_representative_scenarios(
    scenarios: list[dict],
    validation_scenario_count: int | None = None,
) -> tuple[list[dict], list[dict], tuple[int, ...]]:
    """Select a typical scenario first, then add diverse P/Q operating regimes."""

    count = len(scenarios)
    validation_count = (
        representative_validation_count(count)
        if validation_scenario_count is None
        else int(validation_scenario_count)
    )
    if not 1 <= validation_count < count:
        raise ValueError("validation_scenario_count must leave training data")
    features = []
    for scenario in scenarios:
        p = scenario["P_terminal"].sum(axis=1).to_numpy(dtype=float)
        q = scenario["Q_terminal"].sum(axis=1).to_numpy(dtype=float)
        root = scenario["root_voltage"].to_numpy(dtype=float)
        features.append(
            [
                np.mean(p),
                np.std(p),
                np.quantile(p, 0.1),
                np.quantile(p, 0.9),
                np.mean(q),
                np.std(q),
                np.std(np.diff(p)) if len(p) > 1 else 0.0,
                np.std(root),
            ]
        )
    values = np.asarray(features, dtype=float)
    scale = np.where(np.std(values, axis=0) > 1e-12, np.std(values, axis=0), 1.0)
    normalized = (values - np.mean(values, axis=0)) / scale
    selected = [int(np.argmin(np.linalg.norm(normalized, axis=1)))]
    while len(selected) < validation_count:
        distances = np.min(
            np.linalg.norm(
                normalized[:, None, :] - normalized[np.asarray(selected)][None, :, :],
                axis=2,
            ),
            axis=1,
        )
        distances[np.asarray(selected)] = -np.inf
        selected.append(int(np.argmax(distances)))
    validation_indices = tuple(sorted(selected))
    selected_set = set(validation_indices)
    training = [item for index, item in enumerate(scenarios) if index not in selected_set]
    validation = [scenarios[index] for index in validation_indices]
    return training, validation, validation_indices
