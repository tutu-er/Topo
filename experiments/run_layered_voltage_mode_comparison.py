"""Compare physical pseudo-PQV modes with latent layer-common voltage modes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from experiments.common import resource_assignment, selected_scenario_defs, simulate_case_scenario
from experiments.run_edge_detector_then_rnj import _ordered_fit
from experiments.run_latent_tree_diagnostics import _distance_and_depth, _set_score
from experiments.run_matrix_constraint_large_sweep import _terminal_buses
from experiments.run_probabilistic_root_decoupling import _decouple_global_root_mode
from experiments.run_rooted_hierarchical_aggregation import RECIPE
from terminal_case33.data.paper_style_case_bank import CASE_BUILDERS
from terminal_case33.estimation.multiscenario import fit_projected_sensitivity, preprocess_scenarios
from terminal_case33.estimation.recipes import apply_preprocessing_recipe
from terminal_case33.estimation.preprocessing import squared_voltage_drop_from_observed_root
from terminal_case33.graph.rooted_hierarchy import (
    PseudoCluster,
    aggregate_rooted_scenarios,
    build_local_cluster_scenarios,
    expand_pseudo_clades_with_local_reidentification,
    rooted_clades,
)
from terminal_case33.graph.rooted_neighbor_joining import rooted_neighbor_joining, shared_paths_from_distances
from terminal_case33.utils.io import ensure_dir, write_json


PSEUDO_CONFIGS = (
    ("mean_vsq", "mean_vsq", 0.0),
    ("deembed_shrink_025", "deembedded_vsq", 0.25),
    ("deembed_shrink_050", "deembedded_vsq", 0.50),
    ("deembed_full", "deembedded_vsq", 1.00),
)

LATENT_CONFIGS = (
    ("latent_free_unregularized", 0.0, 0.0, 0.0, 20),
    ("latent_free_smooth", 0.0, 1.0, 1e-6, 8),
    ("latent_pseudo_prior_025", 0.25, 1.0, 1e-6, 8),
    ("latent_pseudo_prior_1", 1.0, 1.0, 1e-6, 8),
    ("latent_pseudo_prior_5", 5.0, 1.0, 1e-6, 8),
)


@dataclass(frozen=True)
class LatentCommonModeFit:
    """One local sensitivity fit with a time-varying common voltage-drop mode."""

    r_matrix: np.ndarray
    x_matrix: np.ndarray
    common_modes: tuple[pd.Series, ...]
    r2_score: float
    condition_number: float
    mean_mode_standard_error: float


def _simulate_detailed_pool(
    case_key: str,
    t_count: int,
    data_replicate: int,
    maximum_scenarios: int,
    pq_noise_rel: float,
    v_noise_rel: float,
) -> tuple[object, list[dict]]:
    """Generate noisy measured scenarios while retaining AC truth for diagnostics."""

    net = CASE_BUILDERS[case_key]()
    assignment = resource_assignment(case_key, net)
    scenarios = []
    for definition in selected_scenario_defs(max_scenarios=maximum_scenarios):
        simulation = simulate_case_scenario(
            net=net,
            assignment=assignment,
            T=t_count,
            seed=int(definition["seed"]) + 100_003 * data_replicate,
            pq_noise_rel=pq_noise_rel,
            v_noise_rel=v_noise_rel,
            root_voltage_mean=1.02,
            root_voltage_sigma=float(definition["root_voltage_sigma"]),
            profile_scenario=str(definition["profile_scenario"]),
        )
        if not simulation["ac_converged"]:
            raise RuntimeError(f"AC power flow did not converge for {case_key}")
        scenarios.append(
            {
                "name": f"{definition['name']}_rep{data_replicate}",
                "P_terminal": simulation["P_terminal"],
                "Q_terminal": simulation["Q_terminal"],
                "V_terminal": simulation["V_terminal"],
                "root_voltage": simulation["root_voltage"],
                "drop_target": simulation["drop_target"],
                "P_true": simulation["P_true"],
                "Q_true": simulation["Q_true"],
                "V_bus_true": simulation["V_bus_true"],
                "branch_p_from_true": simulation["branch_p_from_true"],
                "branch_q_from_true": simulation["branch_q_from_true"],
            }
        )
    return net, scenarios


def _true_root_boundaries(net, terminals: list[int]) -> dict[frozenset[int], int]:
    """Map each physical root-child branch to its downstream terminal set."""

    graph = net.to_networkx_graph()
    terminal_set = set(terminals)
    result = {}
    for child in list(graph.neighbors(net.root_bus)):
        graph.remove_edge(net.root_bus, child)
        group = frozenset(terminal_set & nx.node_connected_component(graph, child))
        graph.add_edge(net.root_bus, child)
        if group:
            result[group] = int(child)
    return result


def _clusters_from_root_groups(
    root_groups: set[frozenset[int]],
    terminal_count: int,
) -> list[PseudoCluster]:
    """Represent non-singleton estimated root branches as pseudo nodes."""

    clusters = []
    for index, group in enumerate(sorted(root_groups, key=lambda item: tuple(sorted(item)))):
        if len(group) <= 1:
            continue
        clusters.append(
            PseudoCluster(
                pseudo_id=900000 + index,
                members=group,
                confidence=1.0,
                frozen_clades=tuple(),
                frozen_sibling_pairs=tuple(),
            )
        )
    return clusters


def _fit_rnj_from_scenarios(
    scenarios: list[dict],
    terminals: list[int],
    root: int,
    tolerance_factor: float,
) -> tuple[np.ndarray, np.ndarray, object]:
    """Fit ordered R/X and reconstruct one rooted latent tree."""

    r_matrix, x_matrix, distance, depth = _ordered_fit(scenarios)
    tolerance = tolerance_factor * max(float(np.median(depth)), 1e-12)
    tree = rooted_neighbor_joining(
        shared_paths_from_distances(distance, depth),
        depth,
        terminals,
        root,
        tolerance,
    )
    return r_matrix, x_matrix, tree


def _physical_pseudo_prediction(
    scenarios: list[dict],
    terminals: list[int],
    root: int,
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    clusters: list[PseudoCluster],
    voltage_mode: str,
    deembedding_weight: float,
    tolerance_factor: float,
    fit_upper: bool,
    oracle_boundaries: dict[frozenset[int], int] | None = None,
) -> tuple[set[frozenset[int]], list[dict], dict[int, frozenset[int]]]:
    """Build independent pseudo P/Q/V modes and fit local, optionally upper, RNJ."""

    pseudo_scenarios, pseudo_members = aggregate_rooted_scenarios(
        scenarios,
        terminals,
        r_matrix,
        x_matrix,
        clusters,
        voltage_mode,
        deembedding_weight,
    )
    if oracle_boundaries is not None:
        for source, pseudo in zip(scenarios, pseudo_scenarios, strict=True):
            for pseudo_id, group in pseudo_members.items():
                boundary = oracle_boundaries.get(group)
                if boundary is not None:
                    pseudo["V_terminal"][pseudo_id] = source["V_bus_true"][boundary]
            pseudo["drop_target"] = squared_voltage_drop_from_observed_root(
                pseudo["V_terminal"],
                pseudo["root_voltage"],
            )
    pseudo_nodes = list(pseudo_members)
    pseudo_tree = None
    if fit_upper:
        _, _, pseudo_tree = _fit_rnj_from_scenarios(
            pseudo_scenarios,
            pseudo_nodes,
            root,
            tolerance_factor,
        )
    local_clades = {}
    for cluster in clusters:
        local_scenarios = build_local_cluster_scenarios(
            scenarios,
            pseudo_scenarios,
            cluster,
        )
        members = sorted(cluster.members)
        _, _, local_tree = _fit_rnj_from_scenarios(
            local_scenarios,
            members,
            cluster.pseudo_id,
            tolerance_factor,
        )
        local_clades[cluster.pseudo_id] = rooted_clades(
            local_tree.edges,
            cluster.pseudo_id,
            members,
        )
    if fit_upper:
        predicted = expand_pseudo_clades_with_local_reidentification(
            pseudo_tree.edges,
            root,
            pseudo_members,
            clusters,
            local_clades,
        )
    else:
        predicted = {
            cluster.members
            for cluster in clusters
            if 1 < len(cluster.members) < len(terminals)
        }
        predicted.update(
            clade
            for cluster_clades in local_clades.values()
            for clade in cluster_clades
        )
    predicted = {
        clade for clade in predicted if 1 < len(clade) < len(terminals)
    }
    return predicted, pseudo_scenarios, pseudo_members


def _smooth_common_mode(
    residual: np.ndarray,
    prior: np.ndarray,
    prior_weight: float,
    smoothness_weight: float,
) -> np.ndarray:
    """Solve the scalar common mode with prior and temporal smoothness penalties."""

    sample_count, output_count = residual.shape
    system = (output_count + prior_weight) * np.eye(sample_count)
    if sample_count > 1 and smoothness_weight > 0.0:
        difference = np.eye(sample_count, k=0) - np.eye(sample_count, k=-1)
        difference = difference[1:]
        system += smoothness_weight * (difference.T @ difference)
    rhs = residual.sum(axis=1) + prior_weight * prior
    return np.linalg.solve(system, rhs)


def fit_latent_common_mode_sensitivity(
    scenarios: list[dict],
    members: list[int],
    prior_modes: list[pd.Series] | None,
    prior_weight: float,
    smoothness_weight: float,
    iterations: int = 8,
    alpha: float = 1e-6,
) -> LatentCommonModeFit:
    """Jointly estimate local R/X and one time-varying common mode per scenario."""

    local_raw = []
    for scenario in scenarios:
        local_raw.append(
            {
                "name": scenario["name"],
                "P_terminal": scenario["P_terminal"].loc[:, members],
                "Q_terminal": scenario["Q_terminal"].loc[:, members],
                "drop_target": scenario["drop_target"].loc[:, members],
            }
        )
    prepared = preprocess_scenarios(local_raw, RECIPE)
    prepared_priors = []
    modes = []
    for index, scenario in enumerate(prepared):
        target = scenario["drop_target"]
        if prior_modes is None:
            prior = np.zeros(len(target), dtype=float)
            initial = target.mean(axis=1).to_numpy(dtype=float)
        else:
            raw_prior = apply_preprocessing_recipe(
                prior_modes[index],
                RECIPE,
                len(prior_modes[index]),
            )
            prior = raw_prior.loc[target.index].to_numpy(dtype=float)
            initial = prior.copy()
        prepared_priors.append(prior)
        modes.append(initial)

    r_matrix = np.zeros((len(members), len(members)))
    x_matrix = np.zeros_like(r_matrix)
    condition_number = float("nan")
    for _ in range(iterations):
        adjusted = []
        for scenario, mode in zip(prepared, modes, strict=True):
            adjusted.append(
                {
                    "name": scenario["name"],
                    "P_terminal": scenario["P_terminal"],
                    "Q_terminal": scenario["Q_terminal"],
                    "drop_target": scenario["drop_target"].subtract(mode, axis=0),
                }
            )
        r_matrix, x_matrix, _, condition_number = fit_projected_sensitivity(
            adjusted,
            alpha=alpha,
            constraint_mode="ordered",
        )
        updated_modes = []
        for scenario, prior in zip(prepared, prepared_priors, strict=True):
            p = scenario["P_terminal"].to_numpy(dtype=float)
            q = scenario["Q_terminal"].to_numpy(dtype=float)
            target = scenario["drop_target"].to_numpy(dtype=float)
            residual = target - p @ r_matrix.T - q @ x_matrix.T
            updated_modes.append(
                _smooth_common_mode(
                    residual,
                    prior,
                    prior_weight,
                    smoothness_weight,
                )
            )
        change = max(
            float(np.linalg.norm(new - old) / max(np.linalg.norm(old), 1e-12))
            for new, old in zip(updated_modes, modes, strict=True)
        )
        modes = updated_modes
        if change <= 1e-6:
            break

    residual_blocks = []
    target_blocks = []
    mode_standard_errors = []
    mode_series = []
    for scenario, mode in zip(prepared, modes, strict=True):
        p = scenario["P_terminal"].to_numpy(dtype=float)
        q = scenario["Q_terminal"].to_numpy(dtype=float)
        target = scenario["drop_target"].to_numpy(dtype=float)
        prediction = p @ r_matrix.T + q @ x_matrix.T + mode[:, None]
        residual = target - prediction
        residual_blocks.append(residual)
        target_blocks.append(target)
        mode_standard_errors.append(
            np.std(residual, axis=1, ddof=0) / np.sqrt(max(len(members), 1))
        )
        mode_series.append(pd.Series(mode, index=scenario["drop_target"].index))
    residual = np.vstack(residual_blocks)
    target = np.vstack(target_blocks)
    residual_sum = float(np.sum(residual**2))
    total_sum = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    r2_score = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0
    return LatentCommonModeFit(
        r_matrix=r_matrix,
        x_matrix=x_matrix,
        common_modes=tuple(mode_series),
        r2_score=r2_score,
        condition_number=condition_number,
        mean_mode_standard_error=float(np.mean(np.concatenate(mode_standard_errors))),
    )


def _latent_mode_prediction(
    scenarios: list[dict],
    terminals: list[int],
    root_groups: set[frozenset[int]],
    clusters: list[PseudoCluster],
    prior_pseudo_scenarios: list[dict],
    prior_weight: float,
    smoothness_weight: float,
    alpha: float,
    iterations: int,
    tolerance_factor: float,
) -> tuple[set[frozenset[int]], list[dict]]:
    """Fit one latent common mode per root branch and reconstruct local RNJ trees."""

    cluster_by_members = {cluster.members: cluster for cluster in clusters}
    predicted = set()
    diagnostics = []
    for group in root_groups:
        if 1 < len(group) < len(terminals):
            predicted.add(group)
        members = sorted(group)
        if len(members) < 3:
            continue
        cluster = cluster_by_members.get(group)
        if cluster is None:
            prior_modes = None
            local_root = -3000000 - len(diagnostics)
        else:
            prior_modes = [
                pseudo["drop_target"][cluster.pseudo_id]
                for pseudo in prior_pseudo_scenarios
            ]
            local_root = cluster.pseudo_id
        fit = fit_latent_common_mode_sensitivity(
            scenarios,
            members,
            prior_modes,
            prior_weight,
            smoothness_weight,
            alpha=alpha,
            iterations=iterations,
        )
        distance, depth, _ = _distance_and_depth(
            fit.r_matrix,
            fit.x_matrix,
            "RX_75R_25X",
        )
        tolerance = tolerance_factor * max(float(np.median(depth)), 1e-12)
        tree = rooted_neighbor_joining(
            shared_paths_from_distances(distance, depth),
            depth,
            members,
            local_root,
            tolerance,
        )
        predicted.update(rooted_clades(tree.edges, local_root, members))
        diagnostics.append(
            {
                "group": group,
                "fit": fit,
            }
        )
    return predicted, diagnostics


def _mode_error_rows(
    case_key: str,
    scenario_count: int,
    t_count: int,
    data_replicate: int,
    method: str,
    scenarios: list[dict],
    pseudo_scenarios: list[dict],
    pseudo_members: dict[int, frozenset[int]],
    true_boundaries: dict[frozenset[int], int],
    root: int,
) -> list[dict]:
    """Compare estimated pseudo modes and aggregate P/Q with AC boundary truth."""

    rows = []
    for pseudo_id, group in pseudo_members.items():
        child = true_boundaries.get(group)
        if child is None:
            continue
        edge_label = f"{root}->{child}"
        for scenario, pseudo in zip(scenarios, pseudo_scenarios, strict=True):
            estimated_v = pseudo["V_terminal"][pseudo_id]
            true_v = scenario["V_bus_true"][child]
            estimated_p = pseudo["P_terminal"][pseudo_id]
            estimated_q = pseudo["Q_terminal"][pseudo_id]
            true_p = scenario["branch_p_from_true"][edge_label]
            true_q = scenario["branch_q_from_true"][edge_label]

            def relative_rmse(estimate: pd.Series, truth: pd.Series) -> float:
                error = estimate.to_numpy(dtype=float) - truth.to_numpy(dtype=float)
                scale = np.sqrt(np.mean(truth.to_numpy(dtype=float) ** 2))
                return float(np.sqrt(np.mean(error**2)) / max(scale, 1e-12))

            rows.append(
                {
                    "case": case_key,
                    "scenario_count": scenario_count,
                    "t_count": t_count,
                    "data_replicate": data_replicate,
                    "method": method,
                    "scenario": scenario["name"],
                    "pseudo_id": pseudo_id,
                    "members": ",".join(map(str, sorted(group))),
                    "boundary_bus": child,
                    "v_magnitude_rmse": float(
                        np.sqrt(np.mean((estimated_v.to_numpy() - true_v.to_numpy()) ** 2))
                    ),
                    "v_squared_rmse": float(
                        np.sqrt(
                            np.mean(
                                (estimated_v.pow(2).to_numpy() - true_v.pow(2).to_numpy()) ** 2
                            )
                        )
                    ),
                    "p_relative_rmse": relative_rmse(estimated_p, true_p),
                    "q_relative_rmse": relative_rmse(estimated_q, true_q),
                }
            )
    return rows


def _latent_true_mode_rmse(
    fit: LatentCommonModeFit,
    scenarios: list[dict],
    group: frozenset[int],
    true_boundaries: dict[frozenset[int], int],
) -> float:
    """Compare an estimated preprocessed common mode with the true AC boundary drop."""

    child = true_boundaries.get(group)
    if child is None:
        return float("nan")
    errors = []
    for scenario, estimated in zip(scenarios, fit.common_modes, strict=True):
        true_raw = scenario["root_voltage"].pow(2) - scenario["V_bus_true"][child].pow(2)
        true_mode = apply_preprocessing_recipe(true_raw, RECIPE, len(true_raw))
        common_index = estimated.index.intersection(true_mode.index)
        errors.append(
            estimated.loc[common_index].to_numpy(dtype=float)
            - true_mode.loc[common_index].to_numpy(dtype=float)
        )
    values = np.concatenate(errors)
    return float(np.sqrt(np.mean(values**2)))




def run(
    output_dir: str | Path = "outputs/layered_voltage_mode_comparison",
    cases: list[str] | None = None,
    scenario_counts: list[int] | None = None,
    t_counts: list[int] | None = None,
    data_replicates: int = 1,
    pq_noise_rel: float = 0.005,
    v_noise_rel: float = 0.0002,
    tolerance_factor: float = 0.16,
    root_mode_quantile: float = 0.10,
) -> dict:
    """Run physical pseudo-mode and latent common-mode layer tests."""

    out_dir = ensure_dir(output_dir)
    selected_cases = cases or list(CASE_BUILDERS)
    selected_scenarios = scenario_counts or [3, 5]
    selected_t = t_counts or [48, 96]
    maximum_scenarios = max(selected_scenarios)
    topology_rows = []
    mode_rows = []
    latent_rows = []
    total = len(selected_cases) * len(selected_t) * data_replicates * len(selected_scenarios)
    completed = 0

    for case_key in selected_cases:
        for t_count in selected_t:
            for data_replicate in range(data_replicates):
                net, scenario_pool = _simulate_detailed_pool(
                    case_key,
                    t_count,
                    data_replicate,
                    maximum_scenarios,
                    pq_noise_rel,
                    v_noise_rel,
                )
                terminals = _terminal_buses(net)
                true_clades = rooted_clades(net.closed_edges(), net.root_bus, terminals)
                true_boundaries = _true_root_boundaries(net, terminals)
                for scenario_count in selected_scenarios:
                    scenarios = scenario_pool[:scenario_count]
                    r_matrix, x_matrix, distance, depth = _ordered_fit(scenarios)
                    _, _, root_groups, _ = _decouple_global_root_mode(
                        distance,
                        depth,
                        terminals,
                        net.root_bus,
                        tolerance_factor,
                        root_mode_quantile,
                    )
                    clusters = _clusters_from_root_groups(root_groups, len(terminals))
                    _, _, base_tree = _fit_rnj_from_scenarios(
                        scenarios,
                        terminals,
                        net.root_bus,
                        tolerance_factor,
                    )
                    baseline = rooted_clades(base_tree.edges, net.root_bus, terminals)
                    precision, recall, f1_score = _set_score(baseline, true_clades)
                    topology_rows.append(
                        {
                            "case": case_key,
                            "scenario_count": scenario_count,
                            "t_count": t_count,
                            "data_replicate": data_replicate,
                            "method": "baseline_rnj",
                            "precision": precision,
                            "recall": recall,
                            "f1": f1_score,
                            "exact": baseline == true_clades,
                            "root_partition_exact": set(root_groups) == set(true_boundaries),
                        }
                    )

                    pseudo_by_method = {}
                    for label, voltage_mode, weight in PSEUDO_CONFIGS:
                        predicted, pseudo_scenarios, pseudo_members = _physical_pseudo_prediction(
                            scenarios,
                            terminals,
                            net.root_bus,
                            r_matrix,
                            x_matrix,
                            clusters,
                            voltage_mode,
                            weight,
                            tolerance_factor,
                            fit_upper=False,
                        )
                        pseudo_by_method[label] = pseudo_scenarios
                        precision, recall, f1_score = _set_score(predicted, true_clades)
                        topology_rows.append(
                            {
                                "case": case_key,
                                "scenario_count": scenario_count,
                                "t_count": t_count,
                                "data_replicate": data_replicate,
                                "method": f"physical_local_{label}",
                                "precision": precision,
                                "recall": recall,
                                "f1": f1_score,
                                "exact": predicted == true_clades,
                                "root_partition_exact": set(root_groups) == set(true_boundaries),
                            }
                        )
                        mode_rows.extend(
                            _mode_error_rows(
                                case_key,
                                scenario_count,
                                t_count,
                                data_replicate,
                                label,
                                scenarios,
                                pseudo_scenarios,
                                pseudo_members,
                                true_boundaries,
                                net.root_bus,
                            )
                        )

                    oracle_prediction, oracle_pseudo, oracle_members = (
                        _physical_pseudo_prediction(
                            scenarios,
                            terminals,
                            net.root_bus,
                            r_matrix,
                            x_matrix,
                            clusters,
                            "mean_vsq",
                            0.0,
                            tolerance_factor,
                            fit_upper=False,
                            oracle_boundaries=true_boundaries,
                        )
                    )
                    precision, recall, f1_score = _set_score(
                        oracle_prediction,
                        true_clades,
                    )
                    topology_rows.append(
                        {
                            "case": case_key,
                            "scenario_count": scenario_count,
                            "t_count": t_count,
                            "data_replicate": data_replicate,
                            "method": "physical_local_oracle_v",
                            "precision": precision,
                            "recall": recall,
                            "f1": f1_score,
                            "exact": oracle_prediction == true_clades,
                            "root_partition_exact": set(root_groups) == set(true_boundaries),
                        }
                    )
                    mode_rows.extend(
                        _mode_error_rows(
                            case_key,
                            scenario_count,
                            t_count,
                            data_replicate,
                            "oracle_v",
                            scenarios,
                            oracle_pseudo,
                            oracle_members,
                            true_boundaries,
                            net.root_bus,
                        )
                    )

                    upper_prediction, _, _ = _physical_pseudo_prediction(
                        scenarios,
                        terminals,
                        net.root_bus,
                        r_matrix,
                        x_matrix,
                        clusters,
                        "deembedded_vsq",
                        0.50,
                        tolerance_factor,
                        fit_upper=True,
                    )
                    precision, recall, f1_score = _set_score(
                        upper_prediction,
                        true_clades,
                    )
                    topology_rows.append(
                        {
                            "case": case_key,
                            "scenario_count": scenario_count,
                            "t_count": t_count,
                            "data_replicate": data_replicate,
                            "method": "physical_upper_deembed_shrink_050",
                            "precision": precision,
                            "recall": recall,
                            "f1": f1_score,
                            "exact": upper_prediction == true_clades,
                            "root_partition_exact": set(root_groups) == set(true_boundaries),
                        }
                    )

                    prior_pseudo = pseudo_by_method["deembed_shrink_050"]
                    for label, prior_weight, smoothness_weight, alpha, iterations in LATENT_CONFIGS:
                        predicted, diagnostics = _latent_mode_prediction(
                            scenarios,
                            terminals,
                            root_groups,
                            clusters,
                            prior_pseudo,
                            prior_weight,
                            smoothness_weight,
                            alpha,
                            iterations,
                            tolerance_factor,
                        )
                        precision, recall, f1_score = _set_score(predicted, true_clades)
                        topology_rows.append(
                            {
                                "case": case_key,
                                "scenario_count": scenario_count,
                                "t_count": t_count,
                                "data_replicate": data_replicate,
                                "method": label,
                                "precision": precision,
                                "recall": recall,
                                "f1": f1_score,
                                "exact": predicted == true_clades,
                                "root_partition_exact": set(root_groups) == set(true_boundaries),
                            }
                        )
                        for diagnostic in diagnostics:
                            fit = diagnostic["fit"]
                            true_mode_rmse = _latent_true_mode_rmse(
                                fit,
                                scenarios,
                                diagnostic["group"],
                                true_boundaries,
                            )
                            latent_rows.append(
                                {
                                    "case": case_key,
                                    "scenario_count": scenario_count,
                                    "t_count": t_count,
                                    "data_replicate": data_replicate,
                                    "method": label,
                                    "members": ",".join(map(str, sorted(diagnostic["group"]))),
                                    "r2_score": fit.r2_score,
                                    "condition_number": fit.condition_number,
                                    "mean_mode_standard_error": fit.mean_mode_standard_error,
                                    "true_common_mode_rmse": true_mode_rmse,
                                }
                            )
                    completed += 1
                    print(
                        f"[layer-mode] {completed}/{total} case={case_key} "
                        f"scenarios={scenario_count} T={t_count}",
                        flush=True,
                    )

    topology = pd.DataFrame(topology_rows)
    mode_errors = pd.DataFrame(mode_rows)
    latent_diagnostics = pd.DataFrame(latent_rows)
    for frame in (topology, mode_errors, latent_diagnostics):
        frame["regime"] = np.where(
            frame["scenario_count"].ge(3),
            "normal_multiscenario",
            "single_scenario_stress",
        )
    topology.to_csv(out_dir / "topology_results.csv", index=False)
    mode_errors.to_csv(out_dir / "pseudo_mode_errors.csv", index=False)
    latent_diagnostics.to_csv(out_dir / "latent_mode_diagnostics.csv", index=False)
    topology_summary = (
        topology.groupby("method", as_index=False)
        .agg(
            runs=("f1", "size"),
            mean_f1=("f1", "mean"),
            exact_rate=("exact", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
        )
        .sort_values(["mean_f1", "exact_rate"], ascending=False)
    )
    topology_summary.to_csv(out_dir / "topology_summary.csv", index=False)
    topology_by_regime = (
        topology.groupby(["regime", "method"], as_index=False)
        .agg(
            runs=("f1", "size"),
            mean_f1=("f1", "mean"),
            exact_rate=("exact", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
        )
        .sort_values(["regime", "mean_f1"], ascending=[True, False])
    )
    topology_by_regime.to_csv(out_dir / "topology_summary_by_regime.csv", index=False)
    by_case = (
        topology.groupby(["case", "regime", "method"], as_index=False)
        .agg(mean_f1=("f1", "mean"), exact_rate=("exact", "mean"))
    )
    by_case.to_csv(out_dir / "topology_summary_by_case.csv", index=False)
    mode_summary = (
        mode_errors.groupby("method", as_index=False)
        .agg(
            evaluated_modes=("v_squared_rmse", "size"),
            mean_v_magnitude_rmse=("v_magnitude_rmse", "mean"),
            mean_v_squared_rmse=("v_squared_rmse", "mean"),
            mean_p_relative_rmse=("p_relative_rmse", "mean"),
            mean_q_relative_rmse=("q_relative_rmse", "mean"),
        )
        if not mode_errors.empty
        else pd.DataFrame()
    )
    mode_summary.to_csv(out_dir / "pseudo_mode_summary.csv", index=False)
    mode_by_regime = (
        mode_errors.groupby(["regime", "method"], as_index=False)
        .agg(
            evaluated_modes=("v_squared_rmse", "size"),
            mean_v_magnitude_rmse=("v_magnitude_rmse", "mean"),
            mean_v_squared_rmse=("v_squared_rmse", "mean"),
            mean_p_relative_rmse=("p_relative_rmse", "mean"),
            mean_q_relative_rmse=("q_relative_rmse", "mean"),
        )
        if not mode_errors.empty
        else pd.DataFrame()
    )
    mode_by_regime.to_csv(out_dir / "pseudo_mode_summary_by_regime.csv", index=False)
    latent_summary = (
        latent_diagnostics.groupby("method", as_index=False)
        .agg(
            fits=("r2_score", "size"),
            mean_r2=("r2_score", "mean"),
            median_condition_number=("condition_number", "median"),
            mean_mode_standard_error=("mean_mode_standard_error", "mean"),
            mean_true_common_mode_rmse=("true_common_mode_rmse", "mean"),
        )
        if not latent_diagnostics.empty
        else pd.DataFrame()
    )
    latent_summary.to_csv(out_dir / "latent_mode_summary.csv", index=False)
    latent_by_regime = (
        latent_diagnostics.groupby(["regime", "method"], as_index=False)
        .agg(
            fits=("r2_score", "size"),
            mean_r2=("r2_score", "mean"),
            median_condition_number=("condition_number", "median"),
            mean_mode_standard_error=("mean_mode_standard_error", "mean"),
            mean_true_common_mode_rmse=("true_common_mode_rmse", "mean"),
        )
        if not latent_diagnostics.empty
        else pd.DataFrame()
    )
    latent_by_regime.to_csv(out_dir / "latent_mode_summary_by_regime.csv", index=False)
    metrics = {
        "condition_count": total,
        "measurement_noise": {"pq_relative": pq_noise_rel, "voltage_relative": v_noise_rel},
        "topology_summary": topology_summary.to_dict(orient="records"),
        "topology_summary_by_regime": topology_by_regime.to_dict(orient="records"),
        "pseudo_mode_summary": mode_summary.to_dict(orient="records"),
        "pseudo_mode_summary_by_regime": mode_by_regime.to_dict(orient="records"),
        "latent_mode_summary": latent_summary.to_dict(orient="records"),
        "latent_mode_summary_by_regime": latent_by_regime.to_dict(orient="records"),
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# Layered voltage-mode comparison\n\n"
        "Physical methods create one independent pseudo P/Q/V time series per estimated "
        "root branch. Latent methods treat each branch common voltage-drop mode as a "
        "time-varying optimization variable with optional pseudo-voltage prior.\n\n"
        "## Topology\n\n"
        + topology_summary.to_string(index=False)
        + "\n\n## Topology by data regime\n\n"
        + topology_by_regime.to_string(index=False)
        + "\n\n## Pseudo-mode AC error\n\n"
        + mode_summary.to_string(index=False)
        + "\n\n## Latent common-mode fit\n\n"
        + latent_summary.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/layered_voltage_mode_comparison")
    parser.add_argument("--cases", nargs="*", choices=list(CASE_BUILDERS), default=None)
    parser.add_argument("--scenario-counts", nargs="*", type=int, default=None)
    parser.add_argument("--T-counts", nargs="*", type=int, default=None)
    parser.add_argument("--data-replicates", type=int, default=1)
    parser.add_argument("--pq-noise-rel", type=float, default=0.005)
    parser.add_argument("--v-noise-rel", type=float, default=0.0002)
    parser.add_argument("--tolerance-factor", type=float, default=0.16)
    parser.add_argument("--root-mode-quantile", type=float, default=0.10)
    args = parser.parse_args()
    run(
        output_dir=args.output,
        cases=args.cases,
        scenario_counts=args.scenario_counts,
        t_counts=args.T_counts,
        data_replicates=args.data_replicates,
        pq_noise_rel=args.pq_noise_rel,
        v_noise_rel=args.v_noise_rel,
        tolerance_factor=args.tolerance_factor,
        root_mode_quantile=args.root_mode_quantile,
    )


if __name__ == "__main__":
    main()
