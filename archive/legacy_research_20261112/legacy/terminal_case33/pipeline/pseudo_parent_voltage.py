"""Joint latent pseudo-parent voltage and local sensitivity estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from terminal_case33.estimation.preprocessing import (
    squared_voltage_drop_from_observed_root,
)
from terminal_case33.graph.rooted_hierarchy import (
    PseudoCluster,
    aggregate_rooted_scenarios,
)


@dataclass(frozen=True)
class JointPseudoParentFit:
    """Joint local sensitivity and per-sample pseudo-parent voltage fit."""

    pseudo_id: int
    members: tuple[int, ...]
    r_local: np.ndarray
    x_local: np.ndarray
    pseudo_voltage_squared: tuple[pd.Series, ...]
    residual_nrmse: float
    iterations: int
    converged: bool
    gauge_r: float
    gauge_x: float
    common_mode_method: str
    anchor_strength: float
    prior_weight: float
    smoothness_weight: float


def regularized_joint_fit_options() -> dict[str, Any]:
    """Return the validated 3x96 regularized-joint parameter preset."""

    return {
        "gauge_quantile": 0.05,
        "optimization_gauge_quantile": 0.0,
        "anchor_strength": 0.10,
        "prior_weight": 1.0,
        "smoothness_weight": 1.0,
        "common_mode_method": "huber",
    }


def _project_local_sensitivity(
    matrix: np.ndarray,
    enforce_psd: bool = False,
) -> np.ndarray:
    """Project onto symmetric, nonnegative, diagonally ordered matrices."""

    result = np.maximum(0.5 * (matrix + matrix.T), 0.0)
    if enforce_psd:
        eigenvalues, eigenvectors = np.linalg.eigh(result)
        result = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        result = np.maximum(0.5 * (result + result.T), 0.0)
    if len(result) > 1:
        for index in range(len(result)):
            other = np.delete(result[index], index)
            result[index, index] = max(result[index, index], float(np.max(other)))
    return result


def _gauge_value(matrix: np.ndarray, quantile: float) -> float:
    """Estimate the common root-to-cluster path from off-diagonal entries."""

    if len(matrix) < 2:
        return 0.0
    values = matrix[np.triu_indices(len(matrix), 1)]
    if not values.size:
        return 0.0
    return max(float(np.quantile(values, quantile)), 0.0)


def _remove_common_path_gauge(
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    common_drop: np.ndarray,
    p_values: np.ndarray,
    q_values: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Move an unidentifiable all-ones R/X component into the common mode."""

    common_r = _gauge_value(r_matrix, quantile)
    common_x = _gauge_value(x_matrix, quantile)
    r_adjusted = np.maximum(r_matrix - common_r, 0.0)
    x_adjusted = np.maximum(x_matrix - common_x, 0.0)
    drop_adjusted = (
        common_drop
        + common_r * np.sum(p_values, axis=1)
        + common_x * np.sum(q_values, axis=1)
    )
    return r_adjusted, x_adjusted, drop_adjusted, common_r, common_x


def _profile_location(residual: np.ndarray, method: str, huber_delta: float) -> np.ndarray:
    """Profile one common location from each row of a residual matrix."""

    if method == "mean":
        return np.mean(residual, axis=1)
    if method == "median":
        return np.median(residual, axis=1)
    if method != "huber":
        raise ValueError("common_mode_method must be 'mean', 'median', or 'huber'")
    location = np.median(residual, axis=1)
    scale = 1.4826 * np.median(
        np.abs(residual - location[:, None]),
        axis=1,
    )
    fallback = np.std(residual, axis=1)
    scale = np.maximum(scale, np.maximum(0.25 * fallback, 1e-10))
    for _ in range(8):
        error = residual - location[:, None]
        threshold = huber_delta * scale[:, None]
        weight = np.minimum(1.0, threshold / np.maximum(np.abs(error), 1e-15))
        updated = np.sum(weight * residual, axis=1) / np.maximum(
            np.sum(weight, axis=1),
            1e-15,
        )
        if float(np.max(np.abs(updated - location))) <= 1e-12:
            location = updated
            break
        location = updated
    return location


def _solve_common_mode(
    residual: np.ndarray,
    prior: np.ndarray | None,
    prior_weight: float,
    smoothness_weight: float,
    method: str,
    huber_delta: float,
) -> np.ndarray:
    """Solve a robust, prior-anchored, temporally smooth common mode."""

    count, output_count = residual.shape
    profiled = _profile_location(residual, method, huber_delta)
    if prior_weight == 0.0 and smoothness_weight == 0.0:
        return profiled
    system = (output_count + prior_weight) * np.eye(count)
    if count > 1 and smoothness_weight > 0.0:
        difference = np.eye(count, k=0) - np.eye(count, k=-1)
        difference = difference[1:]
        system += smoothness_weight * (difference.T @ difference)
    rhs = output_count * profiled
    if prior_weight > 0.0:
        if prior is None:
            raise ValueError("a pseudo-parent prior is required when prior_weight > 0")
        rhs += prior_weight * prior
    return np.linalg.solve(system, rhs)


def fit_joint_pseudo_parent_voltage(
    scenarios: list[dict],
    members: list[int],
    pseudo_id: int,
    initial_r: np.ndarray,
    initial_x: np.ndarray,
    ridge: float = 1e-7,
    anchor_strength: float = 0.0,
    gauge_quantile: float = 0.20,
    optimization_gauge_quantile: float = 0.0,
    common_mode_method: str = "mean",
    huber_delta: float = 1.5,
    prior_pseudo_voltage_squared: list[pd.Series] | tuple[pd.Series, ...] | None = None,
    prior_weight: float = 0.0,
    smoothness_weight: float = 0.0,
    enforce_psd: bool = False,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
) -> JointPseudoParentFit:
    r"""Jointly estimate local R/X and a regularized pseudo-parent mode.

    For every member terminal i the raw squared-voltage-drop model is

        v0^2(t) - vi^2(t) = g(t) + p(t)^T ri + q(t)^T xi + ei(t).

    The common mode has an optional de-embedded prior and first-difference
    penalty. anchor_strength shrinks local R/X toward the globally fitted
    submatrix on a curvature-relative scale. The root-to-cluster all-ones
    sensitivity is not identifiable from local contrasts, so it is removed once
    using gauge_quantile and the optimization then uses a zero lower-envelope
    gauge.
    """

    if not scenarios or not members:
        raise ValueError("scenarios and members must be nonempty")
    if (
        ridge < 0.0
        or anchor_strength < 0.0
        or prior_weight < 0.0
        or smoothness_weight < 0.0
        or max_iterations <= 0
        or tolerance <= 0.0
        or huber_delta <= 0.0
    ):
        raise ValueError("invalid joint pseudo-parent solver settings")
    if not 0.0 <= gauge_quantile <= 1.0:
        raise ValueError("gauge_quantile must be in [0, 1]")
    if not 0.0 <= optimization_gauge_quantile <= 1.0:
        raise ValueError("optimization_gauge_quantile must be in [0, 1]")
    if prior_pseudo_voltage_squared is not None and (
        len(prior_pseudo_voltage_squared) != len(scenarios)
    ):
        raise ValueError("pseudo-parent prior count must match scenario count")

    member_list = [int(node) for node in members]
    p_blocks: list[np.ndarray] = []
    q_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    prior_drop_blocks: list[np.ndarray | None] = []
    roots: list[np.ndarray] = []
    indices: list[pd.Index] = []
    for scenario_index, scenario in enumerate(scenarios):
        p = scenario["P_terminal"].loc[:, member_list].to_numpy(dtype=float)
        q = scenario["Q_terminal"].loc[:, member_list].to_numpy(dtype=float)
        voltage = scenario["V_terminal"].loc[:, member_list].to_numpy(dtype=float)
        root = scenario["root_voltage"].to_numpy(dtype=float)
        p_blocks.append(p)
        q_blocks.append(q)
        y_blocks.append(root[:, None] ** 2 - voltage**2)
        roots.append(root)
        indices.append(scenario["V_terminal"].index)
        if prior_pseudo_voltage_squared is None:
            prior_drop_blocks.append(None)
        else:
            prior_vsq = (
                prior_pseudo_voltage_squared[scenario_index]
                .reindex(scenario["V_terminal"].index)
                .to_numpy(dtype=float)
            )
            prior_drop_blocks.append(root**2 - prior_vsq)

    p_values = np.vstack(p_blocks)
    q_values = np.vstack(q_blocks)
    target = np.vstack(y_blocks)
    size = len(member_list)
    r_matrix = _project_local_sensitivity(
        np.asarray(initial_r, dtype=float).copy(),
        enforce_psd,
    )
    x_matrix = _project_local_sensitivity(
        np.asarray(initial_x, dtype=float).copy(),
        enforce_psd,
    )
    if r_matrix.shape != (size, size) or x_matrix.shape != (size, size):
        raise ValueError("initial local sensitivities must match cluster size")

    initial_drop = np.mean(
        target - p_values @ r_matrix.T - q_values @ x_matrix.T,
        axis=1,
    )
    r_matrix, x_matrix, _drop, gauge_r, gauge_x = _remove_common_path_gauge(
        r_matrix,
        x_matrix,
        initial_drop,
        p_values,
        q_values,
        gauge_quantile,
    )
    anchor_r = r_matrix.copy()
    anchor_x = x_matrix.copy()

    design = np.column_stack([p_values, q_values])
    data_curvature = (
        2.0 * float(np.linalg.norm(design, ord=2) ** 2) / max(target.size, 1)
    )
    anchor_curvature = anchor_strength * data_curvature
    lipschitz = data_curvature + anchor_curvature + 2.0 * ridge
    step = 0.9 / max(lipschitz, 1e-12)

    common_blocks = [
        _solve_common_mode(
            target_block - p_block @ r_matrix.T - q_block @ x_matrix.T,
            prior_block,
            prior_weight,
            smoothness_weight,
            common_mode_method,
            huber_delta,
        )
        for target_block, p_block, q_block, prior_block in zip(
            y_blocks,
            p_blocks,
            q_blocks,
            prior_drop_blocks,
            strict=True,
        )
    ]
    common_drop = np.concatenate(common_blocks)
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        previous_r = r_matrix.copy()
        previous_x = x_matrix.copy()
        previous_common = common_drop.copy()

        common_blocks = [
            _solve_common_mode(
                target_block - p_block @ r_matrix.T - q_block @ x_matrix.T,
                prior_block,
                prior_weight,
                smoothness_weight,
                common_mode_method,
                huber_delta,
            )
            for target_block, p_block, q_block, prior_block in zip(
                y_blocks,
                p_blocks,
                q_blocks,
                prior_drop_blocks,
                strict=True,
            )
        ]
        common_drop = np.concatenate(common_blocks)
        residual = (
            common_drop[:, None]
            + p_values @ r_matrix.T
            + q_values @ x_matrix.T
            - target
        )
        scale = 2.0 / max(target.size, 1)
        gradient_r = (
            scale * residual.T @ p_values
            + 2.0 * ridge * (r_matrix - anchor_r)
            + anchor_curvature * (r_matrix - anchor_r)
        )
        gradient_x = (
            scale * residual.T @ q_values
            + 2.0 * ridge * (x_matrix - anchor_x)
            + anchor_curvature * (x_matrix - anchor_x)
        )
        r_matrix = _project_local_sensitivity(
            r_matrix - step * gradient_r,
            enforce_psd,
        )
        x_matrix = _project_local_sensitivity(
            x_matrix - step * gradient_x,
            enforce_psd,
        )
        r_matrix, x_matrix, common_drop, _shift_r, _shift_x = (
            _remove_common_path_gauge(
                r_matrix,
                x_matrix,
                common_drop,
                p_values,
                q_values,
                optimization_gauge_quantile,
            )
        )
        matrix_change = max(
            float(np.max(np.abs(r_matrix - previous_r))),
            float(np.max(np.abs(x_matrix - previous_x))),
        )
        mode_change = float(
            np.linalg.norm(common_drop - previous_common)
            / max(np.linalg.norm(previous_common), 1e-12)
        )
        if max(matrix_change, mode_change) <= tolerance:
            converged = True
            break

    common_blocks = [
        _solve_common_mode(
            target_block - p_block @ r_matrix.T - q_block @ x_matrix.T,
            prior_block,
            prior_weight,
            smoothness_weight,
            common_mode_method,
            huber_delta,
        )
        for target_block, p_block, q_block, prior_block in zip(
            y_blocks,
            p_blocks,
            q_blocks,
            prior_drop_blocks,
            strict=True,
        )
    ]
    common_drop = np.concatenate(common_blocks)
    prediction = common_drop[:, None] + p_values @ r_matrix.T + q_values @ x_matrix.T
    residual = target - prediction
    centered = target - target.mean(axis=0, keepdims=True)
    nrmse = float(
        np.sqrt(np.sum(residual**2) / max(float(np.sum(centered**2)), 1e-15))
    )

    pseudo_series = []
    for root, index, common in zip(roots, indices, common_blocks, strict=True):
        pseudo_vsq = np.maximum(root**2 - common, 1e-12)
        pseudo_series.append(pd.Series(pseudo_vsq, index=index, name=int(pseudo_id)))
    return JointPseudoParentFit(
        pseudo_id=int(pseudo_id),
        members=tuple(member_list),
        r_local=r_matrix,
        x_local=x_matrix,
        pseudo_voltage_squared=tuple(pseudo_series),
        residual_nrmse=nrmse,
        iterations=int(iterations),
        converged=converged,
        gauge_r=float(gauge_r),
        gauge_x=float(gauge_x),
        common_mode_method=common_mode_method,
        anchor_strength=float(anchor_strength),
        prior_weight=float(prior_weight),
        smoothness_weight=float(smoothness_weight),
    )


def aggregate_rooted_scenarios_joint(
    scenarios: list[dict],
    terminals: list[int],
    r_matrix: np.ndarray,
    x_matrix: np.ndarray,
    clusters: list[PseudoCluster],
    *,
    fit_options: dict[str, Any] | None = None,
    blend_with_deembedded: float = 1.0,
    deembedding_weight: float = 0.15,
) -> tuple[list[dict], dict[int, frozenset[int]], dict[int, JointPseudoParentFit]]:
    """Aggregate P/Q and estimate each non-singleton pseudo voltage jointly.

    A blend of one uses the joint estimate and zero uses the deterministic
    de-embedded estimate. Intermediate values shrink the joint squared voltage
    toward that physical anchor.
    """

    if not 0.0 <= blend_with_deembedded <= 1.0:
        raise ValueError("blend_with_deembedded must be in [0, 1]")
    options = dict(fit_options or {})
    position = {int(node): index for index, node in enumerate(terminals)}
    clustered = set().union(*(cluster.members for cluster in clusters)) if clusters else set()
    pseudo_members = {cluster.pseudo_id: cluster.members for cluster in clusters}
    next_id = 910000
    for terminal in terminals:
        if terminal not in clustered:
            pseudo_members[next_id] = frozenset([terminal])
            next_id += 1

    needs_reference = (
        blend_with_deembedded < 1.0 or float(options.get("prior_weight", 0.0)) > 0.0
    )
    reference_scenarios: list[dict] | None = None
    if needs_reference:
        reference_scenarios, reference_members = aggregate_rooted_scenarios(
            scenarios,
            terminals,
            r_matrix,
            x_matrix,
            clusters,
            "deembedded_vsq",
            deembedding_weight=deembedding_weight,
        )
        if reference_members != pseudo_members:
            raise RuntimeError("joint and de-embedded pseudo memberships differ")

    fits: dict[int, JointPseudoParentFit] = {}
    for pseudo_id, members in pseudo_members.items():
        children = sorted(members)
        if len(children) == 1:
            continue
        member_positions = [position[node] for node in children]
        prior = None
        if reference_scenarios is not None:
            prior = [
                pseudo["V_terminal"][pseudo_id].pow(2)
                for pseudo in reference_scenarios
            ]
        fits[pseudo_id] = fit_joint_pseudo_parent_voltage(
            scenarios,
            children,
            pseudo_id,
            r_matrix[np.ix_(member_positions, member_positions)],
            x_matrix[np.ix_(member_positions, member_positions)],
            prior_pseudo_voltage_squared=prior,
            **options,
        )

    aggregated = []
    for scenario_index, scenario in enumerate(scenarios):
        p_terminal = scenario["P_terminal"].loc[:, terminals]
        q_terminal = scenario["Q_terminal"].loc[:, terminals]
        v_terminal = scenario["V_terminal"].loc[:, terminals]
        p_pseudo = pd.DataFrame(index=p_terminal.index)
        q_pseudo = pd.DataFrame(index=q_terminal.index)
        v_pseudo_sq = pd.DataFrame(index=v_terminal.index)
        for pseudo_id, members in pseudo_members.items():
            children = sorted(members)
            p_pseudo[pseudo_id] = p_terminal[children].sum(axis=1)
            q_pseudo[pseudo_id] = q_terminal[children].sum(axis=1)
            if len(children) == 1:
                v_pseudo_sq[pseudo_id] = v_terminal[children[0]].pow(2)
                continue
            joint_vsq = fits[pseudo_id].pseudo_voltage_squared[scenario_index]
            if reference_scenarios is None:
                v_pseudo_sq[pseudo_id] = joint_vsq
            else:
                reference_vsq = reference_scenarios[scenario_index]["V_terminal"][
                    pseudo_id
                ].pow(2)
                v_pseudo_sq[pseudo_id] = (
                    blend_with_deembedded * joint_vsq
                    + (1.0 - blend_with_deembedded) * reference_vsq
                )
        v_pseudo = np.sqrt(np.maximum(v_pseudo_sq, 1e-12))
        aggregated.append(
            {
                "name": scenario["name"],
                "P_terminal": p_pseudo,
                "Q_terminal": q_pseudo,
                "V_terminal": v_pseudo,
                "root_voltage": scenario["root_voltage"],
                "drop_target": squared_voltage_drop_from_observed_root(
                    v_pseudo,
                    scenario["root_voltage"],
                ),
            }
        )
    return aggregated, pseudo_members, fits