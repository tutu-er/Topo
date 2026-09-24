"""Sensitivity estimators used by the literature comparison baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from terminal_case33.estimation.matrix_constraints import project_tree_covariance_matrix


@dataclass(frozen=True)
class CurrentSensitivityEstimate:
    """Reduced current-to-voltage sensitivity and common transformer mode."""

    R: np.ndarray
    X: np.ndarray
    transformer_mode: np.ndarray
    residuals: np.ndarray
    r2_score: float
    condition_number: float
    iterations: int


@dataclass(frozen=True)
class PartialMeterImpedanceEstimate:
    """Pengwah 2024 impedance blocks for smart and interval meters."""

    R_SS: np.ndarray
    R_SI: np.ndarray
    X_SS: np.ndarray
    X_SI: np.ndarray
    residuals: np.ndarray
    r2_score: float
    condition_number: float


def _aligned_arrays(
    voltage: pd.DataFrame,
    active_power: pd.DataFrame,
    reactive_power: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = list(voltage.columns)
    index = voltage.index.intersection(active_power.index).intersection(reactive_power.index)
    if len(index) < 3:
        raise ValueError("at least three aligned samples are required")
    v = voltage.loc[index, columns].to_numpy(dtype=float)
    p = active_power.loc[index, columns].to_numpy(dtype=float)
    q = reactive_power.loc[index, columns].to_numpy(dtype=float)
    return v, p, q


def _project_blocks(coefficients: np.ndarray, node_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_matrix = project_tree_covariance_matrix(coefficients[:node_count, :].T, margin_ratio=0.0, max_iterations=5, tolerance=1e-7)
    x_matrix = project_tree_covariance_matrix(coefficients[node_count:, :].T, margin_ratio=0.0, max_iterations=5, tolerance=1e-7)
    projected = coefficients.copy()
    projected[:node_count, :] = r_matrix.T
    projected[node_count:, :] = x_matrix.T
    return projected, r_matrix, x_matrix


def fit_current_sensitivity(
    voltage: pd.DataFrame,
    active_power: pd.DataFrame,
    reactive_power: pd.DataFrame,
    transformer_mode: str = "constant",
    transformer_regularization: float = 0.1,
    max_iterations: int = 10,
    tolerance: float = 1e-8,
) -> CurrentSensitivityEstimate:
    """Fit Pengwah/Flynn current sensitivity from consecutive differences.

    ``constant`` reproduces the pre-Flynn assumption. ``free_regularized``
    estimates one common transformer-voltage difference per time sample as in
    Flynn et al. (2023), with an L2 penalty on that common mode.
    """

    if transformer_mode not in {"constant", "free_regularized"}:
        raise ValueError("transformer_mode must be constant or free_regularized")
    if transformer_regularization < 0.0:
        raise ValueError("transformer_regularization must be nonnegative")
    voltage_array, active, reactive = _aligned_arrays(voltage, active_power, reactive_power)
    current_real = active / np.maximum(voltage_array, 1e-6)
    current_imag = reactive / np.maximum(voltage_array, 1e-6)
    design = np.hstack([np.diff(current_real, axis=0), np.diff(current_imag, axis=0)])
    # Positive load-current changes lower nodal voltage, hence -Delta V has
    # positive path-resistance/reactance coefficients.
    target = -np.diff(voltage_array, axis=0)
    common = np.zeros(len(target), dtype=float)
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    coefficients, r_matrix, x_matrix = _project_blocks(coefficients, voltage_array.shape[1])
    step = 1.0 / max(float(np.linalg.norm(design, ord=2)) ** 2, 1e-15)
    iterations = 1
    outer_iterations = max_iterations if transformer_mode == "free_regularized" else 1
    for iteration in range(outer_iterations):
        previous = coefficients.copy()
        # Projected gradient solves the constrained least-squares problem;
        # projecting an unconstrained fit is not the constrained optimum.
        for _ in range(10):
            residual_now = design @ coefficients + common[:, None] - target
            proposal = coefficients - step * (design.T @ residual_now)
            proposal, proposal_r, proposal_x = _project_blocks(proposal, voltage_array.shape[1])
            old_objective = float(np.sum(residual_now**2))
            new_objective = float(np.sum((design @ proposal + common[:, None] - target) ** 2))
            if new_objective <= old_objective + 1e-15:
                change = np.linalg.norm(proposal - coefficients, ord="fro")
                coefficients, r_matrix, x_matrix = proposal, proposal_r, proposal_x
                if change <= tolerance * max(np.linalg.norm(coefficients, ord="fro"), 1e-15):
                    break
            else:
                step *= 0.5
        if transformer_mode == "free_regularized":
            raw_common = np.mean(target - design @ coefficients, axis=1)
            node_count = voltage_array.shape[1]
            common = raw_common * node_count / (node_count + transformer_regularization)
        iterations = iteration + 1
        relative_change = np.linalg.norm(coefficients - previous, ord="fro")
        relative_change /= max(np.linalg.norm(previous, ord="fro"), 1e-15)
        if relative_change <= tolerance:
            break
    prediction = design @ coefficients + common[:, None]
    residual = target - prediction
    residual_sum = float(np.sum(residual**2))
    total_sum = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    r2_score = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0
    return CurrentSensitivityEstimate(
        R=r_matrix,
        X=x_matrix,
        transformer_mode=common,
        residuals=residual,
        r2_score=r2_score,
        condition_number=float(np.linalg.cond(design)),
        iterations=iterations,
    )


def fit_partial_meter_impedance(
    root_voltage: pd.Series,
    smart_voltage: pd.DataFrame,
    smart_active_power: pd.DataFrame,
    smart_reactive_power: pd.DataFrame,
    interval_active_power: pd.DataFrame,
    refine_iterations: int = 50,
) -> PartialMeterImpedanceEstimate:
    """Fit the Pengwah 2024 ``R_SS, R_SI, X_SS, X_SI`` model.

    Interval-meter voltage and reactive power are deliberately not used.
    Their real current is approximated by active power at nominal 1 p.u.; the
    unidentifiable ``X_SI`` block is returned as zero, matching the paper's
    near-unity-power-factor approximation.
    """

    smart_columns = list(smart_voltage.columns)
    interval_columns = list(interval_active_power.columns)
    index = root_voltage.index.intersection(smart_voltage.index)
    index = index.intersection(smart_active_power.index).intersection(smart_reactive_power.index)
    index = index.intersection(interval_active_power.index)
    if len(index) < 3 or not smart_columns:
        raise ValueError("insufficient aligned smart-meter samples")
    v_s = smart_voltage.loc[index, smart_columns].to_numpy(dtype=float)
    p_s = smart_active_power.loc[index, smart_columns].to_numpy(dtype=float)
    q_s = smart_reactive_power.loc[index, smart_columns].to_numpy(dtype=float)
    p_i = interval_active_power.loc[index, interval_columns].to_numpy(dtype=float)
    y = root_voltage.loc[index].to_numpy(dtype=float)[:, None] - v_s
    ir_s = p_s / np.maximum(v_s, 1e-6)
    ix_s = q_s / np.maximum(v_s, 1e-6)
    design = np.hstack([ir_s, p_i, ix_s])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    ns, ni = len(smart_columns), len(interval_columns)

    def project(candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r_ss = project_tree_covariance_matrix(candidate[:ns, :].T, margin_ratio=0.0)
        r_si = np.maximum(0.0, candidate[ns : ns + ni, :].T)
        r_si = np.minimum(r_si, np.diag(r_ss)[:, None]) if ni else r_si
        x_ss = project_tree_covariance_matrix(candidate[ns + ni :, :].T, margin_ratio=0.0)
        candidate[:ns, :] = r_ss.T
        candidate[ns : ns + ni, :] = r_si.T
        candidate[ns + ni :, :] = x_ss.T
        return r_ss, r_si, x_ss

    r_ss, r_si, x_ss = project(coefficients)
    step = 1.0 / max(float(np.linalg.norm(design, ord=2)) ** 2, 1e-15)
    for _ in range(refine_iterations):
        previous = coefficients.copy()
        gradient = design.T @ (design @ coefficients - y)
        proposal = coefficients - step * gradient
        proposal_r_ss, proposal_r_si, proposal_x_ss = project(proposal)
        if np.sum((design @ proposal - y) ** 2) <= np.sum((design @ coefficients - y) ** 2) + 1e-15:
            coefficients = proposal
            r_ss, r_si, x_ss = proposal_r_ss, proposal_r_si, proposal_x_ss
        else:
            step *= 0.5
        change = np.linalg.norm(coefficients - previous, ord="fro")
        if change <= 1e-8 * max(np.linalg.norm(previous, ord="fro"), 1e-15):
            break
    residual = y - design @ coefficients
    residual_sum = float(np.sum(residual**2))
    total_sum = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2))
    return PartialMeterImpedanceEstimate(
        R_SS=r_ss,
        R_SI=r_si,
        X_SS=x_ss,
        X_SI=np.zeros((ns, ni), dtype=float),
        residuals=residual,
        r2_score=1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0,
        condition_number=float(np.linalg.cond(design)),
    )
