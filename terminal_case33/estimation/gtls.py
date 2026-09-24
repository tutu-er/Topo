"""Separable generalized total least-squares sensitivity estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from terminal_case33.estimation.matrix_constraints import (
    project_ordered_sensitivity_matrix,
    project_tree_covariance_matrix,
)


@dataclass(frozen=True)
class GTLSSensitivityEstimate:
    """Sensitivity matrices and diagnostics from column-whitened GTLS."""

    r_matrix: np.ndarray
    x_matrix: np.ndarray
    r2_score: float
    augmented_condition_number: float
    response_subspace_condition_number: float
    correction_ratio: float
    predictor_noise_scale: np.ndarray
    response_noise_scale: np.ndarray
    model_mismatch_scale: np.ndarray


def _column_rms(value: np.ndarray) -> np.ndarray:
    """Return stable per-column RMS values."""

    rms = np.sqrt(np.mean(np.asarray(value, dtype=float) ** 2, axis=0))
    positive = rms[rms > 0.0]
    floor = 0.01 * float(np.median(positive)) if positive.size else 1e-12
    return np.maximum(rms, max(floor, 1e-12))


def _project_matrix(matrix: np.ndarray, mode: str) -> np.ndarray:
    """Apply the same physical outer constraints as the OLS estimator."""

    symmetric = np.maximum(0.0, 0.5 * (matrix + matrix.T))
    if mode == "basic":
        return symmetric
    if mode == "ordered":
        return project_ordered_sensitivity_matrix(symmetric)
    if mode == "tree_covariance":
        return project_tree_covariance_matrix(symmetric)
    raise ValueError("unknown sensitivity constraint mode")


def fit_separable_gtls_sensitivity(
    scenarios: list[dict],
    pq_noise_relative_std: float,
    voltage_noise_relative_std: float,
    constraint_mode: str = "ordered",
    shrinkage_to_ols: float = 0.0,
) -> GTLSSensitivityEstimate:
    """Fit reduced R/X by column-whitened multiresponse GTLS.

    Scenario means are removed before fitting, which profiles out one voltage
    intercept per operating scenario. P/Q and squared-voltage-drop columns are
    whitened by noise scales estimated from measured RMS magnitudes. The SVD
    solution is globally optimal for this separable, column-homoscedastic GTLS
    approximation; it is not exact for sample-dependent relative meter noise.

    ``shrinkage_to_ols`` blends the unprojected GTLS coefficient matrix toward
    OLS before physical projection. Zero gives pure GTLS and one gives OLS.
    """

    if not scenarios:
        raise ValueError("at least one scenario is required")
    if pq_noise_relative_std <= 0.0 or voltage_noise_relative_std <= 0.0:
        raise ValueError("meter noise standard deviations must be positive")
    if not 0.0 <= shrinkage_to_ols <= 1.0:
        raise ValueError("shrinkage_to_ols must be in [0, 1]")
    columns = list(scenarios[0]["P_terminal"].columns)
    n = len(columns)
    predictors = []
    predictor_measurements = []
    responses = []
    voltage_squared = []
    for scenario in scenarios:
        p = scenario["P_terminal"].loc[:, columns].to_numpy(dtype=float)
        q = scenario["Q_terminal"].loc[:, columns].to_numpy(dtype=float)
        drop = scenario["drop_target"].loc[:, columns].to_numpy(dtype=float)
        predictors.append(
            np.hstack(
                [
                    p - p.mean(axis=0, keepdims=True),
                    q - q.mean(axis=0, keepdims=True),
                ]
            )
        )
        predictor_measurements.append(np.hstack([p, q]))
        responses.append(drop - drop.mean(axis=0, keepdims=True))
        if "V_terminal" in scenario:
            voltage = scenario["V_terminal"].loc[:, columns].to_numpy(dtype=float)
            voltage_squared.append(voltage**2)

    design = np.vstack(predictors)
    target = np.vstack(responses)
    predictor_scale = pq_noise_relative_std * _column_rms(np.vstack(predictor_measurements))
    if voltage_squared:
        voltage_rms = _column_rms(np.vstack(voltage_squared))
        meter_response_scale = 2.0 * voltage_noise_relative_std * voltage_rms
    else:
        meter_response_scale = 2.0 * voltage_noise_relative_std * np.ones(n)
    ols_coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    ols_residual_variance = np.mean(
        (target - design @ ols_coefficients) ** 2,
        axis=0,
    )
    propagated_predictor_variance = predictor_scale**2 @ (ols_coefficients**2)
    model_mismatch_variance = np.maximum(
        0.0,
        ols_residual_variance - meter_response_scale**2 - propagated_predictor_variance,
    )
    response_scale = np.sqrt(meter_response_scale**2 + model_mismatch_variance)
    predictor_floor = max(float(np.median(predictor_scale)) * 1e-3, 1e-12)
    response_floor = max(float(np.median(response_scale)) * 1e-3, 1e-12)
    predictor_scale = np.maximum(predictor_scale, predictor_floor)
    response_scale = np.maximum(response_scale, response_floor)

    whitened = np.hstack(
        [
            design / predictor_scale[None, :],
            target / response_scale[None, :],
        ]
    )
    _left, singular_values, right_transpose = np.linalg.svd(
        whitened,
        full_matrices=False,
    )
    right = right_transpose.T
    noise_subspace = right[:, -n:]
    predictor_subspace = noise_subspace[: 2 * n]
    response_subspace = noise_subspace[2 * n :]
    whitened_coefficients = -predictor_subspace @ np.linalg.pinv(response_subspace)
    gtls_coefficients = (whitened_coefficients * response_scale[None, :]) / predictor_scale[:, None]
    coefficients = (
        1.0 - shrinkage_to_ols
    ) * gtls_coefficients + shrinkage_to_ols * ols_coefficients
    r_matrix = _project_matrix(coefficients[:n].T, constraint_mode)
    x_matrix = _project_matrix(coefficients[n:].T, constraint_mode)
    projected_coefficients = np.vstack([r_matrix.T, x_matrix.T])
    prediction = design @ projected_coefficients
    residual_sum = float(np.sum((target - prediction) ** 2))
    total_sum = float(np.sum((target - target.mean(axis=0)) ** 2))
    r2_score = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0
    corrected = np.hstack(
        [
            design / predictor_scale[None, :],
            prediction / response_scale[None, :],
        ]
    )
    correction_ratio = np.linalg.norm(whitened - corrected, ord="fro")
    correction_ratio /= max(np.linalg.norm(whitened, ord="fro"), 1e-15)
    return GTLSSensitivityEstimate(
        r_matrix=r_matrix,
        x_matrix=x_matrix,
        r2_score=r2_score,
        augmented_condition_number=float(np.linalg.cond(whitened)),
        response_subspace_condition_number=float(np.linalg.cond(response_subspace)),
        correction_ratio=float(correction_ratio),
        predictor_noise_scale=predictor_scale,
        response_noise_scale=response_scale,
        model_mismatch_scale=np.sqrt(model_mismatch_variance),
    )
