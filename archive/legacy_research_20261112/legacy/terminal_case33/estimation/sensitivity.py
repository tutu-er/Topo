"""Estimate reduced voltage sensitivity matrices from smart-meter data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SensitivityEstimate:
    """Regression output for reduced sensitivity estimation."""

    R_hat: pd.DataFrame
    X_hat: pd.DataFrame
    residuals: pd.DataFrame
    condition_number: float
    r2_score: float
    coefficient_uncertainty: pd.DataFrame


def _preprocess(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "difference":
        return df.diff().dropna()
    if mode == "demean":
        return df - df.mean(axis=0)
    if mode == "highpass":
        return df - df.rolling(window=8, min_periods=1, center=True).mean()
    raise ValueError(f"unknown preprocess: {mode}")


def _preprocess_series(series: pd.Series, mode: str) -> pd.Series:
    """Apply the same temporal preprocessing to a one-dimensional signal."""

    if mode == "difference":
        return series.diff().dropna()
    if mode == "demean":
        return series - series.mean()
    if mode == "highpass":
        return series - series.rolling(window=8, min_periods=1, center=True).mean()
    raise ValueError(f"unknown preprocess: {mode}")


def estimate_reduced_sensitivity(
    V_terminal: pd.DataFrame,
    P_terminal: pd.DataFrame,
    Q_terminal: pd.DataFrame,
    method: str = "ridge",
    preprocess: str = "difference",
    include_root_voltage_mode: bool = True,
    root_voltage: pd.Series | None = None,
    alpha: float = 1e-4,
) -> SensitivityEstimate:
    """Estimate R/X in ``Delta V ~= -R Delta P_load - X Delta Q_load``.

    The returned matrices follow the load-positive convention: a positive load
    coefficient that lowers voltage is reported as positive ``R_hat``/``X_hat``.
    When ``include_root_voltage_mode`` is true, ``root_voltage`` should be the
    measured slack/root voltage; if omitted, the terminal-voltage mean is used
    as a proxy common mode.
    """

    V = _preprocess(V_terminal, preprocess)
    P = _preprocess(P_terminal, preprocess).loc[V.index]
    Q = _preprocess(Q_terminal, preprocess).loc[V.index]
    cols = list(V.columns)
    feature_blocks = [P.to_numpy(), Q.to_numpy()]
    if include_root_voltage_mode:
        if root_voltage is None:
            common = V.mean(axis=1)
        else:
            common = _preprocess_series(root_voltage, preprocess).loc[V.index]
        common = common.to_numpy()[:, None]
        feature_blocks.append(common)
    A = np.hstack(feature_blocks)
    Y = V.to_numpy()
    if method not in {"ols", "ridge", "elasticnet", "huber"}:
        raise ValueError(f"unknown method: {method}")
    reg = alpha * np.eye(A.shape[1]) if method in {"ridge", "elasticnet", "huber"} else np.zeros((A.shape[1], A.shape[1]))
    if include_root_voltage_mode:
        reg[-1, -1] = 0.0
    coef = np.linalg.pinv(A.T @ A + reg) @ A.T @ Y
    y_hat = A @ coef
    residual = Y - y_hat
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    cond = float(np.linalg.cond(A))
    n = len(cols)
    r_hat = -coef[:n, :].T
    x_hat = -coef[n : 2 * n, :].T
    sigma2 = ss_res / max(1, A.shape[0] - A.shape[1])
    cov_diag = np.diag(np.linalg.pinv(A.T @ A + reg)) * sigma2
    uncertainty = np.sqrt(np.maximum(cov_diag[: 2 * n], 0.0)).reshape(2, n).T
    uncertainty_df = pd.DataFrame(uncertainty, index=cols, columns=["P_coef_std", "Q_coef_std"])
    return SensitivityEstimate(
        R_hat=pd.DataFrame(r_hat, index=cols, columns=cols),
        X_hat=pd.DataFrame(x_hat, index=cols, columns=cols),
        residuals=pd.DataFrame(residual, index=V.index, columns=cols),
        condition_number=cond,
        r2_score=float(r2),
        coefficient_uncertainty=uncertainty_df,
    )


def estimate_squared_voltage_sensitivity(
    V_terminal: pd.DataFrame,
    P_terminal: pd.DataFrame,
    Q_terminal: pd.DataFrame,
    root_voltage: pd.Series | None = None,
    include_root_voltage_mode: bool = True,
    fit_intercept: bool = True,
    method: str = "ridge",
    alpha: float = 1e-8,
    load_positive: bool = True,
) -> SensitivityEstimate:
    """Fit ``V^2 ~= C_P P + C_Q Q + a v0^2 + b`` by least squares.

    The fitted model follows the simple squared-voltage objective
    ``||V^2 - (R P + X Q + V0)||``. Because this project uses positive load
    ``P/Q`` values, physical voltage-drop coefficients are negative in the raw
    regression. When ``load_positive`` is true, the returned ``R_hat`` and
    ``X_hat`` are sign-flipped so that larger positive entries mean larger
    load-to-voltage-drop sensitivity and can be used in
    ``d_ij = R_ii + R_jj - 2 R_ij``.
    """

    cols = list(V_terminal.columns)
    common_index = V_terminal.index.intersection(P_terminal.index).intersection(Q_terminal.index)
    V = V_terminal.loc[common_index, cols]
    P = P_terminal.loc[common_index, cols]
    Q = Q_terminal.loc[common_index, cols]
    feature_blocks = [P.to_numpy(), Q.to_numpy()]
    unregularized_cols: list[int] = []
    if include_root_voltage_mode:
        if root_voltage is None:
            common = V.pow(2).mean(axis=1)
        else:
            common = root_voltage.loc[common_index].pow(2)
        unregularized_cols.append(sum(block.shape[1] for block in feature_blocks))
        feature_blocks.append(common.to_numpy()[:, None])
    if fit_intercept:
        unregularized_cols.append(sum(block.shape[1] for block in feature_blocks))
        feature_blocks.append(np.ones((len(common_index), 1)))
    A = np.hstack(feature_blocks)
    Y = V.pow(2).to_numpy()
    if method not in {"ols", "ridge"}:
        raise ValueError(f"unknown method: {method}")
    reg = alpha * np.eye(A.shape[1]) if method == "ridge" else np.zeros((A.shape[1], A.shape[1]))
    for idx in unregularized_cols:
        reg[idx, idx] = 0.0
    coef = np.linalg.pinv(A.T @ A + reg) @ A.T @ Y
    y_hat = A @ coef
    residual = Y - y_hat
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    cond = float(np.linalg.cond(A))
    n = len(cols)
    sign = -1.0 if load_positive else 1.0
    r_hat = sign * coef[:n, :].T
    x_hat = sign * coef[n : 2 * n, :].T
    sigma2 = ss_res / max(1, A.shape[0] - A.shape[1])
    cov_diag = np.diag(np.linalg.pinv(A.T @ A + reg)) * sigma2
    uncertainty = np.sqrt(np.maximum(cov_diag[: 2 * n], 0.0)).reshape(2, n).T
    uncertainty_df = pd.DataFrame(uncertainty, index=cols, columns=["P_coef_std", "Q_coef_std"])
    return SensitivityEstimate(
        R_hat=pd.DataFrame(r_hat, index=cols, columns=cols),
        X_hat=pd.DataFrame(x_hat, index=cols, columns=cols),
        residuals=pd.DataFrame(residual, index=V.index, columns=cols),
        condition_number=cond,
        r2_score=float(r2),
        coefficient_uncertainty=uncertainty_df,
    )


def estimate_coupled_squared_voltage_sensitivity(
    scenarios: list[dict],
    include_root_voltage_mode: bool = True,
    scenario_fixed_effects: bool = True,
    method: str = "ridge",
    alpha: float = 1e-8,
    load_positive: bool = True,
) -> SensitivityEstimate:
    """Estimate one shared squared-voltage sensitivity matrix from scenarios.

    Each scenario dict must contain ``V_terminal``, ``P_terminal``,
    ``Q_terminal`` and may contain ``root_voltage`` and ``name``. The regression
    stacks all time samples from all scenarios and fits one common pair of
    ``P/Q`` coefficient matrices:

    ``V_s^2 ~= C_P P_s + C_Q Q_s + a V0_s^2 + gamma_s + error``.

    ``gamma_s`` is an optional scenario fixed effect. It absorbs scenario-level
    offsets while preserving a shared physical ``R/X`` estimate. With positive
    load data, returned ``R_hat`` and ``X_hat`` are sign-flipped by default.
    """

    if not scenarios:
        raise ValueError("at least one scenario is required")
    first_v = scenarios[0]["V_terminal"]
    cols = list(first_v.columns)
    feature_blocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    index_parts: list[str] = []
    scenario_names: list[str] = []
    for s_idx, scenario in enumerate(scenarios):
        name = str(scenario.get("name", f"scenario_{s_idx}"))
        scenario_names.append(name)
        V_terminal = scenario["V_terminal"]
        P_terminal = scenario["P_terminal"]
        Q_terminal = scenario["Q_terminal"]
        root_voltage = scenario.get("root_voltage")
        common_index = V_terminal.index.intersection(P_terminal.index).intersection(Q_terminal.index)
        V = V_terminal.loc[common_index, cols]
        P = P_terminal.loc[common_index, cols]
        Q = Q_terminal.loc[common_index, cols]
        parts = [P.to_numpy(), Q.to_numpy()]
        if include_root_voltage_mode:
            if root_voltage is None:
                common = V.pow(2).mean(axis=1)
            else:
                common = root_voltage.loc[common_index].pow(2)
            parts.append(common.to_numpy()[:, None])
        feature_blocks.append(np.hstack(parts))
        targets.append(V.pow(2).to_numpy())
        index_parts.extend([f"{name}:{idx}" for idx in common_index])
    base_features = np.vstack(feature_blocks)
    Y = np.vstack(targets)
    extra_blocks: list[np.ndarray] = []
    if scenario_fixed_effects:
        indicator = np.zeros((base_features.shape[0], len(scenarios)))
        start = 0
        for idx, block in enumerate(feature_blocks):
            stop = start + block.shape[0]
            indicator[start:stop, idx] = 1.0
            start = stop
        extra_blocks.append(indicator)
    else:
        extra_blocks.append(np.ones((base_features.shape[0], 1)))
    A = np.hstack([base_features, *extra_blocks])
    if method not in {"ols", "ridge"}:
        raise ValueError(f"unknown method: {method}")
    reg = alpha * np.eye(A.shape[1]) if method == "ridge" else np.zeros((A.shape[1], A.shape[1]))
    first_extra = base_features.shape[1]
    reg[first_extra:, first_extra:] = 0.0
    if include_root_voltage_mode:
        reg[2 * len(cols), 2 * len(cols)] = 0.0
    coef = np.linalg.pinv(A.T @ A + reg) @ A.T @ Y
    y_hat = A @ coef
    residual = Y - y_hat
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    cond = float(np.linalg.cond(A))
    n = len(cols)
    sign = -1.0 if load_positive else 1.0
    r_hat = sign * coef[:n, :].T
    x_hat = sign * coef[n : 2 * n, :].T
    sigma2 = ss_res / max(1, A.shape[0] - A.shape[1])
    cov_diag = np.diag(np.linalg.pinv(A.T @ A + reg)) * sigma2
    uncertainty = np.sqrt(np.maximum(cov_diag[: 2 * n], 0.0)).reshape(2, n).T
    uncertainty_df = pd.DataFrame(uncertainty, index=cols, columns=["P_coef_std", "Q_coef_std"])
    residual_index = pd.Index(index_parts, name="scenario_time")
    return SensitivityEstimate(
        R_hat=pd.DataFrame(r_hat, index=cols, columns=cols),
        X_hat=pd.DataFrame(x_hat, index=cols, columns=cols),
        residuals=pd.DataFrame(residual, index=residual_index, columns=cols),
        condition_number=cond,
        r2_score=float(r2),
        coefficient_uncertainty=uncertainty_df,
    )


def estimate_symmetric_voltage_drop_sensitivity(
    scenarios: list[dict],
    scenario_fixed_effects: bool = True,
    method: str = "ridge",
    alpha: float = 1e-8,
    nonnegative_rx: bool = True,
) -> SensitivityEstimate:
    """Fit a physically constrained voltage-drop model with symmetric R/X.

    The fitted target is the squared-voltage drop from the observed root:

    ``V0_s(t)^2 - V_i,s(t)^2 ~= sum_j R_ij P_j,s(t) + sum_j X_ij Q_j,s(t) + gamma_i,s``.

    ``R`` and ``X`` are parameterized by their upper-triangular entries, so the
    returned matrices are exactly symmetric. The model assumes positive P/Q are
    load quantities and therefore uses positive voltage-drop coefficients.
    """

    if not scenarios:
        raise ValueError("at least one scenario is required")
    cols = list(scenarios[0]["V_terminal"].columns)
    n = len(cols)
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    pair_index = {pair: idx for idx, pair in enumerate(pairs)}
    pair_count = len(pairs)
    fixed_count = len(scenarios) * n if scenario_fixed_effects else n
    var_count = 2 * pair_count + fixed_count

    total_rows = 0
    prepared = []
    for s_idx, scenario in enumerate(scenarios):
        V_terminal = scenario["V_terminal"].loc[:, cols]
        P_terminal = scenario["P_terminal"].loc[:, cols]
        Q_terminal = scenario["Q_terminal"].loc[:, cols]
        drop_target = scenario.get("drop_target")
        root_voltage = scenario.get("root_voltage")
        common_index = V_terminal.index.intersection(P_terminal.index).intersection(Q_terminal.index)
        if drop_target is not None:
            common_index = common_index.intersection(drop_target.index)
        elif root_voltage is not None:
            common_index = common_index.intersection(root_voltage.index)
        else:
            raise ValueError("each scenario needs either drop_target or root_voltage")
        V = V_terminal.loc[common_index]
        P = P_terminal.loc[common_index]
        Q = Q_terminal.loc[common_index]
        root = root_voltage.loc[common_index] if root_voltage is not None else None
        drop = drop_target.loc[common_index, cols] if drop_target is not None else None
        prepared.append((s_idx, str(scenario.get("name", f"scenario_{s_idx}")), V, P, Q, root, drop))
        total_rows += len(common_index) * n

    A = np.zeros((total_rows, var_count), dtype=float)
    y = np.zeros(total_rows, dtype=float)
    row_labels: list[str] = []
    row = 0
    for s_idx, name, V, P, Q, root, drop in prepared:
        p_arr = P.to_numpy(dtype=float)
        q_arr = Q.to_numpy(dtype=float)
        v_arr = V.to_numpy(dtype=float)
        root_sq = root.to_numpy(dtype=float) ** 2 if root is not None else None
        drop_arr = drop.to_numpy(dtype=float) if drop is not None else None
        for t_pos, t_label in enumerate(V.index):
            for obs in range(n):
                for inj in range(n):
                    pair = (inj, obs) if inj <= obs else (obs, inj)
                    idx = pair_index[pair]
                    A[row, idx] += p_arr[t_pos, inj]
                    A[row, pair_count + idx] += q_arr[t_pos, inj]
                if scenario_fixed_effects:
                    A[row, 2 * pair_count + s_idx * n + obs] = 1.0
                else:
                    A[row, 2 * pair_count + obs] = 1.0
                y[row] = drop_arr[t_pos, obs] if drop_arr is not None else root_sq[t_pos] - v_arr[t_pos, obs] ** 2
                row_labels.append(f"{name}:{t_label}:{cols[obs]}")
                row += 1

    if method not in {"ols", "ridge"}:
        raise ValueError(f"unknown method: {method}")
    reg = alpha * np.eye(var_count) if method == "ridge" else np.zeros((var_count, var_count))
    reg[2 * pair_count :, 2 * pair_count :] = 0.0
    normal = A.T @ A + reg
    rhs = A.T @ y
    if nonnegative_rx:
        active = np.zeros(var_count, dtype=bool)
        constrained_count = 2 * pair_count
        coef = np.zeros(var_count, dtype=float)
        for _ in range(30):
            free = ~active
            coef_next = np.zeros(var_count, dtype=float)
            coef_next[free] = np.linalg.pinv(normal[np.ix_(free, free)]) @ rhs[free]
            new_active = active.copy()
            new_active[:constrained_count] |= coef_next[:constrained_count] < 0.0
            coef_next[new_active] = 0.0
            if np.array_equal(new_active, active):
                coef = coef_next
                break
            active = new_active
            coef = coef_next
    else:
        coef = np.linalg.pinv(normal) @ rhs
    y_hat = A @ coef
    residual_vec = y - y_hat
    ss_res = float(np.sum(residual_vec**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    cond = float(np.linalg.cond(A))

    r_hat = np.zeros((n, n), dtype=float)
    x_hat = np.zeros((n, n), dtype=float)
    for (i, j), idx in pair_index.items():
        r_hat[i, j] = r_hat[j, i] = coef[idx]
        x_hat[i, j] = x_hat[j, i] = coef[pair_count + idx]

    residual_wide = pd.DataFrame(
        residual_vec.reshape(-1, n),
        index=pd.Index(row_labels[::n], name="scenario_time"),
        columns=cols,
    )
    sigma2 = ss_res / max(1, total_rows - var_count)
    cov_diag = np.diag(np.linalg.pinv(normal)) * sigma2
    uncertainty = np.zeros((n, 2), dtype=float)
    for obs in range(n):
        diag_pair = pair_index[(obs, obs)]
        uncertainty[obs, 0] = np.sqrt(max(cov_diag[diag_pair], 0.0))
        uncertainty[obs, 1] = np.sqrt(max(cov_diag[pair_count + diag_pair], 0.0))
    uncertainty_df = pd.DataFrame(uncertainty, index=cols, columns=["P_coef_std", "Q_coef_std"])
    return SensitivityEstimate(
        R_hat=pd.DataFrame(r_hat, index=cols, columns=cols),
        X_hat=pd.DataFrame(x_hat, index=cols, columns=cols),
        residuals=residual_wide,
        condition_number=cond,
        r2_score=float(r2),
        coefficient_uncertainty=uncertainty_df,
    )
