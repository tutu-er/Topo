"""固定候选拓扑的非线性 AC P/Q/V 加权最小二乘。

线路 R/X 跨时段共享，分表真实 P/Q 和可选根电压逐时段拟合。每次目标
计算调用现有 AC 潮流，物理平衡不是可被优化器牺牲的软罚项。输入网络
的闭合边代表待检验候选；其 R/X 是初值，函数不会读取 scenario truth。
所有量测及标准差为 pu，线路参数及其界为 ohm。标准差由调用方给定并
冻结，不从本次残差反估。返回 L=sum(r**2)，而非 SciPy 的 cost=L/2。

这是局部最小二乘，不能提供全局最优或拓扑正确性保证。训练完成后可用
fit_impedances=False 在独立数据上校正 P/Q、计算冻结参数下的检验分数。
根电压 sigma=None 表示可信的固定根电压；其余纳入的通道需正标准差。
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, eye, hstack, issparse, kron

from rnj_wzzt.estimation.unmetered_inputs import (
    SourceInputs, _positions, _series_on_index, _terminal_values, prepare_source_inputs,
)
from rnj_wzzt.models.ac_powerflow import _oriented_tree, solve_ac_power_flow_timeseries
from rnj_wzzt.models.network import TerminalizedNetwork


@dataclass(frozen=True)
class ACMeasurementSigma:
    """绝对标准差；不是相对噪声率，也不是物理根电压的波动幅度。

    p/q/v 可为正标量或带时间/终端标签的 DataFrame。其余通道可为正
    标量或带时间标签的 Series；None 表示根电压固定或不使用该总表通道。
    对角权重假定量测误差不相关；相关误差需要另行实现协方差白化。
    """

    p: float | pd.DataFrame
    q: float | pd.DataFrame
    v: float | pd.DataFrame
    root_v: float | pd.Series | None = None
    master_p: float | pd.Series | None = None
    master_q: float | pd.Series | None = None


@dataclass
class ACMeasurementFit:
    """局部拟合结果；success=False 时仍保留真实损失，不能据此正常评分。"""

    loss: float
    initial_loss: float
    success: bool
    status: int
    message: str
    nfev: int
    njev: int | None
    optimality: float
    fitted_net: TerminalizedNetwork
    fitted: dict
    standardized_residuals: dict
    per_time: pd.DataFrame
    ac: dict
    diagnostics: dict


def _positive_scalar(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _aligned_frame(value, reference: pd.DataFrame, name: str) -> np.ndarray:
    values = _terminal_values(value, name)
    rows = _positions(value.index, reference.index, f"{name} time")
    columns = _positions(value.columns, reference.columns, f"{name} terminals")
    return values[np.ix_(rows, columns)]


def _sigma_array(value, reference: pd.DataFrame, name: str, *, terminal: bool) -> np.ndarray:
    if terminal and isinstance(value, pd.DataFrame):
        result = _aligned_frame(value, reference, name)
    elif not terminal and isinstance(value, pd.Series):
        result = _series_on_index(value, reference.index, name)
    elif np.isscalar(value):
        shape = reference.shape if terminal else (len(reference),)
        result = np.full(shape, _positive_scalar(value, name))
    else:
        raise ValueError(f"{name} must be a positive scalar or a labelled {'DataFrame' if terminal else 'Series'}")
    if not np.isfinite(result).all() or (result <= 0.0).any():
        raise ValueError(f"{name} must be finite and positive; zero-noise hard constraints are not supported")
    return result


def _validate_network(net: TerminalizedNetwork, terminals: pd.Index) -> pd.Index:
    if not isinstance(net, TerminalizedNetwork):
        raise ValueError("net must be a TerminalizedNetwork candidate")
    _positive_scalar(net.base_kv, "base_kv")
    _positive_scalar(net.base_mva, "base_mva")
    buses = net.buses["bus_id"].tolist()
    for name, labels in (("bus", buses), ("terminal", terminals.tolist()), ("root", [net.root_bus])):
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in labels):
            raise ValueError(f"{name} labels must be integer bus IDs")
    if len(set(buses)) != len(buses) or not net.branches.index.is_unique:
        raise ValueError("network bus IDs and branch row labels must be unique")
    if net.root_bus not in buses or len(buses) < 2:
        raise ValueError("candidate must contain a root and at least one non-root bus")
    if net.root_bus in terminals or not set(terminals).issubset(buses):
        raise ValueError("terminal columns must identify existing non-root buses")
    if "bus_type" in net.buses:
        expected = set(net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"])
        if expected and set(terminals) != expected:
            raise ValueError("P/Q/V must cover all observed terminal buses")
    active = net.branches.loc[net.branches["is_true_closed"].astype(bool)]
    edges = []
    for row in active.itertuples():
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               for v in (row.from_bus, row.to_bus)):
            raise ValueError("branch endpoints must be integer bus IDs")
        if row.from_bus not in buses or row.to_bus not in buses or row.from_bus == row.to_bus:
            raise ValueError("branch endpoints must be distinct existing buses")
        edges.append(tuple(sorted((row.from_bus, row.to_bus))))
    if len(set(edges)) != len(edges):
        raise ValueError("duplicate closed branches are not supported by radial AC")
    impedances = active[["r_ohm", "x_ohm"]].to_numpy(dtype=float)
    if not np.isfinite(impedances).all() or (impedances < 0.0).any():
        raise ValueError("candidate R/X must be finite and nonnegative")
    _oriented_tree(net)  # Connectivity, acyclicity, root orientation, finite pu values.
    return active.index.copy()


def _impedance_bounds(value, initial: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite (lower, upper) pair in ohm") from exc
    if values.shape != (2,) or not np.isfinite(values).all() or not 0.0 <= values[0] < values[1]:
        raise ValueError(f"{name} must satisfy 0 <= lower < upper < infinity, in ohm")
    if ((initial < values[0]) | (initial > values[1])).any():
        raise ValueError(f"initial impedance is outside {name}")
    return np.full_like(initial, values[0]), np.full_like(initial, values[1])


def _joint_jacobian_rank(jacobian, time_count: int, residual_width: int,
                         state_width: int, shared_width: int, *, rtol: float = 1e-7) -> int:
    """消去逐时段满列秩的 P/Q/(root) 量测块，仅对共享参数做小型秩计算。

    每个状态都有正 sigma 的直接量测，其列独立。完整 J 的秩等于这些
    状态列数加上投影到其正交补后的共享 R/X 列秩，避免巨大稠密 SVD。
    完整 QR 直接给出正交补，避免 B-Q(Q.T@B) 的消去误差制造伪秩。
    相对容差仅用于差分 Jacobian 的数值诊断，不是物理可辨识性证书。
    """
    if shared_width == 0:
        return time_count * state_width
    projected = []
    for t in range(time_count):
        rows = slice(t * residual_width, (t + 1) * residual_width)
        columns = slice(shared_width + t * state_width, shared_width + (t + 1) * state_width)
        local = jacobian[rows, columns]
        shared = jacobian[rows, :shared_width]
        local = local.toarray() if issparse(local) else np.asarray(local)
        shared = shared.toarray() if issparse(shared) else np.asarray(shared)
        orthogonal, _ = np.linalg.qr(local, mode="complete")
        projected.append(orthogonal[:, state_width:].T @ shared)
    singular_values = np.linalg.svd(np.vstack(projected), compute_uv=False)
    shared_rank = int(np.count_nonzero(singular_values > rtol * singular_values[0])) if singular_values.size else 0
    return time_count * state_width + shared_rank


def fit_ac_pqv(
    net: TerminalizedNetwork,
    observed: Mapping[str, object],
    *,
    sigma: ACMeasurementSigma,
    fit_impedances: bool = True,
    r_bounds_ohm: tuple[float, float] | None = None,
    x_bounds_ohm: tuple[float, float] | None = None,
    source: SourceInputs | None = None,
    source_bus_id: int | None = None,
    max_nfev: int = 100,
    ac_max_iter: int = 200,
    ac_tol: float = 1e-12,
    tol: float = 1e-8,
) -> ACMeasurementFit:
    """固定闭合边拓扑，联合拟合共享线路 R/X 与逐时段真实量测状态。

    observed 必须含 P_terminal/Q_terminal/V_terminal、root_voltage；索引
    严格按 P_terminal 对齐，不截断。可选总表仅在提供对应 sigma 时使用。
    不同时计入由这些电压派生的 drop_target，避免重复利用同一量测。
    fit_impedances=True 要求显式有限 ohm 边界；False 固定传入网络参数，
    可用于独立验证。优化内部使用物理量的增量和 3-point 实数差分，
    不使用不适合 conj/abs 的 complex-step。AC失败显式报错，不能计为好分。
    source 为当前观测网格上的固定 pu 额外负荷，不是要拟合的分表负荷。
    源位置及其幅值在此固定；若由同批量测估计，统计校准需包含该步骤。
    """
    required = ("P_terminal", "Q_terminal", "V_terminal", "root_voltage")
    if not isinstance(observed, Mapping) or any(k not in observed for k in required):
        raise ValueError("observed must contain terminal P/Q/V and root_voltage")
    if not isinstance(sigma, ACMeasurementSigma):
        raise ValueError("sigma must be ACMeasurementSigma")
    if not isinstance(fit_impedances, (bool, np.bool_)):
        raise ValueError("fit_impedances must be boolean")
    max_nfev = _positive_integer(max_nfev, "max_nfev")
    ac_max_iter = _positive_integer(ac_max_iter, "ac_max_iter")
    ac_tol = _positive_scalar(ac_tol, "ac_tol")
    tol = _positive_scalar(tol, "tol")
    if tol <= np.finfo(float).eps:
        raise ValueError("tol must exceed machine precision")
    reference = observed["P_terminal"]
    p_obs = _terminal_values(reference, "P_terminal")
    q_obs = _aligned_frame(observed["Q_terminal"], reference, "Q_terminal")
    v_obs = _aligned_frame(observed["V_terminal"], reference, "V_terminal")
    root_obs = _series_on_index(observed["root_voltage"], reference.index, "root_voltage")
    if (v_obs <= 0.0).any() or (root_obs <= 1e-4).any():
        raise ValueError("measured voltage magnitudes must be positive, root_voltage > 1e-4")
    active_index = _validate_network(net, reference.columns)
    working_net = deepcopy(net)
    if fit_impedances:
        # Pandas may reject fractional assignments into integer input columns.
        working_net.branches = working_net.branches.astype({"r_ohm": float, "x_ohm": float})
    initial_r = working_net.branches.loc[active_index, "r_ohm"].to_numpy(dtype=float, copy=True)
    initial_x = working_net.branches.loc[active_index, "x_ohm"].to_numpy(dtype=float, copy=True)
    time_count, n = p_obs.shape
    edge_count = len(active_index)
    sigmas = {
        "P_terminal": _sigma_array(sigma.p, reference, "sigma.p", terminal=True),
        "Q_terminal": _sigma_array(sigma.q, reference, "sigma.q", terminal=True),
        "V_terminal": _sigma_array(sigma.v, reference, "sigma.v", terminal=True),
    }
    measurements = {"P_terminal": p_obs, "Q_terminal": q_obs, "V_terminal": v_obs}
    if sigma.root_v is not None:
        sigmas["root_voltage"] = _sigma_array(sigma.root_v, reference, "sigma.root_v", terminal=False)
        measurements["root_voltage"] = root_obs
    elif observed.get("root_observation") == "noisy":
        raise ValueError("noisy root_voltage requires a positive sigma.root_v")
    for key, scale in (("P0_measured", sigma.master_p), ("Q0_measured", sigma.master_q)):
        if scale is not None:
            if key not in observed:
                raise ValueError(f"{key} is required when its sigma is supplied")
            measurements[key] = _series_on_index(observed[key], reference.index, key)
            sigmas[key] = _sigma_array(scale, reference, f"sigma.{key}", terminal=False)
    if source is None:
        if source_bus_id is not None:
            raise ValueError("source_bus_id requires source amplitudes")
        source_p, source_q = np.zeros(time_count), np.zeros(time_count)
    else:
        if (isinstance(source_bus_id, (bool, np.bool_))
                or not isinstance(source_bus_id, (int, np.integer))
                or source_bus_id == net.root_bus or source_bus_id not in set(net.buses["bus_id"])):
            raise ValueError("source_bus_id must identify an existing non-root bus")
        prepared_source = prepare_source_inputs([observed], [source])
        source_p, source_q = prepared_source.p[0], prepared_source.q[0]

    state_width = 2 * n + int(sigma.root_v is not None)
    state_lower = np.full((time_count, state_width), -np.inf)
    state_upper = np.full((time_count, state_width), np.inf)
    state_scale = np.c_[sigmas["P_terminal"], sigmas["Q_terminal"]]
    if sigma.root_v is not None:
        state_lower[:, -1] = 1e-4 - root_obs
        state_scale = np.c_[state_scale, sigmas["root_voltage"]]
    shared_width = 2 * edge_count if fit_impedances else 0
    if fit_impedances:
        lower_r, upper_r = _impedance_bounds(r_bounds_ohm, initial_r, "r_bounds_ohm")
        lower_x, upper_x = _impedance_bounds(x_bounds_ohm, initial_x, "x_bounds_ohm")
        lower = np.r_[lower_r - initial_r, lower_x - initial_x, state_lower.ravel()]
        upper = np.r_[upper_r - initial_r, upper_x - initial_x, state_upper.ravel()]
        variable_scale = np.r_[np.maximum(initial_r, 1e-3 * (upper_r - lower_r)),
                               np.maximum(initial_x, 1e-3 * (upper_x - lower_x)), state_scale.ravel()]
    else:
        if r_bounds_ohm is not None or x_bounds_ohm is not None:
            raise ValueError("impedance bounds are unused when fit_impedances=False")
        lower, upper, variable_scale = state_lower.ravel(), state_upper.ravel(), state_scale.ravel()
    x0 = np.zeros(shared_width + time_count * state_width)
    channel_keys = tuple(measurements)
    residual_width = sum(n if measurements[k].ndim == 2 else 1 for k in channel_keys)
    local_pattern = kron(eye(time_count, format="csr"), np.ones((residual_width, state_width)), format="csr")
    shared_pattern = np.zeros((residual_width, shared_width))
    shared_pattern[2 * n:3 * n] = 1.0
    master_start = 3 * n + int(sigma.root_v is not None)
    shared_pattern[master_start:] = 1.0
    sparsity = hstack((csr_matrix(np.tile(shared_pattern, (time_count, 1))), local_pattern), format="csr")
    bus_ids = working_net.buses["bus_id"].tolist()
    ac_evaluations = 0
    cached_x, cached_residual, cached_payload = None, None, None

    def evaluate(parameters):
        nonlocal ac_evaluations, cached_x, cached_residual, cached_payload
        if cached_x is not None and np.array_equal(parameters, cached_x):
            return cached_residual
        if fit_impedances:
            working_net.branches.loc[active_index, "r_ohm"] = initial_r + parameters[:edge_count]
            working_net.branches.loc[active_index, "x_ohm"] = initial_x + parameters[edge_count:shared_width]
        states = parameters[shared_width:].reshape(time_count, state_width)
        p_true = p_obs + states[:, :n]
        q_true = q_obs + states[:, n:2 * n]
        root_true = root_obs + states[:, -1] if sigma.root_v is not None else root_obs.copy()
        p_frame = pd.DataFrame(p_true, index=reference.index, columns=reference.columns)
        q_frame = pd.DataFrame(q_true, index=reference.index, columns=reference.columns)
        p_physical = p_frame.reindex(columns=bus_ids, fill_value=0.0).copy()
        q_physical = q_frame.reindex(columns=bus_ids, fill_value=0.0).copy()
        if source is not None:
            p_physical.loc[:, source_bus_id] += source_p
            q_physical.loc[:, source_bus_id] += source_q
        ac_evaluations += 1
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            ac = solve_ac_power_flow_timeseries(
                working_net, p_physical, q_physical, v_root=root_true,
                max_iter=ac_max_iter, tol=ac_tol,
            )
        if not bool(ac["converged"].all()):
            raise RuntimeError("AC power flow did not converge during PQV fitting; no valid score returned")
        heads = [item["label"] for item in ac["metadata"]["branch_edges"]
                 if item["from_bus"] == int(working_net.root_bus)]
        p0 = ac["branch_p_from_pu"][heads].sum(axis=1).to_numpy()
        q0 = ac["branch_q_from_pu"][heads].sum(axis=1).to_numpy()
        values = {
            "P_terminal": p_true, "Q_terminal": q_true,
            "V_terminal": ac["V_bus_mag"].loc[:, reference.columns].to_numpy(),
            "root_voltage": root_true, "P0_measured": p0, "Q0_measured": q0,
        }
        residual_channels = {key: (values[key] - measurements[key]) / sigmas[key] for key in channel_keys}
        residual = np.concatenate([residual_channels[key].reshape(time_count, -1) for key in channel_keys], axis=1)
        if not np.isfinite(residual).all():
            raise FloatingPointError("nonfinite AC measurement residuals")
        cached_x = np.array(parameters, copy=True)
        cached_residual = residual.ravel()
        cached_payload = (ac, values, residual_channels, p_physical, q_physical)
        return cached_residual

    initial_residual = evaluate(x0).copy().reshape(time_count, residual_width)
    solution = least_squares(
        evaluate, x0, jac="3-point", jac_sparsity=sparsity, method="trf",
        bounds=(lower, upper), x_scale=variable_scale, loss="linear",
        # Shared R/X can be weakly excited. Loose inner LSMR solves otherwise
        # cause many tiny outer steps despite a nearly linear local model.
        tr_options={"atol": 1e-10, "btol": 1e-10, "maxiter": 1000},
        max_nfev=max_nfev, ftol=tol, xtol=tol, gtol=tol,
    )
    residual = evaluate(solution.x).copy().reshape(time_count, residual_width)
    ac, values, residual_channels, p_physical, q_physical = cached_payload
    balances = {}
    for symbol, physical in (("p", p_physical), ("q", q_physical)):
        head = values["P0_measured" if symbol == "p" else "Q0_measured"]
        mismatch = head - physical.sum(axis=1).to_numpy() - ac[f"branch_loss_{symbol}_pu"].sum(axis=1).to_numpy()
        balances[symbol] = float(np.max(np.abs(mismatch)))
        if (np.abs(mismatch) > 1e-7 * np.maximum(1.0, np.abs(head))).any():
            raise RuntimeError("fitted AC feeder power balance failed; no valid score returned")
    loss_by_time = np.sum(residual ** 2, axis=1)
    initial_by_time = np.sum(initial_residual ** 2, axis=1)
    rank = _joint_jacobian_rank(solution.jac, time_count, residual_width, state_width, shared_width)
    per_time = pd.DataFrame({
        "initial_loss": initial_by_time, "loss": loss_by_time,
        "residual_count": residual_width,
        "mean_squared_residual": loss_by_time / residual_width,
        "quality_score": 1.0 / (1.0 + loss_by_time / residual_width),
    }, index=reference.index.copy())

    def labelled(key, value):
        if value.ndim == 2:
            return pd.DataFrame(value.copy(), index=reference.index.copy(), columns=reference.columns.copy())
        return pd.Series(value.copy(), index=reference.index.copy(), name=key)

    return ACMeasurementFit(
        loss=float(loss_by_time.sum()), initial_loss=float(initial_by_time.sum()),
        success=bool(solution.success), status=int(solution.status), message=str(solution.message),
        nfev=int(solution.nfev), njev=None if solution.njev is None else int(solution.njev),
        optimality=float(solution.optimality), fitted_net=deepcopy(working_net),
        fitted={key: labelled(key, value) for key, value in values.items()},
        standardized_residuals={key: labelled(key, value) for key, value in residual_channels.items()},
        per_time=per_time, ac=ac,
        diagnostics={
            "solver": "scipy_least_squares_trf", "jacobian": "3-point_sparse",
            "lsmr_atol": 1e-10, "lsmr_btol": 1e-10, "lsmr_maxiter": 1000,
            "loss_definition": "sum_squared_standardized_residuals", "scipy_cost": float(solution.cost),
            "residual_count": int(residual.size), "parameter_count": int(len(x0)),
            "jacobian_rank": rank, "degrees_of_freedom": int(residual.size - rank),
            "jacobian_rank_method": "local_QR_complement_then_shared_SVD",
            "jacobian_rank_rtol": 1e-7,
            "active_bound_count": int(np.count_nonzero(solution.active_mask)),
            "fit_impedances": bool(fit_impedances), "measurement_channels": channel_keys,
            "ac_evaluations": ac_evaluations, "ac_converged": True,
            "p_balance_max_abs_pu": balances["p"], "q_balance_max_abs_pu": balances["q"],
            "global_optimality_certified": False, "quality_score_is_probability": False,
            "scope": "candidate topology with fixed source location and amplitude",
        },
    )


__all__ = ["ACMeasurementSigma", "ACMeasurementFit", "fit_ac_pqv"]
