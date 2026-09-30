"""额外负荷辨识的输入层：观测差额、1.01 倍率近似、幅值校验与对齐。

职责边界：
* scenario/unmetered_load.py 生成物理场景与量测；本文件只接收观测或外部幅值。
* 总分表差额 = 额外负荷 + 线损 + 量测误差，不能直接当作源真值。
* approx_source_inputs 按固定 1.01 倍率假设构造近似幅值，不辨识真实线损。
* 本文件不读取 truth/net，不选择源位置，也不调用 MILP。
* 所有功率均为 pu；kW/kvar 到 pu 的转换由调用方显式完成。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MeterBalance:
    """带符号的总分表差额；仍包含线损与量测误差。"""

    p_gap_pu: pd.Series
    q_gap_pu: pd.Series


@dataclass(frozen=True)
class SourceInputs:
    """一个场景的待校验固定源幅值，不包含源位置。

    p_pu/q_pu 必须覆盖该场景所有观测时刻；prepare_source_inputs 要求
    P 非负、Q 允许正负。approx_source_inputs 的原始近似可含负 P，
    调用方可显式选择 project，再交给 prepare_source_inputs。
    provenance 记录幅值来源，例如 oracle_test 或 meter_estimate，
    仅供审计，不触发任何估计或真值读取。校验在 prepare_source_inputs 中进行。
    """

    p_pu: pd.Series
    q_pu: pd.Series
    provenance: str


@dataclass(frozen=True)
class PreparedSourceInputs:
    """供后续源模型使用的数组：每个场景一个 shape=(T_s,) 的块。"""

    p: tuple[np.ndarray, ...]
    q: tuple[np.ndarray, ...]
    time_indices: tuple[pd.Index, ...]
    provenance: tuple[str, ...]

    @property
    def all_zero(self) -> bool:
        """仅标记 P/Q 全零；由后续求解入口决定是否走无源路径。"""
        return all(not np.any(p != 0.0) and not np.any(q != 0.0)
                   for p, q in zip(self.p, self.q, strict=True))


def _positions(labels: pd.Index, reference: pd.Index, name: str) -> np.ndarray:
    """允许重排；不补齐、不截断、不插值。"""
    if not labels.is_unique or len(labels) != len(reference):
        raise ValueError(f"{name} must contain the same unique labels")
    positions = labels.get_indexer(reference)
    if (positions < 0).any():
        raise ValueError(f"{name} must contain the same unique labels")
    return positions


def _numeric_copy(value: pd.Series | pd.DataFrame, name: str) -> np.ndarray:
    try:
        values = value.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def _terminal_values(value: object, name: str) -> np.ndarray:
    if not isinstance(value, pd.DataFrame) or value.empty:
        raise ValueError(f"{name} must be a nonempty DataFrame")
    if not value.index.is_unique or not value.columns.is_unique:
        raise ValueError(f"{name} time and terminal labels must be unique")
    return _numeric_copy(value, name)


def _series_on_index(value: object, index: pd.Index, name: str) -> np.ndarray:
    if not isinstance(value, pd.Series):
        raise ValueError(f"{name} must be a Series")
    positions = _positions(value.index, index, name)
    return _numeric_copy(value, name)[positions]


def _meter_channels(
    observed: Mapping[str, object],
) -> tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """统一校验并对齐量测，返回时间索引、分表 P/Q 总量、总表 P/Q。"""
    required = ("P_terminal", "Q_terminal", "P0_measured", "Q0_measured")
    if not isinstance(observed, Mapping) or any(key not in observed for key in required):
        raise ValueError("observed must contain terminal P/Q and measured master P/Q")
    p_frame, q_frame = observed["P_terminal"], observed["Q_terminal"]
    # 先验证每个值，避免 pandas.sum 默认忽略缺失分表读数。
    p = _terminal_values(p_frame, "P_terminal")
    q = _terminal_values(q_frame, "Q_terminal")
    q_rows = _positions(q_frame.index, p_frame.index, "Q_terminal time")
    q_cols = _positions(q_frame.columns, p_frame.columns, "Q_terminal terminals")
    q = q[np.ix_(q_rows, q_cols)]
    p0 = _series_on_index(observed["P0_measured"], p_frame.index, "P0_measured")
    q0 = _series_on_index(observed["Q0_measured"], p_frame.index, "Q0_measured")
    return p_frame.index.copy(), p.sum(axis=1), q.sum(axis=1), p0, q0


def compute_meter_balance(observed: Mapping[str, object]) -> MeterBalance:
    """计算 P0-sum(P_i)、Q0-sum(Q_i)，结果按 P_terminal 的行顺序返回。

    输入为 bundle.observed，必须保留 P0_measured/Q0_measured。
    此函数不会截掉负差额，也不会扣除任何假定线损。
    """
    index, p_sum, q_sum, p0, q0 = _meter_channels(observed)
    return MeterBalance(
        p_gap_pu=pd.Series(p0 - p_sum, index=index.copy(), name="P_meter_gap_pu"),
        q_gap_pu=pd.Series(q0 - q_sum, index=index.copy(), name="Q_meter_gap_pu"),
    )


def prepare_source_inputs(
    scenarios: Sequence[Mapping[str, object]],
    source_inputs: Sequence[SourceInputs] | None,
) -> PreparedSourceInputs | None:
    """校验外部幅值，按每个场景 P_terminal 的时间顺序生成独立数组副本。

    scenarios 可为原始 observed 列表，也可为 align_scenarios 的结果。
    source_inputs 按场景列表顺序一一对应，不能自动广播。
    None 原样返回；显式全零输入保留，并由 all_zero 标记。
    本函数不从总分表差额自动生成源幅值。
    """
    if source_inputs is None:
        return None
    if not isinstance(source_inputs, Sequence) or isinstance(source_inputs, (str, bytes)):
        raise ValueError("source_inputs must be a sequence of SourceInputs or None")
    if not scenarios or len(source_inputs) != len(scenarios):
        raise ValueError("source_inputs must have one entry per nonempty scenario list")

    p_blocks, q_blocks, indices, provenance = [], [], [], []
    for scenario, source in zip(scenarios, source_inputs, strict=True):
        if not isinstance(scenario, Mapping) or "P_terminal" not in scenario:
            raise ValueError("each scenario must contain P_terminal")
        p_frame = scenario["P_terminal"]
        _terminal_values(p_frame, "P_terminal")
        if not isinstance(source, SourceInputs):
            raise ValueError("each source input must be SourceInputs")
        if not isinstance(source.provenance, str) or not source.provenance.strip():
            raise ValueError("source provenance must be a nonempty string")
        p = _series_on_index(source.p_pu, p_frame.index, "source p_pu")
        q = _series_on_index(source.q_pu, p_frame.index, "source q_pu")
        if (p < 0.0).any():
            raise ValueError("source p_pu must be nonnegative; estimate it explicitly before preparation")
        p_blocks.append(p)
        q_blocks.append(q)
        indices.append(p_frame.index.copy())
        provenance.append(source.provenance)
    return PreparedSourceInputs(tuple(p_blocks), tuple(q_blocks), tuple(indices), tuple(provenance))


def approx_source_inputs(
    observed: Mapping[str, object],
    *,
    negative_p_policy: Literal["keep", "project"] = "keep",
) -> SourceInputs:
    """按用户指定的 1.01 倍率公式构造近似源幅值，返回独立 Series。

    uP = (P0 - 1.01 * sum(P_i)) / 1.01，Q 同理。
    对应假设为 P0 ≈ 1.01 * (sum(P_i) + uP)，不读取真实线损。
    observed 必须含 P_terminal/Q_terminal/P0_measured/Q0_measured；
    全部功率单位为 pu，输出按 P_terminal 的时间顺序排列。

    keep：默认完整保留有符号结果，与原始公式一致。
    project：显式取有功正部，Q 仍保留正负；provenance 标记该投影。
    prepare_source_inputs 仍拒绝负 P，因此含负值的 keep 结果不能直接准备。
    此固定倍率只是近似假设，不构成损耗校准或窃电判定。
    """
    if negative_p_policy not in {"keep", "project"}:
        raise ValueError("negative_p_policy must be 'keep' or 'project'")
    approx_loss_ratio = 1.01
    index, p_sum, q_sum, p0, q0 = _meter_channels(observed)
    p_source_approx = pd.Series(
        (p0 - p_sum * approx_loss_ratio) / approx_loss_ratio,
        index=index.copy(), name="P_unmetered_approx_pu",
    )
    q_source_approx = pd.Series(
        (q0 - q_sum * approx_loss_ratio) / approx_loss_ratio,
        index=index.copy(), name="Q_unmetered_approx_pu",
    )
    if not np.isfinite(p_source_approx.to_numpy()).all() or not np.isfinite(q_source_approx.to_numpy()).all():
        raise ValueError("approximate source amplitudes must be finite")
    provenance = "approx_source_inputs"
    if negative_p_policy == "project":
        p_source_approx = p_source_approx.clip(lower=0.0)
        provenance = "approx_source_inputs_projected"
    return SourceInputs(p_source_approx, q_source_approx, provenance)
__all__ = [
    "MeterBalance", "SourceInputs", "PreparedSourceInputs",
    "compute_meter_balance", "prepare_source_inputs", "approx_source_inputs",
]
