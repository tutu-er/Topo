"""AC PQV 残差的统计辅助函数；不求解潮流，也不判断拓扑正确概率。

这里的 loss 是 L=sum(标准化残差**2)，不是 scipy least_squares 的 cost=L/2。
已知噪声协方差、局部正则内点及正确模型下，卡方参照只作局部线性近似。
非线性可行局部解的 L 是全局最小值的上界，不能作为全局拒绝证书。

经验秩要求预先冻结同一个评分过程，并且 H0 校准分数和待测分数可交换。
局部优化器也可校准，但结论只适用于匹配的校准分布；未拒绝不等于模型为真。
若线路或评分规则由数据选择，应使用独立验证，或校准完整选择过程。
比较多条线路还须处理多重检验；本模块不自动执行选择或校正。
同批时段共同拟合共享 R/X 会耦合逐时段分数；估计通过率时，应先冻结训练
得到的网络参数，再使用独立验证时段。拒绝这一固定参数模型，不等于排除
该拓扑下所有可能的 R/X；后者需要处理整个复合零假设。

二项检验和通过率区间要求独立同分布的 Bernoulli 事件及预先冻结的规则。
相关时段不能直接视为独立试验。多个分数复用同一经验秩校准库时，边际
误拒绝率不等于条件误拒绝率，不能直接把拒绝次数当作 Binomial(T, alpha)。
可改为对预先定义的整个窗口统计量做匹配窗口的秩校准，保留窗口内相关性。
不得静默删去失败拟合后再声称结论适用于所有时段。

References:
    https://arxiv.org/abs/2107.07511 (exchangeability and marginal calibration)
    https://www.itl.nist.gov/div898/strd/nls/data/LINKS/c-nelson.shtml
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html
    https://arxiv.org/abs/1303.1288 (exact binomial confidence bounds)
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal, Sequence

import numpy as np
from scipy.stats import beta, binomtest, chi2


def _finite_real(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _counts(successes: int, trials: int) -> tuple[int, int]:
    successes = _nonnegative_integer(successes, "successes")
    trials = _nonnegative_integer(trials, "trials")
    if trials == 0 or successes > trials:
        raise ValueError("counts must satisfy 0 <= successes <= trials and trials > 0")
    return successes, trials


def empirical_upper_tail_pvalue(score: float, null_scores: Sequence[float]) -> float:
    """返回 (1 + count(null_score >= score))/(B+1)，较大 L 表示较差拟合。

    接受单时段或预先定义的窗口评分；校准和待测必须采用同一算法与单位。
    分数须有限非负；失败拟合须预先定义处理规则，不能在本函数中丢弃。
    包含相等分数，使重复值下的检验保持保守；最小可分辨 p 为 1/(B+1)。
    有效性依赖可交换性，返回值不是拓扑/线路为真的概率。
    """
    score = _finite_real(score, "score")
    if score < 0.0:
        raise ValueError("score must be nonnegative")
    values = np.asarray(null_scores)
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iuf":
        raise ValueError("null_scores must be a nonempty one-dimensional real numeric sequence")
    values = values.astype(float)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("null_scores must be finite and nonnegative")
    return float((1 + np.count_nonzero(values >= score)) / (values.size + 1))


def binomial_exceedance_test(
    successes: int,
    trials: int,
    null_rate: float,
    *,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
) -> float:
    """返回独立同分布事件次数的精确二项 p 值，默认检验超阈率过高。

    successes 可表示 L 超过固定阈值的时段数。null_rate 必须有独立依据；
    不能直接用共享校准库的边际 conformal alpha 代替条件超阈概率。
    greater 检验 rate > null_rate（对 rate <= null_rate 的复合零假设也有效）。
    本函数只计算给定检验；不验证独立性，不做多重/连续查看校正。
    """
    successes, trials = _counts(successes, trials)
    null_rate = _finite_real(null_rate, "null_rate")
    if not 0.0 <= null_rate <= 1.0:
        raise ValueError("null_rate must lie in [0, 1]")
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError("alternative must be greater, less, or two-sided")
    return float(binomtest(successes, trials, null_rate, alternative=alternative).pvalue)


def one_sided_rate_lower_bound(
    successes: int, trials: int, *, confidence_level: float = 0.95,
) -> float:
    """独立验证样本通过率的单侧 Clopper-Pearson 下界。

    通过规则必须事先冻结。例如 success 表示 L <= threshold，返回值描述
    该采样分布下的通过概率，不是线路正确概率。冻结规则可以来自另一份
    校准数据；本区间针对它在独立验证时段上的实际通过率。
    在固定样本量的独立同分布 Bernoulli 模型下，覆盖率至少 confidence_level。
    """
    successes, trials = _counts(successes, trials)
    confidence_level = _finite_real(confidence_level, "confidence_level")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if successes == 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence_level, successes, trials - successes + 1))


@dataclass(frozen=True)
class ChiSquareWLSDiagnostic:
    """局部卡方诊断；不包含 accept/reject 或拓扑正确概率字段。"""

    loss: float
    residual_count: int
    jacobian_rank: int
    dof: int
    reduced_loss: float | None
    approx_pvalue: float | None


def chi_square_wls_diagnostic(
    loss: float, residual_count: int, jacobian_rank: int,
) -> ChiSquareWLSDiagnostic:
    """计算自由度 m-rank(J) 的局部近似参照，L 必须是标准化残差平方和。

    J 是对本次实际拟合的连续状态变量的白化残差 Jacobian；不能忽略同窗
    重新拟合的 P/Q、根电压、跨时段共享 R/X 或其他参数；共享 R/X 时应使用
    完整联合残差和 Jacobian，不能逐时段重复扣自由度。固定 R/X 不占本次拟合自由度。
    要求已知且正确的噪声协方差、正确模型和可用的正则内点线性近似。
    活跃边界、强非线性、秩退化、噪声尺度由同残差估计或数据选择模型等
    情况需要另行处理。数值收敛不提供全局最优性，也不保证该分布近似。
    dof=0 返回 None，不制造无剩余约束时的检验。
    """
    loss = _finite_real(loss, "loss")
    if loss < 0.0:
        raise ValueError("loss must be nonnegative")
    residual_count = _nonnegative_integer(residual_count, "residual_count")
    jacobian_rank = _nonnegative_integer(jacobian_rank, "jacobian_rank")
    if residual_count == 0 or jacobian_rank > residual_count:
        raise ValueError("counts must satisfy 0 <= jacobian_rank <= residual_count and residual_count > 0")
    dof = residual_count - jacobian_rank
    return ChiSquareWLSDiagnostic(
        loss=loss,
        residual_count=residual_count,
        jacobian_rank=jacobian_rank,
        dof=dof,
        reduced_loss=loss / dof if dof else None,
        approx_pvalue=float(chi2.sf(loss, dof)) if dof else None,
    )


__all__ = [
    "ChiSquareWLSDiagnostic", "binomial_exceedance_test",
    "chi_square_wls_diagnostic", "empirical_upper_tail_pvalue",
    "one_sided_rate_lower_bound",
]
