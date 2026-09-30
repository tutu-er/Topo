"""固定正确拓扑下，共享线路 R/X 的 AC 量测拟合与独立验证示例。

运行：在 rnj_wzzt_core 目录执行 python -X utf8 demo_ac_measurement_fit.py。
本例只调用已有接口，不实现优化算法。两天先生成无噪声量测，再由不同
随机种子添加已知、固定绝对标准差的 pu 高斯噪声；不读取 bundle.truth。
这是已知正确物理拓扑下的数值演示，不是拓扑恢复或统计保证的验证。
"""

from copy import deepcopy
from time import perf_counter

import numpy as np
from scipy.stats import chi2

from rnj_wzzt.estimation.ac_measurement_fit import ACMeasurementSigma, fit_ac_pqv
from rnj_wzzt.scenario.unmetered_load import simulate_unmetered_scenario


CASE_KEY = "paper15"
T_COUNT = 12  # 每天从完整 96 点物理过程均匀抽取 12 点；可改为 8 等因数。
MAX_NFEV = 150
# 以下是绝对 pu 标准差，直接用于造噪声与拟合权重，不从残差估计。
SIGMA = ACMeasurementSigma(
    p=2e-4, q=1e-4, v=1e-4, root_v=None, master_p=4e-4, master_q=2e-4,
)
# 显式固定的 ohm 搜索范围；reference paper15 的线路初值均在此范围内。
R_BOUNDS_OHM = (0.0, 0.15)
X_BOUNDS_OHM = (0.0, 0.15)


def _noisy_observations(replicate: int, noise_seed: int):
    """只从 observed 构造输入；物理日与量测噪声均与另一数据集独立。"""
    net, bundle = simulate_unmetered_scenario(
        CASE_KEY, replicate=replicate, t_count=T_COUNT, scenario_suite="reference",
        source=None, root_observation="exact", pq_noise_rel=0.0, v_noise_rel=0.0,
        master_p_noise_rel=0.0, master_q_noise_rel=0.0,
    )
    rng = np.random.default_rng(noise_seed)
    observed = {
        "name": f"{bundle.observed['name']}_noise{noise_seed}",
        "root_voltage": bundle.observed["root_voltage"].copy(),
        "root_observation": "exact",
    }
    for key, std in (
        ("P_terminal", SIGMA.p), ("Q_terminal", SIGMA.q), ("V_terminal", SIGMA.v),
        ("P0_measured", SIGMA.master_p), ("Q0_measured", SIGMA.master_q),
    ):
        clean = bundle.observed[key]
        observed[key] = clean + rng.normal(0.0, std, size=clean.shape)
    return net, observed


def _print_fit(label: str, fit, elapsed: float):
    diagnostic = fit.diagnostics
    rank = diagnostic["jacobian_rank"]
    dof = diagnostic["degrees_of_freedom"]
    active = diagnostic["active_bound_count"]
    print(f"{label}：耗时 {elapsed:.3f} s；success={fit.success}，status={fit.status}")
    print(f"  L 初值={fit.initial_loss:.6f}，最终={fit.loss:.6f}；{fit.message}")
    print(f"  Jacobian 秩={rank}，局部自由度={dof}，活动边界数={active}")
    print(f"  optimality={fit.optimality:.3e}，优化函数次数={fit.nfev}，"
          f"实际 AC 调用次数={diagnostic['ac_evaluations']}")
    print(f"  AC 平衡误差 P={diagnostic['p_balance_max_abs_pu']:.3e} pu，"
          f"Q={diagnostic['q_balance_max_abs_pu']:.3e} pu")
    if fit.success and dof > 0:
        lo, hi = chi2.ppf([0.025, 0.975], dof)
        print(f"  局部 χ²({dof}) 参考：L/df={fit.loss / dof:.4f}，"
              f"名义 95% 中心区间=[{lo:.3f}, {hi:.3f}]")
        if active:
            print("  存在活动边界，上述局部 χ²近似可能失真。")
    else:
        print("  未收敛或局部自由度不足：保留损失与状态，不作 χ²解释。")


def _print_impedances(physical_net, initial_net, fitted_net):
    active = physical_net.branches["is_true_closed"]
    for name in ("r_ohm", "x_ohm"):
        print(f"{name} 摘要（最小 / 中位 / 最大，单位 Ω）：")
        for label, net in (("物理设定（仅对照）", physical_net),
                           ("拟合初值", initial_net), ("训练拟合", fitted_net)):
            values = net.branches.loc[active, name].to_numpy(dtype=float)
            print(f"  {label}：{values.min():.6f} / {np.median(values):.6f} / {values.max():.6f}")


def main():
    start = perf_counter()
    physical_net, training = _noisy_observations(replicate=0, noise_seed=2026093001)
    _, validation = _noisy_observations(replicate=1, noise_seed=2026093002)
    candidate = deepcopy(physical_net)
    candidate.branches.loc[:, "r_ohm"] *= 0.8
    candidate.branches.loc[:, "x_ohm"] *= 1.2

    print(f"{CASE_KEY}：已知正确拓扑，每天 {T_COUNT} 个观测点，"
          f"{training['P_terminal'].shape[1]} 个分表，max_nfev={MAX_NFEV}")
    print("训练/验证：replicate=0/1，噪声种子=2026093001/2026093002；根电压固定精确值。")
    print("绝对 pu 标准差：P=2e-4，Q=1e-4，V=1e-4，P0=4e-4，Q0=2e-4。")
    print(f"训练阻抗范围：R={R_BOUNDS_OHM} Ω，X={X_BOUNDS_OHM} Ω；初值 R×0.8、X×1.2。")

    stage_start = perf_counter()
    try:
        trained = fit_ac_pqv(
            candidate, training, sigma=SIGMA, fit_impedances=True,
            r_bounds_ohm=R_BOUNDS_OHM, x_bounds_ohm=X_BOUNDS_OHM,
            max_nfev=MAX_NFEV,
        )
    except (RuntimeError, FloatingPointError, ValueError) as exc:
        print(f"训练失败（{perf_counter() - stage_start:.3f} s），无有效分数：{exc}")
        return 2
    _print_fit("训练：共享 R/X，逐时段校正 P/Q", trained, perf_counter() - stage_start)
    _print_impedances(physical_net, candidate, trained.fitted_net)
    if not trained.success:
        print("训练未收敛；下面仅冻结当前参数做诊断验证，不视为正常统计评分。")

    stage_start = perf_counter()
    try:
        validated = fit_ac_pqv(
            trained.fitted_net, validation, sigma=SIGMA, fit_impedances=False,
            max_nfev=MAX_NFEV,
        )
    except (RuntimeError, FloatingPointError, ValueError) as exc:
        print(f"验证失败（{perf_counter() - stage_start:.3f} s），无有效分数：{exc}")
        return 2
    _print_fit("独立验证：冻结训练 R/X，逐时段校正 P/Q", validated, perf_counter() - stage_start)
    print("χ²仅是局部线性、正则条件下的参考，不输出拓扑正确概率，也不判定统计通过。")
    print("验证仍拟合本日潜在 P/Q；上述自由度未计入训练阻抗的不确定性。")
    print(f"总耗时：{perf_counter() - start:.3f} s；这是数值演示，不是拓扑恢复验证。")
    return 0 if trained.success and validated.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
