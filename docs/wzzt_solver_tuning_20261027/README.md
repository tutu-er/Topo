# WZZT 求解容差与 AC 辅助判据：2026-10-27 小规模试验

本次目标是减少“已找到可用结构、继续证明极小优化差距”的计算花费，并检查 AC 残差能否辅助路径选模。实现隔离在 `research_experiments/wzzt_solver_tuning/`；原核心、既有算例、用户已有 AC 修改均保持原状。

## MIP gap 的含义

最小化问题的可行候选目标 U 给最优值上界，分支定界下界为 B。Gurobi 使用

`gap = abs(U-B)/abs(U)`。

`MIPGap=0.01` 表示允许 1% 的相对优化间隙；`0` 要求将相对间隙压到零，同时还可能因绝对间隙容差停止。原包装器同时设置 `MIPGapAbs=1e-10`，所以也不等于无限精度。该百分数描述单次 MILP 目标值的未决范围，与拓扑错误率、噪声比例或整个贪心算法的全局最优性不同。定义见 [Gurobi 参数说明](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#mipgap)。

原核心的接收函数还要求严格的 1e-10 数值证书。仅把求解器 gap 改到 1%，原函数仍可能拒绝它。因此本次同时实现显式近似接收规则，保留原始 status、上下界、实际 gap、原约束和整数可行性检查，不把近似解标记为 exact。

## 推荐作为下一轮起点的实验设置

| 项目 | 工作值与规则 |
|---|---|
| 最终 gap 目标 | **0.01% (`1e-4`)**；大增益满足上下界条件时提前退出，无需等到该 gap |
| 每次 / 每方法预算 | 最多 60 秒 / 240 秒；包含 AC 评分；耗尽时保留未决状态 |
| 增益尺度 | `epsilon = 1e-10 + 0.001 * current_MAE`；0.1% 为本轮工作值，尚未跨网络校准为最优 |
| 大增益的提前接收 | 候选原约束和整数可行；实际增益超过 epsilon；优化未决范围不超过实际增益的一半 |
| 无增益停止 | 有效下界证明最大可能增益不超过 epsilon |
| 未决 | 不满足以上条件时收紧 gap；耗尽预算则保留模型并报告 incomplete |
| AC 辅助选点 | 每个路径点做相同预算的局部共享阻抗 AC 拟合；按四个时间块的电压 MSE，取最小值加一个分块标准误差范围内支撑最少的点 |

这些工作值用于生成下一轮可复核结果。当前三条件试验不足以确定通用最优参数。

先试过 1% 粗解，以及 1%→0.1%→0.01% 的分阶段收紧。分阶段版本能解决小增益时的错误停机问题，但重新建模求解会重复计算；在 Pengwah18 上耗尽 240 秒时只接受4步，而不分阶段版本接受8步。因此当前研究入口默认采用**一次求解 + 增益界提前退出**：设置 0.01% 的最终目标，小增益时保留必要分辨率，大增益时由 callback 结束求解。新增默认值只作用于本研究入口，原生产核心不变。

## 按增益决定，而非只看整体目标的百分比

令当前损失为 C，本次扩展的可行候选为 U，有效下界为 B，则

`C-U <= 最优单步增益 <= C-B`。

定义 g=C-U、G=C-B。接收要求 `g>epsilon` 且 `U-B<=0.5*g`，因此 `G<=1.5*g`：该可行候选至少实现当前单步域内最佳可能改善的 2/3。这是目标增益界；候选支撑依然是近似选择，整条路径仍为贪心搜索。

停止要求 `G<=epsilon`。如果候选比当前模型差，但 B 仍较低，则结果是未决，继续细化或保留 incomplete。只凭“当前拿到的候选没改善”停止，会漏掉尚未求出的改善解。

例如训练 MAE 约 4.45e-4 时，1% 优化间隙约为 4.45e-6；已有假簇的目标改善约 7.15e-7，前者约为后者六倍。因而 1% 适合先找候选，精细停止要结合增益上下界。

实现：[policy.py](../../research_experiments/wzzt_solver_tuning/policy.py)、[bounded_solver.py](../../research_experiments/wzzt_solver_tuning/bounded_solver.py)。

## 试验范围与原始结果

固定三项既有无源 AC96 条件：Soumalas11 完整 RNJ 普通初值、Paper15 singleton 初值、Pengwah18 singleton 初值。先重放三项原始首扩展的相同 MILP 矩阵，每项比较 0、1%、5% 三档 gap，各 90 秒；不提供历史 primal 热启动。随后对相同三个初值执行增益界路径试验，另执行带 gap 细化的版本。没有改观测值、读取真源位置或按真实 F1 选择参数。

每一步保留全部原始终端；已有 RX75 尺度来自同一前置回归并固定。此小驱动采用固定 R/X 系数盒 [0,2]，保留原初始支持；达到盒界时单列受限状态。它没有 swap 或安全零权剪枝，不能当作完整核心在生产默认参数下的重跑。结果用于判断近似接收和辅助选点的作用。

原始输出在 [本地结果目录](../../outputs/wzzt_solver_tuning_20261027/)，含冻结协议、代码副本、原始线性约束、primal、整数域、上下界、路径、AC 拟合状态与核验记录。最终数值表见下节及同目录 CSV。

## 实测结果与未完成分母

9/9 项同矩阵 gap 对照返回，9/9 有原约束可行候选；6/9 达到指定 gap，Pengwah18 的三档均在90秒结束时未达到目标。8/8 项路径试验返回，其中4项达到“下界证明无显著单步增益”，4项保持未决；这些是复用三个条件的配置对照，不是8次独立随机重复。

| 固定首扩展 | gap=0 | gap=1% | gap=5% | 观察 |
|---|---|---|---|---|
| Soumalas11 整树初值 | 1.25秒，获严格目标 | 1.13秒，实际gap0.887% | 1.12秒，实际gap3.509% | 后两候选均比当前模型差；下界仍允许改善，须继续细化 |
| Paper15 singleton | 30.45秒 | 34.28秒 | 28.63秒 | 三档候选目标相同；单改 gap 没有稳定提速证据 |
| Pengwah18 singleton | 90秒，gap30.12% | 90秒，gap31.63% | 90秒，gap32.11% | 三档均已取得与归档已证最优值相同的候选；未决花费主要在证明下界 |

按增益界提前退出，在 Pengwah18 的首扩展约37–39秒取得同一候选并接收，退出时gap仍约44.85%；在线依据是有界增益保证，归档最优值仅用于事后核对。Paper15 首扩展约6.6–9.7秒接收的目标为0.0014508305，相对归档最优0.0014417627高约0.63%；保留这项近似代价。

下表为全部路径试验。F1为同一固定 RX75 分辨率口径，对完整物理真簇事后评分，未参与选点。

| 规则 | 条件 | 实测秒 | 求解次数 | 最后点F1 | AC选点F1 | 停止 |
|---|---|---:|---:|---:|---:|---|
| 1%+增益界 | Soumalas11 | 2.07 | 1 | 1.0000 | 1.0000 | 未决 |
| 1%+增益界 | Paper15 | 138.07 | 9 | 0.3077 | 0.3636 | 未决 |
| 1%+增益界 | Pengwah18 | 240.21 | 9 | 0.6154 | 0.6667 | 未决，方法预算耗尽 |
| 分阶段收紧 | Soumalas11 | 9.20 | 4 | 0.8889 | 1.0000 | 下界无显著增益 |
| 分阶段收紧 | Paper15 | 159.06 | 12 | 0.2857 | 0.3077 | 下界无显著增益 |
| 分阶段收紧 | Pengwah18 | 240.08 | 6 | 0.4000 | 0.4000 | 未决，方法预算耗尽 |
| 一次求解+增益界 | Soumalas11 | 4.68 | 2 | 0.8889 | 1.0000 | 下界无显著增益 |
| 一次求解+增益界 | Paper15 | 89.83 | 10 | 0.2857 | 0.3077 | 下界无显著增益 |

一次求解版本只补测了 Soumalas11 和 Paper15，2/2 返回且达到单步无显著增益。Pengwah18 该版本尚未补测，不把前两种版本的结果记为其完成结果。时间为同机非独占实测，各阶段有交叠；可核对具体求解次数和重复求解机制，时间不作隔离硬件速度排名。

Soumalas11 的 AC 规则选择 path0，避免在正确初值上新增假簇。Paper15 的最终恢复仍较差；AC选点只小幅改善F1，尚未解决早期错误支持锁定。当前证据支持采用增益界减少精确证明花费，并将 AC 作辅助选点；不足以将 AC 通过当作正确拓扑证书。

逐项表：[gap_profiles.csv](gap_profiles.csv)、[outcomes.csv](outcomes.csv)、[全部48个路径点](path_points.csv)。原始向量、整数可行性、目标、增益界、AC平衡及实际生产者/输入哈希的独立检查 **429/429通过**，见[核验记录](independent_audit.json)。其中用独立标量潮流核重算了48个路径点的全部96点电压残差和有功/无功平衡。冻结运行依赖的34个源码文件已核验，见[环境与源码回执](runtime_verification.json)。6项针对反例和电气解析解的测试通过。

## AC 判据的具体范围

从当前下游支撑构造径向候选网络，线路标幺阻抗取平方电压响应原子权重的一半。所有终端 P/Q 和根电压使用观测值；内部零注入。每个候选以相同 12 次函数评估预算局部拟合共享线路 R/X，再计算全部 96 点的相对电压残差、AC 收敛、馈线功率平衡与分块 MSE。未收敛的局部拟合仍保留已取得的 AC 可行点，并明确记录优化预算状态。

全部 96 点仍是拟合数据，四块 SE 只作描述性选模尺度；这里没有独立验证集、统计覆盖保证或拓扑正确概率。AC 收敛表示候选可作潮流计算；错误拓扑也可能取得小残差。该准则无法自动撤销搜索早期锁定的正权交叉簇。未知 Source 的 AC 位置/幅值联合适配不在本次无源试验内。

实现：[ac_screen.py](../../research_experiments/wzzt_solver_tuning/ac_screen.py)。

## 复现

命令需要现存冻结运行目录；默认目标目录已存在时拒绝覆盖。重跑时为 `--output-dir` 指定全新目录。

```powershell
& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_solver_tuning.profile_gaps --output-dir outputs/<新目录>/gap_profiles
& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_solver_tuning.analyze_profiles --output-dir outputs/<新目录>/gap_profiles
& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_solver_tuning.run_bounded_paths --coarse --output-dir outputs/<新目录>/bounded_paths
& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_solver_tuning.run_bounded_paths --refine --output-dir outputs/<新目录>/refined_paths
& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_solver_tuning.run_bounded_paths --single-pass --jobs soumalas11__no_source__RNJ+WZZT_tree_init paper15__no_source__WZZT --output-dir outputs/<新目录>/single_pass_paths
& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_solver_tuning.summarize --root outputs/<新目录>
```

回归检查：`python -B -m pytest -q research_experiments/tests/test_wzzt_solver_tuning.py`。包括“差候选不能证明无改善”的反例、2/3 增益界、缺失上下界、整数可行性，以及以单线路解析 DistFlow 独立核验 AC 权重因子2。
