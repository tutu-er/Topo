# RX75-RNJ 与 L1-wzzT 独立核心

本目录是当前算法的唯一维护位置，可以单独复制、安装和运行，不导入父项目源码。
默认流程为：多场景约束最小二乘估计 R、X → RX75 共享路径分数 → RNJ 变体 →
循环分块 bootstrap → 筛选可信 RNJ 边界子树并固定其原终端支撑 → MILP 自由搜索相容的新支撑 → 独立验证集选模型。

根节点电压已观测，所有末端共用同一根参考：`Y[t,i] = V_root(t)^2 - V_i(t)^2`。
两个回归阶段均直接拟合 `Y = P @ R.T + Q @ X.T`，不估计截距或节点偏置。
默认使用原始 P/Q/Y，不去均值；节点电压可以不同，公共参考根电压可以随时间变化。

## 运行

在本目录执行：

```powershell
conda activate Topo
python -m pip install -e .
python run.py
```

也可以继续使用 `python -m pip install -r requirements.txt`，该文件引用同一项目依赖声明。
默认使用 SciPy/HiGHS，运行依赖仅为 NumPy、Pandas、SciPy、NetworkX。
可选 Gurobi 后端：`python -m pip install -e ".[gurobi]"`，运行时增加
`--milp-solver gurobi`，并确保许可证可用。模型、参数和代码解释见
[MILP 求解器接口](docs/milp_solvers.md)。

```powershell
python run.py --scenario-suite reference --cases paper15 --output outputs/reference_paper15 --time-limit 30
python run.py --scenario-suite legacy --output outputs/legacy_replay
python -m pip install -e ".[test]"
python -m pytest -q
```

普通 `run.py` 默认使用 `reference` 套件并写入 `outputs/reference`：线路库倍率 1，三种不同工况，
根电压为均值 1.02 p.u.、背景波动尺度 0.003 p.u. 的相关过程，根表计相对误差标准差 0.02%。
仍运行四个案例 paper15、soumalas11、flynn16、pengwah18，每个案例使用 3×96 个训练样本、
独立验证样本和 100 次 bootstrap。`--time-limit` 是每次 LP/MILP 求解的时间上限，并非全流程预算。
新旧设置、独立控制的波动/量测/根可观测性以及压力测试见[算例设置](docs/scenario_design.md)。
场景核验脚本不启动 bootstrap 或 MILP；低数据量从完整 96 点曲线抽取。
`pipeline.run()` 是唯一计算入口和默认参数来源，负责数据准备、RNJ、MILP、验证选模及结果写出。
`cli.py` 负责命令行解析和参数转发，`run.py` 负责启动 CLI；不另存一套算法参数。
普通 CLI、历史高级入口与直接 Python 调用默认均为 `reference`、完整 RNJ＋MILP，
输出 `outputs/reference`，不默认运行纯 MILP 对照。
仅运行 RNJ 筛选需显式使用 `--selection-only`；增加纯 MILP 对照需使用 `--run-baseline`。
Python 调用对应 `selection_only=True`、`run_baseline=True`。这些选项不改变默认主线。
历史 `cli.run`、`cli.advanced_main` 和 `pipeline.main` 继续转发到同一流程。
旧 `--run-milp` 仅作为取消 `--selection-only` 的别名保留，不再隐含纯 MILP 对照；对照需显式 `--run-baseline`。
低层六位置参数 `_simulate_pool` 保留 legacy 数据生成协议以复现研究；它不再决定主流程默认值。
支撑搜索统一为自由扩展；历史白名单 MILP 需用相应运行保存的源码快照重放。

## 目录用途

| 目录 | 内容 |
|---|---|
| `rnj_wzzt/` | 正式运行代码；算法只在这里维护 |
| `experiments/`、`scripts/` | 历史研究路径的薄转发层；实现位于父仓库 `research_experiments/` |
| `tests/` | 主线数值与集成测试；原 `meter_theft_pilot/` 因冻结校验保留原位、不自动收集 |
| `docs/` | 数学说明、审查指南和历史研究记录 |
| `outputs/` | 本地实验结果与复现源码快照，不属于运行依赖 |

独立研究实现与对应测试已移至父仓库 [research_experiments/](../research_experiments/README.md)。
其中 `rnj/scripts/run_matched_rnj_tuning.py` 继续位于主实验源码指纹范围之外。
历史研究的时序变换、聚合及矩阵修正均在该研究目录；正式流程只做原始数据对齐。
旧研究脚本仅在完整仓库中提供兼容转发；单独复制本目录仍可运行主线与核心测试。
`__pycache__/` 和 `.pytest_cache/` 是可再生成的运行缓存。

负非对角项 / log-det 实验使用父仓库 `research_experiments/rnj/run_negative_offdiag_logdet.py`，
原 `experiments/test_negative_offdiag_logdet.py` 已更名。它额外依赖 CVXPY 和 CLARABEL，
可用 `--self-test` 执行内置解析校验，不属于默认核心回归测试。
当前回归/RNJ 研究需加 `--skip-milp`；旧的原生白名单 MILP 对照需用历史源码快照，
脚本会拒绝该旧模式，避免自动改变实验搜索域。

## 源码职责

```text
run.py                         固定主线入口
rnj_wzzt/
├─ cli.py                      统一命令行解析与参数转发
├─ pipeline.py                 回归、RNJ 边界筛选和原终端 MILP 补全的流程编排
├─ reporting.py                真值评分和 MILP 结果记录
├─ data/                       四个案例的网络及资源配置
├─ scenario/
│  ├─ settings.py              算例套件、根波动、观测与阻抗配置
│  ├─ simulation.py            AC 场景模拟、量测噪声及训练/验证样本
│  ├─ profiles.py              异质终端 P/Q 曲线
│  └─ validate_scenario.py     网络与终端观测假设检查
├─ models/                     网络对象、AC 潮流及理想 R/X 物理模型
├─ estimation/
│  ├─ preprocessing.py         已观测根的平方电压降目标
│  ├─ multiscenario.py          原始数据标签对齐、堆叠、初始化及回归诊断
│  ├─ constrained_least_squares.py  对称/非负/有序约束的凸 QP
│  ├─ laminar_l1_milp.py        通用版：数据、约束建模、求解与前向路径
│  └─ gurobi_milp.py            gurobipy 版：标准矩阵输入与状态转换
└─ graph/
   ├─ sensitivity_geometry.py  固定 RX75 归一化与共享路径分数
   ├─ rooted_neighbor_joining.py  RNJ 变体的递归合并
   ├─ bootstrap.py             循环分块重采样和边界子树选择工具
   └─ rooted_hierarchy.py      从 RNJ 树提取非平凡根子树支撑
```

建议从 [结构分析与审查顺序](docs/code_structure.md) 开始。
`estimation/multiscenario.py` 处理原始数据的标签对齐、堆叠和有序 R/X QP 调用；
时序配方及其场景变换位于父仓库 `research_experiments/rnj/temporal_preprocessing.py`，用于研究复现。
MILP 实现集中在两个文件：`laminar_l1_milp.py` 包含完整模型与流程，默认走
SciPy/HiGHS；`gurobi_milp.py` 提供 `solver="gurobi"` 的标准矩阵适配。
R/X 数学模型与数值修正见 [回归审查报告](docs/rx_regression_review.md)。

公共模态、公共模态与双 ridge、当前回归的配对实验见
[公共模态对比结果](docs/common_mode_comparison.md)。实验入口为
`experiments/run_common_mode_comparison.py`，默认主线不随实验切换。

## 算法和兼容边界

RNJ 前的默认 R/X 回归使用平方损失与固定线性约束，是凸 QP；RNJ 后的 L1-MILP
使用绝对值损失并搜索新的树结构原子。两者属于不同阶段。
有序约束不保证矩阵对应某棵树，bootstrap 频率也不是校准后的拓扑后验。

核心场景生成器和主流程均只接受根观测 `exact` 或 `noisy`。`unobserved` 的固定标称根
失配对照保存在父仓库研究目录，核心拒绝该模式。旧 `fixed_intercepts` 输入及结果的
`intercepts` 字段已删除；验证只计算给定 R/X 的原始残差，不在验证集重新校准。
MILP 已联合优化所有权重，普通扩展直接使用输出参数；保留初始化和剪枝时必要的 LP。

RNJ+MILP 将筛选后的 RNJ 块作为固定 `initial_supports`，系数参与联合重估，
并保留全部原终端 singleton；后续 MILP 直接优化二进制支撑，不要求新支撑来自完整 RNJ 树。
主流程不聚合 P/Q、不构造伪终端，也不做电压去嵌入。历史聚合实现位于
`experiments/rooted_aggregation.py`，供独立消融和结果复现使用。
核心已移除白名单模式及兼容别名：`allowed_supports`、`candidate_supports`、
`candidate_pool_mode`、`support_search_mode` 不再是输入参数，相应 CLI 开关也已删除。
传入旧参数会报错；输出中的 `support_search_mode="unrestricted"` 仅记录固定的搜索语义。
独立研究的有限池 LP 枚举保留在 `experiments/rooted_ablation_support.py`，不进入正式模型。

`fit_projected_sensitivity` 只接受 `constraint_mode="ordered"`，保留历史四返回值接口。
`constraint_refine_iterations` 是正整数求解器迭代预算（默认 500）。
依赖旧 `basic` 或 `tree_covariance` 模式的历史对照脚本需用对应源码快照运行。

父项目的两个主线命令及相应算法模块继续转发到这里。研究用矩阵修正与诊断
由父项目实现；部分预处理和层次图工具仍供研究脚本使用。

根电压建模、低样本效果与限时联合求解见[根电压与低样本 MILP 审查](docs/root_voltage_low_sample_milp.md)。

## 固定拓扑后的 AC P/Q/V 联合拟合

`estimation/ac_measurement_fit.py` 提供 `fit_ac_pqv`：固定候选树的连接关系，
联合拟合跨时段共享的线路 R/X、逐时段真实分表 P/Q，以及可选的真实根电压。
它是独立可调用的后处理入口，尚未自动接入 `pipeline.run()` 或 MILP 的候选选择。
现有 MILP 的 L1 目标和这里的 AC 加权平方损失是两个不同目标。

令候选拓扑为 G，线路参数为 theta=(r,x)，真实分表负荷为 p_t、q_t；
固定源位置为 b，固定额外负荷为 uP_t、uQ_t。内部潮流计算使用
`p_physical = p_meter + e_b*uP`、`q_physical = q_meter + e_b*uQ`，
其中 p_meter/q_meter 是待拟合的潜在真实分表负荷，不是直接固定为带噪观测。
没有源时 u=0。非终端节点默认零注入，指定源节点除外。

优化目标是：

```text
min L = sum_t,i [ (p_ti-P_obs_ti)^2/sigmaP_ti^2
                +(q_ti-Q_obs_ti)^2/sigmaQ_ti^2
                +(V_AC_ti(theta,p_t,q_t,v0_t,u_t)-V_obs_ti)^2/sigmaV_ti^2 ]
        + 可选的根电压、总表 P0/Q0 标准化残差平方
s.t. 每条闭合线路的 R/X 位于调用方给定的有限非负区间。
```

每次目标计算都用 `models/ac_powerflow.py` 的径向 AC 后推前推潮流求电压、
根注入和损耗，物理平衡不是可被牺牲的罚项。调用 SciPy 的稀疏三点差分
Jacobian + trust-region reflective 非线性最小二乘；这是利用梯度/Jacobian
的局部优化，不提供全局最优证明。AC 任一试探点不收敛会明确报错；达到
优化预算则返回 `success=False` 和实际损失，不自动给出正常通过结论。
内部稀疏 LSMR 子问题采用 atol=btol=1e-10、maxiter=1000，避免弱激励的
共享 R/X 导致外层大量微小步。`success=True` 表示满足某个停止条件；
应同时检查 `message/status/optimality`，例如 ftol 停止不等于梯度已达到 gtol。
第一版不尝试从失败 AC 点恢复，边界和初值需要处于可求解的运行区间。

输入与输出：

| 参数/字段 | 含义与格式 |
|---|---|
| `net` | 候选 `TerminalizedNetwork`；`is_true_closed` 在此表示候选闭合边；R/X 是物理线路 Ω 初值，不是终端灵敏度矩阵或 MILP 原子系数 |
| `observed` | 原始 `bundle.observed`；P/Q/V 为 T×n DataFrame，列为物理终端 bus ID；`root_voltage` 为 T 长 Series；标签严格对齐 |
| `sigma` | `ACMeasurementSigma(p=...,q=...,v=...)`；绝对标准差 pu，均须正；可为标量或与量测同标签的数据，不能直接填相对噪声率 |
| `sigma.root_v` | 给正标准差则根电压参与拟合；None 表示已知准确根电压；标记 noisy 的根量测必须提供标准差 |
| `sigma.master_p/master_q` | 提供正标准差才将对应 `P0_measured/Q0_measured` 纳入目标，否则忽略该量测 |
| `r_bounds_ohm/x_bounds_ohm` | 每类参数一个 `(lower,upper)`，所有闭合线路共用；有限、非负且包含初值；应来自物理先验 |
| `source/source_bus_id` | 可选 `SourceInputs` 及一个非根物理节点；幅值 pu 按观测时刻对齐；固定 P>=0、Q 可正负；此步不搜索源位置或更新源幅值 |
| `fit_impedances` | 默认 True；False 冻结传入的线路 R/X，但仍校正该批量测的潜在 P/Q/根电压，可用于独立验证 |
| `result.loss` | L=sum(标准化残差²)，不是 SciPy cost；`cost=L/2`；返回 `initial_loss` 供比较 |
| `result.fitted_net` | 独立网络副本，包含拟合后的物理 R/X；不修改输入网络 |
| `result.fitted/ac` | 各拟合量测通道以及完整 AC 电压、支路功率、损耗、收敛记录 |
| `result.per_time` | 逐时段 L、量测数 m、L/m，以及 `quality_score=1/(1+L/m)`；分数越高越好，但不是概率 |
| `result.diagnostics` | 完整残差数、参数数、数值 Jacobian 秩、自由度、活跃界数、AC 次数及功率平衡误差 |

不要传 `align_scenarios` 删减后的字典：这里需要保留原始 V、根电压及可选总表。
不再叠加 `drop_target` 残差，因为它由同一电压量测计算，会重复利用信息。
有功负荷允许负值以支持 DER。线路拓扑、隐藏节点、终端 ID 及源节点映射必须由
调用方明确构造；不能把终端灵敏度 R/X 直接填入物理网络。函数不会读取 truth。
多场景共同拟合时可先将各量测沿时间拼接，使用唯一的 `(scenario,time)` MultiIndex，
所有通道与 sigma/source 同步拼接，终端列保持一致。

```python
from rnj_wzzt.estimation.ac_measurement_fit import ACMeasurementSigma, fit_ac_pqv

# 下面的标准差与边界只是接口示例，应替换为表计规格和线路物理先验。
fit = fit_ac_pqv(
    candidate_net, train_observed,
    sigma=ACMeasurementSigma(p=0.001, q=0.001, v=0.0002, root_v=0.0002),
    r_bounds_ohm=(0.0, 0.1), x_bounds_ohm=(0.0, 0.1), max_nfev=200,
    # source=train_source, source_bus_id=candidate_source_bus,
)
if not fit.success:
    raise RuntimeError(fit.message)
validation = fit_ac_pqv(
    fit.fitted_net, independent_observed, fit_impedances=False,
    sigma=ACMeasurementSigma(p=0.001, q=0.001, v=0.0002, root_v=0.0002),
    # source=validation_source, source_bus_id=candidate_source_bus,
)
if not validation.success:
    raise RuntimeError(validation.message)
print(fit.initial_loss, fit.loss, validation.success, validation.loss)
```

`approx_source_inputs` 的 1.01 倍率估计可以显式传入，但其偏差也会进入 AC 残差。
如果源幅值由同批总分表量测估算，源与量测误差相关；固定幅值后的普通卡方参照
不能自动处理这些相关性，统计校准必须覆盖整个估计流程。当前目标仅实现对角
协方差权重，不支持任意相关量测的协方差白化，也不自动估计噪声标准差。

### L 与假设检验的关系

`estimation/ac_validation.py` 提供四个显式统计辅助接口，不自动宣称线路正确。

- `chi_square_wls_diagnostic(L,m,rank)`：返回 `dof=m-rank(J)`、`L/dof` 和
  卡方上尾近似 p 值。需要已知正确噪声模型、正则内点及局部线性近似；
  活跃界、秩退化、强非线性、同数据选模或估计噪声尺度时不能直接使用。
  R/X 跨时段共享，必须用完整联合 Jacobian，不能每个时段重复扣除这些参数。
  输出秩以局部 QR 正交补 + 共享块 SVD 计算，相对容差 1e-7；是数值诊断，
  不是物理参数可辨识性证明。局部极小值也不能作为全局排除候选的证书。
- `empirical_upper_tail_pvalue(L,null_scores)`：返回
  `(1 + count(null_scores >= L))/(B+1)`。校准和待测必须可交换、采用相同冻结
  评分流程，且 H0 校准分布匹配；可使用整个独立日/窗口的 L 以保留窗口内相关性。
- `binomial_exceedance_test(k,N,q0)`：在独立同分布事件模型下，检验超阈次数
  是否显著高于 q0；q0 必须有独立依据，不能把共用经验校准库的边际 alpha
  直接当作条件超阈率。同一校准库可让不同测试时段的拒绝事件相关。
- `one_sided_rate_lower_bound(k,N,confidence_level=0.95)`：对事先冻结的通过规则，
  用独立验证样本计算通过率的单侧 Clopper-Pearson 下界。59/59 个独立样本通过
  对应下界 95.05%；100/100 对应 97.05%；299/299 对应 99.00%。这描述未来
  同分布样本的通过率，不是“线路正确的概率”。相邻 15 分钟点不能默认独立。

若想由“持续通过”推断线路本身，需要另行证明错误线路模型的通过概率至多 q0。
在这一可证的分离条件和独立性下，错误线路连续 N 次通过的概率至多 q0**N。
没有该条件，错误拓扑也可能通过调整 R/X 拟合相同终端量测，任意多次低残差
都不自动提供线路正确性保证。整体残差偏大也不能直接定位哪一条线路有误。
冻结拟合 R/X 后的拒绝仅针对这个固定参数模型，不能排除该拓扑下所有其他 R/X。

运行可复现实例和新增回归：

```powershell
python demo_ac_measurement_fit.py
python -m pytest tests/test_ac_measurement_fit.py tests/test_ac_validation.py -q
```

本机默认示例实跑（paper15，12 时段、15 分表，已知正确拓扑，无额外源）：

| 阶段 | 初始 L | 最终 L | 停止状态 |
|---|---:|---:|---|
| 训练，共享 R/X | 16615.524533 | 182.984697 | 13 次外层调用，ftol 停止 |
| 独立验证，冻结训练 R/X | 366.325559 | 241.638801 | 6 次外层调用，ftol 停止 |

训练有 6 个活跃边界，optimality=0.8299，未达到 gtol；其停止含义是损失变化
足够小，不是严格一阶或全局最优性证明。验证有 204 个局部剩余自由度，
L/df=1.1845；冻结参数的这一参照不包含训练 R/X 的不确定性，不能据此断言
线路是否正确。求解后的参数也并非逐条精确恢复。不同数值库可能有微小差异。

拟合测试使用独立两节点解析 AC 方程作为数值参照，覆盖参数恢复、P/Q/root 校正、
固定源与总表、标签/单位约束、失败状态及数值秩；统计测试覆盖尾概率、计数边界
和独立计算的区间结果。示例展示正确已知拓扑的数值拟合，尚不是错误线路检出的
功效验证。参考 [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)、
[可交换校准](https://arxiv.org/abs/2107.07511) 和
[精确二项区间](https://arxiv.org/abs/1303.1288)。

## 根信息、结构消融与人工核查研究

2026-09-08 的完整研究见[统一研究报告](docs/research_study_20260908.md)：固定4网络、36个低样本主条件和12个候选扩展条件，分别报告根部、末端、R/X、运行时间与失败。当时结果支持RNJ候选+wzzT的条件性优势；自动冻结/收缩并非该批实验的最优配置。该结论与结果记录属于历史配置，当前主流程见上文。

- [根量测锚定模型与528个回归结果](docs/root_information_model.md)：准确根、噪根、无根量测及公共项/RX正则。
- [先验语义与人工核查排序](docs/prior_acquisition_design.md)：区分完整clade、末端兄弟、根分区和物理接线事实。
- `experiments/run_root_information_study.py`：独立调参/测试的根信息研究。
- `experiments/run_rooted_ablation.py --include-one-edit`：配对消融、固定搜索预算和完整诊断。

两个研究入口可指定`--output`到新目录；默认正式结果、图和原始数组的导航见统一研究报告。

## 独立测试与优势边界压力研究

新增独立数值参照、图与物理性质、MILP枚举和实验协议测试，覆盖最大128终端、弱支路、激励秩亏、异常观测、候选漏失及错误冻结。测试发现并修复负节点标签与自动隐藏节点编号冲突。

- [测试覆盖与复现协议](docs/independent_testing_20260910.md)
- [1508个压力条件的研究结果](docs/advantage_stress_results_20260910.md)
- [几何层分层审阅](docs/advantage_geometry_review.md)
- [优势的理论条件与反例](docs/advantage_theory_boundaries.md)

运行入口为 `experiments/run_advantage_stress.py --output outputs/my_new_stress_run`；生产默认不随本次压力实验更换。
