# RNJ-wzzT 代码审核指南

更新日期：2026-09-28。本文只描述独立运行权威实现 **rnj_wzzt_core/rnj_wzzt**；
父项目同名脚本是兼容转发，不是第二套算法。

## 一条主调用链

~~~mermaid
flowchart TD
    A[run.py → cli.main: 解析命令行] --> B[pipeline.run]
    B --> C[_prepare_case: 独立训练与验证数据]
    C --> D[preprocessing: 构造已观测根平方电压降 Y]
    D --> E[multiscenario.align_scenarios: 对齐原始 P/Q/Y]
    E --> F[_select_case_rnj: fit_projected_sensitivity]
    F --> G[constrained_least_squares: ordered R/X QP]
    G --> H[sensitivity_geometry + rooted_neighbor_joining: RX75/RNJ]
    H --> I[bootstrap: 重拟合并筛选边界 cherry]
    I --> J[固定原终端 singleton 与筛选后的 RNJ 支撑]
    J --> K[_fit_milp_variants: 原终端 P/Q/Y]
    K --> L[laminar_l1_milp: 数据、模型、求解与前向扩展]
    L --> Q[gurobi_milp: 可选 Gurobi 矩阵适配]
    L --> M[独立验证集选择路径点]
    M --> N[reporting: 结构评分和求解证据]
~~~

**pipeline.run** 现在只负责阶段编排、逐案例落盘和最终配置汇总。数据准备、RNJ
选择和 MILP 完成分别有单一入口，人工审核时不需要在一个长循环中同时跟踪二十多个局部变量。
它同时定义所有计算默认值；`cli.py` 解析命令行，仅将用户明确给出的选项转发给它。
CLI、直接 Python 调用及历史入口默认统一为 reference 场景、完整 RNJ＋MILP、
不运行纯 MILP 对照、输出 `outputs/reference`。历史高级入口没有独立的一套默认配置。
`preprocessing.py` 只把根/终端电压变成目标 `drop_target`；`align_scenarios` 对齐原始
`P/Q/drop_target`，`fit_projected_sensitivity` 才估计 R/X。核心 `multiscenario.py`
只承担原始数据对齐与有序 QP 调用；历史研究的时序配方及 `preprocess_scenarios`
位于 `research_experiments/rnj/temporal_preprocessing.py`（父仓库），不进入此主流程。
`rooted_hierarchy.rooted_clades` 将 RNJ 树转换为终端支撑；伪终端聚合实现已移至
`research_experiments/rnj/rooted_aggregation.py`（父仓库），主流程保持原终端数据和标签。
研究基线所用的独立矩阵修正与诊断位于父项目
`research_experiments/rnj/matrix_constraints.py`，不属于核心有序最小二乘。

核心场景生成器只接受 `exact/noisy` 根观测；固定标称根的 `unobserved` 对照由父仓库
`research_experiments/rnj/scenario_observation.py` 提供。资源表必须显式提供客户类型和
PV/风电/小机组容量，已删除按节点编号推断资源的旧回退。
`network.injection_buses` 和 `candidate_edges` 两个未使用接口已移除；当前 MILP 搜索终端支撑。
曲线统计表、场景真值诊断和完整 AC 输出继续保留。

## 推荐阅读顺序

| 顺序 | 文件或入口 | 先核对什么 |
|---|---|---|
| 1 | README.md；rnj_wzzt/cli.py: main；pipeline.py: run | CLI 解析用户输入；pipeline 定义统一默认值和计算流程 |
| 2 | rnj_wzzt/pipeline.py: run | 只看主流程、训练/验证分离、输出时点 |
| 3 | pipeline.py: _validate_run_options、_prepare_case、_select_case_rnj、_fit_milp_variants | 参数边界、三个阶段的输入输出，以及 RNJ 支持在何处变成固定支持 |
| 4 | scenario/settings.py、simulation.py、profiles.py；models/ac_powerflow.py | 功率符号、量纲、随机种子、根电压过程、量测噪声、AC 数据生成 |
| 5 | estimation/preprocessing.py、multiscenario.py、constrained_least_squares.py | 已观测公共根的平方电压降、无截距设计矩阵、R/X 对称/非负/有序约束 |
| 6 | graph/sensitivity_geometry.py、rooted_neighbor_joining.py、bootstrap.py | RX75 归一化、共享路径、合并阈值、bootstrap 候选筛选 |
| 7 | graph/rooted_hierarchy.py: rooted_clades；pipeline.py: _fit_milp_variants | RNJ clade 转换、固定初始支撑及原终端 MILP 输入；实验聚合模块不进入主流程 |
| 8 | estimation/laminar_l1_milp.py: fit_laminar_l1_sensitivity | 主循环依次做扩展、固定族重拟合、扩界重启、零权重剪枝、路径选择 |
| 8a | 同文件：build_extension_model、build_fixed_model | 按顺序检查残差、非空、乘积线性化、层状及不重复约束 |
| 8b | 同文件：数据类型、normalize_supports、evaluate_l1_matrices | 输入校验、支撑规则、矩阵重建和验证误差 |
| 8c | estimation/gurobi_milp.py | 与 HiGHS 共用的矩阵输入、原始状态映射和求解界 |
| 9 | reporting.py、tests/、docs/research_study_20260908.md | 真值隔离、候选覆盖率、拓扑分数、最新消融与正式默认是否一致 |

**laminar_l1_milp.py** 集中保存数据类型、校验、模型构造、求解与前向路径，按逻辑分段阅读。
`_run_milp`、`_solve_fixed_prepared`、`_solve_extension_prepared`、`_make_path_point`
仍在主模块中定义和调用，`BoundedPath` 等实验适配器继续在这里替换它们。
实验适配器自己管理有限池枚举，核心不再接收 `candidate_supports`。
公共类型和工具函数均在该模块中定义；模型内部的残差 helper 直接接收 `L1Model`。
通用矩阵版和 gurobipy 版共用数据、模型与选模规则；只有最后的求解调用不同。
`gurobi_milp.py` 可直接接收标准 LP/MILP 数组，不依赖层状支撑的定义。

训练和验证表格在拟合入口转换为 `_PreparedScenarios` 后复用，估计系数上界及各个
验证路径点不再重复转换数据。残差约束直接按场景、时刻、输出终端索引，省去扁平目标
及场景列号的中间数组。下界无收益判断、支撑标签转换、上三角索引就地表达。
每轮扩展的 MILP 已联合拟合所有权重，按实际矩阵预测复核目标后直接使用；
固定支撑 LP 仅用于初始化、扩界重启和必要的近零原子剪枝。

运行源码、独立研究、回归测试和本地结果的目录边界见 [README](../README.md#目录用途)。
实验入口统一使用 `run_` 前缀；`research_experiments/rnj/scripts/run_matched_rnj_tuning.py` 留在独立目录，
避免补充脚本进入主实验的源码指纹集合。历史复现使用各次运行保存的源码快照。

## 审核时应先卡住的四个语义

1. **RNJ 支撑作为预设保留。** `initial_supports` 在 MILP 中是固定支持，不是暖启动。
   混合变体将全部原终端 singleton 和被选 RNJ 块放入初始族；不收缩终端，也不改写
   P/Q/Y。固定的是支撑结构，R/X 权重仍参与联合重估。
2. **新支撑直接自由搜索。** 核心不接收候选白名单；MILP 直接决定新支撑的二进制
   成员，并要求与已有 RNJ 预设相容。结果中的 support_search_mode=unrestricted
   是固定语义记录，不再是可选输入。
3. **bootstrap 频率不是后验概率。** 当前只从完整拟合树已有的边界 cherry 中选择稳定
   块；重采样中出现但完整树缺失的 cherry 会记录在候选表，却不会进入 selected。
4. **电压拟合优不等于拓扑真。** 独立验证 MAE 只在给定搜索路径内选模型；有序矩阵
   约束也是树结构的必要型松弛，不证明矩阵必对应某棵树。

## 数据与数学核对点

- 每个场景的 P_terminal、Q_terminal、drop_target 形状均为“时间点 × 终端”。
- 目标为已观测 V_root² - V_terminal²；先按 P 的时间索引对齐 Q 和目标，默认保留原始水平。
- 多场景共享 R、X；没有场景或末端截距，不内部中心化，不在验证集校准偏置。
- RX75 是分别归一化后的 0.75 R + 0.25 X 共享路径组合；R/X 不是比值。
- RNJ 是带双锚分组、中位数更新、裁剪和根停止的有根变体；默认容差为根深度中位数的
  0.16 倍。
- 单步 MILP 的最优证书只覆盖当步给定搜索域；完整前向扩展不因此成为全局树搜索。
- 真值只进入 reporting.py 的评分和审计字段，不应流入候选生成、验证选择或优化器。

## 2026-09-28 已观测根模型

- L2 与 L1 均使用 `Y = P Rᵀ + Q Xᵀ`；删除自由截距、场景指示列、中位数校准及相关结果字段。
- 正式流程直接对齐原始 P/Q/Y，不再传递 raw 空配方；历史研究所需的显式时序变换保留在 `research_experiments/rnj/temporal_preprocessing.py`（父仓库），不会自动去掉绝对观测水平。
- 主流程拒绝 unobserved 根；训练和验证不再为复用截距而改写名称或要求一一对应。
- 普通 MILP 扩展直接使用全部权重，删除同一支撑族的无条件 LP 复解。
- 本轮模型变化与验证记录在 `artifacts/observed_root_model_20260928/`；此前结果需使用当时源码重放。

## 2026-09-27 整理内容（历史记录）

- MILP 运行代码由四个文件合并为两个：`laminar_l1_milp.py` 集中数据、模型与流程，`gurobi_milp.py` 独立适配求解器。
- 数据与模型辅助文件已并回主模块，移除跨文件转发导入；用逻辑分段代替细粒度文件拆分。
- LP 与 MILP 共用基础变量组；R/X 乘积约束共用同一循环；固定族重拟合和路径记录不再重复传递整套参数。
- `L1Model` 把变量索引与约束收在一起；当时版本还含场景截距，已在次日移除。
- 删除核心白名单整理、候选选择器约束、参数别名、CLI 开关及旧候选报告字段。
- 冻结和收缩语义、数值容差、扩界重启、超时处理与验证选择规则保留。
- 纯 MILP 对照的报告不再错误记录 RNJ 预设；RNJ+MILP 继续记录实际保留的 RNJ 块。
- 预处理先对齐后变换，拒绝缺失/重复时间索引，修正乱序 Q/Y 的差分与滚动计算。
- 取消重复可行化、未使用的截距计算、PSD 循环中的伪逆诊断及多余 Series/DataFrame 转换。
- 去嵌入仅对需要的非单点簇构造簇内差分；clade 在过滤大小前检查成员是否合法。
- LinDistFlow 支持观测×注入的矩形灵敏度，复用节点路径与边阻抗。
- AC 入口拒绝非有限输入与非法停止参数，修正 NaN 误报收敛；场景检查观测终端标志。
- 本轮全核心检查的修改前源码与验证记录在 `artifacts/core_audit_20260927/`。
- 早期拆分的验证记录见 [重构记录](core_refactor_20260927.md)；后续合并的源码快照与检查保存在 `artifacts/milp_file_merge_20260927/`。

## 2026-09-08 整理内容（历史记录）

- pipeline.run 从约 267 行降到约 162 行，拆为数据准备、RNJ 选择、MILP 完成三个
  可单独检查的阶段；两份只读阶段记录代替平行局部变量。
- RNJ 汇总行和 bootstrap 候选行移入 reporting.py，算法编排不再内嵌评价字段拼装。
- 参数在创建输出目录前集中校验；此前 bootstrap_replicates=0 会在深层除零，
  block_length=0 会在重采样中失败。
- coefficient_bound 现在写入 metrics.json，可从结果文件还原 MILP 的 R/X 上界。
- 保持每完成一个案例写 RNJ CSV、每完成一个 MILP 变体写 MILP CSV 的原有故障恢复行为。
- 没有改动 R/X QP、RNJ 合并、bootstrap 抽样、收缩公式或 MILP 数值模型。

## 2026-09-08 验证（历史记录）

- 独立核心测试：159 passed。
- 父项目完整兼容测试：123 passed。
- 两个最小真实流程：reference 场景、paper15、4 个观测点、1 次 bootstrap；分别覆盖
  仅 RNJ 选择和 1 秒预算的完整 RNJ+wzzT 写出。

测试通过证明本次职责整理与接口检查未破坏现有回归样例，不构成算法正确性或统计泛化证明。
