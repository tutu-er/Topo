# RNJ-wzzT 代码审核指南

更新日期：2026-09-08。本文只描述独立运行权威实现 **rnj_wzzt_core/rnj_wzzt**；
父项目同名脚本是兼容转发，不是第二套算法。

## 一条主调用链

~~~mermaid
flowchart TD
    A[run.py / cli.run] --> B[pipeline.run]
    B --> C[_prepare_case: 独立训练与验证数据]
    C --> D[preprocessing + multiscenario]
    D --> E[constrained_least_squares: 有序 R/X QP]
    E --> F[_select_case_rnj]
    F --> G[sensitivity_geometry: RX75]
    G --> H[rooted_neighbor_joining]
    H --> I[bootstrap: 边界 cherry 稳定频率]
    I --> J[_fit_milp_variants]
    J --> K[rooted_hierarchy: 可选收缩/去嵌入]
    J --> L[laminar_l1_milp: 有限候选前向扩展]
    L --> M[独立验证集选择路径点]
    M --> N[reporting: 结构评分和求解证据]
~~~

**pipeline.run** 现在只负责阶段编排、逐案例落盘和最终配置汇总。数据准备、RNJ
选择和 MILP 完成分别有单一入口，人工审核时不需要在一个长循环中同时跟踪二十多个局部变量。

## 推荐阅读顺序

| 顺序 | 文件或入口 | 先核对什么 |
|---|---|---|
| 1 | README.md；rnj_wzzt/cli.py: run | 正式默认值；注意普通入口和高级/历史入口的默认场景不同 |
| 2 | rnj_wzzt/pipeline.py: run | 只看主流程、训练/验证分离、输出时点 |
| 3 | pipeline.py: _validate_run_options、_prepare_case、_select_case_rnj、_fit_milp_variants | 参数边界、三个阶段的输入输出，以及 RNJ 支持在何处变成固定支持 |
| 4 | scenario/settings.py、simulation.py、profiles.py；models/ac_powerflow.py | 功率符号、量纲、随机种子、根电压过程、量测噪声、AC 数据生成 |
| 5 | estimation/preprocessing.py、multiscenario.py、constrained_least_squares.py | 平方电压降、场景中心化、设计矩阵、R/X 对称/非负/有序约束 |
| 6 | graph/sensitivity_geometry.py、rooted_neighbor_joining.py、bootstrap.py | RX75 归一化、共享路径、合并阈值、bootstrap 候选筛选 |
| 7 | pipeline.py 的收缩/候选映射函数；graph/rooted_hierarchy.py | 伪终端映射、去嵌入、候选能否切穿已收缩块 |
| 8 | estimation/laminar_l1_milp.py: fit_laminar_l1_sensitivity | 先读公开函数和主循环，再读固定族拟合、单步扩展 MILP、剪枝及求解诊断 |
| 9 | reporting.py、tests/、docs/research_study_20260908.md | 真值隔离、候选覆盖率、拓扑分数、最新消融与正式默认是否一致 |

**laminar_l1_milp.py** 较长，但实验辅助代码会替换其中若干内部求解步骤；当前没有为缩短
文件而机械拆开，避免破坏可复现实验和 monkeypatch 兼容边界。

## 审核时应先卡住的四个语义

1. **RNJ 支持是否应冻结。** initial_supports 在 MILP 中是固定支持，不是暖启动。
   因而即使 contract_blocks=False，混合变体仍会冻结被选 RNJ 块；开启收缩只是进一步
   改写数据和终端集合。最新消融中“RNJ 候选+wzzT、不冻结”平均结果更好，正式默认
   是否继续冻结应由人工明确决定，不能从变量名 initializer 推断为可撤销初始化。
2. **候选覆盖不是恢复保证。** 默认 candidate_pool_mode=rnj 只搜索完整 RNJ 树
   映射出的有限 clade；候选池缺少真 clade 时，MILP 无法创造它。rnj_one_edit
   扩池仍受层状约束、前向路径和时间预算限制。
3. **bootstrap 频率不是后验概率。** 当前只从完整拟合树已有的边界 cherry 中选择稳定
   块；重采样中出现但完整树缺失的 cherry 会记录在候选表，却不会进入 selected。
4. **电压拟合优不等于拓扑真。** 独立验证 MAE 只在给定搜索路径内选模型；有序矩阵
   约束也是树结构的必要型松弛，不证明矩阵必对应某棵树。

## 数据与数学核对点

- 每个场景的 P_terminal、Q_terminal、drop_target 形状均为“时间点 × 终端”。
- 目标为 V_root² - V_terminal²；P、Q、目标必须使用一致时序变换。
- 多场景共享 R、X；场景中心化精确消去各场景截距，不能因前处理已有 demean 而删除。
- RX75 是分别归一化后的 0.75 R + 0.25 X 共享路径组合；R/X 不是比值。
- RNJ 是带双锚分组、中位数更新、裁剪和根停止的有根变体；默认容差为根深度中位数的
  0.16 倍。
- 单步 MILP 的最优证书只覆盖当步给定候选域；完整前向扩展不因此成为全局树搜索。
- 真值只进入 reporting.py 的评分和审计字段，不应流入候选生成、验证选择或优化器。

## 2026-09-08 整理内容

- pipeline.run 从约 267 行降到约 162 行，拆为数据准备、RNJ 选择、MILP 完成三个
  可单独检查的阶段；两份只读阶段记录代替平行局部变量。
- RNJ 汇总行和 bootstrap 候选行移入 reporting.py，算法编排不再内嵌评价字段拼装。
- 参数在创建输出目录前集中校验；此前 bootstrap_replicates=0 会在深层除零，
  block_length=0 会在重采样中失败。
- coefficient_bound 现在写入 metrics.json，可从结果文件还原 MILP 的 R/X 上界。
- 保持每完成一个案例写 RNJ CSV、每完成一个 MILP 变体写 MILP CSV 的原有故障恢复行为。
- 没有改动 R/X QP、RNJ 合并、bootstrap 抽样、收缩公式或 MILP 数值模型。

## 验证

- 独立核心测试：159 passed。
- 父项目完整兼容测试：123 passed。
- 两个最小真实流程：reference 场景、paper15、4 个观测点、1 次 bootstrap；分别覆盖
  仅 RNJ 选择和 1 秒预算的完整 RNJ+wzzT 写出。

测试通过证明本次职责整理与接口检查未破坏现有回归样例，不构成算法正确性或统计泛化证明。
