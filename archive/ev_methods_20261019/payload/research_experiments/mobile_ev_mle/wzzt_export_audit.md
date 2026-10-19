# WZZT 实际导出契约及额外信息审计

核查的是当前冻结的三末端 `propose → build_passive_candidates → fit_source_aware_wzzt` 路径，以及 `adaptive_history → fit_candidate` 的参数初值路径。代码哈希及六个 `pilot_v3` 历史的实际接口记录保存在 `wzzt_export_audit.json`。只用历史 P/Q/V 和根电压构造提案；真候选在提案完成之后才用于离线覆盖核对。

**当前实际状态：`uses_structure=True`、`uses_impedance_initial=False`、`estimates_source=False`。** WZZT/RNJ 提供可修正结构和邻域种子；当前冻结路径没有把 WZZT 的线路权重或灵敏度矩阵接入 AC 初值。默认源输入为空，不构造 WZZT 源变量。这不影响后续 AC 模型通过总表和同一物理源接口联合估计逐时源状态。

## 1. 源码证据

| 事实 | 代码位置及实际行为 |
|---|---|
| proposal 只读取历史末端 P/Q/V 与已知根电压 | `mobile_ev_mle/candidates.py:19–25`，构造 `Y=Vroot²−Vterminal²`。未把 `master_p/master_q` 或源 nominal 传入 bridge。 |
| 当前 WZZT 不估计源 | `mobile_ev_probing/passive_bridge_v9.py:161–175` 默认 `source_inputs=None`；229–234 调 adapter 时传入此值。`rnj_wzzt_core/rnj_wzzt/estimation/laminar_l1_source.py` 的 source preparation 对 None/全零幅值回退普通模型。 |
| 结构可以替换 | bridge:229–239 使用空 `certified_supports`，接收每个 accepted path 的支撑；245–265 根据未认证排名生成删/换/终端交换邻域，`replaceable=True`、`ac_evidence=None`。 |
| 三末端最终 bank 不靠真拓扑补全 | `candidates.py:34–39` 在提案结构后声明式加入 star 与全部三个 cherry。完整性仅针对该简化 star/cherry 类；扩网并不自动继承此覆盖。 |
| 高层丢弃矩阵与线路权重 | bridge:235 只保存 `result.summary()`，238–239 只把 `fit.support_indices` 放入候选。adapter `SourceFitResult.summary():134–150` 没有导出 `r_matrix/x_matrix/r_values/x_values/h/g_r/g_x`。 |
| 实际 AC 初值并非 WZZT 权重 | `adaptive_history.py:231–235,252` 初次调用 `fit_candidate(initial=None)`；`likelihood.py:355–364` 用校准末端 P/Q 初始化状态，源用 `master−sum terminal` 初始化，逐边 R=.012 pu、X=.006 pu。后续暖启动和平坦方向来自 AC 历史拟合。 |

`passive_metadata=proposal` 仅被历史上下文保存为审计信息，没有参与 R/X 初值赋值。初次源估计的 `master−sum terminal` 尚含线路损耗，只是初值；实际完整 AC 联合拟合会解释损耗，不能把这个差直接称作已估计源真值。

## 2. 内部可获得项与实际导出项

| 信息 | adapter 内部对象 | bridge/propose 实际返回 | 可合理用途 |
|---|---|---|---|
| 结构支撑、路径、来源 | `SourceFitSolution.support_indices`、`SourceFitResult.path` | 候选支撑；origin、rank；最终 WZZT 支撑、accepted path length；可修正 bank | 保留结构分歧、生成局部候选、分配计算预算。 |
| 原子 R/X 权重 | `r_values/x_values` | **未导出** | 以后可做额外 AC 初值；须按当前 squared-voltage 系数单位核对映射与支撑对应，先做完整 AC 验证。 |
| R/X 灵敏度矩阵 | `r_matrix/x_matrix` | **未导出** | 以后可作近似响应、结构差异诊断或初始化；不能替代完整 AC 白化导数。 |
| RNJ root depth/shared paths | RNJ 内部局部 `r` 和 `sensitivity_geometry(...)` | **矩阵和 geometry 未导出**；只导出 r²、rank、condition、solver scalar diagnostics | 可增加几何分歧和激励缺口诊断，但其数据驱动不确定性尚未校准。 |
| 源路径及响应 | `h/g_r/g_x` | **未导出**；当前默认内部均为零 | 只有提供非零、外部声明的源 nominal 才有活动源路径。当前不能声称从 WZZT 得到源估计。 |
| 残差 | 内部按 `P Rᵀ+Q Xᵀ` 构造；未存成 dataclass 逐点字段 | WZZT train/adjusted MAE；候选 contrast MAE/RMSE；RNJ solver squared-residual sum 等**标量**。逐点残差未导出 | 可检查近似失配/异常模式；不是额外 AC 似然。 |
| 权重或概率 | unweighted L1 系数/残差目标，RNJ constrained least squares | 没有 P/Q/V Gaussian covariance weights、候选概率或 bootstrap 覆盖权重 | 不能把排名、r² 或 MAE 转成拓扑概率。 |
| 求解界 | 每次 diagnostics 的 objective、dual bound、mip gap/status，以及物理回代检查 | `solver_attempts` 保留这些实际字段；固定普通 LP 记录的 dual/mip 字段可以为 None | 界的范围是有界固定族或单原子扩展，整个 forward path 仍是贪心；不是全拓扑/完整 AC MLE 的全局界。 |
| 工程边界 | R/X atom/path bounds、外部 certified lower bounds | summary 留 bound active、R/X upper及可选 path/lower bounds | 只在外部依据成立的工程域内解释；当前 nominal 参数范围不构成统计置信集。 |

当前 RNJ 与 contrast 排名采用 (P+0.5Q\) 和固定 `kappa=.5`。正式生成器逐边 R/X 独立变化，并未满足所有线路统一 X/R=.5；这里只是提案近似，其误差通过可修正候选及完整 AC 拟合处理。contrast 逐时去除末端平均响应，允许任意逐时共同输出变化来排名；它并不是用户要求的固定物理接口源模型。该共同输出补偿不能成为主设计指标或物理证据。

## 3. 六历史实际 schema 核验

本次 probe 的六次 WZZT 均 `source_inputs_is_None=True`、source error=0，内部 `h/g_r/g_x` 全零。六次 accepted path length=3、stop=`max_atoms`。重新计算 `Y−P Rᵀ−Q Xᵀ` 的 MAE 与内部 train MAE 相符；此逐点残差是审计重构，原 bridge 仍未返回它。

| 历史 | raw RNJ（tolerance=0） | centered RNJ（tolerance=0） | 最终 WZZT 简化 clade |
|---|---|---|---|
| 710009 | {0,1} | {0,2} | {0,2} |
| 710010 | {0,1} | {0,2} | {0,2} |
| 710011 | {0,1} | {0,1} | {0,1} |
| 710012 | {0,1} | {0,1} | {0,2} |
| 710013 | {0,1} | {0,1} | {0,2} |
| 710014 | {1,2} | {0,2} | {1,2} |

原始 proposal 含邻域生成的 star 和全部 cherry，因此“所有候选的结构分歧”为全部三种 clade，不等同于“RNJ/WZZT 原始提案都产生了全部三种”。JSON 另存按 origin 区分的原始结构集合，避免混淆。六次最终三末端 bank 均保留声明简化类中的真候选；这不是 WZZT 在更大网络中的覆盖证书。

实际 passive schema 9 顶层有候选、diagnostics、generated/retained/base/discarded counts、config、seconds、coverage、`selection_uses_truth=False`、`ranking_is_ac_evidence=False` 等。candidate 有 supports/rank/origins、contrast MAE/RMSE、rank-fit status/optimality、`ac_evidence=None`、空 certified supports、`replaceable=True`。详见保存 JSON 的原始 keys 和 full proposal。

## 4. 本轮文档中需要校正的具体位置

以下文字把拟议的初值用途写成了当前已实施路径，应由 root 原位修订；本审计没有修改这些文件。

| 位置 | 当前表述 | 建议准确表述 |
|---|---|---|
| `docs/mobile_ev_probing_report_v2.tex:1052` | “WZZT/RNJ 仅用于候选、邻域和初值” | 当前用于可修正候选与邻域；R/X 权重初值尚未接通，AC 初值由 likelihood 默认值及已拟合历史状态提供。 |
| `research_experiments/mobile_ev_mle/adaptive_milp_protocol.md:18` | “WZZT 用于候选、邻域和初值” | 改成上述已实施状态，并把权重初值用途列为未验证改进。 |
| `research_experiments/mobile_ev_mle/adaptive_statistics_math.md:198` | “原历史导出的 WZZT/RNJ 用于候选/初值” | 当前主路径只有候选/邻域；作为初值是允许的后续数值改进，不是本轮已实现事实。 |

`symmetric_ac_local_math.md:103` 的“初值**可**来自历史数据挑出的有限 atom”是条件性方案，未直接声称当前已经接通；可再注明本轮冻结主路径未用 WZZT atom 权重。其他 docs 中 `initial_supports` 是不同主线的结构初始化，不能据此断言本研究 AC 阻抗使用了 WZZT 权重。

## 5. 后续怎样利用额外信息而不重复计似然

1. 导出带单位、支撑、源码哈希和失败状态的矩阵/权重作为**额外数值初值**，保留默认初值、多初值与历史平坦方向展开。WZZT 在同一历史上的权重不能被当成独立测量先验，也不能把其残差再乘入同一历史的 AC likelihood。
2. 用 origin-specific 结构分歧安排局部候选和计算预算；旧 WZZT 支撑始终可删除/交换/替换。不能删除难解释的竞争者以提高 MILP 分数。更大网络的候选漏真率要在独立实验中评价。
3. 如采用 WZZT 灵敏度来快速估计 EV 响应，先验证系数单位与 AC 一阶关系，再以完整 AC 白化导数/联合拟合回代。共同源的位置必须维持当前声明物理接口；不能让近似器的任意源路径替代这个接口。
4. 如果以后输入 AC 历史拟合得到的源 nominal，那只是同历史驱动的提案输入。输入估计误差、与末端读数的相关性、非负 P 限制以及原外允许误差域的物理含义都要单独说明。输出盒不能自动成为统计覆盖或新的独立信息。

当前 AC 历史 profile 矩阵、状态补偿方向、活动边界和多展开点已经提供更直接的参数可补偿性信息。WZZT 的补充价值主要是结构搜索和可解释的初始化诊断。接通权重初值是否降低耗时、是否改变最佳已发现局部似然，以及候选压缩是否漏真，仍是待验证方案；本轮没有实施扩网或追加权重初值。
