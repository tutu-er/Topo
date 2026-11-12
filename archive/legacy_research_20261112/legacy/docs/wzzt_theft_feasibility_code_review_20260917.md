# 偷电识别两模块方案：代码核对、可行性评价与相关工作

日期：2026-09-17。本文核对 rnj_wzzt_core 核心代码与[两模块设计](wzzt_credibility_joint_theft_model_20260917.md)、[隐藏节点可辨识性](wzzt_hidden_node_theft_identifiability_20260917.md)、[漏计负荷一般笔记](wzzt_unmetered_load_identifiability_20260917.md)三份设计文档的引用一致性，评价“模块 1 统一可信度评价 + 模块 2 wzzᵀ 偷电联合优化”的可行性，并补充相关工作定位。

状态：代码核对完成（逐行比对）；可行性评价为推导与工程判断，尚无实现或实验；文献为本次检索定位，未完成全面查新。

## 1. 核心代码核对结果

逐行核对结论：三份设计文档对代码的事实性引用全部属实；下表给出精确位置，个别行号范围作了修正。

| 设计文档的声明 | 核对结果（本次实际打开阅读） |
|---|---|
| lin_distflow 丢弃 injection_terminals，不能直接表达矩形响应 | 属实。[lin_distflow.py](/rnj_wzzt_core/rnj_wzzt/models/lin_distflow.py "citation") 第 33 行 `del injection_terminals`；第 50–56 行 R/X 逐对由公共根路径阻抗求和构造，即 wzzᵀ 的闭式等价；第 57–59 行平方电压模式乘 2。 |
| 场景验证禁止 hidden_internal 带负荷 | 属实。[validate_scenario.py](/rnj_wzzt_core/rnj_wzzt/scenario/validate_scenario.py "citation") 第 34–43 行：非根负荷必须是叶节点，hidden_internal 带负荷或被观测均报错。 |
| 默认逐日去均值，丢失恒定分量 | 属实。[preprocessing.py](/rnj_wzzt_core/rnj_wzzt/estimation/preprocessing.py "citation") 第 8 行 RECIPE 为 daily_demean，第 32–46 行实现逐段去均值。 |
| 回归再按场景中心化并消除自由截距 | 属实。[multiscenario.py](/rnj_wzzt_core/rnj_wzzt/estimation/multiscenario.py "citation") 第 156–157 行对设计与目标逐场景去均值；第 182–185 行另行恢复每场景截距。文档原引 128–184 为整个函数范围，精确作用行为 156–157。 |
| MILP 原子为支持集指标外积，特征为下游功率和 | 属实。[laminar_l1_milp.py](/rnj_wzzt_core/rnj_wzzt/estimation/laminar_l1_milp.py "citation") 第 525–543 行 `_fixed_atom_features`：对每个支持集取 `(p @ 1_S) * 1_S`，即 w z zᵀ 原子；第 480–522 行 `build_matrices_from_atoms` 显式按原子累加 R/X。 |
| 拟合含场景×输出的自由截距 | 属实。[laminar_l1_milp.py](/rnj_wzzt_core/rnj_wzzt/estimation/laminar_l1_milp.py "citation") 第 643–645 行每个观测方程含截距变量；第 694–698 行截距变量上下界为 ±∞。 |
| 仿真分开保存真实/量测 P/Q/V，无漏报机制、无首端总表 | 属实。[simulation.py](/rnj_wzzt_core/rnj_wzzt/scenario/simulation.py "citation") 第 200–259 行：P_true/Q_true/V_true 与带噪量测分开保存，根电压表有 exact/noisy 两档；无任何漏计负荷生成机制，返回量中无馈线首端 P/Q 总表。 |
| MILP 求解器为 SciPy HiGHS，要求原始/对偶最优性证书 | 属实。第 581–618 行 `_run_milp` 调 `scipy.optimize.milp`，设 mip_rel_gap 与 mip_abs_gap；第 199–252 行有最优性证书与对偶界检查。 |
| 扩展步每次精确选一个与 family 相容的新原子并全量重估 | 属实。第 778–1024 行 `_solve_extension_prepared` / `solve_best_laminar_extension_l1` 实现有限候选域上的单原子选择与 fully-corrective 重估。 |

补充核对（设计文档未引用、与落地直接相关）：

- [bootstrap.py](/rnj_wzzt_core/rnj_wzzt/graph/bootstrap.py "citation") 已有 circular moving-block bootstrap（块长 4）与不相交块选择设施，模块 1 的“完整流程重跑校准”可复用该重采样机制，但现有 bootstrap 只服务于 clade 支持度，不是异常统计量的零分布校准。
- [simulation.py](/rnj_wzzt_core/rnj_wzzt/scenario/simulation.py "citation") 的噪声模型为相对噪声（`pq_noise_rel * |p_true|`、`v_noise_rel * |V_true|`），模块 1 的 Σ_t 可据此构造异方差权重，但 P/Q 噪声经 R/X 传播进残差的问题（设计文档 §2.2 的 EIV 问题）在现有代码中未被任何模块处理。
- 扩展步 MILP 在四个算例上单次耗时 3.6–143 秒（上限 1800 秒），见 outputs/mainline/。这是模块 2 加入 T×|H| 个二元变量前的基线耗时。

## 2. 可行性评价

总判断：**两模块架构在当前代码地基上可行，且首选路线（总分表差值固定幅值 + 整数位置 + R/X 重估）能整体停留在 MILP 内，与现有求解设施同构；主要风险不在优化可解性，而在可辨识边界与校准纪律。**

### 2.1 模块 2（偷电联合优化）：分三档可行性

| 版本 | 优化类别 | 与现有代码的关系 | 可行性判断 |
|---|---|---|---|
| A. 冻结 R/X、逐位置非负幅值扫描（闭式 Δ） | 每位置一维凸问题 | 纯新增诊断层，不动 MILP | 高。可作基线与初始化，但已被设计文档正确定位为“不是最终模型”。 |
| B. 固定幅值 â_t + 整数位置 s_ht + R/X 重估 | MILP（v^r=r·b 四约束精确线性化） | 与 `_solve_extension_prepared` 同构：新增 T×|H| 二元变量与每时刻 ≤1 单源约束；`_VariableBuilder`/`_ConstraintBuilder` 直接支持 | 高，是首选。注意计算量：二元变量数 = T×|H|，96 点 × 十数个候选位置即上千二元变量，超过现有单原子扩展的规模，需实测 HiGHS 表现；可先按时刻分块求解再合并。 |
| C. 自由幅值 a_ht + 整数位置 + R/X 重估 | 混合整数双线性（非凸） | 现有 L1-MILP 框架不能直接容纳 r×a 项 | 中。只能作对照：交替优化（两凸/MILP 子问题）生成候选，或 McCormick + 空间分支认证界；须按设计文档 §5 报告 [LB,UB] 区间而非单点。 |

关键边界（沿用设计文档推导，本次复核其逻辑成立）：

- “只增负荷”约束必须施加在**原始**有功上；现有 daily_demean + 场景截距会系统性吸收恒定偷电（§1 已核对代码机制），所以模块 2 的输入必须保留绝对量通道，或显式声明只辨识时变分量。
- 缺少独立总功率锚点时存在尺度混淆反例（r₀ 与 γ(r₀+κx₀) 等价），设计文档 §6 的推导本次复核成立。这决定了版本 B 的总分表输入不是工程可选项，而是打破该等价类的信息条件。
- 候选响应列成比例时位置不可辨（无观测侧支路、共同干线），模块 2 只能输出区域级位置集合；这不是求解失败，必须由模块 1 如实报告。

### 2.2 模块 1（统一可信度评价）：可行，但工作量集中在校准而非损失本身

- 数据损失（高斯加权 L2 或 L1 + 尺度归一化）可直接构造；现有代码的残差、截距、R² 与验证集一标准误差选择（pipeline 层）已具备大部分计算件。
- 难点是设计文档 §2.4 的三条纪律在当前代码中没有现成支撑：
  1. **零分布校准**：需要在无偷电数据上重跑“R/X 估计 → RNJ → MILP → 位置扫描”全流程取 Δ 统计量分布。bootstrap.py 提供重采样件，但“全流程重跑”的编排层不存在，需新建。
  2. **H0/H1 公平重估**：两者必须在同一观测、同一误差模型、同一参数域各自重估 R/X；现有管线没有“固定树、重估边权”的独立入口，`solve_fixed_support_l1`（第 743 行）恰好就是该入口的天然基础——给 family 固定、只优化系数，H0/H1 都可以用它实现。
  3. **总分表残差的联合评分**：â_t 与电压残差共享分表 P/Q 误差，模块 1 须处理该相关性（设计文档 §0.3），这需要新建协方差构造，现无可复用件。
- “类似 EM”的概率化版本（设计文档 §2.3 混合模型）可行但非必需；确定性损失 + 全流程校准先于 EM 落地是正确次序。

### 2.3 两模块耦合处的三个工程缺口

1. **偷电生成机制缺失**：simulation.py 需新增“在指定内部/末端位置注入未计量正负荷、AC 潮流按实际负荷求解、量测只暴露报告值”的通道，且不能触发 validate_scenario 的 hidden-load 禁令——应新建独立 scenario 生成器而非放宽验证。
2. **绝对量通道缺失**：daily_demean 与自由截距会吸收恒定偷电；需要 raw 预处理路径 + 受约束截距（已有 recipe "raw"，但 multiscenario 的场景中心化仍会吸收，需改）。
3. **首端总表缺失**：版本 B 依赖的 m_t^P 需要在 simulation.py 增加带噪首端 P/Q 表（线损由此进入模型），这是打破尺度等价类的关键新增观测。

## 3. 相关工作调研与定位

本次检索补充 + 设计文档已有引用，按与本文方案的接近程度排序：

1. **Pengwah, Razzaghi, Andrew (2023), TPWRD**：先估计电压灵敏度、再估计真实用电并与报告值比较的无模型 NTL 检测——与本方案“R/X 估计 + 漏计量恢复”最同构，且其馈线算例 pengwah18 已是本项目主测试算例之一（README 主结果表）。直接的后续对照对象。[Monash 大学论文记录](https://research.monash.edu/en/publications/model-less-non-technical-loss-detection-using-smart-meter-data/ "citation")
2. **Gao, Foggo, Yu (2019), TII**：用电量—电压物理回归做异常排序，不要求完整网络参数。确立“电压+功率检测偷电”本身非新贡献。[作者全文](https://intra.ece.ucr.edu/~nyu/papers/2019-Electricity-Theft "citation")
3. **Bhela, Kekatos, Veeramachaneni (2019), arXiv:1806.08834**：受控注入下未计量负荷的可辨识性分析（图匹配/雅可比满秩，局部可辨识）。与本方案的可辨识边界推导互补，但其量测配置含隐藏节点附近观测，不能直接移植。[作者预印本](https://arxiv.org/abs/1806.08834v2 "citation")
4. **Kekatos 等拓扑学习综述（Learning Distribution Grid Topologies: A Tutorial, NSF PAR 10435069）**：确认已有将拓扑检测建模为 MILP/MIQP 的工作（线电流失配 MILP、树结构 MIQP），但均假设负荷伪量测已知，不处理未计量注入。本方案“拓扑候选与偷电解释竞争”不在其范围。[NSF 全文](https://par.nsf.gov/servlets/purl/10435069 "citation")
5. **Messinis 等式 NTL 系统综述（Energies 2020, 13(18):4727）**：潮流法（中央观察表 vs 分表和）与状态估计/坏数据法是网络型 NTL 检测两大路线；证实“总分表差值锚定 + 技术线损扣除”是工业标准做法，也确认其已知难点正是线损精确计算——与本方案 §0.4 的线损固定条件一致。[MDPI Energies 综述](https://www.mdpi.com/1996-1073/13/18/4727 "citation")
6. **LinDistFlow 参数优化（Taheri, Gupta, Molzahn 2024, arXiv:2404.05125）**：用灵敏度优化 LinDistFlow 系数/偏置逼近非线性潮流，与本项目 R/X 回归同族；其“拓扑变化下重估参数”的实验设计可借鉴到模块 2 的 R/X 重估验证。[arXiv 论文](https://arxiv.org/html/2404.05125v2 "citation")

**贡献边界判断**：逐条看，“用电压灵敏度检测 NTL”“用总分表差锚定总量”“拓扑 MILP”各自都有先例；尚未在文献中见到的是四者的交集——**wzzᵀ 树原子结构下，拓扑/边权/单源未计量负荷的联合竞争解释 + 显式可辨识等价类（比例列、尺度混淆、多源等价）+ 经全流程校准的相容集合输出（允许拒绝判断）**。这是可投稿的新增空间，但 §2 列出的等价类证明与校准实验是必要条件，缺一不可。

## 4. 建议落地次序（代码级）

1. simulation.py 新增偷电注入通道 + 首端 P/Q 总表（带噪），独立 scenario 生成器，不动 validate_scenario。
2. 模块 1 第一版：raw 预处理 + 受约束截距的数据损失；用 solve_fixed_support_l1 实现固定树 H0/H1 重估入口。
3. 模块 2 版本 B：仿 `_solve_extension_prepared` 新建 theft MILP（v^r=r·b 精确线性化、Σ_h s_ht≤1、R/X 自由），先单场景 96 点实测 HiGHS 耗时。
4. 零分布校准编排层：无偷电数据全流程重跑 N 次取 Δ 分布与告警阈值；再跑六组最小实验（正常、内部单源、末端单源、位置切换、比例歧义、参数偏差无偷电）。
5. 版本 C（自由幅值）作为对照：交替优化生成候选 + McCormick 界认证，报告区间而非点值。

每步的验收标准沿用设计文档 §2.4 输出清单：数据损失、H0/H1 经校准改善、相容树/位置集合、参数与电量区间、敏感性、求解完整性标志。
