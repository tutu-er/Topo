# 末端智能表、隐藏节点配电网拓扑辨识：近期进展与新研究路线

> 检索与核对日期：2026-09-01（Asia/Shanghai）  
> 研究边界：径向低压配电网；末端或部分节点智能表；电压幅值与有功/无功（或电流）时间序列；隐藏零注入分支点；拓扑、相别与线路参数联合辨识。  
> 配套证明：current_mainline_complete_proofs.tex/.pdf。本文不重复证明全文，而以已经校正的共享路径、RNJ、四点条件、伪边界聚合和 EIV 结论为数学起点。

## 1. 结论先行

截至 2026-09-01，最需要正视的变化不是“又出现了一个分类器”，而是以下五条主线正在合流：

1. **端用户 \(P/Q/V\) 下的隐藏树全重建已经出现非常接近本项目的直接竞争路线。** 2026 年的 full-branch iterative grouping（FBIG）工作同样先估计带物理约束的加权约化拉普拉斯逆，再由末端量测识别父子/兄弟关系，最后用非线性 DistFlow 修正拓扑和阻抗。因此，“末端 \(P/Q/V\)+约束 \(R/X\)+分组恢复+非线性修正”本身已不足以构成独立的新颖性叙事。
2. **三相、相别、接户线和真实智能表数据从附加项变成了主问题。** 2024--2025 年的工作已把 Steiner 隐藏树、三相耦合、未知相角近似、相别识别和真实五分钟智能表数据放到统一建模中。
3. **数据缺陷被显式建模。** 缺测、低覆盖率、量化、带宽限制、异步与 PV 诱导的相关性偏移不再只是实验噪声，而是进入 EM、误差界、时间段选择和在线更新算法。
4. **工程场景从“从零发现整棵树”分化为“校正已有但过时的拓扑”。** 2025--2026 年的真实 DSO 案例更强调可解释的相别/馈线核验、状态估计残差定位和局部修复。
5. **当前领域仍明显缺少可靠的结构不确定性输出。** 大多数结果仍输出单棵树或分类准确率；“候选集中包含真树的概率”“每个 clade 的同时置信结论”“何时应该拒绝给出唯一拓扑”仍是可形成贡献的空缺。

据此，本项目最有价值的下一步不是继续堆叠重构器，而是把现有 RNJ/NJ/RG/GTLS/AC 主线改造成：

> **同时置信距离 → 不排除真树的 RNJ/四元组候选完备化 → 协方差最优且交叉拟合的伪边界聚合 → 留出 AC 预测排序 → 可拒判的拓扑置信集。**

这一方向既继承现有代码和证明，又能避开 2026 年 FBIG 工作最直接的重叠。

## 2. 现有主线的准确定位

当前实现可抽象为

\[
\Delta \widetilde v_t
=-R\,\Delta \widetilde p_t-X\,\Delta \widetilde q_t+\eta_t,
\qquad
R_{ij}=\sum_{e\in P(r,i)\cap P(r,j)}r_e,
\quad
X_{ij}=\sum_{e\in P(r,i)\cap P(r,j)}x_e .
\]

在有序、对称、非负与 EIV/GTLS 约束下估计 \(R,X\)，再直接构造共享路径分数

\[
S_{ij}=\alpha R_{ij}+(1-\alpha)\beta X_{ij}
\]

和加性距离

\[
d_{ij}=S_{ii}+S_{jj}-2S_{ij}.
\]

之后使用已知根 RNJ，并将 NJ、RG、四元组和一层 clade 聚合作为候选生成器；每个固定候选再进行结构化 EIV 参数拟合和留出 AC 预测似然排序。隐藏节点不按任意编号评价，而按可识别的 rooted terminal clade 评价。

这条主线相对于多数近期工作的优势是：

- 明确区分 terminal-equivalent tree、可识别的隐藏 Steiner 树和给定候选物理树；
- 候选生成与物理排序分离，不让某一个启发式分支直接覆盖最终结论；
- 已处理公共根电压模态、\(P/Q\) 测量误差、固定树 EIV 拟合和 AC 模型失配；
- 以 rooted clade 而不是隐藏节点标签评价，避免不可识别标签造成虚假的错误；
- 已有 RNJ 安全阈值、四点间隙和伪边界误差传播的完整证明基础。

当前最关键的不足是：

- RNJ 和 clade 接受规则仍主要由固定阈值、bootstrap persistence 或经验 margin 决定，尚无统一的**同时置信事件**；
- 候选集很强，但没有“真树为何应在候选集中”的覆盖保证；
- 伪边界电压仍未完全利用候选子节点误差之间的协方差，且结构选择与伪量测生成可能共享噪声；
- 当前 generalized posterior 是有用的排序分数，但不是自动校准的贝叶斯后验或频率学置信集；
- 三相耦合、未知相别、缺测、量化和异步尚未统一进入主模型；
- 若已有 GIS/历史拓扑，当前全重建模式没有充分利用“仅局部错误”的强先验。

## 3. 2024--2026 年直接相关进展

### 3.1 与当前主线高度重叠的工作

#### (A) 2026：末端 \(P/Q/V\) 的拓扑与阻抗联合估计

Zhang 等在 *Sustainable Energy, Grids and Networks* 发表的工作仅使用端用户电压幅值和有功/无功数据：先估计带物理约束的加权约化拉普拉斯逆，再用 full-branch iterative grouping 同时判断父子和兄弟关系，最后以非线性 DistFlow 迭代修正拓扑和阻抗。论文页面列出的卷期时间为 2026 年 9 月。[原始出版页](https://www.sciencedirect.com/science/article/pii/S2352467726002675)

**与本项目的关系。** 这是当前最直接的竞争基线。二者共享“端用户 \(P/Q/V\)、隐藏分支点、受约束灵敏度/阻抗矩阵、分组重构、非线性物理修正”骨架。本项目仍有差异：EIV/GTLS、多个结构生成器的候选并集、known-root RNJ、clade 规范化、留出 AC predictive NLL 和显式候选不确定性。但论文叙事必须把贡献放到这些差异及其理论保证上。

**必须采取的动作。** 在继续宣称主线改进前，先实现或尽可能忠实复现 FBIG，至少比较：同一 \(R/X\) 输入时的重构误差；同一原始 \(P/Q/V\) 时的端到端误差；候选 oracle recall；运行时间；非线性修正对拓扑和参数的分别贡献。

#### (B) 2024：三相 Steiner 隐藏树、灵敏度约束与 RG/backtracking

Fang 等从仅有电压幅值、电流幅值、功率因数及 lead/lag 标志的智能表数据，联合估计三相电压灵敏度和变压器电压，再使用增强 recursive grouping、定向 backtracking 和候选选择恢复带接户线的 Steiner 树；其模型显式考虑未知电压相角近似和中性线引起的相间耦合。[原始出版页](https://www.sciencedirect.com/science/article/pii/S0142061524001704)；[作者机构全文](https://minerva-access.unimelb.edu.au/server/api/core/bitstreams/26bfcb36-fcc3-4676-9b6d-9099b38c3e13/content)

**与本项目的关系。** 它说明“三相化”不能只把单相 \(R/X\) 复制三次。相间耦合、相别映射、变压器电压和接户线需要进入同一灵敏度估计；同时它也是 RG/backtracking 的强基线。

#### (C) 2024/2025：真实三相智能表数据构建完整 LV 模型

Karunarathne 等使用澳大利亚真实低压网络的 60 天、5 分钟分辨率 \(V/I\) 数据，通过多元线性回归、聚类和 Spearman 相关识别拓扑、阻抗和相组；真实案例有 14 个用户，论文报告由所得模型进行潮流计算时电压误差约为 ±1 V。[IEEE 原始页面](https://doi.org/10.1109/TPWRS.2024.3474672)

**与本项目的关系。** 这不是大规模隐藏树理论的终点，但它提高了实验可信度门槛：仅在合成 AC 数据上报告 clade F1 已不足以说明工程可用性；至少需要真实负荷曲线驱动、缺测/异步回放和最终潮流预测误差。

### 3.2 数据不完美与低可观测性

#### (D) 2025：量化量测下的拓扑学习误差界

Talkington 等把量化器直接写入非线性测量模型，而不是简单等价为加性高斯噪声；其结果指出，学习误差与量化 bin 宽度成比例，并在每个节点样本数对节点数呈对数增长时，对网络规模呈次线性增长。[arXiv 原文](https://arxiv.org/abs/2508.05620)；该工作也列在 [PSERC 2025 项目报告](https://documents.pserc.wisc.edu/documents/publications/reports/2025_reports/T_67_Executive_Summary__1_.pdf) 中。

**与本项目的关系。** 当前智能表误差模型应增加“量化区间似然”而非只增加连续噪声；同时可把量化 bin 宽度显式加入距离半径和 RNJ 可恢复条件。

#### (E) 2025：缺测下的线路参数 EM

Kapoor 等仅使用节点功率和电压幅值，以 EM 处理随机缺测和整节点缺测，并同时估计未观测节点的电压；测试包括 IEEE 37 节点、噪声以及真实用户负荷驱动的仿真。[原始出版页](https://www.sciencedirect.com/science/article/pii/S0142061525001851)

**与本项目的关系。** EM 可作为 \(R/X\) 估计层的缺测适配器，但不能单独解决 terminal-only 隐藏分支的结构不确定性。更合理的组合是：EM/状态空间模型补全充分统计量，随后仍由置信 RNJ/四元组生成结构候选。

#### (F) 2024--2025：一般图的部分可观测结构学习

Peng 等研究只观测部分节点时从平滑信号恢复图，分别使用列稀疏和低秩约束，并给出一阶算法的线性收敛保证。[arXiv 原文](https://arxiv.org/abs/2410.05707)

**与本项目的关系。** 该工作不是配电隐藏树识别的替代品；但“隐藏节点影响 = 稀疏局部修正或低秩边缘化效应”可以作为缺测灵敏度矩阵的正则项或初始化器。

### 3.3 快速在线、时间频率和黑箱检测

#### (G) 2025：hierarchical subset sum（HSSP）

Xu 和 Chen 依据各层节点功率守恒，将多层拓扑恢复转化为层级 subset-sum，并以多时刻投票提高抗噪性；论文和代码均公开。[arXiv 原文](https://arxiv.org/abs/2507.16924)；[代码](https://github.com/chennnnnyize/HSSP)

**适用边界非常重要。** 论文问题设定假定所有底层节点都有同步功率数据，并使用父节点总功率与子节点功率和的关系。它不能直接替代“只有末端表、内部父节点完全不可观”的当前任务。若未来能获得变压器/馈线/台区总表，它可作为毫秒到秒级的变更触发器或候选约束。

#### (H) 2025：小波能量特征与非合成网络验证

García 等仅使用能量智能表数据，以小波提取多分辨率时频特征，识别馈线和相别关系；在 11 个非合成网络上评估可观测率、数据长度、误差和 RES 渗透，论文报告多数情形准确率超过 95%，且 80% 以上覆盖、30 天小时数据时可达到 98%。[原始出版页](https://www.sciencedirect.com/science/article/pii/S0378779625001099)

**与本项目的关系。** 小波特征很适合做 feeder/phase 预分区和 PV/负荷时段选择，但其输出粒度与隐藏 Steiner 树重建不同，不能用一个 accuracy 数字直接和 rooted-clade F1 比较。

#### (I) 2025：wavelet-GAT 拓扑检测

Zhang 等把节点图转换为 line graph，再用电压幅值的小波高低频分量驱动 GAT，在 IEEE 33/118/367 节点上报告 99.01%/96.57%/94.76% 的检测准确率。[原始出版页](https://www.sciencedirect.com/science/article/pii/S0378779625004390)

**与本项目的关系。** 这类方法通常针对给定母线/线路集合上的开关状态或有限拓扑类别检测，不等于未知隐藏节点、未知候选线路下的 candidate-free 树恢复。可将其作为快速状态分类基线，但不应混为同一可识别性问题。

### 3.4 从全重建走向校正、核验和真实 DSO 工作流

#### (J) 2025：低于 20% 智能表覆盖率的真实相别核验

Théate 等在比利时 DSO 的真实区域上，用电压时序相关和相别聚类核验拓扑；智能表覆盖率低于 20%，输出强调可解释性和对含混记录的识别，而不是黑箱单标签。[作者机构页面与全文](https://orbi.uliege.be/handle/2268/327542)

#### (K) 2026：PV 条件下仅用电压幅值校正已有拓扑

Liu 等不是从零恢复整棵树，而是利用已有但不完整的拓扑和历史运行记录，依次修正开关状态、用户--馈线关系和相别；其时间段选择策略通过选取负荷主导、PV 注入较弱的时段减轻 PV 造成的虚假相关，测试使用荷兰真实智能表数据。[原始出版页](https://www.sciencedirect.com/science/article/pii/S2352467726002031)

#### (L) 2026：用状态估计残差的空间模式定位拓扑错误

Castin 等在真实 DSO 网络上利用状态估计残差的地理/空间模式发现潜在拓扑错误和错误相别分配，目标是给运维人员指出“哪里可能错”，而不是盲目重建整个网络。[作者机构页面与全文](https://orbi.uliege.be/handle/2268/342287)

**共同启示。** 实际 DSO 更需要“保持可信区域、只解锁异常区域、给出证据与拒判”而不是每次从零输出一棵完全不同的树。这与当前 local/backbone RNJ、留出 AC 残差和 rooted clade 表示天然兼容。

## 4. 近期工作的能力边界对照

| 路线 | 主要量测 | 隐藏分支点 | 三相/相别 | 缺测/量化 | 输出 | 对当前工作的直接作用 |
|---|---|---:|---:|---:|---|---|
| 2026 FBIG + nonlinear refinement | 末端 \(P/Q/|V|\) | 是 | 论文主结果需按原文复核 | 未作为核心 | 单树+阻抗 | 必须复现的直接基线 |
| 2024 三相 sensitivity + RG/backtracking | \(|V|,|I|,\mathrm{PF}\) | 是，Steiner 树 | 是 | 非核心 | 候选树 | 三相扩展与 RG 强基线 |
| 2024/25 真实三相模型构建 | 真实 \(V/I\) | 通过 direct/semi-direct/indirect 节点 | 是 | 真实噪声 | 完整模型 | 提高真实验证标准 |
| 2025 量化误差界 | \(P,Q,V\) 的有限精度值 | 依模型 | 非核心 | 量化核心 | 误差界+拓扑/参数 | 为置信半径加入 bin 宽度 |
| 2025 EM 缺测参数估计 | \(P,Q,|V|\) | 不是结构核心 | 非核心 | 缺测核心 | 参数与未测状态 | 作为前端充分统计量估计器 |
| 2025 HSSP | 各层/各节点功率 | 否，父节点需有总量测 | 否 | 噪声 | 快速连接关系 | 有父级总表时作触发器/约束 |
| 2025 wavelet / wavelet-GAT | 电能或 \(|V|\) | 通常否 | 相别或已知线路状态 | 部分 | 分类/连接标签 | 预分区和快速基线 |
| 2025/26 DSO 核验与校正 | 部分 \(|V|\)+已有记录 | 不以全重建为目标 | 是 | 是 | 异常区域/修正 | 建立 prior-aware local repair 模式 |

## 5. 新思路一：同时置信 RNJ + 四元组候选完备化

### 5.1 为什么这是第一优先级

当前失败往往不是最终 AC 排序器分不清，而是真树从未进入有限候选集。固定阈值 \(\tau\) 或经验 bootstrap 稳定度能提高平均精度，却不能回答：

- 某个 clade 为什么足够确定？
- 一个不确定 clade 应该被删除，还是应保留多个完成方式？
- 候选集在什么事件下必然包含真树？

应把“选择一棵树”改成“先构造一个不排除真树的结构置信域”。

### 5.2 同时距离区间

通过按天/场景的 block bootstrap、multiplier bootstrap 或 EIV sandwich covariance，构造

\[
\Pr\!\left(
\forall i<j:\;|\widehat d_{ij}-d_{ij}|\le r_{ij}
\right)\ge 1-\delta.
\tag{1}
\]

一种实现是先估计标准差 \(s_{ij}\)，再从重抽样分布取得最大统计量分位数

\[
c_{1-\delta}
=Q_{1-\delta}\!\left[
\max_{i<j}
\frac{|\widehat d^{(b)}_{ij}-\widehat d_{ij}|}{s_{ij}}
\right],
\qquad r_{ij}=c_{1-\delta}s_{ij}.
\tag{2}
\]

这一步必须保留时间相关性；逐行 iid bootstrap 会低估慢变负荷和公共电压模态造成的不确定性。

### 5.3 用区间替代固定 RNJ 阈值

对 RNJ 的兄弟不变量，定义

\[
A_{ab;k\ell}
=(d_{ak}-d_{bk})-(d_{a\ell}-d_{b\ell}).
\tag{3}
\]

若 \(a,b\) 是兄弟，则对任意外部 \(k,\ell\)，有 \(A_{ab;k\ell}=0\)。由 (1)，

\[
|\widehat A_{ab;k\ell}-A_{ab;k\ell}|
\le
r_{ak}+r_{bk}+r_{a\ell}+r_{b\ell}
=:R_{ab;k\ell}.
\tag{4}
\]

因此可采用三值决策：

- 若所有区间 \([\widehat A-R,\widehat A+R]\) 都含 0，则关系**未被拒绝**；
- 若至少一个区间排除 0，则该兄弟关系**被拒绝**；
- 只有当估计的父边长度下界也为正时，才把它标为**确定 clade**。

在同时覆盖事件 (1) 上，真兄弟关系不会被错误拒绝。若某个假关系存在一个 \(k,\ell\) 满足

\[
|A_{ab;k\ell}|>2R_{ab;k\ell},
\tag{5}
\]

则对应区间必排除 0，假关系会被拒绝。这是异方差版的 RNJ separation condition；它比统一 \(\tau\) 更适合不同终端噪声差异很大的智能表。

### 5.4 四元组置信间隙

对四个终端 \(a,b,c,d\)，定义

\[
S_1=d_{ab}+d_{cd},\quad
S_2=d_{ac}+d_{bd},\quad
S_3=d_{ad}+d_{bc}.
\tag{6}
\]

若真实 split 为 \(ab\mid cd\)，则

\[
S_2=S_3=S_1+2w,
\tag{7}
\]

其中 \(w>0\) 是内部边长度。令每个和的区间半径为相应两个 \(r_{ij}\) 之和，则保守下界

\[
\underline G_{ab\mid cd}
=\min(\widehat S_2-R_2,\widehat S_3-R_3)
-(\widehat S_1+R_1)
\tag{8}
\]

若 \(\underline G_{ab\mid cd}>0\)，则该 split 在事件 (1) 上必然正确。对于统一误差 \(r_{ij}\le\varepsilon\)，(7) 的观测 gap 误差不超过 \(4\varepsilon\)，故 \(w>2\varepsilon\) 足以保证 split 符号不翻转。

### 5.5 候选完备化

将 rooted clade 视为终端子集。任意一棵有根树的 clade 家族必须是 laminar 的：任意两个 clade 要么互不相交，要么一个包含另一个。构造候选时：

1. 固定所有由 (8) 证明的 clade/split；
2. 删除所有被置信区间明确反驳的关系；
3. 对剩余含混关系枚举或 beam-search 若干兼容的 laminar completion；
4. 把现有 RNJ、NJ、RG、聚合/反聚合树作为高权重种子，而不是唯一来源；
5. 每个 completion 进入相同的 fixed-tree EIV + held-out AC 排序。

若树生成器枚举所有与“确定关系”和“非拒绝关系”相容的 laminar completion，则由 (1) 立即得到

\[
\Pr(T_\star\in\mathcal C)\ge 1-\delta,
\tag{9}
\]

其中 \(\mathcal C\) 是候选集。实际有限 beam 会破坏严格完备性，因此必须同时报告：无限制小规模枚举的 coverage、有限 beam 的 candidate oracle recall，以及 beam 宽度--运行时间曲线。

### 5.6 贡献边界

这一方向的潜在贡献不是“又一种 RNJ”，而是：

- 将 EIV 距离不确定性传播到 RNJ 和四元组结构判断；
- 从 point estimate 转为含真树概率可分析的 candidate confidence region；
- 把现有多重候选生成器组织成有理论边界的 completion 算法；
- 允许在证据不足时拒绝输出唯一拓扑。

## 6. 新思路二：协方差最优、交叉拟合的伪边界聚合

### 6.1 现有平均/中位数为何不足

设一个已识别 clade 的隐藏边界节点为 \(h\)，从其 \(k\) 个子终端分别反推边界电压，得到

\[
z_t=v_{h,t}\mathbf 1+e_t,
\qquad \operatorname{Cov}(e_t)=K.
\tag{10}
\]

公共根电压误差、同一 \(R/X\) 回归误差、共享线路损耗和同一 clade 的结构选择会使 \(e_t\) 强相关。此时简单平均的方差不是一般意义下的“单点方差除以 \(k\)”，而是

\[
\operatorname{Var}(a^\top z_t)=a^\top K a,
\qquad \mathbf 1^\top a=1.
\tag{11}
\]

### 6.2 GLS 最优聚合

若 \(K\succ0\)，所有线性无偏聚合器中方差最小者为

\[
a^\star
=\frac{K^{-1}\mathbf 1}{\mathbf 1^\top K^{-1}\mathbf 1},
\qquad
\operatorname{Var}((a^\star)^\top z_t)
=\frac{1}{\mathbf 1^\top K^{-1}\mathbf 1}.
\tag{12}
\]

证明由带约束二次型的拉格朗日条件直接得到。若 \(K\) 病态，使用 shrinkage covariance

\[
K_\gamma=(1-\gamma)\widehat K+\gamma\operatorname{diag}(\widehat K)
\tag{13}
\]

或 Moore--Penrose 逆，并通过外层验证选择 \(\gamma\)。

### 6.3 与 latent common mode 的最优收缩

令 \(\widehat v^{\rm GLS}_{h,t}\) 为 (12)，\(m_{h,t}\) 为当前 latent common-mode 估计。组合

\[
\widehat v^{(\lambda)}_{h,t}
=(1-\lambda)\widehat v^{\rm GLS}_{h,t}+\lambda m_{h,t}.
\tag{14}
\]

设 \(e_g=\widehat v^{\rm GLS}_{h,t}-v_{h,t}\)、\(e_m=m_{h,t}-v_{h,t}\)，则不要求无偏时的 MSE 最优权重为

\[
\lambda^\star
=\Pi_{[0,1]}
\frac{\mathbb E[e_g(e_g-e_m)]}
     {\mathbb E[(e_m-e_g)^2]}.
\tag{15}
\]

若两者无偏，记方差为 \(\sigma_g^2,\sigma_m^2\)，协方差为 \(\sigma_{gm}\)，则

\[
\lambda^\star
=\Pi_{[0,1]}
\frac{\sigma_g^2-\sigma_{gm}}
{\sigma_g^2+\sigma_m^2-2\sigma_{gm}}.
\tag{16}
\]

这把当前观察到的事实统一起来：hard pseudo voltage 可能因模型偏差而变差，自由 common mode 又可能不可识别；最佳方案应由估计风险在两者之间连续收缩，而不是固定 0.25/0.5/1 权重。

### 6.4 交叉拟合防止“用同一噪声选择并证明自己”

把时间块分成 \(A,B\)：

1. 用 \(A\) 估计 \(R/X\)、选择 clade、估计 \(K\) 和聚合权重；
2. 在 \(B\) 生成 pseudo \(P/Q/V\)，重建 backbone 并计算预测分数；
3. 交换 \(A,B\)，合并候选和分数。

这样可减少两类偏差：clade 因偶然噪声被选中后在同一数据上显得更稳定；聚合权重对样本协方差过拟合。代价是有效样本量下降，因此实验必须比较 ordinary fit、2-fold cross-fit 和按天 leave-one-block-out。

### 6.5 与恢复条件的连接

若聚合后的边界误差通过灵敏度回归映射为距离误差

\[
\|\widehat d-d\|_\infty
\le \varepsilon_{\rm base}
+L_v\max_{h,t}|\widehat v_{h,t}-v_{h,t}|,
\tag{17}
\]

则降低伪边界误差会扩大已证明的 RNJ 安全区间

\[
2\varepsilon\le\tau<\lambda_\star-2\varepsilon.
\tag{18}
\]

注意：MSE 下降本身不能推出 (17) 的高概率上界；需要对子高斯尾部、block bootstrap 或稳健截断另行给出条件。这一点应在论文中明确区分。

## 7. 新思路三：残差门控的局部拓扑修复

### 7.1 两种工作模式

- **无先验模式：** 当前 candidate-free hidden-tree identification，从零构造候选。
- **有先验模式：** 给定 GIS/历史拓扑 \(T_0\)，先检测异常区域，只在局部解锁并修复，可信 clade 保持冻结。

有先验模式更贴近 2025--2026 年 DSO 工作流，也能显著缩小组合搜索空间。

### 7.2 残差定位与局部候选

在 \(T_0\) 上进行 fixed-tree EIV/AC 拟合，定义节点或 clade 的标准化留出残差

\[
R_i=\sum_{t\in\mathcal H}
\frac{(v_{i,t}-\widehat v_{i,t}(T_0))^2}{\widehat\sigma_{i,t}^2},
\tag{19}
\]

并把相邻高残差终端聚合为待解锁区域 \(\mathcal U\)。只在 \(\mathcal U\) 与其一圈边界内运行置信 RNJ、四元组 completion 和局部 clade reattachment。

### 7.3 带编辑先验的物理排序

\[
\widehat T
=\arg\min_{T\in\mathcal C(\mathcal U)}
\Bigl\{
\mathcal L_{\rm AC}^{\rm hold}(T)
+\lambda_{\rm edit}d_{\rm edit}(T,T_0)
\Bigr\}.
\tag{20}
\]

\(d_{\rm edit}\) 可取 rooted-clade symmetric difference、受约束 RF 距离或实际开关/接线操作成本。若真树 \(T_\star\in\mathcal C(\mathcal U)\)，则相对于任意假树 \(T\)，选择真树的充分条件是

\[
\mathcal L_{\rm AC}^{\rm hold}(T)
-\mathcal L_{\rm AC}^{\rm hold}(T_\star)
>
\lambda_{\rm edit}
\bigl[d_{\rm edit}(T_\star,T_0)-d_{\rm edit}(T,T_0)\bigr].
\tag{21}
\]

这清楚显示编辑先验的风险：\(\lambda_{\rm edit}\) 过大时会保护错误历史记录。它应由已知开关事件/人工核验样本选择，而不能用测试真值调参。

### 7.4 输出形式

工程输出不应只有一棵树，而应包括：

- 保持不变的可信 clade；
- 待核验的局部区域；
- 每个建议编辑对留出 AC NLL 的改善；
- 编辑前后潮流预测误差和约束违例；
- 若多个局部候选无法区分，明确输出“需人工或主动探测”。

## 8. 新思路四：三相多视图共享隐藏树

### 8.1 共享路径的块矩阵形式

在线性化三相四线制模型中，终端 \(i,j\) 间的灵敏度块可写为

\[
S_{ij}=\sum_{e\in P(r,i)\cap P(r,j)} Z_e^{(3)},
\tag{22}
\]

其中 \(Z_e^{(3)}\) 包含自阻抗、互阻抗和中性线等效影响。拓扑共享，但各相/相对的观测质量不同。

为得到可用于树重建的正边长标量视图，可取若干半正定投影 \(W_m\succeq0\)：

\[
\rho^{(m)}_{ij}=\operatorname{tr}(W_m\operatorname{Re}S_{ij}),
\qquad
d^{(m)}_{ij}
=\rho^{(m)}_{ii}+\rho^{(m)}_{jj}-2\rho^{(m)}_{ij}.
\tag{23}
\]

只要每条物理边在所选投影下具有非负增量，每个 \(d^{(m)}\) 都对应同一棵树上的不同加性边权。

### 8.2 联合重构

对每个视图估计半径 \(r^{(m)}_{ij}\)，用标准化联合分数

\[
\mathcal S(T)=
\sum_m\omega_m
\sum_{i<j}
\frac{(\widehat d^{(m)}_{ij}-d^{(m)}_{ij}(T))^2}
{(s^{(m)}_{ij})^2}
\tag{24}
\]

选择共享 clade 结构，而允许各视图有独立边长。若相别未知，则交替进行：相别软分配 → 三相 sensitivity 约束估计 → 共享树候选 → 不平衡 AC 留出排序。

### 8.3 理论目标

证明可沿单视图 RNJ 结果推广：若存在权重 \(\omega_m\) 使联合距离误差小于联合最小可辨识边长的一半，则共享拓扑可恢复。更强的目标是说明：即使任一单相视图都不足，只要不同视图的失效位置不相同，联合置信规则仍可确定更多 clade。

### 8.4 风险

- 相别错误会污染块矩阵并制造系统偏差，不是简单独立噪声；
- 中性线和不平衡负荷可能使某些投影边长接近零；
- 真实智能表常无电压相角，必须采用与 Fang 等相近的角度近似或电流分解；
- 三相方向应在单相置信候选集完成后开展，否则同时改变过多假设，难以归因。

## 9. 新思路五：量化、缺测、异步的一体化在线模型

### 9.1 量化区间似然

若智能表只上报 bin \([a_{it},b_{it}]\)，对潜在连续量 \(Y_{it}\sim N(\mu_{it}(T,R,X),\sigma_i^2)\)，其正确似然项是

\[
\Pr(a_{it}\le Y_{it}\le b_{it})
=\Phi\!\left(\frac{b_{it}-\mu_{it}}{\sigma_i}\right)
-\Phi\!\left(\frac{a_{it}-\mu_{it}}{\sigma_i}\right),
\tag{25}
\]

而不是把 bin 中心当作无误差连续值。缺测时删除该因子；若只缺一类 \(P/Q/V\)，保留其余因子。

### 9.2 异步与平均窗口

为每个表引入离散时移 \(\delta_i\) 和积分窗口 \(w_i\)：

\[
\widetilde y_{i,t}
=\frac1{w_i}\int_{t+\delta_i-w_i}^{t+\delta_i}y_i(s)\,ds+\epsilon_{i,t}.
\tag{26}
\]

先用日周期、事件边缘或 feeder-common voltage mode 粗配准 \(\delta_i\)，再在 EIV 拟合中把剩余偏移作为区间/随机效应。不能把所有异步误差都交给 \(R/X\) 吸收，否则会系统性扭曲 shared-path matrix。

### 9.3 在线变更检测

维护滑动窗口距离和 clade 置信区间；当留出残差、确定 clade 冲突数或候选置信集直径超过阈值时触发局部更新。若有父级总表，再调用 HSSP 快速确定可能变化的层级；没有父级总表时仍使用局部置信 RNJ。

## 10. 关于“拓扑后验”和“置信集”的表述规范

现有 GTLS generalized posterior 可作为候选权重和排序工具，但除非明确给出生成模型、先验和似然，它不应直接解释为频率学覆盖概率。

建议区分三种输出：

1. **同时区间诱导的候选覆盖：** 在距离区间假设有效时，由 (9) 给出 \(T_\star\in\mathcal C\) 的概率下界；这是当前最可实现的理论目标。
2. **未来量测预测集：** 依据留出/时间块校准，保证未来 \(P/Q/V\) 落入预测区间；它验证候选的预测充分性，但不等于真拓扑覆盖。
3. **有标签的 conformal topology set：** 只有当校准样本含已知真实拓扑（例如仿真、已记录开关事件或人工核验馈线）时，才可对拓扑标签构造 marginal coverage。无标签真实馈线不能凭 conformal 名称自动得到结构真值覆盖。

## 11. 推荐的论文主线与贡献拆分

### 主论文：Uncertainty-calibrated latent-tree identification

建议聚焦四个部件：

1. block/EIV 同时距离区间；
2. interval RNJ + confidence quartet completion；
3. 现有 fixed-tree EIV/GTLS + held-out AC ranking；
4. 唯一树、候选集或拒判三种输出。

建议核心定理：

- 同时距离覆盖 ⇒ 真兄弟/clade 不被拒绝；
- quartet lower gap 为正 ⇒ split 正确；
- 完备 laminar completion ⇒ 候选覆盖 (9)；
- 在 held-out loss 有正分离且候选含真树时，最终选择一致；
- 将量化 bin、EIV 和伪边界误差加入统一 \(r_{ij}\) 分解。

### 方法论文/主论文第二部分：Cross-fitted covariance-aware aggregation

把 (12)--(18) 与现有 one-layer clade aggregation、latent common mode 结合。贡献点应是“相关误差下的最优无偏边界估计 + 数据复用偏差控制 + 对拓扑恢复 margin 的传播”，而不是泛泛声称“聚合降低噪声”。

### 工程扩展：Prior-aware local repair

使用状态估计/AC 留出残差定位、编辑先验和局部 RNJ 修复，对接真实 DSO 的 topology verification/correction 场景。若能获得真实开关记录或人工核验结果，这一方向最容易形成有工程说服力的案例。

### 后续扩展：Three-phase multi-view + quantized online

三相与在线数据缺陷都很重要，但不宜同时塞进第一篇理论主论文。先在单相条件下把 candidate coverage 做扎实，再推广块矩阵共享树。

## 12. 实验设计：必须回答的问题

### 12.1 基线

- 当前 ordered OLS/RX75/RNJ 完整基线；
- 当前 NJ、RG、RNJ-grid、GTLS candidate posterior、aggregation/decomposition 主线；
- 2026 FBIG + nonlinear refinement；
- 2024 三相 enhanced RG/backtracking（在三相实验中）；
- HSSP（仅在提供父级/全节点功率表的适用场景）；
- wavelet 或 wavelet-GAT（只在相同输出粒度的 feeder/phase 或开关状态任务中）。

### 12.2 数据条件

- 现有 paper15、soumalas11、flynn16、pengwah18、小型 canonical feeder；
- terminalized case33、IEEE 37/123 和不平衡 OpenDSS 馈线；
- 真实负荷曲线驱动的 AC 数据；若无法公开真实拓扑，至少做真实量测的 predictive validation；
- 随机缺测与整节点缺测；
- 8/10/12/16 bit 量化和真实表计精度；
- 表间时移、不同平均窗口、根电压公共误差；
- PV 高渗透、反向潮流、hidden load leakage、不同 \(R/X\) 异质性；
- 先验拓扑只有 1/2/4 个局部错误的 correction 场景。

### 12.3 指标

- rooted-clade precision/recall/F1、exact recovery、RF/编辑距离；
- **candidate oracle recall**：真树或真 clade 是否进入候选集；
- simultaneous interval empirical coverage；
- 候选集大小、RF 直径和拒判率；
- 唯一输出条件下的错误率（selective risk）；
- 线路参数误差、pseudo boundary RMSE、区间宽度校准；
- held-out AC NLL、电压 RMSE、约束违例；
- 运行时间、beam 宽度和内存；
- topology correction 的错误定位率、建议编辑数和人工核验负担。

### 12.4 关键消融

1. fixed \(\tau\) vs pairwise standard error vs simultaneous max-statistic；
2. 只有 RNJ 候选 vs RNJ+NJ/RG vs confidence quartet completion；
3. 简单平均/中位数 vs diagonal precision weighting vs full-covariance GLS；
4. in-sample 聚合 vs 2-fold cross-fit；
5. hard pseudo vs free common mode vs 最优收缩；
6. 不含真树时的 ranker 行为 vs 含真树时的 ranker 行为；
7. 全局重建 vs residual-gated local repair；
8. 连续高斯噪声近似 vs 正确量化区间似然。

所有超参数必须由训练/验证数据或不含真值的稳定性规则选择；测试拓扑真值只用于最终评价。

## 13. 最短可执行路线

### 阶段 0：竞争基线审计

- 实现 2026 FBIG 的最小忠实版本；
- 在相同 \(R/X\) 和相同 \(P/Q/V\) 两层比较；
- 确认当前主线剩余的独特贡献究竟来自候选覆盖、EIV 还是 AC 排序。

### 阶段 1：同时置信距离与 interval RNJ

- 在现有 sweep 上保存每个 block 的 \(R/X,d\) 重拟合结果；
- 实现 max-statistic 半径 \(r_{ij}\)；
- 将 sibling/parent tests 改为三值关系：accept / reject / unresolved；
- 先在小树全枚举验证 (9)。

### 阶段 2：quartet completion 与候选覆盖实验

- 生成确定 split、矛盾 split 和未决 split；
- 实现 laminar compatibility 检查与 beam completion；
- 首先优化 candidate oracle recall，再看最终 MAP；
- 对每次失败区分“候选缺失”和“排序错误”。

### 阶段 3：协方差最优伪边界与 cross-fit

- 导出每个 child-to-boundary 误差向量；
- 估计 shrinkage \(K_\gamma\) 和 (12)；
- 实现 (15) 的 out-of-fold 收缩；
- 检查 pseudo RMSE 改善是否实际传导到距离半径与 clade coverage。

### 阶段 4：prior-aware repair

- 从真树人工制造 1/2/4 个可解释局部错误作为 \(T_0\)；
- 用 held-out residual 定位并仅解锁局部；
- 比较全重建的运行时间、误改率和正确修复率；
- 若有真实 DSO 数据，再将输出改成面向人工核验的 edit report。

### 阶段 5：三相与量化扩展

- 先复现 Fang 等三相 sensitivity + RG 基线；
- 再把 interval candidate set 推广到多视图；
- 最后引入量化区间似然、缺测 mask 和在线 change detection。

## 14. 建议立即停止或降级的叙事

- 不再把“端用户 \(P/Q/V\)+隐藏节点+约束矩阵+迭代分组”单独写成新颖贡献；2026 FBIG 已高度重叠。
- 不把 bootstrap persistence 称为“贝叶斯后验概率”；它是经验稳定度，除非另有生成模型。
- 不宣称 clade 平均必然带来 \(1/k\) 方差下降；相关误差下正确量是 \(a^\top K a\)。
- 不把开关状态分类 accuracy 与 candidate-free 隐藏树 exact recovery 放在同一表格中直接排名。
- 不在没有带真拓扑校准样本时宣称 conformal topology coverage。
- 不用真实测试拓扑选择 \(\tau\)、beam width、收缩权重或候选数量。

## 15. 最终建议

如果只选择一个最值得推进的方向，建议是：

> **以 2026 FBIG 为强直接基线，完成“同时置信 RNJ + 四元组 laminar completion + EIV/AC 物理排序”的候选置信集方法；随后用协方差最优、交叉拟合的伪边界聚合缩小区间。**

它最充分利用现有实现与数学证明，也精准回应了近期文献尚未解决的问题：不是只追求更高的单树准确率，而是明确说明哪些结构已被数据证明、哪些仍含混、候选集为何应包含真树，以及何时必须拒绝给出唯一答案。
