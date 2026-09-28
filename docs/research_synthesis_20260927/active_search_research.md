# 最优寻线：从结构分歧排序走向可辨识性驱动、面向处置损失的主动查证

日期：2026-09-27。定位：窃电、线路图质量评价与最优寻线的联合研究分报告。本文核对当前本地代码并检索原始论文；新数学式属于本报告的形式化推导或拟议方法，除注明外不是现有实现，也不宣称新定理已经获得学术首创性。没有改动生产代码、历史结果或旧查询脚本。

## 1. 最值得推进的主线

把问题写成“在拓扑、线路参数、未计量负荷和计量偏差都可能解释异常时，怎样用最少的新增证据支持正确处置”。现有 RNJ+MILP 提供结构解释；图质量模块指出哪些解释仍不能排除；寻线模块决定下一次测什么、激励谁、查哪个设备；窃电模块的最终输出可以是区域级嫌疑、排除窃电、优先校表或证据不足。

值得研究的核心不是再发明一个熵/成本排序，而是：

1. 明确哪些错误解释在现有测量和允许动作下永远不可区分，给出可计算的证据。
2. 对可区分但很接近的解释，计算最低查证成本的参照，并设计能接近该参照的动作分配。
3. 优先消除会改变处置的解释；允许多棵图、多个参数解保留在同一处置等价类内。
4. 把候选缺失和模型失配作为需要另行检验的对象，避免所有候选一致错误时提前停止。

仅有“损失驱动主动学习”已经不足以声称新颖。Huang 等的 AISTATS 2026 论文已经系统提出按下游决策损失设计采集目标，并讨论加权 Bregman 损失的可计算形式。Topo 的贡献应落在物理混淆机制、受限动作、跨模态查证和停止证据上。[Loss-Driven Bayesian Active Learning](https://proceedings.mlr.press/v300/huang26b.html)

## 2. 先分清四种“寻线”

| 层次 | 真正选择的动作 | 新增什么信息 | 典型代价 | 最合适的数学问题 |
|---|---|---|---|---|
| 主动读取 | 新增总表/中间表记录、读某组电压/电流、延长观测窗口 | 已存在运行状态下的新增观测 | 通信、采样时长、传感器部署 | 传感器选择、序贯实验设计 |
| 主动激励 | 指定可控逆变器的 P/Q 扰动、短时编码序列 | 干预后的跨节点响应 | 能量/无功使用、时间、响应偏离 | 受限模型判别实验设计 |
| 人工核查 | 查具名开关/电缆、核验表计、现场确认完整下游组 | 独立的物理或计量事实 | 出车、路程、人员、作业窗口 | 自适应诊断与带路由的查询规划 |
| 计算搜索 | 扩大结构候选、搜索新支撑、提高求解精度 | 从原观测提取更多计算证据 | CPU、MILP 时间、内存 | 分支定界、列生成/支撑搜索 |

计算搜索本身不会创造物理信息。更多算力能消除“还没搜到”的不确定性，但不能消除观测等价。反过来，人工查线可以确认当前算法隐藏节点不能表达的实物设备；这种事实不能强行塞成一个算法负整数节点 ID。

建议第一篇论文只实现“短时 P/Q 探测 + 总表/中间表增测 + 简单独立核查”中的两类。人工车辆路径和高频传播可作为后续扩展，避免同时承担四种不同物理与成本模型。

## 3. 当前仓库已经做了什么，缺口在哪里

本节以 2026-09-27 实际文件为准。工作区存在既有修改，本报告没有假定 Git 历史版本就是当前接口。

### 3.1 可直接复用的部分

- `rnj_wzzt_core/README.md`：当前主线为约束 R/X 估计、RX75-RNJ、分块 bootstrap、固定可信 RNJ 边界子树、MILP 自由相容支撑扩展和独立验证选模型。
- `rnj_wzzt_core/rnj_wzzt/estimation/laminar_l1_milp.py` 的 `fit_laminar_l1_sensitivity`：有 `initial_supports`；每次扩展搜索与当前 laminar 族相容的全部非空新支撑，保留固定块且联合重估系数。一次有界单原子扩展的最优性证书不等于完整前向路径的全局最优性。
- `rnj_wzzt_core/rnj_wzzt/pipeline.py::_select_boundary_blocks`：每次重采样确实生成 `sampled_clades`，但仅累计 `_boundary_cherries(sampled_clades)`，没有把所有完整候选树保存在输出中。边界频率不能相乘成联合树概率。
- `artifacts/prior_acquisition_design_20260908/rank_topology_queries.py`：已经能对完整有限树集合计算查询前后经验结构分歧，按收益/成本排序，给回答分支、代表树和输入校验。这是可复用基线，不应包装为本次新创新。
- `docs/literature_textbook_20260911/chapters/probing.tex` 及 `sources_probing.md`、`audit_probing.md`：已有逆变器探测、特征电流、高频传播与端口脉冲压缩的分流说明和可辨识性边界。

### 3.2 必须修正的文档漂移

`rnj_wzzt_core/docs/prior_acquisition_design.md` 写于 2026-09-08，其中“candidate_supports 有限候选接口”和相应调用示例已与当前主线不一致。当前 README 明确核心已移除 `allowed_supports`、`candidate_supports` 及相关兼容别名；当前函数签名也没有这些参数。旧文仍有价值的是查询语义、成本、候选覆盖、负先验和假先验实验设计，不宜直接照抄其主线接入代码。

当前主线仍不能直接用通用负先验表达“不是某 clade”“两具名设备无直接连接”。完整候选过滤可以用于独立实验；若需要约束自由支撑搜索，必须另实现明确约束，并验证初始冻结块的冲突和撤销，不能假装外部删掉一个候选列表就约束了 unrestricted 搜索。

### 3.3 现在真正缺少的环节

1. 完整联合假设：树、线路参数、窃电/未计量负荷、计量偏差，以及能映射到实物的锚点。
2. 各动作的条件观测模型和成本模型。
3. 回答有误、未知、过期和模型失配的记录与更新机制。
4. 查询后从原始数据重拟合、撤销错误冻结块的闭环。
5. 适应性采样和停止条件下仍有效的统计证据。
6. 能辨别候选池共同错误的独立检查。

## 4. 联合状态、动作、目标与等价类

令联合状态为

\[
\theta=(T,z,h,b,\eta),
\]

其中 T 为目标图，z 为线路参数，h 为未计量注入及其位置，b 为计量偏差，η 为其他运行干扰。T 的语义必须预先固定：物理树、可辨识最简树、端口等值或统计图只能选择其一作为结构目标。现场窃电标签不是 h≠0 的数学同义词：合法漏表负荷和偏置也能形成相似电气异常。

在历史 \(\mathcal H_t=\{(a_s,y_s):s\le t\}\) 下，动作

\[
a_t=(\text{mode},\text{location},u_t,\text{duration},\text{sensor set})
\]

产生观测 \(Y_t\sim P_\theta(\cdot\mid a_t,\mathcal H_{t-1})\)。短时线性化示例为

\[
\Delta v_O=R_{OC}(T,z)u_P+X_{OC}(T,z)u_Q+G_\theta\Delta\eta+\epsilon.
\]

O 是测压集合，C 是可控集合；此处 R/X 表示选定电压变量与功率符号约定下的响应系数，实际代码需统一电压幅值/平方幅值及注入/负荷符号。不能从文献直接复制一个系数 2 进入生产数据。

**定义 A：观测等价。** 两状态若对允许的全部可预测动作策略产生相同观测分布，就属于同一观测等价类。有限静态动作且条件独立时，可检查每个动作的分布是否相同；有持续干扰或动作耦合时，应比较整段联合分布。

**定义 B：任务等价。** 若在声明的决策空间和损失下，两个状态不要求不同处置，可合并为同一目标类 g(θ)。例如多个内部串联设备配置对“是否先校验总表”和“先查哪个区域”产生同一最优决定，但不一定是同一物理图。

观测等价类和任务等价类不同。想要识别任务，至少不能存在“观测完全等价、处置要求相反”的状态对。这个条件比让所有 R/X 元素估得更准更直接。

决策 d 可为“更正区域归属”“先核验某表”“追加中间表”“输出未计量区域集合”“保持待定”。损失可预先写为

\[
L(d,\theta)=\lambda_{\rm false}L_{\rm false}(d,\theta)
+\lambda_{\rm miss}L_{\rm miss}(d,\theta)
+\lambda_{\rm topo}L_{\rm topo}(d,T)
+\lambda_{\rm op}L_{\rm op}(d,\theta).
\]

权重由研究任务定义或灵敏度分析给定，不能用测试集挑选。建议同时报告各分量，避免一个合成总分掩盖误指认上升。

## 5. 最有用的新数学：干扰参数投影后的可区分性

### 5.1 先估计参数 vs 先区分解释

D-optimal 设计希望减小一个模型内参数置信椭球体积；T-optimal/KL 判别设计希望让竞争模型在重新选择其参数后仍不能拟合同一响应。后者更贴近“这是拓扑错还是未计量负荷”。T-optimal 不是新概念，已有系统的 Bayesian 与半无限规划研究；本研究可贡献的是电网候选、干扰空间和受限动作下的具体形式。[Dette 等，2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4793413/)、[T-optimal 的双重自适应离散算法](https://link.springer.com/article/10.1007/s00180-023-01370-4)

给定两个固定候选的局部近似，整段采集数据白化后满足

\[
\widetilde Y=\widetilde m_i(U)+\widetilde G\xi_i+\varepsilon,
\qquad\varepsilon\sim N(0,I).
\]

若两个候选的可自由调整干扰差都落在同一线性空间 col(\(\widetilde G\))，则竞争候选重新拟合干扰后的最小平方距离为

\[
2D_{ij}^{\rm prof}(U)
=\min_{\xi}\|\widetilde m_i(U)-\widetilde m_j(U)-\widetilde G\xi\|^2
=\|P_{\widetilde G}^{\perp}[\widetilde m_i(U)-\widetilde m_j(U)]\|^2.
\]

这是投影最小二乘恒等式。对相同已知协方差、固定均值的高斯分布，平方距离的一半等于 KL；有估计干扰时该式首先是局部剖面判别量，不能自动当成已经校准的贝叶斯因子或置信度。

若 \(\widetilde G\) 随候选变化，可使用拼接列空间 \([\widetilde G_i,\widetilde G_j]\) 得到对应自由线性干扰差的投影；若干扰有非负/容量/平滑约束，则应解受约束最小化，不能继续用无约束投影替代精确距离。

**关键解释：** 若候选响应差恰好位于可允许的偏差/隐藏负荷空间中，则再降低传感器噪声也不能解混。需要改变激励方向、增加独立总表/中间表通道，或限制干扰模型并证明限制可信。

### 5.2 Fisher 信息中的同一机制

局部连续参数记为 \(\beta\)（需要辨识）和 \(\eta\)（干扰）。已知高斯噪声下设联合雅可比 \([J_\beta,J_\eta]\)，信息矩阵分块为 \(I\)。有效信息为

\[
I_{\beta\mid\eta}
=I_{\beta\beta}-I_{\beta\eta}I_{\eta\eta}^{\dagger}I_{\eta\beta}
=J_\beta^T\Sigma^{-1/2}P_{\Sigma^{-1/2}J_\eta}^{\perp}\Sigma^{-1/2}J_\beta.
\]

使用广义逆时要按投影解释并检查可估计方向；逆矩阵形式的 CRLB 需要额外正则与满秩条件。离散拓扑没有一般的普通导数，本式用于固定结构的局部参数或连续嵌入诊断，不能凭 Fisher 满秩证明跨树的全局唯一性。

一个可执行指标是重要候选对的 \(D_{ij}^{\rm prof}(U)\)，另一个是有效信息最小特征值。先做小规模精确枚举，检查哪个指标真正预测实际判别错误，不能只展示优化目标变大。

### 5.3 动作集合本身可能抹掉全部差异

若两候选在观测-可控端口之间的响应差为 \(\Delta R_{OC},\Delta X_{OC}\)，而所有允许动作满足

\[
\Delta R_{OC}u_P+\Delta X_{OC}u_Q=0,
\]

则这些动作没有线性区分能力。常见情形是：两候选仅差 \(\alpha\mathbf1\mathbf1^T\)，但所有有功动作都能量平衡，即 \(\mathbf1^Tu_P=0\)。此时提高平衡激励幅值无济于事；必须改动作空间或测量通道。

被动数据若始终 q=κp，只暴露 R+κX；独立无功方向有机会拆分 R/X。但这要求无功能够独立改变，且背景负荷/根电压/电压相关负荷没有同步产生可仿造的响应。它不是任意系统中“加 Q 必然恢复拓扑”的定理。

隐藏恒定负荷在短时差分中可能被完全消除。激励能先识别网络响应，但若两假设具有相同端口响应且只差不响应激励的隐藏负荷，单靠差分响应可能完全看不到后者。必须与基线能量平衡、总表或其他非差分观测结合。

## 6. 动作选择：先做可解释的小型设计，再做复杂闭环

### 6.1 固定候选、已知观测分布的成本信息率

先考虑有限假设 \(i=1,\ldots,K\)、有限动作 A，每次动作已知成本 c(a)>0。假设 \(P_i(Y_t\in\cdot\mid\mathcal H_{t-1},a_t)=P_i^{a_t}(\cdot)\)，即给定过去和当前动作后使用固定的逐次观测核；这不要求对整个自适应动作序列条件化后仍独立。记 \(D_{ij}(a)=\mathrm{KL}(P_i^a\|P_j^a)\)。在真实候选 i 下，区分不同任务类的最优信息/成本率为

\[
J_i=\max_{w\in\Delta_A}
\min_{j:g(j)\ne g(i)}
\frac{\sum_a w_aD_{ij}(a)}{\sum_a w_ac(a)}.
\]

给定所有 D 值，改用每单位成本采样量 x 后是一个线性规划：

\[
\max_{x\ge0,\gamma}\gamma\quad
\mathrm{s.t.}\quad\sum_a c(a)x_a=1,\quad
\sum_aD_{ij}(a)x_a\ge\gamma\quad\forall j:g(j)\ne g(i).
\]

由逐动作 KL 累积与二元变化测度论证，若策略对每个候选均满足 \(P_i(\text{输出正确任务类})\ge1-\delta\)，其中 δ<1/2、未决/不终止计入失败，并满足有限期望总成本及相应绝对连续条件，则

\[
\mathbb E_i[\mathrm{cost}]\ge
\frac{\operatorname{kl}(1-\delta,\delta)}{J_i}.
\]

这里必须要求正确作出结论的概率，不能仅要求错误声明概率不超过 δ；零成本始终输出 unresolved 的策略满足后者，却不满足上述下界的前提。动作分布的 KL 必须对应同一动作成本包含的整批观测：若动作是 n 次独立同方差样本的均值，其协方差是 Σ/n，不能同时把 KL 乘 n 再重复计数。

这可作为**已知模型条件下的 oracle 下界**，不是现有 RNJ 搜索的运行复杂度下界。J_i=0 揭示不可分辨方向；真实候选未知时，按当前估计反复解 LP 只是策略原型，还需要强制探索与停止分析。

数学上这属于 controlled sensing，多假设不是老虎机“臂”。一条观测通常同时比较许多拓扑，并不会给某棵拓扑生成一个独立的固定均值奖励。Nitinawarat 等对主动多假设检验和非均匀动作成本已有基础理论；best-arm 的 Track-and-Stop 只能作为分配思想与对照，不能直接粘贴其最优性声明。[Controlled Sensing](https://arxiv.org/abs/1205.0858)、[非等成本和 Markov 观测扩展](https://arxiv.org/abs/1310.1844)、[Track-and-Stop](https://proceedings.mlr.press/v49/garivier16a.html)

### 6.2 面向处置的 Bayes 风险下降

若具有经独立检查的先验和似然，定义

\[
\mathcal R(p)=\min_d\mathbb E_{\theta\sim p}L(d,\theta),\qquad
V(a)=\mathcal R(p_t)-\mathbb E_{Y\mid a,\mathcal H_t}\mathcal R(p_{t+1}).
\]

选 \(V(a)/c(a)\) 是一步贪心；若比较收益与新增成本，应统一单位并最大化 \(V(a)-c(a)\)。前者适合成本约束的启发式，后者适合单步经济决策。两者都不是一般的多步最优策略。

若只有 bootstrap/方法候选权重，可以计算完全相同的有限加权式，但必须称“候选内决策分歧收益”。没有可靠似然时，不应称“预计真实误判下降”或“后验风险”。

### 6.3 不愿给概率时的保守版本

给定可信状态集合 C_t，定义

\[
\mathcal R_{\rm wc}(C_t)=\min_d\sup_{\theta\in C_t}L(d,\theta).
\]

可比较动作在各观测分支后的最坏风险或 minimax regret。集合 C_t 若只是有限候选内逼近，得到的最坏值可能低估真实最坏值；只有它覆盖真实状态，才能把“保守”解释为相对于真实系统的保证。

2026 年已有 Maximin Robust Bayesian Experimental Design 工作，针对信息论约束下的模型失配给出稳健实验设计；“把目标改成 max-min”也不是自动的新贡献。应给出与电网干扰集合、动作物理约束和新证据通道对应的特殊结构。[2026 预印本，v2 为 2026-05-12](https://arxiv.org/abs/2603.14094)

### 6.4 路由与动作相关性

人工查线的费用更接近 c(a_t,s_t)，s_t 包含人员所在位置、已打开设备、作业窗口等。先查 A 再查 B 可能比两次独立出车便宜；当前可行动作也可能随回答变化。用单独 c(a) 的收益比无法证明路径最优。第一阶段可以固定位置成本；第二阶段再对小场景用动态规划/整数规划求全局参照，对大场景用滚动规划，并单列路由收益。

## 7. 为什么不能随意宣称子模性和贪心近似保证

固定线性高斯模型、已知独立传感器噪声和适当的信息目标，可能产生 log-det 类型的次模结构；GP sensor placement 已有经典近似理论。自适应次模性进一步要求每个已实现历史下的条件边际收益具有递减性，不等于“平均信息越看越少”。[Krause 等，2008](https://www.jmlr.org/beta/papers/v9/krause08a.html)、[Golovin 与 Krause，2011](https://arxiv.org/abs/1003.3967)

**独立可手算反例。** 令 U、V 是独立公平 bit，任务标签 Z=U xor V。有两个无噪声动作 a_U、a_V，分别揭示 U 与 V。若只想判定 Z，零一 Bayes 风险初始为 1/2；读任何一个 bit 后仍为 1/2；读完两个后为 0。因此

\[
V(\{a_U\})=V(\{a_V\})=0,\qquad
V(\{a_U,a_V\})=1/2.
\]

第二个动作在已经取得第一个回答后才变得有价值，直接违反收益递减。对目标 Z 的互信息也为 0、0、1 bit。这个反例没有声称是电网模型，只证明“决策风险收益必然子模”的一般论断错误。

电网中类似互补来自：先核表计再定位、两个方向激励才能分离 R/X、总表与下游表联合形成平衡方程、先解除候选歧义才知道该测哪条支路。因而必须将两步 lookahead 与一步贪心比较。

如果想用“等价类边切割”获得可证明近似，EC2/ECED 是值得研究的方向：对不同任务类的候选对赋权，动作尽量切除跨类混淆边。但 ECED 已有理论，且二元测试、持久噪声、测试噪声相互独立以及部分定理中噪声率不随隐藏根因改变等条件不能省略。连续电压、可重复的新噪声、相关偏置不自动满足。其原论文也明确目标错误概率一般不具备自适应次模性。[Chen 等，2017，原文 §2–4 与 Theorem 1](https://yuxinchen.org/files/papers/chen17eced-ejs.pdf)

## 8. 停止规则：效率算法与有效证据分开

### 8.1 一个可严格证明的有限简单假设起点

固定 K 个简单假设，每个 \(p_i(y_t\mid a_t,\mathcal H_{t-1})\) 都是其真实条件密度，所有假设使用同一条基于已观测历史的动作策略。随机动作所用的新随机数也纳入取观测前的信息，且不额外访问真实隐藏状态。定义

\[
\Lambda_{j/i}(t)=\prod_{s=1}^{t}
\frac{p_j(Y_s\mid a_s,\mathcal H_{s-1})}
{p_i(Y_s\mid a_s,\mathcal H_{s-1})}.
\]

在 i 为真时，\(\Lambda_{j/i}\) 是初值 1 的非负鞅（若支持条件不足，可用对应超鞅版本）。设置 B=(K−1)/δ。只有某个 j 对所有任务类不同的 i 都满足 \(\Lambda_{j/i}(t)\ge B\)，才输出 g(j)。

证明很短：真实状态为 i，输出错误类别必须存在 j≠i 曾使 \(\Lambda_{j/i}\ge B\)。由 Ville 不等式及至多 K−1 项并集界，错误类别概率不超过 \((K−1)/B=δ\)。动作如何自适应选择不改变该论证，只要条件似然确实正确且动作不使用未来观测。

这是经典检验鞅原理在任务类上的直接应用，不是新提出的统计定理。[Test Martingales, Bayes Factors and p-Values](https://arxiv.org/abs/0912.4269)、[time-uniform confidence sequences](https://arxiv.org/abs/1810.08240)

优点是“动作选得是否好”和“错误声明概率是否受控”可以分开。该界严格控制的是 \(P_i(\text{某时刻输出错误任务类})\le\delta\)，并不保证有限时间停止，也不保证正确作出结论的概率达到 1−δ。坏动作可以永不停止，零成本始终 unresolved 也可以满足此错误声明界；因此不能仅凭它调用 §6.1 的查证成本下界。若要比较该下界，还需证实终止/正确声明条件，或把未决计入失败。预算用尽、所有可行动作不能分离时应输出 unresolved；不能以最高似然候选强行补成一次已认证决定。

### 8.2 不能跨越的适用边界

- 使用同一批数据拟合 R/X、噪声方差、候选拓扑后把 plug-in likelihood 当已知密度，不满足以上简单假设条件。
- 多个拓扑各自重拟合连续参数所得 GLR，一般不是检验鞅。
- bootstrap 权重、MAE 差、投票比例不能替代似然比。
- 只在每轮检验上用 0.05 后反复查看，不控制总体错误率。
- 采集过程中不断扩充 K 或重新挑选候选，不能无说明地保留原阈值；要预先定义候选宇宙和错误预算，或另用有效的复合假设过程。
- 若真状态不在候选集合中，有限假设检验没有真实模型保证。即使某候选比所有其他错误候选都好，也仍可能共同错误。

复合 controlled sensing 文献已处理特定单参数指数族等模型，但与非线性 AC、多节点隐藏负荷和连续参数约束相差很大。第一版应明确限定为经过生成模型验证的有限简单假设仿真，再逐步扩展。[Deshmukh 等，2019/2021](https://arxiv.org/abs/1910.12697)

### 8.3 可执行的三类终止状态

1. `decision_certified`：满足预先声明的模型与序贯证据条件，输出任务类和有效范围。
2. `budget_exhausted_or_unresolved`：达到预算或全部允许动作的可分离能力不足，保留等价类。
3. `model_or_candidate_failure`：独立拟合检查失败、所有候选对新增数据解释很差、现场回答无候选支持；启动扩充候选/复核模型，不给确定拓扑。

还可提供“所有 C_t 中的状态都允许同一安全处置”的鲁棒决策终止，但该保证取决于 C_t 是否真包含实际状态，以及模型约束是否可信。

## 9. 与窃电、图质量组成完整故事

闭环建议按以下顺序进行：

1. **生成解释。** 从被动 P/Q/V、总表与历史图产生候选；在无窃电/有未计量负荷/计量偏差/拓扑错误等模型族间保留替代解释。不给“窃电模型更灵活、拟合更好”直接赋予真实性。
2. **评价有害歧义。** 图质量不仅报 F1 或电压 MAE，还报任务类分歧、重要候选对的可区分性、固定测试工况的处置损失。
3. **挑新增证据。** 能用小成本测量打破歧义就先测；对端口观测等价的结构改用总表/中间表/实物核查；若动作仍不能分离则停止在可辨识区域。
4. **重新估计。** 新回答具有独立证据来源；保留其误差模型。纠正冻结块时必须回到原始终端，不在不可逆收缩后偷偷忽略矛盾。
5. **输出处置与证据。** 可以输出“下游区域存在未解释功率，先复核表计”和多个可行接线解释；实际窃电指认需要进一步现场事实支持。

图质量可定义为任务相关的矩阵而非单一分数：行是结构/未计量/偏差解释对，列是可行动作，元素是判别量与成本。它直接告诉用户“当前图哪里不确定、这种不确定是否影响决策、哪个新增证据最便宜”。这比把每条线的 bootstrap 稳定度涂色更接近论文的新问题。

可以把实际科学问题浓缩为：**哪些局部图歧义会导致异常归因错误，哪些观测能以最低代价消除这些歧义？**

## 10. 可证伪实验设计

### 10.1 第一阶段：小规模精确 oracle

4–8 末端的小树、2–3 种异常机制、有限 P/Q 动作和二元人工回答；固定已知噪声模型。完整枚举或保留已知覆盖的状态集合，动态规划给出小预算最优策略，LP 给出成本信息率下界。

至少比较：随机动作、最大候选熵下降、旧结构分歧/成本、D-optimal 参数设计、剖面判别设计、任务风险/成本、两步 lookahead、oracle 全状态最优。

精确对照回答三个问题：新目标是否减少实际错误处置；贪心是否被互补动作击败；停止规则是否达到声明错误率。统计错误率需要足够独立重复和区间，不能用少数全正确种子宣称 1% 控制。

### 10.2 第二阶段：现有 AC 仿真上的现实失配

固定训练/验证/测试场景与种子，选择动作时不接触真结构与测试标签。用与推断模型不同的 AC 仿真器产生动作响应，逐项加入：背景负荷漂移、根电压变化、独立 Q 激励受限、ZIP 负荷、电压/电流偏差、错误先验、未建模合法负荷、候选真值遗漏、相别误差和不平衡三相。

动作方案应采用同预算、同动作可行域、同平均注入能量或同扰动时长；不能让主动方法得到更多总表数据却把所有增益归于“更优排序”。

### 10.3 四个必须保留的失败对照

| 可证伪假设 | 故意构造的反例 | 若主张失败，应如何报告 |
|---|---|---|
| 判别投影优于普通参数设计 | 无干扰、所有参数方向同等关系到任务 | 预期优势可能消失；证明改进来自目标差异 |
| 多模态查证优于仅探测 | 增加的总表有大偏置或与下游同步错误 | 不能掩盖误差，说明额外通道何时无价值 |
| 候选一致可停 | 所有候选缺失同一真实支路/隐藏负荷 | 测候选外报警召回率与误报率 |
| 贪心接近最优 | 只有两步联合测量才有任务信息 | 用 exact DP 量化成本差距，禁止套次模界 |

### 10.4 主要指标

- 达到正确任务类的实际成本/时间；未解决率与超预算率。
- 错误指认、漏检、区域大小；候选真值/任务类覆盖率。
- 未直接告知结构的 clade/支路归属恢复，而不仅全结构分数。
- 独立工况 MAE、AC 可行性与实际处置损失；图正确率和运行质量分开。
- 序贯证据的覆盖/错误率及置信区间；模型失配下明确记录失效幅度。
- 查询前预测收益与实际风险下降的校准散点。
- 动作优化、重新拟合、MILP 超时的耗时分解；失败和超时进入分母。

保留直接告知事实集合 K，未告知结构评价应剔除 K 及其逻辑蕴含，以免把现场答案本身计为算法“新恢复”。

## 11. 最小实现方案与论文拆分

### 11.1 推荐新增的独立研究层

不急于改主线，在新实验目录实现五个接口：

```python
JointHypothesis = {tree_id, canonical_clades, rx_parameters,
                   anomaly_model, nuisance_parameters, physical_anchor_map}
Action = {action_id, mode, target_ids, p_pattern, q_pattern,
          duration, sensor_ids, cost, feasibility_model}
predict_observation(hypothesis, action, history)
score_actions(hypotheses, history, loss, action_set)
update_and_check(hypotheses, action, observation, evidence_model)
```

必须额外保存 `weight_semantics`（经验权重/贝叶斯后验/无权集合）、`candidate_scope`、`likelihood_version`、`noise_calibration_split`、`stop_reason`、每次回答的原始值及是否“无法判断”。新文件可读取现有估计结果，旧查询脚本作为基线保持不变。

先在有限假设 oracle 中完成 `predict_observation` 与精确停止规则，再把真实候选生成器接入。否则不容易分清失败来自搜索缺失、物理模型、动作选择还是统计证据。

### 11.2 优先级

**首选论文故事：**“拓扑与异常混淆下，受限主动测量能辨识什么，最低成本是多少，如何选择新增证据并允许拒绝结论。”三个贡献块分别是不可分辨条件、目标导向设计和经匹配对照验证的闭环。先聚焦隐藏负荷与参数/局部结构混淆，不同时加入复杂车辆路由和高频电磁传播。

**第二篇或扩展：**“物理图不确定下的图质量证书与查询推荐。”把结构稳定度转换为任务相关的歧义诊断矩阵，同时提供候选内与候选外证据。但若只有一种经验颜色图，没有有效性或处置收益证明，论文强度有限。

**不推荐当前作为主攻：**“大模型/RL 自动最优查线”“任意场景高频注入重建全物理图”“bootstrap 后验 + 熵贪心 + 次模最优”。这些方向要么已有成熟框架，要么先决物理/统计条件不足，容易掩盖当前最有价值的可辨识性问题。

## 12. 原始文献、阅读程度与创新边界

下表不是宣称穷尽 2026 文献；用于建立最相关比较边界。网页抓取时间不当作论文发表时间。A=打开全文并核对本报告相关部分；B=打开原始摘要/元数据；C=仓库已有全文审查记录并在本轮复核关键来源。

| 文献/原始链接 | 时间与阅读程度 | 可借用内容 | 本工作必须超过或明确区别的部分 |
|---|---|---|---|
| [Cavraro & Kekatos, Graph Algorithms for Topology Identification using Power Grid Probing](https://arxiv.org/abs/1803.04506) | 2018；B+C | 叶节点探测、全测压与部分测压的不同恢复对象 | 全叶可控/全节点测压假设，不能推广成任意终端配置物理全恢复 |
| [Inverter Probing for Power Distribution Network Topology Processing](https://arxiv.org/abs/1802.06027) | 2019 期刊；B+C | 拓扑恢复和状态核验的优化表述 | 主动采集本身已很成熟；新增点须是异常混淆与查证成本 |
| [Smart Inverter Grid Probing for Learning Loads, Part II](https://arxiv.org/html/1806.08836) | 2019 期刊；A | 受逆变器/网络约束的探测设计、雅可比条件 | 本文对象是已给图下的隐负荷估计；不是联合未知图与异常已解决 |
| [Pulse Compression Probing for Tracking Distribution Feeder Models](https://arxiv.org/abs/2305.19465) | 2023；B+C | 编码信号与端口动态模型辨识 | 端口模型不是内部物理树，不混用 quasi-steady R/X |
| [Bayesian T-optimal discriminating designs](https://pmc.ncbi.nlm.nih.gov/articles/PMC4793413/) | 2015；A | 竞争模型重新调参后的判别设计 | T-optimal 与 max-min 判别不是新概念 |
| [Computing T-optimal designs via nested semi-infinite programming and twofold adaptive discretization](https://link.springer.com/article/10.1007/s00180-023-01370-4) | 2023 在线论文；A | 判别设计的嵌套优化和求解途径 | 大问题必须报告数值最优范围，不把局部解叫全球最优 |
| [Controlled Sensing for Multihypothesis Testing](https://arxiv.org/abs/1205.0858) | 2013 TAC；B | 受控观测、动作选择、序贯停止 | 需要模型条件，不能把 AC+未知扰动直接视为简单假设 |
| [Controlled Markovian Observations and Non-Uniform Control Cost](https://arxiv.org/abs/1310.1844) | 2013 预印本；B | 非等成本控制与时间相关观测 | 人工路由状态、负荷记忆要明确建模 |
| [Sequential Controlled Sensing for Composite Multihypothesis Testing](https://arxiv.org/abs/1910.12697) | 2019 预印本；B | 某些复合假设的有效序贯设计 | 其单参数指数族等假设不是自由 AC 隐变量模型 |
| [Optimal Best Arm Identification with Fixed Confidence](https://proceedings.mlr.press/v49/garivier16a.html) | COLT 2016；B | 信息分配与固定置信停止思想 | 一棵树不是独立奖励臂，不能机械迁移 |
| [Adaptive Submodularity](https://arxiv.org/abs/1003.3967) | JAIR 2011；A | 条件边际收益递减、适用时的近似保证 | 决策风险、相关噪声、互补测量必须单独证明 |
| [Near-optimal Bayesian active learning with correlated and noisy tests](https://yuxinchen.org/files/papers/chen17eced-ejs.pdf) | EJS 2017；A，§2–4、Theorem 1 | 任务等价类、ECED 的噪声折扣 | 持久二元噪声等条件不能省略 |
| [Online learning for robust voltage control under uncertain grid topology](https://arxiv.org/abs/2306.16674) | TSG 2024；A，通过作者 2026 学位论文对应章核对 | 学习一致模型集合与鲁棒控制相结合 | “拓扑不确定也可做决策”与“首个闭环”均不能作为新奇主张 |
| [A data-driven approach for topology correction in low voltage distribution networks with PVs](https://arxiv.org/html/2506.20238v3) | 2025-06 首发、2026-03-27 v3；A 浏览 | 当前拓扑校正与 PV 背景 | 要比较纠错候选、物理校验、测量假设，不能泛称旧方法都静态 |
| [Active Learning of Model Discrepancy with Bayesian Experimental Design](https://arxiv.org/abs/2502.05372) | CMAME 446, 2025, 118198；B | 模型误差与主动设计的耦合 | 自适应设计在错模型下会自我强化；本任务需保留未解释残差 |
| [Loss-Driven Bayesian Active Learning](https://arxiv.org/html/2604.11995v1) | AISTATS 2026；A，§3、Theorem 1 | 按任务损失设计数据采集 | 通用损失驱动不是本文新贡献；离散树/处置约束也不自动满足解析 Bayes-act 条件 |
| [Maximin Robust Bayesian Experimental Design](https://arxiv.org/abs/2603.14094) | 2026-03-14，v2 05-12；B | 信息论不确定集下稳健设计 | 只加一个 max-min 目标不足；尚未核对其全部定理，不能直接转用保证 |
| [Time-uniform confidence sequences](https://arxiv.org/abs/1810.08240) | Annals of Statistics 2021；B | 适应性停止需要 time-uniform 证据 | 不等于现有 residual/GLR 已满足鞅条件 |

补充检索发现的 2026 年 SOP/Transformer/AC 残差二阶段拓扑方法，其正式出版社页面正文未能完整打开，本报告不引用其准确率数字，也不据此做全面优劣判断。搜索中还出现期卷日期为 2026 年 12 月的条目，晚于本报告日期，未把它当作已经完整验证的现有成果。

## 13. 可复用 Scholar / IEEE Xplore 检索式

Google Scholar 使用引号限定术语，按 2024–2026 年过滤，再回到作者稿或出版社核对：

```text
"distribution topology" "active probing" identifiability
"inverter probing" "load" "identifiability"
"topology identification" "optimal experimental design"
"electricity theft" "topology uncertainty"
"unmetered load" "active" identification distribution
"controlled sensing" "composite" "non-uniform cost"
"equivalence class determination" "noisy tests"
"decision-focused" "experimental design"
"T-optimal" "model discrimination" nuisance
"Bayesian experimental design" "model discrepancy"
"topology correction" "PV" distribution 2025 2026
```

IEEE Xplore Advanced Search 分开检索，避免 active distribution network 仅指含 DER 网络而非主动探测：

```text
("All Metadata":"topology identification") AND
(("All Metadata":"inverter probing") OR ("All Metadata":"signal injection"))

("All Metadata":"distribution") AND
(("All Metadata":"unmetered load") OR ("All Metadata":"electricity theft")) AND
(("All Metadata":"identifiability") OR ("All Metadata":"active"))

("All Metadata":"controlled sensing") AND
(("All Metadata":"hypothesis testing") OR ("All Metadata":"cost"))

("All Metadata":"topology uncertainty") AND
(("All Metadata":"voltage control") OR ("All Metadata":"sensor placement"))
```

先按 DOI/标题去重；记录首发日期、版本日期、图语义、观测集、可控集、噪声条件、是否包含真实结构、理论目标和代码。Google Scholar 引用数与检索摘要不能作为理论正确性证据。

## 14. 当前结论的证据等级

- **已核实：** 当前核心自由支撑接口；旧先验文档的接口漂移；已有查询脚本的经验分歧语义；完整 bootstrap 树未在该边界选择函数输出中保留；主要原始文献的上述适用范围。
- **本报告直接推导：** 线性干扰投影、有限已知模型的成本信息率 LP、简单假设任务类停止界、XOR 非次模反例。它们是标准理论的具体整理，不以此声称首创。
- **待实现并验证：** 真实 Topo 候选生成到多模态查询的全闭环；带未知连续参数的有效停止；AC 模型失配下的实际收益；现场成本与实际硬件探测。
- **不作保证：** 全物理树可恢复、窃电身份可唯一归因、候选内权重已校准、全局最优寻线路径、任何实际电网动作已经执行。
