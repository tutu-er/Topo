# 线路图质量：从分数走向可停止的结构证书与任务风险

调研与现场代码核查日期：2026-09-27。本文是本轮自主研究的“线路图质量”分报告，仅新增本文件，不改算法、基线、已有测试和结果。本文没有完成电网覆盖率实验；下述实验均明确标为待执行。

**建议主线：利用现有 RNJ/L1-wzzT 提供计算起点，构造允许窃电及参数误差的联合相容集合；用可核查的求解界排除结构，用集合的投影报告确定/未决区域，再按实际巡检或运行损失评价图质量。最值得新增的部分是“求解中途也能诚实输出”的算法与验证，不是再把一个残差归一化成概率。**

这里“诚实”指：在明确模型与校准条件成立时，保留所承诺的重复抽样覆盖/错误控制；不指结果一定精确、集合一定小，也不指电网真实噪声已被验证符合模型。

## 1. 旧基础、新增推导、当前证据

| 层次 | 内容 | 本轮定位 |
|---|---|---|
| 已有研究文档 | 四点距离区间、图置信集合、全体 clade 的同时投影、最坏结构损失、Universal Inference、内/外近似方向 | 已在 `topology_graph_probability_theory_20260923.md` 第 3–5 节讨论；不包装为本轮新发现 |
| 已有研究文档 | 隐藏二度节点、缺少激励、近似不可辨识、弱信号导致诚实集合不得不模糊 | 已在 `topology_edge_probability_impossibility_20260925.md` 讨论 |
| 已有研究文档 | 分数的独立概率校准、Brier/log loss、`D0/D1` 与预测贡献区别 | 已在 `topology_edge_score_probability_calibration_20260925.md` 等讨论 |
| 本轮进一步推导 | 用负对数似然 LB/UB 给出证据区间与三态计算输出；固定数据上随计算时间提高证书而无需重新支付统计显著性水平 | 下文给出简短证明，属于已知统计原理的条件性推论，不是宣称原创统计定理 |
| 本轮进一步推导 | 树内电阻杠杆分数退化；当前 laminar 原子的任务加权电气影响有解析式 | 代数推导，可直接实现指标，但它不是边存在概率 |
| 新研究设计 | 联合拓扑—隐匿负荷外集、主动核查、任务损失停止规则，以及“物理未决/计算未决”分离 | 尚需算法实现、独立枚举 oracle 与 AC 压力实验 |
| 本轮实查 | 当前 core 接口、诊断字段、近期原文/出版信息 | 下文标明访问范围；没有复现这些文献的实验 |

### 1.1 现场代码已发生变化

2026-09-27 读取当前工作区后，以以下事实为准：

- `rnj_wzzt_core/README.md:103-117`：RNJ 前 R/X 回归与 RNJ 后 L1-MILP 是不同阶段；bootstrap 频率不是已校准拓扑后验。当前核心已经移除 `allowed_supports`、`candidate_supports` 等白名单输入；后续 MILP 直接优化新的二进制支撑。
- `rnj_wzzt_core/rnj_wzzt/estimation/laminar_l1_milp.py:1-11`：新支撑须与当前 family 相容；单次有界扩展的精确性不等于整条前向路径全局最优。
- 同文件 `SolverDiagnostics`（约第 35 行）已有 `objective`、`dual_bound`、`mip_gap`、状态、时间与模型规模，适合保留优化证据。
- `solve_fixed_support_l1`（约第 947 行）对给定支撑联合拟合权重；`solve_best_laminar_extension_l1`（约第 1022 行）求一项扩展；`fit_laminar_l1_sensitivity`（约第 1104 行）返回前向路径。
- `initial_supports` 是保留的结构块，其支撑约束后续扩展；即使新支撑搜索已经 unrestricted，也不能推出整个拓扑类被穷尽。`max_atoms`、阻抗上界、保留结构和模型语义仍限定声明域。

因此，旧文档里“正式 core 只在 RNJ 白名单选新支撑”的描述已经过时。与此同时，旧文档对**有限候选作为置信集内近似**的数学警告仍然成立；现在它主要适用于只检查前向路径、候选图缓存、固定初始化家族或不完整备择搜索的做法。

现有求解器字段本身不提供统计证书。把它们接到本文规则前，还必须核对：当前优化问题究竟覆盖哪个原假设、目标是否真为所声明的负对数似然、界是否对整个声明域有效。

## 2. 论文必须先决定“图”的语义

设观测端口为 O，拓扑为 T，连续电气参数为 θ，未登记注入/窃电过程为 h，系统偏差为 b。统一写成

\[
Y_t=f_{T,\theta}(u_t,h_t)+b_t+\varepsilon_t.
\]

P/Q、主动信号幅值若有测量误差，就应把真实输入作为潜变量或区间变量；不能同时把它们当无误差回归量，又声称覆盖其真实不确定性。

建议按照实验信息量选择以下目标，而不是统一称作“物理边”：

| 目标 | 所需信息/约束 | 适合质量输出 |
|---|---|---|
| 已知节点集上的物理线路/开关 | 节点身份、允许物理连接、电气模型及足够观测/激励 | 同时有效的存在/不存在/未决状态 |
| 隐藏节点树的终端 clade 或 split | 根参考与终端可辨识条件；处理零长边及二度隐藏节点 | 部分解析树、确定 clade、尚未分辨的终端区域 |
| 端口响应/Kron 等价类 | 只要求重现观测端口的输入输出 | 响应误差界、任务误差界；不宣称还原全部物理节点 |
| 统计依赖图 | 指定概率模型与混杂假设 | 依赖结构质量；不能直接读作物理接线 |

消去隐藏节点会产生 Schur 补图；它可能带有原网络中不存在的直接端口边。Kron 化还要求正确处理隐藏注入项，因此“把未知负荷消去”并不会让负荷的影响自动消失。[Dörfler–Bullo 原文](https://motion.me.ucsb.edu/pdf/2011d-db.pdf)

**联合窃电时尤需注意：**若两个 `(T,θ,h)` 在可用实验下产生同样观测分布，最多只能识别它们的等价类。把 h 放宽可以避免把窃电误判为错图，但也可能使很多图都能解释数据；这时增大量测数不一定有用，改变探测位置或加入总表约束才可能改变可辨识性。

## 3. 路线 A：先做可验证的集合反演，不依赖后验概率

### 3.1 从观测误差区域得到联合相容集合

【旧思路的可执行化】先声明一个覆盖真实联合量测误差的区域 \(\mathcal E_\alpha\)，满足

\[
\Pr\{\varepsilon\in\mathcal E_\alpha\}\ge1-\alpha.
\]

给定物理声明域 \(\Theta\)，包括允许拓扑、阻抗范围、输入误差、偏差和隐匿负荷限制，定义

\[
\mathcal K_\alpha(D)=\{(T,\theta,h,b):Y-f_{T,\theta}(U,h)-b\in\mathcal E_\alpha\}.
\]

若真实配置属于声明域，覆盖事件上真配置直接属于 K。拓扑集合是其投影

\[
\mathcal C_\alpha(D)=\operatorname{proj}_T\mathcal K_\alpha(D).
\]

这条推理不要求 T 连续，也不要求唯一可辨识。误差区域可以来自确定性仪表界、经过证明的浓缩界、已知联合噪声分布；若只是 plug-in bootstrap 或仿真分位数，则只能按实际校准证据宣称，不能自动写作分布无关有限样本结论。

**实现起步优先级：**先使用线性响应 + 有界误差或已知协方差的完整枚举小问题。先验证集合反演和边投影的逻辑，再处理 AC、EIV、未知协方差与时间相关性。

### 3.2 带窃电容忍的集合如何定义才有用

建议至少画一张“容忍水平—结构解析率”的曲线：

\[
\mathcal H(s,H)=\{h: \text{可疑区域数}\le s,\quad h\ge0,\quad \|h\|\le H\}.
\]

这里非负性只适用于已约定符号的未计量耗电模型；双向 DER、表计符号/偏置问题需要扩展，不能默认都满足。另用总表平衡、事件起止或平滑性形成可审计约束，避免 h 对每时每节点完全自由而把任意错图都吸收。

如果随着 H 增大，某 clade 从确定变未决，这表明原确定性依赖“没有足够大隐匿负荷”的假设。该曲线本身很有研究价值，比只展示一个最优窃电解诚实且可解释。

空集合的含义是当前模型、误差范围或计算结果不相容；它不能独立证明窃电。漏掉根电压误差、线损或仪表偏置也可能产生空集。

## 4. 路线 B：把统计证据与求解界直接连接

### 4.1 Split likelihood 作为严谨基准

将独立案例分为训练 A 与验证 B。条件于 A，固定一个关于 B 数据的正规化预测密度 \(q_A\)。对每张图 T 设

\[
E_T(D_B)=\frac{q_A(D_B)}{\sup_{\eta\in\Theta_T}p_{T,\eta}(D_B)},
\]

其中 η 必须包括原假设允许的阻抗、隐匿负荷、噪声等 nuisance。若 `(T*,η*)` 是真配置，分母不小于真密度，因此

\[
\mathbb E_{T^*,\eta^*}[E_{T^*}\mid A]\le
\int q_A(y)\,dy=1.
\]

由 Markov 不等式，\(\mathcal C_\alpha=\{T:E_T\le1/\alpha\}\) 覆盖真图概率至少 \(1-\alpha\)。这是 Universal Inference 的应用形式，不是本项目创造的新检验。[Wasserman–Ramdas–Balakrishnan 原文](https://par.nsf.gov/servlets/purl/10179988)

“正规化”是实质要求。任意 `exp(-validation_mae)` 不是自动成立的密度；高斯模型若协方差依赖参数，要保留 log determinant 与其他正规化项；L1 目标只有在合适的独立 Laplace 等观测模型及正确尺度下才有直接似然解释。

### 4.2 本轮推导：三态计算输出与安全方向

令

\[
m_T=\inf_{\eta\in\Theta_T}[-\log p_{T,\eta}(D_B)],\qquad J_q=-\log q_A(D_B).
\]

假设求解器/松弛算法给出可靠界

\[
\mathrm{LB}_T\le m_T\le\mathrm{UB}_T.
\]

则直接有

\[
\boxed{\exp(\mathrm{LB}_T-J_q)\le E_T\le\exp(\mathrm{UB}_T-J_q).}
\]

宜在 log 尺度比较，以免指数溢出。阈值为 \(c_\alpha=\log(1/\alpha)\)：

| 计算状态 | 判据 | 可以说什么 |
|---|---|---|
| 认证排除 | LB − Jq > cα | 已知该 T 会被本统计规则排除 |
| 认证保留 | UB − Jq ≤ cα | 已知该 T 会被本统计规则保留；**不是接受它为真** |
| 计算未决 | LB − Jq ≤ cα < UB − Jq，或缺少有效界 | 现有计算尚不能确定它在精确规则中的状态；必须保留到外集 |

因此安全外集可定义为

\[
\mathcal C^{\rm out}_\alpha
=\{T:\mathrm{LB}_T-J_q\le c_\alpha\},
\qquad \mathcal C_\alpha\subseteq\mathcal C^{\rm out}_\alpha.
\]

未访问的 T 可概念上取 LB = −∞，从而仍留在外集；实际不必显式列举，可用尚未剪枝的搜索树节点表示它们。反之，把仅找到的候选 T 输出为 C 会把外集偷换成内集。

**容易颠倒的方向：**对负对数似然最小化，局部可行解给 UB；代入分母会低估 null 最大似然，放大 E，可能错误排除真图。用于安全排除必须是全域 LB，而不是最低已找到残差。

**对当前代码的具体限制：**`dual_bound` 目前对应一个固定家族的一项扩展问题；它不是“所有无此 clade 的树”或“所有合法窃电过程”的全域 LB。要做边检验，必须另建覆盖整个相应原假设的 oracle，或用更宽域的松弛得到保守 LB。

数值上还需保存可行性容差、界容差与目标重算结果；靠近阈值的案例保留为未决。浮点求解器的 `status=optimal` 不自动等于经过精确算术验证的数学证书。若要做强保证，应明确界的数值验证方式或采用向保守方向修正的容差。

### 4.2.1 可以具体实现的全域 oracle

先在 4–6 个终端上独立枚举全部约定的约化层次树；每张树固定 z 后，R/X 系数及截距是线性预测参数。已知尺度独立 Laplace 噪声对应带正确比例常数的 L1/LP；已知协方差 Gaussian 噪声对应凸二次问题。这样能取得每张 T 的精确基准及可靠界，不必先解决大规模非凸 AC。

下一阶段研究全家族 MILP：设置足够数量的原子槽，同时优化所有支撑 z，而不是仅追加一个原子；用二进制乘积变量与有界系数线性化，强制成对支撑嵌套/不交，允许空槽并固定其系数为零。n 个终端上的互异非空 laminar 家族最多含 2n−1 个集合，这给出一个有限槽上界，但具体 root/leaf/stem 约定仍需核对。交换槽的对称性、隐藏零长边的规范化、巨大的系数界都影响实际可解性。

这个新 oracle 可以让 RNJ/wzzT 提供 incumbent 加速，但不能把其初始支撑永久固定为真；否则会把全域假设重新缩回某个搜索分支。也不能用“某槽 z 存在”代表真实边存在而忽略其 r/x 都为零的情形：应采用明确的零长边收缩语义，或给出有物理依据的最小可检测作用阈值。

上述 MILP 可执行性针对已知输入的线性响应模型。若同时让隐匿注入幅值与阻抗自由变化，响应中会出现两个连续变量的乘积，不能继续无条件称为同一个精确 MILP；可先用固定注入模板构造基准，再研究有界全局松弛，并把放松误差计入证书。

如果需要候选预筛选，须另证明真目标被筛掉的概率至多 β；再与 α 级规则结合，最直接的是 1−α−β 的覆盖下界。没有该筛选证据，候选外部必须作为未决保留，不能默认 β=0。

### 4.3 真正可增加的接口：计算随时停止

【本轮条件性推论】保持 D、训练/验证划分和 q 固定。设第 k 次计算产生 LB\(_T^{(k)}\le m_T\)。可以用历史最大下界

\[
\overline{\mathrm{LB}}_T^{(k)}=\max_{j\le k}\mathrm{LB}_T^{(j)}
\]

构造逐步缩小的外集。每个 k 都有 \(\mathcal C_\alpha\subseteq\mathcal C^{\rm out,(k)}_\alpha\)，所以在**任意按计算进展决定的停止时刻 K**，仍然覆盖同一个精确 C；不会因为反复查看 MIP gap 而额外耗费 α。

证明是逐数据点的集合包含关系，与 K 是否依赖数据无关。这使“先花 5 秒输出粗证书，再花 5 分钟减少计算未决”成为合理产品形态。

它不许可在验证集上反复改 q、挑最有利的数据划分或新增量测后仍使用同一固定样本规则。**计算时间的可停止性，与量测时间的可停止性是两件事。**

### 4.4 新量测随时停止需要 e-process 或显著性预算

对于主动实验，令动作 a\(_t\) 在看到 Y\(_t\) 前由历史选择，q\(_t\)(·|历史,a\(_t\)) 是正规化预测密度。正确指定条件量测模型后，可研究

\[
E_t(T)=\frac{\prod_{s\le t}q_s(Y_s\mid\mathcal F_{s-1},a_s)}
{\sup_{\eta\in\Theta_T}\prod_{s\le t}p_{T,\eta}(Y_s\mid\mathcal F_{s-1},a_s)}.
\]

对真 η，分母不小于真联合似然，E\(_t\) 被该真模型下的似然比检验鞅支配，可用 anytime-valid 方法控制“某时刻误排真图”。保守计算下界同样不能放大 E。[Ramdas 等的 SAVI 原文](https://safestatistics.com/wp-content/uploads/2023/10/RamdasSAVIStatScience23.pdf)

这是研究设计。动作安全约束、时间相关误差的条件模型、窗口重用和未知协方差都必须进入证明。简单每过一天再做一次 5% 固定样本检验不提供跨天 5% 保证。若序贯模型暂难建立，可先事前规定批次并分配 \(\sum_t\alpha_t\le\alpha\)。

## 5. 从整集合到图质量：四类量，不合成虚假总概率

### 5.1 同时结构陈述

对非空外集 C，令 S(T) 为可比较的 clade 集，定义

\[
S^- =\bigcap_{T\in C}S(T),\qquad
S^+ =\bigcup_{T\in C}S(T).
\]

在真图属于 C 的同一个事件上，同时满足

\[
S^-\subseteq S(T^*)\subseteq S^+.
\]

所以“必有/必无/未决”是整个集合的投影；从同一个已有效的公共图集合投影所有边，不必再机械逐边 Bonferroni。若改为分别构造许多边检验，则需要另行处理多重性。集合为空时，禁止利用空集交集宣布所有边都确定。

### 5.2 把物理未决与计算未决分开

- **统计/物理未决：**至少两种互斥结构已经有认证保留的可行配置；当前数据/模型不能由这条规则分开它们。
- **计算未决：**尚未排除的相反结构只存在于未完成搜索的分支，尚未找到认证保留的代表。

前者通常需要新信息，后者可能只需要更多算力或更强松弛。注意“认证保留”只是关于所选统计规则，不能证明自然界绝对不可辨识；真正不可辨识需要观测分布相同的证明。

建议分别报告 `resolved_clade_fraction`、`witnessed_ambiguity_fraction`、`computational_unknown_fraction`。这比单独报告 solver success 或边置信度热图更有辨识力。

### 5.3 面向任务的质量，需要保留连续参数集合

只有拓扑投影 C 不够评估电压/损耗/窃电量风险；同一 T 下的不同 θ、h 也可能导致显著后果。应使用联合外集 K。

对最终决策 a（例如巡检哪一组用户、在哪注入特征信号或允许多少功率），给定事先定义的损失 \(L(a;T,\theta,h)\)，计算可靠上界

\[
U_L(D,a)\ge\sup_{(T,\theta,h)\in K}L(a;T,\theta,h).
\]

若 a 也是数据决定的，在同一个覆盖事件上仍有真损失 ≤ U\(_L\)。以 `U_L≤τ` 决定接受，可控制“接受且真实损失超过 τ”的联合事件。**这不自动等于接受子集内的条件错误率 ≤α。**

若需要与各真实网络最优动作比较，使用 regret：

\[
\sup_{\xi\in K}\{c(a;\xi)-\inf_{a'\in\mathcal A(\xi)}c(a';\xi)\},\quad \xi=(T,\theta,h).
\]

最坏损失求解是最大化问题：找到一个危险场景只给下界；安全接受需要上界。不能把“没搜到更坏的”解释为风险已被界住。

### 5.4 一个能形成论文图的现象

【待实验验证】在弱支路/零流工况，结构 F1 可能差但运行损失小；在承载大量功率的骨干，少错一条边可能使 F1 很高而巡检定位或电压决策损失很大。因此同时展示：

1. clade/物理边的结构误差；
2. 指定工况或输入区域内的响应/决策损失；
3. 外集证书对真实损失的覆盖与保守程度；
4. 为达到任务容差所需新增探测/巡检成本。

主张“此图足以支持这项决策”通常比主张“所有线路均已找对”更可执行，也更适合连接窃电与寻线。

## 6. 图论方向的一个实用筛选：有效电阻不是普适置信度

### 6.1 谱近似保证的是二次型

对同节点集、同参考的正权电阻网络，若

\[
(1-\epsilon)L\preceq\widehat L\preceq(1+\epsilon)L,
\]

则能控制所有电位向量的能量二次型。接地后在正定子空间取逆，有

\[
\frac1{1+\epsilon}L^{-1}\preceq\widehat L^{-1}
\preceq\frac1{1-\epsilon}L^{-1}.
\]

这是电阻响应近似的有力工具；它不等于逐条物理边恢复。谱稀疏化正是有意删边但近似保留响应的例子。[Spielman–Srivastava 原文](https://www.cs.cmu.edu/~15859n/RelatedWork/Spielman-Strivastava.pdf)

AC 复导纳、三相耦合、非线性负荷与当前 R/X 共享路径矩阵不直接满足上述全部条件。直接做交流图的 Loewner 排序之前，要先选择可证明的实对称算子或局部线性化对象。

### 6.2 本轮推导：树上的电阻杠杆分数全部等于 1

令边电导为 w\(_e\)，端点间有效电阻为 R\(_{\rm eff}(e)\)。树上该两端只有唯一通路，该通路就是 e，因此

\[
R_{\rm eff}(e)=1/w_e,\qquad w_eR_{\rm eff}(e)=1.
\]

所以将谱稀疏化常用的电阻杠杆分数直接拿来给径向网的现存边排序，会得到所有边相同。它衡量在该给定图中的谱不可替代性，不是现存边是否真的存在的证据。

有效电阻仍可用于传感设计、端口响应距离或网状候选图的冗余分析，但应把图、权值与任务明确给定，不能把它叫成恢复边概率。

### 6.3 对现有 laminar 原子更直接的任务影响量

【本轮代数推导】当前模型中，一条约化 clade 原子有指示向量 \(z\in\{0,1\}^n\)，贡献

\[
R_e=r_ezz^\top,\qquad X_e=x_ezz^\top.
\]

取 \(u=[p^\top,q^\top]^\top\)，\(a_e=[r_ez^\top,x_ez^\top]^\top\)。在保持其余参数固定时，移去该原子的响应差为

\[
\Delta y_e=z(a_e^\top u).
\]

对任务权重 \(W\succeq0\) 及参考输入二阶矩 \(M_u=E[uu^\top]\)，有

\[
\boxed{I_e=E[\|\Delta y_e\|_W^2]
=(z^\top Wz)(a_e^\top M_ua_e).}
\]

证明：\(\|z(a_e^\top u)\|_W^2=(z^\top Wz)(a_e^\top u)^2\)，再取期望即可。若用协方差代替二阶矩，需另外加入均值项，或明确 u 已中心化。

仅有有功、W=I 时，\(I_e=r_e^2|S_e|z^\top M_pz\)。矩阵本身的 Frobenius 影响为 \(\|r_ezz^\top\|_F=r_e|S_e|\)。前者体现实际负荷激励，后者只体现静态响应算子。

**适用边界：**这是固定参数移去响应原子的影响量；不是合法物理断线试验（真实断线可能导致失供），也不是约束无边后重新拟合整个网络的 `D0`。重新拟合可产生补偿，须另算并保留比较。高 I 表示这个原子对所选任务重要，不能证明它属于真实物理图。

可执行扩展：对 K 中存在的同一可比 clade，计算 I 的区间；优先核查“状态未决且最坏任务影响高”的区域。这比仅按最小 bootstrap 频率安排巡检更贴近实际目标。

## 7. 如何把三点组成同一故事

推荐故事不是“三个独立模块的串联精度提升”，而是一个不断缩小歧义的过程：

1. **被动建模：**现有 RNJ/wzzT 给高质量起点；联合模型容忍仪表误差、未计量注入与根参考问题。
2. **质量诊断：**构造联合外集，识别已确定结构、窃电位置等价区域，以及计算未决。
3. **动作选择：**在安全和成本约束下，选择能缩小高损失歧义的探测/核查。
4. **更新与停止：**达到任务风险容差便停止；仍允许保留与当前任务无关的拓扑歧义。

一个有说服力的场景是：“低残差模型把窃电吸收到阻抗/拓扑；联合外集同时保留错图与窃电解释；一次选定位置的特征电流/总表核查使两类解释分离；最终只认证可辨识的窃电区域与关键连接”。它需要独立控制组来证明主动动作优于随机、度数、残差和信息矩阵准则。

### 7.1 最值得投的质量子论文

暂定题目：**面向主动核查的配电拓扑部分辨识：具有统计覆盖与计算中途停止保证的结构证书**。

最少需要三项实质贡献：

- 为当前终端 clade/隐匿注入模型设计可扩展的全域 null 松弛/分支定界，使未搜索结构保留在证书中；
- 给出“固定数据的计算停止”与“自适应新增数据”的不同保证，并以反例证明错误界方向确实破坏覆盖；
- 证明/实验展示任务质量与 F1/残差不同，主动动作能以较少成本达到同一覆盖和任务容差。

只把标准 split LRT 套到若干枚举图、加一张置信热图，创新性有限。若全域计算始终不具备实用性，应缩小题目到小规模局部区域/固定物理候选图的认证核查，避免声称大规模全网保证。

## 8. 近期相邻研究：实际差异在哪里

以下仅使用已访问的原始论文、作者全文或出版方内容作为实质依据；不以搜索摘要代替定理审核。访问日期均为 2026-09-27。

| 工作 | 已核查内容 | 与本方案的关系/差异 |
|---|---|---|
| Deka、Kekatos、Cavraro，Learning Distribution Grid Topologies: A Tutorial，2023 | 原文入口与作者摘要；系统梳理被动/主动、观测配置、物理可辨识性 | 证明“被动+主动拓扑学习”已是成熟框架；新增需放在覆盖/联合歧义/计算证书 |
| Tomaselli 等，EPSR 235:110636，2024；PSCC 作者全文 | 以可训练指数族表示地理网络分布、MCMC 采样；正文也讨论归一化常数的有限背景图近似 | 适合作为合法图先验/生成器方向；其分布不是本项目固定真图的频率置信集，MCMC 样本也不是外集 |
| Li 等，arXiv:2508.05791，2025-08-07 v1 | 全文 III-D 明确用 DBI 与相关性变换加权形成 confidence score；其目标涉及户变连接和相别 | 工程多源融合是相邻路线；目前读到的 score 构造本身不提供本文意义的同时覆盖证明。不要把其应用对象直接当成隐藏物理线路的完整恢复 |
| Integrated data-driven topology reconstruction and risk-aware reconfiguration…，Scientific Reports，2025-08-20 | 出版方全文显示已提出拓扑推断、风险、运行决策的组合故事 | 不可把“拓扑+风险+决策”三词组合作为独占创新；需用可检验的统计和求解保证区分 |
| Park、Balakrishnan、Wasserman，Biometrika 113(2):asaf070，2026；online 2025-11-12 | 原文推断目标是失配模型中的投影/近似投影，方法与条件随散度不同 | 是重要数学扩展，但证明投影覆盖不等于真实 AC 物理拓扑覆盖 |

原始链接分别为：[拓扑学习教程](https://arxiv.org/abs/2206.10837)、[地理图概率模型论文](https://www.sciencedirect.com/science/article/pii/S0378779624005224)、[PSCC 全文](https://pscc-central.epfl.ch/repo/papers/2024/2024_127.pdf)、[多源 confidence 全文](https://arxiv.org/html/2508.05791v1)、[拓扑与风险重构全文](https://www.nature.com/articles/s41598-025-15440-8)、[Robust universal inference 原文](https://academic.oup.com/biomet/article/113/2/asaf070/8321921)。

检索另命中两项，当前只作待核查线索：

- **Confidence-Aware Topology Identification in Low-Voltage Distribution Networks: A Multi-Source Fusion Method Based on Weakly Supervised Learning**，Energies 19(6):1503，2026。[出版方入口](https://www.mdpi.com/1996-1073/19/6/1503)。出版方页面/PDF 本轮多次访问失败；未按原文完成方法或定理审查，因此不据此宣称已有或没有频率覆盖保证。
- **A generative-discriminative framework for distribution network topology identification**，Applied Energy 426 Part C:128743。[出版方入口](https://www.sciencedirect.com/science/article/pii/S0306261926013991)。可见出版方摘要描述候选生成与潮流可信度判别，但刊期显示 2026-12，晚于当前日期，online 日期尚未核实；不作为截至今日已正式发表的定论。其思想接近当前生成候选—物理验证，后续投稿前应核对全文与时间。

### 8.1 Robust universal inference 的正确使用边界

如果真实 AC 分布不在所选线性树模型族中，普通 split likelihood 的正确指定保证不能直接使用。Robust universal inference 提供针对指定散度下模型投影的办法；原文区分精确与近似投影保证，并指出投影本身可能不稳定。[Park 等原文](https://academic.oup.com/biomet/article/113/2/asaf070/8321921)

因此有两种严肃目标：

1. 保留物理解释：显式扩大模型，包括有界 AC 偏差、EIV、未计量负荷，证明真配置仍在域内；代价是集合可能更宽。
2. 接受模型投影目标：研究“对某类运行任务最好的线性等效图”，清楚把结论写成近似模型质量，而非真实物理线路认证。

不能同时享受一个很窄的失配模型和物理真图覆盖，而不支付额外假设。

## 9. 独立验证矩阵：先让错误主张被实验抓出来

下面是建议执行计划，不是完成结果。优先新建隔离研究模块，调用公开 core 接口，保留当前基线；禁止覆盖 `meter_theft_pilot/`。

| 试验层 | 独立 oracle/真值 | 必须改变的因素 | 主要检验 |
|---|---|---|---|
| 代数单元 | 手算 clade 原子与独立矩阵计算 | r/x、W、非零均值、相关 P/Q | 影响量解析式是否等于直接二阶矩积分；均值遗漏反例 |
| 极小拓扑全枚举 | 已知节点树用 Prüfer 序列；隐藏树另用独立层次枚举 | n=4–6、初始块正确/错误、短边/零边 | 全域最优与每次扩展最优是否分清；找到漏真图时保留失败 |
| 精确统计 oracle | 线性高斯已知协方差或有界误差 | 样本量、噪声、弱信号、共线激励 | 图集合覆盖；全部确定 clade 的同时误判率；集合宽度 |
| 求解方向反例 | 完整枚举得到真实 mT | 正确 LB、错误 incumbent、不同 timeout | LB 外集不比精确集合漏得更多；错误方向能导致过度拒绝 |
| 候选覆盖反例 | 人为把真图移出候选/冻结错 clade | 候选域大小与初始化 | 候选内“高 confidence”能否与全域错误并存；不能删掉该组 |
| 不可辨识反例 | 二度隐藏节点拆分、等效注入或零激励 | 更多样本 vs 新探测 | 更多同类数据不应产生虚假的必有/必无；新动作才改变等价类 |
| 模型失配 | 独立 AC 潮流生成器，推断仍线性 | 根参考误差、三相、EIV、相关噪声、重尾、坏表 | 名义覆盖失效应显式展示；扩大偏差域是否恢复覆盖并使集合变宽 |
| 窃电耦合 | 独立植入未计量注入与错线/错参 | h 强度、稀疏度、位置、总表偏差 | 单纯错图解释、单纯窃电解释和联合模型的混淆矩阵 |
| 主动核查 | 同一候选与预算、同一可行动作集合 | random、影响量、EIG、最坏歧义减少 | 在相同覆盖/误报控制下的探测成本、漏检、停止时间 |
| 实际部署总体 | 以网络/馈线为组的独立留出 | 不同馈线/设备/工况 | 仿真概率/区间迁移能力；不把同一图的边当独立样本 |

建议最少记录：`target_semantics`、完整物理域与参数范围、`nuisance_domain`、噪声/时间假设、训练/验证划分、LB/UB 与对应原假设、残差正规化方式、界容差、未搜索分支、timeout、真图是否在声明域、覆盖、确定率、任务损失、计算与量测成本。

若使用 Monte Carlo 估计 95% 覆盖，应按独立重复实验报告二项区间或适合聚类的区间；“500 次里大约 95%”只是有限仿真证据，不构成对真实数据的分布无关定理。候选漏真图、AC 潮流失败、优化超时和没有有效界都须计入，而非从平均结果中剔除。

### 9.1 两周内最小可投稿性检查

第一阶段先跑完全可控的小问题：固定候选物理边集、4–6 节点、已知误差分布、独立枚举；实现精确集合、LB 外集、错误 incumbent 三个对照。若这一步不能稳定复现理论方向，停止增加算法复杂度。

第二阶段连接当前 R/X atom：在独立多网络上画“计算预算—计算未决”和“探测预算—物理未决”两张曲线；加入一个窃电混淆反例及总表/特征信号解除歧义的案例。

第三阶段才上 AC 与真实数据：保留所有覆盖下降，把它们用于确定需要多大的偏差容忍域。若外集始终大到完全无用，应改为局部区域核查问题，或转向任务等效响应质量论文。

## 10. 文献与可复现检索

以下检索用于复核而非声称已经全库穷尽；非电气统计论文在 Xplore 可能只能查到引用。

| 原始工作/原文 | Scholar 检索 | IEEE Xplore 检索 |
|---|---|---|
| [Universal inference, PNAS 2020, DOI 10.1073/pnas.1922664117](https://par.nsf.gov/servlets/purl/10179988) | [精确题名](https://scholar.google.com/scholar?q=%22Universal+inference%22+Wasserman+Ramdas+Balakrishnan) | [题名与作者](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Universal%20inference%22%20Wasserman) |
| [Game-Theoretic Statistics and Safe Anytime-Valid Inference, Statistical Science 2023](https://safestatistics.com/wp-content/uploads/2023/10/RamdasSAVIStatScience23.pdf) | [精确题名](https://scholar.google.com/scholar?q=%22Game-Theoretic+Statistics+and+Safe+Anytime-Valid+Inference%22) | [精确题名](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Game-Theoretic%20Statistics%20and%20Safe%20Anytime-Valid%20Inference%22) |
| [Robust universal inference for misspecified models, Biometrika 2026, DOI 10.1093/biomet/asaf070](https://academic.oup.com/biomet/article/113/2/asaf070/8321921) | [精确题名](https://scholar.google.com/scholar?q=%22Robust+universal+inference+for+misspecified+models%22) | [精确题名](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Robust%20universal%20inference%20for%20misspecified%20models%22) |
| [Graph Sparsification by Effective Resistances, Spielman–Srivastava](https://www.cs.cmu.edu/~15859n/RelatedWork/Spielman-Strivastava.pdf) | [精确题名](https://scholar.google.com/scholar?q=%22Graph+Sparsification+by+Effective+Resistances%22) | [精确题名](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Graph%20Sparsification%20by%20Effective%20Resistances%22) |
| [Kron Reduction of Graphs with Applications to Electrical Networks, IEEE TCAS-I 2013](https://motion.me.ucsb.edu/pdf/2011d-db.pdf) | [精确题名](https://scholar.google.com/scholar?q=%22Kron+Reduction+of+Graphs+with+Applications+to+Electrical+Networks%22) | [精确题名](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Kron%20Reduction%20of%20Graphs%20with%20Applications%20to%20Electrical%20Networks%22) |
| [Learning Distribution Grid Topologies: A Tutorial, IEEE TSG 2023](https://arxiv.org/abs/2206.10837) | [精确题名](https://scholar.google.com/scholar?q=%22Learning+Distribution+Grid+Topologies%3A+A+Tutorial%22) | [精确题名](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Learning%20Distribution%20Grid%20Topologies%22) |
| [Learning probability distributions over georeferenced distribution grid models, EPSR 2024](https://www.sciencedirect.com/science/article/pii/S0378779624005224) | [精确题名](https://scholar.google.com/scholar?q=%22Learning+probability+distributions+over+georeferenced+distribution+grid+models%22) | [精确题名](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Learning%20probability%20distributions%20over%20georeferenced%20distribution%20grid%20models%22) |
| [From Imperfect Signals to Trustworthy Structure, arXiv:2508.05791](https://arxiv.org/html/2508.05791v1) | [题名](https://scholar.google.com/scholar?q=%22From+Imperfect+Signals+to+Trustworthy+Structure%22) | [题名](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22From%20Imperfect%20Signals%20to%20Trustworthy%20Structure%22) |

用于继续查新、排查“是否已有相同方法”的组合词：

- Scholar：`"distribution topology" "confidence set"`；`"topology identification" "partial identification"`；`"universal inference" "mixed integer"`；`"topology identification" "e-value"`；`"active learning" "confidence set" "power distribution"`；`"electricity theft" "topology uncertainty"`。
- IEEE Xplore Advanced Search：`("All Metadata":"topology identification") AND ("All Metadata":"confidence")`；`("All Metadata":"distribution network") AND ("All Metadata":"hypothesis testing")`；`("All Metadata":"topology") AND ("All Metadata":"optimal experimental design")`；`("All Metadata":"non-technical loss") AND ("All Metadata":"topology")`。
- 谱/任务质量：`"distribution network" "spectral approximation"`；`"effective resistance" "sensor placement"`；`"decision focused" "topology identification"`；`"minimax regret" "network uncertainty"`。

## 11. 现在可以与不可以写在论文里的话

**已有材料支持的表述：**现有 RNJ/L1-wzzT 可提供相容结构搜索的高质量候选与可审计单步求解诊断；理论上可以在明确观测模型下将统计外集、求解界及任务损失连接；不可辨识时保留歧义是正确行为；本文给出了研究协议及条件性推导。

**尚不能声称：**已得到真实 AC 电网物理边的校准概率；当前 core 已实现全域拓扑置信集；单次 MILP dual bound 认证完整网络；95% 图集合覆盖等于任意一条边 95% 存在；低残差排除窃电；超时未发现反例等于不存在反例；在仿真总体上有效的校准能直接迁移真实馈线；主动寻线在未证明的条件下具有全局最优探测成本。

优先做的是一个能被完全枚举小系统验证的诚实外集 oracle，再逐步增加物理复杂性。若这一步成立，线路图质量就能成为窃电定位与最优寻线之间的数学接口，而不只是一张事后着色图。
