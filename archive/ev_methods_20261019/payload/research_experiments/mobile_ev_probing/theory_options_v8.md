# 被动辨识优先、AC 相容性与按需 EV 补证：理论及方案审查

日期：2026-10-12。范围：审查 v7 的数学边界，提出可实施的 v8 路线；本文件不报告尚未执行的数值结果，也不把局部非线性优化当作全局排除证书。以下命题为本项目自行表述和证明，不是引文原定理。文献阅读仅覆盖最后一节列明的部分。

## 1. 直接判断

1. **v7 的局部线性机制合理，但不是完整的 AC 拓扑辨识算法。** 历史共享参数、共同源空间耦合、竞争参数充分重拟合构成一致的问题；端点波形定理只在静态线性、同权、固定能量和简单功率约束下成立。v7 不应被表述为已经解决 AC、未知源位置或生产 WZZT 全流程。
2. **EV 应当只补充已有数据不能提供的结构证据。** 先运行被动 WZZT/候选生成与物理相容性检查，输出首选结构、保留候选和仍有争议的边；只有任务尚未解决才选择动作。被动数据若缺少激励或不同候选本来等价，不能理论保证先获得唯一正确图。
3. **候选排除依赖“不存在任何合法解释”的证据。** 大残差、低排名、错误模型局部求解失败或较小 T 分数都不足以排除。正的、经核验的全局下界可以排除；找到满足原 AC 方程和误差域的点可以证明相容；两者之间是未解决。
4. **选址应针对当前争议结构。** 原 v7 用有限节点/波形库和历史约束 T 分数；这是一种可检验的选择启发式。未实现有意义的选时：静态背景下同一功率多重集合的排列完全等价。AC、时变背景、接入与运行约束引入之后才需要实际选时。

目前最应该修正的是流程与保证接口，而不是叠加更复杂的“置信度”。

## 2. AC 模型最小表达

候选拓扑为有限集合 \(\mathcal H\)，线路参数 \(\theta_h\in\Theta_h\) 在一段辨识窗口内固定。时刻 \(t\) 的 AC 状态为 \(x_t\)（例如全部复电压的实、虚部），实际净有无功注入为 \(z_t\)，未计量源为 \(s_t\)。使用已约定的注入正负号，将物理方程写为

\[
F_h(x_t,z_t,s_t;\theta_h)=0,\qquad
y_t=M(x_t)+e_t. \tag{1}
\]

\(F_h\) 可以是完整单相 AC 节点功率平衡；若采用三相模型，变量和方程同时相应扩展。\(M\) 是实际测量映射，例如所测节点的平方电压幅值。域约束为

\[
\theta_h\in\Theta_h,\quad x_t\in X_{h,t},\quad
z_t\in Z_t(\widehat z_t),\quad s_t\in S_{h,t},\quad e_t\in E_t. \tag{2}
\]

其中 \(Z_t\) 表示 PQ 测量允许误差；\(S_{h,t}\) 表示源的已知位置、幅值及真实空间作用；\(E_t\) 表示电压观测误差。EV 已执行后应使用实际注入及跟踪误差，而不是把命令值默认为实际 PQ。线路参数域、状态域和未测源域必须覆盖拟宣称的候选；人为收紧以获得非平凡界会改变结论对象。

式 (1) 的“共同源”是一个在物理节点进入功率平衡的变量；它在不同电压输出上的影响由 AC 方程共同决定，不能预先替换成若干独立输出扰动。

### 2.1 历史集合、候选集合与单调性

对实际观测索引集 \(D\)，定义

\[
\Theta_h^D=\{\theta_h\in\Theta_h:
\forall t\in D\ \exists x_t,z_t,s_t,e_t\text{ 满足 (1)--(2)}\},
\qquad
\mathcal K_D=\{h\in\mathcal H:\Theta_h^D\ne\varnothing\}. \tag{3}
\]

这里每个竞争拓扑可以重新拟合其自身参数和源轨迹；“源共同”指同一个候选同一时刻各输出来自同一个物理源，不是要求所有竞争候选使用同一个源数值。

**命题 1（不依赖线性的历史保留）。** 若新旧数据属于同一个固定拓扑/参数窗口、所有原约束不变，且时刻间的辅助变量没有跨边界约束，则

\[
\Theta_h^{H\cup N}=\Theta_h^H\cap\Theta_h^N,
\qquad \mathcal K_{H\cup N}\subseteq\mathcal K_H. \tag{4}
\]

证明：联合解释的限制给出两个集合中的同一 \(\theta_h\)。反过来，给定交集中的同一参数，旧、新各自存在的状态、源和误差序列可以拼接成联合解释。候选集合包含关系随即成立。证毕。

若源有能量总量、动态或跨时段平滑约束，须保留完整联合可行集及相应边界状态；只保存 \(\theta_h\) 后简单取交集可能错误。若历史期间拓扑切换，则应先分段或显式引入切换模型；不能用命题 1 把不同时期拼成固定拓扑。

**命题 2（真实候选保留及停止）。** 若真实拓扑 \(h_*\in\mathcal H\)，真实固定参数及每个时刻的状态、注入、源、误差均在声明域内，则 \(h_*\in\mathcal K_D\)。因此：

- 若 \(\mathcal K_D=\{h\}\)，则在这些条件及候选范围内 \(h=h_*\)，不再需要为相同辨识目标派 EV。
- 对任何预先定义的结构函数 \(q(h)\)（例如一条开关边是否存在），若 \(\{q(h):h\in\mathcal K_D\}\) 只有一个值，则该局部结构已经确定，即使全图还不唯一。
- 若 \(\mathcal K_D=\varnothing\)，至少一项覆盖/物理/数据假设失败。应检查模型或扩充候选，不能把最小残差候选称为已确认。

证明：真实过程直接给出式 (3) 的可行见证；其余结论为集合包含的直接推论。证毕。该命题不赋予任何后验概率。即使下一次动作是历史的函数，逐路径论证仍成立；它不需要将动作视为随机独立样本。

若域覆盖仅在一个概率至少 \(1-\alpha\) 的**全程联合事件**上成立，上述全程真实候选保留也只在该事件上成立。每时刻单独 95% 的区间不能直接变成任意次数自适应试验的 95% 全程保证。

### 2.2 排除必须使用下界；相容必须使用原模型见证

为简洁取 \(E_t=\{e:|e_i|\le\epsilon_{it}\}\)，定义归一或同单位的额外松弛

\[
\tau_h(D)=\inf\{\tau\ge0:
F_h=0,\ (\theta_h,x_t,z_t,s_t)\text{ 满足物理域},
\ |y_t-M(x_t)|\le\epsilon_t+\tau\mathbf1\ \forall t\in D\}. \tag{5}
\]

在紧域、连续方程且可行时最小值达到。因此 \(\tau_h=0\) 等价于相容；\(\tau_h>0\) 等价于不相容。若采用多种量纲的残差，须先用固定公开尺度归一，不应把 kW 与 pu 直接相加。

设外松弛的可行域包含 (5) 所有 AC 可行点，且其最优值具有经验证的下界 \(L_h\)；原 AC 可行点给出上界 \(U_h\)。则

\[
L_h\le\tau_h(D)\le U_h. \tag{6}
\]

实际接口应至少输出三种状态：

| 状态 | 所需证据 | 含义 |
|---|---|---|
| 已排除 | 扣除数值验证误差后的 \(L_h>0\) | 声明域内无解释 |
| 已找到相容解释 | 原 AC 方程及原误差约束均经核验的可行见证 | 该候选确实尚不能排除 |
| 未解决 | 无正下界，又无相容见证 | 可能是真实歧义，也可能仅是松弛/求解不足 |

浮点 NLP 残差很小通常是数值见证，不自动构成严格存在性证明；若要使用“严格 AC 可行”，还应使用区间验证/解析可行构造，或明确数值容差及物理误差的保守吸收方式。局部 NLP 的正目标只给上界，不能证明排除。LP 对偶变量有意义的对象是该外松弛的约束敏感性与可验证下界；它不是原 AC/MILP 离散拓扑的概率或完整影子价格。

对固定误差与物理域，增加实际观测还满足 \(\tau_h(H\cup N)\ge\tau_h(H)\)：联合可行点投影到历史仍可行。实现中如每轮重建不同外包络，计算下界未必单调；保存旧有效约束可以避免人为丢失证据。

## 3. 为什么线性共同源投影不能直接当作 AC 理论

v7 使用 \(y_t=R_h p_t+a_h\bar s_t\mathbf1+w_t\mathbf1+e_t\)，所以 \(C\mathbf1=0\) 精确消去源。这依赖“源以固定共同加性方向进入平方电压”的模型结构。

一个不需要三相复杂性的反例已经足够。考虑纯电阻、单位功率因数的 AC 特例：共同上游电压为实数 \(V>0\)，末端 \(i\) 经电阻 \(r_i>0\) 消耗恒功率 \(P_i>0\)。高电压解满足

\[
P_i=\frac{V_i(V-V_i)}{r_i},\qquad
V_i=\frac{V+\sqrt{V^2-4r_iP_i}}2,\qquad V^2>4r_iP_i. \tag{7}
\]

于是

\[
\frac{\partial V_i^2}{\partial V}
=V_i\left(1+\frac{V}{\sqrt{V^2-4r_iP_i}}\right). \tag{8}
\]

只要各支路 \(r_iP_i\) 不同，这些导数一般不同。共同干线处的源使 \(V\) 改变，末端平方电压差也随之改变。源在物理上仍是同一个源，但其 AC 输出效应一般不再是 \(w\mathbf1\)。例如 \(V=1\)、\(r_1P_1=0.01\)、\(r_2P_2=0.04\) 时，式 (8) 分别为 2.0002083116 和 2.0038022951；本次以步长 \(10^{-6}\) 的中心差分核对，误差均小于 \(10^{-10}\)。这只是解析反例的数值自检，不是配网算例效果。近轻载情形这个差异可能小；“小”应由量化误差或实验支持，不能写成“严格为零”。

因此可选两条清楚的路径：

1. 用原 AC 方程进行竞争重拟合和最终核验；线性对比仅用于廉价的动作预筛。
2. 在声明 AC 域内验证 \(y=Rp+Xq+g(s)+r_{\rm AC}\) 的余项集合，并在设计/排除时保留其空间与时间耦合；独立扩大每个输出盒可能重新抹去收益。

无验证的局部雅可比不能代替 AC 外包络。已知网络下可观测性的泛型秩结果，也不是未知拓扑下唯一辨识的证明。

## 4. 三套有先后关系的实施方案

### 方案 A：被动集合先筛选，有限 AC 动作补证（立即优先）

保留生产 WZZT 的被动候选排序；另接一个清楚的物理检查层，不要求 WZZT 本身直接承担概率或全局证书。具体次序为：

1. 历史 PQV → WZZT 首选结构和近邻竞争结构 → 各候选历史参数域及三状态证据。
2. 若首要辨识目标已在保留集合上一致，停止。若没有候选保留，诊断模型；若候选生成覆盖未经证明，输出仅为所列候选内的结论。
3. 仅对仍有争议的候选，按可接入节点、SOC、电量、时段和功率/无功能力建立有限动作库。无 V2G 设备时只用非负有功充电；不能把任意 Q 或放电能力赋给普通 EV。
4. 从历史相容参考出发，以完整 AC 预测与竞争重拟合选择动作；若使用有限参考委员会，权重是设计权重，不是候选概率。
5. 执行后用实际 PQV 追加历史，重新计算 (5)--(6)。继续、停止或报告未解决。

对参考预测 \(\mu_{r,s}(A)\)（\(s\) 是互斥的背景/源场景）可使用

\[
T_{r\to h,s}^{\rm AC}(A)=
\inf_{\theta_h\in\Theta_h^H,\ \eta_h\in\mathcal N_{h,s}(A)}
\|\mu_{r,s}(A)-\mu_h(A;\theta_h,\eta_h)\|_W^2, \tag{9}
\]

其中 \(\eta_h\) 代表候选允许重新拟合的未来源及输入误差，\(W\) 是预先确定的度量。若评价的是“实际只执行一个场景后，错误模型重新拟合还剩多少差异”，就应在每个场景内部重拟合，再平均；让互斥未来共享一个竞争参数点计算的是另一种设计目标，不能混同。已实际发生的历史与动作必须共享线路参数。

式 (9) 是 Dette 式竞争重拟合思路的直接迁移，但 AC 内层非凸。局部 NLP 得到的是 (9) 的上界，会高估可区分性；只能作为未认证的选址分数。若使用外包络求内层下界，应连同松弛紧度报告，不能用一个下界代替“准确最优分数”。最终排除仍由实际数据的 (6) 决定。

**最小算例目标：** 富历史、弱历史、弱结构三组；比较无 EV、按需 EV、始终 EV、固定点、同能量恒充。报告被动首选正确率、候选保留数、自动免 EV 比例、追加后排除数、真实候选误排数、上下界与时间。不要以故意缺乏信息的常数历史代替通常被动数据的全部表现。

**价值/局限：** 路径最短、接口清楚；创新主要来自结构化源与历史重拟合在实际 AC 条件下的收益证据，流程组合本身并不新。

### 方案 B：只辨识有争议的局部结构，测试互补节点（第二优先）

定义目标结构 \(q(h)\)，只对 \(q(h)\ne q(k)\) 的候选对设计输入；相同目标结构之间的参数差异无需耗费 EV 能量。输出

\[
\mathcal Q_D=\{q(h):h\in\overline{\mathcal K}_D\}, \tag{10}
\]

其中 \(\overline{\mathcal K}_D\) 是全部尚未被可靠排除的候选，必须包含所有未解决项。\(|\mathcal Q_D|=1\) 才在声明范围内确认该局部目标；候选银行里缺失真结构会破坏保证。

与单节点最高分相比，值得检验的是两个节点的**互补性**。令固定参考下动作 A、B 的非负连续损失为 \(L_A(\theta),L_B(\theta)\)，竞争参数属于同一个非空紧历史域。每个动作的源变量可独立最小化并吸收入损失，前提是没有被省略的跨动作源约束。则

\[
\min_\theta\{L_A(\theta)+L_B(\theta)\}
\ge \min_\theta L_A(\theta)+\min_\theta L_B(\theta). \tag{11}
\]

**命题 3（单点均无分离，联合仍可能分离）。** 若两项单独最小值均为零，则联合最小值严格大于零，当且仅当

\[
\{\theta:L_A(\theta)=0\}\cap\{\theta:L_B(\theta)=0\}=\varnothing. \tag{12}
\]

证明：式 (11) 来自逐点下界。交集非空时同一参数使和为零。交集为空时和处处为正；由连续性与紧性，其达到的最小值严格为正。证毕。

直觉：每个地点单独都能被错误拓扑用某套参数解释，但两套解释可能彼此矛盾；线路参数不能随 EV 地点重置。因此移动到第二个地点的意义可以是消除参数重拟合的自由度，而不仅是把激励做得更大。

该命题是基本集合性质，不是新普遍定理；其研究价值在于能否对具体拓扑争议找到可解释且低成本的互补节点。式 (11) 并未保证拆分同电量预算优于集中单点，必须用相同能量、总时间和出行成本比较。移动 EV 的两个地点是先后观测，模型须确认拓扑/参数在中间保持不变。

可用局部残差对源/参数雅可比做快速评分。若局部重拟合允许 \(\delta\beta\) 无约束，\(r\approx d-J\delta\beta\)，则

\[
\min_{\delta\beta}\|d-J\delta\beta\|_W^2
=\|(I-P_{W^{1/2}J})W^{1/2}d\|_2^2. \tag{13}
\]

\(P\) 为正交投影；这是最小二乘消去干扰方向的直接结果。它关注特定候选的结构差异 \(d\)，无需全方向最小奇异值。源/参数边界活跃时应使用有界最小二乘；AC 状态变动使 \(J\) 改变，式 (13) 只是局部选址近似，不能支持全域排除。

### 方案 C：先区分“证据不足”和“计算不足”，再决定派车（第三优先）

若候选的 \(L_h=0<U_h\)，尚不能确定这是物理上真的可混淆，还是 AC 松弛过宽/非线性求解未找到可行点。派 EV 之前允许执行较便宜的计算动作：保留旧 cuts、对阻抗盒分支、收紧电流/电压域、加强耦合包络。另一个动作才是采集新 PQV。

这形成一个可解释的资源选择：**增加数据，或改善对已有数据的数学验证。** 不宜现在就加上一个未经校准的“预期信息收益”复杂随机优化器；先使用固定计算预算、明确 gap 门槛的规则并做消融。

如果同一观测上的有效嵌套外松弛为 \(\mathcal R_{k+1}\subseteq\mathcal R_k\)，且始终包含原 AC 可行域，则它们的最小目标满足 \(\ell_{k+1}\ge\ell_k\)，依然是原问题下界。求解器返回的近似界不一定严格单调，但可保留历次有效下界的最大值。有限次收紧不保证得到正界。

**检验方式：** 在同一个被动未决实例上比较固定时间收紧、一次 EV、先收紧后 EV；计入求解时间、派车数、能量、最终可核验排除数与未解决数。若收紧即可确认，EV 无需执行；若存在两个经验证的 AC 相容解释，新增有针对性观测才有明确物理意义。

**创新边界：** AC 外包络、模型否证、主动诊断和计算/采集权衡均已有基础。可研究的是本任务里识别两类瓶颈并降低无效移动探测的具体算法与证据，不能未经定向全面检索声称“首次”。

## 5. 如何组织一篇强关联文献/报告

主线只需要一个问题：**已有 PQV 留下的局部拓扑歧义，能否由少量可执行 EV 观测消除，同时保留真实候选且计入 AC 与源误差？**

论证次序为：被动表现及未决结构 → AC/源/观测域 → 相容性和三状态接口 → 按需动作设计 → 实際联合检验 → 免 EV、成功补证、仍不可分、计算不足四类结果。把端点定理留作理想条件下的简化原则，避免使其统领 AC/时变算例。

最有价值的下一步不是为每个模块另造“置信度”，而是让设计目标和最终证据对象接近：针对当前争议边/拓扑，用实际 AC 竞争重拟合测量动作效果，输出数据确实支持到哪一层。

## 6. 原始文献核查与实际阅读范围

本次定向检索覆盖：被动/主动配网拓扑、有限监测的局部结构、AC probing 与非线性模型区分。以下仅是与三套方案直接相关的原始来源，不声称穷尽。已主动考虑 HKU Library；本次所需正文均有开放作者稿，无需进入订阅会话，也没有新增下载文件。

| 来源 | 本次实际读到的部分 | 对本项目的准确作用 |
|---|---|---|
| Dette & Titoff, *Optimal discrimination designs*, AoS 37(4), 2009. [DOI](https://doi.org/10.1214/08-AOS635), [开放稿](https://arxiv.org/pdf/0908.1912). | PDF pp.3--5，§2式 (2.2)--(2.7)，尤其固定参考与竞争重拟合、局部备择及 sequential/Bayesian/maximin 说明。 | 方案 A 的设计基础。原文已包含历史/顺序思路的背景；不能把改成 AC 预测后直接继承其统计功效解释。 |
| Cavraro & Kekatos, *Graph Algorithms for Topology Identification using Power Grid Probing*, IEEE CSL, 2018. [作者稿](https://arxiv.org/pdf/1803.04506). | 本次打开正文；方法详细范围沿用 v7 已核对的§II--VI，v8 未新增读取其全部证明。 | 已有主动拓扑与 reduced topology 基础。v8 只将其作为已核对背景，不据本次打开动作声称重新阅读全文。 |
| Cavraro, Bernstein, Kekatos & Zhang, *Real-Time Identifiability of Power Distribution Network Topologies with Limited Monitoring*, IEEE CSL 4(2):325--330, 2020. [DOI](https://doi.org/10.1109/LCSYS.2019.2926101), [作者正文](https://engineering.purdue.edu/~kekatos/papers/L-CSS-2019.pdf). | PDF pp.1--3，§II--IV开头：线性化式 (1)，历史功率协方差及 Assumption 1，候选目标式 (11)，观测岛定义；未完整核对后续证明。 | 表明针对有限监测/局部连接设计传感位置已有工作。其理想可识别性使用统计、线性化和已知线路信息，不能替代本项目 AC 集合证书。 |
| Bhela, Kekatos & Veeramachaneni, *Smart Inverter Grid Probing for Learning Loads: Part I—Identifiability Analysis*, TPWRS, 2019. [开放稿](https://arxiv.org/pdf/1806.08834). | PDF pp.1--4，§II：AC 二次方程 (1)、跨时段负荷不变式 (4)、ZIP讨论、Remarks 1--4 与 Definition 1；未核对全部可观测性证明。 | AC probing 有直接理论基础，但作者目标是在已知网络下推断未计量负荷。秒级恒负荷、已知拓扑和忽略调压动作不能无条件移到6小时 EV 场景。 |
| Bhela et al., *Part II—Probing Injection Design*, TPWRS, 2019. [DOI](https://doi.org/10.1109/TPWRS.2019.2906306), [开放稿](https://arxiv.org/pdf/1806.08836). | PDF pp.1--3，§II设备/注入库、安全筛选与未知负荷范围；阅读Jacobian病态和状态多样性说明。 | 支持先组织可执行动作再选择实验。其负荷估计/状态多样性目标不同于局部候选区分，不能把全参数条件数直接当作我们的目标。 |
| Scott, Findeisen, Braatz & Raimondo, *Input design for guaranteed fault diagnosis using zonotopes*, Automatica 50:1580--1589, 2014. [DOI](https://doi.org/10.1016/j.automatica.2014.03.016), [作者稿](https://web.mit.edu/braatzgroup/input_design_for_guaranteed_fault_diagnosis_using_zonotopes.pdf). | 正文PDF pp.3--4，式 (14)--(20)、Definition 1、Theorem 3：耦合输出集、观察后相容与事前两两集分离。 | 对比“观察后排除”和“任意扰动下事前保证”不同量词。作者也明确：非全分离动作仍可能在实际输出上完成诊断。因此当前不必强求全域鲁棒分离才允许试验。 |
| Kanishka Singh, Yuhao Ding, Necmiye Ozay & Sze Zheng Yong, *Input Design for Nonlinear Model Discrimination via Affine Abstraction*, IFAC-PapersOnLine 51(16):175--180, 2018. [DOI](https://doi.org/10.1016/j.ifacol.2018.08.030), [作者稿](https://web.eecs.umich.edu/~necmiye/pubs/SinghDOY_adhs18.pdf). | PDF pp.1,3--4，Problem 1.2式 (17)、外近似保留性质、§4 Theorem 1 的假设及 §5 Theorem 2。 | 非线性模型的外近似+可区分输入早已有研究。可迁移的是集合包含逻辑；其仿射抽象计算依赖额外条件，不能把 KKT 或一个局部最优点自动解释为 AC 全域包络。 |

补充现场可行性线索：Huang, Burroughs, Klauber & Sun, *Power Distribution System Characterization with Active Probing: Real-World Testing and Analysis*, IEEE OAJPE 10:86--92, 2023，DOI [10.1109/OAJPE.2022.3227908](https://doi.org/10.1109/OAJPE.2022.3227908)。本次只核对出版社作者接受稿的摘要搜索内容，**没有读取实验全文**；摘要报告 GRP 主动扰动与 µPMU 的实际配网测试。它支持探测设备的工程研究背景，但不能证明普通 EV、15分钟采样或当前候选算法在现场有效。

## 7. 对当前算例实现的核查清单

- 先报告被动结果，再报告 EV 的增量；富历史主动免 EV 是正确行为，不算算法失败。
- 区分 AC 数据生成、AC 参数重拟合、AC 外包络覆盖三件事；只用 AC 仿真产生数据不代表算法具有 AC 保证。
- 每个阶段共享实际窗口的线路参数；输出箱、重置历史参数作为消融，不能与主模型混淆。
- 源轨迹可变但必须按物理位置共享；共同源投影在 AC 下的泄漏需量化。
- 只根据历史选择参考/候选/动作；生成真值只用于评价。
- 用相同时间、电量、设备能力与测量数量比较节点/曲线；双地点必须计入出行或明确只做信息机制测试。
- 报告排除下界、原模型见证及未解决，保存弱边、错误参考、大误差、未知源位置和求解失败。
- 先验证方案 A 的完整流程，再用方案 B 的互补地点建立更强中心结果；方案 C 用来防止把松弛不足误当作必须增加物理实验。

## 8. 与本 checkout 实际 WZZT/RNJ 的接入位置（只读代码核查）

本节核对的是 `C:\Users\23761\.codex\worktrees\36a1\Topo` 中的源文件；没有运行或修改生产算法，也不把其他 checkout 的版本视为相同。v8 三候选原型不是下表整个流程的实现。

### 8.1 已有接口与不能混淆的语义

| 文件/函数 | 本 checkout 实际行为 | 建议接法及限制 |
|---|---|---|
| `rnj_wzzt_core/rnj_wzzt/estimation/multiscenario.py::align_scenarios`、`fit_projected_sensitivity` | 对齐带终端标签和时刻索引的 `P_terminal/Q_terminal/drop_target`；原始根参考平方电压降直接堆叠，拟合有序对称非负 R/X，返回 `(R, X, fit_r2, condition)`。 | 用于被动候选生成。`fit_r2` 与 `[P Q]` 条件数是拟合诊断，不能当作图正确率或 EV 效用。当前源码不去均值；旧文档的 `daily_demean` 叙述不代表本版本。 |
| `rnj_wzzt_core/rnj_wzzt/pipeline.py::_rx75_rnj_clades`、`_select_boundary_blocks` | 用 RX75 共享路径和根深度产生 RNJ；时间块 bootstrap 计算边界 clade 重现率，再取互不交的 top-k。 | 可取得被动首选结构与排序线索；名称为 `confidence` 的值实际是 bootstrap 重现频率。其私有接口不宜直接扩成长期公共 API。 |
| 同文件 `_fit_milp_variants`（约319--337行）；`estimation/laminar_l1_milp.py::fit_laminar_l1_sensitivity` | `selected_supports` 被传入 `initial_supports`；后者明确作为**固定结构块**保留，虽然边权重新拟合。每步完整求解的是当前 laminar 家族的一次新增原子问题，整体仍是前向搜索。 | 不能把这个固定块流程直接称为“EV 可以纠正全部 RNJ 错误”。未有外部结构证明的 RNJ 块应只作为候选提议；确认流程需保留不包含该块的竞争候选或从 singleton 重启。一步 MILP 最优不等于全部拓扑全局最优。 |
| `rnj_wzzt_core/rnj_wzzt/pipeline.py::run` | 通过 `_prepare_case` 模拟指定标准算例，再训练、验证及输出 metrics。 | 这是算例运行器，不是接收现场历史数组的入口；拟议 adapter 应直接接收观测并调用估计函数，不能让 `run` 重新生成数据。 |
| `research_experiments/mobile_ev_probing/wzzt_adapter.py::fit_source_aware_wzzt` | 从 singleton 和调用方传入的 `certified_supports` 重启；每次重新选择完整源路径，保存搜索路径和每次求解诊断。 | 最接近研究接入点，但 `certified_supports` 的有效性由调用方负责，函数不认证它们；未认证 RNJ 结果不能填入此字段。 |
| 同文件 `fit_fixed_source_supports` | 给定一个支持族，重新拟合全部 R/X 并选择源路径；求解未经最优性检查不返回接受解。返回 `SourceFitSolution`，包括 `support_indices/r_matrix/x_matrix/h/g_r/g_x/train_mae/adjusted_train_mae/diagnostics`。 | 用于竞争候选同口径重拟合和排名；失败要保留为未解决。其 `source_p_error/source_q_error` 是按输出放松的幅值外包络，不是 v7 精确共同源集合，也不是 AC 证书。 |
| `terminal_case33/pipeline/rnj_candidates.py::generate_rnj_candidates` | 在4种 R/X 度量、6种 RNJ 容差上构造不同树，按 rooted clade 集去重；`rooted_path_incidence` 生成根定向路径矩阵。 | 已有低成本候选扩展方法，可在研究层复用或薄包装；它是有限候选网格，不覆盖全拓扑。不同参数化的物理候选不能仅凭等价 clade 就无条件合并证书域。 |
| `research_experiments/mobile_ev_probing/observation_study_v5.py::candidate_bank` | 对资产参考图枚举不同父节点的两末端接线交换；`topology_key` 计算 clade key。遇到可能等价的物理成员会拒绝静默去重。 | 有明确资产图时，这是实际可用的局部竞争域模板；不读取真实边权来评分，但资产参考本身是先验结构。它不是任意未知隐节点树的完整候选生成器。 |
| `research_experiments/mobile_ev_probing/observation_lp_v5.py::solve_observation_lp` | 接收候选物理图、原观测、`ObservationDomainV5/ObservationErrorBoxV5`，计算 AC 外 LP 的额外电压松弛；未知源位置被按输出外放松。 | 可作为独立 AC 排除层；零界/失败均未解决。其未知源外放松可能保守，不能宣称已保留 v7 的全部共同源收益。 |

### 8.2 最小新 adapter 合同（建议，尚未实现）

拟新增研究层 `passive_probe_adapter`，而非修改生产 `pipeline.run`。一个入口接收：

```text
prepare_passive_state(
    observed_scenarios, terminal_labels, root_label,
    engineering_parameter_domain, observation_error_model,
    source_model, asset_candidate_bank=None, candidate_settings=...
) -> PassiveProbeState
```

其中原观测必须保留根电压、终端 PQV、主表 PQ（如源模型使用）、单位和时间索引；WZZT 使用的 `drop_target=V_root^2-V_terminal^2` 由这些观测生成。`source_model` 应明确是 v7 已知共同源、v5 未知位置外包络，还是未来完整 AC 源模型。不能根据求解便利隐式切换。

返回对象至少分为以下字段，禁止只输出一个“confidence”：

```text
passive_best_id, passive_rank_scores
candidates[{id, provenance, canonical_clades, physical_graph_or_family,
            parameter_domain_id, fitted_parameters, source_fit,
            train_score, heldout_score, solver_diagnostics}]
evidence[{id, status, valid_lower_bound, ac_witness,
          numerical_tolerance, model_domain_id, observation_ids}]
retained_candidate_ids, disputed_clades_or_edges
scope_description, candidate_coverage_proved
stop_reason, next_action_required
```

建议后续入口为 `choose_probe(state, action_library, forecast_scenarios)` 和 `append_observations_and_refit(state, actual_observations)`。选择器只接收前一个状态及可获得预测，不接收生成真值；每次追加实际观测重新拟合共享参数，并保留全部未解决候选。最终唯一性结论附 `scope_description`：例如“给定资产图所有单次末端交换”，不能写成全图无其他解释。

### 8.3 怎样扩展竞争候选而不永久固定 RNJ

第一批候选可取：原始完整 RNJ、R/X-容差网格的不同树、WZZT 前向路径的若干家族，以及合法资产接线交换。每个候选用相同历史拟合/验证口径重新估计，保留出处。对于最有争议的 RNJ 块，再加入删除该块、改变局部合并、或对应叶/子树换接的合法 laminar 候选；这些是拟新增的局部动作，不是当前代码已有的全域搜索证明。

“保留排名较差候选”与“认证它不能成立”是两件事：为节约设计计算可以只给部分候选评分，但所有没有可靠排除的候选应留在结果范围中。候选未被纳入动作设计不能随后当作已排除。只把有独立结构证明的块传入 `certified_supports`；对其他 RNJ 块应允许删除或替换。

若候选只是约化 clade 家族，转入 AC 层还需显式构造物理图、隐藏节点、各边参数域和允许内部注入策略。`R/X` 矩阵或 `support_indices` 本身不足以自动确定这些 AC 假设。已有资产图时优先用资产图及局部换接，接口更短也更易核验。

### 8.4 已有被动结果能支持什么

本次只读取已有文件，没有重跑或匹配完整源码/输入哈希，因此以下是**历史报告记录**，不作为本轮当前版本复现成功：

- `docs/rx75_rnj_milp_hybrid_results_table.tex` 记录3场景×96点含噪 AC 数据：4个算例完整 RNJ clade F1 为1、1、1、0.9333，有限候选联合结果均为1。它说明“被动先给较好结构，EV补局部证据”有历史动机；不能证明现场或全参数域可靠。
- `rnj_wzzt_core/docs/matched_rnj_tuning_20260910.md` 记录较广压力测试中默认 RNJ 的36个 AC 条件平均F1为0.723805802；验证MAE选阈值后为0.668703634，同时预测MAE改善。这个负结果说明预测拟合和拓扑正确不是同一目标，不能凭较低MAE直接确认图。
- `docs/core_module_code_20260919/evidence/smoke/metrics.json` 是仅8点、bootstrap两次的旧版本 smoke；其配置含 `daily_demean/contract_blocks/candidate_pool_mode`，与本 checkout 当前 `pipeline.run` 不一致。不能把该文件当作当前主线效果，也不能把其中F1=0外推为主算法普遍失败。

正式接入前应新增当前版本的被动基线复验，保存配置、源码hash、观测、候选覆盖与结构指标，再评价EV增量。v8 三候选测试只证明补证机制和控制流程，不能替代这一步。
