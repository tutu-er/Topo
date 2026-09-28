# 窃电分线：从未计量负荷辨识到面向定位任务的图质量与增补测量

研究日期：2026-09-27。本文结合当日代码只读审计、历史结果文件核验和已打开的原始文献。新方向均为研究设计，尚未成为完成的算法或实验结论；没有修改原 MILP、校准、测试、缓存或既有结果。

## 1. 最值得推进的论文问题

建议把主要问题收敛为：**在配电网结构和阻抗也有误差时，如何辨识未计量正负荷的可定位区域，并用同一个辨识边界决定下一处应增补什么测量？**

这个问题自然连接三条线：线路图质量决定隐蔽接入点之间是否可分；窃电模型将这种可分性转化为相容区域；主动寻线或测量增补专门消除阻碍定位的图结构歧义。论文的闭环是“辨识—承认不确定—选择有价值的观测—缩小不确定”，而不是把三个独立模块拼成流水线。

电气量首先识别的是**未计量负荷**。若合法未登记负荷和非法接入在同一处产生相同 P/Q 轨迹、连接和噪声规律，二者的电气观测分布相同；任何仅依赖这些电气观测的算法都无法判断其法律属性。实务中的“窃电”标签还需要台账核验或现场证据。这个边界可以成为问题定义的一部分，不妨碍做高质量的区域定位论文。

最有希望的增量并不是“电压+总表+MILP”，而是**在图误差与负荷异常可能互相解释时，给出可检验的辨识条件、可拒答的区域输出以及有成本约束的下一步观测建议**。

## 2. 当前已有资产：比早期 pilot 成熟，但范围仍有限

下表均以当日文件为准；旧记忆中的部分实现状态已过时。

| 已有内容 | 本轮核验与可复用点 | 不能据此宣称 |
|---|---|---|
| 独立 theft 包的逐时单源 MILP | [theft_model.py](D:/0-github_workspace/Topo/theft_wzzt/theft_wzzt/theft/theft_model.py:120)，固定外部幅值、二元位置与有界 R/X 乘积 | 连续自由幅值与 R/X 同时估计仍是同一个 MILP |
| 固定窗口位置的逐候选 LP | [regularized.py](D:/0-github_workspace/Topo/theft_wzzt/paper_study_v2/regularized.py:13)，H0 和每个位置共享收缩先验，分开记录电压、平衡、先验增益 | 等价求解原逐时移动位置 MILP；收缩一定改善定位 |
| measured-only 的保守推断入口 | [detect_conservative.py](D:/0-github_workspace/Topo/theft_wzzt/detect_conservative.py:19)，检查测量/树协议、源文件哈希、strength=10、199 样本最大分数规则 | 任意时间窗、任意漂移、条件于已告警的区域覆盖或现场窃电概率 |
| V3 冻结研究 | [summary.json](D:/0-github_workspace/Topo/theft_wzzt/outputs/theft_paper_v3/summary.json:1) 明确 expected=completed=3171、errors=[]；[run_study.py](D:/0-github_workspace/Topo/theft_wzzt/paper_study_v3/run_study.py:27) 定义角色与种子 | 本轮重新执行过 3171 个场景；三个自建网络就是标准 IEEE 复现 |
| 65037 三相公共网络试验 | [effect_assessment_20260920.json](D:/0-github_workspace/Topo/theft_wzzt/outputs/theft_real65037/effect_assessment_20260920.json:1) 核验 320 次重放，既有最优集合和幅值 MAE 全部匹配 | 320 个独立现场窃电标签；53 母线都已测过定位 |
| 矩形 observed-to-hidden 灵敏度接口 | 当前 [lin_distflow.py](D:/0-github_workspace/Topo/rnj_wzzt_core/rnj_wzzt/models/lin_distflow.py:18) 真的使用 injection_terminals，并按观测/注入两集合构造矩形 R/X | 旧说明“参数被丢弃”继续适用；整个 production pipeline 已支持隐藏负荷 |
| core 场景验证 | [validate_scenario.py](D:/0-github_workspace/Topo/rnj_wzzt_core/rnj_wzzt/scenario/validate_scenario.py:39) 仍将 hidden_internal 带负荷判违规 | 原 topology suite 已验证隐藏负荷场景 |
| 同时多源 | [theft_simulation.py](D:/0-github_workspace/Topo/theft_wzzt/theft_wzzt/theft/theft_simulation.py:156) 拒绝同一时刻多位置为正；[已有多源分析](D:/0-github_workspace/Topo/docs/theft_multisource_wzzt_research_assessment_20260920.md:81) 已做固定字典代数歧义研究 | 已实现多源 AC 求解、校准或恢复试验 |

V3 在三个合成网络上的随机事件点定位为 88/100、73/100、76/100，区域集合包含真区域均为 100/100，正常误报为 0/100、0/100、1/100；这些是预设混合事件分布的测试频数，有有限样本误差。既有 [readiness 文档](D:/0-github_workspace/Topo/docs/theft_paper_readiness_20260919.md:25) 给出相应区间、区域大小和幅值覆盖，不应只摘“100%”。三棵训练得到的约简树恰好都正确，因此这批数据不能回答错误拓扑如何影响窃电定位。

65037 的固定历史参数、既有 z 选择共 280/320 唯一选中真实约化区域，317/320 的最小目标集合包含真区域，320/320 的前三排序包含真区域；这只是已有两处位置、两个历史模型的配对仿真结果。更值得作为新论文起点的失败是：不平衡母线 5 的某历史模型把通往电表 39 的相参数估成零，导致两个位置响应退化；它属于**估计模型的退化**，不能直接叫物理不可辨识。具体见 [效果分析](D:/0-github_workspace/Topo/docs/theft_real_feeder_effect_assessment_20260920.md:29)。

联合优化证据尤其要谨慎：本轮直接核验 JSON 中 H1_theft 为 total=24、optimal=0、incomplete=24；历史拟合 4/4、H0 36/36 最优不能冲淡这个事实。应将“有限时间求解器找不到更好可行点”和“数学模型即使求到最优也不可辨识”分开研究。

当日 core 与独立 theft 包有各自演进。`theft_wzzt/theft_wzzt/theft/identified_tree.py` 仍显式调用自己的候选池构造；不能把 core 最近的自由相容支撑扩展状态自动套到独立包。后续论文需要冻结两个版本的接口、哈希与候选定义。

## 3. 查新后的边界：哪些故事已经有人讲过

这里的文献只用于界定研究问题，不把阅读摘要当成数值复现。

| 原始工作及本轮阅读层级 | 已有内容 | 对本项目的约束 |
|---|---|---|
| Weckx 等，ISGT Europe 2012，[作者机构全文](https://lirias.kuleuven.be/bitstream/123456789/366219/1/Parameter%20%2BIdentification%2Bof%2BUnknown%2BRadial%2BGrids%2Bfor%2BTheft%2Bdetection.pdf)，已打开摘要与定位章节 | 从智能电表估计未知/不确定径向网的线性模型，并用于绕表定位 | “先辨识未知网络再查窃电”本身不是新问题 |
| Pengwah、Razzaghi、Andrew，IEEE TPWRD 2023，[Monash 官方摘要与书目](https://research.monash.edu/en/publications/model-less-non-technical-loss-detection-using-smart-meter-data/)，DOI 10.1109/TPWRD.2023.3280551 | WLS 电压灵敏度、实际用电估计和阈值告警，采用 European LV 及随机网络 | “无需详尽线路模型”“电压灵敏度发现窃电”已有直接邻近工作 |
| Carmona-Pardo 等，PSCC 2026，[官方全文](https://pscc.epfl.ch/modules/request.php?module=oc_program&action=view.php&id=34&file=1/34.pdf)，已打开 III–V 节 | 已知拓扑/导线/相别，联合表计与首端测量，三相四线灵敏度和凸 MIQP，定位可见/不可见接入 | 物理模型、隐藏接入、整数选择和优化证书都不足以单独构成创新 |
| Leite、Mantovani，IEEE TSG 2018，[UNESP 原始机构记录](https://repositorio.unesp.br/entities/publication/653fbca8-b698-47f8-a46b-ecb120705df1)，已读官方摘要 | 多变量监测后用 A-star 路径搜索定位异常消费点 | “检测后再寻路”也有先例；我们的寻线应是新信息获取的设计问题 |
| Bhela、Kekatos、Veeramachaneni，IEEE TPWRS 2019 Part I，[作者全文](https://arxiv.org/html/1806.08834v2)，已读模型、恒定负荷条件与 ZIP 小节 | 已知网络下，用多时刻潮流耦合和图匹配检验非计量负荷局部可观性 | 不能把已有一般秩/匹配条件直接包装成新理论 |
| 同作者 Part II，[作者预印本](https://arxiv.org/abs/1806.08836)，已读原始摘要 | 在逆变器/电压约束下设计 probing，考虑未知负荷与噪声 | 新意需要落实到离散图/位置假设、干扰因素和实际定位成本 |

PSCC 论文采用真实网络参数上的生成事件；正文 III 节明确已知拓扑、导线类型、相别，V 节明确 generated fraud scenarios。不要将其摘要中的 real-world data 误读为有现场窃电真值。其模型对相间耦合、中性线和多相供电的刻画比当前标量主线更完整，因此必须作为主要对照之一。

搜索还发现同题 EPSR 条目 DOI `10.1016/j.epsr.2026.113599`；本轮直接打开出版商页失败，搜索摘要把卷期标为 2027-02，因此本文以已核实的 PSCC 2026 原文论断为准，不依据该索引记录追加“新期刊版有何改进”的结论。

## 4. 方向 A：对定位有意义的线路图质量，而不是单一拟合分数

### 4.1 核心模型

令 O 为观测端口集合，H 为未计量负荷候选，T 为窗口长度，m=|O|。用根电压减去端口电压构造正压降。在固定线性模型下，已计量功率与隐蔽正负荷满足

\[
y_t=R_{OO}p_t+X_{OO}q_t+g_h a_t+B_t\beta+\varepsilon_t,
\qquad g_h=R_{Oh}+\kappa X_{Oh},\quad a_t\ge0.
\]

这里 \(\kappa=Q/P\) 是暂时固定的模型条件，\(B_t\beta\) 表示参考电压偏差、合法未计量负荷的已声明结构等干扰；不应任意增大 B，否则窃电也会被消掉。若将 squared-voltage 作为观测，应统一 R/X 中的因子 2。

总表平衡必须保留有符号量：

\[
m_t=P_{0,t}^{m}-\sum_i P_{i,t}^{m}=\ell_t+u_t+a_t+\eta_t.
\]

\(u_t\) 是合法未计量项；\(\widehat a_t=\max(m_t-\widehat\ell_t,0)\) 只是 H1 的估计输入。由于 y、m 和估计功率可能共用表计误差，不能把“总表支持”和“电压支持”当成独立证据相乘。

### 4.2 可立即推导的辨识命题

在一棵固定候选树附近，将 R/X 小变化写成局部线性干扰 \(J\delta\theta\)，堆叠窗口并暂时固定幅值轨迹 a，得到

\[
r=d_h+A_h\nu_h+\varepsilon,\qquad d_h=a\otimes g_h,
\]

其中 \(A_h=[J_h\ B]\)。联合 R/X 重估时，隐藏注入经过参数化响应，通常使 Jacobian 依赖 h，不能不加说明地假定所有候选共享 J。若 \(\Sigma\succ0\) 是声明的误差协方差，令 \(W=\Sigma^{-1/2}\)、\(M_{hk}=[WA_h\ WA_k]\)、\(P_{hk}=I-M_{hk}M_{hk}^{\dagger}\)。

**命题 A1（固定线性模型、无界线性干扰的精确版本）。** 两个候选 h、k 可由不同干扰参数产生同一无噪声观测，当且仅当

\[
P_{hk}W(d_h-d_k)=0.
\]

证明只有一行：等价于 \(d_h-d_k\in\operatorname{col}([A_h\ A_k])\)，从而存在两组干扰参数使两种预测相等。若 \(A_h=A_k=A\)，就退化为共同投影 \(I-(WA)(WA)^\dagger\)。它解释了为什么给每条线路更自由的 R/X 变量可能降低最优目标却削弱位置识别。该投影是已知线性代数工具，研究贡献应来自树结构下的可解释条件与真实量测约束。

**适用边界。** 原参数模型非线性时，J 版本只是局部一阶诊断；有非负/上下界时，无界投影可能过分悲观。此时应计算受限距离

\[
\delta_{hk}=\inf_{\nu_h\in\mathcal N_h,\nu_k\in\mathcal N_k}
\left\|W\{d_h+A_h\nu_h-d_k-A_k\nu_k\}\right\|_2.
\]

幅值也未知时，须把允许的非负幅值轨迹或总表一致区间放进两个预测集合；若两个集合都允许零幅值，其最小距离必为零，所以应先固定检测能力要求（例如窗口能量至少 E_min），不能把这个零误判为所有正事件都不可定位。

**命题 A2（有界误差下的操作解释）。** 若两个候选预测集合距离大于 \(2\epsilon\)，且白化后误差范数不超过 \(\epsilon\)，则两者的误差扩张集合不相交。这个结果可以用来设计“足以区分”的测量，而不是凭分数差越大越好。\(\epsilon\) 若来自统计校准，还需单独说明其覆盖条件。

### 4.3 怎样定义图质量

将每张图的质量至少拆为三项：正常窗口预测误差；异常位置之间在干扰消除后的分离能力；真位置是否落在候选域中。只报告正常 R² 或 MAE 会漏掉“对正常数据很准，却把两个重要区域变成同一响应”的图。

可给每张候选图一个任务诊断表：

- 哪些候选区域完全等效，哪些是参数边界导致的近等效；
- 在指定幅值/持续时间下，重要位置对的 \(\delta_{hk}\) 与噪声半径的比值；
- 不同图给出一致结论的区域与图依赖区域；
- 改错哪条支撑后能显著增加最弱的定位分离度。

此处“近”等效关系通常不满足传递性，不能简单按小于某阈值的两两距离合并后声称得到数学等价类。零距离的预测集合重叠也未必形成等价关系；应报告图、相容集合或明确的商空间条件。

### 4.4 论文级可证伪实验

1. 人工构造两个正常 MAE 几乎相同的树，一个保留关键弱支路、一个把其权重压成零，检验定位集合是否不同；直接复现 65037 母线 5 机制。
2. 在相同参数数目或相同预测误差下，对比固定参数、共同倍率、逐边自由重估、收缩模型。核心结果不是谁的训练误差更低，而是谁在同样误报预算下给出更小且覆盖可靠的区域。
3. 各种错误单独注入：漏一个支撑、错一个挂接、零化弱边、根电压偏移、未知 Q/P、合法未登记负荷。保留真实源不在候选域中的试验，并允许输出“域外/模型不适配”。
4. 预先冻结分离度指标，预测哪组会失败，再用独立 AC 场景验证。若分离度对失败不具有解释或预测能力，方向 A 的主要主张就被削弱。

**优先级：最高。** 最多复用既有代码，能直接解释已有负面结果，也最容易把图质量和窃电形成同一篇论文。

## 5. 方向 B：从相容区域反推“下一次应该测哪里”

### 5.1 从一个解转为联合解释集合

保留 \(\mathcal C=\{(G,h,\theta,a):\text{观测与误差模型相容}\}\)，其图 G 可以不同。位置报告取该集合在物理可映射区域上的投影；不同图的任意隐藏节点编号不能直接求并集。已有 `region_label` 是真值评估映射，不是现场已知的隐藏物理地址，见 [identified_tree.py](D:/0-github_workspace/Topo/theft_wzzt/theft_wzzt/theft/identified_tree.py:133)。

如果 \(\mathcal G\) 只是候选内近似，则整个集合方法也条件于真解释在域内。简单合并多张图的候选位置可能变得更保守，但不自动获得全局覆盖率；若每张图都用同一数据拟合、筛选和校准，更不能沿用单图阈值。

### 5.2 观测设计的最小形式

动作 s 可以是临时内部电压测点、某支路电流/功率测点、现场核验一段连接，或者逆变器探测；每种动作需要自己的观测方程与成本。对两种仍相容且现场处置不同的解释 c、c'，计算新增观测的预测差 \(\Delta_s(c,c')\)，并扣除其误差/扰动不确定性。

在第一版中不必使用未经验证的概率先验。可按预算选择测点，使最难分的关键解释对也尽量分开：

\[
\max_{S:\sum_{s\in S}cost_s\le B}\quad
\min_{(c,c')\in\mathcal P}\sum_{s\in S}
\frac{\Delta_s(c,c')^2}{\sigma_s^2}.
\]

该式只在新增噪声可独立标准化、每个解释给出固定预测时成立；相关误差应使用整体 Mahalanobis 距离，参数不确定时内层还要对允许参数求最坏分离。也可把目标改成减少现场核验的最坏剩余工作量。互信息需要概率模型；候选频数或 bootstrap 频率不是可直接代入的真实先验。

小规模候选时可以枚举动作集合建立独立 oracle；大规模再考虑贪心、MILP 或启发式。上式的 max-min 目标不自动具有次模性，不能无条件套用贪心 1-1/e 保证。

### 5.3 必须保留的主动探测反例

固定线性灵敏度、同一个隐藏负荷在探测前后不变时，

\[
y^{(s)}=R_{OO}(p+u_s)+X_{OO}(q+v_s)+g_h a,
\quad
y^{(s)}-y^{(0)}=R_{OO}u_s+X_{OO}v_s.
\]

差分中 h 和 a 完全消失。**因此，仅用这种差分观测，探测输入不能直接打破固定线性模型中隐藏负荷的位置等效。** 若噪声也与位置无关，差分观测的似然不含位置参数，其位置 Fisher 信息为零。

这不与 Bhela 等的非线性潮流探测相矛盾：其使用已知网络的多状态绝对方程及负荷恒定耦合，结构与此处的固定灵敏度纯差分不同。我们的探测可以先改善 R/X 或拓扑估计，从而改善绝对残差定位；新增内部测量也可直接区分原端口等效的解释。若主张依靠 AC 或 ZIP 的工作点变化增强分辨率，需要量化其效应是否超过噪声和模型误差，不能只援引“非线性存在”。

### 5.4 最小实验和判负标准

- 先用一个确定的端口等效位置对，分别比较随机逆变器差分、保留绝对电压的探测、内部电压补测、支路功率补测；线性反例中随机差分应失败。
- 测点选择比较随机、按最小 MAE、按最大响应方差、按最弱假设对分离；预算完全相同，评价次数、行程或采样成本。
- 每个策略采用同样的停止规则、置信条件和测量误差，报告总成本与定位失败，而不只比较一次动作后的熵下降。
- 在 AC 上若主动注入带来的差异小于合理偏置，优先选择内部测量或现场拓扑核验；这也是有价值的研究结论。

**优先级：高，适合作为方向 A 的后半篇或第二篇论文。** 第一版从临时测量位置选择开始，比同时引入逆变器控制、未知 ZIP、多源和在线校准更容易形成可信闭环。

## 6. 方向 C：从唯一源定位转向“哪些子树有多少未计量负荷”

这是中期扩展，先不替换单源主线。固定字典 D 和非负幅值 x 下，观测为 \(r=Dx+\epsilon\)，根表再约束总量。唯一源恢复不成立时，仍可能可靠估计某些子树总量 \(q_S=\sum_{h\in S}x_h\)。

对每个关心的子树，分别解

\[
\underline q_S=\min_{x\in\mathcal F(r)}\sum_{h\in S}x_h,
\qquad
\overline q_S=\max_{x\in\mathcal F(r)}\sum_{h\in S}x_h,
\]

其中 \(\mathcal F(r)\) 包括误差约束、非负性、总表区间、可选源数上限以及声明的图/参数范围。区间完全大于零的区域才是所有相容解释都需要额外负荷的区域；某个候选排名第一与此不同。

既有 2026-09-20 工作已经给出固定字典多源等效与内部测量分离例子。新贡献应是**在不可唯一恢复时仍可认证哪些聚合量、误差与图不确定性如何影响这些量、补哪一个测点最能缩短关键区间**，不应把原有稀疏秩检查重新命名为新算法。

固定 R/X、L1 或线性误差集合下，上下界可由 LP 计算；加入 K 稀疏性可成 MILP；R/X 与自由幅值同时变化产生双线性，必须清楚报告松弛界或局部解。对区间外推的统计覆盖也必须重新校准，不能沿用旧单源阈值。

最小实验：1 源与 2 源在端口上严格等效；加入同一子树的内部测量；比较点定位、可能位置并集、必然正负荷子树及区间宽度。若逐点位置不可辨识但高层子树总量稳定，就有值得报告的部分辨识结果。

**优先级：中。** 数学内容较强，但新仿真器、两源校准和双线性边界都需要额外工作；不宜把它加入第一篇论文的必需承诺。

## 7. 完整故事与论文安排

首选论文可以命名为：**拓扑与阻抗不确定下配电网未计量负荷的区域辨识及增补测量**。三个主贡献应分别回答：

1. 为什么正常预测误差不能充分评价用于异常定位的图；给出树结构下的干扰混淆与位置分离条件。
2. 如何同时保留图、参数、位置的可行解释，并在合理误差范围内输出区域或拒答；不能做的唯一物理节点主张主动排除。
3. 如何利用这些解释选择下一处测量，用等预算 AC 实验验证定位精度和调查成本的共同改善。

实验主表应是“正常图正确/错误 × 正常/异常 × 噪声/偏置 × 观测预算”，而不是越来越多的同类随机种子。原有三网络数据可以作为机制与回归背景；主要新证据应来自 65037 和至少一个公开三相网络、最接近方法的同测量复现、模型错配与候选遗漏。

必备对照包括：总表单独告警；固定 R/X 位置枚举；自由重估和收缩；Pengwah 的可获得实现或认真复现；PSCC 2026 模型在同样已知参数条件下的版本。若本方法使用额外内部候选、额外电压或可信正常历史，对照也需获得同样信息，或单独标明信息优势。

两篇分开写也合理：第一篇只做“图不确定下可辨识区域与任务质量”，第二篇做“部分辨识驱动的序贯观测”。是否拆分由真实实验体量决定，不宜预先承诺三个主题必须各成一篇。

## 8. 建议执行顺序与停止条件

| 阶段 | 可交付物 | 达标标准 | 若不达标如何转向 |
|---|---|---|---|
| 第一阶段：数学与复现 | 分离度推导、退化反例、65037 两历史模型诊断表、当前代码协议哈希 | 指标能解释已知响应零化失败；严格区分局部与全局条件 | 用直接预测集合距离替代过弱的线性投影指标 |
| 第二阶段：图错配实验 | 预设错挂/漏支撑/弱边/根偏差，独立开发与测试 | 相容区域比单点更诚实，并在固定覆盖要求下有实际大小优势 | 若集合几乎总是全域，降低贡献为不可辨识诊断，不强称可定位 |
| 第三阶段：补测闭环 | 1–3 个动作预算的枚举 oracle、随机与最大方差对照 | 同样错误预算下减少现场/测量成本；保留失败与域外事件 | 改成传感器布点论文，或指出纯 probing 对此任务无效 |
| 第四阶段：投稿前证据 | 三相近邻方法复现、参数与算力预算、外层校准重复、候选遗漏 | 增益来自已声明新机制而非额外信息或调参 | 暂缓“可靠窃电定位”主张，以方法与边界为主 |

没有本轮实验支持的百分比提升，也不预估投稿接受概率。优先验证关键机制与负面边界；若这些不能成立，增加场景数量不会救活论文主线。

## 9. 可复现检索式

以下可直接放入 Google Scholar；应同时记录检索日期、时间范围和读到摘要还是全文。

```text
"electricity theft" "unknown topology" voltage
"Parameter Identification of Unknown Radial Grids for Theft Detection"
"Model-less non-technical loss detection using smart meter data"
"invisible fraud" "low-voltage" optimization
"unmetered loads" identifiability nuisance topology
"non-technical losses" "uncertain topology" localization
"Smart Inverter Grid Probing for Learning Loads"
"distribution network" "partial identification" load
"electricity theft" "sensor placement" localization
```

IEEE Xplore 可先用以下 All Metadata 组合，再分别限定 Transactions on Smart Grid、Power Delivery、Power Systems；不要一次把所有限制堆叠到无结果。

```text
("All Metadata":"non-technical loss" OR "All Metadata":"electricity theft")
AND ("All Metadata":"voltage" OR "All Metadata":"power flow")
AND ("All Metadata":"localization" OR "All Metadata":"location")

("All Metadata":"unmetered" OR "All Metadata":"non-metered")
AND ("All Metadata":"identifiability" OR "All Metadata":"observability")

"All Metadata":"distribution network"
AND ("All Metadata":"active probing" OR "All Metadata":"sensor placement")
AND ("All Metadata":"topology" OR "All Metadata":"load")
```

本轮已证实足够近的早期与 2026 工作，因此本文只给“有潜力的差异化方向”，不声称完成穷尽查新。

## 10. 审计记录与记忆来源

本轮完成的是现状读取、统计 JSON 核验、原始文献阅读和数学研究设计；没有重跑历史 suites，也没有实现或验证新方向的 AC 优势。历史 pilot 的单次 HiGHS smoke 只代表当时 pilot，不能覆盖当下独立包的更多成果；反过来，V3 的完成也不能替代错误拓扑、多源和实测验证。

另外新增并实际执行了独立的 [theft_algebra_check.py](D:/0-github_workspace/Topo/docs/research_synthesis_20260927/theft_algebra_check.py)。该脚本不导入 production、不写既有输出，7 个断言通过：在一个 5 节点树中，分叉点与无观测侧支的端口响应差为 0；新增侧支内部电压后差为 1；线性主动注入差分对两位置的差仍为 0；参考偏置可完全吸收公共电压异常，而加入总表通道后存在性信号范数为 1；一个参数干扰方向可把原本为 1 的位置差消为 0。这只是确定性代数机制核验，不能外推为 AC 可恢复或统计检出率。

复现命令（不产生文件）：

```powershell
& 'D:\apps\miniconda3\envs\Topo\python.exe' docs/research_synthesis_20260927/theft_algebra_check.py
```

前期查阅记忆仅用于定位文件与保护边界：`MEMORY.md:390–442`、`MEMORY.md:222–269` 及 `rollout_summaries/2026-09-16T17-52-48-tjGz-topo_hidden_node_theft_two_module_pilot.md`。实现结论已由当前文件重新核验，特别更正了矩形灵敏度接口的旧状态。对应 rollout：`01a0ab59-9064-7d53-a520-1a6a4faed353`、`01a0b9d1-09b4-7c53-bc26-e084c6412985`。
