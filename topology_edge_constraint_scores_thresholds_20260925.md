# 线路存在性证据：AC 误差贡献、自由模型与检验阈值

日期：2026-09-25

问题：使用线路存在/不存在对交流模型拟合误差的贡献作为指标是否合理？有没有直接先例？相对于自由状态的约束代价是否更合适？阈值如何建立？

结论：合理，而且已有直接文献。建议把指标解释为“当前数据对线路状态约束的支持/反对证据”，核心输出采用两维约束代价，而不是直接生成存在概率。最贴近的研究线包括 PaToPa 的删线后重拟合似然比较、广义状态估计的开关约束检验、标准化拉格朗日乘子、AC 停运候选误差间隔。

本文只新增研究说明；未修改既有算法、测试、数据或报告，未进行新的数值验证。经典 1998/2000 两篇原文的访问限制在文献表中明确列出；数学公式中哪些是独立推导也单独标注。

## 1. 已有研究做到哪一步

### 1.1 PaToPa：最接近“删线—重拟合—比较误差”

Yu、Weng、Rajagopal (2018)，PaToPa: A Data-Driven Parameter and Topology Joint Estimation Framework in Distribution Grids，IEEE TPWRS 33(4):4335–4347，DOI 10.1109/TPWRS.2017.2778194。

原文 §V、Theorem 2、Algorithm 1：从候选连接的参数估计出发，按电导排序删除候选边组，重拟合 EIV 模型，比较似然。小噪声分离论证使用无噪声设计矩阵满列秩及最优值连续性。实际算法用验证数据及 20% 似然容忍规则；后者不是置信水平。基础量测包括 P/Q/V/相角，缺角采用近零近似，不能等同于无角度完整 AC 联合估计。[作者全文](https://arxiv.org/pdf/1705.08870)；[机构书目](https://experts.azregents.edu/en/publications/patopa-a-data-driven-parameter-and-topology-joint-estimation-fram/)

因此，“删除真实边会增加最佳解释代价”的基本思路有直接先例；其成立依赖可辨识性、模型和噪声条件。

### 1.2 GSE：自由变量与开/闭约束

Kekatos、Giannakis (2012)，Joint Power System State Estimation and Breaker Status Identification，NAPS，DOI 10.1109/NAPS.2012.6336364。

论文把母线电压和开关电流一起作为状态；开路约束为电流零，理想闭合开关约束为两端电压差零。通过分组稀疏惩罚处理可疑状态。相量 V/I 下为线性 AC 模型，常规 PQV 下采用迭代线性化。正则权重所表达的 confidence 不等于校准概率。[作者全文](https://engineering.purdue.edu/~kekatos/papers/NAPS2012.pdf)

Monticelli (2000)，Testing equality constraint hypotheses in weighted least squares state estimators，IEEE TPWRS 15(3):950–954，DOI 10.1109/59.871718。摘要明确将 J-test 扩展到等式约束假设并用于开关状态；本次未获得其全文，下面的精确公式是独立线性高斯推导，不冒称逐式核验该论文。[IEEE 入口](https://ieeexplore.ieee.org/document/871718)

### 1.3 标准化乘子：约束影响的低成本指标

Clements、Simões Costa (1998)，Topology error identification using normalized Lagrange multipliers，IEEE TPWRS 13(2):347–353，DOI 10.1109/59.667350。方法把开关状态建模为运行约束，标准化对应拉格朗日乘子以识别错误状态；本次未获取该篇全文。[IEEE 入口](https://ieeexplore.ieee.org/document/667350)

可核验的一手后续材料是 Zhu、Abur 的 PSERC 2005 报告 Detecting Circuit Breaker Status Errors in Substations，式 (2.23) 给出 λ_i/sqrt(Cov(λ)_ii)，仿真使用 3.0 阈值。[PSERC 全文](https://documents.pserc.wisc.edu/documents/publications/reports/2005_reports/T-17_Part-2-Final-Report_Oct-2005.pdf)

### 1.4 AC 误差间隔与未决状态

Dai、Tate (2020)，Line Outage Identification Based on AC Power Flow and Synchronized Measurements，IEEE PES General Meeting，DOI 10.1109/PESGM41954.2020.9281410。

对每个单线路停运候选计算预期与实际电压相量变化之差，取最小误差候选；最优与次优误差间隔太小则输出 inconclusive。原文明确阈值按误识别/未决的取舍经验选择。其场景为已知网络和 PMU 量测下的停运定位。[作者全文](https://arxiv.org/pdf/2011.09566)

### 1.5 “影响小”并不等于“不存在”

Donmez、Abur (2022)，Enhancing topology error detection via multiple measurement scans，EPSR 213:108458，DOI 10.1016/j.epsr.2022.108458。

其局部估计使用标准化断路器流量阈值 3。原文 §5 明确讨论闭合但电流很小的开关可能被判为 open，解释为对状态估计影响小。这为区分“物理存在”与“当前工况下有可观测作用”提供直接案例。[大学托管全文](https://old.curent.utk.edu/2023IndustryConference/papers/Journal_PS_30.pdf)

## 2. 先把 AC 误差定义正确

记实际量测堆叠为 z，所有状态、隐藏相角和允许估计的参数记为 η。对于模型族 M，定义：

\[
J_M=\inf_{\eta\in M}
[z-h_M(\eta)]^\top\Sigma^{-1}[z-h_M(\eta)].
\]

所有比较应保持相同的量测集合、噪声协方差、其他参数范围与物理条件。这里 J 没有乘 1/2。

这比直接相加 P/Q/V 原始误差更适合统计解释：不同单位、精度和相关性通过 Σ 统一。独立误差的特例是：

\[
J=\sum_i(r_{P,i}/\sigma_{P,i})^2
+\sum_i(r_{Q,i}/\sigma_{Q,i})^2
+\sum_i(r_{V,i}/\sigma_{V,i})^2.
\]

这仍然只是一个观测模型。若 P/Q 本身有噪声，不能一面把它们当精确潮流输入，一面把输出误差当作独立 V 噪声。可以把真实注入、相角等作为状态，在精确 AC 约束下共同拟合量测。跨节点、跨时间相关需要完整协方差或有依据的近似。

如果各模型还估计不同的 Σ，完整高斯负对数似然包含 log det Σ；只比较平方残差已经不是完整似然比。

若采用 L1、Huber 等鲁棒损失，仍可定义同样的约束代价，但卡方推导不能原样照搬。

## 3. “自由状态”有三种不同含义

### 3.1 自由选择二元状态：没有增加信息

若自由仅指让 b_e∈{0,1} 自行选择：

\[
J_F=\min(J_0,J_1).
\]

于是：

\[
J_0-J_F=\max(J_0-J_1,0),\qquad
J_1-J_F=\max(J_1-J_0,0).
\]

它只是原差值的正负部分，不是一个额外的诊断参照。

### 3.2 自由估计连续导纳：适合区分标称参数与断开

例如用 y_e=a_e y_{\rm nom} 表示支路，比较 a=0、a=1 与 a 自由。

- a=0 表示该简化支路断开。
- a=1 表示具有标称导纳的支路。
- a 自由表示连续包容模型。

自由解 a=0.6 不是存在概率 0.6；它也可能反映阻抗错误、噪声或其他模型失配。

特别要区分：H1“存在且导纳等于标称值”与 H1“存在且导纳可以重新估计”。前者被拒绝，不能直接推出边不存在。后者若允许导纳任意逼近零，H0 落在 H1 的闭包里，精确有/无很难稳定区分；需要物理参数下界或明确的工程等价阈值。

如果自由导纳模型恰好就是 H0 与 H1 的并集或闭包，最优值层面仍可能退化为二元自由；“连续”不自动意味着新增诊断能力。

### 3.3 释放目标支路本构关系：推荐的诊断参照

【以下是借鉴 GSE 的研究设计，尚未在当前代码实现】

简化无并联支路时，保留节点 KCL、其他线路关系以及量测模型，仅将目标支路复电流 I_e 当作自由状态：

\[
H_F:\ I_e\text{ 自由，不强制目标支路的欧姆关系};
\]

\[
H_0:\ I_e=0;
\]

\[
H_1:\ I_e=y_e(V_i-V_j).
\]

I_e 对两端节点的注入符号相反，仍参与 KCL。对理想断路器，闭合条件改为 V_i−V_j=0，不能把“理想开关”与“有限阻抗线路”混为一谈。线路含并联电纳、互感、变比时，应按真实两端口关系处理。

若 y_e 未知，应在 H1 内按合理物理范围拟合，并在多个时段共享线路参数。H_F 每个时段增加的复电流自由度会增长，因此它是诊断超模型，不代表真实的被动电网；不能无限释放所有边而忽略可观测性。

## 4. 推荐保留两个约束代价

只要 H0、H1 的可行域均包含在 HF 内，定义：

\[
\boxed{D_{e,0}=J_{e,0}-J_{e,F}},\qquad
\boxed{D_{e,1}=J_{e,1}-J_{e,F}}.
\]

两者理论上非负。解释方向非常重要：

- D0 大：强制“没有该边”显著损害拟合，是反对无边的证据。
- D1 大：强制“有该边且符合所给物理关系”显著损害拟合，是反对该有边模型的证据。
- 某个 D 小：该约束与当前数据相容，不是已经证实其为真。

| D0 相对自身阈值 | D1 相对自身阈值 | 合理输出 |
|---|---|---|
| 大 | 小 | 无边被拒绝，有边仍相容；支持保留 |
| 小 | 大 | 有边模型被拒绝，无边仍相容；支持去除/检查参数 |
| 小 | 小 | 两状态都相容；未决，可能激励不足 |
| 大 | 大 | 两状态都不相容；检查参数、其他拓扑、坏数据或模型遗漏 |

还要检查整体拟合是否合理。若 JF 本身已经异常大，两个 D 很小也不能证明整个模型可信，因为三种模型可能共同存在系统性误差。

原先的有符号指标仍可保留用于排序：

\[
S_e=J_0-J_1=D_0-D_1.
\]

保留 (D0,D1) 多出的信息，是能够区分“两者都相容”和“两者都不相容”。这两种情形可能有相近的 S，却需要完全不同的处理。

不推荐把 S/J1 作为首选统计量：分母接近零会不稳定，比例受其他量测贡献影响，也没有一般通用分布。

## 5. 为什么它有统计基础：约束损失、Wald 与乘子相联系

【本节为标准线性高斯模型的独立推导；不是声称已逐式核验 Monticelli 原文】

设：

\[
z=H\eta+\varepsilon,\qquad
\varepsilon\sim N(0,\Sigma),
\]

Σ 已知且正定，H 满列秩。自由 WLS 解为 ηhat，其协方差：

\[
K=(H^\top\Sigma^{-1}H)^{-1}.
\]

检验 q 个独立等式：

\[
A\eta=a,\qquad {\rm rank}(A)=q.
\]

令 c=Aηhat−a，V=AKA^T。二次目标展开并求最小约束修正，得到：

\[
\boxed{
J_C-J_F=c^\top V^{-1}c
\sim\chi_q^2\quad(H_C\text{ 为真}).
}
\]

如果拉格朗日函数对目标使用 1/2 J，那么 λ=V^{-1}c，乘子协方差为 V 的逆矩阵，从而：

\[
J_C-J_F=
\lambda^\top{\rm Cov}(\lambda)^{-1}\lambda.
\]

因此约束目标增量、标准化约束残差的 Wald 统计量、标准化乘子的联合平方，在这个模型中是一回事。

这给出两种实现：

1. 精确重拟合 H0/H1/HF，直接计算 D；
2. 在可用的增广状态估计模型下，以归一化乘子进行快速筛查，再对重点线路精确重拟合。

单条线路常对应多个实数约束，宜使用完整协方差的组统计量。不能把一组相关乘子逐个取绝对值后就当成一个已校准的线路概率。

在非线性 AC 中，上式通常只有局部或渐近意义；参数处于边界、模型不可辨识、随样本量增加的状态 nuisance 或拓扑选择，都会影响其适用性。裸乘子还会随约束单位改变，应当标准化。

## 6. 阈值应怎样选择

### 6.1 正则线性/局部高斯基准

在第 5 节的精确条件下：

\[
D_b>\chi^2_{q_b,1-\alpha}
\quad\Longrightarrow\quad
\text{拒绝状态 }b.
\]

例如单个独立实数约束、5% 显著性时为 3.841；两个独立约束时为 5.991。这些只是条件成立时的参照，不是“一条电网线路通用阈值”。

q 是有效独立约束的秩，可能涉及电流实虚部、多时段关系、共同阻抗参数。不是“删一条边，所以 q=1”。

对 a≥0 的导纳幅度、H0:a=0 等边界问题，经典 Wilks 定理一般不能直接用。Self–Liang (1987) 给出了边界下不同极限分布；最简单情况有混合卡方，但零边时阻抗方向不可辨识等情形会更复杂。[原文](https://www.stat.cmu.edu/~brian/763-2015/week06/papers/self-liang-1987.pdf)

### 6.2 当前 AC 研究建议：分别按两个状态做参数模拟标定

对每个 b∈{0,1}：

1. 在 Hb 下估计状态、线路参数及允许的量测噪声结构。
2. 在该拟合模型与相同量测布置/工况设计下生成重复数据。
3. 每次重新求解 HF、H0、H1；若前面有数据驱动选边或调参，应重做相同流程，或用独立数据冻结它。
4. 形成 D_b 的模拟零假设分布，取相应高分位数作为 τ_b。
5. 在额外独立、已知真值的网络/工况上报告错误拒绝率、漏检率、未决率和模型失配敏感性。

这种参数 bootstrap 是模型依赖、通常近似的标定。它不能自动对所有未知参数、所有网络统一保证 α。需要覆盖弱激励、相关噪声、参数误差、坏数据等合理情形；插件估计错误会带入阈值。

同时检验很多边，还需要多重检验控制；单个“3σ”不是全网正确率。相邻时段、同一网络内的多条边，也不能不加区分地算作独立校准样本。

### 6.3 严格但保守的参考：噪声能量上界

【直接推导，不依赖 Wilks；计算和模型前提很强】

设真实均值属于 Hb，全部 N 维量测噪声 ε∼N(0,Σ)，Σ 已知。由于真实状态是 Hb 的可行点：

\[
J_b\le\varepsilon^\top\Sigma^{-1}\varepsilon\sim\chi_N^2.
\]

又因 JF≥0：

\[
0\le D_b\le J_b.
\]

因此：

\[
P_b\{J_b>\chi^2_{N,1-\alpha}\}\le\alpha,
\qquad
P_b\{D_b>\chi^2_{N,1-\alpha}\}\le\alpha.
\]

这里只允许纯 WLS；同批数据估计 Σ、加入正则罚项或存在未建模偏差时，不能直接沿用该精确声明。它对非线性/离散模型仍成立，但通常很保守，不能像正则理论那样直接把 N 改成“量测数减参数数”。

拒绝时需全局最优值或正确方向的认证界：局部 AC 求解器返回的可行目标通常是 J_b 的上界，数值很大不能证明所有可行状态都拟合不好。若需要认证 D_b>τ，可使用

\[
{\rm LB}(J_b)-{\rm UB}(J_F)>\tau.
\]

只用各模型两个局部解的差值，属于数值证据，不具有上述严格拒绝保证。

## 7. 什么指标“更合适”：按目的选择

| 目的 | 指标 | 优点 | 解释边界 |
|---|---|---|---|
| 线路候选排序 | S_e=J0−J1 | 直观、保留倾向方向 | 不能发现两状态都差；不是概率 |
| 状态兼容性诊断 | (D0,D1)=(J0−JF,J1−JF) | 区分支持、未决、模型失配 | 需真正包容自由模型和各自阈值 |
| 大网络低成本筛查 | 标准化乘子向量的二次型 | 利用局部曲率及噪声尺度 | 局部近似，受可观测性影响 |
| 防止训练误差过拟合 | 独立数据上的加权误差增量 | 衡量样本外解释收益 | 预测贡献不等于物理存在性 |
| 判断是否值得增加量测 | 噪声归一化的模型分离/有效信息 | 揭示弱激励和参数补偿 | 需可信模型，通常是条件性诊断 |

若计算成本允许，优先采用“(D0,D1) 作为主要诊断，S 用于排序，归一化乘子用于预筛”。不必强行融合成一个 0–1 数值。

与正则化阈值比较：L1/group-Lasso 的 λ、AIC/BIC 复杂度惩罚，可以帮助模型选择，但其数值不自动成为边存在置信度。若用 AIC/BIC，参数数目、有效样本量、边界和 nuisance 条件也应正确处理；相同边数的两棵树，普通参数计数惩罚还可能完全相同。

## 8. 面向当前拓扑研究必须固定的范围

1. **条件边证据还是全局边证据？** 固定其他边只改变 e，结论以其他边正确为条件。若要全局边结论，J0/J1 应分别在所有允许的无 e/有 e 拓扑族内优化，或明确候选池范围。
2. **径向性。** 删除一条树边会断连，加一条会成环。比较合法树时往往需要换边、重排其他连接，不能把非法结构的拟合恶化当作该边的独立贡献。
3. **参数误差。** H1 固定错误阻抗而 H0 允许其他参数调整，会把阻抗错误混成拓扑证据。应采用一致、明确的参数范围。
4. **噪声与结构作用。** 很小的 S/D 不证明物理边不存在；可能是电压差小、负荷弱、量测位置不合适或有可替代路径。
5. **原始图与约化图。** 终端观测下，某些隐藏物理边只通过等效导纳体现；指标应对应实际可辨识对象。
6. **自由模型有效性。** 若释放过多自由电流使模型完全吸收量测噪声，统计代价和约束秩要相应调整，不能仍视作只增加一个边参数。
7. **未知导纳趋零。** 若“有边”允许任意趋近零的作用，“有边”与“无边”贴边；证明精确不存在需额外分离，或改为实际可忽略的作用阈值。

## 9. 可以形成论文的具体主张

可行的主张是：

> 在给定观测与物理模型下，将目标支路的有/无状态写成共同自由模型上的不同约束，构造噪声标准化的双向剖面拟合代价，并通过约束检验或完整模拟标定区分存在支持、缺失支持、未决和模型不相容。

不宜只把“删线后残差变化”或“相对自由状态的代价”本身作为全新思想。文献已经存在这些基本路线。是否有实质创新，要落在具体的未知参数、隐藏节点、仅 PQV、全 AC、合法树替代、错误率验证与计算效率问题上，并通过对应对照实验支持。

推荐最先做的一个小实验设计（未执行）：

- 同一网络与同一真实参数，分别生成有边、无边、闭合但弱激励、错误阻抗、量测坏数据五类场景；
- 保持量测位置、样本数和噪声等级可比；
- 比较 S、(D0,D1)、标准化乘子、验证集误差差四种输出；
- 看单一 S 是否把“两个都坏”误当作“有明显胜者”；
- 对阈值单独标定并在独立情景测试，记录误报、漏检、未决和求解失败；不得只在成功辨识样本上统计。

## 10. 可复制检索词

| 对应文献/路线 | Google Scholar | IEEE Xplore Advanced Search |
|---|---|---|
| PaToPa | "PaToPa" "topology" "likelihood" | "Document Title":"PaToPa" |
| 等式约束检验 | "Testing equality constraint hypotheses in weighted least squares state estimators" | "Document Title":"Testing equality constraint hypotheses in weighted least squares state estimators" |
| 自由开关变量与联合估计 | "Joint Power System State Estimation and Breaker Status Identification" | "Document Title":"Joint Power System State Estimation and Breaker Status Identification" |
| 标准化拉格朗日乘子 | "topology error identification" "normalized Lagrange multipliers" | "All Metadata":"topology error" AND "All Metadata":"Lagrange multipliers" |
| AC 停运误差间隔 | "Line Outage Identification Based on AC Power Flow and Synchronized Measurements" | "Document Title":"Line Outage Identification Based on AC Power Flow and Synchronized Measurements" |
| 多时段与弱激励 | "Enhancing topology error detection via multiple measurement scans" | "All Metadata":"topology error" AND "All Metadata":"multiple measurement scans" |
| 未知线路参数混淆 | "Identification of Network Parameter Errors" Zhu Abur | "Document Title":"Identification of Network Parameter Errors" |
| 约束统计的推广 | "profile likelihood" "topology identification" | "All Metadata":"topology identification" AND "All Metadata":"likelihood" |

Donmez–Abur 属于 Elsevier 期刊，Xplore 对应项是主题扩展查询，不保证原文收录。Google Scholar/Xplore 搜索语句用于复现查找，不代表已在这两个网站逐项运行搜索。

补充参考：Zhu、Abur (2006)，Identification of Network Parameter Errors，IEEE TPWRS 21(2):586–592，DOI 10.1109/TPWRS.2006.873419；以参数约束的标准化乘子处理参数错误，可用于理解参数—拓扑混淆，但不是直接证明某条边存在。[PSERC 原文](https://documents.pserc.wisc.edu/documents/publications/papers/2006_general_publications/zhu-abur.pdf)
