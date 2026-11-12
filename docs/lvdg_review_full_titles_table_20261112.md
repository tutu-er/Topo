# 综述扩展文献全名、研究特点与入文安排

整理日期：2026-11-12。范围严格对应上一轮输出的29篇扩展文献；前7篇主线文献继续参见[完整综述第13节](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#review-library-selection)。

研究特点依据上一轮已复核的关键正文；“综述中的位置与写法”为本次组织建议，可据论文上下文改写。下列均为正文定向阅读，具体页码和版本见文末；不表示逐页阅读全文。论文题名保留英文全名，题名链接指向原始论文、作者版本或正式书目页。

符号：P/Q为有功/无功，V为电压幅值；SM为智能电表，μPMU为同步相量装置，RG为递归分组，EV为电动汽车，DER为分布式能源。

## 1. 传统拓扑错误与候选开关

| 序号 | 文献全名与书目信息 | 研究特点与主要条件 | 在综述中的位置与写法 |
| --- | --- | --- | --- |
| 1 | **Srivastava等，2022**。[An Optimization-Based Topology Error Detection Method for Power System State Estimation](https://doi.org/10.1016/j.epsr.2022.107914)。Electric Power Systems Research。 | 在已有网络和SCADA/PMU量测下，利用状态估计残差与混合整数优化定位候选线路拓扑错误。 | 传统方法段，简述。可写：此类基于状态估计的拓扑错误处理依托已有网络模型，主要定位线路状态或模型错误；再转入末端未知结构问题。 |
| 2 | **Farajollahi等，2020**。[Topology Identification in Distribution Systems Using Line Current Sensors: An MILP Approach](https://doi.org/10.1109/TSG.2019.2933006)。IEEE Transactions on Smart Grid。 | 用部分线路电流相量、负荷量测或伪量测建立开关状态MILP，并讨论传感器布置和多周期误差；需要候选图及导纳。 | 量测与优化方法段，重点对照。说明设备代价与传感器类型、位置和覆盖有关，比较线路传感器与普通用户电表提供的信息。 |
| 3 | **Karimi与Natarajan，2021**。[Joint Topology Identification and State Estimation in Unobservable Distribution Grids](https://doi.org/10.1109/TSG.2021.3102179)。IEEE Transactions on Smart Grid。 | 将压缩感知状态恢复与候选开关估计联合建模，使用MILP、凸松弛和时间先验；节点集及导纳框架已知。 | 低可观测性段。说明有限量测可通过稀疏表示和历史信息补充，并把已知节点的不可观状态与未知物理节点分别列入比较表。 |

## 2. 非相量、混合仪表与量测误差

| 序号 | 文献全名与书目信息 | 研究特点与主要条件 | 在综述中的位置与写法 |
| --- | --- | --- | --- |
| 4 | **Zhang等，2020**。[Topology Identification and Line Parameter Estimation for Non-PMU Distribution Network: A Numerical Method](https://doi.org/10.1109/TSG.2020.2979368)。IEEE Transactions on Smart Grid。 | 在平衡网络等值下，由P/Q/电压幅值回归初始导纳与连边，再用AC潮流和Newton–Raphson精化；节点集给定。 | 一般配网的非相量方法段，重点介绍。可写：普通非相量量测已能支撑已知节点间的拓扑与参数联合估计；随后引出内部节点缺失的问题。 |
| 5 | **Srinivas与Wu，2022**。[Topology and Parameter Identification of Distribution Network Using Smart Meter and µPMU Measurements](https://doi.org/10.1109/TIM.2022.3175043)。IEEE Transactions on Instrumentation and Measurement。 | 结合智能电表、部分μPMU和伪量测，以UKF与Newton–Raphson交替估计，并分析μPMU位置。 | 混合量测段。用来将设备要求拆成覆盖、精度、安装位置和伪量测误差，而非统一归为高成本方案。 |
| 6 | **Brouillon等，2021**。[Bayesian Error-in-Variables Models for the Identification of Power Networks](https://doi.org/10.48550/arXiv.2107.04480)。库内2021技术报告/预印本。 | 用电压、电流相量的双边噪声模型进行MLE/MAP导纳估计，引入稀疏性和线路先验；目标为全观测或约化网络。 | 数据质量段，重点对照。说明输入量噪声也会造成估计偏差，讨论误差模型与结构先验如何影响参数和连边估计。 |
| 7 | **Yu等，2018**。[PaToPaEM: A Data-Driven Parameter and Topology Joint Estimation Framework for Time Varying System in Distribution Grids](https://doi.org/10.48550/arXiv.1812.06619)。库内2018预印本。 | 用P/Q/电压幅值及相角样本，以EM交替估计工况标签和各工况模型，同时处理双边量测误差。 | 多时段与拓扑变化段。可写：历史数据可能混合不同运行结构，需要同时辨识工况归属及各工况模型；另说明采样期拓扑固定的条件。 |

## 3. 统计图模型与网状结构

| 序号 | 文献全名与书目信息 | 研究特点与主要条件 | 在综述中的位置与写法 |
| --- | --- | --- | --- |
| 8 | **Weng等，2017**。[Distributed Energy Resources Topology Identification via Graphical Modeling](https://doi.org/10.1109/TPWRS.2016.2628876)。IEEE Transactions on Power Systems。 | 由电压概率图模型构造互信息最大生成树，并讨论幅值数据与部分网状扩展；采用注入独立等近似。 | 统计方法段，谱系代表。介绍从相关性到互信息树的思路，并交代统计依赖与真实物理连接之间需要模型条件。 |
| 9 | **Deka等，2020**。[Graphical Models in Meshed Distribution Grids: Topology Estimation, Change Detection & Limitations](https://doi.org/10.1109/TSG.2020.2978541)。IEEE Transactions on Smart Grid。 | 利用电压幅值和相角的逆协方差区分一跳、二跳关系，研究网状结构、线路变化和噪声；结构恢复有环长等条件。 | 统计方法段，重点比较。用其结构分析解释统计图中的附加边及物理支路筛选，再讨论径向/网状与噪声条件。 |

## 4. 三相不平衡与相别联合辨识

| 序号 | 文献全名与书目信息 | 研究特点与主要条件 | 在综述中的位置与写法 |
| --- | --- | --- | --- |
| 10 | **Bariya等，2021**。[Guaranteed Phase & Topology Identification in Three Phase Distribution Grids](https://doi.org/10.1109/TSG.2021.3061392)。IEEE Transactions on Smart Grid。 | 从三相线性模型出发，以多相电压统计联合识别相别与拓扑，给出判别条件；采用全部节点电压。 | 三相方法段，重点比较。与Fang2024并列，分别比较三相耦合处理、相别信息及内部节点量测覆盖。 |
| 11 | **Soltani等，2023**。[Simultaneous Robust State Estimation, Topology Error Processing, and Outage Detection for Unbalanced Distribution Systems](https://doi.org/10.1109/TPWRS.2022.3181118)。IEEE Transactions on Power Systems。 | 在已知线路与参数下，联合估计逐相开关、状态和停电区域；包含三相互阻及混合量测，案例含伪μPMU。 | 现代配网拓扑处理段。展示不平衡网络中状态、开关与停电的联合处理；验证应称实际馈线模型上的综合测试。 |

## 5. 隐藏节点、异质电缆与规模

| 序号 | 文献全名与书目信息 | 研究特点与主要条件 | 在综述中的位置与写法 |
| --- | --- | --- | --- |
| 12 | **Soumalas等，2017**。[A data driven approach to distribution network topology identification](https://doi.org/10.1109/PTC.2017.7981061)。2017 IEEE Manchester PowerTech（阅读作者稿）。 | 由根和叶端同步P/Q/电压估计灵敏度，再用距离与Prüfer序列恢复零注入内部点；依赖线型、邻近距离等先验。 | 隐藏节点段，早期代表。放在Park/Pengwah路线之前，说明末端灵敏度可连接到树层析，同时列出同步、采样和线路先验。 |
| 13 | **Feng等，2026**。[Adaptive optimal-weight method for topology identification and line parameter estimation of large-scale low-voltage distribution grids](https://doi.org/10.1016/j.ijepes.2026.112103)。International Journal of Electrical Power & Energy Systems。 | 通过拓扑编码矩阵、双向层次聚类和先验引导的启发式优化，联合恢复潜在接点、线型及参数；已知电缆库。 | 隐藏节点段，近期直接对照。讨论异质线型、噪声与规模，并比较电缆库和软先验带来的收益及搜索代价。 |

## 6. PLC、谐波与专用信号设备

| 序号 | 文献全名与书目信息 | 研究特点与主要条件 | 在综述中的位置与写法 |
| --- | --- | --- | --- |
| 14 | **Ahmed等，2013**。[Power Line Communications for Low-Voltage Power Grid Tomography](https://doi.org/10.1109/TCOMM.2013.111613.130238)。IEEE Transactions on Communications。 | 叶端PLC设备进行测距，再通过树层析推断内部支点和线路长度；依赖传播模型与测距质量，主要为数值验证。 | 信号/通信方法段，重点加入。可写：端到端通信信号也可提供内部结构信息，设备覆盖要求取决于测距和层析机制。 |
| 15 | **Chen等，2021**。[Topology Identification in Distribution Networks using Harmonic Synchrophasor Measurements](https://doi.org/10.1109/TSG.2020.3038214)。IEEE Transactions on Smart Grid。 | 联合基波与谐波相量构建候选开关MILP，并分析谐波源和传感器位置对可观测性的影响。 | 专门量测段。具体说明谐波带来何种额外信息，以及相应仪表能力、信号源和布置要求。 |
| 16 | **Duan等，2024**。[Signal Injection-Based Topology Identification for Low-Voltage Distribution Networks Considering Missing Data](https://doi.org/10.3390/en17092060)。Energies。 | 补全设备特征信号记录，再按包含关系恢复层级；依赖注入/检测终端、PLC和边缘终端。 | 部署与数据缺失段。将设备记录缺失补全作为工程路线，并与完全未安装仪表的物理分支点分开讨论。 |

## 7. 量化精度、异步与结果可信度

| 序号 | 文献全名与书目信息 | 研究特点与主要条件 | 在综述中的位置与写法 |
| --- | --- | --- | --- |
| 17 | **Talkington等，2026**。[Error Bounds for Radial Network Topology Learning From Quantized Measurements](https://doi.org/10.1109/TPWRS.2026.3666699)。IEEE Transactions on Power Systems。 | 分析量化步长、样本数与连续边权估计误差，采用固定功率因数、抖动量化等条件；节点维度给定。 | 量测精度段，重点加入。把笼统的高精度要求转为误差随量化和样本量变化的讨论；精确连边恢复还需分离条件。 |
| 18 | **Xu等，2023**。[Distribution Network Topology Identification Using Asynchronous Transformer Monitoring Data](https://doi.org/10.1109/TIA.2022.3212030)。IEEE Transactions on Industry Applications。 | 面向中压候选网络，以插值伪量测处理异步TTU数据，并以MIQCP估计开关及未测相角。 | 异步量测段。分别比较相角缺失、时间不同步和通信延迟，说明异步处理所依赖的网络模型。 |
| 19 | **Liu等，2026**。[Confidence-Aware Topology Identification in Low-Voltage Distribution Networks: A Multi-Source Fusion Method Based on Weakly Supervised Learning](https://doi.org/10.3390/en19061503)。Energies。 | 融合多种归属算法的弱标签，输出表计级置信分数和核查排序；实际台区案例重点评估错误分流。 | 结果可靠性段，短引。说明辨识输出可进一步用于现场核查优先级，比较总体准确率、错误召回及置信校准。 |

## 8. 边缘实现、GNN与物理引导学习

| 序号 | 文献全名与书目信息 | 研究特点与主要条件 | 在综述中的位置与写法 |
| --- | --- | --- | --- |
| 20 | **Zhao等，2020**。[Full-Scale Distribution System Topology Identification Using Markov Random Field](https://doi.org/10.1109/TSG.2020.2995164)。IEEE Transactions on Smart Grid。 | 以电压序列构建Markov随机场，结合增量学习和多终端并行计算；模型变量对应采集节点。 | 统计方法与边缘实现段。说明被动统计学习可结合增量更新和分布式部署；full-scale的MV/LV范围与隐藏节点能力分列。 |
| 21 | **Ebtia等，2025**。[Power Distribution Network Topology Detection Using Dual-Graph Structure Graph Neural Network Model](https://doi.org/10.1109/TSG.2024.3512456)。IEEE Transactions on Smart Grid。 | 以双图结构和GraphSAGE学习线路通断，区分训练与推断图输入，并测试未见工况；候选基础设施已知。 | 学习方法段，选作主要GNN代表。比较训练标签、μPMU覆盖、推断所需信息，以及未见工况测试的具体范围。 |
| 22 | **Duan等，2026**。[Adaptive Physics-Informed Data-Driven Topology Identification for Distribution Networks](https://doi.org/10.1109/TII.2026.3684153)。IEEE Transactions on Industrial Informatics。 | 将监督支路预测与线性潮流残差结合，使用已知候选图、参数及部分同步相量；新拓扑实验含少样本微调。 | 物理引导学习段。说明物理信息如何进入损失函数，比较支路级与整网指标，并区分微调适应与零样本泛化。 |

## 9. 树重建的数学基础

| 序号 | 文献全名与书目信息 | 研究特点与主要条件 | 在综述中的位置与写法 |
| --- | --- | --- | --- |
| 23 | **Buneman，1974**。[A note on the metric properties of trees](https://doi.org/10.1016/0095-8956(74)90047-1)。Journal of Combinatorial Theory, Series B。 | 证明满足四点条件的精确距离可以嵌入加权树，并可引入内部点。 | 方法基础，简引。解释端点距离为何能编码内部树结构；电气距离的构造与误差分析由具体电网方法承担。 |
| 24 | **Ni与Tatikonda，2011**。[Network Tomography Based on Additive Metrics](https://doi.org/10.1109/TIT.2011.2168901)。IEEE Transactions on Information Theory。 | 利用端到端可加距离恢复逻辑路由树及链路性能，给出一致性和误差半径条件。 | 树层析/NJ方法来源。介绍从可加距离到内部树的算法与稳定性，再说明所用电气距离如何满足这些条件。 |
| 25 | **Choi等，2011**。[Learning Latent Tree Graphical Models](https://www.jmlr.org/papers/v12/choi11b.html)。Journal of Machine Learning Research 12(49):1771–1812。 | 以加性信息距离进行递归分组，恢复最小潜树；讨论隐节点至少三邻居和结构一致性等条件。 | 潜树与RG方法基础。解释兄弟/父子分组及度数条件；把统计潜树假设与电气观测模型的对应关系单独说明。 |
| 26 | **Dörfler与Bullo，2011**。[Kron Reduction of Graphs with Applications to Electrical Networks](https://doi.org/10.48550/arXiv.1102.2950)。库内2011预印本。 | 对加权无向图的Laplacian作Schur补消元，研究端口等效及图、谱和有效电阻性质；按库内2011预印本阅读。 | 问题定义段。定义观测端口等效模型，再说明恢复原始物理内部结构还需哪些信息和条件。 |

## 10. 现代LVDG、局部容量与EV激励

| 序号 | 文献全名与书目信息 | 研究特点与主要条件 | 在综述中的位置与写法 |
| --- | --- | --- | --- |
| 27 | **Kumarawadu等，2025**。[Smart meter data-driven dynamic operating envelopes for DERs](https://doi.org/10.1016/j.apenergy.2025.125469)。Applied Energy。 | 由智能电表数据估计含中性线、互阻的三相响应模型，计算DER运行边界；测试量测由时序潮流生成。 | 重要性段，重点引用。用具体应用说明三相电气响应信息的价值，同时说明部分控制任务可直接使用端口响应模型。 |
| 28 | **Bailey等，2025**。[Electric Vehicles and the Energy Transition: Unintended Consequences of Time-of-Use Pricing](https://doi.org/10.1257/aeri.20240476)。American Economic Review: Insights。 | 随机现场干预比较分时电价与托管充电；以真实充电记录和虚拟变压器评估同步性及局部容量。 | EV应用动机。说明统一价格可能造成局部同步充电，从而引出电气归属和局部网络约束的重要性。 |
| 29 | **Bailey等，2025**。[Show Me the Money! A Field Experiment on Electric Vehicle Charge Timing](https://doi.org/10.1257/pol.20230653)。American Economic Journal: Economic Policy。 | 以随机奖励、信息处理和撤回奖励实验研究家庭EV充电时移及其持续性。 | 价格型EV探测的背景，条件性引用。支持激励可改变时序的现实依据，并将群体行为效应与逐节点功率执行区分。 |

## 11. 建议的段落连接方式

1. **应用动机**：先用动态运行包络或局部EV容量说明需要哪些电气信息，再界定本文追求完整物理拓扑、分支结构还是端口模型。

2. **已有任务与一般低压方法**：从状态估计/候选开关转向非相量回归、混合量测与统计图模型，比较节点清单、候选图和设备覆盖要求。

3. **聚焦隐藏节点**：以Park、Zhang潜树和Pengwah—Flynn—Fang为核心，加入Soumalas、Feng及PLC层析，分别比较零/非零注入、节点数量、参数先验和信号来源。

4. **方法适用条件**：将不平衡、噪声、异步、计算部署和学习泛化集中比较，研究缺口落在实际未满足的条件组合。

数学基础按所采用的方法简引；学习方法选代表比较；价格行为文献仅在主动EV激励确实进入方法时展开。

## 12. 版本与阅读定位

- Brouillon与PaToPaEM按库内2021报告、2018预印本说明；未以未核验的其他版本替代。

- Choi正式书目为JMLR2011，关键正文按库内2010预印本阅读。

- Ni的2011为正式书目年份；阅读页码属于库内arXiv:0809.0158v1长稿，其首页另有2018排版日期。

- Dörfler/Bullo按库内2011预印本阅读；Chen按作者稿阅读、采用2021期刊书目。

| 序号 | 原综述记录 | 实际阅读位置 |
| --- | --- | --- |
| 1 | [R014](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r014) | PDF p.1摘要/任务；PDF pp.2–3 §2 SE残差、关键支路与关键支路对；PDF p.6 TOMLAB MINLP；PDF p.8 §7结论 |
| 2 | [R112](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r112) | PDF p.2贡献与传感器特点；PDF p.3式(2)–(12)通断模型/Taylor近似；PDF p.4式(24)、§III-A Theorem1；PDF pp.6–7传感误差与测试设置 |
| 3 | [R068](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r068) | PDF p.2贡献/Markov与30%数据；PDF pp.3–4 §II–III测量、稀疏基和开关导纳；PDF p.5 §IV-B连续凸松弛 |
| 4 | [R058](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r058) | PDF p.2贡献/两阶段框架；PDF pp.3–5模型、回归与NR；PDF p.9测试设置 |
| 5 | [R073](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r073) | PDF p.2贡献与比较；PDF pp.3–5测量模型和UKF/NR；PDF p.9缺SM处25%功率伪量测实验 |
| 6 | [R070](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r070) | 技术报告PDF p.3 §2.2 Assumption1；PDF p.4 §3–4 Assumptions2–3；PDF pp.4–5极坐标/直角坐标噪声；PDF pp.7–8 §6.1 MAP |
| 7 | [R048](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r048) | PDF pp.2–3 §II–III量测/误差/潜工况；PDF p.7 Algorithm1输入、Remarks4–5相角设备与K |
| 8 | [R116](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r116) | PDF pp2–4 §II–III、Lemma1、Theorem1；p11幅值数据与讨论；p12 Chow–Liu/KL近似附录 |
| 9 | [R120](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r120) | PDF pp2–4 §§II–III、Assumption1、Theorem1；p7测量噪声与阈值条件；pp8–9 AC仿真与结论 |
| 10 | [R117](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r117) | PDF pp3–4模型及式(7)(8)；p5 Theorem2/3、Algorithm1全节点输入；pp6–7仿真与讨论 |
| 11 | [R115](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r115) | PDF p.3 §II模型与贡献；PDF pp.4–7逐相开关big-M及近似；PDF p.9 §III–IV时间融合/实际馈线/已知连接与参数/pseudo μPMU；PDF pp.11–12停电及坏数据试验、零电流不可辨识 |
| 12 | [R041](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r041) | PDF pp.1–3（§II–IV，Fig.2–3） |
| 13 | [R015](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r015) | PDF pp.1–4（§1、§2.2、Fig.2、式(3)–(7)） |
| 14 | [R024](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r024) | PDF pp.1–3：贡献、modem/branch-node图模型、双向测距；p.2明确数值验证。 |
| 15 | [R114](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r114) | 作者稿PDF pp.1–3：摘要与§II概念/可观测性、§III式(1)–(7)及MILP；公开arXiv摘要再次核实。 |
| 16 | [R004](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r004) | PDF pp.1–4、p.13；出版社全文§2–4、Fig.9附近未辨识设备与现场检查。 |
| 17 | [R108](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r108) | PDF pp.1–4：p.2 Assumption1及量化模型；p.3 Lemma1/Theorem1；p.4 Numerical Results与C_fit式(14)。 |
| 18 | [R019](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r019) | PDF pp.1–3：摘要，p.2 MV场景与异步原因，p.3延时纠正和MIQCP框架；p.8结论。 |
| 19 | [R107](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r107) | PDF p5 latent label与LF定义；pp7–12弱标签与D-S融合；p14实际广东台区：4分支127表、6天15min；式(33)–(35)三项指标；p19消融及错误核查收益 |
| 20 | [R022](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r022) | PDF pp2–3边缘计算与方法分工；pp6–7增量/并行模型；p9实际系统案例：15min更新、10天输入 |
| 21 | [R098](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r098) | PDF p2 TD定义与已知基础设施/阻抗；p3 TableI及归纳/传导区别；pp7–9实验场景与未见工况；pp13–14噪声与离群点；p16 240-Bus及结论 |
| 22 | [R111](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r111) | PDF pp2–5 TableI、§II–III、式(8)–(21)；p11新拓扑适应：20新拓扑+100原拓扑各15组样本，微调后评估；p11 Conclusion |
| 23 | [R013](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r013) | PDF pp.1–2，Theorems 1–2 |
| 24 | [R023](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r023) | PDF pp.1,4 §2，pp.32–33 §8 |
| 25 | [R031](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r031) | 本地PDF p7 §2.3 C1–C2；p10 Proposition3与式(12)加性；p11 §4 Recursive Grouping |
| 26 | [R032](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r032) | PDF pp.1–3，式(1.1)–(1.2)，p.25 §4 |
| 27 | [R010](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r010) | PDF pp.1、3–6：模型中性线/互阻、DOE约束；p.6 §6明确49连接点、模型仿真生成SM数据。 |
| 28 | [R144](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r144) | 正式PDF pp.1、5–6实验随机化/Virtual Transformers，p.10结果，p.15虚拟变压器限制；AEA官方页面已核实。 |
| 29 | [R146](../archive/legacy_research_20261112/legacy/docs/lvdg_hidden_nodes_literature_review_20261111.md#r146) | 正式PDF pp.1、4–7随机化/二阶段撤回，p.10经验方法，p.24稳健性；AEA官方页面已核实。 |
