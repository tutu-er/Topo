# 低压配电网隐藏节点拓扑辨识文献综述与逐篇阅读记录

整理日期：2026-11-11。研究对象：低压配电网（low-voltage distribution grid，LVDG），重点为用户端可测、内部接线或分支节点缺少量测的场景。“边缘”在本文件中指配电网末端和用户侧；边缘计算的通信与部署问题仅在对应文献中讨论。

本文件同时提供综述论证框架、方法比较、写作建议和逐篇阅读记录。主要结论是：**已有研究直接处理低压配电网的未知隐藏分支点；综述应从辨识对象和可用信息出发，比较各方法在何种条件下恢复何种结构。** 研究缺口应落到具体量测、先验、隐藏节点类型、物理近似及噪声条件。

## 目录

- [1. 文献范围与阅读依据](#review-scope)
- [2. 问题定义与比较维度](#review-definitions)
- [3. 建议的综述结构](#review-outline)
- [4. 一般拓扑辨识方法的组织方式](#review-general)
- [5. 隐藏节点方法的重点比较](#review-hidden)
- [6. EV与需求响应文献的使用方式](#review-ev)
- [7. 研究缺口与贡献表述](#review-gaps)
- [8. 可改写进论文的综述段落](#review-draft)
- [9. 全部文献逐篇记录](#review-records)
- [10. 库外补充文献](#review-supplemental)
- [11. 版本合并与书目修正](#review-bibliography)
- [12. 外部补检结果与新增文献](#review-search-update)

<a id="review-scope"></a>

## 1. 文献范围与阅读依据

### 1.1 覆盖范围

本次以 Zotero“拓扑辨识”“拓扑辨识-EV”及其子目录为主要文献集，优先复用本地已有英文 PDF。公共原始来源用于核对书目、补充缺失摘要及追溯两篇直接相关工作。

| 项目 | 数量与含义 |
| --- | --- |
| Zotero原始记录 | 154条；“拓扑辨识”120条，“拓扑辨识-EV”34条，各自含子目录并按条目键去重 |
| 初步归并 | 按同DOI或同题名合并为142组；其中仍有题名变化的预印本与正式版，例如R039与R045，应合引 |
| 正文定向阅读 | 136组；阅读具体问题设定、方法、假设、定理条件或结论，实际范围列于各项记录 |
| 摘要及概述筛选 | 6组：R007、R011、R012、R027、R030、R086；其中R086未取得全文，其余主要为外围文献筛选 |
| 首轮库外补查 | 2篇：SUP01读到方法正文；SUP02仅读官方摘要 |
| 本轮外部补检 | 25项（SUP03–SUP27）：11项正文定向阅读，8项摘要或部分预览，6项仅书目待取原文，见第12节 |

原142组的136/6阅读统计不包含本轮外部增补；新增项分别标明阅读层级。

“142组”是本次记录组织单位，不能直接写成142篇相互独立的研究。R001等通用理论、窃电、安全和机器学习综述保留在逐篇记录中，但主文应按实际论证需要取舍。

### 1.2 阅读与来源标记

- **正文定向阅读**：结论仅对应列出的实际页码、章节、假设或算法；不表示整篇逐段精读。
- **摘要及概述筛选**：用于判断主题与概括明确陈述的方法，不据此补写未读的算法或证明细节。
- **原文方法与条件**：整理作者在所读范围内采用的模型、输入和程序。
- **综述位置与本文判断**：本文件对归类、可比性和写法的分析；涉及额外代数推理时单列。

PDF页码从1开始，可能与印刷页不同。逐篇记录在可定位时提供DOI、出版社或作者存档等稳定链接，并给出原文定位；少数外围条目的公开书目入口仍待补。作者稿、预印本和正式版的已知差异在对应条目说明。未进行覆盖全部文献的勘误、撤稿及最终卷期专项核验。

### 1.3 全文与版本缺口

1. **R086，PLGAN 2024**：本地无PDF。Springer官方页面可读摘要；HKU目录查询未显示匹配记录，页面同时提示服务异常。具体量测与训练条件仍待正文。
2. **SUP02，中文潜树与聚类搜索2025**：已核官方中英文摘要和书目，正文访问受阻。具体量测、零注入及节点度数条件待核。
3. **R121/R142**：为同DOI重复条目，R121已有全文；该EV论文不存在本次全文缺口。
4. **R008/R009**：库中标2027卷期，DOI含2026。引用时应分别核对在线发表时间和卷期年份，保留当前书目状态。

<a id="review-definitions"></a>

## 2. 问题定义与比较维度

### 2.1 先明确“拓扑辨识”的输出

| 任务 | 输出 | 典型已有信息 | 与末端隐藏树的关系 |
| --- | --- | --- | --- |
| 户变、馈线或相别归属 | 用户所属类别 | 已有变压器/馈线/相别集合 | 可支撑分区，通常不还原区内全部线路 |
| 线路或开关状态检测 | 已知支路的通断 | 节点、候选线路，常含参数 | 难点在状态选择，节点数量通常预先给定 |
| 已知节点间结构与参数估计 | 连边和导纳/阻抗 | 节点清单及这些节点的量测 | 可降低线路先验要求，未必生成未知内部点 |
| 约化网络辨识 | 观测端口间的等效导纳或灵敏度 | 部分节点的输入输出 | 保留端口行为；原内部结构需另行恢复 |
| 隐藏物理节点恢复 | 内部节点、连接及可能的边参数 | 末端量测，节点数可能未知 | 是本综述的核心目标 |

“partial observability”“latent”“hidden”“full-scale”等术语须按原文对象理解。例如R068针对既有节点的未测状态与开关；R048的潜变量是样本所属工况；R106的目标为Kron约化导纳。对应证据见[R068](#r068)、[R048](#r048)、[R106](#r106)。

### 2.2 隐藏节点至少分为三种

1. **零注入接线或分支点**：无负荷和发电，仅连接线路。Pengwah、Fang、Guo等工作的内部点主要属于此类。
2. **带未知注入的内部节点**：有负荷或发电，但未测量。Park等交叉矩/树距离方法允许此类节点，同时采用跨节点注入不相关等统计条件；Anguluri结构分解方法另有隐藏节点空间分隔条件。
3. **已知用户或设备缺少高分辨率数据**：实体存在或位置类别已知，可仍有低频电量、伪量测或图属性。R089的interval用户、R104的已有分支箱实体分别属于不同信息条件。

这三类在信息量和可辨识性上不同，比较表应逐篇明确。

### 2.3 建议统一的术语和符号

| 符号或术语 | 本文件含义 |
| --- | --- |
| V | 电压幅值；电压平方或相量会另行写明 |
| P、Q、I | 有功注入、无功注入、电流；方向与具体论文约定对应 |
| SM | 智能电表，实际可提供V/P/Q/I、功率因数或电量的哪几项须逐篇检查 |
| μPMU | 微型同步相量测量装置，可提供幅值与相角等同步信息 |
| RG | recursive grouping，递归分组，由节点关系逐步恢复树及内部点 |
| Steiner树/最小潜树 | 为连接观测节点引入内部点的树，通常将不可分辨的串联结构约化 |
| LTM | latent tree model，潜树概率模型 |
| EM、BIC | 期望最大化参数估计；贝叶斯信息准则，用于模型拟合与复杂度权衡 |
| Kron约化 | 消去内部节点后保留边界端口关系的网络等效，详见R032 |

### 2.4 最小树与逐个物理设备恢复

一个简单的等效性例子可以说明恢复目标：若没有量测和注入的内部节点仅连接两条串联线路，端点观测只反映总阻抗

$$
z_{\mathrm{eq}}=z_1+z_2.
$$

保持总和不变，可改变分段参数或将该度2节点并入一条边。这个例子说明：恢复可辨识的最小树与恢复每个物理接线设备是两个目标。若论文要求后者，应说明额外的设备信息、距离、参数或量测如何解除等效性。Park等工作的度数条件与约化讨论可作为直接来源，见[R055](#r055)；一般网络消元性质见[R032](#r032)。

### 2.5 方法比较应使用的维度

| 维度 | 需要记录的具体问题 |
| --- | --- |
| 节点与线路先验 | 节点数是否已知？候选线、开关位置、相别、阻抗或电缆库是否给定？ |
| 观测覆盖 | 全节点、全叶端、部分用户、混合高低频电表，还是仅探测端？ |
| 量测与控制 | V/P/Q/I/功率因数、相角、谐波、特征信号、可控逆变器或充电事件？ |
| 隐藏节点语义 | 零注入、未知负荷、已知实体缺数据，还是统计潜变量？ |
| 物理模型 | 单相/平衡、显式三相耦合、公共中性线、线性化或AC、根电压是否变化？ |
| 数据条件 | 时间对齐、采样率、噪声/量化、缺失机制、统计不相关或回归可识别条件？ |
| 恢复对象 | 归属、开关状态、最小树、完整候选图、约化导纳、边参数？ |
| 证据 | 条件性定理、可验证求解界、模拟、实测、局部拟合或启发式候选搜索？ |

<a id="review-outline"></a>

## 3. 建议的综述结构

### 3.1 第一段：末端低压网连接信息的重要性

围绕DER、EV和灵活负荷进入用户侧后的具体任务展开：潮流计算、线路负载评估、故障和异常定位、三相电压分析，以及按位置分配运行空间。网络连接决定共同路径阻抗及局部相互影响，仅有总负荷无法代替这些空间信息。

推荐来源：[R087](#r087)用于总体技术背景；[R010](#r010)用于运行包络；[R035](#r035)用于已知网络上的非技术损耗检测；[R109](#r109)用于安全评估。EV现场文献[R144](#r144)、[R152](#r152)可说明统一激励下的同步充电与局部容量问题，保留原文虚拟/合成容量评估的范围。

电压控制文献[R085](#r085)、[R092](#r092)也表明，一些控制目标可通过等效输入输出模型或模型集合完成。因此重要性段应聚焦确实使用连接结构的任务。

### 3.2 第二段：从已有方法过渡到本研究任务

把“高压方法不适用于LVDG”具体化为模型与信息前提的不同：

- [R034](#r034)使用PMU角度/频率和慢振荡动态等值；[R036](#r036)采用DC潮流及固定节点集。
- [R014](#r014)、[R059](#r059)在已有网络上检测拓扑错误或候选单线停运。
- [R019](#r019)、[R068](#r068)、[R112](#r112)等在候选物理图上联合估计开关和状态。

这些工作的优化、统计和学习工具可供借鉴；末端无相角、内部节点数未知的任务需要重新建立变量域、物理关系和可辨识条件。按这些前提过渡，比按发表早晚或电压等级简单切分更准确。

### 3.3 第三段：一般低压方法及其适用条件

建议按“能量/电压数据包含什么结构信息”组织：守恒子空间、统计依赖、物理灵敏度/导纳、主动测距/扰动、监督图学习。每类先用一两句介绍机制，再用一两篇代表说明量测覆盖和结构输出。

过渡句可以围绕内部节点缺测提出：已有方法中，一部分在给定量测节点间连边或识别已知候选支路；针对无量测内部接线点，需要具有潜节点生成或内部结构恢复机制的方法。

### 3.4 第四段：集中比较LVDG与隐藏节点方法

按三条主线展开：

1. **潜树概率建模与结构搜索**：Zhang2022、中文2025摘要线索；讨论电压输入、隐节点数量、EM/BIC及统计—物理映射。
2. **电气灵敏度/加性距离与树重建**：Park、Pengwah、Flynn、Fang、Guo、Feng；比较零注入或未知注入、源电压、三相耦合、混合电表、参数先验和分组规则。
3. **主动层析和结构分解**：PLC、逆变器、Wallbox及Anguluri；比较测距、控制、相角和隐藏节点空间条件。

这些方法的输入和恢复对象并不完全相同，宜给条件对照表，不宜直接排统一性能名次。

### 3.5 第五段：从本文实际满足的条件提出研究缺口

先用一句话明确本文的信息契约，例如“仅用户侧哪些量测可得、哪些内部结构未知、采用何种物理模型、输出哪种可辨识结构”。随后选择最接近的已有方法，说明本文具体改变了哪个条件或改善了什么结果。贡献应由推导、实现和验证分别支撑；本轮文献整理未新增算法实现或实验。

<a id="review-general"></a>

## 4. 一般拓扑辨识方法的组织方式

| 路线 | 基本机制 | 代表记录 | 与隐藏节点任务的主要联系 |
| --- | --- | --- | --- |
| 电量守恒/PCA | 从多时刻守恒约束估计低维约束子空间，再解释电表树 | R039/R045 | 全节点装表条件适合作为内部缺测问题的引入 |
| 电压相关、互信息、MRF | 从电压依赖或逆协方差推断连接 | R002、R022、R081、R116、R120 | 需逐篇核对统计边与物理边的对应，以及是否生成潜节点 |
| 非相量物理回归 | 用P/Q/V或I估计导纳、灵敏度，再精化连接/参数 | R058、R066、R073、R075 | 降低相角需求，但固定节点集、多工况类别与未知隐点数要分开 |
| 特征电流、谐波或线路电流 | 利用传播路径、KCL和特定信号增强连边约束 | R004、R060、R063、R069、R077、R112、R114 | 设备成本具体到收发、传感、同步和部署覆盖；部分PLC方法可恢复未装设备内部点 |
| 主动资源探测 | 已知功率扰动与电压响应提供灵敏度信息 | R020、R049、R056、R082 | 比较可控位置、观测位置和探测期间背景负荷条件 |
| 监督学习与图学习 | 在标注工况、候选图或设备图上预测类别/边 | R062、R076、R078、R083、R086、R097、R098、R101、R104、R110、R111 | “新节点泛化”“缺属性”“潜表示”需与未知物理接线点生成分别描述 |
| 约化拓扑或等效模型 | 学习观测端口之间的导纳/灵敏度 | R010、R070、R085、R092、R106 | 能满足部分下游任务，原内部物理结构仍需额外信息 |
| 置信集与多源融合 | 输出候选集或融合不同识别标签/分数 | R091、R107 | 明确校准对象、同分布条件和输出语义，避免把评分当成一般拓扑正确概率 |

### 4.1 设备要求应分三层写

- **新增硬件**：PLC测距、TMR传感器、智能开关、μPMU或专用脉冲设备。
- **数据服务**：普通SM是否上传V/P/Q/I、是否包含功率因数和符号、时间分辨率与同步精度、通信可靠性。
- **操作权限**：逆变器P/Q调节、直接充电启停、用户合同内调度，或仅能广播价格。

普通智能表方法也可能有较高数据要求；使用主动资源的方法则可能复用已有装置。综述应报告具体要求及其用途，不凭算法名称判断现场成本。

### 4.2 三类容易混淆的“部分信息”

1. R089的部分SM覆盖仍配有其他用户低频电量；完全无用户记录是更少的信息。
2. R104借已有异构图为分支箱建立表示；已知设备实体与未知设备数量不同。
3. R106在观测端口估计Kron约化导纳；约化连边不能直接解释成原电网物理支路。

<a id="review-hidden"></a>

## 5. 隐藏节点方法的重点比较

### 5.1 核心文献对照表

| 文献 | 主要机制 | 量测/先验与隐藏节点条件 | 适合综述承担的作用 |
| --- | --- | --- | --- |
| [R038 Deka2016](#r038) | 叶端电压方差、三元组关系、自底向上恢复 | 叶端统计、候选边及阻抗，隐藏节点度数>2 | 早期仅末端量测的隐藏拓扑路线 |
| [R044 Park2018](#r044)、[R055扩展稿](#r055) | V/P/Q交叉矩→电气距离→RG | 全叶端可测；内部可有未知注入；跨节点注入不相关、相关二阶矩非退化、隐点度≥3 | 无需给定隐藏点数量和内部阻抗的关键基线 |
| [R074 Zhang2022](#r074) | 离散电压潜树、结构搜索、局部EM与BIC | 用户历史电压；未知隐节点；统计树近似 | voltage-only潜树方案，应重点比较 |
| [R079 Pengwah2022](#r079) | 物理约束灵敏度、RG与回溯 | 用户V/I/功率因数；零注入内部点；恒定源电压 | 普通表计与Steiner树直接方案 |
| [R084 Flynn2023](#r084) | 联合拟合源电压变化和灵敏度，改进RG | 端点SM；在分组中创建新父节点 | 放松恒定源电压条件 |
| [R089 Pengwah2024](#r089) | SM子树＋interval用户定位及补分支 | 混合高低频量测；零注入内部点；interval模型另有近似 | 部分SM覆盖的具体信息契约 |
| [R090 Fang2024](#r090) | 显式三相耦合灵敏度、源电压、RG/回溯 | 单相用户V/I/功率因数及符号；零注入内部点 | 三相中性线耦合与隐藏树直接基线 |
| [R096 Guo2025](#r096) | 电压平方P/Q灵敏度分级及插点，后续参数/控制 | 用户V/P/Q；零负荷内部点；省略度1/2点，近似跨相灵敏度为0 | 近期低压用户量测方案 |
| [R015 Feng2026](#r015) | TLM优化＋双向层次聚类＋自适应搜索 | 端点V/I幅值与功率因数、电缆库和软结构先验 | 不同电缆参数与隐藏结构联合估计 |
| [R041 Soumalas2017](#r041) | 灵敏度→距离→Prüfer树层析 | 叶端P/Q/V；零注入内部点；源电压及离散距离条件 | 早期物理距离路线 |
| [R024 Ahmed/Lampe2013](#r024) | PLC超分辨测距＋树层析 | 叶端modem；内部可不装modem | 额外测距信息帮助内部结构恢复 |
| [R020 Cavraro2018](#r020) | 逆变器探测灵敏度与公共路径分组 | 全叶端探测；仅部分电压时恢复可辨分支约化树 | 主动控制位置和观测覆盖的比较 |
| [R006 Hotz2022](#r006) | EV充电启停阶跃→公共路径阻抗→插点 | wallbox协调V/I、足够事件、平衡等值等 | EV测量用于隐藏拓扑的直接先例 |
| [R109安全评估2026](#r109) | TLM迭代拟合和层次聚类，再做触电模拟 | 叶端数据、径向树及建模近似 | 连接隐藏树恢复与安全应用 |
| [SUP01 Anguluri2022卷](#sup01) | 电压逆协方差的结构化稀疏/低秩分解 | 幅值＋相角；隐藏点不能相邻或是叶点；注入分布条件 | 隐藏节点空间条件与相量信息路线 |

### 5.2 潜树概率模型：最接近“仅电压＋未知隐点”的比较对象

R074通过用户电压的离散统计分布建立潜树，使用结构算子产生候选、EM估参，再由BIC选择模型。它直接回应隐藏点数量和连接未知的问题，综述应详细介绍其输入、树模型近似和搜索流程。所读§II-B已说明一跳依赖近似与实际更远依赖的差异；具体证据定位见[R074](#r074)。

SUP02的官方摘要报告潜树、聚类搜索和BIC；当前只能将其作为存在该路线及后续发展的证据。与R074的精确算子、量测、复杂度或保证差异需补全文后判断。

### 5.3 灵敏度与加性树距离：先估电气关系，再恢复内部点

这类方法将末端输入输出映射为公共路径量或成对电气距离，再用RG、层次聚类或树层析恢复内部结构。需要分别检查两个环节：

1. **电气量的估计条件**：可用V/P/Q/I、源电压、相别、三相耦合、噪声和参数先验。
2. **树重建条件**：距离是否满足所用加性关系、隐藏节点类型和度数、阈值、短弱支路、候选回溯或聚类规则。

Park的方法明确以跨节点二阶统计条件隔离内部注入；Pengwah/Fang/Guo等直接恢复零注入接点，但分别处理不同的源电压、量测和三相条件。因此“可处理hidden”应细化为“在所列条件下恢复哪类内部点”。

### 5.4 统计独立、零协方差与回归秩条件要分开

以下为实际阅读中的重要校正：

- **R044/R055**：跨节点P/Q注入不相关是正式模型假设；联合R/X交叉矩求解还涉及二阶矩非退化条件。
- **R079**：原文使用“independent load profiles”，未将其形式化为统计独立、零协方差或设计矩阵满秩。应保留其原文范围。
- **R090**：原文给出伪逆/约束最小二乘求解，所读正文未明确列出满秩或同相/正交电流不共线条件。
- **R096**：用多时刻[P;Q;参考电压]回归灵敏度；“不同注入独立”的偏导表述不能直接转写为统计分布独立条件。

**本文件的代数分析示例**：令 $S_P,S_Q$ 分别表示电压对有功、无功注入的灵敏度。若样本中始终满足 $Q=\kappa P$，线性响应中的两项可合并为 $(S_P+\kappa S_Q)P$；仅凭这组回归数据通常不足以分别识别 $S_P$ 与 $S_Q$。这个判断针对系数分离；物理约束或额外信息可能进一步区分参数，拓扑也可能只依赖某个可识别组合。因此不能仅由共线性直接宣布某篇方法的拓扑恢复失败。各篇原文报告与本文件判断在逐篇记录中分列。

### 5.5 主动探测、测距与结构分解：比较新增信息的来源

PLC提供传播距离，逆变器提供已知P/Q扰动，Wallbox提供充电事件的电流阶跃，结构分解方法使用幅值和相角的联合协方差。它们改变了可用信息，不应只按“额外设备贵”归为一类。

R020在部分电压观测下明确讨论可辨内部点和约化树；R049/R056的主保证则采用全节点电压，观测条件要按具体论文说明。SUP01的符号关系定理以正确分解分量为前提，其结论仍把分解恢复的理论条件列为未来工作；应把结构关系与算法整体保证分别报告。

### 5.6 外部补检后需要加入的比较

- **非零隐藏注入**：在Park之外，Li2021的FOLS/RG、Benzerga和Marulli2021的低压部分量测方法也值得比较；前者采用hidden负荷相对平稳条件，后两者分别使用地理/RX先验或隐藏点分隔等条件。[SUP03](#sup03)、[SUP04](#sup04)、[SUP05](#sup05)
- **约化后恢复原树**：Yuan等在零注入、树结构、hidden度数等条件下由约化导纳反推物理内部结构，并允许hidden相邻。应将其与仅输出约化模型的方法分开。[SUP08](#sup08)
- **近期末端数据与异步问题**：马尚2024、Zhang2026和Gao2026分别提供免阈值RG、末端幅值/功率、异步AMI的直接路线；本轮对应证据为摘要或预览，详细条件仍待正文。[SUP09](#sup09)、[SUP10](#sup10)、[SUP11](#sup11)
- **度2恢复的额外先验**：Mohammadi2025采用相同且已知段电阻，其度2节点处理与未知串联阻抗的等效性问题有不同信息前提。[SUP12](#sup12)
- **工程先验和数据质量**：GIS与幅值融合、精度/采样敏感性研究适合补一般方法段，不能直接算作纯端点未知节点算法。[SUP13](#sup13)、[SUP14](#sup14)

<a id="review-ev"></a>

## 6. EV与需求响应文献的使用方式

EV目录主要为两处论证服务：局部低压约束的重要性，以及可用激励/控制的行为条件。除Wallbox等直接研究外，价格模型、CRF依赖图、潜类别用户和图代理模型的对象是需求行为，不计为隐藏电气接线点算法。

### 6.1 优先保留的EV文献

| 记录 | 可支持的论述 | 证据范围 |
| --- | --- | --- |
| [R144](#r144) | TOU可能诱导同步充电，局部容量问题与单车时移目标不同 | 现场充电数据＋虚拟变压器聚合评估 |
| [R146](#r146) | 金钱奖励可改变充电时机，激励撤除后的行为也需考虑 | 随机现场实验 |
| [R137](#r137) | 用户先选择服务，再由站端调度；行为模型可由随机价格实验估计 | 行为参数来自现场；后续联合优化改善由模拟评估 |
| [R148](#r148) | 价格相关的随机响应及闭环价格更新 | 理论模型与算法，稳定/近最优结果采用相应条件 |
| [R138](#r138) | 用户选择、随机到达与资源超限风险 | 概率界依赖到达和分布模型 |
| [R123](#r123) | 在线学习价格响应、用户统计依赖和随机充电选择 | 条件随机场与在线更新模型 |
| [R152](#r152) | 容量收费、调速和时移的不同机制 | 随机现场响应；聚合容量采用合成约束 |
| [R006](#r006) | 充电事件可直接提供拓扑信息 | LVDG隐藏树辨识论文，应放入方法主线 |

### 6.2 三种控制条件

1. **直接功率调度**：可决定设备启停或功率，但仍有设备和运行约束。
2. **用户选合同后调度**：用户先选服务或期限，运营者在合同内执行。
3. **价格间接诱导**：用户自主响应，实际注入需观测或建模，不能把价格数值直接等同于已实现功率。

若本文利用EV进行主动辨识，应说明哪一种控制权成立、哪些注入被实际量测、响应分布如何进入设计，以及最终拓扑结论依赖什么观测。价格响应造成的共同时间变化还需要与被动方法的注入统计条件相对照。

<a id="review-gaps"></a>

## 7. 研究缺口与贡献表述

### 7.1 候选问题及其已有覆盖

| 候选比较问题 | 已有文献覆盖 | 确定本文缺口时仍需回答 |
| --- | --- | --- |
| 隐藏节点数量和位置未知 | R074、R044/R055、R079等 | 本文新增的是节点类型、信息条件、可靠性还是求解能力？ |
| 仅用户量测 | R074、R079、R090、R096等 | 具体仅有哪些量？是否全用户都有？是否要求高频同步？ |
| 源电压变化 | R084、R090等 | 本文如何处理源电压，与相同输入的基线相比改善什么？ |
| 三相不平衡和耦合 | R090、R096采用不同近似；R010提供端口模型 | 是否显式保留跨相/中性线？恢复的是哪种结构？ |
| 部分SM覆盖 | R089 | 其他用户是否仍有低频电量？完全缺记录情形如何定义？ |
| 内部未知注入 | R044/R055、SUP01 | 本文是否同样允许未知负荷，其统计/分隔条件是什么？ |
| 噪声、短弱支路和缺失 | 多篇分别报告约束、正则、回溯或模拟 | 区分有限实验、参数误差界和结构恢复保证；本文提供哪种证据？ |
| EV主动信息 | R006及一般逆变器探测路线 | 本文是否引入价格自主响应、部分控制或不同观测条件？ |
| 最优性/正确性证据 | 原文定理、启发式搜索和有限候选评估分别存在 | 求解最优、模型拟合最优和真实拓扑可识别性分别如何支撑？ |

### 7.2 建议改写的表述

| 容易过强的写法 | 更准确的写法 |
| --- | --- |
| 传统高压方法不适用于低压配电网 | 相关方法采用DC/同步相量动态模型或已知候选线路，末端无相角及未知内部节点的场景需要调整模型与信息条件。 |
| 现有低压方法无法辨识隐藏节点 | 已有潜树、树距离、主动层析和结构分解方法恢复不同类型的内部节点，其覆盖、注入和物理假设各异。 |
| 现有方法设备要求高 | 具体指出需要μPMU相角、PLC测距、特征电流收发或可控逆变器；普通表计方法另比较量测项目、采样和同步。 |
| 首次用EV辨识低压拓扑 | 已有Wallbox充电事件方法；创新应落到实际控制/观测条件或已验证能力。 |
| 低残差或全局MILP最优证明拓扑正确 | 给定模型内的拟合/求解结论与物理可识别性分别论证，并写清替代结构域和误差预算。 |
| 弱标签或BIC分数给出拓扑正确概率 | 报告其评分、融合或模型选择对象；概率解释需要对应模型或校准依据。 |

### 7.3 最值得优先补齐的证据

- 与本文输入、隐藏节点类型和恢复目标最接近的基线，而非只选设备需求明显不同的方法。
- 对使用“独立”“满秩”“缺测”“未知参数”等词的明确含义与原文定位。
- 各项理论结论的假设和量词范围；仿真、实测、启发式候选评估分别记录。
- 核心论文最终发表版与所用作者稿之间的分页/方法差异，以及SUP02、R086的全文。
- 本文各贡献的实际状态：已推导、已实现、已验证分别说明，不用本次文献整理代替实验支持。

<a id="review-draft"></a>

## 8. 可改写进论文的综述段落

以下为基于本次阅读的中文组织示例，引用ID对应后文逐篇记录。最终稿应按目标期刊格式替换引用，并根据本文实际信息条件调整最后一段。

### 8.1 重要性与任务边界

> 分布式电源、电动汽车及灵活负荷逐步进入低压配电网末端，使运行分析需要进一步描述用户之间的电气连接和局部相互影响。拓扑及线路参数为潮流计算、三相电压分析、故障定位和位置相关的运行约束提供基础[R087](#r087)、[R010](#r010)、[R109](#r109)。在用户侧量测逐步普及而内部接线设备缺乏监测的场景中，关键问题是如何由可用末端数据恢复内部连接结构。

> 拓扑研究涉及不同的先验和输出目标。已有方法包括利用同步相量辨识动态等值网络、在已知线路基础设施上判断开关或停运状态，以及在给定节点集合上联合估计连边和参数[R034](#r034)、[R014](#r014)、[R059](#r059)、[R058](#r058)。当内部节点数量与连接均未知时，还需要考虑潜在物理节点的生成和可辨识性。因此，下文按照可用量测、隐藏节点类型及恢复对象比较相关方法。

### 8.2 一般方法到隐藏节点方法

> 利用常规计量数据的研究主要从能量守恒、统计依赖和物理输入输出关系提取结构信息。PCA方法由电量守恒估计电表网络的约束子空间，图模型和相关分析根据电压统计关系构建连接，非相量回归则由P/Q/V等量测估计导纳或灵敏度[R045](#r045)、[R022](#r022)、[R116](#r116)、[R058](#r058)、[R066](#r066)。这些工作的节点覆盖与结构输出不同；对于内部无量测节点，应进一步检查算法是否包含潜节点插入或物理内部结构恢复步骤。

> 针对隐藏节点，已有研究形成了概率潜树和物理树重建等路线。Zhang等以用户历史电压建立潜树，通过结构搜索、EM和BIC选择未知内部节点及连接[R074](#r074)。Park等由叶端电压与功率交叉统计构建加性电气距离，利用递归分组联合恢复内部节点和线路参数[R044](#r044)、[R055](#r055)。Pengwah等及后续工作以用户表计数据估计灵敏度，并分别处理源电压变化、混合分辨率量测和三相耦合[R079](#r079)、[R084](#r084)、[R089](#r089)、[R090](#r090)；Guo等通过敏感性分级逐层插入内部节点，Feng等结合TLM与层次聚类联合估计隐藏拓扑和参数[R096](#r096)、[R015](#r015)。

### 8.3 新增信息与研究定位

> 主动测距和设备扰动提供了不同的信息来源。PLC层析利用叶端传播距离恢复未装modem的分支结构，逆变器探测利用已知注入与电压响应识别公共路径，Wallbox方法则利用充电启停的电流阶跃估计共享线路阻抗[R024](#r024)、[R020](#r020)、[R006](#r006)。结构分解方法进一步利用电压幅值和相角的联合统计恢复隐藏节点连接，同时采用隐藏节点分隔等条件[SUP01](#sup01)。这些方法的量测、控制权限与隐藏节点假设应分别比较。

> 本文定位可采用以下句式补齐实际内容：本文考虑【具体末端量测】可得、【哪些节点/线路信息】未知的低压配电场景，目标是恢复【最小树/其他明确结构】。相对于【输入和目标最接近的工作】，本文在【实际改变的条件或新增能力】上进行研究，并以【推导/算法实现/既有验证的具体范围】支撑相应结论。

最后一个段落是研究定位模板，其方括号内容仍需由本文的实际方法和证据填写。

<a id="review-records"></a>

## 9. 全部文献逐篇记录

后续记录覆盖142个初步归并组，并保留对应的154条原始记录编号。每项依次给出书目、方法、输入、隐藏节点/任务、条件、综述用途及实际阅读范围。作者稿与正式版的差异、明确的原文条件和本文件的额外判断在对应条目单列。

### 9.1 文献索引

| 编号 | 文献题名 | 记录日期/年份 | 阅读层级 |
| --- | --- | --- | --- |
| [R001](#r001) | Bayesian 𝑇-optimal discriminating designs | 未记录 | 正文定向阅读 |
| [R002](#r002) | Grid Topology Identiﬁcation via Distributed Statistical Hypothesis Testing | 未记录 | 正文定向阅读 |
| [R003](#r003) | Power Theft Localization using Voltage Measurements from Distribution Feeder Nodes | 未记录 | 正文定向阅读 |
| [R004](#r004) | Signal Injection-Based Topology Identification for Low-Voltage Distribution Networks Considering Missing Data | 2024（PDF首页核对） | 正文定向阅读 |
| [R006](#r006) | TOPOLOGY ESTIMATION IN LOW VOLTAGE GRIDS USING WALLBOX CHARGING DATA RECORDINGS | 2022（CIRED作者存档） | 正文定向阅读 |
| [R007](#r007) | Waypoint-based Topology Inference | 未记录 | 摘要及概述筛选 |
| [R008](#r008) | Efficient policy adaptation for voltage control under unknown topology changes | 01/2027 | 正文定向阅读 |
| [R009](#r009) | A topological model-driven optimization approach for detecting and locating electricity fraud in low-voltage networks | 02/2027 | 正文定向阅读 |
| [R010](#r010) | Smart meter data-driven dynamic operating envelopes for DERs | 04/2025 | 正文定向阅读 |
| [R011](#r011) | False data injection attacks against state estimation in electric power grids | 05/2011 | 摘要及概述筛选 |
| [R012](#r012) | Optimal experimental design: Formulations and computations | 07/2024 | 摘要及概述筛选 |
| [R013](#r013) | A note on the metric properties of trees | 08/1974 | 正文定向阅读 |
| [R014](#r014) | An Optimization-Based Topology Error Detection Method for Power System State Estimation | 08/2022 | 正文定向阅读 |
| [R015](#r015) | Adaptive optimal-weight method for topology identification and line parameter estimation of large-scale low-voltage distribution grids | 08/2026 | 正文定向阅读 |
| [R016](#r016) | A review on topology identification methods and applications in distribution networks | 09/2024 | 正文定向阅读 |
| [R017](#r017) | Data-physics hybrid-driven two-stage topology identification for active distribution networks with soft open points | 09/2026 | 正文定向阅读 |
| [R018](#r018) | Tractable Bayesian learning of tree belief networks | 1/2006 | 正文定向阅读 |
| [R019](#r019) | Distribution Network Topology Identification Using Asynchronous Transformer Monitoring Data | 1/2023 | 正文定向阅读 |
| [R020](#r020) | Graph Algorithms for Topology Identification Using Power Grid Probing | 10/2018 | 正文定向阅读 |
| [R021](#r021) | A unified framework for solving a general class of conditional and robust set-membership estimation problems | 11/2014 | 正文定向阅读 |
| [R022](#r022) | Full-Scale Distribution System Topology Identification Using Markov Random Field | 11/2020 | 正文定向阅读 |
| [R023](#r023) | Network Tomography Based on Additive Metrics | 12/2011 | 正文定向阅读 |
| [R024](#r024) | Power Line Communications for Low-Voltage Power Grid Tomography | 12/2013 | 正文定向阅读 |
| [R026](#r026) | Identification and ranking of customers with energy theft using hypothesis tests | 12/2023 | 正文定向阅读 |
| [R027](#r027) | The EM algorithm for graphical association models with missing data | 2/1995 | 摘要及概述筛选 |
| [R028](#r028) | The Variational Bayesian EM Algorithm for Incomplete Data: With Application to Scoring Graphical Model Structures | 2003-7-3 | 正文定向阅读 |
| [R029](#r029) | Minimizing Effective Resistance of a Graph | 2008 | 正文定向阅读 |
| [R030](#r030) | Optimal discrimination designs | 2009-8-1 | 摘要及概述筛选 |
| [R031](#r031) | Learning Latent Tree Graphical Models | 2010-09-14 | 正文定向阅读 |
| [R032](#r032) | Kron Reduction of Graphs with Applications to Electrical Networks | 2011-02-15 | 正文定向阅读 |
| [R033](#r033) | The Balanced Minimum Evolution Problem. | 2012-04-01 | 正文定向阅读 |
| [R034](#r034) | Topology identification for dynamic equivalent models of large power system networks | 2013-06 | 正文定向阅读 |
| [R035](#r035) | Non-Technical Loss Detection Using State Estimation and Analysis of Variance | 2013-08 | 正文定向阅读 |
| [R036](#r036) | Blind topology identification for power systems | 2013-10 | 正文定向阅读 |
| [R037](#r037) | Robust $T$-optimal discriminating designs | 2013-8-1 | 正文定向阅读 |
| [R038](#r038) | Learning Topology of Distribution Grids using only Terminal Node Measurements | 2016-08-17 | 正文定向阅读 |
| [R039](#r039) | Identifying Topology of Power Distribution Networks Based on Smart Meter Data | 2016-09-09 | 正文定向阅读 |
| [R040](#r040) | A general framework for updating belief distributions | 2016-11 | 正文定向阅读 |
| [R041](#r041) | A data driven approach to distribution network topology identification | 2017-06 | 正文定向阅读 |
| [R042](#r042) | Power grid probing for load learning: Identifiability over multiple time instances | 2017-12 | 正文定向阅读 |
| [R043](#r043) | Electricity Theft Detection and Localization in Grid-Tied Microgrids | 2018-05 | 正文定向阅读 |
| [R044](#r044) | Exact Topology and Parameter Estimation in Distribution Grids with Minimal Observability | 2018-06 | 正文定向阅读 |
| [R045](#r045) | Identifying Topology of Low Voltage Distribution Networks Based on Smart Meter Data | 2018-09 | 正文定向阅读 |
| [R047](#r047) | Network Learning via Multiagent Inverse Transportation Problems | 2018-12 | 正文定向阅读 |
| [R048](#r048) | PaToPaEM: A Data-Driven Parameter and Topology Joint Estimation Framework for Time Varying System in Distribution Grids | 2018-12-17 | 正文定向阅读 |
| [R049](#r049) | Inverter Probing for Power Distribution Network Topology Processing | 2019-02-20 | 正文定向阅读 |
| [R050](#r050) | Bayesian topology identification of linear dynamic networks | 2019-03-14 | 正文定向阅读 |
| [R052](#r052) | Smart Inverter Grid Probing for Learning Loads: Part I — Identifiability Analysis | 2019-09 | 正文定向阅读 |
| [R053](#r053) | Smart Inverter Grid Probing for Learning Loads: Part II—Probing Injection Design | 2019-09 | 正文定向阅读 |
| [R054](#r054) | Universal inference | 2020 | 正文定向阅读 |
| [R055](#r055) | Learning with End-Users in Distribution Grids: Topology and Parameter Estimation | 2020-03-01 | 正文定向阅读 |
| [R056](#r056) | An MILP Approach for Distribution Grid Topology Identification using Inverter Probing | 2020-04-07 | 正文定向阅读 |
| [R057](#r057) | Detection of False Data Injection Attacks in Smart Grids Based on Graph Signal Processing | 2020-06 | 正文定向阅读 |
| [R058](#r058) | Topology Identification and Line Parameter Estimation for Non-PMU Distribution Network: A Numerical Method | 2020-09 | 正文定向阅读 |
| [R059](#r059) | Line Outage Identification Based on AC Power Flow and Synchronized Measurements | 2020-11-18 | 正文定向阅读 |
| [R060](#r060) | A Method of Low Voltage Topology Identification | 2020-12 | 正文定向阅读 |
| [R061](#r061) | Learning Latent Variable Dynamic Graphical Models by Confidence Sets Selection | 2020-12 | 正文定向阅读 |
| [R062](#r062) | Power System Network Topology Identification Based on Knowledge Graph and Graph Neural Network | 2021-02-17 | 正文定向阅读 |
| [R063](#r063) | Low-voltage distribution network topology identification method based on characteristic current | 2021-04 | 正文定向阅读 |
| [R064](#r064) | Topology Learning Aided False Data Injection Attack without Prior Topology Information | 2021-07 | 正文定向阅读 |
| [R066](#r066) | Distribution Network Admittance Matrix Estimation With Linear Regression | 2021-09 | 正文定向阅读 |
| [R067](#r067) | An Adaptive-Importance-Sampling-Enhanced Bayesian Approach for Topology Estimation in an Unbalanced Power Distribution System | 2021-10-18 | 正文定向阅读 |
| [R068](#r068) | Joint Topology Identification and State Estimation in Unobservable Distribution Grids | 2021-11 | 正文定向阅读 |
| [R069](#r069) | Low-voltage overhead lines topology identification method based on high-frequency signal injection | 2021-11-30 | 正文定向阅读 |
| [R070](#r070) | Bayesian Error-in-Variables Models for the Identification of Power Networks | 2021-12-19 | 正文定向阅读 |
| [R072](#r072) | Time-uniform, nonparametric, nonasymptotic confidence sequences | 2021-4-1 | 正文定向阅读 |
| [R073](#r073) | Topology and Parameter Identification of Distribution Network Using Smart Meter and µPMU Measurements | 2022 | 正文定向阅读 |
| [R074](#r074) | Low-Voltage Distribution Grid Topology Identification With Latent Tree Model | 2022-05 | 正文定向阅读 |
| [R075](#r075) | A general topology identification framework for distribution systems using smart meter and μ-PMU measurements | 2022-07-01 | 正文定向阅读 |
| [R076](#r076) | A Dual-Graph Attention-Based Approach for Identifying Distribution Network Topology | 2022-10 | 正文定向阅读 |
| [R077](#r077) | Research on Topology Recognition Technology Based on Intelligent Measurement Switches | 2022-11-25 | 正文定向阅读 |
| [R078](#r078) | Distribution Network Topology Identification with Graph Transformer Neural Network | 2022-12 | 正文定向阅读 |
| [R079](#r079) | Topology Identification of Radial Distribution Networks Using Smart Meter Data | 2022-12 | 正文定向阅读 |
| [R081](#r081) | Topology and Impedance Identification Method of Low-Voltage Distribution Network Based on Smart Meter Measurements | 2022-4-29 | 正文定向阅读 |
| [R082](#r082) | Power Distribution System Characterization With Active Probing: Real-World Testing and Analysis | 2023 | 正文定向阅读 |
| [R083](#r083) | Distribution System Topology Identification Using Graph Neural Networks | 2023-05 | 正文定向阅读 |
| [R084](#r084) | An Improved Algorithm for Topology Identification of Distribution Networks Using Smart Meter Data and Its Application for Fault Detection | 2023-09 | 正文定向阅读 |
| [R085](#r085) | Online Voltage Regulation With Minimum Disturbance for Distribution Grid Without System Knowledge | 2024 | 正文定向阅读 |
| [R086](#r086) | Topology Identification of Distribution Networks Based on Physics-Informed Latent Graph Attention Network | 2024 | 摘要及概述筛选 |
| [R087](#r087) | Learning Distribution Grid Topologies: A Tutorial | 2024-01 | 正文定向阅读 |
| [R088](#r088) | Learning on Multimodal Graphs: A Survey | 2024-02-07 | 正文定向阅读 |
| [R089](#r089) | Topology Identification of Distribution Networks With Partial Smart Meter Coverage | 2024-04 | 正文定向阅读 |
| [R090](#r090) | Three-phase voltage sensitivity estimation and its application to topology identification in low-voltage distribution networks | 2024-07-01 | 正文定向阅读 |
| [R091](#r091) | Distribution Network Topology Identification Based on CNN and Improved Conformal Prediction | 2024-09 | 正文定向阅读 |
| [R092](#r092) | Online Learning for Robust Voltage Control Under Uncertain Grid Topology | 2024-09 | 正文定向阅读 |
| [R093](#r093) | Graph neural networks for multi-view learning: a taxonomic review | 2024-10-21 | 正文定向阅读 |
| [R095](#r095) | Significant inference and confidence sets for graphical models | 2025 | 正文定向阅读 |
| [R096](#r096) | Topology and Parameter Identification of Three-Phase Low-Voltage Distribution Network and Voltage Control Strategy Based on User Smart Meter Measurements: A Data-Driven Approach | 2025 | 正文定向阅读 |
| [R097](#r097) | A power grid topology detection method based on edge graph attention neural network | 2025-02-01 | 正文定向阅读 |
| [R098](#r098) | Power Distribution Network Topology Detection Using Dual-Graph Structure Graph Neural Network Model | 2025-03 | 正文定向阅读 |
| [R100](#r100) | Constrained Branching Search for Topology Identification Stream Computing With Lightweight Implementation | 2025-07 | 正文定向阅读 |
| [R101](#r101) | Research on power system topology identification and fault location based on graph neural network | 2025-08 | 正文定向阅读 |
| [R102](#r102) | Fast Distribution Grid Topology Estimation via Subset Sum | 2025-10 | 正文定向阅读 |
| [R103](#r103) | Maximum Likelihood Estimation for Line Parameters in Distribution Grids Based on Expectation Maximization Algorithm | 2025-11 | 正文定向阅读 |
| [R104](#r104) | A Topology Identification Strategy of Low-Voltage Distribution Grids Based on Feature-Enhanced Graph Attention Network | 2025/1 | 正文定向阅读 |
| [R105](#r105) | Utilising Smart-Meter Harmonic Data for Low-Voltage Network Topology Identification | 2025/1 | 正文定向阅读 |
| [R106](#r106) | Autonomous Topology Identification in Distribution Networks Using Smart Meter Data | 2026 | 正文定向阅读 |
| [R107](#r107) | Confidence-Aware Topology Identification in Low-Voltage Distribution Networks: A Multi-Source Fusion Method Based on Weakly Supervised Learning. | 2026-03-15 | 正文定向阅读 |
| [R108](#r108) | Error Bounds for Radial Network Topology Learning From Quantized Measurements | 2026-05 | 正文定向阅读 |
| [R109](#r109) | Electric Shock Simulation and Risk Assessment in Low-Voltage Distribution Networks Under Unknown Topology: A Two-Stage Approach Based on Smart Meter Data. | 2026-06-01 | 正文定向阅读 |
| [R110](#r110) | Anomaly detection and topology identification of distribution network based on conditional variational autoencoder | 2026-06-29 | 正文定向阅读 |
| [R111](#r111) | Adaptive Physics-Informed Data-Driven Topology Identification for Distribution Networks | 2026-08 | 正文定向阅读 |
| [R112](#r112) | Topology Identification in Distribution Systems Using Line Current Sensors: An MILP Approach | 3/2020 | 正文定向阅读 |
| [R114](#r114) | Topology Identification in Distribution Networks using Harmonic Synchrophasor Measurements | 5/2021 | 正文定向阅读 |
| [R115](#r115) | Simultaneous Robust State Estimation, Topology Error Processing, and Outage Detection for Unbalanced Distribution Systems | 5/2023 | 正文定向阅读 |
| [R116](#r116) | Distributed Energy Resources Topology Identification via Graphical Modeling | 7/2017 | 正文定向阅读 |
| [R117](#r117) | Guaranteed Phase & Topology Identification in Three Phase Distribution Grids | 7/2021 | 正文定向阅读 |
| [R118](#r118) | Distribution Network Topology Identification Using Smart Meter Data and Considering the Same-Bus-Different-Feeder Condition | 8/2023 | 正文定向阅读 |
| [R120](#r120) | Graphical Models in Meshed Distribution Grids: Topology Estimation, Change Detection & Limitations | 9/2020 | 正文定向阅读 |
| [R121](#r121) | Dynamic incentives for electric vehicles charging at supermarket stations: Causal insights on demand flexibility | 12/2025 | 正文定向阅读 |
| [R122](#r122) | An Optimized EV Charging Model Considering TOU Price and SOC Curve | 2012-3 | 正文定向阅读 |
| [R123](#r123) | Real-Time Load Elasticity Tracking and Pricing for Electric Vehicle Charging | 2015-5 | 正文定向阅读 |
| [R124](#r124) | Dynamic Pricing and Distributed Energy Management for Demand Response | 2016-02-29 | 正文定向阅读 |
| [R125](#r125) | Modeling demand flexibility of electric vehicles | 2017-10 | 正文定向阅读 |
| [R126](#r126) | Risk-Sensitive Learning and Pricing for Demand Response | 2018-06-18 | 正文定向阅读 |
| [R127](#r127) | Control of Charging of Electric Vehicles Through Menu-Based Pricing | 2018-11 | 正文定向阅读 |
| [R128](#r128) | Stochastic Dynamic Pricing for EV Charging Stations With Renewable Integration and Energy Storage | 2018-3 | 正文定向阅读 |
| [R129](#r129) | Behavioral Modeling of Electric Vehicles Using Price Elasticities | 2018-5 | 正文定向阅读 |
| [R130](#r130) | Supporting power balance in Microgrids with Uncertain Production using Electric Vehicles and Indirect Control | 2019 | 正文定向阅读 |
| [R131](#r131) | A Distributed Online Pricing Strategy for Demand Response Programs | 2019-01 | 正文定向阅读 |
| [R132](#r132) | Inverse optimization with kernel regression: Application to the power forecasting and bidding of a fleet of electric vehicles | 2021-10 | 正文定向阅读 |
| [R133](#r133) | Optimal Pricing of Vehicle-to-Grid Services Using Disaggregate Demand Models | 2021-2-19 | 正文定向阅读 |
| [R134](#r134) | Data-Driven Approach for Analyzing Spatiotemporal Price Elasticities of EV Public Charging Demands Based on Conditional Random Fields | 2021-9 | 正文定向阅读 |
| [R135](#r135) | Demand response model identification and behavior forecast with OptNet | 2022-6-28 | 正文定向阅读 |
| [R136](#r136) | Field Experimental Evidence on the Effect of Pricing on Residential Electricity Conservation | 2023-12 | 正文定向阅读 |
| [R137](#r137) | Learning and optimizing charging behavior at PEV charging stations: Randomized pricing experiments, and joint power and price optimization | 2023-12 | 正文定向阅读 |
| [R138](#r138) | Resource aware pricing for electric vehicle charging | 2023-2 | 正文定向阅读 |
| [R139](#r139) | Unraveling the effect of electricity price on electric vehicle charging behavior: A case study in Shenzhen, China | 2024-11 | 正文定向阅读 |
| [R140](#r140) | Optimal Vehicle Charging in Bilevel Power-Traffic Networks via Charging Demand Function | 2024-5 | 正文定向阅读 |
| [R141](#r141) | Learning and Optimization for Price-Based Demand Response of Electric Vehicle Charging | 2024-7-10 | 正文定向阅读 |
| [R143](#r143) | The Impact of Dynamic Prices on Electric Vehicle Public Charging Demand: Evidence from a Nationwide Natural Field Experiment | 2025-12 | 正文定向阅读 |
| [R144](#r144) | Electric Vehicles and the Energy Transition: Unintended Consequences of Time-of-Use Pricing | 2025-12-1 | 正文定向阅读 |
| [R145](#r145) | V2G for Frequency Regulation Service: A Stackelberg Game Approach Considering Endogenous Uncertainties | 2025-2 | 正文定向阅读 |
| [R146](#r146) | Show Me the Money! A Field Experiment on Electric Vehicle Charge Timing | 2025-5-1 | 正文定向阅读 |
| [R147](#r147) | The impacts of rate surge on electric vehicle charging behaviors: Evidence from California | 2025-8 | 正文定向阅读 |
| [R148](#r148) | Demand Response Under Stochastic, Price-Dependent User Behavior | 2026-03-16 | 正文定向阅读 |
| [R149](#r149) | V2G Regulation in Distribution Networks With Decision-Dependent Uncertainties: A Chance-Constrained Learning Method | 2026-2 | 正文定向阅读 |
| [R150](#r150) | Learning the Price Response of Spatiotemporal EV Charging Flow: A Graph-Attentive Surrogate Model | 2026-4 | 正文定向阅读 |
| [R151](#r151) | A Comprehensive Review of Consumer Models in Price-Based Demand Response and Their Applications to Electric Vehicles | 2026-6-11 | 正文定向阅读 |
| [R152](#r152) | Reducing peak load from electric vehicles through grid tariffs: Evidence from a randomized field experiment | 2026-8 | 正文定向阅读 |
| [R153](#r153) | Dynamic Behavior Modeling of Price-Responsive Flexible Loads: A Deep Learning-Parameterized Inverse Optimization Approach | 2026-9 | 正文定向阅读 |
| [R154](#r154) | Robust scheduling of electric vehicle charging stations with price-based demand response under decision-dependent uncertainty | 2026-9 | 正文定向阅读 |

### 9.2 隐藏节点专题及直接对照（23组）

<a id="r031"></a>

#### R031 Learning Latent Tree Graphical Models

**书目信息**：Myung Jin Choi; Vincent Y. F. Tan; Animashree Anandkumar; Alan S. Willsky。日期/年份：2010-09-14。

**来源**：[DOI：10.48550/arXiv.1009.2722](https://doi.org/10.48550/arXiv.1009.2722)。

- **方法**：以观测变量的成对信息距离判断亲子/兄弟关系，递归分组RG插入隐变量；CLGrouping先建Chow–Liu树，再局部RG以缩小搜索范围。
- **输入与设备**：观测变量的iid样本；Gaussian情形使用−log|相关系数|，离散情形使用成对概率矩阵行列式距离。
- **隐藏节点与辨识对象**：恢复最小潜树，允许内部观测节点；隐节点度数至少3、相邻变量既不独立也不完全相关。
- **条件与适用边界**：统计潜树基础；电压统计满足物理树Markov性质/加性距离需要另行建立，不能直接套用为配电网保证。
- **综述中的位置**：数学基础

**阅读依据**：正文定向阅读；部分正文：模型、可辨识条件、信息距离与关系判别；未逐页阅读全部42页。

**原文定位**：PDF pp.6–10，§2.3 C1–C2、Proposition1、§3.1–3.2/Lemma4；目录与摘要。

<a id="r038"></a>

#### R038 Learning Topology of Distribution Grids using only Terminal Node Measurements

**书目信息**：Deepjyoti Deka; Scott Backhaus; Michael Chertkov。日期/年份：2016-08-17。

**来源**：[DOI：10.48550/arXiv.1608.05031](https://doi.org/10.48550/arXiv.1608.05031)。

- **方法**：利用叶节点电压差方差及三元组关系，自底向上识别共同父节点与隐藏中间层，做已知候选边集中的拓扑选择。
- **输入与设备**：全部叶端电压幅值样本、叶端P/Q注入二阶统计、缺失中间节点集合、所有候选边阻抗。
- **隐藏节点与辨识对象**：允许相邻的内部节点均无量测；隐藏节点度数>2，节点数/候选线及阻抗已知。
- **条件与适用边界**：LC-PF/近似LinDistFlow；不同节点P/Q注入不相关；度2隐藏串接节点引起非唯一性。
- **综述中的位置**：早期叶端观测隐藏树方法

**阅读依据**：正文定向阅读；部分正文：模型、假设、三元组关系、算法输入、实验设置与结论。

**原文定位**：PDF pp.2–6，§II–IV、Assumptions1–2、Algorithm1输入、§VI；p.7结论。

<a id="r039"></a>

#### R039 Identifying Topology of Power Distribution Networks Based on Smart Meter Data

**书目信息**：Satya Jayadev P; Nirav Bhatt; Ramkrishna Pasumarthy; Aravind Rajeswaran（预印本首页写法；正式版见R045）。日期/年份：2016-09-09。

**来源**：[DOI：10.48550/arXiv.1609.02678](https://doi.org/10.48550/arXiv.1609.02678)。

- **方法**：利用能量守恒约束对表计能量数据做加权PCA/SVD，恢复约束子空间，再通过图论解释重建电表连接及相别。
- **输入与设备**：变电站/馈线/变压器/用户等各节点的时间对齐电量数据，节点和表计清单已知。
- **隐藏节点与辨识对象**：本方法模型把每个节点视作表计；正式版明确假定所有节点安装smart meters，没有未知数量物理隐藏节点恢复步骤。
- **条件与适用边界**：与R045/R119为同作者同方法的预印本/正式版发展，应合并引用。损耗、计量和时钟误差建模影响PCA。
- **综述中的位置**：能量守恒/PCA方法

**阅读依据**：正文定向阅读；预印本部分正文+正式版问题设定交叉核对。

**原文定位**：预印本PDF pp.1–3；正式版R045 PDF pp.4–5 §III-B–E核对全节点电表假设。

**合并引用**：与[R045](#r045)及其重复记录R119属于同一PCA研究的预印本/正式版，主文合引。

<a id="r044"></a>

#### R044 Exact Topology and Parameter Estimation in Distribution Grids with Minimal Observability

**书目信息**：Sejun Park; Deepjyoti Deka; Michael Chertkov。日期/年份：2018-06。

**来源**：[DOI：10.23919/PSCC.2018.8442881](https://doi.org/10.23919/PSCC.2018.8442881)。

- **方法**：由叶端电压与P/Q注入的交叉二阶矩求逆Laplacian子矩阵，形成电阻/电抗加性距离，再用RG同时恢复隐藏节点和线阻抗。
- **输入与设备**：叶端时间对齐电压幅值、有功/无功注入样本；无需隐藏节点数、内部注入统计或候选阻抗。
- **隐藏节点与辨识对象**：允许内部节点有未知注入且不测量；所有缺失内部节点度数≥3、所有叶端被观测。
- **条件与适用边界**：LC-PF、跨节点注入不相关；各叶端P/Q协方差矩阵需可逆（文中式(5)求逆）；有限样本用容差、会影响分组。
- **综述中的位置**：无需已知隐藏节点数/阻抗的核心原始工作

**阅读依据**：正文定向阅读；部分正文：模型、全部关键假设、距离推导、RG步骤和实验设置。

**原文定位**：PDF pp.2–5，§II–III、Assumptions1–2、式(5)–(8)、Algorithms1–2、实验设置。

**版本与书目说明**：题名一致；Zotero作者拼写Seiun Park/Chcrtkov有误，PDF首页为Sejun Park/Michael Chertkov。

<a id="r045"></a>

#### R045 Identifying Topology of Low Voltage Distribution Networks Based on Smart Meter Data

**书目信息**：Satya Jayadev Pappu; Nirav Bhatt; Ramkrishna Pasumarthy; Aravind Rajeswaran。日期/年份：2018-09。

**来源**：[DOI：10.1109/TSG.2017.2680542](https://doi.org/10.1109/TSG.2017.2680542)。

- **方法**：Pappu等将电量守恒式的约束子空间估计为PCA/SVD问题，再据回归/图结构恢复低压电表树与用户相别；显式讨论损耗与时钟误差。
- **输入与设备**：所有节点电表的15/30分钟电量样本；不同噪声时用误差协方差加权。
- **隐藏节点与辨识对象**：§III-B明确smart meters installed at all nodal points；输出是表计网络，不进行未知中间物理节点的插入。
- **条件与适用边界**：R039为2016预印本，R119与R045是相同正式文献的重复条目；综述合并为Pappu et al., TSG2018一条。
- **综述中的位置**：常规低压能量守恒/PCA对照

**阅读依据**：正文定向阅读；部分正文：导论、PCA、图解释、全节点能量测量及误差模型。

**原文定位**：PDF pp.1–5，§III-A–E，尤其p.4 §III-B全节点电表假设；PCA部分pp.2–3。

**合并引用**：预印本见[R039](#r039)；R119是本正式版重复记录。

**对应原记录编号**：R045, R119。

<a id="r049"></a>

#### R049 Inverter Probing for Power Distribution Network Topology Processing

**书目信息**：Guido Cavraro; Vassilis Kekatos。日期/年份：2019-02-20。

**来源**：[DOI：10.48550/arXiv.1802.06027](https://doi.org/10.48550/arXiv.1802.06027)；[稳定来源](https://doi.org/10.1109/TCNS.2019.2901714)。

- **方法**：扰动智能逆变器有功/无功注入，收集电压响应；利用树Laplacian约束构造拓扑/参数非凸拟合与已知线路状态检测，并给出凸替代求解。
- **输入与设备**：探测节点的已知功率扰动、全部节点电压响应；状态验证另用候选线路集及阻抗。
- **隐藏节点与辨识对象**：主保证需要全节点电压；全部叶端可探测和满秩探测序列足以识别完整树。仅部分电压的隐藏节点恢复不在该主方案保证内。
- **条件与适用边界**：近似潮流；探测期间负荷变化进误差项；全拓扑可辨识定理针对原非凸模型，凸松弛仅数值验证。
- **综述中的位置**：主动探测与设备/覆盖代价

**阅读依据**：正文定向阅读；部分正文：贡献/观测条件、探测模型、拓扑拟合及状态验证条件。

**原文定位**：原文arXiv v3 PDF pp.1–4、p.8 §IV/Theorem2；p.2明确a1/a2条件。

**对应原记录编号**：R049, R051。

<a id="r055"></a>

#### R055 Learning with End-Users in Distribution Grids: Topology and Parameter Estimation

**书目信息**：Sejun Park; Deepjyoti Deka; Scott Backhaus; Michael Chertkov。日期/年份：2020-03-01。

**来源**：[正式发表DOI：10.1109/TCNS.2020.2979882](https://doi.org/10.1109/TCNS.2020.2979882)；[DOI：10.48550/arXiv.1803.04812](https://doi.org/10.48550/arXiv.1803.04812)。

- **方法**：统一两条叶端学习路线：电压统计+已知候选阻抗的三元组搜索；以及V/P/Q交叉矩→电阻/电抗距离→RG联合拓扑参数恢复，并分析样本复杂度。
- **输入与设备**：算法1需叶端V、注入统计、候选边/阻抗；算法3只需叶端V/P/Q样本。
- **隐藏节点与辨识对象**：算法3可恢复未知数目的内部节点，内部注入可未知；全部隐藏节点度数≥3、全部叶端被观测；度2节点合并为Kron等效边。
- **条件与适用边界**：正文Assumptions1–3：跨节点注入不相关、P/Q协方差非退化；Theorem3另要求LC-PF、常数深度、有界阻抗、次Gaussian注入。摘要greater than three与正文≥3有差异，采用正文。
- **综述中的位置**：核心隐藏节点基线及保证边界

**阅读依据**：正文定向阅读；部分正文：两个算法、假设、RG、degree2扩展、样本定理和实验设置；未逐页阅读全文。

**原文定位**：PDF pp.3–8，Assumptions1–3、Algorithms1–3、§IV扩展与Theorem3；p.8实验设置。


**本轮书目补充**：正式版为IEEE TCNS7(3):1428–1440(2020)。本项阅读页码仍指所读作者稿，引用正式版时核对最终分页。

<a id="r056"></a>

#### R056 An MILP Approach for Distribution Grid Topology Identification using Inverter Probing

**书目信息**：Sina Taheri; Vassilis Kekatos; Guido Cavraro。日期/年份：2020-04-07。

**来源**：[DOI：10.48550/arXiv.2004.02370](https://doi.org/10.48550/arXiv.2004.02370)。

- **方法**：每个候选拓扑上先LS拟合线路电阻，再把最小拟合拓扑选择转写为含二元线路变量的MILP；必要时加更强连通/径向性约束。
- **输入与设备**：已知逆变器探测功率和所有节点的电压响应；已知节点清单及候选边集，可用完全图。
- **隐藏节点与辨识对象**：全节点电压被显式采用；不是依靠叶端电压发现未知内部节点的方案。
- **条件与适用边界**：无噪声可辨识条件是所有叶端被探测、探测矩阵满行秩；McCormick连续变量盒界文中未分析推导而用经验界。
- **综述中的位置**：主动优化搜索对照

**阅读依据**：正文定向阅读；部分正文：观测假设、LS与MILP构造、变量界和结论。

**原文定位**：PDF pp.1–5，§II-B/Theorem1、§III–IV、结论；p.2 henceforth all buses，p.4盒界说明。

<a id="r058"></a>

#### R058 Topology Identification and Line Parameter Estimation for Non-PMU Distribution Network: A Numerical Method

**书目信息**：Jiawei Zhang; Yi Wang; Yang Weng; Ning Zhang。日期/年份：2020-09。

**来源**：[DOI：10.1109/TSG.2020.2979368](https://doi.org/10.1109/TSG.2020.2979368)。

- **方法**：非相量两步辨识：先用P/Q/V的线性回归估计导纳和初始拓扑，再用专门Newton–Raphson迭代联立AC方程精化参数、恢复相角并修正拓扑。
- **输入与设备**：预先给定节点集的各节点P、Q、V幅值多时刻样本，无需直接测相角。
- **隐藏节点与辨识对象**：方法优化在已知节点集上；所读问题设定未提供未知数量隐藏接线节点的生成/恢复机制。
- **条件与适用边界**：初值和非线性迭代稳定性影响精化；IEEE33/123环网仿真支持该数据范围，避免概括为任意低压隐藏网适用。
- **综述中的位置**：低设备要求的非PMU参数/拓扑辨识

**阅读依据**：正文定向阅读；部分正文：框架、回归方程、NR精化及实验输入。

**原文定位**：PDF pp.2–5，§II–IV，p.9测试设置。

<a id="r068"></a>

#### R068 Joint Topology Identification and State Estimation in Unobservable Distribution Grids

**书目信息**：Hazhar Sufi Karimi; Balasubramaniam Natarajan。日期/年份：2021-11。

**来源**：[DOI：10.1109/TSG.2021.3102179](https://doi.org/10.1109/TSG.2021.3102179)。

- **方法**：将稀疏基下的压缩感知状态恢复与开关状态联合写成MINLP，再给MILP和凸松弛，并用开关拓扑的Markov先验改善稀疏量测情形。
- **输入与设备**：已知量测/投影矩阵下的P/Q/复电压压缩观测、已知稀疏化基、固定导纳与可切换支路导纳。
- **隐藏节点与辨识对象**：处理已知网络节点的不可观状态；拓扑变量是既有开关状态，Y=Ybar+ΣsiYi，节点/候选线路与参数框架已知。
- **条件与适用边界**：30%数据案例对应文中压缩量测设置，不能直接当作30%用户电表覆盖或未知隐藏节点数量恢复保证。
- **综述中的位置**：稀疏量测下开关状态/状态联合估计

**阅读依据**：正文定向阅读；部分正文：问题设定、压缩感知量测、开关导纳参数化与方法结构。

**原文定位**：PDF pp.1、3–4，§II-A–C、式(5)、§III稀疏状态与约束。

<a id="r070"></a>

#### R070 Bayesian Error-in-Variables Models for the Identification of Power Networks

**书目信息**：Jean-Sébastien Brouillon; Emanuele Fabbiani; Pulkit Nahata; Keith Moffat; Florian Dörfler; Giancarlo Ferrari-Trecate。日期/年份：2021-12-19。

**来源**：[DOI：10.48550/arXiv.2107.04480](https://doi.org/10.48550/arXiv.2107.04480)。

- **方法**：用电压和电流两端均有噪声的error-in-variables模型做最大似然导纳估计，再通过MAP/Laplace先验加入稀疏性及已知线路参数信息。
- **输入与设备**：关注节点的μPMU电压/电流相量及传感器误差统计，可加入网络先验。
- **隐藏节点与辨识对象**：Assumption1明确：全节点V/I可测，或只辨识观测节点之间的约化网络；未恢复未知物理内部节点。
- **条件与适用边界**：辨识期导纳矩阵固定；噪声模型与独立采样假设影响MLE/MAP，不能将可信的参数误差处理写成完整物理隐藏树保证。
- **综述中的位置**：相量设备/误差建模对照

**阅读依据**：正文定向阅读；部分正文：测量条件、EIV噪声、问题设定及MAP目标。

**原文定位**：技术报告PDF pp.3–5 §§2.2–4、p.7–8 §6.1 MAP。

<a id="r073"></a>

#### R073 Topology and Parameter Identification of Distribution Network Using Smart Meter and µPMU Measurements

**书目信息**：Vedantham Lakshmi Srinivas; Jianzhong Wu。日期/年份：2022。

**来源**：[DOI：10.1109/TIM.2022.3175043](https://doi.org/10.1109/TIM.2022.3175043)。

- **方法**：交替使用UKF概率估计与Newton–Raphson确定性精化，同时辨识拓扑/线路参数；分析SM/μPMU误差与μPMU安装位置。
- **输入与设备**：节点P/Q/V幅值、部分μPMU相量、缺少SM处的伪量测；节点清单给定。
- **隐藏节点与辨识对象**：支持不同设备和缺数据节点的状态补充；所读方法不发现未知数量的无量测物理接线节点。
- **条件与适用边界**：伪量测和传感器精度是输入条件；用户仪表低要求与额外μPMU覆盖代价应分别描述。
- **综述中的位置**：混合设备的参数辨识

**阅读依据**：正文定向阅读；部分正文：贡献、混合设备与伪量测输入、UKF/NR步骤、噪声案例。

**原文定位**：PDF pp.2、4–5，§I-B、§III迭代；p.9无SM位置用25%误差伪量测的案例。

<a id="r074"></a>

#### R074 Low-Voltage Distribution Grid Topology Identification With Latent Tree Model

**书目信息**：Haipeng Zhang; Jian Zhao; Xiaoyu Wang; Yi Xuan。日期/年份：2022-05。

**来源**：[DOI：10.1109/TSG.2022.3146205](https://doi.org/10.1109/TSG.2022.3146205)。

- **方法**：将用户历史电压离散为类别变量，建立包含未知数目隐变量的潜树模型；用节点/状态增删和节点移动进行三阶段结构搜索，局部EM估参数，BIC权衡拟合和复杂度。
- **输入与设备**：所有用户历史电压幅值序列和用户集合；可不测变压器电压，初始化根状态数固定为1。
- **隐藏节点与辨识对象**：直接针对LVDG未知中间节点数量及连接关系，推断潜在中间设备节点；正文包含0注入节点，主要以电压统计建模。
- **条件与适用边界**：§II-B式(2)明确只保留一跳电压依赖是近似，实际依赖可延伸两跳；局部EM+BIC候选搜索没有所读正文中的完整物理树唯一恢复/全局最优证明。
- **综述中的位置**：最直接LVDG+unknown hidden count+voltage-only基线

**阅读依据**：正文定向阅读；部分正文：物理/统计映射、潜树与离散化、5个搜索算子、正则性、EM/BIC、初始化和结论；未逐页精读全部实验。

**原文定位**：PDF pp.3–6，§II-B、§III、§IV、§V-A；p.11 §VII结论。

<a id="r075"></a>

#### R075 A general topology identification framework for distribution systems using smart meter and μ-PMU measurements

**书目信息**：Li Ma; Lei Wu。日期/年份：2022-07-01。

**来源**：[DOI：10.1016/j.ijepes.2022.108019](https://doi.org/10.1016/j.ijepes.2022.108019)。

- **方法**：基于I=YV稀疏导纳估计改进三阶段启发式，对多拓扑数据同时分类和辨识各导纳；通过两阶段程序选择未知拓扑类别数。
- **输入与设备**：各已知节点复电流/电压多时刻记录，来自SM/μPMU组合；采集点设在配变进线。
- **隐藏节点与辨识对象**：未知的是数据中拓扑状态类别数，并非物理隐藏节点数量；网络节点集和列维数已知。
- **条件与适用边界**：相量、导纳稀疏度上限、噪声和样本数影响辨识；避免把unknown number of topologies误译为unknown number of nodes。
- **综述中的位置**：多运行拓扑/混合数据分类

**阅读依据**：正文定向阅读；部分正文：单拓扑导纳模型、多拓扑状态分类及结论。

**原文定位**：PDF p.2 §2.1式(1)–(2)、p.3稀疏约束、p.4未知拓扑数、p.9结论。

<a id="r079"></a>

#### R079 Topology Identification of Radial Distribution Networks Using Smart Meter Data

**书目信息**：Abu Bakr Pengwah; Luxin Fang; Reza Razzaghi; Lachlan L. H. Andrew。日期/年份：2022-12。

**来源**：[DOI：10.1109/JSYST.2021.3128175](https://doi.org/10.1109/JSYST.2021.3128175)。

- **方法**：先由用户V/I/功率因数估计受物理和树距离约束的电压-电流灵敏度，再以RG加回溯生成若干Steiner树，按灵敏度/电压排序等目标选择候选。
- **输入与设备**：所有用户的瞬时电压幅值、电流幅值、功率因数，先按相分组；不要求相量角。
- **隐藏节点与辨识对象**：显式插入未知内部接线节点并估阻抗；§II-A明确全部无电表节点不注入/消耗功率；恢复的是最小Steiner树。
- **条件与适用边界**：§II-C原文在灵敏度式前使用independent load profiles，但未定义为线性独立/满秩，也未给跨用户统计独立或零协方差的形式条件；不将其写成Park方法那样的统计不相关假设。常数变压器电压被明确采用；噪声下回溯/有限候选为数值求解。
- **原文明确报告的回归条件**：PDF p.3 §II-C式(1)前只写independent load profiles；p.5 §III-A用约束灵敏度拟合(16)，未见rank/共线可识别条件。实验pp.6–7的独立随机采样仅是仿真设定。
- **本文件的额外代数判断**：回归需要足够信息区分系数；普通无约束回归的设计矩阵满秩是充分条件，统计独立并非必要。本文包含物理/树约束，不能据普通回归条件直接宣称该方法在相关负荷下一定失效。
- **综述中的位置**：低压隐藏Steiner树直接基线

**阅读依据**：正文定向阅读；部分正文：输入/zero injection、灵敏度约束、RG/backtracking、实验设定和结论。

**原文定位**：PDF pp.2–5 §II–III、p.7合成hidden degree≥3设置、pp.10–11结论。

**对应原记录编号**：R079, R080。

<a id="r081"></a>

#### R081 Topology and Impedance Identification Method of Low-Voltage Distribution Network Based on Smart Meter Measurements

**书目信息**：Li Tong; Weijian Chai; Dongqi Wu。日期/年份：2022-4-29。

**来源**：[DOI：10.3389/fenrg.2022.895397](https://doi.org/10.3389/fenrg.2022.895397)。

- **方法**：以节点电压Pearson相关形成权重，Kruskal选最大相关生成树；再据电压线性回归、功率流分区和回溯估计变压器—分支—表箱—用户线阻抗。
- **输入与设备**：节点电压/电流/相别及V/P/Q时间序列，分层物理拓扑模型依赖已有可测节点。
- **隐藏节点与辨识对象**：所读算法直接对有电压序列的节点构建相关树，未给出未知数量隐藏接线节点的生成步骤。
- **条件与适用边界**：相关高低依赖负荷、公共电源电压等因素；正文采用忽略电压相角差的线性模型。
- **综述中的位置**：电压相关/MST与分层管理拓扑

**阅读依据**：正文定向阅读；部分正文：相关树、参数模型、案例输入与结论。

**原文定位**：PDF pp.2–4相关树和线性阻抗模型、p.6案例输入、p.7结论。

<a id="r086"></a>

#### R086 Topology Identification of Distribution Networks Based on Physics-Informed Latent Graph Attention Network

**书目信息**：Yutong Zhou; Haiwei Liang; Xudong Zou; Yizhou Ding。日期/年份：2024。

**来源**：[DOI：10.1007/978-981-97-3940-0_4](https://doi.org/10.1007/978-981-97-3940-0_4)。

- **方法**：把候选配网拓扑表成line graph，用其邻接矩阵和节点特征训练physics-informed latent graph attention模型识别拓扑；摘要报告改进IEEE33的分类表现。
- **输入与设备**：摘要仅说明时间截面测量及line graph邻接/特征；详细量测与训练标签需求待正文。
- **隐藏节点与辨识对象**：标题latent不能支持恢复无量测物理节点；现有摘要未说明未知隐藏节点数及位置重建。
- **条件与适用边界**：仅摘要，98.15%是其IEEE33案例报告，不作普遍保证。Springer官方作者为Zhou/Liang/Zou/Ding；Hu/Cao为编辑，不应加入作者。
- **综述中的位置**：学习式候选拓扑识别

**阅读依据**：摘要及概述筛选；仅摘要；本地无PDF，root公开/HKU补取仍有全文缺口。

**原文定位**：Zotero摘要；root核对Springer官方章节摘要：https://link.springer.com/chapter/10.1007/978-981-97-3940-0_4。

**版本与书目说明**：仅官方/Zotero摘要；缺原文。

<a id="r087"></a>

#### R087 Learning Distribution Grid Topologies: A Tutorial

**书目信息**：Deepjyoti Deka; Vassilis Kekatos; Guido Cavraro。日期/年份：2024-01。

**来源**：[DOI：10.1109/TSG.2023.3271902](https://doi.org/10.1109/TSG.2023.3271902)。

- **方法**：教程按相量/智能电表、完整/局部/仅电压量测和主动/被动数据组织拓扑辨识与状态检测；统一物理模型、可辨识条件和求解复杂度。
- **输入与设备**：综述，无单一算法；按传感器、覆盖和物理模型逐类讨论。
- **隐藏节点与辨识对象**：明确部分量测下RG/加性阻抗距离能恢复degree≥3内部节点，度1/2节点被Kron约化；部分相量方案依赖零注入未测节点。
- **条件与适用边界**：适合作分类框架与追溯原始论文；具体方法性能应引原文。教程还引Anguluri等hidden nodes structured norm minimization，本库主清单未见该题，宜补查。
- **综述中的位置**：综述框架/术语与遗漏线索

**阅读依据**：正文定向阅读；部分正文：量测分型、隐藏节点条件、三相扩展与研究展望。

**原文定位**：PDF pp.4–8 §§III–IV量测分型、p.11多相扩展、p.12展望/结论、p.13参考文献。

<a id="r089"></a>

#### R089 Topology Identification of Distribution Networks With Partial Smart Meter Coverage

**书目信息**：Abu Bakr Pengwah; Yasin Zabihinia Gerdroodbari; Reza Razzaghi; Lachlan L. H. Andrew。日期/年份：2024-04。

**来源**：[DOI：10.1109/TPWRD.2024.3354292](https://doi.org/10.1109/TPWRD.2024.3354292)。

- **方法**：为高分辨率SM与仅低分辨率电量interval meters联合估计约化阻抗；先RG恢复SM对应子树，再用SM–interval共同路径阻抗定位interval客户并插入遗漏分支。
- **输入与设备**：SM的V/P/Q及低频interval有功电量；不同时间分辨率通过窗口对齐。
- **隐藏节点与辨识对象**：内部节点明确为零注入分支点；SM覆盖可低于100%，但其余用户仍有interval能量信息；degree2未观测节点在SM子树阶段合并，后续可因interval分支被恢复。
- **条件与适用边界**：interval电压近似1pu、无功近似0/接近单位功率因数；完全无电量信息客户与interval客户应分开。
- **综述中的位置**：部分SM覆盖下隐藏分支及混合电表核心基线

**阅读依据**：正文定向阅读；部分正文：两类量测、近似、阻抗估计、SM树/interval位置恢复、结论。

**原文定位**：PDF pp.2–5 §II与III，p.4 Fig3及RG degree<3说明；pp.6–9实验与结论。

**对应原记录编号**：R089, R113。

<a id="r090"></a>

#### R090 Three-phase voltage sensitivity estimation and its application to topology identification in low-voltage distribution networks

**书目信息**：Luxin Fang; Abu Bakr Pengwah; Lachlan L. H. Andrew; Reza Razzaghi; Mario Andrés Muñoz。日期/年份：2024-07-01。

**来源**：[DOI：10.1016/j.ijepes.2024.109949](https://doi.org/10.1016/j.ijepes.2024.109949)。

- **方法**：联合估计三相电压-电流灵敏度和未知变压器电压，显式保留公共中性线导致的跨相耦合；以改进RG/backtracking和快速候选选择恢复三相Steiner拓扑。
- **输入与设备**：全部单相用户SM的电压幅值、电流幅值、功率因数及超前/滞后指示；相别先识别，不需测相量角。
- **隐藏节点与辨识对象**：未知中间节点是零电流注入接线/分支点；PDF p.3 §2.1在式(1)后明确说明unmetered节点没有电流注入。恢复最小Steiner树并包含服务接入线；目标用户均为单相负荷。
- **条件与适用边界**：电压角/正交分量作投影近似；RG以有限候选回溯进行重建，所读正文未给整体全局恢复保证。
- **原文明确报告的回归条件**：PDF p.5 §2.2用伪逆给出不含所列物理约束的最小二乘解，再引入约束(20)/(21)；未显式给设计矩阵满秩或P/Q（电流同相/正交分量）不共线的条件。全文中的rank仅指候选排序。
- **本文件的额外代数判断**：同相/正交电流或不同客户曲线高度共线会影响无约束灵敏度系数分离，这是回归模型层面的判断；物理约束可能补充信息，本文未证明这种情形下一定无法辨识拓扑。
- **综述中的位置**：三相耦合+隐藏节点+未知源电压直接基线

**阅读依据**：正文定向阅读；部分正文：模型/观测、灵敏度优化、RG/backtracking与候选选择、实验输入和结论。

**原文定位**：PDF pp.3–7 §§2–3建模/约束/RG与回溯、pp.8–9实验条件、p.16结论；PDF p.3 §2.1式(1)后零注入说明。

<a id="r096"></a>

#### R096 Topology and Parameter Identification of Three-Phase Low-Voltage Distribution Network and Voltage Control Strategy Based on User Smart Meter Measurements: A Data-Driven Approach

**书目信息**：Li Guo; Ge Yang; Yixin Liu; Yuxuan Zhang; Zhongguan Wang; Haifeng Yu; Yifei Wang; Chengshan Wang。日期/年份：2025。

**来源**：[DOI：10.1109/TIM.2025.3585219](https://doi.org/10.1109/TIM.2025.3585219)。

- **方法**：从用户V/P/Q回归电压平方对P/Q的灵敏度，以三条物理敏感性判据做相别识别、列内灵敏度分级并逐层插入隐藏节点；随后逆向分层估参数并做损耗补偿及电压控制。
- **输入与设备**：全部用户智能电表V幅值和P/Q多时刻数据；隐藏节点无量测。
- **隐藏节点与辨识对象**：正文明确隐藏节点无负荷；degree1/2无负荷节点省略，针对分支结构。
- **条件与适用边界**：§III-B近似跨相灵敏度为0，与显式跨相耦合模型有条件差别；敏感性层级阈值δΩ需小于线路电阻尺度的约束，噪声/短弱支路会影响层级分离。
- **原文明确报告的回归条件**：PDF p.5 §III-A式(22)直接对多个时刻的[P;Q;vref]做LS求SP、SQ；未显式给样本矩阵秩或P/Q不共线条件。p.4式(2)的不同节点注入independent用于写偏导∂Pj/∂Pi=0，不是以随机变量独立/零协方差形式给出的数据分布假设。
- **本文件的额外代数判断**：若样本中Q与P完全共线，仅式(22)通常不足以分别确定SP与SQ；这是方程可识别性的代数判断，不是原文报告的实验失败，也不能直接推出拓扑不可识别。
- **综述中的位置**：近期用户测量+大量hidden nodes直接基线

**阅读依据**：正文定向阅读；部分正文：隐藏节点定义、敏感性判据/回归、相别与层级重建、参数/控制概述及结论。

**原文定位**：PDF pp.2–6 §II–III，p.3度1/2省略、p.5跨相近似、p.6阈值/插点算法；p.12结论。

<a id="r108"></a>

#### R108 Error Bounds for Radial Network Topology Learning From Quantized Measurements

**书目信息**：Samuel Talkington; Aditya Rangarajan; Pedro A. Alcântara; Line Roald; Daniel K. Molzahn; Daniel R. Fuhrmann。日期/年份：2026-05。

**来源**：[DOI：10.1109/TPWRS.2026.3666699](https://doi.org/10.1109/TPWRS.2026.3666699)。

- **方法**：把抖动量化P测量下的导纳/完全图边权估计写为约束广义LASSO，利用径向树模型的Gaussian width给出随量化步长、节点数和样本数变化的参数估计误差界。
- **输入与设备**：全节点有功功率与电压幅值样本，已知固定功率因数q=κp，量化传感器模型。
- **隐藏节点与辨识对象**：理论估计在给定全部节点/完全图边权维度上，没有未知隐藏节点数的发现机制。
- **条件与适用边界**：误差界作用于连续边权向量；精确物理拓扑/弱边支持恢复还需最小边强度等分离条件。样本设计iid次Gaussian等定理前提应保留。
- **综述中的位置**：量测精度/理论误差边界

**阅读依据**：正文定向阅读；部分正文：量化模型、固定功率因数、估计器、定理和数值设计。

**原文定位**：PDF pp.1–4，Assumption1、§III-A、Lemma1、Theorem1、§III-C和结论。

<a id="r117"></a>

#### R117 Guaranteed Phase & Topology Identification in Three Phase Distribution Grids

**书目信息**：Mohini Bariya; Deepjyoti Deka; Alexandra Von Meier。日期/年份：7/2021。

**来源**：[DOI：10.1109/TSG.2021.3061392](https://doi.org/10.1109/TSG.2021.3061392)。

- **方法**：利用不同相位电压协方差进行相别匹配，再以电压差方差为距离贪心构造三相树；允许节点仅有部分相，理论与数值分别讨论相量/幅值输入。
- **输入与设备**：已知节点集全部节点的多相电压时间序列；理论按高分辨率可去趋势数据处理。
- **隐藏节点与辨识对象**：算法输入明确为all nodes in V的电压；未插入未观测物理隐藏节点。
- **条件与适用边界**：跨节点及跨相电流注入不相关；相别匹配用等方差和线路阻抗匹配条件，拓扑部分可放宽等方差。
- **综述中的位置**：三相/仅电压的全节点对照

**阅读依据**：正文定向阅读；部分正文：线性三相模型、注入统计、匹配条件、拓扑定理与算法输入。

**原文定位**：PDF pp.3–6，§II-B假设(7)(8)、Theorem2/3、Algorithms1–2；全节点输入p.5。


### 9.3 一般配电网拓扑方法与应用（48组）

<a id="r004"></a>

#### R004 Signal Injection-Based Topology Identification for Low-Voltage Distribution Networks Considering Missing Data

**书目信息**：Yilong Duan; Zheng Liu; Yuanyuan Liu; Yong Li。日期/年份：2024（PDF首页核对）。

**来源**：[DOI：10.3390/en17092060](https://doi.org/10.3390/en17092060)。

- **方法**：以末端设备注入特征电流/功率、上游接收的记录包含关系识别树层级；用纵向/横向补全修复缺失记录，再做包含检测。
- **输入与设备**：具注入/检测功能的TID、PLC通信和边缘IIT收集的FSR。
- **隐藏节点与辨识对象**：处理设备记录缺失；物理上未装设备的任意隐藏节点重建未在所读方法中给出。
- **条件与适用边界**：要求对应智能设备和通信覆盖；完全无收发记录的设备仍可能无法定位。
- **综述中的位置**：信号注入/设备覆盖路线

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–4,13（§2、§5）。

**原文定位**：PDF pp.1–4,13（§2、§5）。

**版本与书目说明**：R004/R005为重复；PDF首页补得DOI 10.3390/en17092060、2024。

**对应原记录编号**：R004, R005。

<a id="r006"></a>

#### R006 TOPOLOGY ESTIMATION IN LOW VOLTAGE GRIDS USING WALLBOX CHARGING DATA RECORDINGS

**书目信息**：Christian Hotz; Sergej Baum; Eberhard Waffenschmidt; Ingo Stadler。日期/年份：2022（CIRED作者存档）。

**来源**：[正式发表DOI：10.1049/icp.2022.0823](https://doi.org/10.1049/icp.2022.0823)；[稳定来源](https://zenodo.org/records/10046340/files/2022%20CIRED%20Porto%20Topology%20Estimation.pdf)。

- **方法**：识别EV充电启停电流阶跃，以ΔU/ΔI估计端点共同路径阻抗矩阵，逐次合并共享最长公共路径的节点并插入分支点，重构二叉径向树及阻抗。
- **输入与设备**：各wallbox协调电压和电流记录、足够多充电开关事件、通信；推导采用复电压变化。
- **隐藏节点与辨识对象**：明确插入无量测distribution buses，恢复家庭支线连接点。
- **条件与适用边界**：径向/二叉树、平衡三相等值、充电PF=1；背景负荷波动影响精度，非高斯PV P(U)控制干扰不由所提平均方法处理。
- **综述中的位置**：LVDG+隐藏接线点核心基线

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–4,8（switch events、common lines、reconstruction、Conclusion）。

**原文定位**：PDF pp.1–4,8（switch events、common lines、reconstruction、Conclusion）。

**版本与书目说明**：原PDF题名拼作TOPLOGY；公开自存来源为2022 CIRED Porto，Zotero日期为空。


**本轮书目补充**：TH Köln机构出版清单列正式题名 *Estimating Topologies of Low Voltage Grids from Electric Vehicle Charging Point Measurement Data*，CIRED Porto Workshop2022，804–808页。现有Wallbox稿与该条同作者/方法发展，不新增独立方法计数；本项技术定位仍对应已读Wallbox稿。[机构清单](https://www.th-koeln.de/mam/downloads/deutsch/forschung/projekte___publikationen_2023.pdf)。

<a id="r008"></a>

#### R008 Efficient policy adaptation for voltage control under unknown topology changes

**书目信息**：Jie Feng; Yuanyuan Shi; Deepjyoti Deka。日期/年份：01/2027。

**来源**：[DOI：10.1016/j.epsr.2026.113622](https://doi.org/10.1016/j.epsr.2026.113622)。

- **方法**：基于原拓扑和稀疏切换假设，由电压预测残差检测并区分拓扑/负荷变化，更新电压-无功灵敏度，再在线适配预训练RL电压控制器。
- **输入与设备**：节点电压、逆变器控制输入/功率、原拓扑/原灵敏度。
- **隐藏节点与辨识对象**：节点集合和原拓扑已给定；目标为变化线路检测，不是从叶端数据生成未知内部节点。
- **条件与适用边界**：单相或平衡径向LinDistFlow分析；变化稀疏且短窗负荷变化设定；识别保证需实际激励条件。
- **综述中的位置**：拓扑准确性对控制影响/拓扑变化处理

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–4（§2、§3，Remark 2.I、Lemma 3.IV）。

**原文定位**：PDF pp.1–4（§2、§3，Remark 2.I、Lemma 3.IV）。

**版本与书目说明**：首页标卷262(2027)，接收/在线年份需另核；不能按当前日期称2027已正式刊出。

<a id="r010"></a>

#### R010 Smart meter data-driven dynamic operating envelopes for DERs

**书目信息**：Achala Kumarawadu; M. Imran Azim; Mohsen Khorasany; Reza Razzaghi; Rahmat Heidari。日期/年份：04/2025。

**来源**：[DOI：10.1016/j.apenergy.2025.125469](https://doi.org/10.1016/j.apenergy.2025.125469)。

- **方法**：由用户智能表V/P/Q拟合三相含中性线和互阻的公共路径R/X等效模型，再计算满足网络约束的DER动态运行包络和容量分配。
- **输入与设备**：智能表V/P/Q及相别连接信息。
- **隐藏节点与辨识对象**：在端口阻抗/灵敏度层面建模；所读章节未显式重建隐藏物理分支点。
- **条件与适用边界**：需相别信息和足够数据，电压角近似；三相四线等效的接地条件影响适用性。
- **综述中的位置**：拓扑/参数信息的重要性和等效模型应用

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–4（§3.1–3.2，式(4)–(24)）。

**原文定位**：PDF pp.1–4（§3.1–3.2，式(4)–(24)）。

<a id="r015"></a>

#### R015 Adaptive optimal-weight method for topology identification and line parameter estimation of large-scale low-voltage distribution grids

**书目信息**：Nan Feng; Yuxuan Ding; Yaping Du; Dongchuan Fan; Qianchao Wang。日期/年份：08/2026。

**来源**：[DOI：10.1016/j.ijepes.2026.112103](https://doi.org/10.1016/j.ijepes.2026.112103)。

- **方法**：优化端点topology-encoded line matrix(TLM)，用双向层次聚类恢复拓扑/边阻抗；将电压一致性、电缆阻抗角和结构先验加权成目标，用自适应混沌随机分形搜索+小批梯度优化。
- **输入与设备**：端点V/I幅值及功率因数角（或P/Q），已知有限电缆类型的单位阻抗库。
- **隐藏节点与辨识对象**：§1–2明确联合恢复latent nodes和各边参数；属于用户目标的直接比较对象。
- **条件与适用边界**：径向；依赖电缆库与软结构先验；启发式给最佳已找到解，不能将仿真成功视为全局唯一恢复保证。
- **综述中的位置**：LVDG+隐藏节点与多线型核心基线

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–4（§1、§2.2、Fig.2、式(3)–(7)）。

**原文定位**：PDF pp.1–4（§1、§2.2、Fig.2、式(3)–(7)）。

<a id="r016"></a>

#### R016 A review on topology identification methods and applications in distribution networks

**书目信息**：Farzad Dalavi; Mohamad Esmail Hamedani Golshan; Nikos D. Hatziargyriou。日期/年份：09/2024。

**来源**：[DOI：10.1016/j.epsr.2024.110538](https://doi.org/10.1016/j.epsr.2024.110538)。

- **方法**：按模型驱动/数据驱动、单快照/时间序列、被动/主动整理配电网拓扑辨识，并综述联合参数估计、DER、量测质量和应用。
- **输入与设备**：综述文献，涵盖SM/μPMU等。
- **隐藏节点与辨识对象**：应追溯其引用的原始隐藏节点论文，综述本身不提供新增恢复算法。
- **条件与适用边界**：统一讨论中混有开关状态、全结构、归属等不同任务，引用时须保持任务边界。
- **综述中的位置**：综述框架与重要性背景

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–4,11（Fig.1、Table1、§11–12）。

**原文定位**：PDF pp.1–4,11（Fig.1、Table1、§11–12）。

<a id="r017"></a>

#### R017 Data-physics hybrid-driven two-stage topology identification for active distribution networks with soft open points

**书目信息**：Huanan Yu; Yang Liu; Shiqiang Li; He Wang; Jing Bian; Guoqing Li。日期/年份：09/2026。

**来源**：[DOI：10.1016/j.segan.2026.102461](https://doi.org/10.1016/j.segan.2026.102461)。

- **方法**：利用SOP的电气隔离将网络分区，SHAP选择关键特征后由Transformer预测候选拓扑并作Top-K筛选，最后用AC潮流电压残差验证/修正。
- **输入与设备**：SOP稳态模型、节点量测、候选拓扑标签、已知线路参数。
- **隐藏节点与辨识对象**：在给定分区网络/候选状态中识别，无未知隐藏物理节点生成步骤。
- **条件与适用边界**：面向SOP有源配电网；监督候选覆盖与AC检验参数准确性，固定线路参数为作者指出的后续扩展。
- **综述中的位置**：SOP场景与数据-物理候选校正

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1,18（方法概述、§6结论）。

**原文定位**：PDF pp.1,18（方法概述、§6结论）。

**版本与书目说明**：元数据为2026/9；PDF显示2026-07-29在线。

<a id="r019"></a>

#### R019 Distribution Network Topology Identification Using Asynchronous Transformer Monitoring Data

**书目信息**：Zhiqi Xu; Wei Jiang; Junjun Xu; Dong Wang; Yifei Wang; Zhujian Ou。日期/年份：1/2023。

**来源**：[DOI：10.1109/TIA.2022.3212030](https://doi.org/10.1109/TIA.2022.3212030)。

- **方法**：把候选开关状态与未测相角纳入L2拟合，经线性化/凸松弛形成MIQCP；对异步TTU采样产生的大误差用Lagrange插值伪量测替代。
- **输入与设备**：变压器终端TTU电压/功率幅值、候选网络及线路模型。
- **隐藏节点与辨识对象**：不需要全节点相角，但处理既定物理图的开关状态；未给从无量测端生成内部节点的算法。
- **条件与适用边界**：主要针对中压配电网；依赖候选图/参数，插值在高动态条件下精度受限。
- **综述中的位置**：中压/候选开关模型与异步数据

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3,8（§I–III、Conclusion开头）。

**原文定位**：PDF pp.1–3,8（§I–III、Conclusion开头）。

<a id="r020"></a>

#### R020 Graph Algorithms for Topology Identification Using Power Grid Probing

**书目信息**：Guido Cavraro; Vassilis Kekatos。日期/年份：10/2018。

**来源**：[DOI：10.1109/LCSYS.2018.2846801](https://doi.org/10.1109/LCSYS.2018.2846801)。

- **方法**：控制逆变器有功扰动并回归灵敏度列，再由相同公共路径阻抗对应的level sets递归分组，恢复拓扑及线电阻；无功扰动可类似估计电抗。
- **输入与设备**：满行秩探测矩阵、主动逆变器和同步电压响应。
- **隐藏节点与辨识对象**：全叶节点探测+全节点电压可恢复完整树；仅探测点电压时恢复含可辨分支点的约化树，串联未辨节点合并。
- **条件与适用边界**：径向线性模型；探测时非探测负荷近恒定；部分数据的内部可辨点要求至少两子树均含探测点。
- **综述中的位置**：主动探测+隐藏节点约化树核心基线

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§III–V，式(4)–(8)、Definition1）。

**原文定位**：PDF pp.1–3（§III–V，式(4)–(8)、Definition1）。

**对应原记录编号**：R020, R046。

<a id="r022"></a>

#### R022 Full-Scale Distribution System Topology Identification Using Markov Random Field

**书目信息**：Jian Zhao; Liang Li; Zhao Xu; Xiaoyu Wang; Haobo Wang; Xianjun Shao。日期/年份：11/2020。

**来源**：[DOI：10.1109/TSG.2020.2995164](https://doi.org/10.1109/TSG.2020.2995164)。

- **方法**：对电压差分、分箱和one-hot编码建立MRF，以伪似然和L2正则学习节点依赖，再迭代筛选相关边构造径向拓扑；提供增量/并行边缘计算。
- **输入与设备**：MV终端与LV用户智能表节点电压数据。
- **隐藏节点与辨识对象**：所读方法以采集节点为MRF变量，未提供增添无量测物理节点步骤；full-scale不能直接解读为hidden-node恢复。
- **条件与适用边界**：统计依赖到物理边的对应、量测覆盖和分箱/筛选影响结果；径向输出。
- **综述中的位置**：被动统计/边缘计算

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II、方法贡献列表）。

**原文定位**：PDF pp.1–3（§II、方法贡献列表）。

<a id="r024"></a>

#### R024 Power Line Communications for Low-Voltage Power Grid Tomography

**书目信息**：Mohamed O. Ahmed; Lutz Lampe。日期/年份：12/2013。

**来源**：[DOI：10.1109/TCOMM.2013.111613.130238](https://doi.org/10.1109/TCOMM.2013.111613.130238)。

- **方法**：把PLC端到端测距与网络树层析结合；用传输线模型的超分辨到达时间估计距离，再重构树连接和线长。
- **输入与设备**：树边缘PLC modem双向信号/传递函数与传播模型。
- **隐藏节点与辨识对象**：仅叶端有modem，明确恢复未装modem的内部branch nodes及边长。
- **条件与适用边界**：树状网络、PLC传播/测距模型和可辨距离质量；依赖通信信号能力，不需所有分支点安装仪表。
- **综述中的位置**：PLC+隐藏节点核心基线

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II、§III，Fig.1–2）。

**原文定位**：PDF pp.1–3（§II、§III，Fig.1–2）。

**对应原记录编号**：R024, R025。

<a id="r041"></a>

#### R041 A data driven approach to distribution network topology identification

**书目信息**：Kostas Soumalas; George Messinis; Nikos Hatziargyriou。日期/年份：2017-06。

**来源**：[DOI：10.1109/PTC.2017.7981061](https://doi.org/10.1109/PTC.2017.7981061)。

- **方法**：由叶端P/Q/V估计约化电压灵敏度，转成距离矩阵，再按Prüfer序列及离散邻近距离假设做树层析。
- **输入与设备**：根和用户叶端的同步高频P/Q/V；已知相别与馈线归属，线型/邻近距离信息。
- **隐藏节点与辨识对象**：明确允许无负荷、无监测的中间节点并重建树结构。
- **条件与适用边界**：根电压近恒定；测量质量/同步要求较高；距离整数量化和线型/长度情景先验限制通用性。
- **综述中的位置**：被动灵敏度+隐藏节点核心基线

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II–IV，Fig.2–3）。

**原文定位**：PDF pp.1–3（§II–IV，Fig.2–3）。

<a id="r042"></a>

#### R042 Power grid probing for load learning: Identifiability over multiple time instances

**书目信息**：Siddharth Bhela; Vassilis Kekatos; Sriharsha Veeramachaneni。日期/年份：2017-12。

**来源**：[DOI：10.1109/CAMSAP.2017.8313220](https://doi.org/10.1109/CAMSAP.2017.8313220)。

- **方法**：利用多轮逆变器有功/功率因数扰动，将短时恒定的未测负荷通过耦合潮流方程联合估计，并用结构秩/二部图匹配给局部可辨识充分条件。
- **输入与设备**：已知网络导纳Y，探测节点P/Q/电压幅值，多个短时探测窗口。
- **隐藏节点与辨识对象**：估计已知网络上未测负荷，不是恢复未知拓扑中的物理隐藏节点。
- **条件与适用边界**：已知Y；非探测负荷短时不变；局部/泛型可辨识条件与全局唯一性不同。
- **综述中的位置**：主动探测背景，非直接拓扑基线

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（Definition1、式(1)–(5)、Lemma1）。

**原文定位**：PDF pp.1–3（Definition1、式(1)–(5)、Lemma1）。

<a id="r048"></a>

#### R048 PaToPaEM: A Data-Driven Parameter and Topology Joint Estimation Framework for Time Varying System in Distribution Grids

**书目信息**：Jiafan Yu; Yang Weng; Ram Rajagopal。日期/年份：2018-12-17。

**来源**：[DOI：10.48550/arXiv.1812.06619](https://doi.org/10.48550/arXiv.1812.06619)。

- **方法**：以errors-in-variables处理输入输出测量噪声，并将每个样本所属的拓扑/参数工况作为潜变量，用EM交替聚类工况和估计各工况线参数，再由非零参数确定连边。
- **输入与设备**：节点P/Q/V及相角θ历史数据（相角通常需PMU/μPMU）。
- **隐藏节点与辨识对象**：latent/hidden指样本的工况类别，不是物理无量测接线点。
- **条件与适用边界**：需要相角与节点覆盖；多工况数和初始化影响EM；局部似然优化不等于全局拓扑恢复。
- **综述中的位置**：多工况、噪声联合参数/拓扑

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II–III，测量误差模型和Problem）。

**原文定位**：PDF pp.1–3（§II–III，测量误差模型和Problem）。

<a id="r052"></a>

#### R052 Smart Inverter Grid Probing for Learning Loads: Part I — Identifiability Analysis

**书目信息**：Siddharth Bhela; Vassilis Kekatos; Sriharsha Veeramachaneni。日期/年份：2019-09。

**来源**：[DOI：10.1109/TPWRS.2019.2906316](https://doi.org/10.1109/TPWRS.2019.2906316)。

- **方法**：以主动逆变器多轮扰动耦合潮流方程，分析不装表负荷的局部/泛型可辨识；用网络派生二部图上的最大流检查条件，覆盖相量/幅值、径向/网状和多相。
- **输入与设备**：已知导纳/拓扑、探测节点P/Q和电压幅值或相量。
- **隐藏节点与辨识对象**：目标是已知网络上的non-metered loads；不直接识别未知隐藏节点拓扑。
- **条件与适用边界**：非测负荷在多轮间恒定（或另作ZIP模型）；理论条件依赖量测/控制位置。
- **综述中的位置**：主动探测可辨识背景

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II、式(1)–(5)，§III–V范围说明）。

**原文定位**：PDF pp.1–3（§II、式(1)–(5)，§III–V范围说明）。

<a id="r053"></a>

#### R053 Smart Inverter Grid Probing for Learning Loads: Part II—Probing Injection Design

**书目信息**：Siddharth Bhela; Vassilis Kekatos; Sriharsha Veeramachaneni。日期/年份：2019-09。

**来源**：[DOI：10.1109/TPWRS.2019.2906306](https://doi.org/10.1109/TPWRS.2019.2906306)。

- **方法**：先随机生成逆变器可实现的候选探测向量，借线性化模型对预期负荷区间作鲁棒电压筛选，再选状态响应差异较大的集合以改善未测负荷估计。
- **输入与设备**：已知Y/网络、逆变器P/Q能力边界、负荷区间与电压限制。
- **隐藏节点与辨识对象**：继Part I估计已知网络上的未测负荷，非未知物理拓扑恢复。
- **条件与适用边界**：安全筛选依赖线性模型和负荷范围；状态多样性作为改善条件数的设计准则，文中以conjecture引入，勿当普遍最优设计定理。
- **综述中的位置**：探测设计/成本与安全边界

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II、§III.B–C）。

**原文定位**：PDF pp.1–3（§II、§III.B–C）。

<a id="r060"></a>

#### R060 A Method of Low Voltage Topology Identification

**书目信息**：Chen Xu; Yuan Lei; Yuhang Zou。日期/年份：2020-12。

**来源**：[DOI：10.1109/TOCS50858.2020.9339731](https://doi.org/10.1109/TOCS50858.2020.9339731)。

- **方法**：用HPLC自动组网确认户变关系，再由拓扑终端发特征信号、各节点记录接收事件并由智能终端汇总恢复设备层级。
- **输入与设备**：支持HPLC和零交越拓扑信号收发的低压设备、拓扑终端、智能终端。
- **隐藏节点与辨识对象**：识别已部署拓扑设备的归属与层级；未给任意无设备隐藏节点插入方法。
- **条件与适用边界**：依赖设备改造/收发覆盖和通信；任务包含户变归属，须与完整电气树恢复区分。
- **综述中的位置**：信号注入与户变/层级辨识

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II–III）。

**原文定位**：PDF pp.1–3（§II–III）。

<a id="r062"></a>

#### R062 Power System Network Topology Identification Based on Knowledge Graph and Graph Neural Network

**书目信息**：Changgang Wang; Jun An; Gang Mu。日期/年份：2021-02-17。

**来源**：[DOI：10.3389/fenrg.2020.613331](https://doi.org/10.3389/fenrg.2020.613331)。

- **方法**：将设备/节点/线路及遥测属性整理成知识图谱，结合GNN对候选实体关系/线路状态做链路预测，提高含错误信息时的连接判断能力。
- **输入与设备**：已有设备实体、关系/图结构及节点/线路遥测属性和训练标签。
- **隐藏节点与辨识对象**：网络实体/节点集合已定义；未展示生成没有属性数据的未知物理节点。
- **条件与适用边界**：依赖图数据库/训练覆盖；网络隐藏层与物理隐藏节点语义不同。
- **综述中的位置**：知识图谱+监督链路预测

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（摘要、§2知识图谱构建）。

**原文定位**：PDF pp.1–3（摘要、§2知识图谱构建）。

**对应原记录编号**：R062, R071。

<a id="r063"></a>

#### R063 Low-voltage distribution network topology identification method based on characteristic current

**书目信息**：Xuanping Lai; Min Cao; Siyang Liu; Chen Sun。日期/年份：2021-04。

**来源**：[DOI：10.1109/ACPEE51499.2021.9437092](https://doi.org/10.1109/ACPEE51499.2021.9437092)。

- **方法**：智能感知终端注入微小特征电流，用高灵敏TMR磁传感器检测上游通路，并通过LoRa交换检测结果推断台区连接关系。
- **输入与设备**：特征电流发生器、TMR传感器、智能终端与LoRa通信。
- **隐藏节点与辨识对象**：主要定位装有感知/检测设备的连接；无量测物理接线点恢复未在所读章节给出。
- **条件与适用边界**：需要专门传感/信号设备；小电流可测性和通信覆盖限制现场部署。
- **综述中的位置**：设备要求较高的可引用具体例子

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II、§III.A–B）。

**原文定位**：PDF pp.1–3（§II、§III.A–B）。

<a id="r066"></a>

#### R066 Distribution Network Admittance Matrix Estimation With Linear Regression

**书目信息**：Jiawei Zhang; Peng Wang; Ning Zhang。日期/年份：2021-09。

**来源**：[DOI：10.1109/TPWRS.2021.3090250](https://doi.org/10.1109/TPWRS.2021.3090250)。

- **方法**：由解耦线性潮流消去未测相角，建立P/Q/V幅值与导纳子矩阵之间线性关系，用总最小二乘及开环图零模式迭代恢复Y。
- **输入与设备**：待辨区域节点同步P/Q/V幅值，参考节点信息；不需直接相角。
- **隐藏节点与辨识对象**：所读模型以已有节点为变量；部分区域量测建立在其他区域拓扑已知，未给未知隐藏节点生成算法。
- **条件与适用边界**：单相开环网、DLPF小角近似；需超过2(n−1)+1快照且电压变化大于测量误差。
- **综述中的位置**：非相量导纳回归

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II–IV，式(2)–(11)）。

**原文定位**：PDF pp.1–3（§II–IV，式(2)–(11)）。

<a id="r069"></a>

#### R069 Low-voltage overhead lines topology identification method based on high-frequency signal injection

**书目信息**：Haotian Ge; Bingyin Xu; Xinhui Zhang; Yongjian Bi。日期/年份：2021-11-30。

**来源**：[DOI：10.24425/aee.2021.138261](https://doi.org/10.24425/aee.2021.138261)。

- **方法**：注入5MHz高频正弦信号并利用电感阻隔使其单向传播，以各观察点延迟和反射估算连接关系与线段长度。
- **输入与设备**：高频信号源、约1mH电感、线路观察/检测点与高频传播参数。
- **隐藏节点与辨识对象**：测距/反射支持线路结构分析，但所读段未给纯端点数据下任意隐藏分支树唯一恢复结论。
- **条件与适用边界**：针对低压架空线和高频模型；正文报告MATLAB仿真，估长为粗略值。
- **综述中的位置**：高频信号注入/专门设备

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（摘要、§2–3；方法路线由摘要明确）。

**原文定位**：PDF pp.1–3（摘要、§2–3；方法路线由摘要明确）。

<a id="r076"></a>

#### R076 A Dual-Graph Attention-Based Approach for Identifying Distribution Network Topology

**书目信息**：Guoqiang Lin; Siyan Liu; Danni Shi; Xuan Wang; Shiliang Liu。日期/年份：2022-10。

**来源**：[DOI：10.1109/ICCSNT56096.2022.9972879](https://doi.org/10.1109/ICCSNT56096.2022.9972879)。

- **方法**：用图卷积特征注意力与自注意力top-k池化形成整图表示，再由MLP分类器输出候选拓扑类别。
- **输入与设备**：历史节点/线路特征、图数据和已标注拓扑训练集。
- **隐藏节点与辨识对象**：以既定图节点做拓扑分类；未显式推断未知数量物理隐藏节点。
- **条件与适用边界**：监督训练覆盖与候选类别依赖；IEEE33/57仿真不能外推未知结构生成。
- **综述中的位置**：监督GNN拓扑分类

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II，式(6)–(8)）。

**原文定位**：PDF pp.1–3（§II，式(6)–(8)）。

<a id="r077"></a>

#### R077 Research on Topology Recognition Technology Based on Intelligent Measurement Switches

**书目信息**：Dezhi Xiong; Jingjuan Du。日期/年份：2022-11-25。

**来源**：[DOI：10.3390/electronics11233903](https://doi.org/10.3390/electronics11233903)。

- **方法**：把特征电流调制/解调、编码和通信协议集成到智能量测开关，通过各级开关检测到的特征电流确定低压设备拓扑。
- **输入与设备**：各级智能量测开关、拓扑识别单元、信号收发和通信。
- **隐藏节点与辨识对象**：结构识别随智能开关部署层级；未读到仅用户端数据生成未装设备分支点的能力。
- **条件与适用边界**：需功能集成/部署；所报100%是给定实验测试结果。
- **综述中的位置**：设备集成路线与部署要求

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§2与设计概述）。

**原文定位**：PDF pp.1–3（§2与设计概述）。

<a id="r078"></a>

#### R078 Distribution Network Topology Identification with Graph Transformer Neural Network

**书目信息**：Zixuan Zhao; Ji Qiao; Jiateng Li; Mengjie Shi; Xiaohui Wang。日期/年份：2022-12。

**来源**：[DOI：10.1109/SPIES55999.2022.10082563](https://doi.org/10.1109/SPIES55999.2022.10082563)。

- **方法**：以历史电压/功率和对应拓扑构建图数据集，利用Graph Transformer学习多跳关联与节点嵌入，完成拓扑识别。
- **输入与设备**：历史监控V/P等数据、图信息、相应拓扑标签；缺测做相似工况/双线性插值。
- **隐藏节点与辨识对象**：在既定节点/图数据上识别；文中少量缺测插补与发现未知隐藏接线点不同。
- **条件与适用边界**：依赖标注工况和插补质量；监督图学习的训练外结构范围需单独验证。
- **综述中的位置**：监督Graph Transformer

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II、§III.C）。

**原文定位**：PDF pp.1–3（§II、§III.C）。

<a id="r082"></a>

#### R082 Power Distribution System Characterization With Active Probing: Real-World Testing and Analysis

**书目信息**：Can Huang; Hannah Burroughs; Cecilia Klauber; Chih-Che Sun。日期/年份：2023。

**来源**：[DOI：10.1109/OAJPE.2022.3227908](https://doi.org/10.1109/OAJPE.2022.3227908)。

- **方法**：用Grid Resonance Probe可编程阻性脉冲扰动真实配电系统，并用μPMU响应开展馈线连接判断、阻抗估计与可视化。
- **输入与设备**：专用GRP精密负载/电流记录、μPMU相量与频率。
- **隐藏节点与辨识对象**：真实场站案例聚焦给定节点间连接状态和等值阻抗，非任意隐藏树恢复。
- **条件与适用边界**：需要专用扰动设备和μPMU；验证范围为五节点真实系统案例。
- **综述中的位置**：主动探测设备代价与实测证据

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II、§III，Fig.1–4）。

**原文定位**：PDF pp.1–3（§II、§III，Fig.1–4）。

<a id="r083"></a>

#### R083 Distribution System Topology Identification Using Graph Neural Networks

**书目信息**：Rahul Madbhavi; Balasubramaniam Natarajan; Babji Srinivasan。日期/年份：2023-05。

**来源**：[DOI：10.1109/GTD49768.2023.00055](https://doi.org/10.1109/GTD49768.2023.00055)。

- **方法**：把待判开关线路当作链路预测任务：保留已知固定连边形成输入图，GCN聚合部分节点量测后预测开关边状态。
- **输入与设备**：已知固定/候选开关线路图、节点状态量测与监督样本。
- **隐藏节点与辨识对象**：节点和候选边已知；量测缺失可容忍，未生成未知接线节点。
- **条件与适用边界**：需各连通分量至少有量测；候选图/标签覆盖依赖。
- **综述中的位置**：已知候选开关GNN

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II.B、Fig.1；量测条件检索）。

**原文定位**：PDF pp.1–3（§II.B、Fig.1；量测条件检索）。

<a id="r084"></a>

#### R084 An Improved Algorithm for Topology Identification of Distribution Networks Using Smart Meter Data and Its Application for Fault Detection

**书目信息**：Cassidy Flynn; Abu Bakr Pengwah; Reza Razzaghi; Lachlan L. H. Andrew。日期/年份：2023-09。

**来源**：[DOI：10.1109/TSG.2023.3239650](https://doi.org/10.1109/TSG.2023.3239650)。

- **方法**：联合拟合电流-电压灵敏度和变压器电压变化，再用改进递归分组在父子/兄弟关系间选择并创建新父节点；灵敏度突变还用于故障检测。
- **输入与设备**：用户智能表电压/电流（由表计量推得），无需PMU；历史同步样本。
- **隐藏节点与辨识对象**：§III明确由端点距离关系创建隐藏父节点h、恢复完整径向网络。
- **条件与适用边界**：径向结构与可辨分支点；根电压变化正则权重和噪声影响灵敏度/分组，实测与随机树结果有各自范围。
- **综述中的位置**：LVDG+隐藏节点核心基线

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II–III，式(5)–(12)）。

**原文定位**：PDF pp.1–3（§II–III，式(5)–(12)）。

<a id="r085"></a>

#### R085 Online Voltage Regulation With Minimum Disturbance for Distribution Grid Without System Knowledge

**书目信息**：Hamad Alduaij; Yang Weng。日期/年份：2024。

**来源**：[DOI：10.1109/OAJPE.2024.3412120](https://doi.org/10.1109/OAJPE.2024.3412120)。

- **方法**：由有限测量与无功控制建立局部线性随机模型，用物理核和逐渐降低探索频率的自适应LQR调节电压。
- **输入与设备**：部分节点电压、PV/储能/电容等连续无功控制。
- **隐藏节点与辨识对象**：用等效输入输出模型吸收未测节点贡献，未输出完整物理拓扑。
- **条件与适用边界**：控制建模/可稳定性和设备能力条件；不能作为完整拓扑恢复方法比较。
- **综述中的位置**：替代方案：未知拓扑下直接控制

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II，式(1)–(2)）。

**原文定位**：PDF pp.1–3（§II，式(1)–(2)）。

<a id="r091"></a>

#### R091 Distribution Network Topology Identification Based on CNN and Improved Conformal Prediction

**书目信息**：Bian Ziyue; Long Huan; Zhao Jingtao; Zheng Shu; Zhang Xiaoyan。日期/年份：2024-09。

**来源**：[DOI：10.1109/CICED63421.2024.10753756](https://doi.org/10.1109/CICED63421.2024.10753756)。

- **方法**：用相关性选取电压特征、CNN分类候选开关拓扑，再用改进conformal prediction给候选拓扑预测集，折衷覆盖和集合大小。
- **输入与设备**：SCADA电压幅值、候选拓扑标签与校准数据。
- **隐藏节点与辨识对象**：固定节点/开关工况分类，无未知物理隐藏节点生成步骤。
- **条件与适用边界**：覆盖结论需核查校准/同分布条件；模型输出分数不应直接当任意拓扑正确概率。
- **综述中的位置**：分类不确定性/预测集

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II–III，式(2)及预测集讨论）。

**原文定位**：PDF pp.1–3（§II–III，式(2)及预测集讨论）。

<a id="r092"></a>

#### R092 Online Learning for Robust Voltage Control Under Uncertain Grid Topology

**书目信息**：Christopher Yeh; Jing Yu; Yuanyuan Shi; Adam Wierman。日期/年份：2024-09。

**来源**：[DOI：10.1109/TSG.2024.3383804](https://doi.org/10.1109/TSG.2024.3383804)。

- **方法**：在线保留与观测一致的灵敏度参数凸集合，用nested convex body chasing选模型，并用鲁棒预测控制将电压拉回安全区间。
- **输入与设备**：节点电压/无功控制、初始紧凸参数集合、噪声界和操作限值。
- **隐藏节点与辨识对象**：学习允许网络集合/端口灵敏度而非唯一物理拓扑；不以隐藏节点重建为目标。
- **条件与适用边界**：径向线性化和已知不确定集合/噪声界等条件；有限错误控制保证的对象是电压约束。
- **综述中的位置**：拓扑不确定下直接鲁棒控制

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II、Algorithm1，式(2)–(8)）。

**原文定位**：PDF pp.1–3（§II、Algorithm1，式(2)–(8)）。

<a id="r097"></a>

#### R097 A power grid topology detection method based on edge graph attention neural network

**书目信息**：Chunxia Zhao; Xueping Li; Yao Cai。日期/年份：2025-02-01。

**来源**：[DOI：10.1016/j.epsr.2024.111219](https://doi.org/10.1016/j.epsr.2024.111219)。

- **方法**：先生成重构电网样本，再用多头Edge-GAT联合聚合节点与边特征，检测重构后的线路拓扑。
- **输入与设备**：候选基础图、节点/边特征和仿真拓扑训练样本。
- **隐藏节点与辨识对象**：既定节点/候选边上的检测，未显式添加物理隐藏节点。
- **条件与适用边界**：IEEE14/39/118测试主要是输电型系统；模型复杂度、训练分布和噪声限制，不能视为已验证LVDG隐藏点方案。
- **综述中的位置**：高压/一般GNN检测的任务对照

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§2及方法摘要）。

**原文定位**：PDF pp.1–3（§2及方法摘要）。

<a id="r098"></a>

#### R098 Power Distribution Network Topology Detection Using Dual-Graph Structure Graph Neural Network Model

**书目信息**：Afshin Ebtia; Mohsen Ghafouri; Mourad Debbabi; Marthe Kassouf; Arash Mohammadi。日期/年份：2025-03。

**来源**：[DOI：10.1109/TSG.2024.3512456](https://doi.org/10.1109/TSG.2024.3512456)。

- **方法**：训练时以节点属性相似图生成嵌入、用结构感知损失对齐实际拓扑；推断时以双图结构GNN作归纳链路预测识别带电线路。
- **输入与设备**：μPMU量测、已知线路基础设施/阻抗和训练拓扑图。
- **隐藏节点与辨识对象**：p2将TD定义为已知线路基础设施上的通断检测；归纳新节点/图不等于由无量测数据恢复未知物理隐藏点。
- **条件与适用边界**：依赖μPMU与训练域/可观测性；需保持TD与从零TI的区别。
- **综述中的位置**：μPMU+候选边检测

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§I、TableI）。

**原文定位**：PDF pp.1–3（§I、TableI）。

**对应原记录编号**：R098, R099。

<a id="r100"></a>

#### R100 Constrained Branching Search for Topology Identification Stream Computing With Lightweight Implementation

**书目信息**：Zhuoheng Wang; Jie Gao; Qiushi Cui; Yang Weng。日期/年份：2025-07。

**来源**：[DOI：10.1109/TPWRS.2024.3510940](https://doi.org/10.1109/TPWRS.2024.3510940)。

- **方法**：用电压互信息、上下游电压规律及V/P/Q线性残差构成node-connection约束，通过分支搜索恢复径向连接；加入插入PV定位与流式CommuniDispatch实现。
- **输入与设备**：节点V/P/Q历史/流数据、已知电容器位置及有载调压器信息（存在时）。
- **隐藏节点与辨识对象**：所读模型在已知量测节点间搜索连接，未提供未知数量的无测分支点生成步骤。
- **条件与适用边界**：径向、节点数据及电压次序/线性近似等条件；电容器位置等先验影响约束。
- **综述中的位置**：边缘/流计算、低复杂度结构搜索

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–4（§II.A–B，式(1)–(16)）。

**原文定位**：PDF pp.1–4（§II.A–B，式(1)–(16)）。

<a id="r101"></a>

#### R101 Research on power system topology identification and fault location based on graph neural network

**书目信息**：Zongjin Guo; Ziyi Song; Tianyuan Xu; Yuchen Zhao; Shanglin She; Shuo Liu。日期/年份：2025-08。

**来源**：[DOI：10.1109/ICSECE65727.2025.11256901](https://doi.org/10.1109/ICSECE65727.2025.11256901)。

- **方法**：将电网表示为带节点和边属性的图，GCN联合输出节点状态/故障和边连接预测，损失结合分类与边回归。
- **输入与设备**：节点V/θ/P/Q以及边阻抗、导纳、开关状态等定义特征，图结构和监督标签。
- **隐藏节点与辨识对象**：已知节点/物理图上的状态和连接预测，未涉及无量测未知节点的生成。
- **条件与适用边界**：输入中列入开关状态特征，比较拓扑辨识能力前须澄清哪些状态在推断时可得；仅IEEE33模拟。
- **综述中的位置**：一般GNN/故障定位，非隐藏树核心

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§III.A–C，式(4)–(11)）。

**原文定位**：PDF pp.1–3（§III.A–C，式(4)–(11)）。

<a id="r102"></a>

#### R102 Fast Distribution Grid Topology Estimation via Subset Sum

**书目信息**：Yueyao Xu; Yize Chen。日期/年份：2025-10。

**来源**：[DOI：10.1109/EPEC65543.2025.11230369](https://doi.org/10.1109/EPEC65543.2025.11230369)。

- **方法**：利用径向树的上下游功率求和关系，把子节点归属写成subset-sum，并以层次算法逐级重建连接。
- **输入与设备**：全节点同步功率记录（含父支路汇总量/节点功率关系）。
- **隐藏节点与辨识对象**：原文明确假设各节点完整量测；未推导纯叶端观测下隐藏点插入。
- **条件与适用边界**：径向、功率平衡/损耗近似与足够可区分负荷；噪声和相似子集和影响归属。
- **综述中的位置**：全节点功率守恒/组合搜索

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II.B–C、§III，式(4)–(6)）。

**原文定位**：PDF pp.1–3（§II.B–C、§III，式(4)–(6)）。

<a id="r103"></a>

#### R103 Maximum Likelihood Estimation for Line Parameters in Distribution Grids Based on Expectation Maximization Algorithm

**书目信息**：Shubhankar Kapoor; Adrian G. Wills; Johannes Hendriks; Lachlan Blackhall。日期/年份：2025-11。

**来源**：[DOI：10.35833/MPCE.2024.000910](https://doi.org/10.35833/MPCE.2024.000910)。

- **方法**：在给定径向树上保留线路损耗的非线性DistFlow近似，构建带噪P/Q/V与上游潮流的最大似然问题，用EM估计线阻抗与隐含真实量测。
- **输入与设备**：给定节点/边拓扑、节点P/Q/V幅值及根支路P/Q。
- **隐藏节点与辨识对象**：EM潜变量为真实状态/注入，所读模型估计已知边的参数；不重构未知隐藏拓扑。
- **条件与适用边界**：给定径向结构与非线性损耗近似；EM数值解和实测参数误差结果的适用范围需保留。
- **综述中的位置**：固定拓扑参数估计背景

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II.A、式(6)–(15)）。

**原文定位**：PDF pp.1–3（§II.A、式(6)–(15)）。

<a id="r104"></a>

#### R104 A Topology Identification Strategy of Low-Voltage Distribution Grids Based on Feature-Enhanced Graph Attention Network

**书目信息**：Yang Lei; Fan Yang; Yanjun Feng; Wei Hu; Yinzhang Cheng。日期/年份：2025/1。

**来源**：[DOI：10.3390/en18112821](https://doi.org/10.3390/en18112821)。

- **方法**：把变压器、分支箱、用户建成异构图，以meta-path随机游走和skip-gram强化结构特征，再由F-GAT融合电气/语义特征预测连接。
- **输入与设备**：智能终端节点V/P/Q/I、已有设备实体/类型和图结构、训练标签。
- **隐藏节点与辨识对象**：§3.2针对无用电数据的分支箱通过已有图meta-path建表示；支持已知实体缺电量信息，与未知数量物理隐藏点从端点生成不同。
- **条件与适用边界**：依赖已有异构图/设备信息和监督数据；不能把“提及hidden nodes”自动视作未知隐树恢复证明。
- **综述中的位置**：已知设备图上的部分属性观测

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–6（§2–3，式(1)–(2)）。

**原文定位**：PDF pp.1–6（§2–3，式(1)–(2)）。

<a id="r105"></a>

#### R105 Utilising Smart-Meter Harmonic Data for Low-Voltage Network Topology Identification

**书目信息**：Ali Othman; Neville R. Watson; Andrew Lapthorn; Radnya Mukhedkar。日期/年份：2025/1。

**来源**：[DOI：10.3390/en18133333](https://doi.org/10.3390/en18133333)。

- **方法**：将THD和2–20次谐波电压序列转为相关距离矩阵，通过改进MST-Kruskal重建量测端连接，并评估相别/网络区分与拓扑相似性。
- **输入与设备**：具谐波测量能力的智能表电压指标；可比较RMS电压。
- **隐藏节点与辨识对象**：MST在给定ICP集合上连边；所读算法未加入未量测物理分支节点。
- **条件与适用边界**：需要谐波可测及有区分度传播；主要基于负荷/谐波仿真，不能据此认定普通SM全有该功能。
- **综述中的位置**：谐波特征与仪表功能要求

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3,8–10（§3.1、§4，Algorithm1及算法三阶段）。

**原文定位**：PDF pp.1–3,8–10（§3.1、§4，Algorithm1及算法三阶段）。

**版本与书目说明**：PDF显示2025-06-25出版，与Zotero模糊2025/1日期有差异。

<a id="r106"></a>

#### R106 Autonomous Topology Identification in Distribution Networks Using Smart Meter Data

**书目信息**：Seung-Joo Lee; Ju-Young Ma; Gyeong-Hun Kim; Jae-Won Chang。日期/年份：2026。

**来源**：[DOI：10.1109/TSG.2026.3716690](https://doi.org/10.1109/TSG.2026.3716690)。

- **方法**：先把未观测注入的聚合量和电压残差SVD坐标加入无相角回归，估计观测点Kron约化Y；再用k-means++自动分组和物理一致性校正确定约化连边。
- **输入与设备**：观测节点SM P/Q/V及slack P/Q；无需相量、候选线信息或手动阈值。
- **隐藏节点与辨识对象**：§II式(7)明确估计观测节点间Kron-reduced导纳，未重构被消去的物理hidden节点。
- **条件与适用边界**：部分观测注入的代理回归与聚类假设须单独评估；Kron边不同于原始物理边。
- **综述中的位置**：部分观测的约化拓扑，需与hidden物理树区分

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–5,7（§II、§IV，式(7)–(25)）。

**原文定位**：PDF pp.1–5,7（§II、§IV，式(7)–(25)）。

**版本与书目说明**：阅读附件为accepted author version。

<a id="r107"></a>

#### R107 Confidence-Aware Topology Identification in Low-Voltage Distribution Networks: A Multi-Source Fusion Method Based on Weakly Supervised Learning.

**书目信息**：Siliang Liu; Can Deng; Zenan Zheng; Ying Zhu; Hongxin Lu; Wenze Liu。日期/年份：2026-03-15。

**来源**：[DOI：10.3390/en19061503](https://doi.org/10.3390/en19061503)。

- **方法**：将多种户变/相别/分支识别方法视为弱标签源，EM学习混淆模式和潜在归属标签，再按缺测和熵折扣构造D-S证据，冲突门控融合并输出表计置信分数。
- **输入与设备**：智能表数据、多方法在滑动窗上的离散归属输出和缺测率。
- **隐藏节点与辨识对象**：latent为表计归属标签；识别户变/相别/已有分支类别，未恢复未知分支节点树。
- **条件与适用边界**：依赖弱源误差模型/独立性与共识结构；无真值的融合confidence不自动等于校准的正确概率。
- **综述中的位置**：归属识别融合/可信度与数据质量

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3,6–7（§2–3，式(3)–(15)）。

**原文定位**：PDF pp.1–3,6–7（§2–3，式(3)–(15)）。

<a id="r109"></a>

#### R109 Electric Shock Simulation and Risk Assessment in Low-Voltage Distribution Networks Under Unknown Topology: A Two-Stage Approach Based on Smart Meter Data.

**书目信息**：Zhe Li; Shoukang Luo; Xiaojia Sun; Yang Li; Yubo Zhang; Chakhung Yeung; Yuxuan Ding。日期/年份：2026-06-01。

**来源**：[DOI：10.3390/en19112723](https://doi.org/10.3390/en19112723)。

- **方法**：以叶端P/Q/V/I交替估计复TLM和相角，再以层次聚类逐次创建中间节点恢复径向树和阻抗；随后以MNA-FDTD模拟触电电压并比较接地方式风险。
- **输入与设备**：常规智能表端点P/Q/V/I，根电压；不需同步相角。
- **隐藏节点与辨识对象**：§2.1明确每次合并新增intermediate node，恢复无表分支点。
- **条件与适用边界**：小规模径向例、迭代回归与聚类；唯一性/全局恢复条件未在所读方法段提供，仿真成功须按范围使用。
- **综述中的位置**：LVDG+隐藏节点与安全应用

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–6（§2.1–2.2，式(3)–(12)）。

**原文定位**：PDF pp.1–6（§2.1–2.2，式(3)–(12)）。

<a id="r110"></a>

#### R110 Anomaly detection and topology identification of distribution network based on conditional variational autoencoder

**书目信息**：Rui Xue; Bin Li; Xiang Tang; Qijian Wang; Jinming Chen; Zhi Chen; Hao Jiao; Wu Chen; Wenyue He; Yihang Wang。日期/年份：2026-06-29。

**来源**：[DOI：10.1038/s41598-026-59837-5](https://doi.org/10.1038/s41598-026-59837-5)。

- **方法**：先以带正常/异常标签的CVAE筛除异常样本，再交换生成目标与条件变量，使改进CVAE以量测预测开关拓扑标签，并迁移异常模型参数初始化。
- **输入与设备**：历史量测特征、异常标签与开关/拓扑类别标签。
- **隐藏节点与辨识对象**：潜变量是神经生成表示，拓扑目标为switch-state标签；未加入无量测物理隐藏节点。
- **条件与适用边界**：监督标签/训练分布依赖；异常清洗到拓扑预测的迁移表现需给定数据验证。
- **综述中的位置**：中压数据清洗+分类

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3,6–7,9（§2.2–2.3，式(5)–(6)）。

**原文定位**：PDF pp.1–3,6–7,9（§2.2–2.3，式(5)–(6)）。

**版本与书目说明**：本地附件为Article in Press未编辑版本。

<a id="r111"></a>

#### R111 Adaptive Physics-Informed Data-Driven Topology Identification for Distribution Networks

**书目信息**：Mengzhao Duan; Zhengyang Hu; Junyu Chen; Wenyi Zhang; Yuhong Zhao; Xinyang Su; Yibo Ding; Zhao Xu; Yixiong Jia。日期/年份：2026-08。

**来源**：[DOI：10.1109/TII.2026.3684153](https://doi.org/10.1109/TII.2026.3684153)。

- **方法**：多尺度CNN+通道注意力预测候选支路soft mask，按已知线路参数构造Laplacian并用LCPF电压/相角残差加物理损失，以不确定性权重平衡监督与物理目标。
- **输入与设备**：全节点P/Q、部分μPMU V/θ，候选关联矩阵A和线路r/x及标签。
- **隐藏节点与辨识对象**：明确branch status identification；节点/候选基础图已知，不生成未知物理隐藏点。
- **条件与适用边界**：需参数/候选图和相量量测；物理损失一致性不自动证明域外拓扑正确。
- **综述中的位置**：μPMU+物理引导候选开关分类

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–5（§II–III，式(8)–(21)）。

**原文定位**：PDF pp.1–5（§II–III，式(8)–(21)）。

<a id="r112"></a>

#### R112 Topology Identification in Distribution Systems Using Line Current Sensors: An MILP Approach

**书目信息**：Mohammad Farajollahi; Alireza Shahsavari; Hamed Mohsenian-Rad。日期/年份：3/2020。

**来源**：[DOI：10.1109/TSG.2019.2933006](https://doi.org/10.1109/TSG.2019.2933006)。

- **方法**：将少量非接触线路电流测量与节点注入伪量测放入联合开关/状态模型，经big-M与潮流线性化转为MILP，多快照改善噪声鲁棒性并分析传感器位置。
- **输入与设备**：少量线路电流传感器、P/Q伪量测、已知线路导纳/候选图。
- **隐藏节点与辨识对象**：未测节点的状态可为变量，但物理节点/候选边已知；不恢复未知数量隐藏接线点。
- **条件与适用边界**：候选网和伪量测范围依赖；支持径向/环/孤岛状态不等于从零发现任意物理网。
- **综述中的位置**：电流传感器+候选边MILP

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II，式(1)–(9)）。

**原文定位**：PDF pp.1–3（§II，式(1)–(9)）。

<a id="r114"></a>

#### R114 Topology Identification in Distribution Networks using Harmonic Synchrophasor Measurements

**书目信息**：Lei Chen; Mohammad Farajollahi; Mahdi Ghamkhari; Wei Zhao; Songling Huang; Hamed Mohsenian-Rad。日期/年份：5/2021。

**来源**：[DOI：10.1109/TSG.2020.3038214](https://doi.org/10.1109/TSG.2020.3038214)。

- **方法**：联合基波和谐波电流同步相量，以谐波流通路径约束补充通断信息，再构建MILP，并分析谐波源/传感器布置的可观测性。
- **输入与设备**：可测谐波的PMU、基波/谐波电流、伪注入与候选网络。
- **隐藏节点与辨识对象**：识别既定线路开关状态；没有谐波电流的边单靠谐波不能判通断，需基波补充。
- **条件与适用边界**：谐波源与仪表覆盖/位置、无相应谐振等假设；不是普通SM被动隐藏树方案。
- **综述中的位置**：谐波μPMU设备与可观测性

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II–III，式(1)–(7)）。

**原文定位**：PDF pp.1–3（§II–III，式(1)–(7)）。

**版本与书目说明**：本地原文为arXiv 2020-02-05 v1，对应条目期刊2021。

<a id="r115"></a>

#### R115 Simultaneous Robust State Estimation, Topology Error Processing, and Outage Detection for Unbalanced Distribution Systems

**书目信息**：Zahra Soltani; Shanshan Ma; Mojdeh Khorsand; Vijay Vittal。日期/年份：5/2023。

**来源**：[DOI：10.1109/TPWRS.2022.3181118](https://doi.org/10.1109/TPWRS.2022.3181118)。

- **方法**：采用三相电流-电压AC模型，迭代一阶线性化和big-M表达混相开关，形成单快照MIQP同时作状态估计、拓扑错误处理和停电区域识别，并整合异步SM和同步μPMU。
- **输入与设备**：已知三相网络/参数与候选开关、SM/μPMU及DER/负荷数据。
- **隐藏节点与辨识对象**：量测有限/缺失时估计既定节点状态和开关，未生成未知物理隐藏节点。
- **条件与适用边界**：三相不平衡模型明确含互阻/并联导纳；线性化和量测整合条件影响结果，需区分优化模型与原非线性AC。
- **综述中的位置**：三相候选拓扑/状态/停电联合估计

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II.A、方法贡献）。

**原文定位**：PDF pp.1–3（§II.A、方法贡献）。

<a id="r118"></a>

#### R118 Distribution Network Topology Identification Using Smart Meter Data and Considering the Same-Bus-Different-Feeder Condition

**书目信息**：Zhiqi Xu; Wei Jiang; Junjun Xu; Lei Wu; Minghua Wang; Jinming Chen; Ye Chen; Junbo Zhao。日期/年份：8/2023。

**来源**：[DOI：10.1109/TII.2022.3220855](https://doi.org/10.1109/TII.2022.3220855)。

- **方法**：针对同母线不同馈线电压强相关造成的误连，推导电压-功率依赖，建物理概率网络并用belief propagation推断开关状态。
- **输入与设备**：SM节点V/P/Q、候选开关/物理概率图。
- **隐藏节点与辨识对象**：unobserved在文中为开关状态；未恢复未知隐藏接线节点。
- **条件与适用边界**：避免把电压相关一概视作电气相邻；图模型/依赖关系与候选开关范围需已定。
- **综述中的位置**：电压相关失效反例与候选开关概率推断

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II–III，Fig.1–3）。

**原文定位**：PDF pp.1–3（§II–III，Fig.1–3）。

<a id="r120"></a>

#### R120 Graphical Models in Meshed Distribution Grids: Topology Estimation, Change Detection & Limitations

**书目信息**：Deepjyoti Deka; Saurav Talukdar; Michael Chertkov; Murti V. Salapaka。日期/年份：9/2020。

**来源**：[DOI：10.1109/TSG.2020.2978541](https://doi.org/10.1109/TSG.2020.2978541)。

- **方法**：从线性化复电压波动的逆协方差结构剔除附加统计边，恢复网状电网物理连接并检测线路变化，给出噪声容忍和相关注入影响界。
- **输入与设备**：各节点复电压/幅值与相角时间序列。
- **隐藏节点与辨识对象**：所读理论以各非参考节点电压为变量，未给未知物理hidden节点恢复；边变化检测与结构恢复条件不同。
- **条件与适用边界**：理论假定跨节点注入零均值高斯不相关；结构恢复要求环长>3，变化检测结构条件更弱。
- **综述中的位置**：网状拓扑与统计可辨识条件

**阅读依据**：正文定向阅读；正文定向阅读：PDF pp.1–3（§II–III，Assumption1）。

**原文定位**：PDF pp.1–3（§II–III，Assumption1）。


### 9.4 EV与需求响应（33组）

<a id="r121"></a>

#### R121 Dynamic incentives for electric vehicles charging at supermarket stations: Causal insights on demand flexibility

**书目信息**：Carlos A.M. Silva; José R. Andrade; Amândio Ferreira; André Gomes; Ricardo J. Bessa。日期/年份：12/2025。

**来源**：[DOI：10.1016/j.energy.2025.139407](https://doi.org/10.1016/j.energy.2025.139407)。

- **方法**：分别以双层优化设计超市充电折扣、以双重机器学习估计折扣对充电需求的异质影响；优化与因果评估在文中独立实施。
- **控制权**：间接价格激励；非逐车功率指令
- **响应模型**：观测数据中的平均/异质响应，随小时和星期变化
- **条件与适用边界**：DML因果解释依赖所有相关混杂因素已观测等识别条件；单一超市试点的效应不等于任意节点的确定注入。
- **综述中的位置**：可作价格激励可用性、用户异质性背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,5–7,11；§2.3.4、§2.4、§5。

**版本与书目说明**：正式版Energy 341 (2025) 139407；R142是同DOI重复条目，R121具有全文。

**对应原记录编号**：R121, R142。

<a id="r122"></a>

#### R122 An Optimized EV Charging Model Considering TOU Price and SOC Curve

**书目信息**：Yijia Cao; Shengwei Tang; Canbing Li; Peng Zhang; Yi Tan; Zhikun Zhang; Junxiong Li。日期/年份：2012-3。

**来源**：[DOI：10.1109/tsg.2011.2159630](https://doi.org/10.1109/tsg.2011.2159630)。

- **方法**：把分时电价下的充电费用最小化与SOC相关最大充电功率约束结合，用启发式算法将能量从高价时段移至低价时段。
- **控制权**：用户/充电设备直接执行优化充电计划；电价为外部输入
- **响应模型**：给定SOC、连接时间和电价下的优化响应
- **条件与适用边界**：属于理性优化充电模型，实际用户遵从、随机到离站及功率实现误差不由该模型自动解决。
- **综述中的位置**：传统确定性充电响应模型背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1–3,5；§II、§III、§V。

<a id="r123"></a>

#### R123 Real-Time Load Elasticity Tracking and Pricing for Electric Vehicle Charging

**书目信息**：Nasim Yahya Soltani; Seung-Jun Kim; Georgios B. Giannakis。日期/年份：2015-5。

**来源**：[DOI：10.1109/tsg.2014.2363837](https://doi.org/10.1109/tsg.2014.2363837)。

- **方法**：用条件随机场描述个人充电决策及用户间依赖，以在线镜像下降跟踪价格响应参数，再通过概率推断设计实时价格。
- **控制权**：向个体发送价格；用户决定是否充电
- **响应模型**：二元充电决策的条件概率，含用户依赖和时变参数
- **条件与适用边界**：CRF图表示用户统计依赖，非配电网物理拓扑；功率激励是随机实现，需智能表双向通信和响应记录。
- **综述中的位置**：主动价格激励的随机响应模型与在线学习背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,3–5,10；§II、§III-B、§IV、§VI。

**版本与书目说明**：所读PDF为IEEE已接受待编页版本，非已核对最终分页版。

<a id="r124"></a>

#### R124 Dynamic Pricing and Distributed Energy Management for Demand Response

**书目信息**：Liyan Jia; Lang Tong。日期/年份：2016-02-29。

**来源**：[DOI：10.48550/arXiv.1601.02319](https://doi.org/10.48550/arXiv.1601.02319)。

- **方法**：建立零售商日前小时定价与消费者实时用电的Stackelberg博弈，在热负荷模型下推导聚合需求对全天价格向量的仿射关系并刻画利润与消费者剩余权衡。
- **控制权**：日前价格；用户能源管理器响应
- **响应模型**：热负荷随机模型下的最优期望需求，具有跨时段耦合
- **条件与适用边界**：仿射结构依赖HVAC模型和理性最优响应等条件；该文主体为恒温负荷，并非EV或拓扑恢复算法。
- **综述中的位置**：价格—功率模型与时序耦合的理论背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1–3,7；§II、Theorem 1、§VII。

**版本与书目说明**：本地为arXiv稿。

<a id="r125"></a>

#### R125 Modeling demand flexibility of electric vehicles

**书目信息**：Chirag Ramesh Srivatsa; Desik Rengarajan; Vamsi Krishna Tumuluru。日期/年份：2017-10。

**来源**：[DOI：10.1109/smartgridcomm.2017.8340702](https://doi.org/10.1109/smartgridcomm.2017.8340702)。

- **方法**：针对异质EV的到离站、能量和充电约束，多次求解凸充电优化获得参考及随机电价场景响应，再构造24小时价格弹性矩阵。
- **控制权**：聚合商安排车队充电；价格场景外给
- **响应模型**：优化生成的聚合跨时段响应；随机性主要来自电价场景
- **条件与适用边界**：弹性由给定优化模型生成，未据此验证真实用户对价格的因果响应。
- **综述中的位置**：聚合灵活性及跨时段弹性背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1–4,6；§II、§II-C、§IV。

<a id="r126"></a>

#### R126 Risk-Sensitive Learning and Pricing for Demand Response

**书目信息**：Kia Khezeli; Eilyan Bitar。日期/年份：2018-06-18。

**来源**：[DOI：10.48550/arXiv.1611.07098](https://doi.org/10.48550/arXiv.1611.07098)。

- **方法**：在未知仿射需求减量及加性随机冲击模型下，同时估计需求参数和冲击分位数，以扰动的短视定价平衡学习与风险敏感收益。
- **控制权**：统一减载奖励价格；用户自愿响应
- **响应模型**：未知仿射均值加有界随机冲击
- **条件与适用边界**：理论采用需求线性、随机扰动分布正则性等条件；研究对象是一般减载DR，并非EV拓扑。
- **综述中的位置**：间接激励下学习—控制权衡背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1–2,4–5,7；§II-A、§III、§IV、§VII。

**版本与书目说明**：本地为arXiv:1611.07098v3（2018-06-18）。

<a id="r127"></a>

#### R127 Control of Charging of Electric Vehicles Through Menu-Based Pricing

**书目信息**：Arnob Ghosh; Vaneet Aggarwal。日期/年份：2018-11。

**来源**：[DOI：10.1109/tsg.2017.2698830](https://doi.org/10.1109/tsg.2017.2698830)。

- **方法**：在线向每个到达用户提供“充电电量—完成期限—价格”菜单，用户选合同或拒绝；据服务成本与效用信息分析社会福利和运营利润。
- **控制权**：用户选菜单后充电站直接调度履约
- **响应模型**：合同选择受用户效用影响；已接受合同需满足能量/期限
- **条件与适用边界**：定价最优性依赖可用效用信息及相应假设；用户选择和到达未知，不能直接指定其接受某个探测功率。
- **综述中的位置**：可解释价格激励与选择自主权的背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,5–8,13–15,24；§II、§III、§IV–V、§VII。

**版本与书目说明**：本地PDF为31页DRAFT，不是最终期刊排版版。

<a id="r128"></a>

#### R128 Stochastic Dynamic Pricing for EV Charging Stations With Renewable Integration and Energy Storage

**书目信息**：Chao Luo; Yih-Fang Huang; Vijay Gupta。日期/年份：2018-3。

**来源**：[DOI：10.1109/tsg.2017.2696493](https://doi.org/10.1109/tsg.2017.2696493)。

- **方法**：把多站价格、购电与储能决策放入多时段随机动态规划，在需求、可再生能源和批发电价不确定性下权衡收益、满意度和电网影响，并设置收益风险约束。
- **控制权**：站端定价及储能/购电直接控制
- **响应模型**：随机充电需求；跨站价格响应与储能时序耦合
- **条件与适用边界**：需要需求/随机过程模型与网络影响信息；解决充电站运营，未辨识未知物理拓扑。
- **综述中的位置**：价格激励的随机运营背景，非核心拓扑文献。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,3–4,6–7,11；§II、§V、§VII。

**版本与书目说明**：本地为arXiv:1801.02128v1（2018-01-07）。

<a id="r129"></a>

#### R129 Behavioral Modeling of Electric Vehicles Using Price Elasticities

**书目信息**：Anuradha Kannan; Vamsi Krishna Tumuluru。日期/年份：2018-5。

**来源**：[DOI：10.1109/isgt-asia.2018.8467804](https://doi.org/10.1109/isgt-asia.2018.8467804)。

- **方法**：用凸优化分别生成多个负荷节点上的EV参考充放电及随机价格场景响应，再以价格弹性矩阵描述全天需求迁移和V2G供给灵活性。
- **控制权**：聚合商优化充放电；节点电价外给
- **响应模型**：模型生成的24小时聚合响应，包含充电与放电
- **条件与适用边界**：传输网节点与EV物理/调度参数已给；得到的弹性描述依赖模型，不提供隐藏配电节点恢复。
- **综述中的位置**：聚合V2G与时空弹性背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1–3,6；§II、§IV。

<a id="r130"></a>

#### R130 Supporting power balance in Microgrids with Uncertain Production using Electric Vehicles and Indirect Control

**书目信息**：Frederik Banis; Daniela Guericke; Henrik Madsen; Niels Kjølstad Poulsen。日期/年份：2019。

**来源**：[DOI：10.1016/j.ifacol.2019.08.238](https://doi.org/10.1016/j.ifacol.2019.08.238)。

- **方法**：结合场景随机规划、再调度、模型预测控制与在线系统辨识，以直接可控设备和受价格间接影响的EV共同维持微网功率平衡。
- **控制权**：混合直接控制和价格间接控制
- **响应模型**：带不确定基线及脉冲响应的价格敏感模型；按时间簇更新
- **条件与适用边界**：价格敏感单元的响应需在线估计；微网控制模型已给，主体为频率/功率平衡而非拓扑辨识。
- **综述中的位置**：说明直接与间接控制区别、系统辨识需求。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.2–4,6；§2、式(10)–(11)、§4。

**版本与书目说明**：本地为DTU仓储标注的peer reviewed version，首页另有仓储封面。

<a id="r131"></a>

#### R131 A Distributed Online Pricing Strategy for Demand Response Programs

**书目信息**：Pan Li; Hao Wang; Baosen Zhang。日期/年份：2019-01。

**来源**：[DOI：10.1109/TSG.2017.2739021](https://doi.org/10.1109/TSG.2017.2739021)。

- **方法**：在用户二次响应成本未知且通信受限时，以含噪聚合响应做线性回归，在线更新统一激励价格并给出相应遗憾分析。
- **控制权**：每时隙广播一次价格；用户自主减载
- **响应模型**：理性二次成本导出的线性响应加噪声
- **条件与适用边界**：分析依赖特定用户成本与噪声模型；研究一般DR，不识别物理线路。
- **综述中的位置**：低通信在线价格学习背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,3–5,9；§II-B、§III–IV、§VI。

<a id="r132"></a>

#### R132 Inverse optimization with kernel regression: Application to the power forecasting and bidding of a fleet of electric vehicles

**书目信息**：Ricardo Fernández-Blanco; Juan Miguel Morales; Salvador Pineda; Álvaro Porras。日期/年份：2021-10。

**来源**：[DOI：10.1016/j.cor.2021.105405](https://doi.org/10.1016/j.cor.2021.105405)。

- **方法**：以两步凸估计和核回归学习EV车队聚合功率优化模型的时变上下界及分段边际效用，同时输出功率预测和市场报价曲线。
- **控制权**：预测理性聚合商的价格响应；非逐车直接激励
- **响应模型**：历史价格、功率、驾驶/气象等特征决定的优化代理响应
- **条件与适用边界**：聚合代理模型与样本分布决定可外推范围；预测误差与真实用户随机性仍存在。
- **综述中的位置**：由历史价量数据学习响应模型的背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,7–9,44；§2、§5。

**版本与书目说明**：所读本地稿为51页作者/接受前后稿式排版，非最终卷期分页核对。

<a id="r133"></a>

#### R133 Optimal Pricing of Vehicle-to-Grid Services Using Disaggregate Demand Models

**书目信息**：Charilaos Latinopoulos; Aruna Sivakumar; John W. Polak。日期/年份：2021-2-19。

**来源**：[DOI：10.3390/en14041090](https://doi.org/10.3390/en14041090)。

- **方法**：用离散选择的潜类别用户需求模型表示充电/V2G套餐选择，并用遗传算法和粒子群优化各套餐价格以提高停车充电运营收入。
- **控制权**：间接菜单定价；用户选择套餐后站端提供服务
- **响应模型**：潜类别随机效用/选择概率；停车位及配电容量约束
- **条件与适用边界**：latent class表示用户偏好类型，非隐藏电气节点；仿真收益依赖经验估计的需求模型。
- **综述中的位置**：菜单选择和用户异质性背景，非隐藏节点文献。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,5–6,24；§3、§6。

<a id="r134"></a>

#### R134 Data-Driven Approach for Analyzing Spatiotemporal Price Elasticities of EV Public Charging Demands Based on Conditional Random Fields

**书目信息**：Zhiyuan Bao; Zechun Hu; Daniel M. Kammen; Yifan Su。日期/年份：2021-9。

**来源**：[DOI：10.1109/tsg.2021.3080460](https://doi.org/10.1109/tsg.2021.3080460)。

- **方法**：基于历史公共充电价量数据建立时空条件随机场，以带遗忘因子的参数学习和循环信念传播估计本时段、跨时段及跨区域价格弹性。
- **控制权**：从既有价格记录估计响应；未直接控制个人
- **响应模型**：区域聚合需求的概率模型及空间/时间依赖
- **条件与适用边界**：CRF中的hidden变量是区域充电需求，非配电网隐藏节点；历史价量关联不自动等同随机实验的因果弹性。
- **综述中的位置**：间接激励的时空耦合与统计学习背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,3,9–12；§II、§V、§VI。

<a id="r135"></a>

#### R135 Demand response model identification and behavior forecast with OptNet

**书目信息**：Yuexin Bian; Ningkun Zheng; Yang Zheng; Bolun Xu; Yuanyuan Shi。日期/年份：2022-6-28。

**来源**：[DOI：10.1145/3538637.3538871](https://doi.org/10.1145/3538637.3538871)。

- **方法**：把已知形式的用户优化问题嵌入OptNet可微层，利用历史价格—响应数据梯度学习未知参数，再预测新价格下的需求。
- **控制权**：行为模型辨识；用户按所设优化目标响应
- **响应模型**：可微优化代理，示例包括电池、建筑及EV；参数视为相对稳定
- **条件与适用边界**：参数收敛证明针对二次目标、线性等式约束等限定模型；总体预测/参数恢复不自动推广到所有用户模型。
- **综述中的位置**：价格响应模型辨识方法背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1–3,11；§3–4、§7。

<a id="r136"></a>

#### R136 Field Experimental Evidence on the Effect of Pricing on Residential Electricity Conservation

**书目信息**：Jesse Burkhardt; Kenneth T. Gillingham; Praveen K. Kopalle。日期/年份：2023-12。

**来源**：[DOI：10.1287/mnsc.2020.02074](https://doi.org/10.1287/mnsc.2020.02074)。

- **方法**：在得州住宅开展两年随机现场实验，比较信息、关键峰时高价及夜间低价，以高频分设备量测分解节电与负荷迁移。
- **控制权**：价格/信息干预；家庭自行调整用电
- **响应模型**：实测群体处理效应；包含空调与EV充电迁移
- **条件与适用边界**：随机化针对自愿参加家庭；该区域家庭/气候和设备自动化程度限制外推。
- **综述中的位置**：住宅价格响应的现场证据，非核心拓扑方法。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,5,9–12,31；§2、§6。

**版本与书目说明**：元数据题名为Effect of Pricing，所读PDF题名为Effects of Information and Pricing，首页2021-12-10；需作为早期稿引用，不能假定与2023正式版逐字一致。

<a id="r137"></a>

#### R137 Learning and optimizing charging behavior at PEV charging stations: Randomized pricing experiments, and joint power and price optimization

**书目信息**：Hassan Obeid; Ayse Tugba Ozturk; Wente Zeng; Scott J. Moura。日期/年份：2023-12。

**来源**：[DOI：10.1016/j.apenergy.2023.121862](https://doi.org/10.1016/j.apenergy.2023.121862)。

- **方法**：通过随机价格实验估计用户选择普通/可调度充电及插枪时长的行为模型，再联合优化按时计价和站端功率计划。
- **控制权**：用户通过价格选择服务；选可调度服务后站端直接调度
- **响应模型**：二项logit选择概率及价格相关停留时长
- **条件与适用边界**：行为参数来自现场实验，联合优化收益在后续模拟中评估；结论明确实时闭环优化未在该基础设施上实施。
- **综述中的位置**：最适合连接真实用户选择与主动激励设计的EV背景之一。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,4–5,7,12–13；§3、用户选择模型、§7。

<a id="r138"></a>

#### R138 Resource aware pricing for electric vehicle charging

**书目信息**：Cesar Santoyo; Gustav Nilsson; Samuel Coogan。日期/年份：2023-2。

**来源**：[DOI：10.1016/j.automatica.2022.110733](https://doi.org/10.1016/j.automatica.2022.110733)。

- **方法**：在泊松到达和随机需求/时间价值模型下，研究离散速率菜单或截止期定价，并推导站点车辆数及总功率超限概率上界。
- **控制权**：站点定价；用户理性选择服务速率/期限
- **响应模型**：随机到达、需求、不耐烦和停留时间；聚合功率随机
- **条件与适用边界**：概率界依赖泊松到达、跨用户iid、有界随机参数和已知相应分布等条件；不能直接作为拓扑正确概率。
- **综述中的位置**：可用作随机价格激励及资源风险约束背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1–3,5–6,13；Assumption 1、§3、§5。

**版本与书目说明**：本地为arXiv:2009.10771v1（2020-09-22），元数据为2023 Automatica正式版。

<a id="r139"></a>

#### R139 Unraveling the effect of electricity price on electric vehicle charging behavior: A case study in Shenzhen, China

**书目信息**：Haoxuan Kuang; Xinyu Zhang; Haohao Qu; Linlin You; Rui Zhu; Jun Li。日期/年份：2024-11。

**来源**：[DOI：10.1016/j.scs.2024.105836](https://doi.org/10.1016/j.scs.2024.105836)。

- **方法**：从深圳公共充电记录出发，用相关性和Granger检验筛查，再以图注意力、时间注意力与知识嵌入元学习构建回归模型，通过冲击响应分析时空弹性和溢出。
- **控制权**：观测既有公共价格与充电占用；无随机价格指派
- **响应模型**：时空回归预测及模型内价格冲击响应
- **条件与适用边界**：观测数据、Granger检验及代理模型冲击的识别条件与随机现场实验不同；空间充电需求图非物理线路拓扑。
- **综述中的位置**：价格响应时空迁移的观测证据。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,4,7,10–11；方法框架、§5、§6.2。

<a id="r140"></a>

#### R140 Optimal Vehicle Charging in Bilevel Power-Traffic Networks via Charging Demand Function

**书目信息**：Yufan Zhang; Sujit Dey; Yuanyuan Shi。日期/年份：2024-5。

**来源**：[DOI：10.1109/tsg.2023.3327070](https://doi.org/10.1109/tsg.2023.3327070)。

- **方法**：将电网定价与交通流/充电需求放入双层优化，以多参数规划推导分区的充电需求函数，简化两个运营主体的耦合求解。
- **控制权**：电价间接引导交通与充电决策
- **响应模型**：给定网络和理性交通分配模型的需求函数
- **条件与适用边界**：所读版为单时段和单相网络模型，文中把多时段及不平衡拓展列作未来工作；网络拓扑与参数给定。
- **综述中的位置**：价格—地点选择耦合理论背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1–3,5–6,10；§II–III及末尾讨论。

**版本与书目说明**：本地为arXiv:2304.11284v2（2023-09-05），元数据为2024期刊发表。

<a id="r141"></a>

#### R141 Learning and Optimization for Price-Based Demand Response of Electric Vehicle Charging

**书目信息**：Chengyang Gu; Yuxin Pan; Ruohong Liu; Yize Chen。日期/年份：2024-7-10。

**来源**：[DOI：10.23919/acc60939.2024.10644254](https://doi.org/10.23919/acc60939.2024.10644254)。

- **方法**：用多层感知器学习价格到EV充电需求的映射，再通过可微二次规划层以运营成本反向训练，使预测与站端调度共同优化。
- **控制权**：价格输入下需求预测，站端功率直接调度
- **响应模型**：学习的需求映射及确定性调度代理
- **条件与适用边界**：所读实验为模拟比较；端到端运营损失改进不等于真实用户响应确定或已获得独立激励。
- **综述中的位置**：响应预测与下游目标联合学习背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1–4,6；§II–III、§V。

**版本与书目说明**：本地为arXiv:2404.10311v1（2024-04-16），对应ACC 2024条目。

<a id="r143"></a>

#### R143 The Impact of Dynamic Prices on Electric Vehicle Public Charging Demand: Evidence from a Nationwide Natural Field Experiment

**书目信息**：Louise Bernard; Andy Hackett; Robert Metcalfe; Luca Panzone; Andrew Schein。日期/年份：2025-12。

**来源**：[DOI：10.3386/w34600](https://doi.org/10.3386/w34600)。

- **方法**：在英国大型公共充电平台随机向用户指派降价和环保信息通知，估计短期充电需求处理效应并分析平台切换与消费者福利。
- **控制权**：随机价格/信息通知；用户自主决定地点和时间
- **响应模型**：现场随机实验的群体短期因果响应
- **条件与适用边界**：仅覆盖合作平台可观测需求及临时降价；平台内需求增加混有支付渠道切换，不能全算新电量或跨时段负荷迁移。
- **综述中的位置**：公共充电价格响应的强现场证据，可作为补充。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,13,17,24,45；实验设计、§2.3.1、§4.1、§7。

**版本与书目说明**：元数据NBER工作论文DOI；本地119页稿题名缺少Nationwide一词，属于工作论文版本而非已核验期刊版。

<a id="r144"></a>

#### R144 Electric Vehicles and the Energy Transition: Unintended Consequences of Time-of-Use Pricing

**书目信息**：Megan R. Bailey; David P. Brown; Erica Myers; Blake Shaffer; Frank A. Wolak。日期/年份：2025-12-1。

**来源**：[DOI：10.1257/aeri.20240476](https://doi.org/10.1257/aeri.20240476)。

- **方法**：通过现场实验比较统一TOU定价和中央托管充电，并用虚拟变压器聚合评估充电同步对局部容量的影响。
- **控制权**：TOU为间接价格；managed charging为直接时序控制
- **响应模型**：真实用户充电时移及同步响应
- **条件与适用边界**：实测个体充电叠加到虚拟变压器容量评估，不是实际重建隐藏线路；TOU在该场景产生新同步峰而托管充电可缓解。
- **综述中的位置**：支持低压局部约束和拓扑信息重要性的首选EV实证；也提示独立注入条件需审查。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,4–7,10,15；§I–II、§IV、§V。

<a id="r145"></a>

#### R145 V2G for Frequency Regulation Service: A Stackelberg Game Approach Considering Endogenous Uncertainties

**书目信息**：Jingxiang Wang; Zhaojian Wang; Bo Yang; Feng Liu; Wei Wei; Xinping Guan。日期/年份：2025-2。

**来源**：[DOI：10.1109/tte.2024.3392496](https://doi.org/10.1109/tte.2024.3392496)。

- **方法**：以Stackelberg博弈联合聚合商定价和EV充放电，把期望SOC的不确定性与价格趋势关联，用分布鲁棒机会约束及Cantelli不等式转化为锥约束。
- **控制权**：聚合商价格引导EV最优充放电
- **响应模型**：价格相关均值/矩信息的内生SOC不确定性
- **条件与适用边界**：模型假定EV愿意参与V2G，并给定SOC不确定性对价格的线性均值映射等结构；未直接学习全部真实分布。
- **综述中的位置**：决策相关不确定性建模背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,3,5–7,11；§III–IV、§VI。

<a id="r146"></a>

#### R146 Show Me the Money! A Field Experiment on Electric Vehicle Charge Timing

**书目信息**：Megan R. Bailey; David P. Brown; Blake Shaffer; Frank A. Wolak。日期/年份：2025-5-1。

**来源**：[DOI：10.1257/pol.20230653](https://doi.org/10.1257/pol.20230653)。

- **方法**：在家庭EV现场实验中随机比较非高峰充电奖励、亲社会信息和对照，随后移除奖励以检验行为持续性。
- **控制权**：金钱/信息激励；用户自行决定充电时刻
- **响应模型**：随机实验的个体/群体充电时间处理效应
- **条件与适用边界**：主要证据为该实验群体的时间迁移；移除奖励后行为回退，不能把价格激励视为永久且精确的功率执行器。
- **综述中的位置**：支持价格可激发EV时间灵活性的首选现场证据。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,4–5,7–10,24；§I、§III、§VI。

<a id="r147"></a>

#### R147 The impacts of rate surge on electric vehicle charging behaviors: Evidence from California

**书目信息**：Shulong Luo; Yucheng Wang。日期/年份：2025-8。

**来源**：[DOI：10.1016/j.jebo.2025.107048](https://doi.org/10.1016/j.jebo.2025.107048)。

- **方法**：利用加州Palo Alto公共充电由免费改为充电费加超时停车费的政策变化，以时间断点回归分析会话数量、电量、停车时长和用户构成变化。
- **控制权**：公共费率政策；用户自主响应
- **响应模型**：准实验的收费组合影响及用户异质性
- **条件与适用边界**：价格与停车费同期变化，时间断点识别依赖其他同期因素平滑等条件；不同于随机分配单一电价的实验。
- **综述中的位置**：公共价格响应与需求构成变动的补充证据。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,5,9,17；§4–5、§7。

<a id="r148"></a>

#### R148 Demand Response Under Stochastic, Price-Dependent User Behavior

**书目信息**：Guido Cavraro; Andrey Bernstein; Emiliano Dall'Anese。日期/年份：2026-03-16。

**来源**：[DOI：10.48550/arXiv.2603.15983](https://doi.org/10.48550/arXiv.2603.15983)；[另核正式版DOI：10.1109/LCSYS.2026.3711602](https://doi.org/10.1109/LCSYS.2026.3711602)。

- **方法**：把居民响应建模为价格相关随机变量，结合决策相关分布优化与正则化原对偶更新，以变电站实测功率反馈实时调整价格。
- **控制权**：价格间接控制；变电站功率闭环反馈
- **响应模型**：价格决定分布的随机响应，并允许灵敏度估计误差
- **条件与适用边界**：稳定和近最优结果采用分布、凸性及估计误差等条件；研究调节功率交换而非恢复隐藏拓扑，文中线性网络灵敏度用于算法。
- **综述中的位置**：连接概率价格响应和闭环主动激励最直接的理论背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1–4,7；§II–IV、Algorithm 1、§VI。

**版本与书目说明**：本地为arXiv:2603.15983v1；官方NLR研究记录另列2026正式发表IEEE CSL DOI 10.1109/LCSYS.2026.3711602。

<a id="r149"></a>

#### R149 V2G Regulation in Distribution Networks With Decision-Dependent Uncertainties: A Chance-Constrained Learning Method

**书目信息**：Jingxiang Wang; Zhaojian Wang; Qianwen Xu; Baocong Zhang; Bo Yang; Feng Liu; Xinping Guan。日期/年份：2026-2。

**来源**：[DOI：10.1109/tte.2025.3630493](https://doi.org/10.1109/tte.2025.3630493)。

- **方法**：在配电网V2G双层定价框架中，用分位数回归神经网络学习价格、到离站时间条件下的需求分位数，并嵌入机会约束和混合整数近似求解。
- **控制权**：定价间接影响EV，系统优化协调V2G
- **响应模型**：通过QRNN学习的条件分位数表示决策相关需求不确定性
- **条件与适用边界**：学习到的分位数需数据覆盖与校准；嵌入ReLU模型引入整数变量，已有电网结构和参数为输入。
- **综述中的位置**：概率价格响应的学习建模背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,3,5–7,13；§III-A、§IV、§VI。

<a id="r150"></a>

#### R150 Learning the Price Response of Spatiotemporal EV Charging Flow: A Graph-Attentive Surrogate Model

**书目信息**：Sheng Chen; Han Zhang; Qiuwei Wu; Peiyue Li; Zhinong Wei。日期/年份：2026-4。

**来源**：[DOI：10.1109/tte.2026.3657374](https://doi.org/10.1109/tte.2026.3657374)。

- **方法**：用GCN、双向交叉注意力和Transformer学习多时段耦合交通—配电网优化的价量映射，以代理模型快速预测时空充电流并支持调度。
- **控制权**：服务费/电价间接引导交通和充电
- **响应模型**：从优化样本学习的非线性时空需求响应
- **条件与适用边界**：明确假定交通/配电拓扑静态已知，训练标签依赖集中方可获得网络和EV参数；其图学习不是未知配电拓扑恢复。
- **综述中的位置**：规模化时空响应代理背景，非拓扑辨识证据。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,3,5–6,8–9,11；§II–III、§IV、§V。

<a id="r151"></a>

#### R151 A Comprehensive Review of Consumer Models in Price-Based Demand Response and Their Applications to Electric Vehicles

**书目信息**：Qinhao Li; Suchun Fan; Lai Zhou; Zhongwen Wang; Pan Qi。日期/年份：2026-6-11。

**来源**：[DOI：10.3390/en19122809](https://doi.org/10.3390/en19122809)。

- **方法**：按规则、优化、数据驱动和混合四类梳理价格型需求响应消费者模型，并讨论EV约束、信息/能量/资金交互及研究缺口。
- **控制权**：综述涵盖多种间接价格及消费者优化形式
- **响应模型**：综述，不提供独立新的价格响应因果实验
- **条件与适用边界**：可用于分类与追溯原始论文；方法性能和因果效应应回到对应原始研究。
- **综述中的位置**：写价格响应背景时的组织线索，宜少量引用。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,4–8,33；§2–3、§5。

<a id="r152"></a>

#### R152 Reducing peak load from electric vehicles through grid tariffs: Evidence from a randomized field experiment

**书目信息**：Floris van Montfoort; Peter T. Dijkstra; Machiel Mulder; Dávid Kopányi。日期/年份：2026-8。

**来源**：[DOI：10.1016/j.jeem.2026.103354](https://doi.org/10.1016/j.jeem.2026.103354)。

- **方法**：在600余个荷兰家庭的随机现场实验中比较全时和仅峰时的容量超限收费，分析充电速率、时移与聚合容量超限。
- **控制权**：容量收费间接激励；家庭自行调速/延迟
- **响应模型**：随机实验响应；静态收费与TOU收费作用机制不同
- **条件与适用边界**：聚合电网容量分析采用合成约束；TOU在该场景主要时移且可形成移峰后的新峰，不代表任意收费都降低局部峰值。
- **综述中的位置**：局部配电容量与激励协调重要性的最新现场补充证据。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,4,6,9–10,17；§3–4、§6–7。

<a id="r153"></a>

#### R153 Dynamic Behavior Modeling of Price-Responsive Flexible Loads: A Deep Learning-Parameterized Inverse Optimization Approach

**书目信息**：Lingfeng Tang; Haipeng Xie; Zhaohong Bie。日期/年份：2026-9。

**来源**：[DOI：10.1109/tsg.2026.3672613](https://doi.org/10.1109/tsg.2026.3672613)。

- **方法**：用多任务深度网络参数化动态虚拟储能代理，再以可微逆优化从历史价量及公开天气/日历数据学习时变柔性负荷行为。
- **控制权**：学习用户优化代理；价格由上层应用给定
- **响应模型**：随外部特征变化的动态可行域和优化响应
- **条件与适用边界**：预测代理的时变性不自动提供真实行为概率分布；方法是柔性负荷建模，电网拓扑辨识未作为目标。
- **综述中的位置**：时变价格响应辨识背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.1,4–8,13；§III–IV、§VI。

<a id="r154"></a>

#### R154 Robust scheduling of electric vehicle charging stations with price-based demand response under decision-dependent uncertainty

**书目信息**：Guowei Liu; Zhanxi Zhang; Liming Wang; Jiaming Chen; Wen Tao; Xiaoyuan Xu。日期/年份：2026-9。

**来源**：[DOI：10.1016/j.gloei.2026.06.009](https://doi.org/10.1016/j.gloei.2026.06.009)。

- **方法**：以自/交叉价格弹性表示站间需求迁移，用价格偏差相关的响应误差集合构建两阶段鲁棒充电站调度，并以改进Benders分解求解。
- **控制权**：站间价格引导需求分配；站端再调度
- **响应模型**：回归估计弹性加价格相关集合不确定性
- **条件与适用边界**：弹性估计来自概率交通分配与价格扰动样本；鲁棒集合保证的对象是所设集合内调度，非真实概率覆盖或物理拓扑。
- **综述中的位置**：价格相关不确定性及站间迁移背景。

**阅读依据**：正文定向阅读；正文定向阅读：摘要、所列方法/模型段与结论或讨论。

**原文定位**：PDF pp.2,4–6,8–11,17；§1.3、§3、§5。

**版本与书目说明**：本地为Journal Pre-proofs，封面明确非最终Version of Record，正文存在未修复参考文献域错误。


### 9.5 基础理论和其他应用（38组）

<a id="r001"></a>

#### R001 Bayesian 𝑇-optimal discriminating designs

**书目信息**：。日期/年份：未记录。

**来源**：[稳定来源](https://arxiv.org/abs/1412.2548)。

- **方法**：把参数先验积分离散化为多模型比较，交替增补设计支撑点和优化权重，计算Bayesian T最优辨别设计。
- **输入与设备**：候选回归模型、参数先验、可选实验点
- **隐藏节点与辨识对象**：通用实验设计理论
- **条件与适用边界**：依赖候选模型及先验；所读PDF含2014 arXiv编号和2021排版日期，不能据此把两者当作独立研究。
- **综述中的位置**：仅在写主动探测设计时引用

**阅读依据**：正文定向阅读；正文定向阅读：摘要、引言算法概述、§2模型。

**原文定位**：PDF pp.1,3–5，§2及算法概述。

<a id="r002"></a>

#### R002 Grid Topology Identiﬁcation via Distributed Statistical Hypothesis Testing

**书目信息**：Saverio Bolognani。日期/年份：未记录。

**来源**：本地Zotero英文PDF；按本项题名定位。稳定公开书目入口待补。

- **方法**：从线性化潮流推导电压协方差的稀疏图结构，集中式恢复网络，并将局部连接问题转为分布式统计假设检验。
- **输入与设备**：节点历史电压幅值
- **隐藏节点与辨识对象**：集中式算法要求所有母线电压可测；局部假设检验的任务范围较小。
- **条件与适用边界**：不同节点注入互不相关，另用统一X/R或统一功率因数；模型设根电压恒定。
- **综述中的位置**：被动统计方法与信息条件比较

**阅读依据**：正文定向阅读；正文定向阅读：摘要、模型、Assumptions 1–3、结论。

**原文定位**：PDF pp.1–4，§2–3，p.11 §6。

<a id="r003"></a>

#### R003 Power Theft Localization using Voltage Measurements from Distribution Feeder Nodes

**书目信息**：Kalyan Dasgupta; M Padmanaban; J Hazra。日期/年份：未记录。

**来源**：本地Zotero英文PDF；按本项题名定位。稳定公开书目入口待补。

- **方法**：利用相邻节点电压差及归一化二阶电压差，结合Mahalanobis异常检测定位非法取电区域。
- **输入与设备**：相邻馈线节点或变压器的高精度电压
- **隐藏节点与辨识对象**：任务为窃电定位
- **条件与适用边界**：需相邻关系和精确电压；校准误差经差分累积，原文结论明确指出。
- **综述中的位置**：本次结构重建综述可略去

**阅读依据**：正文定向阅读；正文定向阅读：摘要、公式设定、结论。

**原文定位**：PDF pp.1,4 §2，p.17 §5。

<a id="r007"></a>

#### R007 Waypoint-based Topology Inference

**书目信息**：Yilei Lin; Ting He; Shiqiang Wang; Kevin Chan。日期/年份：未记录。

**来源**：本地Zotero英文PDF；按本项题名定位。稳定公开书目入口待补。

- **方法**：利用经过指定waypoint的端到端探测，先辨识1-1-2路由子结构再合并为单源多目的网络。
- **输入与设备**：通信网络端到端路径/性能观测
- **隐藏节点与辨识对象**：处理通信路由内部结构
- **条件与适用边界**：电气潮流关系需要另建，不能直接作为LVDG算法。
- **综述中的位置**：树层析的跨领域背景，主线可略去

**阅读依据**：摘要及概述筛选；摘要及引言筛选。

**原文定位**：PDF p.1摘要与相关工作。

<a id="r009"></a>

#### R009 A topological model-driven optimization approach for detecting and locating electricity fraud in low-voltage networks

**书目信息**：Rubén Carmona-Pardo; Rafael Morán-Corbacho; Carlos García-Santacruz; Álvaro Rodríguez Del Nozal; Esther Romero-Ramos。日期/年份：02/2027。

**来源**：[DOI：10.1016/j.epsr.2026.113599](https://doi.org/10.1016/j.epsr.2026.113599)。

- **方法**：已知四线制低压网络模型下先按电压偏差筛选可疑线路，再解凸二次优化估计非法连接的附加功率与位置。
- **输入与设备**：已知拓扑/连接和线路参数、智能表电压、馈线首端功率
- **隐藏节点与辨识对象**：推断非法负荷，使用给定基础拓扑
- **条件与适用边界**：期刊卷年2027，DOI及正文版权/在线时间需分开核对；不能写成从零发现隐藏电气连接点。
- **综述中的位置**：准确拓扑的下游应用例子

**阅读依据**：正文定向阅读；正文定向阅读：摘要、任务、两阶段方法、结论。

**原文定位**：PDF pp.1–4 §3–4，p.7 §6。

<a id="r011"></a>

#### R011 False data injection attacks against state estimation in electric power grids

**书目信息**：Yao Liu; Peng Ning; Michael K. Reiter。日期/年份：05/2011。

**来源**：[DOI：10.1145/1952982.1952995](https://doi.org/10.1145/1952982.1952995)。

- **方法**：构造落在状态估计量测模型列空间内的虚假数据向量，使传统残差坏数据检测失效，并研究可攻击表计或攻击预算限制。
- **输入与设备**：已知量测模型及攻击者可修改的表计集合
- **隐藏节点与辨识对象**：网络安全研究
- **条件与适用边界**：攻击模型不是隐藏节点辨识算法。
- **综述中的位置**：本次综述主线可略去

**阅读依据**：摘要及概述筛选；摘要及结论筛选。

**原文定位**：PDF p.1摘要，p.29 §6。

<a id="r012"></a>

#### R012 Optimal experimental design: Formulations and computations

**书目信息**：Xun Huan; Jayanth Jagalur; Youssef Marzouk。日期/年份：07/2024。

**来源**：[DOI：10.1017/S0962492924000023](https://doi.org/10.1017/S0962492924000023)。

- **方法**：综述经典和Bayesian最优实验设计准则、数值近似、设计优化及序贯设计方法。
- **输入与设备**：模型、观测噪声/先验和设计变量
- **隐藏节点与辨识对象**：通用最优实验设计
- **条件与适用边界**：本次只读总览与章节安排，未精读129页全文；库中为arXiv作者稿，页首DOI为占位符。
- **综述中的位置**：若引入主动激励，可用于设计方法背景

**阅读依据**：摘要及概述筛选；摘要和引言筛选。

**原文定位**：PDF pp.1–2，总目录§2–5。

<a id="r013"></a>

#### R013 A note on the metric properties of trees

**书目信息**：Peter Buneman。日期/年份：08/1974。

**来源**：[DOI：10.1016/0095-8956(74)90047-1](https://doi.org/10.1016/0095-8956(74)90047-1)。

- **方法**：证明满足四点条件的距离可嵌入加权树，为由末端距离构造含内部点的树提供基础。
- **输入与设备**：满足相应条件的精确成对距离
- **隐藏节点与辨识对象**：数学上的树内部点，需由电网数据另行估计距离。
- **条件与适用边界**：树度量条件本身不提供电网噪声模型或电气可识别条件。
- **综述中的位置**：hidden节点树重建的基础引用

**阅读依据**：正文定向阅读；正文定向阅读：四点条件与两个定理。

**原文定位**：PDF pp.1–2，Theorems 1–2。

<a id="r014"></a>

#### R014 An Optimization-Based Topology Error Detection Method for Power System State Estimation

**书目信息**：Ankur Srivastava; Saikat Chakrabarti; Joao Soares; Sri Niwas Singh。日期/年份：08/2022。

**来源**：[DOI：10.1016/j.epsr.2022.107914](https://doi.org/10.1016/j.epsr.2022.107914)。

- **方法**：利用状态估计归一化残差及矩阵求逆引理，把拓扑误差检测写成混合整数非线性优化。
- **输入与设备**：SCADA与PMU混合量测、已有网络/状态估计模型
- **隐藏节点与辨识对象**：针对给定基础网络的拓扑错误
- **条件与适用边界**：实验为IEEE14/118，重点单一拓扑误差；传感配置和任务均不同于末端智能表未知树恢复。
- **综述中的位置**：已有线路基础设施下的状态/错误检测

**阅读依据**：正文定向阅读；正文定向阅读：摘要、残差/可观测性设定、结论。

**原文定位**：PDF p.1，§2，p.8 §7。

<a id="r018"></a>

#### R018 Tractable Bayesian learning of tree belief networks

**书目信息**：Marina Meilă; Tommi Jaakkola。日期/年份：1/2006。

**来源**：[DOI：10.1007/s11222-006-5535-3](https://doi.org/10.1007/s11222-006-5535-3)。

- **方法**：采用可分解结构先验和Dirichlet参数先验，通过矩阵树工具对树结构集合进行解析Bayesian求和。
- **输入与设备**：完全观测离散变量数据、指定先验
- **隐藏节点与辨识对象**：所述易处理结论基于complete observations
- **条件与适用边界**：不能将对全观测树的解析结果直接扩展到任意潜节点电网。
- **综述中的位置**：概率树学习的基础，主线可简述

**阅读依据**：正文定向阅读；正文定向阅读：摘要、模型参数化。

**原文定位**：PDF p.1，p.4 §2.2。

<a id="r021"></a>

#### R021 A unified framework for solving a general class of conditional and robust set-membership estimation problems

**书目信息**：Vito Cerone; Jean-Bernard Lasserre; Dario Piga; Diego Regruto。日期/年份：11/2014。

**来源**：[DOI：10.1109/TAC.2014.2351695](https://doi.org/10.1109/TAC.2014.2351695)。

- **方法**：对多项式观测关系和半代数有界不确定集，将条件/鲁棒集合成员估计的min-max问题分两阶段作半定松弛。
- **输入与设备**：多项式模型、误差可行集合
- **隐藏节点与辨识对象**：通用有界误差估计理论
- **条件与适用边界**：不直接输出电网树结构；需要另建物理模型和合法拓扑域。
- **综述中的位置**：若综述讨论确定性误差保证可引用

**阅读依据**：正文定向阅读；正文定向阅读：摘要、方法概述、估计示例。

**原文定位**：PDF p.1摘要，§III及p.21投影估计示例。

<a id="r023"></a>

#### R023 Network Tomography Based on Additive Metrics

**书目信息**：Jian Ni; Sekhar Tatikonda。日期/年份：12/2011。

**来源**：[DOI：10.1109/TIT.2011.2168901](https://doi.org/10.1109/TIT.2011.2168901)。

- **方法**：把通信路由的可加性能量转为树距离，借系统发育距离算法恢复逻辑路由树及链路性能，并给一致性/误差半径条件。
- **输入与设备**：端到端可加距离或能导出距离的观测
- **隐藏节点与辨识对象**：恢复逻辑分支点；串联无分支路段合并。
- **条件与适用边界**：用于电网须先论证可加电气距离的来源和误差。
- **综述中的位置**：末端树层析的算法基础

**阅读依据**：正文定向阅读；正文定向阅读：摘要、网络定义和结论。

**原文定位**：PDF pp.1,4 §2，pp.32–33 §8。

<a id="r026"></a>

#### R026 Identification and ranking of customers with energy theft using hypothesis tests

**书目信息**：Rodrigo F.G. Sau; Tailan R. Garcia; Fernanda C.L. Trindade; Madson C. De Almeida; Bala Venkatesh。日期/年份：12/2023。

**来源**：[DOI：10.1016/j.epsr.2023.109862](https://doi.org/10.1016/j.epsr.2023.109862)。

- **方法**：在配电网状态估计后，对量测误差估计进行假设检验，筛选并排序涉嫌窃电客户。
- **输入与设备**：已知网络上的电压与有/无功量测及其误差模型
- **隐藏节点与辨识对象**：识别不诚实客户/非技术损耗
- **条件与适用边界**：依赖给定网络状态估计；不能作为未知隐藏接线点恢复算法。
- **综述中的位置**：准确模型的下游用途，主线可略去

**阅读依据**：正文定向阅读；正文定向阅读：摘要、误差估计、检验流程。

**原文定位**：PDF pp.1,3–4，§2.2及方法段。

<a id="r027"></a>

#### R027 The EM algorithm for graphical association models with missing data

**书目信息**：Steffen L. Lauritzen。日期/年份：2/1995。

**来源**：[DOI：10.1016/0167-9473(93)E0056-A](https://doi.org/10.1016/0167-9473(93)E0056-A)。

- **方法**：用图模型的概率传播高效计算EM的E步，在缺失数据下估计分层对数线性或递归模型参数。
- **输入与设备**：图模型结构和不完整列联表/数据
- **隐藏节点与辨识对象**：通用统计潜变量
- **条件与适用边界**：EM参数拟合与电网物理隐藏节点结构辨识需分别建立。
- **综述中的位置**：LTM/EM的基础引用

**阅读依据**：摘要及概述筛选；摘要及引言方法概述。

**原文定位**：PDF pp.1–2，§1–2与§3–5概述。

<a id="r028"></a>

#### R028 The Variational Bayesian EM Algorithm for Incomplete Data: With Application to Scoring Graphical Model Structures

**书目信息**：Matthew J Beal; Zoubin Ghahramani。日期/年份：2003-7-3。

**来源**：[DOI：10.1093/oso/9780198526155.003.0025](https://doi.org/10.1093/oso/9780198526155.003.0025)。

- **方法**：交替近似潜变量和参数后验，优化边际似然的变分下界，并用其对Bayesian网络结构评分。
- **输入与设备**：潜变量概率模型和不完整数据
- **隐藏节点与辨识对象**：通用潜变量结构评分
- **条件与适用边界**：所比较的变分下界/BIC是统计评分，需另核实物理网络约束。
- **综述中的位置**：概率潜树模型选择的扩展背景

**阅读依据**：正文定向阅读；正文定向阅读：摘要、引言、结论。

**原文定位**：PDF pp.1–2，p.9 §5。

<a id="r029"></a>

#### R029 Minimizing Effective Resistance of a Graph

**书目信息**：Arpita Ghosh; Stephen Boyd; Amin Saberi。日期/年份：2008。

**来源**：[DOI：10.1137/050645452](https://doi.org/10.1137/050645452)。

- **方法**：在给定图上分配非负电导，最小化所有点对有效电阻之和，建立凸/半定规划及对偶误差界。
- **输入与设备**：已知图结构、电导总预算
- **隐藏节点与辨识对象**：设计已知图的权重
- **条件与适用边界**：不是从观测估计未知拓扑；电阻距离概念可作背景。
- **综述中的位置**：电阻距离理论，主线可略去

**阅读依据**：正文定向阅读；正文定向阅读：摘要、优化定义与方法概述。

**原文定位**：PDF p.2摘要，p.4凸性概述，p.5 §2。

<a id="r030"></a>

#### R030 Optimal discrimination designs

**书目信息**：Holger Dette; Stefanie Titoff。日期/年份：2009-8-1。

**来源**：[DOI：10.1214/08-AOS635](https://doi.org/10.1214/08-AOS635)。

- **方法**：研究T最优回归模型辨别设计，连接最佳一致逼近并刻画支撑点及非唯一最优设计。
- **输入与设备**：竞争回归模型、设计域及参考参数
- **隐藏节点与辨识对象**：通用模型辨别设计
- **条件与适用边界**：辨别最优设计的支撑点可能不足以同时辨识大模型全部参数。
- **综述中的位置**：主动探测的模型竞争依据

**阅读依据**：摘要及概述筛选；摘要与引言筛选。

**原文定位**：PDF pp.1–2，摘要和引言。

<a id="r032"></a>

#### R032 Kron Reduction of Graphs with Applications to Electrical Networks

**书目信息**：Florian Dorfler; Francesco Bullo。日期/年份：2011-02-15。

**来源**：[DOI：10.48550/arXiv.1102.2950](https://doi.org/10.48550/arXiv.1102.2950)。

- **方法**：将内部节点消元为Laplacian的Schur补，分析Kron缩减后的图、谱与有效电阻性质。
- **输入与设备**：给定原网络的导纳/Laplacian分块
- **隐藏节点与辨识对象**：消去内部节点，保留边界端口等效关系。
- **条件与适用边界**：仅估计端口等效模型时，原内部结构仍需额外条件才能恢复。
- **综述中的位置**：说明等效网络与物理隐藏树恢复的区别

**阅读依据**：正文定向阅读；正文定向阅读：摘要、消元模型及结论。

**原文定位**：PDF pp.1–3，式(1.1)–(1.2)，p.25 §4。

<a id="r033"></a>

#### R033 The Balanced Minimum Evolution Problem.

**书目信息**：Daniele Catanzaro; Martine Labbé; Raffaele Pesenti; Juan-José Salazar-González。日期/年份：2012-04-01。

**来源**：[DOI：10.1287/ijoc.1110.0455](https://doi.org/10.1287/ijoc.1110.0455)。

- **方法**：为叶端距离驱动的平衡最小进化树重建构造混合整数线性规划，并用分支规则和有效不等式增强精确求解。
- **输入与设备**：物种间距离矩阵；无根二叉谱系树域
- **隐藏节点与辨识对象**：内部谱系分支节点；非电气模型
- **条件与适用边界**：解决给定BME准则的组合优化；不能直接据此宣称在任意配电数据下恢复真实电网。
- **综述中的位置**：可作为潜树的组合优化背景；LVDG相关工作可省略。

**阅读依据**：正文定向阅读；摘要、建模与有效不等式片段、结论定向阅读。

**原文定位**：PDF pp.1–2,5,18；§3、§7。

<a id="r034"></a>

#### R034 Topology identification for dynamic equivalent models of large power system networks

**书目信息**：Seyedbehzad Nabavi; Aranya Chakrabortty。日期/年份：2013-06。

**来源**：[DOI：10.1109/ACC.2013.6579989](https://doi.org/10.1109/ACC.2013.6579989)。

- **方法**：从PMU角度和频率提取慢振荡子空间，以l2及l1/l2参数估计辨识大电网区域间动态等值拓扑。
- **输入与设备**：同步相角/频率时间序列和动态降阶模型
- **隐藏节点与辨识对象**：恢复的是聚合动态等值节点间连接
- **条件与适用边界**：信息、物理时间尺度与低压末端静态量测不同。
- **综述中的位置**：高压/动态等值路线的简短对照

**阅读依据**：正文定向阅读；正文定向阅读：摘要、引言方法与模型。

**原文定位**：PDF pp.1–2，§II–III方法概述。

<a id="r035"></a>

#### R035 Non-Technical Loss Detection Using State Estimation and Analysis of Variance

**书目信息**：Shih-Che Huang; Yuan-Liang Lo; Chan-Nan Lu。日期/年份：2013-08。

**来源**：[DOI：10.1109/TPWRS.2012.2224891](https://doi.org/10.1109/TPWRS.2012.2224891)。

- **方法**：用三相配电状态估计定位计量异常的变压器/馈线区域，再以方差分析形成客户嫌疑清单并估计非技术损耗。
- **输入与设备**：智能表电压、客户电量及站/馈线功率；已验证客户连接、拓扑及参数；同步量测
- **隐藏节点与辨识对象**：不重建隐藏节点；网络连接和配置为已知输入
- **条件与适用边界**：PDF p2明确假定客户连接、配电参数及配置可用且已验证，并假设无偏正态误差。
- **综述中的位置**：可支持准确网络模型对损耗/异常分析的应用价值；不列为未知拓扑恢复方法。

**阅读依据**：正文定向阅读；摘要及状态估计方法/信息假设定向阅读。

**原文定位**：PDF pp.1–2；方法框架与§II。

<a id="r036"></a>

#### R036 Blind topology identification for power systems

**书目信息**：Xiao Li; H. Vincent Poor; Anna Scaglione。日期/年份：2013-10。

**来源**：[DOI：10.1109/SmartGridComm.2013.6687939](https://doi.org/10.1109/SmartGridComm.2013.6687939)。

- **方法**：在DC潮流P=Bθ下，将未知Laplacian和相角作稀疏正则联合分解，交替更新并利用图约束筛选连接。
- **输入与设备**：各母线有功注入序列
- **隐藏节点与辨识对象**：固定节点集；原文先讨论仅注入数据的非唯一性。
- **条件与适用边界**：依赖DC模型和足够激励，正则化选解不等于任意低压隐藏树可唯一恢复。
- **综述中的位置**：DC近似/信息条件不同的代表

**阅读依据**：正文定向阅读；正文定向阅读：摘要、模型、可辨识性与算法。

**原文定位**：PDF pp.1–3，§II与§III，式(9)–(13)、Algorithm1。

<a id="r037"></a>

#### R037 Robust $T$-optimal discriminating designs

**书目信息**：Holger Dette; Viatcheslav B. Melas; Petr Shpilev。日期/年份：2013-8-1。

**来源**：[DOI：10.1214/13-AOS1117](https://doi.org/10.1214/13-AOS1117)。

- **方法**：针对参数误设会削弱T最优试验辨别能力的问题，构造贝叶斯平均与标准化最大最小辨别设计，并分析多项式回归实例。
- **输入与设备**：竞争回归模型、设计域、参数先验或不确定集合
- **隐藏节点与辨识对象**：非拓扑研究
- **条件与适用边界**：设计准则依赖竞争模型和允许参数范围；可辨别性与拓扑正确性需另行建立。
- **综述中的位置**：仅在综述主动试验/探测设计时作方法背景。

**阅读依据**：正文定向阅读；摘要、局部T准则定义和鲁棒设计路线定向阅读。

**原文定位**：PDF pp.1–3；§2及全篇摘要。

**版本与书目说明**：本地为arXiv:1309.4652v1排版稿，首页含Annals出版信息。

<a id="r040"></a>

#### R040 A general framework for updating belief distributions

**书目信息**：P. G. Bissiri; C. C. Holmes; S. G. Walker。日期/年份：2016-11。

**来源**：[DOI：10.1111/rssb.12158](https://doi.org/10.1111/rssb.12158)。

- **方法**：以数据损失的期望加相对先验的KL散度构造决策目标，得到可在不指定完整似然时使用的广义贝叶斯信念更新。
- **输入与设备**：参数先验与数据损失函数
- **隐藏节点与辨识对象**：非电网/拓扑研究
- **条件与适用边界**：广义后验表达信念更新；若用于线路置信度，损失选择、尺度及覆盖性质需要另行说明。
- **综述中的位置**：可解释基于损失的概率权重，当前LVDG隐藏节点综述主体可省略。

**阅读依据**：正文定向阅读；摘要、决策目标与KL更新推导定向阅读。

**原文定位**：PDF pp.1,3–4；式(7)及相关推导。

<a id="r043"></a>

#### R043 Electricity Theft Detection and Localization in Grid-Tied Microgrids

**书目信息**：Muhammad Tariq; H. Vincent Poor。日期/年份：2018-05。

**来源**：[DOI：10.1109/TSG.2016.2602660](https://doi.org/10.1109/TSG.2016.2602660)。

- **方法**：以随机Petri网描述并网微网运行/窃电事件，结合SVD线阻估计与损耗阈值触发检测和用户定位。
- **输入与设备**：智能表累计数据及线阻/线损估计；微网运行状态
- **隐藏节点与辨识对象**：论文声称检测无需精确网络拓扑，但输出是窃电告警/定位，不是隐藏结构重建
- **条件与适用边界**：随机脆弱性模型含事件率等假设；文中案例的检测结果与普适网络恢复保证不同。
- **综述中的位置**：窃电应用侧文献；避免因摘要提到未知拓扑而误分类为拓扑算法。

**阅读依据**：正文定向阅读；摘要及方法分解/随机Petri网建模定向阅读。

**原文定位**：PDF pp.1,3；§II、Algorithms 1–2路线。

<a id="r047"></a>

#### R047 Network Learning via Multiagent Inverse Transportation Problems

**书目信息**：Susan Jia Xu; Mehdi Nourinejad; Xuebo Lai; Joseph Y. J. Chow。日期/年份：2018-12。

**来源**：[DOI：10.1287/trsc.2017.0805](https://doi.org/10.1287/trsc.2017.0805)。

- **方法**：通过多个旅行者的逆最短路问题，学习能解释异质路线选择的成本及共享网络容量对偶价格。
- **输入与设备**：已知交通网络、个体已选路线和起终点；最短路行为模型
- **隐藏节点与辨识对象**：隐藏量是交通成本/对偶价格，非隐藏电气节点
- **条件与适用边界**：模型以已知网络和行为最优化为基础；作者对实测交通例明确指出准确性不能直接定量确立。
- **综述中的位置**：多主体逆优化背景，当前拓扑综述可省略。

**阅读依据**：正文定向阅读；摘要、逆最短路框架与实测例讨论定向阅读。

**原文定位**：PDF pp.1,7,24；§3.1及实测例讨论。

**版本与书目说明**：本地为29页作者稿式排版，非最终期刊版分页。

<a id="r050"></a>

#### R050 Bayesian topology identification of linear dynamic networks

**书目信息**：Shengling Shi; Giulio Bottegal; Paul M. J. Van den Hof。日期/年份：2019-03-14。

**来源**：[DOI：10.48550/arXiv.1903.06205](https://doi.org/10.48550/arXiv.1903.06205)。

- **方法**：把动态网络模块的脉冲响应建模为高斯过程，通过边际似然/EM估计超参数，并用前向—后向贪心搜索选择传递函数网络连接。
- **输入与设备**：动态系统节点时间序列；独立白噪声和模块先验等模型条件
- **隐藏节点与辨识对象**：主体为传递函数互联结构；不是低压电气隐藏接线点恢复
- **条件与适用边界**：原文明确搜索获得局部最优；统计动态边到物理电网边的对应需要额外物理假设。
- **综述中的位置**：贝叶斯网络结构识别背景，可放外围方法。

**阅读依据**：正文定向阅读；摘要、边际似然分解及搜索算法定向阅读。

**原文定位**：PDF pp.1,3–4；目标分解及贪心搜索。

**版本与书目说明**：本地为arXiv:1903.06205v1。

<a id="r054"></a>

#### R054 Universal inference

**书目信息**：Larry Wasserman; Aaditya Ramdas; Sivaraman Balakrishnan。日期/年份：2020。

**来源**：[DOI：10.1073/pnas.1922664117](https://doi.org/10.1073/pnas.1922664117)。

- **方法**：将样本分割为独立训练与检验部分，使用跨样本似然比构造非正则模型也适用的有限样本检验和置信集。
- **输入与设备**：可计算似然的统计模型和满足分割独立性的数据
- **隐藏节点与辨识对象**：统计推断工具，非拓扑算法
- **条件与适用边界**：需正确指定相应分布/似然并控制分割数据依赖；不能将任意拟合残差直接代入视为有效证书。
- **综述中的位置**：仅在讨论拓扑置信集/统计检验时引用。

**阅读依据**：正文定向阅读；摘要、split likelihood-ratio构造与覆盖推导定向阅读。

**原文定位**：PDF pp.2–4（PDF p1为JSTOR封面）；§1及式(7)。

<a id="r057"></a>

#### R057 Detection of False Data Injection Attacks in Smart Grids Based on Graph Signal Processing

**书目信息**：Elisabeth Drayer; Tirza Routtenberg。日期/年份：2020-06。

**来源**：[DOI：10.1109/JSYST.2019.2927469](https://doi.org/10.1109/JSYST.2019.2927469)。

- **方法**：利用电网导纳图上的图傅里叶变换和高通滤波，检测状态估计电压信号中异常的高频成分，以补充传统残差检测。
- **输入与设备**：给定电网图/导纳矩阵、状态估计输出；相应误差模型
- **隐藏节点与辨识对象**：不恢复隐藏结构，导纳图用于定义图频率
- **条件与适用边界**：检测性能以已知网络和运行信号的图平滑特性为前提。
- **综述中的位置**：准确网络模型在攻击检测中的应用背景。

**阅读依据**：正文定向阅读；摘要、检测信号模型与图谱滤波片段定向阅读。

**原文定位**：PDF pp.1,4–5；式(8)、§III-B及图高通滤波。

<a id="r059"></a>

#### R059 Line Outage Identification Based on AC Power Flow and Synchronized Measurements

**书目信息**：Zhen Dai; Joseph Euzebe Tate。日期/年份：2020-11-18。

**来源**：[DOI：10.48550/arXiv.2011.09566](https://doi.org/10.48550/arXiv.2011.09566)。

- **方法**：逐个仿真候选单线断开后的AC电压相量变化，与PMU实测变化匹配，再用拒绝滤波保留不确定结果。
- **输入与设备**：已有网络模型、待检测线路集合、事件前后同步电压相量
- **隐藏节点与辨识对象**：识别候选线路停运
- **条件与适用边界**：单线故障和已知候选图的检测任务，不能直接扩展为未知低压隐藏树恢复。
- **综述中的位置**：已知候选图上的线路状态检测

**阅读依据**：正文定向阅读；正文定向阅读：摘要、主算法及拒绝滤波设定。

**原文定位**：PDF pp.1–3，§II–III，式(1)–(5)。

<a id="r061"></a>

#### R061 Learning Latent Variable Dynamic Graphical Models by Confidence Sets Selection

**书目信息**：Valentina Ciccone; Augusto Ferrante; Mattia Zorzi。日期/年份：2020-12。

**来源**：[DOI：10.1109/TAC.2020.2970409](https://doi.org/10.1109/TAC.2020.2970409)。

- **方法**：在动态过程的谱估计置信邻域内，联合稀疏项与低秩潜变量项选择，以一个正则化参数平衡观测图稀疏性和潜变量数。
- **输入与设备**：有限样本多元AR过程与谱密度估计；Itakura–Saito散度约束
- **隐藏节点与辨识对象**：低秩潜在随机过程，不自动对应电气接线点
- **条件与适用边界**：统计潜因子模型的置信邻域/选择结果须按其AR和估计假设解释；不提供LVDG线路恢复定理。
- **综述中的位置**：潜变量图模型与不确定性方法背景。

**阅读依据**：正文定向阅读；摘要、问题设定、稀疏加低秩与散度约束定向阅读。

**原文定位**：PDF pp.1–4；§II、式(3)、式(6)–(12)。

<a id="r064"></a>

#### R064 Topology Learning Aided False Data Injection Attack without Prior Topology Information

**书目信息**：Martin Higgins; Jiawei Zhang; Ning Zhang; Fei Teng。日期/年份：2021-07。

**来源**：[DOI：10.1109/PESGM46819.2021.9638211](https://doi.org/10.1109/PESGM46819.2021.9638211)。

- **方法**：从支路/母线功率记录学习攻击者的网络模型，并结合攻击侧伪残差检查构造针对AC状态估计的隐蔽虚假数据注入。
- **输入与设备**：母线/支路功率流及攻击者可取得的量测记录
- **隐藏节点与辨识对象**：目标是辅助攻击的网络知识学习；没有本文范围内隐藏节点恢复结论
- **条件与适用边界**：攻击性能和可观测量测覆盖绑定；应作为网络信息安全应用，不等同于低压端节点可观测条件下的全树重建。
- **综述中的位置**：安全应用外围材料，综述主线可省略。

**阅读依据**：正文定向阅读；摘要、AC模型与攻击侧评估方法片段定向阅读。

**原文定位**：PDF pp.1–3；摘要、AC状态估计/攻击模型及伪残差步骤。

**版本与书目说明**：本地稿首页写PESGM 2021 submitted 2020-11-06，并带2021 IEEE DOI。

**对应原记录编号**：R064, R065。

<a id="r067"></a>

#### R067 An Adaptive-Importance-Sampling-Enhanced Bayesian Approach for Topology Estimation in an Unbalanced Power Distribution System

**书目信息**：Yijun Xu; Jaber Valinejad; Mert Korkali; Lamine Mili; Yajun Wang; Xiao Chen; Zongsheng Zheng。日期/年份：2021-10-18。

**来源**：[DOI：10.48550/arXiv.2110.09030](https://doi.org/10.48550/arXiv.2110.09030)。

- **方法**：用非线性三相潮流建立开关状态和负荷的Bayesian后验，自适应重要性采样近似后验并作MAP估计。
- **输入与设备**：已有候选开关/线路物理参数、有限量测及先验/误差分布
- **隐藏节点与辨识对象**：估计既有节点和候选图内的未测状态
- **条件与适用边界**：有限观测与隐藏状态并不等同于发现未知物理接线节点；库中为2021 arXiv稿。
- **综述中的位置**：不平衡配电网的候选开关状态估计

**阅读依据**：正文定向阅读；正文定向阅读：摘要、候选开关模型、后验与采样。

**原文定位**：PDF pp.1–4，§II–III，式(4)–(9)、Algorithm1。

<a id="r072"></a>

#### R072 Time-uniform, nonparametric, nonasymptotic confidence sequences

**书目信息**：Steven R. Howard; Aaditya Ramdas; Jon McAuliffe; Jasjeet Sekhon。日期/年份：2021-4-1。

**来源**：[DOI：10.1214/20-AOS1991](https://doi.org/10.1214/20-AOS1991)。

- **方法**：通过指数超鞅、拼接和混合边界构造随时间一致有效的置信序列，覆盖次高斯等多种过程与自归一化统计量。
- **输入与设备**：满足相应条件矩母函数/超鞅约束的适应过程
- **隐藏节点与辨识对象**：统计序贯推断工具，非拓扑恢复
- **条件与适用边界**：time-uniform保证有明确过程条件；重复查看数据或自适应采样本身不自动满足这些条件。
- **综述中的位置**：仅在讨论在线采样停止/序贯拓扑置信时引用。

**阅读依据**：正文定向阅读；摘要、方法概览与sub-ψ条件定义定向阅读。

**原文定位**：PDF pp.1,4–5；§1.2、§2、Definition 1。

<a id="r088"></a>

#### R088 Learning on Multimodal Graphs: A Survey

**书目信息**：Ciyuan Peng; Jiayuan He; Feng Xia。日期/年份：2024-02-07。

**来源**：[DOI：10.48550/arXiv.2402.05322](https://doi.org/10.48550/arXiv.2402.05322)。

- **方法**：按多模态图的组织方式以及图卷积、图注意力、图对比学习三类融合技术整理多模态图学习。
- **输入与设备**：带不同模态特征/子图的图数据
- **隐藏节点与辨识对象**：多模态表示学习，未提供电气隐藏节点辨识方法
- **条件与适用边界**：综述中的图与数据模态语义须和配电物理结构区分。
- **综述中的位置**：通用机器学习综述，本次主线建议不纳入。

**阅读依据**：正文定向阅读；摘要和分类定义定向筛选。

**原文定位**：PDF pp.1–2；§2、§3。

**版本与书目说明**：本地为arXiv:2402.05322v1。

<a id="r093"></a>

#### R093 Graph neural networks for multi-view learning: a taxonomic review

**书目信息**：Shunxin Xiao; Jiacheng Li; Jielong Lu; Sujia Huang; Bao Zeng; Shiping Wang。日期/年份：2024-10-21。

**来源**：[DOI：10.1007/s10462-024-10990-1](https://doi.org/10.1007/s10462-024-10990-1)。

- **方法**：按输入的多关系、多属性和混合形式分类GNN多视图学习，介绍表示融合方法、应用与公开数据。
- **输入与设备**：多个邻接关系、多个属性视图或二者并存
- **隐藏节点与辨识对象**：多视图节点表征，非隐藏物理节点恢复
- **条件与适用边界**：以图作为输入或学习表示，能力不能直接转写为未知配电拓扑可辨识性。
- **综述中的位置**：通用GNN综述，当前LVDG主线建议不纳入。

**阅读依据**：正文定向阅读；摘要、输入分类与结论定向筛选。

**原文定位**：PDF pp.1,5,32–33；§2.1、§7。

**对应原记录编号**：R093, R094。

<a id="r095"></a>

#### R095 Significant inference and confidence sets for graphical models

**书目信息**：P. A. Koldanov; A. P. Koldanov。日期/年份：2025。

**来源**：[DOI：10.48550/ARXIV.2509.12292](https://doi.org/10.48550/ARXIV.2509.12292)。

- **方法**：把图结构置信集等价转化为对边相关假设及其备择的同时检验，区分确定存在、确定缺失和暂未确定的结构推断。
- **输入与设备**：统计图模型、参数假设及同时检验程序；示例为高斯图的偏相关
- **隐藏节点与辨识对象**：统计依赖图，不是电气隐藏节点恢复
- **条件与适用边界**：图域和可组合的假设/备择条件决定结果；任意树结构约束或物理线路对应需另行处理。
- **综述中的位置**：可启发拓扑置信边界，但非LVDG方法实证。

**阅读依据**：正文定向阅读；摘要、问题定义与同时检验构造定向阅读。

**原文定位**：PDF pp.1,4–5；§2、§3及式(2)。

**版本与书目说明**：本地为arXiv:2509.12292v1。

<a id="r116"></a>

#### R116 Distributed Energy Resources Topology Identification via Graphical Modeling

**书目信息**：Yang Weng; Yizheng Liao; Ram Rajagopal。日期/年份：7/2017。

**来源**：[DOI：10.1109/TPWRS.2016.2628876](https://doi.org/10.1109/TPWRS.2016.2628876)。

- **方法**：以历史节点电压的互信息为权重建立最大生成树，并推广到部分网状连接及仅电压幅值的近似情形。
- **输入与设备**：待建模节点历史电压，智能表幅值使用近似改造
- **隐藏节点与辨识对象**：算法在给定量测节点集连边，没有插入未知内部节点步骤。
- **条件与适用边界**：用不同节点电流注入独立近似建立树形条件依赖；原文给出对称等负荷情形的失败例。
- **综述中的位置**：低成本被动统计路线的代表

**阅读依据**：正文定向阅读；正文定向阅读：摘要、模型、独立性条件、树算法与反例。

**原文定位**：PDF pp.1–5，§II–III，Theorem1、Algorithm1及§III-C。

<a id="review-supplemental"></a>

## 10. 库外补充文献

下列两篇为围绕隐藏节点原始文献线索的补查，未纳入142组计数。

<a id="sup01"></a>

### SUP01 Grid Topology Identification With Hidden Nodes via Structured Norm Minimization

**书目信息**：Rajasekhar Anguluri; Gautam Dasarathy; Oliver Kosut; Lalitha Sankar。IEEE Control Systems Letters, vol.6, pp.1244–1249, 2022；2021-06-17在线。

**来源**：[DOI：10.1109/LCSYS.2021.3089993](https://doi.org/10.1109/LCSYS.2021.3089993)；[作者开放全文](https://rajanguluri.github.io/assets/pdf/papers/RA-GD-OK-LS-LCSS2021.pdf)；[作者稿备选](https://pmc.ncbi.nlm.nih.gov/articles/PMC9223390/)。

- **方法**：将观测电压幅值/相角逆协方差分为稀疏项、稀疏因子的低秩项和一般低秩项，以结构原子范数联合估计，再依据符号关系恢复含隐节点的连边。
- **输入**：观测节点的电压幅值和相角样本，需估计联合协方差。
- **隐藏节点与恢复对象**：目标包含观测–观测及观测–隐藏连接；hidden不可是叶节点且彼此不相邻。允许隐藏节点有非退化未知注入。
- **条件**：LC-PF；跨节点不相关Gaussian P/Q且各节点协方差非退化；稀疏网络、隐节点少于观测节点；最短环长>3。
- **证据边界**：符号定理以正确分解分量为输入；p.1249明确将分解可恢复性的理论条件留作未来研究。
- **纳入建议**：应补入含hidden节点的方法综述，作为需要相角量测及隐藏节点分隔条件的结构分解路线。其信息条件区别于只用端点电表恢复零注入分支树的方法。

**实际阅读范围**：PMC正文§I–II模型与Assumption1；作者PDF pp.1247–1249的Assumption2、Theorem2、Algorithms1–2、结构范数优化、模拟设置和结论；非逐段精读全部证明。

**原文定位**：p.1245 §II-B Assumption1；p.1247 §III-A Assumption2与Theorem2；pp.1247–1248 §IV和式(12)–(14)；p.1249 §VI。

**版本与获取状态**：作者网站提供的6页IEEE出版版PDF；PMC另有作者稿；在线PDF正文可读，未下载/归档本地PDF。

<a id="sup02"></a>

### SUP02 基于隐树模型和聚类搜索的低压配电网拓扑辨识

**书目信息**：张恒超；曹骏；沈秋英。电力科学与技术学报，2025，40(2)：170–178,195；官方页面在线发布日期2025-06-06。

**来源**：[DOI：10.19781/j.issn.1673-9140.2025.02.018](https://doi.org/10.19781/j.issn.1673-9140.2025.02.018)；[稳定来源](https://dlkjxb.csust.edu.cn/dlgcjs/article/abstract/20250218?st=article_issue)。

- **方法**：摘要提出含隐式节点的Bayesian潜树，以聚类搜索生成候选拓扑，并用BIC评价候选；报告仿真和实验验证。
- **输入**：摘要未具体说明所用量测类型、完整覆盖条件和训练/搜索输入。
- **隐藏节点与恢复对象**：明确研究无法上传运行状态的低压隐式节点；具体是零注入节点还是带负荷未测节点，以及度数条件均待正文。
- **条件**：未在摘要报告，待全文核实。
- **证据边界**：仅支持该研究存在及潜树+聚类搜索+BIC路线；不支持与Zhang2022的精确差异、复杂度或辨识保证判断。
- **纳入建议**：应进入LVDG+hidden节点的直接相关文献核对清单；中文或覆盖至2025年的综述建议纳入。当前可简述摘要路线，详细优劣对比须留待全文。

**实际阅读范围**：官方摘要；未读算法、假设、公式、实验正文。

**原文定位**：官方摘要；官方页面Online发布日期及引用信息。

**版本与获取状态**：正式期刊书目与摘要；正文版本未取得；仅摘要；官方页面全文打开超时，未取得PDF。

<a id="review-bibliography"></a>

## 11. 版本合并与书目修正

### 11.1 合并引用与方法发展线

- R039与R045/R119：同一PCA研究的预印本/正式版；正式版优先引用，阅读定位仍对应所读版本。
- R049/R051：同题逆变器探测工作的预印本与正式版，已归并为一组；正式DOI为10.1109/TCNS.2019.2901714。
- R044与R055：同一方法发展线的会议论文和扩展稿，保留独立记录，按所用结论引用对应版本。
- R121/R142：同DOI重复条目，合并后的论文已有全文。
- 其他同题名或同DOI的原记录编号列在对应条目，不据此增加独立研究数量。

### 11.2 本次已经核对的书目修正

| 记录 | 修正或补充 | 依据 |
| --- | --- | --- |
| R044 | 作者采用Sejun Park、Deepjyoti Deka、Michael Chertkov；库中Seiun/Chcrtkov为误拼 | 原PDF首页 |
| R086 | 作者为Zhou、Liang、Zou、Ding；Hu、Cao为书籍编辑 | Springer官方章节页 |
| R039 | 预印本首页第一作者写Satya Jayadev P；正式版规范姓名见R045 | 预印本与正式版首页 |
| R004 | 补DOI 10.3390/en17092060、年份2024 | 原PDF首页 |
| R006 | 补正式题名Estimating Topologies…、CIRED2022 804–808页及DOI10.1049/icp.2022.0823；原Wallbox稿题名拼写另保留 | 作者Zenodo稿、TH Köln机构出版清单 |
| R055 | 补正式DOI10.1109/TCNS.2020.2979882、7(3):1428–1440(2020)；阅读定位仍为作者稿 | 正式书目及LANL机构报告 |
| R029 | 补DOI 10.1137/050645452 | 原PDF |
| R054 | 补DOI 10.1073/pnas.1922664117 | 原PDF |
| R148 | 已读本地arXiv v1；另核正式DOI 10.1109/LCSYS.2026.3711602 | 本地稿及作者机构研究记录 |

### 11.3 引用前需要保留的版本信息

- R001：PDF含arXiv:1412.2548及不同排版日期，正式发表书目信息仍待核。
- R008/R009：2027卷期与2026 DOI/当前阅读时间分别记录，在线发表日期尚未专项核定。
- R127/R132等：本地为较长DRAFT或作者稿；定位使用所读PDF页。
- R136：所读2021稿题名包含Information and Pricing，与2023条目题名有差异。
- R138：所读2020 arXiv v1，条目对应2023 Automatica正式版。
- R143：所读工作论文题名与条目略异，仍按工作论文版本说明。
- R154：所读Journal Pre-proofs，封面明确不是最终Version of Record。
- R086、SUP02：当前仅摘要；精确方法和假设的比较待全文。
- 少数外围记录无DOI或稳定入口，当前按本地PDF题名定位，正式参考文献表应补齐出版信息。

### 11.4 使用本文件编写论文的顺序

1. 从第2节确定本文的辨识对象和量测契约。
2. 用第3–4节组织一般方法，每类保留最有代表性的原始论文。
3. 以第5节选择输入和输出最接近的隐藏节点基线，并回查逐篇记录的原文定位。
4. 按第6节选择少量EV实证和响应模型，Wallbox放入直接拓扑相关工作。
5. 按第7节将候选缺口与本文实际结果对齐，再改写第8节段落。
6. 按本节修正作者、合并版本，并补齐会影响最终结论的全文与书目缺口。

<a id="review-search-update"></a>

## 12. 外部补检结果与新增文献

### 12.1 检索范围与纳入方式

本轮在2026-11-11按中英文主题、已知作者及原文参考文献补检，覆盖低压隐藏节点、末端幅值/功率、非零未测注入、逆潮流、异步AMI、EV事件、GIS和数据精度。优先使用出版社、作者、大学/实验室仓储及官方期刊平台；聚合页只用于发现线索。与原文档按题名和DOI去重后新增25项，保留不同版本和相近发展线的说明。这是定向查漏，未声称穷尽。

| 阅读状态 | 数量 | 可用于的论述 |
| --- | --- | --- |
| 正文定向阅读 | 11 | 所列章节的方法、输入、明确条件及相应证据边界 |
| 摘要或部分预览 | 8 | 摘要/预览明确陈述的路线；未见公式或假设留待正文 |
| 仅书目 | 6 | 确认研究存在及优先取文，不据题名补写算法 |

公开全文不足的高优先级项保留访问缺口。本地Zotero全库题名关键词检索未命中数篇近期英文候选；本轮浏览器访问未能启动，HKU路径未完成。没有修改Zotero或导入新条目。

### 12.2 优先级总览

| 编号 | 文献 | 建议用途 | 阅读层级 |
| --- | --- | --- | --- |
| [SUP03](#sup03) | Distribution grid impedance & topology estimation with limited or no micro-PMUs | 补入带非零注入、相关性与误差量化路线。 | 正文定向阅读 |
| [SUP04](#sup04) | Low-voltage network topology and impedance identification using smart meter measurements | 比较地理/参数先验与缺表能力的取舍。 | 正文定向阅读 |
| [SUP05](#sup05) | Reconstruction of Low-Voltage Networks with Limited Observability | 与Benzerga2021相邻介绍，保留不同会议版本。 | 正文定向阅读 |
| [SUP06](#sup06) | Joint Estimation of Topology and Injection Statistics in Distribution Grids With Missing Nodes | 补充已知缺测节点集合与未知隐点数的任务差别。 | 正文定向阅读 |
| [SUP07](#sup07) | Learning partially observed meshed distribution grids | 补非径向hidden路线，避免把voltage-only译成仅幅值。 | 正文定向阅读 |
| [SUP08](#sup08) | Inverse Power Flow Problem | 补齐从端口等效反推物理隐藏树的条件性结果。 | 正文定向阅读 |
| [SUP09](#sup09) | 基于智能电表数据的低压配电网拓扑与线路参数联合辨识 | 直接加入中文hidden相关工作，详细比较待正文。 | 摘要或部分预览 |
| [SUP10](#sup10) | A data-driven method for joint estimation of topology and impedance in distribution networks using end-user voltage magnitude and power data | 与Park、Pengwah、Guo等直接对齐信息条件。 | 摘要或部分预览 |
| [SUP11](#sup11) | Robust coupling impedance-based topology identification method for asynchronous low-voltage distribution networks | 异步普通表计hidden主线；作者报告114表硬件原型。 | 摘要或部分预览 |
| [SUP12](#sup12) | A Novel Topology Recovery Method for Low Voltage Distribution Networks | 应补入端点方法，具体说明其degree2与模型条件。 | 正文定向阅读 |
| [SUP13](#sup13) | Information-theoretic grid topology reconstruction using low-precision smart meter data | 用于设备/数据要求段，不作为普遍精度下限。 | 正文定向阅读 |
| [SUP14](#sup14) | Data-Driven Topology Generation with Physics-Guidance in LV Distribution Networks | 补GIS与工程先验融合路线。 | 正文定向阅读 |
| [SUP15](#sup15) | A data-driven topology identification method for low-voltage distribution networks based on the wavelet transform | 一般低压低硬件输入方法对照。 | 摘要或部分预览 |
| [SUP16](#sup16) | A review of topology identification methods and applications for low voltage distribution networks | 补近期低压专门综述和引用追踪入口。 | 摘要或部分预览 |
| [SUP17](#sup17) | 基于虚拟阻抗的低压配电网拓扑识别方法 | 一般低压连接纠错的中文代表。 | 摘要或部分预览 |
| [SUP18](#sup18) | 基于分时感知与分块回溯的低压配电网线路阻抗辨识方法 | 可选参数/数据质量背景。 | 摘要或部分预览 |
| [SUP19](#sup19) | Identification of Distribution Network Topology and Line Parameter Based on Smart Meter Measurements | 一般非相量联合估计对照。 | 正文定向阅读 |
| [SUP20](#sup20) | 基于改进相关熵匹配追踪的配电网拓扑与线路参数辨识方法 | 可选鲁棒估计背景，非hidden主线。 | 正文定向阅读 |
| [SUP21](#sup21) | 基于稀疏回归与聚类修正的配电网拓扑与线路参数联合辨识方法研究 | 近期一般方法，可按篇幅选择。 | 摘要或部分预览 |
| [SUP22](#sup22) | A Heuristic-Search-Based Topology Identification and Parameter Estimation Method in Low Voltage Distribution Grids With Low Observability | R015的同作者前作，应优先取得全文。 | 仅书目 |
| [SUP23](#sup23) | Low-Voltage Distribution Grid Topology Identification and Parameter Estimation with Heterogeneous Cable Sizes and Dense Noise: A Prior Knowledge-Guided Adaptive Search Approach | 对相关缺口有影响，先核日期和独立贡献。 | 仅书目 |
| [SUP24](#sup24) | 基于非线性回归的含隐节点低压配电网参数和拓扑联合辨识 | 中文非线性hidden方向的优先全文项。 | 仅书目 |
| [SUP25](#sup25) | Topology Identification of Low-Voltage Distribution Networks with Latent Nodes Considering Impedance Parameter Constraints | 近期直接主题，取原文后再比较。 | 仅书目 |
| [SUP26](#sup26) | Topology and Parameter Joint Identification in Imbalanced Low-Voltage Distribution Networks Based on Load Characteristic Propagation | 高优先级三相hidden原文核对项。 | 仅书目 |
| [SUP27](#sup27) | A Data-Driven Topology and Parameter Joint Estimation Method in Non-PMU Distribution Networks | 追溯TLM相关方法来源，优先核原文。 | 仅书目 |

### 12.3 直接相关增补

<a id="sup03"></a>

#### SUP03 Distribution grid impedance & topology estimation with limited or no micro-PMUs

**书目信息**：Haoran Li; Yang Weng; Yizheng Liao; Brian Keel; Kenneth E. Brown。International Journal of Electrical Power & Energy Systems 129, 106794, July 2021。

**来源**：[DOI](https://doi.org/10.1016/j.ijepes.2021.106794)；[来源1](https://yzliao.github.io/pub/2021IJPES-Top.pdf)；[来源2](https://doi.org/10.1016/j.ijepes.2021.106794)。

- **方法与对象**：OLS/FOLS估计端口阻抗并分析未测注入误差，再用RG恢复内部结构。
- **量测与隐藏节点**：平衡网可用V幅值/P/Q，三相情形用相量；hidden负荷可非零但相对平稳。
- **关键条件与证据边界**：误差定理针对特定相关模型的未测注入误差，未覆盖全部噪声及线性化误差。
- **综述用途**：补入带非零注入、相关性与误差量化路线。

**阅读层级与范围**：正文定向阅读；正文定向：§1–4模型、观测划分、OLS/FOLS、Cholesky与误差条件、RG；§5讨论误差超界的三相案例。

**定位**：作者PDF pp.3–4（观测/hidden负荷设置），pp.7–8（式1–3输入），pp.12–15（Theorems1–2及误差来源），pp.15–19（RG），pp.23–25（误差界与三相线性化）。

**版本**：作者公开36页稿，首页Preprint submitted to Elsevier / March 7, 2021；正式书目由出版社页核对。

**前序版本**：2019稿 *Robust Hidden Topology Identification in Distribution Systems*，arXiv:1902.01365。与2021文同五作者及相近发展线，存在题名和误差分析变化，主引2021文，保留版本差异。[作者预印本](https://arxiv.org/abs/1902.01365)。

<a id="sup04"></a>

#### SUP04 Low-voltage network topology and impedance identification using smart meter measurements

**书目信息**：Amina Benzerga; Daniele Maruli; Antonio Sutera; Alireza Bahmanyar; Sébastien Mathieu; Damien Ernst。2021 IEEE Madrid PowerTech, June 2021。

**来源**：[DOI](https://doi.org/10.1109/PowerTech46648.2021.9495093)；[来源1](https://orbi.uliege.be/bitstream/2268/258890/1/three_phases_lv_id.pdf)；[来源2](https://orbi.uliege.be/handle/2268/258890)。

- **方法与对象**：电压相关MST给初树，以阻抗恒定性检验选择局部插点构型。
- **量测与隐藏节点**：V/I幅值、坐标、相别、变压器量测、已知R/X；可有非零hidden注入。
- **关键条件与证据边界**：要求每相每馈线有量测；复杂缺表情形含几何与默认阻抗回退。隐藏叶节点的电压及对应支路阻抗不可估，使用辅助节点或合并电流处理。
- **综述用途**：比较地理/参数先验与缺表能力的取舍。

**阅读层级与范围**：正文定向阅读；正文§III–IV全部输入、假设、三种hidden构型优化、失败回退和三相合并；摘要与结论。

**定位**：PDF p.2 §III；pp.2–4 §IV-A式(1)–(10)；pp.4–5 §IV-B；p.6结论。

**版本**：机构仓储6页会议稿；首页第二作者拼写Maruli，机构题录同，相关另一篇使用Marulli。

<a id="sup05"></a>

#### SUP05 Reconstruction of Low-Voltage Networks with Limited Observability

**书目信息**：Daniele Marulli; Sébastien Mathieu; Amina Benzerga; Antonio Sutera; Damien Ernst。2021 IEEE PES Innovative Smart Grid Technologies Europe (ISGT Europe), 18–21 October 2021。

**来源**：[DOI](https://doi.org/10.1109/ISGTEurope52324.2021.9640163)；[来源1](https://orbi.uliege.be/bitstream/2268/238913/1/2021165122.pdf)；[来源2](https://orbi.uliege.be/handle/2268/238913)。

- **方法与对象**：相关MST与支路检验后，在局部比较缺节点构型，估hidden电流和阻抗。
- **量测与隐藏节点**：部分SM V/I幅值、相别与变压器量测；hidden可带注入。
- **关键条件与证据边界**：任意两hidden至少相隔3边；30节点模拟的80%覆盖表现属于经验范围。隐藏叶节点的电压及对应支路阻抗不可估，迭代以辅助节点处理。
- **综述用途**：与Benzerga2021相邻介绍，保留不同会议版本。

**阅读层级与范围**：正文定向阅读；§III问题设定、§IV模型/hidden假设和构型优化，摘要与结论。

**定位**：PDF pp.1–2 §III–IV，p.2 Assumption1，pp.2–3插点，p.5 §VI。

**版本**：ORBi author preprint v2，5页、五作者；v1为四作者，机构记录说明早期版本2019。DOI/作者/题名由Crossref出版社登记核准。

**相关发展线**：与[SUP04](#sup04)作者群和思路相近，但会刊、题名与所列假设不同，保留独立记录并相邻介绍。

<a id="sup06"></a>

#### SUP06 Joint Estimation of Topology and Injection Statistics in Distribution Grids With Missing Nodes

**书目信息**：Deepjyoti Deka; Michael Chertkov; Scott Backhaus。IEEE Transactions on Control of Network Systems 7(3):1391–1403, September 2020。

**来源**：[DOI](https://doi.org/10.1109/TCNS.2020.2977365)；[来源1](https://arxiv.org/pdf/1804.04742)；[来源2](https://experts.arizona.edu/en/publications/joint-estimation-of-topology-and-injection-statistics-in-distribu/)。

- **方法与对象**：电压差二阶矩构观测MST，再补missing点并估其注入统计。
- **量测与隐藏节点**：所读missing算法用V幅值/相角、注入统计、已知missing集合及候选阻抗。
- **关键条件与证据边界**：LC-PF、跨节点注入不相关、hidden度>2及分隔条件。
- **综述用途**：补充已知缺测节点集合与未知隐点数的任务差别。

**阅读层级与范围**：正文定向阅读；§III注入条件，§IV Assumptions2–3与Algorithms2–3输入/原理，有限样本阈值与结论。

**定位**：PDF p.3 Assumption1；pp.4–7 §IV，p.6 Algorithm2及p.7 Algorithm3；p.10结论。

**版本**：arXiv1804.04742v2（2020-01-02），12页；正式书目由作者机构记录核对。

**前序会议**：2017 *Estimating topology and injection statistics in distribution grids with hidden nodes*，DOI10.1109/SmartGridComm.2017.8340739。2020文明确为扩展版，本轮不另计独立方法。[会议书目](https://experts.arizona.edu/en/publications/estimating-topology-and-injection-statistics-in-distribution-grid/)。

<a id="sup07"></a>

#### SUP07 Learning partially observed meshed distribution grids

**书目信息**：Harish Doddi; Deepjyoti Deka; Murti Salapaka。2020 International Conference on Probabilistic Methods Applied to Power Systems (PMAPS), August 2020。

**来源**：[DOI](https://doi.org/10.1109/PMAPS47429.2020.9183648)；[来源1](https://salapakalab.ece.umn.edu/data/31abede1ef8f5a599568.pdf)；[来源2](https://salapakalab.ece.umn.edu/publications/doddi-2020-learning-partially-observed-meshed-distribution-grids/)。

- **方法与对象**：观测电压联合逆协方差作稀疏/低秩分解，再按符号和二跳结构插点。
- **量测与隐藏节点**：电压幅值与相角；含内部hidden。
- **关键条件与证据边界**：跨节点注入不相关、girth≥5、hidden间隔≥3hop；径向可放宽至2hop。分解成功须另评价。
- **综述用途**：补非径向hidden路线，避免把voltage-only译成仅幅值。

**阅读层级与范围**：正文定向阅读；§II–IV统计模型、隐藏位置条件、稀疏低秩估计与Algorithm1；模拟和结论。

**定位**：PDF pp.2–3 Assumptions1–3；pp.3–5 inverse covariance/SDP/Algorithm1，p.5阈值与数值设置。

**版本**：作者实验室6页IEEE会议PDF；DOI、作者、题名由Crossref出版社登记核准。

<a id="sup08"></a>

#### SUP08 Inverse Power Flow Problem

**书目信息**：Ye Yuan; Steven H. Low; Omid Ardakanian; Claire J. Tomlin。IEEE Transactions on Control of Network Systems 10(1):261–273, March 2023；2022接受/在线DOI。

**来源**：[DOI](https://doi.org/10.1109/TCNS.2022.3199084)；[来源1](https://webdocs.cs.ualberta.ca/~oardakan/files/tcns2023.pdf)；[来源2](https://authors.library.caltech.edu/records/99twq-8jn27)。

- **方法与对象**：先辨识Kron约化导纳，再以团结构与递归代数恢复原径向网络。
- **量测与隐藏节点**：同步V/I相量；hidden零注入，可彼此相邻。
- **关键条件与证据边界**：树、hidden度≥3、无shunt及测量秩等条件；正文Remark3将无shunt条件的放宽指向full version，未据此放宽树结构或节点度数条件。
- **综述用途**：补齐从端口等效反推物理隐藏树的条件性结果。

**阅读层级与范围**：正文定向阅读；§III最小二乘/满秩条件、§IV-A零注入与Kron约化、§IV-B Assumption3和clique分解/递归恢复框架，结论。

**定位**：作者PDF p.3 §III；pp.4–5 §IV-A Assumptions1–2；pp.5–7 Assumption3及Theorem1；pp.7–10恢复原Y。

**版本**：作者公开12页接受稿，版权页2022；正式2023书目由Caltech作者仓储核对。原2016预印本题名On the Inverse Power Flow Problem，arXiv1610.06631。

**前序版本**：2016 *On the Inverse Power Flow Problem*，arXiv:1610.06631；正式2023文为本项技术依据。[预印本](https://arxiv.org/abs/1610.06631)。

<a id="sup09"></a>

#### SUP09 基于智能电表数据的低压配电网拓扑与线路参数联合辨识

**书目信息**：马尚；卫志农；黄蔓云；郑玉平；孙国强。电力系统自动化，2024，48(2)：60–70。

**来源**：[DOI](https://doi.org/10.7500/AEPS20230720002)；[来源1](https://doi.org/10.7500/AEPS20230720002)；[来源2](https://opaj.napstic.cn/periodicalArticle/0120240200054088)；[来源3](https://coaa.istic.ac.cn/openjournal/periodicalArticle/0120240200054088)。

- **方法与对象**：功率—电压比逆潮流回归得到阻抗距离，再用免判定阈值的改进RG。
- **量测与隐藏节点**：摘要明确无相角、未知数量零注入隐点；具体量测向量待全文。
- **关键条件与证据边界**：径向线性模型；度数、覆盖和源电压条件未在摘要说明。
- **综述用途**：直接加入中文hidden相关工作，详细比较待正文。

**阅读层级与范围**：摘要或部分预览；中英文摘要及书目；未读算法、公式、假设与实验正文。

**定位**：官方平台摘要与引用信息。

<a id="sup10"></a>

#### SUP10 A data-driven method for joint estimation of topology and impedance in distribution networks using end-user voltage magnitude and power data

**书目信息**：Hengfeng Zhang; Wei Sun; Zhiqiang Zhang; Fei Lin; Qiyue Li; Daoming Mu; Jinzhu Yu。Sustainable Energy, Grids and Networks 47 (September 2026), 102385。

**来源**：[DOI](https://doi.org/10.1016/j.segan.2026.102385)；[来源1](https://www.sciencedirect.com/science/article/pii/S2352467726002675)；[来源2](https://api.crossref.org/works/10.1016/j.segan.2026.102385)。

- **方法与对象**：末端V幅值/P/Q作受物理约束矩阵回归，再全分支分组和非线性潮流精化。
- **量测与隐藏节点**：用户端幅值与功率；内部注入、度数和覆盖条件待全文。
- **关键条件与证据边界**：本次仅官方摘要/引言预览，不能据此写整体恢复保证。
- **综述用途**：与Park、Pengwah、Guo等直接对齐信息条件。

**阅读层级与范围**：摘要或部分预览；官方摘要及可读引言预览；未读方法公式、算法正文和实验表。

**定位**：官方摘要及可读引言预览；未读方法公式、算法正文和实验表。

<a id="sup11"></a>

#### SUP11 Robust coupling impedance-based topology identification method for asynchronous low-voltage distribution networks

**书目信息**：Peng Gao; Gaofeng Zhao; Yang Hu; Yang Li; Rui Liu。Sustainable Energy, Grids and Networks 47 (September 2026), 102509。

**来源**：[DOI](https://doi.org/10.1016/j.segan.2026.102509)；[来源1](https://www.sciencedirect.com/science/article/pii/S2352467726003917)；[来源2](https://api.crossref.org/works/10.1016/j.segan.2026.102509)；[来源3](https://ssrn.com/abstract=6750574)。

- **方法与对象**：对耦合阻抗作对称约束正则化估计，再层次聚类恢复内部分支。
- **量测与隐藏节点**：异步AMI的V/I变化，无新增分支传感器。
- **关键条件与证据边界**：具体异步误差、零注入/度数条件仍待方法正文；凸拟合最优与拓扑正确分别评价。
- **综述用途**：异步普通表计hidden主线；作者报告114表硬件原型。

**阅读层级与范围**：摘要或部分预览；官方摘要、完整可读引言与System model、robust method、simulation、conclusion节预览；无完整核心公式与定理。

**定位**：官方摘要、完整可读引言与System model、robust method、simulation、conclusion节预览；无完整核心公式与定理。

<a id="sup12"></a>

#### SUP12 A Novel Topology Recovery Method for Low Voltage Distribution Networks

**书目信息**：Sina Mohammadi; Van-Hai Bui; Wencong Su。arXiv:2502.18939v1，2025-02-26；作者稿注明获PESGM2025接受，作者出版清单亦列该会议。

**来源**：[DOI](https://doi.org/10.48550/arXiv.2502.18939)；[来源1](https://arxiv.org/html/2502.18939v1)；[来源2](https://arxiv.org/abs/2502.18939)；[来源3](https://sites.google.com/site/wencongsu/Home/publication)。

- **方法与对象**：电压增量精度矩阵构距离，以阈值分组，再用电流和段电阻递推隐点电压。
- **量测与隐藏节点**：全部叶端V/I；零注入内部点；各段相同已知电阻，近似纯电阻。
- **关键条件与证据边界**：阈值由四个模拟网络调得；度2恢复使用额外分段先验，未知阻抗列为后续方向。
- **综述用途**：应补入端点方法，具体说明其degree2与模型条件。

**阅读层级与范围**：正文定向阅读；v1全文的§II-A/B、Algorithm1、§III设置和§IV结论定向阅读；公式及表格仅用于核对模型，不复现算法。

**定位**：§II-B式(1)–(5)、Algorithm1；§III；§IV。


### 12.4 一般方法与综述

<a id="sup13"></a>

#### SUP13 Information-theoretic grid topology reconstruction using low-precision smart meter data

**书目信息**：Daniel T. Speckhard。Scientific Reports16,20547；出版社2026-07-03；所读作者稿arXiv:2505.11517v4，2026-05-11。

**来源**：[DOI](https://doi.org/10.1038/s41598-026-54490-4)；[来源1](https://arxiv.org/html/2505.11517v4)；[来源2](https://www.nature.com/articles/s41598-026-54490-4)；[来源3](https://arxiv.org/abs/2505.11517)。

- **方法与对象**：固定Chow–Liu算法，比较量化、精度、采样和时间窗，并合并重复量测节点。
- **量测与隐藏节点**：建模节点的电压增量；MATPOWER/GridLAB-D数据。
- **关键条件与证据边界**：无未知内部点插入；数据质量结果为所用模拟设置的敏感性证据。
- **综述用途**：用于设备/数据要求段，不作为普遍精度下限。

**阅读层级与范围**：正文定向阅读；作者v4 §2.1、Algorithm1、§2.5–2.7、§3.3–3.4及§4；出版社书目、摘要与方法前段。

**定位**：arXiv v4 §2.1、Algorithm1、§2.5–2.7、§3.3–3.4、§4。

<a id="sup14"></a>

#### SUP14 Data-Driven Topology Generation with Physics-Guidance in LV Distribution Networks

**书目信息**：Dong Liu; Juan S. Giraldo; Peter Palensky; Pedro P. Vergara。ISGT Europe2024，5页；会议2024-10-14至17。

**来源**：[DOI](https://doi.org/10.1109/ISGTEUROPE62998.2024.10863141)；[来源1](https://palensky.org/pdf/Liu2024b.pdf)；[来源2](https://research.tudelft.nl/en/publications/data-driven-topology-generation-with-physics-guidance-in-lv-distr/)。

- **方法与对象**：电压聚类划台变边界，GIS最短路/容量修正生成树，再识别馈线端点。
- **量测与隐藏节点**：V幅值、道路/建筑、变压器位置和馈线数等先验。
- **关键条件与证据边界**：真实网拓扑配仿真电压；输出近似地理电气模型，非纯端点任意hidden恢复。
- **综述用途**：补GIS与工程先验融合路线。

**阅读层级与范围**：正文定向阅读；作者公开5页原文§II三个阶段/Algorithms1–2、§III案例与鲁棒性、§IV结论；机构页核书目。

**定位**：PDF pp.2–3 §II，pp.3–5 §III，p.5 §IV。

<a id="sup15"></a>

#### SUP15 A data-driven topology identification method for low-voltage distribution networks based on the wavelet transform

**书目信息**：Sebastián García; Matteo Fresia; J.M. Mora-Merchán; Alejandro Carrasco; Enrique Personal; Carlos León。Electric Power Systems Research 243 (June 2025), 111517；机构PDF首页Available online 16 February 2025。

**来源**：[DOI](https://doi.org/10.1016/j.epsr.2025.111517)；[来源1](https://www.sciencedirect.com/science/article/pii/S0378779625001099)；[来源2](https://idus.us.es/server/api/core/bitstreams/41a40142-e7e6-43ac-a19c-302fc743094e/content)。

- **方法与对象**：智能表电量的小波时频特征用于馈线和相别归属。
- **量测与隐藏节点**：SM电量；具体方法章未读。
- **关键条件与证据边界**：目标为feeder/phase topology，未知物理接点能力未核。
- **综述用途**：一般低压低硬件输入方法对照。

**阅读层级与范围**：摘要或部分预览；官方摘要/Highlights及机构原PDF首页与引言片段；完整方法章本次未读。

**定位**：官方摘要/Highlights及机构原PDF首页与引言片段；完整方法章本次未读。

<a id="sup16"></a>

#### SUP16 A review of topology identification methods and applications for low voltage distribution networks

**书目信息**：作者全表待核。Energy Reports 15 (June 2026), 108914。

**来源**：[DOI](https://doi.org/10.1016/j.egyr.2025.12.051)；[来源1](https://www.sciencedirect.com/science/article/pii/S2352484725007899)。

- **方法与对象**：按低压层级任务、物理/数据路线和数据可用性整理方法与应用。
- **量测与隐藏节点**：综述；本次仅读官方摘要/Highlights。
- **关键条件与证据边界**：具体方法与保证仍追溯原始论文，作者全表待核。
- **综述用途**：补近期低压专门综述和引用追踪入口。

**阅读层级与范围**：摘要或部分预览；官方摘要与Highlights；未读完整综述章节与135篇引文原文。

**定位**：官方摘要与Highlights；未读完整综述章节与135篇引文原文。

<a id="sup17"></a>

#### SUP17 基于虚拟阻抗的低压配电网拓扑识别方法

**书目信息**：殷涵；宣羿；黄怡；孙智卿；赵健；鲍若愚。电力系统保护与控制，2024，52(3)：83–93。

**来源**：[DOI](https://doi.org/10.19783/j.cnki.pspc.230902)；[来源1](https://doi.org/10.19783/j.cnki.pspc.230902)；[来源2](https://opaj.napstic.cn/periodicalArticle/0120240301653962)。

- **方法与对象**：虚拟阻抗岭回归筛错，再以DDTW校验/纠正设备连接。
- **量测与隐藏节点**：既有单相电表与总表量测，具体配置待全文。
- **关键条件与证据边界**：摘要未报告未知数量内部点插入。
- **综述用途**：一般低压连接纠错的中文代表。

**阅读层级与范围**：摘要或部分预览；摘要及书目；期刊官网打开超时，未读正文。

**定位**：官方平台Abstract及引用文本。

<a id="sup18"></a>

#### SUP18 基于分时感知与分块回溯的低压配电网线路阻抗辨识方法

**书目信息**：王炜韬；赵健；王小宇；边晓燕。现代电力，2023，40(4)：614–622。

**来源**：[DOI](https://doi.org/10.19725/j.cnki.1007-2322.2021.0361)；[来源1](https://doi.org/10.19725/j.cnki.1007-2322.2021.0361)；[来源2](https://xddl.ncepujournal.com/cn/article/pdf/preview/10.19725/j.cnki.1007-2322.2021.0361.pdf)。

- **方法与对象**：拓扑分块、模糊熵校验、量测筛选及含边界的非线性阻抗估计。
- **量测与隐藏节点**：用户SM时序；完整覆盖和先验待全文。
- **关键条件与证据边界**：主要为参数估计与已有结构校验，不能据零注入缺测表述当作从零重建。
- **综述用途**：可选参数/数据质量背景。

**阅读层级与范围**：摘要或部分预览；官方PDF首页摘要、引言及检索返回的式(19)–(22)/算例段片段；未成功打开连续全文。

**定位**：原PDF p.1摘要/引言；式(19)–(22)及§3.1检索片段。

<a id="sup19"></a>

#### SUP19 Identification of Distribution Network Topology and Line Parameter Based on Smart Meter Measurements

**书目信息**：Chong Wang; Zheng Lou; Ming Li; Chaoyang Zhu; Dongsheng Jing。Energies，2024，17(4)：830；2024-02-09。

**来源**：[DOI](https://doi.org/10.3390/en17040830)；[来源1](https://www.mdpi.com/1996-1073/17/4/830)。

- **方法与对象**：线性初值、阈值消边，再以AC迭代和自适应岭回归精化导纳。
- **量测与隐藏节点**：各节点V/P/Q；§5.1实验明确全母线SM。
- **关键条件与证据边界**：给定节点集，无未知内部点生成；相角在模型中恢复。
- **综述用途**：一般非相量联合估计对照。

**阅读层级与范围**：正文定向阅读；摘要、§1、§2约束与消边、§4.1–4.2回归精化、§5.1实验量测设定及参考文献；非逐段精读全部结果。

**定位**：HTML §2式(6)–(9)后阈值消边；§4.1式(31)–(38)；§5.1全母线SM设定。

<a id="sup20"></a>

#### SUP20 基于改进相关熵匹配追踪的配电网拓扑与线路参数辨识方法

**书目信息**：孙蒙恩；龚海明；赵杰；孙蓉蓉；卢慧文；王辉。电气传动，2025，55(6)：81–88；2025-06-20。

**来源**：[DOI](https://doi.org/10.19457/j.1001-2095.dqcd25650)；[来源1](https://castjournals.cast.org.cn/joweb/dqcd/CN/1190568678979678408)。

- **方法与对象**：相关熵损失与改进匹配追踪估导纳，处理模拟高斯/非高斯噪声。
- **量测与隐藏节点**：P/Q、V和相角历史序列。
- **关键条件与证据边界**：固定节点/参数域；有限噪声仿真不构成任意分布保证。
- **综述用途**：可选鲁棒估计背景，非hidden主线。

**阅读层级与范围**：正文定向阅读；摘要、§1式(9)–(13)观测模型、§2求解与相关熵定义、参考文献；未精读全部仿真。

**定位**：HTML §1式(9)–(13)，§2及§2.1，摘要。

<a id="sup21"></a>

#### SUP21 基于稀疏回归与聚类修正的配电网拓扑与线路参数联合辨识方法研究

**书目信息**：张瑞峰；李波；廖凯；叶学顺；周俊；吴磊。电力科学与技术学报，2026，41(4)：38–50。

**来源**：[DOI](https://doi.org/10.19781/j.issn.1673-9140.2026.04.004)；[来源1](https://dlkjxb.csust.edu.cn/dlgcjs/article/issue/2026_41_4)；[来源2](https://dlkjxb.csust.edu.cn/)。

- **方法与对象**：电压预辨识、稀疏回归与聚类修正三步估计连接和参数。
- **量测与隐藏节点**：摘要仅称量测数据，具体配置待正文。
- **关键条件与证据边界**：未知物理hidden恢复未在摘要说明。
- **综述用途**：近期一般方法，可按篇幅选择。

**阅读层级与范围**：摘要或部分预览；官方摘要及书目。

**定位**：官方2026年第41卷第4期目录38–50条目及摘要。


### 12.5 已核书目待取原文

<a id="sup22"></a>

#### SUP22 A Heuristic-Search-Based Topology Identification and Parameter Estimation Method in Low Voltage Distribution Grids With Low Observability

**书目信息**：Nan Feng; Yaping Du; Yuxuan Ding。IEEE Transactions on Smart Grid 15(6), 5826–5839, November 2024。

**来源**：[DOI](https://doi.org/10.1109/TSG.2024.3435069)；[来源1](https://ieeexplore.ieee.org/document/10612997/)；[来源2](https://api.crossref.org/works/10.1109/TSG.2024.3435069)；[来源3](https://orcid.org/0009-0009-5129-9392)。

- **方法与对象**：主源书目已核，原文方法待取。
- **量测与隐藏节点**：题名直接针对低可观测LV联合辨识。
- **关键条件与证据边界**：不将转引摘要中的DTM/随机搜索细节当已核原文。
- **综述用途**：R015的同作者前作，应优先取得全文。

**阅读层级与范围**：仅书目；主源书目；官方IEEE页面要求JS/机器人验证，未读官方摘要或正文。

**定位**：主源书目；官方IEEE页面要求JS/机器人验证，未读官方摘要或正文。

**待补**：原文摘要/全文、具体输入、隐藏节点类型及恢复保证；不得用其他版本或专利无标识替代。

<a id="sup23"></a>

#### SUP23 Low-Voltage Distribution Grid Topology Identification and Parameter Estimation with Heterogeneous Cable Sizes and Dense Noise: A Prior Knowledge-Guided Adaptive Search Approach

**书目信息**：Nan Feng; Yaping Du; Yuxuan Ding; Qianchao Wang。IEEE Transactions on Smart Grid，作者ORCID列2026；准确上线日待核。

**来源**：[DOI](https://doi.org/10.1109/TSG.2026.3738092)；[来源1](https://doi.org/10.1109/TSG.2026.3738092)；[来源2](https://orcid.org/0009-0009-5129-9392)。

- **方法与对象**：作者ORCID/Crossref关联题录可核，原文待取。
- **量测与隐藏节点**：题名涉及电缆异质性、密集噪声和先验搜索。
- **关键条件与证据边界**：具体上线日尚未primary核准，暂作候选，不认定截至本日已正式发表。
- **综述用途**：对相关缺口有影响，先核日期和独立贡献。

**阅读层级与范围**：仅书目；作者ORCID/Crossref关联书目；未读官方摘要或正文。

**定位**：作者ORCID/Crossref关联书目；未读官方摘要或正文。

**待补**：原文摘要/全文、具体输入、隐藏节点类型及恢复保证；不得用其他版本或专利无标识替代。

<a id="sup24"></a>

#### SUP24 基于非线性回归的含隐节点低压配电网参数和拓扑联合辨识

**书目信息**：梁栋；张煜堃；王守相；孙智卿。电力系统及其自动化学报，2021，33(11)：28–36。

**来源**：[来源1](https://ee.hebut.edu.cn/szdw/jsml/L/3c786d6fd3364eb0b31c9ffd7f75cc61.html)；[来源2](https://teacher.hebut.edu.cn/publish/201714757ea7fb8c11edab95286ed488d6f1044/lwcg/%E7%A7%91%E7%A0%94%E8%AE%BA%E6%96%871.html)。

- **方法与对象**：作者大学已核非线性回归联合辨识书目，方法待原文。
- **量测与隐藏节点**：题名含隐节点，具体输入和注入类型未核。
- **关键条件与证据边界**：相关专利不能代替论文正文或证明方法完全相同。
- **综述用途**：中文非线性hidden方向的优先全文项。

**阅读层级与范围**：仅书目；题名、四作者、期刊、卷期页码已核；尚无论文摘要或正文。

**定位**：大学官方作者页科研论文[15]；作者个人站科研论文[9]。

**待补**：原文摘要/全文、具体输入、隐藏节点类型及恢复保证；不得用其他版本或专利无标识替代。

<a id="sup25"></a>

#### SUP25 Topology Identification of Low-Voltage Distribution Networks with Latent Nodes Considering Impedance Parameter Constraints

**书目信息**：Ruiyao Jia; Wenpeng Luan; Bo Liu; Yang Han; Bowei Hu。2026 IEEE PES International Meeting (PES IM)，会议总目录起页768（书目库另记单篇1–5页）。

**来源**：[DOI](https://doi.org/10.1109/PESIM67009.2026.11439036)；[来源1](https://ieeexplore.ieee.org/document/11439036)；[来源2](https://www.proceedings.com/content/084/084963webtoc.pdf)。

- **方法与对象**：出版目录已核latent nodes与阻抗约束主题，方法待原文。
- **量测与隐藏节点**：量测和隐藏节点条件未核。
- **关键条件与证据边界**：IEEE验证阻挡，第三方摘要只作线索。
- **综述用途**：近期直接主题，取原文后再比较。

**阅读层级与范围**：仅书目；书目与目录；未读原始摘要或正文。

**定位**：PES IM2026会议目录p.768条目；IEEE document11439036入口。

**待补**：原文摘要/全文、具体输入、隐藏节点类型及恢复保证；不得用其他版本或专利无标识替代。

<a id="sup26"></a>

#### SUP26 Topology and Parameter Joint Identification in Imbalanced Low-Voltage Distribution Networks Based on Load Characteristic Propagation

**书目信息**：Yanan Zhang; Gan Zhou; Huan Mao; Wei Gu; Yanjun Feng; Rui Bo。IEEE Transactions on Smart Grid17(2),1680–1692，2026；作者机构记录给2025版权年份。

**来源**：[DOI](https://doi.org/10.1109/TSG.2025.3632927)；[来源1](https://mst.elsevierpure.com/en/publications/topology-and-parameter-joint-identification-in-imbalanced-low-vol/)；[来源2](https://doi.org/10.1109/TSG.2025.3632927)。

- **方法与对象**：作者机构书目与hidden nodes关键词已核，原文方法待取。
- **量测与隐藏节点**：题名涉及三相不平衡、负荷特征传播与参数。
- **关键条件与证据边界**：不据第三方摘要补写算法或假设。
- **综述用途**：高优先级三相hidden原文核对项。

**阅读层级与范围**：仅书目；作者大学研究记录的书目与keywords；IEEE正文未取得。

**定位**：作者大学研究记录的书目与keywords；IEEE正文未取得。

**待补**：原文摘要/全文、具体输入、隐藏节点类型及恢复保证；不得用其他版本或专利无标识替代。

<a id="sup27"></a>

#### SUP27 A Data-Driven Topology and Parameter Joint Estimation Method in Non-PMU Distribution Networks

**书目信息**：Xiaoxue Wang; Yikang Zhao; Yue Zhou。IEEE Transactions on Power Systems39(1),1681–1692，2024；DOI登记年份2023。

**来源**：[DOI](https://doi.org/10.1109/TPWRS.2023.3242458)；[来源1](https://ee.hebut.edu.cn/szdw/jsml/W/884dcd321f2244a18ef3ffe205ccb5ca.html)。

- **方法与对象**：具体模型与算法待原文；R015的相关矩阵方法引用指向此文。
- **量测与隐藏节点**：非PMU联合估计；具体观测和hidden能力待核。
- **关键条件与证据边界**：当前只采用作者机构书目。
- **综述用途**：追溯TLM相关方法来源，优先核原文。

**阅读层级与范围**：仅书目；作者大学官网代表性论文[1]核对题名、三作者、期刊、2024与DOI；原文尚未取得。

**定位**：河北工业大学王笑雪教师页“代表性论文”[1]。

**待补**：原文摘要/全文、具体输入、隐藏节点类型及恢复保证；不得用其他版本或专利无标识替代。

### 12.6 暂不扩展为核心基线的线索

- 同作者NAPS2025 *An Efficient Voltage-Based Topology Recovery Method for Distribution Networks* 已见作者出版清单；与SUP12的关系及方法未核，不另计已读独立方法。[作者清单](https://sites.google.com/site/wencongsu/Home/publication)。
- *On Identification of Distribution Grids*（10.1109/TCNS.2019.2891002）的主方案采用全观测，应与SUP08的零注入hidden逆潮流恢复分开。
- 2019三相统计拓扑/相别（10.1049/iet-stg.2018.0291）、2026混合信号相别/概率连通（10.1016/j.epsr.2025.112591）可作一般方法线索，未知物理节点能力尚未核定。
- 新近监督学习、一般电压控制和充电优化只在直接服务论证时纳入；检索到新题名不自动增加隐藏节点算法数量。

### 12.7 对综述论证的实际影响

1. 把“隐藏点必须零注入”改成方法分型：已有允许非零未测负荷的方案，同时具有平稳性、地理/RX或空间分隔等条件。
2. 将普通智能表的数据项目、精度、同步与覆盖分开；已有免相角、异步和精度敏感性研究，应比较具体条件。
3. 将Kron端口等效与在条件下反推原树分开；也把已知段阻抗支持的度2恢复与任意未知串联结构分开。
4. 近期Feng、梁栋、Wang、Jia和三相瞬态论文仍有全文缺口。它们是优先核对清单，不据当前书目认定已解决所有相关问题。
5. 原第8节段落可继续使用，但正式写研究缺口前应把这些最接近文献与本文的实际信息契约逐项对齐。
