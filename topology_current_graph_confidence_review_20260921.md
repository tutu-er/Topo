# 配电网拓扑辨识、特征电流、图论与置信度：文献调研

调研日期：2026-09-21。范围：以中低压配电网，尤其径向低压配电网为主，包含必要的图学习和统计推断基础。

本报告是按问题组织的代表性文献调研，不是声称穷尽数据库的系统综述。来源优先使用出版社、作者全文、arXiv及高校机构仓储。论文结果与本报告自行推导的解释分别标记；未取得全文的条目不据摘要补造定理或实验数字。没有对论文代码或实验做本次复现。

## 1. 先确定辨识对象

“拓扑辨识”至少包括四种任务：用户属于哪个台区、用户属于哪一相、用户属于哪条分支，以及恢复节点之间的完整线路连接/开关状态。前三者正确，不意味着完整物理图已恢复。已知候选线路上的开关状态识别，也不同于没有候选图的结构学习。

应区分三类图：

| 图 | 节点/边的含义 | 主要误区 |
|---|---|---|
| 物理电气图 | 实际母线、接头、线路、开关 | 将用户归属关系当作所有物理线路 |
| 端口等效图 | 消去未观测内部节点后保持端口电气关系的图 | 等效边不一定是实际直连线路，Kron图也不一定是树 |
| 统计依赖图 | 电压/电流随机变量间的依赖或条件依赖 | 统计边不一定对应物理边，可含二跳关系 |

入门综述：Deka, Kekatos, Cavraro, *Learning Distribution Grid Topologies: A Tutorial*, IEEE Transactions on Smart Grid, 15(1):999–1013, 2024（2023年在线发表）。[DOI](https://doi.org/10.1109/TSG.2023.3271902)，[作者预印本](https://arxiv.org/abs/2206.10837)。其价值是按照量测对象、量测位置和主动/被动数据取得方式对齐可辨识性条件，而非仅按算法名称分类。

另一综述：Dalavi, Hamedani Golshan, Hatziargyriou, *A review on topology identification methods and applications in distribution networks*, Electric Power Systems Research, 234:110538, 2024。[DOI](https://doi.org/10.1016/j.epsr.2024.110538)。覆盖状态估计、数据驱动、主动探测、DER、数据质量和开关变化。

## 2. 特征电流不是一个统一的英文术语

| 路线 | 常用英文 | 如何携带拓扑信息 | 主要条件 |
|---|---|---|---|
| 人工编码/脉冲电流 | characteristic current; coded current injection; pulse current injection | 接收端检测终端ID或已知波形，确定祖先/后代集合 | 发射/接收设备位置、同步、信号可区分性、漏检与串扰模型 |
| 被动电流量测 | current measurements; branch current; current signature | 利用KCL、负荷变化、时序相关或事件响应 | 不同用户需要有可区分变化；电流幅值一般不能直接线性相加 |
| 谐波与高频信号 | harmonic synchrophasors; high-frequency signal injection | 利用频域传播、相量关系和网络频率响应 | 需要匹配频段的线路/负荷/耦合模型 |
| 逆变器主动探测 | inverter probing; grid probing; active probing | 调节已知P/Q注入，观察电压响应，估计灵敏度/线路结构 | 激励秩、叶节点覆盖、观测位置、背景负荷变化 |

“current signature”单独搜索经常进入电机故障诊断或非侵入式负荷识别，必须加 distribution network、topology 等限定。主动P/Q探测与高频载波注入的物理模型不同，不能直接套用同一可恢复性定理。

## 3. 特征电流如何转成图结构

以下是便于统一理解的理想化推导，不宣称是某篇论文原式。

对单电源径向树，规定一致的电流方向，在忽略并联泄漏和支路外返回通道、使用有符号电流或相量时，KCL给出

\[
I_e(t)=\sum_{j\in D(e)}i_j(t),
\]

其中D(e)为边e下游的注入终端集合。终端j叠加已知编码c_j(t)后，接收信号可写为

\[
Y=HC+E,\qquad H_{ej}=\mathbf1\{j\in D(e)\}.
\]

若编码矩阵C行满秩，可通过相关解调/最小二乘估计H；理想无噪声时H描述哪些支路承载哪些终端的编码。树上的下游集合满足层叠性：任意两个集合要么不相交，要么一个包含另一个。因而可以从集合包含关系构建祖先关系，进一步提取直接父子关系。

**可辨识性边界：**如果两条边在所有已激励终端上的下游集合相同，它们给出的签名相同。这种探测配置无法区分二者。典型例子是两个无观测、无侧支路的连续内部节点；增加重复采样不能消除这种结构歧义。恢复能力取决于激励位置和接收位置的组合。

高频情况下H通常应替换为复数传递矩阵H(f)：电缆电容、负载阻抗、反射、零线及相间耦合可能破坏二值路径模型。需要对漏检与多路径传播建立模型，不能简单把“收到信号”解释成“物理直连”。

## 4. 图论的四个重点

### 4.1 生成树及径向约束

构造节点对距离或相似度后，可用最小/最大生成树得到连通无环图。生成树仅保证输出形状为树；要保证是真实电网，还需证明所用权重具有正确的物理排序。候选边集合必须包含真边。网状运行应另行处理。

### 4.2 拉普拉斯、有效电阻和共享路径

为避免电压幅值/平方电压中系数2的混用，先在标量电阻网络中定义：B为移去参考节点列后的支路—节点关联矩阵，r_e>0，

\[
L=B^\top\operatorname{diag}(1/r_e)B,\qquad Z=L^{-1}.
\]

树上Z_ij等于根到i与根到j的公共路径电阻之和，因此

\[
d_{ij}=Z_{ii}+Z_{jj}-2Z_{ij}=\sum_{e\in\operatorname{path}(i,j)}r_e.
\]

这把线路恢复转化为加性树距离恢复。电力系统线性化模型中的R/X灵敏度可利用同类结构，但须先统一功率符号、标幺基准和电压变量定义。

### 4.3 隐节点、树距离与Kron消元

仅有叶端观测时，满足正边长、足够激励等条件的最小潜在树模型可用递归分组/邻接类方法恢复。未观测二度节点一般无法唯一分开。

对未观测且满足零注入假设的节点h进行消元，端口o看到

\[
L_{\rm red}=L_{oo}-L_{oh}L_{hh}^{-1}L_{ho}.
\]

Kron消元可能产生稠密等效图；它与“只收缩二度节点、保留分叉点的约化树”不是同一个对象。隐藏非零负荷还会引入等效注入项，不能无条件按零注入消元。

### 4.4 概率图不等于物理图

电压协方差、精度矩阵或互信息图中可能出现二跳关系。因此graphical lasso、Chow–Liu或GNN生成的边，要经过电力物理条件或额外辨识步骤，才能解释为真实线路。图学习的统计保证也不能自动变成电网恢复保证。

## 5. 置信度的对象必须写清楚

| 输出 | 回答的问题 | 不能自动推出 |
|---|---|---|
| 残差、相关系数、SNR、softmax | 模型拟合或分类打分多强 | 当前拓扑正确概率 |
| bootstrap边出现频率 | 数据重采样后边选择有多稳定 | 边是真边的频率学概率或贝叶斯后验 |
| Dempster–Shafer证据评分 | 多来源在所定义融合规则下支持多强 | 具有预设错误率的95%置信保证 |
| 贝叶斯后验P(T\|D) | 给定候选空间、先验和似然后，拓扑的条件概率 | 模型错误或候选真值缺失时仍可信 |
| 频率学置信集/检验反演 | 重复生成数据时，集合覆盖真值的概率 | 给已实现的固定图赋同样概率 |
| 共形预测集合 | 对交换性的新案例，候选标签集合的边际覆盖率 | 单个案例的条件正确率、开放世界新拓扑的覆盖 |
| 优化最优性间隙 | 给定模型及可行域内距最优值多远 | 输出与真实物理图一致 |

### 5.1 贝叶斯拓扑推断

若线路参数和负荷等未知量为theta，合理的拓扑后验形式为

\[
P(T\mid D)\propto P(T)\int P(D\mid T,\theta)P(\theta\mid T)\,d\theta.
\]

只把每个拓扑下的最小残差softmax化，除非有相应概率模型与处理未知参数的推导，否则只是评分。边的后验概率应通过合法整图求和得到：P(e存在\|D)=sum_{T:e属于T}P(T\|D)。边事件有依赖，不能简单相乘得到整图正确概率。

### 5.2 共形预测与测试单位

冻结评分函数，用独立交换的校准案例取得阈值后，可构造候选拓扑标签集合C_alpha(X)，满足相应条件下

\[
P\{T_{\rm new}\in C_\alpha(X_{\rm new})\}\ge1-\alpha.
\]

这里的概率通常是跨新案例的边际覆盖。一个拓扑下连续秒级采样不能当然视为大量独立拓扑案例；可考虑独立台区、独立事件或有合理依据的窗口作为校准单位。alpha=0.05时，标准split conformal的有限非平凡阈值至少需要19个校准单位（按类别校准则每类分别需要），这只是分位数分辨率门槛，不意味着样本已足够可靠。自适应选择探测后若评分流程改变，需对最终流程重新校准或使用专门的自适应理论。

### 5.3 推荐报告指标

同时报告用户台区/相别/分支准确率、边precision/recall、整树完全恢复率；置信评分另报校准曲线、Brier score或对数损失；集合方法另报覆盖率、集合大小、空集率、拒识率。测试必须保留未识别、信号全失、真实拓扑不在候选池和负荷/设备分布变化的案例。不要只在成功识别的子集上计算“总体准确率”。

## 6. 数据库检索方法

Google Scholar可将下文英文题目整体放入英文双引号。IEEE论文可在Xplore中直接查完整题目或DOI。非IEEE出版社的论文不保证有Xplore原文条目；此时使用主题检索查相邻IEEE研究。[Google Scholar官方说明](https://scholar.google.com/intl/en/scholar/help.html)，[IEEE官方搜索说明](https://xplorestaging.ieee.org/Xplorehelp/searching-ieee-xplore/search-tips)。

### Google Scholar主题词

```text
"distribution grid" "topology identification" tutorial
"distribution network" "current injection" topology
"low voltage" "characteristic current" topology
"distribution network" "signal injection" "missing data"
"distribution network" "current signature" topology
"distribution network" "harmonic synchrophasors" topology
"distribution network" "high frequency" "topology identification"
"distribution grid" "inverter probing"
"distribution grid" "latent tree" topology
"distribution grid" "effective resistance" topology
"distribution network" "Kron reduction" identification
"distribution grid" "graphical models" topology
"distribution network" topology Bayesian
"distribution network" topology "hypothesis testing"
"distribution network" topology "confidence"
"distribution network" topology "conformal prediction"
"distribution network" topology "sample complexity"
"配电网" "拓扑辨识" "特征电流"
"低压" "拓扑识别" "信号注入"
"配电网" "拓扑辨识" "置信度"
```

以上宜分别运行，不把所有词塞入同一式。还应轮换 topology identification / topology learning / topology estimation / topology reconstruction / topology detection；相别任务补 phase identification，户变关系补 consumer-transformer connectivity。

### IEEE Xplore主题检索式

以下使用英文半角引号及大写布尔运算符，可放入Basic Search或Command Search，默认元数据检索：

```text
("distribution network" OR "distribution grid") AND ("topology identification" OR "topology learning" OR "topology estimation")

("distribution network" OR "low voltage") AND topology AND ("current injection" OR "characteristic current" OR "signal injection")

("distribution network" OR "distribution grid") AND topology AND (harmonic OR "high frequency" OR "current signature")

("distribution network" OR "distribution grid") AND topology AND ("inverter probing" OR "active probing" OR "grid probing")

("distribution network" OR "distribution grid") AND topology AND ("graphical models" OR "spanning tree" OR "latent tree")

("distribution network" OR "distribution grid") AND ("effective resistance" OR "Kron reduction" OR Laplacian)

("distribution network" OR "distribution grid") AND topology AND (Bayesian OR posterior OR "hypothesis testing")

("distribution network" OR "distribution grid") AND topology AND (confidence OR uncertainty OR calibration)

("distribution network" OR "distribution grid") AND topology AND "conformal prediction"

("distribution network" OR "distribution grid") AND topology AND (identifiability OR "sample complexity" OR "error bounds")
```

先不设年份查理论根文献，再筛2021—2026查应用与新进展。检索量过少时先去掉“low voltage”等场景词，不应因IEEE无条目就判断方向没有工作。

## 7. 文献证据表



共收录27项条目，包括综述、研究论文、专著章节与明确标记的题录线索。证据级别说明本次读到了什么。搜索链接按题名生成，未声称逐条执行两个数据库检索或记录命中数。

### 综述入口

#### R1. Learning Distribution Grid Topologies: A Tutorial

Deepjyoti Deka; Vassilis Kekatos; Guido Cavraro。IEEE Transactions on Smart Grid 15(1):999–1013, 2024；2023年在线发表。DOI：10.1109/TSG.2023.3271902。

- **论文结果：**按量测位置、数据类型与主动/被动方式统一比较拓扑学习；覆盖图搜索、最小二乘、凸优化及混合整数模型。适合作为整个调研的入口。
- **假设/边界：**主要围绕径向单相模型，各论文的测量假设和恢复目标不可混合。
- **证据级别：**作者全文及机构正式题录。[原始来源](https://arxiv.org/abs/2206.10837)；[DOI入口](https://doi.org/10.1109/TSG.2023.3271902)。
- **Google Scholar词条：**`"Learning Distribution Grid Topologies: A Tutorial"`。[打开检索](https://scholar.google.com/scholar?q=%22Learning%20Distribution%20Grid%20Topologies%3A%20A%20Tutorial%22)。
- **IEEE Xplore精确题名词条：**`"Document Title":"Learning Distribution Grid Topologies: A Tutorial"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Document%20Title%22%3A%22Learning%20Distribution%20Grid%20Topologies%3A%20A%20Tutorial%22)。

#### R2. A review on topology identification methods and applications in distribution networks

Farzad Dalavi; Mohamad Esmail Hamedani Golshan; Nikos D. Hatziargyriou。Electric Power Systems Research 234:110538, 2024。DOI：10.1016/j.epsr.2024.110538。

- **论文结果：**综合状态估计、数据驱动、主动探测、DER、数据质量及多开关变化的分类与应用。
- **假设/边界：**用于领域地图；不能把综述中不同方法的数值直接拿来公平排名。
- **证据级别：**出版社摘要及可见正文。[原始来源](https://www.sciencedirect.com/science/article/pii/S0378779624004243)；[DOI入口](https://doi.org/10.1016/j.epsr.2024.110538)。
- **Google Scholar词条：**`"A review on topology identification methods and applications in distribution networks"`。[打开检索](https://scholar.google.com/scholar?q=%22A%20review%20on%20topology%20identification%20methods%20and%20applications%20in%20distribution%20networks%22)。
- **IEEE Xplore相邻主题词条：**`"distribution network" AND topology`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22distribution%20network%22%20AND%20topology)。非IEEE出版物不保证有Xplore原文条目。

### 特征电流与频域测量

#### C1. Topology identification of low voltage distribution network based on current injection method

Haotian Ge; Bingyin Xu; Wengyin Chen; Xinhui Zhang; Yongjian Bi。Archives of Electrical Engineering 70(2):297–306, 2021。DOI：10.24425/aee.2021.136985。

- **论文结果：**通过负荷端依次注入电流及上游接收关系判定连接。仿真为5 A、220 Hz；实验使用240 Hz方波，检测到注入点至电源方向的响应。
- **假设/边界：**证据为仿真与实验室；传播方向结论依赖论文电路模型，不能推广到任意高频耦合、多返回路径或复杂负荷。
- **证据级别：**出版社全文。[原始来源](https://journals.pan.pl/Content/119954/art04.pdf)；[DOI入口](https://doi.org/10.24425/aee.2021.136985)。
- **Google Scholar词条：**`"Topology identification of low voltage distribution network based on current injection method"`。[打开检索](https://scholar.google.com/scholar?q=%22Topology%20identification%20of%20low%20voltage%20distribution%20network%20based%20on%20current%20injection%20method%22)。
- **IEEE Xplore相邻主题词条：**`"distribution network" AND topology AND "current injection"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22distribution%20network%22%20AND%20topology%20AND%20%22current%20injection%22)。非IEEE出版物不保证有Xplore原文条目。

#### C2. Research on Topology Recognition Technology Based on Intelligent Measurement Switches

Dezhi Xiong; Jingjuan Du。Electronics 11(23):3903, 2022。DOI：10.3390/electronics11233903。

- **论文结果：**智能开关生成833.3 Hz默认载波并采用OOK编码。9个开关的实验室系统在96 s内完成辨识，报告该试验准确率100%。
- **假设/边界：**作者明确指出实验拓扑较简单，实际应用通常低于100%；这不是整图正确率的统计置信保证。
- **证据级别：**出版社全文。[原始来源](https://www.mdpi.com/2079-9292/11/23/3903)；[DOI入口](https://doi.org/10.3390/electronics11233903)。
- **Google Scholar词条：**`"Research on Topology Recognition Technology Based on Intelligent Measurement Switches"`。[打开检索](https://scholar.google.com/scholar?q=%22Research%20on%20Topology%20Recognition%20Technology%20Based%20on%20Intelligent%20Measurement%20Switches%22)。
- **IEEE Xplore相邻主题词条：**`topology AND "characteristic current"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=topology%20AND%20%22characteristic%20current%22)。非IEEE出版物不保证有Xplore原文条目。

#### C3. 低压配电物联网台区拓扑识别技术测试研究

刘家齐；刘浩军；童力；周金辉；任广振；马振宇。浙江电力 41(12):12–20, 2022。DOI：10.19585/j.zjdl.202212002。

- **论文结果：**搭建边端交互平台，比较浙江电网主流台区拓扑方案的可靠性、准确率与抗扰能力，选取有功电流注入配合频域检测的工程方案。
- **假设/边界：**出版年是2022，不能用PDF上传路径中的2023充当出版年；比较结论限于文中的参评方案与测试条件，参数不是自动适用的通用标准。
- **证据级别：**期刊PDF首页及摘要；未逐表审核全部数值。[原始来源](https://zjdl.cbpt.cnki.net/portal/journal/portal//client/paper/preview?cache=false&filePath=history%2FEditor%2F2023%2F0109%2Fzjdl%2F89bb0fa4-d177-4f33-8ec3-d1816b03f3b6.pdf)；[DOI入口](https://doi.org/10.19585/j.zjdl.202212002)。
- **Google Scholar词条：**`"低压配电物联网台区拓扑识别技术测试研究"`。[打开检索](https://scholar.google.com/scholar?q=%22%E4%BD%8E%E5%8E%8B%E9%85%8D%E7%94%B5%E7%89%A9%E8%81%94%E7%BD%91%E5%8F%B0%E5%8C%BA%E6%8B%93%E6%89%91%E8%AF%86%E5%88%AB%E6%8A%80%E6%9C%AF%E6%B5%8B%E8%AF%95%E7%A0%94%E7%A9%B6%22)。
- **IEEE Xplore相邻主题词条：**`topology AND "current injection" AND detection`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=topology%20AND%20%22current%20injection%22%20AND%20detection)。非IEEE出版物不保证有Xplore原文条目。

#### C4. Signal Injection-Based Topology Identification for Low-Voltage Distribution Networks Considering Missing Data

Yilong Duan; Zheng Liu; Yuanyuan Liu; Yong Li。Energies 17(9):2060, 2024。DOI：10.3390/en17092060。

- **论文结果：**用编码电流传输设备ID，依据特征信号记录FSR的包含关系恢复树；通过纵向、横向补全处理记录缺失。在实际台区演示，论文提出一轮可缩短至1 min。
- **假设/边界：**完全无收发记录的设备仍可能无法定位；现场发现过SPM未接线，断电失联设备也存在未辨识。其价值是处理部分缺失，不是消除所有缺失带来的不可辨识性。
- **证据级别：**出版社全文。[原始来源](https://www.mdpi.com/1996-1073/17/9/2060/html)；[DOI入口](https://doi.org/10.3390/en17092060)。
- **Google Scholar词条：**`"Signal Injection-Based Topology Identification for Low-Voltage Distribution Networks Considering Missing Data"`。[打开检索](https://scholar.google.com/scholar?q=%22Signal%20Injection-Based%20Topology%20Identification%20for%20Low-Voltage%20Distribution%20Networks%20Considering%20Missing%20Data%22)。
- **IEEE Xplore相邻主题词条：**`topology AND "signal injection" AND "missing data"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=topology%20AND%20%22signal%20injection%22%20AND%20%22missing%20data%22)。非IEEE出版物不保证有Xplore原文条目。


#### C5. Low-voltage distribution network topology identification method based on characteristic current

Xuanping Lai; Min Cao; Siyang Liu; Chen Sun。2021 6th Asia Conference on Power and Electrical Engineering (ACPEE), 1233–1238。DOI：10.1109/ACPEE51499.2021.9437092。

- **论文结果：**直接命中characteristic current关键词；摘要索引描述智能终端与TMR传感器用于特征电流检测。
- **假设/边界：**本次不能核对准确率、检测限或现场规模；作为优先获取全文的直接线索。
- **证据级别：**会议目录核对题目作者页码；全文未取得；DOI与方法摘要仅索引交叉核对。[原始来源](https://www.proceedings.com/content/059/059172webtoc.pdf)；[DOI入口](https://doi.org/10.1109/ACPEE51499.2021.9437092)。
- **Google Scholar词条：**`"Low-voltage distribution network topology identification method based on characteristic current"`。[打开检索](https://scholar.google.com/scholar?q=%22Low-voltage%20distribution%20network%20topology%20identification%20method%20based%20on%20characteristic%20current%22)。
- **IEEE Xplore精确题名词条：**`"Document Title":"Low-voltage distribution network topology identification method based on characteristic current"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Document%20Title%22%3A%22Low-voltage%20distribution%20network%20topology%20identification%20method%20based%20on%20characteristic%20current%22)。

#### C6. Switch Status Identification in Distribution Networks Using Harmonic Synchrophasor Measurements

Lei Chen; Mohammad Farajollahi; Mahdi Ghamkhari; Wei Zhao; Songling Huang; Hamed Mohsenian-Rad。IEEE Transactions on Smart Grid 12(3):2413–2424, 2021；2020年在线发表。DOI：10.1109/TSG.2020.3038214。

- **论文结果：**联合基波与谐波电流同步相量，用MILP辨识开关状态，并分析谐波源和传感器配置的可观测条件。在IEEE 33和123节点系统中优于只使用基波信息的对照方法。
- **假设/边界：**没有谐波电流不自动表示支路断开。模型为径向运行，谐波源位置和传感器覆盖决定可观测性。旧arXiv题名为Topology Identification in Distribution Networks using Harmonic Synchrophasor Measurements（2002.01787）；正式题名已改为Switch Status Identification，不能混淆版本结果。
- **证据级别：**作者正式接收稿全文。[原始来源](https://intra.ece.ucr.edu/~hamed/ChFGhZhHMRjTSG2020.pdf)；[DOI入口](https://doi.org/10.1109/TSG.2020.3038214)。
- **Google Scholar词条：**`"Switch Status Identification in Distribution Networks Using Harmonic Synchrophasor Measurements"`。[打开检索](https://scholar.google.com/scholar?q=%22Switch%20Status%20Identification%20in%20Distribution%20Networks%20Using%20Harmonic%20Synchrophasor%20Measurements%22)。
- **IEEE Xplore精确题名词条：**`"Document Title":"Switch Status Identification in Distribution Networks Using Harmonic Synchrophasor Measurements"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Document%20Title%22%3A%22Switch%20Status%20Identification%20in%20Distribution%20Networks%20Using%20Harmonic%20Synchrophasor%20Measurements%22)。

#### C7. Low-voltage overhead lines topology identification method based on high-frequency signal injection

Haotian Ge; Bingyin Xu; Xinhui Zhang; Yongjian Bi。Archives of Electrical Engineering 70(4):791–800, 2021。DOI：10.24425/aee.2021.138261。

- **论文结果：**注入5 MHz信号并利用电感隔离、传播延时和反射信息推断低压架空线路拓扑及长度。
- **假设/边界：**验证为MATLAB仿真；模型属于高频传输线问题，与低频编码微电流方案应分别讨论。
- **证据级别：**出版社全文。[原始来源](https://journals.pan.pl/Content/121560/PDF/art04.pdf)；[DOI入口](https://doi.org/10.24425/aee.2021.138261)。
- **Google Scholar词条：**`"Low-voltage overhead lines topology identification method based on high-frequency signal injection"`。[打开检索](https://scholar.google.com/scholar?q=%22Low-voltage%20overhead%20lines%20topology%20identification%20method%20based%20on%20high-frequency%20signal%20injection%22)。
- **IEEE Xplore相邻主题词条：**`topology AND "high frequency" AND "signal injection"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=topology%20AND%20%22high%20frequency%22%20AND%20%22signal%20injection%22)。非IEEE出版物不保证有Xplore原文条目。

#### C8. Utilising Smart-Meter Harmonic Data for Low-Voltage Network Topology Identification

Ali Othman; Neville R. Watson; Andrew Lapthorn; Radnya Mukhedkar。Energies 18(13):3333, 2025。DOI：10.3390/en18133333。

- **论文结果：**电压THD及2—20次谐波特征结合修改后的Kruskal生成树算法。合成量测实验中，THD在小于30 min采样间隔下报告拓扑相似度1.0。
- **假设/边界：**这是被动谐波电压特征，不是电流主动注入；合成数据的相似度1.0不是置信度1或现场普遍成功保证。
- **证据级别：**出版社全文。[原始来源](https://www.mdpi.com/1996-1073/18/13/3333)；[DOI入口](https://doi.org/10.3390/en18133333)。
- **Google Scholar词条：**`"Utilising Smart-Meter Harmonic Data for Low-Voltage Network Topology Identification"`。[打开检索](https://scholar.google.com/scholar?q=%22Utilising%20Smart-Meter%20Harmonic%20Data%20for%20Low-Voltage%20Network%20Topology%20Identification%22)。
- **IEEE Xplore相邻主题词条：**`topology AND "smart meter" AND harmonic`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=topology%20AND%20%22smart%20meter%22%20AND%20harmonic)。非IEEE出版物不保证有Xplore原文条目。

### 图论与可辨识性

#### G1. Kron Reduction of Graphs With Applications to Electrical Networks

Florian Dörfler; Francesco Bullo。IEEE Transactions on Circuits and Systems I: Regular Papers 60(1):150–163, 2013。DOI：10.1109/TCSI.2012.2215780。

- **论文结果：**以拉普拉斯矩阵Schur补分析图约化、等效电阻及拓扑性质。是理解端口等效图和隐藏节点问题的理论基础。
- **假设/边界：**并未保证从端口等效图唯一恢复任意内部物理网络。
- **证据级别：**作者全文。[原始来源](https://arxiv.org/abs/1102.2950)；[DOI入口](https://doi.org/10.1109/TCSI.2012.2215780)。
- **Google Scholar词条：**`"Kron Reduction of Graphs With Applications to Electrical Networks"`。[打开检索](https://scholar.google.com/scholar?q=%22Kron%20Reduction%20of%20Graphs%20With%20Applications%20to%20Electrical%20Networks%22)。
- **IEEE Xplore精确题名词条：**`"Document Title":"Kron Reduction of Graphs With Applications to Electrical Networks"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Document%20Title%22%3A%22Kron%20Reduction%20of%20Graphs%20With%20Applications%20to%20Electrical%20Networks%22)。


#### G2. Learning Topology of the Power Distribution Grid with and without Missing Data

Deepjyoti Deka; Scott Backhaus; Michael Chertkov。European Control Conference (ECC), 313–320, 2016。DOI：10.1109/ECC.2016.7810304。

- **论文结果：**在条件成立时，以节点电压幅值差方差为边权，运行树是许可线路图上的最小生成树；MST阶段复杂度O(m log m)，不含统计量估计。
- **假设/边界：**线性无损潮流、径向、跨节点注入不相关等是关键。缺失数据版本另要求隐藏节点相距大于两跳及额外线路/注入资料。电压差方差不应混同加性电阻距离。
- **证据级别：**作者全文。[原始来源](https://arxiv.org/abs/1603.01650)；[DOI入口](https://doi.org/10.1109/ECC.2016.7810304)。
- **Google Scholar词条：**`"Learning Topology of the Power Distribution Grid with and without Missing Data"`。[打开检索](https://scholar.google.com/scholar?q=%22Learning%20Topology%20of%20the%20Power%20Distribution%20Grid%20with%20and%20without%20Missing%20Data%22)。
- **IEEE Xplore精确题名词条：**`"Document Title":"Learning Topology of the Power Distribution Grid with and without Missing Data"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Document%20Title%22%3A%22Learning%20Topology%20of%20the%20Power%20Distribution%20Grid%20with%20and%20without%20Missing%20Data%22)。

#### G3. Topology Estimation Using Graphical Models in Multi-Phase Power Distribution Grids

Deepjyoti Deka; Michael Chertkov; Scott Backhaus。IEEE Transactions on Power Systems 35(3):1663–1673, 2020；2019年在线发表。DOI：10.1109/TPWRS.2019.2897004。

- **论文结果：**证明电压概率图可包含物理边及二跳关系，以条件独立检验分离这些边，再确定叶节点连接；包含三相AC算例。
- **假设/边界：**需要相应线性化模型、同步复电压和跨节点独立注入等假设；概率图边不能直接解释为物理边。
- **证据级别：**作者全文及高校正式题录。[原始来源](https://arxiv.org/abs/1803.06531)；[DOI入口](https://doi.org/10.1109/TPWRS.2019.2897004)。
- **Google Scholar词条：**`"Topology Estimation Using Graphical Models in Multi-Phase Power Distribution Grids"`。[打开检索](https://scholar.google.com/scholar?q=%22Topology%20Estimation%20Using%20Graphical%20Models%20in%20Multi-Phase%20Power%20Distribution%20Grids%22)。
- **IEEE Xplore精确题名词条：**`"Document Title":"Topology Estimation Using Graphical Models in Multi-Phase Power Distribution Grids"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Document%20Title%22%3A%22Topology%20Estimation%20Using%20Graphical%20Models%20in%20Multi-Phase%20Power%20Distribution%20Grids%22)。

#### G4. Learning With End-Users in Distribution Grids: Topology and Parameter Estimation

Sejun Park; Deepjyoti Deka; Scott Backhaus; Michael Chertkov。IEEE Transactions on Control of Network Systems 7(3):1428–1440, 2020。DOI：10.1109/TCNS.2020.2979882。

- **论文结果：**从叶端电压幅值和P/Q样本估计阻抗距离，恢复潜在树与参数，无需预先给定许可线路或隐藏节点数。定理3给出M>C|V|log(|V|/η)时至少1−η的恢复概率。
- **假设/边界：**正文要求隐藏节点度数≥3，节点P/Q协方差非退化、跨节点注入不相关；样本界另需固定深度、有界阻抗及次高斯条件。摘要greater than three与正文有差别，本报告采用正文Assumption 1。恒功率因数可能使双参数识别方程退化。
- **证据级别：**作者全文。[原始来源](https://arxiv.org/abs/1803.04812)；[DOI入口](https://doi.org/10.1109/TCNS.2020.2979882)。
- **Google Scholar词条：**`"Learning With End-Users in Distribution Grids: Topology and Parameter Estimation"`。[打开检索](https://scholar.google.com/scholar?q=%22Learning%20With%20End-Users%20in%20Distribution%20Grids%3A%20Topology%20and%20Parameter%20Estimation%22)。
- **IEEE Xplore精确题名词条：**`"Document Title":"Learning With End-Users in Distribution Grids: Topology and Parameter Estimation"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Document%20Title%22%3A%22Learning%20With%20End-Users%20in%20Distribution%20Grids%3A%20Topology%20and%20Parameter%20Estimation%22)。

#### G5. Graph Algorithms for Topology Identification Using Power Grid Probing

Guido Cavraro; Vassilis Kekatos。IEEE Control Systems Letters 2(4):689–694, 2018。DOI：10.1109/LCSYS.2018.2846801。

- **论文结果：**全叶端激励且全节点测压可恢复完整树；只在激励节点测压时恢复约化树。IEEE37节点单相等值、10000次Monte Carlo中，全观测每探测节点90次测量时报告0.2%拓扑错误率。
- **假设/边界：**噪声和激励配置是实验结果的组成部分；全树与约化树错误率不应直接比较。探测矩阵需满行秩；部分观测不能声称恢复任意隐藏二度节点。
- **证据级别：**作者全文。[原始来源](https://engineering.purdue.edu/~kekatos/papers/L-CSS-2018.pdf)；[DOI入口](https://doi.org/10.1109/LCSYS.2018.2846801)。
- **Google Scholar词条：**`"Graph Algorithms for Topology Identification Using Power Grid Probing"`。[打开检索](https://scholar.google.com/scholar?q=%22Graph%20Algorithms%20for%20Topology%20Identification%20Using%20Power%20Grid%20Probing%22)。
- **IEEE Xplore精确题名词条：**`"Document Title":"Graph Algorithms for Topology Identification Using Power Grid Probing"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Document%20Title%22%3A%22Graph%20Algorithms%20for%20Topology%20Identification%20Using%20Power%20Grid%20Probing%22)。

#### G6. Inverter Probing for Power Distribution Network Topology Processing

Guido Cavraro; Vassilis Kekatos。IEEE Transactions on Control of Network Systems 6(3):980–992, 2019。DOI：10.1109/TCNS.2019.2901714。

- **论文结果：**使用逆变器注入扰动和电压响应，利用树拉普拉斯进行拓扑恢复与线路状态核验。基准算例中探测约40%节点可获得10^-3量级线路状态错误概率。
- **假设/边界：**40%是算例观测，不是任意40%节点皆可恢复的定理；线路状态错误概率不等于整图正确概率。
- **证据级别：**作者全文。[原始来源](https://arxiv.org/abs/1802.06027)；[DOI入口](https://doi.org/10.1109/TCNS.2019.2901714)。
- **Google Scholar词条：**`"Inverter Probing for Power Distribution Network Topology Processing"`。[打开检索](https://scholar.google.com/scholar?q=%22Inverter%20Probing%20for%20Power%20Distribution%20Network%20Topology%20Processing%22)。
- **IEEE Xplore精确题名词条：**`"Document Title":"Inverter Probing for Power Distribution Network Topology Processing"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Document%20Title%22%3A%22Inverter%20Probing%20for%20Power%20Distribution%20Network%20Topology%20Processing%22)。


#### G7. Graphical Models in Meshed Distribution Grids: Topology Estimation, Change Detection & Limitations

Deepjyoti Deka; Saurav Talukdar; Michael Chertkov; Murti V. Salapaka。IEEE Transactions on Smart Grid 11(5):4299–4310, 2020。DOI：10.1109/TSG.2020.2978541。

- **论文结果：**将概率图识别扩展至网状网络；在最短环长大于三等条件下，用精度矩阵块的符号判断物理边，并讨论噪声容忍和拓扑变化。
- **假设/边界：**纯图算法与符号算法的环长条件不同；不能写成无条件恢复任意网状网络。跨节点注入统计假设仍然关键。
- **证据级别：**作者全文。[原始来源](https://arxiv.org/abs/1905.06550)；[DOI入口](https://doi.org/10.1109/TSG.2020.2978541)。
- **Google Scholar词条：**`"Graphical Models in Meshed Distribution Grids: Topology Estimation, Change Detection & Limitations"`。[打开检索](https://scholar.google.com/scholar?q=%22Graphical%20Models%20in%20Meshed%20Distribution%20Grids%3A%20Topology%20Estimation%2C%20Change%20Detection%20%26%20Limitations%22)。
- **IEEE Xplore精确题名词条：**`"Document Title":"Graphical Models in Meshed Distribution Grids: Topology Estimation, Change Detection & Limitations"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Document%20Title%22%3A%22Graphical%20Models%20in%20Meshed%20Distribution%20Grids%3A%20Topology%20Estimation%2C%20Change%20Detection%20%26%20Limitations%22)。

### 置信度与概率推断

#### U1. A Recursive Bayesian Approach for Identification of Network Configuration Changes in Distribution System State Estimation

Ravindra Singh; Efthymios Manitsas; Bikash C. Pal; Goran Strbac。IEEE Transactions on Power Systems 25(3):1329–1336, 2010。DOI：10.1109/TPWRS.2010.2040294。

- **论文结果：**各候选配置分别运行状态估计，递归更新贝叶斯概率，从预存model bank中识别配置变化；以UK Generic Distribution System部分网络演示。
- **假设/边界：**真实配置必须在模型库中；后验依赖先验和观测模型。不能由候选后验接近1推断开放世界真实拓扑已认证。
- **证据级别：**作者机构摘要及题录；未审阅完整实验表。[原始来源](https://www.imperial.ac.uk/electrical-engineering/research/control-and-power/publications/?id=298820&noscript=noscript&respub-t4-action=citation.html)；[DOI入口](https://doi.org/10.1109/TPWRS.2010.2040294)。
- **Google Scholar词条：**`"A Recursive Bayesian Approach for Identification of Network Configuration Changes in Distribution System State Estimation"`。[打开检索](https://scholar.google.com/scholar?q=%22A%20Recursive%20Bayesian%20Approach%20for%20Identification%20of%20Network%20Configuration%20Changes%20in%20Distribution%20System%20State%20Estimation%22)。
- **IEEE Xplore精确题名词条：**`"Document Title":"A Recursive Bayesian Approach for Identification of Network Configuration Changes in Distribution System State Estimation"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Document%20Title%22%3A%22A%20Recursive%20Bayesian%20Approach%20for%20Identification%20of%20Network%20Configuration%20Changes%20in%20Distribution%20System%20State%20Estimation%22)。

#### U2. An Adaptive-Importance-Sampling-Enhanced Bayesian Approach for Topology Estimation in an Unbalanced Power Distribution System

Yijun Xu; Jaber Valinejad; Mert Korkali; Lamine Mili; Yajun Wang; Xiao Chen; Zongsheng Zheng。IEEE Transactions on Power Systems 37(3):2220–2232, 2022；2021年在线发表。DOI：10.1109/TPWRS.2021.3121612。

- **论文结果：**在三相不平衡网络中联合估计拓扑、停电和状态，自适应重要性采样近似后验；测试IEEE123和不平衡1282节点系统。
- **假设/边界：**依赖候选开关结构和负荷/测量概率模型，采样近似存在误差；部分开关可能仍错判，不能等同分布无关覆盖保证。
- **证据级别：**作者全文及机构正式全文。[原始来源](https://arxiv.org/abs/2110.09030)；[DOI入口](https://doi.org/10.1109/TPWRS.2021.3121612)。
- **Google Scholar词条：**`"An Adaptive-Importance-Sampling-Enhanced Bayesian Approach for Topology Estimation in an Unbalanced Power Distribution System"`。[打开检索](https://scholar.google.com/scholar?q=%22An%20Adaptive-Importance-Sampling-Enhanced%20Bayesian%20Approach%20for%20Topology%20Estimation%20in%20an%20Unbalanced%20Power%20Distribution%20System%22)。
- **IEEE Xplore精确题名词条：**`"Document Title":"An Adaptive-Importance-Sampling-Enhanced Bayesian Approach for Topology Estimation in an Unbalanced Power Distribution System"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Document%20Title%22%3A%22An%20Adaptive-Importance-Sampling-Enhanced%20Bayesian%20Approach%20for%20Topology%20Estimation%20in%20an%20Unbalanced%20Power%20Distribution%20System%22)。

#### U3. Grid Topology Identification via Distributed Statistical Hypothesis Testing

Saverio Bolognani（2018章）；Saverio Bolognani、Keith Moffat（2024新版章）。Big Data Application in Power Systems, 281–301, 2018；新版109–134, 2024。DOI：10.1016/B978-0-12-811968-6.00013-9。

- **论文结果：**用局部三节点电压统计检验识别相对结构。2018章在IEEE123骨架及真实住户负荷下，对六组三节点重复1200次，并报告错误率及Wilson置信区间。
- **假设/边界：**依赖不相关注入及统一X/R或功率因数等条件；性能错误率的置信区间不等于当前拓扑正确概率。2024版DOI为10.1016/B978-0-443-21524-7.00012-8，不能将旧版实验写成新版新增实验。
- **证据级别：**2018作者全文；2024版出版社摘要。[原始来源](https://people.ee.ethz.ch/~bsaverio/papers/bolognani-grididentification.pdf)；[DOI入口](https://doi.org/10.1016/B978-0-12-811968-6.00013-9)。
- **Google Scholar词条：**`"Grid Topology Identification via Distributed Statistical Hypothesis Testing"`。[打开检索](https://scholar.google.com/scholar?q=%22Grid%20Topology%20Identification%20via%20Distributed%20Statistical%20Hypothesis%20Testing%22)。
- **IEEE Xplore相邻主题词条：**`"distribution" AND topology AND "hypothesis testing"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22distribution%22%20AND%20topology%20AND%20%22hypothesis%20testing%22)。非IEEE出版物不保证有Xplore原文条目。

#### U4. Bayesian Error-in-Variables Models for the Identification of Distribution Grids

Jean-Sébastien Brouillon; Emanuele Fabbiani; Pulkit Nahata; Keith Moffat; Florian Dörfler; Giancarlo Ferrari-Trecate。IEEE Transactions on Smart Grid 14(2):1289–1299, 2023。DOI：10.1109/TSG.2022.3211546。

- **论文结果：**同时处理电压与电流量测误差，以EIV/MLE及先验得到MAP导纳估计；将已知参数和稀疏知识纳入估计。
- **假设/边界：**MAP点估计不自动提供拓扑置信集；Fisher信息或CRLB也不直接保证离散线路正确。预印本末词Power Networks，正式题名末词Distribution Grids；应按版本检索。
- **证据级别：**作者全文及ETH正式题录。[原始来源](https://arxiv.org/abs/2107.04480)；[DOI入口](https://doi.org/10.1109/TSG.2022.3211546)。
- **Google Scholar词条：**`"Bayesian Error-in-Variables Models for the Identification of Distribution Grids"`。[打开检索](https://scholar.google.com/scholar?q=%22Bayesian%20Error-in-Variables%20Models%20for%20the%20Identification%20of%20Distribution%20Grids%22)。
- **IEEE Xplore精确题名词条：**`"Document Title":"Bayesian Error-in-Variables Models for the Identification of Distribution Grids"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Document%20Title%22%3A%22Bayesian%20Error-in-Variables%20Models%20for%20the%20Identification%20of%20Distribution%20Grids%22)。


#### U5. Error Bounds for Radial Network Topology Learning from Quantized Measurements

Samuel Talkington; Aditya Rangarajan; Pedro A. de Alcântara; Line Roald; Daniel K. Molzahn; Daniel R. Fuhrmann。arXiv:2508.05620, 2025；另核到作者2026稿，正式出版信息本报告未完全核准。

- **论文结果：**提出非渐近参数误差界，形式为O(Δ sqrt(log n/s))，讨论至少0.99概率以及对数级样本要求。Δ为量化步长，s为每节点样本数。
- **假设/边界：**已知固定功率因数、线性模型、独立次高斯测量设计和随机dither量化等条件不可省略。参数L2界转化为精确边恢复仍需最小边权间隔；算例曲线常数的事后拟合不能直接变成部署认证。
- **证据级别：**作者预印本及2026作者全文。[原始来源](https://arxiv.org/abs/2508.05620)。
- **Google Scholar词条：**`"Error Bounds for Radial Network Topology Learning from Quantized Measurements"`。[打开检索](https://scholar.google.com/scholar?q=%22Error%20Bounds%20for%20Radial%20Network%20Topology%20Learning%20from%20Quantized%20Measurements%22)。
- **IEEE Xplore相邻主题词条：**`topology AND "quantized measurements"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=topology%20AND%20%22quantized%20measurements%22)。非IEEE出版物不保证有Xplore原文条目。

#### U6. Confidence-Aware Topology Identification in Low-Voltage Distribution Networks: A Multi-Source Fusion Method Based on Weakly Supervised Learning

Siliang Liu; Can Deng; Zenan Zheng; Ying Zhu; Hongxin Lu; Wenze Liu。Energies 19(6):1503, 2026。DOI：10.3390/en19061503。

- **论文结果：**用弱监督标签模型和Dempster–Shafer融合，为每个电表给出置信评分。实际案例包括4条低压分支、127电表、6天15 min间隔数据。
- **假设/边界：**案例以用户相别连接标签为主，不是全物理线路重建；证据融合评分不是独立真值校准后的有限样本覆盖定理。
- **证据级别：**出版社正文及索引可见全文段落。[原始来源](https://www.mdpi.com/1996-1073/19/6/1503)；[DOI入口](https://doi.org/10.3390/en19061503)。
- **Google Scholar词条：**`"Confidence-Aware Topology Identification in Low-Voltage Distribution Networks: A Multi-Source Fusion Method Based on Weakly Supervised Learning"`。[打开检索](https://scholar.google.com/scholar?q=%22Confidence-Aware%20Topology%20Identification%20in%20Low-Voltage%20Distribution%20Networks%3A%20A%20Multi-Source%20Fusion%20Method%20Based%20on%20Weakly%20Supervised%20Learning%22)。
- **IEEE Xplore相邻主题词条：**`topology AND confidence AND "weakly supervised"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=topology%20AND%20confidence%20AND%20%22weakly%20supervised%22)。非IEEE出版物不保证有Xplore原文条目。

#### U7. Hybrid signal-based phase identification and probabilistic topology inference for low-voltage distribution networks

Kewen Pei; Zhuoheng Wang; Qinglong Liao; Yongfu Li; Shanshan Kuang; Hong Xiang; Qiushi Cui。Electric Power Systems Research 254:112591, 2026。DOI：10.1016/j.epsr.2025.112591。

- **论文结果：**融合电压相似性和有功波动，先进行相别识别，再在图约束下自底向上估计连接概率；用10节点仿真及重庆实际台区数据验证。
- **假设/边界：**Hybrid signal在此是电压/功率信息融合，不是人工特征电流注入；本次可见内容未确立有限样本置信覆盖定理。
- **证据级别：**出版社摘要及可见正文。[原始来源](https://www.sciencedirect.com/science/article/pii/S0378779625011782)；[DOI入口](https://doi.org/10.1016/j.epsr.2025.112591)。
- **Google Scholar词条：**`"Hybrid signal-based phase identification and probabilistic topology inference for low-voltage distribution networks"`。[打开检索](https://scholar.google.com/scholar?q=%22Hybrid%20signal-based%20phase%20identification%20and%20probabilistic%20topology%20inference%20for%20low-voltage%20distribution%20networks%22)。
- **IEEE Xplore相邻主题词条：**`"distribution network" AND topology AND probabilistic`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22distribution%20network%22%20AND%20topology%20AND%20probabilistic)。非IEEE出版物不保证有Xplore原文条目。

### 共形预测基础与待取得全文线索

#### P1. Distribution Network Topology Identification Based on CNN and Improved Conformal Prediction

Bian Ziyue; Long Huan; Zhao Jingtao; Zheng Shu; Zhang Xiaoyan（按会议目录顺序）。2024 China International Conference on Electricity Distribution (CICED), 587–592。DOI：10.1109/CICED63421.2024.10753756。

- **论文结果：**确认存在将CNN和改进共形预测用于配电拓扑辨识的专门会议论文。
- **假设/边界：**未核覆盖定理、校准单位或实验数字；不能据题名宣布其具有某一保证。
- **证据级别：**IEEE题名及会议目录；全文未取得。[原始来源](https://www.proceedings.com/content/077/077452webtoc.pdf)；[DOI入口](https://doi.org/10.1109/CICED63421.2024.10753756)。
- **Google Scholar词条：**`"Distribution Network Topology Identification Based on CNN and Improved Conformal Prediction"`。[打开检索](https://scholar.google.com/scholar?q=%22Distribution%20Network%20Topology%20Identification%20Based%20on%20CNN%20and%20Improved%20Conformal%20Prediction%22)。
- **IEEE Xplore精确题名词条：**`"Document Title":"Distribution Network Topology Identification Based on CNN and Improved Conformal Prediction"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Document%20Title%22%3A%22Distribution%20Network%20Topology%20Identification%20Based%20on%20CNN%20and%20Improved%20Conformal%20Prediction%22)。

#### P2. Distribution Network Topology Identification Based on Co-Training of ResNet and Conformal Prediction

本轮未在primary页面核准作者，暂不填写。2025 8th International Conference on Energy, Electrical and Power Engineering (CEEPE)。DOI：10.1109/CEEPE64987.2025.11033908。

- **论文结果：**确认存在将ResNet协同训练与共形预测用于该任务的后续论文。
- **假设/边界：**不能把通用共形预测保证当成该论文已严格验证的结论；应取得全文审查数据划分与最终预测集合。
- **证据级别：**IEEE题录；全文未取得。[原始来源](https://ieeexplore.ieee.org/document/11033908)；[DOI入口](https://doi.org/10.1109/CEEPE64987.2025.11033908)。
- **Google Scholar词条：**`"Distribution Network Topology Identification Based on Co-Training of ResNet and Conformal Prediction"`。[打开检索](https://scholar.google.com/scholar?q=%22Distribution%20Network%20Topology%20Identification%20Based%20on%20Co-Training%20of%20ResNet%20and%20Conformal%20Prediction%22)。
- **IEEE Xplore精确题名词条：**`"Document Title":"Distribution Network Topology Identification Based on Co-Training of ResNet and Conformal Prediction"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Document%20Title%22%3A%22Distribution%20Network%20Topology%20Identification%20Based%20on%20Co-Training%20of%20ResNet%20and%20Conformal%20Prediction%22)。


#### P3. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification

Anastasios N. Angelopoulos; Stephen Bates。arXiv:2107.07511, 2021起（修订稿）。

- **论文结果：**系统解释共形预测集合及交换性下的边际覆盖保证，可作为拓扑标签集合设计的统计基础。
- **假设/边界：**不是配电网专门结果；本报告第5.2节是该框架向拓扑任务的应用解释，不能替代对P1/P2的全文审核。
- **证据级别：**作者全文。[原始来源](https://arxiv.org/abs/2107.07511)。
- **Google Scholar词条：**`"A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"`。[打开检索](https://scholar.google.com/scholar?q=%22A%20Gentle%20Introduction%20to%20Conformal%20Prediction%20and%20Distribution-Free%20Uncertainty%20Quantification%22)。
- **IEEE Xplore相邻主题词条：**`"conformal prediction" AND "uncertainty quantification"`。[打开检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22conformal%20prediction%22%20AND%20%22uncertainty%20quantification%22)。非IEEE出版物不保证有Xplore原文条目。

## 8. 图论与置信度的直接交叉

G4的有限样本定理给出规定模型下的恢复概率，并不等于每个已输出拓扑的条件正确概率。G5把探测幅值、测量次数、最小电阻和噪声水平联系起来，给出保持层集合分组稳定的条件，再用概率界控制整体错误。U5则提供量化误差下的参数界，但要保证每条边正确，还需边权间隔。因此可辨识性、参数估计精度、离散图恢复概率和个例置信评分应分别报告。

第3节KCL编码模型为综合推导；第4节拉普拉斯/Kron内容的直接来源为G1，树距离及隐节点条件见G4，二跳概率图见G3。第5.2节的共形保证来自P3的通用框架向拓扑标签的应用解释，并非对P1/P2未取得全文部分的结果转述。

补充：单边检验的p值也不是该边错误的概率；从多个单边检验构建整图保证，需要处理多重比较及图约束。bootstrap在本报告中只是建议的稳定性工具，没有将其列成已证明的完整拓扑置信集方法。

## 9. 适合继续深入的研究问题

以下是文献基础上的研究建议，不是已经确认无人研究的空白。

1. **特征电流漏检/误检下的树集合恢复。** 为FSR建立发射和接收错误模型，把层叠集合、连通和径向约束纳入推断；若多个拓扑不可区分则返回集合。分别评估信号检出、单边判断、整树恢复，避免把解码置信度直接当拓扑置信度。C4与G5是直接基础。
2. **布点与激励的可辨识性设计。** 在次数、幅值和接收设备数量受限时，增大最难区分的两个合法拓扑之间的响应间隔；先处理零间隔的结构不可辨识，再研究噪声下的样本需求。G4—G6是理论基础。
3. **工程融合评分的外部校准。** 对U6一类评分，用独立台区/事件真值检验其概率含义；共形方法还需明确候选覆盖、交换性及自适应探测后的校准。P1/P2已有直接应用研究，创新性须精读后判断。

## 10. 阅读顺序与审查重点

建议顺序：R1全景 → C2工程特征电流 → C4缺失记录和集合结构 → G5主动探测保证 → G4末端隐节点与样本界 → U2贝叶斯后验 → U6工程置信评分 → P1/P2全文审核。

逐篇核查辨识对象、候选线路是否已知、隐藏节点的负荷假设、噪声模型、仿真/实验室/现场证据、指标的分母，以及完全缺失与不可辨识样本是否被剔除。本次没有复现论文实验；C5、P1、P2明确未取得全文，U5正式出版信息未仅凭二级索引补全。已保留在线年、正式卷期年及预印本题名差异。

