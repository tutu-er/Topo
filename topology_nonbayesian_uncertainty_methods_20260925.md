# 非贝叶斯拓扑辨识如何处理指标、置信度与不确定性

日期：2026-09-25。本文回答：是否只需寻找“指标→概率”映射，以及其他非贝叶斯拓扑辨识方法实际上如何处理该问题。

**结论：如果目标是输出单条线路存在概率，确实需要概率模型或统计校准；如果目标是提供可靠性保证，则还可以校准检验阈值、构造置信集合、证明恢复概率，完全不必给每条边一个概率。非贝叶斯不等于不使用概率。**

本文为原文核查与理论对照，未实现新的辨识或校准实验。

## 1. 先选概率对象，再选校准方式

设 B_e 为真线路状态，S_e=J_e^(0)−J_e^(1) 是用户的带方向影响指标。

| 输出目标 | 数学对象 | 能回答的问题 |
|---|---|---|
| 跨案例的条件存在概率 | g(s)=P(B_e=1|S_e=s) | 在目标案例总体中，出现这种分数时有多少线路为真 |
| Bayesian 后验边概率 | P(B_e=1|D,先验,模型) | 给定模型、先验与这批数据后的状态不确定性 |
| 边检验 | P_0(S_e>cα)≤α | 无边时有多大概率被错误判为有边 |
| 固定真图的置信集合 | P_D(T*∈Cα(D))≥1−α | 重复采样时集合能否覆盖固定真图 |
| 算法恢复保证 | P_D(That(D)=T*)≥1−δ | 指定模型/样本量下算法成功的频率 |
| 重采样稳定性 | P_*(e∈That(D*)) | 扰动当前样本后算法多常选这条边 |

这些量不能互换。尤其不能把1−p值、测试准确率或bootstrap出现频率直接当作当前边的后验。

上轮写出的贝叶斯公式也是普通条件概率恒等式，并不要求采用 Bayesian 参数推断。频率派预测学习同样可以在有真值的案例总体中估计 g(s)，其中基准存在率可以是总体比例。关键是明示参考总体和部署条件。对同一固定未知图，典型频率派分析把 T* 视为固定未知对象，概率主要针对重复抽样的数据。

## 2. 代表性电网论文实际上怎样做

### 2.1 残差最小化，再报告实验准确率

R1，Srivastava 等 (2022)，An Optimization-Based Topology Error Detection Method for Power System State Estimation。

式(23)–(25)比较候选线路退出后的重估残差并选择最优状态；使用矩阵逆引理降低计算。其主要推导为P–δ解耦线性化，假定单个拓扑错误，候选排除临界支路。

表8中，IEEE14节点指定的6个退出事件辨对6个；118节点的50个事件辨对47个。94%表示有限测试事件的辨对比例，不是某次选中线路有94%概率存在。

**与用户问题的关系：**原始残差已经用于选择，但论文没有建立“任意残差差值→当前边存在概率”的映射。这一类点估计方法本身没有完整解决单实例不确定性。

### 2.2 最大似然检测与 Monte Carlo 线路错误统计

R2，Cavraro–Kekatos (2019)，Inverter Probing for Power Distribution Network Topology Processing。

原文§IV式(30)称 maximum-likelihood detector：在已知候选线路及电阻条件下，寻找满足径向/连通约束、使探测响应加权残差最小的二元线路状态。不能把它未经核实称为GLRT。

其IEEE37节点数值部分开展200次Monte Carlo。表I报告每次实验平均误判线路数，包含漏报和误报；这是重复实验的性能量。摘要里的10^-3量级线路状态错误概率，也不能解释成单次输出边的后验0.999。近似网络模型、量测/负荷变化假设与凸松弛恢复过程均影响结论。

**与用户问题的关系：**似然给评分以观测模型解释，重复实验给算法误差频率；这两步仍不自动构成每条边的概率校准。

### 2.3 明确定义 confidence 为最差类别识别正确率

R3，Sharon 等 (2012)，Topology Identification in Distribution Network with Limited Measurements。

给每个候选拓扑建立近似高斯量测分布，采用最大似然分类器：

$$
c(y)=\arg\max_i p_i(y).
$$

其式(1)定义

$$
{\rm confidence}
=
1-\max_i P_i\{c(Y)\ne i\}.
$$

即：假设第i个拓扑为真，重复生成量测统计误判率，再取最差类别正确率。已知物理候选网、线路参数及负荷均值/协方差是重要前提。该定义是识别系统在模型下的事前可靠性，而非P(T_i|当前y)。

另一个细节：原文最大似然选择的动机是最小化类别错误率之和，不能据此宣称它总是直接最小化最坏类别错误率。

**与用户问题的关系：**可以对自己的整个评分—决策流程做“不同真状态下有多大概率判错”的评估，而不先输出后验概率。

### 2.4 通过样本复杂度证明高概率恢复

R4，Park、Deka、Backhaus、Chertkov (2020)，Learning With End-Users in Distribution Grids: Topology and Parameter Estimation。

论文研究终端量测、隐藏中间节点的树恢复。Theorem3给出在线性耦合潮流及有界深度等条件下，样本数

$$
m>C|V|\log(|V|/\eta)
$$

时，存在合适阈值使Algorithm3以至少1−η的概率恢复真拓扑。条件涉及隐藏节点度数、正且有界的线路参数、注入的尾部与协方差非退化等。

这是从统计量集中性到算法恢复事件的概率保证，不是给已观测分数逐点配概率。原文IV-C还说明，非线性AC失配下即使样本无限增大，也不必使估计距离收敛到其理论树距离。

**复用时的原文审查提示：**主文的跨节点二阶不相关表述不足以原样解释所有附录概率乘积分解；附录还使用节点注入独立性及独立样本。若在自己的模型中迁移该证明，宜明确采取更强的跨节点注入向量独立、时间样本独立条件，或补齐适合相关数据的新证明。不能把该样本量形式当一般AC系统的通用保证。

## 3. 可以采用的其他非贝叶斯推断路线

下面是可迁移的统计理论，不表示上述四篇电网论文均已采用。

### 3.1 对影响指标校准零假设阈值

检验 H0:e不存在，以大 S_e 支持存在。为覆盖线路参数、负荷等复合零假设，理想要求

$$
\sup_{\eta\in H_0}P_\eta(S_e>c_{e,\alpha})\le\alpha.
$$

S_e超过阈值时排除无边状态。这里不需要先验存在率，但需要正确处理零假设下的数据分布、nuisance与数据驱动选择。参数bootstrap可以估计参考分布，但单一plug-in模型下的模拟通常是近似，不能自动替代上述全域上界。

对于未知状态，分别检验B_e=0和B_e=1，可返回{0}、{1}、{0,1}或空集；空集表示两个模型均被排除，需要检查模型/数据，而不是同时断言有边与无边。多条边同时发结论，需处理多重检验，或使用共同置信集合给同时保证。

### 3.2 反演检验得到图置信集合

对各候选T构造在其所有允许参数下有效的p值p_T，定义

$$
C_\alpha(D)=\{T:p_T(D)>\alpha\}.
$$

若真T属于完整模型族，则P(T*∈Cα)≥1−α。它允许保留多个无法区分的拓扑。全体相容图共同包含的边可被确认，其余边保持不确定。

这一固定真参数的置信集合，不需要每个T有后验概率。候选遗漏真图、局部AC求解被误认为全局排除等仍会破坏结论。只测试少量候选时应标明集合覆盖范围。

### 3.3 稳定选择

R5，Meinshausen–Bühlmann (2010)，Stability Selection。

重复子采样、运行结构选择，计算入选频率；在其噪声变量交换性等条件下，可以控制误选数量的期望。它不是普通bootstrap频率自动升级为正确概率，也不是默认控制FDR。原文研究变量选择及Gaussian图模型，物理电网的树边依赖、候选筛选等需要重新核对假设。

### 3.4 共形预测

R6，Shafer–Vovk (2008)，A Tutorial on Conformal Prediction。

在独立同分布/适当可交换的有标签案例上，使用校准分数输出新案例的标签集合；可把拓扑配置或边状态作为标签，得到边际覆盖保证。它可以接入非贝叶斯分类器。

但新网络案例的标签预测集合，与同一固定未知电网的结构参数置信集合不同。把同一电网的相关时间点、相关边或bootstrap副本当作独立标签案例，不会自动得到覆盖保证；共形p值也不是标签后验概率。

### 3.5 非贝叶斯概率校准

R7，Niculescu-Mizil–Caruana (2005)，Predicting Good Probabilities With Supervised Learning。

利用已知真值案例学习g(S)，例如Platt或isotonic，输出指定参考总体下的条件概率。这并不要求Bayesian参数先验。模型是否校准，应在独立测试情景中检查；只有总体校准还不等于对所有噪声/工况子群均校准。

## 4. 对用户方案的建议

核心研究对象最好写成：

> 给定一条线路对全局AC拟合误差的影响统计量，如何构造边状态的有效证据与决策规则，并校准其错误率或条件存在概率？

这比仅说“找一个指标到概率的函数”更明确，因为它同时限定输出对象和保证。

建议把指标本身和推断层分开：

1. 固定评分流程S_e：明确含边/无边时哪些参数与其他边允许重拟合、合法径向换边以及求解精度。
2. 若研究单张未知网且有可信量测误差模型，先考虑边状态检验：构建无边/有边下的分数分布，校准排除阈值。
3. 若希望保留结构歧义，反演为边状态或拓扑集合。
4. 若有代表性真值网络案例且需要逐边百分比，再校准g(S_e)，并说明参考总体、基准率和部署条件。
5. 可同时报告S_e、p值/状态集合、校准概率，但它们是不同字段，不互相替代。

最小应报告四项实验量：
- 误报率：真实无边时错误确认有边的频率。
- 漏报率/检出率：真实有边时能否正确检出。
- 未分辨率：有多少情景仍须保留两种状态。
- 若输出概率，额外报告可靠性图及Brier/log loss。

这次未实现或验证上述程序，也未认定该组合本身是新的论文贡献；原文对照说明已有工作分别处理了其中的不同部分。

## 5. 原始文献与检索入口

以下R1–R4为实际电网方法；R5–R7是可迁移的通用统计/学习理论。非IEEE论文的Xplore查询不表示其全文被收录。

### R1. An Optimization-Based Topology Error Detection Method for Power System State Estimation

A. Srivastava; S. Chakrabarti; J. Soares; S. N. Singh. Electric Power Systems Research 209:107914, 2022. DOI: 10.1016/j.epsr.2022.107914。 [原文](https://research.chalmers.se/publication/529376/file/529376_Fulltext.pdf)。

- Scholar：`"An Optimization-Based Topology Error Detection Method for Power System State Estimation"`。[检索](https://scholar.google.com/scholar?q=%22An%20Optimization-Based%20Topology%20Error%20Detection%20Method%20for%20Power%20System%20State%20Estimation%22)。
- IEEE Xplore：`"An Optimization-Based Topology Error Detection Method for Power System State Estimation"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22An%20Optimization-Based%20Topology%20Error%20Detection%20Method%20for%20Power%20System%20State%20Estimation%22)。

### R2. Inverter Probing for Power Distribution Network Topology Processing

G. Cavraro; V. Kekatos. IEEE Transactions on Control of Network Systems 6(3):980–992, 2019. DOI: 10.1109/TCNS.2019.2901714。 [原文](https://engineering.purdue.edu/~kekatos/papers/TCONES2018.pdf)。

- Scholar：`"Inverter Probing for Power Distribution Network Topology Processing"`。[检索](https://scholar.google.com/scholar?q=%22Inverter%20Probing%20for%20Power%20Distribution%20Network%20Topology%20Processing%22)。
- IEEE Xplore：`"Inverter Probing for Power Distribution Network Topology Processing"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Inverter%20Probing%20for%20Power%20Distribution%20Network%20Topology%20Processing%22)。

### R3. Topology Identification in Distribution Network with Limited Measurements

Y. Sharon; A. M. Annaswamy; A. L. Motto; A. Chakraborty. IEEE PES ISGT, 2012. DOI: 10.1109/ISGT.2012.6175638。 [原文](https://yoav.sharon.pro/papers/isgt12_topology.pdf)。

- Scholar：`"Topology Identification in Distribution Network with Limited Measurements"`。[检索](https://scholar.google.com/scholar?q=%22Topology%20Identification%20in%20Distribution%20Network%20with%20Limited%20Measurements%22)。
- IEEE Xplore：`"Topology Identification in Distribution Network with Limited Measurements"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Topology%20Identification%20in%20Distribution%20Network%20with%20Limited%20Measurements%22)。

### R4. Learning With End-Users in Distribution Grids: Topology and Parameter Estimation

S. Park; D. Deka; S. Backhaus; M. Chertkov. IEEE Transactions on Control of Network Systems 7(3):1428–1440, 2020. DOI: 10.1109/TCNS.2020.2979882。 [原文](https://arxiv.org/pdf/1803.04812)。

- Scholar：`"Learning With End-Users in Distribution Grids: Topology and Parameter Estimation"`。[检索](https://scholar.google.com/scholar?q=%22Learning%20With%20End-Users%20in%20Distribution%20Grids%3A%20Topology%20and%20Parameter%20Estimation%22)。
- IEEE Xplore：`"Learning With End-Users in Distribution Grids: Topology and Parameter Estimation"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Learning%20With%20End-Users%20in%20Distribution%20Grids%3A%20Topology%20and%20Parameter%20Estimation%22)。

### R5. Stability Selection

N. Meinshausen; P. Bühlmann. JRSS B 72(4):417–473, 2010. DOI: 10.1111/j.1467-9868.2010.00740.x。 [原文](https://arxiv.org/abs/0809.2932)。

- Scholar：`"Stability Selection"`。[检索](https://scholar.google.com/scholar?q=%22Stability%20Selection%22)。
- IEEE Xplore：`"Stability Selection"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Stability%20Selection%22)。

### R6. A Tutorial on Conformal Prediction

G. Shafer; V. Vovk. JMLR 9:371–421, 2008. [原文](https://jmlr.org/papers/v9/shafer08a.html)。

- Scholar：`"A Tutorial on Conformal Prediction"`。[检索](https://scholar.google.com/scholar?q=%22A%20Tutorial%20on%20Conformal%20Prediction%22)。
- IEEE Xplore：`"A Tutorial on Conformal Prediction"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22A%20Tutorial%20on%20Conformal%20Prediction%22)。

### R7. Predicting Good Probabilities With Supervised Learning

A. Niculescu-Mizil; R. Caruana. ICML, 2005. DOI: 10.1145/1102351.1102430。 [原文](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)。

- Scholar：`"Predicting Good Probabilities With Supervised Learning"`。[检索](https://scholar.google.com/scholar?q=%22Predicting%20Good%20Probabilities%20With%20Supervised%20Learning%22)。
- IEEE Xplore：`"Predicting Good Probabilities With Supervised Learning"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Predicting%20Good%20Probabilities%20With%20Supervised%20Learning%22)。

主题检索：Scholar用 `"distribution grid" "topology" "sample complexity"`、`"topology error" "hypothesis testing"`、`"line status" "maximum likelihood"`、`"topology" "confidence set"`。IEEE Advanced Search可用 `("All Metadata":"topology") AND ("All Metadata":"sample complexity")`，以及 `("All Metadata":"topology error") AND ("All Metadata":"hypothesis testing")`。
