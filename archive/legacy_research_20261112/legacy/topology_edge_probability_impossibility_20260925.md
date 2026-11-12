# 边存在概率的可辨识性与不可能性：针对 AC 拓扑辨识的理论补充

日期：2026-09-25

本文回答：噪声是否可能使某条边的存在状态无法可靠推断？拓扑的离散性是否意味着不能引入概率？既有增删线 AC 拟合差指标，是否必然能转化为边存在概率？

结论：有严格的不可能性理论，但它们否定的是“无额外假设仍能获得有信息、统一可靠的边判断或概率推断”，不否定概率的定义。最直接的论证工具是观测不可辨识、Le Cam 两点检验下界、弱信号下精确支撑恢复的统一不可能性，以及无分布假设的条件概率推断下界。

本文将原论文结论、标准统计事实的直接证明，以及对 AC 模型的条件迁移分开标注。没有进行电网数值实验，也没有证明用户当前具体数据必然不可辨识。

## 1. 首先明确“某条边的概率”是什么

设固定但未知的物理拓扑为 G，目标边状态为 b_e(G)=1{e∈E(G)}。

同一张固定图的量测噪声，不会令物理边在每次量测中随机出现、消失。由噪声直接得到的概率对象是：

\[
P_{G,\theta}\{\widehat b_e(D)\ne b_e(G)\},
\]

即重复观测下的判断错误率。

要讨论 P(b_e=1|D)，可以采用：

1. Bayesian 模型：对未知图及必要的参数赋予先验，讨论认识上的不确定性；
2. 随机网络/情景总体：不同样本具有真实边标签，定义某一总体中的条件发生率；
3. 对前两者给出概率区间或敏感性范围。

这些定义均合法，但依赖的模型、总体与信息不同。频率学派的 95% 置信覆盖、bootstrap 选边频率、候选图归一化权重，不能自动替代它们。

二元状态本身不是障碍。以两种已知正态观测模型为例，状态虽只有 0/1，后验概率却可以是观测值的平滑函数。反之，连续松弛得到的“边权 0.6”也不会因此成为概率 0.6。

## 2. 不可辨识：最强、最直接的不可能性

### 2.1 定义与证明

设两个允许的物理模型分别有边、无边，整份观测 D 的分布为 P_1、P_0。若

\[
P_0=P_1=P_\star,
\]

则任何可测指标 S(D) 在两个世界中具有相同分布。

对任意判别器 φ(D)∈{0,1}，令 A={φ=1}，立即得到：

\[
P_0(\phi=1)+P_1(\phi=0)
=P_\star(A)+P_\star(A^c)=1.
\]

至少有一个世界的错误率不低于 1/2。即使计算能力无限，也不能消除这项信息缺失。

更直接地考虑“概率是否由观测确定”：任取 π∈[0,1]，构造

\[
B\sim{\rm Bernoulli}(\pi),\qquad
D\mid B=0\sim P_\star,\qquad D\mid B=1\sim P_\star.
\]

所有 π 都产生相同的观测边缘分布 P_D=P_\star，但

\[
P(B=1\mid D)=\pi.
\]

因此，仅从 D 的分布不能识别该条件概率。它不是观测分布唯一决定的量。这是“数据不能决定概率”的精确例子，而不是“概率在数学上不存在”。

对于这两个完全指定、观测等价的 Bayesian 假设，Bayes factor 为 1，后验赔率等于先验赔率。不能未经论证把此结论扩大到任意复合参数模型的边际似然；不同参数先验可能产生不同预测分布。

### 2.2 AC 反例：没有激励

【由 AC 方程直接构造，不作为某篇论文原始定理】

在同一节点集 {0,1,2} 上取两棵树：

\[
T_0=\{01,02\},\qquad T_1=\{01,12\}.
\]

线路为正常串联阻抗，忽略并联支路。令所有复电压相同：

\[
V_0=V_1=V_2=v_\star e^{j\theta_\star}.
\]

于是所有线路电流为零，所有节点 P=Q=0，电压幅值均为 v_\star。两图的完整节点 PQV 相同，而边 02 的真值不同。加入相同噪声机制后，观测分布仍然相同。

这个反例说明：缺乏激励时，更多重复样本未必带来拓扑信息。它不说明一般有负荷电网都不可恢复。

### 2.3 AC 反例：隐藏零注入二度节点

取无并联支路、隐藏节点 h 零注入的电路：

\[
a \xrightarrow{z_1} h \xrightarrow{z_2} b.
\]

端口 a、b 的电压电流关系为：

\[
I_a=\frac{V_a-V_b}{z_1+z_2},\qquad I_b=-I_a.
\]

这与单条串联阻抗 z_1+z_2 的线路完全相同。因此任意端口激励下，端口复电压、电流以及 PQV 都无法区分两种内部结构。若要求节点数相同，可以在直接连接 a-b 的图上添加连接 a-h 的零注入隐藏悬挂支路；端口等价仍成立。

Yuan 等《Inverse Power Flow Problem》的 §IV Example 2 给出了 Kron 约化下不同内部网络对应同一端口导纳的例子，并讨论隐藏节点度数条件。[原文](https://arxiv.org/pdf/1610.06631)

这里真正可恢复的对象可能是约化图、端口等价类或串联等效参数。仅在外部端口注入特征电流，不能打破此严格等价；需要能够改变等价关系的内部量测、内部激励或额外物理约束。

注意：该反例甚至允许端口相角量测。不能把“缺少相角”单独写成普遍不可恢复定理。

## 3. 近似不可辨识：Le Cam 两点检验界

【标准两点检验事实；以下给出直接证明。参考 Yu (1997) 的统计下界综述。】

定义整份数据分布的总变差距离：

\[
\delta={\rm TV}(P_0,P_1)=\sup_A|P_0(A)-P_1(A)|.
\]

对于任意 φ，仍令 A={φ=1}：

\[
\begin{aligned}
P_0(\phi=1)+P_1(\phi=0)
&=1-\{P_1(A)-P_0(A)\}\\
&\ge 1-\delta.
\end{aligned}
\]

故等权平均错误率，以及最坏状态错误率，至少为：

\[
\boxed{\frac{1-\delta}{2}.}
\]

不能说“两个状态下各自的错误率都至少为该值”：永远报 0 的程序，在状态 0 下可以零错误，却在状态 1 下必错。

两分布完全相同，界为 1/2；相距很近，最坏错误便接近 1/2。它限制所有算法，而不只是某种优化方法。

对于指标 S(D)，数据处理关系给出：

\[
{\rm TV}(P_0^S,P_1^S)\le {\rm TV}(P_0,P_1).
\]

因此，AC 最优残差差、sigmoid、归一化或其他固定后处理，都不能凭空增加区分两图的信息。若另有独立、有标签的外部校准数据，那是新增信息；需把它及相应模型放进分析，不能当作“只换一个映射函数”。

还可以把结论写成连续输出形式。对任意 q(D)∈[0,1]，总变差控制有界函数期望差，因此：

\[
E_0q(D)+E_1[1-q(D)]\ge1-\delta.
\]

至少一个世界里，q 对真实 0/1 状态的平均绝对误差不小于 (1−δ)/2。这只否定统一尖锐的输出，不否定诚实表达模糊性的 q=0.5。

参考：[Yu, Assouad, Fano, and Le Cam](https://link.springer.com/chapter/10.1007/978-1-4612-1880-7_29)。

## 4. 高斯量测噪声下，可以直接计算信息下界

【两简单正态假设的直接计算；迁移到 AC 时须核实观测模型】

假定全部量测堆叠后：

\[
D\mid b=0\sim N(\mu_0,\Sigma),\qquad
D\mid b=1\sim N(\mu_1,\Sigma),
\]

其中两均值已知、协方差 Σ 已知且正定。定义噪声归一化分离度：

\[
\Delta^2=(\mu_1-\mu_0)^\top\Sigma^{-1}(\mu_1-\mu_0).
\]

令

\[
L=(\mu_1-\mu_0)^\top\Sigma^{-1}
\left(D-\frac{\mu_0+\mu_1}{2}\right).
\]

在状态 0 下 L∼N(−Δ²/2,Δ²)，在状态 1 下 L∼N(Δ²/2,Δ²)。对称似然比检验阈值为零，两个条件错误率相同，均为：

\[
\boxed{R_\star=\Phi(-\Delta/2).}
\]

它既是等先验最优平均错误，也是该简单二元问题的最优最坏错误。其他任何方法均不能使两种状态的错误同时低于它。

| Δ | 最优错误率 | 最优正确率 |
|---:|---:|---:|
| 0 | 50.00% | 50.00% |
| 0.2 | 46.02% | 53.98% |
| 1 | 30.85% | 69.15% |
| 2 | 15.87% | 84.13% |
| 3.290 | 约 5% | 约 95% |

独立标量重复观测、每次响应差 a、噪声标准差 σ 时：

\[
\Delta=\sqrt n\,|a|/\sigma.
\]

如果每次差异仅为噪声标准差的 2%，100 次独立观测只得到 Δ=0.2，即便所有参数预先已知，最优正确率也仅约 54%。这只是理想化说明，不是当前电网数据的实测结果。相关噪声必须用完整 Σ，不能直接按 √n 累积信息。

这里仍能定义平滑后验。例如 π=P(b=1)，则

\[
P(b=1\mid D)
={\rm logistic}\left[
\log\frac{\pi}{1-\pi}+L
\right].
\]

因此“状态离散，概率不能平滑介入”并不成立。在 Δ 很小时，后验通常难以远离先验；这反映信息不足。

## 5. 离散支撑为何难：缺少最小信号时不能统一恢复

【以下为独立标量反例；与稀疏恢复文献的 beta-min 思路相同】

设独立量测：

\[
Y_i\sim N(\theta,\sigma^2),\qquad b(\theta)=1\{\theta\ne0\}.
\]

比较 θ=0 与 θ=a_n>0。由上一节：

\[
\max\{P_0(\widehat b=1),P_{a_n}(\widehat b=0)\}
\ge\Phi\left(-\frac{\sqrt n\,a_n}{2\sigma}\right).
\]

若模型族允许 a_n=o(n^{-1/2})，右侧趋于 1/2。因此，在允许任意弱非零作用的模型族上，不存在统一一致的精确支撑恢复器。

但对固定 a>0，误差可以随 n 增加而趋零。甚至阈值估计器

\[
\widehat b_n=1\{|\overline Y|>\sigma n^{-1/4}\}
\]

对每个固定 θ 均一致，包括 θ=0。必须区分“逐点可恢复”与“对整个参数族统一可恢复”。

Wainwright (2009) 在带噪高维线性模型中研究任意算法的支撑恢复界，最小非零系数是关键条件。Santhanam 与 Wainwright (2012) 对二元图模型研究图恢复的信息论限制。[稀疏恢复原文](https://statistics.berkeley.edu/sites/default/files/tech-reports/725.pdf)；[图模型原文](https://arxiv.org/pdf/0905.2639)

迁移边界：统计条件依赖图不是电网物理拓扑；高维线性模型中的 beta-min 不能直接替换为线路 r、x 的大小。AC 中应研究的是相同观测设计下，两类物理模型可观测响应的最小分离。

## 6. 要保持诚实覆盖，边状态集合就必须经常模糊

【由第 3 节直接推出的推论；不作为新发现的原论文定理】

设边状态置信集合 C(D)⊆{0,1}，在两个状态下均保证：

\[
P_b\{b\in C(D)\}\ge1-\alpha,\qquad b=0,1.
\]

在 P_0 下，由总变差：

\[
P_0\{1\in C\}\ge P_1\{1\in C\}-\delta
\ge1-\alpha-\delta.
\]

再与 P_0{0∈C}≥1−α 使用并集界，并对 P_1 作对称论证：

\[
\boxed{
P_b\{\{0,1\}\subseteq C(D)\}
\ge \max(0,1-2\alpha-\delta),\quad b=0,1.
}
\]

例如 P_0=P_1、α=0.05 时，一个在两个状态下都诚实有效的集合，至少 90% 的时候必须同时保留“存在”和“不存在”。

所以置信集合并非不能构造；无法保证的是它在不可辨识情形下仍然很窄。把 C={0,1} 标成“无法判定”具有严格含义。对整图置信集投影得到的边状态集合，也应保持这种歧义，而不强制转换成一个精确小数。

## 7. 单个案例的条件概率：确有专门的不可能性论文

这里换成 iid 有标签总体 (X_i,B_i)：X 为情景或分数，B 为边真值；目标为 η_P(x)=P(B=1|X=x)。它不同于一张固定未知图的多次量测。

Barber (2020) 的 Theorem 2 证明：要求对所有联合分布 P 有效的条件概率置信区间，其平均长度有不随 n 消失的下界。具体地，X 无原子且 η_P(x)≡1/2 时：

\[
P\{\eta_P(X_{\rm new})\in C_n(X_{\rm new})\}\ge1-\alpha
\quad\text{对所有 P 有效}
\]

蕴含

\[
E\,{\rm length}\{C_n(X_{\rm new})\}\ge1-\alpha.
\]

这是平均覆盖，已经弱于逐 x 覆盖。限制来自“对所有 P 诚实”，不是禁止在正确的参数模型、已知平滑条件下估计概率。该原文包含匹配构造。[Barber 原文](https://arxiv.org/pdf/2004.09477)

对“指标套 sigmoid”也有针对性结果：Gupta、Podkopaev、Ramdas 的 2022 修订版 Theorem 3，排除一对一后处理映射获得其定义下的统一无分布渐近校准保证；严格单调的非退化 Platt 缩放属于此类。Theorem 4 则在 iid 数据及固定分箱等条件下给出分箱概率的有效界；分箱也可用独立数据确定。[原文 v4](https://arxiv.org/pdf/2006.10564v4)

分箱目标是 P(B=1|X∈A_k)，一般不等于 P(B=1|X=x)。降低分辨率可换取可估计性。版本说明：该文发表于 NeurIPS 2020；此处采用 2022 v4 的正文定理。更一般连续参数映射的不可能性在 v4 正文中属于 conjecture，不能升级为已证结论。

此前推荐的 Platt、isotonic 等经验校准，必须限定为有代表性标签数据与相应假设下的方案，并独立评估。它们不保证任意影响指标都能形成准确的单边概率，也不把不同网络/噪声机制的迁移误差自动消除。

## 8. 对用户 AC-PQV 指标的具体含义

设剖面目标为：

\[
J_b(D)=
\inf_{\substack{G:b_e(G)=b\\\theta,x_{1:n}}}
\sum_{t=1}^n
\|y_t-h_G(\theta,x_t)\|_{\Sigma_t^{-1}}^2,
\qquad S_e(D)=J_0(D)-J_1(D).
\]

这里所有模型必须遵守同一物理约束、观测定义和实验输入。J_b 是整个有边/无边模型族的最优拟合，不是任意两张候选图之差。径向网络的一次加边/删边可能破坏结构约束，常需合法换边。

要论证 S_e 无法给出普适可靠概率，比研究一个 sigmoid 更直接的路线是定义观测均值族：

\[
\mathcal M_{e,b,n}
=\{\mu_n(G,\theta):G,\theta\text{ 允许且 }b_e(G)=b\}.
\]

在共同已知 Gaussian 协方差 Σ_n 下，定义：

\[
\rho_{e,n}
=\inf_{\mu_0\in\mathcal M_{e,0,n},
       \mu_1\in\mathcal M_{e,1,n}}
\|\Sigma_n^{-1/2}(\mu_1-\mu_0)\|_2.
\]

由任取一对允许模型应用两点界，可得：

\[
\inf_{\widehat b}
\sup_{G,\theta}
P_{G,\theta}\{\widehat b\ne b_e(G)\}
\ge\Phi(-\rho_{e,n}/2).
\]

这是一项条件迁移：需保证两类模型、噪声、已知输入及允许参数集合的定义真实可信。inf 不必达到，取趋近该下确界的参数对即可。

由此可分别研究：

- 两均值族相交：存在严格不可辨识点；
- 虽不相交，但 ρ 很小：有限样本中统一准确判断受限；
- 存在序列 ρ_{e,n}→0：该模型族不具备统一精确恢复能力；
- 通过激励、传感器或物理约束建立正的分离：可能恢复，但仍需上界/算法证明。

特别注意：

1. 找到一对很接近的允许模型，即可证明该模型族的最坏困难；它不证明真实网络恰好位于该困难区域。
2. 在有限候选池内没有找到近似等价模型，不能证明全模型族已分离。
3. 两个 AC 非凸优化器返回相似残差，不等于数学上已经证明分布相同。
4. 高斯噪声协方差错误、隐藏负荷模型错误，都可能使统计界失去解释。
5. 一条真实边完全可以在当前工况下影响很小；影响指标直接衡量的是可观测作用，而非先验已定义好的物理存在率。

局部线性化可用于寻找困难方向。先白化观测坐标，将边相关扰动的响应记为 d，其他参数/状态的局部 Jacobian 记为 A。能被其他参数吸收的分量为投影 Π_A d，剩余量为 (I−Π_A)d。剩余很小提示弱可辨识，但这是局部诊断；除非控制线性化误差及全局替代解，不能称为精确 AC 不可能性证明。

## 9. 建议用于论文的论证目标

合适表述：

> 在给定观测与噪声模型下，当有边与无边模型族包含观测等价或任意接近的配置时，边状态不具备统一可辨识性。任何基于这些观测的影响指标及其后处理，均无法保证同时具有低判断错误和高分辨率的不确定性输出。若需要边存在概率，必须明确额外的先验或有标签总体假设。

不宜表述：

> 图是离散的，所以不能定义概率；有噪声，所以边概率没有意义；只要置信度模型无法优化，就证明概率不可加入。

对研究设计而言，可以把目标改成“影响指标 + 可区分性诊断 + 错误率/置信集合”，在没有适当概率模型时保留未决状态。若之后确有可靠的先验或独立标签总体，再加入明确限定解释范围的概率输出。

## 10. 核心论文与可复制检索词

Google Scholar 适合理论跨学科检索；IEEE Xplore 不一定收录统计学原论文。下面区分精确题名查询与相关主题查询，不把主题查询视为已检索到的论文。

| 论文与结果定位 | Google Scholar 查询 | IEEE Xplore 查询 |
|---|---|---|
| Yuan, Low, Ardakanian, Tomlin. *Inverse Power Flow Problem*. IEEE TCNS 10(1):261–273, 2023. DOI 10.1109/TCNS.2022.3199084。Example 2 / 隐藏节点与端口等价。 | "Inverse Power Flow Problem" Yuan Low | "Document Title":"Inverse Power Flow Problem" |
| Yu. *Assouad, Fano, and Le Cam*. 1997, pp.423–435. DOI 10.1007/978-1-4612-1880-7_29。两点法与信息论下界综述。 | "Assouad, Fano, and Le Cam" | "All Metadata":"Le Cam" AND "All Metadata":"graph recovery" （相关主题） |
| Wainwright. *Information-Theoretic Limits on Sparsity Recovery in the High-Dimensional and Noisy Setting*. IEEE TIT 55(12):5728–5741, 2009. DOI 10.1109/TIT.2009.2032816。精确支撑恢复条件。 | "Information-theoretic limits on sparsity recovery in the high-dimensional and noisy setting" | "Document Title":"Information-theoretic limits on sparsity recovery in the high-dimensional and noisy setting" |
| Santhanam & Wainwright. *Information-Theoretic Limits of Selecting Binary Graphical Models in High Dimensions*. IEEE TIT 58(7):4117–4134, 2012。图选择的信息下界。 | "Information-Theoretic Limits of Selecting Binary Graphical Models in High Dimensions" | "Document Title":"Information-Theoretic Limits of Selecting Binary Graphical Models in High Dimensions" |
| Barber. *Is distribution-free inference possible for binary regression?* EJS 14(2):3487–3524, 2020. DOI 10.1214/20-EJS1749。Theorem 2 / 条件概率区间长度下界。 | "Is distribution-free inference possible for binary regression" | "All Metadata":"distribution-free" AND "All Metadata":"conditional probability" （相关主题） |
| Gupta, Podkopaev, Ramdas. *Distribution-free binary classification: prediction sets, confidence intervals and calibration*. NeurIPS 2020，定理以 arXiv v4 (2022) 为准。Theorems 3–4 / 一对一校准限制与分箱正面结果。 | "Distribution-free binary classification" "calibration" | "All Metadata":"probability calibration" AND "All Metadata":"distribution-free" （相关主题） |
| Leeb & Pötscher. *Can One Estimate the Unconditional Distribution of Post-Model-Selection Estimators?* Econometric Theory 24(2):338–376, 2008. DOI 10.1017/S0266466608080158。回归模型选择边界附近的分布估计限制，非所有选择后推断均不可能。 | "Can One Estimate the Unconditional Distribution of Post-Model-Selection Estimators" | "All Metadata":"post-model-selection" AND "All Metadata":"inference" （相关主题） |

补充主来源：

- [Wainwright 作者机构页面](https://statistics.berkeley.edu/tech-reports/725)
- [Santhanam–Wainwright 原文](https://arxiv.org/pdf/0905.2639)
- [Leeb–Pötscher 原文](https://arxiv.org/pdf/0704.1584)
- [Barber 期刊 DOI](https://doi.org/10.1214/20-EJS1749)

推荐扩展组合：

Google Scholar：
- "power distribution" "topology identification" "identifiability"
- "inverse power flow" "Kron reduction" "hidden nodes"
- "graph recovery" "information theoretic lower bounds" "minimum signal"
- "support recovery" "beta-min" "impossibility"
- "conditional probability" "distribution-free" "impossibility"
- "honest confidence sets" "weak identification"

IEEE Xplore：
- "All Metadata":"topology identification" AND "All Metadata":"identifiability"
- "All Metadata":"power distribution" AND "All Metadata":"topology" AND "All Metadata":"information theoretic"
- "All Metadata":"graph recovery" AND "All Metadata":"lower bounds"
- "All Metadata":"topology identification" AND "All Metadata":"active probing"

本补充仅新增研究文档；未修改算法、试验或既有报告。
