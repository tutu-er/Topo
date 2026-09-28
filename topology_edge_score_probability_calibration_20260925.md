# 线路误差影响指标如何连接存在概率与置信度

日期：2026-09-25。聚焦问题：已有指标衡量一条线路对全局拟合误差的影响，如何获得概率解释；只提供参考分数是否有研究价值。

结论：单一指标可以作为概率模型的输入，不要求先证明它本身就是概率。需要增加统计校准或显式生成模型，才能赋予概率含义。若只关心拟合/预测贡献，则可以给贡献指标的置信区间，不必强行变成存在概率。本文为理论解释与研究设计，未实施校准实验。

## 1. 先保持方向与推断目标

设 B_e∈{0,1} 为线路真实状态。把全部连续参数按统一规则重新拟合后，定义

$$
S_e=J_e^{(0)}-J_e^{(1)}.
$$

两侧可以是两张固定备选图，也可以是两个边状态下的全拓扑族；两种含义需要明示。正值偏向有边，负值偏向无边。若现有指标是无符号翻边代价 |S_e|，须同时保留当前估计状态 Bhat_e；可校准 P(B_e=Bhat_e | |S_e|,Bhat_e)，不能把强烈支持无边和强烈支持有边都叫存在概率。

“贡献”有三个不同对象：

| 声明 | 合理统计对象 |
|---|---|
| 删去该线后当前数据拟合恶化 | 当前样本的拟合贡献 S_e |
| 删去该线后未来工况预测更差 | 样本外损失差的期望、中位数或预测分布 |
| 该线路真实存在 | B_e 的条件概率、后验概率或边状态假设检验 |

低贡献不等于线路不存在：激励弱、其他线路/参数可补偿、两个模型观测等价都可能使真边贡献很低。高贡献也可能是在补偿其他拓扑或模型错误。需要理论条件或真值验证来建立两者之间的联系。

## 2. 对任意指标都适用的桥梁：指标分布与贝叶斯公式

明确目标网络/线路/工况的参考总体，令

$$
\pi=P(B_e=1),\qquad
f_b(s)=p(S_e=s\mid B_e=b).
$$

连续分数使用密度，离散分数使用概率质量。则标准贝叶斯公式直接给出

$$
\boxed{
P(B_e=1\mid S_e=s)
=
\frac{\pi f_1(s)}
{\pi f_1(s)+(1-\pi)f_0(s)}.
}
$$

或者写成

$$
\log\frac{P(B_e=1\mid s)}{P(B_e=0\mid s)}
=
\log\frac{\pi}{1-\pi}
+\log\frac{f_1(s)}{f_0(s)}.
$$

这条公式不要求 S 是似然、平方误差或线性模型量；它说明任意有信息的指标都可以被概率校准。缺少的是两类状态下的指标分布及先验/基准存在率，而非一定要更换指标。

例如同一 s 的密度比 f1/f0=10：
- π=0.5 时，存在概率为 10/11≈90.9%；
- π=0.01 时，存在概率为 0.1/1.09≈9.17%。

所以只有“分数很大”或“比其他线路大”不能确定存在概率。同一批边的 min-max 归一化、分位排名、任意 sigmoid 均未给出 f0、f1、π。

若不同噪声、样本量、激励水平下分布不同，可加入事前定义的条件 c：

$$
P(B_e=1\mid S_e=s,c)
=
\frac{\pi(c) f_1(s\mid c)}
{\pi(c) f_1(s\mid c)+[1-\pi(c)]f_0(s\mid c)}.
$$

仅使用 s 得到的是对这些条件混合后的概率，可能总体校准而某些子群失准。

对于同一条固定未知线路重复加噪声或 bootstrap，只能研究分数波动/选择稳定性，不能凭这些重复自动产生“存在/不存在”的真值标签。固定图的 Bayesian 后验仍可定义，但需明确两类物理生成模型、先验及未知参数处理。

## 3. 最贴近现有指标的做法：从独立真值情景学习校准函数

【研究设计】保持现有指标，学习

$$
g(s)\approx E[B_e\mid S_e=s].
$$

不必显式估计 f0、f1。构造真实拓扑已知的独立网络情景，运行完整现有辨识与评分流程，得到 (s_i,b_i)。

具体候选：

1. Logistic/Platt 校准：g(s)=1/[1+exp(−(a s+b))]；a、b 由标注校准集拟合。
2. Isotonic regression：在“分数越大、存在概率越高”的单调假设合理时拟合单调函数。
3. 有必要时使用 g(s,n,noise,excitation,...)，并在独立测试集比较是否改善概率质量。

【文献 R1】Niculescu-Mizil 与 Caruana (2005) 研究 Platt 与 isotonic 的分数到概率转换；后者更灵活但小样本容易过拟合。这不是“预先挑一个看起来合适的 sigmoid”：映射参数必须从数据学习，效果必须另测。

### 3.1 为什么学习概率映射有明确理论基础

【标准条件期望推导；R2 为 proper scoring 理论】设 η(s)=P(B=1|S=s)。固定 s 时，预测概率 q 的条件平方损失为

$$
E[(B-q)^2\mid S=s]
=
\eta(s)[1-\eta(s)]+[q-\eta(s)]^2.
$$

所以唯一最优值为 q=η(s)。对数损失的条件期望

$$
-\eta(s)\log q-[1-\eta(s)]\log(1-q)
$$

也在 q=η(s) 最小。故用 Brier score 或 log loss 学习 g 具有清晰的概率目标。

这是总体风险的结论；有限样本、模型族限制和优化误差仍会使 ghat 与 η 不同。不能声称拟合完成就有严格有限样本逐点校准。

### 3.2 一个指标是否足够

一个指标足以定义 P(B=1|S)，不要求它是充分统计量。但它可能丢失数据中的部分判别信息。

设原始数据为 D，S=S(D)，η_D=P(B=1|D)，η_S=P(B=1|S)，则

$$
E[\eta_D\mid S]=\eta_S,
$$

$$
E[(B-\eta_S)^2]-E[(B-\eta_D)^2]
=
E[(\eta_D-\eta_S)^2]\ge0.
$$

所以单指标概率是一个有定义的条件概率，但不必等于使用全部数据的概率。若 S 对两类完全无区分力，g(s)退化为基准存在率 π；仍可校准，却不具备有用的个体区分能力。

因此同时检查：
- 校准：预测约 0.8 的大量新案例，真实存在比例是否约 80%；
- 区分：真边是否通常获得更高概率，尤其在稀疏图下检查 PR 曲线；
- 概率质量：Brier/log loss 是否优于恒定输出 π 的基线；
- 分组：不同噪声、工况、线路位置等是否出现系统偏差。

### 3.3 针对拓扑数据，校准实验的边界

这些是本文针对用户场景的设计要求：

- 校准集和最终测试集分开；评分算法/超参数若已调试，应避免用同一批网络再报告最终性能。
- 按独立网络、馈线或独立情景分组切分；同一图的边和时间窗不能自动当成独立样本。
- 若只评价“算法已经选出的边”，校准样本也必须复现该筛选流程；目标是选中条件下的正确率。若评价所有候选边，则应按该目标总体取样。
- 真/假边比例必须对应部署参考总体。人为平衡的正负样本会改变基准率；仅在类条件分布保持不变的先验漂移条件下，才可按先验 odds 修正概率。
- 对仅由仿真得到的校准，明确这是该仿真分布下的条件存在概率；迁移现实量测需验证模型差异。
- 多条边的相关性不会阻止定义各自边际概率，但不能将这些边际概率相乘来获得整图概率，也不保证校准后的数值自动满足径向图的联合约束。
- 分箱可靠性图与误差条是验证工具；有限样本区间应按实际独立情景/聚类结构计算，不能把所有相关边按独立二项试验计数。

## 4. 不依赖真值标注校准集的另一条路：似然与模型选择

如果 J 是共同已知协方差下的高斯 Mahalanobis 残差和，两类模型分别 profile 后，

$$
2\log\frac{\sup_\eta p(D\mid B=1,\eta)}
{\sup_\eta p(D\mid B=0,\eta)}
=J^{(0)}-J^{(1)}=S.
$$

这时指标已经是广义似然比统计量，可以研究其抽样分布或构建检验，但不是后验 odds。

【文献 R3】真实贝叶斯后验需要

$$
BF_{10}=
\frac{\int p(D\mid B=1,\eta)\pi_1(\eta)d\eta}
{\int p(D\mid B=0,\eta)\pi_0(\eta)d\eta},
\qquad
P(B=1\mid D)=\frac{\pi BF_{10}}{\pi BF_{10}+1-\pi}.
$$

这里的积分应包含所声明模型中不确定的其余拓扑及连续变量。只取最优值会忽略参数不确定性与体积。对任意量纲/任意权重/加惩罚损失，S/2 也不自动等于 log likelihood ratio。

### 4.1 一个精确的小例子：相同 profile 差值如何对应后验

【本文高斯算例】观测 x|θ∼N(θ,v)。无边模型为 θ=0；有边模型允许 θ 自由。未乘 1/2 的标准化拟合改善为 S=x²/v。

若有边时先验 θ∼N(0,τ²)，令 r=τ²/v，积分后得到

$$
BF_{10}=(1+r)^{-1/2}
\exp\left[\frac{r}{1+r}\frac S2\right].
$$

它同时包含似然改善和参数积分引入的体积项。只有给定 r 及 π 后才能把 S 变成后验概率。这个简单例子说明：即使 S 已有标准化似然意义，仍可能不是裸 logistic(S/2)。

在最特殊的两个完全指定、等先验高斯模型下，没有未知参数需要积分，才有 P(B=1|D)=logistic(S/2)。

## 5. 若目标是“置信”，可以校准检验阈值而不输出概率

在 H0:B=0 下建立 S 的零分布。大分数支持有边时，取 cα 使

$$
\sup_{\eta\in H_0}P_\eta(S>c_\alpha)\le\alpha.
$$

于是“仅当 S>cα 才判有边”可控制误报概率。这里一个经过有效校准的参考阈值已经足够实现该检验目标，不必产生后验百分比。经验拍定阈值则没有这个保证。

在固定设计、已知同方差高斯噪声、常规线性嵌套模型且仅增加一个无约束可辨识系数的例子中，

$$
S=(RSS_0-RSS_1)/\sigma^2\sim\chi^2_1\quad(H_0).
$$

阈值约 3.84 对应 5% 检验水平。它不意味着超过阈值的边有 95% 概率存在。

【文献 R4】AC 换边非嵌套、导纳零边界、线路消失后 nuisance 不可辨识等条件会使常规 Wilks 近似失效。可研究施加零假设的参数模拟/自举、置信集反演或其他有效检验，但必须按实际流程重新拟合和处理多重边筛选；原假设下的单个 plug-in 参数模拟一般不自动提供全域有限样本保证。

最简单的“显著性不是后验”算例：某检验在无边时误报率 5%，有边时检出率 80%。若候选边存在率 π=10%，则

$$
P(B=1\mid {\rm positive})
=
\frac{0.8\times0.1}{0.8\times0.1+0.05\times0.9}
=0.64.
$$

所以“5% 显著性”不能翻译成“该边 95% 存在”。

## 6. 只研究影响值，也有严肃的统计理论

若目前难以校准物理边存在概率，可以把目标明确为“预测贡献的不确定性”。

【研究设计；借鉴 R5 的 LOCO】在训练数据上拟合含边与无边模型，在独立测试工况 i 上计算成对损失差

$$
d_i=\ell(\widehat M_{-e};z_i)-\ell(\widehat M_{+e};z_i).
$$

对 μ_e=E[d_i|训练数据]构造置信区间，或分析中位数/未来损失差。两模型参数不得在测试输出上无约束重新拟合，否则改变了验证目标；AC 测试场景中应明确哪些输入已知、哪些状态由预测求得。

【文献 R5】Lei 等 (2018) 的 leave-one-covariate-out inference 用删变量重拟合后的样本外损失差衡量变量贡献，并研究其推断。将删变量迁移为约束边状态是本文设计，不是原文已证明的电网定理。均值区间、中位数检验、个体预测区间的条件不同，不能混称统一有限样本保证。

这种结果可以支持“该线路在规定预测任务中有稳定贡献”，仍不能直接支持“物理线路存在”。如果论文目标就是重要性评估或测量优先级，这已是合理统计目标；若标题和结论声称存在概率，则还需要第 2–4 节的桥梁。

## 7. 任意损失的广义贝叶斯理论：可作备选，但温度必须说明

【文献 R6】Bissiri–Holmes–Walker 给出以损失连接参数与数据的更新框架，形式为

$$
Q_\lambda(T,\eta\mid D)
\propto
\pi(T,\eta)\exp[-\lambda L(T,\eta;D)].
$$

这是在期望损失与相对于先验的 KL 散度之间权衡的解。λ 是学习率/温度，控制分布集中程度。两固定状态的损失差会进入 log odds，但不是无需假设便发现了物理后验概率。

仅有每条边的局部删边分数未必能组成一致的联合图损失；AC 线路影响通常耦合，不能随意求和再声称是原始 AC 模型。若坚持用该路线，应先定义整个合法图空间上的损失与目标风险最小化图；其目标不自动等于物理真图。

【文献 R7】Syring–Martin 研究通过学习率调整广义后验可信区域的频率覆盖，采用 bootstrap 与随机逼近。它证明校准是一个需要单独处理的问题，不是所有 exp(−loss) 权重自带性质。连续参数可信区域的校准也不自动证明每条离散边的概率校准。对本任务，独立真值情景的分数校准通常更直接。

## 8. 对当前研究最具体的选择

1. 保留现有指标并写明两侧如何重拟合，保持支持有边/无边的方向。
2. 若已有仿真真拓扑：先做跨独立网络的 g(S) 校准，以恒定基准率为对照，再检查是否需要加入噪声、样本量和激励信息。输出名称应为“指定情景分布下的校准边存在概率”。
3. 若主要研究单张真实未知图、没有代表性标注：优先明确量测似然与未知参数分布，选择假设检验或 Bayesian 模型比较。
4. 若现阶段只证明指标与拟合贡献有关：报告参考分数及其稳定性/预测贡献区间，避免使用未经支撑的“95% 存在概率”。这不否定指标价值，只限定研究结论。
5. 校准概率后保留原分数、优化误差和参考总体。AC 局部求解误差可被完整流程校准吸收为算法表现的一部分，但不能据此宣称已得到精确物理 likelihood 或全局拓扑证书。

关键实验产物应是“真边/假边的分数分布＋独立测试可靠性图＋Brier/log loss＋分组表现”，而不仅是恢复图上按分数着色。本文没有实施这些实验，所有流程均为待验证设计。

## 9. 原始文献与可复现检索词

非 IEEE 统计学论文在 IEEE Xplore 可能仅检到引用或相关工作；检索链接不是收录或已获得全文的声明。

### R1. Predicting Good Probabilities With Supervised Learning

Alexandru Niculescu-Mizil; Rich Caruana. ICML, 2005. DOI: 10.1145/1102351.1102430。[原文入口](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)。

概率校准：比较 Platt 与 isotonic；说明分数准确排序与概率可靠性是不同问题。

- Scholar：`"Predicting Good Probabilities With Supervised Learning"`。[检索](https://scholar.google.com/scholar?q=%22Predicting%20Good%20Probabilities%20With%20Supervised%20Learning%22)。
- IEEE Xplore：`"Predicting Good Probabilities With Supervised Learning"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Predicting%20Good%20Probabilities%20With%20Supervised%20Learning%22)。

### R2. Strictly Proper Scoring Rules, Prediction, and Estimation

Tilmann Gneiting; Adrian E. Raftery. JASA 102(477):359–378, 2007. DOI: 10.1198/016214506000001437。[原文入口](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)。

严格适当评分规则：Brier/log loss 的总体最优目标是真实条件概率；可用于学习与评价校准器。

- Scholar：`"Strictly Proper Scoring Rules, Prediction, and Estimation"`。[检索](https://scholar.google.com/scholar?q=%22Strictly%20Proper%20Scoring%20Rules%2C%20Prediction%2C%20and%20Estimation%22)。
- IEEE Xplore：`"Strictly Proper Scoring Rules, Prediction, and Estimation"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Strictly%20Proper%20Scoring%20Rules%2C%20Prediction%2C%20and%20Estimation%22)。

### R3. Bayes Factors

Robert E. Kass; Adrian E. Raftery. JASA 90(430):773–795, 1995. DOI: 10.1080/01621459.1995.10476572。[原文入口](https://www.tandfonline.com/doi/abs/10.1080/01621459.1995.10476572)。

模型后验由先验 odds 与 Bayes factor 连接；Bayes factor 使用边际似然而不是 profile 最大值。

- Scholar：`"Bayes Factors"`。[检索](https://scholar.google.com/scholar?q=%22Bayes%20Factors%22)。
- IEEE Xplore：`"Bayes Factors"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Bayes%20Factors%22)。

### R4. Asymptotic Properties of Maximum Likelihood Estimators and Likelihood Ratio Tests under Nonstandard Conditions

Steven G. Self; Kung-Yee Liang. JASA 82(398):605–610, 1987. DOI: 10.1080/01621459.1987.10478472。[原文入口](https://pages.stat.wisc.edu/~larget/Stat998/Fall2015/Self-Liang-1987.pdf)。

参数边界导致非标准似然比极限；支持检查 AC 增删线问题是否能使用普通卡方阈值。

- Scholar：`"Asymptotic Properties of Maximum Likelihood Estimators and Likelihood Ratio Tests under Nonstandard Conditions"`。[检索](https://scholar.google.com/scholar?q=%22Asymptotic%20Properties%20of%20Maximum%20Likelihood%20Estimators%20and%20Likelihood%20Ratio%20Tests%20under%20Nonstandard%20Conditions%22)。
- IEEE Xplore：`"Asymptotic Properties of Maximum Likelihood Estimators and Likelihood Ratio Tests under Nonstandard Conditions"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Asymptotic%20Properties%20of%20Maximum%20Likelihood%20Estimators%20and%20Likelihood%20Ratio%20Tests%20under%20Nonstandard%20Conditions%22)。

### R5. Distribution-Free Predictive Inference for Regression

Jing Lei; Max G’Sell; Alessandro Rinaldo; Ryan J. Tibshirani; Larry Wasserman. JASA 113(523):1094–1111, 2018. DOI: 10.1080/01621459.2017.1307116。[原文入口](https://www.stat.berkeley.edu/~ryantibs/papers/conformal-jasa.pdf)。

第 6 节 LOCO：删除变量重拟合后，用样本外损失差定义预测贡献。均值/中位数/预测区间须分别遵守其条件。

- Scholar：`"Distribution-Free Predictive Inference for Regression"`。[检索](https://scholar.google.com/scholar?q=%22Distribution-Free%20Predictive%20Inference%20for%20Regression%22)。
- IEEE Xplore：`"Distribution-Free Predictive Inference for Regression"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Distribution-Free%20Predictive%20Inference%20for%20Regression%22)。

### R6. A general framework for updating belief distributions

Pier Giovanni Bissiri; Chris C. Holmes; Stephen G. Walker. JRSS B 78(5):1103–1130, 2016. DOI: 10.1111/rssb.12158。[原文入口](https://arxiv.org/abs/1306.6430)。

用一般损失更新广义信念分布。更新一致性不等于物理边概率校准，学习率必须说明。

- Scholar：`"A general framework for updating belief distributions"`。[检索](https://scholar.google.com/scholar?q=%22A%20general%20framework%20for%20updating%20belief%20distributions%22)。
- IEEE Xplore：`"A general framework for updating belief distributions"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22A%20general%20framework%20for%20updating%20belief%20distributions%22)。

### R7. Calibrating general posterior credible regions

Nicholas Syring; Ryan Martin. Biometrika 106(2):479–486, 2019（在线 2018）. DOI: 10.1093/biomet/asy054。[原文入口](https://academic.oup.com/biomet/article/106/2/479/5237467)。

用学习率调整广义后验可信区域的频率覆盖；不是任意离散边概率的自动校准定理。

- Scholar：`"Calibrating general posterior credible regions"`。[检索](https://scholar.google.com/scholar?q=%22Calibrating%20general%20posterior%20credible%20regions%22)。
- IEEE Xplore：`"Calibrating general posterior credible regions"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Calibrating%20general%20posterior%20credible%20regions%22)。

扩展 Scholar 词条：`"edge existence" "probability calibration"`；`"topology identification" "Bayes factor"`；`"profile likelihood" "topology"`；`"leave-one-covariate-out" "variable importance"`；`"Gibbs posterior" calibration`。

扩展 IEEE Xplore Advanced Search：`("All Metadata":"topology identification") AND ("All Metadata":"probability")`；`("All Metadata":"topology") AND ("All Metadata":"likelihood ratio")`；`("All Metadata":"calibration") AND ("All Metadata":"classifier scores")`。
