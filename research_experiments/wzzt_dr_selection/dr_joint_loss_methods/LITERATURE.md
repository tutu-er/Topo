# 与本问题匹配的方法调研（2026-11-03）

本轮使用公开作者/机构全文；没有需要经HKU补取的核心缺口。以下不是穷尽查新，关注相关高斯、绝对值差、可计算概率与梯度。

1. Nemirovski & Shapiro, Convex Approximations of Chance Constrained Programs, SIOPT 17(4), 969–996, 2006. 作者全文 https://www2.isye.gatech.edu/~nemirovs/SIOPT_Bern_2006.pdf 。已读印刷页971–976、988–990（PDF页3–8、20–22）的结构假设、概率上界和凸性条件。指数矩可给安全界；独立性/凸性条件不能直接转移到我们的相关L1差。这里只借用指数Markov原理，并明确用Holder得到相关安全上界，不宣称凸Bernstein程序。

2. Holger Heitsch, On probabilistic capacity maximization in a stationary gas network, WIAS preprint2540 (2018), later Optimization69 (2020),575–604, DOI10.1080/02331934.2019.1625353. 机构全文 https://www.wias-berlin.de/preprint/2540/wias_preprints_2540.pdf 。已读第5节、Theorem3、Algorithm1（PDF页17–20）。球径分解适用于Borel事件，沿射线允许多个可行区间；作者用球面抽样，并非无积分点的闭式算法。本实现改用确定性方向点，仅称数值近似，不继承未核验的误差或梯度保证。

3. Felix Voigtlaender, A general version of Price's theorem, arXiv1710.03576v2 (2020), DOI10.1007/s10959-020-01017-w. 作者预印本 https://arxiv.org/pdf/1710.03576 。已读第1–3节、Theorem1/Corollary2。支持相关高斯非线性期望对协方差的分析；本轮绝对值Hermite系数及余项上界是本文单列推导和独立小例核验，不写成该文现成定理。

4. Sharif Rahman, Wiener-Hermite Polynomial Expansion for Multivariate Gaussian Probability Measures, arXiv1704.07912v1 (2017). https://arxiv.org/pdf/1704.07912 。已读引言与方法结构，定位第5节作为后续依据。相关输入可经线性变换成为独立高斯；有限阶展开近似不等于尾概率保证。本轮只用可解析的绝对值系数，避免构造全张量高维基。

5. Robert B. Davies, Algorithm AS155: The Distribution of a Linear Combination of chi-squared Random Variables, Applied Statistics29(3),323–333 (1980), DOI10.2307/2346911. 作者PDF https://www.robertnz.net/pdf/lc_chisq.pdf 已定位，网页第一页面文本说明可处理正负卡方加高斯项、用特征函数反演；未阅读全文。二阶近似作为本轮追加数学诊断，采用自行推导特征函数并数值积分，不冒充Davies原代码。Imhof原文机构链接410，未将其列为已读全文。

6. van Ackooij & Perez-Aros (2021), Gradient formulae for probability functions depending on a heterogenous family of constraints, DOI10.5802/ojmo.9. https://www.numdam.org/articles/10.5802/ojmo.9/ 仅摘要/书目信息已核，出版社正文访问403。可列为非凸概率梯度的后续文献，不能声称已核其定理适用本L1差。

判断：没有找到能把当前相关、执行饱和、L1差事件原样化成低保守SOCP/MILP的定理。最直接的匹配是保留联合损失，做确定性概率积分；二阶Hermite近似+一维特征函数反演提供另一个可计算候选，但须核算截断余项。参考电压MILP只解决名义正负号，不能补上未知观测分布。
