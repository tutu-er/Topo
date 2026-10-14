# 连续 EV 设计的非凸性与 MILP 代理：文献核查

核查日期：2026-10-14。范围：Bhela 的注入设计、一般凸二次最大化的困难性、分段线性 MILP 的依据。仅核查相关正文，不声称穷尽文献。复用本地 Bhela 作者稿，其 SHA-256 与原清单一致；其他必要正文来自合法开放页面，未新增 PDF。HKU skill 已考虑，开放正文足够，无需订阅访问。

## 1. 本轮核实的结论

1. Bhela 等确实先建立有限注入库，但其选择过程还包含凸 QP 松弛及随机舍入。只说“作者枚举动作”会遗漏核心算法。
2. 本项目一般形式包含 NP-hard 的凸二次最大化。MILP 将连续非线性目标改为有限分段模型后，能提供可行曲线、求解界及分段误差；它保留组合复杂性。
3. 电网响应在工作点线性化以后，平方响应差仍为二次函数。再用一个切平面替换二次目标，才会丢失大幅激励中的曲率。这两种“线性化”应分别检查。
4. 分段线性、支持方向、二元乘积线性化均有成熟文献。可检验的研究贡献应放在历史共享参数的竞争解释、EV服务约束、可核查计算误差，以及完整AC下的辨识收益，尚不能据本轮有限检索宣称首创。

## 2. Bhela 的实际方法

文献：S. Bhela, V. Kekatos, S. Veeramachaneni, *Smart Inverter Grid Probing for Learning Loads: Part II—Probing Injection Design*, IEEE Transactions on Power Systems, 2019。
DOI：[10.1109/TPWRS.2019.2906306](https://doi.org/10.1109/TPWRS.2019.2906306)。
采用版本：[arXiv:1806.08836v2](https://arxiv.org/pdf/1806.08836v2)，2019-03-20。
本地：papers/mobile_ev_probing/2019_Bhela_Probing_Injection_Design_preprint_v2.pdf。
SHA-256：bf9122023c6e642da80246c45279805a1b8ebb061bd60ec6d43fd72101cef9e4。
实际阅读：§II-A/B/C、§III开头；重点为PDF第3–4页、式(9)–(12)。本地文字与开放正文交叉核实。

作者先在逆变器可行域采样，再以线性潮流及LP筛除对预期负荷域不安全的注入。§II-C从剩余L个动作中挑T个，使其状态两两平方距离之和最大。设y_l为动作l的近似系统状态，D_ll'=||y_l-y_l'||²，选择变量x_l∈{0,1}，则式(11)为

    max x'Dx,  1'x=T.

令Y=[y_1,...,y_L]，c_l=||y_l||²，在固定基数超平面上有

    x'Dx = 2T c'x - 2||Yx||².

右式是凹二次函数。式(12)把x的二元约束放松为[0,1]，求解凸QP，随后使用Bernoulli随机舍入并保留满足基数的高分方案。T=2时可穷举；中等L时作者也提到MIQP。§III的SDP是取得数据后的负荷估计器，作用与动作设计中的QP不同。

这项凹性来自“固定状态库 + 选择权重 + 固定基数”。连续EV功率u本身改变状态，目标||Lu+c||²不具有同样凹性。作者估计的是已知网络下的未知负荷，所用状态多样性目标也不同于两个拓扑充分重拟合后的距离。§II-A明确因短时实验忽略电池SOC；本项目应保留SOC、终端任务和接入时间。

其依据 Cevallos、Eisenbrand、Zenklusen 的 *Max-Sum Diversity Via Convex Programming*，SoCG2016，DOI [10.4230/LIPIcs.SoCG.2016.26](https://doi.org/10.4230/LIPIcs.SoCG.2016.26)，[开放全文](https://drops.dagstuhl.de/storage/00lipics/lipics-vol051-socg2016/LIPIcs.SoCG.2016.26/LIPIcs.SoCG.2016.26.pdf)。本轮读取引言、负型距离说明及凸QP背景；其有限集合/拟阵条件不能自动用于有SOC的连续EV轨迹。

## 3. 一般困难性的简短证明（本项目自行整理）

令C是任意无向图的定向关联矩阵，每条边一行。取

    max_{0≤u≤1} ||Cu||².

目标凸，所以盒上至少存在一个顶点最优解。在u∈{0,1}^n时，每条边贡献(u_i-u_j)²：端点位于两组时贡献1，否则贡献0。因此最优值就是该图的最大割边数。一般矩阵、维数随输入增长的这一类问题是NP-hard；本项目代理允许无竞争变量、零常数项时包含此特例。

Max-Cut及其整数二次形式的原始来源核查：Goemans–Williamson, *Improved Approximation Algorithms for Maximum Cut and Satisfiability Problems Using Semidefinite Programming*, JACM42(6),1115–1145,1995；[作者全文](https://math.mit.edu/~goemans/PAPERS/maxcut-jacm.pdf)，本轮读取§1–2，尤其印刷pp.1119–1120。由Max-Cut到本项目一般代理的映射是上述自行推导。

当前4槽、单节点且有特殊物理矩阵的实例，不因上述一般结论就必然计算困难。秩一平方可用正负两个线性方向处理；特殊时序结构也可能进一步简化。应报告实际求解界与时间。

几何上，固定模型的||Lu+c||是仿射像多面体到原点的距离；允许竞争调整时，是到竞争解释集合的距离。只在额外的对称性或两个独立点选择条件下，才能化为通常的欧氏直径。不要引用多面体边图直径/组合直径的复杂性作为这里的证明。

## 4. 可直接采用的 MILP 数学依据

对于已经正确消去竞争参数、或在已核实活动面内得到的

    q(u)=k+||Lu+c||²,

令w_i=L_i u+c_i，并以操作域上的LP求其有限上下界。对一段[a,b]，平方函数的弦为

    s(w)=(a+b)w-ab,
    s(w)-w²=(w-a)(b-w)∈[0,(b-a)²/4].

SOS2或区间选择二进制可以准确表示分段弦图；u在各段之间仍连续。设每个坐标的最大段宽为h_i，ε=Σh_i²/4，则q≤q_bar≤q+ε。对同一可行域，MILP的全局best-bound给真实二次代理最大值的上界；返回曲线上的真实q给下界。如果MILP求解间隙为δ，则代理最优性损失最多ε+δ。这是固定代理的结论，AC回代误差另算。

源：Bärmann、Burlacu、Hager、Kleinert, *On Piecewise Linear Approximations of Bilinear Terms: Structural Comparison of Univariate and Bivariate Mixed-Integer Programming Formulations*，2021-08-05作者稿，[开放正文](https://optimization-online.org/wp-content/uploads/2021/08/8536.pdf)。本轮读§2.1–2.2及§3.1 Lemma3；其平方插值最大误差正是区间宽²/(4n²)。本项目以上单目标与max-min误差传递为自行整理。

若真实局部成本是q(u)=min_z f(u,z)，直接在MILP里让z与u一起最大化，会改变内层量词。要保持竞争者重拟合，可采用已有支持方向下界/可行竞争解释上界，或正确编码内层最优性条件后再近似平方。只有验证活动面与参数域适用，才能使用单一消元二次式。历史残差的常数及线性项同样保留。

相关实现基础：Huchette–Vielma, *Nonconvex piecewise linear functions: Advanced formulations and simple modeling tools*，[作者稿](https://arxiv.org/pdf/1708.00050)。本轮读取§1–2中的分段函数图与混合整数表示，未复核全部计算实验。该文提供成熟的分段建模框架，不提供本项目AC误差保证。

## 5. 与已有电网 MILP 的区别

Taheri、Kekatos、Cavraro, *An MILP Approach for Distribution Grid Topology Identification using Inverter Probing*，[arXiv:2004.02370](https://arxiv.org/pdf/2004.02370)，2020作者稿；本轮读§III–IV，式(10)、(17)–(21)，PDF第3–4页。作者用MILP在取得的激励数据上恢复拓扑，二元拓扑和连续变量乘积在已给有限界下可用McCormick表示。其连续变量界在正文中说明主要由数值实验支持。它与“通过连续EV曲线最大化未来辨识距离”是不同优化问题。该文的MILP不能直接当作我们的目标精确线性化定理。

## 6. 对本轮实验的建议

把三项误差分开记录：AC响应线性化误差、平方目标的MILP分段误差、内层竞争解释误差。提高线路最小阻抗可能使候选闭包不再靠近零边，也可能排除实际弱支路；应分别做“真实物理支路变强”和“同一真实支路仅提高拟合下界”的实验。前者测试信号条件，后者测试先验失配。参数空间有间隔不保证量测空间有间隔，源、P/Q及其它线路仍可能补偿。此判断为本项目数学分析，不是上述文献直接结论。

