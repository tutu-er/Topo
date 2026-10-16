# 共享线路响应不确定性：模型、证据与EV优化

本轮新增模型。线路范围、完整AC方程和EV运行约束沿用原实验；旧133份科学源码与结果保持冻结。以下区分线性Gaussian模型中的等式、数值积分误差和AC近似，实际校准由新独立历史检验。

## 1. 哪些量随机，哪些量只保留一个中心

用 \(k\) 表示候选拓扑，\(T,V,F\) 分别表示训练、已观测证据和未来。每拓扑的AC训练拟合只提供一个展开点。所有动作共享历史后的一个预测中心，规划时不重新拟合线路。

第 \(e\) 条线路使用交错坐标

\[
\eta_{2e}=\log R_e,\qquad \eta_{2e+1}=\log(X_e/R_e).
\]

原 \(R/X\) 区间和比例范围在此坐标中构成凸多面体 \(P_k\)：限制

\[
\log R_{\min}\le\eta_{2e}\le\log R_{\max},\quad
\log\rho_{\min}\le\eta_{2e+1}\le\log\rho_{\max},\quad
\log X_{\min}\le\eta_{2e}+\eta_{2e+1}\le\log X_{\max}.
\]

先验为 \(N(m_0,C_0)\) 在 \(P_k\) 内的截断分布。未截断均值取公开log区间中点，标准差取对应宽度除以 \(\sqrt{12}\)，边间独立；不提高原最小阻抗。\(Z_0\) 表示先验Gaussian落入 \(P_k\) 的质量，用逐边一维积分计算。

每时段状态 \(z_t=(p_t,q_t,s^P_t,s^Q_t)\) 包含真实末端功率和同一接口的未计量源；计量偏差与根电压已校准固定。\(z_t\) 的先验为

\[
z_t=X_t\beta+r_t,\quad \beta\sim N(0,0.5^2I),\quad
r_t=\phi r_{t-1}+v_t,\quad v_t\sim N(0,Q),\quad
r_0\sim N(0,Q/(1-\phi^2)).
\]

\(X_t\) 为预先声明的日内谐波；\(Q\) 为全通道创新协方差，保留P/Q与源的相关。只从训练原始P/Q和“总表减末端”代理估计 \(\phi,Q\)，随后冻结。这是经验Bayes：不积分超参数的不确定性。源代理含线路损耗，故其过程估计有近似误差；代理不作为第二份源观测，AC拟合源也不作为观测或先验均值。

代理噪声协方差为 \(R_t=A\operatorname{diag}(\sigma_t^2)A^\top\)，其中 \(A\) 执行“总表减末端”。因此末端P误差与源P代理误差的协方差为 \(-\sigma_P^2\)，不能删除。若先用OLS去除谐波，记投影 \(M=I-X X^+\)、差分 \(D_\phi r=r_{t+1}-\phi r_t\)、\(W=D_\phi M\)。在固定 \(\phi\) 下，创新样本协方差的期望为

\[
E[\widehat C]=c_Q Q+\frac1{H-1}\sum_t\|W_{:t}\|^2R_t,\quad
c_Q=\frac{\operatorname{tr}(W K_\phi W^\top)}{H-1},\quad
(K_\phi)_{ij}=\frac{\phi^{|i-j|}}{1-\phi^2}.
\]

程序按此扣除已知噪声并除以 \(c_Q\)，再在平均代理噪声的白化坐标中将创新特征值下限设为0.01。该修正考虑短训练段的谐波拟合自由度；插件 \(\phi\) 仍带估计误差。

## 2. 原始训练只进入一次，边界只作用一次

将全部P/Q/V/总表读数按各自已知标准差白化为 \(y\)。局部AC模型为

\[
y_T\simeq f_T(w_0)+H_T(w-w_0)+\epsilon_T,\quad
w=(\eta,z),\quad\epsilon_T\sim N(0,I).
\]

完整AC给出线路与状态导数。物理线路导数原顺序为 \(R_{all},X_{all}\)，与交错log坐标的链式导数相乘。线路在所有时段共享，状态通过过程核跨时相关。

先忽略边界，解析条件化得到Gaussian后验 \(N(m,C)\)。只对低维线路边缘求截断质量 \(Z_T\) 和矩 \(\bar m_L,\bar C_L\)。若 \(L=\eta\)、\(A=C_{zL}C_{LL}^{-1}\)，Gaussian条件分布给出

\[
\bar m_z=m_z+A(\bar m_L-m_L),\quad
\bar C_{zz}=C_{z|L}+A\bar C_LA^\top,\quad
\bar C_{Lz}=\bar C_LA^\top,
\quad C_{z|L}=C_{zz}-C_{zL}C_{LL}^{-1}C_{Lz}.
\]

这些是声明线性模型中的精确矩公式：先对给定线路条件化，再使用全期望与全方差公式。线路与源的补偿一般使交叉协方差为负。只加“线路方差”而忽略交叉项可能扩大或缩小预测误差。

证据段在训练截断均值处建立新的AC仿射块，但条件化作用于原**未截断**训练Gaussian。分别记未截断Gaussian预测密度为 \(p_G\)，则

\[
p(T\mid k,P_k)=p_G(T\mid k)\frac{Z_T}{Z_0},\qquad
p(V\mid T,k,P_k)=p_G(V\mid T,k)\frac{Z_{TV}}{Z_T}.
\]

证明来自Bayes公式：对域内参数积分等于未截断边际密度乘后验域质量，再除先验域质量。两式相乘，\(Z_T\) 抵消。把训练截断矩重新当Gaussian先验，再施加边界，会更换模型；程序保留底层未截断Gaussian以避免此问题。

均匀结构先验下 \(\log\pi_k\) 由两段log证据相加后用logsumexp归一化；所有候选保留，包括浮点显示概率为0的候选。仅证据权重是单独消融。由于AC展开点及过程超参数由训练决定，所得结构权重属于明确的局部经验Bayes近似，不是原非线性AC模型的精确积分。

截断模块使用条件截断Normal的重要性积分，保存质量、独立Sobol scramble间误差、ESS及失败。ESS比例反映提案效率；达到数值精度需要足够绝对ESS和质量/矩误差。积分不达预定门槛时保存失败，不删困难候选。

## 3. 未来功率、电压与EV响应

只保存历史后 \(\eta,z_F\) 的联合矩 \(\bar m,\bar C\)。每动作 \(u\) 以kW计，末端P中加入 \(u/P_{base}\) 一次。完整AC在 \(\bar m\) 处给出白化均值 \(\mu_k(u)\) 和联合导数 \(F_k(u)=[G_\eta,G_z]\)：

\[
Y\mid k,D,u\ \dot\sim\ N(\mu_k(u),\Sigma_k(u)),\qquad
\Sigma_k(u)=I+F_k(u)\bar C_kF_k(u)^\top.
\]

这里的点为“截断联合矩+AC一阶传播”的Gaussian近似。截断分布和非线性AC的真实预测通常非Gaussian。未来P/Q作为随机读数包含在联合密度；实际观测到P/Q后等价于先评价其边际再条件化其余通道，不连续优化读数，也不重复计分。

同一随机线路与背景分别执行 \(u\) 和 \(u_0\) 时，其物理响应差的协方差为

\[
\operatorname{Cov}(\Delta Y)\simeq [F(u)-F(u_0)]\bar C[F(u)-F(u_0)]^\top.
\]

因此相同动作的差异方差为0；两动作边际方差相加会错失相关性。报告分别保留线路、状态及交叉项。去交叉项消融保留两边缘；“线路已知”使用条件状态方差 \(C_{zz}-C_{zL}C_{LL}^{-1}C_{Lz}\)，两者含义不同。

## 4. 目标与MILP导数来自哪里

目标是声明预测模型下的MAP错误风险

\[
R(u)=E[1-\max_k p(k\mid Y,D,u)].
\]

多候选Gaussian的Bhattacharyya量为

\[
B_{ij}=\tfrac18\delta^\top\bar\Sigma^{-1}\delta+
\tfrac12\log\frac{\det\bar\Sigma}{\sqrt{\det\Sigma_i\det\Sigma_j}},\quad
\delta=\mu_i-\mu_j,\quad\bar\Sigma=(\Sigma_i+\Sigma_j)/2.
\]

对应风险上界 \(C(u)=\sum_{i<j}\sqrt{\pi_i\pi_j}e^{-B_{ij}(u)}\)。证明：逐观测令 \(w_i=\pi_i p_i(y)\)，将最大者记为 \(w_1\)，则

\[
\sum_iw_i-\max_iw_i=\sum_{j>1}w_j
\le\sum_{j>1}\sqrt{w_1w_j}\le\sum_{i<j}\sqrt{w_iw_j}.
\]

积分后得到该界。它的作用域是所声明的Gaussian预测密度。

MILP在当前父计划冻结协方差，线性化均值。\(M_k=\partial\mu_k/\partial u\) 由完整AC的末端P导数除以 \(P_{base}\) 得到；冻结协方差时

\[
\nabla_u B_{ij}=\tfrac14(M_i-M_j)^\top\bar\Sigma^{-1}\delta.
\]

主问题使用保留二次差异的分段代理和运行二进制变量，不把这一梯度直接当全局线性目标。即使当前梯度为0，二次差异仍可能随动作增长。完整传播时协方差也随动作变，

\[
\partial_u\Sigma=(\partial_uF)\bar CF^\top+F\bar C(\partial_uF)^\top;
\]

这需要AC的混合二阶信息。首版MILP冻结它，接受提案时重新完整AC并传播协方差，用共同规划noise积分同一风险；完整风险导数另用有限差分核查。求解器间隙仅对应本次冻结代理。

## 5. 可检验的保证

四组可行域通过允许放电和旅行形成FC/FD/MC/MD。FD与MC互不包含；MD继承所有适用父计划。以同一预测密度、同一积分noise和同一风险选择父计划及新提案，可保证**取得的规划风险**不随扩域增加。独立积分检验其数值排序，独立历史检验实际辨识。

该保证不自动扩展到真实物理生成分布：线路/源过程预测失配、AC线性化与Gaussian矩近似、有限积分、有限MILP预算分别记录。主报告保留失败、未决和候选漏真分母，以独立历史而非重复future noise为统计单位。边界处线路方差通过有界积分获得，不把活动约束直接当零方差。
