# 根连接局部辨识：同一物理参数域、三类评分

## 1. 量测和完整AC

候选k只改变指定三末端的分组，其余连接给定。r、x为跨时段共享的逐边标幺阻抗；p_t、q_t为逐时真实净消耗；b_t为原根电压幅值。观测带波浪号，误差独立、零均值高斯，标准差σ已知。EV/PV若存在，已含在净P/Q。本轮仅普通负荷及已计量PV。

令f_k(r,x,p_t,q_t,b_t)为收敛高电压支的完整AC幅值。FULL评分为

\[
 J_k=\frac12\min_{r,x,\{p_t,q_t,b_t\}}
 \sum_t\left[\|p_t-\tilde p_t\|_{\Sigma_p^{-1}}^2+
 \|q_t-\tilde q_t\|_{\Sigma_q^{-1}}^2+
 \|f_k-\tilde v_t\|_{\Sigma_v^{-1}}^2+
 (b_t-\tilde b_t)^2/\sigma_b^2\right].
\]

这里\(\|d\|_W^2=d^TWd\)。CUT没有根表项，只有三个末端；FOCUS-V只删去非局部电压残差，全网功率仍保留。实际程序返回多初值得到的局部损失及驻点误差。

线路域用\(r=R_c(a+b),\ x=R_c(La+Ub)\)，其中每边a,b≥0、a+b≤1。它恰好是0≤r≤R_c、Lr≤x≤Ur的三角形。r>0时X/R∈[L,U]；r=0则x=0。这个参数域凸、允许真实零边，对AC并不消除非线性。R_c是统一公开资产上限。新L/U为0.05/2，旧0.25/1.5的域是其子集。

AC方程写成F(y,c)=0，其中y是复电压实/虚部，c是待更新量；在潮流Jacobian可逆处，y_c=−F_y^{-1}F_c。计量残差导数按共享线路A_t和逐时状态C_t分块。阻尼最小二乘步先消去逐时状态，再在上述三角形内求共享步；每个接受步重新计算完整AC并检查原J下降。这里是优化计量误差，潮流收敛本身不是最小化J。

## 2. 保留原根信息的解析响应评分

M_{je}=1表示边e位于原根到表计j的路径。线性平方电压响应为

\[
 b_t^2\mathbf1-v_t^2\simeq H_k x_t,
 \quad x_t=(p_t,q_t),\quad
 H_k=2[ M\operatorname{diag}(r)M^T,\ M\operatorname{diag}(x)M^T].
\]

符号x_t是拼接的功率向量，未加时间下标的x仍为线路电抗。为避免混淆，程序内部分别命名powers和reactance。

先校正平方高斯噪声均值：\(y_t=(\tilde b_t^2-\sigma_b^2)\mathbf1-(\tilde v_t^2-\sigma_v^2)\)。在预声明v_ref=1处冻结平方噪声方差，

\[
 S_y=\operatorname{diag}(4v_{ref}^2\sigma_v^2+2\sigma_v^4)
 +(4v_{ref}^2\sigma_b^2+2\sigma_b^4)\mathbf1\mathbf1^T.
\]

共同根表产生非对角项，不能把各压降当独立量测。CUT改用正交对比矩阵D，D1=0，将未知切口平方电压消去；对应H=−2D[Mdiag(r)M^T,Mdiag(x)M^T]和\(S_y=D\operatorname{diag}(4\sigma_v^2+2\sigma_v^4)D^T\)。

设\(S_p=\operatorname{diag}(\sigma_p^2,\sigma_q^2)\)。解析最小化
\(\|\tilde x_t-x_t\|_{S_p^{-1}}^2+\|y_t-Hx_t\|_{S_y^{-1}}^2\)，得到

\[
 Q(H)=\sum_t e_t^T\Omega_t^{-1}e_t,\quad
 e_t=y_t-H\tilde x_t,\quad \Omega_t=S_y+HS_pH^T,
\]
\[
 x_t^*=\tilde x_t+S_pH^T\Omega_t^{-1}e_t,
 \qquad \nabla_HQ=-2\sum_t(\Omega_t^{-1}e_t)(x_t^*)^T.
\]

证明可由严格凸的逐时最小二乘一阶条件代回得到；梯度也可对原目标用包络定理求得，已包含传播协方差变化。这里优化隐含P/Q，不把它们积分成随机先验，因此没有logdet项。平方噪声分布并非高斯、物理响应也有损耗余项；RESP是明确的快速近似，不是原始完整AC似然。

## 3. 凸QP和界

固定正定权重W_t后，H对线路坐标线性，\(Q_W=\sum e_t^TW_te_t\)是凸二次函数，线路域同为上述三角形。先用W=S_y^{-1}求初值，再固定其\(\Omega_t^{-1}\)解一次QP；用于初始化和快速评分，不把数据依赖的固定权重称原AC的严格下界。

对任一可行坐标c及梯度g，凸性给出全局下界

\[
 \min_z Q_W(z)\ge Q_W(c)+\min_{z\in\mathcal C}g^T(z-c).
\]

每边(a,b)三角形顶点为(0,0)、(1,0)、(0,1)，故线性最小值逐边取min(0,g_a,g_b)。右式与可行目标之差给出本次冻结QP的可核对间隙。该界只属于这个QP，不能用来拒绝完整AC候选。

## 4. 误差与排名分别解释

参数诊断对同一真实下游净功率定义\(h_{et}=r_e\sum_{j\downarrow e}p_{jt}+x_e\sum_{j\downarrow e}q_{jt}\)。将拟合和真实R/X代入相同功率比较，隔离参数误差；再用校正后的P/Q比较，展示状态补偿。完整AC电压回代包含全部损耗，不把两种误差差额都归为线性化。

若q≈κp，h≈(r+κx)P；R/X单独误差可以大，而组合及V误差小。小X还会放大X的相对误差。共享线路信息由\(\sum A_t^T(I-C_tC_t^+)A_t\)刻画，仅报告局部秩和弱方向，不自动转成置信区间。

新增pair边允许零时，base包含在pair模型闭包内，复杂模型最小训练J天然不大于base。留出固定线路后评价能检查这类过拟合；近似评分首名一致和真实拓扑首名正确是不同的统计量。独立留出量测及真值审计均在训练封存之后使用。
