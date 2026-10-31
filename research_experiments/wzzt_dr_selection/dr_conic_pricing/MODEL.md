# 数学模型与实现对应

## 符号

| 符号 | 代码 | 含义 |
|---|---|---|
| x_j | amplitudes | 合同 j 的期望响应幅值，优化变量[kW] |
| c_j=x_j/λ_j | prices | 单位实际削减能量奖励，优化结果[货币/kWh] |
| z | factors | 整段共享随机因子，各分量独立 U[-1,1] |
| H | profiles | 归一化、有符号、整段 P/Q 增量形状 |
| F | background_factors | 已含新能源的自然净负荷因子载荷 |
| D | response_factors | 合同相对响应因子载荷 |
| q_j=||D_j||_1 | q | 相对响应的全支持幅度 |
| g_l, β_l | benefit_directions, benefit_offsets | 规划前固定的激励函数 |
| τ | reliable_benefit | 最大化的共同可靠收益下界 |
| ε_l | risk_allocations | 预先分配的违约概率 |
| A,b | device_matrix, device_rhs | 设备实际幅值线性域 |

多个 g 的收益必须有可比较的单位，或在优化前给定明确缩放；程序不按测试表现调整缩放。H 是增量，正负方向遵循净消费约定。

## 随机响应与净负荷

给定 λ_j>0、q_j≤1，实际幅值：

\[
\widetilde x_j=x_j(1+D_jz),\quad x_j=\lambda_jc_j,\quad
(1-q_j)x_j\le\widetilde x_j\le(1+q_j)x_j.
\]

同一合同整段共用该幅值，保留时间能量等式。不同合同可以通过 D 的共同列相关；连续随机幅值不等于二元参与。

\[
Y(x,z)=\mu_0+Hx+M(x)z,\qquad M(x)=F+H\operatorname{diag}(x)D.
\]

因为 E[z]=0、Cov(z)=I/3，

\[
E[Y]=\mu_0+Hx,\quad Cov(Y)=M(x)M(x)^T/3.
\]

展开时包括 F 与响应因子的交叉项；程序不以“随机变量相加”为由删除协方差。默认输出协方差因子与对角，避免 m×m 密集矩阵；小系统可显式请求完整协方差。

## 可靠激励条件

总激励：I_l=β_l+g_l^T Y。设 m_l=E[I_l]，s_l=Std(I_l)，则

\[
m_l(x)=\beta_l+g_l^T(\mu_0+Hx),\qquad
s_l(x)=\|M(x)^Tg_l\|_2/\sqrt3.
\]

要求 P{I_l≥τ,全部 l}≥confidence。固定正 ε_l、Σ ε_l≤1−confidence，实施

\[
m_l(x)-\gamma_l s_l(x)\ge\tau,\quad
\gamma_l=\sqrt{(1-\epsilon_l)/\epsilon_l}.
\]

Cantelli 单边不等式给每个方向违约上界 ε_l，并集界给联合违约上界 Σ ε_l。不要求收益方向独立。单方向 90% 的 γ=3，多方向应分配总风险，不能每个方向各用 10% 后声称联合 90%。

该约束是明确的 SOCP：均值仿射，标准差是仿射映射的二范数。它优化保守下界，不是已知有界分布精确分位数的最优值。小因子数时 Cantelli 可能比全支持最坏界还保守；独立采样用于量化保守程度，不修改解析保证。

### 总激励与增量激励

`benefit_mode='incremental'` 定义同背景配对增量：

\[
I_l^{inc}=g_l^T[Y(x,z)-Y(0,z)]
=g_l^THx+g_l^TH\operatorname{diag}(x)Dz.
\]

其均值和方差不含 μ0、F。背景/新能源随机量仍存在于真实 Y，但在线性配对增量中直接抵消。此模式要求 β=0；不能仍把背景方差加回去。

## 设备与报价约束

\[
0\le x_j\le\lambda_j\bar c_j,\qquad
(1+q_j)x_j\le\bar A_j.
\]

任一设备行 A_r·actual≤b_r 的盒支持精确等价式：

\[
A_rx+\|(A_r\operatorname{diag}(x)D)\|_1\le b_r.
\]

绝对值以线性上图变量编译；同一共享因子作用下没有把每个合同的最坏实现错误地当成互相独立。

原 ResourceDay 的全天/截止任务等式由每列 H 事前检查；功率、累计能量、SOC和可用窗变为上述行。网络电压/热限需要后续 AC，未混称设备约束证明。

## 费用与硬预算

e_j=Δt·Σ_t[-h_j^P(t)]_+ 是每 kW 幅值的削减能量，单位 h。单独计量合同的费用：

\[
C_j=c_je_j\widetilde x_j
=\frac{e_j}{\lambda_j}x_j^2(1+D_jz).
\]

因此期望费用 Σ(e_j/λ_j)x_j²。代码采用凸的硬上界：

\[
\sum_j\frac{e_j}{\lambda_j}(1+q_j)x_j^2\le B.
\]

它是逐合同上界之和，可能保守。作为独立重算，同时报告共享因子下精确盒最坏费用：

\[
C^{box}_{max}=\sum_j\frac{e_j}{\lambda_j}x_j^2
+\left\|\sum_j\frac{e_j}{\lambda_j}x_j^2D_j\right\|_1.
\]

后式用于检查与量化保守性，未作为一般凸约束替换前式。

## 两个优化版本

SOCP：上述约束下 max τ，固定每个物理设备允许的合同类型。独立设备不需二元变量。

MISOCP：新增 y_j∈{0,1}，

\[
x_j\le\bar x_jy_j,\quad\sum_{j\in\mathcal J_i}y_j\le1,
\quad\bar x_j=\min(\lambda_j\bar c_j,\bar A_j/(1+q_j)).
\]

类型互斥来自同设备结算/合同选择，不限制节点分散。未被选择的合同幅值为零；选择了类型但最终零幅值不算实际支付。

第一阶段 max τ。报告最优后，在 τ≥τ1−tol_abs−tol_rel|τ1| 下最小化期望费用。实现最小化其平方根半径，目标单调等价，在零附近比二次目标数值稳定。原始求解状态、上下界、两阶段使用的目标尺度分别保存，不把半径 gap 当货币 gap。

Gurobi `NonConvex=0` 检查凸锥结构。单分量范数精确线性化、零设备行和零因子分量跳过，减少冗余；不裁剪实际响应。第一阶段超时有解时保留有解状态，不宣称最优，不启动第二阶段；两阶段共享总时限。

## 辨识验证边界

上述概率只约束显式 I_l。g_l 的拓扑价值必须由原 RNJ＋WZZT 的独立完整观测验证。原辨识器的交叉替换、全部 R/X 从零共同调权、接受、验证和收缩都不在本锥模型中被偷换为线性收益；它们在执行后原样运行。

若激励达标而最终完整错漏不改善，应审查收益近似与原最终选模环节。该结果不改变“90%激励”概率定义，也不能改写为“90%恢复”。
