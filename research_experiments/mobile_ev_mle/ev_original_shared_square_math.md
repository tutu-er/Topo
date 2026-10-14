# 用共享功率平方压缩见证主问题

本方案针对下一版主问题；当前冻结策略和旧结果保持原样。只读核查见 evidence/ev_original_domain/pilot_v2/shared_square_structure_audit.json.gz。

## 1. 符号与原目标

- \(u_{tj}\)：EV 在时段 \(t\)、地点 \(j\) 的网侧有功功率，单位 kW；充电为正，放电为负。依次排成 \(u\in\mathbb R^K\)，当前 \(K=6\times3=18\)。
- \(\xi\in[L,H]\)：竞争拓扑的线路、历史状态和未来状态参数，使用完整的原绝对参数盒。
- \(\xi_0,S\)：展开点与功率进入未来总 P 的映射；局部变量 \(z=\xi-\xi_0-Su\)。
- \(c,B,A_0,A_k\)：冻结的白化残差系数；白化表示每个计量误差除以其标准差。
- \(i\)：一条合法见证，即对全部可执行 \(u\) 都满足参数盒的显式竞争解释 \(\xi_i(u)=b_i+T_i u\)。

冻结代理为

\[
q(u)=\min_{L\le\xi\le H}
\left\|c+Bu+\left(A_0+\sum_{k=1}^K u_kA_k\right)(\xi-\xi_0-Su)\right\|^2.
\tag{1}
\]

主问题需要覆盖全部动作的上界；内层仍在原域求 QP。给最小化问题一条合法见证，其误差 \(f_i(u)\) 满足 \(q(u)\le f_i(u)\)。

## 2. 为什么当前见证只剩 \(u^2\)

当前优先采用 \(T_i=S\)。于是 \(z_i=b_i-\xi_0\) 为常数，见证残差恰为

\[
r_i(u)=a_i+C_i u,\qquad
a_i=c+A_0z_i,\qquad C_{i,:,k}=B_{:,k}+A_kz_i.
\tag{2}
\]

式 (2) 是冻结代理的代数恒等式。还需两个条件：

1. AC 方程逐时独立，计量白化不混合不同时段。每列 \(C_{i,:,tj}\) 只影响该时段的未来量测行，历史量测行不受 \(u\) 影响。因此，不同时段的列内积为零。
2. 一辆 EV 同一时段至多在一个地点接入。路径和接入约束给出 \(u_{tj}u_{tl}=0\)，\(j\ne l\)。有符号充放电也满足该性质。

记 \(\kappa_i=a_i^\top a_i\)、\(\ell_i=2C_i^\top a_i\)、\(d_{ik}=C_{i,:,k}^\top C_{i,:,k}\ge0\)。展开平方后，跨时项的系数为零，同槽异地点项的乘积为零，故在可执行动作域上

\[
\boxed{f_i(u)=\|a_i+C_i u\|^2
=\kappa_i+\ell_i^\top u+\sum_{k=1}^K d_{ik}u_k^2.}
\tag{3}
\]

同槽异地点内积可以很大，消去依据是操作约束。共享线路参数也不破坏证明：对这一条见证，线路参数已固定在 \(b_i\) 中。若 \(T_i\ne S\)、动态网络或跨时白化存在，需要保留相应项。

## 3. 所有见证共用一组平方上弦

在区间 \([\alpha,\beta]\) 上，平方函数连接线为

\[
s(u)=(\alpha+\beta)u-\alpha\beta,\qquad
s(u)-u^2=(u-\alpha)(\beta-u)\in[0,(\beta-\alpha)^2/4].
\tag{4}
\]

对每个 \(u_k\) 分段，用二进制变量选择区间，使 \(s_k\) 等于该段连接线。由于 \(d_{ik}\ge0\)，

\[
f_i(u)\le f_i^+(u):=\kappa_i+\ell_i^\top u+\sum_k d_{ik}s_k(u_k),\qquad
0\le f_i^+(u)-f_i(u)\le\sum_k d_{ik}h_k^2/4,
\tag{5}
\]

其中 \(h_k\) 是最大分段宽度。所有竞争对与见证共用 \(s_k\)，每条见证只新增线性约束。18 控制各分 4 段，至多 72 个平方分段二进制变量；固定地点仅 6 个活动控制，至多 24 个。路径、SOC 与充放电互斥变量继续保留。

竞争对 \(b\) 的历史基准 SSE 为 \(J_{b,H}\)，可加

\[
\gamma_b\le\tfrac12\bigl(f_i^+(u)+E_i-J_{b,H}\bigr)
\quad\text{对该对全部见证 }i.
\tag{6}
\]

\(E_i\ge0\) 是数值保护量。最大化已冻结的加权平均 \(\sum_b w_b\gamma_b\)，或加 \(\eta\le\gamma_b\) 后最大化 \(\eta\)，都是 MILP。每个 \(f_i^+\ge f_i\ge q\)，有限见证取最小后仍为代理目标上界，求解器最大化 best bound 的方向保持正确。新动作仍需原域 QP 回算下界。

式 (5) 仅控制平方分段误差，有限见证与内层最优解释还有差距，需要继续添加 QP 解释。这里减少 MILP 表达规模，原 max 问题的非凸性和代理与完整 AC 的差别仍在。

## 4. 浮点保护

令 \(\bar u_k=\max(|u_k^-|,|u_k^+|)\)。跨时内积的浮点值为 \(\widehat G_{kl}\)，内积误差界为 \(\delta G_{kl}\)，可以加

\[
E_{\mathrm{cross}}=
2\sum_{t(k)\ne t(l),\,k<l}
(|\widehat G_{kl}|+\delta G_{kl})\bar u_k\bar u_l.
\tag{7}
\]

因此微小的非零跨时项也被向上覆盖。同槽乘积归零基于严格可行域；对求解器近似可行点，另行重算实际跨项。

长度 \(m\) 的内积可用标准舍入界 \(\gamma_n=n\epsilon/(1-n\epsilon)\)，\(\epsilon\) 为单位舍入误差，保守取 \(n=2m+8\)。有

\[
\begin{aligned}
|\widehat\kappa-\kappa|&\le\gamma_n\sum_r a_r^2,\\
|\widehat\ell_k-\ell_k|&\le2\gamma_n\sum_r|a_rC_{rk}|,\\
|\widehat d_k-d_k|&\le\gamma_n\sum_r C_{rk}^2,\\
|\widehat G_{kl}-G_{kl}|&\le\gamma_n\sum_r|C_{rk}C_{rl}|.
\end{aligned}
\tag{8}
\]

前三项分别乘 \(1,\bar u_k,\bar u_k^2\) 后求和，得到 \(E_{\mathrm{coef}}\)。生成式 (2) 的误差也需计入。若 \(|\Delta a|\le e_a\)、\(|\Delta C|\le e_C\)，令 \(e=e_a+e_C\bar u\)、\(M=|a|+|C|\bar u\)，则

\[
\left|\|r_i\|^2-\|\widehat a_i+\widehat C_i u\|^2\right|
\le2M^\top e+\|e\|^2=:E_{\mathrm{form}}.
\tag{9}
\]

用 \(E_i=E_{\mathrm{cross}}+E_{\mathrm{coef}}+E_{\mathrm{form}}\) 向上放宽式 (6)。该保护基于保存的冻结系数和通常浮点模型。有限差分导数和 AC 线性化误差另外由完整 AC 同见证检查处理。新表示没有用残差 QR 压缩，原 QR 诊断界不能直接套用。

## 5. 已保存数据的独立核查

pilot_v2 的 4 个历史、19 个展开轮、948 条见证全部满足 \(T_i=S\)，历史控制列、跨时 Gram 与跨时绝对支撑重叠都严格为零。同槽不同地点 Gram 最大为 72.015，因此按小系数删除并不成立；全部保存动作的相应乘积严格为零。

4620 次见证与保存动作组合中，完整冻结残差平方与式 (3) 最大差为 \(9.095\times10^{-12}\)，保守向上浮点保护最大为 \(1.461\times10^{-8}\)。这些是只读算术核查，尚未实际求解新主问题。

下一版先补标量、负功率、路径活动边界、非零跨时反例，再在开发历史上按同一模型比较 incumbent、best bound、耗时和原域 QP 下界。当前主12历史的策略与结果不回写。
