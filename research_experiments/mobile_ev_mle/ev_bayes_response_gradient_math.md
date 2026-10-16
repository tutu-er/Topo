# Bayes风险和BC的完整方向导数：仅作诊断

本页及 `ev_bayes_response_gradient.py` 不改变已冻结策略。历史后验、线路/状态联合矩固定；动作改变未来AC响应及其传播方差。MILP仍冻结协方差，用AC均值Jacobian构造BC分段代理，完整风险导数目前只用于核对。

## 1. 完整BC导数多出什么

沿一条功率方向写 $u(\alpha)=u_0+\alpha d$，$d$ 的单位为kW；点表示对 $\alpha$ 求导。两候选均值之差为 $\delta=\mu_i-\mu_j$，$S=(\Sigma_i+\Sigma_j)/2$，$v=S^{-1}\delta$。则

\[
\dot B=\underbrace{\tfrac14(\dot\mu_i-\dot\mu_j)^\top v}_{\text{均值项}}+
\underbrace{-\tfrac18v^\top\dot Sv+\tfrac12\operatorname{tr}(S^{-1}\dot S)
-\tfrac14\operatorname{tr}(\Sigma_i^{-1}\dot\Sigma_i+\Sigma_j^{-1}\dot\Sigma_j)}_{\text{协方差项}}.
\]

对二次式与log行列式求微分即得该式。历史概率 $\pi$ 固定时，$C=\sum_{i<j}\sqrt{\pi_i\pi_j}e^{-B_{ij}}$ 的导数为各项 $-\sqrt{\pi_i\pi_j}e^{-B_{ij}}\dot B_{ij}$ 之和。AC均值Jacobian只提供第一项。完整传播 $\Sigma=I+F\bar CF^\top$ 还需要

\[
\dot\Sigma=\dot F\bar CF^\top+F\bar C\dot F^\top.
\]

诊断用完整AC的正负小扰动求此项；它不更新历史线路或源后验。一个反例说明该项有实际含义：若 $\delta(u)=u$、两候选共同标准差为 $e^{2u}$，则 $B=u^2e^{-4u}/8$。在 $u=1$，只算均值得到正导数，完整导数为负。电压差增大仍可能被更快增长的不确定性抵消。

## 2. 原风险的真实梯度

固定历史后，$f_k(y;u)$ 是候选预测密度，$\pi_k$ 为历史概率。MAP错误风险为

\[
R(u)=1-\int\max_k\{\pi_k f_k(y;u)\}\,dy.
\]

若MAP几乎处处唯一且可交换积分和求导，最大值的包络公式给出

\[
\dot R=-\sum_k\pi_k\int_{\{\mathrm{MAP}(y)=k\}}\dot f_k(y;u)dy
=-E[1\{\mathrm{MAP}(Y)=K\}\,\dot{\log f_K(Y;u)}].
\]

Gaussian的固定观测score（log密度对动作的导数）为

\[
\dot{\log f_k(y)}=\dot\mu_k^\top\Sigma_k^{-1}r_k+
\tfrac12\{r_k^\top\Sigma_k^{-1}\dot\Sigma_k\Sigma_k^{-1}r_k-
\operatorname{tr}(\Sigma_k^{-1}\dot\Sigma_k)\},\quad r_k=y-\mu_k.
\]

这是对声明Gaussian风险的完整梯度，可用候选分层抽样估计并报告积分标准误。它是总体积分的梯度，不是某次实际future noise的收益。

另一个核对方式是保持同一标准Normal数组 $\epsilon$，令 $Y_k=\mu_k+L_k\epsilon$、$L_kL_k^\top=\Sigma_k$。其路径导数包含 $\dot Y_k=\dot\mu_k+\dot L_k\epsilon$。对各候选log密度求总导数后，后验满足

\[
\dot p_j=p_j\Big(\dot{\log f_j(Y_k)}-\sum_b p_b\dot{\log f_b(Y_k)}\Big).
\]

在MAP不并列的样本上，风险样本导数为 $-\dot p_{\mathrm{MAP}}$。程序核查该路径估计、共同noise中央有限差分和上述score估计；后两种积分估计的方差不同。等先验、均值为 $\pm a u$、共同标准差 $e^{bu}$ 的解析对照为

\[
R(u)=\Phi(-aue^{-bu}),\qquad R'(u)=-\varphi(aue^{-bu})ae^{-bu}(1-bu).
\]

若 $u_{jt}$ 表示kW，时段长度为 $\Delta t$ 小时，网侧电量的边际辨识价格可定义为 $\lambda_{jt}=-(1/\Delta t)\partial R/\partial u_{jt}$，单位为预期错误率减少/kWh。完整风险梯度具有这个含义，冻结协方差的BC梯度仅为当前代理的价格。正的放电/移动自由度保证同一取得目标可继承父计划；梯度本身不给全局最优保证。

共同平移全部候选均值时，后验和风险不变，完整路径导数为0。零梯度也可能位于二次分离的起点，所以保留二次MILP仍有作用。新诊断限事先指定开发历史0/1的全部 `joint_bc` 父→提案方向；保存零方向和缺失锚点，不按结果选择方向，不接触真实future数据。
