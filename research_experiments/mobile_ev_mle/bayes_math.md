# 方向3：有限 AC 模型的拓扑信息增益

本阶段以48个预先声明的物理原子构成 proper 概率模型。历史后验和未来预测密度通过有限求和计算；未来观测积分使用分层 Monte Carlo。验证顺序是解析 Gaussian 算例、共同有限 AC 算例、现有自由 R/X 联合拟合器的转移检查。

## 1. 物理原子、先验与计量

拓扑标签为 $h\in\mathcal H=\{\{0,1\},\{0,2\}\}$。拓扑 $h$ 的第 $k$ 个原子为

$$
\omega_{hk}=(\beta_{hk},\ell_{hk,1:T},s_{hk,1:T}),\qquad
\beta_{hk}=(r_{hk,e},x_{hk,e})_{e\in E_h}.
$$

$\ell_t=(\ell_t^P,\ell_t^Q)$ 是末端真实基线负荷；$s_t=(s_t^P,s_t^Q)$ 是作用于共同节点1的源项，正号表示耗电。历史 $H$ 和未来 $N$ 使用同一个 $\beta_{hk}$。每条联合基线/源路径覆盖12个历史和8个未来时段；源项可以逐时变化，物理作用点固定。

两个拓扑各有24个原子：Rscale和Xscale独立取 $.85,1.15$，内部支路取弱、中、强三种强度，联合路径取两种。这里的两个比例允许 R/X 分别变化，定义的是有比例结构的有限线路先验；自由逐边 R/X 留给下一阶段的连续拟合转移检查。拓扑先验和原子条件质量满足

$$
\pi_h^0>0,\quad \sum_h\pi_h^0=1,\qquad
w_{hk}^0>0,\quad \sum_k w_{hk}^0=1.
$$

默认 $\pi_h^0=1/2$，各拓扑内24个原子等质量。所有线路、路径、质量和噪声在观察历史前冻结。若扩展到 star，应将重复物理原子合并并相加质量，保留拓扑总质量；这样原子数量不会隐含改变拓扑先验。

已知动作 $a$ 的末端总注入为

$$
p_{it}=\ell^P_{it}+u^P_{it}(a),\qquad
q_{it}=\ell^Q_{it}+u^Q_{it}(a).
$$

EV只在此处加入一次。完整 AC 方程以这些总 P/Q 和节点1的源项计算电压、master P/Q。原始计量向量为末端 P/Q、电压幅值及 master P/Q；均值中的末端 P/Q 也是上述总量。root 电压、计量噪声协方差和校准固定偏差为已知输入。

固定偏差 $b$ 在历史与未来共享。第一阶段采用已校准 $b$ 的点质量概率模型。加入校准不确定性时，可以为 $b$ 设置 proper 有限先验，通过独立校准数据 $C$ 得到

$$
p(b\mid C)\ \propto\ p(b)\,p(C\mid b),
$$

并将其纳入原子；运行数据更新从 $p(b\mid C)$ 开始，校准似然使用一次。master P/Q 和校准数据提供物理参数的观测约束。缺少这些约束时，proper 先验仍可定义后验，但源项/线路的规范自由度会增强先验影响。

## 2. 完整 Gaussian 历史后验

统一白化矩阵为 $W_H=\Sigma_H^{-1/2}$。记

$$
z_H=W_Hy_H,\qquad m_{hk}^H=W_H\mu_{hk}^H,
\qquad D_H=(3n+2)|H|.
$$

代码的 `history_y` 和 `history_means` 就是 $z_H,m_{hk}^H$。白化后的完整 Gaussian 密度为

$$
L_{hk}^H=(2\pi)^{-D_H/2}
\exp\{-\tfrac12\|z_H-m_{hk}^H\|^2\}.
$$

原始计量坐标的密度还乘共同因子 $|\det W_H|$。这个因子影响 evidence 数值，后验质量和信息增益中则相消；API提供 `log_whitening_jacobian` 接收其对数。

设 $\alpha_{hk}^0=\pi_h^0w_{hk}^0$。有限 Bayes 更新为

$$
Z_H=\sum_{h,k}\alpha_{hk}^0L_{hk}^H,
\qquad W_{hk}=\frac{\alpha_{hk}^0L_{hk}^H}{Z_H},
\qquad \pi_h=\sum_kW_{hk}.
$$

算法计算

$$
\log W_{hk}=\log\alpha_{hk}^0-\tfrac12\|z_H-m_{hk}^H\|^2
-\operatorname{logsumexp}_{j,l}
\{\log\alpha_{jl}^0-\tfrac12\|z_H-m_{jl}^H\|^2\}.
$$

每个声明原子的对数质量均保留。极小质量转为浮点概率时可以下溢，输出记录下溢数量，并继续保存对数质量、对数拓扑质量和拓扑条件预测。此更新的概率解释基于声明的有限拓扑、线路和路径先验。

## 3. 未来是联合 Gaussian mixture

未来白化均值为 $m^N_{hk}(a)$，维数 $D_N=(3n+2)|N|$。给定原子时，随机性来自已知独立计量噪声：

$$
Z_N=m^N_{hk}(a)+\varepsilon_N,\qquad
\varepsilon_N\sim N(0,I_{D_N}).
$$

条件化历史后，原子权重为 $W_{hk}$，故

$$
q_h(z\mid H,a)=\sum_k\frac{W_{hk}}{\pi_h}
\varphi_{I_{D_N}}\{z-m^N_{hk}(a)\},
\qquad
q(z\mid H,a)=\sum_h\pi_hq_h(z\mid H,a).
$$

这些密度包括 Gaussian normalizer，使用 log-sum-exp 求和。保持原子不变保证线路在 H/N 中共享；未来均值对该原子的同一源路径求值。条件于原子时误差独立，边缘化后则有

$$
\operatorname{Cov}(Z_N\mid h,H,a)
=I+\operatorname{Cov}_{k\mid h,H}\{m^N_{hk}(a)\}.
$$

共享线路、偏差和路径不确定性因此产生跨通道、跨时段关联。混合分布通常也不是具有上述均值、协方差的单一 Gaussian。

历史 MLE 点代入将参数解释集中于选定点；这里的预测按全部原子后验质量平均。相同协方差 Gaussian 的 $2\mathrm{KL}$ 等于白化均值平方差，这个恒等式适用于原子条件密度。拓扑边缘预测是混合分布，其 KL 通过上述密度计算。

## 4. 信息增益和熵减少的证明

在已经观测的历史条件下，未来拓扑后验为

$$
\pi_h(z,a)=\frac{\pi_hq_h(z\mid H,a)}{q(z\mid H,a)}.
$$

使用自然对数，拓扑信息增益定义为

$$
U(a)=\sum_h\pi_h\int q_h(z\mid H,a)
\log\frac{q_h(z\mid H,a)}{q(z\mid H,a)}\,dz.
$$

代入 Bayes 公式，并使用 $\pi_hq_h=q\pi_h(z,a)$，得到

$$
\begin{aligned}
U(a)
&=\sum_h\pi_h\int q_h(z)
  [\log\pi_h(z,a)-\log\pi_h],dz\\
&=\int q(z)\sum_h\pi_h(z,a)\log\pi_h(z,a)\,dz
  -\sum_h\pi_h\log\pi_h\\
&=\mathcal H(\pi)-\mathbb E_{Z_N\sim q}
  \mathcal H\{\pi(Z_N,a)\}.
\end{aligned}
$$

第一行定义也等于条件互信息 $I(h;Z_N\mid H,a)$，即拓扑条件预测间的加权 Jensen–Shannon 分离。KL非负、条件熵非负给出

$$
0\le U(a)\le\mathcal H(\pi)\le\log|\mathcal H|.
$$

若各 $q_h$ 相同，$U(a)=0$；若仅有一个拓扑，也为0，即使其内部线路原子仍可辨识。单次未来观测的实际熵减少可以为负。无EV观测也可以提供信息，EV比较量为 $U(a)-U(a_{\rm none})$，按实际符号报告。

未来负荷与源项在本模型中按原子路径边缘化。与此不同，未来数据到来后再重新拟合的期望涉及 $\mathbb E\inf_\theta L(Y,\theta)$；实验前固定解释的准则涉及 $\inf_\theta\mathbb E L(Y,\theta)$。前者不大于后者，一般严格不同。例子 $Y\sim N(0,1)$、$L=(Y-\theta)^2$ 的两个值为0与1。EIG使用明确的预测概率模型和后验更新。

## 5. 分层 Monte Carlo 和标准误

对每个拓扑独立生成 $M_h$ 个样本：先按 $W_{hk}/\pi_h$ 抽原子，再抽标准 Gaussian 噪声。令

$$
F_{hm}(a)=\log\frac{q_h(Z_{hm}\mid H,a)}{q(Z_{hm}\mid H,a)},
\qquad
B_{hm}(a)=\mathcal H(\pi)-\mathcal H\{\pi(Z_{hm},a)\}.
$$

两种估计为

$$
\widehat U_F(a)=\sum_h\pi_h\bar F_h(a),\qquad
\widehat U_B(a)=\sum_h\pi_h\bar B_h(a).
$$

有限混合密度由全部原子求和，因此二者对这个有限模型的 $U(a)$ 无偏。不同拓扑 strata 相互独立，固定分层样本量下

$$
\widehat{\mathrm{SE}}_F^2(a)=
\sum_h\frac{\pi_h^2s_{F,h}^2(a)}{M_h},
\qquad
\widehat{\mathrm{SE}}_B^2(a)=
\sum_h\frac{\pi_h^2s_{B,h}^2(a)}{M_h}.
$$

$s^2$ 使用样本方差的 $M_h-1$ 分母。熵等式的差值标准误对同一样本的 $F_{hm}-B_{hm}$ 计算，保留它们的协方差。

所有动作共享条件原子抽样和标准噪声，实现共同随机数比较。动作 $a,c$ 的增益估计及标准误为

$$
\widehat G_{a,c}=\sum_h\pi_h\overline{F_h(a)-F_h(c)},\qquad
\widehat{\mathrm{SE}}_{a,c}^2=
\sum_h\frac{\pi_h^2s_{F(a)-F(c),h}^2}{M_h}.
$$

这个差值方差保留共同随机数的协方差。各拓扑之间仍独立抽样；在拓扑间共享同一噪声会改变分层方差公式。

预算采用每拓扑128、512、2048个样本，以及三组独立积分种子。记录均值、单动作标准误、配对增益标准误和熵等式差值。样本均值的负值保留；信息上界是 $\mathcal H(\pi)$。标准误用于判断计算分辨率，动作分数接近时保留并列及积分预算。设计模拟噪声与真实未来评价噪声使用独立随机流。

## 6. 独立解析验证

构造 $h\in\{-1,1\}$、等先验，$b\sim N(0,\tau_b^2)$、$c\sim N(c_0,\tau_c^2)$。历史观测为 $Y_H=b+\epsilon_H$，未来为 $Y=b+hca+\epsilon_N$。若 $b\mid H\sim N(m_b,v_b)$，则

$$
Y\mid h,H,a\sim N(m_b+hc_0a,s_a^2),\qquad
s_a^2=v_b+a^2\tau_c^2+\sigma_N^2.
$$

设 $\gamma=c_0a/s_a$。两个对称拓扑的条件积分相等，所以

$$
U(a)=\mathbb E_{Z\sim N(0,1)}
\left[\log2-\log\{1+\exp(-2\gamma(\gamma+Z))\}\right].
$$

这个一维式用独立数值积分计算，验证MC均值、非等先验分层方差和共同随机数差值方差。$a=0$ 时为0；$c_0=0$ 时两种预测相同。参数不确定性增加 $s_a$ 并减少模型信息，$\tau_c>0$ 时大功率的有效信噪比趋于 $c_0/\tau_c$。单模型内两个线路均值仍可分离，拓扑信息则严格为0。这分别验证了线路参数信息和拓扑信息的目标。

`test_bayes.py` 还核验完整Gaussian evidence、proper prior、混合密度归一化、原子排列及质量保持的拆分不变性、极小质量保留、单模型边界、负MC值保留、动作顺序及分批计算不变性。有限支持ESS会随原子拆分表示改变；记录它时须同时给出支持定义和原子数。

## 7. 有限 AC 比较、先验敏感性和扩展

共同物理与计量协议见 `three_direction_protocol.md`。每次历史观察先计算同一完整有限后验，再评价11种可执行充电动作以及等长度无EV对照。best constant从可行恒充动作按当前准则选择。真实原子及未来实现噪声仅由测试驱动器使用。

先验敏感性固定全部48个支持点为正质量，分别改变拓扑赔率、弱支路质量及其组合，报告后验、信息增益、配对增益及选址改变。强/弱固定案例描述条件性能；检验概率校准时，应另从所声明的完整先验抽取真拓扑和真原子。候选漏真和偏差错设按模型适配性诊断记录。

有限原子后验和预测完成后，选定动作还要交给现有自由逐边R/X、逐时源项AC联合拟合器，报告取得的局部似然、参数误差和失败。该检查评价有限先验选址向更宽物理参数族的转移。网格细化需要按目标proper连续先验给出权重，并比较 evidence、预测及EIG的稳定性；24个原子本身定义有限模型。当前算法穷尽有限求和，MCMC遍历和连续全域保证不参与本阶段结论。

若 $K$ 是原子数、$A$ 是动作数，历史更新的算术量为 $O(KD_H)$，外层预测积分为 $O(AMKD_N)$。共有48个原子时，混合密度可精确求和并分批计算；没有随机的内层积分或历史硬截断。数值AC失败保留为失败，调整物理支持需要重新冻结模型后再生成数据。

## 8. 接口与来源

- `posterior_weights(history_y, history_means, labels, proper_prior, *, log_whitening_jacobian=0.0)` 返回 `PosteriorWeights`，含原子/拓扑的概率及对数质量、完整白化Gaussian evidence、历史loss、有限支持ESS和 `summary()`。
- `predictive_logpdf(posterior, future_means, values, *, topology=None, batch_size=256)` 返回完整边缘或拓扑条件混合log密度。
- `eig_scores(posterior, future_means_dict, *, samples_per_topology=512, seed=0, batch_size=256)` 返回各动作EIG、标准误、熵减少估计、等式差值、stratum摘要及 `paired_comparisons[baseline]`。该字段中的gain为当前动作减baseline。

Hainy, Price, Restif and Drovandi (2022), *Optimal Bayesian design for model discrimination via classification*, §2，出版PDF第3–4页，式(1)–(5)，给出参数/模型先验、预测证据、期望熵和误分类损失框架。已读上述全文位置。[出版全文](https://api.repository.cam.ac.uk/server/api/core/bitstreams/8888df0c-75a6-4448-b59b-b47ef340858e/content)，[DOI](https://doi.org/10.1007/s11222-022-10078-2)。本文的有限AC路径模型、分层方差、Gaussian解析算例和验证协议为本项目自行推导。

Dette and Titoff (2009), *Optimal discrimination designs*, §2与§4.2，PDF第4页(2.4)、第18页(4.1)，给出固定参考与竞争参数重新拟合的T/KL设计。这里的Bayesian EIG按后验质量边缘化，并以拓扑互信息为目标。[原文](https://arxiv.org/pdf/0908.1912)。
