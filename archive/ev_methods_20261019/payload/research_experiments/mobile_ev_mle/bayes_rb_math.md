# 方向3阶段2：用未来拓扑后验 KL 估计信息增益

`bayes_rb.py` 在同一个 proper 有限 AC 概率模型中计算后验 KL 估计。原始 `bayes.py`、阶段1动作和阶段1结果保持冻结。两种估计具有相同的期望信息目标；阶段2单独保存源哈希、积分种子、选择和验证结果。

## 1. 定义与信息恒等式

所有概率条件于已经观测的历史 $H$。令 $\pi_h=P(h\mid H)>0$，$q_h(z\mid a,H)$ 是拓扑条件未来预测密度，$q(z)=\sum_h\pi_hq_h(z)$，并定义

$$
\pi_h(z)=P(h\mid H,z,a)=\frac{\pi_hq_h(z)}{q(z)}.
$$

预测密度按完整原子后验质量对共享线路和全时段基线/源路径求和，沿用阶段1的计量白化约定。历史与未来使用同一线路；源的物理位置固定为节点1；未来末端总P/Q包含EV一次。

阶段1的随机信息项为

$$
F(h,z)=\log\frac{q_h(z)}{q(z)}
=\log\frac{\pi_h(z)}{\pi_h}.
$$

对给定 $z$ 按其拓扑后验取条件期望，得到阶段2的新随机信息项

$$
\mathcal K_a(z)
=\mathbb E[F(h,Z)\mid Z=z]
=\sum_h\pi_h(z)\log\frac{\pi_h(z)}{\pi_h}
=\mathrm{KL}\{\pi(\cdot\mid H,z,a)\|\pi(\cdot\mid H)\}.
$$

因此二者期望相等：

$$
\begin{aligned}
U(a)&=\mathbb E_{h,Z}F(h,Z)
=\mathbb E_{Z\sim q}\mathcal K_a(Z)\\
&=\mathcal H(\pi)
-\mathbb E_{Z\sim q}\mathcal H\{\pi(Z)\}.
\end{aligned}
$$

给定 $z$ 的 $\mathcal K_a(z)$ 衡量该观测使拓扑概率分布改变多少；平均值 $U(a)$ 衡量实验的预期信息。理论上

$$
0\le\mathcal K_a(z)\le\max_h[-\log\pi_h],\qquad
0\le U(a)\le\mathcal H(\pi).
$$

单个观测的后验 KL 可以超过历史熵，例如对原先低概率拓扑提供强证据的观测。熵上界约束期望信息，有限MC估计越界时保留数值和标准误。双精度的微小负KL舍入残差也保留。

## 2. Rao--Blackwell 方差分解及分层条件

先考虑普通独立同分布的联合混合抽样：每次先抽 $h\sim\pi$，再抽 $Z\sim q_h$。假定 $F(h,Z)$ 二阶可积。由全方差公式，

$$
\operatorname{Var}\{F(h,Z)\}
=\operatorname{Var}\{\mathcal K_a(Z)\}
+\mathbb E_Z\operatorname{Var}\{F(h,Z)\mid Z\}.
$$

于是用 $\mathcal K_a(Z)$ 替换 $F(h,Z)$ 的同样本量普通MC均值，方差不增加。若给定 $Z$ 时信息项仍随拓扑变化，且这种观测有正概率，则改进严格。这是这里的 Rao--Blackwell 结论。

实际代码采用每个拓扑固定 $M_h$ 个样本的分层方案。在拓扑 $h$ 内先抽其条件原子，再抽标准 Gaussian 计量噪声，记结果为 $Z_{hm}$。新估计为

$$
\widehat U_{\rm RB}(a)=
\sum_h\pi_h\frac1{M_h}\sum_{m=1}^{M_h}\mathcal K_a(Z_{hm}).
$$

因为 $\sum_h\pi_hq_h=q$，它仍满足 $\mathbb E\widehat U_{\rm RB}=U$。不同拓扑strata独立时，

$$
\operatorname{Var}(\widehat U_{\rm RB})
=\sum_h\frac{\pi_h^2}{M_h}
\operatorname{Var}_{Z\sim q_h}\{\mathcal K_a(Z)\},\qquad
\widehat{\mathrm{SE}}_{\rm RB}^2
=\sum_h\frac{\pi_h^2s_{\mathcal K,h}^2}{M_h}.
$$

阶段1分层方差用 $\operatorname{Var}_{q_h}\{F(h,Z)\}$ 替换上式的条件方差。普通联合抽样的全方差分解并不直接给出这两个固定分层方差的大小顺序；改进幅度需按当前配置测量。等先验、对称二元Gaussian算例的两个strata具有相同条件分布，普通方差顺序在该算例的等分配分层方案中成立。

## 3. 与熵减少估计并列核验

定义单观测熵减少为

$$
B_a(z)=\mathcal H(\pi)-\mathcal H\{\pi(z)\}.
$$

后验 KL 与熵减少的逐点差为

$$
\mathcal K_a(z)-B_a(z)
=-\sum_h\pi_h(z)\log\pi_h-\mathcal H(\pi).
$$

使用 $\mathbb E_Z\pi_h(Z)=\pi_h$，该差的期望为0。等拓扑先验时，两项逐点相等；不等先验时只要求期望相等。代码用同样本差值计算等式检查及其标准误。

动作 $a,c$ 共享原子抽样和标准噪声。配对信息增益和标准误为

$$
\widehat G_{a,c}=\sum_h\pi_h
\overline{\mathcal K_a(Z_h(a))-\mathcal K_c(Z_h(c))},
\quad
\widehat{\mathrm{SE}}_{a,c}^2
=\sum_h\frac{\pi_h^2}{M_h}s_{\mathcal K(a)-\mathcal K(c),h}^2.
$$

差值方差保留共同随机数的协方差。相比把两个单动作方差相加，它直接反映动作比较的数值分辨率。

## 4. 实现与独立验证

`rb_eig_scores(posterior, future_means, samples_per_topology=2048, seed=0, batch_size=256)` 返回和阶段1兼容的 `eig/eig_se`、`entropy_reduction/entropy_se`、等式差值、stratum摘要及 `paired_comparisons`。`estimator` 字段标明 `posterior_KL_Rao_Blackwell`。

`pointwise_topology_kl(posterior, future_means, values, batch_size=256)` 返回单点或批量的完整后验拓扑 KL。有限密度核调用已冻结 `bayes.py` 的 log-sum-exp 实现；极小拓扑仍有条件抽样和稳定对数质量。`bayes_rb.py` 独立记录源文件哈希，同时将 `bayes.py` 视作依赖。

独立测试用标量 Gaussian 密度和一维积分核验：

- 等先验与不等先验的RB均值、分层标准误和熵等式。
- 拓扑后验KL的逐点定义、标签置换不变性，以及单点KL超过历史熵的合法情形。
- 同预测混合分布与单拓扑情况下的0信息。
- 普通联合抽样的全方差分解及其严格下降。
- 对称Gaussian算例32组独立种子的分层MC实测方差下降。
- 共同随机数动作差值的独立积分与标准误。
- 批次/动作顺序不变性、极小质量保留和微小负舍入残差保留。

## 5. 阶段2验证协议

阶段1主动作来自2048样本的第一个积分种子，其他两个种子提供固定动作的独立数值验证。三个弱案例的阶段1主EIG约 $.0209$--$.0226$ nat，主报告对恒充的增益为 $.00236$--$.00430$ nat。早充与晚充的最大化选择对积分噪声敏感；阶段1冻结动作及其独立验证结果作为原始记录保留。

阶段2单独对三个弱历史运行每拓扑2048和8192样本、各三个独立积分种子。主选择和固定动作验证的种子用途事先声明：若第一个种子选动作，其余两个种子验证该固定动作，避免用同一最大化样本报告选中动作的收益。比较RB、熵估计、配对标准误、独立验证收益及跨种子选择稳定性。动作差仍与数值误差同量级时保留并列。

阶段2使用新源哈希和独立的最终校准/评价噪声流，复用相同历史以比较数值设计方法。有限 family 的后验与null检验按各自统计含义报告；该阶段仍以预先声明的48个原子和固定偏差为概率模型。

## 6. 连续 proper 先验的下一阶段

当前48原子结果在每个拓扑内几乎集中到一个线路/路径原子。连续扩展先回答这种集中对先验分辨率是否稳定，然后扩展网络。

第一步采用一至四维的 proper 连续线路先验参照算例：独立Rscale/Xscale以及独立内部R/X，保留两条路径。用自适应积分建立高精度 evidence/预测参照，检查后验峰值位于节点之间、内层积分加密及EIG变化。它仍是有比例结构的线路族，参数域和先验正常数需明示。

第二步在同一三末端网络上设置逐边独立或明确相关的proper log-R/log-X先验、基线负荷过程先验、同一节点的源P/Q过程先验及共享偏差校准分布。完整预测为

$$
q_h(y_N\mid H,a)=\int
L_N(y_N\mid h,\beta,b,\ell_N+u(a),s_N)
p(\ell_N,s_N\mid\ell_H,s_H)
p(\beta,b,\ell_H,s_H\mid h,H,C)\,d\omega.
$$

该积分保留共享线路、校准一次、源作用点和P/Q/V全通道关联。概率支持应与声明的高电压AC工作支路相容，数值求解失败单独记录。

内层连续积分需独立于外层噪声积分检查。自适应积分或有正确prior/proposal密度比的importance方法可以使用局部拟合作proposal中心，后验权重仍由proper先验与完整似然决定。增加内层节点/粒子数，核对 evidence、后验矩、预测和EIG稳定性；外层标准误仅描述外层积分精度。ESS表示权重集中，MCMC诊断表示已运行链的行为，均须与积分参照或多峰检查一起使用。

## 来源

Hainy et al. (2022), *Optimal Bayesian design for model discrimination via classification*, §2，PDF第3--4页，式(1)--(5)，支持proper模型/参数先验、预测证据、期望熵和误分类效用的框架。[全文](https://api.repository.cam.ac.uk/server/api/core/bitstreams/8888df0c-75a6-4448-b59b-b47ef340858e/content)。本节的后验KL表达、普通/分层方差区分、AC验证协议和连续先验分阶段方案为自行推导。
