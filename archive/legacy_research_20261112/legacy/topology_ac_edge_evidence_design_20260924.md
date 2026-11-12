# 拓扑正确性证书与交流潮流单边证据：对两部分研究构思的数学评估

日期：2026-09-24。

本文回应两个问题：（1）能否构造满足即保证拓扑正确的指标；（2）比较增删线前后、分别最小化交流潮流 PQV 误差的差值，能否衡量边的不确定度并关联置信度。

**判断：两部分可以统一。先以 AC 约束下的最优拟合误差定义结构证据，再以经过校准的观测误差集合或统计检验，把证据转化为可排除的拓扑与可确认的边。原始差值是有价值的统计量，但不是天然的正确概率。**

证据标记：文献结论标为【文献】；从明确假设推出的结论标为【本文条件性推导】；尚未实现验证的路线标为【研究设计】。本文未修改算法、未复现论文实验、未运行 AC 识别实验；下文数值仅为解释公式的算例。

## 1. 第一部分：什么意义下能够“一定没问题”

### 1.1 NJ 的保证也是有条件的

【文献 R1】经典 NJ 的 Atteson 半径结论是：真距离来自满足相应条件的加性树，且估计距离的最大误差小于最短真边长的一半时，NJ 恢复真树拓扑。它不是“NJ 输出了一棵树，因此这棵树正确”。

必须区分：

- 树形、径向、正阻抗、四点残差小：结构或拟合一致性。
- 已知真距离误差上界，并证明误差不能改变分裂：条件性正确证书。
- 噪声误差界以至少 1−α 的概率成立：相应的错误发证概率至多 α。

高斯噪声具有无界尾部，有限样本下不同非退化高斯模型具有重叠支持。若没有额外硬约束，通常不能从一个有限观测值无条件证明物理拓扑必真；可以证明“在给定模型与误差界成立时正确”，或给出错误率受控的发证程序。

### 1.2 一个更贴近用户需求的、可计算的树证书

【本文条件性推导】设目标是同一组叶标签上的约化无根树，真实距离 d* 为树度量。输出一棵所有内部边长度为正的二叉树 That，拟合其距离 d_That，计算

$$
\rho=\|\widehat d-d_{\widehat T}\|_\infty,\qquad
\widehat f=\min_{e\in E_{\rm internal}(\widehat T)}\widehat\ell_e.
$$

假设已有针对真实距离的联合误差保证

$$
\|\widehat d-d^\star\|_\infty\le\varepsilon.
$$

则一个充分条件是

$$
\boxed{\widehat f>2(\varepsilon+\rho).}
$$

**证明。** 三角不等式给出 ||d_That−d*||∞≤δ:=ε+ρ。候选树任意 quartet 的小和与两个大和之差等于两倍 quartet 中间路径长度，至少 2 fhat。每个两距离和的误差至多 2δ，两和之差误差至多 4δ。因此 2 fhat>4δ 保持每个候选 quartet 的严格分裂。相同叶集合上的所有 quartet 唯一确定二叉树拓扑，故两树同拓扑。

这里的 ε 与 ρ 完全不同：ρ 可以从拟合结果计算；ε 必须从量测噪声、回归误差、根参考误差、P/Q 输入误差、线性化偏差等获得。不能拿很小的 ρ 代替 ε。若真树存在多分叉或零长度内边，则应保留未解析结构；不能用硬二叉输出强行制造证书。对有根树须包含已知根参考/根距离，单独叶间距离通常不定位根。隐藏二度节点及不可见支路需要先规定可辨识的约化对象。

同一证明还给出逐边结论：候选内部边长度大于 2(ε+ρ) 时，其 split 必为真树 split；将未通过的内边收缩可形成有保证的部分分辨树。这个候选后验证书适用于任何候选生成算法，不能与 Atteson 原始前验半径或 NJ 的单边半径混称。

例如 ε=0.10、ρ=0.02，则阈值为 0.24；候选最短内边 0.30 满足此充分条件，0.20 则不能由此发证。不能发证不意味着拓扑错误。

### 1.3 直接在 AC 模型上定义正确性证书

令观测集合为 D，合法拓扑族为 G，η 收集线路参数、每时段状态、真实注入、根节点量测自由度以及有界模型偏差。定义预测集合

$$
\mathcal M_T=\{h_T(\eta):\eta\in\Theta_T,\ {\rm AC\ constraints\ hold}\}.
$$

对于给定误差范数与预算 ε，

$$
r_T(D)=\inf_{\mu\in\mathcal M_T}\|D-\mu\|_W,\qquad
\mathcal C_\varepsilon(D)=\{T:r_T(D)\le\varepsilon\}.
$$

【本文条件性推导】若真实拓扑属于 G，真实物理响应属于 M_T*，且真实观测误差确不超过 ε，则 T*∈Cε。因此：

$$
\boxed{
r_{\widehat T}\le\varepsilon,\qquad
\inf_{T\not\sim\widehat T}r_T>\varepsilon
}
$$

足以确认 That 的可辨识等价类。符号 T~That 用于隐藏节点、串联细分等观测等价情形；若所有物理节点和边均可辨识，直接比较 T≠That。

这个证书的关键是**排除全部不等价替代拓扑**。只检验若干候选、所有单边邻居或 AC 局部求解失败，都不等于排除了全部替代。

## 2. 第二部分：应当比较什么“最小 PQV 误差”

### 2.1 同时估计 P、Q、V，而不是让量纲决定结果

观测 z_t=[P_t^m,Q_t^m,|V_t|^m,…]。固定拓扑 T，建议定义

$$
J_T=
\inf_{\theta,\{x_t,s_t\}}
\sum_t
[z_t-h_T(x_t,s_t,\theta)]^\top
\Sigma_t^{-1}
[z_t-h_T(x_t,s_t,\theta)],
\quad F_T(x_t,s_t,\theta)=0.
$$

这里 θ 是跨时段共享的线路参数；x_t 是状态；s_t 是真实注入等时变隐变量；F_T=0 表示 AC 潮流、KCL 等约束。相位参考、已知根量测与参数域必须明确。若存在时间相关性，应堆叠全部量测并用完整协方差 Σ，而非假装各时段独立。

具体要求：

1. P/Q/V 残差按可信噪声协方差加权。单纯标幺化不等于统计白化。
2. 若 P/Q 也有误差，应纳入误差变量模型；把量测 P/Q 固定为准确输入通常只形成条件电压预测误差。
3. 所有比较拓扑采用相同测量集合、噪声口径、线路参数范围、根处理方式。
4. 线路参数共享，不能为每个时刻独立调阻抗以吞掉结构误差。
5. 若 Σ 依赖拓扑/参数，完整高斯 −2 log likelihood 还包含 log det Σ；不能只比较二次项。
6. 若损失含稀疏、线路数或阻抗先验惩罚，要把惩罚和观测似然分开解释。用于防止过拟合的惩罚差不是纯数据证据。
7. 若平方电压由有噪声的幅值变换而来，其误差会有偏且非简单同方差高斯；必须按实际变换建模或明确近似。

### 2.2 固定其他边的比较，与真正的边证据不同

用户最直接的量为

$$
\Delta(T_1,T_0)=J_{T_0}-J_{T_1}.
$$

正值支持 T1，负值支持 T0。这首先是“这两个模型之间”的证据；若其他边固定错了，所比较边可能只是补偿错误。

更强的定义是对边状态进行全局剖面优化：

$$
J_e^{(0)}=\inf_{T\in\mathcal G:e\notin E(T)}J_T,\qquad
J_e^{(1)}=\inf_{T\in\mathcal G:e\in E(T)}J_T,
$$

$$
\boxed{D_e=J_e^{(0)}-J_e^{(1)}.}
$$

其含义是：允许其余拓扑、参数与状态重新调整后，最好的“不含 e”模型与最好的“含 e”模型相差多少。若节点身份不确定，可把 e 替换为一个终端 split/clade，用完全相同的方法分析结构特征。

- D_e>0：拟合证据偏向有边。
- D_e<0：偏向无边。
- D_e 接近 0：两类模型仍难区分。
- 绝对值大：相对分离强；还需看绝对拟合、统计校准和计算误差。

若 That 是合法族中真正的全局最优图，则

$$
m_e=\inf_{T:z_e(T)\ne z_e(\widehat T)}J_T-J_{\widehat T}=|D_e|.
$$

它是最优状态被翻转的代价，丢失了 D_e 的符号。若 That 只是局部解，m_e 可能为负，说明存在更好的相反状态模型。

### 2.3 树中的一删一增，以及 MST 类比的边界

相同节点上的连通树都恰有 n−1 条边，所以两个不同树不可能只差一次单边增或删。合适的邻域是

$$
T'=T-e+f,
$$

其中删 e 形成两个连通块，f 跨越该割。可以先算

$$
m_e^{\rm swap}=\min_{f:\,T-e+f\ {\rm is\ allowed}}[J_{T-e+f}-J_T].
$$

如果还存在隐藏节点重构，局部操作应包括收缩、分裂、NNI 等，并规定约化树意义。若系统允许环网、孤岛、多电源模式，则可以比较单纯增删，但必须改变合法运行模型，不能把删掉负荷支路后的不可供电视为精妙的统计识别结果。

关键图论区别：对固定的可加边权，MST 的环交换条件可以形成全局最优证书；AC 重新拟合后 J_T 依赖整张网，通常不满足可加边权结构。**所有单次换边都不改进，一般只说明该邻域局部最优。** 如果要把局部换边提升成全局证书，需要另证交换性质，不能直接移植 MST 结论。

此外，“增线训练误差必不增”只在两模型真正嵌套、旧模型可由新模型的允许参数恢复时成立。新增已知固定导纳线路、改变连通/径向约束等比较未必嵌套。

## 3. 将两部分统一：绝对相容阈值与边剖面差值

### 3.1 一个不依赖 Wilks 的保守有限样本证书

【本文条件性推导】将全部真实量测堆叠为 z∈R^N，并假设

$$
z=h_{T^\star}(\eta^\star)+\epsilon,\qquad
\epsilon\sim N(0,\Sigma),\qquad \Sigma\succ0
$$

且 Σ 已知、真实 η* 属于允许参数域、AC 与量测模型正确。定义 J_T 为完整 Mahalanobis 残差的全局下确界，取

$$
\kappa_\alpha=\chi^2_{N,1-\alpha},\qquad
\mathcal C_\alpha=\{T:J_T\le\kappa_\alpha\}.
$$

注意这里是**原始量测向量的维数 N，不扣除拟合参数个数**。其覆盖保证不是把非线性最小残差直接当成卡方，而是来自

$$
J_{T^\star}
\le
[z-h_{T^\star}(\eta^\star)]^\top\Sigma^{-1}
[z-h_{T^\star}(\eta^\star)]
=
\epsilon^\top\Sigma^{-1}\epsilon
\sim\chi^2_N.
$$

因此严格得到

$$
\boxed{\Pr(T^\star\in\mathcal C_\alpha)\ge1-\alpha.}
$$

这是将观测噪声椭球反演到拓扑空间。N 是预先固定的量测实维度；数据驱动删选量测和反复查看后择时停止需要另行校准，不能直接沿用此固定样本保证。无需 AC 线性化、无需参数处于内部、无需拟合后残差精确为 χ²_{N−k}；代价是可能保守。候选输出可用同一数据选择，因为保证先建立在整个 Cα 上。

Σ 若未知，不可把当前拟合残差估计的 Σ 无条件代入；需要外部校准或联合不确定性集合。若 Σ 奇异，必须在已知支持子空间上工作、同时约束零噪声方向，再使用相应秩。若模型有偏差，必须将真实偏差纳入允许集合，否则覆盖不成立。正确完整联合高斯 Σ 可以处理相关噪声；它不是靠增加一个“有效样本数”自动修复。

### 3.2 边状态与模型相容性的四种结论

在极小值能达到、或使用真实可行见证确认相容的条件下：

| 不含边最优误差 J_e^(0) | 含边最优误差 J_e^(1) | 合理结论 |
|---|---|---|
| >κ | ≤κ | 所有相容拓扑都含 e：可确认存在 |
| ≤κ | >κ | 所有相容拓扑都不含 e：可确认不存在 |
| ≤κ | ≤κ | 两种状态均相容：未分辨 |
| >κ | >κ | 整个模型族不相容：检查模型/数据/阈值，不发边证书 |

如果原假设“无边”根本不在规定的合法图族中，确认“有边”只是该先验结构约束的结果，应标明并非新增量测证据。

**同一个共同 Cα 导出的所有边结论同时有效**：事件 T*∈Cα 发生时，这些结论都正确。因此

$$
\Pr(\text{至少一条被确认的边状态错误})\le\alpha.
$$

这里不需要对每条边另做 Bonferroni；若改为分别构造许多单边检验或区间，则须另处理多重比较。

与用户的差值直接连接：若当前图含 e 且是全局最优，则

$$
J_e^{(0)}=J_{\widehat T}+m_e.
$$

故在 J_That≤κ 的基础上，

$$
\boxed{m_e>\kappa-J_{\widehat T}}
$$

足以排除所有不含 e 的相容图。这就是“拟合差值转为边证书”的一种严谨答案：**差值需要跨过剩余噪声预算，而且比较的是完整相反状态族。**

### 3.3 相同差值可以得到完全不同的结论

仅作阈值示例，设已校准 κ=120：

| J_e^(1) | J_e^(0) | D_e | 结论 |
|---:|---:|---:|---|
| 100 | 135 | 35 | 可确认有边 |
| 60 | 95 | 35 | 两种状态均相容 |
| 140 | 175 | 35 | 两类均不相容，应检查模型 |

所以将所有边的 D_e 排序，再把最前的边称为“95% 可信”，缺少统计依据。不同边的激励、噪声放大、替代连接数量和参数可调空间都不同。

### 3.4 全图证书

在共同 Cα 非空时，只要全部可能边状态都被唯一确定，或直接证明 Cα 只含一个目标等价类，就能发整图证书。保证是

$$
\Pr(\text{发证且拓扑错误})\le\alpha,
$$

不是自动得到 Pr(拓扑错误 | 发证)≤α；后一个条件错误率还涉及发证概率。不能把频率覆盖直接读成这张给定图的后验概率。

## 4. 为什么差值确实与概率存在理论联系

### 4.1 最干净的双模型例子

【本文推导；标准高斯似然关系】先假设两张图及其全部参数/输入都已固定，产生均值 μ0、μ1；两模型共用已知 Σ，并且只有这两个备选。定义 J_b=(z−μ_b)^TΣ^−1(z−μ_b)，D=J0−J1。则

$$
\log\frac{p(z\mid H_1)}{p(z\mid H_0)}=\frac D2.
$$

给定先验 π1、π0，精确有

$$
\Pr(H_1\mid z)=
\operatorname{logistic}\left(\frac D2+\log\frac{\pi_1}{\pi_0}\right).
$$

等先验时 D=0 对应 0.5，D=2log19≈5.889 对应 0.95。该公式回答“差值能否关联概率”：可以，但这是**两个完全指定的概率模型**的结论。

再设

$$
d^2=(\mu_1-\mu_0)^\top\Sigma^{-1}(\mu_1-\mu_0).
$$

在 H1 下 D~N(d²,4d²)，在 H0 下 D~N(−d²,4d²)。等先验按 D 的符号选择时，两个方向的错误概率均为

$$
\Pr(\text{选错})=\Phi(-d/2).
$$

推导来自展开二次项：在 H1 下 D=d²+2 ε^TΣ^−1(μ1−μ0)。因此决定可分辨性的不是未标准化的电压改变量，而是按噪声衡量的响应分离。d=0 时两者在这些观测上不可辨；重复完全无区分力的激励不能解决。

这些公式不能直接用于“参数在同一份数据上都重新拟合”的 profile 差值。

### 4.2 参数未知后的贝叶斯版本

【文献 R6】需要边际似然

$$
p(D\mid T)=\int p(D\mid T,\eta)\,\pi(\eta\mid T)\,d\eta,
$$

再求

$$
\Pr(e\in T\mid D)=
\frac{\sum_{T:e\in T}\pi(T)p(D\mid T)}
{\sum_T\pi(T)p(D\mid T)}.
$$

取最优参数的 exp(−J_T/2) 不是上述积分。它忽略参数空间体积、先验与模型复杂度；若只在候选池求和，得到的是“真图位于该池”的条件模型概率。不能不说明这些条件便称物理边置信度。AC 拟合不天然按边分解，也不能简单相乘边概率得整图概率。

### 4.3 似然比检验与 bootstrap 版本

对于共同已知 Σ，未加惩罚的 J 差值是两倍 profile log likelihood 差值。经典 Wilks 卡方近似需要可辨识、适当嵌套、固定维度、真参数内部等条件。

AC 单边问题常有：

- 径向换边模型非嵌套。
- 线路导纳为零是参数边界；线路消失后，其某些参数不再可辨。
- r/x、隐状态、根电压与其他线路可相互补偿。
- 随时段增长的隐状态带来额外渐近论证。
- 搜索许多候选后才挑一条边，存在选择效应。

【文献 R4】Self–Liang 说明边界处可出现卡方混合，且某些含边界 nuisance 的情形连普通卡方混合都不是。单个规则非负标量的 0.5χ²0+0.5χ²1 不能无条件套给 AC 拓扑。【文献 R5】Vuong 研究非嵌套模型比较，但其“哪个模型更接近数据分布”也不等于哪个是真实物理图，并须满足原论文条件。

【研究设计】可先做参数 bootstrap 校准：在每个被检验边状态的零假设模型下生成模拟量测，复现参数拟合、拓扑搜索、挑边与调参，形成差值的零分布。单纯在原样本上非参数重采样，然后把“同一条边被选中频率”叫正确概率，不解决该问题。

复合零假设只在一个拟合参数点模拟通常是 plug-in 近似，不自动具有一致有限样本水平；要严谨可在零假设 nuisance 集合取最坏尾概率，或用置信区域加误差预算。时间依赖应由生成模型或有理论支持的块方法处理。报告的 p 值不是“边错误概率”；多边检验应控制 FWER/FDR，并说明控制对象。

### 4.4 分裂似然比：不依赖普通卡方极限的另一条路线

【文献 R7】将独立数据分为 A、B，A 用于构造正规预测密度 q_A；检验边状态 b 的复合零假设时，

$$
E_{e,b}=
\frac{q_A(D_B)}
{\sup_{T:z_e(T)=b,\eta}p_{T,\eta}(D_B)}.
$$

若该零假设正确，则 E[E_{e,b}|A]≤1；故 E≥1/α 可按水平 α 排除该状态。分母必须覆盖完整零假设族。

重要实现细节：q_A 必须对验证观测积分为 1。若在 B 上逐时段优化隐藏状态再把“最小残差指数”当分子，它通常不是正规预测密度。可用事先规定的状态分布积分，或在真实精确输入上合法条件化；训练/验证划分本身不修复错误密度。对很多边分别检验仍须处理多重性，也可先反演共同拓扑置信集。

## 5. 非凸优化误差必须进入边不确定度

设两种边状态的 AC 全局最小误差已得到可靠界

$$
L_b\le J_e^{(b)}\le U_b,\quad b=0,1.
$$

则

$$
\boxed{D_e\in[L_0-U_1,\ U_0-L_1].}
$$

区间跨 0，说明计算上尚未证明有边/无边哪方拟合更好。不能把两个局部求解值当作精确最优值，给出看似精确的统计置信度。

发相容性证书需要：

- 一个原始 AC 可行解 U_b≤κ，可以证明这一侧相容。
- 一个有效的全局下界 L_b>κ，可以证明这一侧不相容。
- 局部求解器失败、超时或返回一个较大 U_b，只能记为未解决。

可研究 SDP/SOCP/区间法/全局分支定界提供下界，但下界必须针对同一个原始量测目标和合法域。带额外核范数惩罚的松弛目标本身不能自动当未惩罚问题下界；AC 电压重构后的值也须验证原问题可行。

【文献 R3】Weng 等 2015 年论文把原问题可行目标与凸松弛下界之差称为 estimation confidence。其正文第 4 节明确是在衡量离全局最优值多远。**这种“计算最优性置信”与“拓扑为真的统计概率”不是同一对象。** 该论文恰好提示本研究需要把两者并列报告。

对 R7 的分裂似然比也要注意界方向：若 F_b 为零假设负对数似然最小值，ℓ_q=−log q_A，则 E=exp(F_b−ℓ_q)。使用 F_b 的可靠下界 L_b 得保守 E_lower=exp(L_b−ℓ_q)；局部解给的上界会夸大排除证据。

## 6. 边证据、数据激励与特征电流如何连接

【本文局部推导】在一个工作点附近，将白化后的 AC 响应写作

$$
\widetilde z\approx\widetilde\mu+a\,s_e+B\eta+\widetilde\epsilon.
$$

s_e 是边状态的局部连续扰动代理，Bη 表示其他参数/状态可吸收的变化。令 P_B=B B^\dagger，则有效结构信息为

$$
I_e^{\rm eff}=a^\top(I-P_B)a.
$$

若 a 完全落在其他自由度的列空间中，I_eff=0：边引起的变化可被 nuisance 吞掉，即使原始响应变化大也不代表可辨识。该式来自局部线性投影，仅是 AC 离散跳变的局部诊断，不是全局拓扑定理。

特征电流/逆变器 P/Q 扰动的作用，是改变响应差及其相对噪声，使剩余候选更容易区分。合理探测设计应针对“最难排除的替代图”，允许对方重新拟合其 nuisance，再比较响应；不是单纯把电流幅值做大。

【文献 R8】Cavraro–Kekatos 的逆变器探测研究将拓扑恢复与线路状态验证分别建模，展示主动激励可支持状态检测；其采用近似网络模型，不能直接作为一般非线性 AC 下完整线路参数未知问题的证书。

还需分开三种“边价值”：辨识证据（两种结构可否区分）、预测贡献（样本外误差改进）、运行影响（电压/潮流/控制决策变化）。小差值可能表示不可辨识，也可能表示边无效；它不自动代表运行上不重要。

## 7. 建议形成的研究问题与最小验证路线

【研究设计】把两部分整理为“基于 AC 误差相容集合的拓扑证书与边状态证据评估”：

1. 明确目标图：已知节点线路状态，还是终端观测下的约化树/split。明确径向与环网假设。
2. 固定量测机制：P/Q/V 噪声、根共模、时间依赖、线路参数先验范围与 AC 偏差。先从已知 Σ 的理想实验建立基准。
3. 以共同 J_T 建立原始噪声集合反演，验证全图/边覆盖。它偏保守，但有直接可审计证明。
4. 用全局边状态 profile 定义 D_e；先在小网络完整枚举合法拓扑。固定拓扑的 AC 非凸性仍须单独处理，枚举不自动等于全局 AC 求解。
5. 在中型网络测试单换边、多个换边与全局/松弛界的差距。保留“邻域未找到反例”“找到相容反例”“全族已排除”三种不同标签。
6. 为降低保守性，再比较 nuisance 校准 bootstrap 或合法分裂似然比。若研究目标必须输出后验边概率，再另建完整贝叶斯模型。
7. 在未分辨边上加入可控特征电流/逆变器探测，检验能否收缩相容集合。
8. 报告每条边：状态、带符号 D_e、[L0−U1,U0−L1]、相容状态集合、采用何种统计保证、搜索覆盖范围、计算时间。不要统一压缩成一个未校准百分比。

建议对照实验：

| 维度 | 必须包含的对照 |
|---|---|
| 边模型 | 固定其余拓扑 vs 其余拓扑允许重构；单换边 vs 多换边 |
| 参数 | 已知阻抗 vs 共享未知阻抗；禁止逐时段任意阻抗 |
| 数据 | 仅自然波动 vs 独立主动激励；强/弱负荷变化；同源相关负荷 |
| 噪声 | 已知独立高斯、根共模、时间相关、P/Q EIV、协方差误设 |
| 结构 | 短内边、多分叉近邻、隐藏二度节点、多处同时接错 |
| 计算 | 局部多启动、有效全局界、超时；保留解不出/候选遗漏 |
| 概率 | 图集合覆盖、任意错误发证率、发证率；若输出后验再画概率校准图 |
| 外部有效性 | 真实 AC 数据，三相不平衡与模型失配；不把匹配仿真当现实保证 |

单看平均识别准确率不足以验证证书。应把“错误发证”与“未发证”分开，并同时报告发证率；一个从不发证的程序错误率也为零，但无识别效用。bootstrap 的选择稳定性与实测覆盖率要分栏。

这个大方向已有广义状态估计、拓扑错误假设检验、似然比较和凸松弛文献，不能将“比较增删线后 AC 残差”本身直接认定为新颖贡献。更清晰的潜在贡献是：在所用终端观测与未知参数条件下，建立联合误差—边剖面—计算界—相容拓扑集合的完整保证，并设计有辨识力的主动探测。是否具有论文新颖性仍需按最终模型进一步查重。

## 8. 最相关论文、结果与检索词

以下每项给出精确标题查询；非 IEEE 论文可能只能在 IEEE Xplore 检到引文或相关工作，Xplore 查询链接不表示已在该库收录全文。

### R1. The Performance of Neighbor-Joining Methods of Phylogenetic Reconstruction

Kevin Atteson. Algorithmica 25:251–278, 1999. DOI: 10.1007/PL00008277。[原文或机构来源](https://doi.org/10.1007/PL00008277)。

经典 NJ 的 l∞ 扰动恢复半径。条件使用真树长度及真实距离误差；不是输出树后的自证。

- Google Scholar：`"The Performance of Neighbor-Joining Methods of Phylogenetic Reconstruction"`。[检索](https://scholar.google.com/scholar?q=%22The%20Performance%20of%20Neighbor-Joining%20Methods%20of%20Phylogenetic%20Reconstruction%22)。
- IEEE Xplore：`"The Performance of Neighbor-Joining Methods of Phylogenetic Reconstruction"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22The%20Performance%20of%20Neighbor-Joining%20Methods%20of%20Phylogenetic%20Reconstruction%22)。

### R2. A Note on the Metric Properties of Trees

Peter Buneman. Journal of Combinatorial Theory, Series B 17(1):48–50, 1974. DOI: 10.1016/0095-8956(74)90047-1。[原文或机构来源](https://homepages.inf.ed.ac.uk/opb/homepagefiles/phylogeny-scans/metricproperties.pdf)。

树度量与四点条件的基础。本文候选内边证书是基于此类结构性质的条件推导，不是原文直接给出的 AC 定理。

- Google Scholar：`"A Note on the Metric Properties of Trees"`。[检索](https://scholar.google.com/scholar?q=%22A%20Note%20on%20the%20Metric%20Properties%20of%20Trees%22)。
- IEEE Xplore：`"A Note on the Metric Properties of Trees"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22A%20Note%20on%20the%20Metric%20Properties%20of%20Trees%22)。

### R3. Convexification of bad data and topology error detection and identification problems in AC electric power systems

Yang Weng; Marija D. Ilić; Qiao Li; Rohit Negi. IET Generation, Transmission & Distribution 9(16):2760–2767, 2015. DOI: 10.1049/iet-gtd.2015.0191。[原文或机构来源](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-gtd.2015.0191)。

凸松弛处理 AC 状态估计/拓扑错误中的局部最优问题。第 4 节 confidence 指原始可行目标与松弛下界间隙；不是拓扑后验。全文已核。

- Google Scholar：`"Convexification of bad data and topology error detection and identification problems in AC electric power systems"`。[检索](https://scholar.google.com/scholar?q=%22Convexification%20of%20bad%20data%20and%20topology%20error%20detection%20and%20identification%20problems%20in%20AC%20electric%20power%20systems%22)。
- IEEE Xplore：`"Convexification of bad data and topology error detection and identification problems in AC electric power systems"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Convexification%20of%20bad%20data%20and%20topology%20error%20detection%20and%20identification%20problems%20in%20AC%20electric%20power%20systems%22)。

### R4. Asymptotic Properties of Maximum Likelihood Estimators and Likelihood Ratio Tests under Nonstandard Conditions

Steven G. Self; Kung-Yee Liang. JASA 82(398):605–610, 1987. DOI: 10.1080/01621459.1987.10478472。[原文或机构来源](https://pages.stat.wisc.edu/~larget/Stat998/Fall2015/Self-Liang-1987.pdf)。

边界参数导致非标准似然比极限分布。说明不能默认每条线一个 χ²1；具体混合权重依赖模型。

- Google Scholar：`"Asymptotic Properties of Maximum Likelihood Estimators and Likelihood Ratio Tests under Nonstandard Conditions"`。[检索](https://scholar.google.com/scholar?q=%22Asymptotic%20Properties%20of%20Maximum%20Likelihood%20Estimators%20and%20Likelihood%20Ratio%20Tests%20under%20Nonstandard%20Conditions%22)。
- IEEE Xplore：`"Asymptotic Properties of Maximum Likelihood Estimators and Likelihood Ratio Tests under Nonstandard Conditions"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Asymptotic%20Properties%20of%20Maximum%20Likelihood%20Estimators%20and%20Likelihood%20Ratio%20Tests%20under%20Nonstandard%20Conditions%22)。

### R5. Likelihood Ratio Tests for Model Selection and Non-Nested Hypotheses

Quang H. Vuong. Econometrica 57(2):307–333, 1989. DOI: 10.2307/1912557。[原文或机构来源](https://authors.library.caltech.edu/records/gx67y-we971)。

非嵌套、重叠、嵌套模型及失配条件下的似然比较。链接为作者机构的 1986 工作论文及期刊版本说明；最接近分布不等于真实物理图。

- Google Scholar：`"Likelihood Ratio Tests for Model Selection and Non-Nested Hypotheses"`。[检索](https://scholar.google.com/scholar?q=%22Likelihood%20Ratio%20Tests%20for%20Model%20Selection%20and%20Non-Nested%20Hypotheses%22)。
- IEEE Xplore：`"Likelihood Ratio Tests for Model Selection and Non-Nested Hypotheses"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Likelihood%20Ratio%20Tests%20for%20Model%20Selection%20and%20Non-Nested%20Hypotheses%22)。

### R6. Bayes Factors

Robert E. Kass; Adrian E. Raftery. JASA 90(430):773–795, 1995. DOI: 10.1080/01621459.1995.10476572。[原文或机构来源](https://www.stat.cmu.edu/~kass/papers/bayesfactors.pdf)。

模型证据以参数积分形成边际似然，再结合先验。最大似然残差权重一般不能替代这一积分。

- Google Scholar：`"Bayes Factors"`。[检索](https://scholar.google.com/scholar?q=%22Bayes%20Factors%22)。
- IEEE Xplore：`"Bayes Factors"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Bayes%20Factors%22)。

### R7. Universal Inference

Larry Wasserman; Aaditya Ramdas; Sivaraman Balakrishnan. PNAS 117(29):16880–16890, 2020. DOI: 10.1073/pnas.1922664117。[原文或机构来源](https://arxiv.org/abs/1912.11436)。

分裂似然比给非规则模型有限样本检验；可用 null 最大似然上界保守计算。AC 应用要先构造正规预测密度和正确观测模型。

- Google Scholar：`"Universal Inference"`。[检索](https://scholar.google.com/scholar?q=%22Universal%20Inference%22)。
- IEEE Xplore：`"Universal Inference"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Universal%20Inference%22)。

### R8. Inverter Probing for Power Distribution Network Topology Processing

Guido Cavraro; Vassilis Kekatos. IEEE Transactions on Control of Network Systems, 2019. DOI: 10.1109/TCNS.2019.2901714。[原文或机构来源](https://arxiv.org/abs/1802.06027)。

在近似网络模型下，将拓扑恢复、线路状态验证分别建模。所有母线电压可测时证明探测末端足以精确恢复；其量测/模型条件不同于一般末端 PQV。

- Google Scholar：`"Inverter Probing for Power Distribution Network Topology Processing"`。[检索](https://scholar.google.com/scholar?q=%22Inverter%20Probing%20for%20Power%20Distribution%20Network%20Topology%20Processing%22)。
- IEEE Xplore：`"Inverter Probing for Power Distribution Network Topology Processing"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Inverter%20Probing%20for%20Power%20Distribution%20Network%20Topology%20Processing%22)。

### R9. Bayesian-Based Hypothesis Testing for Topology Error Identification in Generalized State Estimation

Elizete M. Lourenço; Antonio S. Costa; K. A. Clements. IEEE Transactions on Power Systems 19(2):1206–1215, 2004. DOI: 10.1109/TPWRS.2003.821442。[原文或机构来源](https://doi.org/10.1109/TPWRS.2003.821442)。

开关状态组合的贝叶斯假设检验，含先验、Mahalanobis 项和协方差行列式；通过矩阵修改减少反复状态估计。已知参数与可疑集合覆盖是重要前提。核对作者上传全文。

- Google Scholar：`"Bayesian-Based Hypothesis Testing for Topology Error Identification in Generalized State Estimation"`。[检索](https://scholar.google.com/scholar?q=%22Bayesian-Based%20Hypothesis%20Testing%20for%20Topology%20Error%20Identification%20in%20Generalized%20State%20Estimation%22)。
- IEEE Xplore：`"Bayesian-Based Hypothesis Testing for Topology Error Identification in Generalized State Estimation"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Bayesian-Based%20Hypothesis%20Testing%20for%20Topology%20Error%20Identification%20in%20Generalized%20State%20Estimation%22)。

### R10. An Optimization-Based Topology Error Detection Method for Power System State Estimation

Ankur Srivastava; Saikat Chakrabarti; Joao Soares; Sri Niwas Singh. Electric Power Systems Research 209:107914, 2022. DOI: 10.1016/j.epsr.2022.107914。[原文或机构来源](https://research.chalmers.se/publication/529376/file/529376_Fulltext.pdf)。

逐候选线路退出、更新估计并比较残差，矩阵逆引理降低计算。指定量测下 14 节点 6/6、118 节点 47/50 单退出事件辨对。正文核心推导为 P–δ 解耦线性化，排除临界支路；不是完整非线性 AC-PQV 置信证书。机构全文已核。

- Google Scholar：`"An Optimization-Based Topology Error Detection Method for Power System State Estimation"`。[检索](https://scholar.google.com/scholar?q=%22An%20Optimization-Based%20Topology%20Error%20Detection%20Method%20for%20Power%20System%20State%20Estimation%22)。
- IEEE Xplore：`"An Optimization-Based Topology Error Detection Method for Power System State Estimation"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22An%20Optimization-Based%20Topology%20Error%20Detection%20Method%20for%20Power%20System%20State%20Estimation%22)。

### R11. An Adaptive-Importance-Sampling-Enhanced Bayesian Approach for Topology Estimation in an Unbalanced Power Distribution System

Y. Xu; J. Valinejad; M. Korkali; L. Mili; Y. Wang; X. Chen; Z. Zheng. IEEE Transactions on Power Systems 37(3):2220–2232, 2022（在线 2021）. DOI: 10.1109/TPWRS.2021.3121612。[原文或机构来源](https://vtechworks.lib.vt.edu/server/api/core/bitstreams/20c6ad32-d6b6-4142-9924-8c7b2007da7e/content)。

开关与未知 P/Q 联合后验，自适应重要性采样，测试三相不平衡 IEEE 123 节点及 1282 节点。模型和误差统计是前提；采样后验不能与最小残差归一化混同。机构全文已核。

- Google Scholar：`"An Adaptive-Importance-Sampling-Enhanced Bayesian Approach for Topology Estimation in an Unbalanced Power Distribution System"`。[检索](https://scholar.google.com/scholar?q=%22An%20Adaptive-Importance-Sampling-Enhanced%20Bayesian%20Approach%20for%20Topology%20Estimation%20in%20an%20Unbalanced%20Power%20Distribution%20System%22)。
- IEEE Xplore：`"An Adaptive-Importance-Sampling-Enhanced Bayesian Approach for Topology Estimation in an Unbalanced Power Distribution System"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22An%20Adaptive-Importance-Sampling-Enhanced%20Bayesian%20Approach%20for%20Topology%20Estimation%20in%20an%20Unbalanced%20Power%20Distribution%20System%22)。

### R12. Why Neighbor-Joining Works

Radu Mihaescu; Dan Levy; Lior Pachter. Algorithmica 54(1):1–24, 2009. DOI: 10.1007/s00453-007-9116-4。[原文或机构来源](https://arxiv.org/abs/cs/0602041)。

NJ 的 quartet 方法、恢复保证与边半径。Example 11 说明输入 quartet 排序全正确仍不足以单独保证任意规模 NJ 输出正确；本文候选距离后验证书不依赖 NJ 的运行逻辑。

- Google Scholar：`"Why Neighbor-Joining Works"`。[检索](https://scholar.google.com/scholar?q=%22Why%20Neighbor-Joining%20Works%22)。
- IEEE Xplore：`"Why Neighbor-Joining Works"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Why%20Neighbor-Joining%20Works%22)。

## 9. 按研究问题扩展检索

| 目的 | Google Scholar 词条 | IEEE Xplore Advanced Search 词条 |
|---|---|---|
| 残差与线路状态 | `"topology error" "state estimation" "residual" "line outage"` | `("All Metadata":"topology error") AND ("All Metadata":"state estimation") AND ("All Metadata":"residual")` |
| 单边概率证据 | `"power system" "topology" "likelihood ratio" "hypothesis testing"` | `("All Metadata":"topology") AND ("All Metadata":"likelihood ratio")` |
| 参数补偿与混淆 | `"topology error" "parameter errors" "measurement errors"` | `("All Metadata":"topology error") AND ("All Metadata":"parameter error")` |
| 有界噪声认证 | `"topology identification" "set membership" "bounded noise"` | `("All Metadata":"topology identification") AND ("All Metadata":"bounded noise")` |
| 剖面与非规则检验 | `"profile likelihood" "boundary" "nonidentifiable"` | `("All Metadata":"profile likelihood") AND ("All Metadata":"topology")` |
| AC 计算证书 | `"AC state estimation" "semidefinite relaxation" "lower bound"` | `("All Metadata":"state estimation") AND ("All Metadata":"semidefinite") AND ("All Metadata":"topology")` |
| 主动激励区分拓扑 | `"inverter probing" "line status" "detection"` | `("All Metadata":"inverter probing") AND ("All Metadata":"topology")` |

中文扩展：`拓扑辨识 有界噪声 集合辨识`、`广义状态估计 拓扑错误 假设检验`、`拓扑错误 线路参数误差 联合辨识`、`交流状态估计 凸松弛 全局下界`。

文献原有结果、本文条件推导与尚待验证的研究设计已在正文分开。本文给出的是理论方案和检索依据，未声称任何上述概率保证已在当前仓库实现。
