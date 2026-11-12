# 当前主模型与接入方法

## 1. 先固定候选与评分

目标候选记为t，竞争候选记为j。每个候选由RNJ＋WZZT的历史拟合得到逐时段响应矩阵[R_j,X_j]，之后冻结。以下将新日全部节点时段堆叠为向量P和Y，H_j相应表示I_96⊗[R_j,X_j]的时间扩展，均值和协方差也采用相同堆叠顺序。每槽的P先拼有功、再拼无功，末端节点顺序一致，功率采用标幺值。电压响应为观测根电压平方减节点电压平方，单位pu²。

S_j^h是完整旧验证观测下的物理残差平方和，训练数据用于拟合，不重复加入验证评分。新日有m个标量残差，历史有m_h个，N=m_h+m。定义

\[
M_j=\frac{S_j^h+\|Y-H_jP\|^2}{N}.
\]

当前DR严格选模的公共尺度为K>0，并列容差是epsilon=1e-12/K。使目标比竞争者低出该容差的充分条件为

\[
L_j=\|Y-H_jP\|^2-\|Y-H_tP\|^2>\delta_j^{phys},\qquad
\delta_j^{phys}=N\epsilon-S_j^h+S_t^h.
\]

原并列顺序仍由`selection_mse.choose_mse`保留。目标拓扑胜出只需处理不同支撑的竞争者；同支撑不同参数点胜出仍属于同一拓扑成功。固定目标点严格击败所有不同支撑候选是充分条件，可能比真实并列规则保守。零权重、弱权重和预测等价必须单列，不能按真假标签删候选。

## 2. 条件功率与电压模型

名义资源交付记为u。固定协方差的工作模型为

\[
P(u)=\widehat p+Bu+\xi_P,\qquad \xi_P\sim N(0,\Sigma_P),
\]
\[
Y=H_tP(u)+\varepsilon,\qquad
\varepsilon\sim N(0,\Sigma_V),\qquad \xi_P\perp\varepsilon.
\]

widehat p为完整旧日形成的次日功率预测，B把资源交付映射到节点P/Q。Y的生成条件是目标H_t正确，不需要额外预测一条真实电压曲线。该式还假设冻结参数预测可作为剩余误差分布的中心；真结构资格只确认结构前提，参数误差和AC偏差仍在实际验证中保留。

最新40组的原工作法则：96槽/日、15分钟/槽；96训练＋288验证形成4日逐时均值；每功率通道标准差是预测绝对最大值的5%；同节点P/Q共同因素、节点间独立、时间AR(.7)。剩余电压误差为独立对角模型，节点标准差取目标旧验证残差RMS与2sqrt(2)*sigma_V的较大值。该5%是预测功率误差设定，与P/Q表计0.5%噪声不是同一数。

这是一组明确的工作假设，不是实证概率校准。规划协方差固定；实际乘性DR响应、相关偏差和设备饱和由既有执行器生成并另行验证。它们尚未全部进入上述固定协方差保证。

## 3. 两次积分得到二次约束

令s为新日剩余电压误差标准差的最大值，只除以这一个公共标量，保持未加权物理MSE。定义归一化量

\[
z_j=\frac{(H_t-H_j)P}{s}\sim N(d_j(u),C_j),\quad
e=\varepsilon/s\sim N(0,E),\quad
\delta_j=\delta_j^{phys}/s^2,
\]

其中d_j(u)=(H_t-H_j)(widehat p+Bu)/s，C_j=(H_t-H_j)Sigma_P(H_t-H_j)'/s²，E=Sigma_V/s²为对角矩阵。下列L_j使用归一化单位：

\[
L_j=\|z_j\|^2+2e^\top z_j.
\]

固定lambda=1/4。给定z_j，标量e'z_j为均值0、方差z_j'E z_j的高斯变量，因此

\[
\mathbb E_e[e^{-\lambda L_j}\mid z_j]
=\exp[-z_j^\top A z_j],\qquad
A=\operatorname{diag}(a_i),\quad a_i=\lambda(1-2\lambda E_{ii}).
\]

由于max E_ii=1，有a_i>=1/8>0。再对z_j做高斯配方，令

\[
G_j=I+2A^{1/2}C_jA^{1/2},
\]
\[
T_j(u)=d_j(u)^\top A^{1/2}G_j^{-1}A^{1/2}d_j(u)
+\tfrac12\log\det G_j,
\]

便得到精确积分E exp(-lambda L_j)=exp(-T_j)。最后由非负随机变量的Markov不等式：

\[
\Pr_t(L_j\le\delta_j)
\le\min\{1,\exp[\lambda\delta_j-T_j(u)]\}.
\]

积分精确，保守性来自指数概率上界和固定lambda。d_j关于u仿射，A/G_j/协方差固定，因此T_j是二次函数。逐单元除以各自标准差会改变物理损失，不是这里的公共尺度换算。

## 4. 最小费用与多个竞争者

单个竞争者的90%充分条件为T_j(u)>=lambda*delta_j+log10。保留设备域U与原费用C(u)，单对问题为

\[
\min_{u\in U} C(u),\qquad
T_j(u)\ge\lambda\delta_j+\log 10.
\]

多个竞争者时，**预先固定**风险预算alpha_j>0、sum alpha_j<=0.1，施加

\[
T_j(u)\ge\lambda\delta_j+\log(1/\alpha_j)\quad\text{对每个不同支撑竞争者}.
\]

这些仍是多个二次约束，保证条件于目标正确的整库激励事件至少90%；union上界不要求竞争事件独立。风险预算的简单起点可以是均分，选择办法需在试验前锁定。

原控制器实际优化归一化小时合同请求x，由`device.nominal(x)`产生96槽名义交付u；U是其功率、能量、到离站/截止、效率与终端要求。费用来自原合成合同，非货币实证校准。连续线性设备域对应非凸QCQP；保留原分段设备图和整数变量时是带整数的QCQP。求解器返回可行点、全局界、限时和未决应分开报告。

最新40组对**固定零动作**直接计算所有失败上界之和，全部<=0.1。这是零动作整库证书。若把sum exp(lambda*delta_j-T_j(u))<=0.1直接放进报价优化，一般不再是QCQP。完整多候选非零联合优化尚未接通，不能把已有零点核算称为该优化器。

## 5. 代码接入最容易弄反的一处

`mse_existing_cases.core.load_history_case`返回的`current`是历史winner，`reference`是最近不同族。但`core.build_gate`把`reference`当作正确H1生成器。必须显式指定方向：

```python
# target_index是本次假设正确的目标；competitor_index是待击败的竞争者。
view = dict(data, current=competitor_index,
            reference=target_index, ref=target_index)
gate = core.build_gate(view, beta=0.90, full=False)  # 仅核算零动作
# 非零优化需full=True；多候选固定风险预算用beta=1-alpha_j。
```

`conditional_target.zero_screen.spectral_pair`已按target=data.current正确处理。原两例失败中，最近竞争候选并不正确；禁止直接把默认reference当成真模型。现有`mse_reference_repair`的Hbar分支是另一种模型，保留为参考，主线不必接入。

## 6. 与前面算法设计合并时

先核对新窗口所采用算法的训练目标、接受规则、输出收缩/重拟合及最终选模，再转换delta。训练final-E的tau=.001、提议块数惩罚、旧3%MAE与新strictMSE分别保存。当前公式对应strict物理MSE；不把它直接套到MAE或加权高斯损失。

先用历史生成候选/拟合参数，再做次日规划；报价、源码、资源及随机流冻结后才生成实际新日。实际同一观测至少比较无DR、同费简单轮换和优化DR，报告指定损失事件、整库最终错漏/恢复、费用及来源。同一观察可另交NJ/RNJ/联合方法，区分DR和辨识器的贡献。90%始终是声明条件模型中的激励目标，实际AC达成率另列。
