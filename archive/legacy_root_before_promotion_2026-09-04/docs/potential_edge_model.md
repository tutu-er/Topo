# 候选边稀疏恢复与 AC 精化模型

本文档整理 `innovation_potential.py` 背后的模型，并说明它如何从当前标量势模型升级到可投稿的 AC 复导纳恢复方法。

## 1. 问题设定

给定母线集合和一个候选边集合

\[
\mathcal E_c=\{e=(i,j)\},
\]

目标是在不枚举完整拓扑状态的情况下，从多时刻量测恢复激活边、线路参数和未量测状态。

每条候选边对应复导纳

\[
y_e=g_e-jb_e,\qquad g_e\ge0,\ b_e\ge0,
\]

其中 `b_e` 表示正的电纳幅值，Ybus 的支路导纳写成 `g_e - j b_e`。激活边应形成径向或近径向结构。

## 2. 当前标量势模型

当前 `src/topoident/innovation_potential.py` 使用的是一个标量 Laplacian surrogate。令

\[
L=A^\top\operatorname{diag}(w)A,
\]

其中 `A` 是候选边关联矩阵，`w_e>=0` 是边权。固定 slack 节点后，每个样本满足

\[
p_t=Lx_t.
\]

Smart Meter 和微型 PMU 提供部分节点电势观测：

\[
z_t=Hx_t+\epsilon_t.
\]

算法交替执行：

1. 固定边权 `w`，用 KCL 和量测补全所有节点状态 `x_t`；
2. 固定状态 `x_t`，用非负 least squares 和 IRLS 稀疏惩罚更新 `w`；
3. 对最终权重做最大生成树投影，得到径向拓扑。

该模型抓住了“边电流 = 边权 × 节点势差”的凸核心，因此适合做 warm start。但它还不是完整 AC 拓扑和参数估计。

## 3. AC 复导纳升级

完整 AC 模型应直接使用复电压

\[
V_i^t=|V_i^t|e^{j\theta_i^t}
\]

和 Ybus：

\[
Y(g,b)=A^\top\operatorname{diag}(g-jb)A.
\]

每个时刻满足

\[
S_t=\operatorname{diag}(V_t)\overline{Y(g,b)V_t}.
\]

可建立如下优化：

\[
\begin{aligned}
\min_{V,Z,g,b}\quad
&\sum_t\|M_VV_t-y_t\|_{R^{-1}}^2\\
&+\sum_t\|S_t(Z)-\operatorname{diag}(V_t)
\overline{Y(g,b)V_t}\|_{W^{-1}}^2\\
&+\lambda\sum_e\sqrt{g_e^2+b_e^2}
+\gamma\Phi_{\rm tree}(g,b)
+\eta\Phi_{\rm hidden}(Z).
\end{aligned}
\]

其中：

- 第一项约束电压幅值、相角等量测；
- 第二项约束 AC 潮流残差；
- 第三项执行复导纳组稀疏；
- 第四项约束径向性或连通性；
- 第五项描述隐藏节点注入先验。

## 4. 径向结构约束

只在最后做 MST 投影可能得到物理残差并不最优的树。更严格的做法是在优化过程中约束激活变量

\[
a_e\in[0,1]
\]

落在 spanning-tree polytope：

\[
\sum_e a_e=n-1,
\qquad
\sum_{e\in E(S)}a_e\le |S|-1,\quad \forall S\subset V.
\]

指数多个割约束可通过 cutting-plane 动态加入。另一种方案是使用矩阵树定理构造可微连通 barrier，再在收尾阶段做精确树投影和 AC 参数重估。

## 5. 与微型 PMU 的关系

微型 PMU 的作用不是简单降低噪声，而是降低结构不可辨识性。可用 Fisher information 或后验结构风险评价一个安装集合 `S`：

\[
\mathcal I_S(\theta)=
J_S(\theta)^\top R_S^{-1}J_S(\theta),
\]

其中 `theta` 包括激活边、`g,b` 和隐藏状态。布点目标可写成：

\[
S^\star=
\arg\min_{|S|\le M}
\mathbb E[d_{\rm struct}(\widehat G_S,G^\star)]
+\lambda\operatorname{Cost}(S).
\]

如果结合主动 probing，外层还需要选择安全可执行的注入扰动。

## 6. 可行性和风险

当前原型在 12 节点小网络中能恢复全部真实边，但这主要说明标量 surrogate 有结构信号。要支撑论文结论，还需要解决：

1. **模型误差**：从标量势模型升级到 AC 复导纳；
2. **参数精度**：当前 matched weight MAPE 约 35.53%，需要 AC 重估降低误差；
3. **径向约束**：避免只靠后处理 MST；
4. **候选边依赖**：若候选边集合不可靠，应先用 topology-free latent tree 生成局部候选图；
5. **隐藏注入**：不能把所有中间节点都当成零注入，需要与 Bayesian hidden-load 模块结合；
6. **鲁棒评估**：需要测试噪声、样本数量、PMU 数量、候选边冗余和线路参数扰动。

推荐把该模块定位为：**约化树或候选局部图上的 AC 复导纳稀疏精化器**，而不是独立声称“无先验拓扑恢复”。
