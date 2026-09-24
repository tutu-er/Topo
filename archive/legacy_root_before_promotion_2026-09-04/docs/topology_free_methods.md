# 不依赖候选拓扑库的配电网结构学习方法

## 1. “不使用潜在拓扑信息”的三个强度

1. **不使用完整拓扑库**：不知道若干可能运行的完整开关状态，但知道物理线路候选集。现有 innovation_potential.py 属于这一层，因为它使用了 sorted(library)。
2. **不知道候选线路**：只知道母线集合，在完全图上学习连续边变量。它不需要线路候选集，但仍假设所有真实母线已知。
3. **隐藏母线数量也未知**：仅给出末端 Smart Meter，算法自行生成潜在 Steiner 节点并恢复约化树。这是最严格也最符合当前场景的一层。

第三层存在不可绕过的可辨识性边界。只观察边界节点时，一般网络首先只能确定 Kron-reduced 导纳

\[
Y_{\rm red}=Y_{BB}-Y_{BH}Y_{HH}^{-1}Y_{HB}.
\]

不同内部网络可能产生相同的 \(Y_{\rm red}\)。对于径向正阻抗网络，若隐藏节点度数至少为 3 且终端激励充分，可以从终端间加性距离恢复潜在树；度为 2 的隐藏节点通常只能与相邻线路合并。因此合理目标不是无条件恢复原始每一条边，而是恢复：

- 可辨识的约化潜在树；
- 终端之间的分支划分；
- 串联等效线路参数；
- 增加微型 PMU 后可进一步拆分的网络等价类。

已有工作已经在径向、线性模型和特定注入假设下研究了仅用终端量测恢复拓扑。因此“无需候选拓扑”本身不是足够创新点：

- Deka、Backhaus、Chertkov: https://arxiv.org/abs/1608.05031
- Park、Deka、Chertkov: https://arxiv.org/abs/1710.10727
- Li 等: https://arxiv.org/abs/1902.01365

## 2. 方法 A：终端 Green 矩阵到潜在加性树

这是严格不使用候选线路、完整拓扑库和隐藏节点数量的方法，也是新增原型采用的第一阶段。

在线性势模型中

\[
p=Lx,\qquad L=A^\top\operatorname{diag}(w)A.
\]

固定变电站势为零。若注入只位于终端集合 \(B\)，则

\[
x_B=H_Bp_B,\qquad
H_B=E_BL_{-0,-0}^{-1}E_B^\top.
\]

由多运行点数据估计

\[
\widehat H_B=
\arg\min_H\|X_B-P_BH\|_F^2+\lambda\|H\|_F^2.
\]

随后构造终端间有效电阻距离

\[
\widehat d(i,j)=
\widehat H_{ii}+\widehat H_{jj}-2\widehat H_{ij},
\]

以及变电站到终端的距离 \(\widehat d(0,i)=\widehat H_{ii}\)。在树网络中，该距离等于路径上的阻抗之和。对距离矩阵运行 Neighbor Joining 或递归分组，即可同时生成潜在节点、分支和边长，不需要预先列出拓扑。

当前原型还对时间样本进行 bootstrap，重复估计树并输出 terminal split 支持率。该支持率只是经验结构稳定性，不是贝叶斯后验概率或 conformal 覆盖率。

复杂度约为

\[
O(Tm^2+m^3+B_{\rm boot}m^3),
\]

其中 \(m\) 是终端数。它适合作为无先验拓扑初始化，但仍受线性模型、充分激励、树结构和度 2 节点不可辨识的限制。

## 3. 方法 B：完全图上的连续复导纳和隐藏状态联合学习

若所有母线编号已知但线路未知，可在完全图

\[
\mathcal E_{\rm all}=\{(i,j):0\le i<j<n\}
\]

上放置连续复导纳变量。令

\[
Y(g,b)=A_{\rm all}^\top
\operatorname{diag}(g-jb)A_{\rm all},
\qquad g_e,b_e\ge0.
\]

同时优化未量测母线电压、隐藏注入和边参数：

\[
\begin{aligned}
\min_{V,Z,g,b}\quad
&\sum_t\|M_VV_t-y_t\|_{R^{-1}}^2\\
&+\sum_t\|S_t(Z)-\operatorname{diag}(V_t)
\overline{Y(g,b)V_t}\|_{W^{-1}}^2\\
&+\lambda\sum_e\sqrt{g_e^2+b_e^2}
+\gamma\Phi_{\rm tree}(g,b)
+\eta\Phi_{\rm prior}(Z).
\end{aligned}
\]

第一项是量测一致性，第二项是完整 AC 潮流残差，第三项执行复导纳组稀疏，第四项约束连通径向结构，第五项描述隐藏注入的时空先验。

径向性不应只靠最后一次 MST。更严格的方案是令激活变量 \(a_e\in[0,1]\) 位于 spanning-tree polytope：

\[
\sum_ea_e=n-1,\qquad
\sum_{e\in E(S)}a_e\le |S|-1,\quad\forall S\subset V.
\]

指数多个割约束可用 cutting-plane 动态加入。也可以用矩阵树定理的 log-determinant 连通障碍获得可微松弛，再用精确树投影收尾。

建议使用增广拉格朗日或 ADMM：

1. 固定边变量，求多时刻隐藏状态和隐藏注入；
2. 固定状态，更新复导纳组；
3. 对新发现的环或割违反加入 cutting planes；
4. 更新乘子并逐步降低稀疏温度；
5. 使用 AC 潮流重新估计选中线路参数。

变量规模为 \(O(Tn+n^2)\)，每轮包含稀疏非线性状态估计和图约束优化，复杂度明显高于现有标量 IRLS。终端观测不足时完全图问题高度不适定，因此应与方法 A 的约化树初始化、弱地理先验或主动激励结合。

## 4. 方法 C：基于矩阵树分布的变分拓扑后验

为避免“连续权重再阈值化”，可以直接学习树分布而不枚举完整拓扑：

\[
p_\phi(G\mid X)\propto
\exp\left(\sum_{e\in G}\ell_{\phi,e}(X)\right)
\mathbf1\{G\text{ 是生成树}\}.
\]

矩阵树定理可以计算配分函数和边际边概率；Gumbel-perturb-and-MST 可以产生近似可微的离散树样本。物理解码器根据采样树、线路参数和隐藏注入计算观测似然：

\[
\mathcal L=
\mathbb E_{q_\phi(G,Z,\Xi\mid X)}
[-\log p_\theta(X\mid G,Z,\Xi)]
+\beta\operatorname{KL}(q_\phi\|p).
\]

它输出边际边概率、多棵树样本和多模态结构不确定性，不需要候选完整拓扑库。与普通 GNN 不同，采样结果始终满足生成树结构，而不是逐边独立分类。

主要挑战是离散梯度方差、物理解码器成本、树先验偏差和无标签训练时的后验坍缩。

## 5. 方法 D：无状态库的时变图 fused learning

现有 HMM 依赖预先给定的有限拓扑状态。可以直接估计每个时间窗口的连续图：

\[
\min_{\{Y_t,V_t\}}
\sum_t f_{\rm AC}(Y_t,V_t;X_t)
+\lambda\sum_t\sum_e\|y_{e,t}\|_2
+\rho\sum_{t>1}\sum_e
\|y_{e,t}-y_{e,t-1}\|_2.
\]

fused group penalty 使拓扑大部分时间稳定，只在少数时刻发生少量边变化。加入每个时刻的 spanning-tree 约束后，即可在不知道拓扑类别数的条件下同时完成变化点检测、结构恢复和参数估计。

开断动作还可以写成低秩 branch exchange：

\[
\|a_t-a_{t-1}\|_0\le2,
\]

通过混合整数优化或差分稀疏松弛求解。

## 6. 方法 E：结构学习器内嵌的双层安全实验设计

不使用拓扑库时，微型 PMU 布置不能再最大化“候选拓扑签名距离”。应直接针对结构恢复损失设计传感器和 DER probing：

\[
\min_{S,u_{1:H}}
\mathbb E\left[
d_{\rm struct}
\left(\widehat G_{\omega^\star(S,u)}(X),G^\star\right)
\right]+\lambda|S|,
\]

满足

\[
\omega^\star(S,u)=
\arg\min_\omega\mathcal L_{\rm graph}(\omega;X(S,u)),
\]

以及 AC 电压、逆潮流、逆变器容量和功率质量约束。外层选择固定微型 PMU 和安全激励，内层运行方法 B 或 C。可用隐式微分或展开若干次 ADMM 迭代计算外层梯度。

它比简单 max-min 签名布置更有潜力，因为优化目标是“运行结构学习器后还剩多少结构误差”，并不依赖候选完整拓扑。

## 7. 未知图空间中的 Conformal

Class-conditional conformal 需要有限拓扑标签，不能原样用于未知图空间。可以改为结构损失校准。给定基础估计器 \(\widehat G(X)\) 和有真实开关记录的校准事件，计算

\[
R_i=d_{\rm edit}(\widehat G(X_i),G_i).
\]

取 conformal 分位数 \(q_{1-\alpha}\)，输出隐式结构球：

\[
\mathcal C_\alpha(X)=
\{G:d_{\rm edit}(G,\widehat G(X))
\le q_{1-\alpha}\}.
\]

它不枚举所有图，只给出围绕当前估计的允许编辑半径。也可以对 terminal split loss、Kron-reduced 导纳误差或关键开关风险进行 conformal risk control。没有带真实拓扑标签的校准数据时，不能声称真实拓扑覆盖保证；bootstrap 支持率不能替代 conformal。

## 8. 推荐主线

建议采用层次化方案，而不是继续增加彼此独立的小算法：

1. **无先验结构发现**：方法 A 从末端数据恢复可辨识约化树和 bootstrap split 支持率。
2. **AC 连续精化**：将约化树作为方法 B 的 warm start，在完全图或局部扩展图上联合估计复导纳、隐藏状态和隐藏注入。
3. **结构后验**：方法 C 在高不确定区域采样多棵满足径向性的树，而不是只输出一次 MST。
4. **信息获取**：方法 E 根据结构后验或估计器灵敏度联合选择固定微型 PMU 和安全 probing。
5. **可验证输出**：存在有标签校准事件时，输出结构编辑球或 split-level conformal 风险界。

推荐的 Transactions 主线是：

**面向仅末端量测配电网的约化树发现—完全图 AC 导纳精化—双层安全传感设计。**

真正可辩护的贡献不是“不枚举候选拓扑”一句话，而是：明确可辨识等价类，让算法自行生成隐藏节点，在完整 AC 模型中恢复复导纳，并以结构恢复误差而不是候选签名距离设计微型 PMU 和激励。
