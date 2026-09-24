# 给 GPT 新对话使用的提示词

下面提示词用于让另一个 GPT 对话继续思考：如何在 terminal-only hidden-node 配电网拓扑辨识中系统性提高随机误差下的精度。可以直接复制。

```text
你是一名熟悉配电网拓扑辨识、智能电表量测、LinDistFlow/DistFlow、AC 潮流、隐藏节点树重构、统计学习、图学习和机器学习的科研合作者。请围绕下面问题进行深入分析，并提出可实现的算法路线。

研究对象：
1. 配电网是单根径向树，根节点为配变/馈线出口。
2. 根节点可观测，提供 root voltage V0，内部 hidden nodes 无负荷、无智能表、零注入。
3. 只有 terminal/end-user leaf nodes 有 P/Q/V 智能表和负荷/DER。
4. 数据由逐时段 AC backward-forward sweep 潮流生成，不再直接用 LinDistFlow 合成电压。
5. 量测噪声：P/Q 通常为 0.5% 相对瞬时值；V 噪声约 0.01%--0.1%，当前常用 0.02% 相对瞬时电压幅值；root voltage 有公共模态波动。
6. terminal 负荷曲线包含 residential/commercial/industrial demand、PV、wind、小发电机和混合节点。多场景包括 default、high wind、low wind、no solar、cloudy PV、high load、evening peak、storm front、mixed cloud/wind。
7. 目标不是直接恢复完整物理 hidden tree，而是首先恢复 terminal-equivalent tree；若讨论 hidden-node reconstruction，必须明确额外假设。

当前基线：
1. 构造 squared-voltage drop：
   D_i(t)=V0(t)^2 - V_i(t)^2
2. 拟合多场景共享 R/X：
   D_i,s(t) ≈ sum_j R_ij P_j,s(t) + sum_j X_ij Q_j,s(t) + gamma_i,s
3. 约束：
   R=R^T, X=X^T, R>=0, X>=0
4. 根据 reduced sensitivity 构造距离：
   dR_ij = R_ii + R_jj - 2R_ij
   dX_ij = X_ii + X_jj - 2X_ij
   可选 dRX = normalize(dR)+normalize(dX)
5. 用 MST 或 matrix-tree posterior 得到 terminal-equivalent topology。
6. 当前固定组合 rolling_highpass_w288 + dR 在 4 个论文风格 LV 算例上约恢复 55/56 条 terminal-equivalent MST 边；不滤波 raw_drop + dR 约为 51/56。
7. 目前最优 per-case oracle 可达到 56/56，但 oracle 使用了真实拓扑选滤波/距离，不能作为真实未知拓扑算法。

请你完成以下任务：

A. 从误差来源出发建模：
- V 噪声经过平方变换如何影响 D_i(t)；
- P/Q 自变量噪声带来的 errors-in-variables bias；
- root voltage common mode；
- AC 潮流非线性与 LinDistFlow reduced sensitivity 的系统误差；
- 负荷/PV/风电相关性导致的病态设计矩阵；
- MST 对距离排序误差的放大。

B. 提出不依赖真实拓扑的自适应算法：
- 如何选择滤波窗口或频域频带；
- 如何使用 held-out residual、bootstrap edge stability、condition number、R/X consistency、matrix-tree posterior entropy；
- 是否应该用 ensemble over windows；
- 如何避免选到 residual 低但拓扑错的窗口。

C. 比较并给出数学原理：
- WLS/GLS；
- corrected least squares / total least squares / instrumental variables；
- Huber/Tukey robust regression；
- low-rank common-mode removal；
- bootstrap/stability selection；
- matrix-tree theorem edge marginal；
- tree metric projection / quartet voting；
- R/X 多视角距离融合；
- active probing / optimal experiment design；
- AC residual correction。

D. 调研机器学习/深度学习方向是否适合：
- GNN / topology-aware state estimation；
- physics-informed graphical learning；
- neural relational inference；
- differentiable spanning tree / matrix-tree learning；
- Bayesian neural nets / ensemble uncertainty；
- neural operator or surrogate AC power-flow correction；
- synthetic feeder pretraining / domain randomization。
请判断每类方法对本问题是“主算法”、“辅助模块”还是“不推荐”，并说明原因。

E. 给出可实现方案：
- 设计一个无需 oracle 的 Adaptive Stable Constrained Topology Identification 算法；
- 给出伪代码；
- 给出目标函数；
- 给出评价指标；
- 给出消融实验设计；
- 给出在 Python 项目 terminal_load_only_case33 中应新增哪些模块和接口。

F. 回答关键问题：
1. 是否存在理论上不可辨识的情形？
2. degree-2 hidden chain 为什么不能仅靠 terminal measurements 唯一恢复？
3. 为什么提高电压降、增加主动扰动或多场景激励能提高精度？
4. 为什么黑箱 ML 可能输出相关图而不是物理树？
5. 如何报告不确定性，而不是只报告一棵树？

输出要求：
- 请用中文回答。
- 数学公式要完整。
- 不要只泛泛列方法，要说明每种方法在当前 terminal-only 场景中的具体作用和风险。
- 请明确区分 terminal-equivalent topology 与 full physical hidden-node topology。
- 请给出优先级最高的 3 个可实现改进，并说明为什么。
```

