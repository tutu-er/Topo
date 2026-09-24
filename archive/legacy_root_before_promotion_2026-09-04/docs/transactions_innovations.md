# 面向 Transactions 投稿的创新点评估

本文档评估当前几个创新原型能否发展成论文贡献。结论先行：hidden-load Bayesian、active probing、HMM 和 conformal 都可作为有用组件；DRO 当前证据不足；最值得升格为主线的是“无候选拓扑库 latent tree 初始化 + AC 精化 + 可校准不确定性输出 + 主动信息获取”。

当前实验基于 12 节点小网络和 15 个候选拓扑，主要用于验证机制。若要支撑 Transactions 级别投稿，需要在 IEEE 33/123、OpenDSS 或真实切换事件上补齐更严格的评估。

## 1. 隐藏负荷边缘化 Bayesian likelihood

### 当前原型

代码：`src/topoident/innovation_bayesian.py`  
结果：`results/innovation_suite.json`

现有基线把隐藏节点当成零注入，用有限拓扑库的 WLS 残差做分类。新原型在每个候选拓扑下抽取隐藏负荷先验样本，估计观测签名的均值和协方差，用高斯边缘似然替代零注入 WLS。

当前结果：

- hidden nodes: 5、7；
- PMU buses: 9、11；
- zero-injection WLS accuracy: 46.25%；
- marginalized Bayesian accuracy: 97.92%；
- prior draws: 12；
- test cases: 720。

### 可行性判断

**可行，适合作为核心建模组件。** 该结果说明隐藏注入误差是拓扑误判的主要来源之一，边缘化比硬设零注入更稳健。这个方向和实际配网场景匹配：大量中间节点没有实时负荷量测。

主要风险：

- 当前隐藏负荷先验和测试分布一致，存在仿真同源性；
- 协方差由少量 Monte Carlo draw 估计，维度升高后会不稳定；
- 只处理固定的隐藏负荷节点和固定 PMU 位置；
- 当前是高斯近似，不一定能覆盖重尾、相关负荷和分布漂移。

投稿升级建议：

1. 使用分层负荷先验：按用户类型、时间段、温度或季节建模；
2. 用 shrinkage covariance、low-rank plus diagonal 或 unscented transform 替代小样本协方差；
3. 把隐藏注入、线路参数和量测噪声合并成物理边缘似然；
4. 报告 prior mismatch、节点缺失、PMU 数量和样本数量消融。

建议定位：**隐藏注入不确定性下的拓扑条件物理边缘似然**。

## 2. 主动 probing + 微型 PMU 联合设计

### 当前原型

代码：`src/topoident/innovation_active.py`  
结果：`results/innovation_suite.json`

原型枚举末端节点的正负小功率扰动，并枚举一个 PMU 节点，选择最大化最差拓扑对 Mahalanobis 分离度的联合设计。

当前结果：

- passive Smart Meter separation: 6.22；
- passive best PMU separation: 9.48；
- active Smart Meter separation: 15.59；
- joint active PMU separation: 22.25；
- passive Smart Meter accuracy: 95.63%；
- passive best PMU accuracy: 97.70%；
- active Smart Meter accuracy: 99.20%；
- joint active PMU accuracy: 99.87%。

### 可行性判断

**可行，但必须加入工程约束。** 这个方向的直觉很强：不同拓扑在自然负荷下可能难分，但受控小扰动能显著放大拓扑签名差异。联合优化 PMU 与 probing 比单独布点更像一个完整的传感与实验设计问题。

主要风险：

- 当前 probing 动作没有电压上下限、逆变器容量、用户舒适度、功率质量和调度可行性约束；
- 扰动位置只限末端节点，真实 DER 可控点可能稀疏；
- 优化目标是签名距离，不是最终识别错误率或结构风险；
- 没有考虑负荷响应、通信延迟和执行误差。

投稿升级建议：

1. 将外层设计写成带 AC 潮流约束的安全实验设计；
2. 将目标从 pairwise signature distance 改成预期 conformal 集合大小、结构编辑损失或最小 Chernoff 信息；
3. 加入 DER 容量、电压越限概率和扰动成本；
4. 做“无主动、仅 PMU、仅主动、联合主动 PMU”的完整消融。

建议定位：**面向拓扑辨识的安全主动激励与微型 PMU 联合信息获取**。

## 3. 开关图约束 HMM

### 当前原型

代码：`src/topoident/innovation_hmm.py`  
结果：`results/innovation_suite.json`

原型构造拓扑状态图，相邻状态表示一次 branch exchange。每个时刻有缺失的 Smart Meter / PMU 通道，独立分类器只看当前时刻，HMM 用转移代价和 Viterbi 做时间平滑。

当前结果：

- horizon: 100；
- missing probability: 35%；
- true switches: 11；
- independent accuracy: 69%；
- Viterbi accuracy: 97%；
- independent switch F1: 0.344；
- Viterbi switch F1: 0.818。

### 可行性判断

**工程可行性较强，适合作为动态扩展。** 配电网拓扑通常不会每分钟任意跳变，开关动作稀疏且受操作图约束。HMM 正好利用这一时序先验，尤其适合缺失和异步量测。

主要风险：

- 当前仍依赖预先枚举的有限拓扑状态库；
- 转移代价是手工设定，不是从真实开关日志学习；
- emission 仍是简单签名残差；
- 对异常事件、保护动作和多开关同时动作的鲁棒性未验证。

投稿升级建议：

1. 从开关日志学习转移概率，区分常规操作和故障恢复；
2. 将 emission 换成 hidden-load Bayesian likelihood；
3. 与 topology-free fused graph learning 对比；
4. 报告检测延迟、切换时刻 F1、缺失率扫描和多开关事件性能。

建议定位：**带开关操作图先验的动态拓扑状态估计**。它可作为主线的动态章节，而不是单独主贡献。

## 4. potential-edge 稀疏恢复

### 当前原型

代码：`src/topoident/innovation_potential.py`  
结果：`results/innovation_suite.json`

原型使用候选边集合、标量 Laplacian 势模型和 IRLS group sparsity，交替完成状态补全和边权更新，最后投影到最大生成树。

当前结果：

- candidate edges: 16；
- true edges: 11；
- projected edges: 11；
- matched edges: 11；
- edge precision / recall: 100% / 100%；
- matched weight MAPE: 35.53%；
- objective: 0.00540 降至 0.000498。

### 可行性判断

**作为凸核心或 warm start 可行，作为独立论文创新偏弱。** 它展示了“候选边 + 稀疏权重 + 树投影”可以恢复结构，但当前模型是标量势模型，不是完整 AC 复导纳模型；参数误差也还高。

主要风险：

- 仍使用候选边集合，不是真正无拓扑先验；
- 最后 MST 投影可能破坏物理残差最优性；
- 径向性没有在优化过程中严格约束；
- 标量模型无法完整利用电压幅值、相角和无功信息。

投稿升级建议：

1. 升级到复导纳 `g-jb` 和 AC 潮流残差；
2. 用 spanning-tree polytope、cutting-plane 或矩阵树 barrier 约束径向性；
3. 把该模块作为 topology-free latent tree 之后的 AC refinement；
4. 对比 group lasso、MST 后处理、MISOCP / MINLP 小算例。

建议定位：**从约化树或候选局部图出发的 AC 复导纳稀疏精化**。

## 5. class-conditional conformal 集合 + 自适应 PMU 查询

### 当前原型

代码：`src/topoident/innovation_conformal.py`  
文档：`docs/conformal_topology_method.md`  
结果：`results/innovation_suite.json`

原型对每个拓扑类单独校准 Smart Meter 残差阈值，输出 topology prediction set。当集合大小大于 1 时，再根据候选集合内相角分离度选择一个 PMU 查询节点。

当前结果：

- target miscoverage: 5%；
- Smart Meter Top-1 accuracy: 86.57%；
- conformal set coverage: 97.22%；
- average prediction set size: 2.07；
- adaptive one-PMU accuracy: 87.78%；
- PMU query rate: 45.56%。

### 可行性判断

**集合辨识方向可行，自适应查询部分需要重做校准。** Conformal 的价值不在于提高 Top-1，而在于给出有限样本覆盖意义下的“不排除真实拓扑”集合。当前覆盖率超过 95%，说明分数有基本可校准性。

主要风险：

- 校准样本和测试样本来自同一仿真分布，真实漂移未验证；
- 当前把同一扰动 draw 的多个运行点都当作样本，独立性需要重新定义；
- 查询节点可逐样本变化，更像移动传感或按需调用，不是固定安装 PMU；
- 查询后的 Top-1 没有端到端 conformal 校准，不能声明查询后集合覆盖保证。

投稿升级建议：

1. 将事件窗口作为校准单元，而不是单个运行点；
2. 固定模型、安装集合和在线查询策略后，再做最终校准；
3. 输出查询后的 conformal set，而不是只输出 Top-1；
4. 在季节漂移下加入 weighted conformal 或滚动校准；
5. 报告每个拓扑类的覆盖率、平均集合大小和空集合率。

建议定位：**拓扑集合辨识与集合驱动的信息获取**。

## 6. DRO-CVaR PMU 布点

### 当前原型

代码：`src/topoident/innovation_dro.py`、`src/topoident/innovation_dro_v2.py`  
结果：`results/innovation_5_dro.json`

原型在线路参数扰动集合上最大化 lower-CVaR 分离度，希望得到对参数漂移更稳健的 PMU 布点。

当前结果：

- nominal buses: 9、11；
- robust buses: 10、11；
- nominal shifted-test accuracy: 66.44%；
- robust shifted-test accuracy: 54.44%；
- design parameter sigma: 0.16；
- test parameter sigma: 0.28。

### 可行性判断

**当前证据不支持作为主创新。** robust 目标数值更高，但 shifted-test 识别准确率更低，说明所优化的 lower-CVaR pairwise distance 与最终 profile-likelihood 分类损失不一致，或者 ambiguity set / scoring rule 设计不对。

可能原因：

- 对所有拓扑对和所有扰动混合做 CVaR，可能过度关注无关困难对；
- robust 设计目标和测试分类器不是同一个损失；
- 参数扰动幅度和分布设置较任意；
- 没有加入 hidden-load 不确定性，鲁棒对象不完整。

建议处理：

1. 暂时不要把 DRO-CVaR 作为主贡献；
2. 若保留，应把目标改成最坏情形误分类风险、预期 conformal 集合大小或 worst-case negative log-likelihood；
3. 对比 nominal、Bayesian marginal、DRO 三种模型在相同分类器下的性能；
4. 只有当 robust 方案在分布漂移上稳定优于 nominal 时，再写入论文主线。

## 7. topology-library-free latent tree

### 当前原型

代码：`src/topoident/innovation_topology_free.py`  
文档：`docs/topology_free_methods.md`  
结果：`results/innovation_6_topology_free.json`

该方法只给 learner slack 标签、末端标签和终端注入/电势数据。它先估计终端 Green 矩阵，再从有效电阻距离恢复加性潜在树，算法不使用候选拓扑库、候选边集合或隐藏节点数量。

当前结果：

- uses_candidate_topologies: false；
- uses_candidate_edges: false；
- uses_hidden_bus_count: false；
- inferred latent nodes: 3；
- split precision / recall: 100% / 100%；
- minimum true split bootstrap support: 1.0；
- terminal distance mean relative error: 0.039%；
- additive tree fit RMSE: 2.02e-05。

### 可行性判断

**最值得作为主线之一。** 它回应了有限拓扑库方法的关键弱点：真实运行中未必知道所有可能开关状态。该原型还能明确可辨识性边界：只用末端量测时，度为 2 的隐藏节点通常不可区分，只能恢复串联等效后的约化树。

主要风险：

- 当前是标量电阻 / 势模型，尚未连接完整 AC；
- 网络必须近似径向，且终端激励要足够丰富；
- 只能恢复可辨识 terminal splits 和等效边长，不能无条件恢复原始每一条物理线路；
- 小网络成功不代表 IEEE 123 或非径向网络可直接成功。

投稿升级建议：

1. 把“可辨识约化树”作为理论对象，不声称恢复所有内部度 2 节点；
2. 加入噪声、样本数量、终端数量、隐藏分支深度和负荷相关性消融；
3. 将 latent tree 输出作为 AC 复导纳 refinement 的初始化；
4. 与 Deka / Park / Li 等无拓扑库方法做定位和对比；
5. 用 split-level bootstrap 或 conformal risk control 输出不确定性。

建议定位：**仅末端量测下的可辨识约化树发现**。

## 推荐投稿主线

建议不要把 6 个创新点平铺成 6 个“贡献”。更稳的论文结构是：

1. **Topology-free 结构初始化**：从末端 Green 矩阵恢复可辨识约化树和 split support；
2. **AC 物理精化**：在约化树附近扩展到复导纳、隐藏注入和线路参数联合估计；
3. **不确定性输出**：用 class-conditional conformal 或 split-level conformal 输出集合/结构球；
4. **主动信息获取**：用 PMU + probing 最小化集合大小或结构风险；
5. **动态扩展**：HMM 处理有限候选状态下的时变切换，作为工程场景补充。

可作为核心标题的方向：

**面向仅末端量测配电网的约化树发现、AC 导纳精化与可校准主动拓扑辨识。**

各原型的推荐状态：

| 创新点 | 当前状态 | 建议用途 |
| --- | --- | --- |
| Hidden-load Bayesian | 结果强，假设偏同源 | AC 精化和 conformal 分数的核心 likelihood |
| Active probing + PMU | 结果强，工程约束不足 | 主动信息获取模块 |
| HMM | 动态信号清楚 | 动态拓扑扩展章节 |
| Potential sparse | 结构恢复强，模型简化 | AC refinement warm start |
| Conformal | 集合覆盖有信号 | 不确定性输出主模块 |
| DRO-CVaR | 当前反例明显 | 暂缓或重做目标 |
| Topology-free latent tree | 最具主线价值 | 主贡献入口 |
