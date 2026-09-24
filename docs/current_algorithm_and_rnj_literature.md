# 当前无频域主线与 RNJ 文献审计

## 1. 当前算法边界

当前主流程只处理同步的多场景智能电表 `P/Q/V` 与已知根节点电压。
每个场景的无噪真值由径向 AC backward-forward sweep 逐时段生成，随后叠加
`P/Q` 和电压量测噪声。内部节点为零注入且不可观，根节点标签已知。

主流程不再包含频域筛选、相位自由度或频域候选树。唯一固定时间预处理是
逐场景去均值：

\[
\widetilde z_s(t)=z_s(t)-\frac{1}{T_s}\sum_{\tau=1}^{T_s}z_s(\tau).
\]

它只移除每个场景的常数水平，不改变采样顺序，也不使用真实拓扑选择参数。

## 2. 当前端到端逻辑

### 2.1 Ordered reduced sensitivity

以根节点为参考构造平方电压降：

\[
y_i(t)=v_0^2(t)-v_i^2(t).
\]

多场景堆叠后拟合

\[
Y=PR^\top+QX^\top+B_{\rm scenario}+E,
\]

并对 `R/X` 施加对称、元素非负和对角层次约束：

\[
R=R^\top,\quad R_{ij}\ge 0,\quad
R_{ii}\ge R_{ij},\quad R_{jj}\ge R_{ij},
\]

`X` 同理。实现入口为
`terminal_case33/estimation/baseline.py::fit_complete_rnj_baseline`。

### 2.2 直接 R/X 共享路径分数与加性距离诊断

由 reduced matrix 构造 terminal 间距离：

\[
d^R_{ij}=R_{ii}+R_{jj}-2R_{ij},\qquad
d^X_{ij}=X_{ii}+X_{jj}-2X_{ij}.
\]

当前固定组合先分别归一化，再取

\[
d_{ij}=0.75\bar d^R_{ij}+0.25\bar d^X_{ij}.
\]

但 RNJ 不需要先形成该距离。令 `s_R/s_X` 为两类距离的正元素均值，实际排序量直接是

\[
S_{ij}=0.75R_{ij}/s_R+0.25X_{ij}/s_X,\qquad H_i=S_{ii}.
\]

从同一矩阵构造 `d` 后再算 `(H_i+H_j-d_ij)/2` 会严格返回 `S_ij`，所以代码已删除
这段无信息增量的往返；`d` 仅为 NJ/RG 和加性距离诊断保留。根不是从树中猜测，
而是作为 RNJ 的已知标签显式传入。

### 2.3 Ordered RNJ 基础树与 RNJ 候选网格

Ordered 距离产生一个确定性基础树。GTLS 分支考虑 `P/Q` 自变量误差，并在有限
距离模式和 RNJ 容差网格上产生额外候选。候选以 canonical rooted terminal
clade 去重。RNJ 只负责从估计距离生成隐藏树候选，不直接决定最终树。

### 2.4 NJ/RG 外周检测与一层软聚合

无根 NJ 和 root-augmented RG 只用于提出小规模外周 clade。对含噪基准量测再叠加
较小扰动并做时间块重采样，统计 clade 重现率；同时检查边界 shared-path margin、
归一化边长、GTLS 支持和 quartet 反证。

通过门控的 clade 只生成额外 aggregate/decompose 候选：

1. 簇内 `P/Q` 求和形成 pseudo 注入；
2. 用估计 service drop 从 terminal 电压回推 pseudo-parent 电压；
3. 在 pseudo 层重新拟合 Ordered `R/X` 并运行 RNJ；
4. 以 pseudo-parent 为局部根，在簇内重新辨识；
5. 主干和局部 clade 拼接成完整候选。

未聚合候选始终保留，所以聚合不是不可逆硬收缩。实现位于
`pipeline/peripheral_edge_proposals.py` 和 `pipeline/two_level_aggregation.py`。

### 2.5 固定树物理重估与最终选择

所有候选进入同一个选择器。对候选树 `T` 构造 terminal-to-edge 路径矩阵
`A_T`，并重估非负支路参数：

\[
R_T=A_T\operatorname{diag}(r)A_T^\top,\qquad
X_T=A_T\operatorname{diag}(x)A_T^\top,\qquad r,x\ge0.
\]

之后使用 structured-EIV 权重传播 `P/Q/V` 量测误差，在代表性训练工况上做一次
可接受的 AC 参数修正，并在留出场景上按 predictive NLL 排序。quartet 四点条件
和近零隐藏边只做入选树的后验结构检查。AC 排序不能补出候选池中不存在的树，
因此必须同时报告 candidate-oracle recall。

统一入口为 `pipeline/unified_topology_pipeline.py::identify_topology_unified`。

## 3. RNJ 的数学原理

设根为 `s`，terminal 集合为 `L`。在 reduced sensitivity 模型中直接定义

\[
S_{ij}=c_RR_{ij}+c_XX_{ij},\qquad H_i=S_{ii},
\]

其中系数非负且至少一个为正。在精确树上，`S_ij` 正好是从根 `s` 到
`LCA(i,j)` 的加权路径长度。因此最大 `S` 对应当前活动节点中最深的共同祖先。
RNJ 反复执行：

1. 选取 `S_ij` 最大的活动节点对；
2. 将具有相同 shared-path 深度的兄弟节点归到一个新隐藏父节点；
3. 以该父节点替换整组子节点，并更新它与其余活动节点的 shared-path；
4. 重复直到只剩一个活动节点，再连接到已知根。

这解释了 RNJ 为什么能生成隐藏节点，也解释了根约束为何天然满足：`S` 的每个元素
和对角根深度 `H` 都以已知根为公共路径起点，最终输出根始终保留调用者给定的物理标签。

## 4. 当前实现与原始 RNJ 的差别

`terminal_case33/graph/rooted_neighbor_joining.py` 是面向有噪距离的 general-tree
变体，不是 Ni--Tatikonda 伪代码的逐行复现：

- 原论文 general-tree RNJ 使用已知最小边长下界 `Delta` 构造分组阈值；
- 当前代码使用 `tolerance_factor * median(root_depths)`，默认因子 `0.16`；
- 多子节点组的外部 shared-path 用组内中位数更新，以降低异常距离的影响；
- 零长度 hidden-hidden 边在最后规范化收缩。

因此，Ni--Tatikonda 在已知 `Delta` 下的恢复半径不能直接当作当前默认参数的
理论保证。当前容差是跨合成算例固定的工程参数，应在新网络上用无真值的稳定性、
留出物理误差和候选覆盖诊断，而不能用真实 F1 调参。

共同的可辨识边界仍然成立：仅 terminal 加性距离不能唯一恢复 degree-2 hidden
chains，也不能区分零阻抗串联节点；评价前必须规范化到最小可辨识树。

## 5. RNJ 在相关文献中的使用情况

### 5.1 算法来源：网络层析，而非最初的配电智能电表方法

Ni 与 Tatikonda 的 *Network Tomography Based on Additive Metrics* 将 RNJ 用于
已知源节点的路由树层析。论文从源到终端以及终端间的加性距离恢复 binary/general
tree，给出精确加性度量下的恢复结果、带最小边长条件的误差半径和
`O(n^2 log n)` 复杂度。该论文是当前代码中 rooted shared-path 重构的直接理论来源：

https://arxiv.org/abs/0809.0158

普通 Neighbor Joining 更早来自 Saitou 与 Nei 的系统发育树重构；它通常输出
无根树。RNJ 的关键差别不是换一个 MST，而是把已知源节点的根深度直接纳入
shared-path 重构：

https://doi.org/10.1093/oxfordjournals.molbev.a040454

### 5.2 电网中的直接 RNJ 应用：PLC 拓扑层析

Ahmed 与 Lampe 在低压电网 PLC topology inference 中让末端 PLC modem 提供
线路 ranging/distance 估计，再用 RNJ 变体恢复配变为根的隐藏电网树。这是找到的
最直接“RNJ + 配电拓扑”先例：

https://people.ece.ubc.ca/lampe/Preprints/2013-PLC-Topology-Inference.pdf

但它的观测量是 PLC 传播/测距信息，不是智能电表 `P/Q/V` 灵敏度。因此它支持
“RNJ 可用于配电树加性距离”的结构类比，却不能直接证明当前 `R/X` 估计误差模型、
AC 重排或聚合策略。

### 5.3 与当前量测设定更接近的文献通常使用 RG 或 MST

Park、Deka、Backhaus 与 Chertkov 的 terminal-only hidden-node 配电拓扑工作与
当前可观测性设定更接近，但其精确恢复算法明确采用 Recursive Grouping，而不是
RNJ，并强调隐藏节点度条件：

https://arxiv.org/abs/1803.04812

Deka 等基于电压统计量的配电拓扑方法多采用 conditional-independence 规则或
minimum spanning tree；这些方法主要在可观测节点图上选边，并不等价于 RNJ 的
隐藏父节点生成：

https://arxiv.org/abs/1602.08509

## 6. 文献审计结论

1. RNJ 在“已知根 + terminal 加性距离 + hidden tree”上有明确网络层析理论。
2. RNJ 在配电领域有 PLC ranging 驱动的直接应用，但不是主流智能电表 `P/Q/V`
   灵敏度辨识算法。
3. 与本项目量测条件最接近的 hidden-node 文献更多使用 RG；可观测节点拓扑文献
   更多使用 MST 或条件独立方法。
4. 因此，本项目应把 RNJ 表述为“将网络层析的 rooted additive-tree reconstruction
   迁移到阻抗灵敏度距离”，而不应声称这是智能电表拓扑辨识文献中的标准做法。
5. 当前可能形成贡献的是完整耦合：物理约束 `R/X`、多源隐藏树候选、软聚合/
   局部分解、固定树 EIV 参数重估和留出 AC 物理排序；RNJ 本身不是新算法。
