# 可信部分拓扑先验与人工核查查询设计

审查日期：2026-09-08。权威实现为 rnj_wzzt_core/rnj_wzzt。本文区分已经存在的接口、可增加的实验适配及尚未实现的约束。**本轮没有把新先验或查询策略接入主线；提供设计和独立、可运行的查询排序示例。**

## 1. 先定义人工确认的事实

设终端集合为 \(L\)，\(\mathcal C(T)\) 为有根最简树 \(T\) 的全部非平凡 proper clades。每个 clade 是某个节点完整的下游终端集合，不包含全集 \(L\) 和 singleton。

| 先验类型 | 精确定义 | 不能混同的事实 |
|---|---|---|
| 完整 clade \(S\) | 存在节点，其全部下游终端恰好为 \(S\) | 位于某共同区域或某更大支路中，不确定完整集合 \(S\) |
| 二元 exact cherry \(\{a,b\}\) | 完整的两末端同父组，父节点没有其他终端后代 | 同父但还有第三个兄弟，不意味着二元集合是 clade |
| 多元完整末端组 \(S\) | \(S\in\mathcal C(T)\)，且 \(S\) 内没有非平凡 proper clade | 只知道 \(S\) 是 clade，仍允许内部存在分叉 |
| 一级支路归属 | 指定两终端是否在最简树一级支路分区的同一块，或给出完整分区 | 同一级支路不等于 exact cherry，也不等于二元 clade |
| 已知实物边 \((u,v)\) | 具有跨候选一致的真实设备 ID、连接及根侧信息 | RNJ 的负整数隐藏节点编号不是设备 ID |

本设计采用

\[
\Pi(T)=
\{\mathcal C(T)\text{ 中按包含关系极大的 clade}\}
\cup\{\text{尚未覆盖末端的 singleton}\}.
\]

它忽略所有末端共有的根干线及不可分辨的串联 degree-2 隐藏节点。**存在下游集合为全集 \(L\) 的共同干线，不意味着所有末端属于同一“一级支路”。** 工程侧若要问实际变电站出线或具名开关归属，必须提供设备锚点映射，不能直接替换这一最简树分区。

示例的 exact_cherry 查询接受 \(|S|\ge2\) 的完整外围末端组；\(|S|=2\) 时才是二元 cherry。问题必须明确“父节点没有其他后代”，否则得到的是另一种约束。

## 2. 当前接口及其边界

R/X 回归和 RNJ 本身没有外部拓扑先验参数。可利用的接口在后续 laminar L1 回归：

\[
R=\sum_{S\in\mathcal F}r_S\mathbf1_S\mathbf1_S^\top,\qquad
X=\sum_{S\in\mathcal F}x_S\mathbf1_S\mathbf1_S^\top,\qquad r_S,x_S\ge0.
\]

\(\mathcal F\) 是 laminar family：任意两集合嵌套或不相交。R、X 分别估计，不是比值。

- [fit_laminar_l1_sensitivity](/D:/0-github_workspace/Topo/rnj_wzzt_core/rnj_wzzt/estimation/laminar_l1_milp.py:1182) 的 initial_supports 固定集合，之后共同重估 R/X 系数及截距；这些集合限制后续 laminar 兼容性。
- candidate_supports 仅提供有限候选，不强制采用；候选可以相互交叉，每次选择一个与已接纳集合兼容的扩展。
- [零系数修剪](/D:/0-github_workspace/Topo/rnj_wzzt_core/rnj_wzzt/estimation/laminar_l1_milp.py:1388) 保留全部初始支持。“先验结构留在输出中”不保证对应 \(r_S,x_S\) 为正；当前没有单个先验原子的正下界参数。
- 每步扩展最优性仅限该步候选域，整个前向路径与验证选点不是全部树上的全局最优。
- 现有 PseudoCluster 收缩依赖估计的 R/X 做电压去嵌入。错误块若提前收缩，该次简化问题不能再次分裂它。

### 2.1 标签必须映射为当前列位置

若训练列为 [101,102,103,104]，完整 cherry {102,104} 对应位置 (1,3)：

~~~python
labels = list(training[0]["P_terminal"].columns)
position = {label: index for index, label in enumerate(labels)}
known_support = tuple(sorted(position[label] for label in {102, 104}))
leaf_supports = [(index,) for index in range(len(labels))]

result = fit_laminar_l1_sensitivity(
    training,
    validation_scenarios=validation,  # 名称、顺序、终端列已对齐
    initial_supports=[*leaf_supports, known_support],
    candidate_supports=proposal_pool,
)
~~~

必须校验标签存在、支持去重、初始族 laminar。不能同时冻结相交的 exact cherries {a,b}、{b,c}。有外部可信事实时，应先解决其与 RNJ 冻结块的冲突，不能在冲突的旧收缩结果上继续追加。

已收缩时，使用 [位置映射规则](/D:/0-github_workspace/Topo/rnj_wzzt_core/rnj_wzzt/pipeline.py:202)：原终端集合必须是若干完整 pseudo 块的并。切开冻结块会返回 None；人工若确实否定旧块，应撤销收缩并回到原终端重拟合，不能把 None 静默当作“没有先验”。

### 2.2 各类先验的准确落点

| 已获事实 | 现有接口能做什么 | 仍需适配或额外约束 |
|---|---|---|
| 确信完整 clade \(S\) | 放入 initial_supports | 允许内部细化；零系数另外报告 |
| 确信二元 exact cherry | 固定二元集合 \(S\) | 二元集合没有更小非平凡子集；正物理边仍无下界保证 |
| 确信多元完整末端组 \(S\) | 固定 \(S\) | 还要禁止 \(S\) 内 proper 非平凡 clade，或在可信、不相交时收缩；仅固定 \(S\) 不够 |
| 不确定 clade | 加入 candidate_supports | “允许出现”不是置信度惩罚，也不是概率软先验 |
| 完整一级支路分区 \(G_i\) | 固定非 singleton 的各 \(G_i\) | 仅固定块仍允许若干整块形成更高 proper clade；还需限制候选位于某个 \(G_i\) 内，全集 \(L\) 的共同干线原子可例外 |
| 两末端同一级支路 | 可过滤完整候选树 | 不能固定二元集合替代；需要上层 clade 存在的析取约束 |
| 两末端不同一级支路 | 有限池可排除同时包含二者的 proper clade | 当前 unrestricted 搜索没有通用负先验参数 |
| 实物边及其完整下游集合 \(S\) | 可把切分信息译为支持 \(S\) | 实物端点、串联边数量和长度不由支持原子识别；singleton 叶边支持本已存在 |
| 仅知道实物边端点 | 通常不能直接表达 | 需要稳定设备候选图或端点—clade 锚点映射 |

### 2.3 当前缺失的负先验

当前没有 forbidden_supports、forbidden_root_pairs、具名端点连接约束或查询可靠度参数。有限候选池可在调用前过滤，但**本轮没有把适配层加入主线**：

- 否认 clade \(S\)：排除候选 \(C=S\)，并检查初始族没有冻结它。
- 否认二元 exact cherry：同上。
- 确认多元 exact cherry \(S\)：固定 \(S\)，排除 \(C\subsetneq S,\ |C|>1\)。
- 否认多元 exact cherry：是“没有 clade \(S\)，或者其内部存在真分叉”的析取；直接删除 \(S\) 会过度限制。
- 完整一级支路分区：仅允许 \(C\subseteq G_i\)，并可例外保留 \(C=L\) 的共同干线。
- 两末端不同一级支路：禁止 \(\{a,b\}\subseteq C\subsetneq L\)，不能误禁全集干线。

过滤不能忽略初始支持。若 candidate_supports=None，则搜索不是有限列表，必须增加求解约束，外部删列表无效。先验冲突应保留证据并指出冲突事实，不能悄悄忽略。

## 3. 查询必须以完整树集合为输入

令候选为 \(T_1,\ldots,T_K\)，权重 \(w_i\ge0\)，归一化为 \(p_i=w_i/\sum_jw_j\)。

可用来源是完整 bootstrap 树、不同通道/方法的完整重建树或完整候选解。同一 P/Q/V 派生的 R-RNJ、X-RNJ、RX75-RNJ、MILP 高度相关；**等权方法投票只展示方法分歧，不能称独立证据或校准后验。**

[当前 bootstrap](/D:/0-github_workspace/Topo/rnj_wzzt_core/rnj_wzzt/pipeline.py:73) 在第 104 行只累计边界组频次，没有保留每次完整树的联合结构。因此不能独立相乘边界频率构造拓扑概率。正式接入需保存完整 clade family 或带根边集；只给边界组不满足本示例输入约定。

同一规范化树合并权重；同一记录 ID 重复则报错。algorithmic 模式忽略隐藏节点重编号、全集共同干线及串联 degree-2 隐藏节点。physical 模式保留真实边集不同的候选。

候选一致可能是共同错误。若只有一棵候选，分歧为零仅说明来源没有表达不确定性。出现候选质量为零的人工回答时，应记录覆盖失败、扩大候选并重拟合，不能把答案称为“不可能”。

若要检验候选外共同错误，应预留一部分固定预算用于随机或业务重要位置抽查；这部分探索价值不由下面的候选分歧公式保证，零排序值也不等于无需核查。

## 4. 结构损失、经验风险和核查成本

定义：

- \(\mathcal C(T)\)：全部非平凡 proper clades；
- \(\mathcal P(T)=\{\{a,b\}:a,b\text{ 同属 }\Pi(T)\text{ 一块}\}\)：一级支路同属关系；
- \(\mathcal B(T)\)：\(\mathcal C(T)\) 中包含关系极小的集合，即完整外围末端组。

示例损失为

\[
\ell(T,T')=
\lambda_C|\mathcal C(T)\triangle\mathcal C(T')|
+\lambda_R|\mathcal P(T)\triangle\mathcal P(T')|
+\lambda_B|\mathcal B(T)\triangle\mathcal B(T')|.
\]

权重预先给定。root pair 项强调影响多个末端的上层归属，cherry 项强调末端连接。三项可以重叠，这是显式加权选择，不代表统计独立。

候选经验风险采用

\[
D(p)=\mathbb E_{T,T'\sim p\ {\rm independent}}[\ell(T,T')].
\]

这是两次从候选集合抽树时的期望分歧，**不是当前估计器的测试误差，也不是重拟合后必然多恢复多少线路的预测。** 每个问答分支另外返回分支内期望损失最小的候选 medoid，供人工查看；它不改变上述风险定义。

对是/否查询 \(q\)，令 \(A_q(T)\in\{0,1\}\)，\(m_b=\sum_{i:A_q(T_i)=b}p_i\)。假设回答无误，条件后仅保留相应分支并归一化：

\[
D_{\rm after}(q)=\sum_bm_bD(p\mid A_q=b),\qquad
G(q)=D(p)-D_{\rm after}(q),\qquad
{\rm priority}(q)=\frac{G(q)}{{\rm cost}(q)}.
\]

成本必须为正且单位一致，可为工时或费用。另单独输出未加权 clade 对称差下降 \(G_C(q)\)，字段为 expected_clade_symmetric_difference_reduction。

若结构特征 \(F\) 的出现质量为 \(\pi_F\)，则

\[
D(p)=2\sum_F\lambda_F\pi_F(1-\pi_F).
\]

由全方差公式，

\[
G(q)=2\sum_F\lambda_F
\operatorname{Var}_b[\Pr_{\rm empirical}(F\mid A_q=b)]\ge0.
\]

这是有限权重与特征给定后的恒等式，不依赖后验假设。均衡分割不必最优，结构影响和成本同样决定价值。二元熵 \(h_2(m_1)\) 只作辅助展示。

当前脚本假设人工回答无噪声。若存在误查或过期图纸，需要另给 \(\Pr(Y\mid A_q)\)，按回答似然重加权；这尚未实现。“无法判断”应记录成本且不更新候选，不能硬填“否”。

## 5. 可运行示例和正式输入接口

脚本：[rank_topology_queries.py](/D:/0-github_workspace/Topo/artifacts/prior_acquisition_design_20260908/rank_topology_queries.py)。仅用 Python 标准库，不导入核心算法，不发送外部消息。

~~~json
{
  "root": 0,
  "terminals": [1, 2, 3, 4],
  "edge_node_ids": "algorithmic",
  "loss_weights": {"clade": 1, "root_pair": 2, "exact_cherry": 1},
  "trees": [
    {"id": "method_A", "weight": 1, "clades": [[1, 2], [3, 4]]},
    {"id": "method_B", "weight": 1, "clades": [[1, 3], [2, 4]]}
  ],
  "queries": [
    {"id": "q1", "type": "root_branch_same", "terminals": [1, 2], "cost": 10},
    {"id": "q2", "type": "exact_cherry", "terminals": [1, 2], "cost": 6},
    {"id": "q3", "type": "clade", "terminals": [1, 2, 3], "cost": 25}
  ]
}
~~~

接口约定：

- 每棵树只能给 clades 或 edges 之一。clades 是完整候选的全部非平凡集合，不能只选边界组；允许多分叉树。
- edges 为 [u,v] 或 [u,v,length]，要求连通无环、非根叶恰好为给定末端；排序不使用长度。
- 节点标签为整数或非空字符串；算法隐藏 ID 不能作为实物 ID。
- 省略 queries 会自动产生 union-clade、exact-cherry、root-branch pair 查询，成本默认 1。正式核查应补真实成本。
- physical_edge 查询使用 {"id":"q","type":"physical_edge","edge":[u,v],"cost":...}；必须声明 edge_node_ids="physical"，全部候选提供稳定实物 ID 的边集。
- 实物边差异若只涉及串联 degree-2 设备，clade/root/cherry 特征可能全不变，当前结构损失给出零价值；这不能解释为核查该设备没有工程价值。
- 输出包含两侧质量、候选 ID/数量、条件风险、代表树、成本、排序值。certain_in_candidates 仅表示候选内确定。

在项目根目录运行：

~~~powershell
& D:\apps\miniconda3\envs\Topo\python.exe -B artifacts/prior_acquisition_design_20260908/rank_topology_queries.py artifacts/prior_acquisition_design_20260908/example_candidates.json --output artifacts/prior_acquisition_design_20260908/example_ranking.json --csv artifacts/prior_acquisition_design_20260908/example_ranking.csv
~~~

演示输入包含四个方法记录、三棵不同最简树；重复树合并权重：

| 查询 | yes 质量 | 成本 | clade 对称差下降 | 加权下降/成本 |
|---|---:|---:|---:|---:|
| exact cherry {1,2} | 0.75 | 6 | 1.3333 | 0.8125 |
| 1、2 同一级支路 | 0.75 | 10 | 1.3333 | 0.4875 |
| 1、3 同一级支路 | 0.50 | 15 | 1.0000 | 0.3083 |
| 完整 clade {1,2,3} | 0.25 | 25 | 0.6667 | 0.1017 |
| exact cherry {1,4} | 0 | 6 | 0 | 0 |

这是查询排序演示，不是四个实际馈线的核查收益实测。两个质量 0.75 的问题在此候选中恰好产生同一分割，工程语义仍不同，不能互换先验。

[测试](/D:/0-github_workspace/Topo/artifacts/prior_acquisition_design_20260908/test_rank_topology_queries.py) 覆盖可手算风险、重复树、确定事件、共同干线、clade/cherry 区分、实物 ID、非法权重/成本、交叉 clade、环、未知叶及重复 ID。日志在同目录 tests.log。

## 6. 人工核查后的重新拟合

1. 展示可查证的末端/设备标签、yes/no 代表树、主要差异及成本。隐藏节点只作示意，不能写成现场设备。
2. 记录答案、证据来源、时间、可靠度或无法判断，保留每次查询前的集合与权重。
3. 检查先验一致性。答案在候选中无支持时，扩大候选，不对空集合归一化。
4. 可表达的正先验映射为初始支持；负先验映射为明确候选过滤；无法表达的析取或实物事实保持为完整树过滤/待扩展约束。
5. 从原始观测重建受影响的初始族和候选池，重新拟合所有 R/X 系数及训练截距，用独立测试数据评价。不是只改一次权重或调大某个 \(S_{ab}\)。
6. 更新候选后再排序下一问；多个问题的单次价值不可简单相加。

优先采用不收缩的固定支持。只有可信、完整、互不相交的末端组才考虑收缩，并区分先验错误、去嵌入误差和候选缺失。人工否定旧冻结块时，必须允许从原问题回滚。

## 7. 评价未告知结构及假先验鲁棒性

至少比较无先验、仅扩充候选、硬先验不收缩、具备条件时收缩，以及同预算的随机/熵/结构损失成本查询。

排序不得使用真实拓扑。仿真真值只用于模拟核查答案和最终计分；现场答案来自独立核查。先验不能通过测试集挑选。

令 \(K\) 为直接告知或由先验逻辑蕴含的可评价 clade 事实。前后比较使用同一个 \(K\)，排除这些事实后报告 precision/recall/F1 和对称差。完整一级支路分区还蕴含一些禁止的跨支路 clade，不能把这些直接推论都算成算法新恢复。

正 clade 情形的直观指标为

\[
{\rm recall}_{\rm unknown}=
\frac{|(\widehat{\mathcal C}\setminus K)\cap(\mathcal C^\star\setminus K)|}
{|\mathcal C^\star\setminus K|}.
\]

分母为零时报告无未告知真结构，不填 100% 混入平均。同时保留完整结构成绩、先验正确率、真实树/clade 候选覆盖率、R/X 误差、固定训练截距的测试误差和时间。

“未告知边恢复率”仅在端点有稳定实物 ID 时按物理边评价；终端-only 隐藏树优先使用下游 clade 对应的最简边，不按每次新编号的隐藏节点匹配。

假先验实验固定数据和预算，分别注入少量错误肯定、错误否定、过期一级支路归属，记录：

- 冲突检出、拒绝及回滚次数；
- 未告知结构的错误传播及全结构损失；
- 验证/测试误差恶化、先验 R/X 权重退化为零；
- 候选域或收缩是否排除了真实结构。

零权重可提示先验与数据不一致，也可能由低信噪比或不可辨识导致，不能单凭它推翻现场事实。对假先验既报完整损害，也报未告知部分的传播，避免排除先验本身后掩盖直接错误。

## 8. 本轮实现边界

**已存在/已实现**：固定与候选支持接口；独立完整候选树查询排序、输入校验、两侧分割、代表树、风险成本输出及测试。

**可扩展但未接入**：外部先验台账、正负约束适配、完整 bootstrap 树保存、人工界面、问答后主线重拟合。

**尚无保证**：经验权重后验校准、候选外真值发现、错误回答自动纠正、核查成本准确性、真实未知边恢复量的最优性。有限候选中的分歧下降是可复核计算，不能替代这些保证。
