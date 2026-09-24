# NJ/RG 边缘检测驱动的一层聚合拓扑辨识

## 1. 方法定位

NJ、RG 和聚合都不直接充当最终拓扑。NJ/RG 只提出可信的末端区域；RNJ 负责
带已知根的隐藏树候选生成；固定树 `R/X` 重拟合、EIV 权重、留出 AC 误差与
quartet 条件共同选择最终树。

聚合采用软收缩：

1. 未聚合候选始终保留；
2. 聚合只生成额外候选，不删除原 terminal；
3. 检测器确认区域成员，但不冻结区域内部 sibling/clade；
4. 所有候选使用同一个不读取真值的物理评分器；
5. 真值仅用于实验结束后的 clade F1、完全恢复率和 candidate-oracle 诊断。

## 2. 输入与基础距离

输入为多个含噪 AC 潮流场景，每个场景含 terminal 的 `P/Q/V` 与已知根电压。
当前固定预处理为逐场景去均值。Ordered constrained regression 和 GTLS 产生
reduced `R/X` 候选，再由已知根 RNJ 生成基础隐藏树。

阻抗距离为

\[
d_{ij}^{(R)}=R_{ii}+R_{jj}-2R_{ij},\qquad
d_{ij}^{(X)}=X_{ii}+X_{jj}-2X_{ij}.
\]

默认先归一化，再使用 `0.75 dR + 0.25 dX`。

## 3. NJ/RG 外周区域与扰动稳定性

检测器同时运行：

- 无根 Neighbor Joining，再利用估计根深度放回已知物理根；
- 将物理根加入距离矩阵后的 root-augmented Recursive Grouping。

只提取 terminal 数不超过 `aggregation_max_cluster_size` 的外周 clade。基准量测
本身已经含噪；第 `b` 次 bootstrap 在其上叠加较小的第二层 `P/Q/V` 扰动，并做
循环时间块重采样：

\[
\widehat p_{\rm NJ}(C)=\frac1B\sum_{b=1}^{B}
\mathbf 1\{C\in {\cal C}_{\rm NJ}^{(b)}\}.
\]

候选全集取基础树与 bootstrap 树的 clade 并集，同时检查：

- NJ 或 NJ/RG 重现率；
- shared-path boundary margin；
- 归一化边长；
- Ordered RNJ 与 GTLS 支持；
- quartet 是否显著反证。

通过门控的区域形成至多 `nj_edge_candidate_beam_width` 个组合候选，避免枚举
全部 `2^K` 子集。

## 4. pseudo P/Q/V 与一层分解

对区域 `C_g`，pseudo 注入为

\[
P_g(t)=\sum_{i\in C_g}P_i(t),\qquad
Q_g(t)=\sum_{i\in C_g}Q_i(t).
\]

区域边界平方电压从每个 terminal 回推：

\[
\widetilde v_{g,i}^2(t)=v_i^2(t)
+\sum_{j\in C_g}[R_{ij}-R_{gj}]_+P_j(t)
+\sum_{j\in C_g}[X_{ij}-X_{gj}]_+Q_j(t),
\]

\[
\widehat v_g^2(t)
=(1-\lambda)\operatorname{mean}_{i\in C_g}v_i^2(t)
+\lambda\operatorname{median}_{i\in C_g}\widetilde v_{g,i}^2(t).
\]

默认 `lambda=0.50`（`pipeline/unified_topology_pipeline.py` 中的管线默认值；
`aggregate_rooted_scenarios` 函数签名默认 1.0，以管线值为准）。实现只使用簇内 `P/Q` 计算 service drop，避免估计稠密矩阵
的簇外小误差乘以全网负荷后污染 pseudo 电压。

随后执行：

1. 在 pseudo 节点上重新估计 Ordered `R/X` 与 RNJ 主干；
2. 以 `v_g(t)` 为局部根，对簇内原 terminal 再估计一次 `R/X` 与 RNJ；
3. 由 held-out NRMSE、clade stability、条件数和 `R^2` 门控局部结果；
4. 不合格时回退到基础局部结构；
5. 拼接主干与局部 clade，形成完整候选。

## 5. 全局物理选择

Ordered base、GTLS/RNJ-grid 和所有聚合候选共同进入：

1. 固定候选树上的非负可加 `R/X` 重拟合；
2. 传播 `P/Q/V` 测量误差的 structured-EIV 权重；
3. 代表性留出场景上的 AC predictive NLL；
4. quartet 显著性与近零隐藏边检查；
5. degree-2 hidden chain canonicalization。

最终输出是已知根节点的 canonical rooted terminal clade 集。选择过程不读取
真实拓扑。AC likelihood 只会重排已有候选，因此必须同步报告 candidate-oracle
recall，避免把候选遗漏误判为排序器问题。

## 6. 主要接口

```python
from terminal_case33.pipeline import UnifiedTopologyConfig, identify_topology_unified

config = UnifiedTopologyConfig(
    enable_nj_edge_aggregation=True,
    nj_edge_bootstrap_replicates=8,
    nj_edge_support_threshold=0.875,
    nj_edge_minimum_boundary_margin=0.10,
    nj_edge_minimum_length_ratio=0.02,
    nj_edge_rg_tolerance=0.03,
    nj_edge_require_external_confirmation=True,
    nj_edge_candidate_beam_width=6,
    aggregation_max_cluster_size=5,
    aggregation_deembedding_weight=0.50,
)

result = identify_topology_unified(
    scenarios,
    reference_net,
    pq_noise_relative_std=0.005,
    voltage_noise_relative_std=0.0002,
    config=config,
    seed=0,
)
```

关键审计字段为 `peripheral_proposals.evidence`、
`confirmed_detector_clusters`、`rejected_detector_clusters`、
`detector_aggregations`、`selection_source`、`ac_validated.ac_rerank` 和
`final_clades`。

## 7. 当前测试条件与命令

标准 sweep 使用 AC 真值、必有量测噪声和成对随机种子。常用条件为：

| regime | scenarios | samples/scenario | P/Q noise | V noise |
|---|---:|---:|---:|---:|
| sparse/nominal | 3 | 96 | 0.5% | 0.02% |
| sparse/high | 3 | 96 | 1.0% | 0.05% |
| medium/nominal | 5 | 288 | 0.5% | 0.02% |
| medium/high | 5 | 288 | 1.0% | 0.05% |

移除旧候选分支后，旧版汇总表不再作为当前实现的有效性能证据。应使用当前入口
重新生成结果：

```powershell
conda activate Topo
python -m pip install -e .
python -m experiments.run_unified_topology_pipeline_sweep `
  --output outputs\unified_no_frequency_sweep `
  --cases paper15 soumalas11 flynn16 pengwah18 `
  --data-regimes sparse medium rich `
  --noise-regimes low nominal high `
  --trials 2 --quartet-replicates 12 --nj-bootstrap-replicates 8
```

## 8. 当前边界

1. 聚合只能提高候选覆盖或局部辨识，不能保证每个条件都改善；
2. 稀疏高噪声下的主要失败可能仍是候选池缺少真树；
3. 固定树 EIV/AC 重排是主要计算开销；
4. degree-2 hidden chains 和零阻抗串联节点不能由 terminal 距离唯一恢复；
5. 评价对象必须是 canonical rooted clade，而不是 terminal-only MST。
