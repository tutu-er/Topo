# 三篇拓扑辨识论文的初步复现记录

本文件记录 `experiments/run_paper_reproduction_attempt.py` 对三篇文章的第一轮复现范围。这里的“复现”是工程上可运行的机制复现，不声称复现论文中的私有真实数据、全部随机网络、全部图表或精确数值。

对应原文 PDF 已抽取为文本缓存：

- `docs/paper_notes/soumalas_2017.txt`
- `docs/paper_notes/flynn_2023.txt`
- `docs/paper_notes/pengwah_2024.txt`

## 1. Soumalas et al. 2017

论文核心设定：

- LV radial distribution grid；
- smart meter 安装在 consumer leaf nodes；
- parent/intermediate nodes 不监测；
- root/slack 已知；
- 使用同步 P/Q/V 估计 reduced voltage sensitivity matrix；
- 从 reduced sensitivity 得到 leaf-to-leaf additive distance；
- 通过 Prüfer sequence 和候选 line type/length scenario 枚举恢复拓扑；
- 服务线长度案例包括 30 m，以及 28-32 m 随机扰动；
- 论文显式依赖 LV line type、minimum length、integer/multiple-length 假设。

本项目已复现/对齐：

- `hybrid_leaf` / `all_loads_to_new_terminal_leaves` 可生成 leaf-only smart-meter 场景；
- `lv_cable_soumalas_style` 支持 30 m service line 和 28-32 m 随机长度；
- `build_reduced_sensitivity_matrices` 生成 terminal reduced R/X；
- `impedance_distance_from_reduced_R/X` 生成 additive distance；
- `prufer_style_placeholder` 明确保留 Soumalas-style Prüfer 接口边界；
- 本轮新增 `neighbor_joining_baseline` 作为可运行 additive-distance hidden-tree baseline。

尚未完整复现：

- 从距离矩阵抽取 Prüfer sequence 的整数线长算法；
- 多 candidate line type / length scenario 枚举；
- 用 power-flow voltage error 反选 scenario 的完整循环。

当前建议：先用 NJ/recursive grouping 作为稳健 baseline；如确实要复现 Soumalas Prüfer，需要先明确 line length quantization 和 service/overhead line library。

## 2. Flynn et al. 2023

论文核心设定：

- 使用 smart meter 数据估计 LV radial topology；
- 主要改进之一是 sensitivity estimation 中显式考虑 transformer voltage variation；
- 之后用 improved recursive grouping 进行拓扑估计；
- 还提出 fault detection 和新的 tree similarity metric；
- 仿真包括 20-node、50-80 vertices 的随机网络，以及真实澳洲 smart meter 数据。

本项目已复现/对齐：

- `estimate_reduced_sensitivity(..., include_root_voltage_mode=True)` 加入公共 voltage/root mode；
- `run_paper_reproduction_attempt.py` 对比 root voltage variation 下 `include_root_voltage_mode=False/True` 的 R/X 误差；
- `neighbor_joining_baseline` 提供 latent-tree additive-distance baseline；
- 输出 `metrics.json` 中保留 no-root 和 with-root-mode 的 RMSE、relative error、R2、condition number。

尚未完整复现：

- Flynn improved recursive grouping 的 parent/child、sibling、noise constraint 细节；
- 论文的 fault detection statistic；
- 论文提出的 tree similarity metric；
- 真实 DNSP smart meter 数据实验。

当前建议：下一步优先实现 Flynn 的 improved recursive grouping，因为它和当前 terminal-only hidden-node tree 目标最贴近。

## 3. Pengwah et al. 2024

论文核心设定：

- LV network 中 customers 分成 smart meters `S` 和 interval meters `I`；
- smart meter 有 V/P/Q，interval meter 没有 V 和 Q，只有低分辨率能量/有功信息；
- 先估计包含 smart/interval 的 impedance model；
- 再先恢复 smart-meter subgraph `G_S`；
- 最后把 interval meters 逐个定位到 `G_S` 的节点或边上；
- 论文评价包括随机网络、smart-meter penetration、measurement class、IEEE European LV feeder、mean hop error。

本项目已复现/对齐：

- `run_paper_reproduction_attempt.py` 随机划分 70% smart meters / 30% interval meters；
- 对 smart-meter 子集构造 reduced distance 和 matrix-tree baseline；
- 输出 smart/interval meter 清单、smart subgraph edge scores、edge marginals、MAP tree。

尚未完整复现：

- 使用 interval meter 有功/能量和缺失 V/Q 的阻抗优化；
- interval meter location 的 Case A / Case B 节点/边插入算法；
- mean hop error 指标；
- IEEE European LV feeder 的三相/分相实验。

当前建议：Pengwah 复现应排在 Flynn RG 之后，因为它依赖一个较可靠的 smart-meter subgraph 估计结果。

## 运行命令

```bash
cd D:\0-github_workspace\Topo
python -m experiments.run_paper_reproduction_attempt --output outputs/paper_reproduction_attempts
```

主要输出：

- `metrics.json`
- `report.md`
- `terminalized_case33_lv_style.png`
- `soumalas_nj_latent_edges.csv`
- `pengwah_smart_meters.csv`
- `pengwah_interval_meters.csv`
- `pengwah_smart_edge_scores.csv`
- `pengwah_smart_edge_marginals.csv`
- `pengwah_smart_map_tree.csv`
- `R_true.csv`, `X_true.csv`, `dR_true.csv`, `dX_true.csv`

## 当前复现结论

- Soumalas: 场景和 reduced sensitivity/distance 已复现，Prüfer 部分仍是明确 TODO。
- Flynn: transformer/root voltage variation 的核心改进已能做对比实验，RG/fault/tree metric 仍是 TODO。
- Pengwah: smart/interval meter 的部分可观测框架已建立，interval location 优化仍是 TODO。
