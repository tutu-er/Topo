# 当前 terminalized case33 上的论文算法复现表现

运行命令：

```bash
cd D:\0-github_workspace\Topo
python -m experiments.run_paper_full_reproduction --output outputs/paper_full_reproduction
```

输出目录：`outputs/paper_full_reproduction/`

## 算例

本次比较使用同一个 terminalized IEEE case33 LV-style 场景：

- root bus: 1
- observed terminal nodes: 32
- hidden internal nodes: 28
- load buses: 32
- hidden degree distribution: degree-3 有 25 个，degree-4 有 3 个
- degree-2 hidden chains: 0
- strict terminal-load-only violations: 0

评价基准为真实 reduced sensitivity matrix 生成的 terminal-equivalent MST。注意：terminal-equivalent MST 不是完整物理隐藏节点树，只用于公平比较各算法在 observed terminal 层面的拓扑表现。

## 指标总表

| 算法变体 | R 相对误差 | X 相对误差 | dR RMSE | dX RMSE | terminal MST F1 | NJ 距离 RMSE | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| Soumalas 2017 regulated-root | 1.279e-4 | 2.231e-4 | 4.875e-5 | 8.369e-5 | 1.000 | 3.397e-5 | 根电压恒定，距离法几乎精确 |
| Flynn 2023 baseline, no transformer mode | 48.595 | 135.135 | 10.978 | 21.840 | 0.0645 | 6.754 | 根电压波动下基本崩溃 |
| Flynn 2023 with transformer/common mode | 0.7736 | 0.7420 | 0.0893 | 0.0648 | 0.8387 | 0.0556 | 公共电压模态显著改善拓扑 |
| Pengwah 2024 mixed-meter proxy | - | - | - | - | 0.8095 smart subgraph | - | 70% smart / 30% interval |

## Soumalas et al. 2017

当前实现：

- 使用 terminal-only smart meter 场景；
- root voltage 恒定；
- 从 P/Q/V 估计 reduced R/X；
- 从 R/X 构造 terminal additive distance；
- 用 MST 和 NJ 替代尚未完全实现的 Prüfer integer-length decoding。

表现：

- `relative_error_R = 1.279e-4`
- `relative_error_X = 2.231e-4`
- `terminal_mst_f1 = 1.0`
- `nj_distance_rmse_R = 3.397e-5`

解释：

在当前无噪声、根电压恒定、负荷激励充分的设置下，Soumalas 的 reduced sensitivity -> distance 路线表现非常好。真正尚未复现的是论文里的 Prüfer sequence extraction 和 line type/length scenario enumeration。当前结果说明：如果 sensitivity 矩阵准确，terminal-equivalent topology 很容易恢复。

## Flynn et al. 2023

当前实现比较了两种情况：

1. 不考虑 transformer/root voltage variation；
2. 在 sensitivity regression 中加入公共 root/common voltage mode。

表现：

| 指标 | 无 transformer mode | 有 transformer/common mode |
|---|---:|---:|
| R2 score | 0.08799 | 1.00000 |
| R relative error | 48.595 | 0.7736 |
| X relative error | 135.135 | 0.7420 |
| terminal MST F1 | 0.0645 | 0.8387 |
| NJ distance RMSE | 6.754 | 0.0556 |

解释：

根电压波动会严重污染传统 sensitivity estimation，导致拓扑几乎不可用；加入公共电压模态后，拓扑 F1 从 0.0645 提升到 0.8387。这和 Flynn 论文的核心主张一致：transformer voltage variation 必须显式处理。

但当前还不是 Flynn 完整算法，因为 improved recursive grouping、fault detection 和论文 tree similarity metric 还没有完全实现。当前实现复现的是其最关键的 sensitivity-estimation 改进在本算例上的效果。

## Pengwah et al. 2024

当前实现：

- 将 32 个 terminal meters 随机分成 70% smart meters 和 30% interval meters；
- smart meters: 22 个；
- interval meters: 10 个；
- 对 smart-meter 子图构造 distance/matrix-tree 后验；
- 对 interval meter 用 estimated R 的 nearest-smart proxy 做位置归属。

表现：

- `smart_subgraph_terminal_mst_f1 = 0.8095`
- `smart_subgraph_sum_marginals = 21.0 = 22 - 1`
- `interval nearest-smart accuracy = 0.9`
- `mean_distance_regret_true_R = 9.858e-5`
- `max_distance_regret_true_R = 9.858e-4`

interval meter 定位结果：

| interval bus | predicted nearest smart | true nearest smart | correct |
|---:|---:|---:|---:|
| 22 | 1021 | 1021 | 1 |
| 25 | 1024 | 1024 | 1 |
| 1002 | 1019 | 1019 | 1 |
| 1006 | 1026 | 1007 | 0 |
| 1009 | 1008 | 1008 | 1 |
| 1011 | 1010 | 1010 | 1 |
| 1012 | 1010 | 1010 | 1 |
| 1020 | 1021 | 1021 | 1 |
| 1030 | 1029 | 1029 | 1 |
| 1031 | 1032 | 1032 | 1 |

解释：

Pengwah 路线在 partial AMI 设置下仍能得到较好的 smart subgraph 和 interval 近邻定位。但当前 interval placement 只是 nearest-smart proxy，并不是论文中的完整 convex impedance model + Case A/B edge-splitting algorithm。完整复现需要进一步实现缺失 V/Q 的阻抗优化和基于 `β_n` 的节点/边插入。

## 当前排序

在当前 terminalized case33 上，按已实现部分的表现可粗略排序：

1. Soumalas regulated-root 路线：在理想假设下几乎完美；
2. Flynn with transformer/common mode：面对根电压波动仍能恢复大部分 terminal topology；
3. Pengwah mixed-meter proxy：部分可观测场景下 smart 子图和 interval 近邻定位表现较好；
4. Flynn baseline no transformer mode：根电压波动下表现很差。

## 重要边界

- Soumalas 的 Prüfer integer-length decoding 尚未完整实现；
- Flynn 的 improved recursive grouping 尚未完整实现；
- Pengwah 的 interval-meter impedance optimization 和 Case A/B placement 尚未完整实现；
- 当前拓扑评价使用 terminal-equivalent MST，不等同于完整物理隐藏节点树评价；
- 当前实验主要展示算法机制在本项目算例上的表现，不声称复现论文全部私有数据结果。
