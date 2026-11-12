# RNJ 直接使用 R、X 及组合分数的修正与算例结论

## 1. 修正后的实际排序量

对 reduced sensitivity matrix，非对角元素已经是根到最近公共祖先的公共路径和：

\[
R_{ij}=2\sum_{e\in P(0,i)\cap P(0,j)}r_e,\qquad
X_{ij}=2\sum_{e\in P(0,i)\cap P(0,j)}x_e.
\]

因此 RNJ 直接使用

\[
S_{ij}=c_RR_{ij}+c_XX_{ij},\qquad H_i=S_{ii}
\]

即可。旧流程先构造

\[
d_{ij}=H_i+H_j-2S_{ij}
\]

再计算 `(H_i+H_j-d_ij)/2`，严格返回原来的 `S_ij`，没有增加信息。当前实现已把
RNJ 主路径改为直接输入 `S`；距离仍保留给 NJ/RG、四点条件和加性度量诊断。

## 2. 可行性边界

- 精确树矩阵下，只要 `c_R,c_X` 非负且不同时为零，`S` 仍是同一棵树上的正边权
  shared-path matrix；只改变边长，不改变根向 clade。
- 有噪估计下，单独按 `R_ij` 或 `X_ij` 排序在代数上可行，但统计表现取决于对应回归
  通道的信噪比。不能由精确矩阵的拓扑等价性推出两者在估计数据上同样稳定。
- `R` 与 `X` 的数值尺度不同，混合前必须归一化。当前默认
  `S=0.75 R/s_R + 0.25 X/s_X`，其中 `s_R,s_X` 是相应阻抗距离正元素均值。

## 3. 四个典型系统的真值测试

测试系统为 `paper15`、`soumalas11`、`flynn16`、`pengwah18`。直接分数与旧距离回算
的最大绝对差为 `2.22e-16`（浮点舍入量级）。R-only、X-only、RX75 在四个系统的
精确真值矩阵上均得到 rooted-clade F1 = 1.0。

## 4. 有噪 AC 系统表现

复用大规模矩阵约束消融中的同条件切片：ordered 估计、容差因子 0.16、场景数不少于
3，覆盖四个馈线、3/5/10 场景、48/96/288 点和两个重复；每个排序模式 72 次。

| 直接分数 | 平均 rooted-clade F1 | 平均 sibling F1 | 完全恢复率 |
|---|---:|---:|---:|
| R | 0.8343 | 0.8482 | 16.67% |
| X | 0.5080 | 0.5262 | 0.00% |
| RX75 | 0.9681 | 0.9734 | 72.22% |

另外用修正后的直接分数代码路径新跑了四馈线 smoke benchmark（3 场景、96 点、raw 与
daily-demean 两种预处理，共 8 次/模式）：R、X、RX75 的平均 rooted-clade F1 分别为
`0.7972`、`0.3399`、`0.9833`，根分区成功率分别为 `75%`、`50%`、`100%`；三种模式的
精确真值 oracle 均为 1.0。该新跑结果与 72 次/模式的大样本切片给出的排序结论一致。

X-only 在真值上与 R 同序，但估计后明显更差，说明失败来自 X 通道估计误差和激励条件，
不是 RNJ 排序公式。当前系统应继续以归一化 RX75 为默认；R-only 可作为候选或消融，
X-only 不宜单独作为默认重构量。

### 挪威公开工业配电网

本地公开数据还包含两个真实/匿名化径向拓扑和真实小时有功负荷。真值 R、X 上，直接分数
与距离回算的最大差仍为 `2.22e-16`。
但该数据的等值模型含大量理想零阻抗变压器/连接边：radial 1 为 49/53 条，radial 2
为 12/16 条。它们不产生可辨识的阻抗 split，说明“可以直接排序”并不等于“所有物理边
都能从 terminal 的 R、X 矩阵唯一恢复”。

使用公开真实有功负荷、功率因数补全 Q、AC 生成电压的半实证实验中：

- radial 1（40 个 terminal）在 7/30/90 天下 identifiable rooted-clade F1 均为 0；
- radial 2（5 个 terminal）在 7/30/90 天下 F1 均为 1。

因此大系统失败不是这次删除 `rho` 往返造成的；直接式与旧式数值等价。主要限制是
零阻抗可辨识性、仅有功真实量测下的 X 通道弱辨识，以及大 radial 的激励/模型失配。

## 5. 输出与复现

- `outputs/direct_rx_rnj_analysis/true_R_X_heatmaps.png`：四个典型系统的真值
  `R_ij`、`X_ij` 公共路径矩阵热图。
- `outputs/direct_rx_rnj_analysis/fitted_R_X_RX75_heatmaps.png`：相同有噪 AC 样本拟合得到的
  `R_ij`、`X_ij`，以及归一化组合 RX75 热图。
- `outputs/direct_rx_rnj_analysis/true_R_X_sorted_pair_bars.png`：真值 R、X 的全部终端对
  分数降序柱状图。
- `outputs/direct_rx_rnj_analysis/fitted_R_X_RX75_sorted_pair_bars.png`：拟合 R、X、RX75 的
  全部终端对分数降序柱状图。颜色和柱顶记号由真实矩阵中的 `(R_ij,X_ij)` 等值簇决定；
  同一颜色/记号在三个拟合模式中始终表示同一真实末端对簇。
- `outputs/direct_rx_rnj_analysis/fitted_pair_bars_clustered/`：四个算例各自的可放大簇图。
- `outputs/direct_rx_rnj_analysis/true_vs_fitted_R_sorted_bars.png`：按真实 R 降序且按同一
  terminal pair 对齐的真值/拟合值叠加柱状图，以及 `fitted R - true R` 误差图。宽浅柱为
  真值、窄深柱为拟合值；第三栏将拟合 R 独立按拟合值降序排列。真实 R 应为 0 的末端对
  统一显示为灰黑色并加灰色背景。
- `outputs/direct_rx_rnj_analysis/noisy_ac_R_X_RX75_performance.png`：R、X、RX75 的有噪
  AC 拓扑恢复柱状图。
- `outputs/direct_rx_rnj_analysis/norwegian_true_R_X_heatmaps.png`：挪威两个公开 radial 的
  真实 R、X 公共路径热图。
- `outputs/direct_rx_rnj_analysis/norwegian_true_edge_R_X_bars.png`：公开真实拓扑中正阻抗
  支路的 R、X 柱状图，并注明被省略的理想零阻抗支路数量。
- `outputs/direct_rx_rnj_analysis/norwegian_real_load_rnj_performance.png`：7/30/90 天真实
  有功负荷驱动的 RNJ 可辨识 clade F1。
- `true_value_cluster_membership.csv` 保存各终端对的真实 R、X、簇编号；
  `true_vs_fitted_R_pair_comparison.csv` 保存逐 terminal pair 的真值、拟合值和误差；
  `sorted_terminal_pair_scores.csv` 保存每个算例、每种分数的 terminal pair、数值和降序名次；
  其余 CSV 和 JSON 保存拟合误差及拓扑恢复指标。
- `outputs/direct_rx_rnj_smoke_R`、`outputs/direct_rx_rnj_smoke_X`、
  `outputs/direct_rx_rnj_smoke_RX-75R-25X` 保存修正后代码路径的新跑 smoke benchmark。

复现命令：

```powershell
python -m experiments.run_direct_rx_rnj_analysis `
  --output outputs/direct_rx_rnj_analysis
```
