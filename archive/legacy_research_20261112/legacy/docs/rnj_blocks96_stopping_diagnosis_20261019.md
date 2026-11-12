# Soumalas11 四个完整 tree_init 路径的停止策略核查

日期：2026-10-19。操作：读取既存结果、核对冻结代码、对已保存求解结果作算术重放。没有重新拟合或调用求解器。本文是 Souma 四个条件/预定十六个网络条件、四个 tree_init job/正式六十四个 job 的阶段性诊断；生成本文时正式队列仍在运行，本轮原配置保持冻结。

## 结论与覆盖范围

当前 WZZT 已实现数值增益停止、求解证书检查和 RX75 边尺度下界。Soumalas11 的四个完整 `RNJ+WZZT_tree_init` 路径表明：初始有效拓扑正确后，增加假簇仍能降低全样本训练 MAE；改善量足以通过现有数值增益阈值。该现象对应训练目标与拓扑评价的差异。

本文覆盖以下四个已完成 job，每个 job 使用原来的一天 96 点全量拟合，无验证集、无独立测试集：

| 工况 | 结果来源 | 原生认证路径 K | 最终停止原因 |
| --- | --- | --- | --- |
| no_source | `outputs/rnj_blocks96_io_retry_20261019/jobs/soumalas11__no_source__RNJ+WZZT_tree_init` | 15 → 16 → 17 | `no_significant_one_atom_gain` |
| constant | `outputs/rnj_blocks96_20261019/jobs/soumalas11__constant__RNJ+WZZT_tree_init` | 16 → 17 | 同上 |
| intermittent | `outputs/rnj_blocks96_20261019/jobs/soumalas11__intermittent__RNJ+WZZT_tree_init` | 16 → 17 | 同上 |
| smooth | `outputs/rnj_blocks96_20261019/jobs/soumalas11__smooth__RNJ+WZZT_tree_init` | 16 → 17 | 同上 |

K 包括全部 singleton、初始普通支撑及后续支撑。no_source 原始 job 因记录文件替换失败保留为失败记录；本文使用隔离 I/O 恢复运行的完整结果。原始失败项的 path0 fallback 与其缺失的后续状态记录均未作为完成结果。

以下分析限定这四条路径，不外推为全 64 项结论。

## 冻结实现中的停止规则

核对的原生核心为 `outputs/source_ac288_runtime_20261018/rnj_wzzt/estimation/laminar_l1_milp.py`，SHA-256：

```text
e3a020bc40f7bb46b5875ac9db9e0d4c962c5f24ebae78a0c30743aae4b24fbd
```

本批原始冻结入口 `outputs/rnj_blocks96_20261019/source_snapshot/research_experiments/rnj/run_rnj_blocks96.py:298–315` 在 RX75 约束适配器内调用同一原生 `fit_laminar_l1_sensitivity`，显式传入 `validation_scenarios=None`、实际 initial supports、terminal groups 和按行对齐的 Source 输入。I/O 恢复入口的相同调用位于 `outputs/rnj_blocks96_io_retry_20261019/source_snapshot/research_experiments/rnj/run_rnj_blocks96.py:329–346`；两份入口及 floor 适配器分别按各自 payload 的源码指纹核对。

原生目标是全部端子、全部训练时点绝对残差的均值（核心 `896–904` 行）：

\[
L_k=\frac{1}{96n}\sum_{t=1}^{96}\sum_{i=1}^n
\left|Y_{ti}-\widehat Y_{ti}^{(k)}\right|.
\]

Y 为平方电压降，故 L 和绝对增益阈值的单位均为平方 pu。Source 模型的预测额外包含已知源时序乘以拟合的物理路径响应；每次固定支持拟合和扩展均联合重估源路径 h 与线路系数（核心 `1525–1527`、`1553–1554`、`1662–1674` 行）。Source 使用同一停止规则。

当前数值增益阈值为（核心 `1498–1500`、`1658–1660` 行）：

\[
\delta_k=10^{-10}+10^{-8}\max(|L_k|,\epsilon_{\mathrm{mach}}).
\]

这里的绝对项限制极小目标差异， relative 项随当前训练目标缩放。代码变量名为 `gain_tolerance`；在所核冻结核心和本批入口中没有独立的 `numerical_gain_tolerance` 参数。

同一个 δ 在四个位置参与分支：

| 位置 | 实际判据及作用 | 核心行号 |
| --- | --- | --- |
| 未认证扩展的对偶界 | 若有效下界 b 满足 `L_current - b <= δ`，停止为 `no_significant_one_atom_gain_certified_by_dual_bound`；否则保留当前模型并记 `extension_not_proven_optimal` | 1680–1690 |
| 扩界、剪零之前 | 若 `L_current - L_candidate <= δ`，立即停止 | 1703–1706 |
| 剪零重拟合的接受 | 对非冻结且 r、x 均不大于 `1e-9` 的原子尝试删除；仅当 `L_reduced <= L_candidate + δ` 才采用约简模型 | 1731–1746 |
| 剪零之后 | 若 `L_current - L_after_pruning <= δ`，停止 | 1748–1751 |

数值最优性接受规则另要求成功状态及严格的绝对/相对 gap 证书；相关容差均为 `1e-10`（核心 `37–38`、`289–319` 行）。该证书针对当前有界、分组、带 RX75 下界的单个新原子子问题。当前原生循环每次新增一个支持，没有 two-atom、swap 或 refinement 搜索阶段。

其他终止分支包括：不可行扩展、缺 incumbent、扩界次数达到上限、剪零后重复 family、`max_atoms`、极大层状族。候选触及系数上界的 99.5% 时，允许最多四次按通道翻倍上界并从初始 family 重新搜索；这一检查发生在 provisional gain 检查之后（核心 `1708–1729` 行）。初始化本身触界没有单独触发这段扩界逻辑。

验证集 1SE 是路径结束后的选点规则（核心 `626–636`、`1769–1771` 行），不提前中断 MILP 搜索。没有验证集时直接选择最后认证路径点。本批四个 job 均遵循这个无验证分支。

## 接受假边的实际改善量

以下数值来自保存的核心 path objective 与 extension objective。常数 Source 的一个核心 objective 与矩阵重算 MAE 相差约 `8.47e-11`，处于所用数值证书容差量级；为忠实重放状态机，本表统一使用保存的核心 objective。

| 工况 | 初始 L0 | 首次新增假簇 | 首步 ΔL | ΔL/L0 | 当时 δ | ΔL/δ |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| no_source（恢复） | 4.455535463e-4 | [101, 103] | 7.150917528e-7 | 0.160495% | 1.044555355e-10 | 6845.9 |
| constant | 4.446990383e-4 | [101, 103] | 7.167870795e-7 | 0.161185% | 1.044469904e-10 | 6862.7 |
| intermittent | 4.451128491e-4 | [101, 103] | 7.170825554e-7 | 0.161101% | 1.044511285e-10 | 6865.2 |
| smooth | 4.449720169e-4 | [107, 108] | 3.173753662e-7 | 0.071325% | 1.044497202e-10 | 3038.5 |

按当前证书与增益判据，这些接收决定有充分的数值余量；把它们归因于 `1e-10` 量级的舍入误差不符合保存证据。更低训练 MAE 为候选提供接受依据，事后物理真树比较则将新增支持识别为假簇。

no_source 在第一步后还增加假簇 `[107,108]`。涉及真树的拓扑评分集中列于文末事后附表，未用于下面的阈值重放。

## RX75 下界与噪声尺度截断的作用

本批固定几何校准来自对应工况的前置 ordered/FWL 回归矩阵，先取其对称部分 `R=(R_raw+R_raw.T)/2`、`X=(X_raw+X_raw.T)/2`；以下 R、X 均指对称后的矩阵。因此用原矩阵写距离时，等价形式为 `dR_ij=R_raw_ii+R_raw_jj-R_raw_ij-R_raw_ji`。对应代码位于本批 `source_snapshot/research_experiments/rnj/rx75_edge_floor.py:43–79`：

\[
\mu_R=\operatorname{mean}_{d^R_{ij}>0}d^R_{ij},\quad
\mu_X=\operatorname{mean}_{d^X_{ij}>0}d^X_{ij},\quad
a=0.75/\mu_R,\ b=0.25/\mu_X,
\]
\[
d^R_{ij}=R_{ii}+R_{jj}-2R_{ij},\qquad
\tau=0.16\max\{\operatorname{median}\operatorname{diag}(aR+bX),10^{-12}\}.
\]

每个旧非 singleton 支持的权重须为零或满足 `a*r+b*x >= τ`；新支持须直接满足该下界，并覆盖至少两个原始端子（适配器 `118–149` 行）。singleton 保留。输出分类以 `strength + 1e-10 >= τ` 视为达到分辨率（`88–115` 行）。

这个阈值限制归一化 RX75 几何中的边尺度。它没有估计“新增一条达到分辨率的边最多能够拟合多少噪声”，因此通过下界且降低训练 MAE 的假边仍可被接受。当前原生核心也没有噪声残差预算或显式复杂度惩罚。零权旧支持仍参与无条件层状约束，置零不会自动撤销其结构限制。

## 更大增益阈值能否只靠保存路径复现

一般情况下，final accepted path 不足以严格重放完整停止决策：

1. 扩界前先检查增益。较大 δ 可能在旧 epoch 停止，而原运行已经扩界并清空旧 path；需要 progress 保留的早期 epoch。
2. δ 同时决定是否采用剪零重拟合。较大 δ 可能接受原运行拒绝的约简 family，后续 MILP 的输入便不同，原路径后缀失效。
3. 终止时未接受的候选、其 primal/dual 证书及停止原因不包含在 accepted path 中；仅看相邻已接受点会遗漏这次检查。
4. two-atom 或 refinement 协议需要各自的候选和调度记录，不能套用单原子路径的相邻 ΔL。本批所核核心没有这些算子。

这里的可重放含义是：固定已归档的 solver solutions 与并列最优解选择，重算状态机的阈值分支。新求解器运行可能选择另一个并列最优支持；本文没有重新求解，也没有估算新运行耗时。

四个局部 job 逐项通过以下充分条件，故可以重放预设更大 δ 的前缀：

| 适用条件编号 | 工况 | 扩界次数 | 每次 attempt 中可触发剪零的非冻结原子数 | 固定+扩展 solve 数 | 已接受点数 | 完整最后拒绝 attempt |
| --- | --- | ---: | --- | ---: | ---: | --- |
| A | no_source（恢复） | 0 | 0, 0, 0 | 4 | 3 | 有 |
| B | constant | 0 | 0, 0 | 3 | 2 | 有 |
| C | intermittent | 0 | 0, 0 | 3 | 2 | 有 |
| D | smooth | 0 | 0, 0 | 3 | 2 | 有 |

同时核对了：全部保存 solve 满足核心数值证书判据；没有额外固定支持重拟合；每个 accepted family 恰为上个 family 加本次候选；系数、h 与 objective 对齐；最终 `stopped` 快照的 path/attempts 与 final model 一致。用原生 relative=`1e-8` 重放，四项均复现原生选点和停止原因。任一前置条件失败时，脚本输出 `diverged`，不截取该路径冒充反事实结果。

## 统一阈值敏感性表与全 96 点建议

保留绝对项 `1e-10`，对事先给定的三个十进相对尺度 `{1e-4, 1e-3, 1e-2}` 全部报告。该网格是本次事后敏感性诊断；没有据 F1 从中选定新方法参数。

下表 ΔL 是触发停止的那次实际归档候选增益，δ 在所选当前点计算；负 ΔL 表示强制新增达到 RX75 分辨率的支持后训练目标反而变差。停止时保留当前点，尚需求解这一候选才能作出停止决定。适用条件编号对应上一表，不表示其他 job 也可这样重放。

| 工况 | relative | 所选 index / K | 停止候选实际 ΔL | 当时 δ | 局部适用条件 |
| --- | ---: | ---: | ---: | ---: | --- |
| no_source（恢复） | 1e-4 | 2 / 17 | -5.279665978e-7 | 4.454924054e-8 | A |
| no_source（恢复） | 1e-3 | 1 / 16 | 3.460491971e-7 | 4.449384546e-7 | A |
| no_source（恢复） | 1e-2 | 0 / 15 | 7.150917528e-7 | 4.455635463e-6 | A |
| constant | 1e-4 | 1 / 17 | -2.276061886e-7 | 4.449822512e-8 | B |
| constant | 1e-3 | 1 / 17 | -2.276061886e-7 | 4.440822512e-7 | B |
| constant | 1e-2 | 0 / 16 | 7.167870795e-7 | 4.447090383e-6 | B |
| intermittent | 1e-4 | 1 / 17 | -5.588513317e-7 | 4.453957665e-8 | C |
| intermittent | 1e-3 | 1 / 17 | -5.588513317e-7 | 4.444957665e-7 | C |
| intermittent | 1e-2 | 0 / 16 | 7.170825554e-7 | 4.451228491e-6 | C |
| smooth | 1e-4 | 1 / 17 | -5.065084123e-7 | 4.456546416e-8 | D |
| smooth | 1e-3 | 0 / 16 | 3.173753662e-7 | 4.450720169e-7 | D |
| smooth | 1e-2 | 0 / 16 | 3.173753662e-7 | 4.449820169e-6 | D |

12 项重放状态均为 `replayed_archived_trace`。完整 MAE、逐 attempt 增益/阈值、前置检查及输入 SHA 保存于：

- `tmp/rnj_blocks96_soumalas11_gain_sensitivity_versioned_20261019.json`
- `tmp/rnj_blocks96_soumalas11_gain_sensitivity_versioned_20261019.csv`

复现命令：

```powershell
& 'D:/apps/miniconda3/envs/Topo/python.exe' research_experiments/rnj/replay_soumalas11_gain_thresholds96.py
```

脚本仅使用 Python 标准库，读取上述四个完成 job、不可变 progress 与冻结原生代码；不读取物理真网，不访问 result.metrics 参与计算，不导入项目估计器。JSON 的 `input_sha256` 保存所有实际读取输入的逐文件 SHA，`script_sha256` 保存脚本 SHA；输出仅写入本任务 tmp 文件。版本化入口使用新工件名，先前 `tmp/replay_soumalas11_gain_thresholds_20261019.py` 及其旧 JSON/CSV 保留为生成记录，未覆盖。

本次工件 SHA-256 清单如下，输入的完整清单位于上述 JSON 的 `input_sha256` 字段：

| 工件 | SHA-256 |
| --- | --- |
| `research_experiments/rnj/replay_soumalas11_gain_thresholds96.py` | `86f60cf4b2ff97969da11562986dc0a2e15d79f6292bb1d47fde07a135badd95` |
| `tmp/rnj_blocks96_soumalas11_gain_sensitivity_versioned_20261019.json` | `1a44638ce23b8254f717b2bddbbe07bb03be9b9a9e3e70a925d3b1e226375b23` |
| `tmp/rnj_blocks96_soumalas11_gain_sensitivity_versioned_20261019.csv` | `0d106d7d3dd8d2204c7b21feb6f88ba4d5e332b52e04175902174f5fd119e456` |
| 事后评分来源 `tmp/rnj_blocks96_soumalas11_final_diagnostic_20261019_paths.csv` | `776df683a209a922ea1cad38ac0e0914eb94b76608cb35a62827060654e9398a` |

下一步最小工作是：待结果稳定后，将相同的分支核验和统一网格用于其他完成路径，保留不可重放项。继续采用全 96 点拟合并保留原 lastpoint 对照；阈值敏感性结果单独标记，评价仍使用完整物理真树。

若随后需要一个依数据尺度自动停止的单一规则，可以预声明按已知仪表误差传播得到 MAE 噪声尺度，并在首次认证路径达到该尺度时截断。该方案需纳入实测根电压的跨端子共同误差，并明确 AC 模型偏差和 P/Q 误差的处理；这些条件完成前，它属于待实现的噪声尺度启发式。当前文档和算术重放均未将其接入拟合，也未恢复 validation 划分或引入 BIC。

## 事后拓扑评分附表

本表沿用先前完成的 Souma 物理真网逐路径诊断，作用是说明原生搜索在这四条路径上增加了假簇。它与阈值重放的计算输入和参数选择分开；本报告没有按这些 F1 从三个相对尺度中挑选一个方案。

| 工况 | 原生选点 index | 初始 family / active / resolution F1 | 原生最终 family / active / resolution F1 |
| --- | ---: | --- | --- |
| no_source（恢复） | 2 | 1 / 1 / 1 | 0.8 / 0.8 / 0.8 |
| constant | 1 | 0.888889 / 1 / 1 | 0.8 / 0.888889 / 0.888889 |
| intermittent | 1 | 0.888889 / 1 / 1 | 0.8 / 0.888889 / 0.888889 |
| smooth | 1 | 0.888889 / 1 / 1 | 0.8 / 0.888889 / 0.888889 |

源场景初始普通 family 含零权 `[104,105]`，故 family F1 与有效拓扑 F1 不同。active 使用 `max(r,x)>1e-8`，resolution 使用固定 RX75 τ 分类，真簇均按完整物理树计分。
