# MIQP RNJ＋WZZT → 条件 DR 主线

以后从本目录进入研究。辨识器采用 2026-11-12 主项目的现有简化实现；DR 复用已交接的积分、严格 MSE 评分、设备执行器和单对优化器。历史批量控制器、画图、重复算法和大型观测文件留在原位置。

```text
观测训练 P/Q/Y → RNJ 几何与候选 → W 单支持扩展 → 可选 T / E → 保存自身路径
观测验证 P/Q/Y → 原严格 MSE 选择 → 冻结完整候选库和 R/X
历史功率预测＋原资源 → 给定目标 Ht → 整库零 DR 核算
  ├─ 零动作充分且原设备可行：最小费用 0
  └─ 仍有缺口：已有单对报价；多竞争者非零报价待接通
冻结计划 → 原设备随机响应 / 截断 → 原 AC 含噪观测 → 原整库评分与完整恢复评价
```

## 一套算法，分别记录四个规则

| 环节 | 本目录实际规则 |
|---|---|
| 训练搜索 | `MSE_train / M_ref + .003 × 内部支持数`；提议先固定族 QP，再按统一目标接受 |
| 搜索与停止 | 原 W 逐支持扩展；T 为至多 5 个直接块的局部联合重构；一次 E 释放临时内部支持；保留预算、状态和界冲突 |
| 输出参数 | 保存已接受自身路径及单点保护模型；参数冻结；最终无 `.24` 收缩，无完整 RNJ 整树兜底 |
| 最终选择 | 验证 MSE / `M_ref`；数值并列容差 `1e-12`，再依内部支持数、分数、原序选择 |

原默认 `111`（R＋W＋T＋E）保持；`100/rnj_wzzt` 是 R＋W，`101` 是 R＋W＋E。已有开发回放中 101 的表现更好，尚不足以自动替换默认。`relative_gain=0` 保持原默认。`.003` 的双候选展示阈值只影响展示；DR 始终核算完整冻结库中的所有不同活动拓扑。

**与提示词的版本差异：**旧适配器的 3% MAE／`.16`，后续 MIQP 的 `.24`／E 专用 `.001`，都是历史规则。本目录接当前验证 MSE 版本，物理 MSE 容差为 `M_ref × 1e-12`。详见 [MODEL.md](MODEL.md)。

## 最短使用路径

Python 3.10+；NumPy、SciPy 支持观测读取、RNJ/NJ、QP、评分与零动作核算；结构 MIQP 和非零报价还需 Gurobi 与有效许可。

```powershell
$py = 'D:/apps/miniconda3/envs/Topo/python.exe'
$run = 'D:/0-github_workspace/Topo/rnj_wzzt_dr/run.py'
& $py -I -B $run doctor
& $py -I -B $run check --solver

# 所有结果和报价放私人目录；以下 outputs 默认被 Git 忽略。
& $py -I -B $run fit --train /path/train.npz --validation /path/validation.npz --method 111 --output /private/outputs/fit
& $py -I -B $run zero --result /private/outputs/fit/result.json --history /path/validation.npz --forecast /private/forecast.npz --target 0 --voltage-sigma .0005 --case paper15 --output /private/outputs/zero.json
& $py -I -B $run plan --result /private/outputs/fit/result.json --history /path/validation.npz --forecast /private/forecast.npz --target 0 --voltage-sigma .0005 --case paper15 --output /private/outputs/quote
& $py -I -B $run score --result /private/outputs/fit/result.json --history /path/validation.npz --new /private/actual_new_day.npz --output /private/outputs/selection.json
```

把示例路径换成实际绝对路径。观测 NPZ 包含 `labels, P, Q, V, root`，也可用 `Y` 代替 `V/root`；`Y=root²−V²`，功率为 pu。预测 NPZ 包含 `labels, power`，`power` 是 `[新日时槽, 先全部 P 再全部 Q]`。预测只由已知历史形成，时槽对齐须明确；64/32 划分不能沿用旧四个完整日均值。输入接口没有真树或测试集参数。

目录可整体复制改名，`python -I /copied/name/run.py doctor` 仍能工作。`fit` 保存输入／代码哈希和环境，`score` 检查历史与冻结参数，不重拟合；已有输出目录会被保留。

## 留下的文件

| 位置 | 职责 |
|---|---|
| `identification/` | 现有 `algorithm/optimization/trees/competition` 四文件原字节复制；唯一辨识器 |
| `observations.py`、`history.py` | 观测输入、完整冻结库、原最终选择及 SSE 门槛 |
| `dr/pipeline.py` | 明确目标方向、零动作核算、原单对报价入口 |
| `dr/conditional.py`、`zero.py`、`miqp_mse.py` | 原稠密积分、谱积分、单对带整数 QCQP；移除旧实验导入 |
| `dr/device*`、`resource_adapter.py`、`profiles.py` | 原设备、小时请求图、费用及随机响应 |
| `resources/` | 原四网络合成资源配置；私人拟合参数和报价另存 |
| `compat/` | 原 AC 桥及两个直接辅助文件；实际运行仍需原冻结物理模拟源和网络资产 |
| `evidence/` | 原 DR 交接和零费结果，按旧版本解释 |
| `PROVENANCE.json`、`verify.py` | 来源、抽取符号、改动及哈希；无需旧代码即可核验本包 |

## 已完成与下一项最小工作

当前已整理代码和接通接口，并做确定性的小规模验证，见 [VALIDATION.md](VALIDATION.md)。可用算例资格分两批：

| 既有记录 | 完整真拓扑覆盖／历史选择 | 零 DR 状态 |
|---|---|---|
| 当前简化引擎 111，原 48 输入 | 输出库含真拓扑 38/48，历史选对 38/48；10 组缺完整真族 | 38 组仍需按当前参数／历史算零动作条件 |
| 当前简化引擎 101，同一 48 输入 | 输出库含真拓扑 39/48，历史选对 39/48；9 组缺完整真族 | 39 组仍需核算；与 111 有重叠，不能相加 |
| 旧 DR `.24/QP` 库 | 40/48 有完整真候选；8 组缺完整真族 | 40 组／188 比较已 complete，零动作整库 90% 充分条件全部通过 |

前两行读取既有 [覆盖审计](evidence/current48/coverage_summary.csv)，没有新拟合或零动作计算；它们使用训练 96＋验证 288。原 T 的 4 次下界曾被同域 QP 点推翻，原最终模型已作 QP 闭合，见 [原补充核查](evidence/current48/ERRATA_QP_20261112.md)。后续严格总 96 点／大样本批次有独立数据与完成证据，仍保留在原独立目录，不并入本资格表。旧 paper15 的 17/32 收益属于旧 MAE；同观测严格 MSE 下三策略都是 32/32。

下一步先对**当前版本**的冻结结果核候选覆盖及零动作缺口。只保留库含真拓扑、原要求仍有缺口的算例资格；真值仅用于离线资格与选后评价。若找到多竞争者非零例，再接固定风险预算的联合优化器，随后锁一个有界独立实际 AC 测试。此次没有重跑旧完成批次，也没有启动新研究批量。

90% 指目标正确及所声明误差法则下的条件激励事件。设备参数是合成设定；实际相关响应、截断和 AC 达成率另报。归档旧代码前按 [ARCHIVE.md](ARCHIVE.md) 保留数据和物理模拟源。

## 简单后验权重的既有算例验证

按 `sigma²=min validation MSE`、`s=2 sigma²/N` 的方案，已完成96个历史候选库和4个严格96点补充库的重评分。主组92次最高权重≥90%的方法／输入比较中，75次正确、17次错误；错误全部来自库缺完整真拓扑。库含真的77次均选对。结果、难区分样本和复现入口见 [实验说明](experiments/posterior_mse_20261112/README.md)。这是独立诊断，当前辨识和DR默认规则保持原样。
