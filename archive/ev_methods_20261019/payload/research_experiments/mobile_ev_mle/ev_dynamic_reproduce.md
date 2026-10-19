# 本轮实验如何复核和重跑

工作分支：`codex/ev-train-validation-design`。数学与场景见 `ev_dynamic_math.md`、`ev_dynamic_protocol.md`；结果及归因见 `ev_dynamic_results.md`。以下命令在仓库根目录运行。Python 环境使用 NumPy、SciPy、pytest、Gurobi；画图另需 matplotlib。实际版本由各阶段 `freeze.json` 记录。

## 1. 先恢复证据，再只读复核

紧凑包放在 `research_experiments/mobile_ev_mle/evidence/ev_dynamic/`。恢复到一个新的空目录；工具拒绝覆盖已有文件。归档携带逐原文件哈希和冻结源码快照，不执行快照内的代码。

```powershell
python -m research_experiments.mobile_ev_mle.ev_dynamic_delivery restore --archive research_experiments/mobile_ev_mle/evidence/ev_dynamic/development_raw.tar.xz --output outputs/dynamic_development_restored
python -m research_experiments.mobile_ev_mle.ev_dynamic_delivery restore --archive research_experiments/mobile_ev_mle/evidence/ev_dynamic/calibration_raw.tar.xz --output outputs/dynamic_calibration_restored
python -m research_experiments.mobile_ev_mle.ev_dynamic_delivery restore --archive research_experiments/mobile_ev_mle/evidence/ev_dynamic/evaluation_raw.tar.xz --output outputs/dynamic_evaluation_restored
```

正式校准和评价包的阶段标签分别为 `calibration`、`evaluation`。gzip 原字节重建依赖 Python/zlib 版本；工具会核查实际哈希，失败时保留经过验证的解压原文并明确报错，不把内容相同写成原字节相同。

核查当前84份科学源码是否与某个恢复阶段一致，可复用已有安全审计 API：

```python
from research_experiments.mobile_ev_mle.ev_fixed_train_source_restore import audit_source_bytes
report = audit_source_bytes(
    "outputs/dynamic_development_restored/dev_v2",
    expected_count=84,
)
assert report["success"], report
```

默认只读核查，不改源码；会使用并清理仓库 `outputs` 内的小型数据格式适配文件。若仅存在 Git 换行差异，可在检查报告后显式增加 `restore_line_endings=True`。真实内容变化或缺文件时，安全检查拒绝整批恢复。应先检出本轮对应提交，不能用快照覆盖其他研究阶段的代码。

独立 AC 回放不会重新拟合或选择动作。它从保存状态重新计算 AC 响应、误差、能量、动作及价格见证：

```powershell
python -m research_experiments.mobile_ev_mle.ev_dynamic_review --directory outputs/dynamic_development_restored/dev_v2 --output outputs/dynamic_development_replay.json.gz
```

开发 v1 记录中部分规划缓存缺少动作关联，其回放保留 unavailable；v2 补全了记录后才冻结正式策略。两者原始记录分别保留，不用后版通过覆盖前版缺口。

## 2. 从原始记录重算统计

```powershell
python -m research_experiments.mobile_ev_mle.ev_dynamic_collect --mode summarize --directory outputs/dynamic_development_restored/dev_v2 --output outputs/dynamic_summary.json
python -m research_experiments.mobile_ev_mle.ev_dynamic_collect --mode calibrate --directory outputs/dynamic_calibration_restored/calibration --output outputs/dynamic_thresholds.json
python -m research_experiments.mobile_ev_mle.ev_dynamic_collect --mode evaluate --directory outputs/dynamic_evaluation_restored/evaluation --thresholds outputs/dynamic_thresholds.json --output outputs/dynamic_evaluation.json
python -m research_experiments.mobile_ev_mle.ev_dynamic_analyze --report outputs/dynamic_evaluation.json --raw-directory outputs/dynamic_evaluation_restored/evaluation --output outputs/dynamic_reanalysis
python research_experiments/mobile_ev_mle/evidence/ev_dynamic/verification/independent_statistics.py --output outputs/dynamic_statistics_recheck.json
```

收集器只依赖 Python 标准库，检查源码/配置身份、校准与评价种子不交叉、线路参数固定、候选损失和失败规则。它不调用选择器、AC 拟合器或仿真器。分析器从已保存记录生成图和描述统计。最后一条独立统计命令直接读取仓库交付的阈值和评价报告，默认不导入收集器；本次6262项核对全部一致。

## 3. 重跑策略

把 `OPENBLAS_NUM_THREADS`、`OMP_NUM_THREADS` 和 `MKL_NUM_THREADS` 设为1。输出必须是新的目录；已有 trial 不覆盖。默认主条件是统一精度、96训练点、4验证点、6未来点，包含均值预测消融。

```powershell
python -m research_experiments.mobile_ev_mle.ev_dynamic_study --output outputs/dynamic_new_dev --seeds 991101 991102 991103 991104
```

主阶段种子：开发991101–991104，校准991301–991336，独立评价991201–991212。真实结构依次循环 star、clade_01、clade_02、clade_12。分批运行时用 `--family-offset` 保持这个循环；例如从校准第19个种子991319或评价第7个种子991207开始，offset均为2。重跑不得根据独立评价结果调整策略后仍称为原冻结评价。

求解器按墙钟限时，重新运行的有限时间 MILP 取得计划及耗时可能变化。保存证据的数学/物理回放和统计重算，核验的是已经执行的动作与结果；重跑选择器是另一项检验。总体实验耗时包含训练、多个策略、均值消融、价格诊断、未来评价和回放，不等于部署单一方法所需时间。
