# 校准36历史：紧凑交付与精确恢复

本目录只交付校准与保存状态核查，正式12历史评价仍在运行。独立单位为历史，不是任务数或QP数。

`checks.tar.xz` 包含原freeze、72源字节快照、完整AC/操作和QP复核报告及覆盖/hash检查。
`analysis.tar.xz` 包含完整7张/份逐历史CSV和JSON/Markdown分析，含逐候选R/X、取得loglik、失败与逐任务分数。为缩小工作树，26.6MB逐任务CSV保存在该无损包；常用表另放原analysis目录。
全部原raw仍在D阶段目录 `research_experiments/mobile_ev_mle/evidence/ev_compact/calibration_ensemble_v2`，未删除、改写或加入Git；其36份压缩字节hash见risk标准记录和review。

从仓库根解到不存在的新目录：

```text
python -m research_experiments.mobile_ev_mle.ev_original_evidence_archive restore research_experiments/mobile_ev_mle/evidence/ev_compact/delivery_calibration_v2/checks.tar.xz <新的检查目录> --require-exact
python -m research_experiments.mobile_ev_mle.ev_original_evidence_archive restore research_experiments/mobile_ev_mle/evidence/ev_compact/delivery_calibration_v2/analysis.tar.xz <新的分析目录> --require-exact
```

两包均已独立解到新D目录并核对全部原字节。*_verification.json记录原文件/载荷/压缩包hash与运行时。gzip在不同zlib运行时可能只有载荷一致；--require-exact会明确拒绝把这种情况称作原压缩字节一致。

`risk_calibration_before_evaluation.json.gz`保存36条标准记录及校准阈值（当时evaluation=0）。当前冻结driver内的旧statistical_risk_calibrated=false字段未改写；统计决策由外部collector完成，最后评价会核对同一calibration_id。
