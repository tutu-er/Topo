# 本阶段复核与续接

工作分支 codex/ev-train-validation-design。主要提交：87ca36f数学/协议；07d35a4实现；ebb70bc开发验收及正式冻结。实际生效版本以各目录freeze.json的90份源码哈希及source_snapshot.json.gz为准。正式policy_id为88038a9c92c7f54339415cc60efd00ae650754cf9e565ad9aad3c39382684d7f。

全部研究文件在research_experiments/mobile_ev_mle。先读ev_rx_results.md，数学见ev_rx_math.md、ev_rx_interpretation.md及ev_rx_precision_proof.md。本阶段不改原报告，不覆盖旧实验。

## 证据恢复

evidence/ev_rx内四个*_raw.tar.xz分别保存开发box/cone、正式校准/评价。压缩保存原JSON字节，不改写原记录；SHA清单和恢复报告同行保存。以下从仓库根目录运行，恢复目标必须尚不存在：

```powershell
python -m research_experiments.mobile_ev_mle.ev_original_evidence_archive restore research_experiments/mobile_ev_mle/evidence/ev_rx/calibration_raw.tar.xz outputs/rx_restored_calibration --require-exact
python -m research_experiments.mobile_ev_mle.ev_original_evidence_archive restore research_experiments/mobile_ev_mle/evidence/ev_rx/evaluation_raw.tar.xz outputs/rx_restored_evaluation --require-exact
python -m research_experiments.mobile_ev_mle.ev_dynamic_review --directory outputs/rx_restored_calibration --output outputs/rx_calibration_replay.json
python -m research_experiments.mobile_ev_mle.ev_dynamic_review --directory outputs/rx_restored_evaluation --output outputs/rx_evaluation_replay.json
```

开发归档也用同一restore命令。源码快照只用来校验/恢复声明源码，不自动执行归档中的任意代码。Git版本与当前环境不一致时先按哈希核查。

## 独立统计

```powershell
python -m research_experiments.mobile_ev_mle.ev_rx_statistics_audit --self-test --calibration-directory outputs/rx_restored_calibration --evaluation-directory outputs/rx_restored_evaluation --output outputs/rx_stats_replayed.json
python -m research_experiments.mobile_ev_mle.ev_rx_formal_summary --calibration-directory outputs/rx_restored_calibration --evaluation-directory outputs/rx_restored_evaluation --output-directory outputs/rx_formal_replayed
```

统计审查不导入生产汇总器、AC或优化器；直接重建每候选验证+未来J、M、分族rank阈值和决策。正式汇总先保存校准阈值并回读，再访问评价；若已存在不同字节阈值则拒绝覆盖。校准阈值SHA256为2efb5eaa50440b9881898110167893c82b35cb895920ad484f8bf9aece904492，评价报告为5206a287be591e111df2be4e6a874834e6434aa8b717fe16b2fa5463f665ef69。

## 重跑研究

使用Topo Python环境（本次3.11.15、NumPy/SciPy版本在freeze记录），BLAS/OMP线程设1。运行开发时两个训练域共享种子992101–992104；正式cone校准992301–992336、评价992201–992212。每组family_index按0,1,2,3循环。`ev_rx_study --output 新目录 --seeds ... --training-domain box|cone`会先冻结配置/源码并拒绝覆盖已有trial。

本次校准分3块12历史并行，a/b/c配置及源码快照解析后相同；按原字节合并，merge_manifest.json记录各来源和SHA。校准和评价在策略冻结后并行计算，校准阈值先于读取评价统计保存；运行期间未改变选择器。开发配对统计入口为`ev_rx_analyze --box-directory ... --cone-directory ... --output ...`。

## 当前续接点

本阶段已完成4历史配对开发、36校准、12独立评价及回放。详细原文件保留在D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015/outputs。下一阶段不沿用这些评价历史调参：先给一组训练参数的动作响应风险诊断建立依据，验证预测误差界，再决定是否改MILP目标。新的策略需新校准/评价种子；当前未开展滚动、多轮风险或扩网。
