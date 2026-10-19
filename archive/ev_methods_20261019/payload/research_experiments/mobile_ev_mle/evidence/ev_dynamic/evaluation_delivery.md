# 正式评价记录的打包、恢复与评价复算

2026-10-15。工作目录 D:\0-github_workspace\Topo\outputs\ev_dynamic_stage_20261015，Python 为 D:\apps\miniconda3\envs\Topo\python.exe。操作依次执行，四个数值线程环境变量 OMP_NUM_THREADS、OPENBLAS_NUM_THREADS、MKL_NUM_THREADS、NUMEXPR_NUM_THREADS 均为1，PYTHONDONTWRITEBYTECODE=1。

## 验证结果

- 来源 research_experiments/mobile_ev_mle/outputs/dynamic_evaluation_v2，12个独立评价历史，14种方法。
- 收录15个白名单文件（12个trial、freeze、source_snapshot、checkpoint），原文件合计21,696,128 bytes，解压逻辑内容140073885 bytes。evaluation_raw.tar.xz为5,464,164 bytes，SHA256为5304ee4d7709ff972a07483fce683e3c5d34285a7911f777c605fef6aace833e。
- 恢复目录为stage根 outputs/dynamic_evaluation_restored_v2。15/15原文件的原SHA、恢复SHA与manifest原SHA完全一致；gzip重构全部成功。
- 仅从恢复记录使用原thresholds.json重跑collect evaluate，另存evaluation_restored.json。与原evaluation.json逐内容、逐字节完全一致，没有绝对路径或实质差异；二者SHA均为cfa57273b0aa0c2daf852d9bd154d8533680ad19f797789a6770f783524c4e5d。
- 原阈值SHA及两份评价报告中的threshold_file_sha256均为bf01b55c2f2b05681ecbfb1f43a6af186aebbef9d423da5c12b0b522f42c3af2。thresholds_id与policy_id保持不变。
- 现有evaluation_replay.json.gz为12/12独立AC回放完整通过，本次只读确认并引用；字节恢复不改变其对应内容，未重复该批物理回放。

原始raw、source、thresholds.json、checkpoint与evaluation.json均未修改。没有新增策略、拟合、校准或独立样本。

## 命令

    python -m research_experiments.mobile_ev_mle.ev_dynamic_delivery pack --phase evaluation=research_experiments/mobile_ev_mle/outputs/dynamic_evaluation_v2 --output research_experiments/mobile_ev_mle/evidence/ev_dynamic/evaluation_raw.tar.xz

    python -m research_experiments.mobile_ev_mle.ev_dynamic_delivery restore --archive research_experiments/mobile_ev_mle/evidence/ev_dynamic/evaluation_raw.tar.xz --output outputs/dynamic_evaluation_restored_v2

本次通过同一restore_archive API保存返回报告，并逐文件检查original_sha == restored_sha == manifest_original_sha；15项明细见evaluation_restore_report.json。

    python -m research_experiments.mobile_ev_mle.ev_dynamic_collect --mode evaluate --directory outputs/dynamic_evaluation_restored_v2/evaluation --output research_experiments/mobile_ev_mle/evidence/ev_dynamic/evaluation_restored.json --thresholds research_experiments/mobile_ev_mle/evidence/ev_dynamic/thresholds.json

复算后同时比较原字节、递归JSON内容以及实际阈值文件SHA，完整结果保存在evaluation_restore_comparison.json。

运行库：CPython3.11.15，zlib编译/运行版本均1.3.2。
delivery工具SHA：e267a52f7a1d31101d6c76f77d67283a62faa8009d8d037498157896d2f07c17。
collector工具SHA：98eee6570eed0f0282294097c4b8974fe030d181583c2aee50c10a01f9d86b3f。

## 文件校验

| 文件 | bytes | SHA256 |
| --- | ---: | --- |
| evaluation_raw.tar.xz | 5464164 | 5304ee4d7709ff972a07483fce683e3c5d34285a7911f777c605fef6aace833e |
| evaluation_raw.tar.xz.manifest.json | 9476 | ef43b037737020b4cc74575c33b1c90ff62fda31929d911c9702b46c3f9b756f |
| evaluation_restore_report.json | 6148 | d860e79fcf63ef67d5c3a481e0ecc282aaa42073264de4325c70812f4b601ab2 |
| evaluation.json | 322699 | cfa57273b0aa0c2daf852d9bd154d8533680ad19f797789a6770f783524c4e5d |
| evaluation_restored.json | 322699 | cfa57273b0aa0c2daf852d9bd154d8533680ad19f797789a6770f783524c4e5d |
| evaluation_restore_comparison.json | 823 | 2d0a0f36d77588c2a9d8362af4d8f5ba1a606fcb8a8662c9d25e99c403499e7e |
| evaluation_replay.json.gz | 1802857 | 1d32ef335b9ad46a4cd2ca637e4635a859abe2686f7071ff0f0770f03a4eb9cd |

原始、恢复及逐字节相同的复算副本均留D。Git交付可只使用紧凑包、sidecar、原评价及比较报告，避免重复原始文件。本子任务未执行Git操作。
