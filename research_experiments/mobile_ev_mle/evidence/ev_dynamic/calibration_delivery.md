# 正式校准记录的打包、恢复与阈值复算

2026-10-15。工作目录 D:\0-github_workspace\Topo\outputs\ev_dynamic_stage_20261015，Python 为 D:\apps\miniconda3\envs\Topo\python.exe。所有步骤依次执行；OMP_NUM_THREADS、OPENBLAS_NUM_THREADS、MKL_NUM_THREADS、NUMEXPR_NUM_THREADS 均为 1，PYTHONDONTWRITEBYTECODE=1。

## 结果

- 来源为 research_experiments/mobile_ev_mle/outputs/dynamic_calibration_v2，36 个独立校准历史、每真值族9个、14种方法。
- 白名单文件39个（36个trial、freeze、source_snapshot、checkpoint），原文件共65,754,447 bytes；解压逻辑内容434,900,567 bytes。calibration_raw.tar.xz 为15,875,924 bytes，SHA256为0d542a713a98bf0d5cd5ceb3e34949e8b6841cb545c6bd46abfb32624ca2ec50。
- 恢复到 stage 根 outputs/dynamic_calibration_restored_v2。39/39文件的原文件SHA、恢复SHA、manifest原SHA三方全等，无gzip重构失败。
- 仅从恢复后的记录重跑collect calibrate，另存 calibration_restored_thresholds.json。它与原thresholds.json逐内容和逐字节完全一致，差异列表为空；二者SHA均为bf01b55c2f2b05681ecbfb1f43a6af186aebbef9d423da5c12b0b522f42c3af2。
- thresholds_id保持10c25814334344cdb329d0382c6857d3b2a9b5516ba973e1a35de9c576671eba，policy_id保持32b3eefb13e61804144139a41a8d8e29f3fcf2c3e361b9cee33c4a67abe7e9ae。
- 原thresholds.json、raw、checkpoint及冻结科学源码均未修改；evaluation_start.json和正在执行的评价流程未触碰。复算没有新增拟合、策略或统计样本。

39个文件字节相同，使已有 calibration_replay.json.gz 的36/36完整独立回放结论仍对应同一内容；本次未重复该批物理回放。已有回放文件仅只读核对。

## 命令与核验口径

    python -m research_experiments.mobile_ev_mle.ev_dynamic_delivery pack --phase calibration=research_experiments/mobile_ev_mle/outputs/dynamic_calibration_v2 --output research_experiments/mobile_ev_mle/evidence/ev_dynamic/calibration_raw.tar.xz

    python -m research_experiments.mobile_ev_mle.ev_dynamic_delivery restore --archive research_experiments/mobile_ev_mle/evidence/ev_dynamic/calibration_raw.tar.xz --output outputs/dynamic_calibration_restored_v2

本次恢复调用同一restore_archive API以保存返回报告，并对manifest每项执行：

    SHA256(original_file) == SHA256(restored_file) == manifest.original_sha256

39项比较保存在calibration_restore_report.json。

    python -m research_experiments.mobile_ev_mle.ev_dynamic_collect --mode calibrate --directory outputs/dynamic_calibration_restored_v2/calibration --output research_experiments/mobile_ev_mle/evidence/ev_dynamic/calibration_restored_thresholds.json --alpha 0.1

原阈值先锁定SHA，再与重算文件作原字节比较和递归JSON内容比较；具体结果保存在calibration_thresholds_restore_comparison.json。不存在绝对路径元数据差异或实质差异。

运行库：CPython3.11.15，zlib编译/运行版本均1.3.2。
delivery工具SHA：e267a52f7a1d31101d6c76f77d67283a62faa8009d8d037498157896d2f07c17。
collector工具SHA：98eee6570eed0f0282294097c4b8974fe030d181583c2aee50c10a01f9d86b3f。

## 文件校验

| 文件 | bytes | SHA256 |
| --- | ---: | --- |
| calibration_raw.tar.xz | 15875924 | 0d542a713a98bf0d5cd5ceb3e34949e8b6841cb545c6bd46abfb32624ca2ec50 |
| calibration_raw.tar.xz.manifest.json | 23834 | 90cc99f29c7f8565c8c7dbbbd47486ed96a1ddaa0a5562ec2bf08596d5705c99 |
| calibration_restore_report.json | 15178 | 03c7ba1c3b7a8bb02a11bcaf63672c4a3126d7403faad53d990607d351bcbffc |
| thresholds.json | 702049 | bf01b55c2f2b05681ecbfb1f43a6af186aebbef9d423da5c12b0b522f42c3af2 |
| calibration_restored_thresholds.json | 702049 | bf01b55c2f2b05681ecbfb1f43a6af186aebbef9d423da5c12b0b522f42c3af2 |
| calibration_thresholds_restore_comparison.json | 773 | bbafd2651826263ab33bcc2e5b90642a5bfeae4db872f70c2f514e0730e09b83 |
| calibration_replay.json.gz | 5426567 | 292579b9756bf707518f732d8116b8b516363a7ae95e7839ae1b3c6f9f242f38 |

全部原始与恢复文件仍保留D。紧凑包、sidecar、原阈值和比较报告足以用于Git交付；逐字节相同的重算阈值副本可只保留在D。本子任务未执行Git操作。
