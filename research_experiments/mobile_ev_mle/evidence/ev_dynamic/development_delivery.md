# 开发证据打包、字节恢复与独立回放

2026-10-15。本次只使用已保存的开发记录；未运行策略、重新拟合或校准。全部命令在 D:\0-github_workspace\Topo\outputs\ev_dynamic_stage_20261015 执行，Python 为 D:\apps\miniconda3\envs\Topo\python.exe。OMP_NUM_THREADS、OPENBLAS_NUM_THREADS、MKL_NUM_THREADS、NUMEXPR_NUM_THREADS 均设为 1，PYTHONDONTWRITEBYTECODE=1；打包、恢复和两次回放依次执行。

## 打包与恢复结果

三阶段为 dynamic_dev_v1、dynamic_dev_v2、dynamic_alternating_dev_v2，原始目录均保留在 research_experiments/mobile_ev_mle/outputs/。包内标签分别为 dev_v1、dev_v2、alternating_dev_v2。

仅收 freeze.json、source_snapshot.json.gz、trial*.json.gz、checkpoint.json，共 21 个原文件。原文件合计 19,134,559 bytes；gzip 解压后的逻辑内容 110,594,311 bytes，按内容 SHA 去重后 20 个对象、108,787,172 bytes；tar.xz 包为 5,075,000 bytes。gzip 内容先解压再进入外层 xz，manifest 保存原压缩 SHA、解压 SHA、重构规则和运行库版本。

恢复目录为 outputs/dynamic_development_restored_v2。21/21 文件恢复后与现存原文件和 manifest 的 SHA 三方全等；本机 gzip 重构全部通过，无 raw fallback。逐文件比较见 development_restore_report.json。原始 raw、恢复目录仍留 D；紧凑交付文件不替代原始记录。

运行库：{"implementation": "CPython", "python": "3.11.15", "zlib_compile": "1.3.2", "zlib_runtime": "1.3.2"}。

## 独立回放结果

| 已恢复阶段 | 历史完整通过 | 固定参数拟合状态核验 | 规划 profile 核验 | 价格模型对核验 | unavailable |
| --- | ---: | ---: | ---: | ---: | ---: |
| 统一精度 v2 | 4/4 | 480/480 | 792 | 96 | 0 |
| 交替精度 v2 | 4/4 | 480/480 | 780 | 96 | 0 |

两次回放使用独立前推回代物理实现，核对保存的参数、源、计量和规划见证；没有新增拟合、选择或统计校准。成功表示保存记录可按上述检查重算，未提供全局最优性或现场效果结论。

v1 不重复回放。原 verification/dynamic_dev_v1_replay.json.gz 保留：4 个历史中完整通过为 0/4，已保存拟合状态 448/448 可核验，但 156 个规划 profile 因旧缓存缺少动作关联而 unavailable，已有 216 个规划 profile 可核验。保留该来源限制，未用新版驱动补写旧记录。

## 命令

以下为可复放的相同操作（目标包、sidecar 和恢复目录须不存在或符合空目录要求）：

    python -m research_experiments.mobile_ev_mle.ev_dynamic_delivery pack --phase dev_v1=research_experiments/mobile_ev_mle/outputs/dynamic_dev_v1 --phase dev_v2=research_experiments/mobile_ev_mle/outputs/dynamic_dev_v2 --phase alternating_dev_v2=research_experiments/mobile_ev_mle/outputs/dynamic_alternating_dev_v2 --output research_experiments/mobile_ev_mle/evidence/ev_dynamic/development_raw.tar.xz

本次 pack 初始暂存于 stage 根 evidence/ev_dynamic，完成后只将新包及 sidecar 移入上述研究证据目录；原数据没有移动。

    python -m research_experiments.mobile_ev_mle.ev_dynamic_delivery restore --archive research_experiments/mobile_ev_mle/evidence/ev_dynamic/development_raw.tar.xz --output outputs/dynamic_development_restored_v2

本次通过同一 restore_archive API 保存返回报告，并额外逐文件检查 original_sha == restored_sha == manifest_original_sha，将结果写入 development_restore_report.json；该报告含 21 项比较。

    python -m research_experiments.mobile_ev_mle.ev_dynamic_review --directory outputs/dynamic_development_restored_v2/dev_v2 --output research_experiments/mobile_ev_mle/evidence/ev_dynamic/development_uniform_restored_replay.json.gz

    python -m research_experiments.mobile_ev_mle.ev_dynamic_review --directory outputs/dynamic_development_restored_v2/alternating_dev_v2 --output research_experiments/mobile_ev_mle/evidence/ev_dynamic/development_alternating_restored_replay.json.gz

delivery helper SHA256：e267a52f7a1d31101d6c76f77d67283a62faa8009d8d037498157896d2f07c17。
本次 reviewer SHA256：c313c71862c5b3e326b8739f697fc4fd653b3d21a68876a9a88271b2a1ac7649。

## 交付文件

| 文件 | bytes | SHA256 | 含义 |
| --- | ---: | --- | --- |
| development_raw.tar.xz | 5075000 | 3b76ce0b1f5d83132d9870870a9171049c1567a9af4e32da92c0ea5dedc33991 | 开发原始证据包 |
| development_raw.tar.xz.manifest.json | 12313 | dfd2cc5aae8460b58aeeac92c2281b0019d2bf992ec539f7f2d96b7a3a5e80d5 | 包清单与原始／解压 SHA |
| development_restore_report.json | 8342 | 468d69e2674da2bc2c29ef93f525f9b2aac2da0550fbec502b0d934cc348a05a | 21 个原文件恢复比较 |
| development_uniform_restored_replay.json.gz | 597410 | e41ec4c7246eb8e9757a44730865a4dcbfb874097033664abdd1715fc98afe10 | 恢复后统一精度 v2 独立回放 |
| development_alternating_restored_replay.json.gz | 603101 | 3304e7da0abd0fb13b00b61228ccc1586d46626771ad02a1ffcece3bfec52d49 | 恢复后交替精度 v2 独立回放 |
| verification/dynamic_dev_v1_replay.json.gz | 551146 | f259acbb4960f9ae5e12d7219c11e83da38261de1669a044f8e7a260d2251eff | 原 v1 已有回放，保留 unavailable |

未修改任何冻结科学源码或 source_hashes 文件。本子任务未执行 Git 提交；由主任务将紧凑包、清单、报告和本说明纳入阶段交付。
