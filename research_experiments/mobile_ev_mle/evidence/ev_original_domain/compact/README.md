# 新原域实验的可恢复证据

只压缩存储，原 raw 目录完整保留。各包的独立恢复都逐文件验证了原始字节；科学数据和冻结源码保持原样。

| 版本 | 用途 | 文件数 | 原文件合计(bytes) | tar.xz(bytes) | 剩余比例 |
|---|---|---:|---:|---:|---:|
| pilot_v1 | 开发首条 | 4 | 3,669,009 | 1,375,496 | 37.49% |
| pilot_v2 | 开发4H | 8 | 20,886,457 | 7,592,292 | 36.35% |
| evaluation_v2 | 冻结策略后12H独立评价 | 15 | 67,213,677 | 22,758,228 | 33.86% |
| location_ablation_v1 | 同12H位置对照，24条件 | 17 | 2,884,028 | 1,492,620 | 51.75% |
| pilot_numeric_v2 | 数值域开发4H | 7 | 23,659,138 | 8,079,564 | 34.15% |
| stress_asset_v2 | 资产域压力2H | 5 | 8,113,200 | 3,116,052 | 38.41% |
| stress_numeric_v2 | 数值域压力2H | 5 | 6,172,303 | 2,067,376 | 33.49% |
| 合计 | 存储数量，不相加统计样本 | 61 | 132,597,812 | 46,481,628 | 35.05% |

`evaluation_v2` 的12个历史是冻结策略后的独立评价。位置对照复用这12个历史；三个域的开发/压力结果单独保存，不并入该评价分母。域对照之间还可能使用相同种子，文件数量不代表独立样本数量。

v1 包含freeze、源码快照、首条trial和final review。v2包含freeze、源码快照、4条trial、full review和pilot gate。evaluation包含freeze、源码快照、12条trial和full review。位置包包含freeze、源码快照、12条history、summary、parent review validation和full review。三个域包分别包含freeze、源码快照、全部trial和成功的review_full；文件数为7/5/5。CSV、checkpoint与外层分析表保留在原位置，未重复装入这些包。

每份manifest同时存在于归档内与本目录，记录原文件SHA-256、payload SHA-256、原gzip header/trailer、mtime、OS及压缩级别。旧pilot的验证见 `archive_validation.json`；其余包的验证见 `<版本>_archive_validation.json`，保存实际命令、环境、计时、逐文件原字节比对和存储范围。

归档将 `.gz` 还原为原始JSON字节，不解析重排，再一起写入solid tar.xz。默认xz preset 6使用8MiB字典。所有61/61文件已在全新D目录恢复并通过全字节比对，包括源码快照gzip原字节。评价、位置和三域归档以BelowNormal、单线程运行，可能与其他研究操作重叠；记录的计时用于操作追踪。

## 恢复

在Topo仓库根目录执行。目标必须是不存在的新目录，例如：

```powershell
& 'D:/apps/miniconda3/envs/Topo/python.exe' -m research_experiments.mobile_ev_mle.ev_original_evidence_archive restore 'research_experiments/mobile_ev_mle/evidence/ev_original_domain/compact/evaluation_v2_payloads.tar.xz' 'D:/0-github_workspace/Topo/outputs/restored_original_evaluation_v2' --require-exact --report 'D:/0-github_workspace/Topo/outputs/restored_original_evaluation_v2_report.json'
```

恢复器先验证成员路径，逐文件写入，不调用extractall，也不覆盖已有目录。换用其他Python/zlib版本时，重压后的gzip字节可能不同；payload仍须通过SHA检查，报告保存恢复文件的实际hash与差异。此时 `--require-exact` 返回非零。去掉该选项可以检查payload一致的恢复结果。

归档工具21项独立测试通过，工具位于科学freeze之外。首条v2的机制检查仍在 `evidence/ev_original_archive_check/verification_v1.json`，未被完整包覆盖。恢复冻结源码的换行检查另见 `ev_original_source_restore_README.md`（68源与位置对照70源），本归档恢复本身不覆盖仓库源码。
