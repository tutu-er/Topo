# 新原域实验的可恢复证据

只压缩存储，不改变科学数据或冻结源码。原 raw 目录保留。

| 版本 | 文件数 | 原文件合计(bytes) | tar.xz(bytes) | 剩余比例 |
|---|---:|---:|---:|---:|
| pilot_v1 | 4 | 3,669,009 | 1,375,496 | 37.49% |
| pilot_v2 | 8 | 20,886,457 | 7,592,292 | 36.35% |
| 合计 | 12 | 24,555,466 | 8,967,788 | 36.52% |

v1 包含 freeze、源码快照、首条 trial 和 final review。v2 包含 freeze、源码快照、4条 trial、full review 和 pilot gate。分析表另存，不重复装入归档。

归档将 `.gz` 先还原为原始 JSON **字节**，不解析重排，再与普通文件一起放入一个 solid tar.xz。默认 xz preset 6 使用8MiB字典。每份 manifest 同时存在于归档内与本目录，保存原文件SHA-256、payload SHA-256、原gzip header/trailer、mtime、OS和压缩级别。`archive_validation.json` 保存实际打包/恢复命令、环境、计时和逐文件原字节比对。

已在新的 D 目录恢复，两版12/12文件与 raw **全字节相同**，包括源码快照的 gzip 原字节。v1 打包6.64s、恢复2.51s；v2 打包40.38s、恢复14.68s。归档工具21项独立测试通过，未加入科学 freeze。

## 恢复

在 Topo 仓库根目录执行，目标必须是不存在的新目录。例如：

```powershell
& 'D:/apps/miniconda3/envs/Topo/python.exe' -m research_experiments.mobile_ev_mle.ev_original_evidence_archive restore 'research_experiments/mobile_ev_mle/evidence/ev_original_domain/compact/pilot_v2_payloads.tar.xz' 'D:/0-github_workspace/Topo/outputs/restored_original_pilot_v2' --require-exact --report 'D:/0-github_workspace/Topo/outputs/restored_original_pilot_v2_report.json'
```

恢复器先验证成员路径，再逐文件写入；不调用 `extractall`，不覆盖已有目录。若另一Python/zlib版本重压后gzip字节不同，payload必须仍通过SHA检查，并保存恢复后真实文件hash与差异说明。此时 `--require-exact` 返回非零；报告不会将原gzip hash冒称为新文件hash。去掉该选项可检查payload一致的恢复结果。

首条v2的独立机制压缩检查另存在 `evidence/ev_original_archive_check/verification_v1.json`，未被本交付覆盖。
