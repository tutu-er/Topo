# evaluation_v2 可恢复归档

范围严格为15文件：freeze、68源快照、12条trial和fullreview。未加入checkpoint、partial review、threadpool审计或分析表。预声明清单为 `evaluation_v2_archive_plan.json`；原raw保留。

- 原文件合计：67,213,677 bytes。
- 解压payload合计：1,196,333,664 bytes。
- solid tar.xz：22,758,228 bytes，保留原体积33.86%。
- 全新D目录恢复后，15/15文件与raw全字节一致，包括源码快照原gzip字节；复制到C后归档hash再次一致。
- 归档SHA-256：`23ecd20985275432bf949d6b3c37365aa58af3276ced07a5772670781371c4f1`。

实际运行使用BelowNormal优先级、单线程xz preset6（8MiB字典）；pack129.03s、restore49.14s。与根任务的域压力操作/拟合可能重叠，属于实际运行耗时，不作为隔离压缩基准。

`evaluation_v2_manifest.json` 保存原文件/payload的SHA及gzip头和codec信息。`evaluation_v2_archive_validation.json` 包含实际命令、环境、15文件原字节回验及C复制检查。恢复跨Python/zlib版本时，payload必须一致；若gzip原字节不一致，工具明确报告恢复后hash，不冒称原hash。

在Topo根目录恢复至一个不存在的新目录：

```powershell
& 'D:/apps/miniconda3/envs/Topo/python.exe' -m research_experiments.mobile_ev_mle.ev_original_evidence_archive restore 'research_experiments/mobile_ev_mle/evidence/ev_original_domain/compact/evaluation_v2_payloads.tar.xz' 'D:/0-github_workspace/Topo/outputs/restored_original_evaluation_v2' --require-exact --report 'D:/0-github_workspace/Topo/outputs/restored_original_evaluation_v2_report.json'
```

归档工具独立于科学freeze。本步骤没有运行拟合、选择器或改变原证据。
