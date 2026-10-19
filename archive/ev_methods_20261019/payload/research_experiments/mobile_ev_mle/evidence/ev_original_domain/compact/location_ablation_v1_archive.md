# 位置消融可恢复归档

17个文件：freeze、70源快照、12条history、summary、父评价验证和独立 `review_full.json`。没有加入两个CSV、checkpoint或外围重复分析/图表；所有raw及分析产物原位保留。

freeze实际使用直接70源哈希映射，source_snapshot为 `name -> {sha256,bytes_base64}`。70条解码字节均与freeze一致；父68源保持不变，额外两源为位置消融驱动与协议。位置消融依赖的父评价完整证据为同目录 `evaluation_v2_payloads.tar.xz`，该联系和父归档SHA记录在验证中。

- 原文件合计：2,884,028 bytes；payload合计7,897,123 bytes。
- solid tar.xz：1,492,620 bytes，保留原体积51.75%。
- 全新D目录恢复后17/17文件与原raw全字节一致，包括70源快照原gzip字节；C复制归档hash再次一致。
- 归档SHA：`41e80e60389bf73e851c399cd140ec874c109ef7dcc945ebeb6e903afa8b3637`。

使用BelowNormal优先级、单线程xz preset6，pack5.13s、restore1.46s。与域压力操作/拟合可能重叠，耗时不作为隔离基准。实际清单、原文件/payload哈希、gzip头及codec见manifest；命令、原字节比对和C复制验证见 `location_ablation_v1_archive_validation.json`。

在Topo根目录恢复至一个不存在的新目录：

```powershell
& 'D:/apps/miniconda3/envs/Topo/python.exe' -m research_experiments.mobile_ev_mle.ev_original_evidence_archive restore 'research_experiments/mobile_ev_mle/evidence/ev_original_domain/compact/location_ablation_v1_payloads.tar.xz' 'D:/0-github_workspace/Topo/outputs/restored_original_location_ablation_v1' --require-exact --report 'D:/0-github_workspace/Topo/outputs/restored_original_location_ablation_v1_report.json'
```

归档没有新拟合或选择，也没有修改70个科学源。跨运行环境若gzip原字节不同，payload仍须通过SHA，工具会报告恢复后的真实hash及差异。
