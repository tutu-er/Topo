# 96点比较的最终证据

- [64项逐案例结果](summary.csv)，[完整报告](report.md)，[F1图](f1_heatmap.png)。
- [按方法汇总](aggregate_by_method.csv)，[按源项汇总](aggregate_by_variant_method.csv)。
- [独立复核摘要](audit_summary.json)，[独立重算指标](metrics_recomputed.csv)。
- [固定RNJ边界块](frozen_blocks.csv)，[未接受的可行候选](uncertified_incumbents.csv)。
- [执行协议及源码指纹](protocol.json)，[运行环境](environment.json)。
- [协议签名及源码快照核对](provenance_audit.json)。
- [本地完整原始记录指纹](raw_artifacts_sha256.csv)，[源码快照指纹](source_snapshot_manifest.json)。

完整原始输入、模型、矩阵、逐步路径、求解器日志和43个源码快照保留在仓库本地
`outputs/source_ac96_benchmark_20261018/`，生成数据保留在
`outputs/source_ac96_20261018/`。这些生成目录未纳入Git。
原始记录指纹是本次运行结束后所列文件的SHA256；未列入后续新增文件。

每方法计划分母16，四法总计划分母64。incomplete的最后认证模型仍参与均值，
失败和缺失指标不填零。未认证候选仅作诊断，未替换主表模型。
F1评价非平凡终端簇；R/X误差评价平方电压灵敏度矩阵，而非逐条隐藏线路的阻抗。
