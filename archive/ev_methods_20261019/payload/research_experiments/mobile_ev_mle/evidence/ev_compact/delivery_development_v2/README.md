# v2 开发证据交付

仅包含 ensemble 与 center 两组已完成开发数据，各4个独立历史、每历史11方法。没有导出或读取校准/评价 trial。

|目录|raw trial总大小（MB，十进制）|slim trial总大小（MB）|slim/raw|
|---|---:|---:|---:|
|development_ensemble_v2|234.874|23.125|9.85%|
|development_center_v2|38.192|4.998|13.09%|
|合计|273.066|28.124|10.30%|

上表只计算 trial 压缩文件；冻结源码快照、清单与分析摘要另计。

核对结果：8/8个 raw 压缩字节SHA256在导出和回读后保持一致；两个 freeze.json 与 source_snapshot.json.gz 原样复制。
raw_slim_verification.json 记录298395项字段/算术核对，88条方法结果全部通过。核对覆盖实际 noisy/noiseless fits与计量、H-only全部任务分数/失败、每eval逐对区间和状态、每迭代求解器界/规模/耗时、每轮最佳QP全部物理参数/残差。
verified_method_rows.csv 保留逐历史的实际M、无噪M、最终来源、继承检查、资源和代理界；各子目录 summary_from_slim.md/json 是只用slim重新计算的摘要，与raw关键字段逐项一致。

建议提交：八个slim trial、各目录freeze/source_snapshot/delivery_manifest/DELIVERY_SCOPE、核对脚本与JSON、小型逐历史CSV和摘要。完整raw、重复LP数组与大型重复分析表保留本地，路径和压缩字节SHA256已进入清单。
另应随论文结果保留对应独立科学复核的结论及文件hash；本目录的字段一致性核对没有重新运行AC，也不替代独立AC/优化复核。

slim是明确的删减schema，不是原raw的逐字节恢复包。它保留实际计量和完整实际拟合，可回算主要报告；完整重放被省略的预测拟合细节、非最佳QP状态、见证库及LP证书，需要原raw及冻结源码。
所有原raw保持原位，未覆盖、删除或改写；未修改冻结科学源、protocol、C工作树，也未commit/push。

核对重跑命令（在隔离开发根运行）：
`python -m research_experiments.mobile_ev_mle.evidence.ev_compact.delivery_development_v2.verify_delivery`
