# 开发预检证据

本目录是开发证据，64校准/128独立评价尚未完成。

- `preflight_freeze.json` 与 `preflight_source_snapshot.json.gz`：开发v2的配置、seed及精确源码字节。两份完整开发记录以去重浮点数组ZIP保存，`archive_audit.json`核对JSON类型、浮点IEEE位及原文件hash；压缩格式变化不改变数值。
- `linear_calibration_2048.json`：已知参数线性有界生成式，2048独立历史、逐历史结果和seed。使用 `ev_bayes_response_linear_calibration.py` 复现；`linear_calibration_reproduction.json`验证CLI核心与最初测试核相同。
- `response_development_*.json`、`history_support_development_0_1.json`：AC水平/增量、线路/负荷/源分解，以及固定线路后的历史同接口源profile复算。
- `gradient_development_*.json`：开发v2原提案方向的完整协方差梯度核查。`gradient_frozen_development_*.json`记录最终统一父MIP-start后的开发方向；两组分开，不选择性合并。
- `independent_*audit_000.json`：独立AC/协方差/后验及预选有界积分重放。
- `regression_167.xml` 与 `finish_11.xml`：当时已完成的核心/对照回归与后处理工具测试。
- `frozen_batch.json` 与 `frozen_source_snapshot.json.gz`：最终运行批次，采用统一父MIP-start。它与前述开发v2是不同policy，不能混为同一批实验。

完整开发原记录可还原到全新目录：

```
python -m research_experiments.mobile_ev_mle.ev_bayes_response_restore --input research_experiments/mobile_ev_mle/evidence/ev_bayes_response_preflight_v1 --output-root outputs/response_preflight_restored_v1
```

该命令仅校验、还原源码和数据，不执行还原的源码。`restore_audit.json`记录原始及新压缩hash、解析JSON/浮点位一致性和路径检查。目标目录须不存在。

直接检查单份归档而不恢复：

```
python -m research_experiments.mobile_ev_mle.ev_bayes_small_evidence verify research_experiments/mobile_ev_mle/evidence/ev_bayes_response_preflight_v1/development_000.evidence.zip
```

请结合上层 `ev_bayes_response_preflight_report.md` 阅读。线性生成式通过，不等于当前非线性AC概率已校准；开发原始失败与负例均保留。
