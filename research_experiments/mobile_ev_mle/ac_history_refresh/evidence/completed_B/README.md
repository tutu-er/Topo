# 固定计划修正：最终归档

先读上两级 `results.md` 和 `reproduce.md`。本目录保存原JSON/日志的无损gzip或原有gzip，`archive_manifest.json`同时记录原/压缩文件SHA256及大小。原raw及全部运行日志留D盘，128个结果文件来自相同64份历史的A/B复算；独立校准32、评价32。

- `A_B_fixed_plan_comparison_v1`：同历史/计划的完整比较及操作失败分母。
- `B_probability_main_audit_v1`、`B_probability_delivery_summary_v1`：独立Cholesky评分、校准与数据绑定，正式0差异。
- `history_refresh_audit_main_v1`及receipt：64历史/256候选的完整AC、原目标、接受轨迹和数值驻点，0新优化或积分。
- `fixed_response_audit_v1`：真候选有限EV电压增量与总量预测误差，0AC/拟合/新数据。
- `fixed_response_independent_check_v1/v2`：两份真实历史独立手算。v1的两项额外精确字节假设失败保留；v2明确数值等同恒充的容差，未修改主审计。另有零信号bank检查：电压均值增量逐字节为零，未发现MC虚假收益。
- `predictive_risk_audit_v1`：已存Gaussian预测模型的风险积分；数值积分样本不作为新的物理数据。
- `operational_and_plan_contrasts_v1`：完整分母及恒充/平方/判别的同历史对比。
- `raw_manifest_v1`：全部128原结果及B元数据的字节指纹；大raw未在Git重复存储。
- 其余文件保留原校准/绑定、完成状态、原A运行登记收养及只读后处理的环境修复、费用和日志。

当前阶段完成。物理R/X后验积分与新MILP梯度尚未实现；A/B评价已阅读，后续新方法需另设独立评价。原LaTeX/PDF及pyproject未改。
