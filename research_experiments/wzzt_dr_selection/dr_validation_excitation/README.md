# 固定验证参数的连续价格搜索

最新模型与证据：CONTINUOUS_MODEL.md、CONTINUOUS_REPORT.md，公开 ../evidence/validation_continuous_v1。

- 给定训练报价，17完整历史族用原PoolLP共同估计所有R/X；验证参数固定。
- affine_solver.py精确优化验证日连续价格/合同类型，保留原3%与硬预算。continuous_search.py是有界外层；coarse_exit_reprice.py按当前粗族退出条件复用已拟合表。
- actual_bridge.py只复用原AC与RNJ＋WZZT，不重写辨识器。
- retained_history_audit.py是额外命名的组合敏感性审计，不改生产默认与原实际结果。
- 原MODEL.md/REPORT.md记录上一阶段的全无C族排除与8份旧观测重分析；它强于当前粗族退出，不能混用。

本轮13个新连续训练报价、独立10模型路径和2实际完整路径已全部结束。未找到90%激励达标报价，付费最终错漏未胜自然新增/轮换。原实际四策略全恢复；保留17历史族时四策略均漏1，详见报告。所有已完成目录不可重跑覆盖；恢复路径见上级RECOVERY_INDEX。
