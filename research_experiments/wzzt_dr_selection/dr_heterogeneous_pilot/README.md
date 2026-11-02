# 异质DR小批实际测试

先读REPORT.md（完整结果和分母）、MODEL.md（原L1条件、价格/资源/相关随机性）、PLAN.md与REPAIR_PLAN.md（冻结前范围和一次数值修正）。公开汇总在../evidence/heterogeneous_pilot_v1。

- devices.py：资源配置展开、因果能量可行交付、价格费用和控制链导数。
- planner.py：复用原条件积分，名义计划与随机执行均包含设备控制。
- planning_repair.py：仅价格冻结前的有界初值/可行化修正；均值只用于初始化，原概率目标不变。
- driver.py：prepare/optimize/observe/evaluate；复用原AC、量测和冻结RNJ＋WZZT候选选模，无新拟合/结构搜索。
- test_pilot.py、test_repair.py：12项不同测试。
- post_audit.py：独立只读重放；summarize.py、export_results.py：只导出明确汇总，不导出P/Q/V、完整R/X、u或报价轨迹。

已完成私有目录：

    C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_heterogeneous_pilot_stage/

run_v1只有原规划，已由run_v2替代；run_v2已经144/144完成，勿重启observe。run_v2/{protocol,price_freeze,observations_frozen,predictions_frozen,scored_private,summary,aggregate_diagnostics,independent_audit,ended}.json与actual逐记录提供恢复入口。source_snapshot.zip保存精确运行源；发布源码不回写旧hash。

后续新运行使用新的输出目录与预锁种子。原固定路径采用本研究现有本地档案，外部复现需要相应合法输入。当前效果仅对冻结历史候选的最终选择，不是重新运行完整RNJ＋WZZT搜索；95%规划约束三例均未达，诊断报价与实际6/6结果分别报告。
