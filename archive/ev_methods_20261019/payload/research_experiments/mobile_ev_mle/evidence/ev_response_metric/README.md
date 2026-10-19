# 响应指标与标量MILP证据

2026-10-16。源代码、公式及结论入口为 `../../ev_response_metric_results.md` 和 `../../ev_response_metric_reproduce.md`。所有计量为已声明生成器产生的合成数据，无现场私人数据。

|证据|范围|
|---|---|
|ranking_diagnostics.zip及manifest|旧4开发/16评价降为开发诊断；72条不同曲线，线路/负荷/源替换|
|raw.tar.xz及manifest|101源码冻结的严格两候选9法：4开发、18校准、8评价；38个原文件|
|formal_*.json.gz、formal_summary.md|原正式统计、数值阈值和逐历史决策|
|*_independent_review.json.gz|30份历史独立物理、源、梯度和对比MILP核查|
|launch、calibration_merge、threshold_saved_before_evaluation|运行、合并和先阈值后读取评价的时间记录|
|restore.json、replay_comparison.json|38文件SHA完全恢复；换目录后数值阈值/逐历史决策一致，含路径报告字节有差异|
|conditional_development.zip及manifest|仅4开发、每份32条条件未来；真实过去状态条件化的机制诊断|
|scalar_development.zip及manifest|105源码另冻、4份复用开发；三点及细网格MILP、原始界、AC接受计划及实际结果分别记录|
|integrated_tests.xml、scalar_unit_tests.xml、scalar_independent_tests.xml|220+18+10项测试；248项总计，无重复计数|
|delivery_manifest.json|本目录全部其他交付文件SHA256及字节数|

ZIP归档逐文件校验解压字节，包含各阶段原始记录及源码快照。标量MILP及条件积分没有进入9法正式结果。重新使用其开发历史不增加独立样本数。原始MILP证书的对象是冻结仿射问题，完整AC得分及实际计量裕量另行保留。
