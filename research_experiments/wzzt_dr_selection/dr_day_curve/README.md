# 固定24小时曲线：无预设预算上限

先读[REPORT.md](REPORT.md)及[MODEL.md](MODEL.md)。当前主构造器budget=None，费用为目标和输出；设备、能量和可用时窗继续限制响应。通用adapter仍兼容旧budget=1.6默认，但result_bridge显式传入当前None。无新AC、实际未来或参数拟合。

核心：day_model.py、day_solver.py、adapter.py、runner.py；result_bridge.py/repair_tools.py先存raw再恢复数值能量等式，重算原式目标和资源。

检查：python -B -m pytest -q -p no:cacheprovider research_experiments/wzzt_dr_selection/dr_day_curve --basetemp PRIVATE_NEW_TMP --junitxml PRIVATE_JUNIT

现有批次全部完成，不应重跑。以后另锁新模型协议时才运行runner prepare --output NEW_PRIVATE_ROOT --unit-gate PRIVATE_JUNIT --seconds 60 --cost-seconds 30，然后run/evaluate。prepare要求至少28项通过。命令模块前缀为research_experiments.wzzt_dr_selection.dr_day_curve.runner。

这些命令只测试历史误差条件模型。无上限DTO明确existing_actual_worker_compatible=false，不能直接交给仍固定1.6的旧物理worker冒充完整实际执行。

私有恢复根wzzt_day_curve_stage：model_pilot_uncapped_v2为当前5条件；audit_uncapped_v2独立复核；uncapped_export_v1为公共标量及矩阵审核。model_pilot_v1、paper_adapter_retry_v1、combined_evaluation_v1/v2、audit_v1是旧1.6版、重试及中止。

公开evidence/day_curve_pilot_v1以无上限为主，capped_summary.json单列旧批。audit_sources保留外部一次性核验源码原字节，以私有协议与源zip恢复。曲线、P/Q/V、AC及完整R/X均不上传。
