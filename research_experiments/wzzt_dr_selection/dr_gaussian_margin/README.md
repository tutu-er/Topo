# 恢复入口

先读MODEL.md及REPORT.md，证据evidence/gaussian_margin_v1。实施提交8c726f95。所有3求解、8000选模已结束，不重跑旧记录。

私有根C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_gaussian_margin_stage：run_v1含protocol、source_snapshot、problem.pkl/problem_frozen、corrected/naive/max_diagnostic求解receipt与log、price_freeze、evaluation_draws/scalars、predictions_frozen、ended。现有源和输入哈希在protocol，实际参数矩阵/曲线不发布。

gaussian_socp.py提供prepare/evaluate_coefficients/box_bound/solve_coefficients。runner.py命令prepare/solve/evaluate，--output必须为新目录；已完成run_v1不应重启。PYTHONPATH应含仓库根、rnj_wzzt_core和本模块目录。prepare目前明确锁原paper15条件，参考高斯预测只使用训练history，validation仅算历史损失。

audit_independent.py、audit_evidence.py、audit_labels.py独立读取已保存结果；审计来源及回执hash在公开audit_summary。参考误差翻号恒等式、共模抵消、原式费用下界均独立核过。首轮runner语法和audit布尔比较失败回执与源码单独留存。

当前无合格corrected报价，AC gate=false。下一阶段应先重新审视联合翻号差和Gaussian预测支持，不得将naive或max诊断自动部署到旧有限支撑/能量守恒worker。主识别器未改变。
