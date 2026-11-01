# 冻结DR方案的独立实际模拟核验

先读[REPORT.md](REPORT.md)。本轮独立12新日、5策略、60/60 AC策略日已完成，没有待续求解。既有1/5条件具规划资格，其他4保持2不可行/2未决。

driver.py命令prepare/observe/score分别冻结全部请求、调用原物理worker、冻结预测后标记真值。价格和R/X不再优化。每新日独立追加同一完整历史，原3%/新日2%/w2/.16保持。低费用主版本按规划可行＋最便宜已知见证选择，高费用为预锁敏感性；两层各配同费用全天轮换。

实际批不重跑。若以后另锁新协议，入口为python -B -m research_experiments.wzzt_dr_selection.dr_day_effect.driver，prepare需--output NEW_ROOT、--worker-gate PASSING_WORKER_JUNIT、--driver-gate PASSING_DRIVER_JUNIT；之后observe/score使用同一root。

新物理接口用payment_mode=frozen_plan_bound、optimizer_payment_cap=null和冻结计划导出的payment_bound。这个界校验结算，不限制优化；旧默认1.6不改。零DR执行完整AC/计量，不被不需要的设备组成资格挡掉。

公开标量在evidence/day_effect_actual_v1，可直接用plot_effect.py --source PUBLIC_EVIDENCE --output NEW_FIGURE_DIR重绘。export_scalars.py及audit_sources为事后处理/独立核验，不运行生成器或拟合器。

私有恢复根wzzt_day_effect_stage：actual_v1/protocol/freeze/source_snapshot、requests及actual保存全部来源/价格/随机流/回执/轨迹；scores、predictions_frozen、scored、ended给出顺序。audit_selection_v1含预审856及最终7194复核；audit_actual_v1/results_v1含5017物理/机制复核。public_release_v2、public_figures_v3为发布标量和图；plot_failure_v1及public_figures_v1保留编码失败。

worker/source实际版本为提交a212f761及driver6237518b，以协议261源hash和zip字节为准。后续发布的export/plot/报告不冒称事前运行逻辑。旧报价结果和原无支付上限模型均未覆盖。全P/Q/V/AC、完整R/X、96槽曲线留私有，不上传。
