# 距离增量与时长图表

先读[REPORT.md](REPORT.md)。本目录把原5条件、19/20有效模拟日可视化，并做固定模型的强度/时长反事实；没有新增AC、拟合、辨识或价格搜索。

- 主图：`figures/distance_baseline_increment.*`，距离基值/增量与原激励目标gamma。
- 目标评价：`figures/planning_objective_duration.*`，同一1000工作场景的90%可达增量与达标比例。
- 时长/资源：`figures/duration_same_budget.*`，同1.6预算上限、固定相对报价，延长累计削减时段并等时回补。
- 强度/预算：`figures/strength_and_budget.*`，alpha>1明确超出原预算。
- 辅助核验：`figures/actual_exit_margin_examples.*`，原实际M=0边界及正反例。

每图有PNG、SVG、PDF。L1距离增量是前后两个完整预测差范数之差，不是DR单独响应范数；g90为降序第ceil(.9S)个值，不是上侧90%分位。原完整错漏结果保留在上轮报告及本轮辅助图中。

公共标量位于`../evidence/signal_analysis_v1`。直接重绘不需私有轨迹：

```text
python -m research_experiments.wzzt_dr_selection.dr_signal_analysis.plot_existing --source PUBLIC/observed_figure_data.json --output NEW_FIG_DIR
python -m research_experiments.wzzt_dr_selection.dr_signal_analysis.plot_sensitivity --source PUBLIC/dose_scalars.json --output NEW_SENSITIVITY_DIR
python -m research_experiments.wzzt_dr_selection.dr_signal_analysis.plot_working_objective --source PUBLIC/working_objective_scalars.json --output NEW_OBJECTIVE_DIR
```

输出目录要求新建，防止覆盖。底层私有算术复现需明确传入已有冻结研究根：

```text
python -m research_experiments.wzzt_dr_selection.dr_signal_analysis.dose_duration --root PRIVATE_NEW_DAY_RUN --output NEW_PRIVATE_DOSE_DIR
python -m research_experiments.wzzt_dr_selection.dr_signal_analysis.evaluate_working_objective --root PRIVATE_NEW_DAY_RUN --dose-root PRIVATE_DOSE_DIR --output NEW_PRIVATE_WORKING_DIR
```

旧结果均已完成，通常只重绘，无需重复核算。新增长时动作尚未运行；不能把反事实曲线写成新交付数据或现场概率保证。
