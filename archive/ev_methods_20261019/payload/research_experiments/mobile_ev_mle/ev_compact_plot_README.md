# 保存的 EV 曲线图

`ev_compact_plot.py` 从保存的 raw/slim 记录提取功率和操作状态，输出一页 4 历史 × 3 方法图：恒充、精化平均策略、上限20策略。地点以三色表示，移动与接线分别标出；已连接但零功率的时段保留地点标记。图中同时标注实际 AC 的 SSE 裕量 M 和最终计划来源，区分 MILP 新曲线与继承计划。

必须显式给出四个预声明种子，不按结果挑选曲线。开发接口示例（从仓库根目录运行）：

```text
python -m research_experiments.mobile_ev_mle.ev_compact_plot --directory research_experiments/mobile_ev_mle/evidence/ev_compact/delivery_development_v2/development_ensemble_v2 --kind slim --seeds 989101 989102 989103 989104 --output <新的图目录>
```

同一输入目录同时含 raw 与 slim 时用 `--kind raw` 或 `--kind slim` 明确选择；自动模式拒绝重复来源。工具支持声明的 development/evaluation 阶段，拒绝 calibration。本次只运行了开发样本；评价图预定使用 989201、989202、989203、989204，尚未读取它们。

`figure_data.json` 保留完整浮点功率、逐槽接入/活动状态、实际 M、来源、输入压缩字节 SHA、策略身份、脚本 SHA 和字体来源；`plot_manifest.json` 绑定数据、PDF 和 PNG 的哈希。输出目录必须是新目录，避免覆盖旧研究证据。

功率在每个 15 分钟槽内保持恒定，跨槽值由优化得到，图不对相邻槽插值。M 由已保存的候选联合拟合误差复算：最小错误候选 SSE 减真实候选 SSE；拟合是局部取得值。图不重新拟合、调用优化或选择动作。

最终开发图位于 `evidence/ev_compact/figures_development_ensemble_v2_final/`；首次排版图另存，最终版将分数和来源移到坐标轴上方以避免遮住曲线。

报告插图使用 `--layout report`，输出 8×8.5 英寸、10pt 分数/来源与轴标签；缩到约6.5英寸正文宽度时，两者约8.1pt。尺寸、字号和边距均写入 `figure_data.json` 与 `plot_manifest.json`。默认仍为原12英寸布局。报告尺寸开发图另存在 `evidence/ev_compact/figures_development_report_size/`；旧 `_final` 图、数据和清单保持原字节，匹配旧清单的绘图源码另存为 `plot_source_snapshot.py`。
