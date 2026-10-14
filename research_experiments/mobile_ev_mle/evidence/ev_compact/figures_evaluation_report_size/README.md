# 事前指定四条评价历史的 EV 曲线

绘图种子为事前指定的 `989201,989202,989203,989204`。等待 `evaluation_ensemble_v2` 的 `989201..989212` 共12个完成文件齐备，并核对文件名、字节数及最后修改时间在38.44秒内稳定后，才读取这四条历史的已存曲线；没有按结果换例或查看 root 会话。

```text
python -m research_experiments.mobile_ev_mle.ev_compact_plot --directory research_experiments/mobile_ev_mle/evidence/ev_compact/evaluation_ensemble_v2 --kind raw --seeds 989201 989202 989203 989204 --layout report --output research_experiments/mobile_ev_mle/evidence/ev_compact/figures_evaluation_report_size
```

图为4历史×3方法：恒充、精化平均、上限20。颜色标接入点，正功率充电、负功率放电；底色区分移动与接线。最终来源按保存记录标注，包括继承计划。M为实际联合AC的最小错误候选SSE减真实候选SSE，由已存局部拟合成本复算。

PDF为8×8.5英寸；缩至约6.5英寸正文宽时，M/来源和轴标签约8.1pt。PNG及PDF按正文宽度渲染后均已视觉检查，分数和来源不遮曲线。全部12个方法例图保留，含负M及相同继承曲线。

`figure_data.json` 保留原功率、接入、逐槽状态、M和来源，并绑定四个输入压缩SHA；`plot_manifest.json` 绑定数据/PDF/PNG；`plot_source_snapshot.py` 保存实际绘图源码，SHA为 `3e9c24a98eb382881169e52f54f7cd5ef02a267e1e457aaebfc6f32fac2b3026`；`input_readiness.json` 和 `plot_validation.json` 保存稳定门及检查范围。

这里只展示预声明样例，没有新拟合、优化、动作选择、参数修改或72源改动。总体评价仍按完整12条历史的独立统计报告。
