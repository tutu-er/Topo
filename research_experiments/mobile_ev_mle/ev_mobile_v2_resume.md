# 冻结EV研究交付续接

本轮所有计算、统计和审计均已完成，无活跃controller、worker或后处理任务。**不要重新启动旧批次，不追加样本，不据评价调整策略。** 最终发布与续接关闭状态见`ev_mobile_v2_checkpoint.json`和Git。

先读`ev_mobile_v2_results.md`及`ev_mobile_v2_costs.md`。数学在`ev_mobile_v2_math.md`、`ev_reliability_v2_math.md`；复算与证据层次在`ev_mobile_v2_reproduce.md`。

- 192校准、96独立评价；352来源槽物理回放通过，开发重复槽单列。
- 288份源拟合缓存归属完整，未知/未付费命中0；32预声明空间槽完成。
- 237项汇总测试通过；初次临时目录setup失败及原规划/预测/推断失败均保留。
- 交付14逻辑目录352槽，21包、14预选完整计量压缩代表和两库eval0完整raw；整包/成员/恢复SHA均核对。
- 源码工作树`C:/Users/23761/.codex/worktrees/36a1/Topo`，执行及完整raw保留`D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015`。
- 分支`codex/ev-train-validation-design`，用户已批准向`https://github.com/tutu-er/Topo.git`普通push；仅提交研究相关改动。

最后只需确认本轮最终commit/push与结束自动续接已经完成。如果发布遇到明确阻塞，保留本地提交及证据，记录具体原因；不要将其变成新实验。

下一研究尚未启动。建议先核对动作收益预测排序，拆分线路、负荷、源误差，再减少联合梯度法的重复拟合成本；新策略须重新冻结并独立校准评价。

无关PDF原SHA为`fe89b241285fd81f570a921fe99a49bfdb1d6bca6e0ced0d498a45ef81d5dbfa`；`pyproject.toml`为`67d21b6a3da4629c5cba5c445404fc5ecde154bf6b472ee4fa06f75ceb760e1c`，本轮保持原样。最终证据清单包含所有原字节及实际失败分母。
