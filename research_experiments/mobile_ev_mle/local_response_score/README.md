# 局部响应评分与凸QP

先读[结果](results.md)、[数学推导](math.md)、[计算前方案](protocol.md)。实现只接受带噪末端P/Q/V、已知量测标准差和公开物理域；原3/7天AC批次不变。

- `scoring.py`：候选树响应约束、P/Q解析消元及梯度、原参数域响应拟合、凸QP及显式下界。
- `ac_holdout.py`：固定训练线路，完整AC解释独立带噪留出量测。
- `remainder.py`：局部精确AC恒等式、公共/观测收紧/固定线路余项范围。
- `rejection.py`：含余项box的凸QP和条件性拒绝包络，默认不激活。
- `experiment.py`、`physics.py`、`benchmark.py`：冻结主比较、余项检查及新配对计时。
- `audit.py`、`analysis.py`：独立重算与保留完整分母的汇总。

原始结果：`outputs/local_response_main_v1`。全部主批及费用重放完成；代表证据归档恢复见`evidence/restore_receipt.json`。本轮的快速评分适合筛查，尚未建立稳定的唯一局部拓扑辨识。
