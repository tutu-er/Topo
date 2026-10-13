# C 阶段 MILP 与原动作库优化诊断

**C 的 71 对 81 次唯一正确保留差异同时包含求解受限与代理目标对实际终点不充分的问题，不能单独归因于“MILP 搜索域更大所以应更好”。** refit 未保证选中动作的代理值高于原 11 动作库：255 条完整比较中，原库动作下界高于 refit 动作上界 99 条，反向 16 条，区间重叠 140 条；另 1 条原库选址失败保留在全 256 历史分母内。所有数字来自既有 C trial，未调用优化器或更改冻结策略。

## 同一判据和两层界

源码 adaptive_milp_study.py:142–158 与 164–171 表明，原库遍历全部 11 个动作，用同一 frozen soft-history local-refit proxy 的 q_lower 最大者（动作名作确定性 tie-break）选址；refit 对同样的历史上下文、中心/情景/参数盒与冻结线性化构造进行 MILP 设计搜索。原库是有限基线动作库，未穷举 MILP 全设计域。fixed_mle_milp 固定中心、来源与阻抗范围，属于不同判据，本诊断不把它混入上述比较。

对固定动作 u，每 pair 内层盒约束凸 QP 的 q_lower 是数值切线下界，q_upper 是可行 primal 上界；跨 pair 取最小值，得到所选动作的代理区间。refit 的 proxy_lower 是已找到且经运行约束核查的动作 lower incumbent；proxy_upper 是全设计域最大化目标的上界，初值来自 control-box witness upper，之后仅有可用 secant-master bound 才能收紧。**全域 proxy_upper 不是所选动作 q_upper。** 本诊断从 selected_u_kw 匹配 records，重复的相同控制（包括 +0/−0）取区间交集，256 条均匹配且动作下界等于 proxy_lower。

## 求解状态与动作代理排序

refit 的外层保存 status=complete 共 256，表示选址与预测准入完成；原搜索停止原因须读 proxy_stop_reason：proxy_gap_met=128，budget_or_iteration_limit=128。direction_lower 共 515 次，均 HiGHS status=0 / success=true；secant_upper 共 515 次，均 status=1/time limit，且 incumbent、bound 全部缺失。因此没有一次 secant-master bound 更新，全域 UB 始终保留初始 box upper；“选址完成”不能解释为全设计域代理最优。

| 描述层 | 历史数 | refit动作LB > 库动作UB | 库动作LB > refit动作UB | 重叠 | 库失败 |
|---|---:|---:|---:|---:|---:|
| weak_cherry | 102 | 8 | 40 | 54 | 0 |
| star | 65 | 3 | 21 | 41 | 0 |
| strong_cherry | 89 | 5 | 38 | 45 | 1 |

99 条严格库优区间中，76 条优势 >1e−4，75 条 >0.01，69 条 >1；76 条超过 1e−6×max(1,|两动作代理值|)。这些仅说明数值优势尺度，不修改原规则。128 条 gap_met 中，21 条仍有库动作LB > refit动作UB，但优势多在小尺度；128 条预算/迭代停止中，对应 78 条。库动作下界超过 refit 全域 UB 为 0 条，未观察到这一类全域界矛盾。

## 停止 gap 与新增收益的尺度

源码 ev_information_milp.py:390、395–400 使用

`global_gap=max(0,proxy_upper−proxy_lower)`，`relative_gap=global_gap/max(1,|proxy_lower|)`，允许 gap 为 `1e−4 + 1e−3×max(1,|proxy_lower|)`。

q 的历史残差和未来候选差异一并进入目标（adaptive_history.py:423–427）。因此相对 gap 的分母是总代理代价，不是相对 constant 的新增收益。本诊断的动作增益区间为 `[refit动作LB−constant动作UB, refit动作UB−constant动作LB]`，不以全域 UB 充当 refit 动作分数。

128 条 gap_met 中，43 条动作增益区间包含 0；28 条允许 gap 大于正部增益上界；其中 126 条超过绝对 gap=1e−4，停止依赖相对项。本诊断按最小 proxy_q_lower 选取保存的非线性 rollback witness，future share 中位数约 3.27e−7，n=254，典型历史由历史项占据。这支持“按总代价相对停止可能不能分辨新增收益”这一机制解释；该 witness 比例不等于代理最优值的严格历史/未来分解。

| 历史 | 层 | refit总LB | 全域gap | 允许gap | 相对constant动作增益区间 | 停止原因 |
|---|---|---:|---:|---:|---|---|
| 730007 | weak_cherry | 59.7551 | 0.0024939808 | 0.0598551 | [-0.0023877359, 7.1407401e-07] | proxy_gap_met |
| 730030 | strong_cherry | 91.805474 | 3672.636 | 0.091905474 | [40.719616, 40.719616] | budget_or_iteration_limit |
| 730084 | strong_cherry | 94.05544 | 9.433765 | 0.09415544 | [1.9251512, 1.9251512] | budget_or_iteration_limit |

730007 的总 LB≈59.7551，允许 gap≈0.059855，而新增收益区间为 [−0.00238774,7.14e−7]，全域 gap≈0.00249398；已满足总代价相对停止，却未证实相对 constant 的正收益。730030 的库代理优势≈1317.06、全域 gap≈3672.64，属于明显预算/迭代未闭合例，库唯一正确而 refit 未唯一正确。

## 与 71 对 81 的实际终点关联

两者均唯一正确 67，refit 单独 4，库单独 14，两者均未唯一正确 171；净差为库多 10。库单独 14 条中，6 条代理区间严格库优，8 条重叠。refit 单独 4 条中，1 条代理区间严格库优，3 条重叠；严格 refit 代理优的 16 条没有出现 refit 单独唯一正确。这是保存样本中代理排序不足以决定实际唯一保留终点的直接反例，不能把更高代理值解释成更高拓扑正确概率。

所以求解受限有明确证据（99 条动作代理被基线分离压过、全部 secant 超时无界），同时当前代理与局部 AC 联合拟合/双门限终点不构成单调对应。现有证据没有识别出这 10 条净差各由多少求解误差、线性化、参数盒/历史项或门限效应造成；重叠不等于相同代理值。进一步归因需要预先设计的同判据精度/预算研究，此诊断没有开展新实验或依据评价结果反调策略。

raw_extracts.json.gz 保留全 256 的选中动作区间、完整 11 动作库记录（失败历史只有已保存 1 行）、global界/停止原因/solver记录与实际事件；history_diagnostics.csv 为逐历史表，numbers JSON 为统计与实例，input_hashes.json 保留 256 原 trial、配置、门限审查及冻结源码哈希。所有比较界仅针对声明冻结仿射代理与有限 pair，属于浮点数值证据，不能升级为完整非线性 AC 全局最优证书。
