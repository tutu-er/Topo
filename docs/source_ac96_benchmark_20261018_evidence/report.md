# 一天96点：NJ、RNJ、WZZT与RNJ+WZZT比较

生成时间：2026-10-18T21:01:42+08:00。完整运行记录：64/64项已有结果。

四个合成低压网络，各含无源、常值源、间歇源、平滑时变源；每组使用同一背景日的96个点。源P/Q时序已知，接入位置未知。本次只有一个背景日、一个数据实现，所有方法直接拟合这96点，独立验证点数为0。

F1为保存结果中的非平凡终端簇F1；R/X误差为估计平方电压灵敏度矩阵相对真实LinDistFlow矩阵的Frobenius范数误差。表内误差乘100，以百分数表示。计算时间采用保存的算法总耗时，包含方法实际执行的前置拟合、bootstrap和求解。

每次求解上限60秒，每方法上限300秒，进程保护余量30秒；求解器gurobi，单求解器线程。

## 按方法汇总

每方法计划分母固定为16。均值保留所有有相应数值的结果，包括incomplete时保留的模型；各列括号为实际均值分母。时间均值也保留记录了耗时的失败运行。缺失值为NA，不填零；完全恢复同时列出可评分分母和计划分母。

| 方法 | 模型/计划 | complete/incomplete/failed/未出结果 | 平均F1(n) | 平均R误差%(n) | 平均X误差%(n) | 平均秒(n) | 完全恢复/可评分 | 完全恢复/计划 |
|---|---|---|---|---|---|---|---|---|
| NJ | 16/16 | 16/0/0/0 | 0.9489 (16) | 22.19 (16) | 61.44 (16) | 0.29 (16) | 9/16 | 9/16 |
| RNJ | 16/16 | 16/0/0/0 | 0.8467 (16) | 22.19 (16) | 61.44 (16) | 0.29 (16) | 1/16 | 1/16 |
| WZZT | 16/16 | 4/12/0/0 | 0.1538 (16) | 160.90 (16) | 302.66 (16) | 61.82 (16) | 0/16 | 0/16 |
| RNJ+WZZT | 16/16 | 5/11/0/0 | 0.2835 (16) | 145.08 (16) | 288.86 (16) | 95.26 (16) | 0/16 | 0/16 |

## 按源项和方法汇总

每个源项×方法计划分母为4；可评分分母随缺失结果变化。

| 源项 | 方法 | 模型/计划 | complete/incomplete/failed/未出结果 | 平均F1(n) | 平均R误差%(n) | 平均X误差%(n) | 平均秒(n) | 完全恢复/可评分 | 完全恢复/计划 |
|---|---|---|---|---|---|---|---|---|---|
| 无源 | NJ | 4/4 | 4/0/0/0 | 1.0000 (4) | 19.39 (4) | 51.32 (4) | 0.14 (4) | 4/4 | 4/4 |
| 无源 | RNJ | 4/4 | 4/0/0/0 | 0.9284 (4) | 19.39 (4) | 51.32 (4) | 0.14 (4) | 1/4 | 1/4 |
| 无源 | WZZT | 4/4 | 1/3/0/0 | 0.1538 (4) | 124.10 (4) | 256.13 (4) | 64.87 (4) | 0/4 | 0/4 |
| 无源 | RNJ+WZZT | 4/4 | 2/2/0/0 | 0.3578 (4) | 98.44 (4) | 231.85 (4) | 116.94 (4) | 0/4 | 0/4 |
| 常值源 | NJ | 4/4 | 4/0/0/0 | 0.9173 (4) | 23.89 (4) | 52.29 (4) | 0.41 (4) | 1/4 | 1/4 |
| 常值源 | RNJ | 4/4 | 4/0/0/0 | 0.7981 (4) | 23.89 (4) | 52.29 (4) | 0.40 (4) | 0/4 | 0/4 |
| 常值源 | WZZT | 4/4 | 1/3/0/0 | 0.1538 (4) | 185.93 (4) | 299.25 (4) | 62.24 (4) | 0/4 | 0/4 |
| 常值源 | RNJ+WZZT | 4/4 | 1/3/0/0 | 0.2253 (4) | 175.93 (4) | 298.49 (4) | 100.86 (4) | 0/4 | 0/4 |
| 间歇源 | NJ | 4/4 | 4/0/0/0 | 0.9833 (4) | 19.74 (4) | 52.17 (4) | 0.39 (4) | 3/4 | 3/4 |
| 间歇源 | RNJ | 4/4 | 4/0/0/0 | 0.8732 (4) | 19.74 (4) | 52.17 (4) | 0.40 (4) | 0/4 | 0/4 |
| 间歇源 | WZZT | 4/4 | 1/3/0/0 | 0.1538 (4) | 163.23 (4) | 339.23 (4) | 61.96 (4) | 0/4 | 0/4 |
| 间歇源 | RNJ+WZZT | 4/4 | 1/3/0/0 | 0.2788 (4) | 151.61 (4) | 325.31 (4) | 77.20 (4) | 0/4 | 0/4 |
| 平滑时变源 | NJ | 4/4 | 4/0/0/0 | 0.8950 (4) | 25.74 (4) | 89.97 (4) | 0.23 (4) | 1/4 | 1/4 |
| 平滑时变源 | RNJ | 4/4 | 4/0/0/0 | 0.7872 (4) | 25.74 (4) | 89.97 (4) | 0.23 (4) | 0/4 | 0/4 |
| 平滑时变源 | WZZT | 4/4 | 1/3/0/0 | 0.1538 (4) | 170.35 (4) | 316.04 (4) | 58.20 (4) | 0/4 | 0/4 |
| 平滑时变源 | RNJ+WZZT | 4/4 | 1/3/0/0 | 0.2719 (4) | 154.34 (4) | 299.77 (4) | 86.03 (4) | 0/4 | 0/4 |

## F1图

![F1 heatmap](f1_heatmap.png)

橙框表示incomplete；灰色NA表示尚无可评分结果。

## 逐案例结果

### paper15

| 源项 | 方法 | F1 | R误差% | X误差% | 秒 | status | stop_reason |
|---|---|---:|---:|---:|---:|---|---|
| 无源 | NJ | 1.0000 | 17.11 | 47.50 | 0.08 | complete | tree_inference_completed |
| 无源 | RNJ | 0.9231 | 17.11 | 47.50 | 0.08 | complete | tree_inference_completed |
| 无源 | WZZT | 0.0000 | 173.90 | 508.74 | 60.50 | incomplete | extension_not_proven_optimal |
| 无源 | RNJ+WZZT | 0.2500 | 125.08 | 460.71 | 97.15 | incomplete | extension_not_proven_optimal |
| 常值源 | NJ | 1.0000 | 19.98 | 46.09 | 0.41 | complete | tree_inference_completed |
| 常值源 | RNJ | 0.8571 | 19.98 | 46.09 | 0.40 | complete | tree_inference_completed |
| 常值源 | WZZT | 0.0000 | 189.51 | 471.53 | 63.57 | incomplete | extension_not_proven_optimal |
| 常值源 | RNJ+WZZT | 0.2857 | 149.50 | 468.49 | 100.12 | incomplete | extension_not_proven_optimal |
| 间歇源 | NJ | 1.0000 | 17.84 | 52.03 | 0.38 | complete | tree_inference_completed |
| 间歇源 | RNJ | 0.8571 | 17.84 | 52.03 | 0.39 | complete | tree_inference_completed |
| 间歇源 | WZZT | 0.0000 | 183.82 | 510.46 | 64.51 | incomplete | extension_not_proven_optimal |
| 间歇源 | RNJ+WZZT | 0.2500 | 140.83 | 462.83 | 100.29 | incomplete | extension_not_proven_optimal |
| 平滑时变源 | NJ | 0.9231 | 21.91 | 81.38 | 0.35 | complete | tree_inference_completed |
| 平滑时变源 | RNJ | 0.8571 | 21.91 | 81.38 | 0.35 | complete | tree_inference_completed |
| 平滑时变源 | WZZT | 0.0000 | 182.92 | 478.47 | 63.10 | incomplete | extension_not_proven_optimal |
| 平滑时变源 | RNJ+WZZT | 0.2500 | 138.52 | 463.04 | 73.59 | incomplete | extension_not_proven_optimal |

### soumalas11

| 源项 | 方法 | F1 | R误差% | X误差% | 秒 | status | stop_reason |
|---|---|---:|---:|---:|---:|---|---|
| 无源 | NJ | 1.0000 | 18.43 | 50.47 | 0.06 | complete | tree_inference_completed |
| 无源 | RNJ | 1.0000 | 18.43 | 50.47 | 0.05 | complete | tree_inference_completed |
| 无源 | WZZT | 0.6154 | 10.92 | 40.90 | 29.97 | complete | no_significant_one_atom_gain |
| 无源 | RNJ+WZZT | 0.6154 | 10.92 | 40.90 | 28.00 | complete | no_significant_one_atom_gain |
| 常值源 | NJ | 0.8889 | 23.53 | 56.80 | 0.06 | complete | tree_inference_completed |
| 常值源 | RNJ | 0.8889 | 23.53 | 56.80 | 0.06 | complete | tree_inference_completed |
| 常值源 | WZZT | 0.6154 | 11.12 | 41.71 | 60.46 | complete | no_significant_one_atom_gain |
| 常值源 | RNJ+WZZT | 0.6154 | 11.12 | 41.71 | 64.41 | complete | no_significant_one_atom_gain |
| 间歇源 | NJ | 1.0000 | 18.61 | 50.43 | 0.05 | complete | tree_inference_completed |
| 间歇源 | RNJ | 0.8889 | 18.61 | 50.43 | 0.05 | complete | tree_inference_completed |
| 间歇源 | WZZT | 0.6154 | 10.96 | 40.16 | 55.66 | complete | no_significant_one_atom_gain |
| 间歇源 | RNJ+WZZT | 0.6154 | 10.96 | 40.16 | 37.88 | complete | no_significant_one_atom_gain |
| 平滑时变源 | NJ | 1.0000 | 26.81 | 96.20 | 0.05 | complete | tree_inference_completed |
| 平滑时变源 | RNJ | 0.8889 | 26.81 | 96.20 | 0.06 | complete | tree_inference_completed |
| 平滑时变源 | WZZT | 0.6154 | 11.61 | 42.98 | 46.71 | complete | no_significant_one_atom_gain |
| 平滑时变源 | RNJ+WZZT | 0.6154 | 11.61 | 42.98 | 50.43 | complete | no_significant_one_atom_gain |

### flynn16

| 源项 | 方法 | F1 | R误差% | X误差% | 秒 | status | stop_reason |
|---|---|---:|---:|---:|---:|---|---|
| 无源 | NJ | 1.0000 | 23.22 | 67.40 | 0.18 | complete | tree_inference_completed |
| 无源 | RNJ | 0.8571 | 23.22 | 67.40 | 0.16 | complete | tree_inference_completed |
| 无源 | WZZT | 0.0000 | 96.65 | 108.07 | 108.25 | incomplete | extension_not_proven_optimal |
| 无源 | RNJ+WZZT | 0.3158 | 46.03 | 68.30 | 206.35 | complete | laminar_family_limit |
| 常值源 | NJ | 0.9231 | 28.66 | 64.58 | 0.18 | complete | tree_inference_completed |
| 常值源 | RNJ | 0.5714 | 28.66 | 64.58 | 0.17 | complete | tree_inference_completed |
| 常值源 | WZZT | 0.0000 | 295.65 | 328.61 | 61.19 | incomplete | extension_not_proven_optimal |
| 常值源 | RNJ+WZZT | 0.0000 | 295.65 | 328.61 | 75.39 | incomplete | extension_not_proven_optimal |
| 间歇源 | NJ | 1.0000 | 22.81 | 65.45 | 0.15 | complete | tree_inference_completed |
| 间歇源 | RNJ | 0.9231 | 22.81 | 65.45 | 0.15 | complete | tree_inference_completed |
| 间歇源 | WZZT | 0.0000 | 261.36 | 382.71 | 61.84 | incomplete | extension_not_proven_optimal |
| 间歇源 | RNJ+WZZT | 0.0000 | 261.36 | 382.71 | 78.01 | incomplete | extension_not_proven_optimal |
| 平滑时变源 | NJ | 0.8333 | 29.61 | 116.96 | 0.15 | complete | tree_inference_completed |
| 平滑时变源 | RNJ | 0.6250 | 29.61 | 116.96 | 0.16 | complete | tree_inference_completed |
| 平滑时变源 | WZZT | 0.0000 | 273.32 | 354.87 | 61.38 | incomplete | extension_not_proven_optimal |
| 平滑时变源 | RNJ+WZZT | 0.0000 | 264.63 | 313.62 | 76.55 | incomplete | extension_not_proven_optimal |

### pengwah18

| 源项 | 方法 | F1 | R误差% | X误差% | 秒 | status | stop_reason |
|---|---|---:|---:|---:|---:|---|---|
| 无源 | NJ | 1.0000 | 18.80 | 39.90 | 0.25 | complete | tree_inference_completed |
| 无源 | RNJ | 0.9333 | 18.80 | 39.90 | 0.25 | complete | tree_inference_completed |
| 无源 | WZZT | 0.0000 | 214.92 | 366.79 | 60.76 | incomplete | extension_not_proven_optimal |
| 无源 | RNJ+WZZT | 0.2500 | 211.74 | 357.48 | 136.26 | incomplete | extension_not_proven_optimal |
| 常值源 | NJ | 0.8571 | 23.38 | 41.67 | 0.97 | complete | tree_inference_completed |
| 常值源 | RNJ | 0.8750 | 23.38 | 41.67 | 0.95 | complete | tree_inference_completed |
| 常值源 | WZZT | 0.0000 | 247.45 | 355.15 | 63.75 | incomplete | extension_not_proven_optimal |
| 常值源 | RNJ+WZZT | 0.0000 | 247.45 | 355.15 | 163.50 | incomplete | extension_not_proven_optimal |
| 间歇源 | NJ | 0.9333 | 19.69 | 40.77 | 0.96 | complete | tree_inference_completed |
| 间歇源 | RNJ | 0.8235 | 19.69 | 40.77 | 1.01 | complete | tree_inference_completed |
| 间歇源 | WZZT | 0.0000 | 196.77 | 423.59 | 65.82 | incomplete | extension_not_proven_optimal |
| 间歇源 | RNJ+WZZT | 0.2500 | 193.29 | 415.55 | 92.63 | incomplete | extension_not_proven_optimal |
| 平滑时变源 | NJ | 0.8235 | 24.62 | 65.34 | 0.34 | complete | tree_inference_completed |
| 平滑时变源 | RNJ | 0.7778 | 24.62 | 65.34 | 0.34 | complete | tree_inference_completed |
| 平滑时变源 | WZZT | 0.0000 | 213.55 | 387.82 | 61.61 | incomplete | extension_not_proven_optimal |
| 平滑时变源 | RNJ+WZZT | 0.2222 | 202.60 | 379.42 | 143.55 | incomplete | extension_not_proven_optimal |

## 固定的RNJ边界块

当前保存15个固定块，15个可与已保存评价真值比较，其中6个不匹配；0个最终权重均未超过有效阈值。逐块标签、最终R/X原子权重、有效状态与已保存评分中的去留见[frozen_blocks.csv](frozen_blocks.csv)。运行中尚无最终结果的块仅列标签，最终权重和有效状态保持NA。这些块的正确性仅用于结果解释。

## 未接受的可行候选附表

保存日志中有23个未被算法接受、原数值可行性审计通过、且未获最优性认证的候选。逐事件的状态、可行目标、对偶界、gap、目标值、候选F1和R/X误差见[uncertified_incumbents.csv](uncertified_incumbents.csv)。

这些候选由保存的原子和权重重建，仅在评价阶段读取网络参数计算误差；附表保留各事件的optimality_certified字段。候选属于对应单步子问题的可行解，未认证行的最优性仍未知。主表继续使用算法最后接受的模型，候选附表不计入四种方法的正式成绩。超时回退到单点模型的F1=0，表示严格接受规则下保留模型的结构；日志同时记录已找到的未认证可行候选。

## 指标口径与解释

1. NJ与RNJ使用相同的前置ordered R/X回归，故同一数据组的R/X误差相同。该误差评价回归矩阵；F1评价后续成树结果。WZZT的R/X由最后接受的相容原子构造。
2. 有源NJ/RNJ通过FWL投影估计自由终端源响应，WZZT联合选择物理源路径。跨方法差异包含估计域、平方损失/L1损失与搜索策略的共同影响。常值和间歇源的P/Q列秩为1，平滑源为2；单独的P/Q响应分解依赖实际联合设计秩。
3. NJ的0.16用于成树后的内部边收缩；RNJ的0.16用于递归分组与根连接，两者数值相同而作用不同。阈值已预先固定。
4. WZZT主F1只计max(r,x)>1e-08的非平凡原子；NJ/RNJ计各自阈值处理后的非平凡簇。RNJ+WZZT固定但最终零权重的块仍约束搜索，其主F1按有效原子计分。
5. 源时序秩和回归设计秩是数值诊断；源响应布尔诊断字段不作为联合响应、源位置或拓扑可识别性的证书。bootstrap频率只表示本次重采样稳定性。

complete表示算法按其停止规则结束；WZZT证书对象为固定支撑或一次扩展子问题，整条贪心路径的全局最优性未验证。incomplete行保留最后接受的模型，超时和未认证求解保存在原始job记录；failed/缺失模型的指标保持NA。

主表直接汇总已保存指标。候选附表单独在估计完成后进行物理网络评价；本后处理未重新拟合、运行求解器或调参。
