# T 正则机制诊断：保存剖面的独立后分析

只比较固定 T 域中的训练最优结构；未运行逐步搜索、最终 E、收缩后 LP 或最终路径验证。

已返回 6/6 输入，基数 MILP 50 个，达到设定精度 50 个。

alpha=lambda/f0。计数 q 是候选二元变量数，固定骨架另计常数成本；正权重线路数分别报告。

以下数值区间只比较不同 q 的竞争结构，还必须满足同 q 条件。Flynn 虽有非空的跨基数区间，同 q 竞争解已严格优于真实参考，因此该数值区间不是让真实参考最优的可用参数区间。

| job | true_q | cross_cardinality_necessary_alpha_lower | cross_cardinality_necessary_alpha_upper | same_count_counterexample | true_reference_ruled_out | true_reference_supported_within_tol |
|---|---|---|---|---|---|---|
| main__soumalas11__b2__reference__v0002__N24 | 2 | 0.0078816605 | 0.168379627 | False | False | True |
| main__soumalas11__b3__reference__v0002__N48 | 2 | 0.00923573298 | 0.144180942 | False | False | True |
| main__soumalas11__b3__reference__v0005__N48 | 2 | 0.00916364188 | 0.0311250588 | False | False | True |
| main__soumalas11__b4__reference__v0002__N24 | 2 | 0.00548636872 | 0.0688817247 | False | False | True |
| main__flynn16__b3__reference__v0002__N24 | 2 | 0.00843993668 | 0.0717765016 | True | True | False |
| stress__soumalas11__b2__root_noise__v0005__N48 | 2 | 0.00459977938 | 0.0147973996 | False | False | True |

下表的 raw 指原始系数；threshold_only 只对已存系数应用0.24阈值，没有重新拟合。它们不能当作完整算法成绩。

| alpha | completed | raw_F1 | raw_exact | threshold_only_F1 | threshold_only_exact | global_gap_met |
|---|---|---|---|---|---|---|
| 0.003 | 6 | 0.851851852 | 0 | 0.850925926 | 0 | 6 |
| 0.006 | 6 | 0.931623932 | 2 | 0.944444444 | 3 | 6 |
| 0.01 | 6 | 0.987179487 | 5 | 1 | 6 | 6 |
| 0.02 | 6 | 0.963369963 | 4 | 0.976190476 | 5 | 6 |
| 0.03 | 6 | 0.963369963 | 4 | 0.976190476 | 5 | 6 |

真值仅用于事后评分和参考 LP；alpha-grid 在所有保存基数解中按训练目标最小选择，没有用真值择优。
