# 一日96点：末端联合方法独立对照

已核验 144/144，失败 0，无效 0，待完成 0。

48输入=四网络×四个新种子×三档噪声；每输入训练为一整天96槽，验证为独立三天288槽，测试为独立一天96槽。三方法同预算、同少量RNJ初值、原始WZZT、清零和最终E；仅T开关/末端惩罚系数不同。最终为自身路径的.24收缩重拟合与严格验证，不使用完整RNJ整树备用。

| method | planned | returned | verified | pending | failed | invalid | F1 | exact | exact_fraction_planned | recovery_rate_completed | total_seconds | operational_total_seconds | interrupted_attempt_seconds | interrupted_attempts | interruption_affected_evaluations | median_seconds | complete | MIPs | MIP_timeouts | cap_exceeded | successful_attempt_cap_exceeded | model_solver_failure_events | raw_pool_truth | output_pool_truth | exit_connection_pool_truth_covered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noT | 48 | 48 | 48 | 0 | 0 | 0 | 0.915125 | 33 | 0.6875 | 0.6875 | 45498.8 | 46228.9 | 730.124 | 3 | 3 | 494.769 | True | 310 | 0 | 0 | 0 | 0 | 35 | 33 | 46 |
| T003 | 48 | 48 | 48 | 0 | 0 | 0 | 0.912893 | 33 | 0.6875 | 0.6875 | 43923.2 | 44523.8 | 600.599 | 3 | 3 | 425.287 | True | 244 | 0 | 0 | 0 | 0 | 35 | 33 | 46 |
| T010 | 48 | 48 | 48 | 0 | 0 | 0 | 0.911405 | 33 | 0.6875 | 0.6875 | 44401.3 | 44932.9 | 531.563 | 2 | 2 | 468.965 | True | 240 | 0 | 0 | 0 | 0 | 35 | 33 | 46 |

未完成时，均值仅描述已核验子集；exact_fraction_planned保留计划分母，不能当成未完成算法最终恢复率。total_seconds为最终返回尝试的算法墙钟之和，含公共前端记账；operational_total_seconds另计恢复回执估算的基础设施中断墙钟，不是求解器CPU时间。重试的成功尝试按剩余预算检查，成功时间加中断记账按原始4小时总额另检。

成对时间差只比较最终成功尝试，未加入中断损失；并行CPU争用与基础设施恢复使其不是理想独立运行速度对照。MILP数量和状态来自最终返回尝试，先前中断前缀另存恢复审计，不混入模型求解失败。

## 成对差异

| before | after | planned | paired | delta_F1 | delta_exact | delta_recovery_rate | delta_seconds | F1_wins | F1_ties | F1_losses | new_exact | lost_exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noT | T003 | 48 | 48 | -0.00223214 | 0 | 0 | -1575.53 | 0 | 47 | 1 | 0 | 0 |
| noT | T010 | 48 | 48 | -0.00372024 | 0 | 0 | -1097.43 | 0 | 46 | 2 | 0 | 0 |
| T003 | T010 | 48 | 48 | -0.0014881 | 0 | 0 | 478.107 | 0 | 47 | 1 | 0 | 0 |

## 同种子噪声成组重采样

| before | after | paired_inputs | shared_seed_blocks | case_seed_histories | complete | resamples | seed | scheme | delta_F1_lower95 | delta_F1_upper95 | delta_recovery_rate_lower95 | delta_recovery_rate_upper95 | delta_total_seconds_lower95 | delta_total_seconds_upper95 | interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noT | T003 | 48 | 4 | 16 | True | 10000 | 2026111017 | resample four globally shared profile-seed blocks; each block carries all four fixed cases and all three noise levels | -0.00669643 | 0 | 0 | 0 | -2088.25 | -791.864 | descriptive only: four shared seed groups, four fixed networks; few resampling units; not a topology confidence probability |
| noT | T010 | 48 | 4 | 16 | True | 10000 | 2026111017 | resample four globally shared profile-seed blocks; each block carries all four fixed cases and all three noise levels | -0.00744048 | 0 | 0 | 0 | -3095.59 | 900.731 | descriptive only: four shared seed groups, four fixed networks; few resampling units; not a topology confidence probability |
| T003 | T010 | 48 | 4 | 16 | True | 10000 | 2026111017 | resample four globally shared profile-seed blocks; each block carries all four fixed cases and all three noise levels | -0.00446429 | 0 | 0 | 0 | -2235 | 2760.2 | descriptive only: four shared seed groups, four fixed networks; few resampling units; not a topology confidence probability |

生成器在四个网络之间复用同一物理曲线种子。因此，16个case/seed历史只有4个独立种子组；每次重抽整个共享种子组，同时携带四个网络和三档噪声。区间只描述四个固定网络的种子变化，重采样单位很少，不是拓扑真实性概率。

候选库同时报告完整树是否出现、连接并集是否覆盖真连接，两者不等价。候选导出保留历史已拟合完整响应及来源，未用真拓扑或未来DR数据筛选。对接DR时须另声明strict/.24/refit与3%/.16/no-refit的不同选模合同。
