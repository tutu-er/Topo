# 灵活性弱证据扫描结果

历史条件 72；完成辨识 71。资格统计：{'missing_truth': 64, 'failed': 1, 'weak': 1, 'easy': 6}。

各条件共用部分历史随机流；条件数不等于独立历史样本数。32日确认针对开发阶段选出的条件。

开发：1个历史条件，2条独立新日。

|策略|完成/尝试|平均真拓扑后验|后验≥90%|完整恢复|平均计划费/元|
|---|---:|---:|---:|---:|---:|
|no_dfs|2/2|0.074553|0/2|0/2|0.000000|
|optimized_g0|2/2|0.235032|0/2|0/2|2.246795|
|optimized_g1|2/2|0.153698|0/2|0/2|3.298364|
|rotation_g0|2/2|0.061121|0/2|0/2|2.246795|
|rotation_g1|2/2|0.085583|0/2|0/2|3.298364|

求解统计：{'quotes': 4, 'feasible_quotes': 3, 'failed_or_unresolved': 1, 'status_counts': {'feasible': 3, 'unresolved': 1}, 'source_counts': {'native_QCQP_found_feasible': 3, 'no_feasible_point': 1}, 'globally_closed_count': 0, 'verified_initial_count': 0, 'retained_initial_count': 0, 'native_time_limit_count': 4, 'mean_solver_seconds': 3.005415574996732, 'mean_relative_fee_gap': 0.43697009346642873, 'max_relative_fee_gap': 0.540972630377846, 'mean_verified_initial_fee_improvement_yuan': None, 'same_planned_fee_failures': 0}

独立确认：1个历史条件，32条独立新日。

|策略|完成/尝试|平均真拓扑后验|后验≥90%|完整恢复|平均计划费/元|
|---|---:|---:|---:|---:|---:|
|no_dfs|32/32|0.613427|14/32|20/32|0.000000|
|optimized_g0|32/32|0.754863|16/32|25/32|1.783433|
|optimized_g1|32/32|0.749983|18/32|25/32|5.327395|
|rotation_g0|32/32|0.611818|14/32|20/32|1.783433|
|rotation_g1|32/32|0.643195|15/32|21/32|5.327395|

求解统计：{'quotes': 64, 'feasible_quotes': 49, 'failed_or_unresolved': 15, 'status_counts': {'feasible': 49, 'unresolved': 15}, 'source_counts': {'native_QCQP_improved_initial_point': 10, 'native_QCQP_found_feasible': 39, 'no_feasible_point': 15}, 'globally_closed_count': 0, 'verified_initial_count': 10, 'retained_initial_count': 0, 'native_time_limit_count': 64, 'mean_solver_seconds': 3.0051937218750027, 'mean_relative_fee_gap': 0.36255154412319995, 'max_relative_fee_gap': 0.8584938276194526, 'mean_verified_initial_fee_improvement_yuan': 4.924465619201646, 'same_planned_fee_failures': 0}

逐例数据见per_case.json，缺真候选与错误winner见history_summary.csv。

费用为预设资源报价下的激活补偿，不是整网日运营成本；没有可用线路额定容量，未核查热限。源码、配置、预算和私有原始轨迹的哈希随protocol与各日封存记录保存。
