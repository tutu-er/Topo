# 灵活性弱证据扫描结果

历史条件 72；完成辨识 71。资格统计：{'missing_truth': 64, 'failed': 1, 'weak': 1, 'easy': 6}。

各条件共用部分历史随机流；条件数不等于独立历史样本数。32日确认针对开发阶段选出的条件。

开发：7个历史条件，14条独立新日。

|策略|完成/尝试|平均真拓扑后验|后验≥90%|完整恢复|平均计划费/元|
|---|---:|---:|---:|---:|---:|
|no_dfs|14/14|0.691518|4/6|4/14|0.000000|
|optimized_g0|14/14|0.702619|4/6|4/14|0.198468|
|optimized_g1|14/14|0.696152|4/6|4/14|0.743152|
|rotation_g0|14/14|0.690880|4/6|4/14|0.198468|
|rotation_g1|14/14|0.691346|4/6|4/14|0.743152|

求解统计：{'quotes': 28, 'feasible_quotes': 25, 'failed_or_unresolved': 3, 'status_counts': {'feasible': 25, 'native_infeasible': 3}, 'source_counts': {'native_QCQP_improved_initial_point': 24, 'no_feasible_point': 3, 'native_QCQP_found_feasible': 1}, 'globally_closed_count': 10, 'verified_initial_count': 24, 'retained_initial_count': 0, 'native_time_limit_count': 11, 'mean_solver_seconds': 1.6446040321461624, 'mean_relative_fee_gap': 0.15335159879756521, 'max_relative_fee_gap': 1.0, 'mean_verified_initial_fee_improvement_yuan': 0.046081430537727376, 'same_planned_fee_failures': 0}

独立确认：1个历史条件，32条独立新日。

|策略|完成/尝试|平均真拓扑后验|后验≥90%|完整恢复|平均计划费/元|
|---|---:|---:|---:|---:|---:|
|no_dfs|32/32|0.613427|14/32|20/32|0.000000|
|optimized_g0|32/32|0.640216|14/32|20/32|0.287929|
|optimized_g1|32/32|0.618781|14/32|20/32|0.076030|
|rotation_g0|32/32|0.612934|14/32|20/32|0.287929|
|rotation_g1|32/32|0.613308|14/32|20/32|0.076030|

求解统计：{'quotes': 64, 'feasible_quotes': 64, 'failed_or_unresolved': 0, 'status_counts': {'feasible': 64}, 'source_counts': {'native_QCQP_improved_initial_point': 60, 'native_QCQP_found_feasible': 4}, 'globally_closed_count': 12, 'verified_initial_count': 60, 'retained_initial_count': 0, 'native_time_limit_count': 52, 'mean_solver_seconds': 2.6800485374988057, 'mean_relative_fee_gap': 0.479865671655057, 'max_relative_fee_gap': 1.0, 'mean_verified_initial_fee_improvement_yuan': 0.06820142886986709, 'same_planned_fee_failures': 0}

逐例数据见per_case.json，缺真候选与错误winner见history_summary.csv。

费用为预设资源报价下的激活补偿，不是整网日运营成本；没有可用线路额定容量，未核查热限。源码、配置、预算和私有原始轨迹的哈希随protocol与各日封存记录保存。
