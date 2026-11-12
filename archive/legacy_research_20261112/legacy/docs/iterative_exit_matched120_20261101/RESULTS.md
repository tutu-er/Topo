# 同120输入：迭代零块释放与一次停滞恢复

240/240新求解完成；原300秒结果仅作预算及精度放宽的参照。每方法/输入总账上限4小时，共享RNJ前端成本计入；新版分支共享预算。所有MILP相对gap统一0.0001，单线程；参数未重调。算法内计时含数据准备、求解和验证，不含Python启动/导入；整批并行墙钟另见batch_timing.json。

## 最终结果

| scope | method | inputs | F1 | exact | exact_rate | total_seconds | fresh_seconds | median_seconds | max_seconds |
|---|---|---|---|---|---|---|---|---|---|
| all | after | 120 | 0.828964 | 66 | 0.55 | 58138.3 | 57997.7 | 159.863 | 7609.98 |
| all | before | 120 | 0.825234 | 66 | 0.55 | 29544.7 | 29404.1 | 120.169 | 2874.18 |
| all | historical300 | 120 | 0.814915 | 65 | 0.541667 | 17996.2 | 17855.6 | 131.677 | 303.807 |
| main | after | 108 | 0.844793 | 60 | 0.555556 | 54307.2 | 54180.2 | 155.354 | 7609.98 |
| main | before | 108 | 0.840648 | 60 | 0.555556 | 27461.9 | 27335 | 111.958 | 2874.18 |
| main | historical300 | 108 | 0.829183 | 59 | 0.546296 | 16212 | 16085.1 | 122.875 | 303.807 |
| stress | after | 12 | 0.686508 | 6 | 0.5 | 3831.11 | 3817.49 | 174.01 | 1283.93 |
| stress | before | 12 | 0.686508 | 6 | 0.5 | 2082.79 | 2069.17 | 157.172 | 477.821 |
| stress | historical300 | 12 | 0.686508 | 6 | 0.5 | 1784.18 | 1770.56 | 153.047 | 302.364 |

## 配对变化

| scope | comparison | inputs | delta_F1 | wins | ties | losses | new_exact | lost_exact | validation_improved | topology_improved | validation_improved_F1_worse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | all_recovery_increment | 120 | 0.000257937 | 1 | 118 | 1 | 0 | 0 | 2 | 1 | 1 |
| all | budget_effect | 120 | 0.0103186 | 4 | 113 | 3 | 1 | 0 | 4 | 4 | 1 |
| all | mask_increment | 120 | 0.000257937 | 1 | 118 | 1 | 0 | 0 | 2 | 1 | 1 |
| all | matched_before_after | 120 | 0.00373016 | 2 | 116 | 2 | 0 | 0 | 3 | 2 | 1 |
| all | release_increment | 120 | 0 | 0 | 120 | 0 | 0 | 0 | 0 | 0 | 0 |
| all | zero_path_comparison | 120 | 0.00347222 | 1 | 117 | 2 | 0 | 0 | 2 | 1 | 1 |
| main | all_recovery_increment | 108 | 0.000286596 | 1 | 106 | 1 | 0 | 0 | 2 | 1 | 1 |
| main | budget_effect | 108 | 0.0114651 | 4 | 101 | 3 | 1 | 0 | 4 | 4 | 1 |
| main | mask_increment | 108 | 0.000286596 | 1 | 106 | 1 | 0 | 0 | 2 | 1 | 1 |
| main | matched_before_after | 108 | 0.00414462 | 2 | 104 | 2 | 0 | 0 | 3 | 2 | 1 |
| main | release_increment | 108 | 0 | 0 | 108 | 0 | 0 | 0 | 0 | 0 | 0 |
| main | zero_path_comparison | 108 | 0.00385802 | 1 | 105 | 2 | 0 | 0 | 2 | 1 | 1 |
| stress | all_recovery_increment | 12 | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 |
| stress | budget_effect | 12 | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 |
| stress | mask_increment | 12 | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 |
| stress | matched_before_after | 12 | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 |
| stress | release_increment | 12 | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 |
| stress | zero_path_comparison | 12 | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 |

## 机制触发

| scope | method | inputs | zero_checks | zero_release_events | zero_released_sets | zero_unique_sets | zero_released_true | zero_released_false | forward_zero_events | forward_zero_sets | stalled | deletion_triggered | release_branch_run | masked_branch_run | release_adopted | mask_adopted | deleted_true_inputs | deleted_seed_inputs | inputs_with_zero | inputs_with_forward_zero |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | after | 120 | 44 | 44 | 44 | 44 | 12 | 32 | 44 | 44 | 36 | 36 | 36 | 36 | 2 | 2 | 12 | 13 | 34 | 34 |
| all | before | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| main | after | 108 | 43 | 43 | 43 | 43 | 12 | 31 | 43 | 43 | 32 | 32 | 32 | 32 | 2 | 2 | 11 | 11 | 33 | 33 |
| main | before | 108 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| stress | after | 12 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 4 | 4 | 4 | 4 | 0 | 0 | 1 | 2 | 1 | 1 |
| stress | before | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

zero_path_comparison比较旧算法与新版已计算的baseline延续，包含零块处理及其后续路径变化；新版baseline退出时间由分支共享预算分配，因此不是独立满预算零块消融。release_increment比较零块路径与加入非屏蔽恢复分支；mask_increment再比较加入屏蔽分支，均保留相同RNJ验证保护。触发、最终采用、F1变好和新增完整恢复分别报告；验证改善不等于拓扑改善。

统一输出选择沿用旧方法：0.24收缩后联合重拟合、严格验证MAE、平分少内部集合、公共RNJ库及单点回退。native_final另存新版自身原选择，selection_harmonization列出两种选择口径差异。零块计数指新增显式清理事件；旧方法原有最终退出清理保留，其内部次数未伪记为新增前向清理。

## 核验

{
  "passed": true,
  "returned": 240,
  "paired_inputs": 120,
  "checks": 18390,
  "independent_LPs": 973,
  "source_hashes_verified": true,
  "parent_files_preserved": 30,
  "analyzer_sha256": "760ea4065a0a73a1b4a72ef5bf37501326afe2b52ad7754b3c39d7f9c324e3a2",
  "analysis_seconds": 23.373029499998665
}

求解器status与gap只对应各候选域子问题；不存在全树最优或恢复保证。全输入逐例数据、时间、阶段事件、paired变化及求解状态见CSV/JSON；观测NPZ由catalog和哈希引用。
