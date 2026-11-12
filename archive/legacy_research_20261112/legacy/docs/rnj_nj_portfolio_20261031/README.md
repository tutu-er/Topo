# NJ基线与完整树验证组合

完整结论和组合数学原理见[CONCLUSIONS.md](CONCLUSIONS.md)，结果前规则见[PLAN.md](PLAN.md)。

同一120输入；NJ和选择新增120作业，无新结构MILP。旧四方法按原记录对照。
NJ_native保留固定0.16几何拓扑；NJ_matched使用相同0.24/0.16响应收缩与L1重拟合。组合只选完整预测树；strict为主规则，paired为既有一标准误差敏感性。

## 统一0.24口径

| panel | resolution | method | inputs | F1 | exact | internal | train | validation | test |
|---|---|---|---|---|---|---|---|---|---|
| main | 0.24 | NJ_matched | 108 | 0.653409 | 37 | 5.40741 | 0.00138257 | 0.00150906 | 0.00147087 |
| main | 0.24 | NJ_native | 108 | 0.614796 | 17 | 7.25 | 0.00131026 | 0.00142841 | 0.00139372 |
| main | 0.24 | full_rnj_exit | 108 | 0.820131 | 61 | 5.78704 | 0.00128081 | 0.00138922 | 0.00135349 |
| main | 0.24 | ordinary | 108 | 0.753671 | 49 | 5.96296 | 0.00128731 | 0.0013983 | 0.00136257 |
| main | 0.24 | small3_coarse_exit | 108 | 0.791987 | 57 | 5.87963 | 0.00128346 | 0.001393 | 0.00135794 |
| main | 0.24 | small3_original_exit | 108 | 0.829183 | 59 | 5.74074 | 0.00128121 | 0.00138915 | 0.00135334 |
| main | 0.24 | three_paired | 108 | 0.84414 | 63 | 5.71296 | 0.00128014 | 0.00138724 | 0.00135183 |
| main | 0.24 | three_strict | 108 | 0.842737 | 63 | 5.73148 | 0.00128011 | 0.0013872 | 0.00135162 |
| main | 0.24 | two_paired | 108 | 0.84414 | 63 | 5.71296 | 0.00128014 | 0.00138724 | 0.00135183 |
| main | 0.24 | two_strict | 108 | 0.842737 | 63 | 5.73148 | 0.00128011 | 0.0013872 | 0.00135162 |
| stress | 0.24 | NJ_matched | 12 | 0.531838 | 1 | 5.16667 | 0.00145577 | 0.00162467 | 0.00157164 |
| stress | 0.24 | NJ_native | 12 | 0.446958 | 0 | 7.83333 | 0.00138756 | 0.00153501 | 0.00148654 |
| stress | 0.24 | full_rnj_exit | 12 | 0.694444 | 7 | 5.58333 | 0.00136693 | 0.0015101 | 0.00146324 |
| stress | 0.24 | ordinary | 12 | 0.686508 | 6 | 5.5 | 0.0013669 | 0.00150981 | 0.00146326 |
| stress | 0.24 | small3_coarse_exit | 12 | 0.694444 | 7 | 5.58333 | 0.0013669 | 0.00150918 | 0.00146291 |
| stress | 0.24 | small3_original_exit | 12 | 0.686508 | 6 | 5.5 | 0.00136706 | 0.00151047 | 0.00146356 |
| stress | 0.24 | three_paired | 12 | 0.694444 | 7 | 5.58333 | 0.00136693 | 0.0015101 | 0.00146324 |
| stress | 0.24 | three_strict | 12 | 0.694444 | 7 | 5.58333 | 0.00136693 | 0.0015101 | 0.00146324 |
| stress | 0.24 | two_paired | 12 | 0.694444 | 7 | 5.58333 | 0.00136693 | 0.0015101 | 0.00146324 |
| stress | 0.24 | two_strict | 12 | 0.694444 | 7 | 5.58333 | 0.00136693 | 0.0015101 | 0.00146324 |

## 有限候选事后互补性参照

| panel | resolution | domain | inputs | F1 | exact |
|---|---|---|---|---|---|
| main | 0.16 | three | 108 | 0.843472 | 61 |
| main | 0.16 | two | 108 | 0.840939 | 59 |
| main | 0.24 | three | 108 | 0.851842 | 64 |
| main | 0.24 | two | 108 | 0.848769 | 63 |
| stress | 0.16 | three | 12 | 0.721764 | 7 |
| stress | 0.16 | two | 12 | 0.712505 | 6 |
| stress | 0.24 | three | 12 | 0.700397 | 7 |
| stress | 0.24 | two | 12 | 0.700397 | 7 |

该参照使用真拓扑逐输入取最好候选，未参与部署选择，不是可实现方法或全局拓扑界。

## 选中来源

| panel | resolution | method | selected_source | inputs |
|---|---|---|---|---|
| main | 0.24 | NJ_matched | NJ_matched | 108 |
| main | 0.24 | NJ_native | NJ_native | 108 |
| main | 0.24 | three_paired | NJ_matched | 10 |
| main | 0.24 | three_paired | full_rnj_exit | 71 |
| main | 0.24 | three_paired | small3_original_exit | 27 |
| main | 0.24 | three_strict | NJ_matched | 10 |
| main | 0.24 | three_strict | full_rnj_exit | 73 |
| main | 0.24 | three_strict | small3_original_exit | 25 |
| main | 0.24 | two_paired | full_rnj_exit | 80 |
| main | 0.24 | two_paired | small3_original_exit | 28 |
| main | 0.24 | two_strict | full_rnj_exit | 82 |
| main | 0.24 | two_strict | small3_original_exit | 26 |
| stress | 0.24 | NJ_matched | NJ_matched | 12 |
| stress | 0.24 | NJ_native | NJ_native | 12 |
| stress | 0.24 | three_paired | full_rnj_exit | 12 |
| stress | 0.24 | three_strict | full_rnj_exit | 12 |
| stress | 0.24 | two_paired | full_rnj_exit | 12 |
| stress | 0.24 | two_strict | full_rnj_exit | 12 |

## 计算成本

| panel | inputs | marginal_seconds | recorded_two_pipeline_seconds | new_structural_MILP |
|---|---|---|---|---|
| main | 108 | 0.159374 | 228.216 | 0 |
| stress | 12 | 0.153726 | 198.148 | 0 |

marginal_seconds为共用前端后的NJ解码、固定LP与全部选择的实测额外耗时；recorded_two_pipeline_seconds为两条原搜索流程的记录耗时之和，不是新测端到端组合时间。部署需运行两条结构路径，顺序结构上限300＋300秒，不能视为同一300秒预算。
