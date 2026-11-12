# 四种RNJ／WZZT方法：完整测算与分析

480/480条方法—输入记录已齐并核验。主108输入平均F1最高为少量RNJ小块＋原始WZZT步骤＋末端枚举＋退出，F1=0.829183、完整恢复59/108。
264条沿用已完成且逐字节核验的300秒记录，216条为本轮新增求解；没有把复用写成重跑。四网络、观测、数据划分和最终评价一致，不重新调参。

## 1. 方法和预算

普通版采用此前确认的近期完整RNJ初值，关闭预联合、末端联合和退出，使用原无计数惩罚的逐步扩展；不是生产bootstrap最多两个边界块的配置。其余三项定义见METHODS.md。
总结构预算300秒，单次基础MILP30秒、固定家族LP30秒；扩展版前缀180秒，剩余预算用于退出，完整RNJ预联合最多60秒。普通版全部300秒供单候选搜索使用。共同前端和最终选模LP另计；worker硬上限900秒。
扩展版训练目标为MAE＋0.003 f0×内部连接数，普通版仅为MAE；最终均经0.24有效深度收缩、重新训练拟合和严格验证选择，另报0.16。故本轮评估整个方法包，不是只改变一个模块的消融。

## 2. 主108输入与压力12输入

| method | inputs | F1 | exact | train | validation | test |
|---|---|---|---|---|---|---|
| full_rnj_exit | 108 | 0.820131 | 61 | 0.00128081 | 0.00138922 | 0.00135349 |
| ordinary | 108 | 0.753671 | 49 | 0.00128731 | 0.0013983 | 0.00136257 |
| small3_coarse_exit | 108 | 0.791987 | 57 | 0.00128346 | 0.001393 | 0.00135794 |
| small3_original_exit | 108 | 0.829183 | 59 | 0.00128121 | 0.00138915 | 0.00135334 |

| method | inputs | F1 | exact | train | validation | test |
|---|---|---|---|---|---|---|
| full_rnj_exit | 12 | 0.694444 | 7 | 0.00136693 | 0.0015101 | 0.00146324 |
| ordinary | 12 | 0.686508 | 6 | 0.0013669 | 0.00150981 | 0.00146326 |
| small3_coarse_exit | 12 | 0.694444 | 7 | 0.0013669 | 0.00150918 | 0.00146291 |
| small3_original_exit | 12 | 0.686508 | 6 | 0.00136706 | 0.00151047 | 0.00146356 |

完整恢复数的分母分别为108和12；每个输入权重相同，失败不得移出分母。F1按终端内部集合族计算，完整恢复要求集合族完全一致。

合并全120输入的描述性均值：

| method | inputs | F1 | exact | train | validation | test |
|---|---|---|---|---|---|---|
| full_rnj_exit | 120 | 0.807562 | 68 | 0.00128942 | 0.00140131 | 0.00136446 |
| ordinary | 120 | 0.746955 | 55 | 0.00129527 | 0.00140945 | 0.00137264 |
| small3_coarse_exit | 120 | 0.782232 | 64 | 0.0012918 | 0.00140461 | 0.00136844 |
| small3_original_exit | 120 | 0.814915 | 65 | 0.0012898 | 0.00140129 | 0.00136437 |

## 3. 四个算例分别比较

主面板每个网络27输入。平均F1：

| case | full_rnj_exit | ordinary | small3_coarse_exit | small3_original_exit |
|---|---|---|---|---|
| flynn16 | 0.755806 | 0.658849 | 0.673741 | 0.73319 |
| paper15 | 0.844322 | 0.800786 | 0.821321 | 0.841591 |
| pengwah18 | 0.883394 | 0.797555 | 0.847076 | 0.899064 |
| soumalas11 | 0.797002 | 0.757496 | 0.825808 | 0.842887 |

对应完整恢复数（每项分母27）：

| case | full_rnj_exit | ordinary | small3_coarse_exit | small3_original_exit |
|---|---|---|---|---|
| flynn16 | 12 | 9 | 10 | 11 |
| paper15 | 15 | 13 | 15 | 15 |
| pengwah18 | 17 | 15 | 15 | 16 |
| soumalas11 | 17 | 12 | 17 | 17 |

逐噪声／样本数条件见conditions.csv；本批包含N24/48/96和三档电压噪声，不能只取较容易的一档代表全部。

## 4. 配对胜负与原24输入的推广

| reference | method | inputs | delta | wins | ties | losses | new_exact | lost_exact |
|---|---|---|---|---|---|---|---|---|
| ordinary | full_rnj_exit | 108 | 0.06646 | 33 | 72 | 3 | 13 | 1 |
| ordinary | small3_coarse_exit | 108 | 0.0383155 | 25 | 78 | 5 | 9 | 1 |
| ordinary | small3_original_exit | 108 | 0.0755117 | 33 | 72 | 3 | 11 | 1 |
| full_rnj_exit | small3_coarse_exit | 108 | -0.0281445 | 9 | 78 | 21 | 1 | 5 |
| full_rnj_exit | small3_original_exit | 108 | 0.0090517 | 14 | 84 | 10 | 2 | 4 |
| small3_coarse_exit | small3_original_exit | 108 | 0.0371962 | 18 | 85 | 5 | 3 | 1 |

delta为method减reference；new_exact和lost_exact分别为新增和损失的完整恢复输入数。

| scope | method | inputs | F1 | exact | train | validation | test |
|---|---|---|---|---|---|---|---|
| new_main84 | full_rnj_exit | 84 | 0.806473 | 47 | 0.0013061 | 0.00139381 | 0.00135697 |
| new_main84 | ordinary | 84 | 0.744872 | 36 | 0.00131201 | 0.00140251 | 0.00136529 |
| new_main84 | small3_coarse_exit | 84 | 0.782845 | 43 | 0.0013073 | 0.00139644 | 0.00136004 |
| new_main84 | small3_original_exit | 84 | 0.807117 | 45 | 0.00130663 | 0.0013941 | 0.00135692 |
| new_stress12 | full_rnj_exit | 12 | 0.694444 | 7 | 0.00136693 | 0.0015101 | 0.00146324 |
| new_stress12 | ordinary | 12 | 0.686508 | 6 | 0.0013669 | 0.00150981 | 0.00146326 |
| new_stress12 | small3_coarse_exit | 12 | 0.694444 | 7 | 0.0013669 | 0.00150918 | 0.00146291 |
| new_stress12 | small3_original_exit | 12 | 0.686508 | 6 | 0.00136706 | 0.00151047 | 0.00146356 |
| prior_main24 | full_rnj_exit | 24 | 0.867935 | 14 | 0.00119232 | 0.00137314 | 0.00134129 |
| prior_main24 | ordinary | 24 | 0.784469 | 13 | 0.00120088 | 0.00138359 | 0.00135303 |
| prior_main24 | small3_coarse_exit | 24 | 0.823983 | 14 | 0.0012 | 0.00138094 | 0.00135061 |
| prior_main24 | small3_original_exit | 24 | 0.906412 | 14 | 0.00119226 | 0.00137186 | 0.00134082 |

prior_main24是前轮机制对照；new_main84和new_stress12是原始WZZT组合本轮补齐的输入。三个范围难度不同，不将范围间均值变化解释成同一输入上的算法退化；在每个范围内比较相同输入上的方法差异。参数始终冻结，没有根据这些F1再选择small3、惩罚或收缩阈值。

## 5. 收益来自哪里

| view | method | inputs | F1 | exact | train | validation | test |
|---|---|---|---|---|---|---|---|
| path | full_rnj_exit | 108 | 0.814558 | 60 | 0.0012829 | 0.00139411 | 0.00135817 |
| path | ordinary | 108 | 0.723577 | 45 | 0.00129842 | 0.00141492 | 0.00137804 |
| path | small3_coarse_exit | 108 | 0.793731 | 51 | 0.00130821 | 0.00142132 | 0.00138477 |
| path | small3_original_exit | 108 | 0.83114 | 54 | 0.00129475 | 0.00140384 | 0.00136823 |
| prefix_bank | full_rnj_exit | 108 | 0.791057 | 59 | 0.00128479 | 0.00139392 | 0.00135832 |
| prefix_bank | ordinary | 108 | 0.753671 | 49 | 0.00128731 | 0.0013983 | 0.00136257 |
| prefix_bank | small3_coarse_exit | 108 | 0.754137 | 49 | 0.001287 | 0.0013987 | 0.00136376 |
| prefix_bank | small3_original_exit | 108 | 0.779121 | 54 | 0.0012868 | 0.00139716 | 0.00136165 |
| prefix_path | full_rnj_exit | 108 | 0.780085 | 58 | 0.00129106 | 0.00140271 | 0.00136662 |
| prefix_path | ordinary | 108 | 0.723577 | 45 | 0.00129842 | 0.00141492 | 0.00137804 |
| prefix_path | small3_coarse_exit | 108 | 0.547906 | 13 | 0.00161583 | 0.00173814 | 0.00170948 |
| prefix_path | small3_original_exit | 108 | 0.663377 | 38 | 0.00149322 | 0.00160961 | 0.00158105 |

| panel | resolution | method | origin | selected_stage | inputs |
|---|---|---|---|---|---|
| main | 0.24 | full_rnj_exit | complete_rnj | complete_rnj | 65 |
| main | 0.24 | full_rnj_exit | path | exit | 16 |
| main | 0.24 | full_rnj_exit | path | initial | 2 |
| main | 0.24 | full_rnj_exit | path | rnj_pre_joint | 21 |
| main | 0.24 | full_rnj_exit | path | wzzt_extension | 2 |
| main | 0.24 | full_rnj_exit | path | wzzt_tail_joint | 2 |
| main | 0.24 | ordinary | complete_rnj | complete_rnj | 98 |
| main | 0.24 | ordinary | path | initial | 2 |
| main | 0.24 | ordinary | path | wzzt_extension | 8 |
| main | 0.24 | small3_coarse_exit | complete_rnj | complete_rnj | 78 |
| main | 0.24 | small3_coarse_exit | path | coarse_wzzt | 2 |
| main | 0.24 | small3_coarse_exit | path | exit | 24 |
| main | 0.24 | small3_coarse_exit | path | joint_tail | 4 |
| main | 0.24 | small3_original_exit | complete_rnj | complete_rnj | 70 |
| main | 0.24 | small3_original_exit | path | exit | 21 |
| main | 0.24 | small3_original_exit | path | joint_tail | 11 |
| main | 0.24 | small3_original_exit | path | wzzt_extension | 6 |

最终库包含公共完整RNJ候选与各自路径；complete_rnj表示选择公共RNJ，并非该方法新发现了这个结构。自身路径仍可能包含退出热启动带来的RNJ结构。prefix视角排除了退出阶段，但保留其前缀算法；差值是同一流程的阶段诊断，不自动等于单个模块的独立因果贡献。

## 6. 大块和小块错误

| method | size | TP | FP | FN |
|---|---|---|---|---|
| full_rnj_exit | large | 76 | 11 | 5 |
| full_rnj_exit | small | 436 | 102 | 104 |
| ordinary | large | 70 | 13 | 11 |
| ordinary | small | 402 | 159 | 138 |
| small3_coarse_exit | large | 72 | 15 | 9 |
| small3_coarse_exit | small | 422 | 126 | 118 |
| small3_original_exit | large | 76 | 8 | 5 |
| small3_original_exit | small | 438 | 98 | 102 |

small按终端集合大小≤5定义，与触发联合求解的直属块数≤5不同。假集合若与真集合交叉，会阻挡前向补全；相容的多余细分则可能由收缩删除。上表只统计最终结构，过程中的种子和阻塞需要结合events.csv及原始记录。

## 7. 时间、参与率和求解状态

| panel | method | inputs | returned | basic_inputs | basic_MIPs | basic_timeouts | tail_inputs | exit_gap_met | exit_timeouts | prefix_seconds | exit_seconds | elapsed_seconds | pool_count | all_true_in_exit_pool |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | full_rnj_exit | 108 | 108 | 4 | 4 | 0 | 108 | 89 | 19 | 24.468 | 55.738 | 99.513 | 122.444 | 89 |
| main | ordinary | 108 | 108 | 108 | 257 | 0 | 0 | 0 | 0 | 10.2373 | 0 | 10.546 | unavailable | 0 |
| main | small3_coarse_exit | 108 | 108 | 108 | 297 | 68 | 99 | 58 | 50 | 48.7173 | 175.496 | 260.142 | 246.037 | 94 |
| main | small3_original_exit | 108 | 108 | 108 | 331 | 15 | 97 | 81 | 27 | 23.7734 | 60.3611 | 121.634 | 240.528 | 96 |
| stress | full_rnj_exit | 12 | 12 | 0 | 0 | 0 | 12 | 10 | 2 | 9.33837 | 35.8014 | 51.4016 | 159 | 8 |
| stress | ordinary | 12 | 12 | 12 | 23 | 0 | 0 | 0 | 0 | 5.13205 | 0 | 5.36871 | unavailable | 0 |
| stress | small3_coarse_exit | 12 | 12 | 12 | 32 | 0 | 10 | 9 | 3 | 23.5527 | 110.271 | 135.258 | 282.333 | 9 |
| stress | small3_original_exit | 12 | 12 | 12 | 34 | 0 | 9 | 10 | 2 | 22.7978 | 125.504 | 151.245 | 274.333 | 9 |

basic_inputs只统计实际启动基础MILP的输入，排除了“调用扩展函数后发现已无相容候选”的情况。退出达到设定gap只适用于其候选域；超时记录与可行解全部保留。普通方法没有退出，相关列不构成其失败率。不同退出池和初值同时影响速度，不能将所有耗时变化归为求解器性能。
per_model.csv中普通版的candidate_pool_true_recall只指公共前端池的覆盖，不代表其自由单步MILP的全部可选集合；普通版没有显式退出池，events.csv对应列为空。

## 8. 收缩敏感性与证据边界

| method | inputs | F1 | exact | train | validation | test |
|---|---|---|---|---|---|---|
| full_rnj_exit | 108 | 0.814651 | 55 | 0.00127696 | 0.00138471 | 0.00134952 |
| ordinary | 108 | 0.754474 | 44 | 0.0012819 | 0.00139215 | 0.00135716 |
| small3_coarse_exit | 108 | 0.793245 | 51 | 0.00127909 | 0.00138807 | 0.00135269 |
| small3_original_exit | 108 | 0.822247 | 54 | 0.00127703 | 0.00138443 | 0.00134896 |

数值审查37277项通过；固定家族核验共1199个，新增独立HiGHS LP 321次，复用同输入已核验LP 878个。旧交付66文件哈希保持不变。相关行为测试15项通过。
这些结果覆盖固定的论文系列合成网络、有限噪声和样本设置。训练目标降低、计数惩罚和求解gap分别反映拟合、复杂度和数值搜索，均不能替代拓扑F1与完整恢复。更高平均F1与更多完整恢复也应分别判断。

源代码、冻结协议、逐输入记录、原始分析器、独立LP回执及旧源码恢复映射见records.zip。每条复用记录保留原协议哈希，当前目录的manifest.json用于校验最终交付。
