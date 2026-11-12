# 多区域联合加入退出机制及末端信息诊断

结论、参数取舍和小块原因分析见 [ANALYSIS.md](ANALYSIS.md)。本页为程序生成的完整汇总。

开发16输入、6种退出配置；冻结{'policy': 'compact', 'coefficient': 0.003}和{'rule': 'strict', 'resolution': 0.24}。正式120输入全部返回，独立重解455个家族，核查6376项。

## 主面板及压力面板

| 面板 | 共同选模口径 | 方法 | F1 | 完整恢复 | 平均内部连接数 |
|---|---|---|---:|---:|---:|
| main | matched | baseline20 | 0.7879 | 54/108 | 6.241 |
| main | matched | baseline40 | 0.7894 | 54/108 | 6.241 |
| main | matched | exit20 | 0.7948 | 54/108 | 6.167 |
| main | matched | exit40 | 0.8014 | 54/108 | 6.176 |
| main | matched | historical_joint20 | 0.7813 | 51/108 | 6.287 |
| main | tuned | baseline20 | 0.7847 | 59/108 | 5.843 |
| main | tuned | baseline40 | 0.7861 | 59/108 | 5.843 |
| main | tuned | exit20 | 0.7873 | 59/108 | 5.833 |
| main | tuned | exit40 | 0.7958 | 60/108 | 5.796 |
| stress | matched | baseline20 | 0.6788 | 5/12 | 6.000 |
| stress | matched | baseline40 | 0.6788 | 5/12 | 6.000 |
| stress | matched | exit20 | 0.6788 | 5/12 | 6.000 |
| stress | matched | exit40 | 0.6788 | 5/12 | 6.000 |
| stress | matched | historical_joint20 | 0.6531 | 3/12 | 6.000 |
| stress | tuned | baseline20 | 0.6944 | 7/12 | 5.583 |
| stress | tuned | baseline40 | 0.6944 | 7/12 | 5.583 |
| stress | tuned | exit20 | 0.6944 | 7/12 | 5.583 |
| stress | tuned | exit40 | 0.6944 | 7/12 | 5.583 |

matched对所有方法统一使用严格最小验证MAE及0.16收缩。tuned对所有方法统一使用冻结选模规则和阈值，不能把仅改变后处理的收益算成退出搜索收益。历史行仅作同输入参照；新的baseline20和baseline40均重新运行。

## 时间和扫描消融

| method | inputs | F1 | exact | seconds | attempts | local_solves | timeouts | accepted | train | endpoint_F1 | path_F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| joint_20 | 8 | 0.762821 | 4 | 8.91685 | 2 | 0 | 3 | 9 | 0.0012989 | 0.728766 | 0.762821 |
| joint_40 | 8 | 0.762821 | 4 | 8.06442 | 2 | 0 | 4 | 9 | 0.001299 | 0.736058 | 0.762821 |
| regional20_2 | 8 | 0.714286 | 4 | 11.47 | 15.625 | 13.875 | 30 | 19 | 0.00129831 | 0.662683 | 0.714286 |
| regional40_2 | 8 | 0.714286 | 4 | 11.3588 | 16.5 | 14.875 | 31 | 19 | 0.00129831 | 0.662683 | 0.714286 |
| regional40_5 | 8 | 0.735119 | 4 | 19.5037 | 19.625 | 17.75 | 18 | 21 | 0.00129562 | 0.701145 | 0.735119 |

8个预设开发输入，同一lambda=0.003。regional40_2相对regional20_2改变总预算；regional40_5在同为40秒时改变单次MIP时间，允许求解深度与覆盖数量发生取舍。均使用严格0.16共同选模，没有根据这些F1重新调参。

## 退出模型与预算解释

先运行上轮多区域联合20秒，再将临时内部支撑和候选池放入同一个MILP重选。候选池包含终点处重建的旧联合域、初值、终点与原RNJ池；此前每个已完成前缀树另在最终选模库保留。它不保证包含所有历史轮次中未入选的联合候选。全部响应系数联合重新拟合；单点和全集基础项保留，已核实物理先验才强制保留，本批无额外先验。未增加正阻抗下界。
20/40秒是同一次池MILP的壁钟检查点，只使用该时刻已经找到的可行结构及有效界。连续家族闭合重拟合与验证选模作为所有方法的共同后处理，不计为额外结构搜索，其效果单独核查。20秒前缀已用尽预算的输入，20秒增强版本保留前缀。原20秒算法可能有小幅软预算超出。40秒原方法使用新LP bank重新运行，没有读取退出候选。
solver_events的pooled_exit行中final_solver_status属于整个池求解结束时；20秒输出只按它自己的incumbent_at和bound解释，不把后来的最优状态用作较早时刻证明。
前缀使用历史已冻结的lambda=0.003，退出阶段lambda由开发观测选择。只对退出阶段的同一个目标核查下降，不声称不同惩罚目标跨阶段单调。求解上下界只针对明确候选池，不覆盖所有可能树。旧单区域方法与新池选择的域也不等同。

## 末端信息

| panel | regime | method | regions | F1 | exact | wrong_beats_true | near_zero_linear_gap |
|---|---|---|---|---|---|---|---|
| main | clean_ac | paired | 378 | 0.922222 | 276 | 255 | 0 |
| main | clean_ac | regularized | 378 | 1 | 378 | 255 | 0 |
| main | clean_ac | train | 378 | 0.64418 | 3 | 255 | 0 |
| main | clean_ac | validation | 378 | 0.701058 | 36 | 255 | 0 |
| main | linear | paired | 378 | 1 | 378 | 0 | 0 |
| main | linear | regularized | 378 | 1 | 378 | 0 | 0 |
| main | linear | train | 378 | 1 | 378 | 0 | 0 |
| main | linear | validation | 378 | 1 | 378 | 0 | 0 |
| main | noisy | paired | 378 | 0.869489 | 264 | 225 | 0 |
| main | noisy | regularized | 378 | 0.852734 | 282 | 225 | 0 |
| main | noisy | train | 378 | 0.614462 | 3 | 225 | 0 |
| main | noisy | validation | 378 | 0.76261 | 129 | 225 | 0 |
| stress | clean_ac | paired | 42 | 0.907937 | 29 | 27 | 0 |
| stress | clean_ac | regularized | 42 | 1 | 42 | 27 | 0 |
| stress | clean_ac | train | 42 | 0.640476 | 0 | 27 | 0 |
| stress | clean_ac | validation | 42 | 0.716667 | 6 | 27 | 0 |
| stress | linear | paired | 42 | 1 | 42 | 0 | 0 |
| stress | linear | regularized | 42 | 1 | 42 | 0 | 0 |
| stress | linear | train | 42 | 1 | 42 | 0 | 0 |
| stress | linear | validation | 42 | 1 | 42 | 0 | 0 |
| stress | noisy | paired | 42 | 0.88254 | 33 | 18 | 0 |
| stress | noisy | regularized | 42 | 0.734127 | 29 | 18 | 0 |
| stress | noisy | train | 42 | 0.60873 | 0 | 18 | 0 |
| stress | noisy | validation | 42 | 0.694444 | 11 | 18 | 0 |

诊断已知真实外部结构，完整枚举至多5终端内的全部局部层级，包含区域根边可有可无。它是oracle条件性诊断，不是部署结果，也没有参与开发选择。linear用干净P/Q与真实R/X合成精确线性响应；clean_ac用干净AC量测；noisy用实际带噪量测。
本批linear的420个区域全部恢复正确，错误有效家族无近零剖面间隙；这支持该oracle局部线性问题在本批输入上可区分。clean_ac相对linear的下降体现线性响应模型偏差；noisy进一步叠加量测噪声和有限样本影响。投影后信号与残差比仅是几何诊断，residual_RMS为真实物理R/X的原始预测误差，不是重拟合后的噪声标准差。
判断理想线性可辨识性看未正则的train剖面与错误家族间隙。regularized诊断沿用带噪初始RNJ的惩罚尺度，paired诊断也有简约偏好；它们在理想无噪声下舍弃弱边属于决策规则的偏好，不能算成信息不足。
精确反例：P每列都是同一个时间序列，Q=0，R_true=0.1I+0.2 z_12 z_12^T；星形R_star=diag(0.5,0.5,0.1)。两者对该P产生完全相同的Y，且系数都在0至2内。改用P=I时两者响应不同。因此有条件性的固有信息不足，但不能据此断言本批所有末端线都不可辨识。

## 范围与复现

四个网络为仓库论文启发的合成算例；108主条件与12压力条件复用既有归档，不视作120个独立随机拓扑。正式矩阵未参与本次调参，但此前已分析过，不称新盲测。生产默认未修改。原单区域20秒的历史结果在上一报告中，本批时间消融仅覆盖预设8输入。

```powershell
python -m research_experiments.joint_exit.run freeze outputs/joint_exit_NEW
python -m research_experiments.joint_exit.run ablation outputs/joint_exit_NEW
python -m research_experiments.joint_exit.run development outputs/joint_exit_NEW
python -m research_experiments.joint_exit.run select outputs/joint_exit_NEW
python -m research_experiments.joint_exit.run evaluation outputs/joint_exit_NEW
python -m research_experiments.joint_exit.run information outputs/joint_exit_NEW
python -m research_experiments.joint_exit.run analyze outputs/joint_exit_NEW
python -m research_experiments.joint_exit.diagnostics outputs/joint_exit_NEW --verify-information
python -m research_experiments.joint_exit.run finalize outputs/joint_exit_NEW
```
