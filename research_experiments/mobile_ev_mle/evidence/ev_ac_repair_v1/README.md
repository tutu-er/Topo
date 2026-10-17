# 固定计划AC修正：证据说明

科学方案见 ../../ev_ac_repair_math.md，开发结果见 ../../ev_ac_repair_development_results.md，运行状态见 ../../ev_ac_repair_checkpoint.json。新主批输出根为 D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015/outputs/ev_ac_repair_frozen_v1。

## 已核对的开发证据

- preflight.json记录初始71项测试；main_freeze_preflight.json及main_freeze_tests.xml记录主批前78项测试和150份科学源码哈希。
- development_compact_equivalence.json核对8份原始记录压缩读取前后的全部统计量，原始文件字节保持不变。
- development_full_audit_v4.json核对全部8份开发的AC状态、四组预测接线、物理域、同一噪声和原始文件哈希；physical_audit_selfcheck.json保留3项故障注入。
- development_refit_cost_audit.json独立重放全部保存的拟合状态、局部初值、操作计划与费用。审计费用独立列出。
- development_score_decomposition.json按 log p(PQ,V,总表)=log p(PQ)+log p(V,总表|PQ)拆分评分；该诊断未改变规划或主批评分。
- development_physical_audit.json、development_oracle_mean_audit.json、development_probability_audit.json保留早期两样本的传播重建、真值替换诊断及独立概率核对。真值替换未进入选择器。

## 科学批次与统计口径

开发v3共8份历史，主批另有32校准和32评价。每历史两未来情景各一次计量噪声，另存无未来计量噪声结果；独立统计单位始终是历史。

同一固定计划分别采用单中心G00、仅均值修正G10、仅协方差修正G01、两者修正G11，以及独立校准协方差倍率。它们使用同一近似历史后验，未来传播使用完整AC；历史后验自身尚未改成完整非线性AC后验。

条件AC直接输入实测末端P/Q，比较固定历史线路与历史加EV共享线路重拟合。该口径忽略输入P/Q误差，按独立条件SSE阈值保留候选。后验概率与条件SSE表分开解释。局部求解值是合法解释的目标值，没有全局MLE证书。

主批同域平方计划SQMD与原贝叶斯计划BCMD使用相同电池任务及移动、充放电域；BC继承平方等父计划并有额外预算。本轮首先比较固定动作下的推断修正，规划总耗时也单列，没有宣称等总算时的规划竞赛。

失败历史与缺失计划保留原始记录。操作上必要候选推断失败则保留全部候选；完整样本错误率另报缺失范围。SSE阈值校准使用所有计划均完整的校准历史，独立评价保留计划可用范围的差异，因此校准保证限于声明模拟分布下的经验检验。

## 复算

在执行副本根目录使用 D:/apps/miniconda3/envs/Topo/python.exe。将OPENBLAS_NUM_THREADS、OMP_NUM_THREADS、MKL_NUM_THREADS设为1。

审计脚本位于 scripts/，接受 --root 与 --output，原raw仅只读。physical_audit.py可用 --expected-histories 64 审计整个主批；refit_cost_audit.py使用 --phases calibration evaluation --source research_experiments/mobile_ev_mle；statistical_audit.py的实际选项见其 --help。score_decomposition.py按 --phase 单独拆分评分。不要在统计完成前重新选择倍率或修改科学源码。

完整原始记录、freeze.json、source_snapshot.json.gz和控制器日志保留于各输出根。最终证据包提供清单与哈希；原PDF及pyproject不在本轮提交范围。
