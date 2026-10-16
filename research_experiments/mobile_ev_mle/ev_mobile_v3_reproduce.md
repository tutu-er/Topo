# v3复算与续接

本轮源码分支：codex/ev-train-validation-design；基于完成的v2研究，新增v3文件。执行副本为D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015。原115个科学文件、旧校准阈值和报告PDF不改。Python运行环境为D:/apps/miniconda3/envs/Topo/python.exe；每个数值进程的OPENBLAS_NUM_THREADS、OMP_NUM_THREADS、MKL_NUM_THREADS、NUMEXPR_NUM_THREADS均为1。

## 三层数据

1. 旧96评价、11条代表/解释例及52条路线审查：仅诊断使用，入口ev_mobile_v3_diagnostic_results.md。三因素误差、固定/共享损失恒等式及新投影排序分别保存，未合成因果总量。
2. development1：4条开发历史，原共享投影上界MILP，3秒均保留父计划。源码为Git提交02fcc52c对应字节；D输出outputs/mobile_v3_development1含freeze、四原gzip及物理审计。该阶段保留失败的smoke、未完成代理间隙。
3. pilot2：先增加切线MILP初值及单EV/吞吐范围收紧，四条开发通过后保持算法不变，评价16条新历史。两库均T12、H8/96，4候选、4真结构，种子全部在freeze.json中。每条历史内部各方案配对，开发与评价分开汇总。policy_id为8a96cdca57afb8c2a9c71c53b7326847dc7cace7b7068b0036221616c29cd296。

主指标M为共享线路完整AC后最小错误SSE−真实SSE。M_AIC为加2p后的同类差，p=2×边数，作次级描述。新pilot未重新校准，原v2阈值不挪到新策略；误拒控制未在本pilot声明。

## 运行顺序

在执行副本根目录，以新输出目录运行下列模块；已有freeze不一致时程序停止，不能覆盖旧批次。先验证科学源码哈希，再执行开发、独立回放，最后评价：

```powershell
python -m research_experiments.mobile_ev_mle.ev_mobile_v3_study --output outputs/mobile_v3_pilot2 --freeze-only
python -m research_experiments.mobile_ev_mle.ev_mobile_v3_study --output outputs/mobile_v3_pilot2 --phase development
python -m research_experiments.mobile_ev_mle.ev_mobile_v3_physics_audit --directory outputs/mobile_v3_pilot2 --phase development --output outputs/mobile_v3_pilot2/physics_dev_v1.json
python -m research_experiments.mobile_ev_mle.ev_mobile_v3_study --output outputs/mobile_v3_pilot2 --phase evaluation
python -m research_experiments.mobile_ev_mle.ev_mobile_v3_summary --root outputs/mobile_v3_pilot2 --phase all --require-complete --output outputs/mobile_v3_pilot2/summary_final_v1.json
python -m research_experiments.mobile_ev_mle.ev_mobile_v3_physics_audit --directory outputs/mobile_v3_pilot2 --phase evaluation --output outputs/mobile_v3_pilot2/physics_eval_v1.json
```

后台worker对互不相交的--indices列表分工，启动前检查PID；Windows Start-Process使用WindowStyle Hidden，并保留stdout、stderr、exitcode和PID。无自动循环新批次。

完整科学源码字节随source_snapshot.json.gz保存，避免Windows换行转换使源码哈希不同。恢复到新目录时，rnj_wzzt_core/前缀放根目录，其余源放research_experiments/mobile_ev_mle/；先核对每份SHA256。分段MILP有时间限制，硬件与进程调度会改变取得的计划；保存的实际动作、边界及拟合状态是本轮证据，重新运行的不同结果需另报。

## 审核范围

物理回放使用独立branch-current前推回代和计量/SSE核算；它检验保存的状态、线路/源域、SOC、路线及EV单次注入，未证明非线性参数拟合全局最优。代理audit单列自由最小二乘解释的R/X越界和代理/AC分数变化。

最终证据保留每条历史及失败分母、源码/配置/种子、全部动作/路线、最终拟合、规划界和时间。重复动作按action_key去重可降低存储，完整raw仍留D-stage；compact恢复后的内容要通过相同汇总和物理回放，剔除字段及原raw哈希另列。新策略及其证据仅存研究目录，原PDF/pyproject修改保持原样。

下一轮数学修订见ev_mobile_v3_next_model.md，尚未接入本轮策略。它恢复竞争QP物理参数域，以合法解释及支持LP分别构造MILP上、下界，不把本轮自由投影结果写成已解决模型偏差。
