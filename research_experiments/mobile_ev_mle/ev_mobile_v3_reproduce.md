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

## 完成后的证据索引

简明结果为ev_mobile_v3_results.md，续接点为ev_mobile_v3_checkpoint.json和ev_mobile_v3_resume.md。最终研究证据存于evidence/ev_mobile_v3/final_20261016/：

- pilot2_compact/保留4开发+16评价精简记录、freeze、122文件原字节snapshot、完整最终拟合、规划界/PWL范围、失败、物理审计和工作进程日志。manifest.json逐文件记录原/精简哈希；精简恢复不是被删除缓存的完整raw恢复。
- audits/保留最终127项测试XML、全部精简恢复摘要核查、旧开发回放补充、费用/配对统计，以及三事件完整评分归因的独立验证。
- score_attribution/保留3事件H/验证输入、48未来输入和384套固定/联合解释的556KB小证据包。真实因素替换只在离线诊断，不进入选择器。
- tools/保留本轮只读汇总与发布脚本。最终manifest列源文件、证据文件及保护文件SHA256；Git本身记录对应提交和推送。

新16评价完整汇总为pilot2_compact/summary_final_v2.json。costs_and_pairs_v1.json在audits中，逐历史费用和参数误差按原分母保存。恢复示例见pilot2_compact/attachments/mobile_v3_evidence_usage_v1.md；ev_mobile_v3_evidence.restore把去重的最终evaluation恢复给既有汇总/物理审计器。原raw仍保留执行副本，不能用新版本覆盖旧试验记录。

最终测试命令：

```powershell
python -c "from pathlib import Path;import pytest;files=sorted(str(p) for p in Path('research_experiments/mobile_ev_mle').glob('test_ev_mobile_v3_*.py'));raise SystemExit(pytest.main(files+['-q']))"
```

它验证本轮v3实现与解析性质；旧v2科学源码不因本轮结果改写。本轮未修改或重编译原LaTeX/PDF报告。
