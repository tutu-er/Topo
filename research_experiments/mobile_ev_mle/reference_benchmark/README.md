# 标准线路与公开P/Q时序测试

本轮承接已完成的full_pipeline；先读protocol.md、math.md及sources中的来源审查。来源、单位、内点表示、预测边界和稀疏AC实现已验证；新旧流程共220项测试通过。893份科学源码及配置已冻结，256条件批次正在运行；当前进度与续接见checkpoint.json和resume.md。

已逐条核对原MATPOWER case33bw和CIGRE居民子网。旧paper11/paper15只是项目自建论文风格网络，恢复原放大前阻抗并单列变压器扩展；尚未取得之前五篇论文作者最终单相等效逐线数值，具体缺口保留。

各网保留原生P/Q额定值，再乘SimBench原生LV聚合P/Q曲线。源P/Q在同一物理接口，输入与未来真值分开。SINTEF/CINELDI小时数据已取得，但原负荷CSV单位未证实，暂不猜单位用于原网AC。

新代码与证据只在本目录；旧批次、原PDF和pyproject不修改。源码分支codex/ev-train-validation-design，执行副本D:/0-github_workspace/Topo/outputs/ev_reference_stage_20261018。

## 复现入口

先设PYTHONPATH为执行副本及其rnj_wzzt_core；BLAS、OpenMP和Gurobi每worker一线程。正式输入使用inputs_v2：它仅补正MATPOWER 7.1的来源metadata，所有物理参数、选窗、预测和时序与保留的v1相同。

```powershell
python -m research_experiments.mobile_ev_mle.reference_benchmark.profiles --output outputs/inputs_fresh --profile-directory data_external/simbench/scenario0
python -m research_experiments.mobile_ev_mle.reference_benchmark.controller --output outputs/reference_v1 --prepared outputs/inputs_v2/prepared.json.gz --workers 16
python -m research_experiments.mobile_ev_mle.reference_benchmark.analysis --output outputs/reference_v1
```

第一条用于重新准备输入；逐字复现冻结批次应使用随证据保存的原prepared文件及其哈希。controller按开发、校准、评价顺序运行，在任何评价任务派发前写入校准阈值；失败条件落盘后不换种子重采样。可仅运行`--phases development`检查冻结开发阶段，再以同一目录续接。已有lock/PID运行时不可重复启动。

## 方法及适用范围

P-SQ是公开电压逆方差加权的功率平方基线。本批电压精度各时段相同，所以其权重相同，不能把收益解释为选择高精度时段。G-SQ加入候选AC导数差平方，G-AQ再加入预测背景差异。四嵌套动作域均优化；完整AC实际执行和推断比较保留固定点/可选移动的双向域，以及OBS/恒充/高低/随机对照。F+/M+内部计划不是额外已完成的物理消融。

各方法共享11kW EV、6kWh电池服务任务及同一计量预算。移动是可选项，统一旅行与接线模板尚不是真实路网。六类LV聚合曲线复用使大网普通P/Q回归列相关；按网络报告候选覆盖、最终子树F1、严格恢复、零边诊断、误拒、未决和计算失败。线路拟合使用完整AC条件输出损失，但未传播输入P/Q噪声，局部结果不称全局MLE。

本批给算法的是固定共同PCC接口，并未给出该接口与哪块电压表共点的额外映射。case33母线2、CIGRE R1的共点关系只存在于仿真真值；不能把当前模型说成已使用此资产先验。具体见evidence/development/source_interface_information_boundary.md。
