# 系统比较的复算与证据范围

先读 `ev_systematic_protocol.md`、`ev_systematic_math.md`、`ev_systematic_freeze.md`。结果入口为 `ev_systematic_results.md`。本轮不修改原LaTeX/PDF。

## 环境和科学版本

运行环境为Windows、Topo conda环境；MILP后端为Gurobi13.0.3。正式科学策略由110份源码及配置哈希固定：

`86b811f2c74d80b4c1c1b5721871b01e007a102d8c918f5c25972fcaeacd8656`。

初次完整实现提交为`2f4c2f85`。分析、独立回放和打包工具单独记录SHA，不属于动作/拟合科学策略的110份源码。后续提交中的同名源码可先与freeze核对；证据包另含实验时实际源码的base64快照及逐文件哈希。

运行前在PowerShell设置：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
```

全部命令从项目根目录运行。历史批次的冻结源码不覆盖。所有输出目录使用新路径；需要重跑时换目录，保留原结果。

## 从种子重算整个流程

以下串行命令无需手工合并分片；并行运行可以用互不重叠的`--start/--stop`和不同输出目录。完整流程包含模拟、训练、全部10策略、固定/联合推断及真实误差记录。

```powershell
python -m research_experiments.mobile_ev_mle.ev_systematic_study --phase development --start 0 --stop 64 --output outputs/reproduce_systematic_dev
python -m research_experiments.mobile_ev_mle.ev_systematic_study --phase calibration --start 0 --stop 108 --output outputs/reproduce_systematic_cal
python -m research_experiments.mobile_ev_mle.ev_systematic_study --phase evaluation --start 0 --stop 72 --output outputs/reproduce_systematic_eval
```

原执行用3个单线程worker：开发分片[0,24)、[24,44)、[44,64)，校准[0,36)、[36,72)、[72,108)，评价[0,24)、[24,48)、[48,72)。计时受机器负载影响，有限时限MILP可能返回不同incumbent；不能要求重跑动作字节相同。原始incumbent、界、失败、操作检查和AC接受均在保存记录中。

## 从保存记录重算统计

正式三个规则分别运行，任何命令都先持久保存自己的校准阈值，再首次读取评价记录。新输出目录内会保留原primary、绝对误差双门和训练长度分层三套独立结果。

```powershell
python -m research_experiments.mobile_ev_mle.ev_systematic_analyze --calibration outputs/reproduce_systematic_cal --evaluation outputs/reproduce_systematic_eval --output outputs/reproduce_systematic_primary
python -m research_experiments.mobile_ev_mle.ev_systematic_adequacy --calibration outputs/reproduce_systematic_cal --evaluation outputs/reproduce_systematic_eval --output outputs/reproduce_systematic_adequacy
python -m research_experiments.mobile_ev_mle.ev_systematic_stratified --calibration outputs/reproduce_systematic_cal --evaluation outputs/reproduce_systematic_eval --primary-thresholds outputs/reproduce_systematic_primary/thresholds.json --output outputs/reproduce_systematic_stratified
```

分析命令可以读取analysis-lite恢复文件，重新计算全部方法的阈值、逐历史决策和汇总。路径及输入包SHA是来源字段，会随恢复位置和压缩格式变化；数值分数、阈值、保留集合、分母和统计应保持相同。

## 从证据包恢复和独立核查

提交证据分两层：全部244个开发/校准/评价记录的analysis-lite，和事先选定的16个开发代表记录的完整compact。前者保留统计、曲线、操作、分数、参数误差、成本及来源，不含全部计量和拟合状态；后者保留计量与全部固定/联合拟合状态，可独立回放AC。16个代表按每个开发场景首个index=0,4,…60选取，与好坏结果无关。

原全量raw及全量compact保存在本机D-stage，不随Git提交。manifest保存每个原raw的SHA及其交付文件映射，证据包不声称能还原被删去的完整选择器中间数组。原raw可由冻结源码、配置和种子重新运行得到新的可审查记录。

每组`*_bundles`中按`bundle_manifest.json`核对包与成员SHA，再将该组所有tar.xz解压到同一个新目录。不同组使用不同目录。解压后运行：

```powershell
python -m research_experiments.mobile_ev_mle.ev_systematic_delivery verify --directory outputs/restored_analysis
python -m research_experiments.mobile_ev_mle.ev_systematic_delivery verify --directory outputs/restored_representative
python -m research_experiments.mobile_ev_mle.ev_systematic_review --directory outputs/restored_representative --expected-count 16 --output outputs/restored_representative_review.json.gz
```

原完整244条raw的独立AC/残差/成本回放报告也随证据交付，输入SHA与原raw清单关联；恢复的analysis-lite只能复算统计，不承担完整物理回放。冻结代理的QP、MILP及弦界另有独立算术审计，完整AC实际得分与代理证书分开报告。

## 测试与失败保留

`test_ev_systematic_*.py`覆盖场景/计量配对、训练与未来隔离、共享线路见证、代理上下界、校准、回放及交付。运行时关闭缓存并给pytest一个新的可写`--basetemp`；最终130项测试通过，XML随证据保存。大于250KB的汇总JSON按原字节gzip压缩，evidence_index.json同时保存原JSON与交付文件SHA；分析器和独立审计器支持.json.gz。需要原JSON时解压即可逐字节恢复。

v1开发遇到显式None证书的记账异常，11条完成记录及日志单列保留。修复后按新策略重跑全部64开发槽，正式比较只用v2。数值容差导致的原始MILP提案失败、最终合法回退、局部拟合失败与候选漏真均按原分母单列。
