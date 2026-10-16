# EV移动与误拒研究：复算入口

本轮从`dde0f241`继续，正式算法已冻结为guard60。实现与审计144项测试通过，阶段提交`0706e99a`已推送。正式校准已启动；正式评价结果待完成后回填。本说明不改冻结的115份科学源码。

源码工作树为`C:\Users\23761\.codex\worktrees\36a1\Topo`，执行副本为`D:\0-github_workspace\Topo\outputs\ev_rx_stage_20261015`。下文命令在执行副本目录运行，Python为`D:\apps\miniconda3\envs\Topo\python.exe`。

## 1. 这轮只回答三个问题

1. **原来为什么很多方法都是69/72？** 平方固定点与标量固定点只有25/72实际曲线相同，但原始第一名及保留集合72/72相同。多数收益变化未跨过决策边界。三个共同失败均为短训练star被clade12取代，竞争者多2个线路参数。完整旧批审计在执行副本`outputs/reliability_v2_old_evaluation_audit.json`。
2. **能否减少误拒？** 保留原相对SSE及训练长度条件规则，新增结构条件校准和固定复杂度工作评分`T_k=S_k+2d_k`，其中`S_k`为白化输出SSE，`d_k=2×线路数`。主比较预先定为`label_AIC`与`H_R`；其余三规则完整保留。减少误拒与增加未决分别计数。
3. **移动是否提供净收益？** 固定参数标量距离、截断总任务距离及共享线路预测价格分别比较固定点/移动，并保留平方移动强对照。计入驾驶、接线、能耗及求解成本，分开报告继承计划和新梯度步骤的贡献。

计量保持P/Q校正后直接输入，EV已经包含在总P/Q；输出为V和根表P/Q，偏差与根电压固定，未计量源逐时通过同一接口。参考未来均值每拓扑只用一组训练R/X。B价格法允许竞争模型一组R/X同时解释训练和预测响应，最终推断同样保存固定线路及共享线路两路线。数学见`ev_mobile_v2_math.md`；详细规则见`ev_reliability_v2_math.md`。

## 2. 已完成的开发与成本检查

|版本|开发槽/基础seed|B移动相对B固定点的实际M胜/平/负|B固定/移动规划失败|
|---|---:|---:|---:|
|原无保护30秒，n3|20/8|10/7/3|0/0|
|原无保护30秒，n4|8/8|3/4/1|0/0|
|guard30，n3|20/8|0/20/0|1/1|
|guard30，n4|8/8|0/8/0|7/7|

这里`M=min错误候选SSE−真候选SSE`只用于离线描述。两版本均保留全部28槽；同seed的不同训练/窗口设置是配对槽，不能计成28条独立历史。

guard30的B移动20份完成最终保护检查且通过，其余8份缺完整固定点评分，记录失败并回退；28份均未实际移动。B移动梯度MILP调用10次、接受0次，完整AC提案检查0次。最终收到数据后的联合拟合均完成。该批主要暴露规划预算不足：n4固定点甚至难以完成一次全部候选的预测评分。运行中也有多个求解/审计进程竞争资源，未改算法的基线曲线发生漂移；这些变化不能全归因于guard。

因此在首次正式校准前，将所有优化方法的公共软预算统一为60秒，单次MILP仍2秒；父计划和共享context成本计入方法成本。复用旧开发seed做4槽成本验收：n3一槽、n4三槽，均取得完整B固定/移动预测评分，移动均通过保护；其中一槽完成并接受2次梯度MILP步骤。其余3槽未出现接受的梯度步骤。两槽n4的B移动实际规划成本为81.66和68.71秒，保留软预算超时。**这4槽是成本验收，不是新增独立效果评价，也不纳入正式分母。**

原始目录均在执行副本`outputs/`：

|证据|目录/文件|
|---|---|
|原版开发|`mobile_v2_dev_n3_part0`、`mobile_v2_dev_n3_part1`、`mobile_v2_dev_n4`|
|guard30开发|`mobile_v2_guard_dev_n3_part0`、`mobile_v2_guard_dev_n3_part1`、`mobile_v2_guard_dev_n4`|
|开发小审计|`mobile_v2_guard_development_audit.json`及同名`.md`，全部56份旧/新raw的SHA在其中|
|guard60成本4槽|`mobile_v2_guard60_cost_n3`、`mobile_v2_guard60_cost_n4`|
|正式事前记录|`mobile_v2_guard60_formal_launch.json`，含4槽成本明细、源码哈希及预选代表|

旧版本重算应先恢复各自`source_snapshot.json.gz`中的源码到独立执行目录；当前源码默认guard60，不能用它覆盖旧批记录。

## 3. 正式冻结内容

policy ID：`35e7a11ba65a5366930888272f516c3165355f5dbe20de69919734dfd45e23e3`。配置版本`mobile_reliability_EV_v2_guarded_60`，115份科学源码哈希随各输出目录的`freeze.json`和`source_snapshot.json.gz`保存。

- 每个物理候选库独立生成96份校准、48份评价，n3/n4共288条历史；两库分别校准，候选身份按实际末端数与逐边支持集合确定。
- n3事前混合ordinary96、train32、train8、remote8_H96、remote8_H8及四种真结构；n4为twinblock12四种真结构。每份正式历史独立抽取场景/结构。
- 11法为none、constant、highlow、square_fixed/mobile、scalar_fixed/mobile、cap_fixed/mobile、jointgrad_fixed/mobile。
- 3种推断为fixed_total、fixed_holdout、joint_total。5条规则为global_R、H_R、label_R、global_AIC、label_AIC，每条各自α=.1。
- B移动保护底线为`h_a=min(g_a(B固定点),5.4110869)`，`g_a`是该参考模型的最近竞争预测差。局部MILP提出动作，覆盖全部候选组合的完整AC评分复核保护和平均目标；相同动作允许缓存。
- 仅2个计算worker运行，不同时做大型物理审计。先按n3两个半库完成，再按n4两个半库；校准与评价阶段分离。
- 正式评价首次读取前，先保存全部5规则不可变阈值。正式阶段不改τ、惩罚系数、方法、候选或样本量。

正式seed区间为：n3校准101210001–101210096、评价101220001–101220048；n4校准101310001–101310096、评价101320001–101320048。预选完整代表为每库校准index0及评价index0/12/24/36，另保留原版4个pilot。代表不按结果挑选。

## 4. 计算命令与结果接口

先设置单线程数值库：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
```

每个worker执行一个半库，完成后再切换下一个物理库。以n3校准前半为例：

```powershell
& 'D:/apps/miniconda3/envs/Topo/python.exe' -m research_experiments.mobile_ev_mle.ev_mobile_v2_study --bank n3 --phase calibration --start 0 --stop 48 --output outputs/mobile_v2_formal_cal_n3_part0
```

后半使用`--start 48 --stop 96`及新目录`...part1`。n4同样分两半。评价每库index0–47，两个worker分别0–23、24–47，使用全新输出目录。已有目录与旧记录保留，工具拒绝覆盖。

合并半库时先核对同一个freeze/policy和不重复的index/seed，再按原文件名汇集记录，保存原SHA清单。以下`cal_dir`、`eval_dir`代表**同一物理库**的完整96/48合并目录；代码不自动把开发或成本槽加入正式集合。

```python
from pathlib import Path
from research_experiments.mobile_ev_mle import ev_reliability_v2 as rel
from research_experiments.mobile_ev_mle import ev_mobile_v2_results as res
from research_experiments.mobile_ev_mle import ev_systematic_analyze as io

out = Path("outputs/mobile_v2_formal_n3_analysis")
cal = rel.read_phase(cal_dir, expected_phase="calibration")
thresholds = rel.calibrate(cal, out, alpha=.1, expected_count=96)
# 上一行已保存全部5规则；此后才首次读取评价。
ev = rel.read_phase(eval_dir, expected_phase="evaluation")
decisions = rel.evaluate(ev, thresholds, out, expected_count=48)
# 计算worker结束后提取一次原始选择日志，保存小投影以便反复分析。
metadata = res.extract_selection_metadata(ev, eval_raw_dir)
io._write(out / "selection_metadata.json", metadata)
report = res.analyze(ev, thresholds=thresholds, selection_metadata=metadata)
io._write(out / "results.json", report)
```

结果模块需要：标准record中的全部11法、3种损失、参数误差、操作及计时；已经冻结的5规则阈值；原始selection日志或上面保存的小metadata。缺少selection投影时会明确报告日志缺失，不把缺失当“0次梯度”。统计按case/truth/H拆分，逐历史保留移动−固定M、决策变化、真实moves、驾驶成本、保护拒绝与最终origin。

完成后分别运行两类独立检查：`ev_reliability_v2_audit.independent_verify(cal, ev, thresholds, decisions)`重算全部阈值和保留决策；`ev_mobile_v2_analysis_audit.audit_pair(...)`核对96/48分母、seed隔离及全部依赖成本。完整AC物理回算在计算worker结束后另行执行。

## 5. 数据分层及正式结果回填

完整raw保留在上述D-stage目录，包括计量、全部拟合状态、失败、求解界、路径与来源哈希。交付时保存所有正式记录的小统计投影、selection元数据及全部raw哈希；小投影能复算统计决策，完整预选代表用于回放AC和操作。完整raw与统计投影的能力分别说明，不用小投影代替完整物理证据。

正式结果待完成后填写：每库/每规则/每方法的唯一正确、误拒、未决、计算失败；R/X误差与实际M；移动配对收益；直接梯度入选和继承计划比例；保护拒绝、回退及软预算超时。当前不填入任何预期改善。

已有负面证据保留：旧69相同主要是决策边界饱和；结构条件R单独未修正旧三次star失败；guard30因计算预算退化为固定点；guard60成本验收仍有超时和无新梯度步骤；预测保护只约束工作评分，真实误拒和移动收益由本次独立评价检验。
