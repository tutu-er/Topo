# EV移动与误拒研究：复算入口

本轮冻结实验及审计完成：192校准、96独立评价；352来源槽的完整AC回放、288份源拟合缓存所有权、预声明空间32槽及237项测试均已核对。简明结论见`ev_mobile_v2_results.md`，成本见`ev_mobile_v2_costs.md`。115份科学源码和先于评价保存的阈值保持原字节。统计、原始失败、软件版本和恢复哈希均已归档；不启动新批次。

源码工作树`C:/Users/23761/.codex/worktrees/36a1/Topo`，执行副本`D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015`，Python `D:/apps/miniconda3/envs/Topo/python.exe`。

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

正式结果已填写于`ev_mobile_v2_results.md`及两库完整分析。每库48份独立评价保留全部11方法、3种拟合口径、5条规则；未决、失败、回退、R/X误差、实际M、直接梯度来源和软预算超时均另列。独立物理汇总覆盖352来源槽，其中正式288历史与重复开发槽分开。

已有负面证据保留：旧69相同主要是决策边界饱和；结构条件R单独未修正旧三次star失败；guard30因计算预算退化为固定点；guard60成本验收仍有超时和无新梯度步骤；预测保护只约束工作评分，真实误拒和移动收益由本次独立评价检验。

## 6. 共享缓存费用的独立核对

`ev_mobile_v2_cache_owner_audit.py`按原始调用日志核查源拟合缓存的首次计算者，以及其费用是否已在当前方法的祖先中支付。静态检查发现一个条件缺口：`cap_fixed`可能命中仅由`scalar_mobile`预先计算的曲线，而前者不包含后者费用。正式288条历史的完整审计已完成：exact288、unknown0，cap_fixed的1651次查询中1107次命中，闭包外未付费命中为0、追加费用为0。该源码条件缺口在本批未触发，冻结策略保持原样。

工具按每方法的唯一缓存键补计已知首次生产费用，同时保留重复查询次数、缺失日志和方法分母。补计值描述保存路径的费用重记，独立重跑时的路线、缓存及截止行为可能改变。小统计投影不足以核实缓存费用，须读取完整raw。

```powershell
python -m research_experiments.mobile_ev_mle.ev_mobile_v2_cache_owner_audit --directory outputs/mobile_v2_formal_cal_n3 outputs/mobile_v2_formal_cal_n4 outputs/mobile_v2_formal_eval_n3 outputs/mobile_v2_formal_eval_n4 --expected-count 288 --progress --output outputs/mobile_v2_formal_source_cache_owners_v1.json.gz
```

正式结果另用`ev_mobile_v2_uncertainty.py`输出逐项Wilson区间和配对计数。每库48条历史时，1条改变对应2.08个百分点；例如0/48误拒的95% Wilson区间仍为0–7.41%。主比较固定为`label_AIC`相对`H_R`；同规则下另比较移动与固定点，不按评价挑选规则。失败时保留全部候选的历史仍在分母，失败数另列。

```powershell
python -m research_experiments.mobile_ev_mle.ev_mobile_v2_uncertainty --evaluation-report outputs/mobile_v2_formal_n3_analysis/evaluation_reliability_v2.json --records outputs/mobile_v2_formal_eval_n3 --bank n3 --output outputs/mobile_v2_formal_n3_analysis/uncertainty
```

两库分别执行。区间描述声明模拟分布中的重复历史事件率，各项分别报告，不提供同时覆盖，也不解释为某个拓扑的正确概率。

## 7. 已冻结的正式阈值

`evidence/ev_mobile_v2/formal_thresholds/manifest.json`保存两库原始阈值字节、校准合并清单、独立核查和首次评价启动记录。858组分组与顺序统计量全部一致；两库各96个独立校准seed，不包含开发、pilot或成本验收槽。两库阈值门保存于2026-10-16 06:10:32.132878 UTC，随后才启动评价。过程中的重启沿用这些文件，计算只补缺失记录。阈值与后续评价结果分别交付。


## 8. 最终交付与从新checkout复算

证据目录`evidence/ev_mobile_v2/`各阶段保持原字节：

|目录|内容|
|---|---|
|`stage0/`|原30秒、保护30秒、保护60秒的115份科学源码快照，原失败和开发记录|
|`operations_20261016/`|控制程序恢复、缓存/区间工具测试和原运行错误|
|`formal_thresholds/`|评价启动前保存的两库阈值、858组核对及启动门证据|
|`poststats_20261016/`|两库全部统计、区间/配对、预测排序、空间32槽、初次setup失败与237通过XML|
|`cache_audit_20261016/`|288份源拟合缓存费用和独立汇总核查，完整成本/移动表|
|`delivery/`|14目录352槽统计投影、14预选完整计量压缩代表、2个完整raw原gzip|
|`final_20261016/`|全部物理报告、352槽聚合、打包/恢复验证、最终验收与后处理源码字节|

`delivery/pack_manifest.json`列出21个包及SHA，总53.02 MiB，每包≤20 MiB。先校验包字节，再将所选`tar.xz`解压到新的空目录；`bundle_manifest.json`逐成员给出SHA。分析投影保留352槽的损失、参数误差、操作、成本与失败；14条完整计量压缩代表保留全部真实计量和最终固定/联合状态，但汇总规划中间数组。

两个评价序号0另按原gzip字节分为`.bin`。按`delivery/bundles/selected_complete_raw/raw_bundle_manifest.json`各文件`parts`的offset顺序连接，核对整文件SHA后才能作为完整raw。`.bin`无需解压成tar；恢复后的`.json.gz`包含所有规划数组；同包的freeze与source snapshot等元数据也应按清单恢复到该目录。其余完整raw仍留在D-stage，交付投影不能恢复这些raw全部字节。

解包后的声明、来源及统计投影可只读检查：

```powershell
python -m research_experiments.mobile_ev_mle.ev_mobile_v2_delivery verify --directory <新解包目录>
```

独立物理回放使用包含完整计量与状态的代表或恢复raw目录：

```powershell
python -m research_experiments.mobile_ev_mle.ev_mobile_v2_review --directory <完整证据目录> --expected-count <目录内槽数> --output <新review.json>
```

不同科学版本的115源快照分别保存；要重做选择器，先在独立执行目录恢复对应snapshot的原字节，再用相应版本的配置和seed。仅核对保存状态无需重新生成历史、规划或加噪声。`final_20261016/manifest.json`记录实际工具哈希及全部报告SHA，`final`中的`.gz`需先解压一层才能取得原始文件；原始文件本身可能仍是gzip。

完整物理回放检查58,992保存状态、139,468个AC块和32,964个动作QP。它验证保存的AC方程、白化误差、操作及代理算式；局部拟合和每轮MILP的界按其自身对象解释。未把回放通过写成全局MLE或原非凸动作问题的全局最优证书。

本轮不改原LaTeX/PDF。当前结果、证据及后续建议完成交付后，停止本轮自动续接；后续新研究需另行冻结策略与独立校准评价。
