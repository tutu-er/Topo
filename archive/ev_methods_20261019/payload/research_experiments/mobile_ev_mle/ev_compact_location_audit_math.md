# 冻结动作的电气位置消融

本工具处理紧凑策略的两个冻结主方法：`refined_v2g_mobile`、`capped_v2g_mobile`。复用第16阶段`ev_original_location_ablation.py`的AC生成、同噪声计量和H+未来联合拟合；新增compact配置和实际执行计划账本的适配。主选择器及72个科学源不变。

## 固定什么，改变什么

令 \(u_{tj}\) 为已选动作在时段\(t\)、地点\(j\)的网侧有功，单位kW，正为充电。固定其逐时总功率

\[
U_t=\sum_j u_{tj},\qquad
\widetilde u_{t0}=U_t,\quad\widetilde u_{tj}=0\ (j\ne0).
\]

背景负荷、根电压、逐边R/X、同接口源及计量噪声均取原历史保存值。末端总有功送入AC时为

\[
p_{tj}=p^0_{tj}+u_{tj}/P_B,
\qquad \widetilde p_{tj}=p^0_{tj}+\widetilde u_{tj}/P_B,
\quad P_B=20\ \mathrm{kW}.
\]

EV只加入一次，EV无功为0。根表P/Q及线路损耗由各位置的完整AC重新给出；它们不是被固定的总功率。原动作冻结后才读取未来来生成离线模拟观测，不据未来重选位置或曲线。

逐时EV总有功及网侧净电量相同。单车单时段只接入一处时，绝对值吞吐也相同；程序逐条件核对这三项。

## 取消驾驶的服务变化

原驾驶耗电为\(E_{\rm drive}\)，充放效率为\(\eta_c,\eta_d\)，时长\(\Delta t\)单位小时。定义\(U_t^+=\max(U_t,0)\)、\(U_t^-=\max(-U_t,0)\)，则

\[
E_T=E_0+\Delta t\sum_t(\eta_c U_t^+-U_t^-/\eta_d)-E_{\rm drive},
\]
\[
\widetilde E_T=E_0+\Delta t\sum_t(\eta_c U_t^+-U_t^-/\eta_d)
             =E_T+E_{\rm drive}.
\]

因此这是同网侧曲线的电气位置隔离，不是同终端电量服务的固定/移动设备比较。保存原执行operation的SOC、驾驶与人员成本；另给出无驾驶能量轨迹、原终端任务误差、SOC/功率/吞吐点检查。原策略继承父方案时`guard.accepted=False`也可能有合法执行计划；账本必须读取与最终曲线一致的`record.operation`，不按该guard字段误替换成恒充。

当前冻结操作域初始在地点0且已接线，地点0全时段可用。搬移的假设路径是不驾驶、留在地点0；不求新路线。接入假设与默认SOC界来自被哈希绑定的父域。终端服务不同及任何点约束不满足均记录。

## 计量与联合拟合

相同白噪声\(\epsilon\)、固定校准偏差\(b\)和标准差\(\sigma\)用于两个位置：

\[
y^{d}=\mu^d+b+\sigma\epsilon,
\qquad d\in\{\mathrm{original},\mathrm{remap}\}.
\]

无噪声分支只将未来\(\epsilon\)设0，历史保持原样。竞争候选全部保留4个，每个拟合历史和相应未来，历史/未来共享逐边R/X，源逐时变化但固定接口1。参数盒、两个初值、迭代预算和种子规则与父评价相同。原位置的拟合含失败直接复用；新搬移曲线用完整AC联合重拟合。保存局部取得SSE，不宣称全局MLE。

真候选\(a\)只用于离线评价。记\(J_k^d\)为候选\(k\)取得的联合SSE，

\[
M^d=\min_{k\ne a}J_k^d-J_a^d,\qquad
\Delta M=M^{\rm original}-M^{\rm remap}.
\]

同时保存无噪声\(\Delta M\)、4候选逐通道未来P/Q/V/根表SSE、历史罚损、12个有向半SSE差，以及真候选模型下R/X相对误差。新改接过程没有校准拒绝规则，不套原主策略阈值，不输出其误拒/排错概率。

## 缓存、失败与分母

完全相同的完整功率数组使用精确缓存，负零归一。未移动动作与自身精确复用，若原结果完整，则理论和保存的\(\Delta M=0\)；失败也原样保留。共享缓存减少AC/拟合调用，不增加独立历史。

正式评价固定12条历史，每个方法各12个配对，共24个方法条件。历史是独立统计单位，两个方法和4候选不增加这个数量。缺动作、物理AC失败、电压越界、优化失败、同噪声或原操作审计失败均保留分母和原因；保留可取得的通道数值。最大新增拟合为\(12\times2\times2\times4=192\)次，缓存或物理失败会减少实际调用，尝试次数与返回的fit调用另列。

当前仅授权4条开发历史试跑，正式12H由根任务在独立评价完成后运行；校准历史不作为本消融输入。完整父研究文件及独立审查须先齐全，冻结父policy、配置、压缩输入字节、72源和本工具/复用helper/本说明的哈希及字节快照。每H完成保存检查点；源或输入变化时保存无效结果并停止。

```powershell
python -m research_experiments.mobile_ev_mle.ev_compact_location_audit `
  --input <完整4H开发或12H评价目录，raw或slim> `
  --output <新消融目录> --parent-review <同输入的独立父审查> --execute
```

省略`--execute`只保存冻结清单。正式评价阶段必须提供匹配policy、种子及原raw哈希的独立父审查。结果输出`history_<seed>.json.gz`、`paired_locations.csv`、`candidate_channel_costs.csv`及`summary.json`，不修改父记录。
