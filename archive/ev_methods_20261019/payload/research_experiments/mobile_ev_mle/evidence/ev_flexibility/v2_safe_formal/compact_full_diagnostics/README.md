# 新安全版EV设备策略：紧凑可复核证据

本目录保存39个独立校准历史（984001–984039）和48个独立评价历史（985001–985048），每H仅一个计量质量条件。阈值由校准样本的9方法最大拒绝分数独立重算，κ=6.034651101192644。历史和新增量测共同拟合相同逐边参数；所有原状态和失败保留。

`compact_manifest.json`列出每个原raw SHA和剔除字段路径。仅删除两个长SVD诊断字段：`local_scaled_singular_values`与`local_projected_shared_singular_values`。其余拟合状态、源、物理响应、设备约束、曲线、信息矩阵、梯度、接受过程、调用计数和失败保持原字段结构。原raw仍保留在本地，不重复提交。封套、源码快照、cal/eval摘要与checkpoint均为原记录的精确副本。

独立复核入口只读取本目录记录；原raw可用时还会核验来源转换，原raw不存在时仍可完成数值和统计复核（compact-only测试已验证）。从仓库根目录运行：

```powershell
python -m research_experiments.mobile_ev_mle.ev_flexibility_safe_review --directory research_experiments/mobile_ev_mle/evidence/ev_flexibility/v2_safe_formal/compact_full_diagnostics --phase calibration
python -m research_experiments.mobile_ev_mle.ev_flexibility_safe_review --directory research_experiments/mobile_ev_mle/evidence/ev_flexibility/v2_safe_formal/compact_full_diagnostics --phase evaluation
```

`calibration/safe_independent/`和`evaluation/safe_independent/`包含独立branch-current扫回算、SSE/对数似然、EV一次入计、终端SOC/效率/移动/接入检查、方法CSV、紧凑与原数据科学相等证据；没有调用拟合器或选择器。正式统计由独立模块重新计算，未调用正式summarizer。`safe_review_validation.json`绑定外围16测试和冻结23源码/37safe测试来源。数值回算模块保持此前已验证版本。

| 方法 | 唯一正确 | 安全未决 | 误拒 | 计算失败 | 分母 |
|---|---:|---:|---:|---:|---:|
| none | 9 | 39 | 0 | 0 | 48 |
| constant_fixed | 28 | 20 | 0 | 0 | 48 |
| quality_charge | 26 | 22 | 0 | 0 | 48 |
| charge_fixed | 27 | 21 | 0 | 0 | 48 |
| charge_random | 29 | 19 | 0 | 0 | 48 |
| charge_mobile | 30 | 18 | 0 | 0 | 48 |
| v2g_fixed | 27 | 19 | 2 | 0 | 48 |
| v2g_mobile | 27 | 20 | 1 | 0 | 48 |
| charge_staff | 28 | 20 | 0 | 0 | 48 |

任一9方法误拒 2/48，95%经验二项区间 [0.005086,0.142541]。控制器时间超15s 11/48、整条件超90s 0/48，全部保留在分母。该区间描述此批仿真分布，分组结果和配对比较在评价summary内。

固定设计27/48与恒充28/48、移动设计30/48的结果表明：已实现可行曲线和有效激励，但尚未建立优化曲线稳定优于恒充的结论。V2G及人员开放窗口的收益也需按保存负例评价。移动方案保持相同终端电池任务，同时改变位置、时段及补回驾驶耗电的网侧电量；对比评价完整可执行方案。

似然、设计分数和最小SSE排序分别是拟合与规划量；正式保留集合由独立阈值规则给出。研究整体仍未完成，未声明全局MLE或全局设计最优。
