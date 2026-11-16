# 单步10秒、总计120秒的预算诊断

2026-11-16，新增拟合前锁定。只检查时间分配，不更换辨识算法。

固定沿用 `outputs/identification_contribution_20261116_v1` 的八个输入：
Soumalas11、Pengwah18，各8天训练、h00至h03；每项768点训练、192点验证。
保留该批完整输入清单及运行代码快照，准备时逐个核对SHA256。
输入选取不使用恢复率，不打开真值或第11天数值。

仅重新运行 `100`，共8次。每次总限120秒、输出预留12秒、QP软限300秒不变；
唯一算法设置变化是 `per_mip_seconds` 从120秒改成10秒。最多3个单线程worker。
W的单连接QP预试、未知支持MIQP、接受规则、0.08非末端收缩与最终严格验证选模保持原样。
每步仍可能保留预试好点；这类收益与原生MIQP新增收益分开。
10秒约束施加于每次原生MIQP调用，预试仍计入原总时限，不人为改称整个W步骤最多10秒。

直接复用 `rnj_pool_only_20261116.run.fit_one`，`protocol.version='new'`，
但其算法包从原贡献批的不可变 `source_snapshot/rnj_wzzt_dr` 导入。
新入口和共享执行器另外留源副本及哈希；运行前后均核对。
本批单独执行100，每项共同准备做一次、计入该方法总预算；原比较批的共同准备则物理复用、分别记账。
不重跑RNJ1/RNJ15，不追加E60，不产生新AC或灵活性实验。

默认输出为 `outputs/identification_stepcap_20261116_v1/new`。
每项保存 `cases/<input_id>/started.json`、运行进度、`native.json`（成功时）和 `result.json`。
总表保留全部8项、失败、限时、输出超限、实际初值及原生上下界。
完整或失败任务不重试；若任务目录存在而无最终结果，拒绝自动续算并重给预算。
旧结果及源码均不覆盖。

完成后与原100逐项比较：W步数、停止原因、预试和原生耗时、当前点→初值→原生点→固定QP收尾的目标变化，
以及最终验证选择。恢复评价只在拟合封存后独立复用已有评价数据；RNJ1/RNJ15可引用原结果。
短单步是否更好由全部八项结果决定。该检验使用已经看过的输入，属于开发预算诊断，不称独立确认。

准备命令：

```powershell
& D:/apps/miniconda3/envs/Topo/python.exe -B -m rnj_wzzt_dr.experiments.identification_contribution_20261116.run_step_cap --source-run outputs/identification_contribution_20261116_v1 --out outputs/identification_stepcap_20261116_v1/new --prepare-only
```

根任务审查后运行：

```powershell
& D:/apps/miniconda3/envs/Topo/python.exe -B -m rnj_wzzt_dr.experiments.identification_contribution_20261116.run_step_cap --out outputs/identification_stepcap_20261116_v1/new --resume
```
