# 完整消融恢复索引

工作区：`D:/0-github_workspace/Topo`。任务分支：`codex/iterative-matched120-20261101`。
优化实现与事前协议提交为`88b3cf87`；一次性后分析等待器提交为`5266e3c5`。

- 冻结协议：本目录`protocol.json`，计划8配置×120输入=960；原始定义见`PLAN.md`。
- 实际优化源：`outputs/full_ablation_20261102/source`；每个Python文件的SHA256在冻结协议内。禁止修改该目录或事后改变协议。
- 观测目录与split哈希：`outputs/full_ablation_20261102/catalog.json`和协议。原始NPZ原位引用，不复制进新归档。
- 公共校准：`outputs/full_ablation_20261102/common/<job>/common.json`，字节和损失标尺与前次120输入一致。
- 结果、检查点与进度：`outputs/full_ablation_20261102/jobs/<job>/<method>/`。
- 实时批次状态：`outputs/full_ablation_20261102/state.json`；最终运行分母：`completion.json`。
- 独立后分析：`analysis/verification.json`、逐输入/分组CSV与`analysis/jobs/`。待完成、失败和无效输入保留960计划分母。
- 实际后分析源码：`outputs/full_ablation_20261102/analysis_source`。冻结后只修正live分析器的一条回执规则：预算超限时禁止`passed=True`；优化器与协议保持原样。执行的分析器SHA在核验回执内。
- 一次性后处理等待器：`python -B -m research_experiments.full_ablation.finish outputs/full_ablation_20261102`。只等待运行完成，然后分析、绘图和压缩已有记录；不新增优化。状态在`postprocess.json`，日志在`postprocess.log`。

运行器可重入，已有结果和错误回执不会覆盖：

```powershell
D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.full_ablation.run status outputs/full_ablation_20261102
D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.full_ablation.run run outputs/full_ablation_20261102
```

确认已有管理进程结束后才重新启动，避免并发重复调度。错误结果先审查，不自动删除或重新计作成功。

全部运行结束后，主表、条件边际、交互、六种加入顺序及平均贡献图写入本目录；原始执行源与记录压缩为`records.zip`。
最终图形视觉检查另记回执。旧四算法对照及含RNJ整树备用方案均保留；本批所有最终输出均关闭完整RNJ备用。

本批已完成960/960并通过独立核验：5972次MILP达到所设精度、313个固定家族LP核验、30567项检查；失败、超时、无效及预算超限均为0。易读结论见CONCLUSIONS.md。
实际后处理源还保存了diagnostics.py、summarize.py和绘图图例排版修订；这些只处理已有记录，没有改变冻结优化器、协议或数值结果。图表提供PNG和SVG。
records.zip的实际大小、SHA256和后处理状态以postprocess.json为准，全部交付文件的字节校验见manifest.json。

时间为各输入完整算法墙钟之和，统一加一次公共前端成本，区别于并行批次等待时间。全部配置使用同一4小时总额和前向预算规则；参数不根据运行结果调整。
