# G精度MILP冻结证据与复算

本包对应2份开发与8份独立评价，科学identity为cad92e5b819d5dc91ac82defd8dd07407384df2ecebda1898254c62b52d31a25。简明结论见[results.md](../results.md)，完整方法名见[METHOD_NAMES.md](../../METHOD_NAMES.md)。

- [compact/summary.json](compact/summary.json)：全部方法、FIX/JNT、noisy/clean、逐对比较、R/X误差、失败分母、求解间隙和成本。
- [compact/history_pairs.json](compact/history_pairs.json)：逐历史计划指标、分数变化、实际裕量、部署依赖成本；两未来情景不增加独立历史数。
- [compact/audit.json](compact/audit.json)：1920次保存状态完整AC回放，0问题；没有重新拟合或优化。
- [compact/compact_replay_comparison.json](compact/compact_replay_comparison.json)：精简前后科学统计、审计和逐历史配对完全一致的核对记录。
- [compact/raw_and_export_sha256.json](compact/raw_and_export_sha256.json)：10份原始与精简记录SHA、字节数、原始保留位置。
- [compact/freeze.json](compact/freeze.json)、[compact/source_snapshot.zip](compact/source_snapshot.zip)：实际执行的1220个科学文件哈希和精确源码归档。
- [validation.json](validation.json)：最终60项测试及补充环境版本。
- [restore_validation.json](restore_validation.json)：实际恢复科学源码到新空目录后，再次完成1920次AC审计，统计、审计和逐历史结果与原版一致。
- [compact/development_audit_passed](compact/development_audit_passed/summary.json)：首次通过的开发验收结果，独立于最终评价。

完整raw约33.30MB，保持原字节留在D:/0-github_workspace/Topo/outputs/ev_g_precision_stage_20261017/outputs/g_precision_v1。仓库保存全部10份精简记录，合计8.00MB；包含观测、噪声、参数状态、计划/父计划、现有MILP日志、上下界、块及通道SSE、决策和失败。删除的仅是优化起点中的重复数值数组和可从观测/状态重算的残差向量，逐项保留hash/大小或统计摘要。具体导出规则见[compact_export.py](compact_export.py)。

“完整现有日志”不表示旧接口保存了所有内部诊断：例如本批master_without_valid_operation未保存具体操作违约项，该缺口在results.md披露。原始失败仍在包中，没有用成功替换失败。

精简包回放与原版逐项统计完全一致；只忽略生成时间、记录路径和因精简而改变的记录字节hash。完整源码压缩包SHA为60ee02872946ccf41f09af486e637824d6f8ecb401b6a83fba434fc553bc3b14。analysis.py属于独立后处理，未进入科学策略哈希；实际版本另存compact/analysis_postprocessor.py并记录SHA。

先将源码恢复到新空目录，再复制独立后处理，最后回放（将占位路径替换成绝对路径）：

```powershell
Expand-Archive -LiteralPath '<本包>/compact/source_snapshot.zip' -DestinationPath '<新恢复目录>'
Copy-Item -LiteralPath '<本包>/compact/analysis_postprocessor.py' -Destination '<新恢复目录>/research_experiments/mobile_ev_mle/g_precision/analysis.py'
Set-Location -LiteralPath '<新恢复目录>'
python -m research_experiments.mobile_ev_mle.g_precision.analysis --output '<本包>/compact'
```

该回放仅前向计算已有AC状态，不运行MILP或新的拟合。重新研究性运行的命令见上级README，必须使用新输出目录；时间受限的MILP可能返回不同的合法计划，所以原计划和求解证据已一并保留。

测试环境为Python3.11、NumPy/SciPy及Gurobi13；OMP、MKL、OpenBLAS均单线程。新公式及相关回归测试共60项通过。本包未生成新概率校准结果，未检验光伏/风电、候选漏真或滚动实时规划。
