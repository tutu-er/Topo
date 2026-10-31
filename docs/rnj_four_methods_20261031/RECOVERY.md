# 四方法完整对照：恢复入口

本轮480/480条记录已齐；无worker失败，数值核验通过。264条为逐字节复用，216条为新增。普通方法120输入与扩展后的原始WZZT组合120输入均已完整覆盖，不再停留在前轮24输入范围。当前无需重新启动求解。

先读CONCLUSIONS.md，再读METHODS.md与ANALYSIS.md。结果采用四个固定论文系列合成网络；普通版是近期完整RNJ初值＋无计数惩罚的原逐步扩展，不是生产bootstrap两块初始化。所有方法同观测、同300秒结构上限、同验证与收缩，完整流程的目标／候选域差异保留。

| 内容 | 位置 |
|---|---|
| 当前代码 | research_experiments/rnj_four_methods |
| 当前完整结果 | outputs/rnj_four_methods_20261031 |
| 完成／审查回执 | completion.json、verification.json、delivery_audit.json |
| 最终／自身／前缀／未收缩逐输入结果 | per_model.csv |
| 按算例、N和噪声汇总 | cases.csv、conditions.csv |
| 原24与补齐84主输入／12压力输入 | prior_new_scopes.csv |
| 时间、实际基础MILP调用与退出状态 | events.csv、timing_summary.csv |
| 大小块误差、配对胜负、最终来源 | size_errors.csv、paired.csv、selection_origins.csv |
| 原始记录及源码 | records.zip、source_recovery.json |
| 数据／源码／交付哈希 | protocol.json、scoring_data_manifest.json、manifest.json |

上表路径在项目根D:/0-github_workspace/Topo下解析；CSV和文档位于docs/rnj_four_methods_20261031。完整本地日志与观测文件来源在outputs和catalog.json，未把全部观测重复提交至Git。

实现冻结提交b74db88e；独立评分驱动提交03a9b18e；源码恢复工具阶段提交ea26eabe。当前任务分支codex/rnj-four-methods-20261031。前轮分支codex/rnj-small-seed-20261031及codex/joint-exit-20261031保持。旧交付66文件逐一核验未变。

每条evaluation/job/method/provenance.json区分fresh、previous_long、previous_base；复用记录保留自己的原协议哈希。source/为本轮工作者实际冻结源码，previous_sources/保留旧版本与其不同的文件；source_recovery.json逐文件记录如何重建旧版本。评分和报告源码在归档actual_driver/中，不能将事后分析器误当成优化器的训练输入。

使用D:/apps/miniconda3/envs/Topo/python.exe，PYTHONPATH指向项目rnj_wzzt_core。常用命令：

```powershell
python -B -m research_experiments.rnj_four_methods.run pipeline outputs/rnj_four_methods_20261031
```

已有result/error不会重新调度。若只需读结论，无需运行此命令。重新执行分析会重建records.zip及派生表；报告与交付随后依次由report、delivery模块生成。delivery会阻止向同一zip重复附加，需先由分析器重建归档。不要覆盖原冻结协议或用失败子集改变分母。
