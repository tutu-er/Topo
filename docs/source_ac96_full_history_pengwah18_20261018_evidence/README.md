# Pengwah18无源：完整历史方法、64/32与时限对照

本目录是最终6项运行的小型可提交证据包。已有一天96点保持不变，前64点训练、后32点验证；4项complete、2项超时incomplete，6个模型均保留。独立复核10084项全部通过。

- [六项结果与路径图](paths_report.md)、[汇总CSV](summary.csv)。
- [1SE与训练末点对比](selection_effect.csv)、[两档预算配对](budget_comparison.csv)。
- [完整最终认证路径](path_points.csv)、[含扩界历程的认证路径](all_certified_path_points.csv)；本次未触发扩界。
- [冻结协议](protocol.json)、[实际运行环境](environment.json)、[完成状态](completion.json)。运行代码提交为b8be3844403e15e199611e20f1d0a660f513f5dd；运行时源码快照另保留实际工作区依赖，包含此前用户未提交的相关源文件。所有快照的逐文件指纹见下列原始清单。
- [独立审计摘要](independent_audit_summary.json)、[贪心路径、剪枝、1SE与迭代上限复算](full_history_path_diagnosis_summary.json)。
- [全部原始运行文件的SHA256](raw_artifact_manifest.json)、[本证据包来源](package_manifest.json)。

100倍预算指每次求解60→6000秒、每方法300→30000秒，其他协议一致。两个100倍模型均为F1=0.75，按max_atoms停止；最后迭代计数35而当前族32个原子，未建立无进一步增益的收敛结论。选中点F1=0.75，最后训练点F1=0.5714。RNJ无边界块达到0.75，因此两方法路径相同。

F1同时保留family/active口径，两者本次一致；R/X是响应矩阵相对Frobenius误差。32点用于选模，没有独立测试集。真拓扑只用于保存后的评分与诊断。

原始模型、primal向量、每次求解gap、检查点、输入和源码快照保留在工作区 `outputs/source_ac96_full_history_pengwah18_20261018/`，不将大型生成文件加入Git。完整逐步路径诊断保留在 `outputs/source_ac96_diagnosis_20261018/full_history_path_diagnosis.json`。复制的paths_report.md只修订完结说明和图片链接；数值CSV/PNG与只读后处理原件逐字节相同。

[原因分析与NJ/RNJ源项处理](../source_ac96_diagnosis_20261018.md)
