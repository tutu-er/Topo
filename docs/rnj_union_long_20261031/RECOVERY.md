# 结果和恢复索引

工作区D:/0-github_workspace/Topo，分支codex/rnj-union-long-20261031。主模型与协议已在1867d24f冻结并推送，独立审计和数学说明在deed6dae提交并推送；最终报告、诊断和归档由后续提交保存。不修改原四方法和NJ组合报告。

## 先读这些文件

- CONCLUSIONS.md：120总分母、108／12分组、求解预算和原因分析。
- PLAN.md：读取主结果前冻结的四模型方案。
- MODEL.md：候选并集的一次层级稀疏L1 MILP、条件保证和复杂度边界。
- ADDENDUM.md：主实验之后才增加的验证保护诊断和删除生成器消融。
- comparison.csv：含all120、main、stress，F1、完整恢复率、累计时间和探索标记。
- cost_per_input.csv：方法逐输入成本、时间口径和预算。旧fresh=0指本轮未重跑。
- per_model.csv、cases.csv、conditions.csv、scopes.csv：主结果及原方法的可追踪行。
- objective_diagnostics.csv：真实结构独立LP与联合原始目标比较，仅事后评分。
- verification.json、implementation_check.json、bank_ablation_verification.json：检查回执。
- delivery_audit.json、manifest.json：交付和文件哈希；manifest不包含自身。

## 完成分母

outputs/rnj_union_long_20261031/state.json为complete，completion.json为480/480、failures为空。该目录bank_ablation/completion.json另120/120。验证选择诊断未增加结构求解。不能把600结构作业写成600个独立输入；输入始终为同120个。

两个父结果outputs/rnj_four_methods_20261031与outputs/rnj_nj_portfolio_20261031继续只读保留。新主protocol.json保存父manifest、结果与候选来源的哈希；55个父交付文件核验未变。

## 证据位置

本地完整工作目录outputs/rnj_union_long_20261031：source/为执行用冻结源；pools/为训练候选；jobs/<job>/<method>/为payload、progress、checkpoint、result及worker.log；analysis/为独立LP与各诊断；previous/保存父评分参考；bank_ablation/为补充消融。

records.zip保留上述冻结源、主480和额外120作业的候选／回执／日志、原记录引用、独立审计数据、实际分析脚本与报告。actual_driver/与source/分别保存实际后分析代码和当时执行的冻结优化器，两者不可混淆。观测NPZ和网络CSV不重复打包，位置与哈希在catalog.json、protocol.json和父数据索引；本地源数据继续保留。归档解压只用于恢复记录，不意味着已复制所有原始输入。

原始数据哈希核查使用train、validation、test文件；truth_network是目录，包含branches.csv、buses.csv、metadata.json，不按单文件读取。任何重新生成数据或更改候选的后续实验应使用新目录、新协议，不能覆盖本轮。

## 命令

Python运行时D:/apps/miniconda3/envs/Topo/python.exe，PYTHONPATH设为D:/0-github_workspace/Topo/rnj_wzzt_core。在工作区根目录运行：

```powershell
$env:PYTHONPATH='D:\0-github_workspace\Topo\rnj_wzzt_core'
D:\apps\miniconda3\envs\Topo\python.exe -B -m pytest research_experiments/tests/test_rnj_union_long.py -q
D:\apps\miniconda3\envs\Topo\python.exe -B -m research_experiments.rnj_union_long.finish comparison outputs/rnj_union_long_20261031
D:\apps\miniconda3\envs\Topo\python.exe -B -m research_experiments.rnj_union_long.finish package outputs/rnj_union_long_20261031
```

主训练执行入口run.py、事后主审计analyze.py；补充诊断入口diagnose.py、bank_ablation.py。协议及completed回执已存在，无需再次跑求解器来查看结果。重新打包会改变ZIP时间戳与manifest，属于再生成交付，并非新的实验结果。

## 本轮没有做的改动

没有新增物理阻抗正下界，没有扫描λ或用测试F1选阈值，没有修改AC拟合相关的其他在途文件，没有升级生产默认方案。新预算非无限：每方法每输入4小时、纯WZZT单步1小时；本轮所有新结构MILP在上限前达到指定gap，仍保留单步／有限候选域的证明边界。
