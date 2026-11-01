# 已接入的三条迭代规则

当前研究主线已接入：每次已提交状态重拟合后释放临时零块；前向/末端无接受改善时至多一次训练排名删块并恢复搜索；屏蔽只在单个反事实分支内生效，完成全部训练延续后按验证MAE决定采用。详细数学和参数见DESIGN.md。

## 入口与兼容

- 普通研究WZZT：research_experiments.joint_five.search.search，默认zero_release=True；显式False可复现旧零块保留行为。
- 当前完整研究主线：research_experiments.iterative_exit.engine.fit_mainline。
- rnj_count/rnj_small_seed及纯WZZT搜索入口也补齐逐状态清理；五个相关freeze入口携带清理依赖和匹配接口，隔离导入5/5通过。已冻结历史outputs/source不改写。
- 通用核心fit_laminar_l1_sensitivity本来会清理新增非初始零块；initial_supports硬先验语义保持不变。此轮研究RNJ种子是临时集合，已核实先验需显式hard。

## 验证结果

101项相关回归通过：41项研究代码和60项通用核心检查。覆盖初始化错误零块释放后，普通单候选WZZT能选择原先交叉受阻的正确集合；硬先验保留；数值近零但删后损失超容差时拒绝清理；停滞只触发一次；屏蔽贯穿恢复/退出/输出拟合；验证可接受或拒绝屏蔽提议，不把屏蔽永久写回其他分支。

四个真实算例入口检查使用paper15/soumalas11/flynn16/pengwah18，bundle2/N48/V噪声.0002。4/4正常完成，26个独立HiGHS固定家族LP核验通过。paper15的迭代路径实际释放1个零块；其余3例没有符合条件的零块。四例均正常完成末端求解，未触发停滞反事实，new_branch明确记为not_triggered，而非填入回退结果。

四例原主线和新输出均完整恢复。这是入口/不变量检查，不用于声称恢复率提升；尚未对新策略重新执行完整120输入比较。真实停滞/屏蔽生命周期由专门反例和回归测试验证，之前24输入的退出时机试验也继续保留原协议和结果。

各输入新增运行时间约8.58、44.16、51.39、57.89秒（并行；共同前端记录成本另列）。每输入总账4小时，基线和两个可能的反事实共享，不是每分支各4小时。训练只读P/Q/Y和训练生成的公共候选；test/真拓扑仅评分阶段读取。指标分zero_only_own、zero_only_with_rnj、new_branch、own_output、with_rnj_fallback，始终保留来源。

## 恢复

代码research_experiments/iterative_exit，结果outputs/iterative_zero_exit_20261101。先读completion.json、smoke_verification.json、tests.json。records.zip含执行快照source/、交付时current_source/、源/数据哈希、四个结果与检查回执，不重复保存原始观测NPZ。执行快照和交付文件分别保存，源码SHA以协议为准。交付阶段修正了硬先验released日志、文档输出路径、纯WZZT入口清理与冻结依赖，并重算独立检查、重跑41项研究回归、检查5份隔离快照导入；没有重新求解或修改四例空hard设置的数值策略。

```powershell
$env:PYTHONPATH='D:\0-github_workspace\Topo\rnj_wzzt_core'
D:\apps\miniconda3\envs\Topo\python.exe -B -m research_experiments.iterative_exit.run freeze outputs/new_iterative_policy_check
D:\apps\miniconda3\envs\Topo\python.exe -B -m research_experiments.iterative_exit.run run outputs/new_iterative_policy_check
D:\apps\miniconda3\envs\Topo\python.exe -B -m research_experiments.iterative_exit.run analyze outputs/new_iterative_policy_check
```

freeze必须使用新目录。当前run/analyze为四例检查驱动，文档进入docs下同名目录，避免覆盖已交付协议。重新分析同一检查会更新该目录回执，查看已有结果不需要再次求解。
