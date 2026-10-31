# RNJ/WZZT方案索引与恢复入口

| 方案 | 状态与区别 | 代码／结果／报告 |
|---|---|---|
| 保留方案：完整RNJ骨架＋联合退出 | 冻结a46691ed。RNJ提供完整临时层级，120输入仅4个运行基础WZZT；末端及退出修正。统一0.24主F1=0.7958、60/108。 | research_experiments/joint_exit、rnj_count、joint_five；outputs/joint_exit_20261031；docs/joint_exit_20261031/ANALYSIS.md |
| 新方案：少量RNJ小块＋WZZT骨架 | small3，至多两个≤3终端临时小块。基础优先覆盖较多直属块，≤5后联合枚举；跨界窗口及全终端对参加退出；最终验证收缩。40秒测试已完成，未优于保留方案。 | research_experiments/rnj_small_seed；outputs/rnj_small_seed_20261031；SHORT_RESULTS.md与README.md |
| 新旧方案300秒配对 | 用户追加“大幅放宽求解时间”。原40秒结果保留；新旧各120次。单MIP30秒、前缀180秒、总300秒、LP30秒，两者都用已有较好解热启动。 | long_protocol.json、long_source、long/；完成后LONG_RESULTS.md |
| RNJ小块＋原始基础WZZT | 同一small3种子，直接调用已有original_step；不加半数块约束和强制整块并约束。审查后追加24输入、300秒机制对照，独立保存，不能外推为108主面板。 | base_protocol.json、base_source、base/；完成后ORIGINAL_BASE_RESULTS.md |

项目根目录为 `D:/0-github_workspace/Topo`，上表相对路径均从该目录解析。新方案分支为 `codex/rnj-small-seed-20261031`，保留方案分支 `codex/joint-exit-20261031` 不覆盖。

继续前先读 `outputs/rnj_small_seed_20261031/pipeline_state.json`，再读各 `*_completion.json` 和 `analysis/*verification.json`。未有完成回执的部分保持运行中／未完成，不从计划推断结果。

总入口：

```powershell
$env:PYTHONPATH='D:\0-github_workspace\Topo\rnj_wzzt_core'
D:\apps\miniconda3\envs\Topo\python.exe -B -m research_experiments.rnj_small_seed.pipeline outputs/rnj_small_seed_20261031
```

入口按短时→24热启动接续→独立核验→300秒配对→长时独立核验顺序续跑，已有结果直接保留；如果已有控制进程正在运行，不重复启动。失败见对应error.json，不静默删除或丢弃分母。

原始基础WZZT检查另由 `python -m research_experiments.rnj_small_seed.basecheck queued outputs/rnj_small_seed_20261031` 接续，等待前一流水线全部完成后开始；最新状态为base_state.json、base_completion.json和analysis/base_verification.json。如果已有该等待进程，同样不要重复启动。

解释约定：集合成员数与直属块数不同；原子小块须在修正阶段可打开；最终收缩只删边，替换需要候选重选；公共RNJ回退与WZZT路径分开；候选域最优／低残差不等于拓扑正确。
