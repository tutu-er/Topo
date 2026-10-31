# RNJ/WZZT方案索引与恢复入口

| 方案 | 状态与区别 | 代码／结果／报告 |
|---|---|---|
| 保留方案：完整RNJ骨架＋联合退出 | 冻结a46691ed。RNJ提供完整临时层级，120输入仅4个运行基础WZZT；末端及退出修正。统一0.24主F1=0.7958、60/108。 | research_experiments/joint_exit、rnj_count、joint_five；outputs/joint_exit_20261031；docs/joint_exit_20261031/ANALYSIS.md |
| 大块优先变体：少量RNJ小块＋受限WZZT | small3，至多两个≤3终端临时小块。额外要求基础步骤优先覆盖半数直属块；40秒168/168完成。该限制与原始WZZT不同，不能混称。 | research_experiments/rnj_small_seed；outputs/rnj_small_seed_20261031；SHORT_RESULTS.md与README.md |
| 新旧方案300秒配对 | 完成240/240。单MIP30秒、前缀180秒、总结构300秒、LP30秒；共同热启动。108主输入：完整RNJ F1=0.8201、61/108；大块优先0.7920、57/108。压力12输入均0.6944、7/12。 | long_protocol.json、long_source、long/；LONG_RESULTS.md |
| RNJ小块＋原始基础WZZT | 完成24/24。直接调用original_step；不加半数块和强制整块并约束。F1=0.9064对完整RNJ0.8679，完整恢复均14/24；5胜18平1负。审查后追加对照，不能外推为108主面板。 | base_protocol.json、base_source、base/；ORIGINAL_BASE_RESULTS.md |

项目根目录为 `D:/0-github_workspace/Topo`，上表相对路径均从该目录解析。新方案分支为 `codex/rnj-small-seed-20261031`，保留方案分支 `codex/joint-exit-20261031` 不覆盖。

本轮已全部完成，pipeline_state.json与base_state.json均为complete/passed。综合解释见[ANALYSIS.md](ANALYSIS.md)，最终文件校验见delivery_manifest.json。short_manifest.json只对应249c840e历史检查点，不用于校验后续更新的计划、索引和入口说明。

恢复前先读 `outputs/rnj_small_seed_20261031/pipeline_state.json`，再读各 `*_completion.json` 和 `analysis/*verification.json`。先识别原始基础与大块优先版本，再比较同一分母，避免把修正版24结果写成全120结论。

总入口：

```powershell
$env:PYTHONPATH='D:\0-github_workspace\Topo\rnj_wzzt_core'
D:\apps\miniconda3\envs\Topo\python.exe -B -m research_experiments.rnj_small_seed.pipeline outputs/rnj_small_seed_20261031
```

入口按短时→24热启动接续→独立核验→300秒配对→长时独立核验顺序续跑，已有结果直接保留。当前无待运行搜索，无需重启；失败见对应error.json，不静默删除或丢弃分母。

原始基础WZZT检查另由 `python -m research_experiments.rnj_small_seed.basecheck queued outputs/rnj_small_seed_20261031` 接续；本轮已完成24输入，不自动扩展为120。最新状态为base_state.json、base_completion.json和analysis/base_verification.json。

解释约定：集合成员数与直属块数不同；原子小块须在修正阶段可打开；最终收缩只删边，替换需要候选重选；公共RNJ回退与WZZT路径分开；候选域最优／低残差不等于拓扑正确。
