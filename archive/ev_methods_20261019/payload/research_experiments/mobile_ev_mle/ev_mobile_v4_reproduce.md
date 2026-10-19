# v4 保存证据的复算

本轮有 4 条开发历史、8 条独立评价历史。每条评价历史的 regular/shift 共用同一历史、同一组计划和未来噪声；两种未来不算两条独立历史。先用保存证据复算汇总与物理检查，无需重新求解 MILP 或拟合候选。

## 源码与证据

执行副本为 `D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015`，研究源码位于 `C:/Users/23761/.codex/worktrees/36a1/Topo`。下列命令在含本工具的仓库根目录执行；Python 使用已验证的 Topo 环境。

每个证据目录的 `freeze.json` 记录 128 份科学源码的 SHA-256，`source_snapshot.json.gz` 保存完全对应的字节。128 份由原 122 份与 6 份 v4 科学文件组成。普通文件名相对于 `research_experiments/mobile_ev_mle/` 读取；以 `rnj_wzzt_core/` 开头的文件名相对于仓库根目录读取。不要从其他工作树拼接同名旧文件。`restore` 会逐份核对快照和当前源码，全部一致才继续；源码不符时应在专用复算工作树恢复相应版本。

开发证据 `development2_delivery` 的原打包工具版本为 Git `96917cd1`。其 manifest 中旧 `tool_sha256` 原样保留。新增 restore 工具使用同一压缩格式，并在新恢复目录记录自己的工具哈希；两者用途不同。

## 展开压缩证据，再运行现有 CLI

现有 analysis/audit 读原记录格式，不能直接把 compact 格式当作原记录。先展开到一个不存在的新目录。展开保留 `_evidence_provenance`，包括省略字段的路径、哈希和失败证据；已实现计量、操作及实际 AC 拟合证据完整保留，部分重复的预测拟合中间量保持已声明的省略状态。

以下为已打包开发阶段的例子：

```powershell
$taskPython = 'D:/apps/miniconda3/envs/Topo/python.exe'
$taskPack = 'research_experiments/mobile_ev_mle/evidence/ev_mobile_v4/development2_delivery'
$taskRestored = 'outputs/mobile_v4_restored_check'

& $taskPython -m research_experiments.mobile_ev_mle.ev_mobile_v4_evidence restore --directory $taskPack --output $taskRestored
& $taskPython -m research_experiments.mobile_ev_mle.ev_mobile_v4_analysis --root $taskRestored --phase development --require-complete --output outputs/mobile_v4_recomputed_analysis.json
& $taskPython -m research_experiments.mobile_ev_mle.ev_mobile_v4_audit --directory $taskRestored --phase development --output outputs/mobile_v4_recomputed_audit.json
```

最终评价证据使用对应含 `evaluation/` 的证据目录；将上述 `$taskPack` 改为该目录，两个后续命令的 `--phase` 改为 `evaluation`。恢复命令默认展开证据中实际存在的阶段，也可指定 `--phases evaluation`。所有输出必须使用新路径，不覆盖旧记录。分析生成 JSON 与同名 Markdown；物理审计重算保存态的 AC 方程、功率平衡、计量误差和操作约束，不调用优化器。

## 可选诊断与重新运行

- `ev_mobile_v4_interface_check` 将一个保存计划对应的预测均值送入实际评价接口，检查相同目标下的初始化和局部求解差异。重跑读取原始 raw 并重新拟合候选；首个开发动作的证据已保存，可直接读取。
- `ev_mobile_v4_forecast_audit` 的真值只用于事后分析保存动作。预定评价历史为 0、6；线路、普通负荷和源的八种替换共同分摊同一个电压增量 MSE。该分摊与似然差、排错率分别报告。
- 从冻结配置重新运行研究会产生新的求解时间与局部数值轨迹。原运行记录保留，另建输出目录；不使用评价结果修改冻结策略。当前未重新校准误拒率。
