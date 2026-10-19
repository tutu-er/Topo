# 旧 EV 实验：停止作为新研究入口

2026-10-12 起，新工作从 [mobile_ev_mle](../mobile_ev_mle/README.md) 开始。
目标是固定拓扑 AC 最大似然估计，以及基于 Dette 模型判别设计的 EV 选择。
旧有界误差、外 LP 和线性代理实验保留为历史对照，不再充当这条主线。
旧实验的“未决”不是 MLE 无效的证据；负结果也不等于无效数据。

## 保留与清理

- `results_v6` 至 `results_v9` 及其源码、审计材料保留原路径。已提交的证据未修改。
- 早期 `results`、`results_v2` 至 `results_v5`、两个 20261008 审查目录、`results_v5_three_phase` 已归档；位置、校验和及恢复说明见 [清理记录](../../archive/mobile_ev_pre_mle_20261012/README.md)。
- 会覆盖当前 LaTeX 源文件的旧 `build_report_v2.py`、`update_open_report_v4.py` 已移入该本地归档。早期 README 也保留在归档中。
- 可再生的测试临时目录、Python 缓存和 v9 两批过期 PDF 渲染已删除；最终 PDF、源文件与最终渲染证据保留。
- 默认递归发现不再收集旧测试和结果目录；递归复核旧测试时设置 `TOPO_RUN_LEGACY_EV_TESTS=1`。显式指定测试文件仍可运行，不应指向结果中的测试快照。

## 可复用的组件

| 文件 | 可复用内容与限制 |
|---|---|
| `passive_bridge_v9.py` | 被动 RNJ/WZZT 候选生成，允许修正未经认证的结构；不是完整候选空间 |
| `wzzt_adapter.py` | 旧源项 WZZT 适配；保留为候选生成组件，不是 MLE |
| `ac_general_v9.py` | 径向树构造、AC 潮流和物理核对；`solve_outer` 不属于 MLE 主判定 |
| `integrated_study_v9.py` | 旧对照与数据组织示例；`choose_probe` 是线性代理评分，不是完整 AC 似然设计 |

完整旧算法说明仍见 `README_v6.txt` 至 `README_v9.txt`，仅按需阅读。
没有按版本删除旧源码，因为较新模块仍导入较早模块；盲目删除会破坏复现。

历史复核示例（仓库根目录）：

```powershell
$env:TOPO_RUN_LEGACY_EV_TESTS='1'
& 'D:\apps\miniconda3\envs\Topo\python.exe' -m pytest research_experiments/mobile_ev_probing/test_ac_general_v9.py -q
Remove-Item Env:TOPO_RUN_LEGACY_EV_TESTS
```

全量运行早期实验前必须按清理记录恢复其数据和脚本。不要覆盖冻结结果或当前报告。
