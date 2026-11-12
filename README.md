# MIQP RNJ＋WZZT 与条件 DR

当前研究入口是 [rnj_wzzt_dr/README.md](rnj_wzzt_dr/README.md)。流程为：RNJ 提供候选与热启动，WZZT 用 MIQP 逐步扩展，可选 T/E 重构；验证 MSE 选择后冻结完整候选库，再做条件 DR 规划和原设备／AC 评价。

```powershell
$py = 'D:/apps/miniconda3/envs/Topo/python.exe'
& $py -I -B rnj_wzzt_dr/run.py doctor
& $py -I -B rnj_wzzt_dr/run.py check --solver
& $py -I -B rnj_wzzt_dr/run.py --help
```

辨识和报价的输入、原最终评分、版本差异及当前缺口见主线说明；研究模型不因本次归档改变。零动作核算和原单对报价已接通，多竞争者非零报价仍待接通。

## 目前保留的内容

| 位置 | 用途 |
|---|---|
| [rnj_wzzt_dr/](rnj_wzzt_dr/README.md) | 当前主线代码、必要资源和验收记录 |
| [topology_identification_standalone/](topology_identification_standalone/README.md) | 正在进行的严格 96 点批次、原算例和历史完整验证；运行完成前保持路径 |
| `rnj_wzzt_core/` | 另一路 AC 拟合／证据研究，含未提交代码，保持原位 |
| [research_experiments/](research_experiments/README.md) | 两个未跟踪研究目录；其余旧实验已归档 |
| `outputs/`、`artifacts/`、`data_external/` | 原始数据、冻结物理源和结果证据，保持原路径，不参加默认源码扫描 |
| [KEEP.md](KEEP.md) | 保留理由、关键恢复资产和继续归档条件 |
| [archive/legacy_research_20261112/](archive/legacy_research_20261112/README.md) | 已退出主线的旧代码和文档，带路径／哈希清单及恢复工具 |

旧 `experiments.run_mainline`、`terminal_case33` 和历代研究模块已退出根目录默认入口。需要重放时按归档说明恢复，或使用其原版本的独立检出。本次不将旧 L1/MILP 结果并入新 MSE 主线。

根 `pyproject.toml` 现只打包 `rnj_wzzt_dr`，默认测试也只收集该包。已有 `Topo` 环境、`requirements.txt` 和 `environment.yml` 保留，供现有 AC 研究和物理源使用；没有重装或搬动解释器。
