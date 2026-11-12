# 偷电与未计量负荷实验

本目录提供独立偷电实验的索引和启动入口。当前 RNJ＋MILP 主线不导入本目录，也不调用偷电模型。

## 布局与保留原因

| 位置 | 内容 |
| --- | --- |
| [`../../theft_wzzt/`](../../theft_wzzt/README.md) | 独立研究包，保存当时的拓扑算法基线、偷电 MILP、仿真、校准、测试与实验结果。 |
| [`../../rnj_wzzt_core/tests/meter_theft_pilot/`](../../rnj_wzzt_core/tests/meter_theft_pilot/) | 原始固定幅值偷电小实验，共四个 Python 文件；没有接入生产主线。 |
| [`run.py`](run.py) | 通过子进程转发到原 `theft_wzzt/experiments/run_theft.py`，不另写实验算法。 |

原工作区和 pilot 保留原位置与文件内容。已有冻结协议、校准和检查点记录了源码哈希；原验证脚本还检查 pilot 的四个原始文件路径。实验与绘图脚本按原目录查找 `identified_tree.json` 和 `outputs/`，因此移动这些文件会影响复现与缓存有效性。

启动器使用当前 Python 解释器，将工作目录设置为 `theft_wzzt/`，原样转发参数、标准输入输出和退出码。这样旧实验拥有独立的模块搜索路径，不会与当前主线的同名模块混用。请使用已安装原实验依赖的环境，例如仓库使用的 `Topo` 环境。

## 实验入口

从仓库根目录运行：

```powershell
python research_experiments/theft/run.py --help
python -m research_experiments.theft.run --help
```

两个命令显示原 CLI 帮助，不执行实验。原有阶段全部可用：

| 阶段 | 作用 |
| --- | --- |
| `null --replicates 20` | 无偷电场景的全流程校准。 |
| `exp1` 至 `exp6` | 原六类检测实验，可选 `--tree identified` 或 `--tree true`。 |
| `report` | 汇总原实验结果。 |
| `m7_l1`、`m7_bias`、`m7_report` | M7 线损与固定幅值敏感性研究；沿用原缓存和协议。 |

例如 `python -m research_experiments.theft.run exp2` 与原入口行为相同。实际执行实验或报告阶段时，仍由原脚本写入 `theft_wzzt/outputs/`；本次目录整理未重跑这些阶段。

论文实验的其他入口保留在原工作区中：

```powershell
Set-Location theft_wzzt
python detect_conservative.py --help
python -m paper_study_v3.run_study --help
```

## 测试与历史结果验证

主仓库内新增的入口兼容测试：

```powershell
python -m pytest research_experiments/tests/test_theft_launcher.py
```

原算法测试在独立工作区执行，避免顶层 `experiments` 模块重名：

```powershell
Set-Location theft_wzzt
python -m pytest tests paper_study/test_profiles.py paper_study/test_window_entry.py paper_study_v2/test_regularized.py paper_study_v2/test_entry.py paper_study_v3/test_entry.py
```

历史文件校验入口为仓库根目录的 [`scripts/verify_theft_paper_artifacts.py`](../../scripts/verify_theft_paper_artifacts.py)。它依赖已有完整结果，检查冻结源码、校准签名和 pilot 路径，并会重新写入原 `outputs/theft_paper_v3/audit.json`，因此不属于此次仅检查启动兼容性的测试。
