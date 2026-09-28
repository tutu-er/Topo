# 独立研究实验

当前生产主线仍由 `rnj_wzzt_core/rnj_wzzt/` 实现：原始 P/Q/Y → 有序约束最小二乘 → RX75/RNJ → bootstrap 边界筛选 → 原终端 MILP 补全。本目录保存额外实验；核心包不导入本目录。

| 目录或模块 | 内容 |
|---|---|
| `rnj/rooted_aggregation.py` | RNJ 子树聚合、伪终端、去嵌入及标签展开 |
| `rnj/graph_adapters.py` | 从 clade 建树、从终端距离反推共享路径的研究适配 |
| `rnj/sensitivity_geometry.py` | R、X、不同 R/X 配比和距离诊断 |
| `rnj/temporal_preprocessing.py` | 显式去均值、差分等历史时序配方 |
| `rnj/matrix_constraints.py` | 矩阵可行性修正、树协方差诊断，供研究基线使用 |
| `rnj/scenario_observation.py` | 无根量测的失配对照；在核心已观测根仿真基础上构造固定标称根目标 |
| `rnj/run_*.py`、辅助模块 | 根信息、公共模态、正则化、消融及压力对照 |
| `rnj/scripts/run_matched_rnj_tuning.py` | 在主实验之外追加匹配调参，继续排除在主实验指纹集合之外 |
| [`theft/`](theft/README.md) | 偷电、未计量负荷实验的索引和隔离启动器 |
| `tests/` | 上述实验的数值检查和入口兼容测试 |

## 运行

在仓库根目录和已有 `Topo` 环境中执行：

```powershell
python -m research_experiments.rnj.run_rooted_ablation --help
python -m research_experiments.rnj.run_scenario_benchmark --help
python -m research_experiments.rnj.scripts.run_matched_rnj_tuning --help
python -m research_experiments.theft.run --help
python -m pytest -q rnj_wzzt_core/tests tests research_experiments/tests
```

RNJ 研究默认使用原 `rnj_wzzt_core/outputs/`；显式相对输出路径按调用时的工作目录解释。启动新研究时指定新的输出目录。负非对角项/log-det 实验额外需要 CVXPY 与 CLARABEL；它不属于普通主线依赖。

`rnj_wzzt_core/experiments/` 和 `rnj_wzzt_core/scripts/` 的旧文件仅保留转发，支持已有命令和模块导入。`terminal_case33` 的旧研究接口也转发到本目录；生产算法接口继续转发到核心。独立复制核心目录时，只需使用 `run.py` 和核心测试，额外研究入口需要完整仓库。

## 复现边界

- 本次保留已存在的实验结果、缓存、历史源码快照及校验规则。新实验指纹记录实际实现位置；旧指纹不匹配时仍拒绝复用，应使用当时快照重放旧结果。
- 原 `theft_wzzt/` 是带冻结算法、校准和结果路径的独立工作区，保留内容与位置，通过本目录的子进程入口运行。
- 原 `rnj_wzzt_core/tests/meter_theft_pilot/` 四个文件被论文校验脚本按原路径核对，保留原位；它们没有 `test_*.py`，不会被默认测试收集，也不参与生产流程。
- 共享 QP 内核仍提供残差加权的 `output_transform` 数学接口，公共模态实验复用它。主流程始终使用默认值；公共模态的实验设计、调参与专门测试均在本目录。这样无需复制约束编码和数值求解器。
- 核心场景生成器只接受 `exact/noisy`。研究 benchmark 通过 `scenario_observation` 保留 `unobserved`；物理数据、终端噪声、完整网格诊断与原协议一致。根信息研究原有的 `observation_view` 继续由该研究自身维护。

迁移前源码快照、行为对照和测试记录见 `artifacts/core_reorganization_20260928/`；整理说明见 [核心研究分离记录](../rnj_wzzt_core/docs/research_separation_20260928.md)。
