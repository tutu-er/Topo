# 主线与独立研究分离记录

日期：2026-09-28。本次从已有未提交改动继续整理；比较基准是本次开始时的工作区源码快照，未覆盖此前的模型改动。

## 目录与依赖

生产调用链保持为：`run.py → cli → pipeline → 原始 P/Q/Y 对齐 → ordered R/X QP → RX75/RNJ → bootstrap 边界筛选 → 原终端 L1-MILP → 验证集选模`。

| 代码 | 当前归属 |
|---|---|
| RNJ 递归、零长度内部边规约、根子树提取、bootstrap | 核心 `rnj_wzzt/graph/` |
| 固定 RX75 的根深度、共享路径矩阵 | 核心 `graph/sensitivity_geometry.py` |
| 五种 R/X 模式、终端距离诊断 | `research_experiments/rnj/sensitivity_geometry.py` |
| 从 clade 建树、从距离反推共享路径 | `research_experiments/rnj/graph_adapters.py` |
| 聚合、伪终端、去嵌入、支撑标签展开 | `research_experiments/rnj/rooted_aggregation.py` |
| 历史时序配方、矩阵修正与诊断 | `research_experiments/rnj/temporal_preprocessing.py`、`matrix_constraints.py` |
| 原核心目录的 12 个研究实现及调参脚本 | `research_experiments/rnj/`；调参仍位于独立 `scripts/` |
| 上述研究测试 | `research_experiments/tests/`；必要主线独立 oracle 留在核心测试中 |
| 偷电实验 | 新目录 `research_experiments/theft/` 提供索引与子进程启动器，原工作区保留 |

表中 `research_experiments` 路径均相对于仓库根目录。核心代码不导入研究包；研究代码可复用核心算法。

零长度内部边规约仍保留：它处理 RNJ 输出树的退化边，不聚合终端 P/Q/Y。共享 QP 内核仍保留 `output_transform` 残差加权接口，供研究复用；主线始终使用默认值，公共模态实验协议和专门测试已经迁出。没有为隔离实验而复制 QP 或 MILP 求解器。

## 兼容与保护

- 旧 `rnj_wzzt_core/experiments/`、`scripts/` 为薄转发；模块别名维持原有 monkeypatch 行为。直接 CLI 调用规范模块的 `main()`，确保 Windows spawn 工作进程能加载研究函数。
- `terminal_case33` 的图研究接口、时序配方及矩阵修正转发到新研究包；主线接口继续转发到核心。
- 主研究源码指纹记录实际新实现，仍排除事后调参 `scripts/`。旧指纹不匹配仍拒绝缓存复用，历史结果使用对应源码快照重放。
- `theft_wzzt/` 与原 `tests/meter_theft_pilot/` 的位置、内容均保留。72 个 Python 文件与迁移前 SHA256 一致；未写入历史实验结果或校准缓存。
- 研究入口面向完整源码仓库。独立核心 wheel 不包含研究包、父项目或偷电工作区。

## 验证结果

| 检查 | 结果 |
|---|---|
| 核心、父项目、研究测试完整套件 | 1071 passed，1 skipped，35.38 秒 |
| 跳过项 | log-det 实验旧入口的导入兼容测试；当前环境未安装可选 CVXPY |
| 图几何迁移前后对照 | 575 个输入/模式案例逐字段完全一致；核心 RX75 两字段一致 |
| RNJ、零边规约、rooted_clades、两个迁出适配器 | 函数 AST 与快照一致 |
| 固定四终端 AC 全流程 | 含纯 MILP 与 RNJ+MILP，全部非计时 JSON 字段完全一致；RNJ 保留支撑 `101,102` |
| 算法文件 | pipeline、cli、bootstrap、QP、MILP、Gurobi 适配、多场景拟合和目标预处理共 8 文件 SHA256 与本次起点一致 |
| 独立打包 | wheel 的 27 个 Python 文件与核心源码逐字节一致；隔离导入全部 27 模块，父项目/研究导入数为 0 |
| 旧入口及多进程 | 模块身份、monkeypatch、旧 CLI worker pickle 和真实 spawn 子进程检查通过 |

全量命令（仓库根目录）：

```powershell
python -B -m pytest -q -p no:cacheprovider rnj_wzzt_core/tests tests research_experiments/tests
```

行为对照只排除 `selection_seconds`、`elapsed_seconds`、`attempt_runtimes` 三个计时字段。保留了目标值、求解状态、界、gap、停止原因、支撑和全部误差指标，没有隐藏求解失败或未获证扩展。小案例的两条路径均按原规则返回 `extension_not_proven_optimal`；这是原有证书门槛的停止行为，不是全局拓扑最优性证明。

额外限制如实保留：迁移前 paper15 的 30 秒求解预算出现未获证扩展；一个自建无噪声四终端案例在冻结源码中触发原有 MILP 目标一致性检查（MILP 为 0，物理重算约 `1.16e-7`）。本次结构迁移未修改该数值判据，也未宣称覆盖无噪声近零残差下的所有情形。未重跑完整偷电研究或可选 CVXPY 实验。

## 证据与恢复

本地目录：[`artifacts/core_reorganization_20260928/`](../../artifacts/core_reorganization_20260928/)。

- `before_source.zip`、`before_manifest.json`：迁移前 534 个源码/文档文件及 SHA256，包含当时未提交状态。
- `before_worktree.patch`、`before_status.txt`：本次起点的 Git 差异与状态。
- `behavior_comparison.json`、`normalized_before.json`、`normalized_after.json`：主线行为对照及受保护文件校验。
- `graph_verification.json`、`verify_graph_snapshot.py`：图几何和函数 AST 对照结果及可重放探针。
- `tests.log`、`tests.xml`：完整套件记录。
- `packaging/verification.json`、wheel：独立包验证。
- `migration_manifest.json`：相对于本次起点的新增、改变和迁出路径；`after_worktree.patch` 包含当前全部已跟踪差异，含此前改动。
- `after_source.zip`、`after_manifest.json`：完成迁移后的源码快照与 SHA256。

需要恢复时，先将 `before_source.zip` 解压到独立目录，再按清单恢复所需文件；不要整库覆盖后来继续进行的工作。历史 outputs、缓存及冻结快照未移动。
