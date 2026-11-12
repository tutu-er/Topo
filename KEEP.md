# 当前保留与后续归档条件

核对日期：2026-11-12。本清单区分当前代码、进行中的任务与恢复资产；保持原位不等于继续把它们当作默认主线。

## 当前主线

`rnj_wzzt_dr/` 是唯一的新研究入口。辨识、冻结评分、零动作核算和单对报价的代码闭包位于目录内；它不导入旧 `research_experiments/terminal_case33/experiments`。主线默认和概率口径未因归档改变。

## 任务完成前保留原位

- `topology_identification_standalone/`：其 `outputs/large96_main_20261112_001/state.json` 是当前批次的实时状态；保留全部运行器、冻结 runtime、输入、协议、锁、attempt 和完成记录。当前批次及后处理封存后，才评估该目录整体迁移。
- `rnj_wzzt_core/`：已有 5 个跟踪文件的本地修改，以及 AC 拓扑假设／界／替代结构相关的未跟踪源码、演示和测试。此次未移动、提交或覆盖这些内容。
- `research_experiments/coarse_to_fine/` 与 `research_experiments/wzzt_dr_selection/`：是主工作区的未跟踪研究代码，保留其原路径；后者不等同于 DR 专用工作树中已经交接的后续定价包。

## 作为数据与证据仓保存

以下内容不参与新主线默认代码导入。保留路径可以直接复用已有哈希协议，避免无必要地复制大型私人资产。

| 位置 | 保留理由 |
|---|---|
| `outputs/day96_t_benchmark_20261110/` | 旧训练／验证观测、网络与数据来源 |
| `outputs/mse_exit_tuning_20261111/` | 旧 `.24/QP` 冻结候选及参数，DR 40/188 证据的输入来源 |
| `outputs/miqp_loss_benchmark_20261111/` | MIQP 历史源快照、协议与候选谱系 |
| `outputs/wzzt_paper_20261029_full/source/` 及相关网络元数据 | 原 AC 桥依赖的冻结物理模拟器；原协议可能另指定物理源，以各协议为准 |
| 其余 `outputs/` 历史结果 | 数值档案，含失败／限时／未完成分母；此次未把它们认定为缓存或删除 |
| `outputs/lb/`、`outputs/pf/` | 已注册 Git 工作树，禁止按普通结果目录搬动 |
| `artifacts/` | 旧核验、源码快照与恢复证据仓；保持绝对路径引用 |
| `data_external/` | 外部数据与合法来源记录；部分原始文件未上传 Git |
| 独立目录内的 `data/cases/experiments/outputs` | 当前 96 点任务及旧完整验证的输入和原始证据 |

DR 专用工作树仍在 `C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo`。私人 DR 恢复根在 `C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a`。这两个位置在本次授权的主项目整理范围之外，保持原样。

将来迁移原 AC 资产时，旧完成协议保持原文；在新位置另建明确的源闭包协议并核验，再重放。现有程序不会自动搜索被搬走的原物理源。

## 环境及本地工作文件

`environment.yml`、`requirements.txt`、`.codex-rnj-deps/`、`.vscode/`、Python 安装元数据和运行中的 `tmp/` 保持原位。它们属于环境／工作状态，不计作新的科学算法。仅清理缓存不能替代代码归档，也不迁移可能带绝对路径的本地环境。

## 已移走的内容

早期 `docs/`、旧 `terminal_case33/experiments/scripts/tests/configs`、已结束的研究模块、独立旧偷电工程、根目录旧理论文章、旧 README/pyproject 和 `output/` 编译成品已移至 [本次归档](archive/legacy_research_20261112/README.md)。所有内容保留原字节与映射；原未跟踪／忽略文件只进入本地档案，不随本次提交上传。
