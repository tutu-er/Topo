# EV 研究已暂时封存

2026-10-19。EV 的当前实现、算例和原始证据位于独立的 `codex/ev-train-validation-design` 工作树，本主工作区保留统一入口。

| 阅读入口 | 内容 |
|---|---|
| [封存总览](C:/Users/23761/.codex/worktrees/36a1/Topo/research_experiments/ev/README.md) | 当前状态、阅读顺序与恢复说明 |
| [当前实现](C:/Users/23761/.codex/worktrees/36a1/Topo/research_experiments/ev/IMPLEMENTATION.md) | 实际调用链、统计模型、规划目标与可复用组件 |
| [算例结果](C:/Users/23761/.codex/worktrees/36a1/Topo/research_experiments/ev/RESULTS.md) | 各批次结果、完整分母、失败与未完成部分 |
| [文件作用与输出](C:/Users/23761/.codex/worktrees/36a1/Topo/research_experiments/ev/FILE_GUIDE.md) | 按功能查代码、协议、报告、检查点及证据 |
| [归档清单与核验](C:/Users/23761/.codex/worktrees/36a1/Topo/archive/ev_methods_20261019/README.md) | 原/新路径、逐文件哈希、Git与本机材料边界及恢复工具 |

主要结果：同一平方EV动作的最终推断从固定训练线路改为共享线路联合AC拟合，在一批72份独立评价中唯一正确35→69、误拒10→3。复杂规划相对同操作域平方强基线的稳定收益尚未建立。后续完整6/8末端流程中，G-SQ与P-SQ的校准唯一正确均为24/64，仍受候选漏真、联合拟合可用性和校准样本限制。不同批次的数字分别解释，不合并为总体精度。

旧方法和结果完整封存，后续可能采用新方法；新方法另建目录、先明确模型与评价方案。已按用户确认停止后台标准馈线批次及自动续跑进程：开发保存28/32、校准0/128、评价0/96，保留未完成清单。

D盘原EV执行目录集中到 [outputs/ev_archive_20261019](D:/0-github_workspace/Topo/outputs/ev_archive_20261019/)，按 `runs/` 和 `checks/` 分组。原路径恢复表在上述归档清单中；科学源码、失败和已有结果均保留。本次只整理并核对文件，没有新增科学实验或新方法。

其他设备可从 [EV 分支入口](https://github.com/tutu-er/Topo/tree/codex/ev-train-validation-design/research_experiments/ev) 读取受Git管理的说明与证据。本机原始数据没有全部上传，其范围以清单中的Git归属为准。
