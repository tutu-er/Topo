# EV 研究封存入口

状态：**2026-10-19 暂时封存，后续可能改用新方法。** 当前方法及已有结论保留作对照。新研究另建目录、先确定模型与评价方案；旧提示词和旧报告中的“下一步”作为历史记录保存。

当前最清楚的结果是：历史与新增量测共享线路参数的联合 AC 重拟合，比固定训练线路作最终推断更有效。复杂 EV 规划的额外收益仍不稳定；完整流程中的候选漏真、拟合可用性、响应预测和校准样本不足影响最终结果。

按下面的顺序阅读即可，不必从旧目录的长 README 开始。

| 文件 | 作用和输出 |
|---|---|
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | 当前实现的调用链、统计口径、EV 约束及可复用模块 |
| [RESULTS.md](RESULTS.md) | 主要算例结果、原分母、失败和未完成部分；各批次分开报告 |
| [FILE_GUIDE.md](FILE_GUIDE.md) | 按功能查文件，标注入口、输入和输出，定位历史批次 |
| [完整源码文件目录](../../archive/ev_methods_20261019/FILE_CATALOG.csv) | 每个源码、测试、数学稿、协议与检查点的用途、输出提示、原路径和封存路径 |
| [历史结果入口](../../archive/ev_methods_20261019/RESULTS_INDEX.csv) | 每批 results/report/checkpoint/reproduce 的文件位置 |
| [封存与恢复](../../archive/ev_methods_20261019/README.md) | 全量清单、逐文件 SHA-256、外部执行目录和恢复方式 |

目录分工：`research_experiments/ev/` 只放这组说明；旧 `mobile_ev_mle/` 和 `mobile_ev_probing/` 只留封存提示；源码、测试、报告和原证据完整移入 `archive/ev_methods_20261019/payload/research_experiments/`。D 盘的 EV 执行副本集中到 `outputs/ev_archive_20261019/`，原路径与新路径逐项记录。早期 2026-10-12 归档保留原位并纳入索引。

标准线路/公开时序的 256 条件批次已按用户确认停止，自动续跑器及其所属工作进程同时停止：已保存开发 **28/32**，校准 **0/128**，评价 **0/96**。剩余条件作为未完成保留，不能给出该批次的正式评价结论。封存记录见 [停止回执](../../archive/ev_methods_20261019/STOP_RECEIPT.json)。

本次整理只核对已有源码、保存报告和文件完整性，没有新增科学算例，没有重跑旧优化，也没有重编译原 PDF。历史报告中的测试数属于当时批次；本次验证结果单独记入 [封存核验](../../archive/ev_methods_20261019/VERIFICATION.json)。原 PDF 和 `pyproject.toml` 的已有改动保留原状，不并入封存提交。
