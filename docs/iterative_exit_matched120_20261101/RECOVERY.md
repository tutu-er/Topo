# 恢复与核验

入口协议protocol.json固定120个输入、两份源码、数值精度适配、4小时共同总预算及前端成本。PLAN.md说明控制变量和归因范围，RESULTS.md为总表，per_model/paired_inputs/mechanism_inputs为逐输入证据；verification.json是独立LP/哈希检查回执，solver.csv保留子问题状态与界。result_hashes.json对应全部240个最终训练记录。

records.zip包含sources/before、sources/after两份真正执行的源码，当前分析/交付源码current_source、共同前端与旧300秒记录、240个新结果、checkpoint、进度日志及逐输入独立LP核验。完整原始求解器记录位于归档的analysis/solver_records.json；独立CSV保留简明状态与界，避免另行提交重复的10MB未压缩JSON。归档不包含观测NPZ，原始数据路径及SHA256由catalog/protocol引用。

恢复时先核验manifest.json，再核验协议源码哈希及240个结果哈希。run命令会跳过已有result/error；恢复前须检查state.json中的worker PID，避免与尚在运行的进程重复计算。保留失败记录及原分母，处理错误时另存新协议或记录必要的驱动适配，不覆盖历史源码/结果。

运行命令：python -B -m research_experiments.iterative_benchmark.run run outputs/iterative_exit_matched120_20261101

分析命令：python -B -m research_experiments.iterative_benchmark.analyze outputs/iterative_exit_matched120_20261101

分析器按源码SHA256复用逐输入核验缓存；修改分析源码会使旧缓存失效。该入口要求240个新结果齐全。重新分析会更新报告，不必重做训练求解。
